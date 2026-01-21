"""
累加器与精度全面测试
===================

测试维度：
1. 组件精度：FP8 / FP16 / FP32 Linear
2. 累加模式：Sequential vs Parallel (SNN累加器核心机制)
3. 中间精度：FP32 vs FP64 (针对 Softmax, RMSNorm, LayerNorm, Linear)

目标：
覆盖 MofNeuroSim 项目中关于累加器和精度的所有关键验证点。

作者: MofNeuroSim Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import struct
from collections import defaultdict
import time

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# 工具函数
# =============================================================================

def compute_ulp_error(snn_val, ref_val, precision='fp32'):
    """计算 ULP 误差"""
    if not np.isfinite(ref_val) or not np.isfinite(snn_val):
        return 0 if (np.isnan(ref_val) and np.isnan(snn_val)) or (ref_val == snn_val) else float('inf')

    if ref_val == snn_val:
        return 0

    if precision == 'fp32':
        ref_bits = struct.unpack('>I', struct.pack('>f', float(ref_val)))[0]
        snn_bits = struct.unpack('>I', struct.pack('>f', float(snn_val)))[0]
    elif precision == 'fp16':
        ref_bits = int(np.float16(ref_val).view(np.uint16))
        snn_bits = int(np.float16(snn_val).view(np.uint16))
    elif precision == 'fp8':
        # FP8 E4M3: 简化处理
        return abs(float(snn_val) - float(ref_val)) / max(abs(float(ref_val)), 1e-10)
    else:
        ref_bits = struct.unpack('>I', struct.pack('>f', float(ref_val)))[0]
        snn_bits = struct.unpack('>I', struct.pack('>f', float(snn_val)))[0]

    return abs(int(ref_bits) - int(snn_bits))


def compute_ulp_stats(snn_tensor, ref_tensor, precision='fp32'):
    """计算张量的 ULP 统计信息"""
    snn_flat = snn_tensor.detach().cpu().flatten().numpy()
    ref_flat = ref_tensor.detach().cpu().flatten().numpy()
    
    ulps = []
    for s, r in zip(snn_flat, ref_flat):
        ulps.append(compute_ulp_error(s, r, precision))
    
    # 过滤掉非有限值的 ULP (通常是 Inf/NaN)
    valid_ulps = [u for u in ulps if np.isfinite(u)]
    
    if not valid_ulps:
        return {'max': -1, 'mean': -1, 'le1_rate': 0.0}
        
    return {
        'max': max(valid_ulps),
        'mean': np.mean(valid_ulps),
        'le1_rate': sum(1 for u in valid_ulps if u <= 1) / len(valid_ulps) * 100
    }


def generate_test_values(dim, max_abs=None):
    """生成包含边界值和随机值的测试数据"""
    values = []
    # 边界值
    boundary = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]
    if max_abs:
        boundary.extend([max_abs, -max_abs, max_abs/2])
    else:
        boundary.extend([10.0, -10.0])
        
    # 填充
    values.extend(boundary[:min(dim, len(boundary))])
    remaining = dim - len(values)
    if remaining > 0:
        random_vals = torch.randn(remaining).tolist()
        if max_abs:
            random_vals = [max(-max_abs, min(max_abs, v)) for v in random_vals]
        values.extend(random_vals)
        
    return torch.tensor(values[:dim], dtype=torch.float32, device=device)


# =============================================================================
# 测试模块
# =============================================================================

def test_accumulator_modes():
    """测试 1: 累加器拓扑逻辑验证 (串行链式 vs 并行树状)"""
    print("\n" + "="*70)
    print("测试 1: 累加器拓扑逻辑验证 (串行链式 vs 并行树状)")
    print("目标: 使用高精度 FP64 加法器验证不同归约策略(Topology)的数学正确性")
    print("="*70)

    from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
    from atomic_ops.core.accumulator import SequentialAccumulator, ParallelAccumulator
    from atomic_ops.encoding.converters import float64_to_pulse, pulse_to_float64

    test_dims = [8, 16, 32, 64]

    results = []
    for dim in test_dims:
        # 每次迭代重新实例化，确保无状态残留
        adder_seq = SpikeFP64Adder().to(device)
        adder_par = SpikeFP64Adder().to(device)
        seq_acc = SequentialAccumulator(adder_seq)
        par_acc = ParallelAccumulator(adder_par)

        data = torch.randn(dim).to(device)
        ref_sum = data.sum().item()
        data_pulse = float64_to_pulse(data.double(), device=device)

        # Sequential
        seq_result_pulse = seq_acc.reduce(data_pulse, dim=-2)
        seq_result = pulse_to_float64(seq_result_pulse).item()

        # Parallel
        par_result_pulse = par_acc.reduce(data_pulse, dim=-2)
        par_result = pulse_to_float64(par_result_pulse).item()

        seq_ulp = compute_ulp_error(seq_result, ref_sum, 'fp32')
        par_ulp = compute_ulp_error(par_result, ref_sum, 'fp32')

        results.append({
            'dim': dim,
            'seq_ulp': seq_ulp,
            'par_ulp': par_ulp
        })
        print(f"  dim={dim:3d}: Seq ULP={seq_ulp}, Par ULP={par_ulp}")

    return results


def test_softmax_precision():
    """测试 2: SpikeFP32Softmax (FP32 IO) - 中间累加精度验证"""
    print("\n" + "="*70)
    print("测试 2: SpikeFP32Softmax (FP32 IO) - 中间累加精度验证")
    print("目标: 验证 accum_precision='fp64' 在输入输出保持 FP32 时的精度增益")
    print("="*70)

    from atomic_ops import SpikeFP32Softmax
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32

    # FP32 Softmax (标准版) vs "fp64" Softmax (高精度版)
    # 注意：输入输出均为 FP32, 仅由 accum_precision 控制中间累加逻辑

    test_dims = [64, 128, 256, 512, 1024] # 恢复正常测试维度

    results = []

    for dim in test_dims:
        softmax_fp32 = SpikeFP32Softmax(accum_precision='fp32').to(device)
        softmax_fp64 = SpikeFP32Softmax(accum_precision='fp64').to(device)

        x = generate_test_values(dim, max_abs=10.0).unsqueeze(0)
        ref = torch.softmax(x, dim=-1)
        x_pulse = float32_to_pulse(x)

        # FP32
        softmax_fp32.reset()
        y_fp32 = pulse_to_float32(softmax_fp32(x_pulse))
        stats_fp32 = compute_ulp_stats(y_fp32, ref)

        # FP64
        softmax_fp64.reset()
        y_fp64 = pulse_to_float32(softmax_fp64(x_pulse))
        stats_fp64 = compute_ulp_stats(y_fp64, ref)
        
        results.append({
            'dim': dim,
            'fp32': stats_fp32,
            'fp64': stats_fp64
        })
        print(f"  dim={dim:4d}:")
        print(f"    FP32: Max ULP={stats_fp32['max']:>4}, ≤1-ULP={stats_fp32['le1_rate']:>5.1f}%")
        print(f"    FP64: Max ULP={stats_fp64['max']:>4}, ≤1-ULP={stats_fp64['le1_rate']:>5.1f}%")

    return results


def test_rmsnorm_precision():
    """测试 3: SpikeFP32RMSNormFullFP64 (FP32 IO) - 精度验证"""
    print("\n" + "="*70)
    print("测试 3: SpikeFP32RMSNormFullFP64 (FP32 IO) - 精度验证")
    print("目标: 验证 FP32 IO 接口下，内部全链路 FP64 计算的高精度表现")
    print("="*70)
    
    from atomic_ops import SpikeFP32RMSNormFullFP64
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
    
    test_dims = [64, 128, 256, 512] 
    results = []
    
    for dim in test_dims:
        x = generate_test_values(dim).unsqueeze(0)
        # PyTorch Reference
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-6)
        ref = x / rms
        
        x_pulse = float32_to_pulse(x)
        
        # Test FP64 variant
        rmsnorm = SpikeFP32RMSNormFullFP64(dim, eps=1e-6).to(device)
        y_pulse = rmsnorm(x_pulse)
        y_snn = pulse_to_float32(y_pulse)
        
        stats = compute_ulp_stats(y_snn, ref)
        results.append({'dim': dim, 'stats': stats})
        
        print(f"  dim={dim:3d}: Max ULP={stats['max']}, Mean={stats['mean']:.2f}, ≤1-ULP={stats['le1_rate']:.1f}%")
        
    return results


def test_layernorm_precision():
    """测试 4: SpikeFP32LayerNorm (FP32 IO) - 精度验证"""
    print("\n" + "="*70)
    print("测试 4: SpikeFP32LayerNorm (FP32 IO) - 精度验证")
    print("目标: 验证 FP32 IO 接口下，内部 FP64 累加的高精度表现")
    print("="*70)
    
    from atomic_ops import SpikeFP32LayerNorm
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
    
    test_dims = [64, 128, 256, 512]
    results = []
    
    for dim in test_dims:
        x = generate_test_values(dim).unsqueeze(0)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        ref = (x - mean) / torch.sqrt(var + 1e-6)
        
        x_pulse = float32_to_pulse(x)
        ln = SpikeFP32LayerNorm(accumulator_mode='sequential').to(device)
        y_pulse = ln(x_pulse)
        y_snn = pulse_to_float32(y_pulse)
        
        stats = compute_ulp_stats(y_snn, ref)
        results.append({'dim': dim, 'stats': stats})
        print(f"  dim={dim:3d}: Max ULP={stats['max']}, ≤1-ULP={stats['le1_rate']:.1f}%")
        
    return results


def test_linear_precision_modes():
    """测试 5: SpikeFP32Linear (FP32 IO) - 中间累加精度验证"""
    print("\n" + "="*70)
    print("测试 5: SpikeFP32Linear (FP32 IO) - 中间累加精度验证")
    print("目标: 对比 accum_precision='fp32' vs 'fp64' 对大矩阵乘法精度的影响")
    print("="*70)
    
    from atomic_ops import SpikeFP32Linear_MultiPrecision
    from atomic_ops.encoding.converters import float32_to_pulse, pulse_to_float32
    
    dims = [(64, 64), (128, 128), (256, 256), (512, 512)]
    results = []
    
    for in_dim, out_dim in dims:
        x = torch.randn(1, in_dim).to(device) * 0.5
        ref_x = x.clone()
        w = torch.randn(out_dim, in_dim).to(device) * 0.1
        ref = torch.nn.functional.linear(ref_x, w)
        
        x_pulse = float32_to_pulse(x)
        
        # Test FP32 Accum
        lin_fp32 = SpikeFP32Linear_MultiPrecision(in_dim, out_dim, accum_precision='fp32').to(device)
        lin_fp32.set_weight_from_float(w)
        y_fp32 = pulse_to_float32(lin_fp32(x_pulse))
        stats_fp32 = compute_ulp_stats(y_fp32, ref)
        
        # Test FP64 Accum
        lin_fp64 = SpikeFP32Linear_MultiPrecision(in_dim, out_dim, accum_precision='fp64').to(device)
        lin_fp64.set_weight_from_float(w)
        y_fp64 = pulse_to_float32(lin_fp64(x_pulse))
        stats_fp64 = compute_ulp_stats(y_fp64, ref)
        
        results.append({
            'dim': f"{in_dim}x{out_dim}",
            'fp32': stats_fp32,
            'fp64': stats_fp64
        })
        print(f"  {in_dim}x{out_dim}:")
        print(f"    FP32 Accum: Max ULP={stats_fp32['max']:>4}, ≤1-ULP={stats_fp32['le1_rate']:>5.1f}%")
        print(f"    FP64 Accum: Max ULP={stats_fp64['max']:>4}, ≤1-ULP={stats_fp64['le1_rate']:>5.1f}%")
        
    return results


def main():
    print("MofNeuroSim 全面验证套件")
    print("-" * 30)
    
    try:
        test_accumulator_modes()
    except Exception as e:
        print(f"Test 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_softmax_precision()
    except Exception as e:
        print(f"Test 2 Failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_rmsnorm_precision()
    except Exception as e:
        print(f"Test 3 Failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_layernorm_precision()
    except Exception as e:
        print(f"Test 4 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_linear_precision_modes()
    except Exception as e:
        print(f"Test 5 Failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n测试完成")


if __name__ == '__main__':
    main()
