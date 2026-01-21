"""
全精度 Linear 层对齐测试 (Comprehensive Linear Alignment Test)
==========================================================

验证 MofNeuroSim 所有精度 Linear 层 (FP8, FP16, FP32) 在不同累加模式下的数值行为。
对比基准：PyTorch 的高精度浮点运算 (FP32/FP64)。

覆盖范围:
1. FP8 Linear  (Accum: FP8, FP16, FP32)
2. FP16 Linear (Accum: FP16, FP32)
3. FP32 Linear (Accum: FP32, FP64)
"""
import torch
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import (
    PulseFloatingPointEncoder,
    SpikeFP8Linear_MultiPrecision
)
# 注意：需要从子模块导入 FP16/FP32 Linear，因为它们不一定在顶层 __init__ 暴露
from atomic_ops.linear.fp16.fp16_linear import SpikeFP16Linear_MultiPrecision
from atomic_ops.linear.fp32.fp32_linear import SpikeFP32Linear_MultiPrecision

from atomic_ops.encoding.converters import float16_to_pulse, float32_to_pulse
from atomic_ops.encoding.pulse_decoder import (
    PulseFloatingPointDecoder, PulseFP16Decoder, PulseFP32Decoder
)

def run_test_fp8(device, batch=20, in_dim=16, out_dim=8):
    print("\n" + "-"*60)
    print(f"1. FP8 Linear 测试 (Input=FP8, In={in_dim}, Out={out_dim})")
    print("-"*60)
    
    # 准备数据
    torch.manual_seed(42)
    x_f32 = torch.randn(batch, in_dim, device=device) * 0.5
    w_f32 = torch.randn(out_dim, in_dim, device=device) * 0.5
    
    # 量化为 FP8
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)
    x_in = x_fp8.float()
    w_in = w_fp8.float()
    
    # PyTorch 参考 (nn.Linear FP32)
    ref_linear = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linear.weight.data = w_in
    y_ref = ref_linear(x_in)
    
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    decoder = PulseFloatingPointDecoder().to(device) # FP8 解码器
    decoder_fp16 = PulseFP16Decoder().to(device)
    decoder_fp32 = PulseFP32Decoder().to(device)
    
    x_pulse = encoder(x_in)
    
    modes = [
        ('fp8',  decoder),
        ('fp16', decoder_fp16),
        ('fp32', decoder_fp32)
    ]
    
    for accum_mode, dec in modes:
        print(f"Testing Accumulation: {accum_mode.upper()}")
        try:
            model = SpikeFP8Linear_MultiPrecision(
                in_dim, out_dim, accum_precision=accum_mode
            ).to(device)
            model.set_weight_from_float(w_in, encoder)
            model.reset()
            
            y_pulse = model(x_pulse)
            dec.reset()
            y_snn = dec(y_pulse)
            
            # 对齐检查
            # 注意：FP8 累加会有误差，FP16/32 累加应与 PyTorch FP32 结果一致 (因为 FP8xFP8 无损积也是 FP16 范围)
            # 我们将 PyTorch 结果转为对应输出精度进行比较
            if accum_mode == 'fp8':
                y_target = y_ref.to(torch.float8_e4m3fn).float()
                tol = 1e-1 # FP8 容忍较大
            elif accum_mode == 'fp16':
                y_target = y_ref.to(torch.float16).float()
                tol = 1e-3
            else:
                y_target = y_ref
                tol = 1e-5
                
            match = torch.isclose(y_snn, y_target, atol=tol, rtol=1e-3)
            acc = match.float().mean().item() * 100
            
            # 使用 MSE 辅助判断
            mse = torch.nn.functional.mse_loss(y_snn, y_target).item()
            
            print(f"  -> Match Rate: {acc:5.1f}% | MSE: {mse:.6e}")
            
        except Exception as e:
            print(f"  -> Failed: {e}")

def run_test_fp16(device, batch=20, in_dim=16, out_dim=8):
    print("\n" + "-"*60)
    print(f"2. FP16 Linear 测试 (Input=FP16, In={in_dim}, Out={out_dim})")
    print("-"*60)
    
    torch.manual_seed(42)
    x_f32 = torch.randn(batch, in_dim, device=device) * 0.5
    w_f32 = torch.randn(out_dim, in_dim, device=device) * 0.5
    
    x_fp16 = x_f32.to(torch.float16)
    w_fp16 = w_f32.to(torch.float16)
    
    # PyTorch 参考 (nn.Linear FP32 -> FP16 Result)
    # 即使是 fp16 input, PyTorch nn.Linear 默认会使用高精度核心(除非使用 amp)
    # 为了确切对比，我们显式转为 float 跑 linear 再转回
    ref_linear = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linear.weight.data = w_fp16.float()
    y_ref_f32 = ref_linear(x_fp16.float())
    
    x_pulse = float16_to_pulse(x_fp16)
    
    decoder_fp16 = PulseFP16Decoder().to(device)
    
    modes = ['fp16', 'fp32']
    
    for accum_mode in modes:
        print(f"Testing Accumulation: {accum_mode.upper()}")
        try:
            model = SpikeFP16Linear_MultiPrecision(
                in_dim, out_dim, accum_precision=accum_mode
            ).to(device)
            
            # FP16 Linear 权重设置
            # 如果是 inference mode (accumulation test), weight_pulse 是 buffer
            # 我们需要手动 set_weight_pulse 或者 hack
            # 查看源码发现它没有 set_weight_from_float，而是依赖 buffer 或者 STE parameter
            # 这里我们手动设置 buffer
            w_pulse_val = float16_to_pulse(w_fp16)
            model.weight_pulse = w_pulse_val
            
            model.reset()
            y_pulse = model(x_pulse)
            
            decoder_fp16.reset() # FP16 linear 始终输出 FP16 格式 (见 docstring: 输入输出始终为 FP16)
            y_snn = decoder_fp16(y_pulse)
            
            y_target = y_ref_f32.to(torch.float16).float()
            
            # FP16 累加可能有精度损失，FP32 累加应该更准
            match = torch.isclose(y_snn, y_target, atol=1e-3, rtol=1e-3)
            acc = match.float().mean().item() * 100
            mse = torch.nn.functional.mse_loss(y_snn, y_target).item()
            
            print(f"  -> Match Rate: {acc:5.1f}% | MSE: {mse:.6e}")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            import traceback
            traceback.print_exc()

def run_test_fp32(device, batch=20, in_dim=16, out_dim=8):
    print("\n" + "-"*60)
    print(f"3. FP32 Linear 测试 (Input=FP32, In={in_dim}, Out={out_dim})")
    print("-"*60)
    
    torch.manual_seed(42)
    x_f32 = torch.randn(batch, in_dim, device=device) * 0.5
    w_f32 = torch.randn(out_dim, in_dim, device=device) * 0.5
    
    # PyTorch 参考 (nn.Linear FP64 for gold standard)
    ref_linear = torch.nn.Linear(in_dim, out_dim, bias=False).to(device).double()
    ref_linear.weight.data = w_f32.double()
    y_ref_f64 = ref_linear(x_f32.double())
    y_target = y_ref_f64.float()
    
    x_pulse = float32_to_pulse(x_f32)
    
    decoder_fp32 = PulseFP32Decoder().to(device)
    
    modes = ['fp32', 'fp64']
    
    for accum_mode in modes:
        print(f"Testing Accumulation: {accum_mode.upper()}")
        try:
            model = SpikeFP32Linear_MultiPrecision(
                in_dim, out_dim, accum_precision=accum_mode
            ).to(device)
            
            # 设置权重
            w_pulse_val = float32_to_pulse(w_f32)
            # FP32 Linear 在非 STE 模式下，weight_pulse 是 buffer，没有 set helper
            if hasattr(model, 'weight_pulse'):
                # 如果是 Parameter (STE) 或 Buffer
                if isinstance(model.weight_pulse, torch.nn.Parameter):
                     model.weight_pulse.data.copy_(w_pulse_val)
                else:
                     model.weight_pulse = w_pulse_val
            
            model.reset()
            y_pulse = model(x_pulse)
            
            decoder_fp32.reset()
            y_snn = decoder_fp32(y_pulse)
            
            # FP32 vs FP64 Accum
            # FP32 累加会有普通的 float 误差
            # FP64 累加在大量求和时精度更高
            
            match = torch.isclose(y_snn, y_target, atol=1e-5, rtol=1e-5)
            acc = match.float().mean().item() * 100
            mse = torch.nn.functional.mse_loss(y_snn, y_target).item()
            
            print(f"  -> Match Rate: {acc:5.1f}% | MSE: {mse:.6e}")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    run_test_fp8(device)
    run_test_fp16(device)
    run_test_fp32(device)
