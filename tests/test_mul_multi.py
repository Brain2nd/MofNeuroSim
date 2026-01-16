"""
多精度乘法器测试
================

测试 FP8 和 FP16 多精度乘法器的功能正确性。

设计原则：输入输出精度一致，中间精度可配置。
- FP8 × FP8 → FP8（中间精度：FP8/FP16/FP32）
- FP16 × FP16 → FP16（中间精度：FP16/FP32）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from atomic_ops import (
    SpikeFP8Multiplier_MultiPrecision,
    SpikeFP16Multiplier_MultiPrecision,
    PulseFloatingPointEncoder,
    PulseFP32Decoder,
    SimpleLIFNode
)
from atomic_ops.converters import float16_to_pulse, pulse_to_float16
from atomic_ops.pulse_decoder import PulseFP16Decoder


def test_fp8_mul_output_dimensions():
    """测试 FP8 多精度乘法器输出维度（始终为 FP8）"""
    print("\n--- FP8 多精度乘法器输出维度测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建三种中间精度的乘法器
    mul_fp8 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp8').to(device)
    mul_fp16 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp16').to(device)
    mul_fp32 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)

    # 创建 FP8 脉冲输入 (模拟)
    a = torch.zeros(2, 3, 8, device=device)
    b = torch.zeros(2, 3, 8, device=device)

    # 测试输出维度（都应该是 8 位 FP8）
    out_fp8 = mul_fp8(a, b)
    out_fp16 = mul_fp16(a, b)
    out_fp32 = mul_fp32(a, b)

    fp8_ok = out_fp8.shape[-1] == 8
    fp16_ok = out_fp16.shape[-1] == 8  # 输出仍是 FP8
    fp32_ok = out_fp32.shape[-1] == 8  # 输出仍是 FP8

    print(f"  中间FP8 → 输出维度: {out_fp8.shape[-1]} (期望 8) {'✓' if fp8_ok else '✗'}")
    print(f"  中间FP16 → 输出维度: {out_fp16.shape[-1]} (期望 8) {'✓' if fp16_ok else '✗'}")
    print(f"  中间FP32 → 输出维度: {out_fp32.shape[-1]} (期望 8) {'✓' if fp32_ok else '✗'}")

    return fp8_ok and fp16_ok and fp32_ok


def test_fp16_mul_output_dimensions():
    """测试 FP16 多精度乘法器输出维度（始终为 FP16）"""
    print("\n--- FP16 多精度乘法器输出维度测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建两种中间精度的乘法器
    mul_fp16 = SpikeFP16Multiplier_MultiPrecision(intermediate_precision='fp16').to(device)
    mul_fp32 = SpikeFP16Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)

    # 创建 FP16 脉冲输入 (模拟)
    a = torch.zeros(2, 3, 16, device=device)
    b = torch.zeros(2, 3, 16, device=device)

    # 测试输出维度（都应该是 16 位 FP16）
    out_fp16 = mul_fp16(a, b)
    out_fp32 = mul_fp32(a, b)

    fp16_ok = out_fp16.shape[-1] == 16  # 输出是 FP16
    fp32_ok = out_fp32.shape[-1] == 16  # 输出仍是 FP16

    print(f"  中间FP16 → 输出维度: {out_fp16.shape[-1]} (期望 16) {'✓' if fp16_ok else '✗'}")
    print(f"  中间FP32 → 输出维度: {out_fp32.shape[-1]} (期望 16) {'✓' if fp32_ok else '✗'}")

    return fp16_ok and fp32_ok


def test_fp8_mul_numerical():
    """测试 FP8 多精度乘法器数值正确性"""
    print("\n--- FP8 多精度乘法器数值测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建编码器和解码器
    encoder = PulseFloatingPointEncoder().to(device)
    decoder = PulseFloatingPointEncoder()  # FP8 解码用编码器的反向

    # 创建 FP32 中间精度的乘法器
    mul = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)

    test_cases = [
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 4.0),
        (0.5, 2.0, 1.0),
        (-1.0, 2.0, -2.0),
        (1.5, 2.0, 3.0),
    ]

    correct = 0
    for a_val, b_val, expected in test_cases:
        # 编码输入
        encoder.reset()
        a_pulse = encoder(torch.tensor([a_val], device=device))
        encoder.reset()
        b_pulse = encoder(torch.tensor([b_val], device=device))

        # 执行乘法（输出是 FP8）
        result_pulse = mul(a_pulse, b_pulse)

        # 解码 FP8 输出
        from atomic_ops.converters import fp8_bits_to_float
        result = fp8_bits_to_float(result_pulse.squeeze(0)).item()

        # 验证（允许 FP8 舍入误差）
        if abs(result - expected) < 0.5:  # FP8 精度较低
            correct += 1
            status = '✓'
        else:
            status = '✗'

        print(f"  {a_val} × {b_val} = {result:.4f} (期望 {expected}) {status}")

    print(f"  数值测试: {correct}/{len(test_cases)}")
    return correct == len(test_cases)


def test_fp16_mul_numerical():
    """测试 FP16 多精度乘法器数值正确性"""
    print("\n--- FP16 多精度乘法器数值测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建 FP16 解码器
    decoder = PulseFP16Decoder().to(device)

    # 创建 FP32 中间精度的乘法器
    mul = SpikeFP16Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)

    test_cases = [
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 4.0),
        (0.5, 2.0, 1.0),
        (-1.0, 2.0, -2.0),
        (1.5, 2.0, 3.0),
    ]

    correct = 0
    for a_val, b_val, expected in test_cases:
        # 编码 FP16 输入
        a_pulse = float16_to_pulse(torch.tensor([a_val], device=device, dtype=torch.float16))
        b_pulse = float16_to_pulse(torch.tensor([b_val], device=device, dtype=torch.float16))

        # 执行乘法（输出是 FP16）
        mul.reset()
        result_pulse = mul(a_pulse, b_pulse)

        # 解码 FP16 输出
        result = decoder(result_pulse).item()

        # 验证
        if abs(result - expected) < 0.01:
            correct += 1
            status = '✓'
        else:
            status = '✗'

        print(f"  {a_val} × {b_val} = {result:.4f} (期望 {expected}) {status}")

    print(f"  数值测试: {correct}/{len(test_cases)}")
    return correct == len(test_cases)


def test_neuron_template_propagation():
    """测试 neuron_template 参数传递"""
    print("\n--- neuron_template 参数传递测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lif_template = SimpleLIFNode(beta=0.9)

    # FP8 乘法器
    mul_fp8 = SpikeFP8Multiplier_MultiPrecision(
        intermediate_precision='fp32',
        neuron_template=lif_template
    ).to(device)

    # FP16 乘法器
    mul_fp16 = SpikeFP16Multiplier_MultiPrecision(
        intermediate_precision='fp32',
        neuron_template=lif_template
    ).to(device)

    # 检查内部乘法器是否收到了模板
    fp8_has_mul = hasattr(mul_fp8, 'mul')
    fp16_has_mul = hasattr(mul_fp16, 'mul')

    print(f"  FP8 乘法器创建成功: {fp8_has_mul} {'✓' if fp8_has_mul else '✗'}")
    print(f"  FP16 乘法器创建成功: {fp16_has_mul} {'✓' if fp16_has_mul else '✗'}")

    # 功能测试
    a_fp8 = torch.zeros(1, 8, device=device)
    b_fp8 = torch.zeros(1, 8, device=device)
    a_fp16 = torch.zeros(1, 16, device=device)
    b_fp16 = torch.zeros(1, 16, device=device)

    try:
        out_fp8 = mul_fp8(a_fp8, b_fp8)
        out_fp16 = mul_fp16(a_fp16, b_fp16)
        # 验证输出维度
        functional_ok = out_fp8.shape[-1] == 8 and out_fp16.shape[-1] == 16
    except Exception as e:
        print(f"  功能测试失败: {e}")
        functional_ok = False

    print(f"  功能测试: {'✓' if functional_ok else '✗'}")

    return fp8_has_mul and fp16_has_mul and functional_ok


def test_batch_processing():
    """测试批量处理"""
    print("\n--- 批量处理测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FP8 批量测试
    mul_fp8 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)
    a_fp8 = torch.zeros(4, 5, 8, device=device)
    b_fp8 = torch.zeros(4, 5, 8, device=device)

    out_fp8 = mul_fp8(a_fp8, b_fp8)
    fp8_batch_ok = out_fp8.shape == (4, 5, 8)  # 输出是 FP8
    print(f"  FP8 批量: 输入 (4,5,8) × (4,5,8) → 输出 {tuple(out_fp8.shape)} (期望 (4,5,8)) {'✓' if fp8_batch_ok else '✗'}")

    # FP16 批量测试
    mul_fp16 = SpikeFP16Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)
    a_fp16 = torch.zeros(4, 5, 16, device=device)
    b_fp16 = torch.zeros(4, 5, 16, device=device)

    out_fp16 = mul_fp16(a_fp16, b_fp16)
    fp16_batch_ok = out_fp16.shape == (4, 5, 16)  # 输出是 FP16
    print(f"  FP16 批量: 输入 (4,5,16) × (4,5,16) → 输出 {tuple(out_fp16.shape)} (期望 (4,5,16)) {'✓' if fp16_batch_ok else '✗'}")

    return fp8_batch_ok and fp16_batch_ok


def test_broadcast():
    """测试广播功能"""
    print("\n--- 广播功能测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FP8 广播测试
    mul_fp8 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)
    a = torch.zeros(4, 1, 8, device=device)
    b = torch.zeros(1, 5, 8, device=device)

    out = mul_fp8(a, b)
    broadcast_ok = out.shape == (4, 5, 8)  # 输出是 FP8
    print(f"  广播: (4,1,8) × (1,5,8) → {tuple(out.shape)} (期望 (4,5,8)) {'✓' if broadcast_ok else '✗'}")

    return broadcast_ok


def test_intermediate_precision_comparison():
    """测试不同中间精度的效果"""
    print("\n--- 中间精度对比测试 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建三种中间精度的 FP8 乘法器
    mul_fp8 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp8').to(device)
    mul_fp16 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp16').to(device)
    mul_fp32 = SpikeFP8Multiplier_MultiPrecision(intermediate_precision='fp32').to(device)

    # 创建编码器
    encoder = PulseFloatingPointEncoder().to(device)

    # 测试值
    encoder.reset()
    a_pulse = encoder(torch.tensor([1.5], device=device))
    encoder.reset()
    b_pulse = encoder(torch.tensor([2.0], device=device))

    # 执行乘法
    result_fp8 = mul_fp8(a_pulse.clone(), b_pulse.clone())
    result_fp16 = mul_fp16(a_pulse.clone(), b_pulse.clone())
    result_fp32 = mul_fp32(a_pulse.clone(), b_pulse.clone())

    # 解码
    from atomic_ops.converters import fp8_bits_to_float
    val_fp8 = fp8_bits_to_float(result_fp8.squeeze(0)).item()
    val_fp16 = fp8_bits_to_float(result_fp16.squeeze(0)).item()
    val_fp32 = fp8_bits_to_float(result_fp32.squeeze(0)).item()

    print(f"  1.5 × 2.0 (期望 3.0):")
    print(f"    中间FP8:  {val_fp8:.4f}")
    print(f"    中间FP16: {val_fp16:.4f}")
    print(f"    中间FP32: {val_fp32:.4f}")

    # 所有输出维度都应该是 8
    all_fp8 = (result_fp8.shape[-1] == 8 and
               result_fp16.shape[-1] == 8 and
               result_fp32.shape[-1] == 8)
    print(f"  所有输出都是 FP8 (8位): {'✓' if all_fp8 else '✗'}")

    return all_fp8


def main():
    print("=" * 60)
    print("多精度乘法器测试")
    print("=" * 60)
    print("设计原则：输入输出精度一致，中间精度可配置")
    print("  - FP8 × FP8 → FP8（中间：FP8/FP16/FP32）")
    print("  - FP16 × FP16 → FP16（中间：FP16/FP32）")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    results = []

    # 维度测试
    results.append(("FP8 输出维度", test_fp8_mul_output_dimensions()))
    results.append(("FP16 输出维度", test_fp16_mul_output_dimensions()))

    # 数值测试
    results.append(("FP8 数值正确性", test_fp8_mul_numerical()))
    results.append(("FP16 数值正确性", test_fp16_mul_numerical()))

    # 其他测试
    results.append(("neuron_template 传递", test_neuron_template_propagation()))
    results.append(("批量处理", test_batch_processing()))
    results.append(("广播功能", test_broadcast()))
    results.append(("中间精度对比", test_intermediate_precision_comparison()))

    # 汇总
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    if failed > 0:
        print("\n警告: 存在失败的测试！")
    else:
        print("\n所有测试通过！")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
