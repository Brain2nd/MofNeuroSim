"""
FP16 组件 - 100%纯SNN门电路实现
================================

包含:
- FP16ToFP32Converter: FP16 → FP32 转换器
- SpikeFP16MulToFP32: FP16 × FP16 → FP32 乘法器

FP16: [S | E4..E0 | M9..M0], bias=15
FP32: [S | E7..E0 | M22..M0], bias=127

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn
from .logic_gates import (ANDGate, ORGate, XORGate, NOTGate, MUXGate,
                          RippleCarryAdder)
from .fp32_mul import SpikeFP32Multiplier


class FP16ToFP32Converter(nn.Module):
    """FP16 -> FP32 转换器（100%纯SNN门电路）

    FP16: [S | E4..E0 | M9..M0], bias=15
    FP32: [S | E7..E0 | M22..M0], bias=127

    转换规则 (Normal):
    - sign: 直接复制
    - exp: FP32_exp = FP16_exp + 112 (bias差 = 127 - 15 = 112)
    - mant: 10位扩展为23位（低位补0）

    特殊情况:
    - Zero (E=0, M=0): → Zero FP32
    - Subnormal (E=0, M≠0): 归一化后转换（简化处理为0）
    - Inf (E=31, M=0): → Inf FP32 (E=255)
    - NaN (E=31, M≠0): → NaN FP32 (E=255, M保留)
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # 检测 FP16 E=0 - 单实例
        self.e_or = ORGate(neuron_template=nt)
        self.e_is_zero_not = NOTGate(neuron_template=nt)

        # 检测 E=31 (Inf/NaN) - 单实例
        self.e_all_ones_and = ANDGate(neuron_template=nt)

        # 检测 M≠0 - 单实例
        self.m_or = ORGate(neuron_template=nt)

        # 检测 subnormal (E=0 AND M≠0)
        self.is_subnorm_and = ANDGate(neuron_template=nt)

        # 检测 zero (E=0 AND M=0)
        self.not_m_nonzero = NOTGate(neuron_template=nt)
        self.is_zero_and = ANDGate(neuron_template=nt)

        # 检测 Inf (E=31 AND M=0)
        self.is_inf_and = ANDGate(neuron_template=nt)

        # 检测 NaN (E=31 AND M≠0)
        self.is_nan_and = ANDGate(neuron_template=nt)

        # 8位加法器: FP16_exp (5位扩展) + 112
        self.exp_adder = RippleCarryAdder(bits=8, neuron_template=nt)

        # 最终选择 MUX - 单实例
        self.final_exp_mux = MUXGate(neuron_template=nt)
        self.final_mant_mux = MUXGate(neuron_template=nt)

        # 零/Inf/NaN 处理 MUX - 单实例
        self.zero_mux = MUXGate(neuron_template=nt)
        self.inf_mux = MUXGate(neuron_template=nt)
        self.nan_mux = MUXGate(neuron_template=nt)

    def forward(self, fp16_pulse):
        """
        Args:
            fp16_pulse: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            fp32_pulse: [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        device = fp16_pulse.device
        batch_shape = fp16_pulse.shape[:-1]
        zeros = torch.zeros(batch_shape + (1,), device=device)
        ones = torch.ones(batch_shape + (1,), device=device)

        # 提取 FP16 各部分
        s = fp16_pulse[..., 0:1]
        e4 = fp16_pulse[..., 1:2]
        e3 = fp16_pulse[..., 2:3]
        e2 = fp16_pulse[..., 3:4]
        e1 = fp16_pulse[..., 4:5]
        e0 = fp16_pulse[..., 5:6]
        m = fp16_pulse[..., 6:16]  # [M9..M0]

        # 检测 E=0 (tree reduction)
        e_or_01 = self.e_or(e0, e1)
        e_or_23 = self.e_or(e2, e3)
        e_or_0123 = self.e_or(e_or_01, e_or_23)
        e_nonzero = self.e_or(e_or_0123, e4)
        e_is_zero = self.e_is_zero_not(e_nonzero)

        # 检测 E=31 (11111) (tree reduction)
        e_and_01 = self.e_all_ones_and(e0, e1)
        e_and_23 = self.e_all_ones_and(e2, e3)
        e_and_0123 = self.e_all_ones_and(e_and_01, e_and_23)
        e_is_max = self.e_all_ones_and(e_and_0123, e4)

        # 检测 M≠0 (OR tree for 10 bits)
        m_or_0 = self.m_or(m[..., 0:1], m[..., 1:2])
        m_or_1 = self.m_or(m[..., 2:3], m[..., 3:4])
        m_or_2 = self.m_or(m[..., 4:5], m[..., 5:6])
        m_or_3 = self.m_or(m[..., 6:7], m[..., 7:8])
        m_or_4 = self.m_or(m[..., 8:9], m[..., 9:10])
        m_or_01 = self.m_or(m_or_0, m_or_1)
        m_or_23 = self.m_or(m_or_2, m_or_3)
        m_or_0123 = self.m_or(m_or_01, m_or_23)
        m_nonzero = self.m_or(m_or_0123, m_or_4)

        # 特殊情况检测
        not_m_nz = self.not_m_nonzero(m_nonzero)
        is_zero = self.is_zero_and(e_is_zero, not_m_nz)
        is_subnormal = self.is_subnorm_and(e_is_zero, m_nonzero)
        is_inf = self.is_inf_and(e_is_max, not_m_nz)
        is_nan = self.is_nan_and(e_is_max, m_nonzero)

        # ===== Normal 路径 =====
        # FP16 exp 扩展到 8 位 (LSB first)
        fp16_exp_8bit_lsb = torch.cat([e0, e1, e2, e3, e4, zeros, zeros, zeros], dim=-1)

        # +112 = 0b01110000, LSB first: [0, 0, 0, 0, 1, 1, 1, 0]
        const_112_lsb = torch.cat([zeros, zeros, zeros, zeros, ones, ones, ones, zeros], dim=-1)

        # 加法 (LSB first)
        fp32_exp_raw_lsb, _ = self.exp_adder(fp16_exp_8bit_lsb, const_112_lsb)

        # 转回 MSB first
        fp32_exp_normal = torch.cat([
            fp32_exp_raw_lsb[..., 7:8],
            fp32_exp_raw_lsb[..., 6:7],
            fp32_exp_raw_lsb[..., 5:6],
            fp32_exp_raw_lsb[..., 4:5],
            fp32_exp_raw_lsb[..., 3:4],
            fp32_exp_raw_lsb[..., 2:3],
            fp32_exp_raw_lsb[..., 1:2],
            fp32_exp_raw_lsb[..., 0:1],
        ], dim=-1)

        # 尾数扩展: 10位 → 23位 (低位补0)
        fp32_mant_normal = torch.cat([m] + [zeros] * 13, dim=-1)

        # ===== Subnormal 路径 =====
        # 简化处理: subnormal 转为 0 (完整实现需要前导1检测和归一化)
        subnorm_exp = torch.cat([zeros] * 8, dim=-1)
        subnorm_mant = torch.cat([zeros] * 23, dim=-1)

        # 选择 normal vs subnormal (vectorized)
        is_subnormal_8 = is_subnormal.expand_as(subnorm_exp)
        exp_sel = self.final_exp_mux(is_subnormal_8, subnorm_exp, fp32_exp_normal)

        is_subnormal_23 = is_subnormal.expand_as(subnorm_mant)
        mant_sel = self.final_mant_mux(is_subnormal_23, subnorm_mant, fp32_mant_normal)

        # ===== 零处理 ===== (vectorized)
        zero_result = torch.cat([zeros] * 31, dim=-1)
        em_sel = torch.cat([exp_sel, mant_sel], dim=-1)
        is_zero_31 = is_zero.expand_as(em_sel)
        after_zero = self.zero_mux(is_zero_31, zero_result, em_sel)

        # ===== Inf 处理 ===== (vectorized)
        inf_exp = torch.cat([ones] * 8, dim=-1)
        inf_mant = torch.cat([zeros] * 23, dim=-1)
        inf_result = torch.cat([inf_exp, inf_mant], dim=-1)
        is_inf_31 = is_inf.expand_as(after_zero)
        after_inf = self.inf_mux(is_inf_31, inf_result, after_zero)

        # ===== NaN 处理 ===== (vectorized)
        nan_exp = torch.cat([ones] * 8, dim=-1)
        nan_mant = torch.cat([m] + [zeros] * 13, dim=-1)  # 保留原尾数
        nan_result = torch.cat([nan_exp, nan_mant], dim=-1)
        is_nan_31 = is_nan.expand_as(after_inf)
        after_nan = self.nan_mux(is_nan_31, nan_result, after_inf)

        # 组装 FP32
        fp32_pulse = torch.cat([s, after_nan], dim=-1)

        return fp32_pulse

    def reset(self):
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()


class SpikeFP16MulToFP32(nn.Module):
    """FP16 × FP16 → FP32 乘法器（纯SNN门电路）

    通过 FP16→FP32 转换 + FP32×FP32 乘法实现。
    输出完整精度的 FP32 结果。

    FP16: [S | E4..E0 | M9..M0], bias=15
    FP32: [S | E7..E0 | M22..M0], bias=127

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template

        # FP16 → FP32 转换器
        self.conv_a = FP16ToFP32Converter(neuron_template=nt)
        self.conv_b = FP16ToFP32Converter(neuron_template=nt)

        # FP32 乘法器
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

    def forward(self, A, B):
        """
        Args:
            A, B: [..., 16] FP16 脉冲 [S, E4..E0, M9..M0]
        Returns:
            [..., 32] FP32 脉冲 [S, E7..E0, M22..M0]
        """
        # 支持广播
        A, B = torch.broadcast_tensors(A, B)

        # FP16 → FP32
        A_fp32 = self.conv_a(A)
        B_fp32 = self.conv_b(B)

        # FP32 × FP32 → FP32
        result = self.mul(A_fp32, B_fp32)

        return result

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()
