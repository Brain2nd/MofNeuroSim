"""
FP32 Softmax - 100%纯SNN门电路实现
====================================

Softmax(x_i) = exp(x_i) / sum(exp(x_j))

使用已有的Exp、Adder和Divider组合实现。

作者: MofNeuroSim Project
"""
import torch
import torch.nn as nn

from atomic_ops.core.training_mode import TrainingMode
from atomic_ops.core.accumulator import create_accumulator
from .fp32_exp import SpikeFP32Exp
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp32.fp32_div import SpikeFP32Divider
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter


class SpikeFP32Softmax(nn.Module):
    """FP32 Softmax - 100%纯SNN门电路实现

    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    输入: x [..., N, 32] FP32脉冲，其中N是softmax维度
    输出: Softmax(x) [..., N, 32] FP32脉冲

    Args:
        neuron_template: 神经元模板，None 使用默认 IF 神经元
        training_mode: 训练模式 (None/TrainingMode.STE/TrainingMode.TEMPORAL)（梯度流过）
        accumulator_mode: 累加模式 ('sequential' 或 'parallel')
        accum_precision: 中间累加精度 ('fp32' 或 'fp64')
    """
    def __init__(self, neuron_template=None, training_mode=None, accumulator_mode='sequential', accum_precision='fp32'):
        super().__init__()
        self.training_mode = TrainingMode.validate(training_mode)
        self.accum_precision = accum_precision
        nt = neuron_template

        if accum_precision == 'fp64':
            from atomic_ops.activation.fp64.fp64_exp import SpikeFP64Exp
            from atomic_ops.arithmetic.fp64.fp64_div import SpikeFP64Divider
            
            self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
            self.fp64_exp = SpikeFP64Exp(neuron_template=nt)
            self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
            self.fp64_accumulator = create_accumulator(self.fp64_adder, mode=accumulator_mode)
            self.fp64_divider = SpikeFP64Divider(neuron_template=nt)
            self.fp64_to_fp32 = FP64ToFP32Converter(neuron_template=nt)
        else:
            self.exp = SpikeFP32Exp(neuron_template=nt)
            self.adder = SpikeFP32Adder(neuron_template=nt)
            self.accumulator = create_accumulator(self.adder, mode=accumulator_mode)
            self.divider = SpikeFP32Divider(neuron_template=nt)

    def forward(self, x):
        """
        x: [..., N, 32] FP32脉冲
        Returns: Softmax(x) [..., N, 32] FP32脉冲
        """
        # SNN 前向 (纯门电路)
        with torch.no_grad():
            if self.accum_precision == 'fp64':
                # FP64 Path
                x_fp64 = self.fp32_to_fp64(x)
                exp_x = self.fp64_exp(x_fp64)
                sum_exp = self.fp64_accumulator.reduce(exp_x, dim=-2)
                sum_exp_expanded = sum_exp.unsqueeze(-2).expand_as(exp_x)
                out_fp64 = self.fp64_divider(exp_x, sum_exp_expanded)
                out_pulse = self.fp64_to_fp32(out_fp64)
            else:
                # FP32 Path (Standard)
                exp_x = self.exp(x)
                sum_exp = self.accumulator.reduce(exp_x, dim=-2)
                sum_exp_expanded = sum_exp.unsqueeze(-2).expand_as(exp_x)
                out_pulse = self.divider(exp_x, sum_exp_expanded)

        # 如果训练模式，用 STE 包装以支持梯度
        if TrainingMode.is_ste(self.training_mode) and self.training:
            from atomic_ops.core.ste import ste_softmax
            # pulse 格式的 softmax 维度是 -2 (N 维度)
            return ste_softmax(x, out_pulse, dim=-2)

        return out_pulse

    def reset(self):
        if self.accum_precision == 'fp64':
            self.fp32_to_fp64.reset()
            self.fp64_exp.reset()
            self.fp64_adder.reset()
            self.fp64_accumulator.reset()
            self.fp64_divider.reset()
            self.fp64_to_fp32.reset()
        else:
            self.exp.reset()
            self.adder.reset()
            self.accumulator.reset()
            self.divider.reset()

