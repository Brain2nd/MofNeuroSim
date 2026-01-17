"""
FP32 Linear 层 - 100%纯SNN门电路实现
====================================

FP32 输入输出的全连接层，支持 FP32/FP64 中间累加精度。

架构概述
--------

```
输入脉冲 X[batch, in, 32]    权重脉冲 W[out, in, 32]
        │                           │
        └─────────┬─────────────────┘
                  │
        [FP32×FP32→FP32 乘法器]
                  │
                  ▼
         ┌────────────────────┐
         │   累加器选择       │
         │  ┌──────────────┐  │
         │  │ FP32 加法器  │  │  ← accum_precision='fp32' → 输出FP32
         │  ├──────────────┤  │
         │  │ FP64 加法器  │  │  ← accum_precision='fp64' → 转换 → 输出FP32
         │  └──────────────┘  │
         └────────────────────┘
                  │
                  ▼
         输出脉冲 Y[batch, out, 32]
```

累加精度对比
-----------

| 精度 | 与 PyTorch 对齐 | 相对速度 | 内部位宽 |
|------|-----------------|----------|----------|
| fp32 | 100%            | 快       | 32 位    |
| fp64 | 100%            | 慢       | 64 位    |

**推荐**: 使用 `accum_precision='fp32'` 以获得与 PyTorch 完全一致的结果。

使用示例
--------
```python
# 创建层
linear = SpikeFP32Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    accum_precision='fp32'  # 100% 对齐 PyTorch
)

# 设置权重
linear.set_weight_from_float(weight_tensor)

# 前向传播 (纯脉冲域)
y_pulse = linear(x_pulse)  # [batch, 32, 32]
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn

from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier
from atomic_ops.arithmetic.fp32.fp32_adder import SpikeFP32Adder
from atomic_ops.arithmetic.fp64.fp64_adder import SpikeFP64Adder
from atomic_ops.arithmetic.fp64.fp64_components import FP32ToFP64Converter, FP64ToFP32Converter


class SpikeFP32Linear_MultiPrecision(nn.Module):
    """FP32 Linear 层 - 支持不同中间累加精度

    Y = X @ W^T，其中 X 和 W 都是 FP32 脉冲编码

    输入输出始终为 FP32，中间累加精度可选择以平衡精度和性能。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        accum_precision: 中间累加精度，'fp32' / 'fp64'
            - 'fp32': FP32累加 → FP32输出（与PyTorch一致，推荐）
            - 'fp64': FP64累加 → FP32输出（更高精度）
        neuron_template: 神经元模板，None 使用默认 IF 神经元

    架构:
        输入[FP32] → FP32×FP32→FP32乘法 → 累加[accum_precision] → 输出[FP32]
    """
    def __init__(self, in_features, out_features, accum_precision='fp32', neuron_template=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.accum_precision = accum_precision
        nt = neuron_template

        # FP32 乘法器 (逐元素乘法，共享)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

        if accum_precision == 'fp64':
            # FP64 累加模式
            # FP32 → FP64 转换器
            self.fp32_to_fp64 = FP32ToFP64Converter(neuron_template=nt)
            # FP64 累加器 (单实例，动态扩展机制支持复用)
            self.fp64_adder = SpikeFP64Adder(neuron_template=nt)
            # FP64 → FP32 输出转换器
            self.output_converter = FP64ToFP32Converter(neuron_template=nt)
        else:
            # FP32 累加模式 (单实例，动态扩展机制支持复用)
            self.fp32_adder = SpikeFP32Adder(neuron_template=nt)

        self.register_buffer('weight_pulse', None)

    def set_weight_from_float(self, weight_float):
        """将 float 权重转换为 FP32 脉冲

        Args:
            weight_float: [out_features, in_features] 权重张量
        """
        from atomic_ops.encoding.converters import float32_to_pulse
        assert weight_float.shape == (self.out_features, self.in_features)

        # 编码权重 (使用位级转换，确保精确)
        weight_pulse = float32_to_pulse(weight_float, device=weight_float.device)
        self.weight_pulse = weight_pulse

    def forward(self, x):
        """
        Args:
            x: [..., in_features, 32] 输入 FP32 脉冲
        Returns:
            [..., out_features, 32] 输出 FP32 脉冲（所有模式输出都是FP32）
        """
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float"

        # x: [..., in_features, 32] -> [..., 1, in_features, 32]
        # weight: [out_features, in_features, 32]
        x_expanded = x.unsqueeze(-3)

        # FP32 × FP32 → FP32 逐元素乘法
        # 广播: [..., 1, in_features, 32] × [out_features, in_features, 32]
        #     → [..., out_features, in_features, 32]
        products = self.mul(x_expanded, self.weight_pulse)

        # 累加
        if self.in_features == 1:
            if self.accum_precision == 'fp64':
                return products.squeeze(-2)  # 单元素无需累加，直接返回FP32
            return products.squeeze(-2)

        if self.accum_precision == 'fp64':
            return self._fp64_accumulate(products)
        else:
            return self._fp32_accumulate(products)

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：FP32累加 → 输出FP32脉冲"""
        # 第一个乘积
        acc = products_fp32[..., 0, :]

        # 逐个累加（单实例复用）
        for i in range(1, self.in_features):
            acc = self.fp32_adder(acc, products_fp32[..., i, :])

        return acc

    def _fp64_accumulate(self, products_fp32):
        """FP64 累加：FP32→FP64 → FP64累加 → FP64→FP32 → 输出FP32脉冲"""
        # 转换所有乘积到 FP64
        products_fp64 = self.fp32_to_fp64(products_fp32)

        # 第一个乘积
        acc = products_fp64[..., 0, :]

        # 逐个累加（单实例复用）
        for i in range(1, self.in_features):
            acc = self.fp64_adder(acc, products_fp64[..., i, :])

        # 转换回 FP32 输出
        return self.output_converter(acc)

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()
