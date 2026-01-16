"""
FP32 Linear 层 - 100%纯SNN门电路实现
===================================

FP32 输入输出的全连接层，内部使用 FP32 精度累加。

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
         │   FP32 累加器      │
         └────────────────────┘
                  │
                  ▼
         输出脉冲 Y[batch, out, 32]
```

数学公式
--------

```
Y[b, o] = Σ(X[b, i] × W[o, i]) for i in [0, in_features)

其中:
- X, W 为 FP32 格式 [S | E7..E0 | M22..M0]
- 乘法: FP32×FP32→FP32 (SpikeFP32Multiplier)
- 累加: FP32 + FP32 → FP32 (SpikeFP32Adder)
- 输出: FP32 脉冲
```

使用示例
--------
```python
# 创建层
linear = SpikeFP32Linear(in_features=64, out_features=32)

# 设置权重
linear.set_weight_from_float(weight_tensor, encoder)

# 前向传播 (纯脉冲域)
y_pulse = linear(x_pulse)  # [batch, 32, 32]
```

作者: MofNeuroSim Project
许可: MIT License
"""
import torch
import torch.nn as nn

from .fp32_mul import SpikeFP32Multiplier
from .fp32_adder import SpikeFP32Adder


class SpikeFP32Linear(nn.Module):
    """FP32 Linear 层 - 100%纯SNN实现

    Y = X @ W^T，其中 X 和 W 都是 FP32 脉冲编码

    输入输出都是 FP32，内部使用 FP32 精度累加（最高精度）。

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        neuron_template: 神经元模板，None 使用默认 IF 神经元

    架构:
        输入[FP32] → FP32×FP32→FP32乘法 → FP32累加 → 输出[FP32]
    """
    def __init__(self, in_features, out_features, neuron_template=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        nt = neuron_template

        # FP32 乘法器 (逐元素乘法，共享)
        self.mul = SpikeFP32Multiplier(neuron_template=nt)

        # FP32 累加器：每次累加使用独立实例（避免串行复用）
        self.fp32_adders = nn.ModuleList([
            SpikeFP32Adder(neuron_template=nt) for _ in range(max(1, in_features - 1))
        ])

        self.register_buffer('weight_pulse', None)

    def set_weight_from_float(self, weight_float):
        """将 float 权重转换为 FP32 脉冲

        Args:
            weight_float: [out_features, in_features] 权重张量
        """
        from .converters import float32_to_pulse
        assert weight_float.shape == (self.out_features, self.in_features)

        # 编码权重 (使用位级转换，确保精确)
        weight_pulse = float32_to_pulse(weight_float, device=weight_float.device)
        self.weight_pulse = weight_pulse

    def forward(self, x):
        """
        Args:
            x: [..., in_features, 32] 输入 FP32 脉冲
        Returns:
            [..., out_features, 32] 输出 FP32 脉冲
        """
        assert self.weight_pulse is not None, "需要先调用 set_weight_from_float"

        # x: [..., in_features, 32] -> [..., 1, in_features, 32]
        # weight: [out_features, in_features, 32]
        x_expanded = x.unsqueeze(-3)

        # FP32 × FP32 → FP32 逐元素乘法
        # 广播: [..., 1, in_features, 32] × [out_features, in_features, 32]
        #     → [..., out_features, in_features, 32]
        products = self.mul(x_expanded, self.weight_pulse)

        # FP32 累加
        if self.in_features == 1:
            return products.squeeze(-2)

        return self._fp32_accumulate(products)

    def _fp32_accumulate(self, products_fp32):
        """FP32 累加：顺序累加所有乘积

        Args:
            products_fp32: [..., out_features, in_features, 32] FP32 乘积
        Returns:
            [..., out_features, 32] 累加结果
        """
        # 第一个乘积
        acc = products_fp32[..., 0, :]

        # 逐个累加（使用独立实例）
        for i in range(1, self.in_features):
            acc = self.fp32_adders[i-1](acc, products_fp32[..., i, :])

        return acc

    def reset_all(self):
        """递归reset所有子模块"""
        for module in self.modules():
            if module is not self and hasattr(module, 'reset'):
                module.reset()

    def reset(self):
        """向后兼容"""
        self.reset_all()
