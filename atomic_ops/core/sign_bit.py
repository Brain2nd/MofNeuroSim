"""
符号位检测神经元 (Sign Bit Node)
================================

使用抑制性突触检测输入的符号位。

原理:
- 抑制性突触 (权重=-1)
- 负数输入 → 正电流 → 发放 (1)
- 正数输入 → 负电流 → 不发放 (0)

作者: MofNeuroSim Project
"""

# 从统一的神经元模块导入
from .neurons import SignBitNode

# 保持向后兼容：直接导出 SignBitNode
__all__ = ['SignBitNode']
