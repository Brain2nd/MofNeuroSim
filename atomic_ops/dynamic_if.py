"""
动态阈值 IF 神经元 (Dynamic Threshold IF Node)
=============================================

实现 SAR ADC 风格的二进制扫描编码器核心组件。

原理:
- 阈值从 2^(N-1) 递减到 2^(-NT)
- 使用软复位保留残差，实现逐位二进制扫描

作者: MofNeuroSim Project
"""

# 从统一的神经元模块导入
from .neurons import DynamicThresholdIFNode

# 保持向后兼容：直接导出 DynamicThresholdIFNode
__all__ = ['DynamicThresholdIFNode']
