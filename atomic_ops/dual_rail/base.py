"""
Dual-Rail Base Definitions
==========================

Defines the fundamental types and protocols for True Dual-Rail SNN logic.

Protocol:
    Signal is a tuple (pos, neg) representing a logical value.
    - Logic 1: pos=1, neg=0
    - Logic 0: pos=0, neg=1
    - Null:    pos=0, neg=0 (no data)
    - Invalid: pos=1, neg=1 (error state)

Pure SNN Constraints:
    - All weights must be positive (excitatory).
    - No `1.0 - x` operations allowed in logic gates.
    - NOT operation must be a wire swap.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from atomic_ops.core.neurons import SimpleLIFNode

class DualRailBlock(nn.Module):
    """Base class for all Dual-Rail components."""
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError

    def reset(self):
        # Recursively reset all child modules
        for module in self.modules():
            if hasattr(module, 'reset') and module is not self:
                module.reset()

def create_neuron(template, threshold, v_reset=None, param_shape='auto',
                  trainable_threshold=True, trainable_beta=True, beta=None):
    """从模板创建指定阈值的神经元

    Args:
        template: 神经元模板，None 则创建默认 SimpleLIFNode
        threshold: 目标阈值 (float 或 Tensor)
        v_reset: 复位电压 (None=软复位, 数值=硬复位)
        param_shape: 参数形状 ('auto'=懒加载, None=标量, tuple=指定形状)
        trainable_threshold: 阈值是否可训练（默认True）
        trainable_beta: 泄漏率是否可训练（默认True）
        beta: 泄漏因子 (None=DEFAULT_BETA)

    Returns:
        配置好的神经元实例
    """
    if template is None:
        # 默认使用 SimpleLIFNode，启用可训练参数和懒加载
        return SimpleLIFNode(
            beta=beta,  # None 使用 DEFAULT_BETA (1.0 - 1e-7)
            v_threshold=threshold,
            v_reset=v_reset,
            trainable_beta=trainable_beta,
            trainable_threshold=trainable_threshold,
            param_shape=param_shape
        )
    else:
        node = deepcopy(template)
        # 使用 property setter - 支持标量和张量
        node.v_threshold = threshold
        if hasattr(node, 'v_reset'):
            node.v_reset = v_reset
        # 传播 param_shape 用于延迟初始化
        if hasattr(node, 'param_shape') and param_shape is not None:
            node.param_shape = param_shape
        # SimpleIFNode 兼容性（使用 threshold_shape）
        if hasattr(node, 'threshold_shape') and param_shape is not None:
            node.threshold_shape = param_shape
        return node
