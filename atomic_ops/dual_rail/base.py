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
from ..neurons import SimpleIFNode

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

def create_neuron(template, threshold, v_reset=None):
    """Helper to create a neuron from a template.

    Args:
        template: 神经元模板，None 则创建默认 IF 神经元
        threshold: 目标阈值
        v_reset: 复位电压 (None=软复位, 数值=硬复位)
                 默认为 None (软复位)，保留残差用于跨时间步实验

    Returns:
        配置好的神经元实例
    """
    if template is None:
        return SimpleIFNode(v_threshold=threshold, v_reset=v_reset)
    else:
        node = deepcopy(template)
        node.v_threshold = threshold
        if hasattr(node, 'v_reset'):
            node.v_reset = v_reset
        return node
