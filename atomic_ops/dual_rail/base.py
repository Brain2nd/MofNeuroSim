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
from spikingjelly.activation_based import neuron, surrogate

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

def create_neuron(template, threshold, v_reset=0.0):
    """Helper to create a neuron from a template."""
    if template is None:
        return neuron.IFNode(
            v_threshold=threshold, 
            v_reset=v_reset,
            surrogate_function=surrogate.ATan()
        )
    else:
        node = deepcopy(template)
        node.v_threshold = threshold
        node.v_reset = v_reset
        return node
