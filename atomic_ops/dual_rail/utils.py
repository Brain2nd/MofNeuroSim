"""
Dual-Rail Adapters (Utils)
==========================

The "Bridge" between standard digital/single-rail logic and the True Dual-Rail Core.

This is the ONLY place where mathematical negation (1.0 - x) is permitted,
representing the input interface hardware.
"""

import torch
import torch.nn as nn

class SingleToDual(nn.Module):
    """
    Adapter: Converts Single-Rail input to Dual-Rail output.
    
    Logic:
        pos = x
        neg = 1.0 - x  <-- The only allowed subtraction
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input: Tensor [batch, ...]
        # Output: (pos, neg) pair of Tensors
        return x, 1.0 - x

class DualToSingle(nn.Module):
    """
    Adapter: Converts Dual-Rail input back to Single-Rail output.
    
    Logic:
        Basically returns the 'pos' line.
        Optionally, can enforce validity check (assert pos != neg).
    """
    def __init__(self):
        super().__init__()

    def forward(self, p, n):
        # We simply take the positive rail as the standard digital output
        return p

def to_dual_tensor(x):
    """Functional version of SingleToDual"""
    return x, 1.0 - x

def to_single_tensor(p, n):
    """Functional version of DualToSingle"""
    return p
