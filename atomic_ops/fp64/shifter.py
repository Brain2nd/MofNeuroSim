import torch
import torch.nn as nn
from ..logic_gates import MUXGate, ORGate, ANDGate, NOTGate

class BarrelShifterRight64(nn.Module):
    """64-bit 右移位器 (用于FP64尾数对齐)
    支持移位量 0-63
    输入: data [..., 64], shift [..., 6] (二进制脉冲)
    输出: shifted [..., 64]
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 MUXGate，支持动态扩展
        self.mux = MUXGate(neuron_template=nt)

    def forward(self, data, shift):
        # data: [..., 64]
        # shift: [..., 6] [b5, b4, b3, b2, b1, b0]
        # b5 (32), b4 (16), ..., b0 (1)

        batch_shape = data.shape[:-1]
        device = data.device
        current = data
        zeros_64 = torch.zeros(batch_shape + (64,), device=device)

        for i in range(6):
            s_bit = shift[..., 5-i : 6-i]  # b0, b1, ... b5
            shift_amount = 1 << i

            # 向量化构造 shifted 数据
            # Right shift: 高位填0，数据向高索引移动
            if shift_amount < 64:
                shifted = torch.cat([zeros_64[..., :shift_amount], current[..., :-shift_amount]], dim=-1)
            else:
                shifted = zeros_64

            # 向量化 MUX
            s_bit_64 = s_bit.expand(batch_shape + (64,))
            current = self.mux(s_bit_64, shifted, current)

        return current

    def reset(self):
        self.mux.reset()

class BarrelShifterLeft64(nn.Module):
    """64-bit 左移位器 (用于FP64结果归一化)
    支持移位量 0-63
    输入: data [..., 64], shift [..., 6]
    输出: shifted [..., 64]
    """
    def __init__(self, neuron_template=None):
        super().__init__()
        nt = neuron_template
        # 单实例 MUXGate，支持动态扩展
        self.mux = MUXGate(neuron_template=nt)

    def forward(self, data, shift):
        # shift: [b5, b4, b3, b2, b1, b0]

        batch_shape = data.shape[:-1]
        device = data.device
        current = data
        zeros_64 = torch.zeros(batch_shape + (64,), device=device)

        for i in range(6):
            s_bit = shift[..., 5-i : 6-i]
            shift_amount = 1 << i

            # 向量化构造 shifted 数据
            # Left shift: 低位填0，数据向低索引移动
            if shift_amount < 64:
                shifted = torch.cat([current[..., shift_amount:], zeros_64[..., :shift_amount]], dim=-1)
            else:
                shifted = zeros_64

            # 向量化 MUX
            s_bit_64 = s_bit.expand(batch_shape + (64,))
            current = self.mux(s_bit_64, shifted, current)

        return current

    def reset(self):
        self.mux.reset()
