# MofNeuroSim

[![arXiv](https://img.shields.io/badge/arXiv-2512.07724-b31b1b.svg)](https://arxiv.org/abs/2512.07724)
![Status](https://img.shields.io/badge/Status-Preprint-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**A foundational framework and hardware simulator for Spiking Neural Networks (SNNs) based on MOF chips.**

> ⚠️ **Note:** This repository contains the code for the **preliminary version** of our research. For the methodology and initial results, please refer to our preprint: [**arXiv:2512.07724**](https://arxiv.org/abs/2512.07724). The final version of the paper is currently under work.

---

## 🎯 Overview

MofNeuroSim is a **100% pure spiking neural network** implementation of IEEE floating-point arithmetic. Unlike traditional neuromorphic computing frameworks that focus on approximate computing, we achieve **bit-exact** results (0 ULP error) compared to PyTorch references.

### Core Philosophy

All computations are performed **entirely within the pulse domain**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Pure SNN Computation Domain                  │
│                                                                  │
│  Float Input ──→ [Encoder] ──→ [SNN Gates] ──→ [Decoder] ──→ Float Output
│                      │              │              │             │
│                      ▼              ▼              ▼             │
│                   Pulses         IF/LIF        Pulses            │
│                   (0/1)         Neurons        (0/1)             │
└─────────────────────────────────────────────────────────────────┘
```

### What Makes This Different?

| Traditional Approach | MofNeuroSim |
|---------------------|-------------|
| Uses Python arithmetic (`+`, `-`, `*`) | ❌ Forbidden |
| Approximates floating-point | Bit-exact IEEE-754 |
| Rate/temporal coding | Direct binary pulse representation |
| GPU tensor operations | Pure IF neuron gate circuits |

---

## 🏗️ Architecture

### Pure SNN Principles

We strictly follow these rules:

```python
# ❌ FORBIDDEN - Traditional arithmetic
result = 1 - x                    # Use NOTGate(x)
result = a * b                    # Use ANDGate(a, b)
result = a + b - 2*a*b           # Use XORGate(a, b)

# ✅ REQUIRED - Pure SNN gates
result = not_gate(x)
result = and_gate(a, b)
result = xor_gate(a, b)
```

### Logic Gate Implementation

Each gate is implemented using Integrate-and-Fire neurons with carefully designed thresholds:

| Gate | Formula | Threshold | IF Neurons |
|------|---------|-----------|------------|
| AND  | `H(A + B - 1.5)` | 1.5 | 1 |
| OR   | `H(A + B - 0.5)` | 0.5 | 1 |
| NOT  | `H(1 - A - 0.5)` | 0.5 | 1 (inhibitory) |
| XOR  | `OR(A,B) - AND(A,B)` | - | 3 |
| MUX  | `OR(AND(A,S), AND(B,NOT(S)))` | - | 5 |

### Hierarchical Structure

```
Level 0: IF Neurons (threshold + reset)
    │
Level 1: Logic Gates (AND, OR, NOT, XOR, MUX)
    │
Level 2: Arithmetic Units (Adder, Multiplier)
    │
Level 3: Floating-Point Operators (FP Add, FP Mul, FP Div)
    │
Level 4: Neural Network Layers (Linear, Activation, Normalization)
    │
Level 5: Complete Models (MLP, Transformer components)
```

---

## 📦 Components

### Boundary Components (Float ↔ Pulse)

| Component | Direction | Shape Transform |
|-----------|-----------|-----------------|
| `PulseFloatingPointEncoder` | Float → Pulse | `[...]` → `[..., bits]` |
| `PulseFloatingPointDecoder` | Pulse → Float | `[..., bits]` → `[...]` |
| `PulseFP16Decoder` | Pulse → Float16 | `[..., 16]` → `[...]` |
| `PulseFP32Decoder` | Pulse → Float32 | `[..., 32]` → `[...]` |

### Vectorized Gate Library

Optimized gates for parallel bit-level operations:

| Component | Description | Supports |
|-----------|-------------|----------|
| `VecAND`, `VecOR`, `VecNOT`, `VecXOR` | Basic gates | Arbitrary tensor shapes |
| `VecMUX` | 2-to-1 multiplexer | Bit selection |
| `VecORTree`, `VecANDTree` | Reduction trees | Variable input count |
| `VecHalfAdder`, `VecFullAdder` | Binary adders | Bit-parallel |
| `VecAdder`, `VecSubtractor` | N-bit arithmetic | Configurable width |
| `VecComparator` | Magnitude compare | Multi-bit operands |

### Floating-Point Arithmetic

#### FP8 (E4M3 Format)
| Module | Operation | Precision |
|--------|-----------|-----------|
| `SpikeFP8Multiplier` | A × B | 8-bit |
| `SpikeFP8Adder_Spatial` | A + B | 8-bit |
| `SpikeFP8ReLU` | max(0, x) | 8-bit |
| `SpikeFP8Linear_Fast` | Wx + b (FP8 accum) | 8-bit |
| `SpikeFP8Linear_MultiPrecision` | Wx + b | FP8/16/32 accum |

#### FP16
| Module | Operation | Precision |
|--------|-----------|-----------|
| `SpikeFP16Adder` | A + B | 16-bit |
| `SpikeFP8ToFP16Converter` | FP8 → FP16 | Lossless |

#### FP32
| Module | Operation | Notes |
|--------|-----------|-------|
| `SpikeFP32Adder` | A + B | Bit-exact |
| `SpikeFP32Multiplier` | A × B | Bit-exact |
| `SpikeFP32Divider` | A ÷ B | Newton-Raphson |
| `SpikeFP32Sqrt` | √x | Newton-Raphson |
| `SpikeFP32Exp` | e^x | Taylor series |
| `SpikeFP32Reciprocal` | 1/x | Newton-Raphson |
| `SpikeFP32Sigmoid` | σ(x) | Bit-exact |
| `SpikeFP32Tanh` | tanh(x) | Bit-exact |
| `SpikeFP32GELU` | GELU(x) | Approximation |
| `SpikeFP32SiLU` | x·σ(x) | Bit-exact |
| `SpikeFP32Softmax` | softmax(x) | Full implementation |
| `SpikeFP32LayerNorm` | LayerNorm(x) | With learnable params |
| `SpikeFP32RMSNorm` | RMSNorm(x) | Efficient variant |
| `SpikeFP32Embedding` | Lookup table | MUX-tree based |

#### FP64
| Module | Operation | Notes |
|--------|-----------|-------|
| `SpikeFP64Adder` | A + B | Bit-exact |
| `SpikeFP64Multiplier` | A × B | Bit-exact |
| `SpikeFP64Divider` | A ÷ B | Newton-Raphson |
| `SpikeFP64Sqrt` | √x | Newton-Raphson |
| `SpikeFP64Exp` | e^x | Extended precision |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Brain2nd/MofNeuroSim.git
cd MofNeuroSim

# Install dependencies
pip install torch spikingjelly numpy
```

### Basic Usage

```python
import torch
from atomic_ops import (
    PulseFloatingPointEncoder,
    PulseFloatingPointDecoder,
    SpikeFP8Multiplier,
    float_to_fp8_bits,
    fp8_bits_to_float
)

# Create encoder/decoder
encoder = PulseFloatingPointEncoder()
decoder = PulseFloatingPointDecoder()

# Encode float to pulse
x = torch.tensor([1.5, -2.0, 0.5])
x_pulse = encoder(x)  # Shape: [3, 8]

# Create FP8 multiplier
mul = SpikeFP8Multiplier()

# Multiply in pulse domain
a_pulse = encoder(torch.tensor([2.0]))
b_pulse = encoder(torch.tensor([3.0]))
result_pulse = mul(a_pulse, b_pulse)

# Decode back to float
result = decoder(result_pulse)
print(f"2.0 × 3.0 = {result.item()}")  # 6.0
```

### FP32 Operations

```python
from atomic_ops import (
    SpikeFP32Adder,
    SpikeFP32Multiplier,
    SpikeFP32Sigmoid,
    float32_to_pulse,
    pulse_to_float32
)

# Create FP32 adder
adder = SpikeFP32Adder()

# Convert to pulse representation
a = float32_to_pulse(3.14159, device='cuda')
b = float32_to_pulse(2.71828, device='cuda')

# Pure SNN addition
result = adder(a, b)

# Convert back
value = pulse_to_float32(result)
print(f"π + e = {value}")  # Bit-exact result
```

### Neural Network Layer

```python
from atomic_ops import SpikeFP8Linear_MultiPrecision

# Create linear layer with FP32 accumulation (100% bit-exact)
linear = SpikeFP8Linear_MultiPrecision(
    in_features=784,
    out_features=128,
    accum_precision='fp32'  # 'fp8', 'fp16', or 'fp32'
)

# Load pretrained weights
weights = torch.randn(128, 784) * 0.01
linear.set_weight_from_float(weights, encoder)

# Forward pass (pure SNN)
x_pulse = encoder(torch.randn(32, 784))
y_pulse = linear(x_pulse)  # [32, 128, 8]
y = decoder(y_pulse)       # [32, 128]
```

---

## 🔬 Physical Simulation with LIF Neurons

For hardware simulation of MOF chips, we provide LIF (Leaky Integrate-and-Fire) variants:

```python
from atomic_ops.logic_gates_lif import (
    SimpleLIFNode,
    # LIF-based gates with membrane leakage
)

# LIF neuron with configurable dynamics
lif = SimpleLIFNode(
    tau=10.0,    # Membrane time constant
    v_reset=0.0  # Reset voltage
)
```

### Robustness Testing

```bash
python tests/test_robustness.py
```

Tests include:
- **β scan**: Membrane leakage (0.01 - 1.0)
- **σ scan**: Input noise (Gaussian, 0.0 - 1.0)
- **Temperature effects**: Threshold variation

---

## 📊 Precision Alignment

Comparison with PyTorch `nn.Linear`:

| Accumulation | Alignment | Notes |
|--------------|-----------|-------|
| FP8 | ~50% | Per-step rounding |
| FP16 | ~95% | Reduced error accumulation |
| **FP32** | **100%** | **Bit-exact match** |

---

## 🧪 Testing

We provide comprehensive tests for all components:

```bash
# Core arithmetic
python tests/test_fp8_mul.py          # FP8 multiplication
python tests/test_fp32_adder.py       # FP32 addition
python tests/test_fp64_div.py         # FP64 division

# Activation functions
python tests/test_fp32_gelu.py        # GELU
python tests/test_fp32_tanh.py        # Tanh
python tests/test_fp32_layernorm.py   # LayerNorm

# End-to-end
python tests/test_mnist_e2e.py        # MNIST inference
python tests/test_all_precision_alignment.py  # Full precision test

# Physical simulation
python tests/test_robustness.py       # LIF robustness
```

---

## 📁 Project Structure

```
MofNeuroSim/
├── atomic_ops/                    # Core SNN components
│   ├── __init__.py               # Module exports
│   ├── converters.py             # Float ↔ Pulse utilities
│   │
│   ├── logic_gates.py            # IF-based logic gates
│   ├── logic_gates_lif.py        # LIF-based gates (physical sim)
│   ├── vec_logic_gates.py        # Vectorized parallel gates
│   │
│   ├── floating_point.py         # FP8 encoder
│   ├── pulse_decoder.py          # Multi-precision decoders
│   │
│   ├── fp8_mul.py                # FP8 multiplier
│   ├── fp8_adder_spatial.py      # FP8 adder
│   ├── fp8_linear_fast.py        # FP8 linear (fast)
│   ├── fp8_linear_multi.py       # FP8 linear (multi-precision)
│   ├── fp8_relu.py               # FP8/32/64 ReLU
│   │
│   ├── fp16_adder.py             # FP16 adder
│   ├── fp16_components.py        # FP8↔FP16 converter
│   │
│   ├── fp32_adder.py             # FP32 adder
│   ├── fp32_mul.py               # FP32 multiplier
│   ├── fp32_div.py               # FP32 divider
│   ├── fp32_sqrt.py              # FP32 square root
│   ├── fp32_exp.py               # FP32 exponential
│   ├── fp32_recip.py             # FP32 reciprocal
│   ├── fp32_sigmoid.py           # FP32 sigmoid
│   ├── fp32_tanh.py              # FP32 tanh
│   ├── fp32_gelu.py              # FP32 GELU
│   ├── fp32_silu.py              # FP32 SiLU
│   ├── fp32_softmax.py           # FP32 softmax
│   ├── fp32_layernorm.py         # FP32 layer normalization
│   ├── fp32_rmsnorm.py           # FP32 RMS normalization
│   ├── fp32_embedding.py         # FP32 embedding layer
│   │
│   ├── fp64_adder.py             # FP64 adder
│   ├── fp64_mul.py               # FP64 multiplier
│   ├── fp64_div.py               # FP64 divider
│   ├── fp64_sqrt.py              # FP64 square root
│   ├── fp64_exp.py               # FP64 exponential
│   │
│   ├── sign_bit.py               # Sign detection neuron
│   └── dynamic_if.py             # Dynamic threshold IF
│
├── models/                        # SNN inference models
│   └── mnist_snn_infer.py        # MNIST MLP example
│
├── tests/                         # Comprehensive test suite
│   ├── test_logic_gates.py       # Gate correctness
│   ├── test_vec_logic_gates.py   # Vectorized gates
│   ├── test_fp8_*.py             # FP8 tests
│   ├── test_fp16_*.py            # FP16 tests
│   ├── test_fp32_*.py            # FP32 tests
│   ├── test_fp64_*.py            # FP64 tests
│   ├── test_robustness.py        # Physical simulation
│   └── test_mnist_e2e.py         # End-to-end inference
│
└── README.md
```

---

## 📜 FP8 E4M3 Format Reference

```
Bit layout: [S | E3 E2 E1 E0 | M2 M1 M0]
             ↑   \_________/   \_______/
           Sign    Exponent     Mantissa
                   (4 bits)     (3 bits)

Bias: 7

Normal numbers (E ≠ 0):
    value = (-1)^S × 2^(E-7) × (1 + M/8)

Subnormal numbers (E = 0):
    value = (-1)^S × 2^(-6) × (M/8)

Special values:
    0x00 = +0.0
    0x80 = -0.0
    0x7F = NaN (E=15, M=7)
    0xFF = -NaN
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mofneurosim2024,
  title     = {MofNeuroSim: A Foundational Framework for Spiking Neural 
               Networks Based on MOF Chips},
  author    = {...},
  journal   = {arXiv preprint arXiv:2512.07724},
  year      = {2024},
  url       = {https://arxiv.org/abs/2512.07724}
}
```

---

## 🤝 Contributing

This is a research project in active development. Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure all tests pass (`python tests/test_suite.py`)
4. Commit your changes
5. Push to the branch
6. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - SNN simulation framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

<p align="center">
  <b>MofNeuroSim</b> — Bridging Digital Logic and Neuromorphic Computing
</p>
