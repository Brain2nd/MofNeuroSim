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
| `SpikeFP16MulToFP32` | A × B → FP32 | 16→32-bit |
| `SpikeFP16Linear_MultiPrecision` | Wx + b | FP16/32 accum |
| `FP8ToFP16Converter` | FP8 → FP16 | Lossless |
| `FP16ToFP32Converter` | FP16 → FP32 | Lossless |

#### FP32
| Module | Operation | Notes |
|--------|-----------|-------|
| `SpikeFP32Adder` | A + B | Bit-exact |
| `SpikeFP32Multiplier` | A × B | Bit-exact |
| `SpikeFP32Divider` | A ÷ B | Newton-Raphson |
| `SpikeFP32Linear` | Wx + b | Bit-exact, FP32 accum |
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
pip install torch numpy
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

### FP16/FP32 Linear Layers

```python
from atomic_ops import (
    SpikeFP16Linear_MultiPrecision,
    SpikeFP32Linear,
    float16_to_pulse, pulse_to_float16,
    float32_to_pulse, pulse_to_float32
)

# FP16 Linear with FP32 accumulation (100% bit-exact)
linear_fp16 = SpikeFP16Linear_MultiPrecision(
    in_features=64,
    out_features=32,
    accum_precision='fp32'  # 'fp16' or 'fp32'
)
weights = torch.randn(32, 64, dtype=torch.float16)
linear_fp16.set_weight_from_float(weights)

x_pulse = float16_to_pulse(torch.randn(8, 64, dtype=torch.float16))
y_pulse = linear_fp16(x_pulse)  # [8, 32, 16]
y = pulse_to_float16(y_pulse)   # [8, 32]

# FP32 Linear (100% bit-exact)
linear_fp32 = SpikeFP32Linear(in_features=64, out_features=32)
weights = torch.randn(32, 64, dtype=torch.float32)
linear_fp32.set_weight_from_float(weights)

x_pulse = float32_to_pulse(torch.randn(8, 64))
y_pulse = linear_fp32(x_pulse)  # [8, 32, 32]
y = pulse_to_float32(y_pulse)   # [8, 32]
```

### Transformer Model (Qwen3 Architecture)

```python
from models import SpikeQwen3ForCausalLM, SpikeQwen3Config
from atomic_ops import pulse_to_float32

# Configure model
config = SpikeQwen3Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=172,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
)

# Create 100% pure SNN model
model = SpikeQwen3ForCausalLM(config).to('cuda')

# Load weights from HuggingFace model (optional)
# model.set_weights_from_hf_model(hf_model)

# Forward pass
input_ids = torch.randint(0, 1000, (1, 16), device='cuda')
model.reset()  # Clear neuron states
logits_pulse = model(input_ids)           # [1, 16, 1000, 32] pulse
logits = pulse_to_float32(logits_pulse)   # [1, 16, 1000] float
```

---

## 🔬 Physical Simulation with LIF Neurons

For hardware simulation of MOF chips, we provide IF/LIF neurons with **vectorized parameters**:

### Neuron Types

| Neuron | Equation | Use Case |
|--------|----------|----------|
| `SimpleIFNode` | V += I | Digital logic (bit-exact) |
| `SimpleLIFNode` | V = β×V + I | Physical simulation |
| `DynamicThresholdIFNode` | SAR ADC style | Float encoding |

### Vectorized Parameter Support

**Default behavior**: All gates use `_create_neuron` with `param_shape='auto'` (auto-vectorize on first forward), with bit-exact default values.

```python
from atomic_ops import SimpleIFNode, SimpleLIFNode

# Direct creation: threshold_shape=None (scalar broadcast)
if_neuron = SimpleIFNode(v_threshold=1.0)

# Enable auto-vectorization (used by _create_neuron in gates)
if_neuron = SimpleIFNode(v_threshold=1.0, threshold_shape='auto')

# Enable trainable parameters
if_neuron = SimpleIFNode(
    v_threshold=1.0,
    threshold_shape='auto',      # Auto-detect from input (default)
    trainable_threshold=True     # Learnable
)

# LIF with vectorized beta and threshold
lif_neuron = SimpleLIFNode(
    beta=0.9,                    # Membrane leakage factor
    v_threshold=1.0,
    param_shape='auto',          # Auto-detect shape (default)
    trainable_beta=True,         # Learnable leakage
    trainable_threshold=True     # Learnable threshold
)
```

### neuron_template Mechanism

All gates support `neuron_template` for unified IF/LIF switching:

```python
from atomic_ops import ANDGate, SimpleLIFNode, SpikeFP32Adder

# Default: IF neurons (ideal digital logic)
and_gate = ANDGate()
adder = SpikeFP32Adder()

# Physical simulation: LIF neurons
lif_template = SimpleLIFNode(beta=0.9)
and_gate_lif = ANDGate(neuron_template=lif_template)
adder_lif = SpikeFP32Adder(neuron_template=lif_template)
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

### FP8 Input/Output Linear (`SpikeFP8Linear_MultiPrecision`)
| Accumulation | Alignment | Notes |
|--------------|-----------|-------|
| FP8 | ~38% | Per-step rounding |
| FP16 | **100%** | FP32 accum → FP16 → FP8 |
| **FP32** | **100%** | **Bit-exact match** |

### FP16 Input/Output Linear (`SpikeFP16Linear_MultiPrecision`)
| Accumulation | Alignment | Notes |
|--------------|-----------|-------|
| FP16 | ~95% | Per-step rounding |
| **FP32** | **100%** | **Bit-exact match** |

### FP32 Input/Output Linear (`SpikeFP32Linear`)
| Accumulation | Alignment | Notes |
|--------------|-----------|-------|
| **FP32** | **100%** | **Bit-exact match** |

---

## 🧪 Testing

We provide comprehensive tests for all components:

```bash
# ★ Core test suite (recommended)
python tests/test_suite.py                    # Run all core tests
python tests/test_suite.py --only logic_gates # Test specific category
python tests/test_suite.py --only linear      # Test Linear layers

# ★ Qwen3 Transformer end-to-end test
python tests/test_qwen3_e2e_full.py           # Full model validation
# Output: tests/logs/qwen3_e2e_{timestamp}.json

# Core arithmetic
python tests/test_fp8_mul.py          # FP8 multiplication
python tests/test_fp32_adder.py       # FP32 addition
python tests/test_fp64_div.py         # FP64 division

# Activation functions
python tests/test_fp32_gelu.py        # GELU
python tests/test_fp32_tanh.py        # Tanh
python tests/test_fp32_layernorm.py   # LayerNorm

# End-to-end
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
│   │
│   ├── core/                     # Foundation layer
│   │   ├── neurons.py            # IF/LIF neuron implementations
│   │   ├── logic_gates.py        # Basic gates (AND, OR, NOT, XOR, MUX)
│   │   ├── vec_logic_gates.py    # Vectorized parallel gates
│   │   ├── dynamic_if.py         # Dynamic threshold IF (for encoding)
│   │   └── sign_bit.py           # Sign detection
│   │
│   ├── encoding/                 # Float ↔ Pulse conversion
│   │   ├── converters.py         # float32_to_pulse, pulse_to_float32, etc.
│   │   ├── floating_point.py     # FP8 encoder
│   │   └── pulse_decoder.py      # Multi-precision decoders
│   │
│   ├── arithmetic/               # Floating-point arithmetic
│   │   ├── fp8/                  # FP8 ops (mul, adder)
│   │   ├── fp16/                 # FP16 ops (adder, mul, converters)
│   │   ├── fp32/                 # FP32 ops (adder, mul, div, sqrt, recip)
│   │   └── fp64/                 # FP64 ops (adder, mul, div, sqrt)
│   │
│   ├── activation/               # Activation functions
│   │   ├── fp8/                  # FP8 ReLU
│   │   ├── fp32/                 # FP32 (sigmoid, tanh, gelu, silu, softmax, exp)
│   │   └── fp64/                 # FP64 exp
│   │
│   ├── normalization/            # Normalization layers
│   │   └── fp32/                 # LayerNorm, RMSNorm
│   │
│   ├── linear/                   # Linear layers
│   │   ├── fp8/                  # FP8 linear (multi-precision accum)
│   │   ├── fp16/                 # FP16 linear
│   │   └── fp32/                 # FP32 linear, embedding
│   │
│   ├── attention/                # Attention mechanisms
│   │   ├── rope.py               # Rotary Position Embedding
│   │   └── attention.py          # Multi-head attention
│   │
│   ├── trigonometry/             # Trigonometric functions
│   │   ├── fp32/                 # FP32 sin/cos
│   │   └── fp64/                 # FP64 sin/cos
│   │
│   └── dual_rail/                # Dual-rail logic (experimental)
│
├── models/                        # Complete SNN models
│   ├── __init__.py
│   ├── qwen3_config.py           # Qwen3 configuration
│   ├── qwen3_mlp.py              # SwiGLU MLP (100% SNN)
│   ├── qwen3_attention.py        # Attention with QK-Norm, RoPE, GQA
│   ├── qwen3_decoder_layer.py    # Transformer decoder layer
│   └── qwen3_model.py            # SpikeQwen3ForCausalLM
│
├── tests/                         # Comprehensive test suite
│   ├── test_suite.py             # ★ Core test suite
│   ├── test_qwen3_e2e_full.py    # ★ Qwen3 end-to-end test
│   ├── logs/                     # Test output logs (JSON)
│   └── ...                       # Component tests
│
├── CLAUDE.md                      # Development guidelines
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

- [PyTorch](https://pytorch.org/) - Deep learning framework

---

<p align="center">
  <b>MofNeuroSim</b> — Bridging Digital Logic and Neuromorphic Computing
</p>
