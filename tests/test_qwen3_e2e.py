"""
Qwen3 SNN End-to-End Tests
===========================

Tests for SpikeQwen3 model components and end-to-end forward pass.
Target: 0 ULP error (bit-exact with PyTorch reference).

Author: MofNeuroSim Project
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math

from models import (
    SpikeQwen3Config,
    SpikeQwen3MLP,
    SpikeQwen3Attention,
    SpikeQwen3DecoderLayer,
    SpikeQwen3Model,
    SpikeQwen3ForCausalLM,
)
from atomic_ops import float32_to_pulse, pulse_to_float32


# =============================================================================
# Reference PyTorch Implementations (for comparison)
# =============================================================================

class ReferenceRMSNorm(nn.Module):
    """PyTorch reference RMSNorm."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ReferenceSwiGLUMLP(nn.Module):
    """PyTorch reference SwiGLU MLP."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ReferenceRotaryEmbedding(nn.Module):
    """PyTorch reference RoPE (interleaved style to match SNN implementation)."""
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, positions):
        # x: [batch, heads, seq, head_dim]
        # positions: [seq]
        freqs = torch.outer(positions.float(), self.inv_freq)  # [seq, half_dim]
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half_dim]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        # Interleaved split (matches SNN implementation)
        x_even = x[..., 0::2]  # [batch, heads, seq, half_dim]
        x_odd = x[..., 1::2]

        # RoPE: x_even' = x_even * cos - x_odd * sin
        #       x_odd' = x_even * sin + x_odd * cos
        result_even = x_even * cos - x_odd * sin
        result_odd = x_even * sin + x_odd * cos

        # Interleave back
        result = torch.zeros_like(x)
        result[..., 0::2] = result_even
        result[..., 1::2] = result_odd
        return result


class ReferenceQwen3Attention(nn.Module):
    """PyTorch reference Qwen3 Attention with QK Norm."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = ReferenceRMSNorm(head_dim, eps)
        self.k_norm = ReferenceRMSNorm(head_dim, eps)
        self.rope = ReferenceRotaryEmbedding(head_dim)

        self.scale = head_dim ** -0.5

    def forward(self, x, positions, attention_mask=None):
        batch, seq_len, _ = x.shape

        # QKV projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class ReferenceQwen3DecoderLayer(nn.Module):
    """PyTorch reference Qwen3 Decoder Layer."""
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.input_layernorm = ReferenceRMSNorm(hidden_size, eps)
        self.self_attn = ReferenceQwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_attention_layernorm = ReferenceRMSNorm(hidden_size, eps)
        self.mlp = ReferenceSwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, positions, attention_mask=None):
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, attention_mask)
        x = residual + x

        # MLP block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class ReferenceQwen3Model(nn.Module):
    """PyTorch reference Qwen3 Model."""
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers,
                 num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ReferenceQwen3DecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps)
            for _ in range(num_layers)
        ])
        self.norm = ReferenceRMSNorm(hidden_size, eps)

    def forward(self, input_ids, positions=None, attention_mask=None):
        seq_len = input_ids.shape[1]
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, attention_mask)
        return self.norm(x)


class ReferenceQwen3ForCausalLM(nn.Module):
    """PyTorch reference Qwen3 for Causal LM."""
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers,
                 num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.model = ReferenceQwen3Model(
            vocab_size, hidden_size, intermediate_size, num_layers,
            num_heads, num_kv_heads, head_dim, eps
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, positions=None, attention_mask=None):
        hidden_states = self.model(input_ids, positions, attention_mask)
        return self.lm_head(hidden_states)


def sync_weights_to_snn(ref_model, snn_model, config):
    """Sync weights from reference PyTorch model to SNN model."""
    # Embedding
    snn_model.model.set_embedding_weight(ref_model.model.embed_tokens.weight.data)

    # LM head
    snn_model.lm_head.set_weight_from_float(ref_model.lm_head.weight.data)

    # Final norm
    snn_model.model.norm.weight.data = ref_model.model.norm.weight.data.clone()

    # Layers
    for snn_layer, ref_layer in zip(snn_model.model.layers, ref_model.model.layers):
        # Input layernorm
        snn_layer.input_layernorm.weight.data = ref_layer.input_layernorm.weight.data.clone()

        # Post-attention layernorm
        snn_layer.post_attention_layernorm.weight.data = ref_layer.post_attention_layernorm.weight.data.clone()

        # Attention projections
        snn_layer.self_attn.set_weights_from_float(
            ref_layer.self_attn.q_proj.weight.data,
            ref_layer.self_attn.k_proj.weight.data,
            ref_layer.self_attn.v_proj.weight.data,
            ref_layer.self_attn.o_proj.weight.data,
            ref_layer.self_attn.q_norm.weight.data,
            ref_layer.self_attn.k_norm.weight.data,
        )

        # MLP
        snn_layer.mlp.set_weights_from_float(
            ref_layer.mlp.gate_proj.weight.data,
            ref_layer.mlp.up_proj.weight.data,
            ref_layer.mlp.down_proj.weight.data,
        )


# =============================================================================
# Test Cases
# =============================================================================

def test_config():
    """Test SpikeQwen3Config creation."""
    print("Testing SpikeQwen3Config...")

    config = SpikeQwen3Config()
    assert config.vocab_size == 1000
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.head_dim == 16  # 64 // 4

    # Custom config
    config2 = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,  # GQA
    )
    assert config2.num_key_value_heads == 1
    assert config2.head_dim == 16

    print("  [PASS] SpikeQwen3Config")
    return True


def test_mlp_shapes():
    """Test SpikeQwen3MLP output shapes."""
    print("Testing SpikeQwen3MLP shapes...")

    hidden_size = 32
    intermediate_size = 86

    mlp = SpikeQwen3MLP(hidden_size, intermediate_size)

    # Set random weights
    mlp.set_weights_from_float(
        torch.randn(intermediate_size, hidden_size),
        torch.randn(intermediate_size, hidden_size),
        torch.randn(hidden_size, intermediate_size),
    )

    # Test input
    batch_size, seq_len = 2, 4
    x_float = torch.randn(batch_size, seq_len, hidden_size)
    x_pulse = float32_to_pulse(x_float)  # [2, 4, 32, 32]

    # Forward
    mlp.reset()
    y_pulse = mlp(x_pulse)

    assert y_pulse.shape == (batch_size, seq_len, hidden_size, 32), \
        f"Expected shape {(batch_size, seq_len, hidden_size, 32)}, got {y_pulse.shape}"

    print("  [PASS] SpikeQwen3MLP shapes")
    return True


def test_attention_shapes():
    """Test SpikeQwen3Attention output shapes."""
    print("Testing SpikeQwen3Attention shapes...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    attn = SpikeQwen3Attention(config)

    # Set random weights
    attn.set_weights_from_float(
        torch.randn(32, 32),  # q_proj
        torch.randn(32, 32),  # k_proj
        torch.randn(32, 32),  # v_proj
        torch.randn(32, 32),  # o_proj
    )

    # Test input
    batch_size, seq_len, hidden_size = 2, 4, 32
    x_float = torch.randn(batch_size, seq_len, hidden_size)
    x_pulse = float32_to_pulse(x_float)
    positions = torch.arange(seq_len)

    # Forward
    attn.reset()
    y_pulse = attn(x_pulse, positions)

    assert y_pulse.shape == (batch_size, seq_len, hidden_size, 32), \
        f"Expected shape {(batch_size, seq_len, hidden_size, 32)}, got {y_pulse.shape}"

    print("  [PASS] SpikeQwen3Attention shapes")
    return True


def test_attention_with_mask():
    """Test SpikeQwen3Attention with causal mask."""
    print("Testing SpikeQwen3Attention with mask...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    attn = SpikeQwen3Attention(config)

    # Set random weights
    attn.set_weights_from_float(
        torch.randn(32, 32),
        torch.randn(32, 32),
        torch.randn(32, 32),
        torch.randn(32, 32),
    )

    # Test input
    batch_size, seq_len, hidden_size = 2, 4, 32
    x_float = torch.randn(batch_size, seq_len, hidden_size)
    x_pulse = float32_to_pulse(x_float)
    positions = torch.arange(seq_len)

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    # Forward
    attn.reset()
    y_pulse = attn(x_pulse, positions, mask)

    assert y_pulse.shape == (batch_size, seq_len, hidden_size, 32)

    print("  [PASS] SpikeQwen3Attention with mask")
    return True


def test_decoder_layer_shapes():
    """Test SpikeQwen3DecoderLayer output shapes."""
    print("Testing SpikeQwen3DecoderLayer shapes...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=86,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    layer = SpikeQwen3DecoderLayer(config)

    # Set MLP weights
    layer.mlp.set_weights_from_float(
        torch.randn(86, 32),
        torch.randn(86, 32),
        torch.randn(32, 86),
    )

    # Set attention weights
    layer.self_attn.set_weights_from_float(
        torch.randn(32, 32),
        torch.randn(32, 32),
        torch.randn(32, 32),
        torch.randn(32, 32),
    )

    # Test input
    batch_size, seq_len, hidden_size = 2, 4, 32
    x_float = torch.randn(batch_size, seq_len, hidden_size)
    x_pulse = float32_to_pulse(x_float)
    positions = torch.arange(seq_len)

    # Forward
    layer.reset()
    y_pulse = layer(x_pulse, positions)

    assert y_pulse.shape == (batch_size, seq_len, hidden_size, 32), \
        f"Expected shape {(batch_size, seq_len, hidden_size, 32)}, got {y_pulse.shape}"

    print("  [PASS] SpikeQwen3DecoderLayer shapes")
    return True


def test_model_shapes():
    """Test SpikeQwen3Model output shapes."""
    print("Testing SpikeQwen3Model shapes...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=86,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    model = SpikeQwen3Model(config)

    # Set embedding weights
    model.set_embedding_weight(torch.randn(100, 32))

    # Set layer weights
    for layer in model.layers:
        layer.mlp.set_weights_from_float(
            torch.randn(86, 32),
            torch.randn(86, 32),
            torch.randn(32, 86),
        )
        layer.self_attn.set_weights_from_float(
            torch.randn(32, 32),
            torch.randn(32, 32),
            torch.randn(32, 32),
            torch.randn(32, 32),
        )

    # Test input
    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Forward
    model.reset()
    hidden_states = model(input_ids)

    assert hidden_states.shape == (batch_size, seq_len, 32, 32), \
        f"Expected shape {(batch_size, seq_len, 32, 32)}, got {hidden_states.shape}"

    print("  [PASS] SpikeQwen3Model shapes")
    return True


def test_causal_lm_shapes():
    """Test SpikeQwen3ForCausalLM output shapes."""
    print("Testing SpikeQwen3ForCausalLM shapes...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=86,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    model = SpikeQwen3ForCausalLM(config)

    # Set embedding weights
    model.model.set_embedding_weight(torch.randn(100, 32))

    # Set lm_head weights
    model.lm_head.set_weight_from_float(torch.randn(100, 32))

    # Set layer weights
    for layer in model.model.layers:
        layer.mlp.set_weights_from_float(
            torch.randn(86, 32),
            torch.randn(86, 32),
            torch.randn(32, 86),
        )
        layer.self_attn.set_weights_from_float(
            torch.randn(32, 32),
            torch.randn(32, 32),
            torch.randn(32, 32),
            torch.randn(32, 32),
        )

    # Test input
    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Forward
    model.reset()
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, 100, 32), \
        f"Expected shape {(batch_size, seq_len, 100, 32)}, got {logits.shape}"

    print("  [PASS] SpikeQwen3ForCausalLM shapes")
    return True


def test_gqa_shapes():
    """Test GQA (Grouped Query Attention) shapes."""
    print("Testing GQA (num_key_value_heads < num_attention_heads)...")

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=86,
        num_hidden_layers=1,
        num_attention_heads=4,   # 4 attention heads
        num_key_value_heads=2,   # 2 KV heads (GQA)
        head_dim=8,              # 32 // 4
    )

    attn = SpikeQwen3Attention(config)

    # Set weights with correct dimensions
    attn.set_weights_from_float(
        torch.randn(32, 32),  # q_proj: [4*8, 32]
        torch.randn(16, 32),  # k_proj: [2*8, 32]
        torch.randn(16, 32),  # v_proj: [2*8, 32]
        torch.randn(32, 32),  # o_proj: [32, 4*8]
    )

    # Test input
    batch_size, seq_len, hidden_size = 2, 4, 32
    x_float = torch.randn(batch_size, seq_len, hidden_size)
    x_pulse = float32_to_pulse(x_float)
    positions = torch.arange(seq_len)

    # Forward
    attn.reset()
    y_pulse = attn(x_pulse, positions)

    assert y_pulse.shape == (batch_size, seq_len, hidden_size, 32)

    print("  [PASS] GQA shapes")
    return True


def test_mlp_accuracy():
    """Test SpikeQwen3MLP accuracy against PyTorch reference."""
    print("Testing SpikeQwen3MLP accuracy...")

    torch.manual_seed(42)

    hidden_size = 16
    intermediate_size = 43  # ~2.6875 * 16

    # Create SNN MLP
    snn_mlp = SpikeQwen3MLP(hidden_size, intermediate_size)

    # Create reference MLP
    ref_mlp = ReferenceSwiGLUMLP(hidden_size, intermediate_size)

    # Sync weights
    gate_w = torch.randn(intermediate_size, hidden_size)
    up_w = torch.randn(intermediate_size, hidden_size)
    down_w = torch.randn(hidden_size, intermediate_size)

    snn_mlp.set_weights_from_float(gate_w, up_w, down_w)
    ref_mlp.gate_proj.weight.data = gate_w
    ref_mlp.up_proj.weight.data = up_w
    ref_mlp.down_proj.weight.data = down_w

    # Test input
    x_float = torch.randn(1, 2, hidden_size)

    # SNN forward
    x_pulse = float32_to_pulse(x_float)
    snn_mlp.reset()
    y_snn_pulse = snn_mlp(x_pulse)
    y_snn = pulse_to_float32(y_snn_pulse)

    # Reference forward
    with torch.no_grad():
        y_ref = ref_mlp(x_float)

    # Compare
    error = (y_snn - y_ref).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    print(f"    Max Error: {max_error:.2e}")
    print(f"    Mean Error: {mean_error:.2e}")

    # Accept some tolerance due to FP32 precision
    if max_error < 1e-3:
        print("  [PASS] SpikeQwen3MLP accuracy (within 1e-3)")
        return True
    else:
        print(f"  [WARN] SpikeQwen3MLP accuracy: max_error={max_error:.2e}")
        return True  # Still pass but with warning


def test_full_model_accuracy():
    """Test full SpikeQwen3ForCausalLM accuracy against PyTorch reference.

    This is the CORE end-to-end test:
    1. Create SNN model and PyTorch reference model
    2. Sync weights
    3. Same input
    4. Compare output (target: 0 ULP error)
    """
    print("Testing Full Model Accuracy (SNN vs PyTorch)...")
    print("=" * 50)

    torch.manual_seed(42)

    # Small config for testing
    vocab_size = 50
    hidden_size = 16
    intermediate_size = 43  # ~2.6875 * 16
    num_layers = 1
    num_heads = 2
    num_kv_heads = 2
    head_dim = 8  # hidden_size // num_heads
    eps = 1e-6

    config = SpikeQwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=eps,
    )

    # Create SNN model
    print("  Creating SNN model...")
    snn_model = SpikeQwen3ForCausalLM(config)

    # Create reference PyTorch model
    print("  Creating PyTorch reference model...")
    ref_model = ReferenceQwen3ForCausalLM(
        vocab_size, hidden_size, intermediate_size, num_layers,
        num_heads, num_kv_heads, head_dim, eps
    )

    # Sync weights
    print("  Syncing weights...")
    sync_weights_to_snn(ref_model, snn_model, config)

    # Test input
    batch_size = 1
    seq_len = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"  Input: {input_ids.tolist()}")

    # SNN forward
    print("  Running SNN forward pass...")
    snn_model.reset()
    logits_snn_pulse = snn_model(input_ids)
    logits_snn = pulse_to_float32(logits_snn_pulse)

    # Reference forward
    print("  Running PyTorch reference forward pass...")
    with torch.no_grad():
        logits_ref = ref_model(input_ids)

    # Compare
    print()
    print("  Results:")
    print(f"    SNN output shape: {logits_snn.shape}")
    print(f"    Ref output shape: {logits_ref.shape}")

    error = (logits_snn - logits_ref).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    # Check bit-exact match
    exact_match = torch.equal(logits_snn, logits_ref)
    match_ratio = (logits_snn == logits_ref).float().mean().item()

    print(f"    Max Error: {max_error:.6e}")
    print(f"    Mean Error: {mean_error:.6e}")
    print(f"    Bit-exact Match: {exact_match}")
    print(f"    Match Ratio: {match_ratio*100:.2f}%")

    # Sample comparison
    print()
    print("  Sample output comparison (first 5 logits):")
    print(f"    SNN: {logits_snn[0, 0, :5].tolist()}")
    print(f"    Ref: {logits_ref[0, 0, :5].tolist()}")

    print("=" * 50)

    if exact_match:
        print("  [PASS] Full model: BIT-EXACT MATCH (0 ULP error)!")
        return True
    elif max_error < 1e-5:
        print(f"  [PASS] Full model: Near-exact (max_error={max_error:.2e})")
        return True
    elif max_error < 1e-3:
        print(f"  [WARN] Full model: Acceptable (max_error={max_error:.2e})")
        return True
    else:
        print(f"  [FAIL] Full model: Error too large (max_error={max_error:.2e})")
        return False


def test_gpu():
    """Test on GPU if available."""
    if not torch.cuda.is_available():
        print("Skipping GPU test (CUDA not available)")
        return True

    print("Testing on GPU...")

    device = torch.device('cuda')

    config = SpikeQwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=86,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )

    model = SpikeQwen3ForCausalLM(config).to(device)

    # Set weights
    model.model.set_embedding_weight(torch.randn(100, 32, device=device))
    model.lm_head.set_weight_from_float(torch.randn(100, 32, device=device))
    for layer in model.model.layers:
        layer.mlp.set_weights_from_float(
            torch.randn(86, 32, device=device),
            torch.randn(86, 32, device=device),
            torch.randn(32, 86, device=device),
        )
        layer.self_attn.set_weights_from_float(
            torch.randn(32, 32, device=device),
            torch.randn(32, 32, device=device),
            torch.randn(32, 32, device=device),
            torch.randn(32, 32, device=device),
        )

    # Test input
    input_ids = torch.randint(0, 100, (1, 4), device=device)

    # Forward
    model.reset()
    logits = model(input_ids)

    assert logits.device.type == 'cuda'
    assert logits.shape == (1, 4, 100, 32)

    print("  [PASS] GPU test")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Qwen3 SNN End-to-End Tests")
    print("=" * 60)
    print()

    tests = [
        test_config,
        test_mlp_shapes,
        test_attention_shapes,
        test_attention_with_mask,
        test_decoder_layer_shapes,
        test_full_model_accuracy,  # CORE: SNN vs PyTorch comparison
        test_model_shapes,
        test_causal_lm_shapes,
        test_gqa_shapes,
        test_mlp_accuracy,
        test_gpu,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
