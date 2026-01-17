"""
Qwen3 SNN 真正的端到端测试
==========================

完整模型 SNN vs PyTorch 对比，包含：
1. GPU 执行
2. 逐层进度打印
3. 细粒度误差记录保存到 JSON 文件

运行方式:
    conda activate SNN
    python tests/test_qwen3_e2e_full.py

输出文件:
    tests/logs/qwen3_e2e_{timestamp}.json
"""
import torch
import torch.nn as nn
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import float32_to_pulse, pulse_to_float32
from models import (
    SpikeQwen3Config,
    SpikeQwen3ForCausalLM,
)


# ==============================================================================
# PyTorch 参考实现
# ==============================================================================

class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ReferenceSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x) * torch.sigmoid(self.gate_proj(x))  # SiLU
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ReferenceRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, positions):
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        result_even = x_even * cos - x_odd * sin
        result_odd = x_even * sin + x_odd * cos
        result = torch.zeros_like(x)
        result[..., 0::2] = result_even
        result[..., 1::2] = result_odd
        return result


class ReferenceQwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
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
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class ReferenceQwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.input_layernorm = ReferenceRMSNorm(hidden_size, eps)
        self.self_attn = ReferenceQwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_attention_layernorm = ReferenceRMSNorm(hidden_size, eps)
        self.mlp = ReferenceSwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, positions, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, attention_mask)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ReferenceQwen3ForCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers,
                 num_heads, num_kv_heads, head_dim, eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ReferenceQwen3DecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps)
            for _ in range(num_layers)
        ])
        self.norm = ReferenceRMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, positions=None, attention_mask=None):
        seq_len = input_ids.shape[1]
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, attention_mask)
        x = self.norm(x)
        return self.lm_head(x)


# ==============================================================================
# 权重同步
# ==============================================================================

def sync_weights_to_snn(ref_model, snn_model, config, device):
    """同步权重从 PyTorch 参考模型到 SNN 模型"""
    print("  [权重同步] Embedding...")
    snn_model.model.set_embedding_weight(ref_model.embed_tokens.weight.data.to(device))

    print("  [权重同步] LM Head...")
    snn_model.lm_head.set_weight_from_float(ref_model.lm_head.weight.data.to(device))

    print("  [权重同步] Final Norm...")
    snn_model.model.norm.weight.data = ref_model.norm.weight.data.clone().to(device)

    for i, (snn_layer, ref_layer) in enumerate(zip(snn_model.model.layers, ref_model.layers)):
        print(f"  [权重同步] Layer {i}...")

        # LayerNorm weights
        snn_layer.input_layernorm.weight.data = ref_layer.input_layernorm.weight.data.clone().to(device)
        snn_layer.post_attention_layernorm.weight.data = ref_layer.post_attention_layernorm.weight.data.clone().to(device)

        # Attention weights
        snn_layer.self_attn.set_weights_from_float(
            ref_layer.self_attn.q_proj.weight.data.to(device),
            ref_layer.self_attn.k_proj.weight.data.to(device),
            ref_layer.self_attn.v_proj.weight.data.to(device),
            ref_layer.self_attn.o_proj.weight.data.to(device),
            ref_layer.self_attn.q_norm.weight.data.to(device),
            ref_layer.self_attn.k_norm.weight.data.to(device),
        )

        # MLP weights
        snn_layer.mlp.set_weights_from_float(
            ref_layer.mlp.gate_proj.weight.data.to(device),
            ref_layer.mlp.up_proj.weight.data.to(device),
            ref_layer.mlp.down_proj.weight.data.to(device),
        )


# ==============================================================================
# 误差记录工具
# ==============================================================================

def record_error(name, snn_pulse, ref_float, log_data):
    """记录单个节点的误差"""
    snn_float = pulse_to_float32(snn_pulse)
    error = (snn_float - ref_float).abs()
    max_err = error.max().item()
    mean_err = error.mean().item()
    match_1e6 = (error < 1e-6).float().mean().item() * 100
    match_1e5 = (error < 1e-5).float().mean().item() * 100

    log_data['layer_errors'].append({
        'name': name,
        'max_error': max_err,
        'mean_error': mean_err,
        'match_rate_1e6': match_1e6,
        'match_rate_1e5': match_1e5,
    })

    print(f"        误差: max={max_err:.2e} mean={mean_err:.2e} <1e-6:{match_1e6:5.1f}%")
    return snn_float  # 返回解码后的浮点，用于下一步参考计算


# ==============================================================================
# 带进度和误差记录的前向传播
# ==============================================================================

def snn_forward_with_progress(snn_model, ref_model, input_ids, positions, attention_mask, log_data):
    """SNN 前向传播，带进度打印和逐层误差记录"""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    print("\n  [SNN Forward] 开始...")
    start_time = time.time()

    # 1. Embedding
    t0 = time.time()
    print("    [1/N] Embedding...", end=" ", flush=True)
    snn_emb = snn_model.model.embed_tokens(input_ids)
    with torch.no_grad():
        ref_emb = ref_model.embed_tokens(input_ids)
    print(f"完成 ({time.time()-t0:.2f}s)")
    log_data['embedding_time'] = time.time() - t0
    snn_emb_float = record_error("embedding", snn_emb, ref_emb, log_data)

    # 2. Decoder Layers
    snn_hidden = snn_emb
    ref_hidden = ref_emb  # 使用参考模型的输出继续，以隔离每层误差

    for i, (snn_layer, ref_layer) in enumerate(zip(snn_model.model.layers, ref_model.layers)):
        layer_start = time.time()
        print(f"    [Layer {i}] 开始...", flush=True)

        # 2.1 Input LayerNorm
        t0 = time.time()
        print(f"      - input_layernorm...", end=" ", flush=True)
        snn_residual = snn_hidden
        ref_residual = ref_hidden
        snn_normed = snn_layer.input_layernorm(snn_hidden)
        with torch.no_grad():
            ref_normed = ref_layer.input_layernorm(ref_hidden)
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_input_layernorm", snn_normed, ref_normed, log_data)

        # 2.2 Self Attention
        t0 = time.time()
        print(f"      - self_attn...", end=" ", flush=True)
        snn_attn = snn_layer.self_attn(snn_normed, positions, attention_mask)
        with torch.no_grad():
            ref_attn = ref_layer.self_attn(ref_normed, positions, attention_mask)
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_self_attn", snn_attn, ref_attn, log_data)

        # 2.3 Residual Add
        t0 = time.time()
        print(f"      - residual_add_attn...", end=" ", flush=True)
        snn_hidden = snn_layer.residual_add_attn(snn_residual, snn_attn)
        with torch.no_grad():
            ref_hidden = ref_residual + ref_attn
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_residual_attn", snn_hidden, ref_hidden, log_data)

        # 2.4 Post Attention LayerNorm
        t0 = time.time()
        print(f"      - post_attention_layernorm...", end=" ", flush=True)
        snn_residual = snn_hidden
        ref_residual = ref_hidden
        snn_normed = snn_layer.post_attention_layernorm(snn_hidden)
        with torch.no_grad():
            ref_normed = ref_layer.post_attention_layernorm(ref_hidden)
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_post_attn_layernorm", snn_normed, ref_normed, log_data)

        # 2.5 MLP
        t0 = time.time()
        print(f"      - mlp...", end=" ", flush=True)
        snn_mlp = snn_layer.mlp(snn_normed)
        with torch.no_grad():
            ref_mlp = ref_layer.mlp(ref_normed)
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_mlp", snn_mlp, ref_mlp, log_data)

        # 2.6 Residual Add
        t0 = time.time()
        print(f"      - residual_add_mlp...", end=" ", flush=True)
        snn_hidden = snn_layer.residual_add_mlp(snn_residual, snn_mlp)
        with torch.no_grad():
            ref_hidden = ref_residual + ref_mlp
        print(f"({time.time()-t0:.2f}s)")
        record_error(f"L{i}_residual_mlp", snn_hidden, ref_hidden, log_data)

        layer_time = time.time() - layer_start
        print(f"    [Layer {i}] 完成 (总计 {layer_time:.2f}s)")
        log_data[f'layer_{i}_time'] = layer_time

    # 3. Final Norm
    t0 = time.time()
    print(f"    [Final] norm...", end=" ", flush=True)
    snn_final = snn_model.model.norm(snn_hidden)
    with torch.no_grad():
        ref_final = ref_model.norm(ref_hidden)
    print(f"({time.time()-t0:.2f}s)")
    record_error("final_norm", snn_final, ref_final, log_data)

    # 4. LM Head
    t0 = time.time()
    print(f"    [Final] lm_head...", end=" ", flush=True)
    snn_logits = snn_model.lm_head(snn_final)
    with torch.no_grad():
        ref_logits = ref_model.lm_head(ref_final)
    print(f"({time.time()-t0:.2f}s)")
    record_error("lm_head", snn_logits, ref_logits, log_data)

    total_time = time.time() - start_time
    print(f"\n  [SNN Forward] 完成，总耗时: {total_time:.2f}s")
    log_data['total_forward_time'] = total_time

    return snn_logits, ref_logits


# ==============================================================================
# 主测试函数
# ==============================================================================

def run_e2e_test():
    """运行端到端测试"""

    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'qwen3_e2e_{timestamp}.json')

    log_data = {
        'timestamp': timestamp,
        'config': {},
        'results': {},
        'layer_errors': [],
    }

    print("=" * 70)
    print("Qwen3 SNN 端到端测试 (完整模型)")
    print("=" * 70)

    # 检测设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n[Device] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[Device] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\n[Device] CPU (警告: 会很慢!)")

    log_data['device'] = str(device)

    # 配置
    torch.manual_seed(42)

    vocab_size = 100
    hidden_size = 32
    intermediate_size = 86
    num_layers = 1
    num_heads = 2
    num_kv_heads = 2
    head_dim = 16
    eps = 1e-6

    batch_size = 1
    seq_len = 4

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

    log_data['config'] = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }

    print(f"\n[Config] vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}")
    print(f"[Config] batch={batch_size}, seq_len={seq_len}")

    # 创建模型
    print("\n[1/5] 创建 SNN 模型...")
    snn_model = SpikeQwen3ForCausalLM(config).to(device)
    print(f"      SNN 模型已移动到 {device}")

    print("\n[2/5] 创建 PyTorch 参考模型...")
    ref_model = ReferenceQwen3ForCausalLM(
        vocab_size, hidden_size, intermediate_size, num_layers,
        num_heads, num_kv_heads, head_dim, eps
    ).to(device)
    print(f"      参考模型已移动到 {device}")

    # 同步权重
    print("\n[3/5] 同步权重...")
    sync_weights_to_snn(ref_model, snn_model, config, device)

    # 准备输入 (包含边界值 + 随机值)
    print("\n[4/5] 准备输入...")
    # 边界值: 0 (最小token), vocab_size-1 (最大token)
    # 随机值: 中间随机token
    boundary_tokens = [0, vocab_size - 1]  # 边界值
    random_tokens = torch.randint(1, vocab_size - 1, (seq_len - 2,), device=device).tolist()
    input_ids = torch.tensor([boundary_tokens + random_tokens], device=device)

    positions = torch.arange(seq_len, device=device)
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    print(f"      input_ids: {input_ids.tolist()} (含边界值 0, {vocab_size-1})")
    print(f"      input_ids.device: {input_ids.device}")

    # SNN 前向传播（同时与参考模型逐层对比）
    print("\n[5/5] 前向传播对比...")
    snn_model.reset()

    logits_snn_pulse, logits_ref = snn_forward_with_progress(
        snn_model, ref_model, input_ids, positions, attention_mask, log_data
    )
    logits_snn = pulse_to_float32(logits_snn_pulse)

    # 比较结果
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)

    print(f"\n  SNN 输出形状: {logits_snn.shape}")
    print(f"  Ref 输出形状: {logits_ref.shape}")
    print(f"  SNN 输出设备: {logits_snn.device}")
    print(f"  Ref 输出设备: {logits_ref.device}")

    # 计算误差
    error = (logits_snn - logits_ref).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    match_rate_1e5 = (error < 1e-5).float().mean().item() * 100
    match_rate_1e4 = (error < 1e-4).float().mean().item() * 100

    # 相关性
    corr = torch.corrcoef(torch.stack([logits_snn.flatten(), logits_ref.flatten()]))[0, 1].item()

    print(f"\n  最大误差: {max_error:.6e}")
    print(f"  平均误差: {mean_error:.6e}")
    print(f"  匹配率 (<1e-5): {match_rate_1e5:.2f}%")
    print(f"  匹配率 (<1e-4): {match_rate_1e4:.2f}%")
    print(f"  相关性: {corr:.6f}")

    log_data['results'] = {
        'max_error': max_error,
        'mean_error': mean_error,
        'match_rate_1e5': match_rate_1e5,
        'match_rate_1e4': match_rate_1e4,
        'correlation': corr,
    }

    # 样本对比
    print(f"\n  样本对比 (第一个 token 的前 10 个 logits):")
    print(f"    SNN: {logits_snn[0, 0, :10].tolist()}")
    print(f"    Ref: {logits_ref[0, 0, :10].tolist()}")

    # 误差累积趋势分析
    print("\n" + "=" * 70)
    print("误差累积趋势分析")
    print("=" * 70)
    print(f"\n  {'节点':<25s} | {'最大误差':>12s} | {'平均误差':>12s} | {'<1e-6':>7s}")
    print("-" * 65)
    for err_record in log_data['layer_errors']:
        name = err_record['name']
        max_e = err_record['max_error']
        mean_e = err_record['mean_error']
        m1e6 = err_record['match_rate_1e6']
        print(f"  {name:<25s} | {max_e:>12.2e} | {mean_e:>12.2e} | {m1e6:>6.1f}%")

    # 计算层间累积比例
    layer_max_errors = [e['max_error'] for e in log_data['layer_errors'] if e['name'].startswith('L')]
    if len(layer_max_errors) >= 2:
        first_half = layer_max_errors[:len(layer_max_errors)//2]
        second_half = layer_max_errors[len(layer_max_errors)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        ratio = avg_second / avg_first if avg_first > 0 else 0
        log_data['error_accumulation'] = {
            'first_half_avg': avg_first,
            'second_half_avg': avg_second,
            'ratio': ratio,
            'accumulated': ratio > 1.5
        }
        print(f"\n  前半段平均最大误差: {avg_first:.2e}")
        print(f"  后半段平均最大误差: {avg_second:.2e}")
        print(f"  累积比例: {ratio:.2f}x")
        if ratio > 2:
            print("  [!] 警告: 误差在明显累积")
        elif ratio > 1.5:
            print("  [~] 注意: 误差有轻微累积")
        else:
            print("  [OK] 误差未累积，在机器精度范围内")

    # 保存日志
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"\n[Log] 结果已保存到: {log_file}")

    # 判定
    print("\n" + "=" * 70)
    if match_rate_1e4 >= 99:
        print("✓ 测试通过: 端到端误差在可接受范围内")
        return True
    else:
        print("✗ 测试失败: 误差过大")
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
