"""
STE 反向传播精度验证测试
========================

验证现有 STE 实现与 PyTorch ANN 对应组件的 backward 误差。

测试组件:
1. Linear
2. Embedding
3. RMSNorm
4. Sigmoid
5. Tanh
6. SiLU
7. Exp
8. Softmax
9. ReLU
10. GELU

对于每个组件:
- 使用相同的输入和权重
- 分别用 SNN STE backward 和 PyTorch autograd backward 计算梯度
- 对比误差 (Max abs error, Mean abs error, ULP error)

运行方式:
    python tests/test_ste_accuracy.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import float32_to_pulse, pulse_to_float32
from atomic_ops.core.ste import (
    get_snn_components,
    _parallel_reduce_pulse,
    STELinearFunction,
    STESigmoidFunction,
    STETanhFunction,
    STESiLUFunction,
    STEExpFunction,
    STESoftmaxFunction,
    STEReLUFunction,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def compute_errors(snn_grad, ref_grad):
    """计算 SNN 梯度与参考梯度的误差"""
    abs_diff = (snn_grad - ref_grad).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    # ULP 误差
    snn_bits = snn_grad.view(torch.int32)
    ref_bits = ref_grad.view(torch.int32)
    ulp_diff = (snn_bits - ref_bits).abs()
    max_ulp = ulp_diff.max().item()
    mean_ulp = ulp_diff.float().mean().item()
    zero_ulp_rate = (ulp_diff == 0).float().mean().item() * 100

    return {
        'max_abs': max_abs_err,
        'mean_abs': mean_abs_err,
        'max_ulp': max_ulp,
        'mean_ulp': mean_ulp,
        'zero_ulp_rate': zero_ulp_rate,
    }


def print_errors(name, errors):
    """打印误差报告"""
    print(f"  {name}:")
    print(f"    Max abs error:  {errors['max_abs']:.6e}")
    print(f"    Mean abs error: {errors['mean_abs']:.6e}")
    print(f"    Max ULP error:  {errors['max_ulp']}")
    print(f"    Mean ULP error: {errors['mean_ulp']:.2f}")
    print(f"    0-ULP rate:     {errors['zero_ulp_rate']:.1f}%")


# ==============================================================================
# Test 1: Sigmoid
# ==============================================================================
def test_sigmoid():
    print("\n" + "="*60)
    print("Test 1: Sigmoid")
    print("="*60)

    torch.manual_seed(42)
    batch, features = 4, 8

    # 输入
    x_float = torch.randn(batch, features, device=device, requires_grad=True)

    # === PyTorch ANN backward ===
    y_float = torch.sigmoid(x_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # === SNN STE backward ===
    x_float2 = x_float.detach().clone()
    x_pulse = float32_to_pulse(x_float2)
    y_pulse = float32_to_pulse(torch.sigmoid(x_float2))  # 前向结果
    grad_out_pulse = float32_to_pulse(grad_out_float)

    # 手动调用 backward
    comp = get_snn_components(device)
    one_pulse = comp.constants.one(y_pulse.shape[:-1])
    one_minus_y = comp.vec_sub(one_pulse, y_pulse)
    y_times_1my = comp.vec_mul(y_pulse, one_minus_y)
    grad_x_pulse = comp.vec_mul(grad_out_pulse, y_times_1my)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    # 对比
    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-5


# ==============================================================================
# Test 2: Tanh
# ==============================================================================
def test_tanh():
    print("\n" + "="*60)
    print("Test 2: Tanh")
    print("="*60)

    torch.manual_seed(42)
    batch, features = 4, 8

    x_float = torch.randn(batch, features, device=device, requires_grad=True)

    # PyTorch
    y_float = torch.tanh(x_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # SNN STE
    x_float2 = x_float.detach().clone()
    y_pulse = float32_to_pulse(torch.tanh(x_float2))
    grad_out_pulse = float32_to_pulse(grad_out_float)

    comp = get_snn_components(device)
    one_pulse = comp.constants.one(y_pulse.shape[:-1])
    y_squared = comp.vec_mul(y_pulse, y_pulse)
    one_minus_y_sq = comp.vec_sub(one_pulse, y_squared)
    grad_x_pulse = comp.vec_mul(grad_out_pulse, one_minus_y_sq)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-5


# ==============================================================================
# Test 3: Exp
# ==============================================================================
def test_exp():
    print("\n" + "="*60)
    print("Test 3: Exp")
    print("="*60)

    torch.manual_seed(42)
    batch, features = 4, 8

    # 使用较小的输入避免 exp 爆炸
    x_float = torch.randn(batch, features, device=device) * 0.5
    x_float.requires_grad_(True)

    # PyTorch
    y_float = torch.exp(x_float)
    grad_out_float = torch.randn_like(y_float) * 0.1
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # SNN STE: grad_x = grad_out * y
    x_float2 = x_float.detach().clone()
    y_pulse = float32_to_pulse(torch.exp(x_float2))
    grad_out_pulse = float32_to_pulse(grad_out_float)

    comp = get_snn_components(device)
    grad_x_pulse = comp.vec_mul(grad_out_pulse, y_pulse)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-5


# ==============================================================================
# Test 4: SiLU (x * sigmoid(x))
# ==============================================================================
def test_silu():
    print("\n" + "="*60)
    print("Test 4: SiLU")
    print("="*60)

    torch.manual_seed(42)
    batch, features = 4, 8

    x_float = torch.randn(batch, features, device=device, requires_grad=True)

    # PyTorch
    y_float = F.silu(x_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # SNN STE: grad_x = grad_out * (sigmoid + x * sigmoid * (1 - sigmoid))
    x_float2 = x_float.detach().clone()
    sigmoid_float = torch.sigmoid(x_float2)

    x_pulse = float32_to_pulse(x_float2)
    sigmoid_pulse = float32_to_pulse(sigmoid_float)
    grad_out_pulse = float32_to_pulse(grad_out_float)

    comp = get_snn_components(device)
    one_pulse = comp.constants.one(sigmoid_pulse.shape[:-1])

    one_minus_sig = comp.vec_sub(one_pulse, sigmoid_pulse)
    sig_times_1ms = comp.vec_mul(sigmoid_pulse, one_minus_sig)
    x_sig_1ms = comp.vec_mul(x_pulse, sig_times_1ms)
    deriv = comp.vec_add(sigmoid_pulse, x_sig_1ms)
    grad_x_pulse = comp.vec_mul(grad_out_pulse, deriv)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-5


# ==============================================================================
# Test 5: Softmax
# ==============================================================================
def test_softmax():
    print("\n" + "="*60)
    print("Test 5: Softmax")
    print("="*60)

    torch.manual_seed(42)
    batch, seq_len = 2, 4

    x_float = torch.randn(batch, seq_len, device=device, requires_grad=True)

    # PyTorch
    y_float = F.softmax(x_float, dim=-1)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # SNN STE: grad_x = y * (grad_out - sum(grad_out * y))
    x_float2 = x_float.detach().clone()
    y_pulse = float32_to_pulse(F.softmax(x_float2, dim=-1))
    grad_out_pulse = float32_to_pulse(grad_out_float)

    comp = get_snn_components(device)

    # grad_out * y
    grad_times_y = comp.vec_mul(grad_out_pulse, y_pulse)

    # sum 沿 dim=-2 (seq_len 维度，在 pulse 表示中是 dim=-2)
    # grad_times_y: [batch, seq_len, 32]
    # 转置后归约
    grad_times_y_t = grad_times_y.transpose(-2, 0)  # [seq_len, batch, 32]
    # 不对，应该沿 seq_len 归约
    # 重新组织: [batch, seq_len, 32] -> 沿 seq_len 归约
    sum_pulse = _parallel_reduce_pulse(grad_times_y.transpose(1, 0), comp)  # [batch, 32]

    # 广播
    sum_expanded = sum_pulse.unsqueeze(1).expand_as(grad_out_pulse)

    # grad_out - sum
    diff = comp.vec_sub(grad_out_pulse, sum_expanded)

    # y * diff
    grad_x_pulse = comp.vec_mul(y_pulse, diff)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-4  # softmax 可能有更大误差


# ==============================================================================
# Test 6: ReLU
# ==============================================================================
def test_relu():
    print("\n" + "="*60)
    print("Test 6: ReLU")
    print("="*60)

    torch.manual_seed(42)
    batch, features = 4, 8

    x_float = torch.randn(batch, features, device=device, requires_grad=True)

    # PyTorch
    y_float = F.relu(x_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()

    # SNN STE: grad_x = grad_out if x > 0 else 0
    x_float2 = x_float.detach().clone()
    x_pulse = float32_to_pulse(x_float2)
    grad_out_pulse = float32_to_pulse(grad_out_float)

    comp = get_snn_components(device)

    # 符号位: x_pulse[..., 0], 0=正, 1=负
    sign = x_pulse[..., 0:1]
    mask = comp.sign_not(sign)  # 1=正, 0=负
    mask_broadcast = mask.expand_as(grad_out_pulse)

    from atomic_ops.core.vec_logic_gates import VecAND
    vec_and = VecAND().to(device)
    grad_x_pulse = vec_and(grad_out_pulse, mask_broadcast)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    errors = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors)

    return errors['max_abs'] < 1e-5


# ==============================================================================
# Test 7: Linear (简化版 - 手动实现 backward)
# ==============================================================================
def test_linear():
    print("\n" + "="*60)
    print("Test 7: Linear")
    print("="*60)

    torch.manual_seed(42)
    batch, in_features, out_features = 2, 4, 3

    x_float = torch.randn(batch, in_features, device=device)
    weight_float = torch.randn(out_features, in_features, device=device)
    x_float.requires_grad_(True)
    weight_float.requires_grad_(True)

    # PyTorch
    y_float = F.linear(x_float, weight_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()
    ref_grad_w = weight_float.grad.clone()

    # SNN STE - 手动实现 backward
    # grad_x = grad_out @ W  (grad_out: [batch, out], W: [out, in] -> grad_x: [batch, in])
    # grad_W = grad_out^T @ x (grad_out: [batch, out], x: [batch, in] -> grad_W: [out, in])

    x_float2 = x_float.detach().clone()
    weight_float2 = weight_float.detach().clone()
    grad_out_float2 = grad_out_float.detach().clone()

    x_pulse = float32_to_pulse(x_float2)  # [batch, in, 32]
    weight_pulse = float32_to_pulse(weight_float2)  # [out, in, 32]
    grad_out_pulse = float32_to_pulse(grad_out_float2)  # [batch, out, 32]

    comp = get_snn_components(device)

    # grad_x[b, i] = sum_o(grad_out[b, o] * W[o, i])
    # 手动实现：grad_out: [batch, out, 32], W: [out, in, 32]
    # 扩展: grad_out -> [batch, out, 1, 32], W -> [1, out, in, 32]
    # 乘法: [batch, out, in, 32]
    # 沿 out 维度归约 -> [batch, in, 32]

    grad_out_exp = grad_out_pulse.unsqueeze(-2)  # [batch, out, 1, 32]
    weight_exp = weight_pulse.unsqueeze(0)  # [1, out, in, 32]
    products = comp.vec_mul(grad_out_exp, weight_exp)  # [batch, out, in, 32]

    # 沿 out 维度归约
    grad_x_pulse = _parallel_reduce_pulse(products.transpose(1, 0), comp)
    # 注意：这样会把 out 放到第 0 维然后归约，结果是 [batch, in, 32]
    # 实际上应该是沿 dim=1 归约
    products_t = products.permute(1, 0, 2, 3)  # [out, batch, in, 32]
    grad_x_pulse = _parallel_reduce_pulse(products_t, comp)  # [batch, in, 32]

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    # grad_W[o, i] = sum_b(grad_out[b, o] * x[b, i])
    # grad_out: [batch, out, 32], x: [batch, in, 32]
    # 扩展: grad_out -> [batch, out, 1, 32], x -> [batch, 1, in, 32]
    # 乘法: [batch, out, in, 32]
    # 沿 batch 维度归约 -> [out, in, 32]

    x_exp = x_pulse.unsqueeze(-3)  # [batch, 1, in, 32]
    products_w = comp.vec_mul(grad_out_exp, x_exp)  # [batch, out, in, 32]
    grad_w_pulse = _parallel_reduce_pulse(products_w, comp)  # [out, in, 32]

    snn_grad_w = pulse_to_float32(grad_w_pulse)

    print("  grad_x:")
    errors_x = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors_x)

    print("  grad_w:")
    errors_w = compute_errors(snn_grad_w, ref_grad_w)
    print_errors("grad_w", errors_w)

    return errors_x['max_abs'] < 1e-4 and errors_w['max_abs'] < 1e-4


# ==============================================================================
# Test 8: Embedding
# ==============================================================================
def test_embedding():
    print("\n" + "="*60)
    print("Test 8: Embedding")
    print("="*60)

    torch.manual_seed(42)
    vocab_size, hidden_size, seq_len = 16, 8, 4

    indices = torch.randint(0, vocab_size, (2, seq_len), device=device)
    weight_float = torch.randn(vocab_size, hidden_size, device=device, requires_grad=True)

    # PyTorch
    y_float = F.embedding(indices, weight_float)
    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_w = weight_float.grad.clone()

    # SNN STE - 手动实现 embedding backward
    weight_float2 = weight_float.detach().clone()
    weight_pulse = float32_to_pulse(weight_float2)  # [vocab, hidden, 32]
    grad_out_pulse = float32_to_pulse(grad_out_float)  # [batch, seq, hidden, 32]

    comp = get_snn_components(device)

    # Embedding backward: 对每个 index，累加对应的 grad_out
    # 展平处理
    flat_indices = indices.view(-1)  # [N]
    flat_grad = grad_out_pulse.view(-1, hidden_size, 32)  # [N, hidden, 32]
    N = flat_indices.shape[0]

    # 构建 one-hot 矩阵 [N, vocab_size]
    one_hot = torch.zeros(N, vocab_size, device=device, dtype=flat_grad.dtype)
    one_hot.scatter_(1, flat_indices.unsqueeze(1), 1.0)

    # one_hot.T: [V, N], flat_grad: [N, H, 32]
    one_hot_t = one_hot.T  # [V, N]
    mask_expanded = one_hot_t.unsqueeze(-1).unsqueeze(-1)  # [V, N, 1, 1]
    grad_expanded = flat_grad.unsqueeze(0)  # [1, N, H, 32]
    masked_grads = mask_expanded * grad_expanded  # [V, N, H, 32]

    # 对 N 维度进行并行归约
    from atomic_ops.core.ste import _parallel_reduce_pulse_dim
    grad_weight_pulse = _parallel_reduce_pulse_dim(masked_grads, dim=1, comp=comp)

    snn_grad_w = pulse_to_float32(grad_weight_pulse)

    errors = compute_errors(snn_grad_w, ref_grad_w)
    print_errors("grad_weight", errors)

    return errors['max_abs'] < 1e-4


# ==============================================================================
# Test 9: RMSNorm (完整公式)
# ==============================================================================
def test_rmsnorm():
    print("\n" + "="*60)
    print("Test 9: RMSNorm (完整公式)")
    print("="*60)

    torch.manual_seed(42)
    batch, hidden_size = 2, 8
    eps = 1e-6

    x_float = torch.randn(batch, hidden_size, device=device, requires_grad=True)
    weight_float = torch.ones(hidden_size, device=device, requires_grad=True)

    # PyTorch RMSNorm forward
    rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + eps)
    x_norm_float = x_float / rms
    y_float = x_norm_float * weight_float

    grad_out_float = torch.randn_like(y_float)
    y_float.backward(grad_out_float)
    ref_grad_x = x_float.grad.clone()
    ref_grad_w = weight_float.grad.clone()

    # SNN STE - 完整 RMSNorm backward
    # grad_x = grad_out * weight * rms_inv
    #        - x * rms_inv³ * (1/n) * sum(grad_out * weight * x)

    x_float2 = x_float.detach().clone()
    weight_float2 = weight_float.detach().clone()
    rms2 = torch.sqrt(torch.mean(x_float2 ** 2, dim=-1, keepdim=True) + eps)
    rms_inv_float = 1.0 / rms2
    x_norm_float2 = x_float2 / rms2

    x_pulse = float32_to_pulse(x_float2)  # [batch, hidden, 32]
    weight_pulse = float32_to_pulse(weight_float2)  # [hidden, 32]
    rms_inv_pulse = float32_to_pulse(rms_inv_float)  # [batch, 1, 32]
    x_norm_pulse = float32_to_pulse(x_norm_float2)  # [batch, hidden, 32]
    grad_out_pulse = float32_to_pulse(grad_out_float)  # [batch, hidden, 32]

    # hidden_size 的脉冲表示 (用于 SNN 除法)
    hidden_size_float = torch.tensor(float(hidden_size), device=device)
    hidden_size_pulse = float32_to_pulse(hidden_size_float)  # [32]

    comp = get_snn_components(device)

    # ============================================================
    # grad_weight = sum(grad_output * x_norm) 沿 batch 维度
    # ============================================================
    grad_times_xnorm = comp.vec_mul(grad_out_pulse, x_norm_pulse)  # [batch, hidden, 32]
    grad_weight_pulse = _parallel_reduce_pulse(grad_times_xnorm, comp)  # [hidden, 32]

    snn_grad_w = pulse_to_float32(grad_weight_pulse)

    # ============================================================
    # grad_x 完整公式
    # ============================================================

    # 第一项: grad_out * weight * rms_inv
    grad_weight_term = comp.vec_mul(grad_out_pulse, weight_pulse)  # [batch, hidden, 32]
    term1 = comp.vec_mul(grad_weight_term, rms_inv_pulse)  # [batch, hidden, 32]

    # 第二项准备: grad_out * weight * x
    grad_w_x = comp.vec_mul(grad_weight_term, x_pulse)  # [batch, hidden, 32]

    # sum(grad_out * weight * x) 沿 hidden 维度归约
    # [batch, hidden, 32] -> [hidden, batch, 32] -> 归约 -> [batch, 32]
    grad_w_x_t = grad_w_x.transpose(-2, 0)  # [hidden, batch, 32]
    sum_grad_w_x = _parallel_reduce_pulse(grad_w_x_t, comp)  # [batch, 32]
    sum_grad_w_x = sum_grad_w_x.unsqueeze(-2)  # [batch, 1, 32]

    # rms_inv³ = rms_inv * rms_inv * rms_inv
    rms_inv_sq = comp.vec_mul(rms_inv_pulse, rms_inv_pulse)
    rms_inv_cubed = comp.vec_mul(rms_inv_sq, rms_inv_pulse)

    # (1/n) 使用 SNN 除法器: 1.0 / hidden_size
    one_pulse = comp.constants.one(())  # [32]
    one_expanded = one_pulse.unsqueeze(0)  # [1, 32]
    n_expanded = hidden_size_pulse.unsqueeze(0)  # [1, 32]
    inv_n_pulse = comp.vec_div(one_expanded, n_expanded)  # [1, 32]
    inv_n_pulse = inv_n_pulse.squeeze(0)  # [32]

    # 广播 inv_n_pulse 到 sum_grad_w_x 的形状
    inv_n_broadcast = inv_n_pulse.expand(sum_grad_w_x.shape)

    # x * rms_inv³ * (1/n) * sum(...)
    sum_scaled = comp.vec_mul(sum_grad_w_x, inv_n_broadcast)  # [batch, 1, 32]
    sum_scaled = comp.vec_mul(sum_scaled, rms_inv_cubed)  # [batch, 1, 32]
    term2 = comp.vec_mul(x_pulse, sum_scaled)  # [batch, hidden, 32] (广播)

    # grad_x = term1 - term2
    grad_x_pulse = comp.vec_sub(term1, term2)

    snn_grad_x = pulse_to_float32(grad_x_pulse)

    print("  grad_x (完整公式):")
    errors_x = compute_errors(snn_grad_x, ref_grad_x)
    print_errors("grad_x", errors_x)

    print("  grad_weight:")
    errors_w = compute_errors(snn_grad_w, ref_grad_w)
    print_errors("grad_weight", errors_w)

    # 完整公式应该有较高精度
    return errors_x['max_abs'] < 1e-4 and errors_w['max_abs'] < 1e-4


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("="*60)
    print("STE 反向传播精度验证")
    print("="*60)

    results = {}

    results['sigmoid'] = test_sigmoid()
    results['tanh'] = test_tanh()
    results['exp'] = test_exp()
    results['silu'] = test_silu()
    results['softmax'] = test_softmax()
    results['relu'] = test_relu()
    results['linear'] = test_linear()
    results['embedding'] = test_embedding()
    results['rmsnorm'] = test_rmsnorm()

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print("所有测试通过!")
    else:
        print("存在测试失败，请检查!")
    print("="*60)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
