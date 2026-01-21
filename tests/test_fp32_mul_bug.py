
import torch
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops import PulseFloatingPointEncoder
from atomic_ops.encoding.pulse_decoder import PulseFP32Decoder
from atomic_ops.arithmetic.fp32.fp32_mul import SpikeFP32Multiplier

def test_fp32_mul_reproduction():
    print("="*60)
    print("Testing SpikeFP32Multiplier for Bugs")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Init encoder/decoder/multiplier
    encoder = PulseFloatingPointEncoder(
        exponent_bits=8, mantissa_bits=23,
        scan_integer_bits=10, scan_decimal_bits=10
    ).to(device)
    
    decoder = PulseFP32Decoder().to(device)
    mul = SpikeFP32Multiplier().to(device)
    
    # Test Cases
    print("\n--- 1. Random Normal Inputs ---")
    batch_size = 100
    torch.manual_seed(42)
    a_float = torch.randn(batch_size, 1, device=device)
    b_float = torch.randn(batch_size, 1, device=device)
    
    run_test(a_float, b_float, encoder, decoder, mul, "Random Normal")
    
    print("\n--- 2. Special Values ---")
    # Zeros
    a_zero = torch.tensor([0.0, 0.0, 1.0, 5.0], device=device).unsqueeze(1)
    b_zero = torch.tensor([0.0, 5.0, 0.0, 0.0], device=device).unsqueeze(1)
    run_test(a_zero, b_zero, encoder, decoder, mul, "Zero Handling")
    
    # Inf
    inf = float('inf')
    a_inf = torch.tensor([inf, inf, inf, 1.0, 0.0], device=device).unsqueeze(1)
    b_inf = torch.tensor([1.0, inf, -1.0, inf, inf], device=device).unsqueeze(1) # Inf*1, Inf*Inf, Inf*-1, 1*Inf, 0*Inf(NaN)
    run_test(a_inf, b_inf, encoder, decoder, mul, "Infinity Handling")
    
    # NaN
    nan = float('nan')
    a_nan = torch.tensor([nan, 1.0], device=device).unsqueeze(1)
    b_nan = torch.tensor([1.0, nan], device=device).unsqueeze(1)
    run_test(a_nan, b_nan, encoder, decoder, mul, "NaN Handling")

    print("\n--- 3. Subnormal Checks ---") 
    # Small numbers
    small = 1e-40
    a_sub = torch.tensor([small, small * 1e5], device=device).unsqueeze(1)
    b_sub = torch.tensor([1.0, 1e-5], device=device).unsqueeze(1)
    run_test(a_sub, b_sub, encoder, decoder, mul, "Subnormal Handling")

def run_test(a, b, encoder, decoder, mul, name):
    print(f"\nRunning {name}...")
    
    # Reference
    res_ref = a * b
    
    # SNN
    encoder.reset()
    a_pulse = encoder(a)
    b_pulse = encoder(b)
    
    mul.reset()
    res_pulse = mul(a_pulse, b_pulse)
    
    decoder.reset()
    res_snn = decoder(res_pulse)
    
    # Compare
    # Handle NaN for comparison
    nan_mask = torch.isnan(res_ref)
    mask = ~nan_mask
    
    if nan_mask.any():
        snn_nan = torch.isnan(res_snn)
        nan_match = (nan_mask == snn_nan).all()
        print(f"NaN match: {nan_match}")
        if not nan_match:
             print(f"Ref NaN: {nan_mask}")
             print(f"SNN NaN: {snn_nan}")

    # Numeric compare
    if mask.any():
        diff = torch.abs(res_snn[mask] - res_ref[mask])
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        matches = torch.isclose(res_snn[mask], res_ref[mask], rtol=1e-4, atol=1e-5)
        match_rate = matches.float().mean().item() * 100
        
        print(f"Match Rate: {match_rate:.2f}%")
        print(f"Max Diff: {max_diff:.2e}")
        
        if match_rate < 100:
             print("Mismatches found!")
             mismatch_idx = torch.where(~matches)[0]
             for i in mismatch_idx[:5]:
                 print(f"  Idx {i}: A={a[mask][i].item():.4e}, B={b[mask][i].item():.4e} -> Ref={res_ref[mask][i].item():.4e}, SNN={res_snn[mask][i].item():.4e}")

if __name__ == "__main__":
    test_fp32_mul_reproduction()
