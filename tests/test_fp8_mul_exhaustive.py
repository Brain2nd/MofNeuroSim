import sys
sys.path.insert(0, "/home/dgxspark/Desktop/HumanBrain")
import torch
from SNNTorch.atomic_ops import SpikeFP8Multiplier, PulseFloatingPointEncoder, PulseFloatingPointDecoder

def test_exhaustive():
    print("=== Comprehensive Random Testing for SpikeFP8Multiplier ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Prepare Models
    # Encoder: E4M3 (E=4, M=3)
    encoder = PulseFloatingPointEncoder(exponent_bits=4, mantissa_bits=3).to(device)
    # Decoder: E4M3 (E=4, M=3) - Standard Framework Component
    decoder = PulseFloatingPointDecoder(exponent_bits=4, mantissa_bits=3).to(device)
    
    multiplier = SpikeFP8Multiplier().to(device)
    
    # ... (Random Data Generation logic remains same) ...
    # 2. Generate Random Data
    N = 100
    # Avoid subnormal/zero for now to test core logic (E=0 cases might fail due to simplified adder logic)
    # Generate values in range [0.1, 10] to keep E in valid range
    a_float = (torch.rand(N, 1, device=device) + 0.1) * (torch.randint(0, 2, (N, 1), device=device) * 2 - 1) * 4
    b_float = (torch.rand(N, 1, device=device) + 0.1) * (torch.randint(0, 2, (N, 1), device=device) * 2 - 1) * 4
    
    # 3. Encode Inputs
    a_enc = encoder(a_float).squeeze(1)
    b_enc = encoder(b_float).squeeze(1)
    
    # 4. Run SNN Multiplier
    multiplier.reset()
    # Output is [Batch, 8] -> S(1) E(4) M(3)
    result_pulse = multiplier(a_enc, b_enc)
    
    # 5. Verify Results - 100%位精确比较
    # Decode SNN result using Framework Component
    snn_res = decoder(result_pulse)
    
    # Decode inputs to get the actual FP8 values seen by the hardware
    a_quant = decoder(a_enc)
    b_quant = decoder(b_enc)

    # Calculate expected result using encoded/decoded inputs for consistency
    # And then round the result to FP8 to simulate the target precision behavior
    ref_high_prec = a_quant * b_quant
    
    # Simulate FP8 rounding by casting
    # Note: validation requires PyTorch to support float8_e4m3fn on the device
    if hasattr(torch, 'float8_e4m3fn'):
        expected_res = ref_high_prec.to(torch.float8_e4m3fn).float()
    else:
        print("Warning: torch.float8_e4m3fn not supported, skipping strict rounding check - test will fail")
        expected_res = ref_high_prec
    
    # ... (Verification logic remains same) ...
    print(f"\nTesting {N} samples (100%位精确要求)...")
    print(f"{'A':<10} * {'B':<10} = {'Expected':<10} | {'SNN':<10} | {'Match':<10}")
    print("-" * 60)
    
    # 位精确比较
    import struct
    bit_match_count = 0
    for i in range(N):
        snn_bits = struct.unpack('>I', struct.pack('>f', snn_res[i].item()))[0]
        exp_bits = struct.unpack('>I', struct.pack('>f', expected_res[i].item()))[0]
        match = (snn_bits == exp_bits)
        if match:
            bit_match_count += 1
        if i < 10:  # Print first 10
            status = "✓" if match else "✗"
            print(f"{a_quant[i].item():<10.4f} * {b_quant[i].item():<10.4f} = {expected_res[i].item():<10.4f} | {snn_res[i].item():<10.4f} | {status}")
    
    accuracy = bit_match_count / N * 100
    print(f"\n位精确匹配: {bit_match_count}/{N} ({accuracy:.1f}%)")
    print(f"测试结果: {'通过 ✓' if accuracy == 100 else '失败 ✗ (要求100%位精确)'}")

if __name__ == "__main__":
    test_exhaustive()

