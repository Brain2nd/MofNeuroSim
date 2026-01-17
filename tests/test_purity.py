import torch
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atomic_ops.core.logic_gates import NOTGate, XORGate
from atomic_ops.core.vec_logic_gates import VecNOT, VecXOR

def test_scalar_gates():
    print("\nTesting Scalar Gates (logic_gates.py)...")
    not_gate = NOTGate()
    xor_gate = XORGate()

    # NOT
    assert not_gate(torch.tensor(0.0)).item() == 1.0, "NOT(0) failed"
    assert not_gate(torch.tensor(1.0)).item() == 0.0, "NOT(1) failed"
    print("NOTGate: PASS")

    # XOR
    assert xor_gate(torch.tensor(0.0), torch.tensor(0.0)).item() == 0.0, "XOR(0,0) failed"
    assert xor_gate(torch.tensor(0.0), torch.tensor(1.0)).item() == 1.0, "XOR(0,1) failed"
    assert xor_gate(torch.tensor(1.0), torch.tensor(0.0)).item() == 1.0, "XOR(1,0) failed"
    assert xor_gate(torch.tensor(1.0), torch.tensor(1.0)).item() == 0.0, "XOR(1,1) failed"
    print("XORGate: PASS")

def test_vector_gates():
    print("\nTesting Vector Gates (vec_logic_gates.py)...")
    vec_not = VecNOT()
    vec_xor = VecXOR()

    x = torch.tensor([0.0, 1.0])
    y_not = vec_not(x)
    assert torch.allclose(y_not, torch.tensor([1.0, 0.0])), f"VecNOT failed: {y_not}"
    print("VecNOT: PASS")

    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y_xor = vec_xor(a, b)
    assert torch.allclose(y_xor, torch.tensor([0.0, 1.0, 1.0, 0.0])), f"VecXOR failed: {y_xor}"
    print("VecXOR: PASS")

if __name__ == "__main__":
    try:
        test_scalar_gates()
        test_vector_gates()
        print("\nALL LOGIC TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
