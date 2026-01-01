import subprocess
import os
import sys

# Whitelist of tests
TEST_FILES = [
    "tests/test_logic_gates.py", 
    "tests/test_vec_logic_gates.py",
    "tests/test_fp8_mul.py",
    "tests/test_fp8_adder_spatial.py",
    "tests/test_fp32_adder.py",
    "tests/test_fp32_mul.py",
    "tests/test_fp64_components.py",
    "tests/test_all_fp32_components.py"
]

def run_tests():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We need to run from project_root to ensure imports work if scripts expect it
    # But scripts add project_root to sys.path themselves usually.
    # Let's run from project root.
    
    total = len(TEST_FILES)
    passed = 0
    failed_files = []
    
    print(f"Running {total} test scripts...")
    print("="*60)
    
    for test_file in TEST_FILES:
        print(f"Running {test_file} ... ", end='', flush=True)
        
        try:
            # Run with conda env python if possible, or sys.executable
            # Assuming 'python' is in path from conda run
            cmd = ["python", test_file]
            
            result = subprocess.run(
                cmd, 
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            output = result.stdout + result.stderr
            
            # Criteria for failure:
            # 1. Non-zero exit code
            # 2. "FAIL" string in output (case sensitive? usually scripts print FAIL)
            # 3. "Error" string in output (sometimes)
            
            if result.returncode != 0:
                print("FAILED (Exit Code)")
                print("-" * 20)
                print(output)
                print("-" * 20)
                failed_files.append(test_file)
            elif "FAIL" in output or "FAILED" in output:
                # Be careful, some valid output might contain FAIL
                # checking line by line?
                print("FAILED (Output Check)")
                print("-" * 20)
                print(output)
                print("-" * 20)
                failed_files.append(test_file)
            else:
                print("PASSED")
                passed += 1
                
        except Exception as e:
            print(f"ERROR: {e}")
            failed_files.append(test_file)
            
    print("="*60)
    print(f"Summary: {passed}/{total} Tests Passed")
    
    if failed_files:
        print("Failed tests:")
        for f in failed_files:
            print(f" - {f}")
        return False
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
