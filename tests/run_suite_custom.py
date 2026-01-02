import subprocess
import os
import sys
import glob

def run_tests():
    # Root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(project_root, "tests")
    
    # Find all test files
    test_files = glob.glob(os.path.join(tests_dir, "test_*.py")) + \
                 glob.glob(os.path.join(tests_dir, "verify_*.py"))
    
    # Sort for consistent order
    test_files.sort()
    
    total = len(test_files)
    passed = 0
    failed_files = []
    
    print(f"Discovered {total} test files.")
    print("="*60)
    
    for full_path in test_files:
        filename = os.path.basename(full_path)
        # Skip this runner itself if it was matched (unlikely with test_ prefix but safe check)
        if filename == "run_suite_custom.py":
            continue
            
        print(f"Running {filename} ... ", end='', flush=True)
        
        try:
            # Run with python
            cmd = ["python", full_path]
            
            result = subprocess.run(
                cmd, 
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            output = result.stdout + result.stderr
            
            # Criteria for failure:
            if result.returncode != 0:
                print("FAILED (Exit Code)")
                # Print last few lines of output for context
                lines = output.strip().split('\n')
                print("-" * 20)
                print("\n".join(lines[-20:])) # Show last 20 lines
                print("-" * 20)
                failed_files.append(filename)
            elif "FAIL" in output or "FAILED" in output or "Error" in output:
                # Some scripts might print "Error" in normal logs, need checks
                # This is aggressive matching, but compliant with "find bugs"
                print("FAILED (Output Check)")
                lines = output.strip().split('\n')
                print("-" * 20)
                print("\n".join(lines[-20:]))
                print("-" * 20)
                failed_files.append(filename)
            else:
                print("PASSED")
                passed += 1
                
        except Exception as e:
            print(f"ERROR: {e}")
            failed_files.append(filename)
            
    print("="*60)
    print(f"Summary: {passed}/{total} Tests Passed")
    
    if failed_files:
        print("\nFailed tests:")
        for f in failed_files:
            print(f" - {f}")
        return False
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
