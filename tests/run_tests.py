import unittest
import sys
import os

def main():
    print("==================================================")
    print("RUNNING AUTOMATED UNIT TEST SUITE")
    print("==================================================")
    
    # Set the path to the workspace root so imports resolve correctly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("==================================================")
    if result.wasSuccessful():
        print("[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("[FAILURE] Some tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
