#!/usr/bin/env python3

import sys
import subprocess
import json
from pathlib import Path
import time

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import openai
        print("✓ openai")
    except ImportError as e:
        print(f"✗ openai: {e}")
        return False
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    return True

def test_chain_generation():
    """Test the basic chain generation functionality"""
    print("\nTesting chain generation...")
    
    try:
        from implication_chains import create_seed_statements, Atom, Statement
        
        seeds = create_seed_statements()
        print(f"✓ Created {len(seeds)} seed statements")
        
        # Test atom creation
        atom = Atom("dogs", "are", "mammals", "all", True)
        print(f"✓ Created atom: {atom}")
        
        # Test statement creation
        stmt = Statement("All dogs are mammals", [atom])
        print(f"✓ Created statement: {stmt.text}")
        
        return True
        
    except Exception as e:
        print(f"✗ Chain generation test failed: {e}")
        return False

def test_api_key():
    """Test if OpenAI API key is set"""
    import os
    
    print("\nTesting API configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY='your-key'")
        return False
    
    if api_key.startswith("sk-"):
        print("✓ OpenAI API key format looks correct")
        return True
    else:
        print("⚠ API key format may be incorrect (should start with 'sk-')")
        return False

def test_small_chain():
    """Test generating one small chain (if API key is available)"""
    import os
    from implication_chains import ImplicationChainGenerator, create_seed_statements
    
    print("\nTesting small chain generation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping API test - no API key")
        return True
    
    try:
        generator = ImplicationChainGenerator(api_key)
        seeds = create_seed_statements()
        
        # Generate just one small chain
        print("Generating test chain (this may take a few seconds)...")
        chain = generator.generate_chain(seeds[0], chain_length=1)
        
        print(f"✓ Generated chain with {len(chain)} statements:")
        for i, stmt in enumerate(chain):
            print(f"  {i}: {stmt.text}")
            print(f"     Atoms: {[str(atom) for atom in stmt.atoms]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Small chain test failed: {e}")
        return False

def run_quick_dataset_test():
    """Test dataset building with minimal data"""
    print("\nTesting quick dataset build...")
    
    try:
        import os
        from build_dataset import DatasetBuilder
        from implication_chains import create_seed_statements
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠ Skipping dataset test - no API key")
            return True
        
        # Use just 3 seeds for quick test
        seeds = create_seed_statements()[:3]
        
        builder = DatasetBuilder(api_key, chain_length=2, min_distance=1)
        
        print("Building mini dataset (this may take 1-2 minutes)...")
        result = builder.build_dataset(
            seed_statements=seeds,
            output_file="test_dataset.jsonl",
            target_pairs=50
        )
        
        print(f"✓ Generated {len(result['pairs'])} pairs from {len(result['chains'])} chains")
        
        # Cleanup test file
        if Path("test_dataset.jsonl").exists():
            Path("test_dataset.jsonl").unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Quick dataset test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MVP IMPLICATION CHAINS - SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Chain Generation Test", test_chain_generation),
        ("API Key Test", test_api_key),
        ("Small Chain Test", test_small_chain),
        ("Quick Dataset Test", run_quick_dataset_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running: {test_name}")
        print(f"{'-' * 30}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n⚠ Test interrupted by user")
            break
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to run.")
        print("\nNext steps:")
        print("1. Run full dataset generation: python build_dataset.py")
        print("2. Train the classifier: python train_classifier.py")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)