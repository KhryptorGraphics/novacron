#!/usr/bin/env python3
"""
Quick test to verify legacy wrapper classes work with new DB configuration
"""
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_legacy_compatibility():
    """Test that legacy wrapper classes work with new DB configuration"""
    print("Testing Legacy Wrapper Compatibility")
    print("=" * 40)

    # Test 1: Legacy AutoScaler
    print("\n1. Testing AutoScaler legacy wrapper...")
    try:
        from ai_engine.predictive_scaling import AutoScaler
        scaler = AutoScaler()
        print(f"  ✓ AutoScaler created with DB path: {scaler.db_path}")

        # Test with custom path
        custom_scaler = AutoScaler(db_path='/tmp/legacy_autoscaler.db')
        assert custom_scaler.db_path == '/tmp/legacy_autoscaler.db'
        print(f"  ✓ Custom path accepted: {custom_scaler.db_path}")

    except Exception as e:
        print(f"  ✗ AutoScaler error: {e}")
        return False

    # Test 2: Legacy WorkloadClassifier
    print("\n2. Testing WorkloadClassifier legacy wrapper...")
    try:
        from ai_engine.workload_pattern_recognition import WorkloadClassifier
        classifier = WorkloadClassifier()
        print(f"  ✓ WorkloadClassifier created with DB path: {classifier.db_path}")

        # Test with custom path
        custom_classifier = WorkloadClassifier(db_path='/tmp/legacy_classifier.db')
        assert custom_classifier.db_path == '/tmp/legacy_classifier.db'
        print(f"  ✓ Custom path accepted: {custom_classifier.db_path}")

    except Exception as e:
        print(f"  ✗ WorkloadClassifier error: {e}")
        return False

    print("\n" + "=" * 40)
    print("✓ Legacy wrapper compatibility confirmed!")
    return True

if __name__ == "__main__":
    success = test_legacy_compatibility()
    sys.exit(0 if success else 1)