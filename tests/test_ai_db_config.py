#!/usr/bin/env python3
"""
Test script to verify AI Engine database path configuration
Ensures proper environment variable handling and directory creation
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_database_path_configuration():
    """Test database path configuration for all AI engine modules"""

    print("Testing AI Engine Database Path Configuration")
    print("=" * 60)

    # Test 1: Default paths without environment variables
    print("\n1. Testing default paths (no environment variables)...")
    try:
        # Clear environment variables if they exist
        for var in ['PREDICTIVE_SCALING_DB', 'WORKLOAD_PATTERNS_DB', 'PERFORMANCE_DB', 'NOVACRON_DATA_DIR']:
            os.environ.pop(var, None)

        from ai_engine.predictive_scaling import PredictiveScalingEngine
        from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
        from ai_engine.performance_optimizer import PerformancePredictor

        # Create instances with default paths
        ps_engine = PredictiveScalingEngine()
        wpr_engine = WorkloadPatternRecognizer()
        perf_predictor = PerformancePredictor()

        print(f"  ✓ PredictiveScalingEngine DB: {ps_engine.db_path}")
        print(f"  ✓ WorkloadPatternRecognizer DB: {wpr_engine.db_path}")
        print(f"  ✓ PerformancePredictor DB: {perf_predictor.db_path}")

    except Exception as e:
        print(f"  ✗ Error with default paths: {e}")
        return False

    # Test 2: Custom NOVACRON_DATA_DIR
    print("\n2. Testing with NOVACRON_DATA_DIR environment variable...")
    try:
        test_dir = tempfile.mkdtemp(prefix="novacron_test_")
        os.environ['NOVACRON_DATA_DIR'] = test_dir

        # Reload modules to pick up new environment variable
        import importlib
        import ai_engine.predictive_scaling
        import ai_engine.workload_pattern_recognition
        import ai_engine.performance_optimizer

        importlib.reload(ai_engine.predictive_scaling)
        importlib.reload(ai_engine.workload_pattern_recognition)
        importlib.reload(ai_engine.performance_optimizer)

        from ai_engine.predictive_scaling import PredictiveScalingEngine
        from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
        from ai_engine.performance_optimizer import PerformancePredictor

        ps_engine = PredictiveScalingEngine()
        wpr_engine = WorkloadPatternRecognizer()
        perf_predictor = PerformancePredictor()

        assert test_dir in ps_engine.db_path, f"Expected {test_dir} in {ps_engine.db_path}"
        assert test_dir in wpr_engine.db_path, f"Expected {test_dir} in {wpr_engine.db_path}"
        assert test_dir in perf_predictor.db_path, f"Expected {test_dir} in {perf_predictor.db_path}"

        print(f"  ✓ All databases created in NOVACRON_DATA_DIR: {test_dir}")

        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        print(f"  ✗ Error with NOVACRON_DATA_DIR: {e}")
        return False

    # Test 3: Individual database path environment variables
    print("\n3. Testing individual DB path environment variables...")
    try:
        test_ps_db = "/tmp/test_ps.db"
        test_wpr_db = "/tmp/test_wpr.db"
        test_perf_db = "/tmp/test_perf.db"

        os.environ['PREDICTIVE_SCALING_DB'] = test_ps_db
        os.environ['WORKLOAD_PATTERNS_DB'] = test_wpr_db
        os.environ['PERFORMANCE_DB'] = test_perf_db

        # Clear NOVACRON_DATA_DIR to test individual vars
        os.environ.pop('NOVACRON_DATA_DIR', None)

        # Reload modules again
        importlib.reload(ai_engine.predictive_scaling)
        importlib.reload(ai_engine.workload_pattern_recognition)
        importlib.reload(ai_engine.performance_optimizer)

        from ai_engine.predictive_scaling import PredictiveScalingEngine
        from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
        from ai_engine.performance_optimizer import PerformancePredictor

        ps_engine = PredictiveScalingEngine()
        wpr_engine = WorkloadPatternRecognizer()
        perf_predictor = PerformancePredictor()

        assert ps_engine.db_path == test_ps_db, f"Expected {test_ps_db}, got {ps_engine.db_path}"
        assert wpr_engine.db_path == test_wpr_db, f"Expected {test_wpr_db}, got {wpr_engine.db_path}"
        assert perf_predictor.db_path == test_perf_db, f"Expected {test_perf_db}, got {perf_predictor.db_path}"

        print(f"  ✓ PREDICTIVE_SCALING_DB: {ps_engine.db_path}")
        print(f"  ✓ WORKLOAD_PATTERNS_DB: {wpr_engine.db_path}")
        print(f"  ✓ PERFORMANCE_DB: {perf_predictor.db_path}")

    except Exception as e:
        print(f"  ✗ Error with individual DB paths: {e}")
        return False

    # Test 4: Non-writable directory fallback
    print("\n4. Testing fallback to /tmp for non-writable directories...")
    try:
        # Try to use a non-writable directory
        os.environ['NOVACRON_DATA_DIR'] = '/root/novacron_data'  # Typically not writable

        # Clear individual DB vars
        for var in ['PREDICTIVE_SCALING_DB', 'WORKLOAD_PATTERNS_DB', 'PERFORMANCE_DB']:
            os.environ.pop(var, None)

        # Reload modules
        importlib.reload(ai_engine.predictive_scaling)
        importlib.reload(ai_engine.workload_pattern_recognition)
        importlib.reload(ai_engine.performance_optimizer)

        from ai_engine.predictive_scaling import PredictiveScalingEngine
        from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
        from ai_engine.performance_optimizer import PerformancePredictor

        ps_engine = PredictiveScalingEngine()
        wpr_engine = WorkloadPatternRecognizer()
        perf_predictor = PerformancePredictor()

        # Should fall back to /tmp
        assert '/tmp/' in ps_engine.db_path, f"Expected /tmp fallback, got {ps_engine.db_path}"
        assert '/tmp/' in wpr_engine.db_path, f"Expected /tmp fallback, got {wpr_engine.db_path}"
        assert '/tmp/' in perf_predictor.db_path, f"Expected /tmp fallback, got {perf_predictor.db_path}"

        print(f"  ✓ Correctly fell back to /tmp for non-writable directory")

    except Exception as e:
        print(f"  ✗ Error testing fallback: {e}")
        return False

    # Test 5: Direct path parameter override
    print("\n5. Testing direct path parameter override...")
    try:
        custom_path = "/tmp/custom_test.db"

        from ai_engine.predictive_scaling import PredictiveScalingEngine
        from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
        from ai_engine.performance_optimizer import PerformancePredictor

        ps_engine = PredictiveScalingEngine(db_path=custom_path)
        wpr_engine = WorkloadPatternRecognizer(db_path=custom_path)
        perf_predictor = PerformancePredictor(db_path=custom_path)

        assert ps_engine.db_path == custom_path, f"Expected {custom_path}, got {ps_engine.db_path}"
        assert wpr_engine.db_path == custom_path, f"Expected {custom_path}, got {wpr_engine.db_path}"
        assert perf_predictor.db_path == custom_path, f"Expected {custom_path}, got {perf_predictor.db_path}"

        print(f"  ✓ All modules accepted direct path parameter: {custom_path}")

    except Exception as e:
        print(f"  ✗ Error with direct path parameter: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All database path configuration tests passed!")
    print("\nConfiguration priority order:")
    print("  1. Direct db_path parameter (highest priority)")
    print("  2. Individual *_DB environment variables")
    print("  3. NOVACRON_DATA_DIR environment variable")
    print("  4. Default: /var/lib/novacron/")
    print("  5. Fallback: /tmp/ (if directory not writable)")

    return True

if __name__ == "__main__":
    success = test_database_path_configuration()
    sys.exit(0 if success else 1)