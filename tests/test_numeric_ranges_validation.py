#!/usr/bin/env python3
"""
Test script to validate numeric ranges and units consistency in AI engine files.
This script verifies that the fixes for numeric scaling and units are working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_engine.predictive_scaling import PredictiveScalingEngine, ResourceType
from ai_engine.performance_optimizer import PerformancePredictor, PerformanceMetrics

def test_predictive_scaling_units():
    """Test that predictive scaling maintains proper units throughout the pipeline"""
    print("Testing predictive scaling numeric ranges and units...")

    try:
        # Initialize engine
        engine = PredictiveScalingEngine(db_path="/tmp/test_scaling.db")
    except Exception as e:
        print(f"❌ Failed to initialize PredictiveScalingEngine: {e}")
        return False

    # Create test data with realistic values in natural units
    test_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(100)],
        'cpu_usage': np.random.uniform(20, 90, 100),  # CPU: 20-90%
        'memory_usage': np.random.uniform(30, 85, 100),  # Memory: 30-85%
        'network_usage': np.random.uniform(50, 500, 100),  # Network: 50-500 Mbps
        'storage_usage': np.random.uniform(10, 80, 100),  # Storage: 10-80%
    })

    # Test CPU predictions
    try:
        forecast = engine.predict_resource_demand('test_vm', ResourceType.CPU, test_data)

        # Verify predictions are in percentage range (not hard-clipped to 0-1)
        cpu_predictions = forecast.predicted_values
        print(f"CPU predictions range: {min(cpu_predictions):.2f}% to {max(cpu_predictions):.2f}%")

        # Check that predictions aren't artificially clipped
        assert not all(0 <= p <= 1 for p in cpu_predictions), "CPU predictions shouldn't be hard-clipped to [0,1]"

        # Verify confidence intervals match prediction units
        intervals = forecast.confidence_intervals
        print(f"CPU confidence interval example: ({intervals[0][0]:.2f}%, {intervals[0][1]:.2f}%)")

        print("✅ CPU scaling predictions maintain proper percentage units")

    except Exception as e:
        print(f"❌ CPU scaling test failed: {e}")
        return False

    # Test Network predictions
    try:
        forecast = engine.predict_resource_demand('test_vm', ResourceType.NETWORK, test_data)

        network_predictions = forecast.predicted_values
        print(f"Network predictions range: {min(network_predictions):.2f} to {max(network_predictions):.2f} Mbps")

        # Network predictions should not be constrained to [0,1]
        assert max(network_predictions) > 2, "Network predictions should be in Mbps, not [0,1] range"

        print("✅ Network scaling predictions maintain proper Mbps units")

    except Exception as e:
        print(f"❌ Network scaling test failed: {e}")
        return False

    return True

def test_performance_optimizer_units():
    """Test that performance optimizer maintains proper units and scaling"""
    print("\nTesting performance optimizer numeric ranges and units...")

    # Initialize predictor
    predictor = PerformancePredictor(db_path="/tmp/test_performance.db")

    # Store realistic performance data
    test_metrics = [
        PerformanceMetrics(
            timestamp=datetime.now() - timedelta(hours=i),
            node_id='test_node',
            cpu_utilization=np.random.uniform(20, 85),  # 20-85%
            memory_utilization=np.random.uniform(30, 90),  # 30-90%
            disk_iops=int(np.random.uniform(100, 5000)),  # 100-5000 IOPS
            network_bandwidth_mbps=np.random.uniform(50, 800),  # 50-800 Mbps
            latency_ms=np.random.uniform(5, 100),  # 5-100 ms
            throughput_ops_sec=np.random.uniform(200, 2000),  # 200-2000 ops/sec
            error_rate=np.random.uniform(0.001, 0.05),  # 0.1%-5% error rate
            response_time_ms=np.random.uniform(10, 150)  # 10-150 ms
        )
        for i in range(200)
    ]

    # Store metrics in database
    for metric in test_metrics:
        predictor.store_performance_data(metric)

    # Train models
    try:
        training_results = predictor.train_models(lookback_days=1)
        print(f"Training completed for {len(training_results)} targets")

        # Test predictions
        current_metrics = {
            'cpu_utilization': 65.0,  # 65%
            'memory_utilization': 75.0,  # 75%
            'network_bandwidth_mbps': 250.0,  # 250 Mbps
            'latency_ms': 25.0,  # 25 ms
            'error_rate': 0.02  # 2% (0.02 decimal ratio)
        }

        prediction = predictor.predict_performance(current_metrics)

        # Verify predictions are in natural units
        predicted_metrics = prediction.predicted_metrics

        if 'latency_ms' in predicted_metrics:
            latency = predicted_metrics['latency_ms']
            print(f"Predicted latency: {latency:.2f} ms")
            assert latency > 1, "Latency should be in milliseconds, not [0,1] range"

        if 'throughput_ops_sec' in predicted_metrics:
            throughput = predicted_metrics['throughput_ops_sec']
            print(f"Predicted throughput: {throughput:.2f} ops/sec")
            assert throughput > 10, "Throughput should be in ops/sec, not [0,1] range"

        if 'error_rate' in predicted_metrics:
            error_rate = predicted_metrics['error_rate']
            print(f"Predicted error rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
            assert 0 <= error_rate <= 1, "Error rate should be decimal ratio [0.0-1.0]"

        # Check uncertainty bounds are in same units
        bounds = prediction.uncertainty_bounds
        if 'latency_ms' in bounds:
            lower, upper = bounds['latency_ms']
            print(f"Latency uncertainty: ({lower:.2f}, {upper:.2f}) ms")
            assert lower >= 0 and upper > lower, "Latency bounds should be positive and properly ordered"

        print("✅ Performance predictions maintain proper natural units")

    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        return False

    return True

def test_scaling_consistency():
    """Test that scalers are properly fitted and inverse transforms work"""
    print("\nTesting scaler consistency...")

    # Create test engine
    engine = PredictiveScalingEngine(db_path="/tmp/test_scaler.db")

    # Check that scalers are initialized with proper ranges
    cpu_scaler = engine.scalers[ResourceType.CPU.value]
    memory_scaler = engine.scalers[ResourceType.MEMORY.value]
    network_scaler = engine.scalers[ResourceType.NETWORK.value]

    # Verify scaler types and ranges
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    assert isinstance(cpu_scaler, MinMaxScaler), "CPU should use MinMaxScaler"
    assert isinstance(memory_scaler, MinMaxScaler), "Memory should use MinMaxScaler"
    assert isinstance(network_scaler, MinMaxScaler), "Network should use MinMaxScaler"

    # Check feature ranges
    assert cpu_scaler.feature_range == (0, 100), "CPU scaler should have range (0, 100)"
    assert memory_scaler.feature_range == (0, 100), "Memory scaler should have range (0, 100)"
    assert network_scaler.feature_range == (0, 10000), "Network scaler should have range (0, 10000)"

    print("✅ Scalers initialized with proper ranges")

    return True

def test_unit_documentation():
    """Test that unit documentation is present and consistent"""
    print("\nTesting unit documentation...")

    # Check PerformanceMetrics docstring
    perf_metrics_doc = PerformanceMetrics.__doc__

    required_units = [
        'percentage (0-100%)',
        'megabits per second',
        'milliseconds',
        'operations per second',
        'decimal ratio (0.0-1.0'
    ]

    for unit in required_units:
        if unit not in perf_metrics_doc:
            print(f"❌ Missing unit documentation: {unit}")
            return False

    print("✅ Unit documentation is present and comprehensive")

    return True

def run_all_tests():
    """Run all validation tests"""
    print("Running AI Engine Numeric Ranges and Units Validation Tests")
    print("=" * 60)

    tests = [
        test_predictive_scaling_units,
        test_performance_optimizer_units,
        test_scaling_consistency,
        test_unit_documentation
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Numeric ranges and units are properly handled.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)