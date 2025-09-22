#!/usr/bin/env python3
"""
Test script to verify AI engine fixes are working correctly.
Tests the three critical fixes:
1. predict_sequence method in EnhancedResourcePredictor
2. Anomaly detection endpoint calling detect() directly
3. predict_optimal_host method in MigrationPredictor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import AI models
from models import (
    EnhancedResourcePredictor, AdvancedAnomalyDetector,
    SophisticatedMigrationPredictor, MigrationPredictor
)

def test_predict_sequence():
    """Test the predict_sequence method in EnhancedResourcePredictor"""
    print("Testing predict_sequence method...")

    # Create test data
    historical_data = []
    for i in range(24):  # 24 hours of data
        historical_data.append({
            'timestamp': datetime.now() - timedelta(hours=23-i),
            'cpu_usage': 50 + 10 * np.sin(i * np.pi / 12),  # Simulate daily pattern
            'memory_usage': 60 + 5 * np.sin(i * np.pi / 12),
            'disk_usage': 30 + 2 * np.random.random(),
            'network_usage': 40 + 15 * np.random.random(),
        })

    df = pd.DataFrame(historical_data)

    # Test the method
    predictor = EnhancedResourcePredictor()
    result = predictor.predict_sequence(df, ['cpu_usage', 'memory_usage'], horizon=12)

    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'cpu_usage' in result, "Should contain cpu_usage predictions"
    assert 'memory_usage' in result, "Should contain memory_usage predictions"
    assert 'confidence' in result, "Should contain confidence score"
    assert 'model_used' in result, "Should contain model_used info"
    assert 'feature_importance' in result, "Should contain feature importance"

    # Verify predictions
    cpu_predictions = result['cpu_usage']
    memory_predictions = result['memory_usage']

    assert len(cpu_predictions) == 12, "Should have 12 CPU predictions"
    assert len(memory_predictions) == 12, "Should have 12 memory predictions"
    assert all(isinstance(p, (int, float)) for p in cpu_predictions), "All predictions should be numeric"

    print("✅ predict_sequence test passed")
    return True

def test_anomaly_detector():
    """Test the anomaly detector detect() method"""
    print("Testing anomaly detector detect() method...")

    detector = AdvancedAnomalyDetector()

    # Create test metrics
    test_metrics = {
        'cpu_usage': 85.5,
        'memory_usage': 78.2,
        'disk_usage': 45.1,
        'network_usage': 92.8,
        'response_time': 1200
    }

    # Test the detect method
    result = detector.detect(test_metrics)

    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'is_anomaly' in result, "Should contain is_anomaly flag"
    assert 'anomaly_score' in result, "Should contain anomaly_score"
    assert 'severity' in result, "Should contain severity"
    assert 'anomaly_type' in result, "Should contain anomaly_type"
    assert 'contributing_features' in result, "Should contain contributing_features"

    # Verify data types
    assert isinstance(result['is_anomaly'], bool), "is_anomaly should be boolean"
    assert isinstance(result['anomaly_score'], (int, float)), "anomaly_score should be numeric"
    assert isinstance(result['severity'], str), "severity should be string"
    assert isinstance(result['contributing_features'], list), "contributing_features should be list"

    print("✅ Anomaly detector test passed")
    return True

def test_predict_optimal_host():
    """Test the predict_optimal_host method in MigrationPredictor"""
    print("Testing predict_optimal_host method...")

    predictor = MigrationPredictor()

    # Test data
    vm_id = "vm-test-001"
    target_hosts = ["host-1", "host-2", "host-3"]
    vm_metrics = {
        'vm_cpu_cores': 4,
        'vm_memory_gb': 16,
        'vm_disk_gb': 200,
        'cpu_utilization': 0.7,
        'memory_utilization': 0.6,
        'disk_io_ops': 150
    }
    network_topology = {
        'host-1': {'capacity_score': 0.8, 'bandwidth_mbps': 1000},
        'host-2': {'capacity_score': 0.6, 'bandwidth_mbps': 800},
        'host-3': {'capacity_score': 0.9, 'bandwidth_mbps': 1200},
    }
    sla_requirements = {
        'max_downtime_seconds': 30,
        'min_success_rate': 0.95
    }

    # Test the method
    result = predictor.predict_optimal_host(
        vm_id, target_hosts, vm_metrics, network_topology, sla_requirements
    )

    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'recommended_host' in result, "Should contain recommended_host"
    assert 'migration_time' in result, "Should contain migration_time"
    assert 'downtime' in result, "Should contain downtime"
    assert 'confidence' in result, "Should contain confidence"
    assert 'reasons' in result, "Should contain reasons"
    assert 'score' in result, "Should contain score"

    # Verify data types and values
    assert result['recommended_host'] in target_hosts, "Recommended host should be from target list"
    assert isinstance(result['migration_time'], (int, float)), "Migration time should be numeric"
    assert isinstance(result['downtime'], (int, float)), "Downtime should be numeric"
    assert isinstance(result['confidence'], (int, float)), "Confidence should be numeric"
    assert isinstance(result['reasons'], list), "Reasons should be a list"
    assert isinstance(result['score'], (int, float)), "Score should be numeric"

    assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"
    assert 0 <= result['score'] <= 1, "Score should be between 0 and 1"
    assert len(result['reasons']) > 0, "Should have at least one reason"

    print("✅ predict_optimal_host test passed")
    return True

def test_integration():
    """Test that all fixes work together in a realistic scenario"""
    print("Testing integration scenario...")

    # Test resource prediction sequence
    predictor = EnhancedResourcePredictor()
    historical_data = []
    for i in range(48):  # 48 hours of data
        historical_data.append({
            'timestamp': datetime.now() - timedelta(hours=47-i),
            'cpu_usage': 45 + 20 * np.sin(i * np.pi / 24) + np.random.normal(0, 5),
            'memory_usage': 55 + 15 * np.sin(i * np.pi / 24) + np.random.normal(0, 3),
            'disk_usage': 25 + np.random.normal(0, 2),
            'network_usage': 35 + 25 * np.random.random(),
        })

    df = pd.DataFrame(historical_data)
    resource_prediction = predictor.predict_sequence(df, ['cpu_usage', 'memory_usage'], horizon=24)

    # Test anomaly detection
    detector = AdvancedAnomalyDetector()
    current_metrics = {
        'cpu_usage': resource_prediction['cpu_usage'][0] + 30,  # Simulate spike
        'memory_usage': resource_prediction['memory_usage'][0],
        'disk_usage': 25,
        'network_usage': 40
    }

    anomaly_result = detector.detect(current_metrics)

    # Test migration prediction
    migration_predictor = MigrationPredictor()
    migration_result = migration_predictor.predict_optimal_host(
        "vm-integration-test",
        ["host-a", "host-b"],
        current_metrics,
        {'host-a': {'capacity_score': 0.7}, 'host-b': {'capacity_score': 0.9}},
        {'max_downtime_seconds': 20}
    )

    # Verify all components worked
    assert len(resource_prediction['cpu_usage']) == 24, "Resource prediction should work"
    assert 'is_anomaly' in anomaly_result, "Anomaly detection should work"
    assert migration_result['recommended_host'] in ["host-a", "host-b"], "Migration prediction should work"

    print("✅ Integration test passed")
    return True

def main():
    """Run all tests"""
    print("=== AI Engine Fixes Validation ===\n")

    tests = [
        test_predict_sequence,
        test_anomaly_detector,
        test_predict_optimal_host,
        test_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Test Results ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All AI engine fixes are working correctly!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)