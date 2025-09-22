#!/usr/bin/env python3
"""
Test Go integration layer compatibility.
Verifies that the Python AI engine can properly handle requests from the Go AIIntegrationLayer.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.app import process_request

async def test_go_client_resource_prediction():
    """Test resource prediction request format from Go client"""
    print("🔍 Testing Go client resource prediction format...")

    # This is the format sent by Go AIIntegrationLayer.PredictResourceDemand
    request = {
        'id': 'resource-test-1',
        'service': 'resource_prediction',
        'method': 'predict_demand',
        'data': {
            'node_id': 'node-123',
            'resource_type': 'cpu',
            'horizon_minutes': 60,
            'historical_data': [
                {
                    'timestamp': '2024-01-01T10:00:00Z',
                    'value': 45.5,
                    'metadata': {'source': 'prometheus', 'node': 'worker-1'}
                },
                {
                    'timestamp': '2024-01-01T10:05:00Z',
                    'value': 52.3,
                    'metadata': {'source': 'prometheus', 'node': 'worker-1'}
                },
                {
                    'timestamp': '2024-01-01T10:10:00Z',
                    'value': 48.7,
                    'metadata': {'source': 'prometheus', 'node': 'worker-1'}
                }
            ],
            'context': {'cluster': 'prod', 'environment': 'production'}
        }
    }

    result = await process_request(request)

    # Verify Go client expected response format
    assert result['success'] == True, "Request should succeed"
    assert 'data' in result, "Response should contain data field"
    assert 'confidence' in result, "Response should contain confidence field"
    assert 'model_version' in result, "Response should contain model_version field"
    assert 'process_time' in result, "Response should contain process_time field"

    # Verify ResourcePredictionResponse structure
    data = result['data']
    assert 'predictions' in data, "Data should contain predictions array"
    assert 'confidence' in data, "Data should contain confidence field"
    assert 'model_info' in data, "Data should contain model_info field"

    # Verify model_info structure matches Go expectations
    model_info = data['model_info']
    assert 'name' in model_info, "model_info should contain name"
    assert 'version' in model_info, "model_info should contain version"
    assert 'accuracy' in model_info, "model_info should contain accuracy"

    print("✅ Go client resource prediction format - PASSED")
    return True

async def test_go_client_anomaly_detection():
    """Test anomaly detection request format from Go client"""
    print("🔍 Testing Go client anomaly detection format...")

    # This is the format sent by Go AIIntegrationLayer.DetectAnomalies
    request = {
        'id': 'anomaly-test-1',
        'service': 'anomaly_detection',
        'method': 'detect',
        'data': {
            'resource_id': 'vm-456',
            'metric_type': 'cpu_usage',
            'data_points': [
                {
                    'timestamp': '2024-01-01T10:00:00Z',
                    'value': 95.5,  # High CPU usage - should trigger anomaly
                    'metadata': {'cpu_usage': 95.5, 'memory_usage': 87.2, 'disk_usage': 65.1}
                }
            ],
            'sensitivity': 0.1,
            'context': {'node': 'worker-2', 'alert_threshold': 90.0}
        }
    }

    result = await process_request(request)

    # Verify Go client expected response format
    assert result['success'] == True, "Request should succeed"
    assert 'data' in result, "Response should contain data field"
    assert 'confidence' in result, "Response should contain confidence field"

    # Verify AnomalyDetectionResponse structure
    data = result['data']
    assert 'anomalies' in data, "Data should contain anomalies array"
    assert 'overall_score' in data, "Data should contain overall_score field"
    assert 'baseline' in data, "Data should contain baseline field"
    assert 'model_info' in data, "Data should contain model_info field"

    # Verify anomalies structure if any are detected
    if len(data['anomalies']) > 0:
        anomaly = data['anomalies'][0]
        assert 'timestamp' in anomaly, "Anomaly should contain timestamp"
        assert 'anomaly_type' in anomaly, "Anomaly should contain anomaly_type"
        assert 'severity' in anomaly, "Anomaly should contain severity"
        assert 'score' in anomaly, "Anomaly should contain score"
        assert 'description' in anomaly, "Anomaly should contain description"

    print("✅ Go client anomaly detection format - PASSED")
    return True

async def test_go_client_performance_optimization():
    """Test performance optimization request format from Go client"""
    print("🔍 Testing Go client performance optimization format...")

    # This is the format sent by Go AIIntegrationLayer.OptimizePerformance
    request = {
        'id': 'perf-test-1',
        'service': 'performance_optimization',
        'method': 'optimize_cluster',
        'data': {
            'cluster_id': 'prod-cluster-1',
            'cluster_data': {
                'nodes': 5,
                'total_cpu': 200,
                'total_memory': 512,
                'avg_latency': 125.5,
                'throughput': 850.2
            },
            'goals': ['minimize_latency', 'maximize_throughput', 'optimize_cost'],
            'constraints': {
                'max_cost_increase': 0.15,
                'min_availability': 0.999
            }
        }
    }

    result = await process_request(request)

    # Verify Go client expected response format
    assert result['success'] == True, "Request should succeed"
    assert 'data' in result, "Response should contain data field"

    # Verify PerformanceOptimizationResponse structure
    data = result['data']
    assert 'recommendations' in data, "Data should contain recommendations array"
    assert 'expected_gains' in data, "Data should contain expected_gains field"
    assert 'risk_assessment' in data, "Data should contain risk_assessment field"
    assert 'confidence' in data, "Data should contain confidence field"

    # Verify recommendations structure
    if len(data['recommendations']) > 0:
        rec = data['recommendations'][0]
        assert 'type' in rec, "Recommendation should contain type"
        assert 'target' in rec, "Recommendation should contain target"
        assert 'action' in rec, "Recommendation should contain action"
        assert 'priority' in rec, "Recommendation should contain priority"
        assert 'impact' in rec, "Recommendation should contain impact"
        assert 'confidence' in rec, "Recommendation should contain confidence"

    # Verify risk_assessment structure
    risk = data['risk_assessment']
    assert 'overall_risk' in risk, "Risk assessment should contain overall_risk"
    assert 'risk_factors' in risk, "Risk assessment should contain risk_factors"
    assert 'mitigations' in risk, "Risk assessment should contain mitigations"

    print("✅ Go client performance optimization format - PASSED")
    return True

async def test_go_client_workload_analysis():
    """Test workload analysis request format from Go client"""
    print("🔍 Testing Go client workload analysis format...")

    # This is the format sent by Go AIIntegrationLayer.AnalyzeWorkloadPattern
    request = {
        'id': 'workload-test-1',
        'service': 'workload_pattern_recognition',
        'method': 'analyze_patterns',
        'data': {
            'workload_id': 'web-app-workload',
            'data_points': [
                {'timestamp': '2024-01-01T10:00:00Z', 'cpu': 45.0, 'memory': 60.0, 'requests': 150},
                {'timestamp': '2024-01-01T10:05:00Z', 'cpu': 52.0, 'memory': 65.0, 'requests': 180},
                {'timestamp': '2024-01-01T10:10:00Z', 'cpu': 48.0, 'memory': 62.0, 'requests': 165}
            ],
            'analysis_window': 3600
        }
    }

    result = await process_request(request)

    # Verify Go client expected response format
    assert result['success'] == True, "Request should succeed"
    assert 'data' in result, "Response should contain data field"

    # Verify WorkloadPatternResponse structure
    data = result['data']
    assert 'patterns' in data, "Data should contain patterns array"
    assert 'classification' in data, "Data should contain classification field"
    assert 'seasonality' in data, "Data should contain seasonality field"
    assert 'recommendations' in data, "Data should contain recommendations field"
    assert 'confidence' in data, "Data should contain confidence field"

    # Verify patterns structure
    if len(data['patterns']) > 0:
        pattern = data['patterns'][0]
        assert 'type' in pattern, "Pattern should contain type"
        assert 'start_time' in pattern, "Pattern should contain start_time"
        assert 'end_time' in pattern, "Pattern should contain end_time"
        assert 'confidence' in pattern, "Pattern should contain confidence"

    # Verify seasonality structure
    seasonality = data['seasonality']
    assert 'has_seasonality' in seasonality, "Seasonality should contain has_seasonality"
    assert 'components' in seasonality, "Seasonality should contain components"

    print("✅ Go client workload analysis format - PASSED")
    return True

async def test_model_info_compatibility():
    """Test model info endpoint compatibility"""
    print("🔍 Testing model info endpoint compatibility...")

    # Test model info request - should match Go AIIntegrationLayer.GetModelInfo
    request = {
        'id': 'model-info-test',
        'service': 'model_management',
        'method': 'get_info',
        'data': {
            'model_type': 'resource_prediction'
        }
    }

    result = await process_request(request)

    assert result['success'] == True, "Request should succeed"
    assert 'data' in result, "Response should contain data field"

    # Verify ModelInfo structure
    data = result['data']
    assert 'name' in data, "ModelInfo should contain name"
    assert 'version' in data, "ModelInfo should contain version"
    assert 'training_data' in data, "ModelInfo should contain training_data"
    assert 'accuracy' in data, "ModelInfo should contain accuracy"
    assert 'last_trained' in data, "ModelInfo should contain last_trained"

    print("✅ Model info endpoint compatibility - PASSED")
    return True

async def main():
    """Run all Go integration compatibility tests"""
    print("=" * 60)
    print("🚀 GO INTEGRATION COMPATIBILITY TEST SUITE")
    print("=" * 60)

    tests = [
        ("Resource Prediction", test_go_client_resource_prediction),
        ("Anomaly Detection", test_go_client_anomaly_detection),
        ("Performance Optimization", test_go_client_performance_optimization),
        ("Workload Analysis", test_go_client_workload_analysis),
        ("Model Info", test_model_info_compatibility)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if await test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED - {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✅ ALL GO INTEGRATION TESTS PASSED!")
        print("🎉 AI Engine is fully compatible with Go client!")
    else:
        print(f"⚠️ {failed} tests failed. Please review the errors above.")

    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)