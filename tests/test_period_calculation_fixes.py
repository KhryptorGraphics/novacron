#!/usr/bin/env python3
"""
Test script to verify the frequency/seasonal_period semantics fixes
in workload_pattern_recognition.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the ai_engine directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))

from workload_pattern_recognition import WorkloadPatternRecognizer

def create_synthetic_data(hours=168, period_hours=24, amplitude=0.3, base_level=0.5):
    """Create synthetic workload data with known periodic pattern

    Args:
        hours: Total hours of data
        period_hours: Period of the pattern in hours
        amplitude: Amplitude of the periodic component
        base_level: Base level of the signal
    """
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours)]

    # Create periodic CPU usage pattern
    t = np.arange(hours)
    cpu_usage = base_level + amplitude * np.sin(2 * np.pi * t / period_hours)

    # Add some noise
    noise = np.random.normal(0, 0.1, hours)
    cpu_usage = np.clip(cpu_usage + noise, 0, 1)

    # Create simple patterns for other metrics
    memory_usage = np.random.uniform(0.3, 0.7, hours)
    io_usage = np.random.uniform(0.1, 0.4, hours)
    network_usage = np.random.uniform(0.05, 0.3, hours)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'io_usage': io_usage,
        'network_usage': network_usage
    })

    return data

def test_daily_pattern():
    """Test recognition of daily (24-hour) pattern"""
    print("=== Testing Daily Pattern Recognition ===")

    # Create 7 days of data with 24-hour period
    data = create_synthetic_data(hours=168, period_hours=24)

    recognizer = WorkloadPatternRecognizer()
    pattern = recognizer.analyze_workload("test_vm_daily", data)

    print(f"Pattern ID: {pattern.pattern_id}")
    print(f"Workload Type: {pattern.workload_type}")
    print(f"Pattern Type: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.3f}")
    print(f"Frequency: {pattern.frequency} cycles/hour")
    print(f"Seasonal Period: {pattern.seasonal_period} hours")
    print(f"Expected Period: 24 hours")

    # Verify the period is close to expected
    if pattern.seasonal_period:
        period_error = abs(pattern.seasonal_period - 24) / 24
        print(f"Period Error: {period_error:.1%}")

        if period_error < 0.2:  # Allow 20% tolerance
            print("✅ PASS: Period detection within tolerance")
        else:
            print("❌ FAIL: Period detection outside tolerance")
    else:
        print("❌ FAIL: No seasonal period detected")

    return pattern

def test_weekly_pattern():
    """Test recognition of weekly (168-hour) pattern"""
    print("\n=== Testing Weekly Pattern Recognition ===")

    # Create 4 weeks of data with 168-hour (weekly) period
    data = create_synthetic_data(hours=672, period_hours=168)

    recognizer = WorkloadPatternRecognizer()
    pattern = recognizer.analyze_workload("test_vm_weekly", data)

    print(f"Pattern ID: {pattern.pattern_id}")
    print(f"Workload Type: {pattern.workload_type}")
    print(f"Pattern Type: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.3f}")
    print(f"Frequency: {pattern.frequency} cycles/hour")
    print(f"Seasonal Period: {pattern.seasonal_period} hours")
    print(f"Expected Period: 168 hours")

    # Verify the period is close to expected
    if pattern.seasonal_period:
        period_error = abs(pattern.seasonal_period - 168) / 168
        print(f"Period Error: {period_error:.1%}")

        if period_error < 0.3:  # Allow 30% tolerance for longer periods
            print("✅ PASS: Period detection within tolerance")
        else:
            print("❌ FAIL: Period detection outside tolerance")
    else:
        print("❌ FAIL: No seasonal period detected")

    return pattern

def test_no_pattern():
    """Test handling of random data with no pattern"""
    print("\n=== Testing Random Data (No Pattern) ===")

    # Create random data with no periodic pattern
    hours = 168
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours)]
    cpu_usage = np.random.uniform(0.2, 0.8, hours)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': np.random.uniform(0.3, 0.7, hours),
        'io_usage': np.random.uniform(0.1, 0.4, hours),
        'network_usage': np.random.uniform(0.05, 0.3, hours)
    })

    recognizer = WorkloadPatternRecognizer()
    pattern = recognizer.analyze_workload("test_vm_random", data)

    print(f"Pattern ID: {pattern.pattern_id}")
    print(f"Workload Type: {pattern.workload_type}")
    print(f"Pattern Type: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.3f}")
    print(f"Frequency: {pattern.frequency}")
    print(f"Seasonal Period: {pattern.seasonal_period}")

    # For random data, we expect no strong seasonal pattern
    if pattern.seasonal_period is None:
        print("✅ PASS: No seasonal period detected (expected for random data)")
    else:
        print(f"⚠️  WARNING: Seasonal period detected in random data: {pattern.seasonal_period} hours")

    return pattern

def test_frequency_calculation():
    """Test direct frequency calculation methods"""
    print("\n=== Testing Frequency Calculation Methods ===")

    recognizer = WorkloadPatternRecognizer()

    # Test cases: period_samples -> expected_frequency
    test_cases = [
        (24, 1.0/24),    # Daily pattern: 1 cycle per 24 hours
        (168, 1.0/168),  # Weekly pattern: 1 cycle per 168 hours
        (12, 1.0/12),    # 12-hour pattern: 1 cycle per 12 hours
        (None, None),    # No pattern
        (0, None),       # Invalid period
    ]

    for period_samples, expected_freq in test_cases:
        freq = recognizer._calculate_frequency(period_samples)
        period_hours = recognizer._calculate_seasonal_period_hours(period_samples)

        print(f"Period samples: {period_samples}")
        print(f"Calculated frequency: {freq}")
        print(f"Expected frequency: {expected_freq}")
        print(f"Calculated period hours: {period_hours}")

        # Special case for invalid periods (None or 0)
        if period_samples in (None, 0):
            if freq is None and period_hours is None:
                print("✅ PASS: Calculations correct")
            else:
                print("❌ FAIL: Calculations incorrect")
        else:
            if freq == expected_freq and period_hours == period_samples:
                print("✅ PASS: Calculations correct")
            else:
                print("❌ FAIL: Calculations incorrect")
        print()

def main():
    """Run all tests"""
    print("Testing Workload Pattern Recognition Frequency/Period Semantics")
    print("=" * 60)

    try:
        # Run individual tests
        test_frequency_calculation()
        test_daily_pattern()
        test_weekly_pattern()
        test_no_pattern()

        print("\n" + "=" * 60)
        print("Testing completed. Check results above.")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())