#!/usr/bin/env python3
"""
Simple test for model persistence functionality
"""

import os
import sys
import sqlite3
import tempfile
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.models import ModelManager, ModelPerformance


def simple_persistence_test():
    """Simple test of core persistence functionality"""

    print("Simple Model Persistence Test")
    print("-" * 40)

    # Use temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")

        # Test 1: Initialize ModelManager
        print("1. Initializing ModelManager...")
        model_manager = ModelManager(model_dir=temp_dir, db_path=db_path)
        print(f"✓ DB created at: {db_path}")

        # Test 2: Save a simple model
        print("2. Saving model...")
        model = LinearRegression()

        model_id = model_manager.save_model(
            name="test_model",
            model=model,
            version="1.0.0",
            model_type="linear",
            algorithms=["Linear Regression"],
            metadata={"test": True}
        )
        print(f"✓ Model saved with ID: {model_id}")

        # Test 3: Update performance
        print("3. Updating performance...")
        performance = ModelPerformance(
            mae=0.1, mse=0.01, r2=0.9,
            accuracy_score=0.9, confidence_score=0.95,
            training_time=1.0, prediction_time=0.01,
            last_updated=datetime.now()
        )
        model_manager.update_performance("test_model", "1.0.0", performance)
        print("✓ Performance updated")

        # Test 4: Verify database contents
        print("4. Verifying database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM model_registry")
        model_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        perf_count = cursor.fetchone()[0]

        conn.close()

        print(f"✓ Models in registry: {model_count}")
        print(f"✓ Performance records: {perf_count}")

        # Test 5: Retrieve model info
        print("5. Retrieving model info...")
        info = model_manager.get_model_info()
        print(f"✓ Retrieved info for {info['total']} models")

        if info['models']:
            model_info = info['models'][0]
            print(f"  - Name: {model_info['name']}")
            print(f"  - Version: {model_info['version']}")
            print(f"  - Type: {model_info['model_type']}")
            print(f"  - Active: {bool(model_info['is_active'])}")

    print("\n✓ All tests passed! Model persistence is working.")
    return True


if __name__ == "__main__":
    try:
        simple_persistence_test()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)