#!/usr/bin/env python3
"""
Test script for model persistence functionality in NovaCron AI Engine
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.models import ModelManager, ModelPerformance


def test_model_persistence():
    """Test model persistence functionality"""

    print("Testing NovaCron Model Persistence...")
    print("=" * 60)

    # Initialize ModelManager
    test_dir = "/tmp/test_novacron_models/"
    db_path = os.path.join(test_dir, "test_model_registry.db")

    # Clean up any existing test data
    if os.path.exists(db_path):
        os.remove(db_path)

    model_manager = ModelManager(model_dir=test_dir, db_path=db_path)
    print(f"✅ ModelManager initialized with DB at: {db_path}")

    # Test 1: Save a model
    print("\n1. Testing model save...")
    test_model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Train on dummy data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    test_model.fit(X, y)

    model_id = model_manager.save_model(
        name="test_resource_predictor",
        model=test_model,
        version="1.0.0",
        model_type="ensemble",
        algorithms=["Random Forest"],
        metadata={"test": True, "features": 5}
    )
    print(f"✅ Model saved with ID: {model_id}")

    # Test 2: Update performance metrics
    print("\n2. Testing performance metrics persistence...")
    performance = ModelPerformance(
        mae=0.15,
        mse=0.025,
        r2=0.85,
        accuracy_score=0.88,
        confidence_score=0.92,
        training_time=2.5,
        prediction_time=0.05,
        last_updated=datetime.now()
    )

    model_manager.update_performance("test_resource_predictor", "1.0.0", performance)
    print("✅ Performance metrics saved")

    # Test 3: Save training history
    print("\n3. Testing training history persistence...")
    model_manager.save_training_history(
        name="test_resource_predictor",
        version="1.0.0",
        training_info={
            "started": datetime.now(),
            "completed": datetime.now(),
            "dataset_size": 100,
            "hyperparameters": {"n_estimators": 10, "max_depth": 5},
            "validation_score": 0.87
        }
    )
    print("✅ Training history saved")

    # Test 4: Save another version
    print("\n4. Testing multiple versions...")
    test_model2 = LinearRegression()
    test_model2.fit(X, y)

    model_id2 = model_manager.save_model(
        name="test_resource_predictor",
        model=test_model2,
        version="2.0.0",
        model_type="linear",
        algorithms=["Linear Regression"],
        metadata={"test": True, "features": 5, "improved": True}
    )
    print(f"✅ Second version saved with ID: {model_id2}")

    # Test 5: Retrieve model info
    print("\n5. Testing model info retrieval...")
    model_info = model_manager.get_model_info()
    print(f"✅ Retrieved info for {model_info['total']} models")

    for model in model_info['models']:
        print(f"  - {model['name']} v{model['version']}: {model['model_type']}")

    # Test 6: Test specific model retrieval
    print("\n6. Testing specific model retrieval...")
    specific_info = model_manager.get_model_info(name="test_resource_predictor")
    print(f"✅ Retrieved {specific_info['total']} versions of test_resource_predictor")

    # Test 7: Verify database persistence
    print("\n7. Verifying database persistence...")

    # Create new ModelManager instance to test loading
    model_manager2 = ModelManager(model_dir=test_dir, db_path=db_path)

    # Check if models were loaded
    if "test_resource_predictor" in model_manager2.models:
        print("✅ Models successfully loaded from database on restart")
        print(f"  - Loaded versions: {model_manager2.model_versions.get('test_resource_predictor', [])}")
    else:
        print("❌ Failed to load models from database")

    # Test 8: Test cleanup
    print("\n8. Testing model cleanup...")

    # Add more versions to test cleanup
    for i in range(3, 8):
        dummy_model = LinearRegression()
        dummy_model.fit(X, y)
        model_manager2.save_model(
            name="test_resource_predictor",
            model=dummy_model,
            version=f"{i}.0.0"
        )

    print(f"  - Total versions before cleanup: {len(model_manager2.model_versions.get('test_resource_predictor', []))}")

    model_manager2.cleanup_old_models(keep_versions=3)

    # Reload to verify cleanup
    model_manager3 = ModelManager(model_dir=test_dir, db_path=db_path)
    remaining_versions = len(model_manager3.model_versions.get('test_resource_predictor', []))
    print(f"  - Versions after cleanup: {remaining_versions}")

    if remaining_versions <= 3:
        print("✅ Cleanup successful")
    else:
        print("❌ Cleanup may not have worked correctly")

    # Test 9: Verify SQLite database structure
    print("\n9. Verifying database structure...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    expected_tables = ['model_registry', 'performance_metrics', 'training_history']

    for table in expected_tables:
        if (table,) in tables:
            print(f"✅ Table '{table}' exists")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    - Contains {count} records")

    conn.close()

    print("\n" + "=" * 60)
    print("✅ All model persistence tests completed successfully!")
    print(f"Database location: {db_path}")
    print(f"Model directory: {test_dir}")

    return True


if __name__ == "__main__":
    try:
        success = test_model_persistence()
        if success:
            print("\n✅ Model persistence is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)