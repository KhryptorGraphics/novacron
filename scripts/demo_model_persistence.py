#!/usr/bin/env python3
"""
Demo script showing model persistence functionality in NovaCron AI Engine
"""

import os
import sys
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.models import ModelManager, ModelPerformance


def demo_model_persistence():
    """Demonstrate complete model persistence workflow"""

    print("NovaCron AI Engine - Model Persistence Demo")
    print("=" * 50)

    # Initialize ModelManager with default paths (like in production)
    model_dir = "/tmp/novacron_demo_models/"
    db_path = "/tmp/novacron_demo_models/model_registry.db"

    # Clean up any existing demo data
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"Initializing ModelManager...")
    print(f"  Model Directory: {model_dir}")
    print(f"  Database Path: {db_path}")

    model_manager = ModelManager(model_dir=model_dir, db_path=db_path)
    print("✓ ModelManager initialized with persistence enabled")

    # Demo 1: Save multiple models
    print("\n1. Saving Models to Database")
    print("-" * 30)

    # Create and train sample models
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000)

    models_to_save = [
        {
            "name": "resource_predictor",
            "model": RandomForestRegressor(n_estimators=50, random_state=42),
            "version": "1.0.0",
            "type": "ensemble",
            "algorithms": ["Random Forest"],
            "metadata": {"features": 10, "samples": 1000, "purpose": "resource_prediction"}
        },
        {
            "name": "anomaly_detector",
            "model": RandomForestRegressor(n_estimators=30, random_state=42),
            "version": "2.1.0",
            "type": "anomaly_detection",
            "algorithms": ["Isolation Forest", "Random Forest"],
            "metadata": {"features": 10, "contamination": 0.1, "purpose": "anomaly_detection"}
        },
        {
            "name": "workload_predictor",
            "model": LinearRegression(),
            "version": "1.2.0",
            "type": "time_series",
            "algorithms": ["Linear Regression"],
            "metadata": {"features": 10, "horizon": 24, "purpose": "workload_prediction"}
        }
    ]

    for model_info in models_to_save:
        # Train model
        model_info["model"].fit(X, y)

        # Save to database
        model_id = model_manager.save_model(
            name=model_info["name"],
            model=model_info["model"],
            version=model_info["version"],
            model_type=model_info["type"],
            algorithms=model_info["algorithms"],
            metadata=model_info["metadata"]
        )

        print(f"✓ Saved {model_info['name']} v{model_info['version']} (ID: {model_id})")

    # Demo 2: Add performance metrics
    print("\n2. Recording Performance Metrics")
    print("-" * 35)

    performance_data = [
        ("resource_predictor", "1.0.0", 0.12, 0.92, 45.2, 0.89),
        ("anomaly_detector", "2.1.0", 0.08, 0.94, 32.1, 0.91),
        ("workload_predictor", "1.2.0", 0.15, 0.87, 12.3, 0.85)
    ]

    for name, version, mae, accuracy, train_time, confidence in performance_data:
        performance = ModelPerformance(
            mae=mae,
            mse=mae**2,  # Approximation
            r2=accuracy,
            accuracy_score=accuracy,
            confidence_score=confidence,
            training_time=train_time,
            prediction_time=0.05,
            last_updated=datetime.now()
        )

        model_manager.update_performance(name, version, performance)
        print(f"✓ Recorded performance for {name} v{version} (Accuracy: {accuracy:.2f})")

    # Demo 3: Add training history
    print("\n3. Recording Training History")
    print("-" * 32)

    training_histories = [
        ("resource_predictor", "1.0.0", {"n_estimators": 50, "max_depth": 10}, 0.89),
        ("anomaly_detector", "2.1.0", {"contamination": 0.1, "n_estimators": 30}, 0.91),
        ("workload_predictor", "1.2.0", {"fit_intercept": True, "normalize": False}, 0.85)
    ]

    for name, version, hyperparams, validation_score in training_histories:
        model_manager.save_training_history(
            name=name,
            version=version,
            training_info={
                "started": datetime.now(),
                "completed": datetime.now(),
                "dataset_size": 1000,
                "hyperparameters": hyperparams,
                "validation_score": validation_score
            }
        )
        print(f"✓ Recorded training history for {name} v{version}")

    # Demo 4: Retrieve model information
    print("\n4. Retrieving Model Information")
    print("-" * 33)

    all_models = model_manager.get_model_info()
    print(f"✓ Retrieved information for {all_models['total']} models:")

    for model in all_models['models']:
        print(f"  • {model['name']} v{model['version']}")
        print(f"    Type: {model['model_type']}")
        print(f"    Active: {'Yes' if model['is_active'] else 'No'}")
        print(f"    Avg Accuracy: {model.get('avg_accuracy', 'N/A')}")
        print(f"    Created: {model['created_at']}")
        print()

    # Demo 5: Test persistence across "restart"
    print("5. Testing Persistence Across Restart")
    print("-" * 38)

    # Create new ModelManager instance to simulate restart
    print("Simulating application restart...")
    model_manager_2 = ModelManager(model_dir=model_dir, db_path=db_path)

    # Check if data persisted
    reloaded_models = model_manager_2.get_model_info()
    print(f"✓ After restart: {reloaded_models['total']} models loaded from database")

    for model in reloaded_models['models']:
        print(f"  • {model['name']} v{model['version']} - Type: {model['model_type']}")

    # Demo 6: Database inspection
    print("\n6. Database Inspection")
    print("-" * 22)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check table contents
    tables = ['model_registry', 'performance_metrics', 'training_history']

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"✓ {table}: {count} records")

    # Show sample data
    print("\nSample model registry data:")
    cursor.execute("""
        SELECT name, version, model_type, is_active, created_at
        FROM model_registry
        ORDER BY created_at DESC
        LIMIT 3
    """)

    for row in cursor.fetchall():
        name, version, model_type, is_active, created_at = row
        status = "Active" if is_active else "Inactive"
        print(f"  • {name} v{version} ({model_type}) - {status}")

    conn.close()

    # Demo 7: Version management
    print("\n7. Version Management Demo")
    print("-" * 27)

    # Add more versions of resource_predictor
    for i in range(2, 6):
        new_model = RandomForestRegressor(n_estimators=20+i*10, random_state=42+i)
        new_model.fit(X, y)

        model_id = model_manager_2.save_model(
            name="resource_predictor",
            model=new_model,
            version=f"{i}.0.0",
            model_type="ensemble",
            algorithms=["Random Forest"],
            metadata={"version": f"{i}.0.0", "improved": True}
        )
        print(f"✓ Added resource_predictor v{i}.0.0")

    # Show all versions
    resource_predictor_info = model_manager_2.get_model_info(name="resource_predictor")
    print(f"\nTotal resource_predictor versions: {resource_predictor_info['total']}")

    # Clean up old versions (keep only 3)
    print("\nCleaning up old versions (keeping 3 most recent)...")
    model_manager_2.cleanup_old_models(keep_versions=3)

    # Check remaining versions
    final_info = model_manager_2.get_model_info(name="resource_predictor")
    print(f"✓ After cleanup: {final_info['total']} versions remaining")

    for model in final_info['models']:
        print(f"  • v{model['version']} ({'Active' if model['is_active'] else 'Inactive'})")

    print(f"\n{'='*50}")
    print("✅ Model Persistence Demo Complete!")
    print(f"\nPersistent storage locations:")
    print(f"  📂 Models: {model_dir}")
    print(f"  🗄️  Database: {db_path}")
    print(f"\nKey features demonstrated:")
    print(f"  ✓ Model storage and retrieval")
    print(f"  ✓ Performance metrics tracking")
    print(f"  ✓ Training history logging")
    print(f"  ✓ Cross-session persistence")
    print(f"  ✓ Version management")
    print(f"  ✓ Database-backed model registry")

    return True


if __name__ == "__main__":
    try:
        demo_model_persistence()
        print(f"\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)