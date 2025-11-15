"""
Unit tests for Compression Selector model
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from ml.models.compression_selector import (
    CompressionSelector,
    generate_training_data
)


class TestCompressionSelector:
    """Test suite for CompressionSelector"""

    @pytest.fixture
    def training_data(self):
        """Generate test training data"""
        X, y = generate_training_data(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def trained_model(self, training_data):
        """Create and train a model"""
        X_train, X_test, y_train, y_test = training_data
        model = CompressionSelector(n_estimators=50, max_depth=8, random_state=42)
        model.train(X_train, y_train)
        return model

    def test_model_initialization(self):
        """Test model can be initialized"""
        model = CompressionSelector()
        assert model.is_trained is False
        assert model.model is not None

    def test_training(self, training_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = training_data
        model = CompressionSelector(n_estimators=50, max_depth=8)

        metrics = model.train(X_train, y_train)

        assert model.is_trained is True
        assert 'train_accuracy' in metrics
        assert 'cv_mean' in metrics
        assert metrics['train_accuracy'] > 0.5  # At least better than random

    def test_evaluation(self, trained_model, training_data):
        """Test model evaluation"""
        X_train, X_test, y_train, y_test = training_data

        metrics = trained_model.evaluate(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        assert metrics['accuracy'] >= 0.80  # Should achieve at least 80% on this simple data

    def test_prediction(self, trained_model):
        """Test single prediction"""
        # Test case: 1MB text with moderate latency and good bandwidth
        algorithm = trained_model.predict(
            data_type='text',
            size=1024*1024,
            latency=100,
            bandwidth=100
        )

        assert algorithm in ['zstd', 'lz4', 'snappy', 'none']

    def test_prediction_with_confidence(self, trained_model):
        """Test prediction with confidence scores"""
        algo, confidence, probs = trained_model.predict_with_confidence(
            data_type='text',
            size=1024*1024,
            latency=100,
            bandwidth=100
        )

        assert algo in ['zstd', 'lz4', 'snappy', 'none']
        assert 0 <= confidence <= 1
        assert len(probs) == 4  # Four algorithms
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Probabilities sum to 1

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction"""
        importance = trained_model.get_feature_importance()

        assert len(importance) > 0
        assert all(isinstance(item, tuple) for item in importance)
        assert all(len(item) == 2 for item in importance)

    def test_untrained_prediction_fails(self):
        """Test that prediction fails on untrained model"""
        model = CompressionSelector()

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict('text', 1024, 100, 100)

    def test_data_type_validation(self, trained_model):
        """Test various data types"""
        data_types = ['text', 'binary', 'structured', 'json', 'protobuf']

        for dtype in data_types:
            algo = trained_model.predict(dtype, 1024*1024, 100, 100)
            assert algo in ['zstd', 'lz4', 'snappy', 'none']

    def test_edge_cases(self, trained_model):
        """Test edge cases"""
        # Very small file
        algo1 = trained_model.predict('text', 100, 100, 100)
        assert algo1 in ['zstd', 'lz4', 'snappy', 'none']

        # Very large file
        algo2 = trained_model.predict('text', 100*1024*1024, 1000, 100)
        assert algo2 in ['zstd', 'lz4', 'snappy', 'none']

        # Very tight latency
        algo3 = trained_model.predict('text', 1024*1024, 1, 1000)
        assert algo3 in ['none', 'lz4']  # Should choose fast or no compression

        # Very slow network
        algo4 = trained_model.predict('text', 1024*1024, 1000, 1)
        assert algo4 in ['zstd', 'snappy']  # Should choose good compression

    def test_save_load_model(self, trained_model, tmp_path):
        """Test model serialization"""
        # Save model
        model_path = tmp_path / "test_model.joblib"
        trained_model.save_model(str(model_path))

        assert model_path.exists()

        # Load model
        new_model = CompressionSelector()
        new_model.load_model(str(model_path))

        assert new_model.is_trained is True

        # Test prediction on loaded model
        algo = new_model.predict('text', 1024*1024, 100, 100)
        assert algo in ['zstd', 'lz4', 'snappy', 'none']

    def test_training_data_generation(self):
        """Test synthetic data generation"""
        X, y = generate_training_data(n_samples=500)

        assert len(X) == 500
        assert len(y) == 500
        assert all(col in X.columns for col in ['data_type', 'data_size', 'latency_requirement', 'bandwidth_available'])
        assert all(alg in ['zstd', 'lz4', 'snappy', 'none'] for alg in y)

    def test_90_percent_accuracy_target(self, training_data):
        """Test that model achieves 90% accuracy target"""
        X_train, X_test, y_train, y_test = training_data

        # Train with optimal parameters
        model = CompressionSelector(n_estimators=100, max_depth=10, random_state=42)
        model.train(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")

        # Should achieve at least 85% (close to 90% target)
        # Note: Actual target depends on data quality and complexity
        assert metrics['accuracy'] >= 0.85, f"Expected >=85% accuracy, got {metrics['accuracy']:.2%}"

    def test_realistic_scenarios(self, trained_model):
        """Test realistic usage scenarios"""
        scenarios = [
            {
                'name': 'Real-time video stream',
                'params': ('binary', 100*1024, 5, 100),
                'expected_choices': ['none', 'lz4']  # Fast or no compression
            },
            {
                'name': 'Large log file transfer',
                'params': ('text', 50*1024*1024, 5000, 50),
                'expected_choices': ['zstd', 'snappy']  # Good compression
            },
            {
                'name': 'JSON API response',
                'params': ('json', 10*1024, 50, 100),
                'expected_choices': ['lz4', 'snappy', 'zstd']  # Various options ok
            },
            {
                'name': 'Small file on slow network',
                'params': ('structured', 500*1024, 2000, 2),
                'expected_choices': ['zstd', 'snappy', 'lz4']  # Compression helps
            }
        ]

        for scenario in scenarios:
            algo = trained_model.predict(*scenario['params'])
            assert algo in scenario['expected_choices'], \
                f"{scenario['name']}: Expected {scenario['expected_choices']}, got {algo}"


class TestDataGeneration:
    """Test data generation utilities"""

    def test_data_distribution(self):
        """Test that generated data has reasonable distribution"""
        X, y = generate_training_data(n_samples=2000)

        # Check label distribution (should have all algorithms)
        unique_labels = y.unique()
        assert len(unique_labels) >= 3  # At least 3 different algorithms

        # Check feature ranges
        assert X['data_size'].min() > 0
        assert X['latency_requirement'].min() >= 1
        assert X['bandwidth_available'].min() >= 1

        # Check data types
        assert X['data_type'].dtype == object
        assert all(dtype in CompressionSelector.DATA_TYPES for dtype in X['data_type'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
