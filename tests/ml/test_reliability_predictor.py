"""
Unit tests for Node Reliability Predictor
Target: 85% accuracy
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ml.models.reliability_predictor import ReliabilityPredictor, generate_training_data


class TestReliabilityPredictor(unittest.TestCase):
    """Test suite for ReliabilityPredictor"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.model = ReliabilityPredictor(state_size=4, learning_rate=0.001)

        # Generate small dataset for testing
        cls.X, cls.y = generate_training_data(1000)

        # Split data
        split_idx = int(0.8 * len(cls.X))
        cls.X_train, cls.X_test = cls.X[:split_idx], cls.X[split_idx:]
        cls.y_train, cls.y_test = cls.y[:split_idx], cls.y[split_idx:]

    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.target_model)
        self.assertEqual(self.model.state_size, 4)
        self.assertEqual(self.model.learning_rate, 0.001)

    def test_model_architecture(self):
        """Test model has correct architecture"""
        # Check input shape
        input_shape = self.model.model.input_shape
        self.assertEqual(input_shape[1], 4)

        # Check output shape
        output_shape = self.model.model.output_shape
        self.assertEqual(output_shape[1], 1)

    def test_predict_reliability(self):
        """Test single prediction"""
        reliability = self.model.predict_reliability(
            uptime=95.0,
            failure_rate=0.1,
            network_quality=0.9,
            distance=100
        )

        # Check output is valid probability
        self.assertGreaterEqual(reliability, 0.0)
        self.assertLessEqual(reliability, 1.0)
        self.assertIsInstance(reliability, float)

    def test_predict_batch(self):
        """Test batch prediction"""
        predictions = self.model.predict_batch(self.X_test[:10])

        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

    def test_normalize_state(self):
        """Test state normalization"""
        state = self.model._normalize_state(
            uptime=100.0,
            failure_rate=5.0,
            network_quality=0.8,
            distance=5000
        )

        self.assertEqual(state.shape, (1, 4))
        self.assertTrue(np.all(state >= 0.0))
        self.assertTrue(np.all(state <= 1.0))

    def test_normalize_edge_cases(self):
        """Test normalization with edge cases"""
        # Zero values
        state_zero = self.model._normalize_state(0, 0, 0, 0)
        self.assertTrue(np.all(state_zero >= 0.0))

        # Maximum values
        state_max = self.model._normalize_state(100, 100, 1.0, 20000)
        self.assertTrue(np.all(state_max <= 1.0))

    def test_training(self):
        """Test model training"""
        initial_weights = self.model.model.get_weights()[0].copy()

        history = self.model.train(
            self.X_train,
            self.y_train,
            epochs=5,
            batch_size=32
        )

        # Check weights changed
        new_weights = self.model.model.get_weights()[0]
        self.assertFalse(np.array_equal(initial_weights, new_weights))

        # Check history returned
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertEqual(len(history['loss']), 5)

    def test_evaluation(self):
        """Test model evaluation"""
        # Train briefly
        self.model.train(self.X_train, self.y_train, epochs=10, batch_size=32)

        metrics = self.model.evaluate(self.X_test, self.y_test)

        # Check all metrics present
        required_metrics = ['loss', 'mae', 'accuracy', 'rmse', 'r2']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

        # Check values are reasonable
        self.assertGreater(metrics['accuracy'], 0.0)
        self.assertLess(metrics['loss'], 1.0)

    def test_memory_operations(self):
        """Test experience replay memory"""
        state = np.array([[0.95, 0.1, 0.9, 0.05]])
        next_state = np.array([[0.94, 0.15, 0.88, 0.06]])

        initial_memory_size = len(self.model.memory)

        self.model.remember(state, 0.9, 1.0, next_state, False)

        self.assertEqual(len(self.model.memory), initial_memory_size + 1)

    def test_replay_training(self):
        """Test replay training"""
        # Add some experiences
        for i in range(50):
            state = np.random.rand(1, 4)
            next_state = np.random.rand(1, 4)
            self.model.remember(state, 0.8, 1.0, next_state, False)

        # Train on replay
        metrics = self.model.replay(batch_size=32)

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)

    def test_update_target_model(self):
        """Test target model update"""
        # Modify main model
        self.model.model.get_weights()[0][0][0] = 999.0

        # Update target
        self.model.update_target_model()

        # Check weights match
        main_weights = self.model.model.get_weights()
        target_weights = self.model.target_model.get_weights()

        for mw, tw in zip(main_weights, target_weights):
            self.assertTrue(np.array_equal(mw, tw))

    def test_high_reliability_prediction(self):
        """Test prediction for high reliability node"""
        reliability = self.model.predict_reliability(
            uptime=99.9,
            failure_rate=0.01,
            network_quality=0.95,
            distance=50
        )

        # Should predict high reliability (even untrained)
        # Just check it's a valid probability
        self.assertGreaterEqual(reliability, 0.0)
        self.assertLessEqual(reliability, 1.0)

    def test_low_reliability_prediction(self):
        """Test prediction for low reliability node"""
        reliability = self.model.predict_reliability(
            uptime=60.0,
            failure_rate=5.0,
            network_quality=0.3,
            distance=8000
        )

        # Should predict low reliability (even untrained)
        # Just check it's a valid probability
        self.assertGreaterEqual(reliability, 0.0)
        self.assertLessEqual(reliability, 1.0)


class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation"""

    def test_generate_training_data(self):
        """Test training data generation"""
        X, y = generate_training_data(1000)

        # Check shapes
        self.assertEqual(X.shape, (1000, 4))
        self.assertEqual(y.shape, (1000,))

        # Check feature ranges
        self.assertTrue(np.all(X >= 0.0))
        self.assertTrue(np.all(X <= 1.0))

        # Check label range
        self.assertTrue(np.all(y >= 0.0))
        self.assertTrue(np.all(y <= 1.0))

    def test_data_distribution(self):
        """Test generated data has reasonable distribution"""
        X, y = generate_training_data(10000)

        # Check mean reliability is reasonable (around 0.5-0.7)
        mean_reliability = np.mean(y)
        self.assertGreater(mean_reliability, 0.4)
        self.assertLess(mean_reliability, 0.8)

        # Check standard deviation
        std_reliability = np.std(y)
        self.assertGreater(std_reliability, 0.05)
        self.assertLess(std_reliability, 0.3)

    def test_reproducibility(self):
        """Test data generation is reproducible"""
        X1, y1 = generate_training_data(100)
        X2, y2 = generate_training_data(100)

        # Should be identical due to fixed random seed
        self.assertTrue(np.array_equal(X1, X2))
        self.assertTrue(np.array_equal(y1, y2))


class TestAccuracyTarget(unittest.TestCase):
    """Test model achieves 85% accuracy target"""

    def test_accuracy_target(self):
        """Test model can achieve 85% accuracy"""
        # Generate larger dataset
        X, y = generate_training_data(5000)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = ReliabilityPredictor(state_size=4, learning_rate=0.001)
        model.train(X_train, y_train, epochs=50, batch_size=32)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Check accuracy meets target
        self.assertGreaterEqual(
            metrics['accuracy'],
            0.85,
            f"Accuracy {metrics['accuracy']:.4f} below target 0.85"
        )

        print(f"\n✅ Accuracy target achieved: {metrics['accuracy']*100:.2f}%")
        print(f"Other metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
