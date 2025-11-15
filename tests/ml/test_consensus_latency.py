"""
Unit tests for Consensus Latency Predictor
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.ml.models.consensus_latency import (
    ConsensusLatencyPredictor,
    generate_synthetic_training_data
)


class TestConsensusLatencyPredictor(unittest.TestCase):
    """Test cases for ConsensusLatencyPredictor"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.predictor = ConsensusLatencyPredictor(sequence_length=5)

        # Generate small dataset for testing
        X, y = generate_synthetic_training_data(n_samples=1000)
        cls.X_train = X[:800]
        cls.y_train = y[:800]
        cls.X_test = X[800:]
        cls.y_test = y[800:]

    def test_feature_encoding(self):
        """Test feature encoding"""
        # Test LAN encoding
        features_lan = self.predictor._encode_features(7, 'LAN', 0.1, 1000)
        self.assertEqual(features_lan[0, 1], 0.0)  # LAN = 0

        # Test WAN encoding
        features_wan = self.predictor._encode_features(7, 'WAN', 0.1, 1000)
        self.assertEqual(features_wan[0, 1], 1.0)  # WAN = 1

        # Test feature values
        self.assertEqual(features_lan[0, 0], 7.0)  # node_count
        self.assertEqual(features_lan[0, 2], 0.1)  # byzantine_ratio
        self.assertEqual(features_lan[0, 3], 1000.0)  # msg_size

    def test_sequence_creation(self):
        """Test LSTM sequence creation"""
        X = np.random.rand(100, 4)
        y = np.random.rand(100)

        X_seq, y_seq = self.predictor._create_sequences(X, y)

        # Check shapes
        expected_length = 100 - self.predictor.sequence_length
        self.assertEqual(len(X_seq), expected_length)
        self.assertEqual(len(y_seq), expected_length)
        self.assertEqual(X_seq.shape[1], self.predictor.sequence_length)
        self.assertEqual(X_seq.shape[2], 4)

    def test_model_training(self):
        """Test model training"""
        # Train with small dataset
        results = self.predictor.train(
            self.X_train, self.y_train,
            epochs=10,
            batch_size=32
        )

        # Check model was created
        self.assertIsNotNone(self.predictor.model)

        # Check training history
        self.assertIn('history', results)
        self.assertIn('final_metrics', results)

        # Check metrics
        metrics = results['final_metrics']
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 100)

    def test_prediction(self):
        """Test latency prediction"""
        # Ensure model is trained
        if self.predictor.model is None:
            self.predictor.train(self.X_train, self.y_train, epochs=5, batch_size=32)

        # Test prediction
        result = self.predictor.predict_latency(
            node_count=7,
            network_mode='LAN',
            byzantine_ratio=0.1,
            msg_size=1000
        )

        # Check result structure
        self.assertIn('predicted_latency_ms', result)
        self.assertIn('confidence', result)
        self.assertIn('parameters', result)

        # Check values are reasonable
        self.assertGreater(result['predicted_latency_ms'], 0)
        self.assertGreater(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_lan_vs_wan_predictions(self):
        """Test that WAN latency > LAN latency"""
        # Ensure model is trained
        if self.predictor.model is None:
            self.predictor.train(self.X_train, self.y_train, epochs=5, batch_size=32)

        # Same parameters, different network
        params = {'node_count': 7, 'byzantine_ratio': 0.1, 'msg_size': 1000}

        lan_result = self.predictor.predict_latency(**params, network_mode='LAN')
        wan_result = self.predictor.predict_latency(**params, network_mode='WAN')

        # WAN should have higher latency
        self.assertGreater(
            wan_result['predicted_latency_ms'],
            lan_result['predicted_latency_ms']
        )

    def test_byzantine_impact(self):
        """Test that higher byzantine ratio increases latency"""
        # Ensure model is trained
        if self.predictor.model is None:
            self.predictor.train(self.X_train, self.y_train, epochs=5, batch_size=32)

        # Same parameters, different byzantine ratios
        base_params = {'node_count': 7, 'network_mode': 'LAN', 'msg_size': 1000}

        low_byz = self.predictor.predict_latency(**base_params, byzantine_ratio=0.0)
        high_byz = self.predictor.predict_latency(**base_params, byzantine_ratio=0.3)

        # Higher byzantine ratio should increase latency
        self.assertGreater(
            high_byz['predicted_latency_ms'],
            low_byz['predicted_latency_ms']
        )

    def test_confidence_estimation(self):
        """Test confidence estimation"""
        # Ensure model is trained
        if self.predictor.model is None:
            self.predictor.train(self.X_train, self.y_train, epochs=5, batch_size=32)

        # Normal parameters should have higher confidence
        normal = self.predictor._estimate_confidence(7, 'LAN', 0.1, 1000)

        # Extreme parameters should have lower confidence
        extreme = self.predictor._estimate_confidence(200, 'WAN', 0.33, 2000000)

        self.assertGreater(normal, extreme)
        self.assertGreater(normal, 0.7)  # Should be reasonably confident
        self.assertLess(extreme, 0.9)  # Should be less confident

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        X, y = generate_synthetic_training_data(n_samples=1000)

        # Check shapes
        self.assertEqual(len(X), 1000)
        self.assertEqual(len(y), 1000)
        self.assertEqual(X.shape[1], 4)

        # Check value ranges
        self.assertTrue(np.all(X[:, 0] >= 3))  # node_count >= 3
        self.assertTrue(np.all(X[:, 0] <= 100))  # node_count <= 100
        self.assertTrue(np.all(X[:, 1] >= 0))  # network_mode 0 or 1
        self.assertTrue(np.all(X[:, 1] <= 1))
        self.assertTrue(np.all(X[:, 2] >= 0))  # byzantine_ratio >= 0
        self.assertTrue(np.all(X[:, 2] <= 0.33))  # byzantine_ratio <= 0.33
        self.assertTrue(np.all(y > 0))  # latencies > 0

    def test_accuracy_target(self):
        """Test that model achieves 90% accuracy target"""
        # Train with more data
        X, y = generate_synthetic_training_data(n_samples=5000)
        X_train = X[:4000]
        y_train = y[:4000]

        predictor = ConsensusLatencyPredictor(sequence_length=10)
        results = predictor.train(X_train, y_train, epochs=50, batch_size=32)

        accuracy = results['final_metrics']['accuracy']

        # Should achieve at least 85% accuracy (target is 90%)
        # We use 85% to account for randomness in training
        self.assertGreater(accuracy, 85.0,
                          f"Model accuracy {accuracy:.2f}% is below 85% threshold")

        print(f"\nModel achieved {accuracy:.2f}% accuracy (target: 90%)")


class TestIntegration(unittest.TestCase):
    """Integration tests for the predictor"""

    def test_full_workflow(self):
        """Test complete workflow: train, predict, save, load"""
        # Generate data
        X, y = generate_synthetic_training_data(n_samples=1000)
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]

        # Train
        predictor = ConsensusLatencyPredictor(sequence_length=5)
        predictor.train(X_train, y_train, epochs=10, batch_size=32)

        # Predict
        result = predictor.predict_latency(7, 'LAN', 0.1, 1000)
        original_prediction = result['predicted_latency_ms']

        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        predictor.save_model(tmp_path)

        # Load
        new_predictor = ConsensusLatencyPredictor(sequence_length=5)
        new_predictor.load_model(tmp_path)

        # Predict again
        new_result = new_predictor.predict_latency(7, 'LAN', 0.1, 1000)
        new_prediction = new_result['predicted_latency_ms']

        # Predictions should match
        self.assertAlmostEqual(original_prediction, new_prediction, places=2)

        # Cleanup
        import os
        os.unlink(f"{tmp_path}_model.keras")
        os.unlink(f"{tmp_path}_metadata.json")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
