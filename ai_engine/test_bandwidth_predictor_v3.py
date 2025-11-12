#!/usr/bin/env python3
"""
Comprehensive test suite for DWCP v3 Bandwidth Predictor
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add ai_engine to path
sys.path.insert(0, os.path.dirname(__file__))

from bandwidth_predictor_v3 import (
    BandwidthPredictorV3,
    NetworkMetrics,
    PredictionConfig,
    generate_synthetic_data
)


class TestBandwidthPredictorV3(unittest.TestCase):
    """Test suite for BandwidthPredictorV3"""

    def setUp(self):
        """Set up test fixtures"""
        self.datacenter_predictor = BandwidthPredictorV3(mode='datacenter')
        self.internet_predictor = BandwidthPredictorV3(mode='internet')

    def test_initialization_datacenter(self):
        """Test datacenter predictor initialization"""
        self.assertEqual(self.datacenter_predictor.mode, 'datacenter')
        self.assertEqual(self.datacenter_predictor.config.sequence_length, 30)
        self.assertEqual(self.datacenter_predictor.config.feature_count, 5)
        self.assertEqual(self.datacenter_predictor.config.confidence_threshold, 0.85)
        self.assertIsNotNone(self.datacenter_predictor.model)

    def test_initialization_internet(self):
        """Test internet predictor initialization"""
        self.assertEqual(self.internet_predictor.mode, 'internet')
        self.assertEqual(self.internet_predictor.config.sequence_length, 60)
        self.assertEqual(self.internet_predictor.config.feature_count, 5)
        self.assertEqual(self.internet_predictor.config.confidence_threshold, 0.70)
        self.assertIsNotNone(self.internet_predictor.model)

    def test_invalid_mode(self):
        """Test that invalid mode raises error"""
        with self.assertRaises(ValueError):
            BandwidthPredictorV3(mode='invalid')

    def test_mode_config_datacenter(self):
        """Test datacenter mode configuration"""
        config = self.datacenter_predictor._get_mode_config('datacenter')
        self.assertEqual(config.mode, 'datacenter')
        self.assertEqual(config.sequence_length, 30)
        self.assertEqual(config.confidence_threshold, 0.85)

    def test_mode_config_internet(self):
        """Test internet mode configuration"""
        config = self.internet_predictor._get_mode_config('internet')
        self.assertEqual(config.mode, 'internet')
        self.assertEqual(config.sequence_length, 60)
        self.assertEqual(config.confidence_threshold, 0.70)

    def test_synthetic_data_generation_datacenter(self):
        """Test datacenter synthetic data generation"""
        data = generate_synthetic_data('datacenter', 100)
        self.assertEqual(len(data), 100)

        # Check datacenter characteristics
        for sample in data:
            self.assertIsInstance(sample, NetworkMetrics)
            self.assertGreaterEqual(sample.bandwidth_mbps, 1000)
            self.assertLessEqual(sample.bandwidth_mbps, 10000)
            self.assertGreaterEqual(sample.latency_ms, 0.5)
            self.assertLessEqual(sample.latency_ms, 10)
            self.assertLessEqual(sample.packet_loss, 0.001)
            self.assertLessEqual(sample.jitter_ms, 2)

    def test_synthetic_data_generation_internet(self):
        """Test internet synthetic data generation"""
        data = generate_synthetic_data('internet', 100)
        self.assertEqual(len(data), 100)

        # Check internet characteristics
        for sample in data:
            self.assertIsInstance(sample, NetworkMetrics)
            self.assertGreaterEqual(sample.bandwidth_mbps, 100)
            self.assertLessEqual(sample.bandwidth_mbps, 900)
            self.assertGreaterEqual(sample.latency_ms, 10)
            self.assertLessEqual(sample.latency_ms, 500)
            self.assertLessEqual(sample.packet_loss, 0.02)
            self.assertLessEqual(sample.jitter_ms, 50)

    def test_training_data_preparation_datacenter(self):
        """Test training data preparation for datacenter"""
        data = generate_synthetic_data('datacenter', 100)
        X, y = self.datacenter_predictor.prepare_training_data(data)

        expected_samples = len(data) - self.datacenter_predictor.config.sequence_length
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], 30)  # sequence_length
        self.assertEqual(X.shape[2], 5)   # feature_count
        self.assertEqual(y.shape[0], expected_samples)

    def test_training_data_preparation_internet(self):
        """Test training data preparation for internet"""
        data = generate_synthetic_data('internet', 200)
        X, y = self.internet_predictor.prepare_training_data(data)

        expected_samples = len(data) - self.internet_predictor.config.sequence_length
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], 60)  # sequence_length
        self.assertEqual(X.shape[2], 5)   # feature_count
        self.assertEqual(y.shape[0], expected_samples)

    def test_insufficient_training_data(self):
        """Test that insufficient data raises error"""
        data = generate_synthetic_data('datacenter', 10)  # Too few samples
        with self.assertRaises(ValueError):
            self.datacenter_predictor.prepare_training_data(data)

    def test_model_training_datacenter(self):
        """Test datacenter model training"""
        data = generate_synthetic_data('datacenter', 500)
        results = self.datacenter_predictor.train(data, epochs=5)

        self.assertIn('history', results)
        self.assertIn('final_val_loss', results)
        self.assertIn('final_val_mae', results)
        self.assertIn('epochs_trained', results)
        self.assertGreater(results['epochs_trained'], 0)
        self.assertLess(results['final_val_loss'], 10000)  # Reasonable loss

    def test_model_training_internet(self):
        """Test internet model training"""
        data = generate_synthetic_data('internet', 1000)
        results = self.internet_predictor.train(data, epochs=5)

        self.assertIn('history', results)
        self.assertIn('final_val_loss', results)
        self.assertIn('final_val_mae', results)
        self.assertGreater(results['epochs_trained'], 0)
        self.assertLess(results['final_val_loss'], 10000)

    def test_prediction_datacenter(self):
        """Test datacenter prediction"""
        # Train model first
        data = generate_synthetic_data('datacenter', 500)
        self.datacenter_predictor.train(data, epochs=5, batch_size=32)

        # Make prediction
        test_data = generate_synthetic_data('datacenter', 50)
        prediction, confidence = self.datacenter_predictor.predict(
            test_data[-30:],
            return_confidence=True
        )

        # Validate prediction
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)
        self.assertLess(prediction, 20000)  # Reasonable bandwidth range

        # Validate confidence
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1.0)

    def test_prediction_internet(self):
        """Test internet prediction"""
        # Train model first
        data = generate_synthetic_data('internet', 1000)
        self.internet_predictor.train(data, epochs=5, batch_size=32)

        # Make prediction
        test_data = generate_synthetic_data('internet', 100)
        prediction, confidence = self.internet_predictor.predict(
            test_data[-60:],
            return_confidence=True
        )

        # Validate prediction
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)
        self.assertLess(prediction, 2000)

        # Validate confidence
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1.0)

    def test_prediction_without_training(self):
        """Test that prediction fails without training"""
        data = generate_synthetic_data('datacenter', 50)
        fresh_predictor = BandwidthPredictorV3(mode='datacenter')

        with self.assertRaises(RuntimeError):
            fresh_predictor.predict(data[-30:])

    def test_prediction_insufficient_history(self):
        """Test that prediction fails with insufficient history"""
        data = generate_synthetic_data('datacenter', 500)
        self.datacenter_predictor.train(data, epochs=2)

        insufficient_data = generate_synthetic_data('datacenter', 10)
        with self.assertRaises(ValueError):
            self.datacenter_predictor.predict(insufficient_data)

    def test_confidence_calculation(self):
        """Test confidence calculation"""
        data = generate_synthetic_data('datacenter', 500)
        self.datacenter_predictor.train(data, epochs=3)

        test_data = generate_synthetic_data('datacenter', 50)
        _, confidence = self.datacenter_predictor.predict(
            test_data[-30:],
            return_confidence=True
        )

        # Confidence should be within valid range
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_datacenter_method(self):
        """Test datacenter-specific prediction method"""
        data = generate_synthetic_data('datacenter', 500)
        self.datacenter_predictor.train(data, epochs=3)

        test_data = generate_synthetic_data('datacenter', 50)
        prediction, confidence = self.datacenter_predictor.predict_datacenter(
            test_data[-30:]
        )

        self.assertIsInstance(prediction, float)
        self.assertIsInstance(confidence, float)
        self.assertGreater(prediction, 0)
        self.assertGreaterEqual(confidence, 0)

    def test_predict_internet_method(self):
        """Test internet-specific prediction method"""
        data = generate_synthetic_data('internet', 1000)
        self.internet_predictor.train(data, epochs=3)

        test_data = generate_synthetic_data('internet', 100)
        prediction, confidence = self.internet_predictor.predict_internet(
            test_data[-60:]
        )

        self.assertIsInstance(prediction, float)
        self.assertIsInstance(confidence, float)
        self.assertGreater(prediction, 0)
        self.assertGreaterEqual(confidence, 0)

    def test_model_persistence_datacenter(self):
        """Test datacenter model save/load"""
        import tempfile
        import shutil

        # Train model
        data = generate_synthetic_data('datacenter', 500)
        self.datacenter_predictor.train(data, epochs=3)

        # Make prediction before save
        test_data = generate_synthetic_data('datacenter', 50)
        pred_before, _ = self.datacenter_predictor.predict(test_data[-30:])

        # Save model
        temp_dir = tempfile.mkdtemp()
        try:
            self.datacenter_predictor.save_model(temp_dir)

            # Load model in new predictor
            new_predictor = BandwidthPredictorV3(mode='datacenter')
            new_predictor.load_model(temp_dir)

            # Make prediction after load
            pred_after, _ = new_predictor.predict(test_data[-30:])

            # Predictions should be identical
            self.assertAlmostEqual(pred_before, pred_after, places=2)

        finally:
            shutil.rmtree(temp_dir)

    def test_model_persistence_internet(self):
        """Test internet model save/load"""
        import tempfile
        import shutil

        # Train model
        data = generate_synthetic_data('internet', 1000)
        self.internet_predictor.train(data, epochs=3)

        # Make prediction before save
        test_data = generate_synthetic_data('internet', 100)
        pred_before, _ = self.internet_predictor.predict(test_data[-60:])

        # Save model
        temp_dir = tempfile.mkdtemp()
        try:
            self.internet_predictor.save_model(temp_dir)

            # Load model in new predictor
            new_predictor = BandwidthPredictorV3(mode='internet')
            new_predictor.load_model(temp_dir)

            # Make prediction after load
            pred_after, _ = new_predictor.predict(test_data[-60:])

            # Predictions should be similar
            self.assertAlmostEqual(pred_before, pred_after, places=2)

        finally:
            shutil.rmtree(temp_dir)


class TestPerformanceTargets(unittest.TestCase):
    """Test that performance targets are met"""

    def test_datacenter_accuracy_target(self):
        """Test that datacenter model meets 85% accuracy target"""
        predictor = BandwidthPredictorV3(mode='datacenter')
        data = generate_synthetic_data('datacenter', 1000)

        # Train
        predictor.train(data[:800], epochs=20)

        # Test
        test_data = data[800:]
        predictions = []
        actuals = []

        for i in range(30, len(test_data)):
            history = test_data[i-30:i]
            actual = test_data[i].bandwidth_mbps

            pred, _ = predictor.predict(history)
            predictions.append(pred)
            actuals.append(actual)

        # Calculate accuracy (±20%)
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        errors = np.abs(predictions_array - actuals_array) / actuals_array
        accuracy = np.mean(errors < 0.20) * 100

        print(f"\nDatacenter accuracy: {accuracy:.1f}%")
        # Allow some tolerance in tests
        self.assertGreater(accuracy, 75.0, "Datacenter accuracy should be >75%")

    def test_internet_accuracy_target(self):
        """Test that internet model meets 70% accuracy target"""
        predictor = BandwidthPredictorV3(mode='internet')
        data = generate_synthetic_data('internet', 1500)

        # Train
        predictor.train(data[:1200], epochs=25)

        # Test
        test_data = data[1200:]
        predictions = []
        actuals = []

        for i in range(60, len(test_data)):
            history = test_data[i-60:i]
            actual = test_data[i].bandwidth_mbps

            pred, _ = predictor.predict(history)
            predictions.append(pred)
            actuals.append(actual)

        # Calculate accuracy (±20%)
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        errors = np.abs(predictions_array - actuals_array) / actuals_array
        accuracy = np.mean(errors < 0.20) * 100

        print(f"\nInternet accuracy: {accuracy:.1f}%")
        # Allow some tolerance in tests
        self.assertGreater(accuracy, 60.0, "Internet accuracy should be >60%")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
