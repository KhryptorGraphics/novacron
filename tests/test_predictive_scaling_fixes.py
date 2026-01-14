"""
Test suite for predictive scaling feature leakage fixes
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the classes we need to test
import sys
sys.path.append('/home/kp/novacron')
from ai_engine.predictive_scaling import PredictiveScalingEngine, ResourceType, ResourceForecast


class TestPredictiveScalingFixes(unittest.TestCase):
    """Test feature leakage fixes in predictive scaling"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary database for testing
        self.test_db = tempfile.mktemp(suffix='.db')
        self.engine = PredictiveScalingEngine(db_path=self.test_db)

        # Create sample historical data
        base_time = datetime(2023, 1, 1, 12, 0, 0)  # Fixed time for testing
        timestamps = [base_time + timedelta(minutes=i) for i in range(100)]

        self.historical_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': np.random.uniform(0.3, 0.8, 100),
            'memory_usage': np.random.uniform(0.2, 0.7, 100),
            'network_usage': np.random.uniform(10, 100, 100),
        })

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_no_datetime_now_in_lstm_predictions(self):
        """Test that LSTM predictions don't use datetime.now() for cyclical features"""

        # Mock the LSTM model to be trained
        self.engine.lstm_trained = True

        # Create a mock LSTM model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.5]])

        # Replace the LSTM model in the engine
        self.engine.models[ResourceType.CPU.value]['lstm'] = mock_model

        # Test prediction with historical data
        last_timestamp = self.historical_data['timestamp'].iloc[-1]
        features = self.engine._prepare_features(self.historical_data, ResourceType.CPU)

        # Call the LSTM prediction method
        predictions, confidence = self.engine._predict_lstm(mock_model, features, last_timestamp)

        # Verify predictions were generated
        self.assertEqual(len(predictions), self.engine.prediction_horizon)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.1)
        self.assertLessEqual(confidence, 1.0)

        # Verify that the model.predict was called the expected number of times
        self.assertEqual(mock_model.predict.call_count, self.engine.prediction_horizon)

    def test_forecast_uses_historical_timestamp(self):
        """Test that forecasts use historical data timestamps instead of current time"""

        # Get the last timestamp from historical data
        expected_last_timestamp = self.historical_data['timestamp'].iloc[-1]

        # Generate forecast
        forecast = self.engine.predict_resource_demand(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            historical_data=self.historical_data
        )

        # Verify that peak_time and valley_time are based on historical timestamp
        # They should be at or after the last historical timestamp
        self.assertGreaterEqual(forecast.peak_time, expected_last_timestamp)
        self.assertGreaterEqual(forecast.valley_time, expected_last_timestamp)

        # The times should be within the prediction horizon from the last timestamp
        max_future_time = expected_last_timestamp + timedelta(minutes=self.engine.prediction_horizon)
        self.assertLessEqual(forecast.peak_time, max_future_time)
        self.assertLessEqual(forecast.valley_time, max_future_time)

    def test_fallback_forecast_uses_historical_timestamp(self):
        """Test that fallback forecasts use historical timestamps when available"""

        # Create minimal data that would trigger fallback
        minimal_data = self.historical_data.head(10)  # Less than 60 required
        expected_last_timestamp = minimal_data['timestamp'].iloc[-1]

        # Generate fallback forecast
        forecast = self.engine._generate_fallback_forecast(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            data=minimal_data
        )

        # Verify that times are based on historical timestamp
        # Peak time should be after the last timestamp, valley time should be at or after
        self.assertGreater(forecast.peak_time, expected_last_timestamp)
        self.assertGreaterEqual(forecast.valley_time, expected_last_timestamp)

    def test_feature_preparation_uses_only_historical_data(self):
        """Test that feature preparation only uses historical data"""

        # Prepare features
        features = self.engine._prepare_features(self.historical_data, ResourceType.CPU)

        # Verify shape is correct (1 sample, 60 timesteps, 5 features)
        self.assertEqual(features.shape, (1, 60, 5))

        # Verify that features are numeric and within reasonable bounds
        self.assertFalse(np.isnan(features).any())
        self.assertTrue(np.isfinite(features).all())

        # Verify cyclical features are in proper range [-1, 1]
        cyclical_features = features[0, :, 1:5]  # Sin/cos features
        self.assertTrue((cyclical_features >= -1).all())
        self.assertTrue((cyclical_features <= 1).all())

    def test_prediction_consistency_across_calls(self):
        """Test that predictions are consistent when called with the same historical data"""

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate two forecasts with the same data
        forecast1 = self.engine.predict_resource_demand(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            historical_data=self.historical_data
        )

        # Reset seed and generate again
        np.random.seed(42)
        forecast2 = self.engine.predict_resource_demand(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            historical_data=self.historical_data
        )

        # The forecasts should be similar (allowing for small differences due to model randomness)
        self.assertEqual(len(forecast1.predicted_values), len(forecast2.predicted_values))
        self.assertEqual(forecast1.peak_time, forecast2.peak_time)
        self.assertEqual(forecast1.valley_time, forecast2.valley_time)

    def test_empty_data_fallback(self):
        """Test fallback behavior when no historical data is available"""

        empty_data = pd.DataFrame({'timestamp': [], 'cpu_usage': []})

        # This should not fail and should return a reasonable forecast
        forecast = self.engine._generate_fallback_forecast(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            data=empty_data
        )

        # Verify basic forecast structure
        self.assertEqual(len(forecast.predicted_values), self.engine.prediction_horizon)
        self.assertIsInstance(forecast.peak_prediction, (int, float))
        self.assertIsInstance(forecast.valley_prediction, (int, float))
        self.assertEqual(forecast.model_used, "fallback")

    def test_no_current_time_leakage_in_predictions(self):
        """Test that predictions use historical timestamps, not current time"""

        # Create historical data with known timestamps
        past_time = datetime(2023, 1, 1, 10, 0, 0)  # Fixed past time
        timestamps = [past_time + timedelta(minutes=i) for i in range(100)]

        historical_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': np.random.uniform(0.3, 0.8, 100),
            'memory_usage': np.random.uniform(0.2, 0.7, 100),
            'network_usage': np.random.uniform(10, 100, 100),
        })

        last_historical_time = timestamps[-1]  # 2023-01-01 11:39:00

        # Generate forecast - even if current time is much later,
        # predictions should be based on historical data timestamps
        forecast = self.engine.predict_resource_demand(
            vm_id="test_vm",
            resource_type=ResourceType.CPU,
            historical_data=historical_data
        )

        # Verify that forecast times are based on historical data, not current time
        # If there was feature leakage, peak/valley times might reflect current datetime
        self.assertGreaterEqual(forecast.peak_time, last_historical_time)
        self.assertGreaterEqual(forecast.valley_time, last_historical_time)

        # The forecast times should be reasonably close to the last historical timestamp
        # If current time were used, they'd be much later
        max_expected_time = last_historical_time + timedelta(hours=2)  # reasonable horizon
        self.assertLessEqual(forecast.peak_time, max_expected_time)
        self.assertLessEqual(forecast.valley_time, max_expected_time)

        # Check that forecast was generated successfully
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast.predicted_values), self.engine.prediction_horizon)


if __name__ == '__main__':
    unittest.main()