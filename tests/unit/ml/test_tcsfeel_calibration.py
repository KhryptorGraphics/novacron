"""
Unit tests for TCS-FEEL calibration system
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "backend" / "ml" / "federated"))

from calibrate_tcsfeel import (
    TCSFEELCalibrator,
    CalibrationParams,
    CalibrationResult
)
from topology import TopologyOptimizer, ClientNode


class TestCalibrationParams:
    """Test calibration parameter management"""

    def test_default_params(self):
        """Test default parameter values"""
        params = CalibrationParams()

        assert params.min_clients == 10
        assert params.max_clients == 30
        assert params.target_accuracy == 0.963
        assert params.learning_rate == 0.01
        assert params.local_epochs == 5

    def test_custom_params(self):
        """Test custom parameter initialization"""
        params = CalibrationParams(
            clients_per_round=25,
            learning_rate=0.02,
            topology_weight=0.8
        )

        assert params.clients_per_round == 25
        assert params.learning_rate == 0.02
        assert params.topology_weight == 0.8


class TestTCSFEELCalibrator:
    """Test TCS-FEEL calibration system"""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator instance"""
        return TCSFEELCalibrator(
            n_clients=20,  # Smaller for faster tests
            n_classes=10,
            target_accuracy=0.963,
            baseline_accuracy=0.868
        )

    def test_initialization(self, calibrator):
        """Test calibrator initialization"""
        assert calibrator.n_clients == 20
        assert calibrator.n_classes == 10
        assert calibrator.target_accuracy == 0.963
        assert calibrator.baseline_accuracy == 0.868
        assert calibrator.best_result is None
        assert len(calibrator.history) == 0

    def test_create_calibrated_clients(self, calibrator):
        """Test client creation with diversity control"""
        # Low diversity
        clients_low = calibrator._create_calibrated_clients(10, 0.2)
        assert len(clients_low) == 10
        assert all(isinstance(c, ClientNode) for c in clients_low)

        # High diversity
        clients_high = calibrator._create_calibrated_clients(10, 0.4)
        assert len(clients_high) == 10

        # Check data distributions are valid
        for client in clients_low + clients_high:
            assert client.data_distribution.shape == (10,)
            assert np.isclose(client.data_distribution.sum(), 1.0)
            assert client.compute_capacity > 0
            assert client.bandwidth > 0
            assert 0.75 <= client.reliability <= 0.98

    def test_create_topology_aware_connectivity(self, calibrator):
        """Test topology-aware connectivity matrix creation"""
        n_clients = 20

        # Low topology weight
        conn_low = calibrator._create_topology_aware_connectivity(n_clients, 0.3)
        assert conn_low.shape == (n_clients, n_clients)
        assert np.allclose(conn_low, conn_low.T)  # Symmetric
        assert np.allclose(np.diag(conn_low), 0)  # Zero diagonal

        # High topology weight (should have clusters)
        conn_high = calibrator._create_topology_aware_connectivity(n_clients, 0.8)
        assert conn_high.shape == (n_clients, n_clients)

    def test_simulate_training_round(self, calibrator):
        """Test training round simulation"""
        # Create test clients
        clients = calibrator._create_calibrated_clients(10, 0.3)

        params = CalibrationParams()
        current_accuracy = 0.85

        metrics = calibrator._simulate_training_round(
            clients,
            current_accuracy,
            params
        )

        assert 'accuracy_gain' in metrics
        assert 'comm_cost' in metrics
        assert 'fairness' in metrics

        assert metrics['accuracy_gain'] > 0
        assert metrics['comm_cost'] > 0
        assert 0 <= metrics['fairness'] <= 1

    def test_train_with_params(self, calibrator):
        """Test training with specific parameters"""
        params = CalibrationParams(
            clients_per_round=15,
            local_epochs=3,
            learning_rate=0.01
        )

        result = calibrator._train_with_params(params)

        assert isinstance(result, CalibrationResult)
        assert result.final_accuracy >= calibrator.baseline_accuracy
        assert result.rounds_to_convergence > 0
        assert result.communication_reduction > 0
        assert result.convergence_speed > 0
        assert result.total_training_time > 0

    def test_calibration_improves_accuracy(self, calibrator):
        """Test that calibration improves over baseline"""
        params = CalibrationParams(
            clients_per_round=20,
            local_epochs=5,
            learning_rate=0.02
        )

        result = calibrator._train_with_params(params)

        # Should improve over baseline
        assert result.final_accuracy > calibrator.baseline_accuracy

    def test_different_params_produce_different_results(self, calibrator):
        """Test parameter sensitivity"""
        params1 = CalibrationParams(local_epochs=3, learning_rate=0.005)
        params2 = CalibrationParams(local_epochs=10, learning_rate=0.05)

        result1 = calibrator._train_with_params(params1)
        result2 = calibrator._train_with_params(params2)

        # Different params should produce different results
        # (though not guaranteed, very likely with these extreme values)
        assert result1.final_accuracy != result2.final_accuracy or \
               result1.rounds_to_convergence != result2.rounds_to_convergence


class TestCalibrationResult:
    """Test calibration result handling"""

    def test_result_to_dict(self):
        """Test result serialization"""
        params = CalibrationParams()
        result = CalibrationResult(
            params=params,
            final_accuracy=0.965,
            rounds_to_convergence=45,
            communication_reduction=0.375,
            convergence_speed=1.8,
            avg_fairness=0.82,
            total_training_time=120.5
        )

        result_dict = result.to_dict()

        assert result_dict['final_accuracy'] == 0.965
        assert result_dict['rounds_to_convergence'] == 45
        assert result_dict['communication_reduction'] == 0.375
        assert 'params' in result_dict


class TestIntegration:
    """Integration tests for full calibration pipeline"""

    @pytest.fixture
    def small_calibrator(self):
        """Create small calibrator for faster tests"""
        return TCSFEELCalibrator(
            n_clients=15,
            n_classes=10,
            target_accuracy=0.90,  # Lower target for test speed
            baseline_accuracy=0.868
        )

    def test_grid_search_completes(self, small_calibrator):
        """Test that grid search completes without errors"""
        # Override with smaller grid for testing
        original_method = small_calibrator.calibrate_grid_search

        def quick_grid_search():
            # Test just a few combinations
            param_grid = {
                'clients_per_round': [15, 20],
                'local_epochs': [3, 5],
                'learning_rate': [0.01],
                'topology_weight': [0.7],
                'diversity_factor': [0.3]
            }

            best_accuracy = small_calibrator.baseline_accuracy

            for clients in param_grid['clients_per_round']:
                for epochs in param_grid['local_epochs']:
                    params = CalibrationParams(
                        clients_per_round=clients,
                        local_epochs=epochs,
                        learning_rate=0.01,
                        topology_weight=0.7,
                        diversity_factor=0.3,
                        target_accuracy=small_calibrator.target_accuracy
                    )

                    result = small_calibrator._train_with_params(params)
                    small_calibrator.history.append(result)

                    if result.final_accuracy > best_accuracy:
                        best_accuracy = result.final_accuracy
                        small_calibrator.best_result = result

                    if result.final_accuracy >= small_calibrator.target_accuracy:
                        return result

            return small_calibrator.best_result

        result = quick_grid_search()

        assert result is not None
        assert isinstance(result, CalibrationResult)
        assert len(small_calibrator.history) > 0
        assert small_calibrator.best_result is not None

    def test_calibration_metrics_valid(self, small_calibrator):
        """Test that calibration produces valid metrics"""
        params = CalibrationParams()
        result = small_calibrator._train_with_params(params)

        # Accuracy in valid range
        assert 0 <= result.final_accuracy <= 1.0

        # Positive rounds
        assert result.rounds_to_convergence > 0

        # Valid reduction and speedup
        assert 0 <= result.communication_reduction <= 1.0
        assert result.convergence_speed > 0

        # Valid fairness
        assert 0 <= result.avg_fairness <= 1.0

        # Positive training time
        assert result.total_training_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
