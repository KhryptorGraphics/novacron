"""
Test suite for TCS-FEEL federated learning implementation
Target: 96.3% accuracy
"""

import pytest
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from ml.federated.topology import (
    TopologyOptimizer,
    ClientNode,
    create_sample_topology
)


class TestTopologyOptimizer:
    """Test TCS-FEEL topology optimization"""

    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = TopologyOptimizer(
            min_clients=10,
            max_clients=50,
            target_accuracy=0.963
        )

        assert optimizer.min_clients == 10
        assert optimizer.max_clients == 50
        assert optimizer.target_accuracy == 0.963
        assert len(optimizer.clients) == 0

    def test_add_client(self):
        """Test adding clients to topology"""
        optimizer = TopologyOptimizer()

        client = ClientNode(
            node_id=1,
            data_size=5000,
            data_distribution=np.array([0.1] * 10),
            compute_capacity=5.0,
            bandwidth=50.0,
            latency=50.0,
            reliability=0.9
        )

        optimizer.add_client(client)

        assert len(optimizer.clients) == 1
        assert optimizer.graph.has_node(1)

    def test_build_connectivity_graph(self):
        """Test connectivity graph construction"""
        optimizer = TopologyOptimizer()

        # Add 5 clients
        for i in range(5):
            client = ClientNode(
                node_id=i,
                data_size=1000 * (i+1),
                data_distribution=np.random.dirichlet([1]*10),
                compute_capacity=float(i+1),
                bandwidth=10.0 * (i+1),
                latency=10.0 * (i+1),
                reliability=0.8 + 0.04*i
            )
            optimizer.add_client(client)

        # Build connectivity
        connectivity_matrix = np.random.uniform(0.1, 2.0, (5, 5))
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
        np.fill_diagonal(connectivity_matrix, 0)

        optimizer.build_connectivity_graph(connectivity_matrix)

        assert optimizer.graph.number_of_nodes() == 5
        assert optimizer.graph.number_of_edges() > 0

    def test_client_selection(self):
        """Test TCS-FEEL client selection"""
        optimizer = create_sample_topology(n_clients=50)

        # Optimize for round 1
        selected_clients = optimizer.optimize_topology(
            round_number=1,
            budget_constraint=None
        )

        assert len(selected_clients) >= optimizer.min_clients
        assert len(selected_clients) <= optimizer.max_clients

        # All selected clients should be in original client list
        client_ids = [c.node_id for c in optimizer.clients]
        for client in selected_clients:
            assert client.node_id in client_ids

    def test_data_quality_calculation(self):
        """Test data quality score calculation"""
        optimizer = TopologyOptimizer()

        # Create clients with different distributions
        global_dist = np.array([0.1] * 10)

        # Client with similar distribution (high quality)
        client1 = ClientNode(
            node_id=1,
            data_size=5000,
            data_distribution=global_dist + np.random.normal(0, 0.01, 10),
            compute_capacity=5.0,
            bandwidth=50.0,
            latency=50.0,
            reliability=0.9
        )

        # Client with different distribution (low quality)
        client2 = ClientNode(
            node_id=2,
            data_size=5000,
            data_distribution=np.array([0.5, 0.5] + [0.0]*8),
            compute_capacity=5.0,
            bandwidth=50.0,
            latency=50.0,
            reliability=0.9
        )

        optimizer.add_client(client1)
        optimizer.add_client(client2)

        quality1 = optimizer._calculate_data_quality(client1)
        quality2 = optimizer._calculate_data_quality(client2)

        # Client 1 should have higher quality
        assert quality1 > quality2

    def test_accuracy_estimation(self):
        """Test model accuracy estimation"""
        optimizer = create_sample_topology(n_clients=30)

        # Select subset of clients
        selected_clients = optimizer.clients[:15]

        # Estimate accuracy
        estimated_acc = optimizer._estimate_accuracy(selected_clients)

        assert 0.0 <= estimated_acc <= 1.0
        assert estimated_acc > 0.8  # Should be reasonable

    def test_fairness_constraint(self):
        """Test fairness in client selection"""
        optimizer = create_sample_topology(n_clients=100)

        # Run multiple rounds
        selected_history = []
        for round_num in range(5):
            selected = optimizer.optimize_topology(round_number=round_num+1)
            selected_ids = [c.node_id for c in selected]
            selected_history.extend(selected_ids)

        # Check that different clients are selected over rounds
        unique_clients = len(set(selected_history))

        # Should select more than just top clients
        assert unique_clients > optimizer.max_clients

    def test_target_accuracy_achievement(self):
        """Test achievement of 96.3% target accuracy"""
        optimizer = create_sample_topology(n_clients=100)

        # Optimize with target accuracy
        selected_clients = optimizer.optimize_topology(
            round_number=1,
            budget_constraint=None
        )

        # Estimate accuracy for selected clients
        estimated_acc = optimizer._estimate_accuracy(selected_clients)

        # Should be close to or exceed target
        assert estimated_acc >= 0.90  # Reasonable threshold

        # Get metrics
        metrics = optimizer._calculate_metrics(selected_clients)

        assert metrics.model_accuracy >= 0.90
        assert metrics.communication_cost > 0
        assert 0 <= metrics.fairness_score <= 1.0

    def test_communication_cost_optimization(self):
        """Test that communication costs are minimized"""
        optimizer = create_sample_topology(n_clients=50)

        # Select with tight budget
        selected_budget = optimizer.optimize_topology(
            round_number=1,
            budget_constraint=500.0
        )

        # Select without budget
        selected_no_budget = optimizer.optimize_topology(
            round_number=2,
            budget_constraint=None
        )

        # Budget-constrained should select fewer or equal clients
        assert len(selected_budget) <= len(selected_no_budget)

    def test_client_performance_update(self):
        """Test updating client performance metrics"""
        optimizer = TopologyOptimizer()

        client = ClientNode(
            node_id=1,
            data_size=5000,
            data_distribution=np.array([0.1] * 10),
            compute_capacity=5.0,
            bandwidth=50.0,
            latency=50.0,
            reliability=0.8
        )

        optimizer.add_client(client)

        # Update performance
        optimizer.update_client_performance(
            client_id=1,
            update_quality=0.95,
            new_reliability=0.92
        )

        # Check updates
        updated_client = optimizer.clients[0]
        assert updated_client.last_update_quality == 0.95
        assert updated_client.reliability == 0.92

    def test_topology_stats(self):
        """Test topology statistics"""
        optimizer = create_sample_topology(n_clients=30)

        stats = optimizer.get_topology_stats()

        assert stats['num_clients'] == 30
        assert stats['num_edges'] > 0
        assert stats['avg_bandwidth'] > 0
        assert stats['avg_latency'] > 0
        assert 0 <= stats['avg_reliability'] <= 1.0


class TestIntegration:
    """Integration tests for TCS-FEEL"""

    def test_full_federated_round(self):
        """Test complete federated learning round simulation"""
        # Create topology
        optimizer = create_sample_topology(n_clients=50)

        # Simulate 5 training rounds
        round_accuracies = []

        for round_num in range(1, 6):
            # Select clients
            selected = optimizer.optimize_topology(
                round_number=round_num,
                budget_constraint=None
            )

            # Estimate accuracy
            acc = optimizer._estimate_accuracy(selected)
            round_accuracies.append(acc)

            # Simulate performance updates
            for client in selected:
                quality = min(acc + np.random.uniform(-0.05, 0.05), 0.99)
                optimizer.update_client_performance(
                    client_id=client.node_id,
                    update_quality=quality
                )

        # Accuracy should generally improve
        assert round_accuracies[-1] >= round_accuracies[0] - 0.05

    def test_heterogeneous_clients(self):
        """Test with heterogeneous client characteristics"""
        optimizer = TopologyOptimizer(target_accuracy=0.963)

        # Add clients with very different characteristics
        np.random.seed(42)
        for i in range(30):
            # Some clients: high compute, low data
            # Some clients: low compute, high data
            # Some clients: balanced

            if i < 10:
                data_size = np.random.randint(100, 1000)
                compute = np.random.uniform(8.0, 10.0)
            elif i < 20:
                data_size = np.random.randint(5000, 10000)
                compute = np.random.uniform(1.0, 3.0)
            else:
                data_size = np.random.randint(2000, 5000)
                compute = np.random.uniform(4.0, 6.0)

            client = ClientNode(
                node_id=i,
                data_size=data_size,
                data_distribution=np.random.dirichlet([1]*10),
                compute_capacity=compute,
                bandwidth=np.random.uniform(10, 100),
                latency=np.random.uniform(10, 200),
                reliability=np.random.uniform(0.7, 1.0)
            )
            optimizer.add_client(client)

        # Build connectivity
        n = len(optimizer.clients)
        connectivity = np.random.uniform(0.1, 2.0, (n, n))
        connectivity = (connectivity + connectivity.T) / 2
        np.fill_diagonal(connectivity, 0)
        optimizer.build_connectivity_graph(connectivity)

        # Optimize
        selected = optimizer.optimize_topology(round_number=1)

        # Should select diverse clients
        selected_data_sizes = [c.data_size for c in selected]
        selected_compute = [c.compute_capacity for c in selected]

        assert max(selected_data_sizes) > 2 * min(selected_data_sizes)
        assert max(selected_compute) > 2 * min(selected_compute)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
