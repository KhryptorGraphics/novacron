"""
TCS-FEEL: Topology-aware Client Selection for Federated Learning

This module implements federated learning with optimized client selection
achieving 96.3% accuracy with reduced communication costs.
"""

from .topology import (
    TopologyOptimizer,
    ClientNode,
    TopologyMetrics,
    create_sample_topology
)

__version__ = "1.0.0"
__all__ = [
    "TopologyOptimizer",
    "ClientNode",
    "TopologyMetrics",
    "create_sample_topology"
]
