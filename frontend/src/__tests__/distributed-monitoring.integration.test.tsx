/**
 * Integration tests for distributed monitoring components
 * Sprint 6 - Distributed Supercompute Fabric Infrastructure
 */

import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { jest } from '@jest/globals';

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: WebSocket.OPEN,
};

// Mock the useWebSocket hooks
jest.mock('../hooks/useWebSocket', () => ({
  useDistributedTopologyWebSocket: () => ({
    data: {
      nodes: [
        { id: 'node1', type: 'compute', status: 'active', cluster: 'cluster1' },
        { id: 'node2', type: 'federation', status: 'active', cluster: 'cluster2' }
      ],
      edges: [
        { source: 'node1', target: 'node2', bandwidth: 10000, latency: 5 }
      ],
      clusters: [
        { id: 'cluster1', name: 'Primary Cluster', status: 'healthy' },
        { id: 'cluster2', name: 'Secondary Cluster', status: 'healthy' }
      ]
    },
    isConnected: true,
    error: null,
    send: mockWebSocket.send,
    close: mockWebSocket.close,
    reconnect: jest.fn(),
  }),
  useBandwidthMonitoringWebSocket: () => ({
    data: {
      interfaces: [
        {
          id: 'eth0',
          name: 'Primary Interface',
          throughput: 8500,
          capacity: 10000,
          utilization: 85,
          qosEnabled: true,
          status: 'active'
        }
      ],
      globalMetrics: {
        totalBandwidth: 45200,
        utilizedBandwidth: 32800,
        efficiency: 92.5,
        qosCompliance: 98.2
      }
    },
    isConnected: true,
    error: null,
    send: mockWebSocket.send,
    close: mockWebSocket.close,
    reconnect: jest.fn(),
  }),
  usePerformancePredictionWebSocket: () => ({
    data: {
      predictions: [
        {
          id: 'pred1',
          type: 'resource',
          metric: 'cpu',
          currentValue: 68,
          predictedValue: 75,
          confidence: 94,
          timeHorizon: '1h',
          recommendation: 'Scale horizontally'
        }
      ],
      models: [
        {
          id: 'model1',
          name: 'Resource Predictor',
          type: 'LSTM',
          accuracy: 94.2,
          status: 'active',
          lastTrained: '2024-01-15T10:30:00Z'
        }
      ]
    },
    isConnected: true,
    error: null,
    send: mockWebSocket.send,
    close: mockWebSocket.close,
    reconnect: jest.fn(),
  }),
  useSupercomputeFabricWebSocket: () => ({
    data: {
      jobs: [
        {
          id: 'job1',
          name: 'ML Training Job',
          status: 'running',
          progress: 65,
          cluster: 'cluster1',
          resources: { cpu: 32, memory: 128, gpu: 4 }
        }
      ],
      globalPool: {
        totalCPU: 1024,
        usedCPU: 512,
        totalMemory: 4096,
        usedMemory: 2048,
        totalGPU: 64,
        usedGPU: 24
      },
      memoryFabric: {
        totalCapacity: 10240,
        usedCapacity: 7168,
        replicationFactor: 3,
        consistency: 'strong'
      }
    },
    isConnected: true,
    error: null,
    send: mockWebSocket.send,
    close: mockWebSocket.close,
    reconnect: jest.fn(),
  }),
}));

// Mock API client functions
jest.mock('../lib/api/client', () => ({
  getNetworkTopology: jest.fn().mockResolvedValue({
    data: {
      nodes: [],
      edges: [],
      clusters: []
    },
    error: null
  }),
  getBandwidthMetrics: jest.fn().mockResolvedValue({
    data: {
      interfaces: [],
      globalMetrics: {}
    },
    error: null
  }),
  getResourcePredictions: jest.fn().mockResolvedValue({
    data: [],
    error: null
  }),
  getComputeJobs: jest.fn().mockResolvedValue({
    data: [],
    error: null
  }),
}));

// Import components after mocks
import BandwidthMonitoringDashboard from '../components/monitoring/BandwidthMonitoringDashboard';
import PerformancePredictionDashboard from '../components/monitoring/PerformancePredictionDashboard';
import SupercomputeFabricDashboard from '../components/monitoring/SupercomputeFabricDashboard';
import { NetworkTopology } from '../components/network/NetworkTopology';

describe('Distributed Monitoring Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock global WebSocket constructor
    global.WebSocket = jest.fn().mockImplementation(() => mockWebSocket);
  });

  describe('BandwidthMonitoringDashboard', () => {
    test('renders bandwidth monitoring interface with real-time data', async () => {
      render(<BandwidthMonitoringDashboard />);

      // Check for main dashboard elements
      expect(screen.getByText('Bandwidth Monitoring')).toBeInTheDocument();
      expect(screen.getByText('Network Interfaces')).toBeInTheDocument();
      expect(screen.getByText('QoS Monitoring')).toBeInTheDocument();

      // Wait for real-time data to load
      await waitFor(() => {
        expect(screen.getByText('Primary Interface')).toBeInTheDocument();
      });

      // Check bandwidth metrics
      expect(screen.getByText('45.2 Gbps')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    test('handles WebSocket connection states', async () => {
      render(<BandwidthMonitoringDashboard />);

      // Should show connected state
      await waitFor(() => {
        const connectionStatus = screen.getByText(/Connected/i);
        expect(connectionStatus).toBeInTheDocument();
      });
    });

    test('displays QoS compliance metrics', async () => {
      render(<BandwidthMonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('98.2%')).toBeInTheDocument();
      });
    });
  });

  describe('PerformancePredictionDashboard', () => {
    test('renders AI prediction interface with model metrics', async () => {
      render(<PerformancePredictionDashboard />);

      // Check for main dashboard elements
      expect(screen.getByText('Performance Predictions')).toBeInTheDocument();
      expect(screen.getByText('AI Models')).toBeInTheDocument();

      // Wait for prediction data to load
      await waitFor(() => {
        expect(screen.getByText('Resource Predictor')).toBeInTheDocument();
      });

      // Check prediction accuracy
      expect(screen.getByText('94%')).toBeInTheDocument();
    });

    test('displays prediction recommendations', async () => {
      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Scale horizontally')).toBeInTheDocument();
      });
    });

    test('handles confidence threshold filtering', async () => {
      render(<PerformancePredictionDashboard />);

      // Find confidence threshold slider
      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '90' } });

      // Predictions below 90% confidence should be filtered
      await waitFor(() => {
        expect(screen.getByDisplayValue('90')).toBeInTheDocument();
      });
    });
  });

  describe('SupercomputeFabricDashboard', () => {
    test('renders fabric monitoring with compute jobs', async () => {
      render(<SupercomputeFabricDashboard />);

      // Check for main dashboard elements
      expect(screen.getByText('Supercompute Fabric')).toBeInTheDocument();
      expect(screen.getByText('Compute Jobs')).toBeInTheDocument();

      // Wait for job data to load
      await waitFor(() => {
        expect(screen.getByText('ML Training Job')).toBeInTheDocument();
      });

      // Check resource utilization
      expect(screen.getByText('65%')).toBeInTheDocument();
    });

    test('displays global resource pool metrics', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByText('1024')).toBeInTheDocument(); // Total CPU
        expect(screen.getByText('4096')).toBeInTheDocument(); // Total Memory
      });
    });

    test('shows memory fabric statistics', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByText('10.2 TB')).toBeInTheDocument(); // Total capacity
        expect(screen.getByText('strong')).toBeInTheDocument(); // Consistency level
      });
    });
  });

  describe('NetworkTopology Integration', () => {
    test('renders distributed network topology with federation nodes', async () => {
      render(
        <NetworkTopology
          showDistributed={true}
          showBandwidth={true}
          showPerformanceMetrics={true}
        />
      );

      // Check for distributed features
      expect(screen.getByText('Distributed View')).toBeInTheDocument();

      // Wait for network data to load
      await waitFor(() => {
        expect(screen.getByText('node1')).toBeInTheDocument();
        expect(screen.getByText('node2')).toBeInTheDocument();
      });
    });

    test('displays bandwidth information on edges', async () => {
      render(
        <NetworkTopology
          showDistributed={true}
          showBandwidth={true}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('10 Gbps')).toBeInTheDocument();
      });
    });

    test('shows cluster boundaries and federation status', async () => {
      render(
        <NetworkTopology
          showDistributed={true}
          showPerformanceMetrics={true}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Primary Cluster')).toBeInTheDocument();
        expect(screen.getByText('Secondary Cluster')).toBeInTheDocument();
      });
    });
  });

  describe('Cross-Component Data Flow', () => {
    test('maintains WebSocket connections across multiple components', async () => {
      const { rerender } = render(<BandwidthMonitoringDashboard />);

      // Switch to performance dashboard
      rerender(<PerformancePredictionDashboard />);

      // WebSocket should maintain connection
      expect(mockWebSocket.close).not.toHaveBeenCalled();
    });

    test('handles WebSocket errors gracefully', async () => {
      // Mock WebSocket error
      const errorHook = {
        data: null,
        isConnected: false,
        error: new Error('Connection failed'),
        send: jest.fn(),
        close: jest.fn(),
        reconnect: jest.fn(),
      };

      jest.doMock('../hooks/useWebSocket', () => ({
        useBandwidthMonitoringWebSocket: () => errorHook,
      }));

      render(<BandwidthMonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/connection error/i)).toBeInTheDocument();
      });
    });

    test('synchronizes data updates across related components', async () => {
      // This would test that bandwidth updates in the network topology
      // are reflected in the bandwidth monitoring dashboard
      const container = render(
        <div>
          <NetworkTopology showBandwidth={true} />
          <BandwidthMonitoringDashboard />
        </div>
      );

      await waitFor(() => {
        const bandwidthElements = screen.getAllByText(/gbps/i);
        expect(bandwidthElements.length).toBeGreaterThan(1);
      });
    });
  });

  describe('Performance and Load Testing', () => {
    test('handles high-frequency WebSocket updates', async () => {
      render(<BandwidthMonitoringDashboard />);

      // Simulate rapid updates
      for (let i = 0; i < 100; i++) {
        mockWebSocket.addEventListener.mock.calls[0][1]({
          data: JSON.stringify({
            interfaces: [{
              id: 'eth0',
              throughput: 8500 + i,
              utilization: 85 + (i % 10)
            }]
          })
        });
      }

      // Component should remain stable
      await waitFor(() => {
        expect(screen.getByText('Bandwidth Monitoring')).toBeInTheDocument();
      });
    });

    test('manages memory with large datasets', async () => {
      // Create large mock dataset
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        id: `job${i}`,
        name: `Job ${i}`,
        status: 'running',
        progress: Math.random() * 100
      }));

      jest.doMock('../hooks/useWebSocket', () => ({
        useSupercomputeFabricWebSocket: () => ({
          data: { jobs: largeDataset },
          isConnected: true,
          error: null,
          send: jest.fn(),
          close: jest.fn(),
          reconnect: jest.fn(),
        }),
      }));

      render(<SupercomputeFabricDashboard />);

      // Should handle large datasets without performance issues
      await waitFor(() => {
        expect(screen.getByText('Supercompute Fabric')).toBeInTheDocument();
      });
    });
  });

  describe('Mobile Responsiveness', () => {
    test('adapts layout for mobile viewport', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<BandwidthMonitoringDashboard />);

      await waitFor(() => {
        const dashboard = screen.getByRole('main') || screen.getByText('Bandwidth Monitoring').closest('div');
        expect(dashboard).toHaveClass('grid-cols-1');
      });
    });

    test('provides touch-friendly controls on mobile', async () => {
      // Mock touch device
      Object.defineProperty(window, 'ontouchstart', {
        writable: true,
        configurable: true,
        value: true,
      });

      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        const buttons = screen.getAllByRole('button');
        buttons.forEach(button => {
          expect(button).toHaveStyle('min-height: 44px'); // Touch-friendly size
        });
      });
    });
  });

  describe('Accessibility', () => {
    test('provides proper ARIA labels and roles', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
        expect(screen.getByLabelText(/fabric dashboard/i)).toBeInTheDocument();
      });
    });

    test('supports keyboard navigation', async () => {
      render(<BandwidthMonitoringDashboard />);

      const firstButton = screen.getAllByRole('button')[0];
      firstButton.focus();
      expect(document.activeElement).toBe(firstButton);

      // Test tab navigation
      fireEvent.keyDown(firstButton, { key: 'Tab' });
      expect(document.activeElement).not.toBe(firstButton);
    });

    test('provides screen reader compatible content', async () => {
      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        expect(screen.getByText('94%')).toHaveAttribute('aria-label', 'Prediction accuracy: 94 percent');
      });
    });
  });
});