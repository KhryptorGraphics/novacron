import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { act } from 'react-dom/test-utils';

// Mock WebSocket hook
const mockWebSocketData = {
  data: {
    metrics: {
      totalVMs: 47,
      runningVMs: 42,
      cpuUsage: 68,
      memoryUsage: 72,
      storageUsage: 54,
      networkStatus: 'online',
    },
    topology: {
      nodes: [
        {
          id: 'node1',
          name: 'VM-Web-01',
          type: 'vm',
          status: 'healthy',
          metrics: { cpuUsage: 45, memoryUsage: 60 }
        }
      ],
      edges: [
        {
          source: 'node1',
          target: 'host1',
          type: 'network',
          metrics: { latency: 2.3, bandwidth: 1000 }
        }
      ]
    },
    bandwidth: {
      interfaces: [
        {
          id: 'eth0',
          name: 'Primary Interface',
          utilization: 75,
          capacity: 10000,
          inbound: 7500,
          outbound: 5200
        }
      ],
      aggregated: {
        totalCapacity: 10000,
        totalUtilization: 75,
        peakUtilization: 89,
        averageLatency: 2.3
      },
      history: [
        { timestamp: '2024-01-01T00:00:00Z', utilization: 70, throughput: 7000 },
        { timestamp: '2024-01-01T00:01:00Z', utilization: 75, throughput: 7500 }
      ]
    },
    predictions: {
      resourcePredictions: [
        {
          resourceType: 'cpu',
          currentUsage: 68,
          predictedUsage: 72,
          confidence: 85,
          timeHorizon: '1hr',
          recommendations: ['Consider scaling up if usage continues to increase']
        }
      ],
      workloadPatterns: [
        {
          id: 'pattern1',
          name: 'Daily Peak',
          clusterId: 'cluster1',
          pattern: [
            { hour: 9, cpu: 45, memory: 60, network: 30 },
            { hour: 14, cpu: 75, memory: 80, network: 65 }
          ],
          confidence: 90,
          seasonality: 'daily'
        }
      ]
    },
    fabric: {
      computeJobs: [
        {
          id: 'job1',
          name: 'ML Training Job',
          type: 'ml_training',
          status: 'running',
          priority: 8,
          resources: { cpu: 16, memory: 32, storage: 100 },
          createdAt: '2024-01-01T09:00:00Z',
          updatedAt: '2024-01-01T09:30:00Z',
          progress: 45
        }
      ],
      globalResourcePool: {
        totalCpu: 1000,
        totalMemory: 2000,
        totalGpu: 50,
        totalStorage: 5000,
        availableCpu: 320,
        availableMemory: 560,
        availableGpu: 12,
        availableStorage: 2100,
        utilization: {
          cpu: 68,
          memory: 72,
          gpu: 76,
          storage: 58
        }
      }
    }
  },
  isConnected: true
};

// Mock the WebSocket hooks
jest.mock('@/hooks/useWebSocket', () => ({
  useMonitoringWebSocket: () => mockWebSocketData,
  useDistributedTopologyWebSocket: () => mockWebSocketData,
  useBandwidthMonitoringWebSocket: () => mockWebSocketData,
  usePerformancePredictionWebSocket: () => mockWebSocketData,
  useSupercomputeFabricWebSocket: () => mockWebSocketData,
}));

// Mock dynamic imports
jest.mock('next/dynamic', () => {
  return (fn: () => Promise<any>) => {
    const Component = React.lazy(fn);
    const DynamicComponent = (props: any) => (
      <React.Suspense fallback={<div>Loading...</div>}>
        <Component {...props} />
      </React.Suspense>
    );
    return DynamicComponent;
  };
});

// Import components after mocking
import { BandwidthMonitoringDashboard } from '@/components/monitoring/BandwidthMonitoringDashboard';
import { PerformancePredictionDashboard } from '@/components/monitoring/PerformancePredictionDashboard';
import { SupercomputeFabricDashboard } from '@/components/monitoring/SupercomputeFabricDashboard';
import { NetworkTopology } from '@/components/visualizations/NetworkTopology';

describe('Distributed Monitoring Components', () => {
  beforeEach(() => {
    // Reset any state between tests
    jest.clearAllMocks();
  });

  describe('BandwidthMonitoringDashboard', () => {
    it('renders with WebSocket data', async () => {
      render(<BandwidthMonitoringDashboard />);

      // Check for live status badge
      expect(screen.getByText('Live Updates')).toBeInTheDocument();

      // Check for bandwidth data
      await waitFor(() => {
        expect(screen.getByText('Primary Interface')).toBeInTheDocument();
        expect(screen.getByText('75%')).toBeInTheDocument();
      });
    });

    it('displays aggregated metrics', async () => {
      render(<BandwidthMonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('10,000')).toBeInTheDocument(); // Total capacity
        expect(screen.getByText('75%')).toBeInTheDocument(); // Utilization
      });
    });

    it('shows historical data chart', async () => {
      render(<BandwidthMonitoringDashboard />);

      // Look for chart elements
      await waitFor(() => {
        expect(screen.getByRole('region')).toBeInTheDocument();
      });
    });
  });

  describe('PerformancePredictionDashboard', () => {
    it('renders with prediction data', async () => {
      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Resource Predictions')).toBeInTheDocument();
        expect(screen.getByText('85%')).toBeInTheDocument(); // Confidence
      });
    });

    it('displays workload patterns', async () => {
      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Daily Peak')).toBeInTheDocument();
        expect(screen.getByText('90%')).toBeInTheDocument(); // Pattern confidence
      });
    });

    it('shows predictive recommendations', async () => {
      render(<PerformancePredictionDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/scaling up/i)).toBeInTheDocument();
      });
    });
  });

  describe('SupercomputeFabricDashboard', () => {
    it('renders with fabric data', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Global Resource Pool')).toBeInTheDocument();
        expect(screen.getByText('ML Training Job')).toBeInTheDocument();
      });
    });

    it('displays resource utilization', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByText('68%')).toBeInTheDocument(); // CPU utilization
        expect(screen.getByText('72%')).toBeInTheDocument(); // Memory utilization
      });
    });

    it('shows compute jobs', async () => {
      render(<SupercomputeFabricDashboard />);

      await waitFor(() => {
        expect(screen.getByText('running')).toBeInTheDocument();
        expect(screen.getByText('45%')).toBeInTheDocument(); // Progress
      });
    });
  });

  describe('NetworkTopology', () => {
    it('renders with topology data', async () => {
      const mockData = {
        nodes: mockWebSocketData.data.topology.nodes,
        edges: mockWebSocketData.data.topology.edges,
        clusters: []
      };

      render(
        <NetworkTopology
          data={mockData}
          showDistributed={true}
          showBandwidth={true}
          showPerformanceMetrics={true}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Network Topology')).toBeInTheDocument();
        expect(screen.getByText('Live Updates')).toBeInTheDocument();
      });
    });

    it('shows feature toggles', async () => {
      render(
        <NetworkTopology
          showDistributed={true}
          showBandwidth={true}
          showPerformanceMetrics={true}
        />
      );

      expect(screen.getByText('Distributed View')).toBeInTheDocument();
      expect(screen.getByText('Bandwidth')).toBeInTheDocument();
      expect(screen.getByText('Metrics')).toBeInTheDocument();
    });

    it('has canvas for visualization', async () => {
      render(<NetworkTopology />);

      await waitFor(() => {
        const canvas = screen.getByRole('img', { hidden: true });
        expect(canvas).toBeInTheDocument();
      });
    });
  });

  describe('WebSocket Integration', () => {
    it('all dashboards receive WebSocket updates', async () => {
      const { rerender } = render(<BandwidthMonitoringDashboard />);

      // Verify initial connection
      expect(screen.getByText('Live Updates')).toBeInTheDocument();

      // Mock WebSocket data update
      const updatedData = {
        ...mockWebSocketData.data,
        bandwidth: {
          ...mockWebSocketData.data.bandwidth,
          aggregated: {
            ...mockWebSocketData.data.bandwidth.aggregated,
            totalUtilization: 85
          }
        }
      };

      // Simulate WebSocket update
      act(() => {
        mockWebSocketData.data = updatedData;
        rerender(<BandwidthMonitoringDashboard />);
      });

      await waitFor(() => {
        expect(screen.getByText('85%')).toBeInTheDocument();
      });
    });

    it('handles connection status changes', async () => {
      // Test with disconnected state
      const disconnectedData = {
        ...mockWebSocketData,
        isConnected: false
      };

      jest.mocked(require('@/hooks/useWebSocket')).useMonitoringWebSocket.mockReturnValue(disconnectedData);

      render(<BandwidthMonitoringDashboard />);

      // Should not show live updates badge when disconnected
      expect(screen.queryByText('Live Updates')).not.toBeInTheDocument();
    });
  });
});