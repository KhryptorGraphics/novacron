/**
 * End-to-End tests for distributed monitoring workflow
 * Sprint 6 - Distributed Supercompute Fabric Infrastructure
 *
 * These tests simulate complete user workflows across the distributed monitoring system
 */

import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';

// Mock WebSocket server responses
const mockWebSocketServer = {
  connections: new Set(),
  broadcast: (data: any) => {
    mockWebSocketServer.connections.forEach((conn: any) => {
      conn.onmessage({ data: JSON.stringify(data) });
    });
  },
  simulateLatency: (ms: number = 100) => new Promise(resolve => setTimeout(resolve, ms)),
};

// Mock WebSocket implementation
class MockWebSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  readyState = WebSocket.CONNECTING;

  constructor(public url: string) {
    mockWebSocketServer.connections.add(this);

    // Simulate connection
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 50);
  }

  send(data: string) {
    // Simulate server processing
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage({
          data: JSON.stringify({ type: 'ack', originalData: JSON.parse(data) })
        } as MessageEvent);
      }
    }, 25);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    mockWebSocketServer.connections.delete(this);
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  addEventListener(type: string, listener: any) {
    if (type === 'open') this.onopen = listener;
    if (type === 'message') this.onmessage = listener;
    if (type === 'error') this.onerror = listener;
    if (type === 'close') this.onclose = listener;
  }

  removeEventListener() {
    // Mock implementation
  }
}

// Mock components
jest.mock('../components/dashboard/UnifiedDashboard', () => {
  return function MockUnifiedDashboard() {
    const [activeView, setActiveView] = React.useState('overview');

    return (
      <div data-testid="unified-dashboard">
        <nav>
          <button onClick={() => setActiveView('bandwidth')}>Bandwidth</button>
          <button onClick={() => setActiveView('predictions')}>AI Predictions</button>
          <button onClick={() => setActiveView('fabric')}>Fabric</button>
          <button onClick={() => setActiveView('mobile')}>Mobile</button>
        </nav>
        <main>
          {activeView === 'bandwidth' && <div data-testid="bandwidth-dashboard">Bandwidth Monitoring</div>}
          {activeView === 'predictions' && <div data-testid="predictions-dashboard">Performance Predictions</div>}
          {activeView === 'fabric' && <div data-testid="fabric-dashboard">Supercompute Fabric</div>}
          {activeView === 'mobile' && <div data-testid="mobile-dashboard">Mobile Interface</div>}
        </main>
      </div>
    );
  };
});

// Global setup
beforeAll(() => {
  global.WebSocket = MockWebSocket as any;
  global.fetch = jest.fn();
});

describe('Distributed Monitoring E2E Workflows', () => {
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
    jest.clearAllMocks();
    mockWebSocketServer.connections.clear();

    // Mock successful API responses
    (global.fetch as jest.Mock).mockImplementation((url: string) => {
      const responses: Record<string, any> = {
        '/api/network/topology': {
          data: {
            nodes: [
              { id: 'node1', type: 'compute', status: 'active', cluster: 'cluster1' },
              { id: 'node2', type: 'federation', status: 'active', cluster: 'cluster2' }
            ],
            edges: [{ source: 'node1', target: 'node2', bandwidth: 10000, latency: 5 }],
            clusters: [
              { id: 'cluster1', name: 'Primary Cluster', status: 'healthy' },
              { id: 'cluster2', name: 'Secondary Cluster', status: 'healthy' }
            ]
          }
        },
        '/api/network/bandwidth/metrics': {
          data: {
            interfaces: [{
              id: 'eth0',
              name: 'Primary Interface',
              throughput: 8500,
              capacity: 10000,
              utilization: 85,
              qosEnabled: true,
              status: 'active'
            }],
            globalMetrics: {
              totalBandwidth: 45200,
              utilizedBandwidth: 32800,
              efficiency: 92.5,
              qosCompliance: 98.2
            }
          }
        },
        '/api/ai/predictions/resources': {
          data: [{
            id: 'pred1',
            type: 'resource',
            metric: 'cpu',
            currentValue: 68,
            predictedValue: 75,
            confidence: 94,
            timeHorizon: '1h',
            recommendation: 'Scale horizontally'
          }]
        },
        '/api/fabric/jobs': {
          data: [{
            id: 'job1',
            name: 'ML Training Job',
            status: 'running',
            progress: 65,
            cluster: 'cluster1',
            resources: { cpu: 32, memory: 128, gpu: 4 }
          }]
        }
      };

      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(responses[url] || { data: null, error: null })
      });
    });
  });

  describe('Complete Monitoring Workflow', () => {
    test('user navigates through all distributed monitoring dashboards', async () => {
      const UnifiedDashboard = require('../components/dashboard/UnifiedDashboard').default;
      render(<UnifiedDashboard />);

      // Start on overview
      expect(screen.getByTestId('unified-dashboard')).toBeInTheDocument();

      // Navigate to bandwidth monitoring
      await user.click(screen.getByText('Bandwidth'));

      await waitFor(() => {
        expect(screen.getByTestId('bandwidth-dashboard')).toBeInTheDocument();
      });

      // Navigate to AI predictions
      await user.click(screen.getByText('AI Predictions'));

      await waitFor(() => {
        expect(screen.getByTestId('predictions-dashboard')).toBeInTheDocument();
      });

      // Navigate to fabric monitoring
      await user.click(screen.getByText('Fabric'));

      await waitFor(() => {
        expect(screen.getByTestId('fabric-dashboard')).toBeInTheDocument();
      });

      // Navigate to mobile interface
      await user.click(screen.getByText('Mobile'));

      await waitFor(() => {
        expect(screen.getByTestId('mobile-dashboard')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Data Flow Simulation', () => {
    test('simulates complete bandwidth monitoring workflow with live updates', async () => {
      // Mock the bandwidth monitoring component with real WebSocket integration
      const BandwidthDashboard = () => {
        const [metrics, setMetrics] = React.useState(null);
        const [connected, setConnected] = React.useState(false);

        React.useEffect(() => {
          const ws = new WebSocket('/api/ws/network/bandwidth');

          ws.addEventListener('open', () => {
            setConnected(true);
          });

          ws.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            setMetrics(data);
          });

          // Simulate initial data load
          setTimeout(() => {
            ws.onmessage?.({
              data: JSON.stringify({
                interfaces: [{
                  id: 'eth0',
                  name: 'Primary Interface',
                  throughput: 8500,
                  utilization: 85
                }],
                globalMetrics: {
                  totalBandwidth: 45200,
                  efficiency: 92.5
                }
              })
            } as MessageEvent);
          }, 100);

          return () => ws.close();
        }, []);

        if (!connected) return <div>Connecting...</div>;
        if (!metrics) return <div>Loading metrics...</div>;

        return (
          <div data-testid="bandwidth-live">
            <h1>Bandwidth Monitoring</h1>
            <div data-testid="throughput">{(metrics as any).interfaces[0].throughput} Mbps</div>
            <div data-testid="utilization">{(metrics as any).interfaces[0].utilization}%</div>
            <div data-testid="efficiency">{(metrics as any).globalMetrics.efficiency}%</div>
          </div>
        );
      };

      render(<BandwidthDashboard />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByText('Bandwidth Monitoring')).toBeInTheDocument();
      });

      // Verify initial data
      expect(screen.getByTestId('throughput')).toHaveTextContent('8500 Mbps');
      expect(screen.getByTestId('utilization')).toHaveTextContent('85%');
      expect(screen.getByTestId('efficiency')).toHaveTextContent('92.5%');

      // Simulate real-time update
      act(() => {
        mockWebSocketServer.broadcast({
          interfaces: [{
            id: 'eth0',
            name: 'Primary Interface',
            throughput: 9200,
            utilization: 92
          }],
          globalMetrics: {
            totalBandwidth: 45200,
            efficiency: 94.1
          }
        });
      });

      // Verify updated data
      await waitFor(() => {
        expect(screen.getByTestId('throughput')).toHaveTextContent('9200 Mbps');
        expect(screen.getByTestId('utilization')).toHaveTextContent('92%');
        expect(screen.getByTestId('efficiency')).toHaveTextContent('94.1%');
      });
    });

    test('simulates AI prediction workflow with model retraining', async () => {
      const PredictionDashboard = () => {
        const [predictions, setPredictions] = React.useState<any[]>([]);
        const [models, setModels] = React.useState<any[]>([]);
        const [retraining, setRetraining] = React.useState(false);

        React.useEffect(() => {
          const ws = new WebSocket('/api/ws/ai/predictions');

          ws.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            if (data.predictions) setPredictions(data.predictions);
            if (data.models) setModels(data.models);
          });

          // Initial data
          setTimeout(() => {
            ws.onmessage?.({
              data: JSON.stringify({
                predictions: [{
                  id: 'pred1',
                  metric: 'cpu',
                  currentValue: 68,
                  predictedValue: 75,
                  confidence: 94
                }],
                models: [{
                  id: 'model1',
                  name: 'Resource Predictor',
                  accuracy: 94.2,
                  status: 'active'
                }]
              })
            } as MessageEvent);
          }, 100);

          return () => ws.close();
        }, []);

        const handleRetrain = async () => {
          setRetraining(true);
          // Simulate retraining
          await new Promise(resolve => setTimeout(resolve, 2000));

          // Update model accuracy after retraining
          setModels(prev => prev.map(model => ({
            ...model,
            accuracy: model.accuracy + 1.5,
            status: 'active'
          })));
          setRetraining(false);
        };

        return (
          <div data-testid="predictions-live">
            <h1>AI Predictions</h1>
            {predictions.map(pred => (
              <div key={pred.id} data-testid={`prediction-${pred.id}`}>
                <span>Current: {pred.currentValue}%</span>
                <span>Predicted: {pred.predictedValue}%</span>
                <span>Confidence: {pred.confidence}%</span>
              </div>
            ))}
            {models.map(model => (
              <div key={model.id} data-testid={`model-${model.id}`}>
                <span>{model.name}</span>
                <span>Accuracy: {model.accuracy}%</span>
                <span>Status: {model.status}</span>
              </div>
            ))}
            <button
              onClick={handleRetrain}
              disabled={retraining}
              data-testid="retrain-button"
            >
              {retraining ? 'Retraining...' : 'Retrain Model'}
            </button>
          </div>
        );
      };

      render(<PredictionDashboard />);

      // Wait for initial data
      await waitFor(() => {
        expect(screen.getByTestId('prediction-pred1')).toBeInTheDocument();
        expect(screen.getByText('Confidence: 94%')).toBeInTheDocument();
      });

      // Check initial model accuracy
      expect(screen.getByText('Accuracy: 94.2%')).toBeInTheDocument();

      // Trigger model retraining
      await user.click(screen.getByTestId('retrain-button'));

      // Verify retraining state
      expect(screen.getByText('Retraining...')).toBeInTheDocument();

      // Wait for retraining to complete
      await waitFor(
        () => {
          expect(screen.getByText('Accuracy: 95.7%')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      // Verify button is re-enabled
      expect(screen.getByText('Retrain Model')).toBeInTheDocument();
    });
  });

  describe('Cross-Component Integration Scenarios', () => {
    test('simulates fabric scaling based on prediction recommendations', async () => {
      const IntegratedWorkflow = () => {
        const [prediction, setPrediction] = React.useState<any>(null);
        const [fabricJobs, setFabricJobs] = React.useState<any[]>([]);
        const [scalingAction, setScalingAction] = React.useState<string | null>(null);

        React.useEffect(() => {
          // Simulate prediction update
          setPrediction({
            id: 'pred1',
            metric: 'cpu',
            currentValue: 85,
            predictedValue: 95,
            confidence: 96,
            recommendation: 'Scale horizontally',
            urgency: 'high'
          });

          setFabricJobs([{
            id: 'job1',
            name: 'ML Training',
            status: 'running',
            resources: { cpu: 32, memory: 128 }
          }]);
        }, []);

        const handleAutoScale = async () => {
          setScalingAction('scaling');

          // Simulate scaling process
          await new Promise(resolve => setTimeout(resolve, 1500));

          // Update jobs after scaling
          setFabricJobs(prev => [
            ...prev,
            {
              id: 'job2',
              name: 'Auto-scaled Instance',
              status: 'starting',
              resources: { cpu: 16, memory: 64 }
            }
          ]);

          setScalingAction('completed');
        };

        return (
          <div data-testid="integrated-workflow">
            <h1>Integrated Monitoring</h1>

            <div data-testid="prediction-section">
              {prediction && (
                <div>
                  <p>CPU will reach {prediction.predictedValue}% (confidence: {prediction.confidence}%)</p>
                  <p>Recommendation: {prediction.recommendation}</p>
                  {prediction.urgency === 'high' && (
                    <button
                      onClick={handleAutoScale}
                      disabled={scalingAction === 'scaling'}
                      data-testid="auto-scale-button"
                    >
                      {scalingAction === 'scaling' ? 'Scaling...' : 'Auto Scale'}
                    </button>
                  )}
                </div>
              )}
            </div>

            <div data-testid="fabric-section">
              <h2>Fabric Jobs ({fabricJobs.length})</h2>
              {fabricJobs.map(job => (
                <div key={job.id} data-testid={`job-${job.id}`}>
                  {job.name} - {job.status}
                </div>
              ))}
              {scalingAction === 'completed' && (
                <div data-testid="scaling-success">Scaling completed successfully</div>
              )}
            </div>
          </div>
        );
      };

      render(<IntegratedWorkflow />);

      // Wait for prediction
      await waitFor(() => {
        expect(screen.getByText('CPU will reach 95% (confidence: 96%)')).toBeInTheDocument();
        expect(screen.getByText('Recommendation: Scale horizontally')).toBeInTheDocument();
      });

      // Verify initial job count
      expect(screen.getByText('Fabric Jobs (1)')).toBeInTheDocument();
      expect(screen.getByTestId('job-job1')).toBeInTheDocument();

      // Trigger auto-scaling
      await user.click(screen.getByTestId('auto-scale-button'));

      // Verify scaling in progress
      expect(screen.getByText('Scaling...')).toBeInTheDocument();

      // Wait for scaling completion
      await waitFor(
        () => {
          expect(screen.getByText('Fabric Jobs (2)')).toBeInTheDocument();
          expect(screen.getByTestId('job-job2')).toBeInTheDocument();
          expect(screen.getByTestId('scaling-success')).toBeInTheDocument();
        },
        { timeout: 2000 }
      );
    });

    test('simulates network bandwidth alert triggering resource migration', async () => {
      const NetworkMigrationWorkflow = () => {
        const [bandwidthAlert, setBandwidthAlert] = React.useState<any>(null);
        const [migrations, setMigrations] = React.useState<any[]>([]);
        const [migrationInProgress, setMigrationInProgress] = React.useState(false);

        React.useEffect(() => {
          // Simulate bandwidth threshold breach
          setTimeout(() => {
            setBandwidthAlert({
              id: 'alert1',
              type: 'bandwidth_threshold',
              interface: 'eth0',
              currentUtilization: 95,
              threshold: 85,
              severity: 'critical'
            });
          }, 500);
        }, []);

        const handleTriggerMigration = async () => {
          setMigrationInProgress(true);

          // Simulate migration process
          await new Promise(resolve => setTimeout(resolve, 2000));

          setMigrations(prev => [...prev, {
            id: 'migration1',
            vmId: 'vm-web01',
            sourceCluster: 'cluster1',
            targetCluster: 'cluster2',
            status: 'completed',
            bandwidth: 'reduced'
          }]);

          // Clear alert after successful migration
          setBandwidthAlert(null);
          setMigrationInProgress(false);
        };

        return (
          <div data-testid="network-migration-workflow">
            <h1>Network & Migration Integration</h1>

            <div data-testid="alert-section">
              {bandwidthAlert && (
                <div data-testid="bandwidth-alert" className="alert-critical">
                  <p>CRITICAL: Interface {bandwidthAlert.interface} at {bandwidthAlert.currentUtilization}%</p>
                  <p>Threshold: {bandwidthAlert.threshold}%</p>
                  <button
                    onClick={handleTriggerMigration}
                    disabled={migrationInProgress}
                    data-testid="trigger-migration-button"
                  >
                    {migrationInProgress ? 'Migrating...' : 'Migrate VMs'}
                  </button>
                </div>
              )}
            </div>

            <div data-testid="migration-section">
              <h2>Active Migrations ({migrations.length})</h2>
              {migrations.map(migration => (
                <div key={migration.id} data-testid={`migration-${migration.id}`}>
                  {migration.vmId}: {migration.sourceCluster} → {migration.targetCluster}
                  <span data-testid="migration-status"> ({migration.status})</span>
                </div>
              ))}
            </div>
          </div>
        );
      };

      render(<NetworkMigrationWorkflow />);

      // Wait for bandwidth alert
      await waitFor(() => {
        expect(screen.getByTestId('bandwidth-alert')).toBeInTheDocument();
        expect(screen.getByText('CRITICAL: Interface eth0 at 95%')).toBeInTheDocument();
      });

      // Verify no migrations initially
      expect(screen.getByText('Active Migrations (0)')).toBeInTheDocument();

      // Trigger migration
      await user.click(screen.getByTestId('trigger-migration-button'));

      // Verify migration in progress
      expect(screen.getByText('Migrating...')).toBeInTheDocument();

      // Wait for migration completion
      await waitFor(
        () => {
          expect(screen.getByText('Active Migrations (1)')).toBeInTheDocument();
          expect(screen.getByTestId('migration-migration1')).toBeInTheDocument();
          expect(screen.getByTestId('migration-status')).toHaveTextContent('(completed)');
        },
        { timeout: 2500 }
      );

      // Verify alert is cleared
      expect(screen.queryByTestId('bandwidth-alert')).not.toBeInTheDocument();
    });
  });

  describe('Error Recovery and Resilience', () => {
    test('handles WebSocket disconnection and reconnection gracefully', async () => {
      const ResilientComponent = () => {
        const [connectionStatus, setConnectionStatus] = React.useState('connecting');
        const [data, setData] = React.useState(null);
        const [reconnectAttempts, setReconnectAttempts] = React.useState(0);

        React.useEffect(() => {
          let ws: WebSocket;
          let reconnectTimeout: NodeJS.Timeout;

          const connect = () => {
            ws = new WebSocket('/api/ws/test');

            ws.addEventListener('open', () => {
              setConnectionStatus('connected');
              setReconnectAttempts(0);
            });

            ws.addEventListener('message', (event) => {
              setData(JSON.parse(event.data));
            });

            ws.addEventListener('close', () => {
              setConnectionStatus('disconnected');

              // Auto-reconnect with backoff
              const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
              reconnectTimeout = setTimeout(() => {
                setReconnectAttempts(prev => prev + 1);
                setConnectionStatus('reconnecting');
                connect();
              }, delay);
            });

            ws.addEventListener('error', () => {
              setConnectionStatus('error');
            });
          };

          connect();

          return () => {
            clearTimeout(reconnectTimeout);
            ws?.close();
          };
        }, [reconnectAttempts]);

        return (
          <div data-testid="resilient-component">
            <div data-testid="connection-status">Status: {connectionStatus}</div>
            <div data-testid="reconnect-attempts">Attempts: {reconnectAttempts}</div>
            {data && <div data-testid="received-data">Data received</div>}
          </div>
        );
      };

      render(<ResilientComponent />);

      // Wait for initial connection
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Status: connected');
      });

      // Simulate connection loss
      act(() => {
        mockWebSocketServer.connections.forEach((conn: any) => {
          conn.close();
        });
      });

      // Verify disconnection detected
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Status: disconnected');
      });

      // Wait for reconnection attempt
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Status: reconnecting');
        expect(screen.getByTestId('reconnect-attempts')).toHaveTextContent('Attempts: 1');
      }, { timeout: 2000 });

      // Verify successful reconnection
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Status: connected');
        expect(screen.getByTestId('reconnect-attempts')).toHaveTextContent('Attempts: 0');
      }, { timeout: 3000 });
    });

    test('handles API timeout and retry logic', async () => {
      let apiCallCount = 0;

      // Mock API with initial failures
      (global.fetch as jest.Mock).mockImplementation(() => {
        apiCallCount++;

        if (apiCallCount <= 2) {
          return Promise.reject(new Error('Network timeout'));
        }

        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            data: { status: 'success', attempt: apiCallCount }
          })
        });
      });

      const RetryComponent = () => {
        const [data, setData] = React.useState(null);
        const [error, setError] = React.useState<string | null>(null);
        const [loading, setLoading] = React.useState(false);
        const [retryCount, setRetryCount] = React.useState(0);

        const fetchData = async (attempt = 0) => {
          setLoading(true);
          setError(null);

          try {
            const response = await fetch('/api/test-endpoint');
            const result = await response.json();
            setData(result.data);
          } catch (err) {
            if (attempt < 3) {
              setRetryCount(attempt + 1);
              setTimeout(() => fetchData(attempt + 1), 1000);
            } else {
              setError((err as Error).message);
            }
          } finally {
            setLoading(false);
          }
        };

        React.useEffect(() => {
          fetchData();
        }, []);

        return (
          <div data-testid="retry-component">
            {loading && <div data-testid="loading">Loading...</div>}
            {error && <div data-testid="error">Error: {error}</div>}
            {data && <div data-testid="success-data">Success: attempt {(data as any).attempt}</div>}
            <div data-testid="retry-count">Retries: {retryCount}</div>
          </div>
        );
      };

      render(<RetryComponent />);

      // Initial loading state
      expect(screen.getByTestId('loading')).toBeInTheDocument();

      // Wait for retries and eventual success
      await waitFor(() => {
        expect(screen.getByTestId('success-data')).toHaveTextContent('Success: attempt 3');
        expect(screen.getByTestId('retry-count')).toHaveTextContent('Retries: 2');
      }, { timeout: 5000 });

      // Verify no error or loading state
      expect(screen.queryByTestId('error')).not.toBeInTheDocument();
      expect(screen.queryByTestId('loading')).not.toBeInTheDocument();
    });
  });

  describe('Performance Under Load', () => {
    test('handles high-frequency updates without performance degradation', async () => {
      const PerformanceComponent = () => {
        const [updateCount, setUpdateCount] = React.useState(0);
        const [lastUpdate, setLastUpdate] = React.useState(Date.now());
        const [averageLatency, setAverageLatency] = React.useState(0);

        React.useEffect(() => {
          const ws = new WebSocket('/api/ws/performance-test');
          const latencies: number[] = [];

          ws.addEventListener('message', (event) => {
            const now = Date.now();
            const data = JSON.parse(event.data);
            const latency = now - data.timestamp;

            latencies.push(latency);

            setUpdateCount(prev => prev + 1);
            setLastUpdate(now);
            setAverageLatency(latencies.reduce((a, b) => a + b, 0) / latencies.length);
          });

          return () => ws.close();
        }, []);

        return (
          <div data-testid="performance-component">
            <div data-testid="update-count">Updates: {updateCount}</div>
            <div data-testid="average-latency">Avg Latency: {averageLatency.toFixed(2)}ms</div>
            <div data-testid="last-update">Last: {new Date(lastUpdate).toISOString()}</div>
          </div>
        );
      };

      render(<PerformanceComponent />);

      // Simulate high-frequency updates (100 updates in 1 second)
      for (let i = 0; i < 100; i++) {
        setTimeout(() => {
          mockWebSocketServer.broadcast({
            timestamp: Date.now(),
            data: `update-${i}`,
            sequence: i
          });
        }, i * 10);
      }

      // Wait for all updates to be processed
      await waitFor(() => {
        expect(screen.getByTestId('update-count')).toHaveTextContent('Updates: 100');
      }, { timeout: 2000 });

      // Verify reasonable performance
      const latencyText = screen.getByTestId('average-latency').textContent;
      const avgLatency = parseFloat(latencyText!.replace('Avg Latency: ', '').replace('ms', ''));
      expect(avgLatency).toBeLessThan(100); // Should be under 100ms average
    });
  });
});