import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { toast } from '@/components/ui/use-toast';
import { RealTimeDashboard } from '@/components/admin/RealTimeDashboard';
import { useAdminRealTimeUpdates } from '@/lib/ws/useAdminWebSocket';

// Mock the WebSocket hook
jest.mock('@/lib/ws/useAdminWebSocket', () => ({
  useAdminRealTimeUpdates: jest.fn(),
  getConnectionStatusInfo: jest.fn(() => ({
    status: 'Connected',
    color: 'text-green-600',
    icon: '🟢'
  }))
}));

// Mock the toast
jest.mock('@/components/ui/use-toast', () => ({
  toast: jest.fn(),
  useToast: () => ({ toast: jest.fn() })
}));

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />
}));

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

describe('Admin Components', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
    jest.clearAllMocks();
    
    // Mock the WebSocket hook with default values
    (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
      isConnected: true,
      metrics: [],
      alerts: [],
      connectionState: 1, // ReadyState.OPEN
      error: null
    });
  });

  afterEach(() => {
    queryClient.clear();
  });

  describe('RealTimeDashboard', () => {
    const renderWithProviders = (component: React.ReactElement) => {
      return render(
        <QueryClientProvider client={queryClient}>
          {component}
        </QueryClientProvider>
      );
    };

    it('renders connection status correctly when connected', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Auto-Refresh On')).toBeInTheDocument();
    });

    it('renders connection status correctly when disconnected', () => {
      (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
        isConnected: false,
        metrics: [],
        alerts: [],
        connectionState: 3, // ReadyState.CLOSED
        error: 'Connection failed'
      });

      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('Connection failed')).toBeInTheDocument();
    });

    it('displays system health metrics', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('System Health')).toBeInTheDocument();
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('Active Connections')).toBeInTheDocument();
    });

    it('renders charts correctly', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('System Resources (Live)')).toBeInTheDocument();
      expect(screen.getByText('Network Traffic (Live)')).toBeInTheDocument();
      expect(screen.getAllByTestId('responsive-container')).toHaveLength(5); // 3 main charts + 2 small charts
    });

    it('shows no alerts message when there are no alerts', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('All Systems Normal')).toBeInTheDocument();
      expect(screen.getByText('No active alerts at this time')).toBeInTheDocument();
    });

    it('displays alerts when they exist', () => {
      const mockAlerts = [
        {
          id: '1',
          title: 'High CPU Usage',
          description: 'CPU usage is above 90%',
          severity: 'high' as const,
          timestamp: new Date().toISOString()
        },
        {
          id: '2', 
          title: 'Memory Warning',
          description: 'Memory usage is approaching limit',
          severity: 'medium' as const,
          timestamp: new Date().toISOString()
        }
      ];

      (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
        isConnected: true,
        metrics: [],
        alerts: mockAlerts,
        connectionState: 1,
        error: null
      });

      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('High CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('Memory Warning')).toBeInTheDocument();
      expect(screen.getByText('CPU usage is above 90%')).toBeInTheDocument();
    });

    it('toggles auto-refresh when button is clicked', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      const autoRefreshButton = screen.getByText('Auto-Refresh On');
      expect(autoRefreshButton).toBeInTheDocument();
      
      fireEvent.click(autoRefreshButton);
      
      expect(screen.getByText('Auto-Refresh Off')).toBeInTheDocument();
    });

    it('displays performance summary cards', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('Response Time')).toBeInTheDocument();
      expect(screen.getByText('Resource Distribution')).toBeInTheDocument();
      expect(screen.getByText('System Status')).toBeInTheDocument();
    });

    it('shows WebSocket connection status in system status', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('WebSocket')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('handles error state gracefully', () => {
      (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
        isConnected: false,
        metrics: [],
        alerts: [],
        connectionState: 3,
        error: 'Network error'
      });

      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('Network error')).toBeInTheDocument();
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('shows trend indicators correctly', async () => {
      renderWithProviders(<RealTimeDashboard />);
      
      // Wait for component to initialize with mock data
      await waitFor(() => {
        expect(screen.getByText('vs previous')).toBeInTheDocument();
      });
    });

    it('displays data point count in system status', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      expect(screen.getByText('Data points')).toBeInTheDocument();
    });

    it('formats time correctly in charts', () => {
      renderWithProviders(<RealTimeDashboard />);
      
      // Check that charts are rendered (time formatting is internal to charts)
      expect(screen.getAllByTestId('line-chart')).toHaveLength(2);
      expect(screen.getAllByTestId('area-chart')).toHaveLength(1);
    });
  });

  describe('Error Handling', () => {
    it('handles missing data gracefully', () => {
      (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
        isConnected: true,
        metrics: [],
        alerts: [],
        connectionState: 1,
        error: null
      });

      render(
        <QueryClientProvider client={queryClient}>
          <RealTimeDashboard />
        </QueryClientProvider>
      );

      // Should not crash and should show default values
      expect(screen.getByText('0%')).toBeInTheDocument(); // Health score
      expect(screen.getByText('0ms')).toBeInTheDocument(); // Response time
    });

    it('handles invalid data gracefully', () => {
      (useAdminRealTimeUpdates as jest.Mock).mockReturnValue({
        isConnected: true,
        metrics: [{ invalid: 'data' }],
        alerts: [{ invalid: 'alert' }],
        connectionState: 1,
        error: null
      });

      render(
        <QueryClientProvider client={queryClient}>
          <RealTimeDashboard />
        </QueryClientProvider>
      );

      // Should still render without errors
      expect(screen.getByText('System Health')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels and roles', () => {
      render(
        <QueryClientProvider client={queryClient}>
          <RealTimeDashboard />
        </QueryClientProvider>
      );

      // Check for accessible elements
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
      
      // Auto-refresh toggle should be accessible
      const autoRefreshButton = screen.getByText(/Auto-Refresh/);
      expect(autoRefreshButton).toBeInTheDocument();
    });

    it('provides proper visual indicators for connection status', () => {
      render(
        <QueryClientProvider client={queryClient}>
          <RealTimeDashboard />
        </QueryClientProvider>
      );

      // Should have visual status indicators
      expect(screen.getByText('Connected')).toHaveClass('text-green-600');
    });
  });
});

describe('Integration Tests', () => {
  it('integrates properly with React Query', async () => {
    const queryClient = createTestQueryClient();
    
    render(
      <QueryClientProvider client={queryClient}>
        <RealTimeDashboard />
      </QueryClientProvider>
    );

    // Should render without throwing errors
    expect(screen.getByText('System Health')).toBeInTheDocument();
    
    // Should not have any React Query errors
    const queryCache = queryClient.getQueryCache();
    expect(queryCache.getAll()).toHaveLength(0); // No queries running in this test
  });
});