import { render, screen, waitFor } from '@testing-library/react';

import AnalyticsPage from '@/app/analytics/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/lib/api/hooks/useVMs', () => ({
  useVMs: () => ({
    items: [
      { id: 'vm-1', name: 'Alpha', state: 'running' },
      { id: 'vm-2', name: 'Beta', state: 'stopped' },
    ],
    isLoading: false,
  }),
}));

jest.mock('@/hooks/useVolumes', () => ({
  useVolumes: () => ({
    volumes: [{ id: 'vol-1', name: 'Primary', size: 100, tier: 'HOT', vmId: 'vm-1' }],
    loading: false,
  }),
}));

jest.mock('@/hooks/useSecurity', () => ({
  useSecurityMetrics: () => ({
    metrics: {
      securityScore: 92,
      complianceScore: 88,
      activeThreats: 2,
    },
    loading: false,
  }),
}));

jest.mock('@/lib/api/networks', () => ({
  networkApi: {
    listNetworks: jest.fn().mockResolvedValue([{ id: 'net-1', name: 'Production' }]),
  },
}));

describe('AnalyticsPage', () => {
  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        currentCpuUsage: 22.4,
        currentMemoryUsage: 47.8,
        currentDiskUsage: 59.1,
        currentNetworkUsage: 12.2,
      }),
    }) as jest.Mock;
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('renders the live operational analytics view', async () => {
    render(<AnalyticsPage />);

    expect(screen.getByText('Analytics')).toBeInTheDocument();
    expect(screen.getByText(/Historical trends are intentionally unavailable/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Virtual machines')).toBeInTheDocument();
      expect(screen.getByText('Networks')).toBeInTheDocument();
    });
  });
});
