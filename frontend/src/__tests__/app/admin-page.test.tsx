import { render, screen } from '@testing-library/react';

import AdminPage from '@/app/admin/page';

jest.mock('@/components/accessibility/a11y-components', () => ({
  SkipToMain: () => <div data-testid="skip-to-main" />,
}));

jest.mock('@/components/error-boundary', () => ({
  ErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/theme/theme-toggle', () => ({
  ThemeToggle: () => <div data-testid="theme-toggle" />,
}));

jest.mock('@/components/ui/mobile-navigation', () => ({
  MobileNavigation: () => <div data-testid="mobile-navigation" />,
  DesktopSidebar: () => <div data-testid="desktop-sidebar" />,
}));

jest.mock('@/components/ui/loading-states', () => ({
  DashboardSkeleton: () => <div data-testid="dashboard-skeleton" />,
  RefreshIndicator: () => <div data-testid="refresh-indicator" />,
  LoadingStates: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/ui/progressive-disclosure', () => ({
  LazyTabs: ({ tabs }: { tabs: Array<{ label: string; content: React.ReactNode }> }) => (
    <div>
      {tabs.map((tab) => (
        <div key={tab.label}>
          <span>{tab.label}</span>
          {tab.content}
        </div>
      ))}
    </div>
  ),
}));

jest.mock('@/components/security/SecurityComplianceDashboard', () => ({
  __esModule: true,
  default: () => <div data-testid="security-dashboard">security-dashboard</div>,
}));

jest.mock('@/components/admin/RolePermissionManager', () => ({
  __esModule: true,
  default: () => <div data-testid="role-permission-manager">role-permission-manager</div>,
}));

jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    user: {
      email: 'admin@example.com',
      role: 'admin',
      roles: ['admin'],
    },
    logout: jest.fn(),
  }),
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}));

jest.mock('@/lib/api/security', () => ({
  securityAPI: {
    getAuditTrail: jest.fn().mockResolvedValue({ events: [] }),
    exportSecurityReport: jest.fn().mockResolvedValue(undefined),
  },
}));

describe('AdminPage', () => {
  it('renders only the release-candidate admin tabs', async () => {
    render(<AdminPage />);

    expect(screen.getByText('Security')).toBeInTheDocument();
    expect(screen.getByText('Roles & Permissions')).toBeInTheDocument();
    expect(screen.getByText('Audit')).toBeInTheDocument();
    expect(screen.queryByText('Overview')).not.toBeInTheDocument();
    expect(screen.queryByText('Real-time Monitor')).not.toBeInTheDocument();
    expect(screen.getByTestId('security-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('role-permission-manager')).toBeInTheDocument();
  });
});
