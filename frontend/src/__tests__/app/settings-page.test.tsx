import { render, screen } from '@testing-library/react';

import SettingsPage from '@/app/settings/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/auth/TwoFactorSettings', () => ({
  __esModule: true,
  default: () => <div data-testid="two-factor-settings">two-factor-settings</div>,
}));

jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    user: {
      email: 'admin@novacron.test',
      firstName: 'Nova',
      lastName: 'Admin',
      tenantId: 'default',
      role: 'admin',
      roles: ['admin'],
    },
  }),
}));

describe('SettingsPage', () => {
  it('renders the account and security scoped settings surface', () => {
    render(<SettingsPage />);

    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText(/Account and security preferences only/i)).toBeInTheDocument();
    expect(screen.getByTestId('two-factor-settings')).toBeInTheDocument();
  });
});
