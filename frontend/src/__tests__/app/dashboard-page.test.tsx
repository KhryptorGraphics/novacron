import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';

import DashboardPage from '@/app/dashboard/page';

expect.extend(toHaveNoViolations);

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/dashboard/UnifiedDashboard', () => ({
  __esModule: true,
  default: () => <div data-testid="unified-dashboard">unified-dashboard</div>,
}));

describe('DashboardPage', () => {
  it('delegates to the canonical unified dashboard', () => {
    render(<DashboardPage />);

    expect(screen.getByTestId('unified-dashboard')).toBeInTheDocument();
  });

  it('has no detectable accessibility violations', async () => {
    const { container } = render(<DashboardPage />);

    expect(await axe(container)).toHaveNoViolations();
  });
});
