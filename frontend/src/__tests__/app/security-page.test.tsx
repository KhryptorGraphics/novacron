import { render, screen } from '@testing-library/react';

import SecurityPage from '@/app/security/page';

jest.mock('@/components/security/SecurityComplianceDashboard', () => ({
  __esModule: true,
  default: () => <div data-testid="security-dashboard">security-dashboard</div>,
}));

describe('SecurityPage', () => {
  it('delegates to the canonical security dashboard', () => {
    render(<SecurityPage />);

    expect(screen.getByTestId('security-dashboard')).toBeInTheDocument();
  });
});
