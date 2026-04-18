import { render, screen } from '@testing-library/react';

import VMsPage from '@/app/vms/page';

jest.mock('@/app/core/vms/page', () => ({
  __esModule: true,
  default: () => <div data-testid="core-vms-page">core-vms-page</div>,
}));

describe('VMsPage', () => {
  it('delegates to the canonical core VMs page', () => {
    render(<VMsPage />);

    expect(screen.getByTestId('core-vms-page')).toBeInTheDocument();
  });
});
