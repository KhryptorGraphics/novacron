import { render, screen } from '@testing-library/react';

import VMsPage from '@/app/vms/page';

jest.mock('@/components/vm/CanonicalVMsPage', () => ({
  __esModule: true,
  default: () => <div data-testid="canonical-vms-page">canonical-vms-page</div>,
}));

describe('VMsPage', () => {
  it('delegates to the canonical VMs page', () => {
    render(<VMsPage />);

    expect(screen.getByTestId('canonical-vms-page')).toBeInTheDocument();
  });
});
