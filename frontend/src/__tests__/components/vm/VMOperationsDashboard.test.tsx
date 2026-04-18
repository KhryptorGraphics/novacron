import { render, screen } from '@testing-library/react';

import VMOperationsDashboard from '@/components/vm/VMOperationsDashboard';

jest.mock('@/app/core/vms/page', () => ({
  __esModule: true,
  default: () => <div data-testid="core-vms-page">core-vms-page</div>,
}));

describe('VMOperationsDashboard', () => {
  it('renders the canonical core VM page in snapshot mode', () => {
    render(<VMOperationsDashboard />);

    expect(screen.getByText('Snapshot Mode')).toBeInTheDocument();
    expect(screen.getByText('Realtime VM streams are unavailable')).toBeInTheDocument();
    expect(screen.getByTestId('core-vms-page')).toBeInTheDocument();
  });
});
