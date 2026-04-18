import { render, screen } from '@testing-library/react';

import StoragePage from '@/app/storage/page';

jest.mock('@/components/storage/StorageManagementUI', () => ({
  __esModule: true,
  default: () => <div data-testid="storage-management-ui">storage-management-ui</div>,
}));

describe('StoragePage', () => {
  it('delegates to the canonical storage management UI', () => {
    render(<StoragePage />);

    expect(screen.getByTestId('storage-management-ui')).toBeInTheDocument();
  });
});
