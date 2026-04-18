import { render, screen, waitFor } from '@testing-library/react';

import NetworkPage from '@/app/network/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: jest.fn() }),
}));

jest.mock('@/lib/api/hooks/useVMs', () => ({
  useVMs: () => ({
    items: [{ id: 'vm-1', name: 'Alpha', state: 'running' }],
  }),
}));

jest.mock('@/lib/api/networks', () => ({
  networkApi: {
    listNetworks: jest.fn().mockResolvedValue([
      {
        id: 'net-1',
        name: 'Production',
        type: 'bridged',
        subnet: '192.168.10.0/24',
        gateway: '192.168.10.1',
        status: 'active',
        created_at: '2026-04-18T00:00:00Z',
        updated_at: '2026-04-18T00:00:00Z',
      },
    ]),
    listVmInterfaces: jest.fn().mockResolvedValue([
      {
        id: 'eth0',
        vm_id: 'vm-1',
        network_id: 'net-1',
        name: 'eth0',
        mac_address: '00:16:3e:12:34:56',
        ip_address: '192.168.10.25',
        status: 'active',
        created_at: '2026-04-18T00:00:00Z',
        updated_at: '2026-04-18T00:00:00Z',
      },
    ]),
    createNetwork: jest.fn(),
    deleteNetwork: jest.fn(),
    attachVmInterface: jest.fn(),
    deleteVmInterface: jest.fn(),
  },
}));

describe('NetworkPage', () => {
  it('renders the canonical network inventory and interfaces', async () => {
    render(<NetworkPage />);

    expect(screen.getByText('Network')).toBeInTheDocument();
    expect(
      screen.getByText(/Canonical network inventory and VM interface management backed by/i),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Production')).toBeInTheDocument();
      expect(screen.getByText('Alpha')).toBeInTheDocument();
      expect(screen.getByText('eth0')).toBeInTheDocument();
    });
  });
});
