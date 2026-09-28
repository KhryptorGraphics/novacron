import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import NetworkPage from '@/app/network/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// jest.mock factories are hoisted above these declarations, so they are only
// dereferenced lazily (at render time), which is why the names carry the
// `mock` prefix jest requires.
const mockToast = jest.fn();
jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: mockToast }),
}));

const mockAuthState = { user: { role: 'admin', roles: ['admin'] } as { role: string; roles: string[] } | null };
jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => mockAuthState,
}));

jest.mock('@/lib/api/hooks/useVMs', () => ({
  useVMs: () => ({
    items: [{ id: 'vm-1', name: 'Alpha', state: 'running' }],
  }),
}));

const production = {
  id: '0f3c2a9e-51d4-4b7a-9c2e-7d1f00aa1234',
  name: 'Production',
  bridge: 'ncbr-0f3c2a9e51',
  cidr: '192.168.10.0/24',
  gateway: '192.168.10.1',
  vlan_id: 100,
  mtu: 1500,
  created_by: null,
  vm_count: 2,
  created_at: '2026-04-18T00:00:00Z',
  updated_at: '2026-04-18T00:00:00Z',
};

const lab = {
  ...production,
  id: '9b8a7c6d-5e4f-4a3b-8c2d-1e0f00bb5678',
  name: 'Lab',
  bridge: 'ncbr-9b8a7c6d5e',
  cidr: '10.50.0.0/24',
  gateway: null,
  vlan_id: null,
  mtu: 9000,
  vm_count: 0,
};

const mockNetworkApi = {
  listNetworks: jest.fn(),
  listVmInterfaces: jest.fn(),
  createNetwork: jest.fn(),
  deleteNetwork: jest.fn(),
  attachVmInterface: jest.fn(),
  deleteVmInterface: jest.fn(),
};
jest.mock('@/lib/api/networks', () => ({
  get networkApi() {
    return mockNetworkApi;
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
  mockAuthState.user = { role: 'admin', roles: ['admin'] };
  mockNetworkApi.listNetworks.mockResolvedValue([production, lab]);
  mockNetworkApi.listVmInterfaces.mockResolvedValue([
    {
      id: 'eth0',
      vm_id: 'vm-1',
      network_id: production.id,
      name: 'eth0',
      mac_address: '00:16:3e:12:34:56',
      ip_address: '192.168.10.25',
      status: 'attached',
      created_at: '2026-04-18T00:00:00Z',
      updated_at: '2026-04-18T00:00:00Z',
    },
  ]);
});

describe('NetworkPage', () => {
  it('renders the catalog with bridge, VLAN, MTU and attachment counts', async () => {
    render(<NetworkPage />);

    expect(screen.getByText('Network')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Production')).toBeInTheDocument();
    });
    expect(screen.getByText('ncbr-0f3c2a9e51')).toBeInTheDocument();
    expect(screen.getByText('192.168.10.0/24')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
    expect(screen.getByText('untagged')).toBeInTheDocument();
    expect(screen.getByText('none (L2 only)')).toBeInTheDocument();
    expect(screen.getByText('9000')).toBeInTheDocument();
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('eth0')).toBeInTheDocument();

    // A network with VMs attached cannot be deleted from the UI either.
    const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
    expect(deleteButtons).toHaveLength(2);
    expect(deleteButtons[0]).toBeDisabled();
    expect(deleteButtons[1]).toBeEnabled();
  });

  it('hides create/delete controls from non-admins', async () => {
    mockAuthState.user = { role: 'user', roles: ['user'] };
    render(<NetworkPage />);

    await waitFor(() => {
      expect(screen.getByText('Production')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: /add network/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
  });

  it('submits the create form as the POST /networks payload', async () => {
    mockNetworkApi.createNetwork.mockResolvedValue({
      ...lab,
      id: 'new-id',
      name: 'edge',
      bridge: 'ncbr-newnewnew1',
      cidr: '10.60.0.0/24',
      gateway: '10.60.0.1',
      vlan_id: 200,
      mtu: 1400,
    });
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText('Production')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /add network/i }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'edge' } });
    fireEvent.change(screen.getByLabelText('CIDR'), { target: { value: '10.60.0.0/24' } });
    fireEvent.change(screen.getByLabelText(/gateway/i), { target: { value: '10.60.0.1' } });
    fireEvent.change(screen.getByLabelText(/vlan id/i), { target: { value: '4095' } });
    fireEvent.change(screen.getByLabelText(/mtu/i), { target: { value: '1400' } });
    fireEvent.click(screen.getByRole('button', { name: /create network/i }));

    // Out-of-range VLAN is caught client-side; nothing is sent.
    expect(mockNetworkApi.createNetwork).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Invalid network', variant: 'destructive' }));

    fireEvent.change(screen.getByLabelText(/vlan id/i), { target: { value: '200' } });
    fireEvent.click(screen.getByRole('button', { name: /create network/i }));

    await waitFor(() => {
      expect(mockNetworkApi.createNetwork).toHaveBeenCalledWith({
        name: 'edge',
        cidr: '10.60.0.0/24',
        gateway: '10.60.0.1',
        vlan_id: 200,
        mtu: 1400,
      });
    });
    await waitFor(() => {
      expect(screen.getByText('edge')).toBeInTheDocument();
    });
  });
});
