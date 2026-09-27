import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

import { getVM } from '../vms';
import { useVM } from './useVM';

jest.mock('../vms', () => ({
  getVM: jest.fn(),
}));

const vm = {
  id: 'vm-1',
  name: 'alpha',
  state: 'running',
  node_id: 'node-a',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('useVM', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the bare VM object the backend actually returns', async () => {
    (getVM as jest.Mock).mockResolvedValue(vm);

    const { result } = renderHook(() => useVM('vm-1'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.vm).toEqual(vm);
    expect(result.current.error).toBeNull();
  });

  it('unwraps an { data } envelope if the backend ever wraps the VM', async () => {
    (getVM as jest.Mock).mockResolvedValue({ data: vm, error: null });

    const { result } = renderHook(() => useVM('vm-1'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.vm).toEqual(vm);
  });

  it('surfaces an error envelope instead of an empty VM', async () => {
    (getVM as jest.Mock).mockResolvedValue({ data: null, error: { code: 'HTTP_404', message: 'Not Found' } });

    const { result } = renderHook(() => useVM('vm-1'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.vm).toBeNull();
    expect(result.current.error).toEqual(new Error('Not Found'));
  });
});
