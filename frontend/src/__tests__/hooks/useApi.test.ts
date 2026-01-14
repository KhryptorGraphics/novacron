import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { mockApiResponse, mockApiError } from '@/src/__tests__/utils/test-utils';

// Mock fetch
global.fetch = jest.fn();
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock hook implementation
const useVMs = () => {
  const queryClient = new QueryClient();
  
  const fetchVMs = async () => {
    const response = await fetch('/api/vms');
    if (!response.ok) {
      throw new Error('Failed to fetch VMs');
    }
    return response.json();
  };

  return {
    data: null,
    isLoading: false,
    error: null,
    refetch: fetchVMs,
  };
};

const useCreateVM = () => {
  const createVM = async (vmData: any) => {
    const response = await fetch('/api/vms', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vmData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create VM');
    }
    
    return response.json();
  };

  return {
    mutate: createVM,
    isLoading: false,
    error: null,
  };
};

describe('API Hooks', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('useVMs', () => {
    it('fetches VMs successfully', async () => {
      const mockVMs = [
        { id: 'vm-1', name: 'Test VM 1', state: 'running' },
        { id: 'vm-2', name: 'Test VM 2', state: 'stopped' },
      ];

      mockFetch.mockResolvedValueOnce(mockApiResponse(mockVMs) as any);

      const { result } = renderHook(() => useVMs());
      
      await waitFor(() => {
        result.current.refetch();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/vms');
    });

    it('handles fetch error', async () => {
      mockFetch.mockResolvedValueOnce(mockApiError('Server error', 500) as any);

      const { result } = renderHook(() => useVMs());
      
      await expect(result.current.refetch()).rejects.toThrow('Failed to fetch VMs');
    });
  });

  describe('useCreateVM', () => {
    it('creates VM successfully', async () => {
      const mockVM = { id: 'vm-new', name: 'New VM', state: 'created' };
      const vmData = { name: 'New VM', cpu: 2, memory: 1024 };

      mockFetch.mockResolvedValueOnce(mockApiResponse(mockVM, 201) as any);

      const { result } = renderHook(() => useCreateVM());
      
      const response = await result.current.mutate(vmData);
      expect(response).toEqual(mockVM);
      
      expect(mockFetch).toHaveBeenCalledWith('/api/vms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vmData),
      });
    });

    it('handles create VM error', async () => {
      const vmData = { name: 'New VM', cpu: 2, memory: 1024 };

      mockFetch.mockResolvedValueOnce(mockApiError('Validation error', 400) as any);

      const { result } = renderHook(() => useCreateVM());
      
      await expect(result.current.mutate(vmData)).rejects.toThrow('Failed to create VM');
    });
  });
});
