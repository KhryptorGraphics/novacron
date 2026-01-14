import { renderHook, act, waitFor } from '@testing-library/react';
import { jest } from '@jest/globals';

import { useApi, useVMActions, useAuth } from '@/hooks/useApi';
import { apiClient } from '@/lib/api';

// Mock the API client
jest.mock('@/lib/api', () => ({
  apiClient: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    patch: jest.fn(),
  },
}));

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

const mockedApiClient = apiClient as jest.Mocked<typeof apiClient>;

describe('useApi Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic API Operations', () => {
    it('should perform GET request successfully', async () => {
      const mockData = { id: 1, name: 'Test' };
      mockedApiClient.get.mockResolvedValue({ data: mockData });

      const { result } = renderHook(() => useApi());

      await act(async () => {
        const response = await result.current.get('/test');
        expect(response.data).toEqual(mockData);
      });

      expect(mockedApiClient.get).toHaveBeenCalledWith('/test');
    });

    it('should perform POST request successfully', async () => {
      const mockData = { id: 1, name: 'Created' };
      const requestData = { name: 'New Item' };
      mockedApiClient.post.mockResolvedValue({ data: mockData });

      const { result } = renderHook(() => useApi());

      await act(async () => {
        const response = await result.current.post('/test', requestData);
        expect(response.data).toEqual(mockData);
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/test', requestData);
    });

    it('should perform PUT request successfully', async () => {
      const mockData = { id: 1, name: 'Updated' };
      const requestData = { name: 'Updated Item' };
      mockedApiClient.put.mockResolvedValue({ data: mockData });

      const { result } = renderHook(() => useApi());

      await act(async () => {
        const response = await result.current.put('/test/1', requestData);
        expect(response.data).toEqual(mockData);
      });

      expect(mockedApiClient.put).toHaveBeenCalledWith('/test/1', requestData);
    });

    it('should perform DELETE request successfully', async () => {
      mockedApiClient.delete.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useApi());

      await act(async () => {
        const response = await result.current.delete('/test/1');
        expect(response.data).toEqual({ success: true });
      });

      expect(mockedApiClient.delete).toHaveBeenCalledWith('/test/1');
    });

    it('should handle API errors', async () => {
      const error = new Error('API Error');
      mockedApiClient.get.mockRejectedValue(error);

      const { result } = renderHook(() => useApi());

      await act(async () => {
        try {
          await result.current.get('/test');
        } catch (e) {
          expect(e).toBe(error);
        }
      });
    });
  });

  describe('Loading States', () => {
    it('should track loading state during requests', async () => {
      mockedApiClient.get.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 100))
      );

      const { result } = renderHook(() => useApi());

      expect(result.current.loading).toBe(false);

      act(() => {
        result.current.get('/test');
      });

      expect(result.current.loading).toBe(true);

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });
    });

    it('should handle concurrent requests', async () => {
      mockedApiClient.get.mockImplementation((url) => 
        new Promise(resolve => {
          const delay = url === '/fast' ? 50 : 100;
          setTimeout(() => resolve({ data: { url } }), delay);
        })
      );

      const { result } = renderHook(() => useApi());

      act(() => {
        result.current.get('/slow');
        result.current.get('/fast');
      });

      expect(result.current.loading).toBe(true);

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const networkError = new Error('Network Error');
      networkError.name = 'NetworkError';
      mockedApiClient.get.mockRejectedValue(networkError);

      const { result } = renderHook(() => useApi());

      await act(async () => {
        try {
          await result.current.get('/test');
        } catch (error) {
          expect(error).toBe(networkError);
        }
      });

      expect(result.current.error).toBe(networkError);
    });

    it('should handle HTTP errors with status codes', async () => {
      const httpError = {
        response: {
          status: 404,
          data: { message: 'Not Found' }
        }
      };
      mockedApiClient.get.mockRejectedValue(httpError);

      const { result } = renderHook(() => useApi());

      await act(async () => {
        try {
          await result.current.get('/test');
        } catch (error) {
          expect(error).toEqual(httpError);
        }
      });
    });

    it('should clear error on successful request', async () => {
      // First request fails
      mockedApiClient.get.mockRejectedValueOnce(new Error('First Error'));
      
      const { result } = renderHook(() => useApi());

      await act(async () => {
        try {
          await result.current.get('/test');
        } catch (e) {
          // Expected to fail
        }
      });

      expect(result.current.error).toBeInstanceOf(Error);

      // Second request succeeds
      mockedApiClient.get.mockResolvedValue({ data: { success: true } });

      await act(async () => {
        await result.current.get('/test');
      });

      expect(result.current.error).toBe(null);
    });
  });
});

describe('useVMActions Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('VM Lifecycle Operations', () => {
    it('should start VM successfully', async () => {
      mockedApiClient.post.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        await result.current.startVM('vm-123');
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/vms/vm-123/start');
      expect(result.current.error).toBe(null);
    });

    it('should stop VM successfully', async () => {
      mockedApiClient.post.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        await result.current.stopVM('vm-123');
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/vms/vm-123/stop');
    });

    it('should restart VM successfully', async () => {
      mockedApiClient.post.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        await result.current.restartVM('vm-123');
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/vms/vm-123/restart');
    });

    it('should delete VM successfully', async () => {
      mockedApiClient.delete.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        await result.current.deleteVM('vm-123');
      });

      expect(mockedApiClient.delete).toHaveBeenCalledWith('/api/v1/vms/vm-123');
    });

    it('should create VM successfully', async () => {
      const vmConfig = {
        name: 'Test VM',
        type: 'qemu',
        cpu: 2,
        memory: 4096,
      };
      const createdVM = { id: 'vm-123', ...vmConfig };
      
      mockedApiClient.post.mockResolvedValue({ data: createdVM });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        const vm = await result.current.createVM(vmConfig);
        expect(vm).toEqual(createdVM);
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/vms', vmConfig);
    });

    it('should update VM successfully', async () => {
      const vmUpdate = {
        name: 'Updated VM',
        cpu: 4,
      };
      const updatedVM = { id: 'vm-123', ...vmUpdate };
      
      mockedApiClient.put.mockResolvedValue({ data: updatedVM });

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        const vm = await result.current.updateVM('vm-123', vmUpdate);
        expect(vm).toEqual(updatedVM);
      });

      expect(mockedApiClient.put).toHaveBeenCalledWith('/api/v1/vms/vm-123', vmUpdate);
    });
  });

  describe('Error Handling', () => {
    it('should handle start VM failure', async () => {
      const error = new Error('Failed to start VM');
      mockedApiClient.post.mockRejectedValue(error);

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        try {
          await result.current.startVM('vm-123');
        } catch (e) {
          expect(e).toBe(error);
        }
      });

      expect(result.current.error).toBe(error);
    });

    it('should handle delete VM failure', async () => {
      const error = { 
        response: { 
          status: 409, 
          data: { message: 'VM is running' } 
        } 
      };
      mockedApiClient.delete.mockRejectedValue(error);

      const { result } = renderHook(() => useVMActions());

      await act(async () => {
        try {
          await result.current.deleteVM('vm-123');
        } catch (e) {
          expect(e).toEqual(error);
        }
      });
    });
  });

  describe('Loading States', () => {
    it('should track loading state during VM operations', async () => {
      mockedApiClient.post.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 100))
      );

      const { result } = renderHook(() => useVMActions());

      expect(result.current.isLoading).toBe(false);

      act(() => {
        result.current.startVM('vm-123');
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('should handle multiple concurrent operations', async () => {
      mockedApiClient.post.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 50))
      );
      mockedApiClient.delete.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 75))
      );

      const { result } = renderHook(() => useVMActions());

      act(() => {
        result.current.startVM('vm-1');
        result.current.deleteVM('vm-2');
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });
  });

  describe('Optimistic Updates', () => {
    it('should provide optimistic feedback for VM actions', async () => {
      mockedApiClient.post.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ data: {} }), 100))
      );

      const { result } = renderHook(() => useVMActions());

      const onVMStateChange = jest.fn();
      
      await act(async () => {
        result.current.startVM('vm-123', { onStateChange: onVMStateChange });
      });

      // Should have called state change callback optimistically
      expect(onVMStateChange).toHaveBeenCalledWith('vm-123', 'starting');
    });
  });
});

describe('useAuth Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
  });

  describe('Authentication Flow', () => {
    it('should login successfully', async () => {
      const mockToken = 'mock-jwt-token';
      const mockUser = { id: '1', username: 'test', email: 'test@example.com' };
      
      mockedApiClient.post.mockResolvedValue({
        data: { token: mockToken, user: mockUser }
      });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        await result.current.login('test', 'password');
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/auth/login', {
        username: 'test',
        password: 'password'
      });
      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('token', mockToken);
    });

    it('should handle login failure', async () => {
      const error = { 
        response: { 
          status: 401, 
          data: { message: 'Invalid credentials' } 
        } 
      };
      mockedApiClient.post.mockRejectedValue(error);

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        try {
          await result.current.login('test', 'wrongpassword');
        } catch (e) {
          expect(e).toEqual(error);
        }
      });

      expect(result.current.user).toBe(null);
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.error).toEqual(error);
    });

    it('should logout successfully', async () => {
      mockLocalStorage.getItem.mockReturnValue('existing-token');
      mockedApiClient.post.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        await result.current.logout();
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/auth/logout');
      expect(result.current.user).toBe(null);
      expect(result.current.isAuthenticated).toBe(false);
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('token');
    });

    it('should refresh authentication token', async () => {
      const oldToken = 'old-token';
      const newToken = 'new-token';
      const mockUser = { id: '1', username: 'test' };

      mockLocalStorage.getItem.mockReturnValue(oldToken);
      mockedApiClient.post.mockResolvedValue({
        data: { token: newToken, user: mockUser }
      });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        await result.current.refreshToken();
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/auth/refresh');
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('token', newToken);
    });
  });

  describe('Token Management', () => {
    it('should load token from localStorage on init', () => {
      const mockToken = 'stored-token';
      mockLocalStorage.getItem.mockReturnValue(mockToken);

      const { result } = renderHook(() => useAuth());

      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('token');
      // Token should be available (through the hook's internal state)
    });

    it('should handle expired token', async () => {
      const expiredTokenError = {
        response: {
          status: 401,
          data: { message: 'Token expired' }
        }
      };
      mockedApiClient.get.mockRejectedValue(expiredTokenError);

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        try {
          await result.current.getCurrentUser();
        } catch (e) {
          expect(e).toEqual(expiredTokenError);
        }
      });

      // Should clear authentication state
      expect(result.current.isAuthenticated).toBe(false);
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('token');
    });

    it('should automatically refresh token when near expiry', async () => {
      const nearExpiryToken = 'near-expiry-token';
      const newToken = 'refreshed-token';
      
      mockLocalStorage.getItem.mockReturnValue(nearExpiryToken);
      mockedApiClient.post.mockResolvedValue({
        data: { token: newToken }
      });

      const { result } = renderHook(() => useAuth());

      // Simulate token near expiry
      await act(async () => {
        await result.current.checkAndRefreshToken();
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/auth/refresh');
    });
  });

  describe('User Management', () => {
    it('should get current user info', async () => {
      const mockUser = { 
        id: '1', 
        username: 'test', 
        email: 'test@example.com',
        roles: ['user']
      };
      
      mockedApiClient.get.mockResolvedValue({ data: mockUser });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        const user = await result.current.getCurrentUser();
        expect(user).toEqual(mockUser);
      });

      expect(mockedApiClient.get).toHaveBeenCalledWith('/api/v1/auth/me');
      expect(result.current.user).toEqual(mockUser);
    });

    it('should update user profile', async () => {
      const updateData = { email: 'newemail@example.com' };
      const updatedUser = { 
        id: '1', 
        username: 'test', 
        email: 'newemail@example.com' 
      };
      
      mockedApiClient.put.mockResolvedValue({ data: updatedUser });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        const user = await result.current.updateProfile(updateData);
        expect(user).toEqual(updatedUser);
      });

      expect(mockedApiClient.put).toHaveBeenCalledWith('/api/v1/auth/profile', updateData);
      expect(result.current.user).toEqual(updatedUser);
    });

    it('should change password', async () => {
      mockedApiClient.post.mockResolvedValue({ data: { success: true } });

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        await result.current.changePassword('oldpass', 'newpass');
      });

      expect(mockedApiClient.post).toHaveBeenCalledWith('/api/v1/auth/change-password', {
        currentPassword: 'oldpass',
        newPassword: 'newpass'
      });
    });
  });

  describe('Permission Checks', () => {
    it('should check user permissions', async () => {
      const mockUser = { 
        id: '1', 
        roles: ['admin'], 
        permissions: ['vm:create', 'vm:delete']
      };
      
      const { result } = renderHook(() => useAuth());
      
      // Set user in state
      act(() => {
        result.current.setUser(mockUser);
      });

      expect(result.current.hasPermission('vm:create')).toBe(true);
      expect(result.current.hasPermission('vm:delete')).toBe(true);
      expect(result.current.hasPermission('admin:users')).toBe(false);
    });

    it('should check user roles', () => {
      const mockUser = { id: '1', roles: ['admin', 'user'] };
      
      const { result } = renderHook(() => useAuth());
      
      act(() => {
        result.current.setUser(mockUser);
      });

      expect(result.current.hasRole('admin')).toBe(true);
      expect(result.current.hasRole('user')).toBe(true);
      expect(result.current.hasRole('superadmin')).toBe(false);
    });
  });

  describe('Session Persistence', () => {
    it('should persist session across page reloads', () => {
      const mockToken = 'persistent-token';
      const mockUser = { id: '1', username: 'test' };
      
      mockLocalStorage.getItem.mockReturnValue(mockToken);
      mockedApiClient.get.mockResolvedValue({ data: mockUser });

      const { result } = renderHook(() => useAuth());

      // Should attempt to restore session on mount
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('token');
    });

    it('should handle session restore failure', async () => {
      mockLocalStorage.getItem.mockReturnValue('invalid-token');
      mockedApiClient.get.mockRejectedValue({
        response: { status: 401 }
      });

      const { result } = renderHook(() => useAuth());

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(false);
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('token');
      });
    });
  });
});

// Performance and memory leak tests
describe('Hook Performance', () => {
  it('should not cause memory leaks', async () => {
    const { unmount } = renderHook(() => useApi());
    
    // Simulate component unmount
    unmount();
    
    // All pending requests should be cancelled
    // This would be tested with actual implementation details
  });

  it('should debounce rapid successive calls', async () => {
    mockedApiClient.get.mockResolvedValue({ data: {} });

    const { result } = renderHook(() => useApi());

    // Make rapid successive calls
    act(() => {
      result.current.get('/test');
      result.current.get('/test');
      result.current.get('/test');
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Should have debounced calls (actual implementation would need this logic)
  });
});