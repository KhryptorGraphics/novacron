/**
 * Users API service
 */

import { apiClient } from './api-client';

export interface User {
  id: string;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  status: 'active' | 'inactive' | 'suspended';
  role: string;
  tenant_id?: string;
  two_factor_enabled?: boolean;
  created_at?: string;
  updated_at?: string;
  last_login?: string;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
  role?: string;
  tenant_id?: string;
}

export interface UpdateUserRequest {
  username?: string;
  email?: string;
  first_name?: string;
  last_name?: string;
  role?: string;
  status?: 'active' | 'inactive' | 'suspended';
}

export interface UsersListResponse {
  users: User[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export const usersApi = {
  /**
   * Get all users with pagination
   */
  async getUsers(params?: {
    page?: number;
    limit?: number;
    search?: string;
    status?: string;
    role?: string;
  }): Promise<UsersListResponse> {
    const queryParams = new URLSearchParams();
    
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    if (params?.search) queryParams.append('search', params.search);
    if (params?.status) queryParams.append('status', params.status);
    if (params?.role) queryParams.append('role', params.role);

    const endpoint = `/api/users${queryParams.toString() ? `?${queryParams}` : ''}`;
    return apiClient.get<UsersListResponse>(endpoint);
  },

  /**
   * Get a single user by ID
   */
  async getUser(userId: string): Promise<User> {
    return apiClient.get<User>(`/api/users/${userId}`);
  },

  /**
   * Create a new user
   */
  async createUser(data: CreateUserRequest): Promise<User> {
    return apiClient.post<User>('/api/users', data);
  },

  /**
   * Update a user
   */
  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    return apiClient.put<User>(`/api/users/${userId}`, data);
  },

  /**
   * Delete a user
   */
  async deleteUser(userId: string): Promise<void> {
    return apiClient.delete<void>(`/api/users/${userId}`);
  },

  /**
   * Suspend a user
   */
  async suspendUser(userId: string): Promise<User> {
    return apiClient.post<User>(`/api/users/${userId}/suspend`, {});
  },

  /**
   * Activate a user
   */
  async activateUser(userId: string): Promise<User> {
    return apiClient.post<User>(`/api/users/${userId}/activate`, {});
  },

  /**
   * Reset user password
   */
  async resetPassword(userId: string): Promise<{ message: string }> {
    return apiClient.post<{ message: string }>(`/api/users/${userId}/reset-password`, {});
  },

  /**
   * Enable 2FA for user
   */
  async enable2FA(userId: string): Promise<User> {
    return apiClient.post<User>(`/api/users/${userId}/2fa/enable`, {});
  },

  /**
   * Disable 2FA for user
   */
  async disable2FA(userId: string): Promise<User> {
    return apiClient.post<User>(`/api/users/${userId}/2fa/disable`, {});
  },
};

export default usersApi;

