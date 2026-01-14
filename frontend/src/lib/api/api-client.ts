/**
 * Enhanced API Client with error handling, retries, and loading states
 * Provides a consistent interface for all API calls
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

export interface ApiError {
  message: string;
  status: number;
  code?: string;
  details?: any;
}

export interface ApiResponse<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Get auth token from localStorage
   */
  private getAuthToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('novacron_token');
    }
    return null;
  }

  /**
   * Build headers with auth token if available
   */
  private buildHeaders(customHeaders?: HeadersInit): HeadersInit {
    const headers = { ...this.defaultHeaders, ...customHeaders };
    const token = this.getAuthToken();
    
    if (token) {
      return {
        ...headers,
        'Authorization': `Bearer ${token}`,
      };
    }
    
    return headers;
  }

  /**
   * Handle API errors consistently
   */
  private handleError(error: any, endpoint: string): ApiError {
    console.error(`API Error [${endpoint}]:`, error);

    if (error.response) {
      // Server responded with error status
      return {
        message: error.response.data?.message || error.response.statusText || 'Server error',
        status: error.response.status,
        code: error.response.data?.code,
        details: error.response.data,
      };
    } else if (error.request) {
      // Request made but no response
      return {
        message: 'No response from server. Please check your connection.',
        status: 0,
        code: 'NETWORK_ERROR',
      };
    } else {
      // Error setting up request
      return {
        message: error.message || 'An unexpected error occurred',
        status: 0,
        code: 'CLIENT_ERROR',
      };
    }
  }

  /**
   * Make HTTP request with error handling and retries
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    retries: number = 0
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = this.buildHeaders(options.headers);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      // Handle 401 Unauthorized - token expired
      if (response.status === 401) {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('novacron_token');
          localStorage.removeItem('authUser');
          window.location.href = '/auth/login';
        }
        throw new Error('Authentication required');
      }

      // Handle other error statuses
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw {
          response: {
            status: response.status,
            statusText: response.statusText,
            data: errorData,
          },
        };
      }

      // Parse successful response
      const data = await response.json();
      return data;
    } catch (error) {
      // Retry logic for network errors
      if (retries > 0 && this.shouldRetry(error)) {
        await this.delay(1000 * (3 - retries)); // Exponential backoff
        return this.request<T>(endpoint, options, retries - 1);
      }

      throw this.handleError(error, endpoint);
    }
  }

  /**
   * Determine if request should be retried
   */
  private shouldRetry(error: any): boolean {
    // Retry on network errors or 5xx server errors
    return (
      !error.response ||
      (error.response.status >= 500 && error.response.status < 600)
    );
  }

  /**
   * Delay helper for retries
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'GET' }, 2);
  }

  /**
   * POST request
   */
  async post<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        ...options,
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      },
      1
    );
  }

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        ...options,
        method: 'PUT',
        body: data ? JSON.stringify(data) : undefined,
      },
      1
    );
  }

  /**
   * DELETE request
   */
  async delete<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' }, 1);
  }

  /**
   * PATCH request
   */
  async patch<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        ...options,
        method: 'PATCH',
        body: data ? JSON.stringify(data) : undefined,
      },
      1
    );
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;

