// Authentication service for NovaCron frontend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

interface LoginRequest {
  email: string;
  password: string;
}

interface RegisterRequest {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
}

interface ForgotPasswordRequest {
  email: string;
}

interface AuthResponse {
  token: string;
  expiresAt: string;
  user: {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
    tenantId?: string;
    status: string;
  };
}

interface UserResponse {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  tenantId?: string;
  status: string;
}

class AuthService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  async login(credentials: LoginRequest): Promise<AuthResponse> {
    return this.request<AuthResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  async register(userData: RegisterRequest): Promise<UserResponse> {
    return this.request<UserResponse>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async forgotPassword(data: ForgotPasswordRequest): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async resetPassword(data: { token: string; password: string }): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get current user from token
  getCurrentUser(): UserResponse | null {
    const token = this.getToken();
    if (!token) return null;
    
    try {
      // In a real implementation, you would decode the JWT token
      // For now, we'll try to get user data from localStorage or return demo user
      if (typeof window !== 'undefined') {
        const storedUser = localStorage.getItem('authUser');
        if (storedUser) {
          return JSON.parse(storedUser);
        }
      }
      
      // Fallback demo user - in production this should decode the JWT
      return {
        id: "user-123",
        email: "user@example.com", 
        firstName: "Demo",
        lastName: "User",
        status: "active"
      };
    } catch (error) {
      console.error('Error getting current user:', error);
      return null;
    }
  }

  // Check if user is authenticated
  isAuthenticated() {
    const token = this.getToken();
    return !!token;
  }

  // Get auth token
  getToken() {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('authToken');
    }
    return null;
  }

  // Set auth token and user data
  setToken(token: string, user?: UserResponse) {
    if (typeof window !== 'undefined') {
      localStorage.setItem('authToken', token);
      if (user) {
        localStorage.setItem('authUser', JSON.stringify(user));
      }
    }
  }

  // Remove auth token and user data
  removeToken() {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('authToken');
      localStorage.removeItem('authUser');
    }
  }

  // Logout
  logout() {
    this.removeToken();
    // In a real implementation, you might want to call the logout API endpoint
  }
}

export const authService = new AuthService();
export default authService;