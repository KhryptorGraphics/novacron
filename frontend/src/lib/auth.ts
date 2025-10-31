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
    two_factor_enabled?: boolean;
  };
  requires_2fa?: boolean;
  temp_token?: string;
}

interface UserResponse {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  tenantId?: string;
  status: string;
  two_factor_enabled?: boolean;
}

interface TwoFactorSetupResponse {
  qr_code: string;
  secret: string;
  backup_codes: string[];
}

interface TwoFactorVerifyRequest {
  user_id: string;
  code: string;
  is_backup_code?: boolean;
  temp_token?: string;
}

interface TwoFactorVerifyResponse {
  valid: boolean;
  remaining_backup_codes?: number;
  error?: string;
  token?: string;
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

  // Decode JWT token (simple base64 decode - in production use a proper JWT library)
  private decodeJWT(token: string): any {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) {
        throw new Error('Invalid JWT format');
      }

      // Decode the payload (second part)
      const payload = parts[1];
      const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
      return JSON.parse(decoded);
    } catch (error) {
      console.error('Failed to decode JWT:', error);
      return null;
    }
  }

  // Get current user from token
  getCurrentUser(): UserResponse | null {
    const token = this.getToken();
    if (!token) return null;

    try {
      // Decode JWT token to get user information
      const payload = this.decodeJWT(token);
      if (!payload) {
        // Token is invalid, remove it
        this.removeToken();
        return null;
      }

      // Check if token is expired
      if (payload.exp && payload.exp * 1000 < Date.now()) {
        // Token expired, remove it
        this.removeToken();
        return null;
      }

      // Extract user information from JWT payload
      // The backend JWT should contain: sub (user ID), email, firstName, lastName, etc.
      const user: UserResponse = {
        id: payload.sub || payload.user_id || payload.id,
        email: payload.email || '',
        firstName: payload.firstName || payload.first_name || payload.given_name || '',
        lastName: payload.lastName || payload.last_name || payload.family_name || '',
        tenantId: payload.tenantId || payload.tenant_id,
        status: payload.status || 'active',
        two_factor_enabled: payload.two_factor_enabled || payload['2fa_enabled'] || false
      };

      // Cache user data in localStorage for quick access
      if (typeof window !== 'undefined') {
        localStorage.setItem('authUser', JSON.stringify(user));
      }

      return user;
    } catch (error) {
      console.error('Error getting current user:', error);
      this.removeToken();
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
      // Use consistent token key
      return localStorage.getItem('novacron_token');
    }
    return null;
  }

  // Set auth token and user data
  setToken(token: string, user?: UserResponse) {
    if (typeof window !== 'undefined') {
      // Use consistent token key
      localStorage.setItem('novacron_token', token);
      if (user) {
        localStorage.setItem('authUser', JSON.stringify(user));
      }
    }
  }

  // Remove auth token and user data
  removeToken() {
    if (typeof window !== 'undefined') {
      // Use consistent token key
      localStorage.removeItem('novacron_token');
      localStorage.removeItem('authUser');
      localStorage.removeItem('tempToken');
    }
  }

  // Logout
  logout() {
    this.removeToken();
    // In a real implementation, you might want to call the logout API endpoint
  }

  // 2FA Methods

  // Setup 2FA - generate QR code and secret
  async setup2FA(): Promise<TwoFactorSetupResponse> {
    const token = this.getToken();
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    return this.request<TwoFactorSetupResponse>('/api/auth/2fa/setup', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: user.id,
        account_name: user.email
      }),
    });
  }

  // Verify 2FA setup with code
  async verify2FASetup(data: TwoFactorVerifyRequest): Promise<TwoFactorVerifyResponse> {
    const token = this.getToken();
    const user = this.getCurrentUser();
    if (!user && !data.user_id) {
      throw new Error('User ID required for 2FA verification');
    }
    return this.request<TwoFactorVerifyResponse>('/api/auth/2fa/verify', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: data.user_id || user?.id,
        code: data.code,
        is_backup_code: data.is_backup_code || false
      }),
    });
  }

  // Verify 2FA during login
  async verify2FALogin(data: TwoFactorVerifyRequest): Promise<AuthResponse> {
    return this.request<AuthResponse>('/api/auth/2fa/verify-login', {
      method: 'POST',
      body: JSON.stringify({
        user_id: data.user_id,
        code: data.code,
        is_backup_code: data.is_backup_code || false,
        temp_token: data.temp_token
      }),
    });
  }

  // Enable 2FA after verification
  async enable2FA(code: string): Promise<{ success: boolean; message: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    const token = this.getToken();
    return this.request<{ success: boolean; message: string }>('/api/auth/2fa/enable', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: user.id,
        code: code
      }),
    });
  }

  // Disable 2FA
  async disable2FA(): Promise<{ success: boolean; message: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    const token = this.getToken();
    return this.request<{ success: boolean; message: string }>('/api/auth/2fa/disable', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: user.id
      }),
    });
  }

  // Get 2FA status
  async get2FAStatus(): Promise<{ enabled: boolean; backup_codes_remaining: number; setup?: boolean; setup_at?: string; last_used?: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    const token = this.getToken();
    const response = await this.request<{ enabled: boolean; setup?: boolean; setup_at?: string; last_used?: string; algorithm?: string; digits?: number; period?: number }>('/api/auth/2fa/status?user_id=' + user.id, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    // Transform backend response to match frontend expectations
    return {
      enabled: response.enabled,
      backup_codes_remaining: 0, // Backend doesn't return this in status, need separate call
      setup: response.setup,
      setup_at: response.setup_at,
      last_used: response.last_used
    };
  }

  // Generate new backup codes
  async generateBackupCodes(): Promise<{ backup_codes: string[] }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    const token = this.getToken();
    return this.request<{ backup_codes: string[] }>('/api/auth/2fa/backup-codes', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: user.id
      }),
    });
  }

  // Store temporary token for 2FA flow
  setTempToken(tempToken: string) {
    if (typeof window !== 'undefined') {
      localStorage.setItem('tempToken', tempToken);
    }
  }

  // Get temporary token for 2FA flow
  getTempToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('tempToken');
    }
    return null;
  }

  // Remove temporary token
  removeTempToken() {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('tempToken');
    }
  }
}

export const authService = new AuthService();
export default authService;