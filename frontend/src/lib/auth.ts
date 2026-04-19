import { buildApiUrl } from '@/lib/api/origin';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
}

interface ForgotPasswordRequest {
  email: string;
}

export interface UserResponse {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  tenantId?: string;
  status: string;
  two_factor_enabled?: boolean;
  role?: string;
  roles?: string[];
}

export interface ClusterSummaryResponse {
  id: string;
  name: string;
  tier: string;
  performanceScore: number;
  interconnectLatencyMs: number;
  interconnectBandwidthMbps: number;
  currentNodeCount: number;
  maxSupportedNodeCount: number;
  growthState: string;
  federationState: string;
  degraded: boolean;
  lastEvaluatedAt: string;
  edgeLatencyMs?: number;
  edgeBandwidthMbps?: number;
}

export interface AdmissionResponse {
  admitted: boolean;
  state?: string;
  clusterId?: string;
  role?: string;
  source?: string;
  admittedAt?: string;
  tenantId?: string;
  selected?: boolean;
  cluster?: ClusterSummaryResponse;
}

export interface SessionResponse {
  id: string;
  expiresAt: string;
  createdAt: string;
  lastAccessedAt: string;
  selectedClusterId?: string;
}

export interface AuthResponse {
  token: string;
  refreshToken?: string;
  expiresAt: string;
  user: UserResponse;
  admission: AdmissionResponse;
  memberships: AdmissionResponse[];
  selectedCluster?: ClusterSummaryResponse;
  session: SessionResponse;
  requires_2fa?: boolean;
  temp_token?: string;
}

export interface CurrentUserResponse {
  user: UserResponse;
  admission: AdmissionResponse;
  memberships: AdmissionResponse[];
  selectedCluster?: ClusterSummaryResponse;
  session: SessionResponse;
}

interface OAuthAuthorizationUrlResponse {
  provider: string;
  authorizationUrl: string;
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

const ACCESS_TOKEN_KEY = 'novacron_token';
const REFRESH_TOKEN_KEY = 'novacron_refresh_token';
const AUTH_USER_KEY = 'authUser';
const AUTH_MEMBERSHIPS_KEY = 'authMemberships';
const SELECTED_CLUSTER_KEY = 'selectedCluster';
const AUTH_SESSION_KEY = 'authSession';
const TEMP_TOKEN_KEY = 'tempToken';

class AuthService {
  private normalizeRoles(
    primaryRole?: unknown,
    roleList?: unknown,
  ): { role?: string; roles?: string[] } {
    const normalizedRoles = Array.isArray(roleList)
      ? roleList.filter((role): role is string => typeof role === 'string' && role.length > 0)
      : [];
    const normalizedPrimaryRole = typeof primaryRole === 'string' && primaryRole.length > 0
      ? primaryRole
      : normalizedRoles[0];

    if (normalizedPrimaryRole && !normalizedRoles.includes(normalizedPrimaryRole)) {
      normalizedRoles.unshift(normalizedPrimaryRole);
    }

    return {
      ...(normalizedPrimaryRole ? { role: normalizedPrimaryRole } : {}),
      ...(normalizedRoles.length > 0 ? { roles: normalizedRoles } : {}),
    };
  }

  private buildUserFromClaims(payload: any): UserResponse {
    const { role, roles } = this.normalizeRoles(payload.role, payload.roles);

    return {
      id: payload.sub || payload.user_id || payload.id,
      email: payload.email || payload.metadata?.email || '',
      firstName: payload.firstName || payload.first_name || payload.given_name || payload.metadata?.first_name || '',
      lastName: payload.lastName || payload.last_name || payload.family_name || payload.metadata?.last_name || '',
      status: payload.status || payload.metadata?.status || 'active',
      two_factor_enabled: payload.two_factor_enabled || payload['2fa_enabled'] || false,
      ...((payload.tenantId || payload.tenant_id) ? { tenantId: payload.tenantId || payload.tenant_id } : {}),
      ...(role ? { role } : {}),
      ...(roles ? { roles } : {}),
    };
  }

  private decodeJWT(token: string): any {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) {
        throw new Error('Invalid JWT format');
      }
      const payload = parts[1];
      const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
      return JSON.parse(decoded);
    } catch (error) {
      console.error('Failed to decode JWT:', error);
      return null;
    }
  }

  private getStoredJson<T>(key: string): T | null {
    if (typeof window === 'undefined') {
      return null;
    }

    const value = window.localStorage.getItem(key);
    if (!value) {
      return null;
    }

    try {
      return JSON.parse(value) as T;
    } catch (error) {
      console.error(`Failed to parse stored auth value for ${key}:`, error);
      window.localStorage.removeItem(key);
      return null;
    }
  }

  private setStoredJson(key: string, value: unknown) {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(key, JSON.stringify(value));
  }

  private withAuthorization(headers?: HeadersInit): Record<string, string> {
    const resolved: Record<string, string> = {};
    const mergeHeader = (key: string, value: string) => {
      const existingKey = Object.keys(resolved).find(
        (candidate) => candidate.toLowerCase() === key.toLowerCase(),
      );
      resolved[existingKey ?? key] = value;
    };

    if (headers instanceof Headers) {
      headers.forEach((value, key) => mergeHeader(key, value));
    } else if (Array.isArray(headers)) {
      headers.forEach(([key, value]) => mergeHeader(key, value));
    } else if (headers) {
      Object.entries(headers).forEach(([key, value]) => {
        if (typeof value !== 'undefined') {
          mergeHeader(key, String(value));
        }
      });
    }

    if (!Object.keys(resolved).some((key) => key.toLowerCase() === 'content-type')) {
      resolved['Content-Type'] = 'application/json';
    }

    const token = this.getToken();
    if (token && !Object.keys(resolved).some((key) => key.toLowerCase() === 'authorization')) {
      resolved.Authorization = `Bearer ${token}`;
    }

    return resolved;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = buildApiUrl(endpoint);

    const config: RequestInit = {
      credentials: 'include',
      ...options,
      headers: this.withAuthorization(options.headers),
    };

    const response = await fetch(url, config);
    if (!response.ok) {
      let detail = `HTTP error! status: ${response.status}`;
      try {
        const body = await response.json();
        if (body?.error) {
          detail = body.error;
        }
      } catch {
        const body = await response.text().catch(() => '');
        if (body) {
          detail = body;
        }
      }
      throw new Error(detail);
    }

    return response.json() as Promise<T>;
  }

  private persistUser(user: UserResponse, token?: string) {
    const tokenClaims = token ? this.decodeJWT(token) : null;
    const claimsUser = tokenClaims ? this.buildUserFromClaims(tokenClaims) : null;
    const normalizedRoles = this.normalizeRoles(
      user.role ?? claimsUser?.role,
      user.roles ?? claimsUser?.roles,
    );

    this.setStoredJson(AUTH_USER_KEY, {
      ...claimsUser,
      ...user,
      ...(normalizedRoles.role ? { role: normalizedRoles.role } : {}),
      ...(normalizedRoles.roles ? { roles: normalizedRoles.roles } : {}),
    });
  }

  private persistSessionState(payload: {
    token?: string;
    refreshToken?: string;
    user?: UserResponse;
    memberships?: AdmissionResponse[];
    selectedCluster?: ClusterSummaryResponse;
    session?: SessionResponse;
  }) {
    if (typeof window === 'undefined') {
      return;
    }

    if (payload.token) {
      window.localStorage.setItem(ACCESS_TOKEN_KEY, payload.token);
    }
    if (payload.refreshToken) {
      window.localStorage.setItem(REFRESH_TOKEN_KEY, payload.refreshToken);
    }
    if (payload.user) {
      this.persistUser(payload.user, payload.token);
    }
    if (payload.memberships) {
      this.setStoredJson(AUTH_MEMBERSHIPS_KEY, payload.memberships);
    }
    if (payload.selectedCluster) {
      this.setStoredJson(SELECTED_CLUSTER_KEY, payload.selectedCluster);
    } else if (payload.memberships && payload.memberships.length === 0) {
      window.localStorage.removeItem(SELECTED_CLUSTER_KEY);
    }
    if (payload.session) {
      this.setStoredJson(AUTH_SESSION_KEY, payload.session);
    }
  }

  setSession(response: AuthResponse | CurrentUserResponse, tokenOverride?: string, refreshTokenOverride?: string) {
    if ('token' in response) {
      const resolvedRefreshToken = refreshTokenOverride || response.refreshToken;
      this.persistSessionState({
        token: tokenOverride || response.token,
        ...(resolvedRefreshToken ? { refreshToken: resolvedRefreshToken } : {}),
        user: response.user,
        memberships: response.memberships,
        ...(response.selectedCluster ? { selectedCluster: response.selectedCluster } : {}),
        session: response.session,
      });
      return;
    }

    const currentToken = tokenOverride || this.getToken();
    const currentRefreshToken = refreshTokenOverride || this.getRefreshToken();
    this.persistSessionState({
      ...(currentToken ? { token: currentToken } : {}),
      ...(currentRefreshToken ? { refreshToken: currentRefreshToken } : {}),
      user: response.user,
      memberships: response.memberships,
      ...(response.selectedCluster ? { selectedCluster: response.selectedCluster } : {}),
      session: response.session,
    });
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

  async getGitHubAuthorizationUrl(redirectTo: string = '/dashboard'): Promise<OAuthAuthorizationUrlResponse> {
    return this.request<OAuthAuthorizationUrlResponse>(
      `/api/auth/oauth/github/url?redirect_to=${encodeURIComponent(redirectTo)}`,
    );
  }

  async getCurrentSession(): Promise<CurrentUserResponse> {
    return this.request<CurrentUserResponse>('/api/auth/me', {
      method: 'GET',
    });
  }

  async refresh(): Promise<AuthResponse> {
    const refreshToken = this.getRefreshToken();
    return this.request<AuthResponse>('/api/auth/refresh', {
      method: 'POST',
      body: JSON.stringify(refreshToken ? { refreshToken } : {}),
    });
  }

  async logout(): Promise<void> {
    try {
      await this.request('/api/auth/logout', {
        method: 'POST',
        body: JSON.stringify({}),
      });
    } catch (error) {
      console.warn('Runtime logout failed, clearing local auth state:', error);
    } finally {
      this.removeToken();
    }
  }

  async listSessions(): Promise<SessionResponse[]> {
    return this.request<SessionResponse[]>('/api/auth/sessions', { method: 'GET' });
  }

  async listAdmissions(): Promise<AdmissionResponse[]> {
    return this.request<AdmissionResponse[]>('/api/cluster/admissions', { method: 'GET' });
  }

  async selectCluster(clusterId: string): Promise<CurrentUserResponse> {
    return this.request<CurrentUserResponse>('/api/cluster/admissions/select', {
      method: 'POST',
      body: JSON.stringify({ clusterId }),
    });
  }

  async submitEdgeMetrics(clusterId: string, latencyMs: number, bandwidthMbps: number): Promise<AdmissionResponse[]> {
    return this.request<AdmissionResponse[]>('/api/cluster/edge-metrics', {
      method: 'POST',
      body: JSON.stringify({ clusterId, latencyMs, bandwidthMbps }),
    });
  }

  async restoreSession(): Promise<CurrentUserResponse | null> {
    try {
      const current = await this.getCurrentSession();
      this.setSession(current);
      return current;
    } catch (error) {
      const token = this.getToken();
      const refreshToken = this.getRefreshToken();
      if (!token && !refreshToken) {
        this.removeToken();
        return null;
      }

      try {
        const refreshed = await this.refresh();
        this.setSession(refreshed);
        const current = await this.getCurrentSession();
        this.setSession(current, refreshed.token, refreshed.refreshToken);
        return current;
      } catch (refreshError) {
        console.warn('Failed to restore auth session:', refreshError || error);
        this.removeToken();
        return null;
      }
    }
  }

  async checkEmailAvailability(email: string): Promise<{ available: boolean }> {
    return this.request<{ available: boolean }>(`/api/auth/check-email?email=${encodeURIComponent(email)}`);
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

  getCurrentUser(): UserResponse | null {
    const token = this.getToken();
    if (!token) {
      return this.getStoredJson<UserResponse>(AUTH_USER_KEY);
    }

    const payload = this.decodeJWT(token);
    if (!payload) {
      this.removeToken();
      return null;
    }
    if (payload.exp && payload.exp * 1000 < Date.now()) {
      this.removeToken();
      return null;
    }

    const tokenUser = this.buildUserFromClaims(payload);
    const cachedUser = this.getStoredJson<UserResponse>(AUTH_USER_KEY);
    const user: UserResponse = {
      ...cachedUser,
      ...tokenUser,
      email: tokenUser.email || cachedUser?.email || '',
      firstName: tokenUser.firstName || cachedUser?.firstName || '',
      lastName: tokenUser.lastName || cachedUser?.lastName || '',
      status: tokenUser.status || cachedUser?.status || 'active',
      ...((tokenUser.tenantId || cachedUser?.tenantId) ? { tenantId: tokenUser.tenantId || cachedUser?.tenantId } : {}),
      ...((tokenUser.role || cachedUser?.role) ? { role: tokenUser.role || cachedUser?.role } : {}),
      ...((tokenUser.roles || cachedUser?.roles) ? { roles: tokenUser.roles || cachedUser?.roles } : {}),
    };

    this.persistUser(user, token);
    return user;
  }

  getMemberships(): AdmissionResponse[] {
    return this.getStoredJson<AdmissionResponse[]>(AUTH_MEMBERSHIPS_KEY) || [];
  }

  getSelectedCluster(): ClusterSummaryResponse | null {
    return this.getStoredJson<ClusterSummaryResponse>(SELECTED_CLUSTER_KEY);
  }

  getSession(): SessionResponse | null {
    return this.getStoredJson<SessionResponse>(AUTH_SESSION_KEY);
  }

  getActiveMemberships(): AdmissionResponse[] {
    return this.getMemberships().filter((membership) => membership.admitted || membership.state === 'active');
  }

  isAuthenticated() {
    return !!this.getCurrentUser();
  }

  getToken() {
    if (typeof window !== 'undefined') {
      return window.localStorage.getItem(ACCESS_TOKEN_KEY);
    }
    return null;
  }

  getRefreshToken() {
    if (typeof window !== 'undefined') {
      return window.localStorage.getItem(REFRESH_TOKEN_KEY);
    }
    return null;
  }

  setToken(token: string, user?: UserResponse) {
    const selectedCluster = this.getSelectedCluster();
    const session = this.getSession();
    this.persistSessionState({
      token,
      ...(user ? { user } : {}),
      memberships: this.getMemberships(),
      ...(selectedCluster ? { selectedCluster } : {}),
      ...(session ? { session } : {}),
    });
  }

  removeToken() {
    if (typeof window === 'undefined') {
      return;
    }
    for (const key of [
      ACCESS_TOKEN_KEY,
      REFRESH_TOKEN_KEY,
      AUTH_USER_KEY,
      AUTH_MEMBERSHIPS_KEY,
      SELECTED_CLUSTER_KEY,
      AUTH_SESSION_KEY,
      TEMP_TOKEN_KEY,
    ]) {
      window.localStorage.removeItem(key);
    }
  }

  resolvePostLoginPath(
    memberships: AdmissionResponse[] = this.getMemberships(),
    selectedCluster: ClusterSummaryResponse | null | undefined = this.getSelectedCluster(),
    requestedPath?: string,
  ): string {
    const activeMemberships = memberships.filter((membership) => membership.admitted || membership.state === 'active');
    if (activeMemberships.length === 0) {
      return '/cluster/access';
    }

    const selectedClusterId = selectedCluster?.id || activeMemberships.find((membership) => membership.selected)?.clusterId;
    if (activeMemberships.length > 1 && !selectedClusterId) {
      return '/clusters/select';
    }

    if (requestedPath && requestedPath !== '/dashboard') {
      return requestedPath;
    }

    return '/dashboard';
  }

  shouldSelectCluster(): boolean {
    return this.getActiveMemberships().length > 1 && !this.getSelectedCluster();
  }

  hasClusterAccess(): boolean {
    return this.getActiveMemberships().length > 0;
  }

  async setup2FA(): Promise<TwoFactorSetupResponse> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    return this.request<TwoFactorSetupResponse>('/api/auth/2fa/setup', {
      method: 'POST',
      body: JSON.stringify({
        user_id: user.id,
        account_name: user.email,
      }),
    });
  }

  async verify2FASetup(data: TwoFactorVerifyRequest): Promise<TwoFactorVerifyResponse> {
    const user = this.getCurrentUser();
    if (!user && !data.user_id) {
      throw new Error('User ID required for 2FA verification');
    }
    return this.request<TwoFactorVerifyResponse>('/api/auth/2fa/verify', {
      method: 'POST',
      body: JSON.stringify({
        user_id: data.user_id || user?.id,
        code: data.code,
        is_backup_code: data.is_backup_code || false,
      }),
    });
  }

  async verify2FALogin(data: TwoFactorVerifyRequest): Promise<AuthResponse> {
    return this.request<AuthResponse>('/api/auth/2fa/verify-login', {
      method: 'POST',
      body: JSON.stringify({
        user_id: data.user_id,
        code: data.code,
        is_backup_code: data.is_backup_code || false,
        temp_token: data.temp_token,
      }),
    });
  }

  async enable2FA(code: string): Promise<{ success: boolean; message: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    return this.request<{ success: boolean; message: string }>('/api/auth/2fa/enable', {
      method: 'POST',
      body: JSON.stringify({
        user_id: user.id,
        code,
      }),
    });
  }

  async disable2FA(): Promise<{ success: boolean; message: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    return this.request<{ success: boolean; message: string }>('/api/auth/2fa/disable', {
      method: 'POST',
      body: JSON.stringify({
        user_id: user.id,
      }),
    });
  }

  async get2FAStatus(): Promise<{ enabled: boolean; backup_codes_remaining: number; setup?: boolean; setup_at?: string; last_used?: string }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    const response = await this.request<{ enabled: boolean; setup?: boolean; setup_at?: string; last_used?: string }>('/api/auth/2fa/status?user_id=' + user.id, {
      method: 'GET',
    });
    return {
      enabled: response.enabled,
      backup_codes_remaining: 0,
      ...(typeof response.setup !== 'undefined' ? { setup: response.setup } : {}),
      ...(response.setup_at ? { setup_at: response.setup_at } : {}),
      ...(response.last_used ? { last_used: response.last_used } : {}),
    };
  }

  async generateBackupCodes(): Promise<{ backup_codes: string[] }> {
    const user = this.getCurrentUser();
    if (!user) {
      throw new Error('User not authenticated');
    }
    return this.request<{ backup_codes: string[] }>('/api/auth/2fa/backup-codes', {
      method: 'POST',
      body: JSON.stringify({
        user_id: user.id,
      }),
    });
  }

  setTempToken(tempToken: string) {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(TEMP_TOKEN_KEY, tempToken);
    }
  }

  getTempToken(): string | null {
    if (typeof window !== 'undefined') {
      return window.localStorage.getItem(TEMP_TOKEN_KEY);
    }
    return null;
  }

  removeTempToken() {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(TEMP_TOKEN_KEY);
    }
  }

  storeOAuthCallbackSession(payload: {
    token: string;
    refreshToken?: string;
    user: UserResponse;
    memberships: AdmissionResponse[];
    selectedCluster?: ClusterSummaryResponse;
    session: SessionResponse;
  }) {
    this.setSession({
      token: payload.token,
      ...(payload.refreshToken ? { refreshToken: payload.refreshToken } : {}),
      expiresAt: payload.session.expiresAt,
      user: payload.user,
      admission: payload.memberships.find((membership) => membership.selected)
        || payload.memberships.find((membership) => membership.admitted)
        || { admitted: false },
      memberships: payload.memberships,
      ...(payload.selectedCluster ? { selectedCluster: payload.selectedCluster } : {}),
      session: payload.session,
    });
  }
}

export const authService = new AuthService();
export default authService;
