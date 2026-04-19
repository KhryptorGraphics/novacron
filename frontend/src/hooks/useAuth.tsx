'use client';

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  AdmissionResponse,
  AuthResponse,
  ClusterSummaryResponse,
  CurrentUserResponse,
  SessionResponse,
  UserResponse,
  authService,
} from '@/lib/auth';

interface AuthContextType {
  user: UserResponse | null;
  memberships: AdmissionResponse[];
  selectedCluster: ClusterSummaryResponse | null;
  session: SessionResponse | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  requires2FA: boolean;
  tempToken: string | null;
  needsClusterSelection: boolean;
  hasClusterAccess: boolean;
  login: (email: string, password: string) => Promise<string | null>;
  logout: () => Promise<void>;
  verify2FA: (code: string) => Promise<string>;
  selectCluster: (clusterId: string) => Promise<string>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

function isFullAuthResponse(payload: AuthResponse | CurrentUserResponse): payload is AuthResponse {
  return 'token' in payload;
}

function applyAuthPayload(
  payload: AuthResponse | CurrentUserResponse | null,
  setUser: (value: UserResponse | null) => void,
  setMemberships: (value: AdmissionResponse[]) => void,
  setSelectedCluster: (value: ClusterSummaryResponse | null) => void,
  setSession: (value: SessionResponse | null) => void,
) {
  if (!payload) {
    setUser(null);
    setMemberships([]);
    setSelectedCluster(null);
    setSession(null);
    return;
  }

  if (isFullAuthResponse(payload)) {
    authService.setSession(payload);
  } else {
    authService.setSession(payload);
  }

  setUser(payload.user);
  setMemberships(payload.memberships || []);
  setSelectedCluster(payload.selectedCluster || null);
  setSession(payload.session || null);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [memberships, setMemberships] = useState<AdmissionResponse[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterSummaryResponse | null>(null);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [requires2FA, setRequires2FA] = useState(false);
  const [tempToken, setTempToken] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      const storedTempToken = authService.getTempToken();
      if (storedTempToken) {
        setTempToken(storedTempToken);
        setRequires2FA(true);
      }

      const restored = await authService.restoreSession();
      if (!cancelled) {
        applyAuthPayload(restored, setUser, setMemberships, setSelectedCluster, setSession);
        setIsLoading(false);
      }
    }

    bootstrap().catch((error) => {
      console.error('Failed to bootstrap auth session:', error);
      if (!cancelled) {
        applyAuthPayload(null, setUser, setMemberships, setSelectedCluster, setSession);
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const login = async (email: string, password: string) => {
    const response = await authService.login({ email, password });

    if (response.requires_2fa && response.temp_token) {
      authService.setTempToken(response.temp_token);
      setTempToken(response.temp_token);
      setRequires2FA(true);
      return null;
    }

    applyAuthPayload(response, setUser, setMemberships, setSelectedCluster, setSession);
    setRequires2FA(false);
    setTempToken(null);
    return authService.resolvePostLoginPath(response.memberships, response.selectedCluster);
  };

  const verify2FA = async (code: string) => {
    if (!tempToken) {
      throw new Error('No temporary token available');
    }

    const response = await authService.verify2FALogin({
      user_id: '',
      code,
      temp_token: tempToken,
    });

    if (response.requires_2fa) {
      throw new Error('Two-factor authentication is still required');
    }

    applyAuthPayload(response, setUser, setMemberships, setSelectedCluster, setSession);
    authService.removeTempToken();
    setRequires2FA(false);
    setTempToken(null);
    return authService.resolvePostLoginPath(response.memberships, response.selectedCluster);
  };

  const logout = async () => {
    await authService.logout();
    applyAuthPayload(null, setUser, setMemberships, setSelectedCluster, setSession);
    setRequires2FA(false);
    setTempToken(null);
    router.push('/auth/login');
  };

  const selectCluster = async (clusterId: string) => {
    const response = await authService.selectCluster(clusterId);
    applyAuthPayload(response, setUser, setMemberships, setSelectedCluster, setSession);
    return authService.resolvePostLoginPath(response.memberships, response.selectedCluster);
  };

  const activeMemberships = memberships.filter((membership) => membership.admitted || membership.state === 'active');

  return (
    <AuthContext.Provider value={{
      user,
      memberships,
      selectedCluster,
      session,
      isLoading,
      isAuthenticated: !!user,
      requires2FA,
      tempToken,
      needsClusterSelection: activeMemberships.length > 1 && !selectedCluster,
      hasClusterAccess: activeMemberships.length > 0,
      login,
      logout,
      verify2FA,
      selectCluster,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
