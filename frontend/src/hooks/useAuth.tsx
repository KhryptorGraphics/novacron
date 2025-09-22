import { useState, useEffect, useContext, createContext, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { authService, UserResponse } from '@/lib/auth';

interface AuthContextType {
  user: UserResponse | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  requires2FA: boolean;
  tempToken: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  verify2FA: (code: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [requires2FA, setRequires2FA] = useState(false);
  const [tempToken, setTempToken] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const token = authService.getToken();
    const storedTempToken = authService.getTempToken();

    if (token) {
      const currentUser = authService.getCurrentUser();
      setUser(currentUser);
    } else if (storedTempToken) {
      setTempToken(storedTempToken);
      setRequires2FA(true);
    }

    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await authService.login({ email, password });

      if (response.requires_2fa && response.temp_token) {
        authService.setTempToken(response.temp_token);
        setTempToken(response.temp_token);
        setRequires2FA(true);
      } else {
        authService.setToken(response.token, response.user);
        setUser(response.user);
        setRequires2FA(false);
        setTempToken(null);
      }
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  };

  const verify2FA = async (code: string) => {
    if (!tempToken) throw new Error('No temporary token available');

    try {
      const response = await authService.verify2FALogin({
        code,
        temp_token: tempToken
      });

      authService.setToken(response.token, response.user);
      authService.removeTempToken();
      setUser(response.user);
      setRequires2FA(false);
      setTempToken(null);
    } catch (error) {
      console.error('2FA verification failed:', error);
      throw error;
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
    setRequires2FA(false);
    setTempToken(null);
    router.push('/auth/login');
  };

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      isAuthenticated: !!user,
      requires2FA,
      tempToken,
      login,
      logout,
      verify2FA,
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