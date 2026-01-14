"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { authService } from "@/lib/auth";

interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  tenantId?: string;
  status: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (firstName: string, lastName: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      try {
        // Check if user is already authenticated
        const storedToken = authService.getToken();
        if (storedToken) {
          setToken(storedToken);
          // Try to get current user data
          const currentUser = authService.getCurrentUser();
          if (currentUser) {
            setUser(currentUser);
          } else {
            // Fallback user object for demo purposes
            setUser({
              id: "user-123",
              email: "user@example.com",
              firstName: "John",
              lastName: "Doe",
              status: "active"
            });
          }
        }
      } catch (error) {
        console.error('Failed to initialize auth:', error);
        // Clear invalid token
        authService.removeToken();
      } finally {
        setIsLoading(false);
      }
    };
    
    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    if (!email || !password) {
      throw new Error('Email and password are required');
    }
    
    try {
      const response = await authService.login({ email, password });
      if (response?.token && response?.user) {
        setToken(response.token);
        setUser(response.user);
        authService.setToken(response.token, response.user);
      } else {
        throw new Error('Invalid response from login service');
      }
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  };

  const register = async (firstName: string, lastName: string, email: string, password: string) => {
    if (!firstName || !lastName || !email || !password) {
      throw new Error('All fields are required for registration');
    }
    
    try {
      const response = await authService.register({ firstName, lastName, email, password });
      // After registration, user needs to log in
      return response;
    } catch (error) {
      console.error('Registration failed:', error);
      throw error;
    }
  };

  const logout = () => {
    try {
      authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setToken(null);
      setUser(null);
    }
  };

  const isAuthenticated = !!(token && user);

  const value = {
    user,
    token,
    login,
    register,
    logout,
    isAuthenticated,
    isLoading
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

// Protected route component
export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return <div>Loading...</div>;
  }
  
  if (!isAuthenticated) {
    // In a real implementation, you would redirect to login page
    // For now, we'll just return null
    return null;
  }
  
  return <>{children}</>;
}