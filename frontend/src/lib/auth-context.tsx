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
    // Check if user is already authenticated
    const storedToken = authService.getToken();
    if (storedToken) {
      setToken(storedToken);
      // In a real implementation, you would fetch user data from the token or API
      // For now, we'll set a basic user object
      setUser({
        id: "user-123",
        email: "user@example.com",
        firstName: "John",
        lastName: "Doe",
        status: "active"
      });
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await authService.login({ email, password });
      setToken(response.token);
      setUser(response.user);
      authService.setToken(response.token);
    } catch (error) {
      throw error;
    }
  };

  const register = async (firstName: string, lastName: string, email: string, password: string) => {
    try {
      await authService.register({ firstName, lastName, email, password });
      // After registration, user needs to log in
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    authService.logout();
    setToken(null);
    setUser(null);
  };

  const isAuthenticated = !!token;

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