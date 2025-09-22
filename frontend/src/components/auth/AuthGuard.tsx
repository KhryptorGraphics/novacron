"use client";

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { Icons } from '@/components/ui/icons';

interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
  redirectTo?: string;
}

export default function AuthGuard({
  children,
  requireAuth = true,
  redirectTo = '/auth/login'
}: AuthGuardProps) {
  const { isAuthenticated, isLoading, requires2FA } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (isLoading) return;

    // Public routes that don't require authentication
    const publicRoutes = ['/auth/login', '/auth/register', '/auth/forgot-password', '/auth/reset-password'];
    const isPublicRoute = publicRoutes.some(route => pathname.startsWith(route));

    if (requireAuth && !isAuthenticated && !requires2FA && !isPublicRoute) {
      router.push(redirectTo);
      return;
    }

    // If authenticated but on auth pages, redirect to dashboard
    if (isAuthenticated && isPublicRoute) {
      router.push('/dashboard');
      return;
    }

    // If 2FA is required, only allow 2FA pages
    if (requires2FA && !pathname.startsWith('/auth/')) {
      // Let the login page handle the 2FA flow
      router.push('/auth/login');
      return;
    }
  }, [isAuthenticated, isLoading, requires2FA, pathname, router, requireAuth, redirectTo]);

  // Show loading spinner while checking auth
  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex items-center space-x-2">
          <Icons.spinner className="h-6 w-6 animate-spin" />
          <span className="text-muted-foreground">Loading...</span>
        </div>
      </div>
    );
  }

  // If require auth but not authenticated, don't render children
  if (requireAuth && !isAuthenticated && !requires2FA) {
    return null;
  }

  return <>{children}</>;
}