'use client';

import { useAuth } from '@/hooks/useAuth';
import { useRouter, usePathname } from 'next/navigation';
import { ReactNode, useEffect } from 'react';

interface ProtectedRouteProps {
  children: ReactNode;
  requiredPermissions?: string[];
}

export function ProtectedRoute({ children, requiredPermissions }: ProtectedRouteProps) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) {
      // Store the intended destination
      const returnUrl = encodeURIComponent(pathname || '/dashboard');
      router.push(`/auth/login?returnUrl=${returnUrl}`);
    }
  }, [user, loading, router, pathname]);

  // Show loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!user) {
    return null;
  }

  // TODO: Implement permission checking
  // if (requiredPermissions && !hasPermissions(user, requiredPermissions)) {
  //   return (
  //     <div className="flex items-center justify-center min-h-screen">
  //       <div className="text-center space-y-4">
  //         <h2 className="text-2xl font-bold">Access Denied</h2>
  //         <p className="text-muted-foreground">You don't have permission to access this page</p>
  //       </div>
  //     </div>
  //   );
  // }

  return <>{children}</>;
}

