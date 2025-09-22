"use client";

import { ReactNode } from 'react';
import { useRBAC } from '@/contexts/RBACContext';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ShieldX } from 'lucide-react';

interface RBACGuardProps {
  children: ReactNode;
  requiredRoles?: string[];
  requiredPermissions?: {resource: string, action: string}[];
  fallback?: ReactNode;
  showError?: boolean;
}

export default function RBACGuard({
  children,
  requiredRoles,
  requiredPermissions,
  fallback,
  showError = true
}: RBACGuardProps) {
  const { canAccess } = useRBAC();

  const hasAccess = canAccess(requiredRoles, requiredPermissions);

  if (!hasAccess) {
    if (fallback) {
      return <>{fallback}</>;
    }

    if (showError) {
      return (
        <Alert variant="destructive">
          <ShieldX className="h-4 w-4" />
          <AlertTitle>Access Denied</AlertTitle>
          <AlertDescription>
            You don't have the required permissions to view this content.
          </AlertDescription>
        </Alert>
      );
    }

    return null;
  }

  return <>{children}</>;
}