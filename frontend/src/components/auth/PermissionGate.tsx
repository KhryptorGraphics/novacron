"use client";

import { ReactNode } from 'react';
import { useRBAC } from '@/contexts/RBACContext';

interface PermissionGateProps {
  children: ReactNode;
  resource: string;
  action: string;
  fallback?: ReactNode;
}

export default function PermissionGate({
  children,
  resource,
  action,
  fallback
}: PermissionGateProps) {
  const { hasPermission } = useRBAC();

  if (!hasPermission(resource, action)) {
    return fallback ? <>{fallback}</> : null;
  }

  return <>{children}</>;
}