"use client";

import type { ReactNode } from 'react';
import {
  RBACProvider as CanonicalRBACProvider,
  useRBAC,
  type Permission,
  type Role,
} from '@/contexts/RBACContext';

export { useRBAC, type Permission, type Role } from '@/contexts/RBACContext';

export interface User {
  id: string;
  username: string;
  email: string;
  roles: Role[];
  permissions: string[];
  metadata?: Record<string, unknown>;
}

export interface RBACProviderProps {
  children: ReactNode;
}

export const RBACProvider = ({ children }: RBACProviderProps) => (
  <CanonicalRBACProvider>{children}</CanonicalRBACProvider>
);

export const usePermissions = () => {
  const { permissions } = useRBAC();
  return permissions;
};

export const useHasPermission = (resource: string, action: string) => {
  const { hasPermission } = useRBAC();
  return hasPermission(resource, action);
};

export const useHasRole = (roleName: string) => {
  const { hasRole } = useRBAC();
  return hasRole(roleName);
};

export const useCanAccess = (resource: string, action: string) => {
  const { canAccess } = useRBAC();
  return canAccess(undefined, [{ resource, action }]);
};

export default RBACProvider;
