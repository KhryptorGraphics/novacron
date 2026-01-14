"use client";

import { createContext, useContext, ReactNode } from 'react';
import { useAuth } from '@/hooks/useAuth';

export interface Permission {
  resource: string;
  actions: string[];
}

export interface Role {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
  isSystemRole?: boolean;
}

interface RBACContextType {
  userRoles: Role[];
  permissions: Permission[];
  hasPermission: (resource: string, action: string) => boolean;
  hasRole: (roleName: string) => boolean;
  canAccess: (requiredRoles?: string[], requiredPermissions?: {resource: string, action: string}[]) => boolean;
}

const RBACContext = createContext<RBACContextType | undefined>(undefined);

// Mock roles and permissions data - in production, this would come from your backend
const SYSTEM_ROLES: Role[] = [
  {
    id: 'super-admin',
    name: 'Super Administrator',
    description: 'Full system access with all permissions',
    isSystemRole: true,
    permissions: [
      { resource: '*', actions: ['*'] }
    ]
  },
  {
    id: 'admin',
    name: 'Administrator',
    description: 'System administration with most permissions',
    isSystemRole: true,
    permissions: [
      { resource: 'users', actions: ['create', 'read', 'update', 'delete'] },
      { resource: 'vms', actions: ['create', 'read', 'update', 'delete', 'migrate'] },
      { resource: 'monitoring', actions: ['read', 'configure'] },
      { resource: 'security', actions: ['read', 'configure'] },
      { resource: 'backups', actions: ['create', 'read', 'restore'] },
      { resource: 'settings', actions: ['read', 'update'] }
    ]
  },
  {
    id: 'operator',
    name: 'Operator',
    description: 'VM operations and monitoring access',
    isSystemRole: true,
    permissions: [
      { resource: 'vms', actions: ['read', 'update', 'migrate'] },
      { resource: 'monitoring', actions: ['read'] },
      { resource: 'backups', actions: ['create', 'read'] }
    ]
  },
  {
    id: 'viewer',
    name: 'Viewer',
    description: 'Read-only access to system resources',
    isSystemRole: true,
    permissions: [
      { resource: 'vms', actions: ['read'] },
      { resource: 'monitoring', actions: ['read'] },
      { resource: 'backups', actions: ['read'] }
    ]
  }
];

export function RBACProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();

  // Mock user roles - in production, this would come from user data
  const getUserRoles = (): Role[] => {
    if (!user) return [];

    // For demo purposes, assign roles based on user email or ID
    // In production, this would come from the user object or a separate API call
    const userRoleNames = (user as any).roles || ['viewer']; // Default to viewer

    return SYSTEM_ROLES.filter(role => userRoleNames.includes(role.id));
  };

  const userRoles = getUserRoles();

  // SSR-safe: Ensure userRoles is always an array before calling flatMap
  const permissions: Permission[] = Array.isArray(userRoles)
    ? userRoles.flatMap(role => role?.permissions || [])
    : [];

  const hasPermission = (resource: string, action: string): boolean => {
    if (!user || !Array.isArray(permissions)) return false;

    // Check for super admin wildcard permission
    const hasSuperAdmin = permissions.some(
      perm => perm?.resource === '*' && perm?.actions?.includes('*')
    );

    if (hasSuperAdmin) return true;

    // Check specific permissions
    return permissions.some(perm => {
      if (!perm || !perm.actions) return false;
      const resourceMatch = perm.resource === resource || perm.resource === '*';
      const actionMatch = perm.actions.includes(action) || perm.actions.includes('*');
      return resourceMatch && actionMatch;
    });
  };

  const hasRole = (roleName: string): boolean => {
    if (!Array.isArray(userRoles)) return false;
    return userRoles.some(role => role?.id === roleName || role?.name === roleName);
  };

  const canAccess = (
    requiredRoles?: string[],
    requiredPermissions?: {resource: string, action: string}[]
  ): boolean => {
    if (!user) return false;

    // Check role requirements
    if (Array.isArray(requiredRoles) && requiredRoles.length > 0) {
      const hasRequiredRole = requiredRoles.some(role => hasRole(role));
      if (!hasRequiredRole) return false;
    }

    // Check permission requirements
    if (Array.isArray(requiredPermissions) && requiredPermissions.length > 0) {
      const hasAllPermissions = requiredPermissions.every(
        perm => hasPermission(perm?.resource, perm?.action)
      );
      if (!hasAllPermissions) return false;
    }

    return true;
  };

  return (
    <RBACContext.Provider value={{
      userRoles,
      permissions,
      hasPermission,
      hasRole,
      canAccess,
    }}>
      {children}
    </RBACContext.Provider>
  );
}

export function useRBAC() {
  const context = useContext(RBACContext);
  if (context === undefined) {
    throw new Error('useRBAC must be used within an RBACProvider');
  }
  return context;
}