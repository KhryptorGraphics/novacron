'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { apiClient } from '@/lib/api/client';
import { useWebSocket } from '@/hooks/useWebSocket';

// Types
export interface User {
  id: string;
  username: string;
  email: string;
  roles: Role[];
  permissions: string[];
  metadata?: Record<string, any>;
}

export interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  priority?: number;
}

export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
  conditions?: Record<string, any>;
}

export interface RBACContextType {
  user: User | null;
  roles: Role[];
  permissions: string[];
  loading: boolean;
  error: string | null;
  hasPermission: (permission: string) => boolean;
  hasRole: (roleId: string) => boolean;
  hasAnyRole: (roleIds: string[]) => boolean;
  hasAllRoles: (roleIds: string[]) => boolean;
  hasAnyPermission: (permissions: string[]) => boolean;
  hasAllPermissions: (permissions: string[]) => boolean;
  refreshPermissions: () => Promise<void>;
  canAccess: (resource: string, action: string) => boolean;
}

// Context
const RBACContext = createContext<RBACContextType | undefined>(undefined);

// Provider Props
interface RBACProviderProps {
  children: ReactNode;
  userId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

// Provider Component
export const RBACProvider: React.FC<RBACProviderProps> = ({
  children,
  userId,
  autoRefresh = true,
  refreshInterval = 60000, // 1 minute
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/api/security/events/stream', {
    enabled: autoRefresh,
    reconnect: true,
  });

  // Fetch user permissions
  const fetchUserPermissions = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch permissions for the user
      const permissionsResponse = await apiClient.get(`/api/security/rbac/user/${userId}/permissions`);
      const permissionsData = permissionsResponse.data.permissions || [];

      // Fetch roles for the user
      const rolesResponse = await apiClient.get(`/api/security/rbac/user/${userId}/roles`);
      const rolesData = rolesResponse.data.roles || [];

      // Create a mock user object since the backend doesn't provide user details endpoint
      const userData: User = {
        id: userId,
        username: userId, // Use userId as username fallback
        email: `${userId}@mock.local`, // Mock email for development
        roles: rolesData.map((roleName: string) => ({
          id: roleName,
          name: roleName,
          permissions: [],
        })),
        permissions: permissionsData,
      };

      setUser(userData);
      setRoles(userData.roles);
      setPermissions(permissionsData);
    } catch (err) {
      console.error('Failed to fetch RBAC data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch permissions');
    } finally {
      setLoading(false);
    }
  }, [userId]);

  // Initial fetch
  useEffect(() => {
    fetchUserPermissions();
  }, [fetchUserPermissions]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh || !userId) return;

    const interval = setInterval(fetchUserPermissions, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, userId, refreshInterval, fetchUserPermissions]);

  // Handle WebSocket updates
  useEffect(() => {
    if (!wsData) return;

    // Handle permission update events
    if (wsData.type === 'permission_update' && wsData.userId === userId) {
      fetchUserPermissions();
    }

    // Handle role update events
    if (wsData.type === 'role_update') {
      const affectedUserIds = wsData.affectedUsers || [];
      if (affectedUserIds.includes(userId)) {
        fetchUserPermissions();
      }
    }
  }, [wsData, userId, fetchUserPermissions]);

  // Permission check functions
  const hasPermission = useCallback((permission: string): boolean => {
    return permissions.includes(permission);
  }, [permissions]);

  const hasRole = useCallback((roleId: string): boolean => {
    return roles.some(role => role.id === roleId);
  }, [roles]);

  const hasAnyRole = useCallback((roleIds: string[]): boolean => {
    return roleIds.some(roleId => hasRole(roleId));
  }, [hasRole]);

  const hasAllRoles = useCallback((roleIds: string[]): boolean => {
    return roleIds.every(roleId => hasRole(roleId));
  }, [hasRole]);

  const hasAnyPermission = useCallback((perms: string[]): boolean => {
    return perms.some(perm => hasPermission(perm));
  }, [hasPermission]);

  const hasAllPermissions = useCallback((perms: string[]): boolean => {
    return perms.every(perm => hasPermission(perm));
  }, [hasPermission]);

  const canAccess = useCallback((resource: string, action: string): boolean => {
    const permission = `${resource}:${action}`;
    return hasPermission(permission);
  }, [hasPermission]);

  const refreshPermissions = useCallback(async () => {
    await fetchUserPermissions();
  }, [fetchUserPermissions]);

  const contextValue: RBACContextType = {
    user,
    roles,
    permissions,
    loading,
    error,
    hasPermission,
    hasRole,
    hasAnyRole,
    hasAllRoles,
    hasAnyPermission,
    hasAllPermissions,
    refreshPermissions,
    canAccess,
  };

  return (
    <RBACContext.Provider value={contextValue}>
      {children}
    </RBACContext.Provider>
  );
};

// Hooks
export const useRBAC = (): RBACContextType => {
  const context = useContext(RBACContext);
  if (!context) {
    throw new Error('useRBAC must be used within an RBACProvider');
  }
  return context;
};

export const usePermissions = (): string[] => {
  const { permissions } = useRBAC();
  return permissions;
};

export const useHasPermission = (permission: string): boolean => {
  const { hasPermission } = useRBAC();
  return hasPermission(permission);
};

export const useHasRole = (roleId: string): boolean => {
  const { hasRole } = useRBAC();
  return hasRole(roleId);
};

export const useCanAccess = (resource: string, action: string): boolean => {
  const { canAccess } = useRBAC();
  return canAccess(resource, action);
};

// Guard Components
interface PermissionGateProps {
  permission: string;
  children: ReactNode;
  fallback?: ReactNode;
}

export const PermissionGate: React.FC<PermissionGateProps> = ({
  permission,
  children,
  fallback = null,
}) => {
  const hasPermission = useHasPermission(permission);
  return hasPermission ? <>{children}</> : <>{fallback}</>;
};

interface RoleGateProps {
  role: string;
  children: ReactNode;
  fallback?: ReactNode;
}

export const RoleGate: React.FC<RoleGateProps> = ({
  role,
  children,
  fallback = null,
}) => {
  const hasRole = useHasRole(role);
  return hasRole ? <>{children}</> : <>{fallback}</>;
};

interface AccessGateProps {
  resource: string;
  action: string;
  children: ReactNode;
  fallback?: ReactNode;
}

export const AccessGate: React.FC<AccessGateProps> = ({
  resource,
  action,
  children,
  fallback = null,
}) => {
  const canAccess = useCanAccess(resource, action);
  return canAccess ? <>{children}</> : <>{fallback}</>;
};

// HOCs for class components
export function withRBAC<P extends object>(
  Component: React.ComponentType<P & RBACContextType>
): React.FC<P> {
  return (props: P) => {
    const rbacContext = useRBAC();
    return <Component {...props} {...rbacContext} />;
  };
}

export function withPermission<P extends object>(
  permission: string,
  FallbackComponent?: React.ComponentType<P>
) {
  return (Component: React.ComponentType<P>): React.FC<P> => {
    return (props: P) => {
      const hasPermission = useHasPermission(permission);

      if (hasPermission) {
        return <Component {...props} />;
      }

      if (FallbackComponent) {
        return <FallbackComponent {...props} />;
      }

      return null;
    };
  };
}

export function withRole<P extends object>(
  role: string,
  FallbackComponent?: React.ComponentType<P>
) {
  return (Component: React.ComponentType<P>): React.FC<P> => {
    return (props: P) => {
      const hasRole = useHasRole(role);

      if (hasRole) {
        return <Component {...props} />;
      }

      if (FallbackComponent) {
        return <FallbackComponent {...props} />;
      }

      return null;
    };
  };
}

// Utility function for conditional rendering
export const rbacRender = (
  condition: boolean,
  component: ReactNode,
  fallback: ReactNode = null
): ReactNode => {
  return condition ? component : fallback;
};

// Example usage component
export const RBACDebugPanel: React.FC = () => {
  const { user, roles, permissions, loading, error } = useRBAC();

  if (loading) return <div>Loading RBAC data...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-4 bg-gray-100 rounded">
      <h3 className="text-lg font-semibold mb-2">RBAC Debug Info</h3>
      <div>
        <strong>User:</strong> {user?.username || 'Not logged in'}
      </div>
      <div>
        <strong>Roles:</strong> {roles.map(r => r.name).join(', ') || 'None'}
      </div>
      <div>
        <strong>Permissions:</strong> {permissions.length} total
      </div>
    </div>
  );
};

export default RBACProvider;