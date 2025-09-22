import { useRBAC } from '@/contexts/RBACContext';

export function usePermissions() {
  const { hasPermission, hasRole, canAccess, userRoles, permissions } = useRBAC();

  // Helper functions for common permission checks
  const canCreateVMs = () => hasPermission('vms', 'create');
  const canDeleteVMs = () => hasPermission('vms', 'delete');
  const canMigrateVMs = () => hasPermission('vms', 'migrate');
  const canManageUsers = () => hasPermission('users', 'create') && hasPermission('users', 'update');
  const canConfigureSecurity = () => hasPermission('security', 'configure');
  const canAccessMonitoring = () => hasPermission('monitoring', 'read');
  const canManageBackups = () => hasPermission('backups', 'create');
  const canRestoreBackups = () => hasPermission('backups', 'restore');
  const canUpdateSettings = () => hasPermission('settings', 'update');

  // Role checks
  const isAdmin = () => hasRole('admin') || hasRole('super-admin');
  const isSuperAdmin = () => hasRole('super-admin');
  const isOperator = () => hasRole('operator');
  const isViewer = () => hasRole('viewer');

  return {
    // Core RBAC functions
    hasPermission,
    hasRole,
    canAccess,
    userRoles,
    permissions,

    // VM permissions
    canCreateVMs,
    canDeleteVMs,
    canMigrateVMs,

    // User management permissions
    canManageUsers,

    // Security permissions
    canConfigureSecurity,

    // Monitoring permissions
    canAccessMonitoring,

    // Backup permissions
    canManageBackups,
    canRestoreBackups,

    // Settings permissions
    canUpdateSettings,

    // Role checks
    isAdmin,
    isSuperAdmin,
    isOperator,
    isViewer,
  };
}