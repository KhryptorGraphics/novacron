'use client';

import { useEffect, useMemo, useState } from 'react';
import { Loader2, Save, Shield, UserCheck } from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import { securityAPI, type SecurityPermissionDefinition, type SecurityRoleDefinition } from '@/lib/api/security';

export function RolePermissionManager() {
  const { toast } = useToast();
  const [roles, setRoles] = useState<SecurityRoleDefinition[]>([]);
  const [permissions, setPermissions] = useState<SecurityPermissionDefinition[]>([]);
  const [userId, setUserId] = useState('');
  const [selectedRoles, setSelectedRoles] = useState<string[]>([]);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [loadingAssignments, setLoadingAssignments] = useState(false);
  const [savingAssignments, setSavingAssignments] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      setLoadingCatalog(true);
      setError(null);

      try {
        const [roleDefinitions, permissionDefinitions] = await Promise.all([
          securityAPI.getRoles(),
          securityAPI.getPermissions(),
        ]);

        if (cancelled) {
          return;
        }

        setRoles(roleDefinitions);
        setPermissions(permissionDefinitions);
      } catch (catalogError) {
        if (!cancelled) {
          setError(
            catalogError instanceof Error
              ? catalogError.message
              : 'Failed to load RBAC catalog from the canonical server.',
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingCatalog(false);
        }
      }
    }

    loadCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedRoleDefinitions = useMemo(
    () => roles.filter((role) => selectedRoles.includes(role.id)),
    [roles, selectedRoles],
  );

  const selectedPermissionIds = useMemo(() => {
    const permissionSet = new Set<string>();
    for (const role of selectedRoleDefinitions) {
      for (const permission of role.permissions) {
        permissionSet.add(permission);
      }
    }
    return Array.from(permissionSet).sort();
  }, [selectedRoleDefinitions]);

  const selectedPermissionDefinitions = useMemo(
    () => permissions.filter((permission) => selectedPermissionIds.includes(permission.id)),
    [permissions, selectedPermissionIds],
  );

  const toggleRole = (roleId: string, checked: boolean) => {
    setSelectedRoles((current) =>
      checked ? Array.from(new Set([...current, roleId])) : current.filter((role) => role !== roleId),
    );
  };

  const loadAssignments = async () => {
    if (!userId.trim()) {
      toast({
        title: 'User ID required',
        description: 'Enter a user ID before loading canonical role assignments.',
        variant: 'destructive',
      });
      return;
    }

    setLoadingAssignments(true);
    setError(null);

    try {
      const assignments = await securityAPI.getUserRoleAssignments(userId.trim());
      setSelectedRoles(assignments);
    } catch (assignmentError) {
      setError(
        assignmentError instanceof Error
          ? assignmentError.message
          : 'Failed to load user role assignments.',
      );
    } finally {
      setLoadingAssignments(false);
    }
  };

  const saveAssignments = async () => {
    if (!userId.trim()) {
      toast({
        title: 'User ID required',
        description: 'Enter a user ID before saving canonical role assignments.',
        variant: 'destructive',
      });
      return;
    }

    if (selectedRoles.length === 0) {
      toast({
        title: 'Role required',
        description: 'Select at least one role before saving.',
        variant: 'destructive',
      });
      return;
    }

    setSavingAssignments(true);
    setError(null);

    try {
      const assignments = await securityAPI.assignUserRoles(userId.trim(), selectedRoles);
      setSelectedRoles(assignments);
      toast({
        title: 'Assignments saved',
        description: `Canonical RBAC assignments were updated for user ${userId.trim()}.`,
      });
    } catch (assignmentError) {
      setError(
        assignmentError instanceof Error
          ? assignmentError.message
          : 'Failed to save user role assignments.',
      );
    } finally {
      setSavingAssignments(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-2xl font-semibold">
            <UserCheck className="h-6 w-6" />
            Roles & Permissions
          </h2>
          <p className="text-sm text-muted-foreground">
            Assignment-only RBAC management backed by the canonical security API.
          </p>
        </div>
        <Badge variant="outline">Release Path</Badge>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>RBAC sync failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Load User Assignments</CardTitle>
          <CardDescription>
            Enter a user ID, load the current role assignments, then save an updated canonical role set.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 md:flex-row md:items-end">
          <div className="flex-1 space-y-2">
            <Label htmlFor="rbac-user-id">User ID</Label>
            <Input
              id="rbac-user-id"
              placeholder="e.g. 42"
              value={userId}
              onChange={(event) => setUserId(event.target.value)}
            />
          </div>
          <Button variant="outline" onClick={loadAssignments} disabled={loadingAssignments || loadingCatalog}>
            {loadingAssignments ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Load Assignments
          </Button>
          <Button onClick={saveAssignments} disabled={savingAssignments || loadingCatalog}>
            {savingAssignments ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
            Save Assignments
          </Button>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
        <Card>
          <CardHeader>
            <CardTitle>Canonical Roles</CardTitle>
            <CardDescription>
              The release candidate only supports assigning existing roles. Custom role CRUD stays out of scope.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {loadingCatalog ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading canonical role catalog…
              </div>
            ) : (
              roles.map((role) => (
                <div key={role.id} className="rounded-lg border p-4">
                  <div className="flex items-start gap-3">
                    <Checkbox
                      id={`role-${role.id}`}
                      checked={selectedRoles.includes(role.id)}
                      onCheckedChange={(checked) => toggleRole(role.id, checked === true)}
                    />
                    <div className="space-y-2">
                      <Label htmlFor={`role-${role.id}`} className="flex cursor-pointer items-center gap-2 text-base font-medium">
                        <Shield className="h-4 w-4" />
                        {role.name}
                      </Label>
                      <p className="text-sm text-muted-foreground">{role.description}</p>
                      <div className="flex flex-wrap gap-2">
                        {role.permissions.map((permission) => (
                          <Badge key={permission} variant="secondary">
                            {permission}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Effective Permissions</CardTitle>
            <CardDescription>Permissions implied by the selected canonical roles.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {selectedPermissionDefinitions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Load or choose at least one role to inspect the effective permission set.
              </p>
            ) : (
              selectedPermissionDefinitions.map((permission) => (
                <div key={permission.id} className="rounded-md border p-3">
                  <div className="font-medium">{permission.name}</div>
                  <div className="text-xs text-muted-foreground">{permission.id}</div>
                  <p className="mt-1 text-sm text-muted-foreground">{permission.description}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default RolePermissionManager;
