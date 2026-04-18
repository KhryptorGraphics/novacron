'use client';

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Loader2, Pencil, Shield, Trash2, UserPlus, Users } from 'lucide-react';

import AuthGuard from '@/components/auth/AuthGuard';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';
import { useAuth } from '@/hooks/useAuth';
import { adminApi, type CreateUserRequest, type UpdateUserRequest, type User } from '@/lib/api/admin';

type UserFormState = {
  username: string;
  email: string;
  password: string;
  role: string;
  active: boolean;
};

const emptyForm: UserFormState = {
  username: '',
  email: '',
  password: '',
  role: 'user',
  active: true,
};

function isAdminUser(user: ReturnType<typeof useAuth>['user']) {
  if (!user) {
    return false;
  }

  const roles = new Set([user.role, ...(user.roles || [])].filter(Boolean));
  return roles.has('admin') || roles.has('super-admin');
}

export default function UsersPage() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [form, setForm] = useState<UserFormState>(emptyForm);

  const canManageUsers = isAdminUser(user);

  const loadUsers = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await adminApi.users.list({
        page: 1,
        page_size: 100,
        search: search || undefined,
        role: roleFilter === 'all' ? undefined : roleFilter,
      });
      setUsers(response.users);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load canonical users.');
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!canManageUsers) {
      setLoading(false);
      return;
    }

    void loadUsers();
  }, [canManageUsers, search, roleFilter]);

  const filteredStats = useMemo(() => ({
    total: users.length,
    active: users.filter((entry) => entry.active).length,
    admins: users.filter((entry) => entry.role === 'admin').length,
    operators: users.filter((entry) => entry.role === 'operator').length,
  }), [users]);

  const openCreateDialog = () => {
    setEditingUser(null);
    setForm(emptyForm);
    setDialogOpen(true);
  };

  const openEditDialog = (selectedUser: User) => {
    setEditingUser(selectedUser);
    setForm({
      username: selectedUser.username,
      email: selectedUser.email,
      password: '',
      role: selectedUser.role,
      active: selectedUser.active,
    });
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setEditingUser(null);
    setForm(emptyForm);
  };

  const saveUser = async () => {
    setSaving(true);

    try {
      if (editingUser) {
        const updatePayload: UpdateUserRequest = {
          username: form.username,
          email: form.email,
          role: form.role,
          active: form.active,
        };
        const updatedUser = await adminApi.users.update(editingUser.id, updatePayload);
        setUsers((current) => current.map((entry) => (entry.id === updatedUser.id ? updatedUser : entry)));
        await adminApi.users.assignRoles(updatedUser.id, [form.role]);
        toast({
          title: 'User updated',
          description: `${updatedUser.username} now has the canonical ${form.role} role.`,
        });
      } else {
        const createPayload: CreateUserRequest = {
          username: form.username,
          email: form.email,
          password: form.password,
          role: form.role,
        };
        const createdUser = await adminApi.users.create(createPayload);
        await adminApi.users.assignRoles(createdUser.id, [form.role]);
        setUsers((current) => [createdUser, ...current]);
        toast({
          title: 'User created',
          description: `${createdUser.username} was added through the canonical admin API.`,
        });
      }

      closeDialog();
    } catch (saveError) {
      toast({
        title: editingUser ? 'User update failed' : 'User creation failed',
        description: saveError instanceof Error ? saveError.message : 'Canonical user management request failed.',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const deleteUser = async (selectedUser: User) => {
    if (!window.confirm(`Delete ${selectedUser.username}? This action cannot be undone.`)) {
      return;
    }

    try {
      await adminApi.users.delete(selectedUser.id);
      setUsers((current) => current.filter((entry) => entry.id !== selectedUser.id));
      toast({
        title: 'User deleted',
        description: `${selectedUser.username} was removed from the canonical admin surface.`,
      });
    } catch (deleteError) {
      toast({
        title: 'User deletion failed',
        description: deleteError instanceof Error ? deleteError.message : 'Failed to delete user.',
        variant: 'destructive',
      });
    }
  };

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Users</h1>
            <p className="text-muted-foreground">
              Admin-only user management backed by `/api/admin/users*`.
            </p>
          </div>
          <Button onClick={openCreateDialog} disabled={!canManageUsers}>
            <UserPlus className="mr-2 h-4 w-4" />
            Add User
          </Button>
        </div>

        {!canManageUsers ? (
          <Card>
            <CardHeader>
              <CardTitle>Administrator access required</CardTitle>
              <CardDescription>
                The release candidate only exposes user management to `admin` and `super-admin` roles.
              </CardDescription>
            </CardHeader>
          </Card>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{filteredStats.total}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active</CardTitle>
                  <Badge variant="secondary">{filteredStats.active}</Badge>
                </CardHeader>
                <CardContent>
                  <div className="text-sm text-muted-foreground">Canonical `active` flag from the admin API.</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Administrators</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{filteredStats.admins}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Operators</CardTitle>
                  <Badge variant="outline">{filteredStats.operators}</Badge>
                </CardHeader>
                <CardContent>
                  <div className="text-sm text-muted-foreground">Assignment-only role updates stay on the canonical surface.</div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Filters</CardTitle>
                <CardDescription>Search by username/email and narrow to a canonical role.</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4 md:flex-row">
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search users"
                  className="md:max-w-sm"
                />
                <Select value={roleFilter} onValueChange={setRoleFilter}>
                  <SelectTrigger className="md:max-w-[180px]">
                    <SelectValue placeholder="All roles" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All roles</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                    <SelectItem value="operator">Operator</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                    <SelectItem value="user">User</SelectItem>
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>User Inventory</CardTitle>
                <CardDescription>
                  Bulk import/export, approval queues, and placeholder actions were removed from the release path.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading canonical users…
                  </div>
                ) : error ? (
                  <div className="flex items-center text-sm text-destructive">
                    <AlertTriangle className="mr-2 h-4 w-4" />
                    {error}
                  </div>
                ) : users.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No users matched the current filter.</div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Username</TableHead>
                        <TableHead>Email</TableHead>
                        <TableHead>Role</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {users.map((entry) => (
                        <TableRow key={entry.id}>
                          <TableCell className="font-medium">{entry.username}</TableCell>
                          <TableCell>{entry.email}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{entry.role}</Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant={entry.active ? 'secondary' : 'destructive'}>
                              {entry.active ? 'active' : 'inactive'}
                            </Badge>
                          </TableCell>
                          <TableCell>{new Date(entry.created_at).toLocaleDateString()}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button size="sm" variant="outline" onClick={() => openEditDialog(entry)}>
                                <Pencil className="mr-2 h-4 w-4" />
                                Edit
                              </Button>
                              <Button size="sm" variant="outline" onClick={() => deleteUser(entry)}>
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </>
        )}

        <Dialog open={dialogOpen} onOpenChange={(open) => (!open ? closeDialog() : setDialogOpen(true))}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editingUser ? 'Edit User' : 'Create User'}</DialogTitle>
              <DialogDescription>
                {editingUser
                  ? 'Update the canonical admin user record and its assigned role.'
                  : 'Create a user through the canonical admin API and assign a release-surface role.'}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  value={form.username}
                  onChange={(event) => setForm((current) => ({ ...current, username: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  value={form.email}
                  onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))}
                />
              </div>
              {!editingUser ? (
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={form.password}
                    onChange={(event) => setForm((current) => ({ ...current, password: event.target.value }))}
                  />
                </div>
              ) : null}
              <div className="space-y-2">
                <Label htmlFor="role">Role</Label>
                <Select value={form.role} onValueChange={(value) => setForm((current) => ({ ...current, role: value }))}>
                  <SelectTrigger id="role">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Admin</SelectItem>
                    <SelectItem value="operator">Operator</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                    <SelectItem value="user">User</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {editingUser ? (
                <div className="space-y-2">
                  <Label htmlFor="active">Status</Label>
                  <Select
                    value={form.active ? 'active' : 'inactive'}
                    onValueChange={(value) => setForm((current) => ({ ...current, active: value === 'active' }))}
                  >
                    <SelectTrigger id="active">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="inactive">Inactive</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={closeDialog}>Cancel</Button>
              <Button onClick={saveUser} disabled={saving || !form.username || !form.email || (!editingUser && !form.password)}>
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                {editingUser ? 'Save Changes' : 'Create User'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AuthGuard>
  );
}
