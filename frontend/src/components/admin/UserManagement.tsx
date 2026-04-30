"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FadeIn } from "@/lib/animations";
import { useUsers, useUpdateUser } from "@/lib/api/hooks/useAdmin";
import type { User } from "@/lib/api/types";
import { 
  Users, 
  Search, 
  UserPlus, 
  Mail, 
  Shield, 
  Ban, 
  UserCheck, 
  Eye,
  Edit,
  AlertCircle,
  CheckCircle,
  Clock
} from "lucide-react";

type CanonicalAdminUser = Partial<User> & {
  username?: string;
  active?: boolean;
};

type UserRow = {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
  last_login: string | null;
  login_count: number;
  organization: string;
  two_factor: boolean;
  email_verified: boolean;
};

function normalizeUser(user: CanonicalAdminUser): UserRow {
  const activeStatus = user.active === false ? "suspended" : "active";

  return {
    id: String(user.id || ""),
    name: user.name || user.username || user.email || "Unnamed user",
    email: user.email || "",
    role: user.role || "user",
    status: user.status || activeStatus,
    last_login: user.last_login || null,
    login_count: user.login_count || 0,
    organization: user.organization || "Default",
    two_factor: user.two_factor_enabled || false,
    email_verified: user.email_verified ?? true,
  };
}

export function UserManagement() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [roleFilter, setRoleFilter] = useState("all");
  const userQueryFilters: { search?: string; status?: string; role?: string; pageSize?: number } = { pageSize: 100 };
  if (searchQuery) userQueryFilters.search = searchQuery;
  if (statusFilter !== "all") userQueryFilters.status = statusFilter;
  if (roleFilter !== "all") userQueryFilters.role = roleFilter;
  const usersQuery = useUsers(userQueryFilters);
  const updateUser = useUpdateUser();
  const users = (usersQuery.data?.users || []).map((user) => normalizeUser(user as CanonicalAdminUser));
  const loading = usersQuery.isLoading || updateUser.isPending;

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         user.organization.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || user.status === statusFilter;
    const matchesRole = roleFilter === "all" || user.role === roleFilter;
    
    return matchesSearch && matchesStatus && matchesRole;
  });

  const userStats = {
    total: usersQuery.data?.total || filteredUsers.length,
    active: filteredUsers.filter((user) => user.status === "active").length,
    pending: filteredUsers.filter((user) => user.status === "pending").length,
    twoFactorEnabled: filteredUsers.filter((user) => user.two_factor).length,
  };

  const handleUserAction = async (userId: string, action: string) => {
    const active = action === "activate" || action === "approve";
    await updateUser.mutateAsync({ id: userId, active });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active": return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "suspended": return <Ban className="h-4 w-4 text-red-600" />;
      case "pending": return <Clock className="h-4 w-4 text-yellow-600" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Never";
    return new Date(dateString).toLocaleDateString() + " " + 
           new Date(dateString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Users className="h-6 w-6" />
            User Management
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage user accounts, roles, and permissions
          </p>
        </div>
        
        <Button>
          <UserPlus className="h-4 w-4 mr-2" />
          Add User
        </Button>
      </div>

      {/* User Statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Users</p>
                  <p className="text-2xl font-bold">{userStats.total.toLocaleString()}</p>
                </div>
                <Users className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Users</p>
                  <p className="text-2xl font-bold text-green-600">{userStats.active}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Pending Approval</p>
                  <p className="text-2xl font-bold text-yellow-600">{userStats.pending}</p>
                </div>
                <Clock className="h-8 w-8 text-yellow-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">2FA Enabled</p>
                  <p className="text-2xl font-bold text-purple-600">{userStats.twoFactorEnabled}</p>
                </div>
                <Shield className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search users by name, email, or organization..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="suspended">Suspended</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={roleFilter} onValueChange={setRoleFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Roles</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="moderator">Moderator</SelectItem>
                <SelectItem value="user">User</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Users Table */}
      <FadeIn delay={0.5}>
        <Card>
          <CardHeader>
            <CardTitle>Users ({filteredUsers.length})</CardTitle>
            <CardDescription>
              Manage user accounts and their permissions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {Boolean(usersQuery.error) && (
              <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                Failed to load users from the canonical admin API.
              </div>
            )}
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>User</TableHead>
                    <TableHead>Organization</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Last Login</TableHead>
                    <TableHead>Security</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredUsers.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell>
                        <div className="flex flex-col">
                          <span className="font-medium">{user.name}</span>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {user.email}
                          </span>
                        </div>
                      </TableCell>
                      
                      <TableCell>{user.organization}</TableCell>
                      
                      <TableCell>
                        <Badge 
                          variant={user.role === 'admin' ? 'destructive' : user.role === 'moderator' ? 'secondary' : 'outline'}
                          className="capitalize"
                        >
                          {user.role}
                        </Badge>
                      </TableCell>
                      
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(user.status)}
                          <span className="capitalize">{user.status}</span>
                        </div>
                      </TableCell>
                      
                      <TableCell>
                        <div className="text-sm">
                          {formatDate(user.last_login)}
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            {user.login_count} total logins
                          </div>
                        </div>
                      </TableCell>
                      
                      <TableCell>
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-1 text-sm">
                            <Mail className="h-3 w-3" />
                            <span className={user.email_verified ? "text-green-600" : "text-red-600"}>
                              {user.email_verified ? "Verified" : "Unverified"}
                            </span>
                          </div>
                          <div className="flex items-center gap-1 text-sm">
                            <Shield className="h-3 w-3" />
                            <span className={user.two_factor ? "text-green-600" : "text-gray-600"}>
                              {user.two_factor ? "2FA On" : "2FA Off"}
                            </span>
                          </div>
                        </div>
                      </TableCell>
                      
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {user.status === 'pending' && (
                            <Button
                              size="sm"
                              onClick={() => handleUserAction(user.id, 'approve')}
                              disabled={loading}
                              className="text-xs px-2 h-7"
                            >
                              Approve
                            </Button>
                          )}
                          
                          {user.status === 'active' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleUserAction(user.id, 'suspend')}
                              disabled={loading}
                              className="text-xs px-2 h-7"
                            >
                              Suspend
                            </Button>
                          )}
                          
                          {user.status === 'suspended' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleUserAction(user.id, 'activate')}
                              disabled={loading}
                              className="text-xs px-2 h-7"
                            >
                              Activate
                            </Button>
                          )}
                          
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 w-7 p-0"
                            title={`${user.name} details are available through the canonical user table row`}
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                          
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 w-7 p-0"
                          >
                            <Edit className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Bulk Actions</CardTitle>
          <CardDescription>
            Perform actions on multiple users at once
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm">
              <Mail className="h-4 w-4 mr-2" />
              Send Welcome Email
            </Button>
            <Button variant="outline" size="sm">
              <Shield className="h-4 w-4 mr-2" />
              Enable 2FA Reminder
            </Button>
            <Button variant="outline" size="sm">
              <UserCheck className="h-4 w-4 mr-2" />
              Bulk Role Assignment
            </Button>
            <Button variant="outline" size="sm" className="text-red-600">
              <Ban className="h-4 w-4 mr-2" />
              Bulk Suspension
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
