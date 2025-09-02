"use client";

import { useState } from "react";

// Disable static generation for this page
export const dynamic = 'force-dynamic';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { 
  Users, 
  UserPlus,
  Search,
  MoreHorizontal,
  Edit,
  Trash2,
  Shield,
  Mail,
  Phone,
  Calendar,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Filter,
  Download,
  Upload
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// Mock user data
const mockUsers = [
  {
    id: "user-001",
    firstName: "John",
    lastName: "Doe", 
    email: "john.doe@company.com",
    role: "admin",
    status: "active",
    lastLogin: "2025-04-11T14:30:00Z",
    createdAt: "2024-01-15T10:00:00Z",
    permissions: ["vm.create", "vm.delete", "vm.migrate", "system.configure"],
    phone: "+1-555-0123",
    twoFactorEnabled: true,
    department: "IT Operations"
  },
  {
    id: "user-002",
    firstName: "Sarah",
    lastName: "Johnson",
    email: "sarah.johnson@company.com",
    role: "operator",
    status: "active",
    lastLogin: "2025-04-11T13:15:00Z",
    createdAt: "2024-02-20T09:30:00Z",
    permissions: ["vm.view", "vm.start", "vm.stop", "vm.restart"],
    phone: "+1-555-0124",
    twoFactorEnabled: true,
    department: "Engineering"
  },
  {
    id: "user-003",
    firstName: "Mike",
    lastName: "Chen",
    email: "mike.chen@company.com",
    role: "viewer",
    status: "inactive",
    lastLogin: "2025-04-05T16:45:00Z",
    createdAt: "2024-03-10T14:15:00Z",
    permissions: ["vm.view", "system.view"],
    phone: "+1-555-0125",
    twoFactorEnabled: false,
    department: "Development"
  },
  {
    id: "user-004",
    firstName: "Emma",
    lastName: "Wilson",
    email: "emma.wilson@company.com",
    role: "operator",
    status: "pending",
    lastLogin: null,
    createdAt: "2025-04-10T11:00:00Z",
    permissions: ["vm.view", "vm.start", "vm.stop"],
    phone: "+1-555-0126",
    twoFactorEnabled: false,
    department: "QA"
  }
];

const userRoles = [
  { id: "admin", name: "Administrator", description: "Full system access" },
  { id: "operator", name: "Operator", description: "VM management access" },
  { id: "viewer", name: "Viewer", description: "Read-only access" }
];

export default function UsersPage() {
  const [users, setUsers] = useState(mockUsers);
  const [searchQuery, setSearchQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);

  // Filter users with null checks
  const filteredUsers = Array.isArray(users) ? users.filter(user => {
    if (!user || typeof user !== 'object') return false;
    const matchesSearch = (user.firstName?.toLowerCase() || '').includes(searchQuery.toLowerCase()) ||
                         (user.lastName?.toLowerCase() || '').includes(searchQuery.toLowerCase()) ||
                         (user.email?.toLowerCase() || '').includes(searchQuery.toLowerCase());
    const matchesRole = roleFilter === "all" || user.role === roleFilter;
    const matchesStatus = statusFilter === "all" || user.status === statusFilter;
    
    return matchesSearch && matchesRole && matchesStatus;
  }) : [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "inactive": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      case "pending": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "suspended": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active": return <CheckCircle className="h-3 w-3" />;
      case "inactive": return <XCircle className="h-3 w-3" />;
      case "pending": return <AlertTriangle className="h-3 w-3" />;
      case "suspended": return <XCircle className="h-3 w-3" />;
      default: return <AlertTriangle className="h-3 w-3" />;
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case "admin": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "operator": return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      case "viewer": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const userStats = {
    totalUsers: Array.isArray(users) ? users.length : 0,
    activeUsers: Array.isArray(users) ? users.filter(u => u?.status === "active").length : 0,
    pendingUsers: Array.isArray(users) ? users.filter(u => u?.status === "pending").length : 0,
    adminUsers: Array.isArray(users) ? users.filter(u => u?.role === "admin").length : 0
  };

  const handleUserAction = (userId: string, action: string) => {
    console.log(`${action} user ${userId}`);
    // In a real app, this would call an API
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">User Management</h1>
          <p className="text-muted-foreground">Manage users, roles, and permissions</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Upload className="h-4 w-4 mr-2" />
            Import Users
          </Button>
          <Button onClick={() => setShowCreateDialog(true)}>
            <UserPlus className="h-4 w-4 mr-2" />
            Add User
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{userStats.totalUsers}</div>
            <p className="text-xs text-muted-foreground">
              {userStats.activeUsers} active
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Approval</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{userStats.pendingUsers}</div>
            <p className="text-xs text-muted-foreground">
              Awaiting review
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Administrators</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{userStats.adminUsers}</div>
            <p className="text-xs text-muted-foreground">
              Full access users
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">2FA Enabled</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Array.isArray(users) ? users.filter(u => u?.twoFactorEnabled).length : 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Enhanced security
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search users by name or email..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={roleFilter} onValueChange={setRoleFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Roles</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="operator">Operator</SelectItem>
                <SelectItem value="viewer">Viewer</SelectItem>
              </SelectContent>
            </Select>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Users Table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Department</TableHead>
              <TableHead>Last Login</TableHead>
              <TableHead>2FA</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Array.isArray(filteredUsers) && filteredUsers.length > 0 ? filteredUsers.map((user) => {
              if (!user || !user.id) return null;
              return (
              <TableRow key={user.id}>
                <TableCell>
                  <div className="flex items-center space-x-3">
                    <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-semibold">
                      {(user.firstName || '').charAt(0)}{(user.lastName || '').charAt(0)}
                    </div>
                    <div>
                      <div className="font-medium">{user.firstName || ''} {user.lastName || ''}</div>
                      <div className="text-sm text-muted-foreground">{user.email || ''}</div>
                    </div>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={getRoleColor(user.role)}>
                    <Shield className="h-3 w-3 mr-1" />
                    {user.role}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge className={getStatusColor(user.status)}>
                    {getStatusIcon(user.status)}
                    <span className="ml-1 capitalize">{user.status}</span>
                  </Badge>
                </TableCell>
                <TableCell>{user.department || 'N/A'}</TableCell>
                <TableCell className="text-sm">
                  {user.lastLogin ? new Date(user.lastLogin).toLocaleDateString() : "Never"}
                </TableCell>
                <TableCell>
                  {user.twoFactorEnabled ? (
                    <Badge variant="outline" className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Enabled
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400">
                      <XCircle className="h-3 w-3 mr-1" />
                      Disabled
                    </Badge>
                  )}
                </TableCell>
                <TableCell>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" className="h-8 w-8 p-0">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuLabel>Actions</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem onClick={() => handleUserAction(user.id, "edit")}>
                        <Edit className="mr-2 h-4 w-4" />
                        Edit User
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleUserAction(user.id, "resetPassword")}>
                        <Mail className="mr-2 h-4 w-4" />
                        Reset Password
                      </DropdownMenuItem>
                      {user.status === "pending" && (
                        <DropdownMenuItem onClick={() => handleUserAction(user.id, "approve")}>
                          <CheckCircle className="mr-2 h-4 w-4" />
                          Approve User
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem 
                        className="text-red-600"
                        onClick={() => handleUserAction(user.id, "delete")}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete User
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
              );
            }) : (
              <TableRow>
                <TableCell colSpan={7} className="text-center text-muted-foreground">
                  No users found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>

      {/* Create User Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Add New User</DialogTitle>
            <DialogDescription>
              Create a new user account with appropriate permissions.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name</Label>
                <Input id="firstName" placeholder="John" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name</Label>
                <Input id="lastName" placeholder="Doe" />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <Input id="email" type="email" placeholder="john.doe@company.com" />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="role">Role</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select role" />
                </SelectTrigger>
                <SelectContent>
                  {userRoles.map(role => (
                    <SelectItem key={role.id} value={role.id}>
                      <div className="flex flex-col">
                        <span>{role.name || ''}</span>
                        <span className="text-xs text-muted-foreground">
                          {role.description || ''}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="department">Department</Label>
              <Input id="department" placeholder="Engineering" />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Send Welcome Email</Label>
                <div className="text-sm text-muted-foreground">
                  Email login credentials to user
                </div>
              </div>
              <Switch defaultChecked />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button onClick={() => setShowCreateDialog(false)}>
              Create User
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}