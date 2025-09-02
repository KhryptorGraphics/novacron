"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState } from "react";
import { useUsers, useBulkUserOperation, useCreateUser, useUpdateUser } from "@/lib/api/hooks/useAdmin";
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
import { Checkbox } from "@/components/ui/checkbox";
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/use-toast";
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
  Trash2,
  MoreHorizontal,
  Calendar,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Filter,
  Download,
  Upload,
  Settings,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import { cn } from "@/lib/utils";
import { User } from "@/lib/api/types";
import { FadeIn } from "@/lib/animations";
import { useForm } from "react-hook-form";

interface UserFilters {
  status?: string;
  role?: string;
  search?: string;
  page: number;
  pageSize: number;
}

export default function UserManagementPage() {
  const { toast } = useToast();
  const [filters, setFilters] = useState<UserFilters>({
    status: '',
    role: '',
    search: '',
    page: 1,
    pageSize: 20
  });
  
  const [selectedUsers, setSelectedUsers] = useState<string[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  
  const { data: usersData, isLoading, error } = useUsers({
    status: filters.status || undefined,
    role: filters.role || undefined,
    search: filters.search || undefined,
    page: filters.page,
    pageSize: filters.pageSize
  });
  
  const bulkOperation = useBulkUserOperation();
  const createUser = useCreateUser();
  const updateUser = useUpdateUser();
  
  const { register, handleSubmit, reset, formState: { errors } } = useForm<Partial<User>>();
  
  const users = usersData?.users || [];
  const totalPages = usersData?.totalPages || 1;
  
  const handleFilterChange = (key: keyof UserFilters, value: string | number) => {
    setFilters(prev => ({
      ...prev,
      [key]: value,
      page: key !== 'page' ? 1 : value // Reset page when other filters change
    }));
  };
  
  const handleSelectUser = (userId: string, selected: boolean) => {
    setSelectedUsers(prev => 
      selected 
        ? [...prev, userId]
        : prev.filter(id => id !== userId)
    );
  };
  
  const handleSelectAll = (selected: boolean) => {
    setSelectedUsers(selected ? users.map(user => user.id) : []);
  };
  
  const handleBulkOperation = async (operation: string) => {
    if (selectedUsers.length === 0) {
      toast({
        title: "No users selected",
        description: "Please select users to perform bulk operations.",
        variant: "destructive"
      });
      return;
    }
    
    try {
      await bulkOperation.mutateAsync({
        userIds: selectedUsers,
        operation: operation as any
      });
      
      toast({
        title: "Bulk operation completed",
        description: `Successfully performed ${operation} on ${selectedUsers.length} users.`
      });
      
      setSelectedUsers([]);
    } catch (error) {
      toast({
        title: "Bulk operation failed",
        description: "Please try again or contact support.",
        variant: "destructive"
      });
    }
  };
  
  const handleCreateUser = async (data: Partial<User>) => {
    try {
      await createUser.mutateAsync(data);
      toast({
        title: "User created successfully",
        description: `${data.name} has been added to the system.`
      });
      setShowCreateDialog(false);
      reset();
    } catch (error) {
      toast({
        title: "Failed to create user",
        description: "Please check the form and try again.",
        variant: "destructive"
      });
    }
  };
  
  const handleUpdateUser = async (data: Partial<User>) => {
    if (!editingUser) return;
    
    try {
      await updateUser.mutateAsync({
        id: editingUser.id,
        ...data
      });
      toast({
        title: "User updated successfully",
        description: `${data.name || editingUser.name}'s information has been updated.`
      });
      setEditingUser(null);
      reset();
    } catch (error) {
      toast({
        title: "Failed to update user",
        description: "Please check the form and try again.",
        variant: "destructive"
      });
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-green-500";
      case "suspended": return "bg-red-500";
      case "pending": return "bg-yellow-500";
      case "disabled": return "bg-gray-500";
      default: return "bg-gray-500";
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active": return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "suspended": return <Ban className="h-4 w-4 text-red-600" />;
      case "pending": return <Clock className="h-4 w-4 text-yellow-600" />;
      case "disabled": return <AlertCircle className="h-4 w-4 text-gray-600" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-600" />;
    }
  };
  
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">Failed to load users</h3>
          <p className="text-gray-600 dark:text-gray-400">Please try refreshing the page.</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Users className="h-8 w-8" />
            User Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage user accounts, roles, and permissions across the system
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
          <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
            <DialogTrigger asChild>
              <Button>
                <UserPlus className="h-4 w-4 mr-2" />
                Add User
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Create New User</DialogTitle>
                <DialogDescription>
                  Add a new user to the system with specified role and permissions.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleSubmit(handleCreateUser)} className="space-y-4">
                <div>
                  <Label htmlFor="name">Full Name</Label>
                  <Input 
                    id="name"
                    {...register("name", { required: "Name is required" })}
                    placeholder="John Doe"
                  />
                  {errors.name && (
                    <p className="text-sm text-red-500 mt-1">{errors.name.message}</p>
                  )}
                </div>
                
                <div>
                  <Label htmlFor="email">Email Address</Label>
                  <Input 
                    id="email"
                    type="email"
                    {...register("email", { 
                      required: "Email is required",
                      pattern: {
                        value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                        message: "Invalid email address"
                      }
                    })}
                    placeholder="john@example.com"
                  />
                  {errors.email && (
                    <p className="text-sm text-red-500 mt-1">{errors.email.message}</p>
                  )}
                </div>
                
                <div>
                  <Label htmlFor="role">Role</Label>
                  <Select onValueChange={(value) => register("role").onChange({ target: { value } })}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="viewer">Viewer</SelectItem>
                      <SelectItem value="user">User</SelectItem>
                      <SelectItem value="moderator">Moderator</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="organization">Organization (Optional)</Label>
                  <Input 
                    id="organization"
                    {...register("organization")}
                    placeholder="ACME Corporation"
                  />
                </div>
                
                <div className="flex justify-end gap-2 pt-4">
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={() => {
                      setShowCreateDialog(false);
                      reset();
                    }}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createUser.isPending}>
                    {createUser.isPending ? "Creating..." : "Create User"}
                  </Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Users</p>
                  <p className="text-2xl font-bold">{usersData?.total?.toLocaleString() || 0}</p>
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
                  <p className="text-2xl font-bold text-green-600">
                    {users.filter(u => u.status === 'active').length}
                  </p>
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
                  <p className="text-2xl font-bold text-yellow-600">
                    {users.filter(u => u.status === 'pending').length}
                  </p>
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
                  <p className="text-2xl font-bold text-purple-600">
                    {users.filter(u => u.two_factor_enabled).length}
                  </p>
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
                placeholder="Search by name, email, or organization..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={filters.status} onValueChange={(value) => handleFilterChange('status', value)}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="suspended">Suspended</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="disabled">Disabled</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={filters.role} onValueChange={(value) => handleFilterChange('role', value)}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Roles" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Roles</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="moderator">Moderator</SelectItem>
                <SelectItem value="user">User</SelectItem>
                <SelectItem value="viewer">Viewer</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
      
      {/* Bulk Actions */}
      {selectedUsers.length > 0 && (
        <FadeIn>
          <Card className="border-blue-200 bg-blue-50 dark:bg-blue-950/50 dark:border-blue-800">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-sm font-medium">
                    {selectedUsers.length} user{selectedUsers.length !== 1 ? 's' : ''} selected
                  </span>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => setSelectedUsers([])}
                  >
                    Clear selection
                  </Button>
                </div>
                
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleBulkOperation('send_email')}
                  >
                    <Mail className="h-4 w-4 mr-1" />
                    Email
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleBulkOperation('activate')}
                  >
                    <CheckCircle className="h-4 w-4 mr-1" />
                    Activate
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleBulkOperation('suspend')}
                  >
                    <Ban className="h-4 w-4 mr-1" />
                    Suspend
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      )}
      
      {/* Users Table */}
      <FadeIn delay={0.5}>
        <Card>
          <CardHeader>
            <CardTitle>Users ({users.length})</CardTitle>
            <CardDescription>
              Manage user accounts and their permissions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <Checkbox
                            checked={selectedUsers.length === users.length && users.length > 0}
                            onCheckedChange={handleSelectAll}
                            aria-label="Select all users"
                          />
                        </TableHead>
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
                      {users.map((user) => (
                        <TableRow key={user.id}>
                          <TableCell>
                            <Checkbox
                              checked={selectedUsers.includes(user.id)}
                              onCheckedChange={(checked) => handleSelectUser(user.id, checked as boolean)}
                              aria-label={`Select ${user.name}`}
                            />
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              {user.avatar_url ? (
                                <img 
                                  src={user.avatar_url} 
                                  alt={user.name}
                                  className="h-10 w-10 rounded-full object-cover"
                                />
                              ) : (
                                <div className="h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                                  <span className="text-sm font-medium">
                                    {user.name.split(' ').map(n => n[0]).join('')}
                                  </span>
                                </div>
                              )}
                              <div>
                                <div className="font-medium">{user.name}</div>
                                <div className="text-sm text-gray-600 dark:text-gray-400">{user.email}</div>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>{user.organization || '-'}</TableCell>
                          
                          <TableCell>
                            <Badge 
                              variant={
                                user.role === 'admin' ? 'destructive' : 
                                user.role === 'moderator' ? 'secondary' : 
                                'outline'
                              }
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
                              {user.last_login ? formatDate(user.last_login) : 'Never'}
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
                                <span className={user.two_factor_enabled ? "text-green-600" : "text-gray-600"}>
                                  {user.two_factor_enabled ? "2FA On" : "2FA Off"}
                                </span>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => {
                                  setEditingUser(user);
                                  reset(user);
                                }}
                                className="h-8 w-8 p-0"
                              >
                                <Edit className="h-4 w-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-8 w-8 p-0"
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-8 w-8 p-0"
                              >
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                
                {/* Pagination */}
                <div className="flex items-center justify-between mt-6">
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Showing {((filters.page - 1) * filters.pageSize) + 1} to{' '}
                    {Math.min(filters.page * filters.pageSize, usersData?.total || 0)} of{' '}
                    {usersData?.total || 0} users
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleFilterChange('page', Math.max(1, filters.page - 1))}
                      disabled={filters.page <= 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                      Previous
                    </Button>
                    
                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        const page = i + Math.max(1, filters.page - 2);
                        return page <= totalPages ? (
                          <Button
                            key={page}
                            variant={page === filters.page ? "default" : "outline"}
                            size="sm"
                            onClick={() => handleFilterChange('page', page)}
                            className="h-8 w-8 p-0"
                          >
                            {page}
                          </Button>
                        ) : null;
                      })}
                    </div>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleFilterChange('page', Math.min(totalPages, filters.page + 1))}
                      disabled={filters.page >= totalPages}
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </FadeIn>
      
      {/* Edit User Dialog */}
      <Dialog open={!!editingUser} onOpenChange={(open) => !open && setEditingUser(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Edit User</DialogTitle>
            <DialogDescription>
              Update user information and permissions.
            </DialogDescription>
          </DialogHeader>
          {editingUser && (
            <form onSubmit={handleSubmit(handleUpdateUser)} className="space-y-4">
              <div>
                <Label htmlFor="edit-name">Full Name</Label>
                <Input 
                  id="edit-name"
                  {...register("name", { required: "Name is required" })}
                  defaultValue={editingUser.name}
                />
                {errors.name && (
                  <p className="text-sm text-red-500 mt-1">{errors.name.message}</p>
                )}
              </div>
              
              <div>
                <Label htmlFor="edit-email">Email Address</Label>
                <Input 
                  id="edit-email"
                  type="email"
                  {...register("email", { 
                    required: "Email is required",
                    pattern: {
                      value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                      message: "Invalid email address"
                    }
                  })}
                  defaultValue={editingUser.email}
                />
                {errors.email && (
                  <p className="text-sm text-red-500 mt-1">{errors.email.message}</p>
                )}
              </div>
              
              <div>
                <Label htmlFor="edit-role">Role</Label>
                <Select 
                  defaultValue={editingUser.role}
                  onValueChange={(value) => register("role").onChange({ target: { value } })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="viewer">Viewer</SelectItem>
                    <SelectItem value="user">User</SelectItem>
                    <SelectItem value="moderator">Moderator</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="edit-organization">Organization</Label>
                <Input 
                  id="edit-organization"
                  {...register("organization")}
                  defaultValue={editingUser.organization || ''}
                />
              </div>
              
              <div className="flex justify-end gap-2 pt-4">
                <Button 
                  type="button" 
                  variant="outline" 
                  onClick={() => {
                    setEditingUser(null);
                    reset();
                  }}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={updateUser.isPending}>
                  {updateUser.isPending ? "Updating..." : "Update User"}
                </Button>
              </div>
            </form>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}