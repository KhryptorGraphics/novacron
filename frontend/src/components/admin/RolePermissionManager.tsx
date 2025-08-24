"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
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
import { Textarea } from "@/components/ui/textarea";
import { FadeIn } from "@/lib/animations";
import { 
  UserCheck, 
  Shield, 
  Users, 
  Lock, 
  Plus, 
  Edit, 
  Trash2, 
  Save, 
  X,
  Eye,
  Settings,
  Database,
  Server,
  Network,
  FileText,
  Mail,
  BarChart3,
  AlertTriangle
} from "lucide-react";
import { cn } from "@/lib/utils";

// Permission categories and permissions
const permissionCategories = [
  {
    id: "user_management",
    name: "User Management",
    icon: <Users className="h-4 w-4" />,
    permissions: [
      { id: "users.view", name: "View Users", description: "View user profiles and information" },
      { id: "users.create", name: "Create Users", description: "Add new user accounts" },
      { id: "users.edit", name: "Edit Users", description: "Modify user profiles and settings" },
      { id: "users.delete", name: "Delete Users", description: "Remove user accounts" },
      { id: "users.suspend", name: "Suspend Users", description: "Suspend or activate user accounts" },
      { id: "users.roles", name: "Manage User Roles", description: "Assign and modify user roles" }
    ]
  },
  {
    id: "vm_management",
    name: "VM Management",
    icon: <Server className="h-4 w-4" />,
    permissions: [
      { id: "vms.view", name: "View VMs", description: "View virtual machine information" },
      { id: "vms.create", name: "Create VMs", description: "Create new virtual machines" },
      { id: "vms.edit", name: "Edit VMs", description: "Modify VM configurations" },
      { id: "vms.delete", name: "Delete VMs", description: "Remove virtual machines" },
      { id: "vms.migrate", name: "Migrate VMs", description: "Perform VM migrations" },
      { id: "vms.snapshots", name: "Manage Snapshots", description: "Create and manage VM snapshots" }
    ]
  },
  {
    id: "system_admin",
    name: "System Administration",
    icon: <Settings className="h-4 w-4" />,
    permissions: [
      { id: "system.config", name: "System Configuration", description: "Modify system settings" },
      { id: "system.logs", name: "View System Logs", description: "Access system and audit logs" },
      { id: "system.backup", name: "Backup Management", description: "Manage system backups" },
      { id: "system.maintenance", name: "Maintenance Mode", description: "Enable/disable maintenance mode" },
      { id: "system.updates", name: "System Updates", description: "Apply system updates and patches" }
    ]
  },
  {
    id: "database",
    name: "Database Access",
    icon: <Database className="h-4 w-4" />,
    permissions: [
      { id: "db.view", name: "View Database", description: "Read database tables and records" },
      { id: "db.edit", name: "Edit Database", description: "Modify database records" },
      { id: "db.schema", name: "Schema Management", description: "Modify database structure" },
      { id: "db.backup", name: "Database Backup", description: "Create and restore database backups" }
    ]
  },
  {
    id: "reporting",
    name: "Reports & Analytics",
    icon: <BarChart3 className="h-4 w-4" />,
    permissions: [
      { id: "reports.view", name: "View Reports", description: "Access system reports and analytics" },
      { id: "reports.create", name: "Create Reports", description: "Generate custom reports" },
      { id: "reports.export", name: "Export Data", description: "Export data and reports" },
      { id: "reports.schedule", name: "Schedule Reports", description: "Schedule automated reports" }
    ]
  }
];

// Predefined roles
const defaultRoles = [
  {
    id: 1,
    name: "Super Admin",
    description: "Full system access with all permissions",
    color: "red",
    builtin: true,
    users: 2,
    permissions: permissionCategories.flatMap(cat => cat.permissions.map(p => p.id))
  },
  {
    id: 2,
    name: "Administrator",
    description: "System administration with limited database access",
    color: "orange",
    builtin: true,
    users: 5,
    permissions: [
      "users.view", "users.create", "users.edit", "users.suspend", "users.roles",
      "vms.view", "vms.create", "vms.edit", "vms.migrate", "vms.snapshots",
      "system.config", "system.logs", "system.backup",
      "reports.view", "reports.create", "reports.export"
    ]
  },
  {
    id: 3,
    name: "VM Manager",
    description: "Virtual machine management and operations",
    color: "blue",
    builtin: true,
    users: 12,
    permissions: [
      "users.view",
      "vms.view", "vms.create", "vms.edit", "vms.migrate", "vms.snapshots",
      "reports.view", "reports.create"
    ]
  },
  {
    id: 4,
    name: "Read Only",
    description: "View-only access to system resources",
    color: "green",
    builtin: true,
    users: 45,
    permissions: [
      "users.view", "vms.view", "reports.view"
    ]
  }
];

interface Role {
  id: number;
  name: string;
  description: string;
  color: string;
  builtin: boolean;
  users: number;
  permissions: string[];
}

export function RolePermissionManager() {
  const [roles, setRoles] = useState<Role[]>(defaultRoles);
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [editingRole, setEditingRole] = useState<Role | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [activeTab, setActiveTab] = useState("roles");

  const handleCreateRole = () => {
    const newRole: Role = {
      id: Date.now(),
      name: "New Role",
      description: "",
      color: "gray",
      builtin: false,
      users: 0,
      permissions: []
    };
    setEditingRole(newRole);
    setIsCreating(true);
  };

  const handleEditRole = (role: Role) => {
    if (role.builtin) return;
    setEditingRole({ ...role });
    setIsCreating(false);
  };

  const handleSaveRole = () => {
    if (!editingRole) return;
    
    if (isCreating) {
      setRoles(prev => [...prev, editingRole]);
    } else {
      setRoles(prev => prev.map(role => 
        role.id === editingRole.id ? editingRole : role
      ));
    }
    
    setEditingRole(null);
    setIsCreating(false);
  };

  const handleDeleteRole = (roleId: number) => {
    const role = roles.find(r => r.id === roleId);
    if (!role || role.builtin) return;
    
    if (confirm(`Are you sure you want to delete the role "${role.name}"?`)) {
      setRoles(prev => prev.filter(r => r.id !== roleId));
      if (selectedRole?.id === roleId) {
        setSelectedRole(null);
      }
    }
  };

  const handlePermissionChange = (permissionId: string, checked: boolean) => {
    if (!editingRole) return;
    
    const updatedPermissions = checked
      ? [...editingRole.permissions, permissionId]
      : editingRole.permissions.filter(p => p !== permissionId);
      
    setEditingRole({
      ...editingRole,
      permissions: updatedPermissions
    });
  };

  const getColorClasses = (color: string) => {
    switch (color) {
      case "red": return "bg-red-100 text-red-800 border-red-300";
      case "orange": return "bg-orange-100 text-orange-800 border-orange-300";
      case "blue": return "bg-blue-100 text-blue-800 border-blue-300";
      case "green": return "bg-green-100 text-green-800 border-green-300";
      case "purple": return "bg-purple-100 text-purple-800 border-purple-300";
      case "gray": return "bg-gray-100 text-gray-800 border-gray-300";
      default: return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const getRolePermissionCount = (role: Role) => {
    return role.permissions.length;
  };

  const getTotalPermissions = () => {
    return permissionCategories.reduce((total, cat) => total + cat.permissions.length, 0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <UserCheck className="h-6 w-6" />
            Roles & Permissions
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage user roles and access permissions
          </p>
        </div>
        
        <Button onClick={handleCreateRole} disabled={!!editingRole}>
          <Plus className="h-4 w-4 mr-2" />
          Create Role
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Roles List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Roles ({roles.length})</CardTitle>
              <CardDescription>System roles and their permissions</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-2">
                {roles.map((role) => (
                  <div
                    key={role.id}
                    onClick={() => setSelectedRole(role)}
                    className={cn(
                      "p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors border-l-4",
                      selectedRole?.id === role.id && "bg-blue-50 dark:bg-blue-950",
                      role.color === "red" && "border-red-500",
                      role.color === "orange" && "border-orange-500",
                      role.color === "blue" && "border-blue-500",
                      role.color === "green" && "border-green-500",
                      role.color === "purple" && "border-purple-500",
                      role.color === "gray" && "border-gray-500"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium">{role.name}</h4>
                          {role.builtin && (
                            <Badge variant="outline" className="text-xs">
                              <Lock className="h-3 w-3 mr-1" />
                              Built-in
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {role.description}
                        </p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>{role.users} users</span>
                          <span>{getRolePermissionCount(role)} permissions</span>
                        </div>
                      </div>
                      
                      {!role.builtin && (
                        <div className="flex items-center gap-1 ml-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditRole(role);
                            }}
                            className="h-7 w-7 p-0"
                          >
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteRole(role.id);
                            }}
                            className="h-7 w-7 p-0 text-red-600"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Role Details / Editor */}
        <div className="lg:col-span-2">
          {editingRole ? (
            <FadeIn>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>
                      {isCreating ? "Create New Role" : "Edit Role"}
                    </CardTitle>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setEditingRole(null);
                          setIsCreating(false);
                        }}
                      >
                        <X className="h-4 w-4 mr-2" />
                        Cancel
                      </Button>
                      <Button onClick={handleSaveRole}>
                        <Save className="h-4 w-4 mr-2" />
                        Save
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Role Basic Info */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Role Name</label>
                      <Input
                        value={editingRole.name}
                        onChange={(e) => setEditingRole({
                          ...editingRole,
                          name: e.target.value
                        })}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Color</label>
                      <Select
                        value={editingRole.color}
                        onValueChange={(value) => setEditingRole({
                          ...editingRole,
                          color: value
                        })}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="red">Red</SelectItem>
                          <SelectItem value="orange">Orange</SelectItem>
                          <SelectItem value="blue">Blue</SelectItem>
                          <SelectItem value="green">Green</SelectItem>
                          <SelectItem value="purple">Purple</SelectItem>
                          <SelectItem value="gray">Gray</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Description</label>
                    <Textarea
                      value={editingRole.description}
                      onChange={(e) => setEditingRole({
                        ...editingRole,
                        description: e.target.value
                      })}
                      className="mt-1"
                      rows={2}
                    />
                  </div>

                  {/* Permissions */}
                  <div>
                    <h4 className="font-medium mb-4">Permissions</h4>
                    <div className="space-y-6">
                      {permissionCategories.map((category) => (
                        <div key={category.id} className="border rounded-lg p-4">
                          <h5 className="font-medium flex items-center gap-2 mb-3">
                            {category.icon}
                            {category.name}
                          </h5>
                          <div className="space-y-3">
                            {category.permissions.map((permission) => (
                              <div key={permission.id} className="flex items-start gap-3">
                                <Checkbox
                                  checked={editingRole.permissions.includes(permission.id)}
                                  onCheckedChange={(checked) => 
                                    handlePermissionChange(permission.id, checked as boolean)
                                  }
                                  className="mt-0.5"
                                />
                                <div className="flex-1">
                                  <div className="font-medium text-sm">{permission.name}</div>
                                  <div className="text-xs text-gray-600 dark:text-gray-400">
                                    {permission.description}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          ) : selectedRole ? (
            <FadeIn>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {selectedRole.name}
                        {selectedRole.builtin && (
                          <Badge variant="outline">
                            <Lock className="h-3 w-3 mr-1" />
                            Built-in
                          </Badge>
                        )}
                      </CardTitle>
                      <CardDescription>{selectedRole.description}</CardDescription>
                    </div>
                    {!selectedRole.builtin && (
                      <Button
                        variant="outline"
                        onClick={() => handleEditRole(selectedRole)}
                      >
                        <Edit className="h-4 w-4 mr-2" />
                        Edit Role
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="text-center">
                      <div className="text-2xl font-bold">{selectedRole.users}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Assigned Users</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {getRolePermissionCount(selectedRole)}/{getTotalPermissions()}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Permissions</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-4">Assigned Permissions</h4>
                    <div className="space-y-4">
                      {permissionCategories.map((category) => {
                        const categoryPermissions = category.permissions.filter(p => 
                          selectedRole.permissions.includes(p.id)
                        );
                        
                        if (categoryPermissions.length === 0) return null;
                        
                        return (
                          <div key={category.id} className="border rounded-lg p-4">
                            <h5 className="font-medium flex items-center gap-2 mb-3">
                              {category.icon}
                              {category.name}
                              <Badge variant="outline">
                                {categoryPermissions.length}/{category.permissions.length}
                              </Badge>
                            </h5>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              {categoryPermissions.map((permission) => (
                                <div key={permission.id} className="flex items-center gap-2 text-sm">
                                  <div className="h-2 w-2 bg-green-500 rounded-full" />
                                  {permission.name}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          ) : (
            <Card>
              <CardContent className="p-12 text-center">
                <Shield className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium mb-2">Select a Role</h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Choose a role from the list to view its details and permissions
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}