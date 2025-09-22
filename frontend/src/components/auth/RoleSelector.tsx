"use client";

import { useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useRBAC, Role } from '@/contexts/RBACContext';
import { Shield, Users, Eye, Settings } from 'lucide-react';

interface RoleSelectorProps {
  selectedRoles: string[];
  onRoleChange: (roles: string[]) => void;
  disabled?: boolean;
  showDescription?: boolean;
}

export default function RoleSelector({
  selectedRoles,
  onRoleChange,
  disabled = false,
  showDescription = true
}: RoleSelectorProps) {
  const { userRoles } = useRBAC();

  // Available roles - in production, this might be fetched from an API
  const availableRoles: Role[] = [
    {
      id: 'super-admin',
      name: 'Super Administrator',
      description: 'Full system access with all permissions',
      permissions: []
    },
    {
      id: 'admin',
      name: 'Administrator',
      description: 'System administration with most permissions',
      permissions: []
    },
    {
      id: 'operator',
      name: 'Operator',
      description: 'VM operations and monitoring access',
      permissions: []
    },
    {
      id: 'viewer',
      name: 'Viewer',
      description: 'Read-only access to system resources',
      permissions: []
    }
  ];

  const getRoleIcon = (roleId: string) => {
    switch (roleId) {
      case 'super-admin':
        return <Shield className="h-4 w-4" />;
      case 'admin':
        return <Settings className="h-4 w-4" />;
      case 'operator':
        return <Users className="h-4 w-4" />;
      case 'viewer':
        return <Eye className="h-4 w-4" />;
      default:
        return <Users className="h-4 w-4" />;
    }
  };

  const handleRoleToggle = (roleId: string) => {
    if (selectedRoles.includes(roleId)) {
      onRoleChange(selectedRoles.filter(id => id !== roleId));
    } else {
      onRoleChange([...selectedRoles, roleId]);
    }
  };

  const selectedRole = availableRoles.find(role => role.id === selectedRoles[0]);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Role Assignment</Label>
        <Select
          value={selectedRoles[0] || ''}
          onValueChange={(value) => onRoleChange([value])}
          disabled={disabled}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select a role" />
          </SelectTrigger>
          <SelectContent>
            {availableRoles.map((role) => (
              <SelectItem key={role.id} value={role.id}>
                <div className="flex items-center gap-2">
                  {getRoleIcon(role.id)}
                  <span>{role.name}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selectedRole && showDescription && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              {getRoleIcon(selectedRole.id)}
              {selectedRole.name}
            </CardTitle>
            <CardDescription>{selectedRole.description}</CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="space-y-2">
              <Label className="text-sm font-medium">Key Capabilities:</Label>
              <div className="flex flex-wrap gap-2">
                {selectedRole.id === 'super-admin' && (
                  <>
                    <Badge variant="default">Full System Access</Badge>
                    <Badge variant="default">User Management</Badge>
                    <Badge variant="default">Security Configuration</Badge>
                  </>
                )}
                {selectedRole.id === 'admin' && (
                  <>
                    <Badge variant="default">VM Management</Badge>
                    <Badge variant="default">User Management</Badge>
                    <Badge variant="default">System Configuration</Badge>
                    <Badge variant="default">Backup Management</Badge>
                  </>
                )}
                {selectedRole.id === 'operator' && (
                  <>
                    <Badge variant="secondary">VM Operations</Badge>
                    <Badge variant="secondary">Monitoring Access</Badge>
                    <Badge variant="secondary">Backup Creation</Badge>
                  </>
                )}
                {selectedRole.id === 'viewer' && (
                  <>
                    <Badge variant="outline">Read-Only Access</Badge>
                    <Badge variant="outline">Monitoring View</Badge>
                    <Badge variant="outline">Backup Listing</Badge>
                  </>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {selectedRoles.length > 0 && (
        <div className="space-y-2">
          <Label className="text-sm font-medium">Selected Roles:</Label>
          <div className="flex flex-wrap gap-2">
            {selectedRoles.map((roleId) => {
              const role = availableRoles.find(r => r.id === roleId);
              return role ? (
                <Badge key={roleId} variant="default">
                  {getRoleIcon(roleId)}
                  <span className="ml-1">{role.name}</span>
                </Badge>
              ) : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
}