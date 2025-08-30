"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings, 
  Plus, 
  Edit, 
  Trash2, 
  Shield, 
  Zap, 
  Heart, 
  Scale, 
  AlertCircle,
  CheckCircle,
  Clock,
} from 'lucide-react';

interface OrchestrationPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: number;
  rules: PolicyRule[];
  createdAt: string;
  updatedAt: string;
}

interface PolicyRule {
  type: 'placement' | 'autoscaling' | 'healing' | 'loadbalance' | 'security' | 'compliance';
  enabled: boolean;
  priority: number;
  parameters: Record<string, any>;
}

interface PolicyManagementPanelProps {
  policies: OrchestrationPolicy[];
}

const RULE_TYPE_ICONS = {
  placement: Scale,
  autoscaling: Zap,
  healing: Heart,
  loadbalance: Settings,
  security: Shield,
  compliance: CheckCircle,
};

const RULE_TYPE_COLORS = {
  placement: 'bg-blue-100 text-blue-800',
  autoscaling: 'bg-green-100 text-green-800',
  healing: 'bg-red-100 text-red-800',
  loadbalance: 'bg-purple-100 text-purple-800',
  security: 'bg-yellow-100 text-yellow-800',
  compliance: 'bg-gray-100 text-gray-800',
};

export function PolicyManagementPanel({ policies: initialPolicies }: PolicyManagementPanelProps) {
  const [policies, setPolicies] = useState(initialPolicies);
  const [selectedPolicy, setSelectedPolicy] = useState<OrchestrationPolicy | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Filter and search policies
  const filteredPolicies = policies.filter(policy => {
    const matchesFilter = filter === 'all' || 
      (filter === 'enabled' && policy.enabled) ||
      (filter === 'disabled' && !policy.enabled);
    
    const matchesSearch = policy.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      policy.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const togglePolicyEnabled = async (policyId: string) => {
    try {
      const policy = policies.find(p => p.id === policyId);
      if (!policy) return;

      const response = await fetch(`/api/orchestration/policies/${policyId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !policy.enabled }),
      });

      if (response.ok) {
        setPolicies(policies.map(p => 
          p.id === policyId ? { ...p, enabled: !p.enabled } : p
        ));
      }
    } catch (error) {
      console.error('Failed to toggle policy:', error);
    }
  };

  const deletePolicy = async (policyId: string) => {
    if (!confirm('Are you sure you want to delete this policy?')) return;

    try {
      const response = await fetch(`/api/orchestration/policies/${policyId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setPolicies(policies.filter(p => p.id !== policyId));
      }
    } catch (error) {
      console.error('Failed to delete policy:', error);
    }
  };

  const getRuleIcon = (type: string) => {
    const IconComponent = RULE_TYPE_ICONS[type as keyof typeof RULE_TYPE_ICONS] || Settings;
    return <IconComponent className="h-4 w-4" />;
  };

  const getRuleTypeColor = (type: string) => {
    return RULE_TYPE_COLORS[type as keyof typeof RULE_TYPE_COLORS] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Policy Management</span>
              </CardTitle>
              <CardDescription>
                Configure orchestration policies and rules
              </CardDescription>
            </div>
            <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Policy
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Create New Policy</DialogTitle>
                  <DialogDescription>
                    Define a new orchestration policy with custom rules
                  </DialogDescription>
                </DialogHeader>
                <PolicyForm onSave={() => setIsCreateDialogOpen(false)} />
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <Label htmlFor="search">Search Policies</Label>
              <Input
                id="search"
                placeholder="Search by name or description..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="filter">Filter</Label>
              <Select value={filter} onValueChange={setFilter}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="enabled">Enabled</SelectItem>
                  <SelectItem value="disabled">Disabled</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Policies List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredPolicies.map((policy) => (
          <Card key={policy.id} className={`relative ${!policy.enabled ? 'opacity-60' : ''}`}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CardTitle className="text-lg">{policy.name}</CardTitle>
                  <Badge variant={policy.enabled ? 'success' : 'secondary'}>
                    {policy.enabled ? 'Active' : 'Inactive'}
                  </Badge>
                  <Badge variant="outline">Priority {policy.priority}</Badge>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={policy.enabled}
                    onCheckedChange={() => togglePolicyEnabled(policy.id)}
                  />
                  <Dialog open={isEditDialogOpen && selectedPolicy?.id === policy.id} 
                          onOpenChange={(open) => {
                            setIsEditDialogOpen(open);
                            if (!open) setSelectedPolicy(null);
                          }}>
                    <DialogTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSelectedPolicy(policy)}
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl">
                      <DialogHeader>
                        <DialogTitle>Edit Policy</DialogTitle>
                        <DialogDescription>
                          Modify policy settings and rules
                        </DialogDescription>
                      </DialogHeader>
                      {selectedPolicy && (
                        <PolicyForm 
                          policy={selectedPolicy} 
                          onSave={() => {
                            setIsEditDialogOpen(false);
                            setSelectedPolicy(null);
                          }} 
                        />
                      )}
                    </DialogContent>
                  </Dialog>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => deletePolicy(policy.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <CardDescription>{policy.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Rules */}
                <div>
                  <Label className="text-sm font-medium">Rules ({policy.rules.length})</Label>
                  <div className="mt-2 space-y-2">
                    {policy.rules.map((rule, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center space-x-3">
                          {getRuleIcon(rule.type)}
                          <div>
                            <div className="flex items-center space-x-2">
                              <span className="font-medium capitalize">{rule.type}</span>
                              <Badge 
                                variant="outline" 
                                className={getRuleTypeColor(rule.type)}
                              >
                                {rule.type}
                              </Badge>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Priority: {rule.priority}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant={rule.enabled ? 'success' : 'secondary'}>
                            {rule.enabled ? <CheckCircle className="h-3 w-3" /> : <Clock className="h-3 w-3" />}
                          </Badge>
                        </div>
                      </div>
                    ))}
                    
                    {policy.rules.length === 0 && (
                      <div className="text-center text-muted-foreground py-4">
                        No rules configured
                      </div>
                    )}
                  </div>
                </div>

                {/* Metadata */}
                <div className="text-xs text-muted-foreground border-t pt-3">
                  <div className="flex justify-between">
                    <span>Created: {new Date(policy.createdAt).toLocaleDateString()}</span>
                    <span>Updated: {new Date(policy.updatedAt).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredPolicies.length === 0 && (
        <Card>
          <CardContent className="text-center py-8">
            <AlertCircle className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">
              {searchTerm || filter !== 'all' 
                ? 'No policies match your search criteria'
                : 'No policies configured yet'
              }
            </p>
            {!searchTerm && filter === 'all' && (
              <Button
                className="mt-4"
                onClick={() => setIsCreateDialogOpen(true)}
              >
                Create Your First Policy
              </Button>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Policy Form Component
interface PolicyFormProps {
  policy?: OrchestrationPolicy;
  onSave: () => void;
}

function PolicyForm({ policy, onSave }: PolicyFormProps) {
  const [formData, setFormData] = useState({
    name: policy?.name || '',
    description: policy?.description || '',
    priority: policy?.priority || 5,
    enabled: policy?.enabled ?? true,
  });
  const [rules, setRules] = useState<PolicyRule[]>(policy?.rules || []);
  const [newRuleType, setNewRuleType] = useState<string>('');

  const addRule = () => {
    if (!newRuleType) return;
    
    const newRule: PolicyRule = {
      type: newRuleType as PolicyRule['type'],
      enabled: true,
      priority: 5,
      parameters: {},
    };
    
    setRules([...rules, newRule]);
    setNewRuleType('');
  };

  const removeRule = (index: number) => {
    setRules(rules.filter((_, i) => i !== index));
  };

  const updateRule = (index: number, updates: Partial<PolicyRule>) => {
    setRules(rules.map((rule, i) => i === index ? { ...rule, ...updates } : rule));
  };

  const handleSave = async () => {
    try {
      const policyData = {
        ...formData,
        rules,
        updatedAt: new Date().toISOString(),
        ...(policy ? {} : { 
          id: `policy-${Date.now()}`,
          createdAt: new Date().toISOString() 
        }),
      };

      const url = policy 
        ? `/api/orchestration/policies/${policy.id}`
        : '/api/orchestration/policies';
      
      const method = policy ? 'PUT' : 'POST';

      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(policyData),
      });

      if (response.ok) {
        onSave();
      }
    } catch (error) {
      console.error('Failed to save policy:', error);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="name">Policy Name</Label>
          <Input
            id="name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            placeholder="Enter policy name"
          />
        </div>
        <div>
          <Label htmlFor="priority">Priority</Label>
          <Input
            id="priority"
            type="number"
            min="1"
            max="10"
            value={formData.priority}
            onChange={(e) => setFormData({ ...formData, priority: parseInt(e.target.value) })}
          />
        </div>
      </div>

      <div>
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={formData.description}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          placeholder="Describe what this policy does"
          rows={3}
        />
      </div>

      <div className="flex items-center space-x-2">
        <Switch
          id="enabled"
          checked={formData.enabled}
          onCheckedChange={(enabled) => setFormData({ ...formData, enabled })}
        />
        <Label htmlFor="enabled">Enable policy</Label>
      </div>

      {/* Rules Section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <Label>Rules</Label>
          <div className="flex items-center space-x-2">
            <Select value={newRuleType} onValueChange={setNewRuleType}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select rule type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="placement">Placement</SelectItem>
                <SelectItem value="autoscaling">Auto Scaling</SelectItem>
                <SelectItem value="healing">Self Healing</SelectItem>
                <SelectItem value="loadbalance">Load Balance</SelectItem>
                <SelectItem value="security">Security</SelectItem>
                <SelectItem value="compliance">Compliance</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={addRule} disabled={!newRuleType} size="sm">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="space-y-2 max-h-48 overflow-y-auto">
          {rules.map((rule, index) => (
            <div key={index} className="flex items-center justify-between p-3 border rounded">
              <div className="flex items-center space-x-3">
                {getRuleIcon(rule.type)}
                <div>
                  <div className="font-medium capitalize">{rule.type}</div>
                  <div className="text-sm text-muted-foreground">
                    Priority: {rule.priority}
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={rule.enabled}
                  onCheckedChange={(enabled) => updateRule(index, { enabled })}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => removeRule(index)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>

        {rules.length === 0 && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Add at least one rule to define policy behavior
            </AlertDescription>
          </Alert>
        )}
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={onSave}>
          Cancel
        </Button>
        <Button onClick={handleSave} disabled={!formData.name || rules.length === 0}>
          {policy ? 'Update Policy' : 'Create Policy'}
        </Button>
      </DialogFooter>
    </div>
  );
}

// Helper function moved inside the component file
function getRuleIcon(type: string) {
  const RULE_TYPE_ICONS = {
    placement: Scale,
    autoscaling: Zap,
    healing: Heart,
    loadbalance: Settings,
    security: Shield,
    compliance: CheckCircle,
  };
  
  const IconComponent = RULE_TYPE_ICONS[type as keyof typeof RULE_TYPE_ICONS] || Settings;
  return <IconComponent className="h-4 w-4" />;
}