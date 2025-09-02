"use client";

import { useState } from "react";
import { useSystemConfig, useUpdateConfig, useResourceQuotas, useUpdateResourceQuota } from "@/lib/api/hooks/useAdmin";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/use-toast";
import { 
  Settings, 
  Database, 
  Network, 
  Shield, 
  Server,
  HardDrive,
  Cpu,
  MemoryStick,
  Clock,
  Lock,
  Globe,
  Mail,
  Bell,
  Zap,
  Eye,
  EyeOff,
  Save,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Info,
  Search,
  Plus,
  Trash2,
  Edit
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SystemConfiguration, ResourceQuota } from "@/lib/api/types";
import { FadeIn } from "@/lib/animations";
import { useForm } from "react-hook-form";

// Mock configuration data with comprehensive settings
const mockConfigs: SystemConfiguration[] = [
  // Security Settings
  {
    id: "1",
    category: "security",
    key: "session_timeout",
    value: 1800,
    description: "User session timeout in seconds",
    type: "number",
    required: true,
    sensitive: false,
    updated_at: "2024-08-24T10:30:00Z",
    updated_by: "admin@novacron.io"
  },
  {
    id: "2",
    category: "security", 
    key: "password_min_length",
    value: 12,
    description: "Minimum password length requirement",
    type: "number",
    required: true,
    sensitive: false,
    updated_at: "2024-08-24T09:15:00Z",
    updated_by: "admin@novacron.io"
  },
  {
    id: "3",
    category: "security",
    key: "mfa_required",
    value: true,
    description: "Require multi-factor authentication for all users",
    type: "boolean",
    required: true,
    sensitive: false,
    updated_at: "2024-08-23T14:22:00Z",
    updated_by: "admin@novacron.io"
  },
  // System Settings
  {
    id: "4",
    category: "system",
    key: "max_concurrent_vms",
    value: 500,
    description: "Maximum number of concurrent VMs per node",
    type: "number",
    required: true,
    sensitive: false,
    updated_at: "2024-08-22T11:45:00Z",
    updated_by: "admin@novacron.io"
  },
  {
    id: "5",
    category: "system",
    key: "backup_retention_days",
    value: 30,
    description: "Number of days to retain VM backups",
    type: "number",
    required: true,
    sensitive: false,
    updated_at: "2024-08-21T16:30:00Z",
    updated_by: "admin@novacron.io"
  },
  // Network Settings
  {
    id: "6",
    category: "network",
    key: "default_network_cidr",
    value: "192.168.0.0/16",
    description: "Default network CIDR for new VMs",
    type: "string",
    required: true,
    sensitive: false,
    updated_at: "2024-08-20T13:10:00Z",
    updated_by: "admin@novacron.io"
  },
  {
    id: "7",
    category: "network",
    key: "enable_sdn",
    value: true,
    description: "Enable software-defined networking",
    type: "boolean",
    required: false,
    sensitive: false,
    updated_at: "2024-08-19T08:25:00Z",
    updated_by: "admin@novacron.io"
  },
  // Email Settings
  {
    id: "8",
    category: "email",
    key: "smtp_server",
    value: "smtp.novacron.io",
    description: "SMTP server hostname",
    type: "string",
    required: true,
    sensitive: false,
    updated_at: "2024-08-18T12:00:00Z",
    updated_by: "admin@novacron.io"
  },
  {
    id: "9",
    category: "email",
    key: "smtp_password",
    value: "************",
    description: "SMTP server password",
    type: "string",
    required: true,
    sensitive: true,
    updated_at: "2024-08-18T12:00:00Z",
    updated_by: "admin@novacron.io"
  }
];

const mockQuotas: ResourceQuota[] = [
  {
    id: "quota-1",
    user_id: "user-123",
    organization_id: undefined,
    resource_type: "cpu",
    limit: 32,
    used: 18,
    unit: "cores",
    period: "monthly",
    created_at: "2024-08-01T00:00:00Z",
    updated_at: "2024-08-24T10:30:00Z"
  },
  {
    id: "quota-2", 
    user_id: "user-123",
    organization_id: undefined,
    resource_type: "memory",
    limit: 64,
    used: 42,
    unit: "GB",
    period: "monthly",
    created_at: "2024-08-01T00:00:00Z",
    updated_at: "2024-08-24T10:30:00Z"
  },
  {
    id: "quota-3",
    user_id: undefined,
    organization_id: "org-456",
    resource_type: "storage",
    limit: 1000,
    used: 650,
    unit: "GB",
    period: "monthly",
    created_at: "2024-08-01T00:00:00Z",
    updated_at: "2024-08-24T10:30:00Z"
  }
];

const configCategories = [
  { id: "security", name: "Security", icon: <Shield className="h-4 w-4" />, count: 3 },
  { id: "system", name: "System", icon: <Server className="h-4 w-4" />, count: 2 },
  { id: "network", name: "Network", icon: <Network className="h-4 w-4" />, count: 2 },
  { id: "email", name: "Email", icon: <Mail className="h-4 w-4" />, count: 2 },
  { id: "storage", name: "Storage", icon: <HardDrive className="h-4 w-4" />, count: 0 },
];

export default function SystemConfigurationPage() {
  const { toast } = useToast();
  const [selectedCategory, setSelectedCategory] = useState("security");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSensitive, setShowSensitive] = useState(false);
  const [editingConfig, setEditingConfig] = useState<SystemConfiguration | null>(null);
  const [showConfigDialog, setShowConfigDialog] = useState(false);
  const [showQuotaDialog, setShowQuotaDialog] = useState(false);
  const [pendingChanges, setPendingChanges] = useState<Record<string, any>>({});
  
  // const { data: configs, isLoading: configsLoading } = useSystemConfig(selectedCategory);
  // const { data: quotas, isLoading: quotasLoading } = useResourceQuotas();
  const updateConfig = useUpdateConfig();
  const updateQuota = useUpdateResourceQuota();
  
  const { register, handleSubmit, reset, formState: { errors } } = useForm();
  
  // Use mock data for demo
  const configs = mockConfigs.filter(config => 
    config.category === selectedCategory &&
    (config.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
     config.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );
  const quotas = mockQuotas;
  
  const handleConfigUpdate = async (key: string, value: any, category: string) => {
    try {
      await updateConfig.mutateAsync({ key, value, category });
      toast({
        title: "Configuration updated",
        description: `${key} has been updated successfully.`
      });
      // Remove from pending changes
      const newPending = { ...pendingChanges };
      delete newPending[key];
      setPendingChanges(newPending);
    } catch (error) {
      toast({
        title: "Failed to update configuration",
        description: "Please try again later.",
        variant: "destructive"
      });
    }
  };
  
  const handleConfigChange = (key: string, value: any) => {
    setPendingChanges(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  const saveAllChanges = async () => {
    const changes = Object.entries(pendingChanges);
    if (changes.length === 0) {
      toast({
        title: "No changes to save",
        description: "Make some configuration changes first."
      });
      return;
    }
    
    try {
      await Promise.all(
        changes.map(([key, value]) => {
          const config = configs.find(c => c.key === key);
          return updateConfig.mutateAsync({ key, value, category: config?.category });
        })
      );
      
      toast({
        title: "All changes saved",
        description: `${changes.length} configuration(s) updated successfully.`
      });
      setPendingChanges({});
    } catch (error) {
      toast({
        title: "Failed to save changes",
        description: "Some configurations may not have been updated.",
        variant: "destructive"
      });
    }
  };
  
  const discardChanges = () => {
    setPendingChanges({});
    toast({
      title: "Changes discarded",
      description: "All pending changes have been reverted."
    });
  };
  
  const getConfigValue = (config: SystemConfiguration) => {
    return pendingChanges[config.key] !== undefined ? pendingChanges[config.key] : config.value;
  };
  
  const isConfigChanged = (key: string) => {
    return pendingChanges[key] !== undefined;
  };
  
  const renderConfigInput = (config: SystemConfiguration) => {
    const currentValue = getConfigValue(config);
    const isChanged = isConfigChanged(config.key);
    
    switch (config.type) {
      case "boolean":
        return (
          <div className="flex items-center gap-3">
            <Switch
              checked={currentValue}
              onCheckedChange={(value) => handleConfigChange(config.key, value)}
            />
            <span className="text-sm">{currentValue ? "Enabled" : "Disabled"}</span>
            {isChanged && (
              <Badge variant="outline" className="text-xs">
                Changed
              </Badge>
            )}
          </div>
        );
      
      case "number":
        if (config.key.includes("timeout") || config.key.includes("retention")) {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <Input
                  type="number"
                  value={currentValue}
                  onChange={(e) => handleConfigChange(config.key, parseInt(e.target.value))}
                  className={cn("w-32", isChanged && "border-blue-500")}
                  min="0"
                />
                <span className="text-sm text-gray-600">
                  {config.key.includes("timeout") ? "seconds" : "days"}
                </span>
                {isChanged && (
                  <Badge variant="outline" className="text-xs">
                    Changed
                  </Badge>
                )}
              </div>
            </div>
          );
        }
        return (
          <div className="flex items-center gap-3">
            <Input
              type="number"
              value={currentValue}
              onChange={(e) => handleConfigChange(config.key, parseInt(e.target.value))}
              className={cn("w-32", isChanged && "border-blue-500")}
              min="0"
            />
            {isChanged && (
              <Badge variant="outline" className="text-xs">
                Changed
              </Badge>
            )}
          </div>
        );
      
      case "string":
        if (config.sensitive) {
          return (
            <div className="flex items-center gap-3">
              <Input
                type={showSensitive ? "text" : "password"}
                value={showSensitive ? currentValue : "************"}
                onChange={(e) => handleConfigChange(config.key, e.target.value)}
                className={cn("w-64", isChanged && "border-blue-500")}
                readOnly={!showSensitive}
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowSensitive(!showSensitive)}
              >
                {showSensitive ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
              {isChanged && (
                <Badge variant="outline" className="text-xs">
                  Changed
                </Badge>
              )}
            </div>
          );
        }
        return (
          <div className="flex items-center gap-3">
            <Input
              type="text"
              value={currentValue}
              onChange={(e) => handleConfigChange(config.key, e.target.value)}
              className={cn("w-64", isChanged && "border-blue-500")}
            />
            {isChanged && (
              <Badge variant="outline" className="text-xs">
                Changed
              </Badge>
            )}
          </div>
        );
      
      default:
        return (
          <div className="flex items-center gap-3">
            <Input
              type="text"
              value={currentValue}
              onChange={(e) => handleConfigChange(config.key, e.target.value)}
              className={cn("w-64", isChanged && "border-blue-500")}
            />
            {isChanged && (
              <Badge variant="outline" className="text-xs">
                Changed
              </Badge>
            )}
          </div>
        );
    }
  };
  
  const getResourceIcon = (type: string) => {
    switch (type) {
      case "cpu": return <Cpu className="h-4 w-4" />;
      case "memory": return <MemoryStick className="h-4 w-4" />;
      case "storage": return <HardDrive className="h-4 w-4" />;
      case "network": return <Network className="h-4 w-4" />;
      case "vms": return <Server className="h-4 w-4" />;
      default: return <Settings className="h-4 w-4" />;
    }
  };
  
  const getUsageColor = (used: number, limit: number) => {
    const percentage = (used / limit) * 100;
    if (percentage >= 90) return "text-red-600";
    if (percentage >= 75) return "text-yellow-600";
    return "text-green-600";
  };
  
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Settings className="h-8 w-8" />
            System Configuration
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage system settings, resource quotas, and global configurations
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {Object.keys(pendingChanges).length > 0 && (
            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={discardChanges}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Discard Changes
              </Button>
              <Button onClick={saveAllChanges}>
                <Save className="h-4 w-4 mr-2" />
                Save All Changes ({Object.keys(pendingChanges).length})
              </Button>
            </div>
          )}
        </div>
      </div>
      
      {/* Configuration Tabs */}
      <Tabs defaultValue="settings" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="settings">System Settings</TabsTrigger>
          <TabsTrigger value="quotas">Resource Quotas</TabsTrigger>
        </TabsList>
        
        {/* System Settings Tab */}
        <TabsContent value="settings" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Category Sidebar */}
            <FadeIn delay={0.1}>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Categories</CardTitle>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search configs..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 text-sm"
                    />
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="space-y-1">
                    {configCategories.map((category) => (
                      <button
                        key={category.id}
                        onClick={() => setSelectedCategory(category.id)}
                        className={cn(
                          "w-full flex items-center justify-between px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors",
                          selectedCategory === category.id && "bg-blue-50 dark:bg-blue-950/50 border-r-2 border-blue-500"
                        )}
                      >
                        <div className="flex items-center gap-2">
                          {category.icon}
                          <span className="text-sm font-medium">{category.name}</span>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {category.count}
                        </Badge>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            {/* Configuration Panel */}
            <div className="lg:col-span-3 space-y-6">
              <FadeIn delay={0.2}>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {configCategories.find(c => c.id === selectedCategory)?.icon}
                      {configCategories.find(c => c.id === selectedCategory)?.name} Settings
                    </CardTitle>
                    <CardDescription>
                      Configure {selectedCategory} related settings for the system
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {configs.length === 0 ? (
                        <div className="text-center py-8">
                          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                          <h3 className="text-lg font-medium mb-2">No configurations found</h3>
                          <p className="text-gray-600 dark:text-gray-400">
                            {searchQuery 
                              ? "No configurations match your search query."
                              : "This category doesn't have any configurations yet."
                            }
                          </p>
                        </div>
                      ) : (
                        configs.map((config) => (
                          <div key={config.id} className="border rounded-lg p-4 space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-3">
                                  <h4 className="font-medium">{config.key}</h4>
                                  {config.required && (
                                    <Badge variant="outline" className="text-xs">
                                      Required
                                    </Badge>
                                  )}
                                  {config.sensitive && (
                                    <Badge variant="secondary" className="text-xs">
                                      <Lock className="h-3 w-3 mr-1" />
                                      Sensitive
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                  {config.description}
                                </p>
                              </div>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <div className="flex-1">
                                {renderConfigInput(config)}
                              </div>
                              
                              <div className="text-right">
                                <div className="text-xs text-gray-600 dark:text-gray-400">
                                  Updated: {new Date(config.updated_at).toLocaleDateString()}
                                </div>
                                <div className="text-xs text-gray-600 dark:text-gray-400">
                                  By: {config.updated_by}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </FadeIn>
            </div>
          </div>
        </TabsContent>
        
        {/* Resource Quotas Tab */}
        <TabsContent value="quotas" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium">Resource Quotas</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Manage resource limits and usage tracking
              </p>
            </div>
            
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Create Quota
            </Button>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quota Overview */}
            <FadeIn delay={0.3}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Quota Overview
                  </CardTitle>
                  <CardDescription>System-wide resource utilization</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {quotas.map((quota) => (
                    <div key={quota.id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {getResourceIcon(quota.resource_type)}
                          <span className="font-medium capitalize">
                            {quota.resource_type}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className={cn("font-bold", getUsageColor(quota.used, quota.limit))}>
                            {quota.used} / {quota.limit} {quota.unit}
                          </div>
                          <div className="text-xs text-gray-600">
                            {Math.round((quota.used / quota.limit) * 100)}% used
                          </div>
                        </div>
                      </div>
                      <Progress value={(quota.used / quota.limit) * 100} className="h-2" />
                    </div>
                  ))}
                </CardContent>
              </Card>
            </FadeIn>
            
            {/* Quota Management */}
            <FadeIn delay={0.4}>
              <Card>
                <CardHeader>
                  <CardTitle>Active Quotas</CardTitle>
                  <CardDescription>Manage individual resource quotas</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {quotas.map((quota) => (
                      <div key={quota.id} className="border rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            {getResourceIcon(quota.resource_type)}
                            <span className="font-medium capitalize">
                              {quota.resource_type} Quota
                            </span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                              <Edit className="h-3 w-3" />
                            </Button>
                            <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <div className="text-gray-600">Target</div>
                            <div className="font-medium">
                              {quota.user_id ? `User: ${quota.user_id.slice(-6)}` : 
                               quota.organization_id ? `Org: ${quota.organization_id.slice(-6)}` : 
                               'System'}
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-600">Period</div>
                            <div className="font-medium capitalize">{quota.period}</div>
                          </div>
                        </div>
                        
                        <div className="mt-2">
                          <Progress value={(quota.used / quota.limit) * 100} className="h-1" />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
          
          {/* Quota Alerts */}
          <FadeIn delay={0.5}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Quota Alerts
                </CardTitle>
                <CardDescription>Resource usage warnings and recommendations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-yellow-50 dark:bg-yellow-950/50 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                    <div>
                      <div className="font-medium text-yellow-800 dark:text-yellow-200">Storage Quota Warning</div>
                      <div className="text-sm text-yellow-700 dark:text-yellow-300">
                        Organization quota is at 65% capacity (650GB / 1000GB). Consider increasing limit.
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-950/50 rounded-lg border border-blue-200 dark:border-blue-800">
                    <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <div className="font-medium text-blue-800 dark:text-blue-200">Quota Optimization</div>
                      <div className="text-sm text-blue-700 dark:text-blue-300">
                        CPU quota usage is low (18/32 cores). Consider reallocating to memory resources.
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-950/50 rounded-lg border border-green-200 dark:border-green-800">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                    <div>
                      <div className="font-medium text-green-800 dark:text-green-200">Quota Status Normal</div>
                      <div className="text-sm text-green-700 dark:text-green-300">
                        All other resource quotas are operating within normal parameters.
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </FadeIn>
        </TabsContent>
      </Tabs>
    </div>
  );
}