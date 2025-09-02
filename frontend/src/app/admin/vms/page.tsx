"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState } from "react";
import { useVmTemplates, useCreateVmTemplate, useDeleteVmTemplate } from "@/lib/api/hooks/useAdmin";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
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
  Server, 
  Plus, 
  Search, 
  Settings, 
  Play,
  Pause,
  Square,
  RotateCcw,
  Trash2,
  Copy,
  Edit,
  Eye,
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  Monitor,
  Activity,
  Zap,
  Download,
  Upload,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  Clock
} from "lucide-react";
import { cn } from "@/lib/utils";
import { VmTemplate } from "@/lib/api/types";
import { FadeIn } from "@/lib/animations";
import { useForm } from "react-hook-form";

// Mock VM data for comprehensive management
const mockVMs = [
  {
    id: "vm-001",
    name: "web-server-01",
    status: "running",
    template_id: "tmpl-ubuntu-web",
    cpu_cores: 4,
    memory_mb: 8192,
    disk_gb: 80,
    ip_address: "192.168.1.10",
    node: "node-01",
    uptime: "15d 4h 23m",
    cpu_usage: 45,
    memory_usage: 62,
    disk_usage: 35,
    network_in: 1.2,
    network_out: 0.8,
    created_at: "2024-08-10T09:30:00Z",
    last_backup: "2024-08-23T02:00:00Z"
  },
  {
    id: "vm-002", 
    name: "database-primary",
    status: "running",
    template_id: "tmpl-postgres-db",
    cpu_cores: 8,
    memory_mb: 16384,
    disk_gb: 500,
    ip_address: "192.168.1.20",
    node: "node-02",
    uptime: "30d 12h 45m",
    cpu_usage: 78,
    memory_usage: 89,
    disk_usage: 67,
    network_in: 2.8,
    network_out: 1.5,
    created_at: "2024-07-25T14:15:00Z",
    last_backup: "2024-08-24T01:00:00Z"
  },
  {
    id: "vm-003",
    name: "api-gateway",
    status: "stopped",
    template_id: "tmpl-nginx-lb",
    cpu_cores: 2,
    memory_mb: 4096,
    disk_gb: 40,
    ip_address: "192.168.1.30",
    node: "node-01", 
    uptime: "0d 0h 0m",
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 15,
    network_in: 0,
    network_out: 0,
    created_at: "2024-08-20T11:20:00Z",
    last_backup: "2024-08-22T02:00:00Z"
  },
  {
    id: "vm-004",
    name: "monitoring-stack",
    status: "migrating",
    template_id: "tmpl-monitoring",
    cpu_cores: 4,
    memory_mb: 8192,
    disk_gb: 120,
    ip_address: "192.168.1.40",
    node: "node-03",
    uptime: "7d 18h 12m",
    cpu_usage: 35,
    memory_usage: 55,
    disk_usage: 42,
    network_in: 0.9,
    network_out: 1.2,
    created_at: "2024-08-17T16:45:00Z",
    last_backup: "2024-08-23T03:00:00Z"
  }
];

const vmOperations = [
  { 
    id: "bulk-start", 
    label: "Start Selected", 
    icon: <Play className="h-4 w-4" />, 
    variant: "default" as const 
  },
  { 
    id: "bulk-stop", 
    label: "Stop Selected", 
    icon: <Square className="h-4 w-4" />, 
    variant: "outline" as const 
  },
  { 
    id: "bulk-restart", 
    label: "Restart Selected", 
    icon: <RotateCcw className="h-4 w-4" />, 
    variant: "outline" as const 
  },
  { 
    id: "bulk-migrate", 
    label: "Migrate Selected", 
    icon: <Activity className="h-4 w-4" />, 
    variant: "outline" as const 
  },
  { 
    id: "bulk-backup", 
    label: "Backup Selected", 
    icon: <Download className="h-4 w-4" />, 
    variant: "outline" as const 
  }
];

export default function VMManagementPage() {
  const { toast } = useToast();
  const [selectedVMs, setSelectedVMs] = useState<string[]>([]);
  const [vmFilters, setVmFilters] = useState({
    status: '',
    node: '',
    search: ''
  });
  const [showTemplateDialog, setShowTemplateDialog] = useState(false);
  const [showVMDialog, setShowVMDialog] = useState(false);
  
  const { data: templates, isLoading: templatesLoading } = useVmTemplates();
  const createTemplate = useCreateVmTemplate();
  const deleteTemplate = useDeleteVmTemplate();
  
  const { register, handleSubmit, reset, formState: { errors } } = useForm<Partial<VmTemplate>>();
  
  const filteredVMs = mockVMs.filter(vm => {
    const matchesStatus = !vmFilters.status || vm.status === vmFilters.status;
    const matchesNode = !vmFilters.node || vm.node === vmFilters.node;
    const matchesSearch = !vmFilters.search || 
      vm.name.toLowerCase().includes(vmFilters.search.toLowerCase()) ||
      vm.ip_address.includes(vmFilters.search);
    
    return matchesStatus && matchesNode && matchesSearch;
  });
  
  const handleVMSelect = (vmId: string, selected: boolean) => {
    setSelectedVMs(prev => 
      selected 
        ? [...prev, vmId]
        : prev.filter(id => id !== vmId)
    );
  };
  
  const handleSelectAll = (selected: boolean) => {
    setSelectedVMs(selected ? filteredVMs.map(vm => vm.id) : []);
  };
  
  const handleBulkOperation = async (operation: string) => {
    if (selectedVMs.length === 0) {
      toast({
        title: "No VMs selected",
        description: "Please select VMs to perform bulk operations.",
        variant: "destructive"
      });
      return;
    }
    
    // Simulate API call
    toast({
      title: "Operation initiated",
      description: `${operation.replace('bulk-', '').replace('-', ' ')} operation started for ${selectedVMs.length} VM(s).`
    });
    
    setSelectedVMs([]);
  };
  
  const handleCreateTemplate = async (data: Partial<VmTemplate>) => {
    try {
      await createTemplate.mutateAsync(data);
      toast({
        title: "Template created successfully",
        description: `${data.name} template has been created.`
      });
      setShowTemplateDialog(false);
      reset();
    } catch (error) {
      toast({
        title: "Failed to create template",
        description: "Please check the form and try again.",
        variant: "destructive"
      });
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "text-green-600";
      case "stopped": return "text-red-600";
      case "paused": return "text-yellow-600";
      case "migrating": return "text-blue-600";
      default: return "text-gray-600";
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running": return <Play className="h-4 w-4 text-green-600" />;
      case "stopped": return <Square className="h-4 w-4 text-red-600" />;
      case "paused": return <Pause className="h-4 w-4 text-yellow-600" />;
      case "migrating": return <Activity className="h-4 w-4 text-blue-600" />;
      default: return <AlertTriangle className="h-4 w-4 text-gray-600" />;
    }
  };
  
  const getResourceColor = (usage: number) => {
    if (usage >= 90) return "text-red-600";
    if (usage >= 75) return "text-yellow-600";
    return "text-green-600";
  };
  
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Server className="h-8 w-8" />
            VM Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage virtual machines, templates, and resource allocation
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Dialog open={showTemplateDialog} onOpenChange={setShowTemplateDialog}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Settings className="h-4 w-4 mr-2" />
                Templates
              </Button>
            </DialogTrigger>
          </Dialog>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Create VM
          </Button>
        </div>
      </div>
      
      {/* VM Overview Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total VMs</p>
                  <p className="text-2xl font-bold">{mockVMs.length}</p>
                  <div className="text-xs text-gray-600">
                    {mockVMs.filter(vm => vm.status === 'running').length} running
                  </div>
                </div>
                <Server className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">CPU Cores</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {mockVMs.reduce((acc, vm) => acc + vm.cpu_cores, 0)}
                  </p>
                  <div className="text-xs text-gray-600">
                    Avg {Math.round(mockVMs.reduce((acc, vm) => acc + vm.cpu_usage, 0) / mockVMs.length)}% usage
                  </div>
                </div>
                <Cpu className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Memory</p>
                  <p className="text-2xl font-bold text-green-600">
                    {Math.round(mockVMs.reduce((acc, vm) => acc + vm.memory_mb, 0) / 1024)}GB
                  </p>
                  <div className="text-xs text-gray-600">
                    Avg {Math.round(mockVMs.reduce((acc, vm) => acc + vm.memory_usage, 0) / mockVMs.length)}% usage
                  </div>
                </div>
                <MemoryStick className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Storage</p>
                  <p className="text-2xl font-bold text-orange-600">
                    {Math.round(mockVMs.reduce((acc, vm) => acc + vm.disk_gb, 0))}GB
                  </p>
                  <div className="text-xs text-gray-600">
                    Avg {Math.round(mockVMs.reduce((acc, vm) => acc + vm.disk_usage, 0) / mockVMs.length)}% usage
                  </div>
                </div>
                <HardDrive className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
      
      {/* VM Management Tabs */}
      <Tabs defaultValue="instances" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="instances">VM Instances</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>
        
        {/* VM Instances Tab */}
        <TabsContent value="instances" className="space-y-6">
          {/* Filters */}
          <Card>
            <CardContent className="p-6">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search VMs by name or IP..."
                    value={vmFilters.search}
                    onChange={(e) => setVmFilters(prev => ({ ...prev, search: e.target.value }))}
                    className="pl-10"
                  />
                </div>
                
                <Select 
                  value={vmFilters.status} 
                  onValueChange={(value) => setVmFilters(prev => ({ ...prev, status: value }))}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="All Statuses" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Statuses</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                    <SelectItem value="stopped">Stopped</SelectItem>
                    <SelectItem value="paused">Paused</SelectItem>
                    <SelectItem value="migrating">Migrating</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select 
                  value={vmFilters.node} 
                  onValueChange={(value) => setVmFilters(prev => ({ ...prev, node: value }))}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="All Nodes" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Nodes</SelectItem>
                    <SelectItem value="node-01">Node 01</SelectItem>
                    <SelectItem value="node-02">Node 02</SelectItem>
                    <SelectItem value="node-03">Node 03</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
          
          {/* Bulk Operations */}
          {selectedVMs.length > 0 && (
            <FadeIn>
              <Card className="border-blue-200 bg-blue-50 dark:bg-blue-950/50 dark:border-blue-800">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <span className="text-sm font-medium">
                        {selectedVMs.length} VM{selectedVMs.length !== 1 ? 's' : ''} selected
                      </span>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => setSelectedVMs([])}
                      >
                        Clear selection
                      </Button>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {vmOperations.map((operation) => (
                        <Button
                          key={operation.id}
                          variant={operation.variant}
                          size="sm"
                          onClick={() => handleBulkOperation(operation.id)}
                          className="flex items-center gap-1"
                        >
                          {operation.icon}
                          {operation.label}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          )}
          
          {/* VMs Table */}
          <FadeIn delay={0.5}>
            <Card>
              <CardHeader>
                <CardTitle>Virtual Machines ({filteredVMs.length})</CardTitle>
                <CardDescription>
                  Manage VM instances and their resources
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <input
                            type="checkbox"
                            checked={selectedVMs.length === filteredVMs.length && filteredVMs.length > 0}
                            onChange={(e) => handleSelectAll(e.target.checked)}
                            className="rounded"
                          />
                        </TableHead>
                        <TableHead>VM Details</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Resources</TableHead>
                        <TableHead>Network</TableHead>
                        <TableHead>Performance</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredVMs.map((vm) => (
                        <TableRow key={vm.id}>
                          <TableCell>
                            <input
                              type="checkbox"
                              checked={selectedVMs.includes(vm.id)}
                              onChange={(e) => handleVMSelect(vm.id, e.target.checked)}
                              className="rounded"
                            />
                          </TableCell>
                          
                          <TableCell>
                            <div>
                              <div className="font-medium">{vm.name}</div>
                              <div className="text-sm text-gray-600 dark:text-gray-400">
                                ID: {vm.id}
                              </div>
                              <div className="text-sm text-gray-600 dark:text-gray-400">
                                Node: {vm.node}
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="flex items-center gap-2">
                              {getStatusIcon(vm.status)}
                              <span className={cn("capitalize font-medium", getStatusColor(vm.status))}>
                                {vm.status}
                              </span>
                            </div>
                            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                              Uptime: {vm.uptime}
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="space-y-1 text-sm">
                              <div className="flex items-center gap-2">
                                <Cpu className="h-3 w-3" />
                                <span>{vm.cpu_cores} cores</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <MemoryStick className="h-3 w-3" />
                                <span>{Math.round(vm.memory_mb / 1024)}GB RAM</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <HardDrive className="h-3 w-3" />
                                <span>{vm.disk_gb}GB disk</span>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="space-y-1 text-sm">
                              <div>IP: {vm.ip_address}</div>
                              <div className="flex items-center gap-2">
                                <span className="text-green-600">↓ {vm.network_in} MB/s</span>
                                <span className="text-blue-600">↑ {vm.network_out} MB/s</span>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <span className="text-xs w-12">CPU:</span>
                                <div className="flex-1">
                                  <Progress value={vm.cpu_usage} className="h-1" />
                                </div>
                                <span className={cn("text-xs font-medium", getResourceColor(vm.cpu_usage))}>
                                  {vm.cpu_usage}%
                                </span>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <span className="text-xs w-12">RAM:</span>
                                <div className="flex-1">
                                  <Progress value={vm.memory_usage} className="h-1" />
                                </div>
                                <span className={cn("text-xs font-medium", getResourceColor(vm.memory_usage))}>
                                  {vm.memory_usage}%
                                </span>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <span className="text-xs w-12">Disk:</span>
                                <div className="flex-1">
                                  <Progress value={vm.disk_usage} className="h-1" />
                                </div>
                                <span className={cn("text-xs font-medium", getResourceColor(vm.disk_usage))}>
                                  {vm.disk_usage}%
                                </span>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="flex items-center gap-1">
                              {vm.status === 'running' ? (
                                <Button size="sm" variant="outline" className="h-8 w-8 p-0">
                                  <Square className="h-3 w-3" />
                                </Button>
                              ) : vm.status === 'stopped' ? (
                                <Button size="sm" variant="outline" className="h-8 w-8 p-0">
                                  <Play className="h-3 w-3" />
                                </Button>
                              ) : null}
                              
                              <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                                <Monitor className="h-3 w-3" />
                              </Button>
                              
                              <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                                <Settings className="h-3 w-3" />
                              </Button>
                              
                              <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                                <Eye className="h-3 w-3" />
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
        </TabsContent>
        
        {/* Templates Tab */}
        <TabsContent value="templates" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium">VM Templates</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Manage VM templates for quick deployment
              </p>
            </div>
            
            <Dialog open={showTemplateDialog} onOpenChange={setShowTemplateDialog}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Template
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Create VM Template</DialogTitle>
                  <DialogDescription>
                    Create a new VM template for standardized deployments.
                  </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit(handleCreateTemplate)} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="template-name">Template Name</Label>
                      <Input 
                        id="template-name"
                        {...register("name", { required: "Name is required" })}
                        placeholder="Ubuntu Web Server"
                      />
                      {errors.name && (
                        <p className="text-sm text-red-500 mt-1">{errors.name.message}</p>
                      )}
                    </div>
                    
                    <div>
                      <Label htmlFor="template-os">Operating System</Label>
                      <Select onValueChange={(value) => register("os").onChange({ target: { value } })}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select OS" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ubuntu">Ubuntu</SelectItem>
                          <SelectItem value="centos">CentOS</SelectItem>
                          <SelectItem value="debian">Debian</SelectItem>
                          <SelectItem value="windows">Windows Server</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div>
                    <Label htmlFor="template-description">Description</Label>
                    <Textarea
                      id="template-description"
                      {...register("description")}
                      placeholder="Template description and use case..."
                      rows={3}
                    />
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="template-cpu">CPU Cores</Label>
                      <Input 
                        id="template-cpu"
                        type="number"
                        {...register("cpu_cores", { required: "CPU cores required" })}
                        placeholder="4"
                        min="1"
                        max="32"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="template-memory">Memory (MB)</Label>
                      <Input 
                        id="template-memory"
                        type="number"
                        {...register("memory_mb", { required: "Memory required" })}
                        placeholder="8192"
                        min="512"
                        step="512"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="template-disk">Disk (GB)</Label>
                      <Input 
                        id="template-disk"
                        type="number"
                        {...register("disk_gb", { required: "Disk size required" })}
                        placeholder="80"
                        min="10"
                      />
                    </div>
                  </div>
                  
                  <div className="flex justify-end gap-2 pt-4">
                    <Button 
                      type="button" 
                      variant="outline" 
                      onClick={() => {
                        setShowTemplateDialog(false);
                        reset();
                      }}
                    >
                      Cancel
                    </Button>
                    <Button type="submit" disabled={createTemplate.isPending}>
                      {createTemplate.isPending ? "Creating..." : "Create Template"}
                    </Button>
                  </div>
                </form>
              </DialogContent>
            </Dialog>
          </div>
          
          <FadeIn delay={0.5}>
            <Card>
              <CardContent className="p-6">
                {templatesLoading ? (
                  <div className="flex items-center justify-center h-32">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {templates?.map((template) => (
                      <div key={template.id} className="border rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{template.name}</h4>
                          <Badge variant={template.is_public ? "secondary" : "outline"}>
                            {template.is_public ? "Public" : "Private"}
                          </Badge>
                        </div>
                        
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {template.description}
                        </p>
                        
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>OS: {template.os} {template.os_version}</div>
                          <div>CPU: {template.cpu_cores} cores</div>
                          <div>RAM: {Math.round(template.memory_mb / 1024)}GB</div>
                          <div>Disk: {template.disk_gb}GB</div>
                        </div>
                        
                        <div className="flex items-center justify-between text-xs text-gray-600">
                          <span>Used {template.usage_count} times</span>
                          <span>{new Date(template.created_at).toLocaleDateString()}</span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Button size="sm" variant="outline" className="flex-1">
                            <Copy className="h-3 w-3 mr-1" />
                            Clone
                          </Button>
                          <Button size="sm" variant="outline" className="flex-1">
                            <Edit className="h-3 w-3 mr-1" />
                            Edit
                          </Button>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="p-2"
                            onClick={() => deleteTemplate.mutateAsync(template.id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </FadeIn>
        </TabsContent>
        
        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Resource Utilization
                  </CardTitle>
                  <CardDescription>Current resource usage across all VMs</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">CPU Usage</span>
                      <span className="text-sm">
                        {Math.round(mockVMs.reduce((acc, vm) => acc + vm.cpu_usage, 0) / mockVMs.length)}%
                      </span>
                    </div>
                    <Progress 
                      value={Math.round(mockVMs.reduce((acc, vm) => acc + vm.cpu_usage, 0) / mockVMs.length)} 
                      className="h-2" 
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">Memory Usage</span>
                      <span className="text-sm">
                        {Math.round(mockVMs.reduce((acc, vm) => acc + vm.memory_usage, 0) / mockVMs.length)}%
                      </span>
                    </div>
                    <Progress 
                      value={Math.round(mockVMs.reduce((acc, vm) => acc + vm.memory_usage, 0) / mockVMs.length)} 
                      className="h-2" 
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium">Disk Usage</span>
                      <span className="text-sm">
                        {Math.round(mockVMs.reduce((acc, vm) => acc + vm.disk_usage, 0) / mockVMs.length)}%
                      </span>
                    </div>
                    <Progress 
                      value={Math.round(mockVMs.reduce((acc, vm) => acc + vm.disk_usage, 0) / mockVMs.length)} 
                      className="h-2" 
                    />
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Performance Alerts
                  </CardTitle>
                  <CardDescription>VMs requiring attention</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3 p-3 bg-red-50 dark:bg-red-950/50 rounded-lg border border-red-200 dark:border-red-800">
                      <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-red-800 dark:text-red-200">High Memory Usage</div>
                        <div className="text-sm text-red-700 dark:text-red-300">
                          database-primary: 89% memory utilization
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3 p-3 bg-yellow-50 dark:bg-yellow-950/50 rounded-lg border border-yellow-200 dark:border-yellow-800">
                      <Clock className="h-5 w-5 text-yellow-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-yellow-800 dark:text-yellow-200">High CPU Usage</div>
                        <div className="text-sm text-yellow-700 dark:text-yellow-300">
                          database-primary: 78% CPU utilization
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-950/50 rounded-lg border border-green-200 dark:border-green-800">
                      <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-green-800 dark:text-green-200">All Systems Normal</div>
                        <div className="text-sm text-green-700 dark:text-green-300">
                          Other VMs operating within normal parameters
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}