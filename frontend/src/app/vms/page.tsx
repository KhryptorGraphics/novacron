"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { 
  Server, 
  Play, 
  Pause, 
  Square, 
  RotateCw, 
  Settings, 
  Monitor, 
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  Plus,
  Search,
  Filter,
  Download,
  Upload,
  Trash2,
  ArrowRight,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { VMCreateDialog } from "@/components/vm/VMCreateDialog";
import { VMMigrationDialog } from "@/components/vm/VMMigrationDialog";
import { VMDetailsCard } from "@/components/vm/VMDetailsCard";

// Mock VM data - replace with API calls
const mockVMs = [
  {
    id: "vm-001",
    name: "web-server-01",
    status: "running",
    os: "Ubuntu 22.04 LTS",
    cpu: { cores: 4, usage: 45 },
    memory: { total: 8, used: 5.2, usage: 65 },
    disk: { total: 100, used: 32, usage: 32 },
    network: { rx: 15.2, tx: 8.7 },
    uptime: "5d 12h 34m",
    host: "node-01",
    ip: "192.168.1.10",
    created: "2024-01-15T10:30:00Z"
  },
  {
    id: "vm-002", 
    name: "database-primary",
    status: "running",
    os: "CentOS 8",
    cpu: { cores: 8, usage: 78 },
    memory: { total: 16, used: 12.8, usage: 80 },
    disk: { total: 500, used: 280, usage: 56 },
    network: { rx: 45.1, tx: 67.3 },
    uptime: "12d 8h 15m",
    host: "node-02", 
    ip: "192.168.1.11",
    created: "2024-01-10T14:22:00Z"
  },
  {
    id: "vm-003",
    name: "dev-environment",
    status: "stopped", 
    os: "Ubuntu 20.04 LTS",
    cpu: { cores: 2, usage: 0 },
    memory: { total: 4, used: 0, usage: 0 },
    disk: { total: 50, used: 18, usage: 36 },
    network: { rx: 0, tx: 0 },
    uptime: "0m",
    host: "node-01",
    ip: "192.168.1.12",
    created: "2024-01-20T09:15:00Z"
  },
  {
    id: "vm-004",
    name: "backup-server",
    status: "error",
    os: "Debian 11",
    cpu: { cores: 2, usage: 0 },
    memory: { total: 8, used: 0, usage: 0 },
    disk: { total: 1000, used: 450, usage: 45 },
    network: { rx: 0, tx: 0 },
    uptime: "0m",
    host: "node-03",
    ip: "192.168.1.13",
    created: "2024-01-08T16:45:00Z"
  }
];

export default function VMsPage() {
  const [vms, setVMs] = useState(mockVMs);
  const [filteredVMs, setFilteredVMs] = useState(mockVMs);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [hostFilter, setHostFilter] = useState("all");
  const [selectedVM, setSelectedVM] = useState<string | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showMigrationDialog, setShowMigrationDialog] = useState(false);
  const [migrationVM, setMigrationVM] = useState<string | null>(null);

  // Filter VMs based on search and filters
  useEffect(() => {
    let filtered = vms.filter(vm => {
      const matchesSearch = vm.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           vm.os.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           vm.ip.includes(searchQuery);
      const matchesStatus = statusFilter === "all" || vm.status === statusFilter;
      const matchesHost = hostFilter === "all" || vm.host === hostFilter;
      
      return matchesSearch && matchesStatus && matchesHost;
    });
    
    setFilteredVMs(filtered);
  }, [vms, searchQuery, statusFilter, hostFilter]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "stopped": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      case "error": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "starting": return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running": return <CheckCircle className="h-4 w-4" />;
      case "stopped": return <Square className="h-4 w-4" />;
      case "error": return <AlertCircle className="h-4 w-4" />;
      case "starting": return <Clock className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  const handleVMAction = (vmId: string, action: string) => {
    // Mock VM action - replace with API call
    console.log(`Performing ${action} on VM ${vmId}`);
    // Update VM status locally for demo
    setVMs(prev => prev.map(vm => 
      vm.id === vmId 
        ? { ...vm, status: action === "start" ? "starting" : action === "stop" ? "stopped" : vm.status }
        : vm
    ));
  };

  const handleMigration = (vmId: string) => {
    setMigrationVM(vmId);
    setShowMigrationDialog(true);
  };

  const vmStats = {
    total: vms.length,
    running: vms.filter(vm => vm.status === "running").length,
    stopped: vms.filter(vm => vm.status === "stopped").length,
    error: vms.filter(vm => vm.status === "error").length
  };

  const totalResources = vms.reduce((acc, vm) => ({
    cpu: acc.cpu + vm.cpu.cores,
    memory: acc.memory + vm.memory.total,
    disk: acc.disk + vm.disk.total
  }), { cpu: 0, memory: 0, disk: 0 });

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Virtual Machines</h1>
          <p className="text-muted-foreground">Manage and monitor your virtual infrastructure</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create VM
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total VMs</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{vmStats.total}</div>
            <p className="text-xs text-muted-foreground">
              {vmStats.running} running, {vmStats.stopped} stopped
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total CPU Cores</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalResources.cpu}</div>
            <p className="text-xs text-muted-foreground">
              Allocated across all VMs
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Memory</CardTitle>
            <MemoryStick className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalResources.memory} GB</div>
            <p className="text-xs text-muted-foreground">
              RAM allocated to VMs
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Storage</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalResources.disk} GB</div>
            <p className="text-xs text-muted-foreground">
              Disk space allocated
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
                placeholder="Search VMs by name, OS, or IP..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="stopped">Stopped</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>
            <Select value={hostFilter} onValueChange={setHostFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Host" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Hosts</SelectItem>
                <SelectItem value="node-01">node-01</SelectItem>
                <SelectItem value="node-02">node-02</SelectItem>
                <SelectItem value="node-03">node-03</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* VM List */}
      <Tabs defaultValue="grid" className="w-full">
        <TabsList>
          <TabsTrigger value="grid">Grid View</TabsTrigger>
          <TabsTrigger value="table">Table View</TabsTrigger>
        </TabsList>
        
        <TabsContent value="grid" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredVMs.map((vm) => (
              <VMDetailsCard
                key={vm.id}
                vm={vm}
                onAction={handleVMAction}
                onMigration={handleMigration}
                onSelect={setSelectedVM}
              />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="table" className="space-y-4">
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>OS</TableHead>
                  <TableHead>Host</TableHead>
                  <TableHead>CPU</TableHead>
                  <TableHead>Memory</TableHead>
                  <TableHead>Storage</TableHead>
                  <TableHead>IP Address</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredVMs.map((vm) => (
                  <TableRow key={vm.id}>
                    <TableCell className="font-medium">{vm.name}</TableCell>
                    <TableCell>
                      <Badge className={getStatusColor(vm.status)}>
                        {getStatusIcon(vm.status)}
                        <span className="ml-1 capitalize">{vm.status}</span>
                      </Badge>
                    </TableCell>
                    <TableCell>{vm.os}</TableCell>
                    <TableCell>{vm.host}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm">{vm.cpu.cores} cores</span>
                        <div className="w-16">
                          <Progress value={vm.cpu.usage} className="h-1" />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {vm.cpu.usage}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm">{vm.memory.total} GB</span>
                        <div className="w-16">
                          <Progress value={vm.memory.usage} className="h-1" />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {vm.memory.usage}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm">{vm.disk.total} GB</span>
                        <div className="w-16">
                          <Progress value={vm.disk.usage} className="h-1" />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {vm.disk.usage}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>{vm.ip}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-1">
                        {vm.status === "running" ? (
                          <>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={() => handleVMAction(vm.id, "stop")}
                            >
                              <Square className="h-4 w-4" />
                            </Button>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={() => handleVMAction(vm.id, "restart")}
                            >
                              <RotateCw className="h-4 w-4" />
                            </Button>
                          </>
                        ) : (
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => handleVMAction(vm.id, "start")}
                          >
                            <Play className="h-4 w-4" />
                          </Button>
                        )}
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => handleMigration(vm.id)}
                        >
                          <ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Dialogs */}
      <VMCreateDialog 
        open={showCreateDialog} 
        onOpenChange={setShowCreateDialog}
      />
      
      <VMMigrationDialog
        open={showMigrationDialog}
        onOpenChange={setShowMigrationDialog}
        vmId={migrationVM}
      />
    </div>
  );
}