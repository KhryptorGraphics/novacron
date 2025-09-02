'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { 
  Play, 
  Square, 
  RotateCw, 
  MoveHorizontal, 
  Trash2, 
  Settings, 
  Activity,
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  Shield,
  AlertCircle,
  CheckCircle,
  Clock,
  MoreVertical,
  Download,
  Upload,
  Copy,
  Terminal,
  Eye,
  Gauge
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface VM {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'paused' | 'migrating' | 'error';
  host: string;
  resources: {
    cpu: {
      cores: number;
      usage: number;
    };
    memory: {
      allocated: number;
      used: number;
    };
    disk: {
      allocated: number;
      used: number;
    };
    network: {
      rx: number;
      tx: number;
    };
  };
  os: string;
  ipAddress: string;
  uptime: number;
  lastBackup: string;
  tags: string[];
  alerts: number;
}

interface MigrationTarget {
  id: string;
  name: string;
  available: boolean;
  resources: {
    cpuAvailable: number;
    memoryAvailable: number;
    diskAvailable: number;
  };
  load: number;
}

const VMOperationsDashboard: React.FC = () => {
  const [vms, setVms] = useState<VM[]>([]);
  const [selectedVm, setSelectedVm] = useState<VM | null>(null);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [migrationTargets, setMigrationTargets] = useState<MigrationTarget[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [actionDialog, setActionDialog] = useState<{ open: boolean; action: string; vm: VM | null }>({ 
    open: false, 
    action: '', 
    vm: null 
  });

  // WebSocket connection for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/api/ws/vms');

  useEffect(() => {
    if (wsData) {
      setVms(wsData.vms || []);
    }
  }, [wsData]);

  // Mock data for development
  useEffect(() => {
    const mockVms: VM[] = [
      {
        id: 'vm-001',
        name: 'web-server-01',
        status: 'running',
        host: 'host-01',
        resources: {
          cpu: { cores: 4, usage: 45 },
          memory: { allocated: 8192, used: 6144 },
          disk: { allocated: 100, used: 45 },
          network: { rx: 1024, tx: 512 }
        },
        os: 'Ubuntu 22.04 LTS',
        ipAddress: '192.168.1.101',
        uptime: 86400,
        lastBackup: '2 hours ago',
        tags: ['production', 'web'],
        alerts: 0
      },
      {
        id: 'vm-002',
        name: 'database-01',
        status: 'running',
        host: 'host-02',
        resources: {
          cpu: { cores: 8, usage: 75 },
          memory: { allocated: 16384, used: 14336 },
          disk: { allocated: 500, used: 350 },
          network: { rx: 2048, tx: 1024 }
        },
        os: 'PostgreSQL 15',
        ipAddress: '192.168.1.102',
        uptime: 172800,
        lastBackup: '1 hour ago',
        tags: ['production', 'database'],
        alerts: 1
      },
      {
        id: 'vm-003',
        name: 'dev-environment',
        status: 'stopped',
        host: 'host-01',
        resources: {
          cpu: { cores: 2, usage: 0 },
          memory: { allocated: 4096, used: 0 },
          disk: { allocated: 50, used: 20 },
          network: { rx: 0, tx: 0 }
        },
        os: 'Debian 11',
        ipAddress: '192.168.1.103',
        uptime: 0,
        lastBackup: '1 day ago',
        tags: ['development'],
        alerts: 0
      }
    ];

    const mockTargets: MigrationTarget[] = [
      {
        id: 'host-02',
        name: 'host-02.datacenter.local',
        available: true,
        resources: {
          cpuAvailable: 16,
          memoryAvailable: 32768,
          diskAvailable: 1000
        },
        load: 45
      },
      {
        id: 'host-03',
        name: 'host-03.datacenter.local',
        available: true,
        resources: {
          cpuAvailable: 8,
          memoryAvailable: 16384,
          diskAvailable: 500
        },
        load: 30
      }
    ];

    setVms(mockVms);
    setMigrationTargets(mockTargets);
  }, []);

  const handleVMAction = async (action: string, vm: VM) => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/vms/${vm.id}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        // Update VM status optimistically
        setVms(prev => (prev || []).map(v => 
          v.id === vm.id 
            ? { ...v, status: action === 'start' ? 'running' : action === 'stop' ? 'stopped' : v.status }
            : v
        ));
      }
    } catch (error) {
      console.error(`Failed to ${action} VM:`, error);
    } finally {
      setIsLoading(false);
      setActionDialog({ open: false, action: '', vm: null });
    }
  };

  const handleMigration = async (vm: VM, targetId: string) => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/vms/${vm.id}/migrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targetHost: targetId })
      });
      if (response.ok) {
        setVms(prev => (prev || []).map(v => 
          v.id === vm.id ? { ...v, status: 'migrating' } : v
        ));
      }
    } catch (error) {
      console.error('Migration failed:', error);
    } finally {
      setIsLoading(false);
      setActionDialog({ open: false, action: '', vm: null });
    }
  };

  const getStatusColor = (status: VM['status']) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'stopped': return 'bg-gray-500';
      case 'paused': return 'bg-yellow-500';
      case 'migrating': return 'bg-blue-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: VM['status']) => {
    switch (status) {
      case 'running': return <CheckCircle className="h-4 w-4" />;
      case 'stopped': return <Square className="h-4 w-4" />;
      case 'paused': return <Clock className="h-4 w-4" />;
      case 'migrating': return <MoveHorizontal className="h-4 w-4" />;
      case 'error': return <AlertCircle className="h-4 w-4" />;
      default: return null;
    }
  };

  const filteredVms = vms.filter(vm => {
    if (filter === 'all') return true;
    if (filter === 'running') return vm.status === 'running';
    if (filter === 'stopped') return vm.status === 'stopped';
    if (filter === 'alerts') return vm.alerts > 0;
    return true;
  }).sort((a, b) => {
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    if (sortBy === 'status') return a.status.localeCompare(b.status);
    if (sortBy === 'cpu') return b.resources.cpu.usage - a.resources.cpu.usage;
    if (sortBy === 'memory') return b.resources.memory.used - a.resources.memory.used;
    return 0;
  });

  // Performance data for charts
  const performanceData = [
    { time: '00:00', cpu: 20, memory: 40, network: 10 },
    { time: '04:00', cpu: 35, memory: 45, network: 15 },
    { time: '08:00', cpu: 65, memory: 60, network: 45 },
    { time: '12:00', cpu: 80, memory: 75, network: 60 },
    { time: '16:00', cpu: 70, memory: 70, network: 50 },
    { time: '20:00', cpu: 45, memory: 55, network: 30 },
    { time: '24:00', cpu: 30, memory: 45, network: 20 }
  ];

  const resourceDistribution = [
    { name: 'CPU', value: 65, fill: '#3b82f6' },
    { name: 'Memory', value: 75, fill: '#10b981' },
    { name: 'Disk', value: 45, fill: '#f59e0b' },
    { name: 'Network', value: 30, fill: '#8b5cf6' }
  ];

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold">VM Operations</h1>
          <Badge variant="outline" className={isConnected ? 'bg-green-50' : 'bg-red-50'}>
            <div className={`h-2 w-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
        </div>
        
        <div className="flex items-center gap-2">
          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Filter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All VMs</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="stopped">Stopped</SelectItem>
              <SelectItem value="alerts">With Alerts</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name">Name</SelectItem>
              <SelectItem value="status">Status</SelectItem>
              <SelectItem value="cpu">CPU Usage</SelectItem>
              <SelectItem value="memory">Memory Usage</SelectItem>
            </SelectContent>
          </Select>
          
          <Button variant="outline" size="icon">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total VMs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{vms.length}</div>
            <p className="text-xs text-muted-foreground">
              {vms.filter(v => v.status === 'running').length} running
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">65%</div>
            <Progress value={65} className="h-2 mt-2" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <MemoryStick className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">75%</div>
            <Progress value={75} className="h-2 mt-2" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{vms.reduce((sum, vm) => sum + vm.alerts, 0)}</div>
            <p className="text-xs text-muted-foreground">
              Across {vms.filter(v => v.alerts > 0).length} VMs
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* VM List */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Virtual Machines</CardTitle>
              <CardDescription>Manage and monitor your VM fleet</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {(filteredVms || []).map((vm) => (
                <div
                  key={vm.id}
                  className={`border rounded-lg p-4 hover:bg-accent cursor-pointer transition-colors ${
                    selectedVm?.id === vm.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedVm(vm)}
                >
                  <div className="flex items-start justify-between">
                    <div className="space-y-2 flex-1">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">{vm.name}</h3>
                        <Badge variant="outline" className={`${getStatusColor(vm.status)} text-white`}>
                          {getStatusIcon(vm.status)}
                          <span className="ml-1">{vm.status}</span>
                        </Badge>
                        {vm.alerts > 0 && (
                          <Badge variant="destructive">
                            {vm.alerts} Alert{vm.alerts > 1 ? 's' : ''}
                          </Badge>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">CPU</p>
                          <div className="flex items-center gap-1">
                            <Gauge className="h-3 w-3" />
                            <span className="font-medium">{vm.resources.cpu.usage}%</span>
                          </div>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Memory</p>
                          <div className="flex items-center gap-1">
                            <MemoryStick className="h-3 w-3" />
                            <span className="font-medium">
                              {(vm.resources.memory.used / 1024).toFixed(1)}GB
                            </span>
                          </div>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Disk</p>
                          <div className="flex items-center gap-1">
                            <HardDrive className="h-3 w-3" />
                            <span className="font-medium">{vm.resources.disk.used}GB</span>
                          </div>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Network</p>
                          <div className="flex items-center gap-1">
                            <Network className="h-3 w-3" />
                            <span className="font-medium">
                              ↓{vm.resources.network.rx} ↑{vm.resources.network.tx}
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>{vm.os}</span>
                        <span>{vm.ipAddress}</span>
                        <span>Host: {vm.host}</span>
                      </div>
                      
                      <div className="flex gap-1">
                        {(vm.tags || []).map((tag) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    
                    <div className="flex gap-1">
                      {vm.status === 'stopped' ? (
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActionDialog({ open: true, action: 'start', vm });
                          }}
                        >
                          <Play className="h-4 w-4" />
                        </Button>
                      ) : (
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActionDialog({ open: true, action: 'stop', vm });
                          }}
                        >
                          <Square className="h-4 w-4" />
                        </Button>
                      )}
                      
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          setActionDialog({ open: true, action: 'restart', vm });
                        }}
                      >
                        <RotateCw className="h-4 w-4" />
                      </Button>
                      
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          setActionDialog({ open: true, action: 'migrate', vm });
                        }}
                      >
                        <MoveHorizontal className="h-4 w-4" />
                      </Button>
                      
                      <Button size="icon" variant="ghost">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* VM Details Panel */}
        <div className="space-y-4">
          {selectedVm ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>VM Details</CardTitle>
                  <CardDescription>{selectedVm.name}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="overview">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="overview">Overview</TabsTrigger>
                      <TabsTrigger value="performance">Performance</TabsTrigger>
                      <TabsTrigger value="actions">Actions</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="overview" className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Status</span>
                          <Badge className={getStatusColor(selectedVm.status)}>
                            {selectedVm.status}
                          </Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Operating System</span>
                          <span className="text-sm font-medium">{selectedVm.os}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">IP Address</span>
                          <span className="text-sm font-medium">{selectedVm.ipAddress}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Host</span>
                          <span className="text-sm font-medium">{selectedVm.host}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Uptime</span>
                          <span className="text-sm font-medium">
                            {Math.floor(selectedVm.uptime / 86400)}d {Math.floor((selectedVm.uptime % 86400) / 3600)}h
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Last Backup</span>
                          <span className="text-sm font-medium">{selectedVm.lastBackup}</span>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <h4 className="text-sm font-semibold">Resources</h4>
                        <div className="space-y-2">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>CPU ({selectedVm.resources.cpu.cores} cores)</span>
                              <span>{selectedVm.resources.cpu.usage}%</span>
                            </div>
                            <Progress value={selectedVm.resources.cpu.usage} className="h-2" />
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Memory</span>
                              <span>{(selectedVm.resources.memory.used / 1024).toFixed(1)}/{(selectedVm.resources.memory.allocated / 1024).toFixed(1)}GB</span>
                            </div>
                            <Progress 
                              value={(selectedVm.resources.memory.used / selectedVm.resources.memory.allocated) * 100} 
                              className="h-2" 
                            />
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Disk</span>
                              <span>{selectedVm.resources.disk.used}/{selectedVm.resources.disk.allocated}GB</span>
                            </div>
                            <Progress 
                              value={(selectedVm.resources.disk.used / selectedVm.resources.disk.allocated) * 100} 
                              className="h-2" 
                            />
                          </div>
                        </div>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="performance">
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-semibold mb-2">Resource Usage (24h)</h4>
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={performanceData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="time" />
                              <YAxis />
                              <Tooltip />
                              <Line type="monotone" dataKey="cpu" stroke="#3b82f6" name="CPU" />
                              <Line type="monotone" dataKey="memory" stroke="#10b981" name="Memory" />
                              <Line type="monotone" dataKey="network" stroke="#f59e0b" name="Network" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="actions" className="space-y-2">
                      <Button className="w-full justify-start" variant="outline">
                        <Terminal className="mr-2 h-4 w-4" />
                        Open Console
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Eye className="mr-2 h-4 w-4" />
                        View Logs
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Copy className="mr-2 h-4 w-4" />
                        Clone VM
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Download className="mr-2 h-4 w-4" />
                        Create Snapshot
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Upload className="mr-2 h-4 w-4" />
                        Restore Snapshot
                      </Button>
                      <Button className="w-full justify-start" variant="outline" className="text-red-600">
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete VM
                      </Button>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
              
              {/* Resource Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Resource Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={resourceDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {(resourceDistribution || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-[400px] text-muted-foreground">
                Select a VM to view details
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Action Dialogs */}
      <Dialog open={actionDialog.open} onOpenChange={(open) => setActionDialog({ ...actionDialog, open })}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {actionDialog.action === 'migrate' ? 'Migrate VM' : `Confirm ${actionDialog.action}`}
            </DialogTitle>
            <DialogDescription>
              {actionDialog.action === 'migrate' 
                ? `Select target host for ${actionDialog.vm?.name}`
                : `Are you sure you want to ${actionDialog.action} ${actionDialog.vm?.name}?`
              }
            </DialogDescription>
          </DialogHeader>
          
          {actionDialog.action === 'migrate' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Target Host</Label>
                <Select value={selectedTarget} onValueChange={setSelectedTarget}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select target host" />
                  </SelectTrigger>
                  <SelectContent>
                    {(migrationTargets || []).map((target) => (
                      <SelectItem key={target.id} value={target.id}>
                        <div className="flex items-center justify-between w-full">
                          <span>{target.name}</span>
                          <Badge variant="outline" className="ml-2">
                            {target.load}% load
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              {selectedTarget && migrationTargets.find(t => t.id === selectedTarget) && (
                <Alert>
                  <AlertDescription>
                    <div className="space-y-1 text-sm">
                      <p>Available Resources:</p>
                      <p>• CPU: {migrationTargets.find(t => t.id === selectedTarget)?.resources.cpuAvailable} cores</p>
                      <p>• Memory: {(migrationTargets.find(t => t.id === selectedTarget)?.resources.memoryAvailable || 0) / 1024}GB</p>
                      <p>• Disk: {migrationTargets.find(t => t.id === selectedTarget)?.resources.diskAvailable}GB</p>
                    </div>
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setActionDialog({ open: false, action: '', vm: null })}>
              Cancel
            </Button>
            <Button 
              onClick={() => {
                if (actionDialog.vm) {
                  if (actionDialog.action === 'migrate') {
                    handleMigration(actionDialog.vm, selectedTarget);
                  } else {
                    handleVMAction(actionDialog.action, actionDialog.vm);
                  }
                }
              }}
              disabled={actionDialog.action === 'migrate' && !selectedTarget}
            >
              {isLoading ? 'Processing...' : 'Confirm'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default VMOperationsDashboard;