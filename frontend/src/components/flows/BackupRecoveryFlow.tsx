'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Calendar,
  Clock,
  Save,
  RotateCcw,
  Shield,
  AlertTriangle,
  CheckCircle,
  Play,
  Pause,
  Eye,
  Download,
  Trash2,
  Settings,
  Archive,
  Database,
  HardDrive,
  CloudUpload,
  RefreshCw,
  Search,
  Filter,
  ChevronDown,
  ChevronRight,
  FileText
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

interface BackupPolicy {
  id: string;
  name: string;
  description: string;
  schedule: {
    type: 'hourly' | 'daily' | 'weekly' | 'monthly';
    time: string;
    daysOfWeek?: number[];
    dayOfMonth?: number;
  };
  retention: {
    hourly: number;
    daily: number;
    weekly: number;
    monthly: number;
  };
  storage: {
    tier: 'hot' | 'cold' | 'archive';
    location: 'local' | 'cloud' | 'hybrid';
    encryption: boolean;
    compression: boolean;
  };
  targets: string[];
  active: boolean;
  nextRun: Date;
  lastRun?: Date;
  lastStatus: 'success' | 'failed' | 'running' | 'pending';
}

interface BackupInstance {
  id: string;
  vmId: string;
  vmName: string;
  policyId: string;
  policyName: string;
  type: 'full' | 'incremental' | 'differential';
  status: 'completed' | 'failed' | 'running' | 'queued';
  createdAt: Date;
  completedAt?: Date;
  size: number;
  compression: number;
  duration: number;
  location: string;
  verificationStatus: 'verified' | 'failed' | 'pending' | 'not_verified';
  restorePoints: number;
  tags: string[];
}

interface RestoreJob {
  id: string;
  backupId: string;
  targetVM: string;
  type: 'full' | 'file_level' | 'application';
  status: 'running' | 'completed' | 'failed' | 'pending';
  progress: number;
  startedAt: Date;
  estimatedCompletion?: Date;
  selectedFiles?: string[];
  targetLocation: string;
}

const mockPolicies: BackupPolicy[] = [
  {
    id: 'policy-1',
    name: 'Production Daily Backup',
    description: 'Daily backups for production VMs with 30-day retention',
    schedule: {
      type: 'daily',
      time: '02:00',
    },
    retention: {
      hourly: 0,
      daily: 30,
      weekly: 12,
      monthly: 6
    },
    storage: {
      tier: 'hot',
      location: 'hybrid',
      encryption: true,
      compression: true
    },
    targets: ['vm-001', 'vm-002', 'vm-003'],
    active: true,
    nextRun: new Date(Date.now() + 6 * 60 * 60 * 1000),
    lastRun: new Date(Date.now() - 18 * 60 * 60 * 1000),
    lastStatus: 'success'
  },
  {
    id: 'policy-2',
    name: 'Development Hourly Snapshot',
    description: 'Hourly snapshots for development environment',
    schedule: {
      type: 'hourly',
      time: '00',
    },
    retention: {
      hourly: 24,
      daily: 7,
      weekly: 2,
      monthly: 0
    },
    storage: {
      tier: 'hot',
      location: 'local',
      encryption: false,
      compression: true
    },
    targets: ['vm-004', 'vm-005'],
    active: true,
    nextRun: new Date(Date.now() + 45 * 60 * 1000),
    lastRun: new Date(Date.now() - 15 * 60 * 1000),
    lastStatus: 'success'
  }
];

const mockBackups: BackupInstance[] = [
  {
    id: 'backup-001',
    vmId: 'vm-001',
    vmName: 'web-server-01',
    policyId: 'policy-1',
    policyName: 'Production Daily Backup',
    type: 'full',
    status: 'completed',
    createdAt: new Date(Date.now() - 18 * 60 * 60 * 1000),
    completedAt: new Date(Date.now() - 17 * 60 * 60 * 1000),
    size: 45.2,
    compression: 65,
    duration: 3420,
    location: 'AWS S3 - us-east-1',
    verificationStatus: 'verified',
    restorePoints: 3,
    tags: ['production', 'verified', 'critical']
  },
  {
    id: 'backup-002',
    vmId: 'vm-002',
    vmName: 'app-server-02',
    policyId: 'policy-1',
    policyName: 'Production Daily Backup',
    type: 'incremental',
    status: 'completed',
    createdAt: new Date(Date.now() - 6 * 60 * 60 * 1000),
    completedAt: new Date(Date.now() - 5.5 * 60 * 60 * 1000),
    size: 12.8,
    compression: 45,
    duration: 1680,
    location: 'Local Storage',
    verificationStatus: 'verified',
    restorePoints: 1,
    tags: ['production', 'incremental']
  },
  {
    id: 'backup-003',
    vmId: 'vm-003',
    vmName: 'db-server-01',
    policyId: 'policy-1',
    policyName: 'Production Daily Backup',
    type: 'full',
    status: 'failed',
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    size: 0,
    compression: 0,
    duration: 0,
    location: '',
    verificationStatus: 'failed',
    restorePoints: 0,
    tags: ['production', 'failed']
  }
];

const mockVMs = [
  { id: 'vm-001', name: 'web-server-01', status: 'running' },
  { id: 'vm-002', name: 'app-server-02', status: 'running' },
  { id: 'vm-003', name: 'db-server-01', status: 'running' },
  { id: 'vm-004', name: 'dev-server-01', status: 'running' },
  { id: 'vm-005', name: 'test-server-01', status: 'stopped' }
];

export function BackupRecoveryFlow() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPolicy, setSelectedPolicy] = useState<BackupPolicy | null>(null);
  const [selectedBackup, setSelectedBackup] = useState<BackupInstance | null>(null);
  const [isCreatingPolicy, setIsCreatingPolicy] = useState(false);
  const [isRunningBackup, setIsRunningBackup] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [backupProgress, setBackupProgress] = useState(0);
  const [restoreProgress, setRestoreProgress] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');
  const [expandedBackups, setExpandedBackups] = useState<Set<string>>(new Set());
  
  // New backup policy form state
  const [newPolicy, setNewPolicy] = useState<Partial<BackupPolicy>>({
    name: '',
    description: '',
    schedule: {
      type: 'daily',
      time: '02:00'
    },
    retention: {
      hourly: 0,
      daily: 7,
      weekly: 4,
      monthly: 3
    },
    storage: {
      tier: 'hot',
      location: 'local',
      encryption: true,
      compression: true
    },
    targets: [],
    active: true
  });

  // Restore configuration state
  const [restoreConfig, setRestoreConfig] = useState({
    backupId: '',
    restoreType: 'full',
    targetVM: '',
    targetLocation: 'original',
    selectedFiles: [] as string[],
    restoreToNewVM: false,
    newVMName: '',
    verifyAfterRestore: true
  });

  const { toast } = useToast();

  // Simulate backup progress
  useEffect(() => {
    if (isRunningBackup) {
      const interval = setInterval(() => {
        setBackupProgress(prev => {
          if (prev >= 100) {
            setIsRunningBackup(false);
            toast({
              title: "Backup Completed",
              description: "VM backup has been created successfully.",
            });
            return 100;
          }
          return prev + Math.random() * 5;
        });
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isRunningBackup, toast]);

  // Simulate restore progress
  useEffect(() => {
    if (isRestoring) {
      const interval = setInterval(() => {
        setRestoreProgress(prev => {
          if (prev >= 100) {
            setIsRestoring(false);
            toast({
              title: "Restore Completed",
              description: "VM has been restored successfully.",
            });
            return 100;
          }
          return prev + Math.random() * 3;
        });
      }, 800);
      
      return () => clearInterval(interval);
    }
  }, [isRestoring, toast]);

  const runBackupNow = (policyId: string) => {
    setIsRunningBackup(true);
    setBackupProgress(0);
    toast({
      title: "Backup Started",
      description: "Manual backup job has been initiated.",
    });
  };

  const startRestore = () => {
    if (!restoreConfig.backupId || !restoreConfig.targetVM) {
      toast({
        title: "Invalid Configuration",
        description: "Please select backup and target VM.",
        variant: "destructive"
      });
      return;
    }
    
    setIsRestoring(true);
    setRestoreProgress(0);
    toast({
      title: "Restore Started",
      description: "VM restore operation has begun.",
    });
  };

  const filteredBackups = mockBackups.filter(backup => {
    const matchesSearch = backup.vmName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         backup.policyName.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || backup.status === filterStatus;
    const matchesType = filterType === 'all' || backup.type === filterType;
    
    return matchesSearch && matchesStatus && matchesType;
  });

  const toggleBackupExpansion = (backupId: string) => {
    const newExpanded = new Set(expandedBackups);
    if (newExpanded.has(backupId)) {
      newExpanded.delete(backupId);
    } else {
      newExpanded.add(backupId);
    }
    setExpandedBackups(newExpanded);
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const formatSize = (sizeGB: number) => {
    if (sizeGB >= 1024) {
      return `${(sizeGB / 1024).toFixed(1)} TB`;
    }
    return `${sizeGB.toFixed(1)} GB`;
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold">Backup & Recovery</h2>
          <p className="text-gray-600 mt-1">Protect your VMs with automated backups and point-in-time recovery</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={() => setIsCreatingPolicy(true)}>
            <Settings className="mr-2 h-4 w-4" />
            Create Policy
          </Button>
          <Button>
            <Save className="mr-2 h-4 w-4" />
            Backup Now
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="backups">Backups</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="restore">Restore</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Backups</p>
                    <p className="text-3xl font-bold">247</p>
                  </div>
                  <Archive className="h-8 w-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Success Rate</p>
                    <p className="text-3xl font-bold text-green-600">98.2%</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-600" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Storage Used</p>
                    <p className="text-3xl font-bold">2.4 TB</p>
                  </div>
                  <HardDrive className="h-8 w-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Next Backup</p>
                    <p className="text-3xl font-bold">2h</p>
                  </div>
                  <Clock className="h-8 w-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Recent Backups</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockBackups.slice(0, 5).map(backup => (
                    <div key={backup.id} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center space-x-3">
                        <div className={`p-1 rounded ${
                          backup.status === 'completed' ? 'bg-green-100 text-green-600' :
                          backup.status === 'failed' ? 'bg-red-100 text-red-600' :
                          'bg-yellow-100 text-yellow-600'
                        }`}>
                          {backup.status === 'completed' ? <CheckCircle className="h-4 w-4" /> :
                           backup.status === 'failed' ? <AlertTriangle className="h-4 w-4" /> :
                           <Clock className="h-4 w-4" />}
                        </div>
                        <div>
                          <p className="font-medium">{backup.vmName}</p>
                          <p className="text-sm text-gray-600">
                            {backup.type} • {formatSize(backup.size)}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={backup.status === 'completed' ? 'default' : 'destructive'}>
                          {backup.status}
                        </Badge>
                        <p className="text-xs text-gray-600 mt-1">
                          {backup.createdAt.toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Policy Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {mockPolicies.map(policy => (
                    <div key={policy.id} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center space-x-3">
                        <div className={`p-1 rounded ${
                          policy.active ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-600'
                        }`}>
                          <Shield className="h-4 w-4" />
                        </div>
                        <div>
                          <p className="font-medium">{policy.name}</p>
                          <p className="text-sm text-gray-600">
                            {policy.targets.length} VMs • {policy.schedule.type}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={policy.active ? 'default' : 'secondary'}>
                          {policy.active ? 'Active' : 'Inactive'}
                        </Badge>
                        <p className="text-xs text-gray-600 mt-1">
                          Next: {policy.nextRun.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Running Jobs */}
          {isRunningBackup && (
            <Card>
              <CardHeader>
                <CardTitle>Active Backup Job</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
                      <div>
                        <p className="font-medium">Manual backup - web-server-01</p>
                        <p className="text-sm text-gray-600">Creating full system backup...</p>
                      </div>
                    </div>
                    <Button variant="outline" size="sm">
                      <Pause className="mr-2 h-4 w-4" />
                      Pause
                    </Button>
                  </div>
                  <Progress value={backupProgress} className="w-full" />
                  <div className="flex justify-between text-sm text-gray-600">
                    <span>Progress: {Math.round(backupProgress)}%</span>
                    <span>ETA: {Math.max(0, Math.round((100 - backupProgress) / 10))} minutes</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="backups" className="space-y-6">
          {/* Search and Filter */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search backups by VM name or policy..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                
                <Select value={filterStatus} onValueChange={setFilterStatus}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select value={filterType} onValueChange={setFilterType}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="full">Full</SelectItem>
                    <SelectItem value="incremental">Incremental</SelectItem>
                    <SelectItem value="differential">Differential</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Backup List */}
          <Card>
            <CardHeader>
              <CardTitle>Backup History</CardTitle>
              <CardDescription>View and manage your VM backups</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {filteredBackups.map(backup => (
                  <div key={backup.id} className="border rounded-lg">
                    <div 
                      className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50"
                      onClick={() => toggleBackupExpansion(backup.id)}
                    >
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {expandedBackups.has(backup.id) ? 
                            <ChevronDown className="h-4 w-4" /> : 
                            <ChevronRight className="h-4 w-4" />
                          }
                          <div className={`p-1 rounded ${
                            backup.status === 'completed' ? 'bg-green-100 text-green-600' :
                            backup.status === 'failed' ? 'bg-red-100 text-red-600' :
                            'bg-blue-100 text-blue-600'
                          }`}>
                            {backup.status === 'completed' ? <CheckCircle className="h-4 w-4" /> :
                             backup.status === 'failed' ? <AlertTriangle className="h-4 w-4" /> :
                             <RefreshCw className="h-4 w-4 animate-spin" />}
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex items-center space-x-2">
                            <h4 className="font-medium">{backup.vmName}</h4>
                            <Badge variant="outline">{backup.type}</Badge>
                            {backup.verificationStatus === 'verified' && (
                              <Badge variant="secondary">
                                <Shield className="mr-1 h-3 w-3" />
                                Verified
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-gray-600">
                            {backup.policyName} • {backup.createdAt.toLocaleDateString()} {backup.createdAt.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-right">
                        <div>
                          <p className="font-medium">{formatSize(backup.size)}</p>
                          <p className="text-sm text-gray-600">{backup.compression}% compressed</p>
                        </div>
                        <Badge variant={backup.status === 'completed' ? 'default' : 'destructive'}>
                          {backup.status}
                        </Badge>
                      </div>
                    </div>
                    
                    {expandedBackups.has(backup.id) && (
                      <div className="px-4 pb-4 border-t bg-gray-50">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                          <div>
                            <Label className="text-sm font-medium">Details</Label>
                            <div className="mt-2 space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span>Duration:</span>
                                <span>{formatDuration(backup.duration)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Location:</span>
                                <span>{backup.location || 'Local Storage'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Restore Points:</span>
                                <span>{backup.restorePoints}</span>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <Label className="text-sm font-medium">Tags</Label>
                            <div className="flex flex-wrap gap-1 mt-2">
                              {backup.tags.map(tag => (
                                <Badge key={tag} variant="outline" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div className="flex justify-end space-x-2">
                            {backup.status === 'completed' && (
                              <>
                                <Button variant="outline" size="sm">
                                  <Eye className="mr-2 h-4 w-4" />
                                  View
                                </Button>
                                <Button variant="outline" size="sm">
                                  <RotateCcw className="mr-2 h-4 w-4" />
                                  Restore
                                </Button>
                                <Button variant="outline" size="sm">
                                  <Download className="mr-2 h-4 w-4" />
                                  Download
                                </Button>
                              </>
                            )}
                            <Button variant="outline" size="sm">
                              <Trash2 className="mr-2 h-4 w-4" />
                              Delete
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="policies" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Backup Policies</CardTitle>
              <CardDescription>Configure automated backup schedules and retention policies</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockPolicies.map(policy => (
                  <Card key={policy.id} className="border-l-4 border-l-blue-500">
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-semibold text-lg">{policy.name}</h4>
                          <p className="text-gray-600">{policy.description}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant={policy.active ? 'default' : 'secondary'}>
                            {policy.active ? 'Active' : 'Inactive'}
                          </Badge>
                          <Button variant="outline" size="sm" onClick={() => runBackupNow(policy.id)}>
                            <Play className="mr-2 h-4 w-4" />
                            Run Now
                          </Button>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                          <Label className="text-sm font-medium">Schedule</Label>
                          <p className="text-sm mt-1">
                            {policy.schedule.type} at {policy.schedule.time}
                          </p>
                          <p className="text-xs text-gray-500">
                            Next: {policy.nextRun.toLocaleDateString()} {policy.nextRun.toLocaleTimeString()}
                          </p>
                        </div>
                        
                        <div>
                          <Label className="text-sm font-medium">Retention</Label>
                          <div className="text-sm mt-1 space-y-0.5">
                            {policy.retention.daily > 0 && <p>Daily: {policy.retention.daily}</p>}
                            {policy.retention.weekly > 0 && <p>Weekly: {policy.retention.weekly}</p>}
                            {policy.retention.monthly > 0 && <p>Monthly: {policy.retention.monthly}</p>}
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm font-medium">Storage</Label>
                          <div className="text-sm mt-1">
                            <p>{policy.storage.tier} tier</p>
                            <p>{policy.storage.location}</p>
                            {policy.storage.encryption && (
                              <Badge variant="outline" className="text-xs mt-1">
                                <Shield className="mr-1 h-3 w-3" />
                                Encrypted
                              </Badge>
                            )}
                          </div>
                        </div>
                        
                        <div>
                          <Label className="text-sm font-medium">Targets ({policy.targets.length})</Label>
                          <div className="text-sm mt-1">
                            {policy.targets.slice(0, 2).map(vmId => {
                              const vm = mockVMs.find(v => v.id === vmId);
                              return vm ? <p key={vmId}>{vm.name}</p> : null;
                            })}
                            {policy.targets.length > 2 && (
                              <p className="text-gray-500">+{policy.targets.length - 2} more</p>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex justify-end space-x-2 mt-4 pt-4 border-t">
                        <Button variant="outline" size="sm">
                          <Settings className="mr-2 h-4 w-4" />
                          Edit
                        </Button>
                        <Button variant="outline" size="sm">
                          <FileText className="mr-2 h-4 w-4" />
                          Logs
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="restore" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Restore Configuration */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>Restore Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Select Backup *</Label>
                    <Select
                      value={restoreConfig.backupId}
                      onValueChange={(value) => setRestoreConfig(prev => ({ ...prev, backupId: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Choose backup to restore" />
                      </SelectTrigger>
                      <SelectContent>
                        {mockBackups.filter(b => b.status === 'completed').map(backup => (
                          <SelectItem key={backup.id} value={backup.id}>
                            <div>
                              <div className="font-medium">{backup.vmName}</div>
                              <div className="text-sm text-gray-500">
                                {backup.createdAt.toLocaleDateString()} • {formatSize(backup.size)}
                              </div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Restore Type</Label>
                    <Select
                      value={restoreConfig.restoreType}
                      onValueChange={(value) => setRestoreConfig(prev => ({ ...prev, restoreType: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="full">Full VM Restore</SelectItem>
                        <SelectItem value="file_level">File-Level Restore</SelectItem>
                        <SelectItem value="application">Application Restore</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Target VM *</Label>
                    <Select
                      value={restoreConfig.targetVM}
                      onValueChange={(value) => setRestoreConfig(prev => ({ ...prev, targetVM: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select target VM" />
                      </SelectTrigger>
                      <SelectContent>
                        {mockVMs.map(vm => (
                          <SelectItem key={vm.id} value={vm.id}>
                            <div className="flex items-center justify-between w-full">
                              <span>{vm.name}</span>
                              <Badge variant={vm.status === 'running' ? 'default' : 'secondary'}>
                                {vm.status}
                              </Badge>
                            </div>
                          </SelectItem>
                        ))}
                        <SelectItem value="new">Create New VM</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {restoreConfig.targetVM === 'new' && (
                    <div className="space-y-2">
                      <Label>New VM Name</Label>
                      <Input
                        placeholder="Enter name for new VM"
                        value={restoreConfig.newVMName}
                        onChange={(e) => setRestoreConfig(prev => ({ ...prev, newVMName: e.target.value }))}
                      />
                    </div>
                  )}

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="verify-restore"
                      checked={restoreConfig.verifyAfterRestore}
                      onCheckedChange={(checked) => setRestoreConfig(prev => ({ 
                        ...prev, 
                        verifyAfterRestore: !!checked 
                      }))}
                    />
                    <Label htmlFor="verify-restore">Verify after restore</Label>
                  </div>

                  <Button 
                    onClick={startRestore} 
                    className="w-full"
                    disabled={!restoreConfig.backupId || !restoreConfig.targetVM || isRestoring}
                  >
                    {isRestoring ? (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                        Restoring...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="mr-2 h-4 w-4" />
                        Start Restore
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Restore Progress & Info */}
            <div className="lg:col-span-2">
              {isRestoring ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Restore in Progress</CardTitle>
                    <CardDescription>
                      Restoring {mockBackups.find(b => b.id === restoreConfig.backupId)?.vmName} from backup
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Progress</span>
                        <span>{Math.round(restoreProgress)}%</span>
                      </div>
                      <Progress value={restoreProgress} className="w-full" />
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Restore Type:</span>
                        <span className="ml-2 font-medium">{restoreConfig.restoreType}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Target:</span>
                        <span className="ml-2 font-medium">
                          {mockVMs.find(vm => vm.id === restoreConfig.targetVM)?.name || 'New VM'}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">ETA:</span>
                        <span className="ml-2 font-medium">
                          {Math.max(0, Math.round((100 - restoreProgress) / 5))} minutes
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Verification:</span>
                        <span className="ml-2 font-medium">
                          {restoreConfig.verifyAfterRestore ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    </div>

                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        Target VM will be temporarily unavailable during restore process.
                      </AlertDescription>
                    </Alert>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>Point-in-Time Recovery</CardTitle>
                    <CardDescription>
                      Select a backup and configure restore options
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {restoreConfig.backupId ? (
                      <div className="space-y-4">
                        {(() => {
                          const selectedBackupData = mockBackups.find(b => b.id === restoreConfig.backupId);
                          return selectedBackupData ? (
                            <>
                              <div className="p-4 bg-blue-50 rounded-lg border">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="font-medium">{selectedBackupData.vmName}</h4>
                                  <Badge variant="outline">{selectedBackupData.type} backup</Badge>
                                </div>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                  <div>
                                    <span className="text-gray-600">Created:</span>
                                    <span className="ml-2">{selectedBackupData.createdAt.toLocaleString()}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-600">Size:</span>
                                    <span className="ml-2">{formatSize(selectedBackupData.size)}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-600">Location:</span>
                                    <span className="ml-2">{selectedBackupData.location}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-600">Verification:</span>
                                    <div className="ml-2 inline-flex items-center">
                                      <CheckCircle className="h-3 w-3 text-green-600 mr-1" />
                                      <span>Verified</span>
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {restoreConfig.restoreType === 'file_level' && (
                                <Card>
                                  <CardHeader>
                                    <CardTitle className="text-lg">Select Files to Restore</CardTitle>
                                  </CardHeader>
                                  <CardContent>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                      {[
                                        '/etc/nginx/nginx.conf',
                                        '/var/www/html/index.html',
                                        '/home/user/documents/',
                                        '/var/log/application.log',
                                        '/etc/ssl/certificates/'
                                      ].map(path => (
                                        <div key={path} className="flex items-center space-x-2">
                                          <Checkbox
                                            id={path}
                                            checked={restoreConfig.selectedFiles.includes(path)}
                                            onCheckedChange={(checked) => {
                                              if (checked) {
                                                setRestoreConfig(prev => ({
                                                  ...prev,
                                                  selectedFiles: [...prev.selectedFiles, path]
                                                }));
                                              } else {
                                                setRestoreConfig(prev => ({
                                                  ...prev,
                                                  selectedFiles: prev.selectedFiles.filter(f => f !== path)
                                                }));
                                              }
                                            }}
                                          />
                                          <Label htmlFor={path} className="text-sm font-mono">
                                            {path}
                                          </Label>
                                        </div>
                                      ))}
                                    </div>
                                  </CardContent>
                                </Card>
                              )}
                            </>
                          ) : null;
                        })()}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Database className="mx-auto h-12 w-12 mb-4" />
                        <h3 className="text-lg font-medium mb-2">Select a Backup to Restore</h3>
                        <p>Choose from available backups to begin the restore process</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}