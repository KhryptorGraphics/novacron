'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { 
  HardDrive,
  Plus,
  Trash2,
  Copy,
  Download,
  Upload,
  Settings,
  Shield,
  Clock,
  AlertCircle,
  CheckCircle,
  Database,
  Server,
  Layers,
  Archive,
  RefreshCw,
  ZapOff,
  Zap,
  ChevronRight,
  FolderOpen,
  File,
  Lock,
  Unlock,
  Calendar as CalendarIcon,
  MoreVertical,
  Expand,
  Shrink,
  Activity
} from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { format } from 'date-fns';

interface Volume {
  id: string;
  name: string;
  size: number;
  used: number;
  type: 'ssd' | 'hdd' | 'nvme';
  status: 'online' | 'offline' | 'degraded' | 'syncing';
  encrypted: boolean;
  mountPoint: string;
  vmAttached: string | null;
  created: string;
  lastSnapshot: string | null;
  replicationTarget: string | null;
  iops: number;
  throughput: number;
}

interface StoragePool {
  id: string;
  name: string;
  totalSize: number;
  usedSize: number;
  availableSize: number;
  volumes: number;
  type: 'local' | 'network' | 'cloud';
  redundancy: 'none' | 'raid1' | 'raid5' | 'raid10';
  health: 'healthy' | 'warning' | 'critical';
  devices: Array<{
    name: string;
    size: number;
    status: 'online' | 'offline' | 'failed';
  }>;
}

interface Snapshot {
  id: string;
  volumeId: string;
  name: string;
  size: number;
  created: string;
  type: 'manual' | 'scheduled' | 'automatic';
  retentionDays: number;
  protected: boolean;
}

interface BackupJob {
  id: string;
  name: string;
  source: string;
  destination: string;
  schedule: string;
  lastRun: string;
  nextRun: string;
  status: 'active' | 'paused' | 'failed';
  retention: number;
  compression: boolean;
  encryption: boolean;
}

const StorageManagementUI: React.FC = () => {
  const [volumes, setVolumes] = useState<Volume[]>([]);
  const [pools, setPools] = useState<StoragePool[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [backupJobs, setBackupJobs] = useState<BackupJob[]>([]);
  const [selectedVolume, setSelectedVolume] = useState<Volume | null>(null);
  const [createVolumeDialog, setCreateVolumeDialog] = useState(false);
  const [snapshotDialog, setSnapshotDialog] = useState(false);
  const [backupDialog, setBackupDialog] = useState(false);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());

  // New volume form state
  const [newVolume, setNewVolume] = useState({
    name: '',
    size: 100,
    type: 'ssd',
    pool: '',
    encrypted: false,
    mountPoint: '/mnt/',
    enableSnapshots: false,
    snapshotSchedule: 'daily',
    enableReplication: false,
    replicationTarget: ''
  });

  // Mock data
  useEffect(() => {
    const mockVolumes: Volume[] = [
      {
        id: 'vol-001',
        name: 'system-root',
        size: 500,
        used: 350,
        type: 'nvme',
        status: 'online',
        encrypted: true,
        mountPoint: '/',
        vmAttached: 'vm-001',
        created: '2024-01-15',
        lastSnapshot: '2024-03-20',
        replicationTarget: 'backup-site-01',
        iops: 15000,
        throughput: 2000
      },
      {
        id: 'vol-002',
        name: 'database-storage',
        size: 1000,
        used: 750,
        type: 'ssd',
        status: 'online',
        encrypted: true,
        mountPoint: '/var/lib/postgresql',
        vmAttached: 'vm-002',
        created: '2024-01-20',
        lastSnapshot: '2024-03-21',
        replicationTarget: null,
        iops: 10000,
        throughput: 1500
      },
      {
        id: 'vol-003',
        name: 'backup-storage',
        size: 2000,
        used: 1200,
        type: 'hdd',
        status: 'syncing',
        encrypted: false,
        mountPoint: '/backup',
        vmAttached: null,
        created: '2024-02-01',
        lastSnapshot: '2024-03-19',
        replicationTarget: 'offsite-backup',
        iops: 500,
        throughput: 150
      }
    ];

    const mockPools: StoragePool[] = [
      {
        id: 'pool-001',
        name: 'Fast Storage Pool',
        totalSize: 5000,
        usedSize: 2300,
        availableSize: 2700,
        volumes: 8,
        type: 'local',
        redundancy: 'raid10',
        health: 'healthy',
        devices: [
          { name: 'nvme0', size: 1000, status: 'online' },
          { name: 'nvme1', size: 1000, status: 'online' },
          { name: 'nvme2', size: 1000, status: 'online' },
          { name: 'nvme3', size: 1000, status: 'online' }
        ]
      },
      {
        id: 'pool-002',
        name: 'Archive Pool',
        totalSize: 10000,
        usedSize: 6500,
        availableSize: 3500,
        volumes: 12,
        type: 'network',
        redundancy: 'raid5',
        health: 'warning',
        devices: [
          { name: 'hdd0', size: 2000, status: 'online' },
          { name: 'hdd1', size: 2000, status: 'online' },
          { name: 'hdd2', size: 2000, status: 'failed' },
          { name: 'hdd3', size: 2000, status: 'online' },
          { name: 'hdd4', size: 2000, status: 'online' }
        ]
      }
    ];

    const mockSnapshots: Snapshot[] = [
      {
        id: 'snap-001',
        volumeId: 'vol-001',
        name: 'Daily Backup',
        size: 350,
        created: '2024-03-21T02:00:00Z',
        type: 'scheduled',
        retentionDays: 7,
        protected: false
      },
      {
        id: 'snap-002',
        volumeId: 'vol-002',
        name: 'Pre-upgrade Snapshot',
        size: 750,
        created: '2024-03-20T14:30:00Z',
        type: 'manual',
        retentionDays: 30,
        protected: true
      }
    ];

    const mockBackupJobs: BackupJob[] = [
      {
        id: 'backup-001',
        name: 'Daily Database Backup',
        source: 'vol-002',
        destination: 's3://backups/database',
        schedule: '0 2 * * *',
        lastRun: '2024-03-21T02:00:00Z',
        nextRun: '2024-03-22T02:00:00Z',
        status: 'active',
        retention: 30,
        compression: true,
        encryption: true
      },
      {
        id: 'backup-002',
        name: 'Weekly Full Backup',
        source: 'pool-001',
        destination: 'backup-site-01',
        schedule: '0 3 * * 0',
        lastRun: '2024-03-17T03:00:00Z',
        nextRun: '2024-03-24T03:00:00Z',
        status: 'active',
        retention: 90,
        compression: true,
        encryption: true
      }
    ];

    setVolumes(mockVolumes);
    setPools(mockPools);
    setSnapshots(mockSnapshots);
    setBackupJobs(mockBackupJobs);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': case 'healthy': case 'active': return 'bg-green-500';
      case 'offline': case 'critical': case 'failed': return 'bg-red-500';
      case 'degraded': case 'warning': case 'paused': return 'bg-yellow-500';
      case 'syncing': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'nvme': return <Zap className="h-4 w-4" />;
      case 'ssd': return <HardDrive className="h-4 w-4" />;
      case 'hdd': return <Database className="h-4 w-4" />;
      default: return <Server className="h-4 w-4" />;
    }
  };

  // Chart data
  const storageDistribution = (pools || []).map(pool => ({
    name: pool.name,
    value: pool.usedSize,
    total: pool.totalSize,
    fill: pool.health === 'healthy' ? '#10b981' : pool.health === 'warning' ? '#f59e0b' : '#ef4444'
  }));

  const performanceTrend = [
    { time: '00:00', iops: 5000, throughput: 800 },
    { time: '04:00', iops: 3000, throughput: 500 },
    { time: '08:00', iops: 12000, throughput: 1800 },
    { time: '12:00', iops: 15000, throughput: 2000 },
    { time: '16:00', iops: 14000, throughput: 1900 },
    { time: '20:00', iops: 8000, throughput: 1200 },
    { time: '24:00', iops: 4000, throughput: 600 }
  ];

  const handleCreateVolume = () => {
    // Implementation for creating volume
    console.log('Creating volume:', newVolume);
    setCreateVolumeDialog(false);
  };

  const handleCreateSnapshot = (volumeId: string) => {
    // Implementation for creating snapshot
    console.log('Creating snapshot for volume:', volumeId);
    setSnapshotDialog(false);
  };

  const handleDeleteVolume = (volumeId: string) => {
    // Implementation for deleting volume
    console.log('Deleting volume:', volumeId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Storage Management</h1>
          <p className="text-muted-foreground">Manage volumes, snapshots, and backup policies</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setCreateVolumeDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Volume
          </Button>
          <Button variant="outline" onClick={() => setBackupDialog(true)}>
            <Archive className="mr-2 h-4 w-4" />
            Backup Policy
          </Button>
        </div>
      </div>

      {/* Storage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Storage</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15TB</div>
            <p className="text-xs text-muted-foreground">8.8TB used (59%)</p>
            <Progress value={59} className="h-2 mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Volumes</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{volumes.length}</div>
            <p className="text-xs text-muted-foreground">
              {volumes.filter(v => v.status === 'online').length} online
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Snapshots</CardTitle>
            <Copy className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{snapshots.length}</div>
            <p className="text-xs text-muted-foreground">
              {snapshots.filter(s => s.protected).length} protected
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Backup Jobs</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{backupJobs.length}</div>
            <p className="text-xs text-muted-foreground">
              {backupJobs.filter(j => j.status === 'active').length} active
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="volumes" className="space-y-4">
        <TabsList>
          <TabsTrigger value="volumes">Volumes</TabsTrigger>
          <TabsTrigger value="pools">Storage Pools</TabsTrigger>
          <TabsTrigger value="snapshots">Snapshots</TabsTrigger>
          <TabsTrigger value="backup">Backup & Recovery</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="volumes" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Volume List */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Volumes</CardTitle>
                  <CardDescription>Manage storage volumes and their configurations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {(volumes || []).map((volume) => (
                    <div
                      key={volume.id}
                      className={`border rounded-lg p-4 hover:bg-accent cursor-pointer transition-colors ${
                        selectedVolume?.id === volume.id ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => setSelectedVolume(volume)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center gap-2">
                            <div className="p-1 rounded bg-muted">
                              {getTypeIcon(volume.type)}
                            </div>
                            <h3 className="font-semibold">{volume.name}</h3>
                            <Badge variant="outline" className={`${getStatusColor(volume.status)} text-white`}>
                              {volume.status}
                            </Badge>
                            {volume.encrypted && (
                              <Lock className="h-3 w-3 text-muted-foreground" />
                            )}
                          </div>

                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Size</p>
                              <p className="font-medium">{volume.size}GB</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Used</p>
                              <p className="font-medium">{volume.used}GB ({Math.round(volume.used / volume.size * 100)}%)</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Mount Point</p>
                              <p className="font-medium font-mono text-xs">{volume.mountPoint}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Attached To</p>
                              <p className="font-medium">{volume.vmAttached || 'Unattached'}</p>
                            </div>
                          </div>

                          <Progress value={volume.used / volume.size * 100} className="h-2" />

                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>IOPS: {volume.iops.toLocaleString()}</span>
                            <span>Throughput: {volume.throughput}MB/s</span>
                            {volume.lastSnapshot && (
                              <span>Last snapshot: {new Date(volume.lastSnapshot).toLocaleDateString()}</span>
                            )}
                          </div>
                        </div>

                        <div className="flex flex-col gap-1">
                          <Button size="icon" variant="ghost" onClick={(e) => {
                            e.stopPropagation();
                            setSnapshotDialog(true);
                          }}>
                            <Copy className="h-4 w-4" />
                          </Button>
                          <Button size="icon" variant="ghost">
                            <Settings className="h-4 w-4" />
                          </Button>
                          <Button size="icon" variant="ghost" onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteVolume(volume.id);
                          }}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Volume Details */}
            <div>
              {selectedVolume ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Volume Details</CardTitle>
                    <CardDescription>{selectedVolume.name}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Type</span>
                        <span className="text-sm font-medium uppercase">{selectedVolume.type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Created</span>
                        <span className="text-sm font-medium">{selectedVolume.created}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Encryption</span>
                        <Badge variant={selectedVolume.encrypted ? 'default' : 'outline'}>
                          {selectedVolume.encrypted ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Replication</span>
                        <span className="text-sm font-medium">
                          {selectedVolume.replicationTarget || 'None'}
                        </span>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold mb-2">Storage Usage</h4>
                      <ResponsiveContainer width="100%" height={150}>
                        <PieChart>
                          <Pie
                            data={[
                              { name: 'Used', value: selectedVolume.used },
                              { name: 'Free', value: selectedVolume.size - selectedVolume.used }
                            ]}
                            cx="50%"
                            cy="50%"
                            innerRadius={40}
                            outerRadius={60}
                            paddingAngle={2}
                            dataKey="value"
                          >
                            <Cell fill="#3b82f6" />
                            <Cell fill="#e5e7eb" />
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-2">
                      <Button className="w-full" variant="outline">
                        <Expand className="mr-2 h-4 w-4" />
                        Resize Volume
                      </Button>
                      <Button className="w-full" variant="outline">
                        <Shield className="mr-2 h-4 w-4" />
                        Configure Encryption
                      </Button>
                      <Button className="w-full" variant="outline">
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Setup Replication
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center h-[400px] text-muted-foreground">
                    Select a volume to view details
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="pools" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {(pools || []).map((pool) => (
              <Card key={pool.id}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle>{pool.name}</CardTitle>
                      <CardDescription>
                        {pool.type} • {pool.redundancy.toUpperCase()}
                      </CardDescription>
                    </div>
                    <Badge className={getStatusColor(pool.health)}>
                      {pool.health}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Capacity</span>
                        <span>{pool.usedSize}GB / {pool.totalSize}GB</span>
                      </div>
                      <Progress value={pool.usedSize / pool.totalSize * 100} className="h-2" />
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Volumes</p>
                        <p className="font-medium">{pool.volumes}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Available</p>
                        <p className="font-medium">{pool.availableSize}GB</p>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold mb-2">Storage Devices</h4>
                      <div className="space-y-1">
                        {(pool.devices || []).map((device) => (
                          <div key={device.name} className="flex items-center justify-between text-sm">
                            <span className="font-mono">{device.name}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-muted-foreground">{device.size}GB</span>
                              <Badge variant="outline" className={`text-xs ${
                                device.status === 'online' ? 'text-green-600' :
                                device.status === 'failed' ? 'text-red-600' : 'text-gray-600'
                              }`}>
                                {device.status}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="snapshots" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Snapshots</CardTitle>
                  <CardDescription>Manage volume snapshots and restore points</CardDescription>
                </div>
                <Button onClick={() => setSnapshotDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Create Snapshot
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(snapshots || []).map((snapshot) => (
                  <div key={snapshot.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Copy className="h-4 w-4 text-muted-foreground" />
                        <h4 className="font-semibold">{snapshot.name}</h4>
                        {snapshot.protected && (
                          <Lock className="h-3 w-3 text-muted-foreground" />
                        )}
                        <Badge variant="outline">
                          {snapshot.type}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>Volume: {snapshot.volumeId}</span>
                        <span>Size: {snapshot.size}GB</span>
                        <span>Created: {new Date(snapshot.created).toLocaleString()}</span>
                        <span>Retention: {snapshot.retentionDays} days</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">
                        <Upload className="mr-2 h-4 w-4" />
                        Restore
                      </Button>
                      <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Export
                      </Button>
                      {!snapshot.protected && (
                        <Button variant="ghost" size="sm">
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="backup" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Backup Jobs</CardTitle>
                  <CardDescription>Configure and monitor backup policies</CardDescription>
                </div>
                <Button onClick={() => setBackupDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Create Backup Job
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(backupJobs || []).map((job) => (
                  <div key={job.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Archive className="h-4 w-4 text-muted-foreground" />
                          <h4 className="font-semibold">{job.name}</h4>
                          <Badge className={getStatusColor(job.status)}>
                            {job.status}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">Source</p>
                            <p className="font-medium font-mono text-xs">{job.source}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Destination</p>
                            <p className="font-medium font-mono text-xs">{job.destination}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Schedule</p>
                            <p className="font-medium">{job.schedule}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Retention</p>
                            <p className="font-medium">{job.retention} days</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>Last run: {new Date(job.lastRun).toLocaleString()}</span>
                          <span>Next run: {new Date(job.nextRun).toLocaleString()}</span>
                          {job.compression && <Badge variant="outline" className="text-xs">Compressed</Badge>}
                          {job.encryption && <Badge variant="outline" className="text-xs">Encrypted</Badge>}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button size="icon" variant="ghost">
                          {job.status === 'active' ? 
                            <ZapOff className="h-4 w-4" /> : 
                            <Zap className="h-4 w-4" />
                          }
                        </Button>
                        <Button size="icon" variant="ghost">
                          <Settings className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>IOPS & Throughput</CardTitle>
                <CardDescription>24-hour performance metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Line 
                      yAxisId="left" 
                      type="monotone" 
                      dataKey="iops" 
                      stroke="#3b82f6" 
                      name="IOPS"
                      strokeWidth={2}
                    />
                    <Line 
                      yAxisId="right" 
                      type="monotone" 
                      dataKey="throughput" 
                      stroke="#10b981" 
                      name="Throughput (MB/s)"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Storage Distribution</CardTitle>
                <CardDescription>Usage across storage pools</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={storageDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" name="Used (GB)" fill="#3b82f6" />
                    <Bar dataKey="total" name="Total (GB)" fill="#e5e7eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Create Volume Dialog */}
      <Dialog open={createVolumeDialog} onOpenChange={setCreateVolumeDialog}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Create New Volume</DialogTitle>
            <DialogDescription>
              Configure a new storage volume with desired specifications
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="volume-name">Volume Name</Label>
                <Input
                  id="volume-name"
                  value={newVolume.name}
                  onChange={(e) => setNewVolume({ ...newVolume, name: e.target.value })}
                  placeholder="my-volume"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="mount-point">Mount Point</Label>
                <Input
                  id="mount-point"
                  value={newVolume.mountPoint}
                  onChange={(e) => setNewVolume({ ...newVolume, mountPoint: e.target.value })}
                  placeholder="/mnt/volume"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Size (GB)</Label>
              <div className="flex items-center gap-4">
                <Slider
                  value={[newVolume.size]}
                  onValueChange={(value) => setNewVolume({ ...newVolume, size: value[0] })}
                  min={10}
                  max={5000}
                  step={10}
                  className="flex-1"
                />
                <span className="w-20 text-right font-medium">{newVolume.size} GB</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Storage Type</Label>
              <RadioGroup
                value={newVolume.type}
                onValueChange={(value) => setNewVolume({ ...newVolume, type: value })}
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="nvme" id="nvme" />
                  <Label htmlFor="nvme" className="flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    NVMe (Highest Performance)
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="ssd" id="ssd" />
                  <Label htmlFor="ssd" className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4" />
                    SSD (Balanced)
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="hdd" id="hdd" />
                  <Label htmlFor="hdd" className="flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    HDD (Cost Effective)
                  </Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="encryption">Enable Encryption</Label>
                <Switch
                  id="encryption"
                  checked={newVolume.encrypted}
                  onCheckedChange={(checked) => setNewVolume({ ...newVolume, encrypted: checked })}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="snapshots">Enable Snapshots</Label>
                <Switch
                  id="snapshots"
                  checked={newVolume.enableSnapshots}
                  onCheckedChange={(checked) => setNewVolume({ ...newVolume, enableSnapshots: checked })}
                />
              </div>

              {newVolume.enableSnapshots && (
                <div className="space-y-2 pl-4">
                  <Label>Snapshot Schedule</Label>
                  <Select
                    value={newVolume.snapshotSchedule}
                    onValueChange={(value) => setNewVolume({ ...newVolume, snapshotSchedule: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hourly">Hourly</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="flex items-center justify-between">
                <Label htmlFor="replication">Enable Replication</Label>
                <Switch
                  id="replication"
                  checked={newVolume.enableReplication}
                  onCheckedChange={(checked) => setNewVolume({ ...newVolume, enableReplication: checked })}
                />
              </div>

              {newVolume.enableReplication && (
                <div className="space-y-2 pl-4">
                  <Label>Replication Target</Label>
                  <Select
                    value={newVolume.replicationTarget}
                    onValueChange={(value) => setNewVolume({ ...newVolume, replicationTarget: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select target" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="backup-site-01">Backup Site 01</SelectItem>
                      <SelectItem value="offsite-backup">Offsite Backup</SelectItem>
                      <SelectItem value="cloud-storage">Cloud Storage</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateVolumeDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateVolume}>
              Create Volume
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default StorageManagementUI;