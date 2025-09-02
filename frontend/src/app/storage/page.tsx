"use client";

import { useState, useEffect } from "react";
import { storageApi, type StoragePool, type StorageVolume, type StorageMetrics } from "@/lib/api/storage";

// Disable static generation for this page
export const dynamic = 'force-dynamic';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  HardDrive, 
  Database, 
  FolderOpen, 
  Plus, 
  Search, 
  Download,
  Upload,
  Trash2,
  Settings,
  AlertCircle,
  CheckCircle,
  Activity,
  BarChart3,
  Layers,
  Archive,
  Shield,
  Zap
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

// Extended types for UI-specific data
interface ExtendedStoragePool extends StoragePool {
  usage: number;
  nodes: string[];
  redundancy: string;
  iops: number;
  throughput: number;
  vmsCount: number;
  status: string;
}

interface ExtendedStorageVolume extends Omit<StorageVolume, 'pool_id'> {
  pool_id: string;
  vmName?: string;
  used: number;
  usage: number;
  type: string;
  status: string;
}

export default function StoragePage() {
  const [storagePools, setStoragePools] = useState<ExtendedStoragePool[]>([]);
  const [storageVolumes, setStorageVolumes] = useState<ExtendedStorageVolume[]>([]);
  const [storageMetrics, setStorageMetrics] = useState<StorageMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [poolFilter, setPoolFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");

  // Load storage data from API
  useEffect(() => {
    const loadStorageData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load pools, volumes, and metrics in parallel
        const [pools, volumes, metrics] = await Promise.all([
          storageApi.getStoragePools(),
          storageApi.getStorageVolumes(),
          storageApi.getStorageMetrics()
        ]);
        
        // Transform pools to include UI-specific data
        const extendedPools: ExtendedStoragePool[] = pools.map(pool => ({
          ...pool,
          usage: storageApi.formatUsagePercentage(pool.used_space, pool.total_space),
          nodes: pool.metadata?.nodes ? pool.metadata.nodes.split(',') : ['node-01'],
          redundancy: pool.metadata?.redundancy || 'raid1',
          iops: parseInt(pool.metadata?.iops || '1000'),
          throughput: parseFloat(pool.metadata?.throughput || '1.0'),
          vmsCount: parseInt(pool.metadata?.vms_count || '0'),
          status: pool.metadata?.status || 'healthy'
        }));
        
        // Transform volumes to include UI-specific data
        const extendedVolumes: ExtendedStorageVolume[] = volumes.map(volume => ({
          ...volume,
          vmName: volume.metadata?.vm_name || undefined,
          used: volume.allocation,
          usage: storageApi.formatUsagePercentage(volume.allocation, volume.capacity),
          type: volume.metadata?.type || 'system',
          status: volume.metadata?.status || 'active'
        }));
        
        setStoragePools(extendedPools);
        setStorageVolumes(extendedVolumes);
        setStorageMetrics(metrics);
      } catch (err) {
        console.error('Failed to load storage data:', err);
        setError('Failed to load storage data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    loadStorageData();
  }, []);

  // Filter volumes based on search and filters
  const filteredVolumes = Array.isArray(storageVolumes) ? storageVolumes.filter(volume => {
    if (!volume) return false;
    const matchesSearch = (volume.name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                         (volume.vmName && volume.vmName.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesPool = poolFilter === "all" || volume.pool === poolFilter;
    const matchesType = typeFilter === "all" || volume.type === typeFilter;
    
    return matchesSearch && matchesPool && matchesType;
  }) : [];

  // Filter volumes based on search and filters
  const filteredVolumesByPool = filteredVolumes.filter(volume => {
    const matchesPool = poolFilter === "all" || volume.pool_id === poolFilter;
    return matchesPool;
  });

  // Handler functions for actions
  const handleCreatePool = async () => {
    // TODO: Implement create pool dialog
    console.log('Create pool clicked');
  };

  const handleCreateVolume = async () => {
    // TODO: Implement create volume dialog
    console.log('Create volume clicked');
  };

  const handleDeleteVolume = async (volumeId: string) => {
    if (confirm('Are you sure you want to delete this volume?')) {
      try {
        await storageApi.deleteStorageVolume(volumeId);
        // Refresh volumes
        const volumes = await storageApi.getStorageVolumes();
        const extendedVolumes: ExtendedStorageVolume[] = volumes.map(volume => ({
          ...volume,
          vmName: volume.metadata?.vm_name || undefined,
          used: volume.allocation,
          usage: storageApi.formatUsagePercentage(volume.allocation, volume.capacity),
          type: volume.metadata?.type || 'system',
          status: volume.metadata?.status || 'active'
        }));
        setStorageVolumes(extendedVolumes);
      } catch (err) {
        console.error('Failed to delete volume:', err);
        alert('Failed to delete volume. Please try again.');
      }
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy": 
      case "active": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "warning": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "error": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "mounted": return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
      case "active": return <CheckCircle className="h-4 w-4" />;
      case "warning": return <AlertCircle className="h-4 w-4" />;
      case "error": return <AlertCircle className="h-4 w-4" />;
      case "mounted": return <FolderOpen className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "ssd": return <Zap className="h-4 w-4" />;
      case "hdd": return <HardDrive className="h-4 w-4" />;
      default: return <Database className="h-4 w-4" />;
    }
  };

  const getUsageColor = (usage: number) => {
    if (usage < 70) return "text-green-600 dark:text-green-400";
    if (usage < 85) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const formatBytes = (bytes: number) => {
    return storageApi.formatBytes(bytes);
  };

  const totalStats = storageMetrics ? {
    totalStorage: storageMetrics.total_capacity_bytes,
    usedStorage: storageMetrics.used_capacity_bytes,
    availableStorage: storageMetrics.available_capacity_bytes,
    totalVolumes: storageMetrics.total_volumes,
    activeVolumes: storageMetrics.active_volumes
  } : {
    totalStorage: storagePools.reduce((acc, pool) => acc + pool.total_space, 0),
    usedStorage: storagePools.reduce((acc, pool) => acc + pool.used_space, 0),
    availableStorage: storagePools.reduce((acc, pool) => acc + (pool.total_space - pool.used_space), 0),
    totalVolumes: storageVolumes.length,
    activeVolumes: storageVolumes.filter(vol => vol.status === "active").length
  };

  const overallUsage = totalStats.totalStorage > 0 ? (totalStats.usedStorage / totalStats.totalStorage) * 100 : 0;

  // Show loading state
  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex justify-center items-center min-h-[400px]">
          <div className="text-center">
            <Activity className="h-8 w-8 animate-spin mx-auto mb-4" />
            <p className="text-muted-foreground">Loading storage data...</p>
          </div>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Card className="max-w-md mx-auto mt-20">
          <CardHeader>
            <CardTitle className="flex items-center text-red-600">
              <AlertCircle className="h-5 w-5 mr-2" />
              Error Loading Storage Data
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-4">{error}</p>
            <Button onClick={() => window.location.reload()} className="w-full">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Storage Management</h1>
          <p className="text-muted-foreground">Monitor and manage your storage infrastructure</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => handleCreateVolume()}>
            <Plus className="h-4 w-4 mr-2" />
            Add Volume
          </Button>
          <Button onClick={() => handleCreatePool()}>
            <Plus className="h-4 w-4 mr-2" />
            Create Pool
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Storage</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(totalStats.totalStorage)}</div>
            <p className="text-xs text-muted-foreground">
              {formatBytes(totalStats.usedStorage)} used
            </p>
            <div className="mt-2">
              <Progress value={overallUsage} className="h-1" />
            </div>
            <p className={`text-xs mt-1 ${getUsageColor(overallUsage)}`}>
              {overallUsage.toFixed(1)}% utilized
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Pools</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{storagePools.length}</div>
            <p className="text-xs text-muted-foreground">
              {storagePools.filter(p => p.status === "healthy").length} healthy
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Volumes</CardTitle>
            <FolderOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalStats.activeVolumes}</div>
            <p className="text-xs text-muted-foreground">
              of {totalStats.totalVolumes} total
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Space</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(totalStats.availableStorage)}</div>
            <p className="text-xs text-muted-foreground">
              Ready for allocation
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="pools" className="w-full">
        <TabsList>
          <TabsTrigger value="pools">Storage Pools</TabsTrigger>
          <TabsTrigger value="volumes">Volumes</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>
        
        <TabsContent value="pools" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-2">
            {Array.isArray(storagePools) && storagePools.map((pool) => (
              <Card key={pool.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        {getTypeIcon(pool.type)}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{pool.name}</CardTitle>
                        <p className="text-sm text-muted-foreground">
                          {pool.redundancy.toUpperCase()} • {pool.nodes.join(", ")}
                        </p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(pool.status)}>
                      {getStatusIcon(pool.status)}
                      <span className="ml-1 capitalize">{pool.status}</span>
                    </Badge>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {/* Storage Usage */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Storage Usage</span>
                      <span className={getUsageColor(pool.usage)}>
                        {pool.usage.toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={pool.usage} className="h-2" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{formatBytes(pool.used_space)} used</span>
                      <span>{formatBytes(pool.total_space)} total</span>
                    </div>
                  </div>
                  
                  {/* Performance Metrics */}
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="text-center">
                      <div className="font-medium">{pool.iops.toLocaleString()}</div>
                      <div className="text-muted-foreground">IOPS</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium">{pool.throughput} GB/s</div>
                      <div className="text-muted-foreground">Throughput</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium">{pool.vmsCount}</div>
                      <div className="text-muted-foreground">VMs</div>
                    </div>
                  </div>
                  
                  {/* Actions */}
                  <div className="flex items-center justify-between pt-2 border-t">
                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4 mr-1" />
                        Configure
                      </Button>
                      <Button variant="outline" size="sm">
                        <BarChart3 className="h-4 w-4 mr-1" />
                        Metrics
                      </Button>
                    </div>
                    <Button variant="ghost" size="sm" className="text-muted-foreground">
                      <Activity className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="volumes" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row gap-4 items-center">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search volumes by name or VM..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Select value={poolFilter} onValueChange={setPoolFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Pool" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Pools</SelectItem>
                    {Array.isArray(storagePools) && storagePools.map(pool => (
                      <SelectItem key={pool.id} value={pool.id}>{pool.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={typeFilter} onValueChange={setTypeFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                    <SelectItem value="data">Data</SelectItem>
                    <SelectItem value="archive">Archive</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Volumes Table */}
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Volume Name</TableHead>
                  <TableHead>VM</TableHead>
                  <TableHead>Pool</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Usage</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Array.isArray(filteredVolumes) && filteredVolumes.map((volume) => (
                  <TableRow key={volume.id}>
                    <TableCell className="font-medium">{volume.name}</TableCell>
                    <TableCell>
                      {volume.vmName ? (
                        <span>{volume.vmName}</span>
                      ) : (
                        <span className="text-muted-foreground">Unattached</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {storagePools.find(p => p.id === volume.pool_id)?.name}
                    </TableCell>
                    <TableCell>{formatBytes(volume.capacity)}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <div className="w-16">
                          <Progress value={volume.usage} className="h-1" />
                        </div>
                        <span className={`text-sm ${getUsageColor(volume.usage)}`}>
                          {volume.usage}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="capitalize">
                        {volume.type}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge className={getStatusColor(volume.status)}>
                        {getStatusIcon(volume.status)}
                        <span className="ml-1 capitalize">{volume.status}</span>
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-1">
                        <Button variant="ghost" size="sm">
                          <Settings className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="text-red-600"
                          onClick={() => handleDeleteVolume(volume.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
        
        <TabsContent value="performance" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>IOPS Performance</CardTitle>
                <CardDescription>Input/Output operations per second</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Array.isArray(storagePools) && storagePools.map(pool => (
                    <div key={pool.id} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{pool.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24">
                          <Progress 
                            value={(pool.iops / 20000) * 100} 
                            className="h-2"
                          />
                        </div>
                        <span className="text-sm font-mono w-16 text-right">
                          {pool.iops.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Throughput Performance</CardTitle>
                <CardDescription>Data transfer rates in GB/s</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Array.isArray(storagePools) && storagePools.map(pool => (
                    <div key={pool.id} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{pool.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24">
                          <Progress 
                            value={(pool.throughput / 5) * 100} 
                            className="h-2"
                          />
                        </div>
                        <span className="text-sm font-mono w-16 text-right">
                          {pool.throughput} GB/s
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}