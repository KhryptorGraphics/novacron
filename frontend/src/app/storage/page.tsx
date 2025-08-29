"use client";

import { useState, useEffect } from "react";
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

// Mock storage data
const mockStoragePools = [
  {
    id: "pool-01",
    name: "Primary SSD Pool",
    type: "ssd",
    status: "healthy",
    totalSize: 2000,
    usedSize: 850,
    availableSize: 1150,
    usage: 42.5,
    nodes: ["node-01", "node-02"],
    redundancy: "raid10",
    iops: 15000,
    throughput: 2.5,
    vmsCount: 12
  },
  {
    id: "pool-02", 
    name: "Backup HDD Pool",
    type: "hdd",
    status: "healthy",
    totalSize: 8000,
    usedSize: 3200,
    availableSize: 4800,
    usage: 40,
    nodes: ["node-03", "node-04"],
    redundancy: "raid6",
    iops: 500,
    throughput: 0.8,
    vmsCount: 8
  },
  {
    id: "pool-03",
    name: "Archive Storage",
    type: "hdd",
    status: "warning",
    totalSize: 12000,
    usedSize: 9600,
    availableSize: 2400,
    usage: 80,
    nodes: ["node-05"],
    redundancy: "raid5",
    iops: 200,
    throughput: 0.5,
    vmsCount: 3
  }
];

const mockStorageVolumes = [
  {
    id: "vol-001",
    name: "web-server-01-disk",
    vmId: "vm-001",
    vmName: "web-server-01",
    pool: "pool-01",
    size: 100,
    used: 32,
    usage: 32,
    type: "system",
    status: "active",
    created: "2024-01-15T10:30:00Z"
  },
  {
    id: "vol-002",
    name: "database-primary-disk",
    vmId: "vm-002", 
    vmName: "database-primary",
    pool: "pool-01",
    size: 500,
    used: 280,
    usage: 56,
    type: "system",
    status: "active",
    created: "2024-01-10T14:22:00Z"
  },
  {
    id: "vol-003",
    name: "backup-volume-01",
    vmId: "vm-004",
    vmName: "backup-server",
    pool: "pool-02",
    size: 1000,
    used: 450,
    usage: 45,
    type: "data",
    status: "active",
    created: "2024-01-08T16:45:00Z"
  },
  {
    id: "vol-004",
    name: "archived-data",
    vmId: null,
    vmName: null,
    pool: "pool-03",
    size: 2000,
    used: 1800,
    usage: 90,
    type: "archive",
    status: "mounted",
    created: "2023-12-01T09:00:00Z"
  }
];

export default function StoragePage() {
  const [storagePools, setStoragePools] = useState(mockStoragePools);
  const [storageVolumes, setStorageVolumes] = useState(mockStorageVolumes);
  const [searchQuery, setSearchQuery] = useState("");
  const [poolFilter, setPoolFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");

  // Filter volumes based on search and filters
  const filteredVolumes = storageVolumes.filter(volume => {
    const matchesSearch = volume.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         (volume.vmName && volume.vmName.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesPool = poolFilter === "all" || volume.pool === poolFilter;
    const matchesType = typeFilter === "all" || volume.type === typeFilter;
    
    return matchesSearch && matchesPool && matchesType;
  });

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
    if (bytes === 0) return '0 GB';
    const k = 1000;
    const sizes = ['GB', 'TB', 'PB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const totalStats = {
    totalStorage: storagePools.reduce((acc, pool) => acc + pool.totalSize, 0),
    usedStorage: storagePools.reduce((acc, pool) => acc + pool.usedSize, 0),
    availableStorage: storagePools.reduce((acc, pool) => acc + pool.availableSize, 0),
    totalVolumes: storageVolumes.length,
    activeVolumes: storageVolumes.filter(vol => vol.status === "active").length
  };

  const overallUsage = (totalStats.usedStorage / totalStats.totalStorage) * 100;

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Storage Management</h1>
          <p className="text-muted-foreground">Monitor and manage your storage infrastructure</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Add Volume
          </Button>
          <Button>
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
            {storagePools.map((pool) => (
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
                      <span>{formatBytes(pool.usedSize)} used</span>
                      <span>{formatBytes(pool.totalSize)} total</span>
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
                    {storagePools.map(pool => (
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
                {filteredVolumes.map((volume) => (
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
                      {storagePools.find(p => p.id === volume.pool)?.name}
                    </TableCell>
                    <TableCell>{formatBytes(volume.size)}</TableCell>
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
                        <Button variant="ghost" size="sm" className="text-red-600">
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
                  {storagePools.map(pool => (
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
                  {storagePools.map(pool => (
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