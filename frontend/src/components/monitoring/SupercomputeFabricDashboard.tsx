import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Globe, Cpu, MemoryStick, HardDrive, Network, Zap, Activity,
  Server, Cloud, Database, Users, TrendingUp, Gauge, Brain,
  CheckCircle, AlertCircle, Play, Pause, Square, BarChart3
} from 'lucide-react';
import { ResourceTreemap } from '@/components/visualizations/ResourceTreemap';
import { HeatmapChart } from '@/components/visualizations/HeatmapChart';
import { useSupercomputeFabricWebSocket } from '@/hooks/useWebSocket';
import type {
  ComputeJob,
  GlobalResourcePool,
  MemoryFabric,
  ProcessingFabric,
  FabricMetrics,
  ClusterResourceInventory
} from '@/lib/api/types';

interface SupercomputeFabricDashboardProps {
  globalView?: boolean;
  selectedClusters?: string[];
  refreshInterval?: number;
}

export const SupercomputeFabricDashboard: React.FC<SupercomputeFabricDashboardProps> = ({
  globalView = true,
  selectedClusters = [],
  refreshInterval = 5000,
}) => {
  const [viewMode, setViewMode] = useState<'global' | 'cluster' | 'job'>('global');
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'5min' | '1hr' | '24hr'>('1hr');

  // Real-time fabric data from WebSocket with typed payload
  const { data: fabricData, isConnected } = useSupercomputeFabricWebSocket();

  // Mock global resource pool data
  const mockGlobalResources: GlobalResourcePool = {
    totalClusters: 8,
    totalNodes: 1247,
    totalCPUCores: 49880,
    totalMemoryGB: 199520,
    totalStorageTB: 2984,
    totalGPUs: 624,
    utilization: {
      cpu: 68.5,
      memory: 72.1,
      storage: 54.3,
      gpu: 81.2,
      network: 45.7,
    },
    availability: {
      healthy: 1189,
      degraded: 42,
      failed: 16,
    },
    regions: [
      { name: 'US-East-1', clusters: 3, nodes: 468, utilization: 72 },
      { name: 'US-West-2', clusters: 2, nodes: 312, utilization: 64 },
      { name: 'EU-Central-1', clusters: 2, nodes: 298, utilization: 70 },
      { name: 'AP-South-1', clusters: 1, nodes: 169, utilization: 58 },
    ],
  };

  // Mock distributed compute jobs
  const mockComputeJobs: ComputeJob[] = [
    {
      id: 'job-ml-001',
      name: 'Large Language Model Training',
      type: 'ml_training',
      status: 'running',
      priority: 'high',
      startTime: new Date(Date.now() - 3600000 * 8).toISOString(),
      estimatedDuration: 14400, // 4 hours
      progress: 65,
      resourceAllocation: {
        cpuCores: 256,
        memoryGB: 1024,
        gpuCount: 32,
        storageGB: 500,
        networkBandwidthMbps: 10000,
      },
      clustersInvolved: ['us-east-gpu-01', 'us-east-gpu-02'],
      userSubmitted: 'research-team-ai',
      costEstimate: 450.75,
      completionETA: new Date(Date.now() + 3600000 * 2.5).toISOString(),
    },
    {
      id: 'job-sim-002',
      name: 'Quantum Simulation',
      type: 'simulation',
      status: 'queued',
      priority: 'medium',
      startTime: new Date(Date.now() + 900000).toISOString(), // 15 min from now
      estimatedDuration: 7200, // 2 hours
      progress: 0,
      resourceAllocation: {
        cpuCores: 512,
        memoryGB: 2048,
        gpuCount: 0,
        storageGB: 200,
        networkBandwidthMbps: 1000,
      },
      clustersInvolved: ['us-west-compute-01'],
      userSubmitted: 'quantum-research',
      costEstimate: 125.50,
      completionETA: new Date(Date.now() + 900000 + 7200000).toISOString(),
    },
    {
      id: 'job-render-003',
      name: 'Video Rendering Pipeline',
      type: 'rendering',
      status: 'running',
      priority: 'low',
      startTime: new Date(Date.now() - 1800000).toISOString(), // 30 min ago
      estimatedDuration: 3600, // 1 hour
      progress: 85,
      resourceAllocation: {
        cpuCores: 128,
        memoryGB: 256,
        gpuCount: 16,
        storageGB: 1000,
        networkBandwidthMbps: 5000,
      },
      clustersInvolved: ['eu-central-render-01'],
      userSubmitted: 'media-production',
      costEstimate: 85.25,
      completionETA: new Date(Date.now() + 540000).toISOString(), // 9 min
    },
    {
      id: 'job-analysis-004',
      name: 'Big Data Analytics',
      type: 'analytics',
      status: 'completed',
      priority: 'medium',
      startTime: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
      estimatedDuration: 5400, // 1.5 hours
      progress: 100,
      resourceAllocation: {
        cpuCores: 64,
        memoryGB: 512,
        gpuCount: 0,
        storageGB: 300,
        networkBandwidthMbps: 2000,
      },
      clustersInvolved: ['ap-south-data-01'],
      userSubmitted: 'analytics-team',
      costEstimate: 65.75,
      completionETA: new Date(Date.now() - 600000).toISOString(), // completed 10 min ago
    },
  ];

  // Mock memory fabric data
  const mockMemoryFabric: MemoryFabric = {
    totalMemory: 199520, // GB
    usedMemory: 143764, // GB
    availableMemory: 55756, // GB
    memoryPools: [
      {
        name: 'High-Performance Pool',
        totalGB: 51200,
        usedGB: 38912,
        type: 'ddr5',
        speed: 4800,
        clusters: ['us-east-gpu-01', 'us-east-gpu-02'],
        utilization: 76,
      },
      {
        name: 'Standard Pool',
        totalGB: 102400,
        usedGB: 73728,
        type: 'ddr4',
        speed: 3200,
        clusters: ['us-west-compute-01', 'eu-central-compute-01'],
        utilization: 72,
      },
      {
        name: 'Large Capacity Pool',
        totalGB: 45920,
        usedGB: 31124,
        type: 'ddr4',
        speed: 2666,
        clusters: ['ap-south-data-01'],
        utilization: 68,
      },
    ],
    cacheHitRate: 89.5,
    memoryCoherence: 97.2,
    swapUsage: 2.1,
  };

  // Mock processing fabric data
  const mockProcessingFabric: ProcessingFabric = {
    totalCores: 49880,
    activeCores: 34162,
    idleCores: 15718,
    totalGPUs: 624,
    activeGPUs: 507,
    idleGPUs: 117,
    workDistribution: {
      ml_training: 35,
      simulation: 25,
      analytics: 20,
      rendering: 12,
      other: 8,
    },
    loadBalancingEfficiency: 94.2,
    averageUtilization: 68.5,
    peakUtilization: 91.3,
    throughputOps: 1247000000, // operations per second
  };

  // Mock cluster resource inventories
  const mockClusterInventories: ClusterResourceInventory[] = [
    {
      clusterId: 'us-east-gpu-01',
      clusterName: 'US East GPU Cluster 01',
      region: 'us-east-1',
      nodes: 156,
      cpuCores: 6240,
      memoryGB: 24960,
      storageGB: 374400,
      gpuCount: 312,
      networkBandwidthGbps: 400,
      utilization: { cpu: 74, memory: 81, storage: 52, gpu: 89, network: 45 },
      status: 'healthy',
      specialization: 'gpu_compute',
    },
    {
      clusterId: 'us-east-gpu-02',
      clusterName: 'US East GPU Cluster 02',
      region: 'us-east-1',
      nodes: 128,
      cpuCores: 5120,
      memoryGB: 20480,
      storageGB: 307200,
      gpuCount: 256,
      networkBandwidthGbps: 320,
      utilization: { cpu: 68, memory: 76, storage: 48, gpu: 82, network: 41 },
      status: 'healthy',
      specialization: 'gpu_compute',
    },
    {
      clusterId: 'us-west-compute-01',
      clusterName: 'US West Compute Cluster 01',
      region: 'us-west-2',
      nodes: 184,
      cpuCores: 7360,
      memoryGB: 29440,
      storageGB: 441600,
      gpuCount: 0,
      networkBandwidthGbps: 200,
      utilization: { cpu: 62, memory: 69, storage: 55, gpu: 0, network: 38 },
      status: 'healthy',
      specialization: 'cpu_intensive',
    },
    {
      clusterId: 'eu-central-compute-01',
      clusterName: 'EU Central Compute Cluster 01',
      region: 'eu-central-1',
      nodes: 148,
      cpuCores: 5920,
      memoryGB: 23680,
      storageGB: 355200,
      gpuCount: 56,
      networkBandwidthGbps: 150,
      utilization: { cpu: 71, memory: 73, storage: 59, gpu: 45, network: 52 },
      status: 'degraded',
      specialization: 'mixed_workload',
    },
  ];

  // Use WebSocket data when available, fallback to mocks
  const globalResourcePool = useMemo(() =>
    fabricData?.globalResourcePool ?? mockGlobalResources,
    [fabricData]
  );

  const computeJobs = useMemo(() =>
    fabricData?.computeJobs ?? mockComputeJobs,
    [fabricData]
  );

  const fabricMetricsData = useMemo(() =>
    fabricData?.fabricMetrics ?? null,
    [fabricData]
  );

  // Calculate fabric-wide metrics
  const fabricMetrics = useMemo(() => {
    const totalJobs = computeJobs.length;
    const runningJobs = computeJobs.filter(job => job.status === 'running').length;
    const queuedJobs = computeJobs.filter(job => job.status === 'queued').length;
    const completedJobs = computeJobs.filter(job => job.status === 'completed').length;

    const totalCost = computeJobs.reduce((sum, job) => sum + job.costEstimate, 0);
    const avgJobDuration = computeJobs.reduce((sum, job) => sum + job.estimatedDuration, 0) / totalJobs;

    return {
      totalJobs,
      runningJobs,
      queuedJobs,
      completedJobs,
      totalCost,
      avgJobDuration,
      throughput: mockProcessingFabric.throughputOps,
      efficiency: mockProcessingFabric.loadBalancingEfficiency,
    };
  }, [computeJobs]);

  // Format bytes to human readable
  const formatBytes = (gb: number) => {
    if (gb >= 1024) {
      return `${(gb / 1024).toFixed(1)} TB`;
    }
    return `${gb.toFixed(0)} GB`;
  };

  // Format operations per second
  const formatOps = (ops: number) => {
    if (ops >= 1e9) return `${(ops / 1e9).toFixed(1)}B ops/s`;
    if (ops >= 1e6) return `${(ops / 1e6).toFixed(1)}M ops/s`;
    if (ops >= 1e3) return `${(ops / 1e3).toFixed(1)}K ops/s`;
    return `${ops} ops/s`;
  };

  // Get status icon and color
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="h-4 w-4 text-green-500" />;
      case 'queued': return <Pause className="h-4 w-4 text-yellow-500" />;
      case 'completed': return <CheckCircle className="h-4 w-4 text-blue-500" />;
      case 'failed': return <Square className="h-4 w-4 text-red-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Supercompute Fabric</h2>
          <p className="text-muted-foreground">
            Unified distributed computing infrastructure monitoring and management
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={viewMode} onValueChange={(v: any) => setViewMode(v)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="global">Global View</SelectItem>
              <SelectItem value="cluster">Cluster View</SelectItem>
              <SelectItem value="job">Job View</SelectItem>
            </SelectContent>
          </Select>
          {isConnected && (
            <Badge variant="outline">
              <Activity className="h-3 w-3 mr-1 text-green-500" />
              Live Data
            </Badge>
          )}
        </div>
      </div>

      {/* Global Resource Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{globalResourcePool.totalNodes.toLocaleString()}</div>
            <div className="flex gap-2 mt-2">
              <Badge variant="default" className="text-xs">
                {globalResourcePool.availability.healthy} Healthy
              </Badge>
              <Badge variant="secondary" className="text-xs">
                {globalResourcePool.availability.degraded} Degraded
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Cores</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{globalResourcePool.totalCPUCores.toLocaleString()}</div>
            <Progress value={globalResourcePool.utilization.cpu} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {globalResourcePool.utilization.cpu.toFixed(1)}% utilized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory</CardTitle>
            <MemoryStick className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(globalResourcePool.totalMemoryGB)}</div>
            <Progress value={globalResourcePool.utilization.memory} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {globalResourcePool.utilization.memory.toFixed(1)}% utilized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">GPUs</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{globalResourcePool.totalGPUs}</div>
            <Progress value={globalResourcePool.utilization.gpu} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {globalResourcePool.utilization.gpu.toFixed(1)}% utilized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Throughput</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatOps(mockProcessingFabric.throughputOps)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {mockProcessingFabric.loadBalancingEfficiency.toFixed(1)}% efficiency
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid grid-cols-6 w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="jobs">Compute Jobs</TabsTrigger>
          <TabsTrigger value="memory">Memory Fabric</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="clusters">Clusters</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Resource Treemap */}
            <Card>
              <CardHeader>
                <CardTitle>Global Resource Distribution</CardTitle>
                <CardDescription>Resource allocation across all clusters</CardDescription>
              </CardHeader>
              <CardContent>
                <ResourceTreemap
                  data={globalResourcePool.regions.map(region => ({
                    id: region.name,
                    name: region.name,
                    value: region.nodes,
                    utilization: region.utilization,
                    children: [
                      { id: `${region.name}-cpu`, name: 'CPU', value: region.nodes * 40, utilization: region.utilization },
                      { id: `${region.name}-mem`, name: 'Memory', value: region.nodes * 160, utilization: region.utilization + 5 },
                      { id: `${region.name}-gpu`, name: 'GPU', value: region.nodes * 4, utilization: region.utilization + 15 },
                    ],
                  }))}
                  height={300}
                />
              </CardContent>
            </Card>

            {/* Performance Heatmap */}
            <Card>
              <CardHeader>
                <CardTitle>Cluster Performance Heatmap</CardTitle>
                <CardDescription>Utilization patterns across infrastructure</CardDescription>
              </CardHeader>
              <CardContent>
                <HeatmapChart
                  data={mockClusterInventories.map(cluster => ({
                    x: cluster.region,
                    y: cluster.clusterName.split(' ').slice(-2).join(' '), // Simplified name
                    value: cluster.utilization.cpu,
                    metadata: {
                      nodes: cluster.nodes,
                      specialization: cluster.specialization,
                      status: cluster.status,
                    },
                  }))}
                  height={300}
                />
              </CardContent>
            </Card>
          </div>

          {/* Regional Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Regional Distribution</CardTitle>
              <CardDescription>Compute resources by geographic region</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {globalResourcePool.regions.map((region) => (
                  <div key={region.name} className="p-4 rounded-lg bg-accent/50">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">{region.name}</h4>
                      <Globe className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Clusters:</span>
                        <span>{region.clusters}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Nodes:</span>
                        <span>{region.nodes}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Utilization:</span>
                        <span>{region.utilization}%</span>
                      </div>
                      <Progress value={region.utilization} className="h-2" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Fabric Health */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Fabric Health</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Network Coherence</span>
                    <div className="flex items-center gap-2">
                      <Progress value={mockMemoryFabric.memoryCoherence} className="w-20" />
                      <span className="text-sm">{mockMemoryFabric.memoryCoherence.toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Load Balance</span>
                    <div className="flex items-center gap-2">
                      <Progress value={mockProcessingFabric.loadBalancingEfficiency} className="w-20" />
                      <span className="text-sm">{mockProcessingFabric.loadBalancingEfficiency.toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Cache Hit Rate</span>
                    <div className="flex items-center gap-2">
                      <Progress value={mockMemoryFabric.cacheHitRate} className="w-20" />
                      <span className="text-sm">{mockMemoryFabric.cacheHitRate.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Job Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Jobs:</span>
                    <span>{fabricMetrics.totalJobs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Running:</span>
                    <span className="text-green-600">{fabricMetrics.runningJobs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Queued:</span>
                    <span className="text-yellow-600">{fabricMetrics.queuedJobs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Completed:</span>
                    <span className="text-blue-600">{fabricMetrics.completedJobs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Cost:</span>
                    <span>${fabricMetrics.totalCost.toFixed(2)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Workload Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(mockProcessingFabric.workDistribution).map(([type, percentage]) => (
                    <div key={type} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{type.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={percentage} className="w-16" />
                        <span className="text-sm w-8 text-right">{percentage}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="jobs" className="space-y-4">
          <div className="space-y-4">
            {computeJobs.map((job) => (
              <Card key={job.id} className={selectedJob === job.id ? 'ring-2 ring-primary' : ''}>
                <CardHeader
                  className="cursor-pointer"
                  onClick={() => setSelectedJob(selectedJob === job.id ? null : job.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(job.status)}
                      <div>
                        <CardTitle className="text-base">{job.name}</CardTitle>
                        <CardDescription>
                          {job.type.replace('_', ' ')} • {job.userSubmitted}
                        </CardDescription>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <Badge variant={job.priority === 'high' ? 'destructive' :
                                   job.priority === 'medium' ? 'default' : 'secondary'}>
                        {job.priority.toUpperCase()}
                      </Badge>
                      <div className="text-right">
                        <p className="font-medium">${job.costEstimate.toFixed(2)}</p>
                        <p className="text-sm text-muted-foreground">
                          {job.status === 'completed' ? 'Completed' :
                           job.status === 'running' ? `${job.progress}%` : 'Queued'}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                {selectedJob === job.id && (
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-medium mb-3">Resource Allocation</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">CPU Cores:</span>
                            <span>{job.resourceAllocation.cpuCores}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Memory:</span>
                            <span>{formatBytes(job.resourceAllocation.memoryGB)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">GPUs:</span>
                            <span>{job.resourceAllocation.gpuCount || 'None'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Storage:</span>
                            <span>{formatBytes(job.resourceAllocation.storageGB)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Network:</span>
                            <span>{job.resourceAllocation.networkBandwidthMbps} Mbps</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium mb-3">Job Details</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Started:</span>
                            <span>{new Date(job.startTime).toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Duration:</span>
                            <span>{Math.floor(job.estimatedDuration / 3600)}h {Math.floor((job.estimatedDuration % 3600) / 60)}m</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">ETA:</span>
                            <span>{job.status === 'completed' ? 'Completed' : new Date(job.completionETA).toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Clusters:</span>
                            <span>{job.clustersInvolved.length}</span>
                          </div>
                        </div>

                        {job.status === 'running' && (
                          <div className="mt-4">
                            <div className="flex justify-between text-sm mb-1">
                              <span>Progress</span>
                              <span>{job.progress}%</span>
                            </div>
                            <Progress value={job.progress} />
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="mt-6">
                      <h4 className="text-sm font-medium mb-2">Involved Clusters</h4>
                      <div className="flex flex-wrap gap-2">
                        {job.clustersInvolved.map((clusterId) => (
                          <Badge key={clusterId} variant="outline">
                            {clusterId}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="memory" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardHeader>
                <CardTitle>Memory Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">{formatBytes(mockMemoryFabric.totalMemory)}</div>
                  <div className="text-sm text-muted-foreground">Total Capacity</div>
                  <Progress value={(mockMemoryFabric.usedMemory / mockMemoryFabric.totalMemory) * 100} />
                  <div className="flex justify-between text-xs">
                    <span>Used: {formatBytes(mockMemoryFabric.usedMemory)}</span>
                    <span>Free: {formatBytes(mockMemoryFabric.availableMemory)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cache Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Cache Hit Rate</span>
                      <span>{mockMemoryFabric.cacheHitRate.toFixed(1)}%</span>
                    </div>
                    <Progress value={mockMemoryFabric.cacheHitRate} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Memory Coherence</span>
                      <span>{mockMemoryFabric.memoryCoherence.toFixed(1)}%</span>
                    </div>
                    <Progress value={mockMemoryFabric.memoryCoherence} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Swap Usage</span>
                      <span>{mockMemoryFabric.swapUsage.toFixed(1)}%</span>
                    </div>
                    <Progress value={mockMemoryFabric.swapUsage} />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Pools</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-lg font-semibold">{mockMemoryFabric.memoryPools.length} Active Pools</div>
                <div className="text-sm text-muted-foreground mb-3">Distributed across clusters</div>
                <div className="space-y-2">
                  {mockMemoryFabric.memoryPools.slice(0, 2).map((pool, idx) => (
                    <div key={idx} className="flex justify-between text-sm">
                      <span>{pool.name.split(' ')[0]}:</span>
                      <span>{formatBytes(pool.totalGB)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Memory Pool Details */}
          <div className="space-y-4">
            {mockMemoryFabric.memoryPools.map((pool, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{pool.name}</CardTitle>
                      <CardDescription>
                        {pool.type.toUpperCase()} @ {pool.speed} MHz • {pool.clusters.length} clusters
                      </CardDescription>
                    </div>
                    <Badge variant={pool.utilization > 80 ? 'destructive' : 'default'}>
                      {pool.utilization}% Utilized
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-sm font-medium mb-3">Capacity</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Total:</span>
                          <span>{formatBytes(pool.totalGB)}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Used:</span>
                          <span>{formatBytes(pool.usedGB)}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Available:</span>
                          <span>{formatBytes(pool.totalGB - pool.usedGB)}</span>
                        </div>
                        <Progress value={pool.utilization} className="mt-2" />
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-3">Associated Clusters</h4>
                      <div className="flex flex-wrap gap-2">
                        {pool.clusters.map((clusterId) => (
                          <Badge key={clusterId} variant="outline" className="text-xs">
                            {clusterId}
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Memory Type:</span>
                          <span className="font-medium">{pool.type.toUpperCase()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Speed:</span>
                          <span className="font-medium">{pool.speed} MHz</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="processing" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>CPU Cores</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{mockProcessingFabric.totalCores.toLocaleString()}</div>
                <div className="text-sm text-muted-foreground">Total Available</div>
                <div className="mt-2 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Active:</span>
                    <span className="text-green-600">{mockProcessingFabric.activeCores.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Idle:</span>
                    <span className="text-muted-foreground">{mockProcessingFabric.idleCores.toLocaleString()}</span>
                  </div>
                </div>
                <Progress value={(mockProcessingFabric.activeCores / mockProcessingFabric.totalCores) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>GPU Units</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{mockProcessingFabric.totalGPUs}</div>
                <div className="text-sm text-muted-foreground">Accelerators</div>
                <div className="mt-2 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Active:</span>
                    <span className="text-green-600">{mockProcessingFabric.activeGPUs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Idle:</span>
                    <span className="text-muted-foreground">{mockProcessingFabric.idleGPUs}</span>
                  </div>
                </div>
                <Progress value={(mockProcessingFabric.activeGPUs / mockProcessingFabric.totalGPUs) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Throughput</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatOps(mockProcessingFabric.throughputOps)}</div>
                <div className="text-sm text-muted-foreground">Operations/sec</div>
                <div className="mt-2">
                  <div className="flex justify-between text-sm">
                    <span>Avg Utilization:</span>
                    <span>{mockProcessingFabric.averageUtilization.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Peak Utilization:</span>
                    <span>{mockProcessingFabric.peakUtilization.toFixed(1)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{mockProcessingFabric.loadBalancingEfficiency.toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">Load Balancing</div>
                <Progress value={mockProcessingFabric.loadBalancingEfficiency} className="mt-2" />
                <div className="text-xs text-muted-foreground mt-1">
                  Optimal distribution across fabric
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Workload Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Workload Distribution</CardTitle>
              <CardDescription>Current processing workload breakdown by type</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(mockProcessingFabric.workDistribution).map(([type, percentage]) => (
                  <div key={type}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize">{type.replace('_', ' ')}</span>
                      <span>{percentage}%</span>
                    </div>
                    <Progress value={percentage} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>P2P Network Fabric</CardTitle>
              <CardDescription>Inter-cluster connectivity and performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <Network className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Network fabric visualization will be shown here</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Integration with NetworkTopology component for P2P mesh visualization
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clusters" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {mockClusterInventories.map((cluster) => (
              <Card key={cluster.clusterId}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{cluster.clusterName}</CardTitle>
                      <CardDescription>
                        {cluster.region} • {cluster.nodes} nodes • {cluster.specialization.replace('_', ' ')}
                      </CardDescription>
                    </div>
                    <Badge variant={cluster.status === 'healthy' ? 'default' : 'destructive'}>
                      {cluster.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium mb-2">Resources</h4>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">CPU Cores:</span>
                            <span>{cluster.cpuCores.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Memory:</span>
                            <span>{formatBytes(cluster.memoryGB)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Storage:</span>
                            <span>{formatBytes(cluster.storageGB / 1024)} TB</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">GPUs:</span>
                            <span>{cluster.gpuCount || 'None'}</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium mb-2">Utilization</h4>
                        <div className="space-y-2">
                          <div>
                            <div className="flex justify-between text-xs mb-1">
                              <span>CPU</span>
                              <span>{cluster.utilization.cpu}%</span>
                            </div>
                            <Progress value={cluster.utilization.cpu} className="h-1" />
                          </div>
                          <div>
                            <div className="flex justify-between text-xs mb-1">
                              <span>Memory</span>
                              <span>{cluster.utilization.memory}%</span>
                            </div>
                            <Progress value={cluster.utilization.memory} className="h-1" />
                          </div>
                          <div>
                            <div className="flex justify-between text-xs mb-1">
                              <span>Storage</span>
                              <span>{cluster.utilization.storage}%</span>
                            </div>
                            <Progress value={cluster.utilization.storage} className="h-1" />
                          </div>
                          {cluster.gpuCount > 0 && (
                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span>GPU</span>
                                <span>{cluster.utilization.gpu}%</span>
                              </div>
                              <Progress value={cluster.utilization.gpu} className="h-1" />
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="pt-2 border-t">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-muted-foreground">Network Bandwidth:</span>
                        <span>{cluster.networkBandwidthGbps} Gbps</span>
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-muted-foreground">Specialization:</span>
                        <Badge variant="outline" className="text-xs">
                          {cluster.specialization.replace('_', ' ')}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SupercomputeFabricDashboard;