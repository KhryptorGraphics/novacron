export type ApiError = { code: string; message: string };

export type Pagination = {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  sortBy?: "name" | "createdAt" | "state";
  sortDir?: "asc" | "desc";
};

export type ApiEnvelope<T> = {
  data: T | null;
  error: ApiError | null;
  pagination?: Pagination;
};

export type VM = {
  id: string;
  name: string;
  state: string;
  node_id: string;
  created_at: string;
  updated_at: string;
};

// Admin-specific types
export type User = {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'moderator' | 'user' | 'viewer';
  status: 'active' | 'suspended' | 'pending' | 'disabled';
  created_at: string;
  updated_at: string;
  last_login?: string;
  login_count: number;
  organization?: string;
  two_factor_enabled: boolean;
  email_verified: boolean;
  permissions: string[];
  avatar_url?: string;
};

export type UserSession = {
  id: string;
  user_id: string;
  ip_address: string;
  user_agent: string;
  created_at: string;
  last_activity: string;
  is_current: boolean;
  location?: string;
};

export type AuditLogEntry = {
  id: string;
  user_id: string;
  user_name: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details: Record<string, any>;
  ip_address: string;
  user_agent: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
};

export type SystemMetrics = {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_in: number;
  network_out: number;
  active_connections: number;
  response_time: number;
};

export type SecurityAlert = {
  id: string;
  type: 'authentication' | 'access_control' | 'data_breach' | 'malware' | 'network';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  source: string;
  timestamp: string;
  status: 'new' | 'investigating' | 'resolved' | 'false_positive';
  affected_resources: string[];
  remediation_steps?: string[];
};

export type SystemConfiguration = {
  id: string;
  category: string;
  key: string;
  value: any;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'json' | 'enum';
  options?: string[];
  required: boolean;
  sensitive: boolean;
  updated_at: string;
  updated_by: string;
};

export type ResourceQuota = {
  id: string;
  user_id?: string;
  organization_id?: string;
  resource_type: 'cpu' | 'memory' | 'storage' | 'network' | 'vms';
  limit: number;
  used: number;
  unit: string;
  period?: 'hourly' | 'daily' | 'monthly';
  created_at: string;
  updated_at: string;
};

export type VmTemplate = {
  id: string;
  name: string;
  description: string;
  os: string;
  os_version: string;
  cpu_cores: number;
  memory_mb: number;
  disk_gb: number;
  network_config: Record<string, any>;
  is_public: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
  usage_count: number;
};

export type PerformanceReport = {
  id: string;
  report_type: 'daily' | 'weekly' | 'monthly';
  period_start: string;
  period_end: string;
  metrics: {
    avg_cpu: number;
    avg_memory: number;
    avg_disk_usage: number;
    total_requests: number;
    avg_response_time: number;
    error_rate: number;
    uptime_percentage: number;
  };
  trends: {
    cpu_trend: 'increasing' | 'decreasing' | 'stable';
    memory_trend: 'increasing' | 'decreasing' | 'stable';
    response_time_trend: 'increasing' | 'decreasing' | 'stable';
  };
  generated_at: string;
};

// Distributed Network Types
export type NetworkNode = {
  id: string;
  name: string;
  type: 'vm' | 'host' | 'storage' | 'network' | 'service' | 'cluster' | 'federation';
  status: 'healthy' | 'warning' | 'error' | 'unknown' | 'migrating';
  clusterId?: string;
  region?: string;
  metrics?: {
    cpuUsage?: number;
    memoryUsage?: number;
    diskUsage?: number;
    networkIn?: number;
    networkOut?: number;
    [key: string]: number | undefined;
  };
  position?: { x: number; y: number; fixed?: boolean };
};

export type NetworkEdge = {
  source: string;
  target: string;
  type: 'network' | 'storage' | 'dependency' | 'cluster' | 'federation' | 'migration';
  metrics?: {
    latency?: number;
    bandwidth?: number;
    packetLoss?: number;
    utilization?: number;
    capacity?: number;
    qos?: 'high' | 'medium' | 'low';
  };
  animated?: boolean;
  bidirectional?: boolean;
};

export type ClusterTopology = {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  clusters?: ClusterInfo[];
};

export type FederationTopology = {
  clusters: ClusterInfo[];
  federationLinks: NetworkEdge[];
  globalMetrics: NetworkMetrics;
};

export type NetworkMetrics = {
  totalBandwidth: number;
  utilization: number;
  latency: number;
  packetLoss: number;
  qos: 'high' | 'medium' | 'low';
  timestamp: string;
};

export type BandwidthMetrics = {
  // Summary fields used by dashboard UI
  timestamp?: string;
  totalBandwidth: number; // Mbps
  usedBandwidth: number; // Mbps
  availableBandwidth: number; // Mbps
  utilizationPercent: number; // %
  uploadRate: number; // Mbps
  downloadRate: number; // Mbps
  peakUsage: number; // Mbps
  averageUsage: number; // Mbps

  // Original structured data (optional for backward compatibility)
  interfaces?: Array<{
    id: string;
    name: string;
    utilization: number;
    capacity: number;
    inbound: number;
    outbound: number;
  }>;
  aggregated?: {
    totalCapacity: number;
    totalUtilization: number;
    peakUtilization: number;
    averageLatency: number;
  };
  history?: Array<{
    timestamp: string;
    utilization: number;
    throughput: number;
  }>;
};

export type QoSMetrics = {
  // Direct metrics used by dashboard
  latency: number; // ms
  jitter: number; // ms
  packetLoss: number; // %
  throughput: number; // Mbps
  qosCompliance: number; // %
  priorityQueues: {
    high: { usage: number; latency: number };
    medium: { usage: number; latency: number };
    low: { usage: number; latency: number };
  };

  // Original structured data (optional for backward compatibility)
  interfaceId?: string;
  policies?: Array<{
    id: string;
    name: string;
    priority: number;
    bandwidth: number;
    latency: number;
    jitter: number;
  }>;
  performance?: {
    guaranteedBandwidth: number;
    actualBandwidth: number;
    latency: number;
    jitter: number;
    packetLoss: number;
  };
};

export type NetworkInterface = {
  id: string;
  name: string;
  type: 'ethernet' | 'fiber' | 'wireless' | 'virtual' | 'vpn';
  status: 'up' | 'down' | 'degraded';
  capacity: number;
  utilization: number;
  qosEnabled: boolean;
  nodeId?: string;

  // Fields used in UI (optional where not universally available)
  speed: number; // Mbps
  duplex: 'full' | 'half';
  mtu: number; // bytes
  errorRate: number; // percentage as decimal
  packetsSent: number;
  packetsReceived: number;
  bytesSent: number;
  bytesReceived: number;
};

export type TrafficFlow = {
  id: string;
  source: string;
  target: string; // also available as "destination" for consistency
  destination?: string; // alias for target
  protocol: string;
  port: number;
  bandwidth: number; // Mbps
  latency?: number; // ms
  packets: number;
  bytes: number; // total bytes transferred
  duration?: number; // seconds
  startTime: string; // ISO timestamp
};

export type ResourcePrediction = {
  resourceType: 'cpu' | 'memory' | 'storage' | 'network';
  currentUsage: number;
  predictedUsage: number;
  confidence: number;
  timeHorizon?: string;
  recommendations?: string[];

  // Additional fields used in dashboard
  trend?: 'increasing' | 'decreasing' | 'stable';
  timeToCapacity?: string;
  recommendation?: string;
  factors?: string[];
};

export type WorkloadPattern = {
  id: string;
  name: string;
  clusterId: string;
  pattern: Array<{
    hour: number;
    cpu: number;
    memory: number;
    network: number;
  }>;
  confidence: number;
  seasonality: 'daily' | 'weekly' | 'monthly';
};

export type MigrationPrediction = {
  vmId: string;
  sourceNode: string;
  targetNode: string;
  confidence: number;
  reason: string;
  estimatedDuration: number;
  riskScore: number;
};

export type ScalingRecommendation = {
  resourceType: string;
  action: 'scale_up' | 'scale_down' | 'migrate' | 'optimize';
  confidence: number;
  impact: string;
  priority: 'low' | 'medium' | 'high';
};

export type PerformanceOptimization = {
  type?: string;
  description: string;
  impact?: number | 'low' | 'medium' | 'high';
  implementation?: string;
  cost?: number;

  // Additional fields used in dashboard
  id?: string;
  category?: string;
  title?: string;
  effort?: 'low' | 'medium' | 'high';
  savings?: number;
  performanceGain?: number;
  estimatedTime?: string;
  prerequisites?: string[];
};

export type ComputeJob = {
  id: string;
  name: string;
  type: 'ml_training' | 'simulation' | 'rendering' | 'analysis' | 'batch_processing';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'high' | 'medium' | 'low' | number;
  resources?: {
    cpu: number;
    memory: number;
    gpu?: number;
    storage: number;
  };
  createdAt?: string;
  updatedAt?: string;
  duration?: number;
  progress?: number;

  // Additional fields used in dashboard
  startTime?: string;
  estimatedDuration?: number;
  resourceAllocation?: {
    cpuCores: number;
    memoryGB: number;
    gpuCount: number;
    storageGB: number;
    networkBandwidthMbps: number;
  };
  clustersInvolved?: string[];
  userSubmitted?: string;
  costEstimate?: number;
  completionETA?: string;
};

export type GlobalResourcePool = {
  totalClusters?: number;
  totalNodes?: number;
  totalCPUCores?: number;
  totalMemoryGB?: number;
  totalStorageTB?: number;
  totalGPUs?: number;
  utilization?: {
    cpu: number;
    memory: number;
    storage: number;
    gpu: number;
    network: number;
  };
  availability?: {
    healthy: number;
    degraded: number;
    failed: number;
  };
  regions?: Array<{
    name: string;
    clusters: number;
    nodes: number;
    utilization: number;
  }>;
  // Original fields (optional for backward compatibility)
  totalCpu?: number;
  totalMemory?: number;
  totalGpu?: number;
  totalStorage?: number;
  availableCpu?: number;
  availableMemory?: number;
  availableGpu?: number;
  availableStorage?: number;
};

export type MemoryFabric = {
  totalCapacity: number;
  usedCapacity: number;
  availableCapacity: number;
  nodes: Array<{
    id: string;
    capacity: number;
    used: number;
    bandwidth: number;
    latency: number;
  }>;
  coherencyProtocol: string;
  performance: {
    throughput: number;
    latency: number;
    coherencyOverhead: number;
  };
};

export type ProcessingFabric = {
  totalCores: number;
  activeCores: number;
  nodes: Array<{
    id: string;
    cores: number;
    utilization: number;
    temperature: number;
    power: number;
  }>;
  interconnect: {
    topology: string;
    bandwidth: number;
    latency: number;
  };
  workloadDistribution: {
    balanced: number;
    imbalanced: number;
    efficiency: number;
  };
};

export type FabricMetrics = {
  memory: MemoryFabric;
  processing: ProcessingFabric;
  interconnect: {
    totalBandwidth: number;
    utilization: number;
    latency: number;
    errorRate: number;
  };
  performance: {
    throughput: number;
    efficiency: number;
    scalability: number;
  };
};

export type ClusterInfo = {
  id: string;
  name: string;
  region: string;
  nodeCount: number;
  status: 'healthy' | 'degraded' | 'error';
  bounds?: { x: number; y: number; width: number; height: number };
};

export type CrossClusterMigration = {
  id: string;
  vmId: string;
  sourceCluster: string;
  targetCluster: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number;
  startTime: string;
  estimatedCompletion?: string;
  bandwidth: number;
  transferredData: number;
  totalData: number;
};

export type FederationStatus = {
  connectedClusters: number;
  totalClusters: number;
  syncHealth: 'healthy' | 'degraded' | 'error';
  lastSync: string;
  activeMigrations: number;
  dataConsistency: number;
};

export type ClusterResourceInventory = {
  clusterId: string;
  resources: {
    cpu: { total: number; used: number; available: number };
    memory: { total: number; used: number; available: number };
    storage: { total: number; used: number; available: number };
    gpu?: { total: number; used: number; available: number };
  };
  workloads: {
    vms: number;
    containers: number;
    jobs: number;
  };
};

export type TopologyUpdate = {
  type: 'node_added' | 'node_removed' | 'node_updated' | 'edge_added' | 'edge_removed';
  nodeId?: string;
  edgeId?: string;
  data: any;
  timestamp: string;
};

export type BandwidthUpdate = {
  interfaceId: string;
  utilization: number;
  throughput: number;
  timestamp: string;
};

export type PerformanceUpdatePayload = {
  resourcePredictions: ResourcePrediction[];
  workloadPatterns: WorkloadPattern[];
  migrationPredictions: MigrationPrediction[];
  scalingRecommendations: ScalingRecommendation[];
  performanceOptimizations: PerformanceOptimization[];
};

export type FabricUpdatePayload = {
  globalResourcePool: GlobalResourcePool;
  computeJobs: ComputeJob[];
  fabricMetrics: FabricMetrics;
};

export type PerformanceUpdate = {
  nodeId: string;
  metrics: {
    cpu: number;
    memory: number;
    network: number;
    storage?: number;
  };
  timestamp: string;
};

export type FabricUpdate = {
  type: 'job_started' | 'job_completed' | 'resource_allocated' | 'resource_freed';
  jobId?: string;
  resourceType?: string;
  amount?: number;
  timestamp: string;
};

export type ModelMetrics = {
  modelId: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingTime: number;
  lastTrained: string;
};

export type PredictionResult = {
  model: string;
  prediction: any;
  confidence: number;
  timestamp: string;
  metadata?: Record<string, any>;
};

export type ModelTrainingStatus = {
  modelId: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  progress: number;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  estimatedCompletion?: string;
};

export type OptimizationRecommendation = {
  type: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  effort: 'low' | 'medium' | 'high';
  priority: number;
  implementation: string;
  expectedBenefit: string;
};

export type NetworkConfig = {
  defaultQoS: 'high' | 'medium' | 'low';
  bandwidthAllocation: {
    guaranteed: number;
    burst: number;
    priority: number;
  };
  latencyThresholds: {
    warning: number;
    critical: number;
  };
  monitoringInterval: number;
};

export type BandwidthThresholds = {
  warning: number;
  critical: number;
  interface: string;
  direction: 'inbound' | 'outbound' | 'both';
};

export type PredictionConfig = {
  modelId: string;
  updateInterval: number;
  confidence: number;
  timeHorizon: string;
  features: string[];
};

export type FabricSettings = {
  maxConcurrentJobs: number;
  resourceAllocationStrategy: 'round_robin' | 'least_loaded' | 'priority_based';
  loadBalancing: boolean;
  autoScaling: boolean;
  maintenanceWindow: {
    start: string;
    end: string;
    timezone: string;
  };
};

// Additional types needed by API client
export type SecurityPolicy = {
  id: string;
  name: string;
  type: 'access_control' | 'data_protection' | 'network_security';
  rules: Array<{
    id: string;
    condition: string;
    action: 'allow' | 'deny' | 'log';
    priority: number;
  }>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
};

export type ComplianceReport = {
  id: string;
  type: 'security' | 'privacy' | 'audit';
  status: 'passed' | 'failed' | 'warning';
  score: number;
  findings: Array<{
    severity: 'critical' | 'high' | 'medium' | 'low';
    description: string;
    recommendation: string;
  }>;
  generated_at: string;
  period: {
    start: string;
    end: string;
  };
};

export type AuditLog = AuditLogEntry;

