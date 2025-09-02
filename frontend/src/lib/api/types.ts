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

