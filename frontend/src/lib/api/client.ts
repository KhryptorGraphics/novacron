// API Client for NovaCron Enhanced API
export class ApiClient {
  private baseURL: string;
  private wsURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';
    this.wsURL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8091';

    // Try to get token from localStorage if available
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('novacron_token') || null;
    }
  }

  setToken(token: string | null) {
    this.token = token;
    if (typeof window !== 'undefined' && token) {
      localStorage.setItem('novacron_token', token);
    } else if (typeof window !== 'undefined' && !token) {
      localStorage.removeItem('novacron_token');
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('novacron_token');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    if (response.status === 204) {
      return {} as T; // No content
    }

    return response.json();
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : null,
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : null,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  // WebSocket connection
  connectWebSocket(path: string): WebSocket | null {
    try {
      const url = `${this.wsURL}${path}`;
      const ws = new WebSocket(url);

      // Add authentication if token is available
      if (this.token) {
        ws.addEventListener('open', () => {
          ws.send(JSON.stringify({
            type: 'auth',
            token: this.token
          }));
        });
      }

      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      return null;
    }
  }
}

export const apiClient = new ApiClient();

// ----- Core-mode typed API helpers -----
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8090/api/v1";

import type { ApiError, Pagination, ApiEnvelope } from '@/lib/api/types';

function withParams(path: string, params?: Record<string, string | number | undefined>): string {
  const url = new URL(path, API_BASE);
  const sp = new URLSearchParams();
  if (params) {
    for (const [k,v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  url.search = sp.toString();
  return url.toString();
}

function parsePaginationHeader(res: Response): Pagination | undefined {
  const raw = res.headers.get("X-Pagination");
  if (!raw) return undefined;
  try { return JSON.parse(raw) as Pagination } catch { return undefined }
}

export class ApiHttpError extends Error {
  status: number; code: string; url: string;
  constructor(status: number, code: string, message: string, url: string) { super(message); this.status=status; this.code=code; this.url=url; }
}

/**
 * GET helper for core mode.
 * @param path API path starting with "/" relative to API_BASE
 * @param params Optional query string parameters
 * @param opts Optional options { role } where role defaults to "viewer"
 * @returns ApiEnvelope<T> with pagination populated from X-Pagination header if present
 */
export async function apiGet<T>(path: string, params?: Record<string, string | number | undefined>, opts?: { role?: "viewer" | "operator" }): Promise<ApiEnvelope<T>> {
  try {
    const url = withParams(path, params);
    const role = opts?.role ?? "viewer";
    const res = await fetch(url, { method: "GET", headers: { Accept: "application/json", "X-Role": role }, credentials: "include" });
    
    if (!res.ok) {
      return { data: null, error: { code: `HTTP_${res.status}`, message: res.statusText } };
    }
    
    const env = await res.json() as ApiEnvelope<T>;
    const pg = parsePaginationHeader(res); if (pg) env.pagination = pg;
    if (env.error) throw new ApiHttpError(res.status, env.error.code, env.error.message, url);
    return env;
  } catch (error) {
    return { data: null, error: { code: "FETCH_ERROR", message: error instanceof Error ? error.message : "Unknown error" } };
  }
}

/**
 * POST helper for core mode.
 * @param path API path starting with "/" relative to API_BASE
 * @param body Optional JSON body (will be JSON.stringified if provided)
 * @param opts Optional options { role } where role defaults to "viewer"
 * @returns ApiEnvelope<T>
 */
export async function apiPost<T>(path: string, body?: unknown, opts?: { role?: "viewer" | "operator" }): Promise<ApiEnvelope<T>> {
  const url = new URL(path, API_BASE).toString();
  const role = opts?.role ?? "viewer";
  const res = await fetch(url, { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json", "X-Role": role }, body: body!==undefined?JSON.stringify(body):undefined, credentials: "include" });
  const env = await res.json() as ApiEnvelope<T>;
  if (env.error) throw new ApiHttpError(res.status, env.error.code, env.error.message, url);
  return env;
}

// ----- Distributed System API Functions -----

import type {
  NetworkNode,
  NetworkEdge,
  ClusterTopology,
  BandwidthMetrics,
  QoSMetrics,
  NetworkInterface,
  ResourcePrediction,
  WorkloadPattern,
  MigrationPrediction,
  ComputeJob,
  GlobalResourcePool,
  MemoryFabric,
  ProcessingFabric
} from '@/lib/api/types';

// Network Topology APIs
export const getNetworkTopology = () =>
  apiGet<ClusterTopology>('/network/topology');

export const getNetworkNodes = () =>
  apiGet<NetworkNode[]>('/network/nodes');

export const getNetworkEdges = () =>
  apiGet<NetworkEdge[]>('/network/edges');

export const updateNetworkNode = (nodeId: string, updates: Partial<NetworkNode>) =>
  apiPost<NetworkNode>(`/network/nodes/${nodeId}`, updates, { role: 'operator' });

// Bandwidth Monitoring APIs
export const getBandwidthMetrics = (timeRange?: string) =>
  apiGet<BandwidthMetrics>('/network/bandwidth/metrics', timeRange ? { timeRange } : undefined);

export const getQoSMetrics = (interfaceId?: string) =>
  apiGet<QoSMetrics>('/network/qos/metrics', interfaceId ? { interfaceId } : undefined);

export const getNetworkInterfaces = () =>
  apiGet<NetworkInterface[]>('/network/interfaces');

export const updateQoSPolicy = (interfaceId: string, policy: any) =>
  apiPost<QoSMetrics>(`/network/qos/${interfaceId}`, policy, { role: 'operator' });

// Performance Prediction APIs
export const getResourcePredictions = (timeHorizon?: number) =>
  apiGet<ResourcePrediction[]>('/ai/predictions/resources', timeHorizon ? { timeHorizon } : undefined);

export const getWorkloadPatterns = (clusterId?: string) =>
  apiGet<WorkloadPattern[]>('/ai/patterns/workload', clusterId ? { clusterId } : undefined);

export const getMigrationPredictions = () =>
  apiGet<MigrationPrediction[]>('/ai/predictions/migration');

export const triggerPredictionRefresh = () =>
  apiPost<{ status: string }>('/ai/predictions/refresh', undefined, { role: 'operator' });

// Supercompute Fabric APIs
export const getComputeJobs = (status?: string) =>
  apiGet<ComputeJob[]>('/fabric/jobs', status ? { status } : undefined);

export const getGlobalResourcePool = () =>
  apiGet<GlobalResourcePool>('/fabric/resources/global');

export const getMemoryFabric = () =>
  apiGet<MemoryFabric>('/fabric/memory');

export const getProcessingFabric = () =>
  apiGet<ProcessingFabric>('/fabric/processing');

export const submitComputeJob = (job: Omit<ComputeJob, 'id' | 'status' | 'createdAt' | 'updatedAt'>) =>
  apiPost<ComputeJob>('/fabric/jobs', job, { role: 'operator' });

export const cancelComputeJob = (jobId: string) =>
  apiPost<{ status: string }>(`/fabric/jobs/${jobId}/cancel`, undefined, { role: 'operator' });

// Cross-Cluster Federation APIs
export const getFederationStatus = () =>
  apiGet<{ status: string; connectedClusters: number; syncHealth: string }>('/federation/status');

export const getFederatedClusters = () =>
  apiGet<Array<{ id: string; name: string; status: string; location: string }>>('/federation/clusters');

export const initializeFederation = (clusterConfig: any) =>
  apiPost<{ status: string }>('/federation/initialize', clusterConfig, { role: 'operator' });

export const syncFederationState = () =>
  apiPost<{ status: string }>('/federation/sync', undefined, { role: 'operator' });

// Security and Compliance APIs
export const getSecurityPolicies = () =>
  apiGet<SecurityPolicy[]>('/security/policies');

export const getComplianceReport = (framework?: string) =>
  apiGet<ComplianceReport>('/security/compliance/report', framework ? { framework } : undefined);

export const getAuditLogs = (startTime?: string, endTime?: string) =>
  apiGet<AuditLog[]>('/security/audit/logs',
    startTime && endTime ? { startTime, endTime } : undefined);

export const updateSecurityPolicy = (policyId: string, policy: Partial<SecurityPolicy>) =>
  apiPost<SecurityPolicy>(`/security/policies/${policyId}`, policy, { role: 'operator' });

// System Configuration APIs
export const getSystemConfiguration = () =>
  apiGet<SystemConfiguration>('/system/configuration');

export const updateSystemConfiguration = (config: Partial<SystemConfiguration>) =>
  apiPost<SystemConfiguration>('/system/configuration', config, { role: 'operator' });

export const getSystemHealth = () =>
  apiGet<{ status: string; components: Array<{ name: string; status: string; lastCheck: string }> }>('/system/health');

// Distributed Storage APIs
export const getStorageMetrics = () =>
  apiGet<{
    totalCapacity: number;
    usedCapacity: number;
    availableCapacity: number;
    replicationFactor: number;
    storageHealth: string;
  }>('/storage/metrics');

export const getStoragePools = () =>
  apiGet<Array<{
    id: string;
    name: string;
    type: string;
    capacity: number;
    used: number;
    health: string;
  }>>('/storage/pools');

// Migration and Backup APIs
export const getMigrationJobs = () =>
  apiGet<Array<{
    id: string;
    sourceCluster: string;
    targetCluster: string;
    status: string;
    progress: number;
    vmCount: number;
    startTime: string;
  }>>('/migration/jobs');

export const initiateClusterMigration = (migrationConfig: {
  sourceCluster: string;
  targetCluster: string;
  vmIds: string[];
  migrationStrategy: string;
}) =>
  apiPost<{ jobId: string; status: string }>('/migration/initiate', migrationConfig, { role: 'operator' });

export const getBackupStatus = () =>
  apiGet<{
    activeBackups: number;
    lastBackupTime: string;
    backupHealth: string;
    totalBackupSize: number;
  }>('/backup/status');

// Live Migration APIs
export const getLiveMigrationStatus = () =>
  apiGet<Array<{
    id: string;
    vmId: string;
    sourceNode: string;
    targetNode: string;
    progress: number;
    phase: string;
    bandwidth: number;
  }>>('/migration/live/status');

export const initiateLiveMigration = (vmId: string, targetNode: string, options?: {
  maxDowntime?: number;
  bandwidth?: number;
  priority?: 'low' | 'normal' | 'high';
}) =>
  apiPost<{ migrationId: string; status: string }>(`/migration/live/${vmId}`,
    { targetNode, ...options }, { role: 'operator' });

// Real-time Metrics Stream APIs
export const getMetricsStream = (endpoint: string, options?: {
  interval?: number;
  aggregation?: string;
}) =>
  apiGet<{ streamUrl: string; token: string }>(`/metrics/stream/${endpoint}`, options);

export const getNodeMetrics = (nodeId: string, timeRange?: string) =>
  apiGet<{
    cpu: number[];
    memory: number[];
    network: number[];
    storage: number[];
    timestamps: string[];
  }>(`/metrics/nodes/${nodeId}`, timeRange ? { timeRange } : undefined);

// AI Model Management APIs
export const getAIModels = () =>
  apiGet<Array<{
    id: string;
    name: string;
    type: string;
    status: string;
    accuracy: number;
    lastTrained: string;
  }>>('/ai/models');

export const trainAIModel = (modelId: string, trainingConfig: {
  dataset: string;
  parameters: Record<string, any>;
}) =>
  apiPost<{ taskId: string; status: string }>(`/ai/models/${modelId}/train`, trainingConfig, { role: 'operator' });

export const getModelPrediction = (modelId: string, input: any) =>
  apiPost<{ prediction: any; confidence: number }>(`/ai/models/${modelId}/predict`, { input });
