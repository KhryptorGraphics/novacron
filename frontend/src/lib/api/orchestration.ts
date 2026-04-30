import apiClient from './api-client';
import { buildApiUrl } from './origin';

type ListResponse<T> = T[] | { data?: T[] | null; items?: T[] | null; policies?: T[] | null; models?: T[] | null; events?: T[] | null };

export interface EngineStatus {
  state: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
  startTime: string;
  activePolicies: number;
  eventsProcessed: number;
  metrics: Record<string, unknown>;
}

export interface OrchestrationDecision {
  id: string;
  decisionType: 'placement' | 'scaling' | 'healing' | 'migration' | 'optimization';
  recommendation: string;
  score: number;
  confidence: number;
  explanation: string;
  timestamp: string;
  status: 'pending' | 'executed' | 'failed' | 'cancelled';
}

export interface PolicyRule {
  type: 'placement' | 'autoscaling' | 'healing' | 'loadbalance' | 'security' | 'compliance';
  enabled: boolean;
  priority: number;
  parameters: Record<string, unknown>;
}

export interface OrchestrationPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: number;
  rules: PolicyRule[];
  createdAt: string;
  updatedAt: string;
}

export interface MLModelMetrics {
  modelType: string;
  accuracy: number;
  throughput: number;
  latency: number;
  lastTraining: string;
  version: string;
  status: 'training' | 'deployed' | 'evaluating' | 'error';
  benchmarkResults?: {
    precision: number;
    recall: number;
    f1Score: number;
    auc: number;
  };
  trainingMetrics?: {
    epochs: number;
    trainingLoss: number;
    validationLoss: number;
    trainingTime: string;
  };
}

export interface MetricPoint {
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  networkIO: number;
  diskIO: number;
  decisionsPerMinute: number;
  responseTime: number;
}

export interface ScalingEvent {
  timestamp: string;
  action: 'scale_up' | 'scale_down' | 'no_change';
  vmId: string;
  beforeCount: number;
  afterCount: number;
  reason: string;
  cpuUtilization: number;
  memoryUtilization: number;
  requestRate: number;
  responseTime: number;
}

export interface ScalingMetrics {
  timestamp: string;
  totalVMs: number;
  cpuUtilization: number;
  memoryUtilization: number;
  requestRate: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  scalingEvents: number;
}

function unwrapList<T>(response: ListResponse<T>): T[] {
  if (Array.isArray(response)) {
    return response;
  }

  return response.data ?? response.items ?? response.policies ?? response.models ?? response.events ?? [];
}

function authHeaders(): HeadersInit {
  if (typeof window === 'undefined') {
    return {};
  }

  const token = localStorage.getItem('novacron_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export const orchestrationApi = {
  getStatus: () => apiClient.get<EngineStatus>('/api/orchestration/status'),

  listDecisions: async (limit = 10): Promise<OrchestrationDecision[]> => {
    const response = await apiClient.get<ListResponse<OrchestrationDecision>>(`/api/orchestration/decisions?limit=${limit}`);
    return unwrapList(response);
  },

  listPolicies: async (): Promise<OrchestrationPolicy[]> => {
    const response = await apiClient.get<ListResponse<OrchestrationPolicy>>('/api/orchestration/policies');
    return unwrapList(response);
  },

  createPolicy: (policy: OrchestrationPolicy) =>
    apiClient.post<OrchestrationPolicy>('/api/orchestration/policies', policy),

  updatePolicy: (id: string, policy: OrchestrationPolicy) =>
    apiClient.put<OrchestrationPolicy>(`/api/orchestration/policies/${id}`, policy),

  deletePolicy: (id: string) =>
    apiClient.delete<void>(`/api/orchestration/policies/${id}`),

  listModels: async (): Promise<MLModelMetrics[]> => {
    const response = await apiClient.get<ListResponse<MLModelMetrics>>('/api/orchestration/ml-models');
    return unwrapList(response);
  },

  retrainModel: (modelType: string) =>
    apiClient.post<{ status: string }>(`/api/orchestration/ml-models/${modelType}/retrain`),

  downloadModel: async (modelType: string): Promise<Blob> => {
    const response = await fetch(buildApiUrl(`/api/orchestration/ml-models/${modelType}/download`), {
      headers: authHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to download model: ${response.status}`);
    }

    return response.blob();
  },

  getRealtimeMetrics: () =>
    apiClient.get<Partial<MetricPoint> & Record<string, unknown>>('/api/orchestration/metrics/realtime'),

  listScalingMetrics: async (range: string): Promise<ScalingMetrics[]> => {
    const response = await apiClient.get<ListResponse<ScalingMetrics>>(`/api/orchestration/scaling/metrics?range=${range}`);
    return unwrapList(response);
  },

  listScalingEvents: async (limit = 20): Promise<ScalingEvent[]> => {
    const response = await apiClient.get<ListResponse<ScalingEvent>>(`/api/orchestration/scaling/events?limit=${limit}`);
    return unwrapList(response);
  },
};
