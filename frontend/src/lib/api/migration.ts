import { buildApiV1Url } from '@/lib/api/origin';

export type MigrationJob = {
  id?: string;
  jobId?: string;
  status: string;
  sourceCluster?: string;
  targetCluster?: string;
  vmCount?: number;
  vmIds?: string[];
  migrationStrategy?: string;
  progress?: number;
  createdAt?: string;
  startTime?: string;
};

export type MigrationCheck = {
  name: string;
  status: 'passed' | 'warning' | 'failed' | string;
  message: string;
};

export type MigrationPlan = {
  planId: string;
  status: string;
  sourceCluster: string;
  targetCluster: string;
  vmIds: string[];
  vmCount: number;
  migrationStrategy: string;
  estimatedDurationSeconds: number;
  estimatedDowntimeSeconds: number;
  checks: MigrationCheck[];
  createdAt: string;
};

export type MigrationRequest = {
  sourceCluster: string;
  targetCluster: string;
  vmIds: string[];
  migrationStrategy: string;
  bandwidthMbps?: number;
  maxDowntimeSeconds?: number;
};

function authHeaders(): HeadersInit {
  const token = typeof window !== 'undefined' ? window.localStorage.getItem('novacron_token') : null;
  return {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(buildApiV1Url(path), {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed for ${path}`);
  }
  return response.json() as Promise<T>;
}

export const migrationApi = {
  listJobs: () => request<MigrationJob[]>('/migration/jobs'),
  initiate: (payload: MigrationRequest) =>
    request<MigrationJob>('/migration/initiate', { method: 'POST', body: JSON.stringify(payload) }),
  listPlans: () => request<MigrationPlan[]>('/migration/plans'),
  createPlan: (payload: MigrationRequest) =>
    request<MigrationPlan>('/migration/plans', { method: 'POST', body: JSON.stringify(payload) }),
  preflight: (payload: MigrationRequest & { planId?: string }) =>
    request<{ planId?: string; status: string; checks: MigrationCheck[] }>('/migration/preflight', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  rollback: (jobId: string) =>
    request<{ jobId: string; rollbackId: string; status: string }>(`/migration/jobs/${jobId}/rollback`, {
      method: 'POST',
    }),
};
