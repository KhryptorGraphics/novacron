import { buildApiV1Url } from '@/lib/api/origin';

export type BackupStatus = {
  activeBackups: number;
  lastBackupTime: string;
  backupHealth: string;
  totalBackupSize: number;
};

export type BackupPolicy = {
  id: string;
  name: string;
  enabled: boolean;
  schedule: string;
  retentionDays: number;
  target: string;
  status: string;
  createdAt: string;
  updatedAt: string;
};

export type BackupRun = {
  id: string;
  policyId?: string;
  status: string;
  createdAt: string;
};

export type RestoreJob = {
  id: string;
  backupId: string;
  target: string;
  status: string;
  createdAt: string;
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

export const backupApi = {
  getStatus: () => request<BackupStatus>('/backup/status'),
  listPolicies: () => request<BackupPolicy[]>('/backup/policies'),
  createPolicy: (payload: { name: string; schedule?: string; retentionDays?: number; target?: string; enabled?: boolean }) =>
    request<BackupPolicy>('/backup/policies', { method: 'POST', body: JSON.stringify(payload) }),
  updatePolicy: (id: string, payload: Partial<BackupPolicy>) =>
    request<BackupPolicy>(`/backup/policies/${id}`, { method: 'PUT', body: JSON.stringify(payload) }),
  deletePolicy: (id: string) =>
    request<{ id: string; status: string }>(`/backup/policies/${id}`, { method: 'DELETE' }),
  runPolicy: (id: string) =>
    request<BackupRun>(`/backup/policies/${id}/run`, { method: 'POST' }),
  listBackups: () => request<BackupRun[]>('/backup/backups'),
  restore: (payload: { backupId: string; target?: string }) =>
    request<RestoreJob>('/backup/restore', { method: 'POST', body: JSON.stringify(payload) }),
};
