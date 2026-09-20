import { buildApiUrl } from '@/lib/api/origin';
import { authService } from '@/lib/auth';

export type FabricLinkProfile = {
  rtt_ms: number;
  last_heartbeat: string;
  stale: boolean;
};

export type FabricNode = {
  node_id: string;
  addr: string;
  arch: string;
  cores: number;
  mem_total_mb: number;
  mem_allocated_mb: number;
  storage_total_gb: number;
  storage_free_gb: number;
  vm_count: number;
  reachable: boolean;
  link: FabricLinkProfile | null;
};

export type FabricJob = {
  job_id: string;
  name?: string | null;
  command: string;
  status: string;
  node_id: string;
  vm_id: string;
  created_at: string;
  error?: string | null;
};

export type FabricJobLogs = {
  stdout: string;
  stderr: string;
};

export type FabricJobDetail = FabricJob & {
  logs?: FabricJobLogs;
};

export type FabricJobPlacement = {
  decision: string;
  cost_estimate_s?: number;
  reason: string;
};

export type FabricJobSubmission = {
  job_id: string;
  vm_id: string;
  node_id: string;
  status: string;
  placement?: FabricJobPlacement | null;
};

export type FabricJobRequest = {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  node_id?: string;
  bytes_to_move?: number;
  memory_mb?: number;
  vcpus?: number;
};

export type FabricJobCancelResult = {
  cancelled: boolean;
  status: string;
  error?: string | null;
};

export type FabricTransfer = {
  transfer_id: string;
  status: string;
  kind?: string | null;
  target_node?: string | null;
  bytes_total?: number | null;
  bytes_moved?: number | null;
  measured_bps?: number | null;
  compression?: string | null;
  eta_seconds?: number | null;
};

export type FabricTransferDecisionInputs = {
  link_bps: number;
  sample_ratio: number;
  threshold: number;
};

export type FabricTransferDetail = FabricTransfer & {
  decision_inputs?: FabricTransferDecisionInputs | null;
};

// The fabric routes answer failures with `{"error": "..."}` (400/503), so surface
// that message instead of the raw status when the body carries one.
async function describeFailure(response: Response, path: string): Promise<string> {
  const body = await response.text();

  if (body) {
    try {
      const parsed = JSON.parse(body) as { error?: unknown; message?: unknown };
      if (typeof parsed.error === 'string' && parsed.error) {
        return parsed.error;
      }
      if (typeof parsed.message === 'string' && parsed.message) {
        return parsed.message;
      }
    } catch {
      // Non-JSON bodies fall through to the raw text.
    }

    return body;
  }

  return `Request failed for ${path} (HTTP ${response.status})`;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = authService.getToken();
  const response = await fetch(buildApiUrl(path), {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    throw new Error(await describeFailure(response, path));
  }

  return response.json() as Promise<T>;
}

export const fabricApi = {
  listNodes: async (): Promise<FabricNode[]> => {
    const payload = await request<{ nodes?: FabricNode[] } | null>('/api/cluster/nodes');
    if (!payload) {
      return [];
    }
    return Array.isArray(payload.nodes) ? payload.nodes : [];
  },
  listJobs: async (): Promise<FabricJob[]> => {
    const payload = await request<{ jobs?: FabricJob[] } | null>('/api/compute/jobs');
    if (!payload) {
      return [];
    }
    return Array.isArray(payload.jobs) ? payload.jobs : [];
  },
  getJob: (jobId: string) => request<FabricJobDetail>(`/api/compute/jobs/${encodeURIComponent(jobId)}`),
  submitJob: (body: FabricJobRequest) =>
    request<FabricJobSubmission>('/api/compute/jobs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  cancelJob: (jobId: string) =>
    request<FabricJobCancelResult>(`/api/compute/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    }),
  // The list envelope is not pinned by the contract; accept `{transfers: []}` or a bare array.
  listTransfers: async (): Promise<FabricTransfer[]> => {
    const payload = await request<{ transfers?: FabricTransfer[] } | FabricTransfer[] | null>('/api/transfers');
    if (!payload) {
      return [];
    }
    if (Array.isArray(payload)) {
      return payload;
    }
    return Array.isArray(payload.transfers) ? payload.transfers : [];
  },
  getTransfer: (transferId: string) =>
    request<FabricTransferDetail>(`/api/transfers/${encodeURIComponent(transferId)}`),
};