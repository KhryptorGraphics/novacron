import { buildApiV1Url } from '@/lib/api/origin';

/** One row of the networks catalog (GET /api/v1/networks): a NovaCron-managed host bridge. */
export type CanonicalNetwork = {
  id: string;
  name: string;
  bridge: string;
  cidr: string;
  gateway?: string | null;
  vlan_id?: number | null;
  mtu: number;
  created_by?: string | null;
  vm_count: number;
  created_at: string;
  updated_at: string;
};

export type CreateNetworkPayload = {
  name: string;
  cidr: string;
  gateway?: string | undefined;
  vlan_id?: number | undefined;
  mtu?: number | undefined;
};

export type CanonicalVmInterface = {
  id: string;
  vm_id: string;
  network_id?: string | null;
  name: string;
  mac_address: string;
  ip_address?: string | null;
  status: string;
  created_at: string;
  updated_at: string;
};

function authHeaders(): HeadersInit {
  const token = typeof window !== 'undefined' ? window.localStorage.getItem('novacron_token') : null;
  return {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/** The server answers errors as {"error": "..."}; surface that message, not the raw body. */
async function errorMessage(response: Response, path: string): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text) as { error?: unknown; vm_ids?: unknown };
    if (typeof parsed.error === 'string') {
      return Array.isArray(parsed.vm_ids) && parsed.vm_ids.length > 0
        ? `${parsed.error} (${parsed.vm_ids.join(', ')})`
        : parsed.error;
    }
  } catch {
    // not JSON
  }
  return text || `Request failed for ${path} (${response.status})`;
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
    throw new Error(await errorMessage(response, path));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const networkApi = {
  listNetworks: () => request<CanonicalNetwork[]>('/networks'),
  getNetwork: (id: string) => request<CanonicalNetwork>(`/networks/${id}`),
  createNetwork: (payload: CreateNetworkPayload) =>
    request<CanonicalNetwork>('/networks', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteNetwork: (id: string) =>
    request<{ id: string; name: string; status: string }>(`/networks/${id}`, { method: 'DELETE' }),
  listVmInterfaces: (vmId: string) => request<CanonicalVmInterface[]>(`/vms/${vmId}/interfaces`),
  attachVmInterface: (
    vmId: string,
    payload: { network_id?: string | undefined; name: string; mac_address: string; ip_address?: string | undefined },
  ) =>
    request<CanonicalVmInterface>(`/vms/${vmId}/interfaces`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateVmInterface: (
    vmId: string,
    interfaceId: string,
    payload: { network_id?: string; name?: string; ip_address?: string; status?: string },
  ) =>
    request<CanonicalVmInterface>(`/vms/${vmId}/interfaces/${interfaceId}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteVmInterface: (vmId: string, interfaceId: string) =>
    request<{ id: string; vm_id: string; status: string }>(`/vms/${vmId}/interfaces/${interfaceId}`, {
      method: 'DELETE',
    }),
};
