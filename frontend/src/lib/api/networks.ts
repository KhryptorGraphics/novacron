import { buildApiV1Url } from '@/lib/api/origin';

export type CanonicalNetwork = {
  id: string;
  name: string;
  type: string;
  subnet: string;
  gateway?: string | null;
  status: string;
  created_at: string;
  updated_at: string;
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

export type FirewallRule = {
  id: string;
  name: string;
  networkId?: string;
  direction: 'inbound' | 'outbound' | string;
  action: 'allow' | 'deny' | string;
  protocol: 'tcp' | 'udp' | 'icmp' | 'any' | string;
  source: string;
  destination: string;
  port: string;
  priority: number;
  enabled: boolean;
  hits: number;
  status: string;
  createdAt: string;
  updatedAt: string;
};

export type LoadBalancer = {
  id: string;
  name: string;
  networkId?: string;
  vip?: string;
  port: number;
  algorithm: string;
  type: string;
  status: string;
  backends: Array<Record<string, unknown>>;
  createdAt: string;
  updatedAt: string;
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

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const networkApi = {
  listNetworks: () => request<CanonicalNetwork[]>('/networks'),
  createNetwork: (payload: { name: string; type: string; subnet: string; gateway?: string }) =>
    request<CanonicalNetwork>('/networks', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteNetwork: (id: string) =>
    request<{ id: string; status: string }>(`/networks/${id}`, { method: 'DELETE' }),
  listVmInterfaces: (vmId: string) => request<CanonicalVmInterface[]>(`/vms/${vmId}/interfaces`),
  attachVmInterface: (
    vmId: string,
    payload: { network_id?: string; name: string; mac_address: string; ip_address?: string },
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
  listFirewallRules: () => request<FirewallRule[]>('/network/firewall-rules'),
  createFirewallRule: (payload: Partial<FirewallRule> & { name: string }) =>
    request<FirewallRule>('/network/firewall-rules', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateFirewallRule: (id: string, payload: Partial<FirewallRule>) =>
    request<FirewallRule>(`/network/firewall-rules/${id}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteFirewallRule: (id: string) =>
    request<{ id: string; status: string }>(`/network/firewall-rules/${id}`, { method: 'DELETE' }),
  listLoadBalancers: () => request<LoadBalancer[]>('/network/load-balancers'),
  createLoadBalancer: (payload: Partial<LoadBalancer> & { name: string }) =>
    request<LoadBalancer>('/network/load-balancers', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateLoadBalancer: (id: string, payload: Partial<LoadBalancer>) =>
    request<LoadBalancer>(`/network/load-balancers/${id}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteLoadBalancer: (id: string) =>
    request<{ id: string; status: string }>(`/network/load-balancers/${id}`, { method: 'DELETE' }),
};
