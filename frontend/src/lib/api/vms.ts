import { apiDelete, apiGet, apiPost } from "./client";
import type { VM } from "./types";

export type ListVMsParams = {
  page?: number;
  pageSize?: number;
  sortBy?: "name" | "createdAt" | "state";
  sortDir?: "asc" | "desc";
  state?: string;
  nodeId?: string;
  q?: string;
};

export type CreateVMRequest = {
  name: string;
  node_id?: string;
  cpu_shares?: number;
  memory_mb?: number;
  tags?: Record<string, unknown>;
};

export const listVMs = (params?: ListVMsParams) => apiGet<VM[]>("/vms", params);
export const getVM = (id: string) => apiGet<VM>(`/vms/${id}`);
export const createVM = (payload: CreateVMRequest) =>
  apiPost<VM>("/vms", payload, { role: "operator" });
export const postVMAction = (id: string, action: "start" | "stop" | "restart" | "pause" | "resume", opts?: { role?: "viewer"|"operator" }) =>
  apiPost<VM>(`/vms/${id}/${action}`, undefined, opts);
export const deleteVM = (id: string) =>
  apiDelete<{ id: string; status: string }>(`/vms/${id}`, { role: "operator" });
