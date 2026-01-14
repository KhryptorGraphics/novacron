import { apiGet, apiPost } from "./client";
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

export const listVMs = (params?: ListVMsParams) => apiGet<VM[]>("/vms", params);
export const getVM = (id: string) => apiGet<VM>(`/vms/${id}`);
export const postVMAction = (id: string, action: "start" | "stop" | "restart" | "pause" | "resume", opts?: { role?: "viewer"|"operator" }) =>
  apiPost<VM>(`/vms/${id}/${action}`, undefined, opts);

