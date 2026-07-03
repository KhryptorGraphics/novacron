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

export type CreateVMBody = {
  name: string;
  type?: string;                  // VMType / driver, e.g. "kvm" (multi-arch selector)
  cpu?: number;                   // vCPU shares
  memory?: number;                // MB (backend expects megabytes)
  disk?: number;                  // GB
  image?: string;
  tags?: Record<string, string>;
};

export const listVMs = (params?: ListVMsParams) => apiGet<VM[]>("/vms", params);
export const getVM = (id: string) => apiGet<VM>(`/vms/${id}`);
export const createVM = (body: CreateVMBody) => apiPost<VM>("/vms", body, { role: "operator" });
export const postVMAction = (id: string, action: "start" | "stop" | "restart" | "pause" | "resume", opts?: { role?: "viewer"|"operator" }) =>
  apiPost<VM>(`/vms/${id}/${action}`, undefined, opts);

