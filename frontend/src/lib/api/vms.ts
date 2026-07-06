import { apiGet, apiPost } from "./client";
import type { VM } from "./types";

export type ListVMsParams = {
  page?: number | undefined;
  pageSize?: number | undefined;
  sortBy?: "name" | "createdAt" | "state" | undefined;
  sortDir?: "asc" | "desc" | undefined;
  state?: string | undefined;
  nodeId?: string | undefined;
  q?: string | undefined;
};

export type CreateVMBody = {
  name: string;
  type?: string | undefined;      // VMType / driver, e.g. "kvm" (multi-arch selector)
  cpu?: number | undefined;       // vCPU shares
  memory?: number | undefined;    // MB (backend expects megabytes)
  disk?: number | undefined;      // GB
  image?: string | undefined;
  tags?: Record<string, string> | undefined;
};

export const listVMs = (params?: ListVMsParams) => apiGet<VM[]>("/vms", params);
export const getVM = (id: string) => apiGet<VM>(`/vms/${id}`);
export const createVM = (body: CreateVMBody) => apiPost<VM>("/vms", body, { role: "operator" });
export const postVMAction = (id: string, action: "start" | "stop" | "restart" | "pause" | "resume", opts?: { role?: "viewer"|"operator" | undefined }) =>
  apiPost<VM>(`/vms/${id}/${action}`, undefined, opts);

