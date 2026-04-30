import { apiClient } from "./client";

type ApiDataResponse<T> = {
  success?: boolean;
  data?: T;
};

export interface CronJob {
  id: string;
  name: string;
  schedule: string;
  timezone: string;
  enabled: boolean;
  priority: number;
  max_retries: number;
  timeout: number;
  created_at: string;
  next_run_at?: string;
  last_run_at?: string;
  metadata?: Record<string, unknown>;
}

export interface CreateJobRequest {
  name: string;
  schedule: string;
  timezone?: string;
  enabled?: boolean;
  priority?: number;
  max_retries?: number;
  timeout?: number;
  metadata?: Record<string, unknown>;
}

export interface JobExecution {
  id: string;
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "retrying";
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  result?: unknown;
  error_message?: string;
  attempt_number?: number;
  metadata?: unknown;
}

export interface WorkflowNode {
  id: string;
  name: string;
  type: "job" | "decision" | "parallel" | "subworkflow";
  config: unknown;
  dependencies?: string[];
  next?: string[];
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: Array<{
    from: string;
    to: string;
    condition?: string;
  }>;
  enabled: boolean;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

export interface CreateWorkflowRequest {
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: Array<{
    from: string;
    to: string;
    condition?: string;
  }>;
  enabled?: boolean;
  metadata?: Record<string, unknown>;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  result?: unknown;
  errorMessage?: string;
  currentNode?: string;
  nodeExecutions: Record<string, {
    status: "pending" | "running" | "completed" | "failed";
    startedAt?: string;
    completedAt?: string;
    durationMs?: number;
    result?: unknown;
    errorMessage?: string;
  }>;
  metadata?: Record<string, unknown>;
}

function unwrapData<T>(response: T | ApiDataResponse<T>, fallback: T): T {
  if (response && typeof response === "object" && "data" in response) {
    return (response as ApiDataResponse<T>).data ?? fallback;
  }
  return response as T;
}

export const automationApi = {
  listJobs: async (): Promise<CronJob[]> => {
    const response = await apiClient.get<CronJob[] | ApiDataResponse<CronJob[]>>("/api/jobs");
    return unwrapData(response, []);
  },

  getJob: async (id: string): Promise<CronJob> => {
    const response = await apiClient.get<CronJob | ApiDataResponse<CronJob>>(`/api/jobs/${id}`);
    return unwrapData(response, null as unknown as CronJob);
  },

  createJob: async (job: CreateJobRequest): Promise<CronJob> => {
    const response = await apiClient.post<CronJob | ApiDataResponse<CronJob>>("/api/jobs", job);
    return unwrapData(response, null as unknown as CronJob);
  },

  updateJob: async (id: string, job: Partial<CreateJobRequest>): Promise<CronJob> => {
    const response = await apiClient.put<CronJob | ApiDataResponse<CronJob>>(`/api/jobs/${id}`, job);
    return unwrapData(response, null as unknown as CronJob);
  },

  deleteJob: async (id: string): Promise<boolean> => {
    const response = await apiClient.delete<boolean | { success?: boolean }>(`/api/jobs/${id}`);
    return typeof response === "boolean" ? response : response.success ?? true;
  },

  executeJob: async (id: string): Promise<unknown> => {
    const response = await apiClient.post<unknown | ApiDataResponse<unknown>>(`/api/jobs/${id}/execute`);
    return unwrapData(response, null);
  },

  listWorkflows: async (): Promise<Workflow[]> => {
    const response = await apiClient.get<Workflow[] | ApiDataResponse<Workflow[]>>("/api/workflows");
    return unwrapData(response, []);
  },

  getWorkflow: async (id: string): Promise<Workflow> => {
    const response = await apiClient.get<Workflow | ApiDataResponse<Workflow>>(`/api/workflows/${id}`);
    return unwrapData(response, null as unknown as Workflow);
  },

  createWorkflow: async (workflow: CreateWorkflowRequest): Promise<Workflow> => {
    const response = await apiClient.post<Workflow | ApiDataResponse<Workflow>>("/api/workflows", workflow);
    return unwrapData(response, null as unknown as Workflow);
  },

  updateWorkflow: async (id: string, workflow: Partial<CreateWorkflowRequest>): Promise<Workflow> => {
    const response = await apiClient.put<Workflow | ApiDataResponse<Workflow>>(`/api/workflows/${id}`, workflow);
    return unwrapData(response, null as unknown as Workflow);
  },

  deleteWorkflow: async (id: string): Promise<boolean> => {
    const response = await apiClient.delete<boolean | { success?: boolean }>(`/api/workflows/${id}`);
    return typeof response === "boolean" ? response : response.success ?? true;
  },

  executeWorkflow: async (id: string): Promise<WorkflowExecution> => {
    const response = await apiClient.post<WorkflowExecution | ApiDataResponse<WorkflowExecution>>(`/api/workflows/${id}/execute`);
    return unwrapData(response, null as unknown as WorkflowExecution);
  },

  getWorkflowExecution: async (id: string): Promise<WorkflowExecution> => {
    const response = await apiClient.get<WorkflowExecution | ApiDataResponse<WorkflowExecution>>(`/api/workflows/executions/${id}`);
    return unwrapData(response, null as unknown as WorkflowExecution);
  },
};
