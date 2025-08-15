// API service for NovaCron backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

// VM Types
export interface VMInfo {
  id: string;
  name: string;
  state: string;
  cpu_shares: number;
  memory_mb: number;
  cpu_usage?: number;
  memory_usage?: number;
  network_sent?: number;
  network_recv?: number;
  created_at: string;
  root_fs?: string;
  tags?: Record<string, string>;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
}

export interface CreateVMRequest {
  name: string;
  cpu_shares: number;
  memory_mb: number;
  root_fs?: string;
  network_id?: string;
  tags?: Record<string, string>;
}

// Job Types
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
  metadata?: Record<string, any>;
}

export interface CreateJobRequest {
  name: string;
  schedule: string;
  timezone?: string;
  enabled?: boolean;
  priority?: number;
  max_retries?: number;
  timeout?: number;
  metadata?: Record<string, any>;
}

export interface JobExecution {
  id: string;
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'retrying';
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  result?: any;
  error_message?: string;
  attempt_number?: number;
  metadata?: any;
}

// Workflow Types
export interface WorkflowNode {
  id: string;
  name: string;
  type: 'job' | 'decision' | 'parallel' | 'subworkflow';
  config: any;
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
  metadata?: Record<string, any>;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  result?: any;
  errorMessage?: string;
  currentNode?: string;
  nodeExecutions: Record<string, {
    status: 'pending' | 'running' | 'completed' | 'failed';
    startedAt?: string;
    completedAt?: string;
    result?: any;
    errorMessage?: string;
  }>;
  metadata?: Record<string, any>;
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
  metadata?: Record<string, any>;
}

class APIService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check
  async getHealth(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/health');
  }

  // VM Management
  async listVMs(): Promise<{ vms: number; status: string }> {
    return this.request<{ vms: number; status: string }>('/api/v1/vms');
  }

  async getVM(id: string): Promise<{ id: string; state: string }> {
    return this.request<{ id: string; state: string }>(`/api/v1/vms/${id}`);
  }

  async createVM(vmData: CreateVMRequest): Promise<{ id: string; name: string; status: string }> {
    return this.request<{ id: string; name: string; status: string }>('/api/v1/vms', {
      method: 'POST',
      body: JSON.stringify(vmData),
    });
  }

  async deleteVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/api/v1/vms/${id}`, {
      method: 'DELETE',
    });
  }

  async startVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/api/v1/vms/${id}/start`, {
      method: 'POST',
    });
  }

  async stopVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/api/v1/vms/${id}/stop`, {
      method: 'POST',
    });
  }

  async getVMMetrics(id: string): Promise<{ id: string; cpu_usage: number; memory_usage: number }> {
    return this.request<{ id: string; cpu_usage: number; memory_usage: number }>(`/api/v1/vms/${id}/metrics`);
  }

  // Job Management
  async listJobs(): Promise<CronJob[]> {
    const response = await this.request<{ success: boolean; data: CronJob[] }>('/api/jobs');
    return response.data;
  }

  async getJob(id: string): Promise<CronJob> {
    const response = await this.request<{ success: boolean; data: CronJob }>(`/api/jobs/${id}`);
    return response.data;
  }

  async createJob(jobData: CreateJobRequest): Promise<CronJob> {
    const response = await this.request<{ success: boolean; data: CronJob }>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(jobData),
    });
    return response.data;
  }

  async updateJob(id: string, jobData: Partial<CreateJobRequest>): Promise<CronJob> {
    const response = await this.request<{ success: boolean; data: CronJob }>(`/api/jobs/${id}`, {
      method: 'PUT',
      body: JSON.stringify(jobData),
    });
    return response.data;
  }

  async deleteJob(id: string): Promise<boolean> {
    const response = await this.request<{ success: boolean }> (`/api/jobs/${id}`, {
      method: 'DELETE',
    });
    return response.success;
  }

  async executeJob(id: string): Promise<any> {
    const response = await this.request<{ success: boolean; data: any }> (`/api/jobs/${id}/execute`, {
      method: 'POST',
    });
    return response.data;
  }

  async getJobExecutions(id: string): Promise<JobExecution[]> {
    const response = await this.request<{ success: boolean; data: JobExecution[] }> (`/api/jobs/${id}/executions`);
    return response.data;
  }

  // Workflow Management
  async listWorkflows(): Promise<Workflow[]> {
    const response = await this.request<{ success: boolean; data: Workflow[] }>('/api/workflows');
    return response.data;
  }

  async getWorkflow(id: string): Promise<Workflow> {
    const response = await this.request<{ success: boolean; data: Workflow }>(`/api/workflows/${id}`);
    return response.data;
  }

  async createWorkflow(workflowData: CreateWorkflowRequest): Promise<Workflow> {
    const response = await this.request<{ success: boolean; data: Workflow }>('/api/workflows', {
      method: 'POST',
      body: JSON.stringify(workflowData),
    });
    return response.data;
  }

  async updateWorkflow(id: string, workflowData: Partial<CreateWorkflowRequest>): Promise<Workflow> {
    const response = await this.request<{ success: boolean; data: Workflow }>(`/api/workflows/${id}`, {
      method: 'PUT',
      body: JSON.stringify(workflowData),
    });
    return response.data;
  }

  async deleteWorkflow(id: string): Promise<boolean> {
    const response = await this.request<{ success: boolean }> (`/api/workflows/${id}`, {
      method: 'DELETE',
    });
    return response.success;
  }

  async executeWorkflow(id: string): Promise<WorkflowExecution> {
    const response = await this.request<{ success: boolean; data: WorkflowExecution }> (`/api/workflows/${id}/execute`, {
      method: 'POST',
    });
    return response.data;
  }

  async getWorkflowExecution(id: string): Promise<WorkflowExecution> {
    const response = await this.request<{ success: boolean; data: WorkflowExecution }>(`/api/workflows/executions/${id}`);
    return response.data;
  }

  // WebSocket connection for real-time updates
  createWebSocket(onMessage: (data: any) => void, onError?: (error: Event) => void): WebSocket | null {
    try {
      const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws';
      const ws = new WebSocket(wsUrl);
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };
      
      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      return null;
    }
  }
}

export const apiService = new APIService();
export default apiService;