// API service for NovaCron backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080/api/v1';

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
    return this.request<{ vms: number; status: string }>('/vms');
  }

  async getVM(id: string): Promise<{ id: string; state: string }> {
    return this.request<{ id: string; state: string }>(`/vms/${id}`);
  }

  async createVM(vmData: CreateVMRequest): Promise<{ id: string; name: string; status: string }> {
    return this.request<{ id: string; name: string; status: string }>('/vms', {
      method: 'POST',
      body: JSON.stringify(vmData),
    });
  }

  async deleteVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/vms/${id}`, {
      method: 'DELETE',
    });
  }

  async startVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/vms/${id}/start`, {
      method: 'POST',
    });
  }

  async stopVM(id: string): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>(`/vms/${id}/stop`, {
      method: 'POST',
    });
  }

  async getVMMetrics(id: string): Promise<{ id: string; cpu_usage: number; memory_usage: number }> {
    return this.request<{ id: string; cpu_usage: number; memory_usage: number }>(`/vms/${id}/metrics`);
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