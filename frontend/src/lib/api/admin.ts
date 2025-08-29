import { apiClient } from './client';

// Types for Admin API
export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  role?: string;
}

export interface UpdateUserRequest {
  username?: string;
  email?: string;
  role?: string;
  active?: boolean;
}

export interface UserListResponse {
  users: User[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface TableInfo {
  name: string;
  schema: string;
  row_count: number;
  size: string;
  description: string;
}

export interface QueryResult {
  columns: string[];
  rows: any[][];
  affected: number;
  duration: string;
}

export interface SecurityMetrics {
  total_alerts: number;
  critical_alerts: number;
  failed_logins_24h: number;
  active_sessions: number;
  last_breach?: string;
  alerts_by_type: Record<string, number>;
  alerts_by_severity: Record<string, number>;
  recent_activities: AuditLogEntry[];
  top_ips: IPStatistic[];
}

export interface AuditLogEntry {
  id: number;
  user_id: number;
  username: string;
  action: string;
  resource: string;
  details: Record<string, any>;
  ip: string;
  user_agent: string;
  success: boolean;
  created_at: string;
}

export interface IPStatistic {
  ip: string;
  count: number;
  country?: string;
}

export interface SystemConfig {
  server: {
    api_port: number;
    ws_port: number;
    log_level: string;
    log_format: string;
    environment: string;
    max_connections: number;
    request_timeout_seconds: number;
    enable_cors: boolean;
    tls_enabled: boolean;
    tls_cert_path?: string;
    tls_key_path?: string;
  };
  database: {
    host: string;
    port: number;
    name: string;
    username: string;
    max_connections: number;
    max_idle_connections: number;
    connection_max_lifetime_minutes: number;
    ssl_mode: string;
  };
  security: {
    jwt_expiry_hours: number;
    password_min_length: number;
    require_special_chars: boolean;
    max_login_attempts: number;
    lockout_duration_minutes: number;
    session_timeout_minutes: number;
    require_mfa: boolean;
    allowed_origins: string;
    rate_limit_enabled: boolean;
    rate_limit_rpm: number;
  };
  storage: {
    default_path: string;
    max_disk_usage_gb: number;
    compression_enabled: boolean;
    encryption_enabled: boolean;
    backup_enabled: boolean;
    backup_retention_days: number;
    tiered_storage_enabled: boolean;
  };
  vm: {
    default_driver: string;
    max_vms_per_node: number;
    default_cpu_cores: number;
    default_memory_mb: number;
    default_disk_gb: number;
    migration_enabled: boolean;
    live_migration_enabled: boolean;
    compression_level: number;
    network_optimized: boolean;
    resource_limits: {
      max_cpu_cores: number;
      max_memory_gb: number;
      max_disk_gb: number;
      max_network_mbps: number;
    };
    supported_drivers: string[];
  };
  monitoring: {
    enabled: boolean;
    metrics_interval_seconds: number;
    retention_days: number;
    prometheus_enabled: boolean;
    grafana_enabled: boolean;
    alerting_enabled: boolean;
    webhook_url?: string;
    slack_webhook?: string;
    email_notifications: boolean;
  };
  network: {
    default_subnet: string;
    dhcp_enabled: boolean;
    vlan_support: boolean;
    sdn_enabled: boolean;
    bandwidth_limiting: boolean;
    qos_enabled: boolean;
    firewall_enabled: boolean;
    dns_servers: string;
  };
  last_updated: string;
  updated_by?: string;
}

// Admin API methods
export const adminApi = {
  // User Management
  users: {
    list: (params?: { 
      page?: number; 
      page_size?: number; 
      search?: string; 
      role?: string;
    }): Promise<UserListResponse> => {
      const query = new URLSearchParams();
      if (params?.page) query.set('page', params.page.toString());
      if (params?.page_size) query.set('page_size', params.page_size.toString());
      if (params?.search) query.set('search', params.search);
      if (params?.role) query.set('role', params.role);
      
      return apiClient.get(`/api/admin/users?${query}`);
    },
    
    create: (user: CreateUserRequest): Promise<User> => 
      apiClient.post('/api/admin/users', user),
    
    update: (id: number, user: UpdateUserRequest): Promise<User> =>
      apiClient.put(`/api/admin/users/${id}`, user),
    
    delete: (id: number): Promise<void> =>
      apiClient.delete(`/api/admin/users/${id}`),
    
    assignRoles: (id: number, roles: string[]): Promise<void> =>
      apiClient.post(`/api/admin/users/${id}/roles`, { roles }),
  },

  // Database Administration
  database: {
    tables: (): Promise<{ tables: TableInfo[]; count: number }> =>
      apiClient.get('/api/admin/database/tables'),
    
    tableDetails: (table: string): Promise<{
      table: TableInfo;
      columns: any[];
      indexes: any[];
    }> => apiClient.get(`/api/admin/database/tables/${table}`),
    
    query: (sql: string): Promise<QueryResult> =>
      apiClient.post('/api/admin/database/query', { sql }),
    
    execute: (sql: string): Promise<QueryResult> =>
      apiClient.post('/api/admin/database/execute', { sql }),
  },

  // Security Dashboard
  security: {
    metrics: (): Promise<SecurityMetrics> =>
      apiClient.get('/api/admin/security/metrics'),
    
    alerts: (params?: { 
      page?: number; 
      page_size?: number; 
      severity?: string; 
      status?: string;
    }): Promise<{ alerts: any[]; page: number; count: number }> => {
      const query = new URLSearchParams();
      if (params?.page) query.set('page', params.page.toString());
      if (params?.page_size) query.set('page_size', params.page_size.toString());
      if (params?.severity) query.set('severity', params.severity);
      if (params?.status) query.set('status', params.status);
      
      return apiClient.get(`/api/admin/security/alerts?${query}`);
    },
    
    auditLogs: (params?: { 
      page?: number; 
      page_size?: number; 
      action?: string; 
      user_id?: string;
    }): Promise<{ logs: AuditLogEntry[]; page: number; count: number }> => {
      const query = new URLSearchParams();
      if (params?.page) query.set('page', params.page.toString());
      if (params?.page_size) query.set('page_size', params.page_size.toString());
      if (params?.action) query.set('action', params.action);
      if (params?.user_id) query.set('user_id', params.user_id);
      
      return apiClient.get(`/api/admin/security/audit?${query}`);
    },
    
    policies: (): Promise<{ policies: any[]; count: number }> =>
      apiClient.get('/api/admin/security/policies'),
    
    updatePolicy: (id: number, policy: any): Promise<any> =>
      apiClient.put(`/api/admin/security/policies/${id}`, policy),
  },

  // System Configuration
  config: {
    get: (): Promise<SystemConfig> =>
      apiClient.get('/api/admin/config'),
    
    update: (config: SystemConfig): Promise<SystemConfig> =>
      apiClient.put('/api/admin/config', config),
    
    validate: (config: SystemConfig): Promise<{ valid: boolean; errors?: string[] }> =>
      apiClient.post('/api/admin/config/validate', config),
    
    backup: (description: string): Promise<any> =>
      apiClient.post('/api/admin/config/backup', { description }),
    
    listBackups: (params?: { page?: number; page_size?: number }): Promise<{
      backups: any[];
      page: number;
      count: number;
      has_more: boolean;
    }> => {
      const query = new URLSearchParams();
      if (params?.page) query.set('page', params.page.toString());
      if (params?.page_size) query.set('page_size', params.page_size.toString());
      
      return apiClient.get(`/api/admin/config/backups?${query}`);
    },
    
    restore: (id: number): Promise<{ message: string; config: SystemConfig }> =>
      apiClient.post(`/api/admin/config/restore/${id}`),
  },
};