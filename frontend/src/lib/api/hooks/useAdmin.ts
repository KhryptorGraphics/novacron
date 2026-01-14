import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../client';
import { 
  User, 
  AuditLogEntry, 
  SystemMetrics, 
  SecurityAlert, 
  SystemConfiguration,
  ResourceQuota,
  VmTemplate,
  PerformanceReport,
  UserSession
} from '../types';

// Admin API endpoints
const ADMIN_ENDPOINTS = {
  USERS: '/admin/users',
  USER_SESSIONS: '/admin/users/sessions',
  AUDIT_LOGS: '/admin/audit-logs',
  SYSTEM_METRICS: '/admin/system/metrics',
  SECURITY_ALERTS: '/admin/security/alerts',
  SYSTEM_CONFIG: '/admin/system/config',
  RESOURCE_QUOTAS: '/admin/resource-quotas',
  VM_TEMPLATES: '/admin/vm-templates',
  PERFORMANCE_REPORTS: '/admin/reports/performance'
};

// Query keys
export const ADMIN_QUERY_KEYS = {
  USERS: ['admin', 'users'],
  USER_SESSIONS: (userId?: string) => ['admin', 'user-sessions', userId],
  AUDIT_LOGS: ['admin', 'audit-logs'],
  SYSTEM_METRICS: ['admin', 'system-metrics'],
  SECURITY_ALERTS: ['admin', 'security-alerts'],
  SYSTEM_CONFIG: ['admin', 'system-config'],
  RESOURCE_QUOTAS: ['admin', 'resource-quotas'],
  VM_TEMPLATES: ['admin', 'vm-templates'],
  PERFORMANCE_REPORTS: ['admin', 'performance-reports']
};

// User Management Hooks
export const useUsers = (filters?: { 
  status?: string; 
  role?: string; 
  search?: string;
  page?: number;
  pageSize?: number;
}) => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.USERS, filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters?.status) params.append('status', filters.status);
      if (filters?.role) params.append('role', filters.role);
      if (filters?.search) params.append('search', filters.search);
      if (filters?.page) params.append('page', filters.page.toString());
      if (filters?.pageSize) params.append('pageSize', filters.pageSize.toString());
      
      const response = await apiClient.get<{
        users: User[];
        total: number;
        page: number;
        pageSize: number;
        totalPages: number;
      }>(`${ADMIN_ENDPOINTS.USERS}?${params.toString()}`);
      return response;
    },
    staleTime: 30000, // 30 seconds
    refetchInterval: 60000, // 1 minute
  });
};

export const useCreateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userData: Partial<User>) => {
      return apiClient.post<User>(ADMIN_ENDPOINTS.USERS, userData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.USERS });
    },
  });
};

export const useUpdateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ id, ...userData }: Partial<User> & { id: string }) => {
      return apiClient.put<User>(`${ADMIN_ENDPOINTS.USERS}/${id}`, userData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.USERS });
    },
  });
};

export const useDeleteUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userId: string) => {
      return apiClient.delete(`${ADMIN_ENDPOINTS.USERS}/${userId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.USERS });
    },
  });
};

export const useUserSessions = (userId?: string) => {
  return useQuery({
    queryKey: ADMIN_QUERY_KEYS.USER_SESSIONS(userId),
    queryFn: async () => {
      const endpoint = userId 
        ? `${ADMIN_ENDPOINTS.USER_SESSIONS}/${userId}`
        : ADMIN_ENDPOINTS.USER_SESSIONS;
      return apiClient.get<UserSession[]>(endpoint);
    },
    enabled: !!userId || userId === undefined,
    staleTime: 30000,
  });
};

export const useTerminateSession = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (sessionId: string) => {
      return apiClient.delete(`${ADMIN_ENDPOINTS.USER_SESSIONS}/${sessionId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'user-sessions'] });
    },
  });
};

// Audit Logs
export const useAuditLogs = (filters?: {
  userId?: string;
  action?: string;
  resourceType?: string;
  startDate?: string;
  endDate?: string;
  page?: number;
  pageSize?: number;
}) => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.AUDIT_LOGS, filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters?.userId) params.append('userId', filters.userId);
      if (filters?.action) params.append('action', filters.action);
      if (filters?.resourceType) params.append('resourceType', filters.resourceType);
      if (filters?.startDate) params.append('startDate', filters.startDate);
      if (filters?.endDate) params.append('endDate', filters.endDate);
      if (filters?.page) params.append('page', filters.page.toString());
      if (filters?.pageSize) params.append('pageSize', filters.pageSize.toString());
      
      return apiClient.get<{
        logs: AuditLogEntry[];
        total: number;
        page: number;
        pageSize: number;
        totalPages: number;
      }>(`${ADMIN_ENDPOINTS.AUDIT_LOGS}?${params.toString()}`);
    },
    staleTime: 60000, // 1 minute
  });
};

// System Metrics
export const useSystemMetrics = (timeRange?: '1h' | '6h' | '24h' | '7d' | '30d') => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.SYSTEM_METRICS, timeRange],
    queryFn: async () => {
      const params = timeRange ? `?range=${timeRange}` : '';
      return apiClient.get<SystemMetrics[]>(`${ADMIN_ENDPOINTS.SYSTEM_METRICS}${params}`);
    },
    staleTime: 30000,
    refetchInterval: 30000, // Real-time updates every 30 seconds
  });
};

// Security Alerts
export const useSecurityAlerts = (filters?: {
  status?: string;
  severity?: string;
  type?: string;
  page?: number;
  pageSize?: number;
}) => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.SECURITY_ALERTS, filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters?.status) params.append('status', filters.status);
      if (filters?.severity) params.append('severity', filters.severity);
      if (filters?.type) params.append('type', filters.type);
      if (filters?.page) params.append('page', filters.page.toString());
      if (filters?.pageSize) params.append('pageSize', filters.pageSize.toString());
      
      return apiClient.get<{
        alerts: SecurityAlert[];
        total: number;
        page: number;
        pageSize: number;
        totalPages: number;
      }>(`${ADMIN_ENDPOINTS.SECURITY_ALERTS}?${params.toString()}`);
    },
    staleTime: 30000,
    refetchInterval: 60000,
  });
};

export const useUpdateSecurityAlert = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ id, ...alertData }: Partial<SecurityAlert> & { id: string }) => {
      return apiClient.put<SecurityAlert>(`${ADMIN_ENDPOINTS.SECURITY_ALERTS}/${id}`, alertData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.SECURITY_ALERTS });
    },
  });
};

// System Configuration
export const useSystemConfig = (category?: string) => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.SYSTEM_CONFIG, category],
    queryFn: async () => {
      const params = category ? `?category=${category}` : '';
      return apiClient.get<SystemConfiguration[]>(`${ADMIN_ENDPOINTS.SYSTEM_CONFIG}${params}`);
    },
    staleTime: 300000, // 5 minutes
  });
};

export const useUpdateConfig = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ key, value, category }: { key: string; value: any; category?: string }) => {
      return apiClient.put<SystemConfiguration>(`${ADMIN_ENDPOINTS.SYSTEM_CONFIG}/${key}`, {
        value,
        category,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.SYSTEM_CONFIG });
    },
  });
};

// Resource Quotas
export const useResourceQuotas = (filters?: {
  userId?: string;
  organizationId?: string;
  resourceType?: string;
}) => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.RESOURCE_QUOTAS, filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters?.userId) params.append('userId', filters.userId);
      if (filters?.organizationId) params.append('organizationId', filters.organizationId);
      if (filters?.resourceType) params.append('resourceType', filters.resourceType);
      
      return apiClient.get<ResourceQuota[]>(`${ADMIN_ENDPOINTS.RESOURCE_QUOTAS}?${params.toString()}`);
    },
    staleTime: 60000,
  });
};

export const useUpdateResourceQuota = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ id, ...quotaData }: Partial<ResourceQuota> & { id: string }) => {
      return apiClient.put<ResourceQuota>(`${ADMIN_ENDPOINTS.RESOURCE_QUOTAS}/${id}`, quotaData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.RESOURCE_QUOTAS });
    },
  });
};

// VM Templates
export const useVmTemplates = () => {
  return useQuery({
    queryKey: ADMIN_QUERY_KEYS.VM_TEMPLATES,
    queryFn: async () => {
      return apiClient.get<VmTemplate[]>(ADMIN_ENDPOINTS.VM_TEMPLATES);
    },
    staleTime: 300000, // 5 minutes
  });
};

export const useCreateVmTemplate = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (templateData: Partial<VmTemplate>) => {
      return apiClient.post<VmTemplate>(ADMIN_ENDPOINTS.VM_TEMPLATES, templateData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.VM_TEMPLATES });
    },
  });
};

export const useDeleteVmTemplate = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (templateId: string) => {
      return apiClient.delete(`${ADMIN_ENDPOINTS.VM_TEMPLATES}/${templateId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.VM_TEMPLATES });
    },
  });
};

// Performance Reports
export const usePerformanceReports = (reportType?: 'daily' | 'weekly' | 'monthly') => {
  return useQuery({
    queryKey: [...ADMIN_QUERY_KEYS.PERFORMANCE_REPORTS, reportType],
    queryFn: async () => {
      const params = reportType ? `?type=${reportType}` : '';
      return apiClient.get<PerformanceReport[]>(`${ADMIN_ENDPOINTS.PERFORMANCE_REPORTS}${params}`);
    },
    staleTime: 600000, // 10 minutes
  });
};

// WebSocket hook for real-time admin updates
export const useAdminWebSocket = () => {
  return apiClient.connectWebSocket('/admin/ws');
};

// Bulk operations
export const useBulkUserOperation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ 
      userIds, 
      operation, 
      data 
    }: { 
      userIds: string[]; 
      operation: 'activate' | 'suspend' | 'delete' | 'update_role' | 'send_email';
      data?: any;
    }) => {
      return apiClient.post(`${ADMIN_ENDPOINTS.USERS}/bulk`, {
        userIds,
        operation,
        data,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.USERS });
      queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.AUDIT_LOGS });
    },
  });
};

// System health check
export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['admin', 'system-health'],
    queryFn: async () => {
      return apiClient.get<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        services: Array<{
          name: string;
          status: 'up' | 'down' | 'degraded';
          response_time: number;
          uptime: string;
          error_rate: number;
        }>;
        overall_score: number;
      }>('/admin/system/health');
    },
    staleTime: 30000,
    refetchInterval: 30000,
  });
};