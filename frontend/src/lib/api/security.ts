// Security API service for NovaCron frontend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090';

export interface SecurityEvent {
  id: string;
  timestamp: string;
  type: 'auth' | 'access' | 'modification' | 'anomaly' | 'threat';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  source: string;
  user?: string;
  resource?: string;
  action: string;
  result: 'success' | 'failure' | 'blocked';
  details: string;
  ip?: string;
  location?: string;
  metadata?: Record<string, any>;
}

export interface ComplianceRequirement {
  id: string;
  category: string;
  name: string;
  description: string;
  status: 'compliant' | 'non-compliant' | 'partial' | 'pending';
  severity: 'critical' | 'high' | 'medium' | 'low';
  lastChecked: string;
  evidence?: string[];
  remediationSteps?: string[];
  nextCheck?: string;
}

export interface VulnerabilityScan {
  id: string;
  target: string;
  type: 'network' | 'application' | 'container' | 'infrastructure';
  status: 'running' | 'completed' | 'failed' | 'scheduled';
  startTime: string;
  endTime?: string;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
  findings?: VulnerabilityFinding[];
}

export interface VulnerabilityFinding {
  id: string;
  cve?: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  component: string;
  description: string;
  remediation: string;
  exploitable: boolean;
  cvssScore?: number;
  references?: string[];
}

export interface AccessControl {
  id: string;
  resource: string;
  policy: string;
  type: 'rbac' | 'abac' | 'dac' | 'mac';
  rules: AccessRule[];
  enforced: boolean;
  lastModified: string;
}

export interface AccessRule {
  id: string;
  subject: string;
  action: string;
  effect: 'allow' | 'deny';
  conditions?: Record<string, any>;
}

export interface SecurityMetrics {
  securityScore: number;
  complianceScore: number;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  activeThreats: number;
  blockedThreats: number;
  vulnerabilityCount: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
}

export interface ThreatTrend {
  timestamp: string;
  threats: number;
  blocked: number;
  severity_breakdown: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

export interface ComplianceByCategory {
  category: string;
  compliant: number;
  total: number;
  percentage: number;
}

class SecurityAPIService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;

    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`,
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
      console.error(`Security API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  private getAuthToken(): string {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('authToken') || '';
    }
    return '';
  }

  // Security Events
  async getSecurityEvents(
    limit: number = 100,
    offset: number = 0,
    severity?: string,
    type?: string,
    timeRange?: string
  ): Promise<{ events: SecurityEvent[]; total: number }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });

    if (severity && severity !== 'all') params.append('severity', severity);
    if (type && type !== 'all') params.append('type', type);
    if (timeRange) params.append('timeRange', timeRange);

    return this.request<{ events: SecurityEvent[]; total: number }>(
      `/api/security/events?${params.toString()}`
    );
  }

  async acknowledgeSecurityEvent(eventId: string): Promise<void> {
    return this.request<void>(`/api/security/events/${eventId}/acknowledge`, {
      method: 'POST',
    });
  }

  // Compliance
  async getComplianceRequirements(): Promise<ComplianceRequirement[]> {
    return this.request<ComplianceRequirement[]>('/api/security/compliance/requirements');
  }

  async updateComplianceRequirement(
    id: string,
    updates: Partial<ComplianceRequirement>
  ): Promise<ComplianceRequirement> {
    return this.request<ComplianceRequirement>(`/api/security/compliance/requirements/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async triggerComplianceCheck(requirementId?: string): Promise<{ jobId: string }> {
    const endpoint = requirementId
      ? `/api/security/compliance/check/${requirementId}`
      : '/api/security/compliance/check';

    return this.request<{ jobId: string }>(endpoint, {
      method: 'POST',
    });
  }

  async getComplianceByCategory(): Promise<ComplianceByCategory[]> {
    return this.request<ComplianceByCategory[]>('/api/security/compliance/by-category');
  }

  // Vulnerability Scanning
  async getVulnerabilityScans(): Promise<VulnerabilityScan[]> {
    return this.request<VulnerabilityScan[]>('/api/security/vulnerabilities/scans');
  }

  async startVulnerabilityScan(
    target: string,
    scanType: string
  ): Promise<{ scanId: string }> {
    return this.request<{ scanId: string }>('/api/security/vulnerabilities/scan', {
      method: 'POST',
      body: JSON.stringify({ target, type: scanType }),
    });
  }

  async getScanFindings(scanId: string): Promise<VulnerabilityFinding[]> {
    return this.request<VulnerabilityFinding[]>(`/api/security/vulnerabilities/scans/${scanId}/findings`);
  }

  async markFindingResolved(findingId: string): Promise<void> {
    return this.request<void>(`/api/security/vulnerabilities/findings/${findingId}/resolve`, {
      method: 'POST',
    });
  }

  // Access Controls
  async getAccessControls(): Promise<AccessControl[]> {
    return this.request<AccessControl[]>('/api/security/access-controls');
  }

  async updateAccessControl(
    id: string,
    updates: Partial<AccessControl>
  ): Promise<AccessControl> {
    return this.request<AccessControl>(`/api/security/access-controls/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async testAccessControl(
    resource: string,
    subject: string,
    action: string
  ): Promise<{ allowed: boolean; reason?: string }> {
    return this.request<{ allowed: boolean; reason?: string }>('/api/security/access-controls/test', {
      method: 'POST',
      body: JSON.stringify({ resource, subject, action }),
    });
  }

  // Security Metrics
  async getSecurityMetrics(): Promise<SecurityMetrics> {
    return this.request<SecurityMetrics>('/api/security/metrics');
  }

  async getThreatTrends(
    timeRange: string = '24h',
    granularity: string = '1h'
  ): Promise<ThreatTrend[]> {
    const params = new URLSearchParams({
      timeRange,
      granularity,
    });

    return this.request<ThreatTrend[]>(`/api/security/metrics/threats?${params.toString()}`);
  }

  // Security Configuration
  async getSecurityConfig(): Promise<Record<string, any>> {
    return this.request<Record<string, any>>('/api/security/config');
  }

  async updateSecurityConfig(config: Record<string, any>): Promise<void> {
    return this.request<void>('/api/security/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  // Audit Trail
  async getAuditTrail(
    limit: number = 100,
    offset: number = 0,
    resource?: string,
    action?: string,
    user?: string,
    timeRange?: string
  ): Promise<{ events: SecurityEvent[]; total: number }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });

    if (resource) params.append('resource', resource);
    if (action) params.append('action', action);
    if (user) params.append('user', user);
    if (timeRange) params.append('timeRange', timeRange);

    return this.request<{ events: SecurityEvent[]; total: number }>(
      `/api/security/audit?${params.toString()}`
    );
  }

  // Security Health Check
  async performHealthCheck(): Promise<{
    overall: 'healthy' | 'warning' | 'critical';
    checks: Array<{
      name: string;
      status: 'passed' | 'failed' | 'warning';
      message: string;
      details?: Record<string, any>;
    }>;
  }> {
    return this.request<any>('/api/security/health');
  }

  // Incident Management
  async createSecurityIncident(incident: {
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    type: string;
    affectedSystems?: string[];
  }): Promise<{ incidentId: string }> {
    return this.request<{ incidentId: string }>('/api/security/incidents', {
      method: 'POST',
      body: JSON.stringify(incident),
    });
  }

  async getSecurityIncidents(): Promise<Array<{
    id: string;
    title: string;
    status: 'open' | 'investigating' | 'resolved' | 'closed';
    severity: 'critical' | 'high' | 'medium' | 'low';
    createdAt: string;
    resolvedAt?: string;
  }>> {
    return this.request<any>('/api/security/incidents');
  }
}

export const securityAPI = new SecurityAPIService();
export default securityAPI;