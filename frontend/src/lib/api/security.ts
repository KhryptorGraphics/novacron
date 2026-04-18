// Security API service for NovaCron frontend

import authService from '@/lib/auth';
import { buildApiUrl } from '@/lib/api/origin';

type RawRecord = Record<string, any>;

export class UnsupportedSecurityFeatureError extends Error {
  readonly code = 'UNSUPPORTED_SECURITY_FEATURE';

  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedSecurityFeatureError';
  }
}

export function isUnsupportedSecurityFeatureError(
  error: unknown,
): error is UnsupportedSecurityFeatureError {
  return error instanceof UnsupportedSecurityFeatureError;
}

function unsupportedSecurityFeature(message: string): never {
  throw new UnsupportedSecurityFeatureError(message);
}

export const securityCapabilities = {
  acknowledgeEvents: true,
  triggerComplianceChecks: true,
  remediateFindings: false,
  exportSecurityReports: true,
  manageAccessControls: true,
  manageSecurityConfig: false,
  performHealthChecks: false,
  createIncidents: true,
} as const;

export interface SecurityRoleDefinition {
  id: string;
  name: string;
  description: string;
  permissions: string[];
}

export interface SecurityPermissionDefinition {
  id: string;
  name: string;
  description: string;
}

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

function normalizeSeverity(value: unknown): SecurityEvent['severity'] {
  switch (String(value || '').toLowerCase()) {
    case 'critical':
    case 'high':
    case 'medium':
    case 'low':
    case 'info':
      return value as SecurityEvent['severity'];
    default:
      return 'info';
  }
}

function normalizeComplianceStatus(value: unknown): ComplianceRequirement['status'] {
  switch (String(value || '').toLowerCase()) {
    case 'compliant':
      return 'compliant';
    case 'partial':
    case 'partially_compliant':
      return 'partial';
    case 'pending':
      return 'pending';
    default:
      return 'non-compliant';
  }
}

function normalizeComplianceSeverity(status: ComplianceRequirement['status']): ComplianceRequirement['severity'] {
  switch (status) {
    case 'non-compliant':
      return 'high';
    case 'partial':
      return 'medium';
    case 'pending':
      return 'low';
    default:
      return 'low';
  }
}

function normalizeComplianceRequirements(payload: RawRecord): ComplianceRequirement[] {
  const frameworks = Array.isArray(payload.frameworks) ? payload.frameworks : [];
  const checkedAt =
    typeof payload.last_updated === 'string' ? payload.last_updated : new Date().toISOString();

  return frameworks.map((framework, index) => {
    const status = normalizeComplianceStatus(framework.status);
    const name = String(framework.name || `Framework ${index + 1}`);
    const remediationSteps =
      status === 'compliant'
        ? undefined
        : ['Review the framework findings and remediate the failing controls.'];

    return {
      id: String(framework.id || framework.name || `framework-${index}`),
      category: name,
      name,
      description: `${name} compliance status reported by the backend.`,
      status,
      severity: normalizeComplianceSeverity(status),
      lastChecked: checkedAt,
      ...(remediationSteps ? { remediationSteps } : {}),
    };
  });
}

function aggregateComplianceByCategory(
  requirements: ComplianceRequirement[]
): ComplianceByCategory[] {
  const byCategory = new Map<string, { compliant: number; total: number }>();

  for (const requirement of requirements) {
    const counts = byCategory.get(requirement.category) || { compliant: 0, total: 0 };
    counts.total += 1;
    if (requirement.status === 'compliant') {
      counts.compliant += 1;
    }
    byCategory.set(requirement.category, counts);
  }

  return Array.from(byCategory.entries()).map(([category, counts]) => ({
    category,
    compliant: counts.compliant,
    total: counts.total,
    percentage: counts.total === 0 ? 0 : Math.round((counts.compliant / counts.total) * 100),
  }));
}

function normalizeVulnerabilityFinding(finding: RawRecord, index: number): VulnerabilityFinding {
  const cve = typeof finding.cve === 'string' ? finding.cve : undefined;
  const cvssScore = typeof finding.cvssScore === 'number' ? finding.cvssScore : undefined;
  const references = Array.isArray(finding.references)
    ? finding.references.map(String)
    : undefined;

  return {
    id: String(finding.id || finding.cve || `finding-${index}`),
    title: String(finding.title || finding.name || finding.component || 'Vulnerability finding'),
    severity: normalizeSeverity(finding.severity),
    component: String(finding.component || finding.target || 'unknown'),
    description: String(finding.description || 'No backend description provided.'),
    remediation: String(
      finding.remediation ||
        'Review the backend scan output and remediate the affected component.'
    ),
    exploitable: Boolean(finding.exploitable),
    ...(cve ? { cve } : {}),
    ...(cvssScore !== undefined ? { cvssScore } : {}),
    ...(references ? { references } : {}),
  };
}

function summarizeThreatLevel(
  activeThreats: number,
  counts: SecurityMetrics['vulnerabilityCount']
): SecurityMetrics['threatLevel'] {
  if (activeThreats > 0 || counts.critical > 0) return 'critical';
  if (counts.high > 0) return 'high';
  if (counts.medium > 0) return 'medium';
  return 'low';
}

function deriveSecurityScore(
  complianceScore: number,
  activeThreats: number,
  counts: SecurityMetrics['vulnerabilityCount']
): number {
  const penalty =
    counts.critical * 12 + counts.high * 5 + counts.medium * 2 + activeThreats * 6;

  return Math.max(0, Math.min(100, Math.round(complianceScore - penalty)));
}

class SecurityAPIService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = buildApiUrl(endpoint);
    const token = this.getAuthToken();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const config: RequestInit = {
      headers,
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        if (response.status === 501) {
          unsupportedSecurityFeature('This security capability is not available on the canonical server.');
        }

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
    return authService.getToken() || '';
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
    await this.request(`/api/security/events/${eventId}/acknowledge`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
  }

  // Compliance
  async getComplianceRequirements(): Promise<ComplianceRequirement[]> {
    const response = await this.request<RawRecord>('/api/security/compliance');
    return normalizeComplianceRequirements(response);
  }

  async updateComplianceRequirement(
    id: string,
    updates: Partial<ComplianceRequirement>
  ): Promise<ComplianceRequirement> {
    void id;
    void updates;
    unsupportedSecurityFeature('Compliance requirement updates are not available on the canonical server.');
  }

  async triggerComplianceCheck(requirementId?: string): Promise<{ jobId: string }> {
    return this.request<{ jobId: string }>('/api/security/compliance/check', {
      method: 'POST',
      body: JSON.stringify(
        requirementId ? { requirement_id: requirementId } : {},
      ),
    });
  }

  async getComplianceByCategory(): Promise<ComplianceByCategory[]> {
    const requirements = await this.getComplianceRequirements();
    return aggregateComplianceByCategory(requirements);
  }

  // Vulnerability Scanning
  async getVulnerabilityScans(): Promise<VulnerabilityScan[]> {
    const response = await this.request<RawRecord>('/api/security/vulnerabilities');
    const findings = Array.isArray(response.vulnerabilities)
      ? response.vulnerabilities.map(normalizeVulnerabilityFinding)
      : [];
    const summary = response.summary || {};
    const lastScan = typeof response.last_scan === 'string' ? response.last_scan : undefined;
    const startTime = lastScan || new Date().toISOString();

    if (!lastScan && findings.length === 0) {
      return [];
    }

    return [
      {
        id: lastScan || 'latest-scan',
        target: 'cluster',
        type: 'infrastructure',
        status: 'completed',
        startTime,
        ...(lastScan ? { endTime: lastScan } : {}),
        vulnerabilities: {
          critical: Number(summary.critical || 0),
          high: Number(summary.high || 0),
          medium: Number(summary.medium || 0),
          low: Number(summary.low || 0),
          info: Number(summary.info || 0),
        },
        findings,
      },
    ];
  }

  async startVulnerabilityScan(
    target: string,
    scanType: string
  ): Promise<{ scanId: string }> {
    const response = await this.request<{ scan_id: string }>('/api/security/scan', {
      method: 'POST',
      body: JSON.stringify({ targets: [target], scan_types: [scanType], config: {} }),
    });

    return { scanId: response.scan_id };
  }

  async getScanFindings(scanId: string): Promise<VulnerabilityFinding[]> {
    const response = await this.request<RawRecord>(`/api/security/scan/${scanId}`);
    const findings = response.results?.findings;
    return Array.isArray(findings) ? findings.map(normalizeVulnerabilityFinding) : [];
  }

  async markFindingResolved(findingId: string): Promise<void> {
    void findingId;
    unsupportedSecurityFeature('Vulnerability remediation is not available on the canonical server.');
  }

  // Access Controls
  async getAccessControls(): Promise<AccessControl[]> {
    unsupportedSecurityFeature('Access control management is not available on the canonical server.');
  }

  async updateAccessControl(
    id: string,
    updates: Partial<AccessControl>
  ): Promise<AccessControl> {
    void id;
    void updates;
    unsupportedSecurityFeature('Access control management is not available on the canonical server.');
  }

  async testAccessControl(
    resource: string,
    subject: string,
    action: string
  ): Promise<{ allowed: boolean; reason?: string }> {
    void resource;
    void subject;
    void action;
    unsupportedSecurityFeature('Access control testing is not available on the canonical server.');
  }

  // Security Metrics
  async getSecurityMetrics(): Promise<SecurityMetrics> {
    const [threats, vulnerabilities, compliance] = await Promise.all([
      this.request<RawRecord>('/api/security/threats'),
      this.request<RawRecord>('/api/security/vulnerabilities'),
      this.request<RawRecord>('/api/security/compliance'),
    ]);

    const vulnerabilityCount = {
      critical: Number(vulnerabilities.summary?.critical || 0),
      high: Number(vulnerabilities.summary?.high || 0),
      medium: Number(vulnerabilities.summary?.medium || 0),
      low: Number(vulnerabilities.summary?.low || 0),
      info: Number(vulnerabilities.summary?.info || 0),
    };
    const activeThreats = Array.isArray(threats.threats) ? threats.threats.length : 0;
    const complianceScore = Number(compliance.compliance_score || 0);

    return {
      complianceScore,
      vulnerabilityCount,
      activeThreats,
      blockedThreats: 0,
      threatLevel: summarizeThreatLevel(activeThreats, vulnerabilityCount),
      securityScore: deriveSecurityScore(complianceScore, activeThreats, vulnerabilityCount),
    };
  }

  async getThreatTrends(
    timeRange: string = '24h',
    granularity: string = '1h'
  ): Promise<ThreatTrend[]> {
    void timeRange;
    void granularity;
    return [];
  }

  // Security Configuration
  async getSecurityConfig(): Promise<Record<string, any>> {
    unsupportedSecurityFeature('Security configuration is not available on the canonical server.');
  }

  async getRoles(): Promise<SecurityRoleDefinition[]> {
    const response = await this.request<{ roles: SecurityRoleDefinition[] }>('/api/security/rbac/roles');
    return Array.isArray(response.roles) ? response.roles : [];
  }

  async getPermissions(): Promise<SecurityPermissionDefinition[]> {
    const response = await this.request<{ permissions: SecurityPermissionDefinition[] }>(
      '/api/security/rbac/permissions',
    );
    return Array.isArray(response.permissions) ? response.permissions : [];
  }

  async getUserRoleAssignments(userId: string): Promise<string[]> {
    const response = await this.request<{ roles: string[] }>(`/api/security/rbac/user/${userId}/roles`);
    return Array.isArray(response.roles) ? response.roles : [];
  }

  async assignUserRoles(userId: string, roles: string[]): Promise<string[]> {
    const response = await this.request<{ roles: string[] }>(`/api/security/rbac/user/${userId}/roles`, {
      method: 'POST',
      body: JSON.stringify({ roles }),
    });
    return Array.isArray(response.roles) ? response.roles : [];
  }

  async updateSecurityConfig(config: Record<string, any>): Promise<void> {
    void config;
    unsupportedSecurityFeature('Security configuration is not available on the canonical server.');
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
      `/api/security/audit/events?${params.toString()}`
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
    unsupportedSecurityFeature('Security health checks are not available on the canonical server.');
  }

  // Incident Management
  async createSecurityIncident(incident: {
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    type: string;
    affectedSystems?: string[];
  }): Promise<{ incidentId: string }> {
    const response = await this.request<{ incidentId: string }>('/api/security/incidents', {
      method: 'POST',
      body: JSON.stringify({
        title: incident.title,
        description: incident.description,
        severity: incident.severity,
        type: incident.type,
        affectedSystems: incident.affectedSystems || [],
      }),
    });
    return { incidentId: response.incidentId };
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

  async exportSecurityReport(kind: 'compliance' | 'audit', format: 'json' | 'csv' = 'json'): Promise<void> {
    const endpoint =
      kind === 'audit'
        ? `/api/security/audit/export?format=${format}`
        : `/api/security/compliance/export?format=${format}`;

    const url = buildApiUrl(endpoint);
    const token = this.getAuthToken();
    const response = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });

    if (!response.ok) {
      throw new Error(`Failed to export ${kind} report`);
    }

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${kind}-report.${format}`;
    link.click();
    window.URL.revokeObjectURL(downloadUrl);
  }
}

export const securityAPI = new SecurityAPIService();
export default securityAPI;
