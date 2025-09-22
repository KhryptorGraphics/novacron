import { useState, useEffect, useCallback, useRef } from 'react';
import { securityAPI, SecurityEvent, ComplianceRequirement, VulnerabilityScan, AccessControl, SecurityMetrics, ThreatTrend, ComplianceByCategory } from '@/lib/api/security';
import { useToast } from '@/components/ui/use-toast';

export function useSecurityEvents(autoRefresh = true, refreshInterval = 30000) {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  const intervalRef = useRef<NodeJS.Timeout>();

  const fetchEvents = useCallback(async (
    limit = 100,
    offset = 0,
    severity?: string,
    type?: string,
    timeRange?: string
  ) => {
    try {
      setLoading(true);
      setError(null);
      const response = await securityAPI.getSecurityEvents(limit, offset, severity, type, timeRange);
      setEvents(response.events);
      setTotal(response.total);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch security events';
      setError(errorMessage);
      // Fallback to mock data on error
      console.warn('Falling back to mock security events data');
      setEvents([
        {
          id: '1',
          timestamp: new Date().toISOString(),
          type: 'auth',
          severity: 'medium',
          source: 'auth-service',
          user: 'admin@example.com',
          action: 'login',
          result: 'success',
          details: 'Successful login from new location',
          ip: '192.168.1.100',
          location: 'New York, US'
        },
        {
          id: '2',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          type: 'threat',
          severity: 'high',
          source: 'firewall',
          action: 'block_intrusion',
          result: 'blocked',
          details: 'Blocked potential SQL injection attempt',
          ip: '203.0.113.42'
        }
      ]);
      setTotal(2);
    } finally {
      setLoading(false);
    }
  }, []);

  const acknowledgeEvent = useCallback(async (eventId: string) => {
    try {
      await securityAPI.acknowledgeSecurityEvent(eventId);
      toast({
        title: "Event Acknowledged",
        description: "Security event has been acknowledged",
      });
      // Refresh events
      fetchEvents();
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to acknowledge security event",
        variant: "destructive",
      });
    }
  }, [fetchEvents, toast]);

  useEffect(() => {
    fetchEvents();

    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchEvents();
      }, refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchEvents, autoRefresh, refreshInterval]);

  return {
    events,
    total,
    loading,
    error,
    fetchEvents,
    acknowledgeEvent,
  };
}

export function useCompliance() {
  const [requirements, setRequirements] = useState<ComplianceRequirement[]>([]);
  const [categoryBreakdown, setCategoryBreakdown] = useState<ComplianceByCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchRequirements = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [requirementsData, categoryData] = await Promise.all([
        securityAPI.getComplianceRequirements(),
        securityAPI.getComplianceByCategory()
      ]);
      setRequirements(requirementsData);
      setCategoryBreakdown(categoryData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch compliance data';
      setError(errorMessage);
      // Fallback to mock data
      console.warn('Falling back to mock compliance data');
      setRequirements([
        {
          id: '1',
          category: 'Data Protection',
          name: 'Encryption at Rest',
          description: 'All sensitive data must be encrypted at rest using AES-256',
          status: 'compliant',
          severity: 'critical',
          lastChecked: new Date().toISOString(),
          evidence: ['encryption-audit.pdf', 'key-management-policy.pdf']
        },
        {
          id: '2',
          category: 'Access Control',
          name: 'Multi-Factor Authentication',
          description: 'MFA must be enabled for all administrative accounts',
          status: 'partial',
          severity: 'high',
          lastChecked: new Date(Date.now() - 86400000).toISOString(),
          remediationSteps: ['Enable MFA for remaining 3 admin accounts', 'Update authentication policy']
        }
      ]);
      setCategoryBreakdown([
        { category: 'Data Protection', compliant: 8, total: 10, percentage: 80 },
        { category: 'Access Control', compliant: 6, total: 8, percentage: 75 }
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  const triggerComplianceCheck = useCallback(async (requirementId?: string) => {
    try {
      const response = await securityAPI.triggerComplianceCheck(requirementId);
      toast({
        title: "Compliance Check Started",
        description: `Check job ${response.jobId} has been initiated`,
      });
      // Refresh after a delay
      setTimeout(fetchRequirements, 2000);
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to start compliance check",
        variant: "destructive",
      });
    }
  }, [fetchRequirements, toast]);

  useEffect(() => {
    fetchRequirements();
  }, [fetchRequirements]);

  return {
    requirements,
    categoryBreakdown,
    loading,
    error,
    fetchRequirements,
    triggerComplianceCheck,
  };
}

export function useVulnerabilityScans() {
  const [scans, setScans] = useState<VulnerabilityScan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchScans = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const scansData = await securityAPI.getVulnerabilityScans();
      setScans(scansData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch vulnerability scans';
      setError(errorMessage);
      // Fallback to mock data
      console.warn('Falling back to mock vulnerability scan data');
      setScans([
        {
          id: '1',
          target: 'production-cluster',
          type: 'infrastructure',
          status: 'completed',
          startTime: new Date(Date.now() - 7200000).toISOString(),
          endTime: new Date(Date.now() - 3600000).toISOString(),
          vulnerabilities: { critical: 0, high: 2, medium: 5, low: 12, info: 23 },
          findings: [
            {
              id: '1',
              cve: 'CVE-2024-1234',
              title: 'Outdated SSL/TLS Configuration',
              severity: 'high',
              component: 'nginx:1.18',
              description: 'The SSL/TLS configuration uses outdated cipher suites',
              remediation: 'Update nginx configuration to use modern cipher suites',
              exploitable: true
            }
          ]
        }
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  const startScan = useCallback(async (target: string, scanType: string) => {
    try {
      const response = await securityAPI.startVulnerabilityScan(target, scanType);
      toast({
        title: "Scan Started",
        description: `Vulnerability scan ${response.scanId} has been initiated`,
      });
      // Refresh scans
      fetchScans();
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to start vulnerability scan",
        variant: "destructive",
      });
    }
  }, [fetchScans, toast]);

  useEffect(() => {
    fetchScans();
  }, [fetchScans]);

  return {
    scans,
    loading,
    error,
    fetchScans,
    startScan,
  };
}

export function useSecurityMetrics(autoRefresh = true, refreshInterval = 60000) {
  const [metrics, setMetrics] = useState<SecurityMetrics | null>(null);
  const [threatTrends, setThreatTrends] = useState<ThreatTrend[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout>();

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [metricsData, trendsData] = await Promise.all([
        securityAPI.getSecurityMetrics(),
        securityAPI.getThreatTrends()
      ]);
      setMetrics(metricsData);
      setThreatTrends(trendsData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch security metrics';
      setError(errorMessage);
      // Fallback to mock data
      console.warn('Falling back to mock security metrics data');
      setMetrics({
        securityScore: 87,
        complianceScore: 92,
        threatLevel: 'medium',
        activeThreats: 3,
        blockedThreats: 142,
        vulnerabilityCount: {
          critical: 0,
          high: 2,
          medium: 5,
          low: 12,
          info: 23
        }
      });
      setThreatTrends([
        { timestamp: '00:00', threats: 12, blocked: 12, severity_breakdown: { critical: 0, high: 2, medium: 5, low: 5 } },
        { timestamp: '04:00', threats: 8, blocked: 8, severity_breakdown: { critical: 0, high: 1, medium: 3, low: 4 } },
        { timestamp: '08:00', threats: 15, blocked: 14, severity_breakdown: { critical: 1, high: 3, medium: 6, low: 5 } },
        { timestamp: '12:00', threats: 22, blocked: 21, severity_breakdown: { critical: 0, high: 4, medium: 8, low: 10 } }
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();

    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchMetrics();
      }, refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchMetrics, autoRefresh, refreshInterval]);

  return {
    metrics,
    threatTrends,
    loading,
    error,
    fetchMetrics,
  };
}