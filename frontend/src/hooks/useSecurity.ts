import { useState, useEffect, useCallback, useRef } from 'react';
import { securityAPI, SecurityEvent, ComplianceRequirement, VulnerabilityScan, SecurityMetrics, ThreatTrend, ComplianceByCategory } from '@/lib/api/security';
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
      setEvents([]);
      setTotal(0);
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
      setRequirements([]);
      setCategoryBreakdown([]);
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
      setScans([]);
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
      setMetrics(null);
      setThreatTrends([]);
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
