// React hooks for API integration
import { useState, useEffect, useCallback } from 'react';
import { 
  apiService, 
  HealthStatus 
} from '@/lib/api';
import { fabricApi, type FabricJob, type FabricJobRequest } from '@/lib/api/fabric';

export function useHealth() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const healthData = await apiService.getHealth();
      setHealth(healthData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check health');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return { health, loading, error, refetch: checkHealth };
}

export function useVMs() {
  const [vms, setVMs] = useState<{ vms: number; status: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVMs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const vmData = await apiService.listVMs();
      setVMs(vmData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch VMs');
    } finally {
      setLoading(false);
    }
  }, []);

  const createVM = useCallback(async (vmData: { name: string; vcpus?: number; cpu_shares?: number; memory_mb: number }) => {
    try {
      // cpu_shares stays optional in the wire type; 1024 is the backend
      // manager-path default and keeps the metadata record informative.
      const result = await apiService.createVM({ cpu_shares: 1024, ...vmData });
      await fetchVMs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to create VM');
    }
  }, [fetchVMs]);

  const deleteVM = useCallback(async (id: string) => {
    try {
      const result = await apiService.deleteVM(id);
      await fetchVMs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete VM');
    }
  }, [fetchVMs]);

  const startVM = useCallback(async (id: string) => {
    try {
      const result = await apiService.startVM(id);
      await fetchVMs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to start VM');
    }
  }, [fetchVMs]);

  const stopVM = useCallback(async (id: string) => {
    try {
      const result = await apiService.stopVM(id);
      await fetchVMs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to stop VM');
    }
  }, [fetchVMs]);

  useEffect(() => {
    fetchVMs();
  }, [fetchVMs]);

  return { 
    vms, 
    loading, 
    error, 
    refetch: fetchVMs,
    createVM,
    deleteVM,
    startVM,
    stopVM
  };
}

export function useVMMetrics(vmId: string | null) {
  const [metrics, setMetrics] = useState<{ id: string; cpu_usage: number; memory_usage: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    if (!vmId) return;
    
    try {
      setLoading(true);
      setError(null);
      const metricsData = await apiService.getVMMetrics(vmId);
      setMetrics(metricsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch VM metrics');
    } finally {
      setLoading(false);
    }
  }, [vmId]);

  useEffect(() => {
    if (vmId) {
      fetchMetrics();
      // Fetch metrics every 10 seconds
      const interval = setInterval(fetchMetrics, 10000);
      return () => clearInterval(interval);
    }
    return undefined;
  }, [fetchMetrics, vmId]);

  return { metrics, loading, error, refetch: fetchMetrics };
}

// Fabric Job Hooks (using /api/compute/jobs canonical backend endpoint)
export function useFabricJobs() {
  const [jobs, setJobs] = useState<FabricJob[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const jobData = await fabricApi.listJobs();
      setJobs(jobData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch fabric jobs');
    } finally {
      setLoading(false);
    }
  }, []);

  const submitJob = useCallback(async (jobData: FabricJobRequest) => {
    try {
      const result = await fabricApi.submitJob(jobData);
      await fetchJobs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to submit fabric job');
    }
  }, [fetchJobs]);

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      const result = await fabricApi.cancelJob(jobId);
      await fetchJobs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to cancel fabric job');
    }
  }, [fetchJobs]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  return { 
    jobs, 
    loading, 
    error, 
    refetch: fetchJobs,
    submitJob,
    cancelJob
  };
}

export function useFabricJob(jobId: string | null) {
  const [job, setJob] = useState<FabricJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;
    
    try {
      setLoading(true);
      setError(null);
      const jobData = await fabricApi.getJob(jobId);
      setJob(jobData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch fabric job');
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const cancelJob = useCallback(async () => {
    if (!jobId) return;
    
    try {
      const result = await fabricApi.cancelJob(jobId);
      setJob(prev => prev ? { ...prev, status: 'cancelled' } : null);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to cancel fabric job');
    }
  }, [jobId]);

  useEffect(() => {
    if (jobId) {
      fetchJob();
    }
  }, [fetchJob, jobId]);

  return { 
    job, 
    loading, 
    error, 
    refetch: fetchJob,
    cancelJob
  };
}

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<Event | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    
    const connect = () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/ws/metrics`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          setConnected(true);
          setError(null);
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setLastMessage(data);
          } catch (e) {
            setLastMessage(event.data);
          }
        };
        
        ws.onerror = (err) => {
          setError(err);
        };
        
        ws.onclose = () => {
          setConnected(false);
          // Reconnect after 5 seconds
          setTimeout(connect, 5000);
        };
      } catch (err) {
        setError(err instanceof Event ? err : null);
      }
    };

    connect();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  return { connected, lastMessage, error };
}