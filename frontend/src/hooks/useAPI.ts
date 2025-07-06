// React hooks for API integration
import { useState, useEffect, useCallback } from 'react';
import { apiService, VMInfo, HealthStatus } from '@/lib/api';

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
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
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

  const createVM = useCallback(async (vmData: { name: string; cpu_shares: number; memory_mb: number }) => {
    try {
      const result = await apiService.createVM(vmData);
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
  }, [fetchMetrics, vmId]);

  return { metrics, loading, error, refetch: fetchMetrics };
}

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    const ws = apiService.createWebSocket(
      (data) => {
        setLastMessage(data);
        setConnected(true);
      },
      (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      }
    );

    if (ws) {
      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      
      return () => {
        ws.close();
      };
    }
  }, []);

  return { connected, lastMessage };
}