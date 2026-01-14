// React hooks for API integration
import { useState, useEffect, useCallback } from 'react';
import { 
  apiService, 
  VMInfo, 
  HealthStatus, 
  CronJob, 
  CreateJobRequest, 
  Workflow, 
  CreateWorkflowRequest, 
  WorkflowExecution 
} from '@/lib/api';

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

// Job Hooks
export function useJobs() {
  const [jobs, setJobs] = useState<CronJob[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const jobData = await apiService.listJobs();
      setJobs(jobData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs');
    } finally {
      setLoading(false);
    }
  }, []);

  const createJob = useCallback(async (jobData: CreateJobRequest) => {
    try {
      const result = await apiService.createJob(jobData);
      await fetchJobs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to create job');
    }
  }, [fetchJobs]);

  const updateJob = useCallback(async (id: string, jobData: Partial<CreateJobRequest>) => {
    try {
      const result = await apiService.updateJob(id, jobData);
      await fetchJobs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to update job');
    }
  }, [fetchJobs]);

  const deleteJob = useCallback(async (id: string) => {
    try {
      const result = await apiService.deleteJob(id);
      await fetchJobs(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete job');
    }
  }, [fetchJobs]);

  const executeJob = useCallback(async (id: string) => {
    try {
      const result = await apiService.executeJob(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to execute job');
    }
  }, []);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  return { 
    jobs, 
    loading, 
    error, 
    refetch: fetchJobs,
    createJob,
    updateJob,
    deleteJob,
    executeJob
  };
}

export function useJob(id: string | null) {
  const [job, setJob] = useState<CronJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJob = useCallback(async () => {
    if (!id) return;
    
    try {
      setLoading(true);
      setError(null);
      const jobData = await apiService.getJob(id);
      setJob(jobData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch job');
    } finally {
      setLoading(false);
    }
  }, [id]);

  const updateJob = useCallback(async (jobData: Partial<CreateJobRequest>) => {
    if (!id) return;
    
    try {
      const result = await apiService.updateJob(id, jobData);
      setJob(result);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to update job');
    }
  }, [id]);

  const deleteJob = useCallback(async () => {
    if (!id) return;
    
    try {
      const result = await apiService.deleteJob(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete job');
    }
  }, [id]);

  const executeJob = useCallback(async () => {
    if (!id) return;
    
    try {
      const result = await apiService.executeJob(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to execute job');
    }
  }, [id]);

  useEffect(() => {
    if (id) {
      fetchJob();
    }
  }, [fetchJob, id]);

  return { 
    job, 
    loading, 
    error, 
    refetch: fetchJob,
    updateJob,
    deleteJob,
    executeJob
  };
}

// Workflow Hooks
export function useWorkflows() {
  const [workflows, setWorkflows] = useState<Workflow[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflows = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const workflowData = await apiService.listWorkflows();
      setWorkflows(workflowData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch workflows');
    } finally {
      setLoading(false);
    }
  }, []);

  const createWorkflow = useCallback(async (workflowData: CreateWorkflowRequest) => {
    try {
      const result = await apiService.createWorkflow(workflowData);
      await fetchWorkflows(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to create workflow');
    }
  }, [fetchWorkflows]);

  const updateWorkflow = useCallback(async (id: string, workflowData: Partial<CreateWorkflowRequest>) => {
    try {
      const result = await apiService.updateWorkflow(id, workflowData);
      await fetchWorkflows(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to update workflow');
    }
  }, [fetchWorkflows]);

  const deleteWorkflow = useCallback(async (id: string) => {
    try {
      const result = await apiService.deleteWorkflow(id);
      await fetchWorkflows(); // Refresh the list
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete workflow');
    }
  }, [fetchWorkflows]);

  const executeWorkflow = useCallback(async (id: string) => {
    try {
      const result = await apiService.executeWorkflow(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to execute workflow');
    }
  }, []);

  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);

  return { 
    workflows, 
    loading, 
    error, 
    refetch: fetchWorkflows,
    createWorkflow,
    updateWorkflow,
    deleteWorkflow,
    executeWorkflow
  };
}

export function useWorkflow(id: string | null) {
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflow = useCallback(async () => {
    if (!id) return;
    
    try {
      setLoading(true);
      setError(null);
      const workflowData = await apiService.getWorkflow(id);
      setWorkflow(workflowData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch workflow');
    } finally {
      setLoading(false);
    }
  }, [id]);

  const updateWorkflow = useCallback(async (workflowData: Partial<CreateWorkflowRequest>) => {
    if (!id) return;
    
    try {
      const result = await apiService.updateWorkflow(id, workflowData);
      setWorkflow(result);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to update workflow');
    }
  }, [id]);

  const deleteWorkflow = useCallback(async () => {
    if (!id) return;
    
    try {
      const result = await apiService.deleteWorkflow(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete workflow');
    }
  }, [id]);

  const executeWorkflow = useCallback(async () => {
    if (!id) return;
    
    try {
      const result = await apiService.executeWorkflow(id);
      return result;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to execute workflow');
    }
  }, [id]);

  useEffect(() => {
    if (id) {
      fetchWorkflow();
    }
  }, [fetchWorkflow, id]);

  return { 
    workflow, 
    loading, 
    error, 
    refetch: fetchWorkflow,
    updateWorkflow,
    deleteWorkflow,
    executeWorkflow
  };
}

export function useWorkflowExecution(id: string | null) {
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchExecution = useCallback(async () => {
    if (!id) return;
    
    try {
      setLoading(true);
      setError(null);
      const executionData = await apiService.getWorkflowExecution(id);
      setExecution(executionData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch execution');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    if (id) {
      fetchExecution();
      // Poll for updates every 5 seconds
      const interval = setInterval(fetchExecution, 5000);
      return () => clearInterval(interval);
    }
  }, [fetchExecution, id]);

  return { 
    execution, 
    loading, 
    error, 
    refetch: fetchExecution
  };
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