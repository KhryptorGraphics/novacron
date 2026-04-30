"use client";

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  orchestrationApi,
  type EngineStatus,
  type OrchestrationPolicy,
  type MetricPoint,
} from '../orchestration';

function errorMessage(error: unknown): string | null {
  if (!error) {
    return null;
  }
  return error instanceof Error ? error.message : 'Orchestration API request failed';
}

function normalizeMetricPoint(data: Partial<MetricPoint> & Record<string, unknown>): MetricPoint {
  return {
    timestamp: typeof data.timestamp === 'string' ? data.timestamp : new Date().toISOString(),
    cpuUsage: Number(data.cpuUsage ?? data.cpu_usage ?? 0),
    memoryUsage: Number(data.memoryUsage ?? data.memory_usage ?? 0),
    networkIO: Number(data.networkIO ?? data.network_io ?? 0),
    diskIO: Number(data.diskIO ?? data.disk_io ?? 0),
    decisionsPerMinute: Number(data.decisionsPerMinute ?? data.decisions_per_minute ?? 0),
    responseTime: Number(data.responseTime ?? data.response_time ?? 0),
  };
}

async function optional<T>(request: Promise<T>, fallback: T): Promise<T> {
  try {
    return await request;
  } catch {
    return fallback;
  }
}

export function useOrchestrationDashboard() {
  const dashboardQuery = useQuery({
    queryKey: ['orchestration', 'dashboard'],
    queryFn: async () => {
      const [engineStatus, recentDecisions, policies, mlModels] = await Promise.all([
        orchestrationApi.getStatus(),
        optional(orchestrationApi.listDecisions(10), []),
        optional(orchestrationApi.listPolicies(), []),
        optional(orchestrationApi.listModels(), []),
      ]);

      return {
        engineStatus,
        recentDecisions,
        policies,
        activePolicies: policies.filter((policy) => policy.enabled),
        mlModels,
      };
    },
    refetchInterval: 30_000,
  });

  return {
    engineStatus: dashboardQuery.data?.engineStatus ?? null,
    recentDecisions: dashboardQuery.data?.recentDecisions ?? [],
    policies: dashboardQuery.data?.policies ?? [],
    activePolicies: dashboardQuery.data?.activePolicies ?? [],
    mlModels: dashboardQuery.data?.mlModels ?? [],
    loading: dashboardQuery.isLoading,
    error: errorMessage(dashboardQuery.error),
    refetch: dashboardQuery.refetch,
  };
}

export function useScalingMetrics(timeRange: string) {
  const scalingQuery = useQuery({
    queryKey: ['orchestration', 'scaling', timeRange],
    queryFn: async () => {
      const [scalingData, recentEvents] = await Promise.all([
        orchestrationApi.listScalingMetrics(timeRange),
        orchestrationApi.listScalingEvents(20),
      ]);
      return { scalingData, recentEvents };
    },
    refetchInterval: 30_000,
  });

  return {
    scalingData: scalingQuery.data?.scalingData ?? [],
    recentEvents: scalingQuery.data?.recentEvents ?? [],
    loading: scalingQuery.isLoading,
    error: errorMessage(scalingQuery.error),
    refetch: scalingQuery.refetch,
  };
}

export function useRealtimeOrchestrationMetrics(autoRefresh: boolean) {
  const metricsQuery = useQuery({
    queryKey: ['orchestration', 'metrics', 'realtime'],
    queryFn: async () => normalizeMetricPoint(await orchestrationApi.getRealtimeMetrics()),
    refetchInterval: autoRefresh ? 5_000 : false,
  });

  return {
    metric: metricsQuery.data ?? null,
    loading: metricsQuery.isLoading,
    error: errorMessage(metricsQuery.error),
    refetch: metricsQuery.refetch,
  };
}

export function usePolicyMutations() {
  const queryClient = useQueryClient();
  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['orchestration'] });

  const createMutation = useMutation({
    mutationFn: orchestrationApi.createPolicy,
    onSuccess: invalidate,
  });
  const updateMutation = useMutation({
    mutationFn: ({ id, policy }: { id: string; policy: OrchestrationPolicy }) => orchestrationApi.updatePolicy(id, policy),
    onSuccess: invalidate,
  });
  const deleteMutation = useMutation({
    mutationFn: orchestrationApi.deletePolicy,
    onSuccess: invalidate,
  });

  return {
    createPolicy: (policy: OrchestrationPolicy) => createMutation.mutateAsync(policy),
    updatePolicy: (id: string, policy: OrchestrationPolicy) => updateMutation.mutateAsync({ id, policy }),
    deletePolicy: (id: string) => deleteMutation.mutateAsync(id),
  };
}

export function useModelMutations() {
  const queryClient = useQueryClient();
  const retrainMutation = useMutation({
    mutationFn: orchestrationApi.retrainModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['orchestration'] }),
  });

  return {
    retrainModel: (modelType: string) => retrainMutation.mutateAsync(modelType),
    downloadModel: orchestrationApi.downloadModel,
  };
}

export type { EngineStatus, OrchestrationPolicy };
