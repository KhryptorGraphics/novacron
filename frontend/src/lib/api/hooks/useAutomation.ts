"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  automationApi,
  type CreateJobRequest,
  type CreateWorkflowRequest,
} from "../automation";

function errorMessage(error: unknown): string | null {
  if (!error) {
    return null;
  }
  return error instanceof Error ? error.message : "API request failed";
}

export function useJobs() {
  const queryClient = useQueryClient();
  const jobsQuery = useQuery({
    queryKey: ["automation", "jobs"],
    queryFn: automationApi.listJobs,
    staleTime: 5_000,
  });

  const createMutation = useMutation({
    mutationFn: automationApi.createJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "jobs"] }),
  });
  const updateMutation = useMutation({
    mutationFn: ({ id, job }: { id: string; job: Partial<CreateJobRequest> }) => automationApi.updateJob(id, job),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "jobs"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: automationApi.deleteJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "jobs"] }),
  });
  const executeMutation = useMutation({
    mutationFn: automationApi.executeJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "jobs"] }),
  });

  return {
    jobs: jobsQuery.data ?? null,
    loading: jobsQuery.isLoading,
    error: errorMessage(jobsQuery.error),
    refetch: jobsQuery.refetch,
    createJob: (job: CreateJobRequest) => createMutation.mutateAsync(job),
    updateJob: (id: string, job: Partial<CreateJobRequest>) => updateMutation.mutateAsync({ id, job }),
    deleteJob: (id: string) => deleteMutation.mutateAsync(id),
    executeJob: (id: string) => executeMutation.mutateAsync(id),
  };
}

export function useJob(id: string | null) {
  const queryClient = useQueryClient();
  const jobQuery = useQuery({
    queryKey: ["automation", "job", id],
    queryFn: () => automationApi.getJob(id as string),
    enabled: Boolean(id),
    staleTime: 5_000,
  });
  const updateMutation = useMutation({
    mutationFn: (job: Partial<CreateJobRequest>) => automationApi.updateJob(id as string, job),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: () => automationApi.deleteJob(id as string),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "jobs"] }),
  });
  const executeMutation = useMutation({
    mutationFn: () => automationApi.executeJob(id as string),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation"] }),
  });

  return {
    job: jobQuery.data ?? null,
    loading: jobQuery.isLoading,
    error: errorMessage(jobQuery.error),
    refetch: jobQuery.refetch,
    updateJob: (job: Partial<CreateJobRequest>) => updateMutation.mutateAsync(job),
    deleteJob: () => deleteMutation.mutateAsync(),
    executeJob: () => executeMutation.mutateAsync(),
  };
}

export function useWorkflows() {
  const queryClient = useQueryClient();
  const workflowsQuery = useQuery({
    queryKey: ["automation", "workflows"],
    queryFn: automationApi.listWorkflows,
    staleTime: 5_000,
  });

  const createMutation = useMutation({
    mutationFn: automationApi.createWorkflow,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "workflows"] }),
  });
  const updateMutation = useMutation({
    mutationFn: ({ id, workflow }: { id: string; workflow: Partial<CreateWorkflowRequest> }) => automationApi.updateWorkflow(id, workflow),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "workflows"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: automationApi.deleteWorkflow,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "workflows"] }),
  });
  const executeMutation = useMutation({
    mutationFn: automationApi.executeWorkflow,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "workflows"] }),
  });

  return {
    workflows: workflowsQuery.data ?? null,
    loading: workflowsQuery.isLoading,
    error: errorMessage(workflowsQuery.error),
    refetch: workflowsQuery.refetch,
    createWorkflow: (workflow: CreateWorkflowRequest) => createMutation.mutateAsync(workflow),
    updateWorkflow: (id: string, workflow: Partial<CreateWorkflowRequest>) => updateMutation.mutateAsync({ id, workflow }),
    deleteWorkflow: (id: string) => deleteMutation.mutateAsync(id),
    executeWorkflow: (id: string) => executeMutation.mutateAsync(id),
  };
}

export function useWorkflow(id: string | null) {
  const queryClient = useQueryClient();
  const workflowQuery = useQuery({
    queryKey: ["automation", "workflow", id],
    queryFn: () => automationApi.getWorkflow(id as string),
    enabled: Boolean(id),
    staleTime: 5_000,
  });
  const updateMutation = useMutation({
    mutationFn: (workflow: Partial<CreateWorkflowRequest>) => automationApi.updateWorkflow(id as string, workflow),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: () => automationApi.deleteWorkflow(id as string),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation", "workflows"] }),
  });
  const executeMutation = useMutation({
    mutationFn: () => automationApi.executeWorkflow(id as string),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["automation"] }),
  });

  return {
    workflow: workflowQuery.data ?? null,
    loading: workflowQuery.isLoading,
    error: errorMessage(workflowQuery.error),
    refetch: workflowQuery.refetch,
    updateWorkflow: (workflow: Partial<CreateWorkflowRequest>) => updateMutation.mutateAsync(workflow),
    deleteWorkflow: () => deleteMutation.mutateAsync(),
    executeWorkflow: () => executeMutation.mutateAsync(),
  };
}

export function useWorkflowExecution(id: string | null) {
  const executionQuery = useQuery({
    queryKey: ["automation", "workflow-execution", id],
    queryFn: () => automationApi.getWorkflowExecution(id as string),
    enabled: Boolean(id),
    refetchInterval: 5_000,
  });

  return {
    execution: executionQuery.data ?? null,
    loading: executionQuery.isLoading,
    error: errorMessage(executionQuery.error),
    refetch: executionQuery.refetch,
  };
}
