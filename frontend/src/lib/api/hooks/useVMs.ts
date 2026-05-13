'use client';

import { useQuery } from '@tanstack/react-query';
import { listVMs, type ListVMsParams } from '../vms';
import type { VM } from '../types';

export function useVMs(params?: ListVMsParams) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['vms', params],
    queryFn: () => listVMs(params),
    staleTime: 5_000,
  });
  const payload = data?.data;

  return {
    items: Array.isArray(payload) ? payload as VM[] : [],
    pagination: data?.pagination || { page: 1, pageSize: 10, total: 0, totalPages: 0 },
    isLoading,
    error,
  };
}
