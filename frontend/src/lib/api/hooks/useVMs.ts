"use client";

import { useQuery } from "@tanstack/react-query";
import { listVMs, type ListVMsParams } from "../vms";
import type { VM } from "../types";

export function useVMs(params?: ListVMsParams) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["vms", params],
    queryFn: () => listVMs(params),
    staleTime: 5_000,
  });

  // The canonical client assumes an { data, error } envelope, but the backend
  // VM endpoints return a bare array. Accept both so real inventory renders
  // instead of silently showing an empty list.
  // ponytail: tolerate both shapes here, not in the shared apiGet helper.
  const payload = data as any;
  const items: VM[] = Array.isArray(payload)
    ? (payload as VM[])
    : Array.isArray(payload?.data)
      ? (payload.data as VM[])
      : [];

  // apiGet never throws — it returns { data: null, error } on HTTP/network
  // failure — so useQuery.error stays null. Surface that envelope error, or a
  // 401/500 would render as an empty inventory instead of the error state.
  const envelopeError =
    !Array.isArray(payload) && payload?.error
      ? new Error(payload.error.message || payload.error.code || "Failed to load virtual machines")
      : null;

  return {
    items,
    pagination: data?.pagination || { page: 1, pageSize: 10, total: 0, totalPages: 0 },
    isLoading,
    error: error ?? envelopeError,
  };
}

