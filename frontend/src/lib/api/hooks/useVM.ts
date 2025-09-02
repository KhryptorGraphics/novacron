"use client";

import { useQuery } from "@tanstack/react-query";
import { getVM } from "../vms";
import type { VM } from "../types";

export function useVM(id: string) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["vm", id],
    queryFn: () => getVM(id),
    enabled: Boolean(id),
  });

  return {
    vm: (data?.data ?? null) as VM | null,
    isLoading,
    error,
  };
}

