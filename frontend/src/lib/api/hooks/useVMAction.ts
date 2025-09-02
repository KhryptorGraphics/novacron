"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { postVMAction } from "../vms";
import type { VM } from "../types";

export function useVMAction(id: string, action: "start" | "stop" | "restart" | "pause" | "resume", options?: { role?: "viewer" | "operator" }) {
  const qc = useQueryClient();
  return useMutation({
    mutationKey: ["vm-action", id, action, options?.role],
    mutationFn: () => postVMAction(id, action, { role: options?.role }),
    onMutate: async () => {
      // placeholder for optimistic update
    },
    onSuccess: async () => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["vm", id] }),
        qc.invalidateQueries({ queryKey: ["vms"] }),
      ]);
    },
  });
}

