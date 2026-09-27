'use client';

import { useQuery } from '@tanstack/react-query';
import { getVM } from '../vms';
import type { VM } from '../types';

function isVM(value: unknown): value is VM {
  return typeof value === 'object' && value !== null && 'id' in value && typeof value.id === 'string';
}

function envelopeErrorMessage(value: unknown): string | null {
  if (typeof value !== 'object' || value === null || !('error' in value)) {
    return null;
  }
  const error = value.error;
  if (typeof error !== 'object' || error === null) {
    return null;
  }
  const message = 'message' in error && typeof error.message === 'string' ? error.message : '';
  const code = 'code' in error && typeof error.code === 'string' ? error.code : '';
  return message || code || 'Failed to load virtual machine';
}

export function useVM(id: string) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['vm', id],
    queryFn: () => getVM(id),
    enabled: Boolean(id),
  });

  // The canonical client assumes an { data, error } envelope, but GET
  // /vms/{id} returns a bare VM object (mirrors useVMs.ts's array handling
  // for GET /vms). Accept both so a real VM's detail page renders.
  const payload: unknown = data;
  let vm: VM | null = null;
  if (isVM(payload)) {
    vm = payload;
  } else if (typeof payload === 'object' && payload !== null && 'data' in payload && isVM(payload.data)) {
    vm = payload.data;
  }

  // apiGet never throws — it returns { data: null, error } on HTTP/network
  // failure — so useQuery.error stays null; surface that envelope error.
  const envelopeMessage = vm === null ? envelopeErrorMessage(payload) : null;

  return {
    vm,
    isLoading,
    error: error ?? (envelopeMessage ? new Error(envelopeMessage) : null),
  };
}
