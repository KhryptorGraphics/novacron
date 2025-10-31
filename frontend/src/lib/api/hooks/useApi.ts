/**
 * React hooks for API calls with loading and error states
 */

import { useState, useEffect, useCallback } from 'react';
import { apiClient, ApiError } from '../api-client';

export interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  refetch: () => Promise<void>;
}

/**
 * Hook for GET requests with automatic fetching
 */
export function useApi<T>(
  endpoint: string,
  options?: {
    enabled?: boolean;
    dependencies?: any[];
  }
): UseApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const enabled = options?.enabled !== false;
  const dependencies = options?.dependencies || [];

  const fetchData = useCallback(async () => {
    if (!enabled) return;

    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.get<T>(endpoint);
      setData(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [endpoint, enabled]);

  useEffect(() => {
    fetchData();
  }, [fetchData, ...dependencies]);

  return {
    data,
    loading,
    error,
    refetch: fetchData,
  };
}

/**
 * Hook for mutations (POST, PUT, DELETE, PATCH)
 */
export interface UseMutationState<T, V = any> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  mutate: (variables: V) => Promise<T | null>;
  reset: () => void;
}

export function useMutation<T, V = any>(
  method: 'POST' | 'PUT' | 'DELETE' | 'PATCH',
  endpoint: string | ((variables: V) => string),
  options?: {
    onSuccess?: (data: T) => void;
    onError?: (error: ApiError) => void;
  }
): UseMutationState<T, V> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(
    async (variables: V): Promise<T | null> => {
      setLoading(true);
      setError(null);

      try {
        const url = typeof endpoint === 'function' ? endpoint(variables) : endpoint;
        let result: T;

        switch (method) {
          case 'POST':
            result = await apiClient.post<T>(url, variables);
            break;
          case 'PUT':
            result = await apiClient.put<T>(url, variables);
            break;
          case 'DELETE':
            result = await apiClient.delete<T>(url);
            break;
          case 'PATCH':
            result = await apiClient.patch<T>(url, variables);
            break;
          default:
            throw new Error(`Unsupported method: ${method}`);
        }

        setData(result);
        options?.onSuccess?.(result);
        return result;
      } catch (err) {
        const apiError = err as ApiError;
        setError(apiError);
        options?.onError?.(apiError);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [method, endpoint, options]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    data,
    loading,
    error,
    mutate,
    reset,
  };
}

/**
 * Hook for paginated data
 */
export interface UsePaginatedApiState<T> extends UseApiState<T> {
  page: number;
  pageSize: number;
  totalPages: number;
  totalItems: number;
  nextPage: () => void;
  prevPage: () => void;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
}

export function usePaginatedApi<T>(
  endpoint: string,
  initialPageSize: number = 20
): UsePaginatedApiState<T> {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [totalPages, setTotalPages] = useState(0);
  const [totalItems, setTotalItems] = useState(0);

  const paginatedEndpoint = `${endpoint}?page=${page}&limit=${pageSize}`;

  const { data, loading, error, refetch } = useApi<any>(paginatedEndpoint, {
    dependencies: [page, pageSize],
  });

  useEffect(() => {
    if (data) {
      setTotalPages(data.total_pages || Math.ceil((data.total || 0) / pageSize));
      setTotalItems(data.total || 0);
    }
  }, [data, pageSize]);

  const nextPage = useCallback(() => {
    if (page < totalPages) {
      setPage(p => p + 1);
    }
  }, [page, totalPages]);

  const prevPage = useCallback(() => {
    if (page > 1) {
      setPage(p => p - 1);
    }
  }, [page]);

  return {
    data,
    loading,
    error,
    refetch,
    page,
    pageSize,
    totalPages,
    totalItems,
    nextPage,
    prevPage,
    setPage,
    setPageSize,
  };
}

