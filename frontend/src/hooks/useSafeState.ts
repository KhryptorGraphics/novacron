'use client';

import { useState, useCallback, Dispatch, SetStateAction } from 'react';

/**
 * A safe version of useState that prevents null/undefined issues
 * and provides default values for common operations
 */
export function useSafeState<T>(
  initialState: T | (() => T),
  defaultValue?: T
): [T, Dispatch<SetStateAction<T>>] {
  const [state, setState] = useState<T>(() => {
    try {
      const initial = typeof initialState === 'function' 
        ? (initialState as () => T)() 
        : initialState;
      
      // Ensure we never have null/undefined for objects and arrays
      if (initial === null || initial === undefined) {
        if (defaultValue !== undefined) {
          return defaultValue;
        }
        // Provide sensible defaults based on type inference
        if (Array.isArray(initial)) {
          return [] as unknown as T;
        }
        if (typeof initial === 'object') {
          return {} as T;
        }
      }
      
      return initial;
    } catch (error) {
      console.error('Error initializing state:', error);
      return defaultValue ?? ([] as unknown as T);
    }
  });

  const setSafeState = useCallback<Dispatch<SetStateAction<T>>>((newState) => {
    setState((prevState) => {
      try {
        const nextState = typeof newState === 'function'
          ? (newState as (prevState: T) => T)(prevState)
          : newState;
        
        // Prevent setting to null/undefined
        if (nextState === null || nextState === undefined) {
          console.warn('Attempted to set state to null/undefined, using default value');
          return defaultValue ?? prevState;
        }
        
        return nextState;
      } catch (error) {
        console.error('Error updating state:', error);
        return prevState;
      }
    });
  }, [defaultValue]);

  return [state, setSafeState];
}

/**
 * Safe array operations hook
 */
export function useSafeArray<T>(initialArray: T[] = []) {
  const [array, setArray] = useSafeState<T[]>(initialArray, []);

  const push = useCallback((item: T) => {
    setArray((prev) => [...(prev || []), item]);
  }, [setArray]);

  const remove = useCallback((index: number) => {
    setArray((prev) => {
      if (!prev || !Array.isArray(prev)) return [];
      return prev.filter((_, i) => i !== index);
    });
  }, [setArray]);

  const update = useCallback((index: number, item: T) => {
    setArray((prev) => {
      if (!prev || !Array.isArray(prev)) return [];
      const newArray = [...prev];
      if (index >= 0 && index < newArray.length) {
        newArray[index] = item;
      }
      return newArray;
    });
  }, [setArray]);

  const clear = useCallback(() => {
    setArray([]);
  }, [setArray]);

  return {
    array,
    setArray,
    push,
    remove,
    update,
    clear,
    length: array?.length || 0,
    isEmpty: !array || array.length === 0,
  };
}

/**
 * Safe object operations hook
 */
export function useSafeObject<T extends Record<string, any>>(initialObject: T = {} as T) {
  const [object, setObject] = useSafeState<T>(initialObject, {} as T);

  const updateField = useCallback(<K extends keyof T>(key: K, value: T[K]) => {
    setObject((prev) => ({
      ...(prev || {}),
      [key]: value,
    }));
  }, [setObject]);

  const updateMultiple = useCallback((updates: Partial<T>) => {
    setObject((prev) => ({
      ...(prev || {}),
      ...updates,
    }));
  }, [setObject]);

  const removeField = useCallback(<K extends keyof T>(key: K) => {
    setObject((prev) => {
      if (!prev) return {} as T;
      const { [key]: _, ...rest } = prev;
      return rest as T;
    });
  }, [setObject]);

  const clear = useCallback(() => {
    setObject({} as T);
  }, [setObject]);

  const hasField = useCallback(<K extends keyof T>(key: K): boolean => {
    return object && key in object;
  }, [object]);

  return {
    object,
    setObject,
    updateField,
    updateMultiple,
    removeField,
    clear,
    hasField,
    keys: Object.keys(object || {}),
    values: Object.values(object || {}),
    entries: Object.entries(object || {}),
  };
}

/**
 * Safe async state hook with loading and error handling
 */
export function useSafeAsyncState<T>(initialState?: T) {
  const [data, setData] = useSafeState<T | undefined>(initialState);
  const [loading, setLoading] = useSafeState(false);
  const [error, setError] = useSafeState<Error | null>(null);

  const execute = useCallback(async (asyncFunction: () => Promise<T>) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await asyncFunction();
      setData(result);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [setData, setError, setLoading]);

  const reset = useCallback(() => {
    setData(initialState);
    setLoading(false);
    setError(null);
  }, [initialState, setData, setError, setLoading]);

  return {
    data,
    loading,
    error,
    execute,
    reset,
    isSuccess: !loading && !error && data !== undefined,
    isError: !loading && error !== null,
    isLoading: loading,
    isIdle: !loading && !error && data === undefined,
  };
}