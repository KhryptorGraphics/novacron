'use client';

import * as React from 'react';
import { RefreshCw, Loader2 } from 'lucide-react';

// Dashboard skeleton loader
export function DashboardSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-8 bg-gray-200 rounded w-1/4" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-32 bg-gray-200 rounded" />
        ))}
      </div>
      <div className="h-64 bg-gray-200 rounded" />
    </div>
  );
}

// Refresh indicator
export function RefreshIndicator({
  isRefreshing = false,
  onRefresh,
  lastUpdated,
}: {
  isRefreshing?: boolean;
  size?: number;
  onRefresh?: (() => void) | (() => Promise<void>);
  lastUpdated?: Date;
}) {
  return (
    <button
      type="button"
      onClick={() => { void onRefresh?.(); }}
      title={lastUpdated ? `Last updated: ${lastUpdated.toLocaleTimeString()}` : undefined}
      className="inline-flex items-center text-muted-foreground hover:text-foreground"
    >
      <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
    </button>
  );
}

// Loading states component. `state` accepts either the string status or a
// {isLoading,...} object; while loading it renders `loadingComponent` (or a
// spinner), otherwise it renders `children`.
type LoadingStateObject = {
  isLoading?: boolean;
  isError?: boolean;
  isSuccess?: boolean;
  message?: string;
};

export function LoadingStates({
  state = 'loading',
  size = 'default',
  loadingComponent,
  children,
}: {
  state?: 'loading' | 'success' | 'error' | 'idle' | LoadingStateObject;
  size?: 'small' | 'default' | 'large';
  loadingComponent?: React.ReactNode;
  children?: React.ReactNode;
}) {
  const sizeClass =
    size === 'small' ? 'h-4 w-4' : size === 'large' ? 'h-8 w-8' : 'h-6 w-6';

  const isLoading =
    typeof state === 'object' ? state.isLoading === true : state === 'loading';

  if (isLoading) {
    return <>{loadingComponent ?? <Loader2 className={`${sizeClass} animate-spin`} />}</>;
  }

  if (typeof state === 'object') {
    if (state.isError) {
      return <div className={`${sizeClass} text-red-500 flex items-center justify-center`}>✗</div>;
    }
    return <>{children}</>;
  }

  switch (state) {
    case 'success':
      return <div className={`${sizeClass} text-green-500 flex items-center justify-center`}>✓</div>;
    case 'error':
      return <div className={`${sizeClass} text-red-500 flex items-center justify-center`}>✗</div>;
    default:
      return <>{children}</>;
  }
}