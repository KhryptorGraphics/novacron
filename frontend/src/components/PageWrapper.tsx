'use client';

import React, { ReactNode, Suspense } from 'react';
import ErrorBoundary from './ErrorBoundary';

interface PageWrapperProps {
  children: ReactNode;
  pageName?: string;
  showLoader?: boolean;
}

const LoadingSpinner = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
  </div>
);

const PageErrorFallback = ({ pageName }: { pageName?: string }) => (
  <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
    <div className="text-center">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
        Error Loading {pageName || 'Page'}
      </h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        There was a problem loading this page. Please try refreshing.
      </p>
      <button
        onClick={() => window.location.reload()}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Refresh Page
      </button>
    </div>
  </div>
);

export default function PageWrapper({ 
  children, 
  pageName,
  showLoader = true 
}: PageWrapperProps) {
  return (
    <ErrorBoundary
      fallback={<PageErrorFallback pageName={pageName} />}
      onError={(error, errorInfo) => {
        // Log page-specific errors
        console.error(`Error in ${pageName || 'page'}:`, error, errorInfo);
      }}
    >
      <Suspense fallback={showLoader ? <LoadingSpinner /> : null}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
}

// Export a higher-order component for easy page wrapping
export function withPageWrapper<P extends object>(
  Component: React.ComponentType<P>,
  pageName?: string
) {
  return function WrappedComponent(props: P) {
    return (
      <PageWrapper pageName={pageName}>
        <Component {...props} />
      </PageWrapper>
    );
  };
}