"use client";

import React from "react";
import { AlertTriangle, RefreshCw, Home, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  showDetails: boolean;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<ErrorFallbackProps>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  resetKeys?: Array<string | number>;
  resetOnPropsChange?: boolean;
}

interface ErrorFallbackProps {
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  resetError: () => void;
  showDetails: boolean;
  toggleDetails: () => void;
}

// Default Error Fallback Component
function DefaultErrorFallback({ 
  error, 
  errorInfo, 
  resetError,
  showDetails,
  toggleDetails
}: ErrorFallbackProps) {
  const router = useRouter();
  
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-2xl w-full">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 sm:p-8">
          {/* Error Icon and Title */}
          <div className="flex flex-col items-center text-center mb-6">
            <div className="h-20 w-20 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center mb-4">
              <AlertTriangle className="h-10 w-10 text-red-600 dark:text-red-400" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              Oops! Something went wrong
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              We encountered an unexpected error. Don't worry, your data is safe.
            </p>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm font-mono text-red-800 dark:text-red-300">
                {error.message || "An unknown error occurred"}
              </p>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <Button
              onClick={resetError}
              className="flex-1"
              variant="default"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Try Again
            </Button>
            <Button
              onClick={() => router.push("/")}
              className="flex-1"
              variant="outline"
            >
              <Home className="mr-2 h-4 w-4" />
              Go to Home
            </Button>
          </div>
          
          {/* Error Details Toggle */}
          <button
            onClick={toggleDetails}
            className="w-full text-left text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span>Technical Details</span>
              {showDetails ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </div>
          </button>
          
          {/* Error Stack Trace */}
          {showDetails && errorInfo && (
            <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Stack Trace:
              </h3>
              <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-auto max-h-48">
                {errorInfo.componentStack}
              </pre>
              {error?.stack && (
                <>
                  <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mt-4 mb-2">
                    Error Stack:
                  </h3>
                  <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-auto max-h-48">
                    {error.stack}
                  </pre>
                </>
              )}
            </div>
          )}
        </div>
        
        {/* Help Text */}
        <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-6">
          If this problem persists, please contact support with the error details above.
        </p>
      </div>
    </div>
  );
}

// Main Error Boundary Class Component
export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false
    };
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error
    };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to console in development
    if (process.env.NODE_ENV === "development") {
      console.error("Error Boundary caught an error:", error, errorInfo);
    }
    
    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);
    
    // Update state with error info
    this.setState({
      errorInfo
    });
    
    // In production, you might want to log to an error reporting service
    // logErrorToService(error, errorInfo);
  }
  
  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    const { resetKeys, resetOnPropsChange } = this.props;
    const { hasError } = this.state;
    
    // Reset error boundary when resetKeys change
    if (hasError && resetKeys) {
      const hasResetKeyChanged = resetKeys.some(
        (key, index) => key !== prevProps.resetKeys?.[index]
      );
      
      if (hasResetKeyChanged) {
        this.resetError();
      }
    }
    
    // Reset on any props change if specified
    if (hasError && resetOnPropsChange && prevProps !== this.props) {
      this.resetError();
    }
  }
  
  resetError = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false
    });
  };
  
  toggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails
    }));
  };
  
  render() {
    const { hasError, error, errorInfo, showDetails } = this.state;
    const { children, fallback: FallbackComponent = DefaultErrorFallback } = this.props;
    
    if (hasError) {
      return (
        <FallbackComponent
          error={error}
          errorInfo={errorInfo}
          resetError={this.resetError}
          showDetails={showDetails}
          toggleDetails={this.toggleDetails}
        />
      );
    }
    
    return children;
  }
}

// Async Error Boundary for Suspense
interface AsyncErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function AsyncErrorBoundary({ 
  children, 
  fallback 
}: AsyncErrorBoundaryProps) {
  return (
    <ErrorBoundary
      fallback={({ resetError }) => (
        <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
          <AlertTriangle className="h-12 w-12 text-yellow-500 mb-4" />
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Failed to load content
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 text-center">
            We couldn't load this section. Please check your connection and try again.
          </p>
          <Button onClick={resetError} size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </div>
      )}
    >
      <React.Suspense fallback={fallback}>
        {children}
      </React.Suspense>
    </ErrorBoundary>
  );
}

// Network Error Component
export function NetworkError({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
      <div className="h-20 w-20 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
        <svg
          className="h-10 w-10 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0"
          />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        No Internet Connection
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 text-center max-w-sm">
        Please check your internet connection and try again.
      </p>
      {onRetry && (
        <Button onClick={onRetry} size="sm">
          <RefreshCw className="mr-2 h-4 w-4" />
          Try Again
        </Button>
      )}
    </div>
  );
}

// 404 Error Component
export function NotFoundError() {
  const router = useRouter();
  
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8">
      <div className="text-center">
        <h1 className="text-9xl font-bold text-gray-200 dark:text-gray-800">
          404
        </h1>
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mt-4 mb-2">
          Page Not Found
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="flex gap-4 justify-center">
          <Button onClick={() => router.back()} variant="outline">
            Go Back
          </Button>
          <Button onClick={() => router.push("/")}>
            <Home className="mr-2 h-4 w-4" />
            Go Home
          </Button>
        </div>
      </div>
    </div>
  );
}

// Permission Error Component
export function PermissionError({ 
  message = "You don't have permission to access this resource" 
}: { 
  message?: string 
}) {
  const router = useRouter();
  
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
      <div className="h-20 w-20 rounded-full bg-yellow-100 dark:bg-yellow-900/20 flex items-center justify-center mb-4">
        <svg
          className="h-10 w-10 text-yellow-600 dark:text-yellow-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
          />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Access Denied
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 text-center max-w-sm">
        {message}
      </p>
      <Button onClick={() => router.push("/")} size="sm">
        <Home className="mr-2 h-4 w-4" />
        Go to Home
      </Button>
    </div>
  );
}