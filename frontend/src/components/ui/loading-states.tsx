"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Loader2, RefreshCw, AlertCircle } from "lucide-react";

// Skeleton component for loading states
interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "text" | "circular" | "rectangular";
  width?: string | number;
  height?: string | number;
  lines?: number;
}

export function Skeleton({
  className,
  variant = "default",
  width,
  height,
  lines = 1,
  ...props
}: SkeletonProps) {
  const baseClasses = "animate-pulse bg-muted rounded";
  
  const variants = {
    default: "h-4",
    text: "h-4",
    circular: "rounded-full",
    rectangular: "rounded-lg",
  };

  if (variant === "text" && lines > 1) {
    return (
      <div className={cn("space-y-2", className)} {...props}>
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className={cn(
              baseClasses,
              variants.text,
              index === lines - 1 ? "w-3/4" : "w-full"
            )}
            style={{
              width: index === lines - 1 ? "75%" : width,
              height,
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={cn(baseClasses, variants[variant], className)}
      style={{ width, height }}
      {...props}
    />
  );
}

// Loading spinner with different sizes and variants
interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  variant?: "default" | "dots" | "pulse" | "bounce";
  className?: string;
  label?: string;
}

export function LoadingSpinner({
  size = "md",
  variant = "default",
  className,
  label,
}: LoadingSpinnerProps) {
  const sizes = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
  };

  const spinnerVariants = {
    default: (
      <Loader2 className={cn("animate-spin", sizes[size], className)} />
    ),
    dots: (
      <div className={cn("flex gap-1", className)}>
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={cn(
              "rounded-full bg-current",
              size === "sm" ? "w-1.5 h-1.5" : size === "lg" ? "w-2.5 h-2.5" : "w-2 h-2"
            )}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 1.4,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </div>
    ),
    pulse: (
      <motion.div
        className={cn(
          "rounded-full bg-current",
          sizes[size],
          className
        )}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.7, 1, 0.7],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
        }}
      />
    ),
    bounce: (
      <motion.div
        className={cn("flex gap-1", className)}
      >
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={cn(
              "rounded-full bg-current",
              size === "sm" ? "w-1.5 h-1.5" : size === "lg" ? "w-2.5 h-2.5" : "w-2 h-2"
            )}
            animate={{
              y: ["0%", "-50%", "0%"],
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: i * 0.1,
            }}
          />
        ))}
      </motion.div>
    ),
  };

  return (
    <div className="flex flex-col items-center gap-2">
      {spinnerVariants[variant]}
      {label && (
        <span className="text-sm text-muted-foreground animate-pulse">
          {label}
        </span>
      )}
    </div>
  );
}

// Loading overlay for entire components/pages
interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  loadingText?: string;
  variant?: "default" | "blur" | "skeleton";
  className?: string;
}

export function LoadingOverlay({
  isLoading,
  children,
  loadingText,
  variant = "default",
  className,
}: LoadingOverlayProps) {
  return (
    <div className={cn("relative", className)}>
      {children}
      
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className={cn(
              "absolute inset-0 z-10 flex items-center justify-center",
              variant === "blur" && "backdrop-blur-sm",
              variant === "default" && "bg-background/80"
            )}
          >
            <div className="flex flex-col items-center gap-3 p-6 bg-card rounded-lg shadow-lg border">
              <LoadingSpinner size="lg" />
              {loadingText && (
                <p className="text-sm text-muted-foreground text-center">
                  {loadingText}
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Skeleton components for common patterns
export function CardSkeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="nova-card p-6 space-y-4">
      <div className="flex items-center gap-3">
        <Skeleton variant="circular" width={40} height={40} />
        <div className="space-y-2 flex-1">
          <Skeleton width="60%" height={16} />
          <Skeleton width="40%" height={12} />
        </div>
      </div>
      <Skeleton variant="text" lines={lines} />
    </div>
  );
}

export function TableSkeleton({ 
  rows = 5, 
  columns = 4 
}: { 
  rows?: number; 
  columns?: number; 
}) {
  return (
    <div className="nova-card">
      <div className="p-4 border-b">
        <Skeleton width="30%" height={20} />
      </div>
      <div className="divide-y">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="flex gap-4 p-4">
            {Array.from({ length: columns }).map((_, colIndex) => (
              <div key={colIndex} className="flex-1">
                <Skeleton height={16} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <Skeleton width="30%" height={32} />
        <Skeleton width="60%" height={16} />
      </div>
      
      {/* Metrics cards */}
      <div className="nova-grid-dashboard">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="nova-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <Skeleton width="40%" height={16} />
              <Skeleton variant="circular" width={24} height={24} />
            </div>
            <Skeleton width="60%" height={28} />
            <Skeleton width="80%" height={12} />
          </div>
        ))}
      </div>
      
      {/* Charts */}
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
        <div className="nova-card p-6 space-y-4">
          <Skeleton width="40%" height={20} />
          <Skeleton variant="rectangular" height={200} />
        </div>
        <div className="nova-card p-6 space-y-4">
          <Skeleton width="40%" height={20} />
          <Skeleton variant="rectangular" height={200} />
        </div>
      </div>
    </div>
  );
}

// Progressive disclosure component
interface ProgressiveDisclosureProps {
  isLoading: boolean;
  children: React.ReactNode;
  skeleton: React.ReactNode;
  delay?: number;
  className?: string;
}

export function ProgressiveDisclosure({
  isLoading,
  children,
  skeleton,
  delay = 0,
  className,
}: ProgressiveDisclosureProps) {
  return (
    <div className={className}>
      <AnimatePresence mode="wait">
        {isLoading ? (
          <motion.div
            key="skeleton"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ delay, duration: 0.2 }}
          >
            {skeleton}
          </motion.div>
        ) : (
          <motion.div
            key="content"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: delay + 0.1, duration: 0.3 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Error state component
interface ErrorStateProps {
  error: string | Error;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({ error, onRetry, className }: ErrorStateProps) {
  const errorMessage = typeof error === "string" ? error : error.message;
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "flex flex-col items-center justify-center gap-4 p-8 text-center",
        className
      )}
    >
      <div className="flex items-center justify-center w-12 h-12 rounded-full bg-destructive/10 text-destructive">
        <AlertCircle className="w-6 h-6" />
      </div>
      
      <div className="space-y-2">
        <h3 className="text-lg font-semibold">Something went wrong</h3>
        <p className="text-sm text-muted-foreground max-w-md">
          {errorMessage}
        </p>
      </div>
      
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Try Again
        </button>
      )}
    </motion.div>
  );
}

// Loading button state
interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export const LoadingButton = React.forwardRef<HTMLButtonElement, LoadingButtonProps>(
  ({ isLoading = false, loadingText, children, disabled, className, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex items-center justify-center gap-2 transition-all",
          isLoading && "cursor-not-allowed",
          className
        )}
        {...props}
      >
        <AnimatePresence mode="wait" initial={false}>
          {isLoading ? (
            <motion.span
              key="loading"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="inline-flex items-center gap-2"
            >
              <Loader2 className="w-4 h-4 animate-spin" />
              {loadingText || "Loading..."}
            </motion.span>
          ) : (
            <motion.span
              key="content"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
            >
              {children}
            </motion.span>
          )}
        </AnimatePresence>
      </button>
    );
  }
);
LoadingButton.displayName = "LoadingButton";