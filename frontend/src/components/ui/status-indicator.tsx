"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Info, 
  Clock, 
  Loader2,
  Circle
} from "lucide-react";

const statusIndicatorVariants = cva(
  "inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-200",
  {
    variants: {
      variant: {
        success: "nova-status-success",
        warning: "nova-status-warning", 
        error: "nova-status-error",
        info: "nova-status-info",
        neutral: "bg-muted text-muted-foreground border border-border",
        pending: "bg-warning/10 text-warning border border-warning/20 animate-pulse",
        loading: "bg-muted text-muted-foreground border border-border",
      },
      size: {
        sm: "px-2 py-0.5 text-xs",
        md: "px-2.5 py-1 text-xs", 
        lg: "px-3 py-1.5 text-sm",
      },
    },
    defaultVariants: {
      variant: "neutral",
      size: "md",
    },
  }
);

const iconMap = {
  success: CheckCircle2,
  warning: AlertTriangle,
  error: XCircle,
  info: Info,
  neutral: Circle,
  pending: Clock,
  loading: Loader2,
};

export interface StatusIndicatorProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof statusIndicatorVariants> {
  showIcon?: boolean;
  icon?: React.ComponentType<{ className?: string }>;
}

const StatusIndicator = React.forwardRef<HTMLDivElement, StatusIndicatorProps>(
  ({ className, variant, size, showIcon = true, icon, children, ...props }, ref) => {
    const IconComponent = icon || (variant ? iconMap[variant] : iconMap.neutral);
    const isLoading = variant === "loading";

    return (
      <div
        className={cn(statusIndicatorVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      >
        {showIcon && IconComponent && (
          <IconComponent 
            className={cn(
              "h-3 w-3 flex-shrink-0",
              isLoading && "animate-spin",
              size === "sm" && "h-2.5 w-2.5",
              size === "lg" && "h-3.5 w-3.5"
            )} 
          />
        )}
        {children}
      </div>
    );
  }
);
StatusIndicator.displayName = "StatusIndicator";

// VM Status specific component
export interface VMStatusProps extends Omit<StatusIndicatorProps, "variant"> {
  status: "running" | "stopped" | "paused" | "migrating" | "error" | "unknown";
}

const vmStatusMap: Record<VMStatusProps["status"], StatusIndicatorProps["variant"]> = {
  running: "success",
  stopped: "neutral", 
  paused: "warning",
  migrating: "pending",
  error: "error",
  unknown: "neutral",
};

export const VMStatus = React.forwardRef<HTMLDivElement, VMStatusProps>(
  ({ status, children, ...props }, ref) => {
    const statusText = children || status.charAt(0).toUpperCase() + status.slice(1);
    
    return (
      <StatusIndicator
        variant={vmStatusMap[status]}
        ref={ref}
        {...props}
      >
        {statusText}
      </StatusIndicator>
    );
  }
);
VMStatus.displayName = "VMStatus";

// Migration Status component
export interface MigrationStatusProps extends Omit<StatusIndicatorProps, "variant"> {
  status: "preparing" | "copying" | "syncing" | "completed" | "failed" | "cancelled";
  progress?: number;
}

const migrationStatusMap: Record<MigrationStatusProps["status"], StatusIndicatorProps["variant"]> = {
  preparing: "pending",
  copying: "loading",
  syncing: "loading", 
  completed: "success",
  failed: "error",
  cancelled: "neutral",
};

export const MigrationStatus = React.forwardRef<HTMLDivElement, MigrationStatusProps>(
  ({ status, progress, children, ...props }, ref) => {
    const statusText = children || (
      progress !== undefined && (status === "copying" || status === "syncing")
        ? `${status.charAt(0).toUpperCase() + status.slice(1)} (${progress}%)`
        : status.charAt(0).toUpperCase() + status.slice(1)
    );
    
    return (
      <StatusIndicator
        variant={migrationStatusMap[status]}
        ref={ref}
        {...props}
      >
        {statusText}
      </StatusIndicator>
    );
  }
);
MigrationStatus.displayName = "MigrationStatus";

// Health Check Status component
export interface HealthStatusProps extends Omit<StatusIndicatorProps, "variant"> {
  status: "healthy" | "degraded" | "unhealthy" | "maintenance" | "unknown";
}

const healthStatusMap: Record<HealthStatusProps["status"], StatusIndicatorProps["variant"]> = {
  healthy: "success",
  degraded: "warning",
  unhealthy: "error", 
  maintenance: "info",
  unknown: "neutral",
};

export const HealthStatus = React.forwardRef<HTMLDivElement, HealthStatusProps>(
  ({ status, children, ...props }, ref) => {
    const statusText = children || status.charAt(0).toUpperCase() + status.slice(1);
    
    return (
      <StatusIndicator
        variant={healthStatusMap[status]}
        ref={ref}
        {...props}
      >
        {statusText}
      </StatusIndicator>
    );
  }
);
HealthStatus.displayName = "HealthStatus";

export { StatusIndicator, statusIndicatorVariants };