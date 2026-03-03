"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricsCardProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number;
  changeLabel?: string;
  status?: "success" | "warning" | "error" | "neutral";
  icon?: React.ReactNode;
  className?: string;
}

export function MetricsCard({
  title,
  value,
  unit,
  change,
  changeLabel,
  status = "neutral",
  icon,
  className
}: MetricsCardProps) {
  const getTrendIcon = () => {
    if (!change) return null;
    if (change > 0) return <TrendingUp className="h-4 w-4" />;
    if (change < 0) return <TrendingDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };
  
  const getTrendColor = () => {
    if (!change) return "text-gray-500";
    if (status === "success") return change > 0 ? "text-green-600" : "text-red-600";
    if (status === "error") return change > 0 ? "text-red-600" : "text-green-600";
    return change > 0 ? "text-blue-600" : "text-orange-600";
  };
  
  const getStatusColor = () => {
    const colors = {
      success: "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400",
      warning: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400",
      error: "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400",
      neutral: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    };
    return colors[status];
  };
  
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
          {title}
        </CardTitle>
        {icon && (
          <div className={cn("p-2 rounded-lg", getStatusColor())}>
            {icon}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold">{value}</span>
          {unit && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {unit}
            </span>
          )}
        </div>
        
        {(change !== undefined || changeLabel) && (
          <div className="flex items-center gap-2 mt-2">
            {change !== undefined && (
              <div className={cn("flex items-center gap-1", getTrendColor())}>
                {getTrendIcon()}
                <span className="text-sm font-medium">
                  {Math.abs(change)}%
                </span>
              </div>
            )}
            {changeLabel && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {changeLabel}
              </span>
            )}
          </div>
        )}
        
        {/* Visual indicator bar */}
        {status !== "neutral" && (
          <div className={cn(
            "absolute bottom-0 left-0 right-0 h-1",
            status === "success" && "bg-green-500",
            status === "warning" && "bg-yellow-500",
            status === "error" && "bg-red-500"
          )} />
        )}
      </CardContent>
    </Card>
  );
}