"use client";

import { RefreshCw, Loader2 } from "lucide-react";

// Dashboard skeleton loader
export function DashboardSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-8 bg-gray-200 rounded w-1/4"></div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-32 bg-gray-200 rounded"></div>
        ))}
      </div>
      <div className="h-64 bg-gray-200 rounded"></div>
    </div>
  );
}

// Refresh indicator
export function RefreshIndicator({ 
  isRefreshing = false,
  size = 16 
}: { 
  isRefreshing?: boolean;
  size?: number;
}) {
  return (
    <RefreshCw 
      className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} 
    />
  );
}

// Loading states component
export function LoadingStates({
  state = "loading",
  size = "default"
}: {
  state?: "loading" | "success" | "error" | "idle";
  size?: "small" | "default" | "large";
}) {
  const getSizeClass = () => {
    switch (size) {
      case "small": return "h-4 w-4";
      case "large": return "h-8 w-8";
      default: return "h-6 w-6";
    }
  };

  const sizeClass = getSizeClass();

  switch (state) {
    case "loading":
      return <Loader2 className={`${sizeClass} animate-spin`} />;
    case "success":
      return <div className={`${sizeClass} text-green-500 flex items-center justify-center`}>✓</div>;
    case "error":
      return <div className={`${sizeClass} text-red-500 flex items-center justify-center`}>✗</div>;
    default:
      return null;
  }
}