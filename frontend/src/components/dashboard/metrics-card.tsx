"use client";

import React from "react";

interface MetricsCardProps {
  title: string;
  value: string;
  icon: string;
  trend?: "up" | "down" | "stable";
}

export function MetricsCard({ title, value, icon, trend = "stable" }: MetricsCardProps) {
  // Get trend color and icon
  const getTrendDetails = () => {
    switch (trend) {
      case "up":
        return {
          color: "text-green-500",
          icon: "↑",
          text: "Increasing",
        };
      case "down":
        return {
          color: "text-red-500",
          icon: "↓",
          text: "Decreasing",
        };
      case "stable":
      default:
        return {
          color: "text-blue-500",
          icon: "→",
          text: "Stable",
        };
    }
  };

  const trendDetails = getTrendDetails();

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-medium text-gray-500 dark:text-gray-400">{title}</h3>
          <p className="text-3xl font-bold mt-1">{value}</p>
        </div>
        <div className="text-3xl bg-blue-100 dark:bg-blue-900 p-3 rounded-lg">
          {icon}
        </div>
      </div>
      <div className={`flex items-center mt-4 ${trendDetails.color}`}>
        <span className="mr-1">{trendDetails.icon}</span>
        <span className="text-sm">{trendDetails.text}</span>
      </div>
    </div>
  );
}
