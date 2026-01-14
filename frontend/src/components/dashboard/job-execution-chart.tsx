"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useJobs } from "@/hooks/useAPI";

export function JobExecutionChart() {
  const { jobs } = useJobs();
  
  // Calculate execution frequency by hour for the last 24 hours
  const executionData = Array.from({ length: 24 }, (_, i) => {
    const hour = new Date();
    hour.setHours(hour.getHours() - (23 - i));
    return {
      hour: hour.getHours(),
      count: Math.floor(Math.random() * 10), // Placeholder data
    };
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Job Executions (24h)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 flex items-end justify-between pt-4">
          {executionData.map((data, index) => (
            <div key={index} className="flex flex-col items-center flex-1 px-1">
              <div 
                className="w-full bg-blue-500 rounded-t hover:bg-blue-600 transition-colors"
                style={{ height: `${data.count * 10}%` }}
              />
              <span className="text-xs text-gray-500 mt-1">{data.hour}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}