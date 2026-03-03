"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useWorkflows } from "@/hooks/useAPI";

export function WorkflowExecutionChart() {
  const { workflows } = useWorkflows();
  
  // Calculate execution frequency by day for the last 7 days
  const executionData = Array.from({ length: 7 }, (_, i) => {
    const day = new Date();
    day.setDate(day.getDate() - (6 - i));
    return {
      day: day.toLocaleDateString('en-US', { weekday: 'short' }),
      count: Math.floor(Math.random() * 5), // Placeholder data
    };
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Workflow Executions (7d)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 flex items-end justify-between pt-4">
          {executionData.map((data, index) => (
            <div key={index} className="flex flex-col items-center flex-1 px-1">
              <div 
                className="w-full bg-green-500 rounded-t hover:bg-green-600 transition-colors"
                style={{ height: `${data.count * 20}%` }}
              />
              <span className="text-xs text-gray-500 mt-1">{data.day}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}