"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export function ExecutionStatistics() {
  // Mock statistics data
  const stats = {
    totalExecutions: 1247,
    successRate: 98.2,
    avgDuration: 4200, // ms
    failedExecutions: 22,
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution Statistics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Total Executions</span>
              <span className="font-medium">{stats.totalExecutions}</span>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Success Rate</span>
              <span className="font-medium">{stats.successRate}%</span>
            </div>
            <Progress value={stats.successRate} className="h-2" />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Average Duration</span>
              <span className="font-medium">{(stats.avgDuration / 1000).toFixed(2)}s</span>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Failed Executions</span>
              <span className="font-medium text-red-500">{stats.failedExecutions}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}