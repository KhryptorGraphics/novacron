"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { JobList } from "@/components/dashboard/job-list";
import { WorkflowList } from "@/components/dashboard/workflow-list";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useWebSocket } from "@/hooks/useAPI";

export function SchedulingDashboard() {
  const { connected, lastMessage } = useWebSocket();
  const [activeTab, setActiveTab] = useState("jobs");

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">Scheduling Dashboard</h2>
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      {lastMessage && (
        <Card>
          <CardHeader>
            <CardTitle>Real-time Updates</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-gray-100 p-4 rounded-md text-sm overflow-x-auto">
              {JSON.stringify(lastMessage, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
      
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="jobs">Jobs</TabsTrigger>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
        </TabsList>
        <TabsContent value="jobs" className="space-y-4">
          <JobList />
        </TabsContent>
        <TabsContent value="workflows" className="space-y-4">
          <WorkflowList />
        </TabsContent>
      </Tabs>
    </div>
  );
}