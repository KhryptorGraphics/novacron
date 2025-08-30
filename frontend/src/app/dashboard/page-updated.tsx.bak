"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  WorkflowVisualization 
} from "@/components/dashboard/workflow-visualization";
import { 
  WorkflowExecutionMonitor 
} from "@/components/dashboard/workflow-execution-monitor";
import { JobList } from "@/components/dashboard/job-list";
import { WorkflowList } from "@/components/dashboard/workflow-list";
import { JobExecutionChart } from "@/components/dashboard/job-execution-chart";
import { WorkflowExecutionChart } from "@/components/dashboard/workflow-execution-chart";
import { ExecutionTimeline } from "@/components/dashboard/execution-timeline";
import { ExecutionStatistics } from "@/components/dashboard/execution-statistics";
import { useWebSocket, useJobs, useWorkflows } from "@/hooks/useAPI";
import { JobExecution } from "@/lib/api";

export default function ComprehensiveDashboard() {
  const { connected, lastMessage } = useWebSocket();
  const { jobs, refetch: refetchJobs, loading: jobsLoading } = useJobs();
  const { workflows, refetch: refetchWorkflows, loading: workflowsLoading } = useWorkflows();
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const [recentExecutions, setRecentExecutions] = useState<JobExecution[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds

  // Handle real-time updates from WebSocket
  useEffect(() => {
    if (lastMessage) {
      // Refresh data when we get real-time updates
      refetchJobs();
      refetchWorkflows();
      
      // Add to recent executions if it's an execution update
      if (lastMessage.type === 'job_execution' || lastMessage.type === 'workflow_execution') {
        // In a real implementation, we would update the recent executions list
        console.log('Real-time execution update:', lastMessage);
      }
    }
  }, [lastMessage, refetchJobs, refetchWorkflows]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      refetchJobs();
      refetchWorkflows();
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, refetchJobs, refetchWorkflows]);

  // Calculate execution statistics
  const executionStats = {
    totalJobs: jobs?.length || 0,
    enabledJobs: jobs?.filter(job => job.enabled).length || 0,
    totalWorkflows: workflows?.length || 0,
    enabledWorkflows: workflows?.filter(workflow => workflow.enabled).length || 0,
    recentFailures: 0, // This would be populated from execution data
    averageDuration: 0 // This would be calculated from execution data
  };

  // Show dashboard tour for new users
  useEffect(() => {
    const hasSeenTour = localStorage.getItem('dashboardTourCompleted');
    if (!hasSeenTour) {
      // In a real implementation, we would show a tour component
      console.log('Showing dashboard tour');
    }
  }, []);
    <div className="flex flex-col gap-6 p-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">NovaCron Dashboard</h1>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`h-3 w-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => { refetchJobs(); refetchWorkflows(); }}
              className="text-sm border rounded p-1 hover:bg-gray-100"
              disabled={jobsLoading || workflowsLoading}
            >
              {jobsLoading || workflowsLoading ? 'Refreshing...' : 'Refresh'}
            </button>
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="mr-1"
              />
              Auto-refresh
            </label>
            <select 
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="text-sm border rounded p-1"
              disabled={!autoRefresh}
            >
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={60}>1m</option>
              <option value={300}>5m</option>
            </select>
          </div>
          <div className="relative">
            <input
              type="text"
              placeholder="Search executions..."
              className="text-sm border rounded p-1 pl-8"
            />
            <svg
              className="w-4 h-4 absolute left-2 top-1.5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <select className="text-sm border rounded p-1">
            <option>All Executions</option>
            <option>Running</option>
            <option>Completed</option>
            <option>Failed</option>
          </select>
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
      
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="jobs">Jobs</TabsTrigger>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
                <div className="h-4 w-4 text-muted-foreground">📈</div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{executionStats.totalJobs}</div>
                <p className="text-xs text-muted-foreground">{executionStats.enabledJobs} enabled</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Workflows</CardTitle>
                <div className="h-4 w-4 text-muted-foreground">🔄</div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{executionStats.totalWorkflows}</div>
                <p className="text-xs text-muted-foreground">{executionStats.enabledWorkflows} enabled</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Running Tasks</CardTitle>
                <div className="h-4 w-4 text-muted-foreground">⚡</div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">0</div>
                <p className="text-xs text-muted-foreground">0 active now</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <div className="h-4 w-4 text-muted-foreground">✅</div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">100%</div>
                <p className="text-xs text-muted-foreground">No recent failures</p>
              </CardContent>
            </Card>
          </div>
          
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span>API Server</span>
                    <Badge variant="default">Online</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Database</span>
                    <Badge variant="default">Online</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Redis</span>
                    <Badge variant="default">Online</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Message Queue</span>
                    <Badge variant="default">Online</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <ExecutionStatistics />
          </div>
          
          <JobExecutionChart />
          
          <WorkflowExecutionChart />
          
          <ExecutionTimeline />
        </TabsContent>
        
        <TabsContent value="jobs" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Jobs</h2>
            <div className="flex space-x-2">
              <select className="text-sm border rounded p-1">
                <option>All Status</option>
                <option>Enabled</option>
                <option>Disabled</option>
              </select>
              <select className="text-sm border rounded p-1">
                <option>All Types</option>
                <option>Cron</option>
                <option>Interval</option>
              </select>
            </div>
          </div>
          <JobList />
        </TabsContent>
        
        <TabsContent value="workflows" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Workflows</h2>
            <div className="flex space-x-2">
              <select className="text-sm border rounded p-1">
                <option>All Status</option>
                <option>Enabled</option>
                <option>Disabled</option>
              </select>
            </div>
          </div>
          <WorkflowList />
        </TabsContent>
        
        <TabsContent value="visualization" className="space-y-4">
          {selectedWorkflow ? (
            <WorkflowVisualization workflowId={selectedWorkflow} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Workflow Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Please select a workflow from the Workflows tab to visualize it.</p>
                <Button 
                  className="mt-4"
                  onClick={() => {
                    // In a real implementation, you would set the selected workflow
                    // For now, we'll just show a message
                    alert("Please select a workflow from the Workflows tab first");
                  }}
                >
                  Select Workflow
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="monitoring" className="space-y-4">
          {selectedExecution ? (
            <WorkflowExecutionMonitor executionId={selectedExecution} />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Execution Monitoring</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Please select an execution from the Workflows tab to monitor it.</p>
                <Button 
                  className="mt-4"
                  onClick={() => {
                    // In a real implementation, you would set the selected execution
                    // For now, we'll just show a message
                    alert("Please select an execution from the Workflows tab first");
                  }}
                >
                  Select Execution
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}