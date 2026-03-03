"use client";

import { useState, useEffect } from "react";
import { useWorkflowExecution } from "@/hooks/useAPI";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { format } from "date-fns";

interface ExecutionNode {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  errorMessage?: string;
}

export function WorkflowExecutionMonitor({ executionId }: { executionId: string }) {
  const { execution, loading, error } = useWorkflowExecution(executionId);
  const [nodes, setNodes] = useState<ExecutionNode[]>([]);

  useEffect(() => {
    if (execution) {
      // Convert node executions to array format
      const nodeArray: ExecutionNode[] = Object.entries(execution.nodeExecutions).map(([id, nodeExec]) => ({
        id,
        name: `Node ${id}`, // In a real implementation, you'd get the actual node name
        status: nodeExec.status,
        startedAt: nodeExec.startedAt,
        completedAt: nodeExec.completedAt,
        durationMs: nodeExec.durationMs,
        errorMessage: nodeExec.errorMessage
      }));
      
      setNodes(nodeArray);
    }
  }, [execution]);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Execution Monitor</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Loading execution data...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Execution Monitor</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500">Error loading execution: {error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!execution) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Execution Monitor</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Execution not found</p>
        </CardContent>
      </Card>
    );
  }

  // Calculate overall progress
  const totalNodes = nodes.length;
  const completedNodes = nodes.filter(node => node.status === 'completed').length;
  const progress = totalNodes > 0 ? (completedNodes / totalNodes) * 100 : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution Monitor</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Overall execution status */}
          <div className="border rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium">Overall Status</h3>
              <Badge 
                variant={execution.status === 'completed' ? 'default' : execution.status === 'failed' ? 'destructive' : 'secondary'}
              >
                {execution.status}
              </Badge>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} />
            </div>
            
            <div className="grid grid-cols-2 gap-4 mt-4">
              {execution.startedAt && (
                <div>
                  <p className="text-sm text-gray-500">Started</p>
                  <p>{format(new Date(execution.startedAt), "MMM dd, yyyy HH:mm:ss")}</p>
                </div>
              )}
              {execution.completedAt && (
                <div>
                  <p className="text-sm text-gray-500">Completed</p>
                  <p>{format(new Date(execution.completedAt), "MMM dd, yyyy HH:mm:ss")}</p>
                </div>
              )}
              {execution.durationMs && (
                <div>
                  <p className="text-sm text-gray-500">Duration</p>
                  <p>{execution.durationMs}ms</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Node executions */}
          <div>
            <h3 className="font-medium mb-3">Node Executions</h3>
            <div className="space-y-3">
              {nodes.map(node => (
                <div key={node.id} className="border rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium">{node.name}</h4>
                    <Badge 
                      variant={node.status === 'completed' ? 'default' : node.status === 'failed' ? 'destructive' : 'secondary'}
                    >
                      {node.status}
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {node.startedAt && (
                      <div>
                        <p className="text-gray-500">Started</p>
                        <p>{format(new Date(node.startedAt), "HH:mm:ss")}</p>
                      </div>
                    )}
                    {node.completedAt && (
                      <div>
                        <p className="text-gray-500">Completed</p>
                        <p>{format(new Date(node.completedAt), "HH:mm:ss")}</p>
                      </div>
                    )}
                    {node.durationMs && (
                      <div>
                        <p className="text-gray-500">Duration</p>
                        <p>{node.durationMs}ms</p>
                      </div>
                    )}
                  </div>
                  
                  {node.errorMessage && (
                    <div className="mt-2">
                      <p className="text-red-500 text-sm">Error: {node.errorMessage}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}