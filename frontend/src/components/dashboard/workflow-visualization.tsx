"use client";

import { useState, useEffect, useRef } from "react";
import { useWorkflow, useWorkflowExecution } from "@/hooks/useAPI";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle 
} from "@/components/ui/dialog";
import { format } from "date-fns";

interface Node {
  id: string;
  name: string;
  type: 'job' | 'decision' | 'parallel' | 'subworkflow';
  x: number;
  y: number;
  status?: 'pending' | 'running' | 'completed' | 'failed';
}

interface Edge {
  id: string;
  source: string;
  target: string;
}

export function WorkflowVisualization({ workflowId }: { workflowId: string }) {
  const { workflow, loading, error, executeWorkflow } = useWorkflow(workflowId);
  const { execution } = useWorkflowExecution(workflowId);
  const [isExecuting, setIsExecuting] = useState(false);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [isNodeDialogOpen, setIsNodeDialogOpen] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const handleExecute = async () => {
    if (!workflow) return;
    
    try {
      setIsExecuting(true);
      await executeWorkflow();
    } catch (err) {
      console.error("Failed to execute workflow:", err);
    } finally {
      setIsExecuting(false);
    }
  };

  // Convert workflow data to visualization format
  const nodes: Node[] = workflow?.nodes.map((node: any, index: number) => ({
    id: node.id,
    name: node.name,
    type: node.type,
    x: 100 + (index % 3) * 200,
    y: 100 + Math.floor(index / 3) * 150,
    status: execution?.nodeExecutions[node.id]?.status || 'pending'
  })) || [];

  const edges: Edge[] = workflow?.edges.map((edge: any, index: number) => ({
    id: `edge-${index}`,
    source: edge.from,
    target: edge.to
  })) || [];

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Loading workflow...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500">Error loading workflow: {error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!workflow) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Workflow not found</p>
        </CardContent>
      </Card>
    );
  }

  // Get node by ID
  const getNode = (id: string) => nodes.find(node => node.id === id);

  // Get status color
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'running': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-300';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>Workflow Visualization: {workflow.name}</CardTitle>
          <div className="flex space-x-2">
            <Button 
              onClick={handleExecute} 
              disabled={isExecuting}
              variant="default"
            >
              {isExecuting ? "Executing..." : "Execute"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="border rounded-lg p-4 h-96 overflow-auto">
          <svg 
            ref={svgRef}
            width="100%" 
            height="100%" 
            className="min-w-full min-h-full"
          >
            {/* Render edges */}
            {edges.map(edge => {
              const sourceNode = getNode(edge.source);
              const targetNode = getNode(edge.target);
              
              if (!sourceNode || !targetNode) return null;
              
              return (
                <line
                  key={edge.id}
                  x1={sourceNode.x + 50}
                  y1={sourceNode.y + 25}
                  x2={targetNode.x}
                  y2={targetNode.y + 25}
                  stroke="#94a3b8"
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
              );
            })}
            
            {/* Arrow marker definition */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon
                  points="0 0, 10 3.5, 0 7"
                  fill="#94a3b8"
                />
              </marker>
            </defs>
            
            {/* Render nodes */}
            {nodes.map(node => (
              <g 
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={() => {
                  const nodeData = workflow.nodes.find((n: any) => n.id === node.id);
                  if (nodeData) {
                    setSelectedNode({
                      ...nodeData,
                      execution: execution?.nodeExecutions[node.id]
                    });
                    setIsNodeDialogOpen(true);
                  }
                }}
                className="cursor-pointer hover:opacity-80"
              >
                <rect
                  width="100"
                  height="50"
                  rx="5"
                  ry="5"
                  className={`${getStatusColor(node.status)} fill-opacity-20 stroke-opacity-80`}
                  stroke={node.status === 'running' ? '#3b82f6' : '#94a3b8'}
                  strokeWidth="2"
                />
                <text
                  x="50"
                  y="25"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs font-medium fill-foreground"
                >
                  {node.name}
                </text>
                <text
                  x="50"
                  y="40"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs fill-foreground"
                >
                  {node.type}
                </text>
              </g>
            ))}
          </svg>
        </div>
        
        {/* Workflow execution status */}
        {execution && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-medium">Execution Status</h3>
                <p className="text-sm text-gray-500">
                  Started: {execution.startedAt ? format(new Date(execution.startedAt), "MMM dd, yyyy HH:mm:ss") : "N/A"}
                </p>
                {execution.completedAt && (
                  <p className="text-sm text-gray-500">
                    Completed: {format(new Date(execution.completedAt), "MMM dd, yyyy HH:mm:ss")}
                  </p>
                )}
              </div>
              <Badge 
                variant={execution.status === 'completed' ? 'default' : execution.status === 'failed' ? 'destructive' : 'secondary'}
              >
                {execution.status}
              </Badge>
            </div>
            
            {execution.durationMs && (
              <p className="text-sm mt-2">
                Duration: {execution.durationMs}ms
              </p>
            )}
          </div>
        )}
        
        {/* Node details dialog */}
        <Dialog open={isNodeDialogOpen} onOpenChange={setIsNodeDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {selectedNode?.name} ({selectedNode?.type})
              </DialogTitle>
            </DialogHeader>
            {selectedNode && (
              <div className="grid gap-4 py-4">
                <div>
                  <h4 className="font-medium mb-2">Configuration</h4>
                  <pre className="bg-gray-100 p-3 rounded-md text-sm overflow-x-auto">
                    {JSON.stringify(selectedNode.config, null, 2)}
                  </pre>
                </div>
                
                {selectedNode.execution && (
                  <div>
                    <h4 className="font-medium mb-2">Execution Status</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Status:</span>
                        <Badge variant={selectedNode.execution.status === 'completed' ? 'default' : selectedNode.execution.status === 'failed' ? 'destructive' : 'secondary'}>
                          {selectedNode.execution.status}
                        </Badge>
                      </div>
                      {selectedNode.execution.startedAt && (
                        <p>Started: {format(new Date(selectedNode.execution.startedAt), "MMM dd, yyyy HH:mm:ss")}</p>
                      )}
                      {selectedNode.execution.completedAt && (
                        <p>Completed: {format(new Date(selectedNode.execution.completedAt), "MMM dd, yyyy HH:mm:ss")}</p>
                      )}
                      {selectedNode.execution.durationMs && (
                        <p>Duration: {selectedNode.execution.durationMs}ms</p>
                      )}
                      {selectedNode.execution.errorMessage && (
                        <div>
                          <p className="text-red-500">Error:</p>
                          <p className="text-red-500 text-sm">{selectedNode.execution.errorMessage}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}