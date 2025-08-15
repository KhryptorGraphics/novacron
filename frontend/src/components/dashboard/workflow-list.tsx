"use client";

import { useState } from "react";
import { useWorkflows } from "@/hooks/useAPI";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { format } from "date-fns";

export function WorkflowList() {
  const { workflows, loading, error, createWorkflow, updateWorkflow, deleteWorkflow, executeWorkflow, refetch } = useWorkflows();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingWorkflow, setEditingWorkflow] = useState<any>(null);
  const [newWorkflow, setNewWorkflow] = useState({
    name: "",
    description: "",
    enabled: true,
    nodes: "[]",
    edges: "[]",
    metadata: ""
  });

  const handleCreateWorkflow = async () => {
    try {
      const nodes = newWorkflow.nodes ? JSON.parse(newWorkflow.nodes) : [];
      const edges = newWorkflow.edges ? JSON.parse(newWorkflow.edges) : [];
      const metadata = newWorkflow.metadata ? JSON.parse(newWorkflow.metadata) : {};
      
      await createWorkflow({
        name: newWorkflow.name,
        description: newWorkflow.description,
        enabled: newWorkflow.enabled,
        nodes,
        edges,
        metadata
      });
      
      setIsCreateDialogOpen(false);
      setNewWorkflow({
        name: "",
        description: "",
        enabled: true,
        nodes: "[]",
        edges: "[]",
        metadata: ""
      });
    } catch (err) {
      console.error("Failed to create workflow:", err);
    }
  };

  const handleUpdateWorkflow = async () => {
    if (!editingWorkflow) return;
    
    try {
      const nodes = editingWorkflow.nodes ? JSON.parse(editingWorkflow.nodes) : [];
      const edges = editingWorkflow.edges ? JSON.parse(editingWorkflow.edges) : [];
      const metadata = editingWorkflow.metadata ? JSON.parse(editingWorkflow.metadata) : {};
      
      await updateWorkflow(editingWorkflow.id, {
        name: editingWorkflow.name,
        description: editingWorkflow.description,
        enabled: editingWorkflow.enabled,
        nodes,
        edges,
        metadata
      });
      
      setIsEditDialogOpen(false);
      setEditingWorkflow(null);
    } catch (err) {
      console.error("Failed to update workflow:", err);
    }
  };

  const handleDeleteWorkflow = async (id: string) => {
    if (confirm("Are you sure you want to delete this workflow?")) {
      try {
        await deleteWorkflow(id);
      } catch (err) {
        console.error("Failed to delete workflow:", err);
      }
    }
  };

  const handleExecuteWorkflow = async (id: string) => {
    try {
      await executeWorkflow(id);
      // Refresh workflows to show updated status
      refetch();
    } catch (err) {
      console.error("Failed to execute workflow:", err);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Management</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Loading workflows...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Management</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500">Error loading workflows: {error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>Workflow Management</CardTitle>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>Create Workflow</Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Create New Workflow</DialogTitle>
              </DialogHeader>
              <div className="grid gap-4 py-4 max-h-96 overflow-y-auto">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="workflow-name"
                    value={newWorkflow.name}
                    onChange={(e) => setNewWorkflow({...newWorkflow, name: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-description" className="text-right">
                    Description
                  </Label>
                  <Textarea
                    id="workflow-description"
                    value={newWorkflow.description}
                    onChange={(e) => setNewWorkflow({...newWorkflow, description: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-nodes" className="text-right">
                    Nodes
                  </Label>
                  <Textarea
                    id="workflow-nodes"
                    value={newWorkflow.nodes}
                    onChange={(e) => setNewWorkflow({...newWorkflow, nodes: e.target.value})}
                    className="col-span-3"
                    placeholder='[{"id": "1", "name": "task1", "type": "job", "config": {}}]'
                    rows={5}
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-edges" className="text-right">
                    Edges
                  </Label>
                  <Textarea
                    id="workflow-edges"
                    value={newWorkflow.edges}
                    onChange={(e) => setNewWorkflow({...newWorkflow, edges: e.target.value})}
                    className="col-span-3"
                    placeholder='[{"from": "1", "to": "2"}]'
                    rows={3}
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-metadata" className="text-right">
                    Metadata
                  </Label>
                  <Textarea
                    id="workflow-metadata"
                    value={newWorkflow.metadata}
                    onChange={(e) => setNewWorkflow({...newWorkflow, metadata: e.target.value})}
                    className="col-span-3"
                    placeholder='{"key": "value"}'
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="workflow-enabled" className="text-right">
                    Enabled
                  </Label>
                  <Switch
                    id="workflow-enabled"
                    checked={newWorkflow.enabled}
                    onCheckedChange={(checked) => setNewWorkflow({...newWorkflow, enabled: checked})}
                  />
                </div>
              </div>
              <Button onClick={handleCreateWorkflow}>Create Workflow</Button>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Description</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Created</TableHead>
              <TableHead>Updated</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {workflows && workflows.map((workflow) => (
              <TableRow key={workflow.id}>
                <TableCell className="font-medium">{workflow.name}</TableCell>
                <TableCell>{workflow.description || "No description"}</TableCell>
                <TableCell>
                  <Badge variant={workflow.enabled ? "default" : "destructive"}>
                    {workflow.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </TableCell>
                <TableCell>
                  {format(new Date(workflow.createdAt), "MMM dd, yyyy")}
                </TableCell>
                <TableCell>
                  {format(new Date(workflow.updatedAt), "MMM dd, yyyy")}
                </TableCell>
                <TableCell>
                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => {
                        setEditingWorkflow({
                          ...workflow,
                          nodes: JSON.stringify(workflow.nodes, null, 2),
                          edges: JSON.stringify(workflow.edges, null, 2),
                          metadata: workflow.metadata ? JSON.stringify(workflow.metadata, null, 2) : ""
                        });
                        setIsEditDialogOpen(true);
                      }}
                    >
                      Edit
                    </Button>
                    <Button 
                      size="sm" 
                      onClick={() => handleExecuteWorkflow(workflow.id)}
                    >
                      Run
                    </Button>
                    <Button 
                      size="sm" 
                      variant="destructive"
                      onClick={() => handleDeleteWorkflow(workflow.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        
        {/* Edit Workflow Dialog */}
        <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Edit Workflow</DialogTitle>
            </DialogHeader>
            {editingWorkflow && (
              <div className="grid gap-4 py-4 max-h-96 overflow-y-auto">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="edit-workflow-name"
                    value={editingWorkflow.name}
                    onChange={(e) => setEditingWorkflow({...editingWorkflow, name: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-description" className="text-right">
                    Description
                  </Label>
                  <Textarea
                    id="edit-workflow-description"
                    value={editingWorkflow.description}
                    onChange={(e) => setEditingWorkflow({...editingWorkflow, description: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-nodes" className="text-right">
                    Nodes
                  </Label>
                  <Textarea
                    id="edit-workflow-nodes"
                    value={editingWorkflow.nodes}
                    onChange={(e) => setEditingWorkflow({...editingWorkflow, nodes: e.target.value})}
                    className="col-span-3"
                    placeholder='[{"id": "1", "name": "task1", "type": "job", "config": {}}]'
                    rows={5}
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-edges" className="text-right">
                    Edges
                  </Label>
                  <Textarea
                    id="edit-workflow-edges"
                    value={editingWorkflow.edges}
                    onChange={(e) => setEditingWorkflow({...editingWorkflow, edges: e.target.value})}
                    className="col-span-3"
                    placeholder='[{"from": "1", "to": "2"}]'
                    rows={3}
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-metadata" className="text-right">
                    Metadata
                  </Label>
                  <Textarea
                    id="edit-workflow-metadata"
                    value={editingWorkflow.metadata}
                    onChange={(e) => setEditingWorkflow({...editingWorkflow, metadata: e.target.value})}
                    className="col-span-3"
                    placeholder='{"key": "value"}'
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-workflow-enabled" className="text-right">
                    Enabled
                  </Label>
                  <Switch
                    id="edit-workflow-enabled"
                    checked={editingWorkflow.enabled}
                    onCheckedChange={(checked) => setEditingWorkflow({...editingWorkflow, enabled: checked})}
                  />
                </div>
              </div>
            )}
            <Button onClick={handleUpdateWorkflow}>Update Workflow</Button>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}