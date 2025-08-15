"use client";

import { useState } from "react";
import { useJobs } from "@/hooks/useAPI";
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

export function JobList() {
  const { jobs, loading, error, createJob, updateJob, deleteJob, executeJob, refetch } = useJobs();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingJob, setEditingJob] = useState<any>(null);
  const [newJob, setNewJob] = useState({
    name: "",
    schedule: "",
    timezone: "UTC",
    enabled: true,
    priority: 5,
    max_retries: 3,
    timeout: 30000,
    metadata: ""
  });

  const handleCreateJob = async () => {
    try {
      const metadata = newJob.metadata ? JSON.parse(newJob.metadata) : {};
      await createJob({
        ...newJob,
        metadata
      });
      setIsCreateDialogOpen(false);
      setNewJob({
        name: "",
        schedule: "",
        timezone: "UTC",
        enabled: true,
        priority: 5,
        max_retries: 3,
        timeout: 30000,
        metadata: ""
      });
    } catch (err) {
      console.error("Failed to create job:", err);
    }
  };

  const handleUpdateJob = async () => {
    if (!editingJob) return;
    
    try {
      const metadata = editingJob.metadata ? JSON.parse(editingJob.metadata) : {};
      await updateJob(editingJob.id, {
        ...editingJob,
        metadata
      });
      setIsEditDialogOpen(false);
      setEditingJob(null);
    } catch (err) {
      console.error("Failed to update job:", err);
    }
  };

  const handleDeleteJob = async (id: string) => {
    if (confirm("Are you sure you want to delete this job?")) {
      try {
        await deleteJob(id);
      } catch (err) {
        console.error("Failed to delete job:", err);
      }
    }
  };

  const handleExecuteJob = async (id: string) => {
    try {
      await executeJob(id);
      // Refresh jobs to show updated status
      refetch();
    } catch (err) {
      console.error("Failed to execute job:", err);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Job Management</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Loading jobs...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Job Management</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500">Error loading jobs: {error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>Job Management</CardTitle>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>Create Job</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Job</DialogTitle>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="name"
                    value={newJob.name}
                    onChange={(e) => setNewJob({...newJob, name: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="schedule" className="text-right">
                    Schedule
                  </Label>
                  <Input
                    id="schedule"
                    value={newJob.schedule}
                    onChange={(e) => setNewJob({...newJob, schedule: e.target.value})}
                    className="col-span-3"
                    placeholder="* * * * *"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="timezone" className="text-right">
                    Timezone
                  </Label>
                  <Input
                    id="timezone"
                    value={newJob.timezone}
                    onChange={(e) => setNewJob({...newJob, timezone: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="priority" className="text-right">
                    Priority
                  </Label>
                  <Input
                    id="priority"
                    type="number"
                    value={newJob.priority}
                    onChange={(e) => setNewJob({...newJob, priority: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="max_retries" className="text-right">
                    Max Retries
                  </Label>
                  <Input
                    id="max_retries"
                    type="number"
                    value={newJob.max_retries}
                    onChange={(e) => setNewJob({...newJob, max_retries: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="timeout" className="text-right">
                    Timeout (ms)
                  </Label>
                  <Input
                    id="timeout"
                    type="number"
                    value={newJob.timeout}
                    onChange={(e) => setNewJob({...newJob, timeout: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="metadata" className="text-right">
                    Metadata
                  </Label>
                  <Textarea
                    id="metadata"
                    value={newJob.metadata}
                    onChange={(e) => setNewJob({...newJob, metadata: e.target.value})}
                    className="col-span-3"
                    placeholder='{"key": "value"}'
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="enabled" className="text-right">
                    Enabled
                  </Label>
                  <Switch
                    id="enabled"
                    checked={newJob.enabled}
                    onCheckedChange={(checked) => setNewJob({...newJob, enabled: checked})}
                  />
                </div>
              </div>
              <Button onClick={handleCreateJob}>Create Job</Button>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Schedule</TableHead>
              <TableHead>Timezone</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Next Run</TableHead>
              <TableHead>Priority</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs && jobs.map((job) => (
              <TableRow key={job.id}>
                <TableCell className="font-medium">{job.name}</TableCell>
                <TableCell>{job.schedule}</TableCell>
                <TableCell>{job.timezone}</TableCell>
                <TableCell>
                  <Badge variant={job.enabled ? "default" : "destructive"}>
                    {job.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </TableCell>
                <TableCell>
                  {job.next_run_at ? format(new Date(job.next_run_at), "MMM dd, yyyy HH:mm") : "N/A"}
                </TableCell>
                <TableCell>{job.priority}</TableCell>
                <TableCell>
                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => {
                        setEditingJob({
                          ...job,
                          metadata: job.metadata ? JSON.stringify(job.metadata, null, 2) : ""
                        });
                        setIsEditDialogOpen(true);
                      }}
                    >
                      Edit
                    </Button>
                    <Button 
                      size="sm" 
                      onClick={() => handleExecuteJob(job.id)}
                    >
                      Run
                    </Button>
                    <Button 
                      size="sm" 
                      variant="destructive"
                      onClick={() => handleDeleteJob(job.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        
        {/* Edit Job Dialog */}
        <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Job</DialogTitle>
            </DialogHeader>
            {editingJob && (
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-name" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="edit-name"
                    value={editingJob.name}
                    onChange={(e) => setEditingJob({...editingJob, name: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-schedule" className="text-right">
                    Schedule
                  </Label>
                  <Input
                    id="edit-schedule"
                    value={editingJob.schedule}
                    onChange={(e) => setEditingJob({...editingJob, schedule: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-timezone" className="text-right">
                    Timezone
                  </Label>
                  <Input
                    id="edit-timezone"
                    value={editingJob.timezone}
                    onChange={(e) => setEditingJob({...editingJob, timezone: e.target.value})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-priority" className="text-right">
                    Priority
                  </Label>
                  <Input
                    id="edit-priority"
                    type="number"
                    value={editingJob.priority}
                    onChange={(e) => setEditingJob({...editingJob, priority: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-max_retries" className="text-right">
                    Max Retries
                  </Label>
                  <Input
                    id="edit-max_retries"
                    type="number"
                    value={editingJob.max_retries}
                    onChange={(e) => setEditingJob({...editingJob, max_retries: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-timeout" className="text-right">
                    Timeout (ms)
                  </Label>
                  <Input
                    id="edit-timeout"
                    type="number"
                    value={editingJob.timeout}
                    onChange={(e) => setEditingJob({...editingJob, timeout: parseInt(e.target.value)})}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-metadata" className="text-right">
                    Metadata
                  </Label>
                  <Textarea
                    id="edit-metadata"
                    value={editingJob.metadata}
                    onChange={(e) => setEditingJob({...editingJob, metadata: e.target.value})}
                    className="col-span-3"
                    placeholder='{"key": "value"}'
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="edit-enabled" className="text-right">
                    Enabled
                  </Label>
                  <Switch
                    id="edit-enabled"
                    checked={editingJob.enabled}
                    onCheckedChange={(checked) => setEditingJob({...editingJob, enabled: checked})}
                  />
                </div>
              </div>
            )}
            <Button onClick={handleUpdateJob}>Update Job</Button>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}