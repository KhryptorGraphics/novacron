"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useVMs } from "@/lib/api/hooks/useVMs";
import { useVmTemplates } from "@/lib/api/hooks/useAdmin";
import { createVM, deleteVM, postVMAction } from "@/lib/api/vms";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import {
  Activity,
  AlertTriangle,
  Cpu,
  HardDrive,
  MemoryStick,
  Play,
  Plus,
  RotateCcw,
  Search,
  Server,
  Settings,
  Square,
  Trash2,
} from "lucide-react";
import { FadeIn } from "@/lib/animations";
import type { VM } from "@/lib/api/types";
import type { ListVMsParams } from "@/lib/api/vms";

type AdminVM = {
  id: string;
  name: string;
  status: string;
  node: string;
  created_at: string;
  updated_at: string;
};

function toAdminVM(vm: VM): AdminVM {
  return {
    id: vm.id,
    name: vm.name,
    status: vm.state || "unknown",
    node: vm.node_id || "unassigned",
    created_at: vm.created_at,
    updated_at: vm.updated_at,
  };
}

function statusColor(status: string): string {
  switch (status) {
    case "running":
      return "text-green-600";
    case "stopped":
      return "text-red-600";
    case "paused":
      return "text-yellow-600";
    case "migrating":
    case "creating":
      return "text-blue-600";
    default:
      return "text-gray-600";
  }
}

export default function VMManagementPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState({ status: "all", node: "all", search: "" });
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [isMutating, setIsMutating] = useState(false);
  const [newVM, setNewVM] = useState({ name: "", cpu_shares: 1, memory_mb: 1024 });

  const vmQueryParams: ListVMsParams = { pageSize: 100 };
  if (filters.status !== "all") vmQueryParams.state = filters.status;
  if (filters.node !== "all") vmQueryParams.nodeId = filters.node;
  if (filters.search) vmQueryParams.q = filters.search;

  const { items, isLoading, error } = useVMs(vmQueryParams);
  const { data: templates, isLoading: templatesLoading } = useVmTemplates();

  const vms = items.map(toAdminVM).filter((vm) => {
    const search = filters.search.toLowerCase();
    return !search || vm.name.toLowerCase().includes(search) || vm.id.toLowerCase().includes(search);
  });

  const stats = {
    total: vms.length,
    running: vms.filter((vm) => vm.status === "running").length,
    stopped: vms.filter((vm) => vm.status === "stopped").length,
    unassigned: vms.filter((vm) => vm.node === "unassigned").length,
  };

  const refreshVMs = () => queryClient.invalidateQueries({ queryKey: ["vms"] });

  const runVMAction = async (vm: AdminVM, action: "start" | "stop" | "restart" | "delete") => {
    setIsMutating(true);
    try {
      if (action === "delete") {
        await deleteVM(vm.id);
      } else if (action === "restart") {
        await postVMAction(vm.id, "stop", { role: "operator" });
        await postVMAction(vm.id, "start", { role: "operator" });
      } else {
        await postVMAction(vm.id, action, { role: "operator" });
      }
      await refreshVMs();
      toast({ title: "VM action completed", description: `${action} completed for ${vm.name}.` });
    } catch (err) {
      toast({
        title: "VM action failed",
        description: err instanceof Error ? err.message : `Unable to ${action} ${vm.name}.`,
        variant: "destructive",
      });
    } finally {
      setIsMutating(false);
    }
  };

  const handleCreateVM = async () => {
    if (!newVM.name.trim()) {
      toast({ title: "VM name required", description: "Enter a VM name before creating it.", variant: "destructive" });
      return;
    }

    setIsMutating(true);
    try {
      await createVM({
        name: newVM.name.trim(),
        cpu_shares: newVM.cpu_shares,
        memory_mb: newVM.memory_mb,
      });
      await refreshVMs();
      setShowCreateDialog(false);
      setNewVM({ name: "", cpu_shares: 1, memory_mb: 1024 });
      toast({ title: "VM creation requested", description: `${newVM.name} is being created by the canonical runtime.` });
    } catch (err) {
      toast({
        title: "Failed to create VM",
        description: err instanceof Error ? err.message : "The canonical VM API rejected the request.",
        variant: "destructive",
      });
    } finally {
      setIsMutating(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Server className="h-8 w-8" />
            VM Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Manage canonical virtual machines and runtime-backed templates.
          </p>
        </div>

        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Create VM
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create VM</DialogTitle>
              <DialogDescription>Create a VM through the canonical runtime API.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="vm-name">Name</Label>
                <Input id="vm-name" value={newVM.name} onChange={(event) => setNewVM((prev) => ({ ...prev, name: event.target.value }))} />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="vm-cpu">CPU shares</Label>
                <Input
                  id="vm-cpu"
                  type="number"
                  min={1}
                  value={newVM.cpu_shares}
                  onChange={(event) => setNewVM((prev) => ({ ...prev, cpu_shares: Number(event.target.value) || 1 }))}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="vm-memory">Memory MB</Label>
                <Input
                  id="vm-memory"
                  type="number"
                  min={128}
                  step={128}
                  value={newVM.memory_mb}
                  onChange={(event) => setNewVM((prev) => ({ ...prev, memory_mb: Number(event.target.value) || 1024 }))}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>Cancel</Button>
              <Button onClick={handleCreateVM} disabled={isMutating}>{isMutating ? "Creating..." : "Create VM"}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total VMs</p>
                  <p className="text-2xl font-bold">{stats.total}</p>
                  <div className="text-xs text-gray-600">{stats.running} running</div>
                </div>
                <Server className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Running</p>
                  <p className="text-2xl font-bold text-green-600">{stats.running}</p>
                  <div className="text-xs text-gray-600">Active runtime state</div>
                </div>
                <Activity className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Stopped</p>
                  <p className="text-2xl font-bold text-red-600">{stats.stopped}</p>
                  <div className="text-xs text-gray-600">Ready for start</div>
                </div>
                <Square className="h-8 w-8 text-red-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Unassigned</p>
                  <p className="text-2xl font-bold text-orange-600">{stats.unassigned}</p>
                  <div className="text-xs text-gray-600">No node reported</div>
                </div>
                <AlertTriangle className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>

      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search VMs by name or ID..."
                value={filters.search}
                onChange={(event) => setFilters((prev) => ({ ...prev, search: event.target.value }))}
                className="pl-10"
              />
            </div>
            <Select value={filters.status} onValueChange={(value) => setFilters((prev) => ({ ...prev, status: value }))}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="stopped">Stopped</SelectItem>
                <SelectItem value="creating">Creating</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filters.node} onValueChange={(value) => setFilters((prev) => ({ ...prev, node: value }))}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All nodes" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Nodes</SelectItem>
                {Array.from(new Set(vms.map((vm) => vm.node))).map((node) => (
                  <SelectItem key={node} value={node}>{node}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <FadeIn>
        <Card>
          <CardHeader>
            <CardTitle>VM Instances ({vms.length})</CardTitle>
            <CardDescription>Runtime-backed VM inventory from the canonical API.</CardDescription>
          </CardHeader>
          <CardContent>
            {Boolean(error) && (
              <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                Failed to load VM inventory from the canonical API.
              </div>
            )}
            {isLoading ? (
              <div className="py-12 text-center text-sm text-gray-600">Loading VMs...</div>
            ) : vms.length === 0 ? (
              <div className="py-12 text-center">
                <Server className="mx-auto h-12 w-12 text-muted-foreground" />
                <h3 className="mt-4 text-lg font-semibold">No VMs found</h3>
                <p className="text-sm text-muted-foreground mt-1">Create a VM or adjust filters.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Node</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {vms.map((vm) => (
                      <TableRow key={vm.id}>
                        <TableCell>
                          <div>
                            <div className="font-medium">{vm.name}</div>
                            <div className="text-xs text-muted-foreground">{vm.id}</div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className={statusColor(vm.status)}>{vm.status}</Badge>
                        </TableCell>
                        <TableCell>{vm.node}</TableCell>
                        <TableCell>{new Date(vm.updated_at).toLocaleString()}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            {vm.status === "running" ? (
                              <Button size="sm" variant="outline" onClick={() => runVMAction(vm, "stop")} disabled={isMutating}>
                                <Square className="h-3 w-3 mr-1" />
                                Stop
                              </Button>
                            ) : (
                              <Button size="sm" variant="outline" onClick={() => runVMAction(vm, "start")} disabled={isMutating}>
                                <Play className="h-3 w-3 mr-1" />
                                Start
                              </Button>
                            )}
                            <Button size="sm" variant="outline" onClick={() => runVMAction(vm, "restart")} disabled={isMutating || vm.status !== "running"}>
                              <RotateCcw className="h-3 w-3 mr-1" />
                              Restart
                            </Button>
                            <Button size="sm" variant="destructive" onClick={() => runVMAction(vm, "delete")} disabled={isMutating}>
                              <Trash2 className="h-3 w-3 mr-1" />
                              Delete
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>

      <FadeIn delay={0.5}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              VM Templates
            </CardTitle>
            <CardDescription>Template inventory from the admin API.</CardDescription>
          </CardHeader>
          <CardContent>
            {templatesLoading ? (
              <div className="py-8 text-sm text-gray-600">Loading templates...</div>
            ) : !templates?.length ? (
              <div className="py-8 text-sm text-gray-600">No VM templates are available.</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {templates.map((template) => (
                  <Card key={template.id}>
                    <CardHeader>
                      <CardTitle className="text-base">{template.name}</CardTitle>
                      <CardDescription>{template.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="grid grid-cols-3 gap-3 text-sm">
                      <div className="flex items-center gap-1">
                        <Cpu className="h-4 w-4" />
                        {template.cpu_cores} CPU
                      </div>
                      <div className="flex items-center gap-1">
                        <MemoryStick className="h-4 w-4" />
                        {Math.round(template.memory_mb / 1024)} GB
                      </div>
                      <div className="flex items-center gap-1">
                        <HardDrive className="h-4 w-4" />
                        {template.disk_gb} GB
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  );
}
