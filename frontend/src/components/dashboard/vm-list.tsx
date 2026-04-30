'use client';

import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { OSSelector } from "@/components/dashboard/os-selector";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { useToast } from "@/components/ui/use-toast";
import {
  Play,
  Square,
  RotateCw,
  Trash2,
  MoreVertical,
  Plus,
  Pause,
  ExternalLink,
  Copy,
  Server,
  HardDrive,
  Cpu,
  MemoryStick
} from "lucide-react";
import { cn } from "@/lib/utils";
import { createVM, deleteVM, postVMAction, type ListVMsParams } from "@/lib/api/vms";
import { useVMs } from "@/lib/api/hooks/useVMs";
import type { VM } from "@/lib/api/types";
import { formatDistanceToNow } from 'date-fns';

// Types
interface VMProps {
  id: string;
  name: string;
  state: string;
  node_id: string;
  owner: string;
  spec: {
    vcpu: number;
    memory_mb: number;
    disk_mb: number;
    type: string;
    image: string;
  };
  created_at: string;
  updated_at: string;
  tags: Record<string, string>;
  process_info?: {
    cpu_usage_percent: number;
    memory_usage_mb: number;
  };
}

interface VMListProps {
  title?: string;
  limit?: number;
  filter?: string;
  onVMClick?: (vm: VMProps) => void;
}

// State color mapping
const stateColors = {
  running: "bg-green-500",
  stopped: "bg-slate-500",
  paused: "bg-amber-500",
  error: "bg-red-500",
  creating: "bg-blue-500",
  unknown: "bg-purple-500",
  migrating: "bg-indigo-500",
  restarting: "bg-cyan-500",
  deleting: "bg-rose-500",
};

function toVMProps(vm: VM): VMProps {
  return {
    id: vm.id,
    name: vm.name,
    state: vm.state || "unknown",
    node_id: vm.node_id || "unassigned",
    owner: "system",
    spec: {
      vcpu: 0,
      memory_mb: 0,
      disk_mb: 0,
      type: "canonical",
      image: "runtime-managed",
    },
    created_at: vm.created_at,
    updated_at: vm.updated_at,
    tags: {},
  };
}

// VM List Component
export function VMList({ title = "Virtual Machines", limit, filter, onVMClick }: VMListProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const vmQueryParams: ListVMsParams = {};
  if (limit !== undefined) vmQueryParams.pageSize = limit;
  if (filter) vmQueryParams.q = filter;
  const { items, isLoading: isVMsLoading, error: vmsError } = useVMs(vmQueryParams);
  const [isMutating, setIsMutating] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    action: string;
    vmId?: string;
    vmName?: string;
  }>({
    open: false,
    action: "",
  });

  // New VM form state
  const [newVM, setNewVM] = useState({
    name: "",
    type: "container",
    image: "",
    vcpu: 1,
    memory_mb: 1024,
    disk_mb: 10240,
  });

  const vms = items
    .map(toVMProps)
    .filter((vm) => {
      if (!filter) return true;
      const normalizedFilter = filter.toLowerCase();
      return (
        vm.name.toLowerCase().includes(normalizedFilter) ||
        vm.state.toLowerCase() === normalizedFilter ||
        vm.node_id.toLowerCase() === normalizedFilter ||
        vm.spec.type.toLowerCase() === normalizedFilter
      );
    })
    .slice(0, limit || undefined);

  const isLoading = isVMsLoading || isMutating;

  // VM operations
  const handleVMOperation = async (vmId: string, operation: string) => {
    setIsMutating(true);

    try {
      switch (operation) {
        case "start":
        case "stop":
          await postVMAction(vmId, operation, { role: "operator" });
          break;
        case "restart":
          await postVMAction(vmId, "stop", { role: "operator" });
          await postVMAction(vmId, "start", { role: "operator" });
          break;
        case "delete":
          await deleteVM(vmId);
          break;
        case "pause":
          throw new Error("Pause is not exposed by the canonical VM API yet");
        default:
          throw new Error(`Unsupported VM operation: ${operation}`);
      }

      await queryClient.invalidateQueries({ queryKey: ["vms"] });

      toast({
        title: "Success",
        description: `VM ${operation} operation completed successfully`,
      });
    } catch (error) {
      console.error(`Failed to ${operation} VM:`, error);
      toast({
        title: "Error",
        description: `Failed to ${operation} VM. Please try again.`,
        variant: "destructive",
      });
    } finally {
      setIsMutating(false);
      setConfirmDialog({ open: false, action: "" });
    }
  };

  // Create new VM
  const handleCreateVM = async () => {
    setIsMutating(true);

    try {
      // Validate form
      if (!newVM.name || !newVM.image) {
        throw new Error("Name and image are required");
      }

      await createVM({
        name: newVM.name,
        cpu_shares: newVM.vcpu,
        memory_mb: newVM.memory_mb,
        tags: {
          type: newVM.type,
          image: newVM.image,
          disk_mb: newVM.disk_mb,
          created_by: "ui",
        },
      });
      await queryClient.invalidateQueries({ queryKey: ["vms"] });

      toast({
        title: "Success",
        description: `VM ${newVM.name} created successfully`,
      });

      // Reset form and close dialog
      setNewVM({
        name: "",
        type: "container",
        image: "",
        vcpu: 1,
        memory_mb: 1024,
        disk_mb: 10240,
      });
      setShowCreateDialog(false);
    } catch (error) {
      console.error('Failed to create VM:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to create VM",
        variant: "destructive",
      });
    } finally {
      setIsMutating(false);
    }
  };

  // Get VM details
  const openVMDetails = (vm: VMProps) => {
    if (onVMClick) {
      onVMClick(vm);
    }
  };

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle>{title}</CardTitle>
          <Button onClick={() => setShowCreateDialog(true)} variant="outline" size="sm">
            <Plus className="h-4 w-4 mr-2" />
            New VM
          </Button>
        </div>
        <CardDescription>
          Manage and monitor virtual machines across your infrastructure
        </CardDescription>
      </CardHeader>
      <CardContent>
        {vmsError ? (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            Failed to load virtual machines from the canonical API.
          </div>
        ) : isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
          </div>
        ) : vms.length === 0 ? (
          <div className="text-center py-12">
            <Server className="mx-auto h-12 w-12 text-muted-foreground" />
            <h3 className="mt-4 text-lg font-semibold">No VMs found</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {filter ? "Try changing your filter criteria" : "Create a new VM to get started"}
            </p>
            {!filter && (
              <Button
                onClick={() => setShowCreateDialog(true)}
                variant="outline"
                className="mt-4"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create VM
              </Button>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table className="border-collapse border-spacing-0">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">Status</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Resources</TableHead>
                  <TableHead>Node</TableHead>
                  <TableHead>Uptime</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {vms.map((vm) => (
                  <TableRow
                    key={vm.id}
                    onClick={() => openVMDetails(vm)}
                    className="cursor-pointer hover:bg-muted/50"
                  >
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <div
                          className={cn(
                            "h-3 w-3 rounded-full",
                            stateColors[vm.state as keyof typeof stateColors] || "bg-gray-500"
                          )}
                          aria-hidden="true"
                        />
                        <span className="capitalize text-xs">{vm.state}</span>
                      </div>
                    </TableCell>
                    <TableCell className="font-medium">
                      {vm.name}
                      {Object.keys(vm.tags).length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {Object.entries(vm.tags).map(([key, value]) => (
                            <Badge key={key} variant="outline" className="text-xs">
                              {key}:{value}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </TableCell>
                    <TableCell>{vm.spec.type}</TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1 text-xs">
                        <div className="flex items-center">
                          <Cpu className="h-3 w-3 mr-1" />
                          <span>{vm.spec.vcpu} vCPU</span>
                          {vm.process_info && (
                            <span className="text-muted-foreground ml-1">
                              ({vm.process_info.cpu_usage_percent.toFixed(1)}%)
                            </span>
                          )}
                        </div>
                        <div className="flex items-center">
                          <MemoryStick className="h-3 w-3 mr-1" />
                          <span>{(vm.spec.memory_mb / 1024).toFixed(1)} GB</span>
                          {vm.process_info && (
                            <span className="text-muted-foreground ml-1">
                              ({(vm.process_info.memory_usage_mb / 1024).toFixed(1)} GB used)
                            </span>
                          )}
                        </div>
                        <div className="flex items-center">
                          <HardDrive className="h-3 w-3 mr-1" />
                          <span>{(vm.spec.disk_mb / 1024).toFixed(1)} GB</span>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{vm.node_id}</Badge>
                    </TableCell>
                    <TableCell>
                      {vm.state === "running" ? (
                        formatDistanceToNow(new Date(vm.updated_at), { addSuffix: true })
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <span className="sr-only">Open menu</span>
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuLabel>Actions</DropdownMenuLabel>
                          {vm.state === "running" && (
                            <>
                              <DropdownMenuItem
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setConfirmDialog({
                                    open: true,
                                    action: "stop",
                                    vmId: vm.id,
                                    vmName: vm.name,
                                  });
                                }}
                              >
                                <Square className="mr-2 h-4 w-4" />
                                Stop
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setConfirmDialog({
                                    open: true,
                                    action: "restart",
                                    vmId: vm.id,
                                    vmName: vm.name,
                                  });
                                }}
                              >
                                <RotateCw className="mr-2 h-4 w-4" />
                                Restart
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleVMOperation(vm.id, "pause");
                                }}
                              >
                                <Pause className="mr-2 h-4 w-4" />
                                Pause
                              </DropdownMenuItem>
                            </>
                          )}
                          {(vm.state === "stopped" || vm.state === "paused") && (
                            <DropdownMenuItem
                              onClick={(e) => {
                                e.stopPropagation();
                                handleVMOperation(vm.id, "start");
                              }}
                            >
                              <Play className="mr-2 h-4 w-4" />
                              Start
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              openVMDetails(vm);
                            }}
                          >
                            <ExternalLink className="mr-2 h-4 w-4" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              // Copy VM ID to clipboard
                              navigator.clipboard.writeText(vm.id);
                              toast({
                                title: "Copied",
                                description: "VM ID copied to clipboard",
                              });
                            }}
                          >
                            <Copy className="mr-2 h-4 w-4" />
                            Copy ID
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              setConfirmDialog({
                                open: true,
                                action: "delete",
                                vmId: vm.id,
                                vmName: vm.name,
                              });
                            }}
                            className="text-red-600"
                          >
                            <Trash2 className="mr-2 h-4 w-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
      {!limit && vms.length > 0 && (
        <CardFooter className="flex justify-between">
          <div className="text-sm text-muted-foreground">
            Showing {vms.length} virtual machines
          </div>
          <Button variant="outline" size="sm" onClick={() => {}}>
            View All
          </Button>
        </CardFooter>
      )}

      {/* Create VM Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Create New VM</DialogTitle>
            <DialogDescription>
              Configure and create a new virtual machine.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="vm-name" className="text-right">
                Name
              </Label>
              <Input
                id="vm-name"
                value={newVM.name}
                onChange={(e) => setNewVM({ ...newVM, name: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="vm-type" className="text-right">
                Type
              </Label>
              <Select
                value={newVM.type}
                onValueChange={(value) => setNewVM({ ...newVM, type: value })}
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select VM Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="container">Container</SelectItem>
                  <SelectItem value="containerd">Containerd</SelectItem>
                  <SelectItem value="kvm">KVM</SelectItem>
                  <SelectItem value="process">Process</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {newVM.type === "kvm" ? (
              <div className="grid grid-cols-4 items-center gap-4">
                <div className="col-span-4">
                  <OSSelector
                    value={newVM.image}
                    onChange={(value) => setNewVM({ ...newVM, image: value })}
                    className="grid grid-cols-4 items-center gap-4"
                  />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="vm-image" className="text-right">
                  Image
                </Label>
                <Input
                  id="vm-image"
                  value={newVM.image}
                  onChange={(e) => setNewVM({ ...newVM, image: e.target.value })}
                  className="col-span-3"
                  placeholder={
                    newVM.type === "container" ? "nginx:latest" :
                    newVM.type === "containerd" ? "docker.io/library/alpine:latest" :
                    "bash -c 'sleep infinity'"
                  }
                />
              </div>
            )}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="vm-vcpu" className="text-right">
                vCPU
              </Label>
              <div className="flex items-center col-span-3">
                <Input
                  id="vm-vcpu"
                  type="number"
                  min="1"
                  max="32"
                  value={newVM.vcpu}
                  onChange={(e) => setNewVM({ ...newVM, vcpu: parseInt(e.target.value) || 1 })}
                  className="w-20"
                />
                <span className="ml-2 text-sm text-muted-foreground">cores</span>
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="vm-memory" className="text-right">
                Memory
              </Label>
              <div className="flex items-center col-span-3">
                <Input
                  id="vm-memory"
                  type="number"
                  min="128"
                  max="65536"
                  step="128"
                  value={newVM.memory_mb}
                  onChange={(e) => setNewVM({ ...newVM, memory_mb: parseInt(e.target.value) || 1024 })}
                  className="w-24"
                />
                <span className="ml-2 text-sm text-muted-foreground">MB ({(newVM.memory_mb / 1024).toFixed(1)} GB)</span>
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="vm-disk" className="text-right">
                Disk
              </Label>
              <div className="flex items-center col-span-3">
                <Input
                  id="vm-disk"
                  type="number"
                  min="1024"
                  max="1048576"
                  step="1024"
                  value={newVM.disk_mb}
                  onChange={(e) => setNewVM({ ...newVM, disk_mb: parseInt(e.target.value) || 10240 })}
                  className="w-24"
                />
                <span className="ml-2 text-sm text-muted-foreground">MB ({(newVM.disk_mb / 1024).toFixed(1)} GB)</span>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateVM} disabled={isLoading}>
              {isLoading ? (
                <>
                  <span className="animate-spin mr-2">⟳</span>
                  Creating...
                </>
              ) : (
                "Create VM"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onOpenChange={(open) =>
          setConfirmDialog({ ...confirmDialog, open })
        }
      >
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>
              {confirmDialog.action === "delete"
                ? "Delete VM"
                : confirmDialog.action === "stop"
                ? "Stop VM"
                : confirmDialog.action === "restart"
                ? "Restart VM"
                : "Confirm Action"}
            </DialogTitle>
            <DialogDescription>
              {confirmDialog.action === "delete"
                ? "Are you sure you want to delete this VM? This action cannot be undone."
                : confirmDialog.action === "stop"
                ? "Are you sure you want to stop this VM? Any running processes will be terminated."
                : confirmDialog.action === "restart"
                ? "Are you sure you want to restart this VM? This will cause a brief downtime."
                : "Please confirm this action."}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <p className="text-sm">
              VM: <span className="font-semibold">{confirmDialog.vmName}</span>
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              ID: {confirmDialog.vmId}
            </p>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmDialog({ open: false, action: "" })}
            >
              Cancel
            </Button>
            <Button
              variant={confirmDialog.action === "delete" ? "destructive" : "default"}
              onClick={() => {
                if (confirmDialog.vmId && confirmDialog.action) {
                  handleVMOperation(confirmDialog.vmId, confirmDialog.action);
                }
              }}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="animate-spin mr-2">⟳</span>
                  Processing...
                </>
              ) : (
                confirmDialog.action === "delete"
                  ? "Delete"
                  : confirmDialog.action === "stop"
                  ? "Stop"
                  : confirmDialog.action === "restart"
                  ? "Restart"
                  : "Confirm"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
