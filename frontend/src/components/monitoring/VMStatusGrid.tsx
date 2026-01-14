"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ResponsiveDataTable } from "@/components/ui/responsive-table";
import { 
  Server, 
  PlayCircle, 
  PauseCircle, 
  StopCircle,
  RefreshCw,
  MoreVertical,
  Cpu,
  HardDrive,
  Activity
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface VM {
  id: string;
  name: string;
  status: "running" | "stopped" | "paused" | "migrating";
  cpu: number;
  memory: number;
  disk: number;
  network: {
    in: number;
    out: number;
  };
  uptime: string;
  host: string;
}

interface VMStatusGridProps {
  vms: VM[];
  onVMAction?: (vmId: string, action: string) => void;
  loading?: boolean;
}

export function VMStatusGrid({ vms, onVMAction, loading = false }: VMStatusGridProps) {
  const [selectedVMs, setSelectedVMs] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  
  const getStatusColor = (status: VM["status"]) => {
    const colors = {
      running: "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400",
      stopped: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400",
      paused: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400",
      migrating: "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
    };
    return colors[status];
  };
  
  const getStatusIcon = (status: VM["status"]) => {
    const icons = {
      running: <PlayCircle className="h-4 w-4" />,
      stopped: <StopCircle className="h-4 w-4" />,
      paused: <PauseCircle className="h-4 w-4" />,
      migrating: <RefreshCw className="h-4 w-4 animate-spin" />
    };
    return icons[status];
  };
  
  const handleSelectVM = (vmId: string) => {
    setSelectedVMs(prev =>
      prev.includes(vmId)
        ? prev.filter(id => id !== vmId)
        : [...prev, vmId]
    );
  };
  
  // Table columns configuration
  const columns = [
    {
      key: "name",
      header: "VM Name",
      priority: "high" as const,
      render: (value: string, vm: VM) => (
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-gray-500" />
          <span className="font-medium">{value}</span>
        </div>
      )
    },
    {
      key: "status",
      header: "Status",
      priority: "high" as const,
      render: (_: any, vm: VM) => (
        <Badge className={getStatusColor(vm.status)}>
          <span className="flex items-center gap-1">
            {getStatusIcon(vm.status)}
            {vm.status}
          </span>
        </Badge>
      )
    },
    {
      key: "cpu",
      header: "CPU",
      priority: "medium" as const,
      align: "center" as const,
      render: (value: number) => (
        <div className="flex items-center justify-center gap-1">
          <Cpu className="h-3 w-3 text-gray-500" />
          <span>{value}%</span>
        </div>
      )
    },
    {
      key: "memory",
      header: "Memory",
      priority: "medium" as const,
      align: "center" as const,
      render: (value: number) => `${value}%`
    },
    {
      key: "disk",
      header: "Disk",
      priority: "low" as const,
      align: "center" as const,
      render: (value: number) => (
        <div className="flex items-center justify-center gap-1">
          <HardDrive className="h-3 w-3 text-gray-500" />
          <span>{value}%</span>
        </div>
      )
    },
    {
      key: "host",
      header: "Host",
      priority: "low" as const,
      render: (value: string) => (
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {value}
        </span>
      )
    },
    {
      key: "actions",
      header: "Actions",
      priority: "high" as const,
      align: "center" as const,
      render: (_: any, vm: VM) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {vm.status === "running" && (
              <>
                <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "pause")}>
                  <PauseCircle className="mr-2 h-4 w-4" />
                  Pause
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "stop")}>
                  <StopCircle className="mr-2 h-4 w-4" />
                  Stop
                </DropdownMenuItem>
              </>
            )}
            {vm.status === "stopped" && (
              <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "start")}>
                <PlayCircle className="mr-2 h-4 w-4" />
                Start
              </DropdownMenuItem>
            )}
            {vm.status === "paused" && (
              <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "resume")}>
                <PlayCircle className="mr-2 h-4 w-4" />
                Resume
              </DropdownMenuItem>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "restart")}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Restart
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "migrate")}>
              Migrate
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onVMAction?.(vm.id, "console")}>
              Console
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )
    }
  ];
  
  // Mobile card renderer
  const renderMobileCard = (vm: VM) => {
    if (!vm) return null;
    
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Server className="h-5 w-5 text-gray-500" />
            <span className="font-semibold">{vm.name || 'Unknown'}</span>
          </div>
          <Badge className={getStatusColor(vm.status || 'stopped')}>
            <span className="flex items-center gap-1">
              {getStatusIcon(vm.status || 'stopped')}
              {vm.status || 'stopped'}
            </span>
          </Badge>
        </div>
      
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="text-center">
          <p className="text-gray-500 dark:text-gray-400">CPU</p>
          <p className="font-semibold">{vm.cpu || 0}%</p>
        </div>
        <div className="text-center">
          <p className="text-gray-500 dark:text-gray-400">Memory</p>
          <p className="font-semibold">{vm.memory || 0}%</p>
        </div>
        <div className="text-center">
          <p className="text-gray-500 dark:text-gray-400">Disk</p>
          <p className="font-semibold">{vm.disk || 0}%</p>
        </div>
      </div>
      
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500 dark:text-gray-400">
          Host: {vm.host || 'Unknown'}
        </span>
        <span className="text-gray-500 dark:text-gray-400">
          Uptime: {vm.uptime || 'N/A'}
        </span>
      </div>
      
      <div className="flex gap-2">
        {vm.status === "running" ? (
          <>
            <Button
              size="sm"
              variant="outline"
              className="flex-1"
              onClick={() => onVMAction?.(vm.id, "pause")}
            >
              <PauseCircle className="mr-1 h-3 w-3" />
              Pause
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="flex-1"
              onClick={() => onVMAction?.(vm.id, "stop")}
            >
              <StopCircle className="mr-1 h-3 w-3" />
              Stop
            </Button>
          </>
        ) : (
          <Button
            size="sm"
            variant="outline"
            className="flex-1"
            onClick={() => onVMAction?.(vm.id, "start")}
          >
            <PlayCircle className="mr-1 h-3 w-3" />
            Start
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onVMAction?.(vm.id, "more")}
        >
          <MoreVertical className="h-4 w-4" />
        </Button>
      </div>
    </div>
    );
  };
  
  if (viewMode === "table") {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Virtual Machines</CardTitle>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setViewMode("grid")}
              >
                Grid View
              </Button>
              {selectedVMs.length > 0 && (
                <Badge variant="secondary">
                  {selectedVMs.length} selected
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveDataTable
            data={Array.isArray(vms) ? vms : []}
            columns={columns}
            mobileCard={renderMobileCard}
            onRowClick={(vm) => handleSelectVM(vm.id)}
          />
        </CardContent>
      </Card>
    );
  }
  
  // Grid view
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Virtual Machines</CardTitle>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setViewMode("table")}
            >
              Table View
            </Button>
            {selectedVMs.length > 0 && (
              <Badge variant="secondary">
                {selectedVMs.length} selected
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.isArray(vms) && vms.length > 0 ? vms.map((vm) => {
            if (!vm || !vm.id) return null;
            return (
              <div
                key={vm.id}
                className="p-4 rounded-lg border bg-white dark:bg-gray-800 dark:border-gray-700 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => handleSelectVM(vm.id)}
              >
                {renderMobileCard(vm)}
              </div>
            );
          }) : (
            <div className="col-span-full text-center py-8 text-muted-foreground">
              No virtual machines found
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}