"use client";

import { useEffect, useState } from "react";
import { PieChart, LineChart } from "@/components/dashboard/charts";
import { SystemStatus } from "@/components/dashboard/system-status";
import { NodeList } from "@/components/dashboard/node-list";
import { VMList } from "@/components/dashboard/vm-list";
import { MetricsCard } from "@/components/dashboard/metrics-card";
import { useToast } from "@/components/ui/use-toast";
import { useHealth, useVMs, useWebSocket } from "@/hooks/useAPI";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Wifi, WifiOff, Server, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";

// Data types
interface Node {
  id: string;
  name: string;
  role: string;
  state: string;
  address: string;
  port: number;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  vm_count: number;
  joined_at: string;
}

interface VM {
  id: string;
  name: string;
  state: string;
  pid: number | null;
  cpu_usage: number;
  memory_usage: number;
  created_at: string;
  ip_address: string | null;
}

interface DashboardData {
  nodes: Node[];
  vms: VM[];
  system_status: {
    total_nodes: number;
    total_vms: number;
    total_cpu_usage: number;
    total_memory_usage: number;
    total_disk_usage: number;
  };
  metrics_history: {
    timestamp: string;
    cpu_usage: number;
    memory_usage: number;
  }[];
}

export default function DashboardPage() {
  const [selectedTab, setSelectedTab] = useState("overview");
  const { toast } = useToast();
  
  // Use real API hooks
  const { health, loading: healthLoading, error: healthError, refetch: refetchHealth } = useHealth();
  const { vms, loading: vmsLoading, error: vmsError, refetch: refetchVMs } = useVMs();
  const { connected: wsConnected, lastMessage } = useWebSocket();

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (lastMessage) {
      console.log('Received WebSocket message:', lastMessage);
      refetchVMs();
    }
  }, [lastMessage, refetchVMs]);

  // Show errors as toasts
  useEffect(() => {
    if (healthError) {
      toast({
        title: "Health Check Failed",
        description: healthError,
        variant: "destructive",
      });
    }
  }, [healthError, toast]);

  useEffect(() => {
    if (vmsError) {
      toast({
        title: "Failed to Load VMs",
        description: vmsError,
        variant: "destructive",
      });
    }
  }, [vmsError, toast]);

  // Create data structure for components
  const data: DashboardData | null = health && vms ? {
    nodes: [
      {
        id: "node-001",
        name: "novacron-server",
        role: "master",
        state: health.status === "healthy" ? "running" : "error",
        address: "localhost",
        port: 8080,
        cpu_usage: 10.5,
        memory_usage: 2048000000,
        disk_usage: 10240000000,
        vm_count: vms.vms || 0,
        joined_at: health.timestamp,
      },
    ],
    vms: [], // VMs will be populated by VMList component directly
    system_status: {
      total_nodes: 1,
      total_vms: vms.vms || 0,
      total_cpu_usage: 10.5,
      total_memory_usage: 2048000000,
      total_disk_usage: 10240000000,
    },
    metrics_history: Array.from({ length: 24 }, (_, i) => ({
      timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
      cpu_usage: Math.random() * 50 + 10,
      memory_usage: Math.random() * 40 + 30,
    })),
  } : null;

  const loading = healthLoading || vmsLoading;

  const handleRefresh = () => {
    refetchHealth();
    refetchVMs();
    toast({
      title: "Refreshing Data",
      description: "Fetching latest information from server...",
    });
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="h-16 w-16 animate-spin rounded-full border-4 border-t-blue-500"></div>
          <p className="text-lg font-medium">Loading NovaCron Dashboard...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center">
          <Server className="h-16 w-16 text-gray-400" />
          <h2 className="text-2xl font-bold">Unable to Connect</h2>
          <p className="text-gray-600 max-w-md">
            Cannot connect to the NovaCron server. Please ensure the server is running on localhost:8080.
          </p>
          <Button onClick={handleRefresh} className="mt-4">
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry Connection
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      {/* Header */}
      <div className="flex items-center justify-between space-y-2">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-muted-foreground">
            Monitor and manage your virtual machines
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            {wsConnected ? (
              <Badge variant="default" className="bg-green-500">
                <Wifi className="mr-1 h-3 w-3" />
                Connected
              </Badge>
            ) : (
              <Badge variant="destructive">
                <WifiOff className="mr-1 h-3 w-3" />
                Disconnected
              </Badge>
            )}
          </div>
          
          {/* Server Status */}
          <div className="flex items-center gap-2">
            <Badge variant={health?.status === "healthy" ? "default" : "destructive"}>
              <Activity className="mr-1 h-3 w-3" />
              {health?.status === "healthy" ? "Healthy" : "Error"}
            </Badge>
          </div>
          
          {/* Refresh Button */}
          <Button onClick={handleRefresh} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Nodes"
          value={data.system_status.total_nodes.toString()}
          description="Active hypervisor nodes"
          icon="🖥️"
        />
        <MetricsCard
          title="Total VMs"
          value={data.system_status.total_vms.toString()}
          description="Running virtual machines"
          icon="📦"
        />
        <MetricsCard
          title="CPU Usage"
          value={`${data.system_status.total_cpu_usage.toFixed(1)}%`}
          description="Average across all nodes"
          icon="⚡"
        />
        <MetricsCard
          title="Memory Usage"
          value={`${(data.system_status.total_memory_usage / (1024 * 1024 * 1024)).toFixed(1)}GB`}
          description="Total memory allocated"
          icon="💾"
        />
      </div>

      {/* Tabs */}
      <div className="space-y-4">
        <div className="border-b">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: "overview", label: "Overview" },
              { id: "vms", label: "Virtual Machines" },
              { id: "nodes", label: "Nodes" },
              { id: "monitoring", label: "Monitoring" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id)}
                className={`whitespace-nowrap border-b-2 py-2 px-1 text-sm font-medium ${
                  selectedTab === tab.id
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="space-y-4">
          {selectedTab === "overview" && (
            <div className="grid gap-4 md:grid-cols-2">
              <SystemStatus data={data.system_status} />
              <div className="space-y-4">
                <LineChart data={data.metrics_history} />
                <PieChart data={data.system_status} />
              </div>
            </div>
          )}

          {selectedTab === "vms" && (
            <VMList />
          )}

          {selectedTab === "nodes" && (
            <NodeList nodes={data.nodes} />
          )}

          {selectedTab === "monitoring" && (
            <div className="grid gap-4 md:grid-cols-2">
              <LineChart data={data.metrics_history} />
              <div className="space-y-4">
                <SystemStatus data={data.system_status} />
                <PieChart data={data.system_status} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}