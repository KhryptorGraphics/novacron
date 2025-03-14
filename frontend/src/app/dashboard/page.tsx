"use client";

import { useEffect, useState } from "react";
import { PieChart, LineChart } from "@/components/dashboard/charts";
import { SystemStatus } from "@/components/dashboard/system-status";
import { NodeList } from "@/components/dashboard/node-list";
import { VMList } from "@/components/dashboard/vm-list";
import { MetricsCard } from "@/components/dashboard/metrics-card";
import { useToast } from "@/components/ui/use-toast";

// Mock data types
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
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<DashboardData | null>(null);
  const [selectedTab, setSelectedTab] = useState("overview");
  const { toast } = useToast();

  useEffect(() => {
    // Fetch dashboard data from API
    // In a real app, this would be a fetch call to the API
    setTimeout(() => {
      // Mock data
      setData({
        nodes: [
          {
            id: "node-001",
            name: "hypervisor-1",
            role: "master",
            state: "running",
            address: "192.168.1.100",
            port: 7700,
            cpu_usage: 10.5,
            memory_usage: 2048000000,
            disk_usage: 10240000000,
            vm_count: 1,
            joined_at: "2025-03-12T09:00:00Z",
          },
          {
            id: "node-002",
            name: "hypervisor-2",
            role: "worker",
            state: "running",
            address: "192.168.1.101",
            port: 7700,
            cpu_usage: 5.2,
            memory_usage: 1024000000,
            disk_usage: 5120000000,
            vm_count: 1,
            joined_at: "2025-03-12T09:05:00Z",
          },
        ],
        vms: [
          {
            id: "vm-001",
            name: "web-server",
            state: "running",
            pid: 12345,
            cpu_usage: 5.2,
            memory_usage: 128000000,
            created_at: "2025-03-12T10:00:00Z",
            ip_address: "10.0.0.2",
          },
          {
            id: "vm-002",
            name: "database",
            state: "stopped",
            pid: null,
            cpu_usage: 0.0,
            memory_usage: 0,
            created_at: "2025-03-12T11:00:00Z",
            ip_address: null,
          },
        ],
        system_status: {
          total_nodes: 2,
          total_vms: 2,
          total_cpu_usage: 15.7,
          total_memory_usage: 3072000000,
          total_disk_usage: 15360000000,
        },
        metrics_history: Array.from({ length: 24 }, (_, i) => ({
          timestamp: new Date(Date.now() - i * 3600000).toISOString(),
          cpu_usage: 10 + Math.random() * 20,
          memory_usage: 2048000000 + Math.random() * 1024000000,
        })).reverse(),
      });
      setLoading(false);
    }, 1500);
  }, []);

  const handleNodeAction = (nodeId: string, action: string) => {
    // In a real app, this would call the API
    toast({
      title: "Node Action",
      description: `${action} node ${nodeId}`,
    });
  };

  const handleVMAction = (vmId: string, action: string) => {
    // In a real app, this would call the API
    toast({
      title: "VM Action",
      description: `${action} VM ${vmId}`,
    });
  };

  if (loading || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <div className="h-16 w-16 animate-spin rounded-full border-4 border-t-blue-500"></div>
        <p className="text-lg font-medium mt-4">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">NovaCron Dashboard</h1>
          <p className="text-gray-500">Manage your distributed hypervisor infrastructure</p>
        </div>
        <div className="flex space-x-2 mt-4 lg:mt-0">
          <button 
            className={`px-4 py-2 rounded-md ${selectedTab === 'overview' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setSelectedTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`px-4 py-2 rounded-md ${selectedTab === 'nodes' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setSelectedTab('nodes')}
          >
            Nodes
          </button>
          <button 
            className={`px-4 py-2 rounded-md ${selectedTab === 'vms' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setSelectedTab('vms')}
          >
            VMs
          </button>
        </div>
      </div>

      {selectedTab === 'overview' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <MetricsCard 
              title="Total Nodes" 
              value={data.system_status.total_nodes.toString()} 
              icon="🖥️" 
              trend="stable"
            />
            <MetricsCard 
              title="Total VMs" 
              value={data.system_status.total_vms.toString()} 
              icon="🖧" 
              trend="up"
            />
            <MetricsCard 
              title="CPU Usage" 
              value={`${data.system_status.total_cpu_usage.toFixed(1)}%`} 
              icon="📊" 
              trend="up"
            />
            <MetricsCard 
              title="Memory Usage" 
              value={`${(data.system_status.total_memory_usage / 1024 / 1024 / 1024).toFixed(1)} GB`} 
              icon="📈" 
              trend="stable"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">CPU Usage (Last 24 Hours)</h2>
              <div className="h-64">
                <LineChart 
                  data={data.metrics_history.map(d => ({ 
                    x: new Date(d.timestamp).toLocaleTimeString(), 
                    y: d.cpu_usage 
                  }))} 
                  xLabel="Time" 
                  yLabel="CPU %" 
                />
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Memory Usage (Last 24 Hours)</h2>
              <div className="h-64">
                <LineChart 
                  data={data.metrics_history.map(d => ({ 
                    x: new Date(d.timestamp).toLocaleTimeString(), 
                    y: d.memory_usage / 1024 / 1024 / 1024 
                  }))} 
                  xLabel="Time" 
                  yLabel="Memory (GB)" 
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Resource Usage by Node</h2>
              <div className="h-64">
                <PieChart 
                  data={data.nodes.map(node => ({ 
                    name: node.name, 
                    value: node.cpu_usage 
                  }))} 
                  label="CPU Usage" 
                />
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">VM Status Distribution</h2>
              <div className="h-64">
                <PieChart 
                  data={[
                    { name: 'Running', value: data.vms.filter(vm => vm.state === 'running').length },
                    { name: 'Stopped', value: data.vms.filter(vm => vm.state === 'stopped').length },
                  ]} 
                  label="VM Status" 
                />
              </div>
            </div>
          </div>
        </>
      )}

      {selectedTab === 'nodes' && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Hypervisor Nodes</h2>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-md">Add Node</button>
          </div>
          <NodeList nodes={data.nodes} onAction={handleNodeAction} />
        </div>
      )}

      {selectedTab === 'vms' && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Virtual Machines</h2>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-md">Create VM</button>
          </div>
          <VMList vms={data.vms} onAction={handleVMAction} />
        </div>
      )}
    </div>
  );
}
