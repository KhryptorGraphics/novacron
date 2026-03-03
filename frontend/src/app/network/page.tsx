"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  Network, 
  Wifi, 
  Router, 
  Globe,
  Plus, 
  Search, 
  Settings,
  AlertCircle,
  CheckCircle,
  Activity,
  BarChart3,
  Zap,
  Shield,
  ArrowUp,
  ArrowDown,
  Clock
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Line } from "react-chartjs-2";

// Mock network data
const mockNetworks = [
  {
    id: "net-001",
    name: "Production Network",
    type: "bridged",
    vlan: 100,
    subnet: "192.168.1.0/24",
    gateway: "192.168.1.1",
    dns: ["8.8.8.8", "8.8.4.4"],
    status: "active",
    connectedVMs: 8,
    bandwidth: { total: 1000, used: 450 },
    latency: 2.1,
    packetLoss: 0.01,
    uptime: "99.9%"
  },
  {
    id: "net-002", 
    name: "DMZ Network",
    type: "isolated",
    vlan: 200,
    subnet: "10.0.1.0/24",
    gateway: "10.0.1.1",
    dns: ["10.0.1.10"],
    status: "active",
    connectedVMs: 3,
    bandwidth: { total: 1000, used: 120 },
    latency: 1.8,
    packetLoss: 0.02,
    uptime: "99.8%"
  },
  {
    id: "net-003",
    name: "Storage Network",
    type: "host-only",
    vlan: 300,
    subnet: "172.16.0.0/24", 
    gateway: "172.16.0.1",
    dns: ["172.16.0.10"],
    status: "active",
    connectedVMs: 5,
    bandwidth: { total: 10000, used: 3200 },
    latency: 0.5,
    packetLoss: 0.001,
    uptime: "99.99%"
  },
  {
    id: "net-004",
    name: "Management Network",
    type: "nat",
    vlan: 400,
    subnet: "192.168.100.0/24",
    gateway: "192.168.100.1",
    dns: ["192.168.100.10"],
    status: "warning",
    connectedVMs: 2,
    bandwidth: { total: 100, used: 45 },
    latency: 5.2,
    packetLoss: 0.1,
    uptime: "98.5%"
  }
];

const mockNetworkInterfaces = [
  {
    id: "eth0",
    name: "eth0",
    host: "node-01",
    mac: "00:16:3e:12:34:56",
    ip: "192.168.1.10",
    network: "net-001",
    speed: "1 Gbps",
    duplex: "full",
    status: "up",
    rxBytes: 1024000000,
    txBytes: 512000000,
    rxPackets: 850000,
    txPackets: 720000,
    rxErrors: 0,
    txErrors: 0
  },
  {
    id: "eth1", 
    name: "eth1",
    host: "node-01",
    mac: "00:16:3e:12:34:57",
    ip: "172.16.0.10",
    network: "net-003",
    speed: "10 Gbps",
    duplex: "full", 
    status: "up",
    rxBytes: 5120000000,
    txBytes: 3840000000,
    rxPackets: 2100000,
    txPackets: 1800000,
    rxErrors: 1,
    txErrors: 0
  },
  {
    id: "eth2",
    name: "eth2", 
    host: "node-02",
    mac: "00:16:3e:12:34:58",
    ip: "192.168.1.11",
    network: "net-001", 
    speed: "1 Gbps",
    duplex: "full",
    status: "up",
    rxBytes: 768000000,
    txBytes: 384000000,
    rxPackets: 640000,
    txPackets: 560000,
    rxErrors: 2,
    txErrors: 1
  }
];

export default function NetworkPage() {
  const [networks, setNetworks] = useState(mockNetworks);
  const [interfaces, setInterfaces] = useState(mockNetworkInterfaces);
  const [searchQuery, setSearchQuery] = useState("");
  const [networkFilter, setNetworkFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  // Filter based on search and filters
  const filteredNetworks = networks.filter(network => {
    const matchesSearch = network.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         network.subnet.includes(searchQuery);
    const matchesStatus = statusFilter === "all" || network.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "warning": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "error": 
      case "down": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "up": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
      case "up": return <CheckCircle className="h-4 w-4" />;
      case "warning": return <AlertCircle className="h-4 w-4" />;
      case "error":
      case "down": return <AlertCircle className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getNetworkTypeIcon = (type: string) => {
    switch (type) {
      case "bridged": return <Globe className="h-4 w-4" />;
      case "isolated": return <Shield className="h-4 w-4" />;
      case "host-only": return <Network className="h-4 w-4" />;
      case "nat": return <Router className="h-4 w-4" />;
      default: return <Network className="h-4 w-4" />;
    }
  };

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const getBandwidthColor = (usage: number) => {
    if (usage < 70) return "text-green-600 dark:text-green-400";
    if (usage < 85) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const totalStats = {
    totalNetworks: networks.length,
    activeNetworks: networks.filter(n => n.status === "active").length,
    totalVMs: networks.reduce((acc, n) => acc + n.connectedVMs, 0),
    avgLatency: (networks.reduce((acc, n) => acc + n.latency, 0) / networks.length).toFixed(1)
  };

  // Mock bandwidth chart data
  const chartData = {
    labels: Array.from({length: 24}, (_, i) => `${i}:00`),
    datasets: [
      {
        label: 'Inbound (Mbps)',
        data: Array.from({length: 24}, () => Math.random() * 800 + 100),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Outbound (Mbps)', 
        data: Array.from({length: 24}, () => Math.random() * 600 + 50),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Bandwidth (Mbps)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time (24h)'
        }
      }
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Network Management</h1>
          <p className="text-muted-foreground">Monitor and configure your virtual networks</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Add Interface
          </Button>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Create Network
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Networks</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalStats.totalNetworks}</div>
            <p className="text-xs text-muted-foreground">
              {totalStats.activeNetworks} active
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Connected VMs</CardTitle>
            <Wifi className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalStats.totalVMs}</div>
            <p className="text-xs text-muted-foreground">
              Across all networks
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalStats.avgLatency}ms</div>
            <p className="text-xs text-muted-foreground">
              Cross-network average
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Throughput</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4.8 Gbps</div>
            <p className="text-xs text-muted-foreground">
              Current aggregate
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="networks" className="w-full">
        <TabsList>
          <TabsTrigger value="networks">Networks</TabsTrigger>
          <TabsTrigger value="interfaces">Interfaces</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>
        
        <TabsContent value="networks" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row gap-4 items-center">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search networks by name or subnet..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Networks Grid */}
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-2">
            {filteredNetworks.map((network) => (
              <Card key={network.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        {getNetworkTypeIcon(network.type)}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{network.name}</CardTitle>
                        <p className="text-sm text-muted-foreground">
                          {network.type} • VLAN {network.vlan}
                        </p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(network.status)}>
                      {getStatusIcon(network.status)}
                      <span className="ml-1 capitalize">{network.status}</span>
                    </Badge>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {/* Network Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="font-medium">Subnet</div>
                      <div className="text-muted-foreground">{network.subnet}</div>
                    </div>
                    <div>
                      <div className="font-medium">Gateway</div>
                      <div className="text-muted-foreground">{network.gateway}</div>
                    </div>
                    <div>
                      <div className="font-medium">Connected VMs</div>
                      <div className="text-muted-foreground">{network.connectedVMs}</div>
                    </div>
                    <div>
                      <div className="font-medium">Uptime</div>
                      <div className="text-muted-foreground">{network.uptime}</div>
                    </div>
                  </div>
                  
                  {/* Bandwidth Usage */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Bandwidth Usage</span>
                      <span className={getBandwidthColor((network.bandwidth.used / network.bandwidth.total) * 100)}>
                        {((network.bandwidth.used / network.bandwidth.total) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={(network.bandwidth.used / network.bandwidth.total) * 100} className="h-2" />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{network.bandwidth.used} Mbps used</span>
                      <span>{network.bandwidth.total} Mbps total</span>
                    </div>
                  </div>
                  
                  {/* Performance Metrics */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="font-medium">Latency</div>
                      <div className={`${network.latency > 5 ? 'text-red-600' : 'text-green-600'}`}>
                        {network.latency}ms
                      </div>
                    </div>
                    <div>
                      <div className="font-medium">Packet Loss</div>
                      <div className={`${network.packetLoss > 0.05 ? 'text-red-600' : 'text-green-600'}`}>
                        {network.packetLoss}%
                      </div>
                    </div>
                  </div>
                  
                  {/* Actions */}
                  <div className="flex items-center justify-between pt-2 border-t">
                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4 mr-1" />
                        Configure
                      </Button>
                      <Button variant="outline" size="sm">
                        <BarChart3 className="h-4 w-4 mr-1" />
                        Stats
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="interfaces" className="space-y-4">
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Interface</TableHead>
                  <TableHead>Host</TableHead>
                  <TableHead>Network</TableHead>
                  <TableHead>IP Address</TableHead>
                  <TableHead>Speed</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>RX/TX</TableHead>
                  <TableHead>Errors</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {interfaces.map((interface_) => (
                  <TableRow key={interface_.id}>
                    <TableCell className="font-medium">
                      <div className="flex flex-col">
                        <span>{interface_.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {interface_.mac}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>{interface_.host}</TableCell>
                    <TableCell>
                      {networks.find(n => n.id === interface_.network)?.name}
                    </TableCell>
                    <TableCell>{interface_.ip}</TableCell>
                    <TableCell>
                      <div className="flex flex-col">
                        <span>{interface_.speed}</span>
                        <span className="text-xs text-muted-foreground">
                          {interface_.duplex}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge className={getStatusColor(interface_.status)}>
                        {getStatusIcon(interface_.status)}
                        <span className="ml-1">{interface_.status}</span>
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="text-sm">
                        <div className="flex items-center gap-1">
                          <ArrowDown className="h-3 w-3 text-blue-500" />
                          <span>{formatBytes(interface_.rxBytes)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <ArrowUp className="h-3 w-3 text-green-500" />
                          <span>{formatBytes(interface_.txBytes)}</span>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="text-sm">
                        <div className={`${interface_.rxErrors > 0 ? 'text-red-600' : 'text-green-600'}`}>
                          RX: {interface_.rxErrors}
                        </div>
                        <div className={`${interface_.txErrors > 0 ? 'text-red-600' : 'text-green-600'}`}>
                          TX: {interface_.txErrors}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Button variant="ghost" size="sm">
                        <Settings className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
        
        <TabsContent value="monitoring" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Network Bandwidth Usage</CardTitle>
              <CardDescription>24-hour network traffic overview</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <Line data={chartData} options={chartOptions} />
              </div>
            </CardContent>
          </Card>
          
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Top Consumers</CardTitle>
                <CardDescription>VMs using most bandwidth</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { name: "database-primary", usage: 680, percentage: 35 },
                    { name: "web-server-01", usage: 450, percentage: 23 },
                    { name: "backup-server", usage: 380, percentage: 20 },
                    { name: "dev-environment", usage: 200, percentage: 10 }
                  ].map((vm, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{vm.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20">
                          <Progress value={vm.percentage} className="h-2" />
                        </div>
                        <span className="text-sm w-12 text-right">
                          {vm.usage} MB/s
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Network Health</CardTitle>
                <CardDescription>Current status indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Overall Health</span>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Excellent
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Average Latency</span>
                    <span className="text-sm text-green-600">{totalStats.avgLatency}ms</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Packet Loss</span>
                    <span className="text-sm text-green-600">0.02%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Network Utilization</span>
                    <span className="text-sm text-yellow-600">68%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Active Connections</span>
                    <span className="text-sm">{totalStats.totalVMs} VMs</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}