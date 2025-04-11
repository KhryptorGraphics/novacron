import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { format } from 'date-fns';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { 
  AlertCircle, 
  AlertTriangle, 
  Clock, 
  HardDrive, 
  Info, 
  Layers, 
  RefreshCw, 
  Server, 
  Cpu, 
  MemoryStick,
  Network,
  Database,
  Activity
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

// Types
export interface Metric {
  name: string;
  value: number;
  timestamp: string;
  tags: Record<string, string>;
}

export interface Alert {
  id: string;
  name: string;
  description: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  status: 'firing' | 'resolved' | 'acknowledged';
  startTime: string;
  endTime?: string;
  labels: Record<string, string>;
  value: number;
  resource: string;
}

export interface VMMetrics {
  vmId: string;
  name: string;
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkRx: number;
  networkTx: number;
  iops: number;
  status: 'running' | 'stopped' | 'error' | 'unknown';
}

interface TimeRangeOption {
  label: string;
  value: string;
  seconds: number;
}

// Constants
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080/api';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws';

const TIME_RANGES: TimeRangeOption[] = [
  { label: 'Last 15 minutes', value: '15m', seconds: 15 * 60 },
  { label: 'Last hour', value: '1h', seconds: 60 * 60 },
  { label: 'Last 6 hours', value: '6h', seconds: 6 * 60 * 60 },
  { label: 'Last 24 hours', value: '24h', seconds: 24 * 60 * 60 },
  { label: 'Last 7 days', value: '7d', seconds: 7 * 24 * 60 * 60 },
];

// Helper functions
const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

const formatPercentage = (value: number): string => {
  return `${Math.round(value)}%`;
};

const getSeverityColor = (severity: Alert['severity']): string => {
  switch (severity) {
    case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    case 'error': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20 dark:text-orange-400';
    case 'warning': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/20 dark:text-amber-400';
    case 'info': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

const getSeverityIcon = (severity: Alert['severity']) => {
  switch (severity) {
    case 'critical': return <AlertCircle className="h-4 w-4" />;
    case 'error': return <AlertCircle className="h-4 w-4" />;
    case 'warning': return <AlertTriangle className="h-4 w-4" />;
    case 'info': return <Info className="h-4 w-4" />;
    default: return <Info className="h-4 w-4" />;
  }
};

const getStatusColor = (status: Alert['status']): string => {
  switch (status) {
    case 'firing': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    case 'acknowledged': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
    case 'resolved': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

const getVmStatusColor = (status: VMMetrics['status']): string => {
  switch (status) {
    case 'running': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
    case 'stopped': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/20 dark:text-amber-400';
    case 'error': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

// Components
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  change?: number;
  sparklineData?: number[];
}> = ({ title, value, icon, change, sparklineData }) => {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <div className="h-4 w-4 text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && (
          <p className={`text-xs ${change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {change > 0 ? '+' : ''}{change}% from last period
          </p>
        )}
        {sparklineData && (
          <div className="h-10 mt-2">
            <Line
              data={{
                labels: sparklineData.map((_, i) => ''),
                datasets: [
                  {
                    data: sparklineData,
                    borderColor: 'rgba(59, 130, 246, 0.8)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false,
                  },
                  tooltip: {
                    enabled: false,
                  },
                },
                scales: {
                  x: {
                    display: false,
                  },
                  y: {
                    display: false,
                    min: Math.min(...sparklineData) * 0.8,
                    max: Math.max(...sparklineData) * 1.2,
                  },
                },
              }}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

const AlertItem: React.FC<{ alert: Alert; onAcknowledge: (id: string) => void }> = ({ 
  alert, 
  onAcknowledge 
}) => {
  return (
    <div className={`p-4 rounded-lg mb-3 ${getSeverityColor(alert.severity)}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0 mt-0.5">
          {getSeverityIcon(alert.severity)}
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium">{alert.name}</h3>
          <div className="mt-1 text-sm">
            <p>{alert.description}</p>
          </div>
          <div className="mt-2 flex justify-between items-center">
            <div className="flex space-x-2 text-xs">
              <Badge variant="outline" className={getStatusColor(alert.status)}>
                {alert.status}
              </Badge>
              <span className="text-xs text-muted-foreground">
                <Clock className="h-3 w-3 inline mr-1" />
                {format(new Date(alert.startTime), 'MMM d, HH:mm:ss')}
              </span>
              <span className="text-xs">{alert.resource}</span>
            </div>
            {alert.status === 'firing' && (
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => onAcknowledge(alert.id)}
              >
                Acknowledge
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const VMMetricsTable: React.FC<{ vms: VMMetrics[] }> = ({ vms }) => {
  return (
    <Table>
      <TableCaption>List of monitored virtual machines</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>CPU</TableHead>
          <TableHead>Memory</TableHead>
          <TableHead>Disk</TableHead>
          <TableHead>Network</TableHead>
          <TableHead>IOPS</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {vms.map((vm) => (
          <TableRow key={vm.vmId}>
            <TableCell className="font-medium">{vm.name}</TableCell>
            <TableCell>
              <Badge variant="outline" className={getVmStatusColor(vm.status)}>
                {vm.status}
              </Badge>
            </TableCell>
            <TableCell>
              <div className="flex flex-col">
                <span className="text-sm">{formatPercentage(vm.cpuUsage)}</span>
                <Progress value={vm.cpuUsage} className="h-1 w-24" />
              </div>
            </TableCell>
            <TableCell>
              <div className="flex flex-col">
                <span className="text-sm">{formatPercentage(vm.memoryUsage)}</span>
                <Progress value={vm.memoryUsage} className="h-1 w-24" />
              </div>
            </TableCell>
            <TableCell>
              <div className="flex flex-col">
                <span className="text-sm">{formatPercentage(vm.diskUsage)}</span>
                <Progress value={vm.diskUsage} className="h-1 w-24" />
              </div>
            </TableCell>
            <TableCell>
              <div className="text-sm">
                <span className="block">↑ {formatBytes(vm.networkTx)}/s</span>
                <span className="block">↓ {formatBytes(vm.networkRx)}/s</span>
              </div>
            </TableCell>
            <TableCell>{vm.iops} IOPS</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

// Main Dashboard Component
const MonitoringDashboard: React.FC = () => {
  const { toast } = useToast();
  const [timeRange, setTimeRange] = useState<string>(TIME_RANGES[1].value);
  const [activeTab, setActiveTab] = useState<string>('overview');
  
  // API Queries
  const { 
    data: systemMetrics,
    isLoading: isLoadingMetrics,
    refetch: refetchMetrics,
  } = useQuery({
    queryKey: ['systemMetrics', timeRange],
    queryFn: async () => {
      const rangeSeconds = TIME_RANGES.find(r => r.value === timeRange)?.seconds || 3600;
      const response = await fetch(`${API_URL}/monitoring/metrics?timeRange=${rangeSeconds}`);
      if (!response.ok) {
        throw new Error('Failed to fetch system metrics');
      }
      return response.json();
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { 
    data: alerts,
    isLoading: isLoadingAlerts,
    refetch: refetchAlerts,
  } = useQuery({
    queryKey: ['alerts'],
    queryFn: async () => {
      const response = await fetch(`${API_URL}/monitoring/alerts`);
      if (!response.ok) {
        throw new Error('Failed to fetch alerts');
      }
      return response.json();
    },
    refetchInterval: 15000, // Refetch every 15 seconds
  });

  const { 
    data: vms,
    isLoading: isLoadingVMs,
    refetch: refetchVMs,
  } = useQuery({
    queryKey: ['vms'],
    queryFn: async () => {
      const response = await fetch(`${API_URL}/monitoring/vms`);
      if (!response.ok) {
        throw new Error('Failed to fetch VM metrics');
      }
      return response.json();
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // WebSocket for real-time updates
  const { lastMessage } = useWebSocket(`${WS_URL}/monitoring`, {
    onOpen: () => {
      console.log('WebSocket connected');
    },
    onError: (event) => {
      console.error('WebSocket error:', event);
      toast({
        title: 'WebSocket Error',
        description: 'Failed to connect to real-time updates',
        variant: 'destructive',
      });
    },
    shouldReconnect: () => true,
  });

  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage !== null) {
      const data = JSON.parse(lastMessage.data);
      
      // Handle different message types
      switch (data.type) {
        case 'metric':
          // Update metrics in real-time
          refetchMetrics();
          break;
        case 'alert':
          // Show toast for new alerts and refetch alerts list
          if (data.alert && data.alert.status === 'firing') {
            toast({
              title: `${data.alert.severity.toUpperCase()}: ${data.alert.name}`,
              description: data.alert.description,
              variant: 'destructive',
            });
          }
          refetchAlerts();
          break;
        case 'vm':
          // Update VM metrics
          refetchVMs();
          break;
      }
    }
  }, [lastMessage, refetchMetrics, refetchAlerts, refetchVMs, toast]);

  // Handle alert acknowledgment
  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`${API_URL}/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to acknowledge alert');
      }
      
      refetchAlerts();
      toast({
        title: 'Alert Acknowledged',
        description: 'The alert has been acknowledged',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'An unknown error occurred',
        variant: 'destructive',
      });
    }
  };

  // Prepare chart data
  const cpuData = systemMetrics?.cpuTimeseriesData || Array(24).fill(0).map(() => Math.random() * 100);
  const memoryData = systemMetrics?.memoryTimeseriesData || Array(24).fill(0).map(() => Math.random() * 100);
  const diskData = systemMetrics?.diskTimeseriesData || Array(24).fill(0).map(() => Math.random() * 100);
  const networkData = systemMetrics?.networkTimeseriesData || Array(24).fill(0).map(() => Math.random() * 100);
  
  const timeLabels = systemMetrics?.timeLabels || 
    Array(24).fill(0).map((_, i) => format(new Date(Date.now() - (23 - i) * 3600 * 1000), 'HH:mm'));

  const chartData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'CPU Usage (%)',
        data: cpuData,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4,
      },
      {
        label: 'Memory Usage (%)',
        data: memoryData,
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        tension: 0.4,
      },
      {
        label: 'Disk Usage (%)',
        data: diskData,
        borderColor: 'rgba(255, 159, 64, 1)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.4,
      },
      {
        label: 'Network (Mbps)',
        data: networkData,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  // Alert statistics
  const alertStats = React.useMemo(() => {
    if (!alerts) return { critical: 0, error: 0, warning: 0, info: 0 };
    
    return {
      critical: alerts.filter((a: Alert) => a.severity === 'critical' && a.status === 'firing').length,
      error: alerts.filter((a: Alert) => a.severity === 'error' && a.status === 'firing').length,
      warning: alerts.filter((a: Alert) => a.severity === 'warning' && a.status === 'firing').length,
      info: alerts.filter((a: Alert) => a.severity === 'info' && a.status === 'firing').length,
    };
  }, [alerts]);

  // VM statistics
  const vmStats = React.useMemo(() => {
    if (!vms) return { running: 0, stopped: 0, error: 0, total: 0, avgCpu: 0, avgMemory: 0, avgDisk: 0 };
    
    const running = vms.filter((vm: VMMetrics) => vm.status === 'running').length;
    const stopped = vms.filter((vm: VMMetrics) => vm.status === 'stopped').length;
    const error = vms.filter((vm: VMMetrics) => vm.status === 'error').length;
    const total = vms.length;
    
    const avgCpu = vms.reduce((acc: number, vm: VMMetrics) => acc + vm.cpuUsage, 0) / (total || 1);
    const avgMemory = vms.reduce((acc: number, vm: VMMetrics) => acc + vm.memoryUsage, 0) / (total || 1);
    const avgDisk = vms.reduce((acc: number, vm: VMMetrics) => acc + vm.diskUsage, 0) / (total || 1);
    
    return { running, stopped, error, total, avgCpu, avgMemory, avgDisk };
  }, [vms]);

  // Handling refresh
  const handleRefresh = () => {
    refetchMetrics();
    refetchAlerts();
    refetchVMs();
    toast({
      title: 'Dashboard Refreshed',
      description: 'The monitoring data has been refreshed',
    });
  };

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">Monitoring Dashboard</h1>
        <div className="flex items-center space-x-4">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map((range) => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="vms">Virtual Machines</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="System CPU Usage"
              value={formatPercentage(systemMetrics?.currentCpuUsage || 45)}
              icon={<Cpu />}
              change={systemMetrics?.cpuChangePercentage || 5.2}
              sparklineData={cpuData.slice(-10)}
            />
            <MetricCard
              title="Memory Usage"
              value={formatPercentage(systemMetrics?.currentMemoryUsage || 72)}
              icon={<MemoryStick />}
              change={systemMetrics?.memoryChangePercentage || -2.1}
              sparklineData={memoryData.slice(-10)}
            />
            <MetricCard
              title="Disk Usage"
              value={formatPercentage(systemMetrics?.currentDiskUsage || 58)}
              icon={<HardDrive />}
              change={systemMetrics?.diskChangePercentage || 1.8}
              sparklineData={diskData.slice(-10)}
            />
            <MetricCard
              title="Network Usage"
              value={`${Math.round(systemMetrics?.currentNetworkUsage || 125)} Mbps`}
              icon={<Network />}
              change={systemMetrics?.networkChangePercentage || 12.5}
              sparklineData={networkData.slice(-10)}
            />
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card className="col-span-2 row-span-2">
              <CardHeader>
                <CardTitle>System Metrics Over Time</CardTitle>
                <CardDescription>
                  Resource usage trends for the selected time period
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Line data={chartData} options={chartOptions} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Virtual Machines</CardTitle>
                <CardDescription>Status of managed VMs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center pb-4">
                  <div className="text-3xl font-bold">{vmStats.total}</div>
                  <p className="text-xs text-muted-foreground">Total VMs</p>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-green-100 p-2 rounded-md dark:bg-green-900/20">
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">{vmStats.running}</div>
                    <p className="text-xs text-green-600 dark:text-green-400">Running</p>
                  </div>
                  <div className="bg-amber-100 p-2 rounded-md dark:bg-amber-900/20">
                    <div className="text-xl font-bold text-amber-600 dark:text-amber-400">{vmStats.stopped}</div>
                    <p className="text-xs text-amber-600 dark:text-amber-400">Stopped</p>
                  </div>
                  <div className="bg-red-100 p-2 rounded-md dark:bg-red-900/20">
                    <div className="text-xl font-bold text-red-600 dark:text-red-400">{vmStats.error}</div>
                    <p className="text-xs text-red-600 dark:text-red-400">Error</p>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" size="sm" className="w-full" onClick={() => setActiveTab('vms')}>
                  View Details
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Active Alerts</CardTitle>
                <CardDescription>Current system alerts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center pb-4">
                  <div className="text-3xl font-bold">
                    {alertStats.critical + alertStats.error + alertStats.warning + alertStats.info}
                  </div>
                  <p className="text-xs text-muted-foreground">Total Alerts</p>
                </div>
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="bg-red-100 p-2 rounded-md dark:bg-red-900/20">
                    <div className="text-xl font-bold text-red-600 dark:text-red-400">
                      {alertStats.critical}
                    </div>
                    <p className="text-xs text-red-600 dark:text-red-400">Critical</p>
                  </div>
                  <div className="bg-orange-100 p-2 rounded-md dark:bg-orange-900/20">
                    <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
                      {alertStats.error}
                    </div>
                    <p className="text-xs text-orange-600 dark:text-orange-400">Error</p>
                  </div>
                  <div className="bg-amber-100 p-2 rounded-md dark:bg-amber-900/20">
                    <div className="text-xl font-bold text-amber-600 dark:text-amber-400">
                      {alertStats.warning}
                    </div>
                    <p className="text-xs text-amber-600 dark:text-amber-400">Warning</p>
                  </div>
                  <div className="bg-blue-100 p-2 rounded-md dark:bg-blue-900/20">
                    <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                      {alertStats.info}
                    </div>
                    <p className="text-xs text-blue-600 dark:text-blue-400">Info</p>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" size="sm" className="w-full" onClick={() => setActiveTab('alerts')}>
                  View Details
                </Button>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        {/* Virtual Machines Tab */}
        <TabsContent value="vms" className="space-y-6">
          <div className="grid gap-4 grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle>VM Summary</CardTitle>
                <CardDescription>Overview of managed virtual machines</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center">
                  <div className="text-center">
                    <div className="text-2xl font-bold">{vmStats.total}</div>
                    <p className="text-xs text-muted-foreground">Total VMs</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">{vmStats.running}</div>
                    <p className="text-xs text-green-600 dark:text-green-400">Running</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{vmStats.stopped}</div>
                    <p className="text-xs text-amber-600 dark:text-amber-400">Stopped</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Average Resource Usage</CardTitle>
                <CardDescription>Across all running VMs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">CPU</span>
                      <span className="text-sm">{formatPercentage(vmStats.avgCpu)}</span>
                    </div>
                    <Progress value={vmStats.avgCpu} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Memory</span>
                      <span className="text-sm">{formatPercentage(vmStats.avgMemory)}</span>
                    </div>
                    <Progress value={vmStats.avgMemory} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Disk</span>
                      <span className="text-sm">{formatPercentage(vmStats.avgDisk)}</span>
                    </div>
                    <Progress value={vmStats.avgDisk} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Actions</CardTitle>
                <CardDescription>VM management operations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" size="sm" className="w-full justify-start">
                  <Server className="mr-2 h-4 w-4" />
                  Provision New VM
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start">
                  <Activity className="mr-2 h-4 w-4" />
                  System Diagnostics
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start">
                  <Database className="mr-2 h-4 w-4" />
                  Storage Management
                </Button>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Virtual Machines</CardTitle>
              <CardDescription>All monitored virtual machines in the system</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingVMs ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">Loading VM metrics...</p>
                </div>
              ) : vms && vms.length > 0 ? (
                <VMMetricsTable vms={vms} />
              ) : (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No virtual machines found.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <div className="grid gap-4 grid-cols-4">
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Alert Status</CardTitle>
                <CardDescription>Active alerts by severity</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm flex items-center">
                      <span className="h-3 w-3 rounded-full bg-red-500 inline-block mr-2"></span>
                      Critical
                    </span>
                    <Badge variant="outline" className="bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400">
                      {alertStats.critical}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm flex items-center">
                      <span className="h-3 w-3 rounded-full bg-orange-500 inline-block mr-2"></span>
                      Error
                    </span>
                    <Badge variant="outline" className="bg-orange-100 text-orange-600 dark:bg-orange-900/20 dark:text-orange-400">
                      {alertStats.error}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm flex items-center">
                      <span className="h-3 w-3 rounded-full bg-amber-500 inline-block mr-2"></span>
                      Warning
                    </span>
                    <Badge variant="outline" className="bg-amber-100 text-amber-600 dark:bg-amber-900/20 dark:text-amber-400">
                      {alertStats.warning}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm flex items-center">
                      <span className="h-3 w-3 rounded-full bg-blue-500 inline-block mr-2"></span>
                      Info
                    </span>
                    <Badge variant="outline" className="bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400">
                      {alertStats.info}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Alert Timeline</CardTitle>
                <CardDescription>Alert activity over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-56">
                  <Bar 
                    data={{
                      labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
                      datasets: [
                        {
                          label: 'Critical',
                          data: [2, 1, 3, 0, 2, 1, alertStats.critical],
                          backgroundColor: 'rgba(220, 38, 38, 0.8)',
                        },
                        {
                          label: 'Error',
                          data: [3, 2, 4, 1, 3, 2, alertStats.error],
                          backgroundColor: 'rgba(234, 88, 12, 0.8)',
                        },
                        {
                          label: 'Warning',
                          data: [5, 4, 6, 3, 5, 4, alertStats.warning],
                          backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        },
                        {
                          label: 'Info',
                          data: [7, 6, 8, 5, 7, 6, alertStats.info],
                          backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        x: {
                          stacked: true,
                        },
                        y: {
                          stacked: true,
                          beginAtZero: true,
                        },
                      },
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Current Alerts</CardTitle>
              <CardDescription>Active and recent alerts</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingAlerts ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">Loading alerts...</p>
                </div>
              ) : alerts && alerts.length > 0 ? (
                <div className="space-y-2">
                  {alerts.map((alert: Alert) => (
                    <AlertItem 
                      key={alert.id} 
                      alert={alert} 
                      onAcknowledge={handleAcknowledgeAlert} 
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No alerts found.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid gap-4 grid-cols-1">
            <Card>
              <CardHeader>
                <CardTitle>Resource Usage Analysis</CardTitle>
                <CardDescription>Pattern analysis and trend detection</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-medium mb-2">CPU Trend Analysis</h4>
                    <div className="h-60">
                      <Line
                        data={{
                          labels: timeLabels,
                          datasets: [
                            {
                              label: 'CPU Usage (%)',
                              data: cpuData,
                              borderColor: 'rgba(75, 192, 192, 1)',
                              backgroundColor: 'rgba(75, 192, 192, 0.2)',
                              tension: 0.4,
                              fill: true,
                            },
                          ],
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            y: {
                              beginAtZero: true,
                              max: 100,
                            },
                          },
                        }}
                      />
                    </div>
                    <div className="mt-2">
                      <p className="text-sm text-muted-foreground">
                        {systemMetrics?.cpuAnalysis || "CPU usage shows standard workday pattern with peaks during business hours."}
                      </p>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Memory Allocation</h4>
                    <div className="h-60">
                      <Doughnut
                        data={{
                          labels: ['In Use', 'Available', 'Reserved', 'Cached'],
                          datasets: [
                            {
                              data: [
                                systemMetrics?.memoryInUse || 65,
                                systemMetrics?.memoryAvailable || 15,
                                systemMetrics?.memoryReserved || 10,
                                systemMetrics?.memoryCached || 10,
                              ],
                              backgroundColor: [
                                'rgba(153, 102, 255, 0.8)',
                                'rgba(75, 192, 192, 0.8)',
                                'rgba(255, 159, 64, 0.8)',
                                'rgba(54, 162, 235, 0.8)',
                              ],
                              borderWidth: 1,
                            },
                          ],
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                        }}
                      />
                    </div>
                    <div className="mt-2">
                      <p className="text-sm text-muted-foreground">
                        {systemMetrics?.memoryAnalysis || "Memory allocation is healthy with sufficient available memory for peak operations."}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Predictive Insights</CardTitle>
                <CardDescription>AI-powered system predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-1 flex items-center">
                        <Server className="h-4 w-4 mr-2" />
                        Resource Forecasting
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        Based on current trends, system resources will be sufficient for the next 30 days. 
                        Consider adding additional storage capacity within 45 days as growth trends indicate 
                        80% usage by that time.
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-1 flex items-center">
                        <AlertTriangle className="h-4 w-4 mr-2" />
                        Anomaly Detection
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        Hypervisor node 2 is showing slight performance degradation patterns compared to 
                        historical baseline. This may indicate early hardware issues or resource contention.
                        Scheduled diagnostic recommended.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-1 flex items-center">
                        <Activity className="h-4 w-4 mr-2" />
                        Performance Optimization
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        VM resource rebalancing across nodes could improve overall system performance by 
                        an estimated 12-15%. Database-heavy VMs on node 3 would benefit from memory 
                        reallocation from underutilized compute-optimized VMs.
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-1 flex items-center">
                        <Layers className="h-4 w-4 mr-2" />
                        Workload Patterns
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        System has identified recurring peak usage patterns every Tuesday and Thursday 
                        between 2-4 PM UTC. Consider scheduling non-essential maintenance outside these 
                        windows and potentially allocating additional resources during peaks.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;
