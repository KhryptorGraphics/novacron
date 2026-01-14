"use client";

import { useState, useEffect } from "react";

// Disable static generation for this page
export const dynamic = 'force-dynamic';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown,
  Activity, 
  Clock,
  Zap,
  Database,
  Server,
  Users,
  Calendar,
  Download,
  RefreshCw,
  Target,
  AlertCircle,
  CheckCircle,
  PieChart,
  LineChart,
  BarChart
} from "lucide-react";
import { Line, Bar, Doughnut, Pie } from "react-chartjs-2";
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
  Filler
} from 'chart.js';

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
  ArcElement,
  Filler
);

// Generate mock data
const generateTimeSeriesData = (points: number, min: number, max: number) => {
  return Array.from({ length: points }, () => Math.floor(Math.random() * (max - min) + min));
};

const timeLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
const weekLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState("24h");
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate data refresh
    await new Promise(resolve => setTimeout(resolve, 1500));
    setRefreshing(false);
  };

  // Performance Metrics Data
  const performanceData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'CPU Usage (%)',
        data: generateTimeSeriesData(24, 20, 80),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
      },
      {
        label: 'Memory Usage (%)',
        data: generateTimeSeriesData(24, 30, 90),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true,
      }
    ]
  };

  // Resource Utilization Data
  const resourceUtilizationData = {
    labels: ['CPU', 'Memory', 'Storage', 'Network'],
    datasets: [{
      data: [68, 75, 45, 32],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
      ],
      borderColor: [
        'rgb(59, 130, 246)',
        'rgb(16, 185, 129)',
        'rgb(245, 158, 11)',
        'rgb(239, 68, 68)',
      ],
      borderWidth: 2,
    }]
  };

  // VM Distribution Data
  const vmDistributionData = {
    labels: ['Running', 'Stopped', 'Error', 'Starting'],
    datasets: [{
      data: [12, 3, 1, 2],
      backgroundColor: [
        'rgba(34, 197, 94, 0.8)',
        'rgba(156, 163, 175, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(59, 130, 246, 0.8)',
      ],
      borderColor: [
        'rgb(34, 197, 94)',
        'rgb(156, 163, 175)',
        'rgb(239, 68, 68)',
        'rgb(59, 130, 246)',
      ],
      borderWidth: 2,
    }]
  };

  // Network Traffic Data
  const networkTrafficData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'Inbound (Mbps)',
        data: generateTimeSeriesData(24, 50, 400),
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
      },
      {
        label: 'Outbound (Mbps)',
        data: generateTimeSeriesData(24, 30, 300),
        backgroundColor: 'rgba(16, 185, 129, 0.8)',
      }
    ]
  };

  // Storage Usage Trend
  const storageUsageData = {
    labels: monthLabels,
    datasets: [
      {
        label: 'Storage Usage (GB)',
        data: generateTimeSeriesData(12, 1000, 5000),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        fill: true,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
    },
  };

  // Mock analytics metrics
  const analyticsMetrics = {
    totalVMs: 18,
    vmGrowth: 12.5,
    avgCPUUsage: 64,
    cpuTrend: -5.2,
    avgMemoryUsage: 72,
    memoryTrend: 8.1,
    storageUsage: 3200,
    storageTrend: 15.6,
    networkThroughput: 245,
    networkTrend: -3.4,
    uptime: 99.8,
    uptimeTrend: 0.2,
    incidents: 3,
    incidentTrend: -40,
  };

  const getTrendColor = (trend: number) => {
    if (trend > 0) return "text-green-600 dark:text-green-400";
    if (trend < 0) return "text-red-600 dark:text-red-400";
    return "text-gray-600 dark:text-gray-400";
  };

  const getTrendIcon = (trend: number) => {
    if (trend > 0) return <TrendingUp className="h-4 w-4" />;
    if (trend < 0) return <TrendingDown className="h-4 w-4" />;
    return <Activity className="h-4 w-4" />;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h1>
          <p className="text-muted-foreground">Performance insights and trend analysis</p>
        </div>
        <div className="flex gap-2 items-center">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button 
            variant="outline" 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total VMs</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsMetrics.totalVMs}</div>
            <div className={`flex items-center text-xs ${getTrendColor(analyticsMetrics.vmGrowth)}`}>
              {getTrendIcon(analyticsMetrics.vmGrowth)}
              <span className="ml-1">{Math.abs(analyticsMetrics.vmGrowth)}% from last month</span>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg CPU Usage</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsMetrics.avgCPUUsage}%</div>
            <div className={`flex items-center text-xs ${getTrendColor(analyticsMetrics.cpuTrend)}`}>
              {getTrendIcon(analyticsMetrics.cpuTrend)}
              <span className="ml-1">{Math.abs(analyticsMetrics.cpuTrend)}% from last hour</span>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Uptime</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsMetrics.uptime}%</div>
            <div className={`flex items-center text-xs ${getTrendColor(analyticsMetrics.uptimeTrend)}`}>
              {getTrendIcon(analyticsMetrics.uptimeTrend)}
              <span className="ml-1">{Math.abs(analyticsMetrics.uptimeTrend)}% improvement</span>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Incidents</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsMetrics.incidents}</div>
            <div className={`flex items-center text-xs ${getTrendColor(analyticsMetrics.incidentTrend)}`}>
              {getTrendIcon(analyticsMetrics.incidentTrend)}
              <span className="ml-1">{Math.abs(analyticsMetrics.incidentTrend)}% fewer</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="performance" className="w-full">
        <TabsList>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>
        
        <TabsContent value="performance" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Performance Over Time */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <LineChart className="h-5 w-5" />
                      Performance Metrics
                    </CardTitle>
                    <CardDescription>CPU and memory usage over time</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Line data={performanceData} options={chartOptions} />
                </div>
              </CardContent>
            </Card>

            {/* Resource Utilization */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="h-5 w-5" />
                  Resource Utilization
                </CardTitle>
                <CardDescription>Current system resource usage</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Doughnut data={resourceUtilizationData} options={doughnutOptions} />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Performance Insights */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Insights</CardTitle>
              <CardDescription>AI-powered analysis and recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="h-4 w-4 text-green-500" />
                    <span className="font-medium text-green-700 dark:text-green-400">Optimization Opportunity</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    VM vm-003 is consistently underutilized (avg 15% CPU). Consider downsizing 
                    to reduce resource waste and optimize costs.
                  </p>
                </div>
                
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertCircle className="h-4 w-4 text-yellow-500" />
                    <span className="font-medium text-yellow-700 dark:text-yellow-400">Performance Warning</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Network latency on node-02 has increased by 25% over the past week. 
                    Check network configuration and consider load balancing.
                  </p>
                </div>
                
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="h-4 w-4 text-blue-500" />
                    <span className="font-medium text-blue-700 dark:text-blue-400">Best Practice</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Storage I/O patterns show excellent distribution. Current RAID configuration 
                    is optimal for the workload.
                  </p>
                </div>
                
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                    <span className="font-medium text-green-700 dark:text-green-400">Trend Analysis</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Overall system performance has improved 12% this month due to recent 
                    optimizations and proactive maintenance.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="resources" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* VM Status Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  VM Status Distribution
                </CardTitle>
                <CardDescription>Current state of all virtual machines</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Pie data={vmDistributionData} options={doughnutOptions} />
                </div>
              </CardContent>
            </Card>

            {/* Storage Usage Trend */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Storage Growth
                </CardTitle>
                <CardDescription>Storage usage over the past year</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <Line data={storageUsageData} options={chartOptions} />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Resource Summary */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Memory Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">{analyticsMetrics.avgMemoryUsage}%</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Used: 46.2 GB</span>
                    <span>Total: 64 GB</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                    <div className="bg-blue-600 h-2 rounded-full" style={{width: `${analyticsMetrics.avgMemoryUsage}%`}}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Storage Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">{Math.round((analyticsMetrics.storageUsage / 10000) * 100)}%</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Used: {analyticsMetrics.storageUsage} GB</span>
                    <span>Total: 10 TB</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{width: `${Math.round((analyticsMetrics.storageUsage / 10000) * 100)}%`}}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Network Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">{analyticsMetrics.networkThroughput} Mbps</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Current</span>
                    <span>Max: 1 Gbps</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: `${(analyticsMetrics.networkThroughput / 1000) * 100}%`}}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="network" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart className="h-5 w-5" />
                Network Traffic Analysis
              </CardTitle>
              <CardDescription>Inbound and outbound network traffic patterns</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <Bar data={networkTrafficData} options={chartOptions} />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Top Network Consumers</CardTitle>
                <CardDescription>VMs with highest network usage</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { name: "database-primary", usage: 156, percentage: 45 },
                    { name: "web-server-01", usage: 89, percentage: 25 },
                    { name: "backup-server", usage: 67, percentage: 19 },
                    { name: "dev-environment", usage: 34, percentage: 11 }
                  ].filter(vm => vm && vm.name).map((vm, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{vm.name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                          <div className="bg-blue-600 h-2 rounded-full" style={{width: `${vm.percentage}%`}}></div>
                        </div>
                        <span className="text-sm w-16 text-right">{vm.usage} Mbps</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Network Health</CardTitle>
                <CardDescription>Key network performance indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Average Latency</span>
                    <span className="text-sm font-medium text-green-600">2.3ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Packet Loss</span>
                    <span className="text-sm font-medium text-green-600">0.01%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Jitter</span>
                    <span className="text-sm font-medium text-green-600">0.5ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Throughput</span>
                    <span className="text-sm font-medium text-green-600">847 Mbps</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Error Rate</span>
                    <span className="text-sm font-medium text-green-600">0.002%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="trends" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Usage Trends</CardTitle>
                <CardDescription>Monthly resource usage patterns</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">CPU Usage Trend</span>
                      <span className="text-sm text-red-500">↑ 8.2%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div className="bg-red-500 h-2 rounded-full" style={{width: '68%'}}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Memory Usage Trend</span>
                      <span className="text-sm text-green-500">↓ 3.1%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div className="bg-green-500 h-2 rounded-full" style={{width: '72%'}}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Storage Growth</span>
                      <span className="text-sm text-yellow-500">↑ 15.6%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div className="bg-yellow-500 h-2 rounded-full" style={{width: '32%'}}></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">Network Usage</span>
                      <span className="text-sm text-blue-500">↑ 5.4%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div className="bg-blue-500 h-2 rounded-full" style={{width: '25%'}}></div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Forecast</CardTitle>
                <CardDescription>Predicted resource needs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Storage Capacity</span>
                      <span className="text-sm text-yellow-600">Warning</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Based on current growth rate, storage will reach 80% capacity in 4 months.
                    </p>
                  </div>
                  
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">CPU Demand</span>
                      <span className="text-sm text-green-600">Normal</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      CPU usage is stable and well within capacity limits.
                    </p>
                  </div>
                  
                  <div className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Memory Usage</span>
                      <span className="text-sm text-green-600">Optimizing</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Memory usage is decreasing due to recent optimizations.
                    </p>
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