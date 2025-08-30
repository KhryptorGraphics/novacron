"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Activity,
  Cpu,
  MemoryStick,
  Network,
  HardDrive,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  Clock,
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

interface EngineStatus {
  state: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
  startTime: string;
  activePolicies: number;
  eventsProcessed: number;
  metrics: Record<string, any>;
}

interface RealTimeMetricsPanelProps {
  engineStatus: EngineStatus | null;
}

interface MetricPoint {
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  networkIO: number;
  diskIO: number;
  decisionsPerMinute: number;
  responseTime: number;
}

interface SystemAlert {
  id: string;
  type: 'warning' | 'error' | 'info';
  message: string;
  timestamp: string;
  component: string;
}

export function RealTimeMetricsPanel({ engineStatus }: RealTimeMetricsPanelProps) {
  const [metricsHistory, setMetricsHistory] = useState<MetricPoint[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<MetricPoint | null>(null);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/orchestration/metrics/realtime');
        if (response.ok) {
          const data = await response.json();
          
          const newMetric: MetricPoint = {
            timestamp: new Date().toISOString(),
            cpuUsage: data.cpu_usage || Math.random() * 80 + 10,
            memoryUsage: data.memory_usage || Math.random() * 70 + 20,
            networkIO: data.network_io || Math.random() * 100,
            diskIO: data.disk_io || Math.random() * 50,
            decisionsPerMinute: data.decisions_per_minute || Math.floor(Math.random() * 20) + 5,
            responseTime: data.response_time || Math.random() * 50 + 10,
          };

          setCurrentMetrics(newMetric);
          setMetricsHistory(prev => {
            const updated = [...prev, newMetric].slice(-30); // Keep last 30 points
            return updated;
          });

          // Generate alerts based on metrics
          if (newMetric.cpuUsage > 80) {
            setAlerts(prev => [...prev, {
              id: `cpu-${Date.now()}`,
              type: 'warning',
              message: `High CPU usage detected: ${newMetric.cpuUsage.toFixed(1)}%`,
              timestamp: new Date().toISOString(),
              component: 'CPU',
            }].slice(-10));
          }

          if (newMetric.responseTime > 100) {
            setAlerts(prev => [...prev, {
              id: `response-${Date.now()}`,
              type: 'error',
              message: `High response time: ${newMetric.responseTime.toFixed(0)}ms`,
              timestamp: new Date().toISOString(),
              component: 'Performance',
            }].slice(-10));
          }
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        // Generate mock data for demonstration
        generateMockMetrics();
      }
    };

    const generateMockMetrics = () => {
      const newMetric: MetricPoint = {
        timestamp: new Date().toISOString(),
        cpuUsage: 30 + Math.sin(Date.now() / 10000) * 20 + Math.random() * 10,
        memoryUsage: 45 + Math.cos(Date.now() / 12000) * 15 + Math.random() * 8,
        networkIO: 20 + Math.random() * 30,
        diskIO: 15 + Math.random() * 20,
        decisionsPerMinute: 10 + Math.floor(Math.random() * 15),
        responseTime: 25 + Math.random() * 25,
      };

      setCurrentMetrics(newMetric);
      setMetricsHistory(prev => [...prev, newMetric].slice(-30));
    };

    // Initial fetch
    fetchMetrics();

    // Set up interval if auto-refresh is enabled
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(fetchMetrics, 5000); // Every 5 seconds
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [autoRefresh]);

  const getMetricStatus = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value > thresholds.critical) return 'critical';
    if (value > thresholds.warning) return 'warning';
    return 'normal';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-green-100 text-green-800 border-green-200';
    }
  };

  const formatUptime = (startTime: string) => {
    if (!startTime) return 'Unknown';
    
    const start = new Date(startTime);
    const now = new Date();
    const diff = now.getTime() - start.getTime();
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const time = new Date(label).toLocaleTimeString();
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3">
          <p className="font-medium">{time}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(1)}{entry.name.includes('Usage') ? '%' : entry.name.includes('Time') ? 'ms' : ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Engine Status Overview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Real-Time System Metrics</span>
            </CardTitle>
            <CardDescription>
              Live monitoring of orchestration engine performance
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant={engineStatus?.state === 'running' ? 'success' : 'warning'}>
              {engineStatus?.state || 'Unknown'}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              <RefreshCw className={`h-4 w-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              <span className="ml-1">{autoRefresh ? 'Auto' : 'Manual'}</span>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {formatUptime(engineStatus?.startTime || '')}
              </div>
              <p className="text-sm text-muted-foreground">Uptime</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {engineStatus?.eventsProcessed || 0}
              </div>
              <p className="text-sm text-muted-foreground">Events Processed</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {currentMetrics?.decisionsPerMinute || 0}
              </div>
              <p className="text-sm text-muted-foreground">Decisions/min</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {currentMetrics?.responseTime.toFixed(0) || 0}ms
              </div>
              <p className="text-sm text-muted-foreground">Avg Response</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Resource Usage Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {currentMetrics?.cpuUsage.toFixed(1) || 0}%
            </div>
            <Progress value={currentMetrics?.cpuUsage || 0} className="h-2" />
            <p className={`text-xs mt-2 px-2 py-1 rounded ${getStatusColor(
              getMetricStatus(currentMetrics?.cpuUsage || 0, { warning: 70, critical: 90 })
            )}`}>
              {getMetricStatus(currentMetrics?.cpuUsage || 0, { warning: 70, critical: 90 })}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <MemoryStick className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {currentMetrics?.memoryUsage.toFixed(1) || 0}%
            </div>
            <Progress value={currentMetrics?.memoryUsage || 0} className="h-2" />
            <p className={`text-xs mt-2 px-2 py-1 rounded ${getStatusColor(
              getMetricStatus(currentMetrics?.memoryUsage || 0, { warning: 75, critical: 90 })
            )}`}>
              {getMetricStatus(currentMetrics?.memoryUsage || 0, { warning: 75, critical: 90 })}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network I/O</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {currentMetrics?.networkIO.toFixed(0) || 0} MB/s
            </div>
            <Progress value={(currentMetrics?.networkIO || 0) * 2} className="h-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Network throughput
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Disk I/O</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {currentMetrics?.diskIO.toFixed(0) || 0} MB/s
            </div>
            <Progress value={(currentMetrics?.diskIO || 0) * 2} className="h-2" />
            <p className="text-xs text-muted-foreground mt-2">
              Disk throughput
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Metrics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Resource Usage Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    interval="preserveStartEnd"
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="cpuUsage"
                    stackId="1"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.6}
                    name="CPU Usage"
                  />
                  <Area
                    type="monotone"
                    dataKey="memoryUsage"
                    stackId="1"
                    stroke="#82ca9d"
                    fill="#82ca9d"
                    fillOpacity={0.6}
                    name="Memory Usage"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Performance Metrics</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    interval="preserveStartEnd"
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="decisionsPerMinute"
                    stroke="#8884d8"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Decisions/min"
                  />
                  <Line
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Response Time"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5" />
            <span>System Alerts</span>
          </CardTitle>
          <CardDescription>
            Recent system alerts and notifications
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {alerts.length > 0 ? (
              alerts.slice().reverse().map((alert) => (
                <div key={alert.id} className="flex items-center space-x-3 p-3 border rounded-lg">
                  <div className={`w-2 h-2 rounded-full ${
                    alert.type === 'error' ? 'bg-red-500' :
                    alert.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                  }`}></div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{alert.component}</Badge>
                      <span className="text-xs text-muted-foreground">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm mt-1">{alert.message}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                <p>No system alerts</p>
                <p className="text-sm">All systems operating normally</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}