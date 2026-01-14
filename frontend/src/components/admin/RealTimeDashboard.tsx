"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useAdminRealTimeUpdates, getConnectionStatusInfo } from "@/lib/ws/useAdminWebSocket";
import { 
  Activity, 
  Users, 
  Server, 
  Shield, 
  Database,
  Network,
  AlertTriangle,
  CheckCircle,
  Wifi,
  WifiOff,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus
} from "lucide-react";
import { cn } from "@/lib/utils";
import { FadeIn } from "@/lib/animations";
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
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from "recharts";

interface RealTimeMetrics {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_in: number;
  network_out: number;
  active_connections: number;
  response_time: number;
}

// Mock real-time data generator
const generateRealtimeData = (): RealTimeMetrics => ({
  timestamp: new Date().toISOString(),
  cpu_usage: Math.floor(Math.random() * 30) + 40, // 40-70%
  memory_usage: Math.floor(Math.random() * 25) + 55, // 55-80%
  disk_usage: Math.floor(Math.random() * 15) + 60, // 60-75%
  network_in: Math.random() * 5 + 1, // 1-6 MB/s
  network_out: Math.random() * 3 + 0.5, // 0.5-3.5 MB/s
  active_connections: Math.floor(Math.random() * 100) + 150, // 150-250
  response_time: Math.floor(Math.random() * 50) + 25 // 25-75ms
});

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00'];

export const RealTimeDashboard = () => {
  const { isConnected, metrics, alerts, connectionState, error } = useAdminRealTimeUpdates();
  const [realtimeData, setRealtimeData] = useState<RealTimeMetrics[]>([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  
  // Simulate real-time data updates
  useEffect(() => {
    if (isAutoRefresh) {
      const interval = setInterval(() => {
        setRealtimeData(prev => {
          const newData = generateRealtimeData();
          const updated = [...prev, newData];
          // Keep only last 20 data points
          return updated.slice(-20);
        });
      }, 2000); // Update every 2 seconds
      
      return () => clearInterval(interval);
    }
  }, [isAutoRefresh]);
  
  // Initialize with some data
  useEffect(() => {
    const initialData = Array.from({ length: 10 }, () => generateRealtimeData());
    setRealtimeData(initialData);
  }, []);
  
  const connectionInfo = getConnectionStatusInfo(connectionState);
  const latestMetrics = realtimeData[realtimeData.length - 1];
  
  const systemHealthScore = latestMetrics ? 
    Math.round((100 - latestMetrics.cpu_usage + (100 - latestMetrics.memory_usage) + (100 - latestMetrics.disk_usage)) / 3) : 
    0;
  
  const getHealthColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };
  
  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="h-3 w-3 text-red-500" />;
    if (current < previous) return <TrendingDown className="h-3 w-3 text-green-500" />;
    return <Minus className="h-3 w-3 text-gray-500" />;
  };
  
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };
  
  return (
    <div className="space-y-6">
      {/* Connection Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Wifi className="h-5 w-5 text-green-600" />
            ) : (
              <WifiOff className="h-5 w-5 text-red-600" />
            )}
            <span className={cn("text-sm font-medium", connectionInfo.color)}>
              {connectionInfo.status}
            </span>
          </div>
          
          {error && (
            <Badge variant="destructive" className="text-xs">
              {error}
            </Badge>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            className={cn(
              "flex items-center gap-2",
              isAutoRefresh && "bg-green-50 border-green-200 text-green-700"
            )}
          >
            <RefreshCw className={cn("h-4 w-4", isAutoRefresh && "animate-spin")} />
            {isAutoRefresh ? "Auto-Refresh On" : "Auto-Refresh Off"}
          </Button>
        </div>
      </div>
      
      {/* Real-time Metrics Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">System Health</p>
                  <p className={cn("text-xl font-bold", getHealthColor(systemHealthScore))}>
                    {systemHealthScore}%
                  </p>
                  {realtimeData.length > 1 && (
                    <div className="flex items-center gap-1 mt-1">
                      {getTrendIcon(systemHealthScore, 75)}
                      <span className="text-xs text-gray-600">vs previous</span>
                    </div>
                  )}
                </div>
                <Activity className="h-6 w-6 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">CPU Usage</p>
                  <p className="text-xl font-bold">{latestMetrics?.cpu_usage || 0}%</p>
                  {realtimeData.length > 1 && (
                    <div className="flex items-center gap-1 mt-1">
                      {getTrendIcon(latestMetrics?.cpu_usage || 0, realtimeData[realtimeData.length - 2]?.cpu_usage || 0)}
                      <span className="text-xs text-gray-600">
                        {Math.abs((latestMetrics?.cpu_usage || 0) - (realtimeData[realtimeData.length - 2]?.cpu_usage || 0)).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
                <Server className="h-6 w-6 text-purple-600" />
              </div>
              <div className="mt-2">
                <Progress value={latestMetrics?.cpu_usage || 0} className="h-1" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Memory Usage</p>
                  <p className="text-xl font-bold">{latestMetrics?.memory_usage || 0}%</p>
                  {realtimeData.length > 1 && (
                    <div className="flex items-center gap-1 mt-1">
                      {getTrendIcon(latestMetrics?.memory_usage || 0, realtimeData[realtimeData.length - 2]?.memory_usage || 0)}
                      <span className="text-xs text-gray-600">
                        {Math.abs((latestMetrics?.memory_usage || 0) - (realtimeData[realtimeData.length - 2]?.memory_usage || 0)).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
                <Database className="h-6 w-6 text-green-600" />
              </div>
              <div className="mt-2">
                <Progress value={latestMetrics?.memory_usage || 0} className="h-1" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Active Connections</p>
                  <p className="text-xl font-bold">{latestMetrics?.active_connections || 0}</p>
                  {realtimeData.length > 1 && (
                    <div className="flex items-center gap-1 mt-1">
                      {getTrendIcon(latestMetrics?.active_connections || 0, realtimeData[realtimeData.length - 2]?.active_connections || 0)}
                      <span className="text-xs text-gray-600">
                        {Math.abs((latestMetrics?.active_connections || 0) - (realtimeData[realtimeData.length - 2]?.active_connections || 0))}
                      </span>
                    </div>
                  )}
                </div>
                <Users className="h-6 w-6 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
      
      {/* Real-time Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.5}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4" />
                System Resources (Live)
              </CardTitle>
              <CardDescription>Real-time CPU, Memory, and Disk usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={realtimeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis 
                      dataKey="timestamp" 
                      tick={{ fontSize: 10 }}
                      tickFormatter={formatTime}
                      interval="preserveStartEnd"
                    />
                    <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
                    <Tooltip 
                      labelFormatter={(value) => formatTime(value as string)}
                      formatter={(value: number, name: string) => [
                        `${value.toFixed(1)}%`,
                        name.replace('_', ' ').toUpperCase()
                      ]}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="cpu_usage" 
                      stroke="#8b5cf6" 
                      strokeWidth={2}
                      dot={false}
                      name="CPU"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory_usage" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={false}
                      name="Memory"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="disk_usage" 
                      stroke="#f59e0b" 
                      strokeWidth={2}
                      dot={false}
                      name="Disk"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Network className="h-4 w-4" />
                Network Traffic (Live)
              </CardTitle>
              <CardDescription>Real-time network throughput monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={realtimeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis 
                      dataKey="timestamp" 
                      tick={{ fontSize: 10 }}
                      tickFormatter={formatTime}
                      interval="preserveStartEnd"
                    />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip 
                      labelFormatter={(value) => formatTime(value as string)}
                      formatter={(value: number, name: string) => [
                        `${value.toFixed(2)} MB/s`,
                        name.replace('_', ' ').replace('network ', 'Network ').toUpperCase()
                      ]}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="network_in" 
                      stackId="1" 
                      stroke="#3b82f6" 
                      fill="#3b82f6"
                      fillOpacity={0.6}
                      name="Inbound"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="network_out" 
                      stackId="1" 
                      stroke="#ef4444" 
                      fill="#ef4444"
                      fillOpacity={0.6}
                      name="Outbound"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
      
      {/* Real-time Alerts */}
      <FadeIn delay={0.7}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="h-4 w-4" />
              Live System Alerts
            </CardTitle>
            <CardDescription>Real-time security and system alerts</CardDescription>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-green-600 mb-2">All Systems Normal</h3>
                  <p className="text-gray-600 dark:text-gray-400">No active alerts at this time</p>
                </div>
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {alerts.slice(0, 5).map((alert, index) => (
                  <div 
                    key={alert.id || index} 
                    className={cn(
                      "flex items-start gap-3 p-3 rounded-lg border",
                      alert.severity === 'critical' && "bg-red-50 border-red-200 dark:bg-red-950/50 dark:border-red-800",
                      alert.severity === 'high' && "bg-orange-50 border-orange-200 dark:bg-orange-950/50 dark:border-orange-800",
                      alert.severity === 'medium' && "bg-yellow-50 border-yellow-200 dark:bg-yellow-950/50 dark:border-yellow-800",
                      alert.severity === 'low' && "bg-blue-50 border-blue-200 dark:bg-blue-950/50 dark:border-blue-800"
                    )}
                  >
                    <AlertTriangle className={cn(
                      "h-4 w-4 mt-0.5",
                      alert.severity === 'critical' && "text-red-600",
                      alert.severity === 'high' && "text-orange-600",
                      alert.severity === 'medium' && "text-yellow-600",
                      alert.severity === 'low' && "text-blue-600"
                    )} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="font-medium text-sm">{alert.title || 'System Alert'}</p>
                        <Badge 
                          variant={
                            alert.severity === 'critical' ? 'destructive' : 
                            alert.severity === 'high' ? 'destructive' :
                            alert.severity === 'medium' ? 'secondary' : 
                            'outline'
                          }
                          className="text-xs"
                        >
                          {alert.severity}
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                        {alert.description || 'No description available'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {alert.timestamp ? formatTime(alert.timestamp) : 'Just now'}
                      </p>
                    </div>
                  </div>
                ))}
                
                {alerts.length > 5 && (
                  <div className="text-center pt-2">
                    <Button variant="ghost" size="sm">
                      View All {alerts.length} Alerts
                    </Button>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>
      
      {/* Performance Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <FadeIn delay={0.8}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Response Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mb-2">
                {latestMetrics?.response_time || 0}ms
              </div>
              <div className="h-16">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={realtimeData.slice(-10)}>
                    <Line 
                      type="monotone" 
                      dataKey="response_time" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.9}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Resource Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'CPU', value: latestMetrics?.cpu_usage || 0 },
                        { name: 'Memory', value: latestMetrics?.memory_usage || 0 },
                        { name: 'Disk', value: latestMetrics?.disk_usage || 0 },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={45}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {[
                        { name: 'CPU', value: latestMetrics?.cpu_usage || 0 },
                        { name: 'Memory', value: latestMetrics?.memory_usage || 0 },
                        { name: 'Disk', value: latestMetrics?.disk_usage || 0 },
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={1.0}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">System Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm">WebSocket</span>
                <Badge variant={isConnected ? "secondary" : "destructive"}>
                  {isConnected ? "Connected" : "Disconnected"}
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm">Auto-refresh</span>
                <Badge variant={isAutoRefresh ? "secondary" : "outline"}>
                  {isAutoRefresh ? "Active" : "Paused"}
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm">Data points</span>
                <Badge variant="outline">
                  {realtimeData.length}
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm">Last update</span>
                <span className="text-xs text-gray-600">
                  {latestMetrics ? formatTime(latestMetrics.timestamp) : 'Never'}
                </span>
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
    </div>
  );
};