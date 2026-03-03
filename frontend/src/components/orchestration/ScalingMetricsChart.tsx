"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
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
  ComposedChart,
} from 'recharts';
import { TrendingUp, Zap, Activity, RefreshCw } from 'lucide-react';

interface ScalingEvent {
  timestamp: string;
  action: 'scale_up' | 'scale_down' | 'no_change';
  vmId: string;
  beforeCount: number;
  afterCount: number;
  reason: string;
  cpuUtilization: number;
  memoryUtilization: number;
  requestRate: number;
  responseTime: number;
}

interface ScalingMetrics {
  timestamp: string;
  totalVMs: number;
  cpuUtilization: number;
  memoryUtilization: number;
  requestRate: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  scalingEvents: number;
}

export function ScalingMetricsChart() {
  const [scalingData, setScalingData] = useState<ScalingMetrics[]>([]);
  const [recentEvents, setRecentEvents] = useState<ScalingEvent[]>([]);
  const [timeRange, setTimeRange] = useState('1h');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchScalingMetrics = async () => {
      try {
        const [metricsRes, eventsRes] = await Promise.all([
          fetch(`/api/orchestration/scaling/metrics?range=${timeRange}`),
          fetch('/api/orchestration/scaling/events?limit=20'),
        ]);

        if (metricsRes.ok) {
          const metrics = await metricsRes.json();
          setScalingData(metrics);
        }

        if (eventsRes.ok) {
          const events = await eventsRes.json();
          setRecentEvents(events);
        }
      } catch (err) {
        console.error('Failed to fetch scaling metrics:', err);
        // Generate mock data for demonstration
        generateMockData();
      } finally {
        setLoading(false);
      }
    };

    fetchScalingMetrics();
    
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchScalingMetrics, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const generateMockData = () => {
    const now = new Date();
    const interval = timeRange === '1h' ? 5 * 60 * 1000 : timeRange === '6h' ? 30 * 60 * 1000 : 60 * 60 * 1000;
    const points = timeRange === '1h' ? 12 : timeRange === '6h' ? 12 : 24;

    const metrics: ScalingMetrics[] = [];
    const events: ScalingEvent[] = [];

    for (let i = points - 1; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * interval);
      
      // Simulate scaling patterns
      const hour = timestamp.getHours();
      const baseLoad = 0.3 + 0.4 * Math.sin((hour - 6) * Math.PI / 12);
      const randomVariation = 0.1 * (Math.random() - 0.5);
      
      const cpuUtil = Math.max(0.1, Math.min(0.95, baseLoad + randomVariation));
      const memUtil = Math.max(0.1, Math.min(0.9, cpuUtil * 0.8 + randomVariation * 0.5));
      const requestRate = 100 + cpuUtil * 500 + Math.random() * 100;
      const responseTime = 50 + cpuUtil * 200 + Math.random() * 50;
      const throughput = requestRate * (1 - Math.min(0.1, cpuUtil * 0.1));
      const errorRate = Math.min(0.05, cpuUtil > 0.8 ? (cpuUtil - 0.8) * 0.25 : 0);
      
      // Determine VM count based on load
      let vmCount = Math.ceil(cpuUtil * 10) + 2;
      if (i < points - 1) {
        const prevCount = metrics[metrics.length - 1].totalVMs;
        // Smooth transitions
        vmCount = Math.max(1, Math.min(12, prevCount + (Math.random() > 0.7 ? (Math.random() > 0.5 ? 1 : -1) : 0)));
      }

      metrics.push({
        timestamp: timestamp.toISOString(),
        totalVMs: vmCount,
        cpuUtilization: cpuUtil * 100,
        memoryUtilization: memUtil * 100,
        requestRate,
        responseTime,
        throughput,
        errorRate: errorRate * 100,
        scalingEvents: Math.random() > 0.8 ? 1 : 0,
      });

      // Generate scaling events
      if (i < points - 2 && Math.random() > 0.7) {
        const prevCount = i === points - 1 ? vmCount : metrics[metrics.length - 2].totalVMs;
        if (vmCount !== prevCount) {
          events.push({
            timestamp: timestamp.toISOString(),
            action: vmCount > prevCount ? 'scale_up' : 'scale_down',
            vmId: `vm-${Math.random().toString(36).substr(2, 9)}`,
            beforeCount: prevCount,
            afterCount: vmCount,
            reason: vmCount > prevCount ? 
              `High ${cpuUtil > 0.8 ? 'CPU' : 'memory'} utilization detected` :
              'Low resource utilization, scaling down',
            cpuUtilization: cpuUtil * 100,
            memoryUtilization: memUtil * 100,
            requestRate,
            responseTime,
          });
        }
      }
    }

    setScalingData(metrics);
    setRecentEvents(events.slice(-10).reverse());
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'scale_up': return 'success';
      case 'scale_down': return 'warning';
      default: return 'secondary';
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'scale_up': return '↗';
      case 'scale_down': return '↘';
      default: return '→';
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3">
          <p className="font-medium">{new Date(label).toLocaleTimeString()}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(1)}{entry.name.includes('Rate') || entry.name.includes('Time') ? '' : '%'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <RefreshCw className="h-6 w-6 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Scaling Overview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Auto-Scaling Metrics</span>
            </CardTitle>
            <CardDescription>
              Real-time scaling decisions and resource utilization
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">1h</SelectItem>
                <SelectItem value="6h">6h</SelectItem>
                <SelectItem value="24h">24h</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {scalingData.length > 0 ? scalingData[scalingData.length - 1].totalVMs : 0}
              </div>
              <p className="text-sm text-muted-foreground">Active VMs</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {scalingData.length > 0 ? scalingData[scalingData.length - 1].cpuUtilization.toFixed(1) : 0}%
              </div>
              <p className="text-sm text-muted-foreground">CPU Utilization</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {scalingData.length > 0 ? scalingData[scalingData.length - 1].requestRate.toFixed(0) : 0}
              </div>
              <p className="text-sm text-muted-foreground">Requests/sec</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {recentEvents.filter(e => e.action !== 'no_change').length}
              </div>
              <p className="text-sm text-muted-foreground">Recent Scaling Events</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Resource Utilization Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Resource Utilization</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={scalingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    interval="preserveStartEnd"
                  />
                  <YAxis yAxisId="utilization" domain={[0, 100]} />
                  <YAxis yAxisId="vms" orientation="right" domain={[0, 'dataMax']} />
                  <Tooltip content={<CustomTooltip />} />
                  
                  <Area
                    yAxisId="utilization"
                    type="monotone"
                    dataKey="cpuUtilization"
                    stackId="1"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.6}
                    name="CPU %"
                  />
                  <Area
                    yAxisId="utilization"
                    type="monotone"
                    dataKey="memoryUtilization"
                    stackId="1"
                    stroke="#82ca9d"
                    fill="#82ca9d"
                    fillOpacity={0.6}
                    name="Memory %"
                  />
                  <Line
                    yAxisId="vms"
                    type="monotone"
                    dataKey="totalVMs"
                    stroke="#ff7c7c"
                    strokeWidth={3}
                    dot={{ r: 4 }}
                    name="VM Count"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Performance Metrics</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={scalingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp"
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    interval="preserveStartEnd"
                  />
                  <YAxis yAxisId="rate" />
                  <YAxis yAxisId="time" orientation="right" />
                  <Tooltip content={<CustomTooltip />} />
                  
                  <Bar
                    yAxisId="rate"
                    dataKey="requestRate"
                    fill="#8884d8"
                    fillOpacity={0.6}
                    name="Request Rate"
                  />
                  <Line
                    yAxisId="time"
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#ff7c7c"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Response Time (ms)"
                  />
                  <Line
                    yAxisId="rate"
                    type="monotone"
                    dataKey="errorRate"
                    stroke="#ffc658"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="Error Rate %"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Scaling Events */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Scaling Events</CardTitle>
          <CardDescription>
            Latest automatic scaling decisions and their triggers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentEvents.map((event, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <Badge variant={getActionColor(event.action)}>
                      {getActionIcon(event.action)} {event.action.replace('_', ' ')}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {new Date(event.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-sm font-medium">
                    {event.beforeCount} → {event.afterCount} VMs
                  </div>
                </div>
                
                <p className="text-sm mb-3">{event.reason}</p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div>
                    <span className="text-muted-foreground">CPU: </span>
                    <span className="font-medium">{event.cpuUtilization.toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Memory: </span>
                    <span className="font-medium">{event.memoryUtilization.toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Requests/s: </span>
                    <span className="font-medium">{event.requestRate.toFixed(0)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Response: </span>
                    <span className="font-medium">{event.responseTime.toFixed(0)}ms</span>
                  </div>
                </div>
              </div>
            ))}
            
            {recentEvents.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                No recent scaling events
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}