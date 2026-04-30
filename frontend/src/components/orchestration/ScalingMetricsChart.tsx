"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useScalingMetrics } from '@/lib/api/hooks/useOrchestration';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  Bar,
  ComposedChart,
} from 'recharts';
import { TrendingUp, Zap, Activity, RefreshCw, AlertTriangle } from 'lucide-react';

export function ScalingMetricsChart() {
  const [timeRange, setTimeRange] = useState('1h');
  const { scalingData, recentEvents, loading, error, refetch } = useScalingMetrics(timeRange);

  const getActionColor = (action: string) => {
    switch (action) {
      case 'scale_up': return 'default';
      case 'scale_down': return 'secondary';
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

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
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
            <Button variant="outline" size="sm" onClick={() => refetch()}>
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
