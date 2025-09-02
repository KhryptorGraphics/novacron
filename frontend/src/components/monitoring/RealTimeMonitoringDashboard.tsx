'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Activity,
  AlertTriangle,
  Bell,
  BellOff,
  CheckCircle,
  Clock,
  Database,
  Globe,
  HardDrive,
  Info,
  Loader2,
  MemoryStick,
  Network,
  RefreshCw,
  Server,
  Settings,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
  Zap,
  AlertCircle,
  BarChart3,
  LineChart as LineChartIcon
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { useWebSocket } from '@/hooks/useWebSocket';

interface Metric {
  id: string;
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  status: 'healthy' | 'warning' | 'critical';
  threshold: {
    warning: number;
    critical: number;
  };
  history: Array<{ timestamp: number; value: number }>;
}

interface SystemAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  description: string;
  timestamp: string;
  source: string;
  acknowledged: boolean;
  actions?: Array<{ label: string; action: string }>;
}

interface HealthCheck {
  service: string;
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  lastCheck: string;
  uptime: number;
}

const RealTimeMonitoringDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([]);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [selectedMetricView, setSelectedMetricView] = useState('grid');
  const [alertFilter, setAlertFilter] = useState('all');
  const metricsRef = useRef<{ [key: string]: number[] }>({});

  // WebSocket connection for real-time data
  const { data: wsData, isConnected } = useWebSocket('/api/ws/monitoring');

  // Generate mock real-time data
  useEffect(() => {
    const generateMetrics = (): Metric[] => {
      const now = Date.now();
      return [
        {
          id: 'cpu-usage',
          name: 'CPU Usage',
          value: 45 + Math.random() * 20,
          unit: '%',
          trend: Math.random() > 0.5 ? 'up' : 'down',
          trendValue: Math.random() * 5,
          status: 'healthy',
          threshold: { warning: 70, critical: 90 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: 40 + Math.random() * 30
          }))
        },
        {
          id: 'memory-usage',
          name: 'Memory Usage',
          value: 65 + Math.random() * 15,
          unit: '%',
          trend: 'stable',
          trendValue: 0.5,
          status: 'warning',
          threshold: { warning: 60, critical: 85 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: 60 + Math.random() * 20
          }))
        },
        {
          id: 'disk-io',
          name: 'Disk I/O',
          value: 120 + Math.random() * 80,
          unit: 'MB/s',
          trend: 'up',
          trendValue: 12,
          status: 'healthy',
          threshold: { warning: 300, critical: 400 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: 100 + Math.random() * 100
          }))
        },
        {
          id: 'network-throughput',
          name: 'Network Throughput',
          value: 850 + Math.random() * 300,
          unit: 'Mbps',
          trend: 'up',
          trendValue: 8,
          status: 'healthy',
          threshold: { warning: 1500, critical: 1800 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: 800 + Math.random() * 400
          }))
        },
        {
          id: 'response-time',
          name: 'API Response Time',
          value: 45 + Math.random() * 30,
          unit: 'ms',
          trend: 'down',
          trendValue: -5,
          status: 'healthy',
          threshold: { warning: 100, critical: 200 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: 40 + Math.random() * 40
          }))
        },
        {
          id: 'error-rate',
          name: 'Error Rate',
          value: Math.random() * 2,
          unit: '%',
          trend: 'down',
          trendValue: -0.3,
          status: 'healthy',
          threshold: { warning: 2, critical: 5 },
          history: Array.from({ length: 20 }, (_, i) => ({
            timestamp: now - (19 - i) * 60000,
            value: Math.random() * 3
          }))
        }
      ];
    };

    const generateAlerts = (): SystemAlert[] => {
      const alertTemplates = [
        {
          severity: 'warning' as const,
          title: 'High Memory Usage',
          description: 'Memory usage on host-02 has exceeded 75%',
          source: 'host-02'
        },
        {
          severity: 'info' as const,
          title: 'Backup Completed',
          description: 'Daily backup completed successfully',
          source: 'backup-service'
        },
        {
          severity: 'error' as const,
          title: 'API Endpoint Down',
          description: 'Health check failed for /api/v2/users',
          source: 'api-gateway'
        }
      ];

      return alertTemplates.slice(0, Math.floor(Math.random() * 3) + 1).map((template, index) => ({
        ...template,
        id: `alert-${Date.now()}-${index}`,
        timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
        acknowledged: false,
        actions: template.severity === 'error' ? [
          { label: 'Restart Service', action: 'restart' },
          { label: 'View Logs', action: 'logs' }
        ] : undefined
      }));
    };

    const generateHealthChecks = (): HealthCheck[] => {
      return [
        {
          service: 'API Gateway',
          status: 'healthy',
          latency: 12 + Math.random() * 8,
          lastCheck: new Date().toISOString(),
          uptime: 99.99
        },
        {
          service: 'Database',
          status: 'healthy',
          latency: 5 + Math.random() * 5,
          lastCheck: new Date().toISOString(),
          uptime: 99.95
        },
        {
          service: 'Cache Service',
          status: Math.random() > 0.8 ? 'degraded' : 'healthy',
          latency: 2 + Math.random() * 3,
          lastCheck: new Date().toISOString(),
          uptime: 99.90
        },
        {
          service: 'Message Queue',
          status: 'healthy',
          latency: 8 + Math.random() * 7,
          lastCheck: new Date().toISOString(),
          uptime: 99.97
        },
        {
          service: 'Storage Backend',
          status: 'healthy',
          latency: 15 + Math.random() * 10,
          lastCheck: new Date().toISOString(),
          uptime: 99.99
        }
      ];
    };

    // Initial data
    setMetrics(generateMetrics());
    setAlerts(generateAlerts());
    setHealthChecks(generateHealthChecks());

    // Auto-refresh
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(() => {
        setMetrics(generateMetrics());
        if (Math.random() > 0.7) {
          setAlerts(prev => [...generateAlerts(), ...prev].slice(0, 10));
        }
        setHealthChecks(generateHealthChecks());
      }, refreshInterval);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'warning': case 'degraded': return 'text-yellow-500';
      case 'critical': case 'error': case 'down': return 'text-red-500';
      case 'info': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4" />;
      case 'warning': case 'degraded': return <AlertTriangle className="h-4 w-4" />;
      case 'critical': case 'error': case 'down': return <AlertCircle className="h-4 w-4" />;
      case 'info': return <Info className="h-4 w-4" />;
      default: return null;
    }
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-500" />;
      default: return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  const filteredAlerts = (alerts || []).filter(alert => {
    if (alertFilter === 'all') return true;
    if (alertFilter === 'unacknowledged') return !alert.acknowledged;
    return alert.severity === alertFilter;
  });

  // Prepare chart data
  const chartData = (metrics[0]?.history || []).map((point, index) => ({
    time: new Date(point.timestamp).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    }),
    cpu: metrics[0]?.history?.[index]?.value || 0,
    memory: metrics[1]?.history?.[index]?.value || 0,
    disk: (metrics[2]?.history?.[index]?.value || 0) / 5,
    network: (metrics[3]?.history?.[index]?.value || 0) / 20
  }));

  const radarData = [
    { metric: 'CPU', value: metrics[0]?.value || 0 },
    { metric: 'Memory', value: metrics[1]?.value || 0 },
    { metric: 'Disk', value: Math.min((metrics[2]?.value || 0) / 4, 100) },
    { metric: 'Network', value: Math.min((metrics[3]?.value || 0) / 18, 100) },
    { metric: 'Response', value: 100 - Math.min((metrics[4]?.value || 0), 100) },
    { metric: 'Errors', value: 100 - (metrics[5]?.value || 0) * 20 }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold">System Monitoring</h1>
          <Badge variant="outline" className={isConnected ? 'bg-green-50' : 'bg-red-50'}>
            {isConnected ? <Wifi className="h-3 w-3 mr-1" /> : <WifiOff className="h-3 w-3 mr-1" />}
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
        </div>
        
        <div className="flex items-center gap-2">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-[100px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="5m">5 min</SelectItem>
              <SelectItem value="15m">15 min</SelectItem>
              <SelectItem value="1h">1 hour</SelectItem>
              <SelectItem value="6h">6 hours</SelectItem>
              <SelectItem value="24h">24 hours</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="flex items-center gap-2">
            <Switch
              checked={autoRefresh}
              onCheckedChange={setAutoRefresh}
              id="auto-refresh"
            />
            <Label htmlFor="auto-refresh" className="text-sm">
              Auto-refresh
            </Label>
          </div>
          
          <Button variant="outline" size="icon" onClick={() => window.location.reload()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          
          <Button variant="outline" size="icon">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {(metrics || []).map((metric) => (
          <Card key={metric.id}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">{metric.name}</span>
                {getTrendIcon(metric.trend)}
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold">
                  {metric.value.toFixed(metric.unit === '%' ? 0 : 1)}
                </span>
                <span className="text-sm text-muted-foreground">{metric.unit}</span>
              </div>
              <div className={`text-xs mt-1 ${metric.trend === 'up' ? 'text-green-500' : metric.trend === 'down' ? 'text-red-500' : 'text-gray-500'}`}>
                {metric.trend === 'up' ? '+' : metric.trend === 'down' ? '-' : '±'}
                {Math.abs(metric.trendValue).toFixed(1)}%
              </div>
              <div className="mt-2 h-1 bg-gray-200 rounded overflow-hidden">
                <div 
                  className={`h-full ${
                    metric.value > metric.threshold.critical ? 'bg-red-500' :
                    metric.value > metric.threshold.warning ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(metric.value / metric.threshold.critical * 100, 100)}%` }}
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Metrics Charts */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Performance Metrics</CardTitle>
                  <CardDescription>Real-time system performance</CardDescription>
                </div>
                <Tabs value={selectedMetricView} onValueChange={setSelectedMetricView}>
                  <TabsList>
                    <TabsTrigger value="grid">Grid</TabsTrigger>
                    <TabsTrigger value="chart">Chart</TabsTrigger>
                    <TabsTrigger value="radar">Radar</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </CardHeader>
            <CardContent>
              {selectedMetricView === 'chart' && (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="cpu" stroke="#3b82f6" name="CPU %" strokeWidth={2} />
                    <Line type="monotone" dataKey="memory" stroke="#10b981" name="Memory %" strokeWidth={2} />
                    <Line type="monotone" dataKey="disk" stroke="#f59e0b" name="Disk (scaled)" strokeWidth={2} />
                    <Line type="monotone" dataKey="network" stroke="#8b5cf6" name="Network (scaled)" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
              
              {selectedMetricView === 'radar' && (
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar name="Current" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              )}
              
              {selectedMetricView === 'grid' && (
                <div className="grid grid-cols-2 gap-4">
                  {(metrics || []).slice(0, 4).map((metric) => (
                    <div key={metric.id} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">{metric.name}</span>
                        <span className={`text-sm ${getStatusColor(metric.status)}`}>
                          {getStatusIcon(metric.status)}
                        </span>
                      </div>
                      <ResponsiveContainer width="100%" height={80}>
                        <AreaChart data={metric.history.slice(-10)}>
                          <Area 
                            type="monotone" 
                            dataKey="value" 
                            stroke="#3b82f6" 
                            fill="#3b82f6" 
                            fillOpacity={0.2}
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Health Checks */}
          <Card>
            <CardHeader>
              <CardTitle>Service Health</CardTitle>
              <CardDescription>Real-time service status and uptime</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(healthChecks || []).map((check) => (
                  <div key={check.service} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-full ${
                        check.status === 'healthy' ? 'bg-green-100' :
                        check.status === 'degraded' ? 'bg-yellow-100' : 'bg-red-100'
                      }`}>
                        <Server className={`h-4 w-4 ${getStatusColor(check.status)}`} />
                      </div>
                      <div>
                        <p className="font-medium">{check.service}</p>
                        <p className="text-sm text-muted-foreground">
                          Uptime: {check.uptime}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{check.latency.toFixed(1)}ms</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(check.lastCheck).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts Panel */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Alerts</CardTitle>
                <Select value={alertFilter} onValueChange={setAlertFilter}>
                  <SelectTrigger className="w-[140px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="error">Errors</SelectItem>
                    <SelectItem value="warning">Warnings</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                    <SelectItem value="unacknowledged">Unread</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px] pr-4">
                <div className="space-y-3">
                  {filteredAlerts.length > 0 ? (
                    filteredAlerts.map((alert) => (
                      <Alert key={alert.id} className={`${
                        alert.severity === 'critical' || alert.severity === 'error' ? 'border-red-200' :
                        alert.severity === 'warning' ? 'border-yellow-200' : ''
                      }`}>
                        <div className="flex items-start gap-3">
                          <div className={`mt-0.5 ${getStatusColor(alert.severity)}`}>
                            {getStatusIcon(alert.severity)}
                          </div>
                          <div className="flex-1">
                            <AlertTitle className="text-sm font-medium">
                              {alert.title}
                            </AlertTitle>
                            <AlertDescription className="mt-1 text-xs">
                              {alert.description}
                            </AlertDescription>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="outline" className="text-xs">
                                {alert.source}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {new Date(alert.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            {alert.actions && (
                              <div className="flex gap-2 mt-2">
                                {(alert.actions || []).map((action) => (
                                  <Button
                                    key={action.action}
                                    size="sm"
                                    variant="outline"
                                    className="h-6 text-xs"
                                  >
                                    {action.label}
                                  </Button>
                                ))}
                              </div>
                            )}
                          </div>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6"
                            onClick={() => {
                              setAlerts(prev => (prev || []).map(a => 
                                a.id === alert.id ? { ...a, acknowledged: true } : a
                              ));
                            }}
                          >
                            {alert.acknowledged ? <BellOff className="h-3 w-3" /> : <Bell className="h-3 w-3" />}
                          </Button>
                        </div>
                      </Alert>
                    ))
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      No alerts to display
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Refresh Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Refresh Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Refresh Interval</Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[refreshInterval / 1000]}
                    onValueChange={(value) => setRefreshInterval(value[0] * 1000)}
                    min={1}
                    max={60}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-sm font-medium w-12">{refreshInterval / 1000}s</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="notifications">Push Notifications</Label>
                <Switch id="notifications" />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="sound">Alert Sounds</Label>
                <Switch id="sound" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default RealTimeMonitoringDashboard;