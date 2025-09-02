"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  useSystemMetrics, 
  usePerformanceReports, 
  useUsers,
  useAuditLogs 
} from "@/lib/api/hooks/useAdmin";
import { 
  BarChart3,
  TrendingUp,
  TrendingDown,
  Users,
  Activity,
  Server,
  Database,
  Network,
  Shield,
  Clock,
  Zap,
  Download,
  Calendar,
  AlertTriangle,
  CheckCircle,
  Eye
} from "lucide-react";
import { FadeIn } from "@/lib/animations";
import { cn } from "@/lib/utils";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from "recharts";

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00'];

// Mock data generators for comprehensive analytics
const generateMetricsData = () => {
  return Array.from({ length: 24 }, (_, i) => ({
    time: `${i.toString().padStart(2, '0')}:00`,
    cpu: Math.floor(Math.random() * 40) + 30,
    memory: Math.floor(Math.random() * 30) + 50,
    disk: Math.floor(Math.random() * 20) + 60,
    network_in: Math.floor(Math.random() * 100) + 50,
    network_out: Math.floor(Math.random() * 80) + 30,
    response_time: Math.floor(Math.random() * 100) + 50,
    active_connections: Math.floor(Math.random() * 200) + 100,
  }));
};

const generateUserActivityData = () => {
  return Array.from({ length: 7 }, (_, i) => ({
    day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i],
    active_users: Math.floor(Math.random() * 300) + 200,
    new_registrations: Math.floor(Math.random() * 20) + 5,
    login_sessions: Math.floor(Math.random() * 500) + 300,
  }));
};

const generateResourceUsageData = () => {
  return [
    { name: 'CPU', value: 65, color: '#8884d8' },
    { name: 'Memory', value: 78, color: '#82ca9d' },
    { name: 'Disk', value: 45, color: '#ffc658' },
    { name: 'Network', value: 32, color: '#ff7300' },
  ];
};

export default function AdminAnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d' | '30d'>('24h');
  const [reportType, setReportType] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  
  const { data: systemMetrics, isLoading: metricsLoading } = useSystemMetrics(timeRange);
  const { data: performanceReports, isLoading: reportsLoading } = usePerformanceReports(reportType);
  const { data: usersData } = useUsers({ pageSize: 1000 });
  const { data: auditData } = useAuditLogs({ pageSize: 100 });
  
  // Generate mock data for comprehensive visualization
  const metricsChartData = useMemo(() => generateMetricsData(), [timeRange]);
  const userActivityData = useMemo(() => generateUserActivityData(), []);
  const resourceUsageData = useMemo(() => generateResourceUsageData(), []);
  
  // Calculate key metrics
  const keyMetrics = useMemo(() => {
    const users = usersData?.users || [];
    const totalUsers = users.length;
    const activeUsers = users.filter(u => u.status === 'active').length;
    const pendingUsers = users.filter(u => u.status === 'pending').length;
    const twoFactorUsers = users.filter(u => u.two_factor_enabled).length;
    
    return {
      totalUsers,
      activeUsers,
      pendingUsers,
      twoFactorUsers,
      activeUserPercentage: totalUsers > 0 ? (activeUsers / totalUsers) * 100 : 0,
      twoFactorPercentage: totalUsers > 0 ? (twoFactorUsers / totalUsers) * 100 : 0,
    };
  }, [usersData]);
  
  const systemHealthScore = useMemo(() => {
    const avgCpu = metricsChartData.reduce((acc, item) => acc + item.cpu, 0) / metricsChartData.length;
    const avgMemory = metricsChartData.reduce((acc, item) => acc + item.memory, 0) / metricsChartData.length;
    const avgResponseTime = metricsChartData.reduce((acc, item) => acc + item.response_time, 0) / metricsChartData.length;
    
    // Simple health score calculation
    const cpuScore = Math.max(0, 100 - avgCpu);
    const memoryScore = Math.max(0, 100 - avgMemory);
    const responseScore = Math.max(0, 100 - (avgResponseTime / 10));
    
    return Math.round((cpuScore + memoryScore + responseScore) / 3);
  }, [metricsChartData]);
  
  const getHealthColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };
  
  const getHealthBadge = (score: number) => {
    if (score >= 80) return { variant: "secondary" as const, text: "Excellent", color: "bg-green-100 text-green-800" };
    if (score >= 60) return { variant: "secondary" as const, text: "Good", color: "bg-yellow-100 text-yellow-800" };
    return { variant: "destructive" as const, text: "Needs Attention", color: "bg-red-100 text-red-800" };
  };
  
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <BarChart3 className="h-8 w-8" />
            System Analytics
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Comprehensive analytics and performance insights
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={(value: any) => setTimeRange(value)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="6h">Last 6 Hours</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>
      
      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">System Health</p>
                  <p className={`text-2xl font-bold ${getHealthColor(systemHealthScore)}`}>
                    {systemHealthScore}%
                  </p>
                  <Badge className={getHealthBadge(systemHealthScore).color}>
                    {getHealthBadge(systemHealthScore).text}
                  </Badge>
                </div>
                <Activity className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Users</p>
                  <p className="text-2xl font-bold text-green-600">{keyMetrics.activeUsers}</p>
                  <div className="flex items-center gap-1 text-sm text-gray-600">
                    <TrendingUp className="h-3 w-3" />
                    {keyMetrics.activeUserPercentage.toFixed(1)}% of total
                  </div>
                </div>
                <Users className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Response Time</p>
                  <p className="text-2xl font-bold">
                    {Math.round(metricsChartData.reduce((acc, item) => acc + item.response_time, 0) / metricsChartData.length)}ms
                  </p>
                  <div className="flex items-center gap-1 text-sm text-green-600">
                    <TrendingDown className="h-3 w-3" />
                    12% improvement
                  </div>
                </div>
                <Zap className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Security Score</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {keyMetrics.twoFactorPercentage.toFixed(0)}%
                  </p>
                  <div className="text-sm text-gray-600">2FA adoption rate</div>
                </div>
                <Shield className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
      
      {/* Analytics Tabs */}
      <Tabs defaultValue="performance" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="users">User Analytics</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
        </TabsList>
        
        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    System Resources Over Time
                  </CardTitle>
                  <CardDescription>CPU, Memory, and Disk usage patterns</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={metricsChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          dataKey="cpu" 
                          stroke="#8884d8" 
                          strokeWidth={2}
                          name="CPU %"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="memory" 
                          stroke="#82ca9d" 
                          strokeWidth={2}
                          name="Memory %"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="disk" 
                          stroke="#ffc658" 
                          strokeWidth={2}
                          name="Disk %"
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
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5" />
                    Network Traffic & Response Times
                  </CardTitle>
                  <CardDescription>Network utilization and system responsiveness</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={metricsChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Area 
                          type="monotone" 
                          dataKey="network_in" 
                          stackId="1" 
                          stroke="#8884d8" 
                          fill="#8884d8"
                          name="Network In (MB/s)"
                        />
                        <Area 
                          type="monotone" 
                          dataKey="network_out" 
                          stackId="1" 
                          stroke="#82ca9d" 
                          fill="#82ca9d"
                          name="Network Out (MB/s)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
          
          <FadeIn delay={0.7}>
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics Summary</CardTitle>
                <CardDescription>Key performance indicators and trends</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Average CPU Usage</span>
                      <span className="text-sm">
                        {Math.round(metricsChartData.reduce((acc, item) => acc + item.cpu, 0) / metricsChartData.length)}%
                      </span>
                    </div>
                    <Progress value={Math.round(metricsChartData.reduce((acc, item) => acc + item.cpu, 0) / metricsChartData.length)} />
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Average Memory Usage</span>
                      <span className="text-sm">
                        {Math.round(metricsChartData.reduce((acc, item) => acc + item.memory, 0) / metricsChartData.length)}%
                      </span>
                    </div>
                    <Progress value={Math.round(metricsChartData.reduce((acc, item) => acc + item.memory, 0) / metricsChartData.length)} />
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Average Disk Usage</span>
                      <span className="text-sm">
                        {Math.round(metricsChartData.reduce((acc, item) => acc + item.disk, 0) / metricsChartData.length)}%
                      </span>
                    </div>
                    <Progress value={Math.round(metricsChartData.reduce((acc, item) => acc + item.disk, 0) / metricsChartData.length)} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </FadeIn>
        </TabsContent>
        
        {/* User Analytics Tab */}
        <TabsContent value="users" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle>User Activity Trends</CardTitle>
                  <CardDescription>Daily active users and registration trends</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={userActivityData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="day" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="active_users" fill="#8884d8" name="Active Users" />
                        <Bar dataKey="new_registrations" fill="#82ca9d" name="New Registrations" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle>User Status Distribution</CardTitle>
                  <CardDescription>Current user account statuses</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 border rounded-lg">
                      <CheckCircle className="h-8 w-8 text-green-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-green-600">{keyMetrics.activeUsers}</div>
                      <div className="text-sm text-gray-600">Active Users</div>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <Clock className="h-8 w-8 text-yellow-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-yellow-600">{keyMetrics.pendingUsers}</div>
                      <div className="text-sm text-gray-600">Pending Approval</div>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <Shield className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-purple-600">{keyMetrics.twoFactorUsers}</div>
                      <div className="text-sm text-gray-600">2FA Enabled</div>
                    </div>
                    
                    <div className="text-center p-4 border rounded-lg">
                      <Users className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-blue-600">{keyMetrics.totalUsers}</div>
                      <div className="text-sm text-gray-600">Total Users</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
        
        {/* Resources Tab */}
        <TabsContent value="resources" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle>Resource Utilization</CardTitle>
                  <CardDescription>Current system resource allocation</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={resourceUsageData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value }) => `${name}: ${value}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {resourceUsageData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle>Resource Alerts & Recommendations</CardTitle>
                  <CardDescription>System optimization suggestions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-950/50 rounded-lg border border-green-200 dark:border-green-800">
                      <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-green-800 dark:text-green-200">CPU Usage Normal</div>
                        <div className="text-sm text-green-700 dark:text-green-300">
                          CPU utilization is within optimal range
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3 p-3 bg-yellow-50 dark:bg-yellow-950/50 rounded-lg border border-yellow-200 dark:border-yellow-800">
                      <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-yellow-800 dark:text-yellow-200">Memory Optimization</div>
                        <div className="text-sm text-yellow-700 dark:text-yellow-300">
                          Consider increasing memory allocation for better performance
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-950/50 rounded-lg border border-blue-200 dark:border-blue-800">
                      <Eye className="h-5 w-5 text-blue-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-blue-800 dark:text-blue-200">Storage Monitoring</div>
                        <div className="text-sm text-blue-700 dark:text-blue-300">
                          Disk usage is stable, no immediate action required
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
        
        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6">
          <FadeIn delay={0.5}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  Security Overview
                </CardTitle>
                <CardDescription>System security metrics and alerts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-6 border rounded-lg">
                    <div className="text-3xl font-bold text-green-600 mb-2">98.5%</div>
                    <div className="text-sm font-medium mb-1">Security Score</div>
                    <div className="text-xs text-gray-600">Above industry average</div>
                  </div>
                  
                  <div className="text-center p-6 border rounded-lg">
                    <div className="text-3xl font-bold text-yellow-600 mb-2">3</div>
                    <div className="text-sm font-medium mb-1">Active Alerts</div>
                    <div className="text-xs text-gray-600">Requires attention</div>
                  </div>
                  
                  <div className="text-center p-6 border rounded-lg">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {keyMetrics.twoFactorPercentage.toFixed(0)}%
                    </div>
                    <div className="text-sm font-medium mb-1">2FA Adoption</div>
                    <div className="text-xs text-gray-600">Security compliance</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </FadeIn>
        </TabsContent>
      </Tabs>
    </div>
  );
}