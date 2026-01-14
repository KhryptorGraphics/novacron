"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricsCard } from "@/components/monitoring/MetricsCard";
import { AnimatedCounter, FadeIn, StaggeredList } from "@/lib/animations";
import { 
  Users, 
  Database, 
  Shield, 
  AlertTriangle, 
  Server, 
  Activity,
  HardDrive,
  Network,
  Clock,
  TrendingUp,
  TrendingDown,
  UserCheck,
  UserX,
  Eye,
  Lock
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

// Mock admin metrics data
const adminMetrics = {
  users: {
    total: 1247,
    active: 892,
    pending: 23,
    suspended: 12,
    growth: 8.5
  },
  security: {
    loginAttempts: 15432,
    failedLogins: 234,
    sessions: 167,
    alerts: 5
  },
  system: {
    uptime: "99.97%",
    diskUsage: 67,
    memoryUsage: 73,
    cpuUsage: 45,
    networkThroughput: 2.4
  },
  database: {
    connections: 45,
    maxConnections: 100,
    queryTime: 12.3,
    slowQueries: 3
  }
};

const systemHealth = [
  { service: "Authentication Service", status: "healthy", uptime: "99.98%", responseTime: "45ms" },
  { service: "Database Cluster", status: "healthy", uptime: "99.95%", responseTime: "12ms" },
  { service: "VM Management", status: "warning", uptime: "99.87%", responseTime: "156ms" },
  { service: "Storage Backend", status: "healthy", uptime: "99.99%", responseTime: "23ms" },
  { service: "Network Overlay", status: "healthy", uptime: "99.94%", responseTime: "8ms" }
];

const recentAlerts = [
  { id: 1, type: "security", severity: "high", message: "Multiple failed login attempts detected", time: "2 min ago" },
  { id: 2, type: "performance", severity: "medium", message: "VM migration queue exceeding threshold", time: "15 min ago" },
  { id: 3, type: "system", severity: "low", message: "Scheduled maintenance reminder", time: "1 hour ago" },
  { id: 4, type: "security", severity: "high", message: "Unusual API access pattern detected", time: "2 hours ago" }
];

export function AdminMetrics() {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefreshMetrics = async () => {
    setRefreshing(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setRefreshing(false);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "destructive";
      case "medium": return "secondary";
      case "low": return "outline";
      default: return "outline";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy": return "text-green-600";
      case "warning": return "text-yellow-600";
      case "error": return "text-red-600";
      default: return "text-gray-600";
    }
  };

  return (
    <div className="space-y-8">
      {/* System Overview Metrics */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">System Overview</h2>
          <Button 
            variant="outline" 
            onClick={handleRefreshMetrics}
            disabled={refreshing}
          >
            {refreshing ? "Refreshing..." : "Refresh Metrics"}
          </Button>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <FadeIn delay={0.1}>
            <MetricsCard
              title="Total Users"
              value={<AnimatedCounter value={adminMetrics.users.total} />}
              change={adminMetrics.users.growth}
              changeLabel="vs last month"
              status="success"
              icon={<Users className="h-5 w-5" />}
            />
          </FadeIn>
          
          <FadeIn delay={0.2}>
            <MetricsCard
              title="System Uptime"
              value={adminMetrics.system.uptime}
              change={0.02}
              changeLabel="vs last month"
              status="success"
              icon={<Clock className="h-5 w-5" />}
            />
          </FadeIn>
          
          <FadeIn delay={0.3}>
            <MetricsCard
              title="Security Alerts"
              value={<AnimatedCounter value={adminMetrics.security.alerts} />}
              change={-15}
              changeLabel="vs last week"
              status="warning"
              icon={<Shield className="h-5 w-5" />}
            />
          </FadeIn>
          
          <FadeIn delay={0.4}>
            <MetricsCard
              title="DB Connections"
              value={`${adminMetrics.database.connections}/${adminMetrics.database.maxConnections}`}
              change={8}
              changeLabel="active sessions"
              status="success"
              icon={<Database className="h-5 w-5" />}
            />
          </FadeIn>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* User Statistics */}
        <FadeIn delay={0.5}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserCheck className="h-5 w-5" />
                User Statistics
              </CardTitle>
              <CardDescription>Active user engagement and account status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span>Active Users</span>
                <div className="flex items-center gap-2">
                  <span className="font-semibold">{adminMetrics.users.active}</span>
                  <TrendingUp className="h-4 w-4 text-green-600" />
                </div>
              </div>
              <Progress value={(adminMetrics.users.active / adminMetrics.users.total) * 100} className="h-2" />
              
              <div className="flex justify-between items-center">
                <span>Pending Approvals</span>
                <Badge variant="secondary">{adminMetrics.users.pending}</Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span>Suspended Accounts</span>
                <Badge variant="destructive">{adminMetrics.users.suspended}</Badge>
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        {/* System Resources */}
        <FadeIn delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                System Resources
              </CardTitle>
              <CardDescription>Current system resource utilization</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span>CPU Usage</span>
                  <span className="font-semibold">{adminMetrics.system.cpuUsage}%</span>
                </div>
                <Progress value={adminMetrics.system.cpuUsage} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <span>Memory Usage</span>
                  <span className="font-semibold">{adminMetrics.system.memoryUsage}%</span>
                </div>
                <Progress value={adminMetrics.system.memoryUsage} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <span>Disk Usage</span>
                  <span className="font-semibold">{adminMetrics.system.diskUsage}%</span>
                </div>
                <Progress value={adminMetrics.system.diskUsage} className="h-2" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Service Health */}
        <FadeIn delay={0.7}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Service Health
              </CardTitle>
              <CardDescription>Status of critical system services</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {systemHealth.map((service, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <div className={`h-3 w-3 rounded-full ${
                        service.status === 'healthy' ? 'bg-green-500' : 
                        service.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                      <div>
                        <span className="font-medium">{service.service}</span>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {service.uptime} uptime • {service.responseTime}
                        </div>
                      </div>
                    </div>
                    <Badge variant={service.status === 'healthy' ? 'secondary' : 'destructive'}>
                      {service.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        {/* Recent Alerts */}
        <FadeIn delay={0.8}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Recent Alerts
              </CardTitle>
              <CardDescription>Latest system alerts and notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentAlerts.map((alert) => (
                  <div key={alert.id} className="flex items-start gap-3 p-3 rounded-lg border">
                    <div className="flex items-center gap-2 flex-1">
                      <Badge variant={getSeverityColor(alert.severity) as any} className="shrink-0">
                        {alert.severity}
                      </Badge>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{alert.message}</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">{alert.time}</p>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
    </div>
  );
}