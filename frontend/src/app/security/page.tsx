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
  Shield, 
  Key, 
  Lock,
  Eye,
  AlertTriangle,
  CheckCircle,
  Users,
  FileText,
  Settings,
  Search,
  Plus,
  Activity,
  Globe,
  Server,
  Database,
  Network
} from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Mock security data
const mockSecurityAlerts = [
  {
    id: "alert-001",
    type: "authentication",
    severity: "high",
    title: "Multiple failed login attempts",
    description: "5 failed login attempts from IP 192.168.1.100",
    timestamp: "2025-04-11T14:30:00Z",
    status: "active",
    source: "auth-system",
    affectedResource: "admin-portal"
  },
  {
    id: "alert-002",
    type: "network", 
    severity: "medium",
    title: "Unusual network traffic",
    description: "Detected high outbound traffic from VM vm-003",
    timestamp: "2025-04-11T13:45:00Z",
    status: "investigating",
    source: "network-monitor",
    affectedResource: "vm-003"
  },
  {
    id: "alert-003",
    type: "access",
    severity: "low",
    title: "New admin user created",
    description: "Admin user 'john.doe' was created by 'admin'",
    timestamp: "2025-04-11T12:15:00Z", 
    status: "resolved",
    source: "user-management",
    affectedResource: "user-system"
  }
];

const mockAuditLogs = [
  {
    id: "log-001",
    timestamp: "2025-04-11T14:35:00Z",
    user: "admin",
    action: "vm.create",
    resource: "vm-005",
    details: "Created new VM with 4 CPU cores, 8GB RAM",
    ipAddress: "192.168.1.50",
    userAgent: "Mozilla/5.0...",
    result: "success"
  },
  {
    id: "log-002",
    timestamp: "2025-04-11T14:20:00Z",
    user: "operator",
    action: "vm.migrate",
    resource: "vm-001",
    details: "Migrated VM from node-01 to node-02",
    ipAddress: "192.168.1.51",
    userAgent: "Mozilla/5.0...",
    result: "success"
  },
  {
    id: "log-003", 
    timestamp: "2025-04-11T14:00:00Z",
    user: "unknown",
    action: "auth.login",
    resource: "admin-portal",
    details: "Failed login attempt with username 'admin'",
    ipAddress: "192.168.1.100",
    userAgent: "curl/7.68.0",
    result: "failed"
  }
];

const mockSecurityPolicies = [
  {
    id: "policy-001",
    name: "Password Policy",
    type: "authentication",
    status: "enabled",
    description: "Minimum 12 characters, complexity requirements",
    lastUpdated: "2025-03-15T10:00:00Z",
    affectedUsers: 25
  },
  {
    id: "policy-002", 
    name: "Network Access Control",
    type: "network",
    status: "enabled",
    description: "Restrict VM network access based on security groups",
    lastUpdated: "2025-03-10T14:30:00Z",
    affectedUsers: 15
  },
  {
    id: "policy-003",
    name: "Data Encryption",
    type: "data",
    status: "enabled", 
    description: "Encrypt all VM storage and network traffic",
    lastUpdated: "2025-02-28T09:15:00Z",
    affectedUsers: 30
  }
];

export default function SecurityPage() {
  const [securityAlerts, setSecurityAlerts] = useState(mockSecurityAlerts);
  const [auditLogs, setAuditLogs] = useState(mockAuditLogs);
  const [searchQuery, setSearchQuery] = useState("");
  const [alertFilter, setAlertFilter] = useState("all");
  const [logFilter, setLogFilter] = useState("all");

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "high": return "bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400";
      case "medium": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "low": return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": 
      case "enabled": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "investigating": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "resolved": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "disabled": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "authentication": return <Key className="h-4 w-4" />;
      case "network": return <Network className="h-4 w-4" />;
      case "access": return <Users className="h-4 w-4" />;
      case "data": return <Database className="h-4 w-4" />;
      default: return <Shield className="h-4 w-4" />;
    }
  };

  const securityStats = {
    totalAlerts: securityAlerts.length,
    activeAlerts: securityAlerts.filter(a => a.status === "active").length,
    highSeverityAlerts: securityAlerts.filter(a => a.severity === "high" || a.severity === "critical").length,
    auditEvents: auditLogs.length,
    failedLogins: auditLogs.filter(log => log.action.includes("login") && log.result === "failed").length,
    activePolicies: mockSecurityPolicies.filter(p => p.status === "enabled").length
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Security Dashboard</h1>
          <p className="text-muted-foreground">Monitor security events and manage access policies</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Add Policy
          </Button>
          <Button>
            <Shield className="h-4 w-4 mr-2" />
            Security Scan
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Security Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{securityStats.totalAlerts}</div>
            <p className="text-xs text-muted-foreground">
              {securityStats.activeAlerts} active
            </p>
            {securityStats.highSeverityAlerts > 0 && (
              <div className="mt-2">
                <Badge variant="destructive" className="text-xs">
                  {securityStats.highSeverityAlerts} high priority
                </Badge>
              </div>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed Logins</CardTitle>
            <Key className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{securityStats.failedLogins}</div>
            <p className="text-xs text-muted-foreground">
              Last 24 hours
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Policies</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{securityStats.activePolicies}</div>
            <p className="text-xs text-muted-foreground">
              of {mockSecurityPolicies.length} total
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Audit Events</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{securityStats.auditEvents}</div>
            <p className="text-xs text-muted-foreground">
              Recent activity
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Security Health Status */}
      {securityStats.activeAlerts > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Security Alert</AlertTitle>
          <AlertDescription>
            You have {securityStats.activeAlerts} active security alert{securityStats.activeAlerts > 1 ? 's' : ''} 
            that require attention. {securityStats.highSeverityAlerts > 0 && 
            `${securityStats.highSeverityAlerts} are high priority.`}
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="alerts" className="w-full">
        <TabsList>
          <TabsTrigger value="alerts">Security Alerts</TabsTrigger>
          <TabsTrigger value="audit">Audit Logs</TabsTrigger>
          <TabsTrigger value="policies">Security Policies</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
        </TabsList>
        
        <TabsContent value="alerts" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row gap-4 items-center">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search alerts..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Select value={alertFilter} onValueChange={setAlertFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Severity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severity</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Alerts List */}
          <div className="space-y-3">
            {securityAlerts.map((alert) => (
              <Card key={alert.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
                        {getTypeIcon(alert.type)}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{alert.title}</CardTitle>
                        <p className="text-sm text-muted-foreground">
                          {alert.source} • {alert.affectedResource}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getSeverityColor(alert.severity)}>
                        {alert.severity}
                      </Badge>
                      <Badge className={getStatusColor(alert.status)}>
                        {alert.status}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent>
                  <p className="text-sm mb-3">{alert.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {new Date(alert.timestamp).toLocaleString()}
                    </span>
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        Investigate
                      </Button>
                      <Button variant="outline" size="sm">
                        Dismiss
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="audit" className="space-y-4">
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>User</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Resource</TableHead>
                  <TableHead>IP Address</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead>Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {auditLogs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell className="text-sm">
                      {new Date(log.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-medium">{log.user}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono">
                        {log.action}
                      </Badge>
                    </TableCell>
                    <TableCell>{log.resource}</TableCell>
                    <TableCell className="font-mono">{log.ipAddress}</TableCell>
                    <TableCell>
                      <Badge className={log.result === "success" ? 
                        "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400" :
                        "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"
                      }>
                        {log.result}
                      </Badge>
                    </TableCell>
                    <TableCell className="max-w-xs truncate">{log.details}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
        
        <TabsContent value="policies" className="space-y-4">
          <div className="grid gap-4">
            {mockSecurityPolicies.map((policy) => (
              <Card key={policy.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        {getTypeIcon(policy.type)}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{policy.name}</CardTitle>
                        <p className="text-sm text-muted-foreground">
                          {policy.type} policy • {policy.affectedUsers} users
                        </p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(policy.status)}>
                      {policy.status}
                    </Badge>
                  </div>
                </CardHeader>
                
                <CardContent>
                  <p className="text-sm mb-3">{policy.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      Last updated: {new Date(policy.lastUpdated).toLocaleDateString()}
                    </span>
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4 mr-1" />
                        Configure
                      </Button>
                      <Button variant="outline" size="sm">
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="compliance" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Compliance Status</CardTitle>
                <CardDescription>Current compliance with security standards</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">ISO 27001</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={95} className="w-24 h-2" />
                    <span className="text-sm text-green-600">95%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">SOC 2</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={88} className="w-24 h-2" />
                    <span className="text-sm text-green-600">88%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">PCI DSS</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={72} className="w-24 h-2" />
                    <span className="text-sm text-yellow-600">72%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">GDPR</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={90} className="w-24 h-2" />
                    <span className="text-sm text-green-600">90%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Security Recommendations</CardTitle>
                <CardDescription>Actions to improve security posture</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    <span className="text-sm">Enable 2FA for all admin users</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                    <span className="text-sm">Update encryption keys (90 days old)</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm">Network segmentation configured</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    <span className="text-sm">Schedule security vulnerability scan</span>
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