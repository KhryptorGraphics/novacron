"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FadeIn } from "@/lib/animations";
import { 
  FileText, 
  Search, 
  Download, 
  Filter, 
  Calendar,
  User,
  Shield,
  Database,
  Server,
  Settings,
  Eye,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Globe,
  Activity,
  Lock,
  Unlock,
  UserPlus,
  UserMinus,
  Edit,
  Trash2
} from "lucide-react";
import { cn } from "@/lib/utils";

// Mock audit log data
const auditLogs = [
  {
    id: 1,
    timestamp: "2024-08-24T14:30:15Z",
    user: "admin@novacron.io",
    action: "USER_LOGIN",
    resource: "Authentication System",
    resourceId: "session_12345",
    ip: "192.168.1.100",
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    status: "success",
    details: "Successful login with 2FA",
    category: "authentication",
    severity: "info"
  },
  {
    id: 2,
    timestamp: "2024-08-24T14:28:42Z",
    user: "manager@company.com",
    action: "VM_CREATED",
    resource: "Virtual Machine",
    resourceId: "vm_web_server_03",
    ip: "203.0.113.42",
    userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    status: "success",
    details: "Created new VM: web-server-03 with 4GB RAM, 2 vCPUs",
    category: "vm_management",
    severity: "info"
  },
  {
    id: 3,
    timestamp: "2024-08-24T14:25:18Z",
    user: "operator@startup.io",
    action: "CONFIG_CHANGED",
    resource: "System Configuration",
    resourceId: "config_security_settings",
    ip: "198.51.100.75",
    userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    status: "success",
    details: "Updated password policy: minimum length changed from 8 to 12 characters",
    category: "system_config",
    severity: "warning"
  },
  {
    id: 4,
    timestamp: "2024-08-24T14:20:33Z",
    user: "hacker@malicious.com",
    action: "LOGIN_FAILED",
    resource: "Authentication System",
    resourceId: null,
    ip: "192.0.2.100",
    userAgent: "curl/7.68.0",
    status: "failed",
    details: "Failed login attempt: invalid credentials (attempt 15/5)",
    category: "authentication",
    severity: "critical"
  },
  {
    id: 5,
    timestamp: "2024-08-24T14:18:07Z",
    user: "admin@novacron.io",
    action: "USER_SUSPENDED",
    resource: "User Account",
    resourceId: "user_suspicious_activity",
    ip: "192.168.1.100",
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    status: "success",
    details: "Suspended user account due to suspicious activity patterns",
    category: "user_management",
    severity: "warning"
  },
  {
    id: 6,
    timestamp: "2024-08-24T14:15:22Z",
    user: "data-analyst@company.com",
    action: "DATA_EXPORTED",
    resource: "Database",
    resourceId: "export_user_data_q3_2024",
    ip: "10.0.0.45",
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    status: "success",
    details: "Exported 15,432 user records for Q3 2024 analysis",
    category: "data_access",
    severity: "info"
  },
  {
    id: 7,
    timestamp: "2024-08-24T14:12:55Z",
    user: "vm-operator@datacenter.net",
    action: "VM_MIGRATED",
    resource: "Virtual Machine",
    resourceId: "vm_database_primary",
    ip: "172.16.0.25",
    userAgent: "NovaCron-CLI/2.1.0",
    status: "success",
    details: "Migrated VM from node-01 to node-03, migration time: 4m 23s",
    category: "vm_management",
    severity: "info"
  },
  {
    id: 8,
    timestamp: "2024-08-24T14:10:10Z",
    user: "security-admin@novacron.io",
    action: "IP_BLOCKED",
    resource: "Security System",
    resourceId: "block_rule_automated_001",
    ip: "192.168.1.50",
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    status: "success",
    details: "Automatically blocked IP 192.0.2.100 after 25 failed login attempts",
    category: "security",
    severity: "warning"
  }
];

const logCategories = [
  { value: "all", label: "All Categories" },
  { value: "authentication", label: "Authentication" },
  { value: "user_management", label: "User Management" },
  { value: "vm_management", label: "VM Management" },
  { value: "system_config", label: "System Configuration" },
  { value: "data_access", label: "Data Access" },
  { value: "security", label: "Security Events" }
];

const severityLevels = [
  { value: "all", label: "All Severities" },
  { value: "critical", label: "Critical" },
  { value: "warning", label: "Warning" },
  { value: "info", label: "Info" }
];

export function AuditLogs() {
  const [logs, setLogs] = useState(auditLogs);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [selectedLog, setSelectedLog] = useState<any>(null);

  const filteredLogs = logs.filter(log => {
    const matchesSearch = 
      log.user.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.resource.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.details.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.ip.includes(searchQuery);
    
    const matchesCategory = categoryFilter === "all" || log.category === categoryFilter;
    const matchesSeverity = severityFilter === "all" || log.severity === severityFilter;
    const matchesStatus = statusFilter === "all" || log.status === statusFilter;
    
    return matchesSearch && matchesCategory && matchesSeverity && matchesStatus;
  });

  const getActionIcon = (action: string) => {
    switch (action) {
      case "USER_LOGIN": return <Lock className="h-4 w-4 text-blue-600" />;
      case "LOGIN_FAILED": return <XCircle className="h-4 w-4 text-red-600" />;
      case "USER_CREATED": return <UserPlus className="h-4 w-4 text-green-600" />;
      case "USER_SUSPENDED": return <UserMinus className="h-4 w-4 text-orange-600" />;
      case "VM_CREATED": return <Server className="h-4 w-4 text-green-600" />;
      case "VM_MIGRATED": return <Activity className="h-4 w-4 text-blue-600" />;
      case "CONFIG_CHANGED": return <Settings className="h-4 w-4 text-orange-600" />;
      case "DATA_EXPORTED": return <Download className="h-4 w-4 text-purple-600" />;
      case "IP_BLOCKED": return <Shield className="h-4 w-4 text-red-600" />;
      default: return <FileText className="h-4 w-4 text-gray-600" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "bg-red-500";
      case "warning": return "bg-yellow-500";
      case "info": return "bg-blue-500";
      default: return "bg-gray-500";
    }
  };

  const getSeverityVariant = (severity: string) => {
    switch (severity) {
      case "critical": return "destructive";
      case "warning": return "secondary";
      case "info": return "outline";
      default: return "outline";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success": return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "failed": return <XCircle className="h-4 w-4 text-red-600" />;
      default: return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
  };

  const exportLogs = () => {
    const csvContent = [
      "Timestamp,User,Action,Resource,IP,Status,Severity,Details",
      ...filteredLogs.map(log => 
        `"${log.timestamp}","${log.user}","${log.action}","${log.resource}","${log.ip}","${log.status}","${log.severity}","${log.details}"`
      )
    ].join("\n");
    
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `audit_logs_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <FileText className="h-6 w-6" />
            Audit Logs
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Track system events and user activities
          </p>
        </div>
        
        <Button onClick={exportLogs}>
          <Download className="h-4 w-4 mr-2" />
          Export Logs
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search logs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                {logCategories.map(cat => (
                  <SelectItem key={cat.value} value={cat.value}>
                    {cat.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                {severityLevels.map(level => (
                  <SelectItem key={level.value} value={level.value}>
                    {level.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="success">Success</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>
            
            <div className="flex items-center justify-center">
              <Badge variant="outline">
                {filteredLogs.length} entries
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Audit Logs Table */}
      <FadeIn>
        <Card>
          <CardHeader>
            <CardTitle>System Audit Trail</CardTitle>
            <CardDescription>
              Chronological record of all system activities and events
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Resource</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Severity</TableHead>
                    <TableHead>Source IP</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.map((log) => {
                    const timestamp = formatTimestamp(log.timestamp);
                    
                    return (
                      <TableRow key={log.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                        <TableCell>
                          <div className="text-sm">
                            <div className="font-medium">{timestamp.date}</div>
                            <div className="text-gray-600 dark:text-gray-400">{timestamp.time}</div>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <User className="h-4 w-4 text-gray-400" />
                            <span className="font-mono text-sm">{log.user}</span>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getActionIcon(log.action)}
                            <span className="font-medium text-sm">
                              {log.action.replace(/_/g, ' ')}
                            </span>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <div className="text-sm">
                            <div className="font-medium">{log.resource}</div>
                            {log.resourceId && (
                              <div className="text-gray-600 dark:text-gray-400 font-mono text-xs">
                                {log.resourceId}
                              </div>
                            )}
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(log.status)}
                            <span className={cn(
                              "capitalize text-sm font-medium",
                              log.status === "success" ? "text-green-600" : "text-red-600"
                            )}>
                              {log.status}
                            </span>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <Badge variant={getSeverityVariant(log.severity) as any}>
                            {log.severity}
                          </Badge>
                        </TableCell>
                        
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Globe className="h-3 w-3 text-gray-400" />
                            <span className="font-mono text-sm">{log.ip}</span>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setSelectedLog(log)}
                            className="h-7 w-7 p-0"
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Log Detail Modal */}
      {selectedLog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <Card className="w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  {getActionIcon(selectedLog.action)}
                  Log Entry Details
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedLog(null)}
                >
                  <XCircle className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Timestamp:</span>
                  <div className="mt-1">{formatTimestamp(selectedLog.timestamp).date} {formatTimestamp(selectedLog.timestamp).time}</div>
                </div>
                
                <div>
                  <span className="font-medium">User:</span>
                  <div className="mt-1 font-mono">{selectedLog.user}</div>
                </div>
                
                <div>
                  <span className="font-medium">Action:</span>
                  <div className="mt-1">{selectedLog.action.replace(/_/g, ' ')}</div>
                </div>
                
                <div>
                  <span className="font-medium">Status:</span>
                  <div className="mt-1 flex items-center gap-2">
                    {getStatusIcon(selectedLog.status)}
                    <span className="capitalize">{selectedLog.status}</span>
                  </div>
                </div>
                
                <div>
                  <span className="font-medium">Resource:</span>
                  <div className="mt-1">{selectedLog.resource}</div>
                </div>
                
                <div>
                  <span className="font-medium">Severity:</span>
                  <div className="mt-1">
                    <Badge variant={getSeverityVariant(selectedLog.severity) as any}>
                      {selectedLog.severity}
                    </Badge>
                  </div>
                </div>
                
                <div>
                  <span className="font-medium">Source IP:</span>
                  <div className="mt-1 font-mono">{selectedLog.ip}</div>
                </div>
                
                <div>
                  <span className="font-medium">Category:</span>
                  <div className="mt-1 capitalize">{selectedLog.category.replace(/_/g, ' ')}</div>
                </div>
              </div>
              
              {selectedLog.resourceId && (
                <div>
                  <span className="font-medium">Resource ID:</span>
                  <div className="mt-1 font-mono text-sm bg-gray-100 dark:bg-gray-800 p-2 rounded">
                    {selectedLog.resourceId}
                  </div>
                </div>
              )}
              
              <div>
                <span className="font-medium">Details:</span>
                <div className="mt-1 text-sm bg-gray-100 dark:bg-gray-800 p-3 rounded">
                  {selectedLog.details}
                </div>
              </div>
              
              <div>
                <span className="font-medium">User Agent:</span>
                <div className="mt-1 text-xs font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded break-all">
                  {selectedLog.userAgent}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}