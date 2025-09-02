"use client";

import { useState, useEffect } from "react";
import { useSecurityAlerts, useUpdateSecurityAlert, useAuditLogs, useUsers } from "@/lib/api/hooks/useAdmin";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { 
  Shield, 
  AlertTriangle, 
  Eye, 
  CheckCircle, 
  XCircle,
  Clock,
  Search,
  Filter,
  Download,
  RefreshCw,
  Activity,
  Lock,
  Unlock,
  Key,
  Database,
  Network,
  Server,
  User,
  Settings,
  Ban,
  UserCheck,
  Mail,
  Globe,
  Smartphone,
  Wifi
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SecurityAlert, AuditLogEntry } from "@/lib/api/types";
import { FadeIn } from "@/lib/animations";
import { useForm } from "react-hook-form";

// Mock security data
const securityMetrics = {
  overallScore: 94,
  threatsBlocked: 1247,
  vulnerabilitiesFound: 3,
  authenticationAttempts: 15432,
  failedLogins: 234,
  suspiciousActivities: 12,
  firewallRules: 156,
  activeConnections: 892,
  sslCertStatus: "valid",
  lastScan: "2024-08-24T10:30:00Z"
};

const threatCategories = [
  { name: "Malware", count: 45, severity: "high", trend: "decreasing" },
  { name: "Phishing", count: 23, severity: "medium", trend: "stable" },
  { name: "Brute Force", count: 12, severity: "high", trend: "increasing" },
  { name: "Data Breach", count: 2, severity: "critical", trend: "decreasing" },
  { name: "Unauthorized Access", count: 8, severity: "high", trend: "stable" },
];

const complianceChecks = [
  { name: "GDPR Compliance", status: "passing", score: 98, description: "Data protection and privacy" },
  { name: "SOC2 Type II", status: "passing", score: 96, description: "Security and availability controls" },
  { name: "ISO 27001", status: "warning", score: 87, description: "Information security management" },
  { name: "PCI DSS", status: "passing", score: 94, description: "Payment card data security" },
  { name: "HIPAA", status: "not_applicable", score: null, description: "Healthcare data protection" },
];

export default function SecurityCenterPage() {
  const { toast } = useToast();
  const [alertFilters, setAlertFilters] = useState({
    status: '',
    severity: '',
    type: '',
    page: 1,
    pageSize: 20
  });
  const [selectedAlert, setSelectedAlert] = useState<SecurityAlert | null>(null);
  const [showAlertDialog, setShowAlertDialog] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  const { data: alertsData, isLoading: alertsLoading, refetch: refetchAlerts } = useSecurityAlerts(alertFilters);
  const { data: auditData } = useAuditLogs({ pageSize: 50 });
  const { data: usersData } = useUsers({ pageSize: 1000 });
  const updateAlert = useUpdateSecurityAlert();
  
  const { register, handleSubmit, reset } = useForm();
  
  // Auto-refresh alerts every 30 seconds
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        refetchAlerts();
      }, 30000);
      
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refetchAlerts]);
  
  const alerts = alertsData?.alerts || [];
  
  const handleAlertAction = async (alertId: string, action: 'resolve' | 'investigate' | 'dismiss') => {
    try {
      let status: SecurityAlert['status'];
      switch (action) {
        case 'resolve':
          status = 'resolved';
          break;
        case 'investigate':
          status = 'investigating';
          break;
        case 'dismiss':
          status = 'false_positive';
          break;
      }
      
      await updateAlert.mutateAsync({ id: alertId, status });
      
      toast({
        title: "Alert updated",
        description: `Alert has been marked as ${status.replace('_', ' ')}.`
      });
    } catch (error) {
      toast({
        title: "Failed to update alert",
        description: "Please try again later.",
        variant: "destructive"
      });
    }
  };
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "destructive";
      case "high": return "destructive";
      case "medium": return "secondary";
      case "low": return "outline";
      default: return "outline";
    }
  };
  
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
      case "high":
        return <AlertTriangle className="h-4 w-4" />;
      case "medium":
        return <Eye className="h-4 w-4" />;
      case "low":
        return <CheckCircle className="h-4 w-4" />;
      default:
        return <Eye className="h-4 w-4" />;
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case "new": return "text-red-600";
      case "investigating": return "text-yellow-600";
      case "resolved": return "text-green-600";
      case "false_positive": return "text-gray-600";
      default: return "text-gray-600";
    }
  };
  
  const getComplianceColor = (status: string, score?: number | null) => {
    if (status === "not_applicable") return "text-gray-600";
    if (status === "passing" && score && score >= 90) return "text-green-600";
    if (status === "warning" || (score && score >= 70)) return "text-yellow-600";
    return "text-red-600";
  };
  
  const getSecurityScore = () => {
    const users = usersData?.users || [];
    const twoFactorEnabled = users.filter(u => u.two_factor_enabled).length;
    const totalUsers = users.length;
    const twoFactorPercentage = totalUsers > 0 ? (twoFactorEnabled / totalUsers) * 100 : 0;
    
    const criticalAlerts = alerts.filter(a => a.severity === 'critical' && a.status === 'new').length;
    const alertsPenalty = Math.min(criticalAlerts * 5, 20); // Max 20 point penalty
    
    const baseScore = 85;
    const twoFactorBonus = Math.min(twoFactorPercentage * 0.15, 15); // Max 15 point bonus
    
    return Math.max(0, Math.min(100, baseScore + twoFactorBonus - alertsPenalty));
  };
  
  const securityScore = getSecurityScore();
  
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Shield className="h-8 w-8" />
            Security Center
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Monitor security threats, compliance, and system protection
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Label htmlFor="auto-refresh" className="text-sm">Auto-refresh</Label>
            <input 
              id="auto-refresh"
              type="checkbox" 
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
          </div>
          <Button variant="outline" onClick={() => refetchAlerts()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Security Report
          </Button>
        </div>
      </div>
      
      {/* Security Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Security Score</p>
                  <p className={`text-2xl font-bold ${
                    securityScore >= 90 ? 'text-green-600' : 
                    securityScore >= 70 ? 'text-yellow-600' : 
                    'text-red-600'
                  }`}>
                    {securityScore.toFixed(0)}%
                  </p>
                  <div className="mt-2">
                    <Progress value={securityScore} className="h-2" />
                  </div>
                </div>
                <Shield className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Alerts</p>
                  <p className="text-2xl font-bold text-red-600">
                    {alerts.filter(a => a.status === 'new').length}
                  </p>
                  <div className="text-xs text-gray-600">
                    {alerts.filter(a => a.severity === 'critical').length} critical
                  </div>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Threats Blocked</p>
                  <p className="text-2xl font-bold text-green-600">{securityMetrics.threatsBlocked}</p>
                  <div className="text-xs text-gray-600">Last 30 days</div>
                </div>
                <Lock className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
        
        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Failed Logins</p>
                  <p className="text-2xl font-bold text-yellow-600">{securityMetrics.failedLogins}</p>
                  <div className="text-xs text-gray-600">Last 24 hours</div>
                </div>
                <Unlock className="h-8 w-8 text-yellow-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>
      
      {/* Security Tabs */}
      <Tabs defaultValue="alerts" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="alerts">Security Alerts</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
          <TabsTrigger value="threats">Threat Analysis</TabsTrigger>
          <TabsTrigger value="access">Access Control</TabsTrigger>
        </TabsList>
        
        {/* Security Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          {/* Filters */}
          <Card>
            <CardContent className="p-6">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search alerts..."
                    className="pl-10"
                  />
                </div>
                
                <Select 
                  value={alertFilters.status} 
                  onValueChange={(value) => setAlertFilters(prev => ({ ...prev, status: value, page: 1 }))}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="All Statuses" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Statuses</SelectItem>
                    <SelectItem value="new">New</SelectItem>
                    <SelectItem value="investigating">Investigating</SelectItem>
                    <SelectItem value="resolved">Resolved</SelectItem>
                    <SelectItem value="false_positive">False Positive</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select 
                  value={alertFilters.severity} 
                  onValueChange={(value) => setAlertFilters(prev => ({ ...prev, severity: value, page: 1 }))}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="All Severities" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Severities</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select 
                  value={alertFilters.type} 
                  onValueChange={(value) => setAlertFilters(prev => ({ ...prev, type: value, page: 1 }))}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="All Types" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Types</SelectItem>
                    <SelectItem value="authentication">Authentication</SelectItem>
                    <SelectItem value="access_control">Access Control</SelectItem>
                    <SelectItem value="data_breach">Data Breach</SelectItem>
                    <SelectItem value="malware">Malware</SelectItem>
                    <SelectItem value="network">Network</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
          
          {/* Alerts Table */}
          <FadeIn delay={0.5}>
            <Card>
              <CardHeader>
                <CardTitle>Security Alerts ({alerts.length})</CardTitle>
                <CardDescription>
                  Active security threats and incidents requiring attention
                </CardDescription>
              </CardHeader>
              <CardContent>
                {alertsLoading ? (
                  <div className="flex items-center justify-center h-64">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Alert</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Severity</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Source</TableHead>
                          <TableHead>Time</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {alerts.map((alert) => (
                          <TableRow key={alert.id}>
                            <TableCell>
                              <div>
                                <div className="font-medium">{alert.title}</div>
                                <div className="text-sm text-gray-600 dark:text-gray-400 truncate max-w-xs">
                                  {alert.description}
                                </div>
                              </div>
                            </TableCell>
                            
                            <TableCell>
                              <Badge variant="outline" className="capitalize">
                                {alert.type.replace('_', ' ')}
                              </Badge>
                            </TableCell>
                            
                            <TableCell>
                              <Badge 
                                variant={getSeverityColor(alert.severity) as any}
                                className="flex items-center gap-1 w-fit"
                              >
                                {getSeverityIcon(alert.severity)}
                                {alert.severity}
                              </Badge>
                            </TableCell>
                            
                            <TableCell>
                              <span className={cn("capitalize", getStatusColor(alert.status))}>
                                {alert.status.replace('_', ' ')}
                              </span>
                            </TableCell>
                            
                            <TableCell>{alert.source}</TableCell>
                            
                            <TableCell>
                              <div className="text-sm">
                                {new Date(alert.timestamp).toLocaleDateString()}
                                <div className="text-xs text-gray-600 dark:text-gray-400">
                                  {new Date(alert.timestamp).toLocaleTimeString()}
                                </div>
                              </div>
                            </TableCell>
                            
                            <TableCell>
                              <div className="flex items-center gap-1">
                                {alert.status === 'new' && (
                                  <>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={() => handleAlertAction(alert.id, 'investigate')}
                                      className="text-xs px-2 h-7"
                                    >
                                      Investigate
                                    </Button>
                                    <Button
                                      size="sm"
                                      onClick={() => handleAlertAction(alert.id, 'resolve')}
                                      className="text-xs px-2 h-7"
                                    >
                                      Resolve
                                    </Button>
                                  </>
                                )}
                                
                                <Dialog>
                                  <DialogTrigger asChild>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      className="h-7 w-7 p-0"
                                    >
                                      <Eye className="h-3 w-3" />
                                    </Button>
                                  </DialogTrigger>
                                  <DialogContent className="max-w-2xl">
                                    <DialogHeader>
                                      <DialogTitle className="flex items-center gap-2">
                                        {getSeverityIcon(alert.severity)}
                                        {alert.title}
                                      </DialogTitle>
                                      <DialogDescription>
                                        Alert details and remediation information
                                      </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-4">
                                      <div>
                                        <Label>Description</Label>
                                        <p className="text-sm mt-1">{alert.description}</p>
                                      </div>
                                      
                                      <div className="grid grid-cols-2 gap-4">
                                        <div>
                                          <Label>Type</Label>
                                          <p className="text-sm mt-1 capitalize">{alert.type.replace('_', ' ')}</p>
                                        </div>
                                        <div>
                                          <Label>Severity</Label>
                                          <p className="text-sm mt-1 capitalize">{alert.severity}</p>
                                        </div>
                                        <div>
                                          <Label>Source</Label>
                                          <p className="text-sm mt-1">{alert.source}</p>
                                        </div>
                                        <div>
                                          <Label>Status</Label>
                                          <p className="text-sm mt-1 capitalize">{alert.status.replace('_', ' ')}</p>
                                        </div>
                                      </div>
                                      
                                      {alert.affected_resources.length > 0 && (
                                        <div>
                                          <Label>Affected Resources</Label>
                                          <div className="flex flex-wrap gap-1 mt-1">
                                            {alert.affected_resources.map((resource, index) => (
                                              <Badge key={index} variant="secondary" className="text-xs">
                                                {resource}
                                              </Badge>
                                            ))}
                                          </div>
                                        </div>
                                      )}
                                      
                                      {alert.remediation_steps && (
                                        <div>
                                          <Label>Remediation Steps</Label>
                                          <ol className="text-sm mt-1 space-y-1 list-decimal list-inside">
                                            {alert.remediation_steps.map((step, index) => (
                                              <li key={index}>{step}</li>
                                            ))}
                                          </ol>
                                        </div>
                                      )}
                                      
                                      <div className="flex justify-end gap-2 pt-4">
                                        <Button
                                          variant="outline"
                                          onClick={() => handleAlertAction(alert.id, 'dismiss')}
                                        >
                                          Mark as False Positive
                                        </Button>
                                        <Button
                                          onClick={() => handleAlertAction(alert.id, 'resolve')}
                                        >
                                          Mark as Resolved
                                        </Button>
                                      </div>
                                    </div>
                                  </DialogContent>
                                </Dialog>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          </FadeIn>
        </TabsContent>
        
        {/* Compliance Tab */}
        <TabsContent value="compliance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle>Compliance Status</CardTitle>
                  <CardDescription>Current compliance with security standards</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {complianceChecks.map((check, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`h-3 w-3 rounded-full ${
                          check.status === 'passing' ? 'bg-green-500' :
                          check.status === 'warning' ? 'bg-yellow-500' :
                          check.status === 'not_applicable' ? 'bg-gray-500' :
                          'bg-red-500'
                        }`} />
                        <div>
                          <div className="font-medium">{check.name}</div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {check.description}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        {check.score !== null ? (
                          <div className={`font-bold ${getComplianceColor(check.status, check.score)}`}>
                            {check.score}%
                          </div>
                        ) : (
                          <div className="text-gray-400 text-sm">N/A</div>
                        )}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle>Security Policies</CardTitle>
                  <CardDescription>Active security policies and configurations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Key className="h-5 w-5 text-blue-600" />
                        <div>
                          <div className="font-medium">Password Policy</div>
                          <div className="text-sm text-gray-600">Min 12 chars, complexity required</div>
                        </div>
                      </div>
                      <Badge variant="secondary">Enabled</Badge>
                    </div>
                    
                    <div className="flex justify-between items-center p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Shield className="h-5 w-5 text-green-600" />
                        <div>
                          <div className="font-medium">Multi-Factor Authentication</div>
                          <div className="text-sm text-gray-600">TOTP, SMS, Hardware tokens</div>
                        </div>
                      </div>
                      <Badge variant="secondary">Enforced</Badge>
                    </div>
                    
                    <div className="flex justify-between items-center p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Lock className="h-5 w-5 text-purple-600" />
                        <div>
                          <div className="font-medium">Session Management</div>
                          <div className="text-sm text-gray-600">30min timeout, secure cookies</div>
                        </div>
                      </div>
                      <Badge variant="secondary">Active</Badge>
                    </div>
                    
                    <div className="flex justify-between items-center p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Database className="h-5 w-5 text-orange-600" />
                        <div>
                          <div className="font-medium">Data Encryption</div>
                          <div className="text-sm text-gray-600">AES-256 at rest, TLS 1.3 in transit</div>
                        </div>
                      </div>
                      <Badge variant="secondary">Enabled</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
        
        {/* Threat Analysis Tab */}
        <TabsContent value="threats" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle>Threat Categories</CardTitle>
                  <CardDescription>Analysis of security threats by category</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {threatCategories.map((category, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className={`h-3 w-3 rounded-full ${
                            category.severity === 'critical' ? 'bg-red-500' :
                            category.severity === 'high' ? 'bg-orange-500' :
                            category.severity === 'medium' ? 'bg-yellow-500' :
                            'bg-green-500'
                          }`} />
                          <div>
                            <div className="font-medium">{category.name}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">
                              {category.count} incidents this month
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge 
                            variant={getSeverityColor(category.severity) as any}
                            className="mb-1"
                          >
                            {category.severity}
                          </Badge>
                          <div className={`text-xs ${
                            category.trend === 'increasing' ? 'text-red-600' :
                            category.trend === 'decreasing' ? 'text-green-600' :
                            'text-gray-600'
                          }`}>
                            {category.trend}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle>Threat Intelligence</CardTitle>
                  <CardDescription>Latest threat intelligence and recommendations</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-3 bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded-lg">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                        <div>
                          <div className="font-medium text-red-800 dark:text-red-200">High Risk Alert</div>
                          <div className="text-sm text-red-700 dark:text-red-300">
                            New ransomware campaign detected targeting similar organizations
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-950/50 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                      <div className="flex items-start gap-2">
                        <Eye className="h-4 w-4 text-yellow-600 mt-0.5" />
                        <div>
                          <div className="font-medium text-yellow-800 dark:text-yellow-200">Monitoring</div>
                          <div className="text-sm text-yellow-700 dark:text-yellow-300">
                            Increased phishing attempts observed across industry
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-blue-50 dark:bg-blue-950/50 border border-blue-200 dark:border-blue-800 rounded-lg">
                      <div className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-blue-600 mt-0.5" />
                        <div>
                          <div className="font-medium text-blue-800 dark:text-blue-200">Recommendation</div>
                          <div className="text-sm text-blue-700 dark:text-blue-300">
                            Update security awareness training to include latest threats
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
        
        {/* Access Control Tab */}
        <TabsContent value="access" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <FadeIn delay={0.5}>
              <Card>
                <CardHeader>
                  <CardTitle>Authentication Methods</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Key className="h-4 w-4" />
                      <span className="text-sm">Password</span>
                    </div>
                    <Badge variant="secondary">100%</Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Smartphone className="h-4 w-4" />
                      <span className="text-sm">SMS 2FA</span>
                    </div>
                    <Badge variant="secondary">
                      {usersData?.users ? 
                        Math.round((usersData.users.filter(u => u.two_factor_enabled).length / usersData.users.length) * 100) :
                        0
                      }%
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4" />
                      <span className="text-sm">TOTP App</span>
                    </div>
                    <Badge variant="secondary">45%</Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Wifi className="h-4 w-4" />
                      <span className="text-sm">Hardware Token</span>
                    </div>
                    <Badge variant="secondary">12%</Badge>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.6}>
              <Card>
                <CardHeader>
                  <CardTitle>Role Distribution</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Administrators</span>
                    <Badge variant="destructive">
                      {usersData?.users ? usersData.users.filter(u => u.role === 'admin').length : 0}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Moderators</span>
                    <Badge variant="secondary">
                      {usersData?.users ? usersData.users.filter(u => u.role === 'moderator').length : 0}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Users</span>
                    <Badge variant="outline">
                      {usersData?.users ? usersData.users.filter(u => u.role === 'user').length : 0}
                    </Badge>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Viewers</span>
                    <Badge variant="outline">
                      {usersData?.users ? usersData.users.filter(u => u.role === 'viewer').length : 0}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
            
            <FadeIn delay={0.7}>
              <Card>
                <CardHeader>
                  <CardTitle>Session Statistics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Active Sessions</span>
                    <span className="font-bold text-green-600">{securityMetrics.activeConnections}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Failed Logins (24h)</span>
                    <span className="font-bold text-red-600">{securityMetrics.failedLogins}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Suspicious Activities</span>
                    <span className="font-bold text-yellow-600">{securityMetrics.suspiciousActivities}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Total Auth Attempts</span>
                    <span className="font-bold">{securityMetrics.authenticationAttempts.toLocaleString()}</span>
                  </div>
                </CardContent>
              </Card>
            </FadeIn>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}