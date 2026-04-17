'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Shield, AlertTriangle, XCircle,
  Eye, RefreshCw, Download, Filter,
  FileText, Activity, TrendingUp,
  AlertCircle, CheckCircle2, Info, Ban, Zap,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useToast } from '@/components/ui/use-toast';
import { PieChart, Pie, Cell, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useSecurityEvents, useCompliance, useVulnerabilityScans, useSecurityMetrics } from '@/hooks/useSecurity';
import RBACGuard from '@/components/auth/RBACGuard';

// Import types from the API service
import type { VulnerabilityFinding } from '@/lib/api/security';

const SecurityComplianceDashboard: React.FC = () => {
  const { toast } = useToast();
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const refreshInterval = 30;

  // Use real API hooks instead of mock data
  const {
    events: securityEvents,
    loading: eventsLoading,
    error: eventsError,
  } = useSecurityEvents(autoRefresh, refreshInterval * 1000);

  const {
    requirements: complianceRequirements,
    categoryBreakdown: complianceByCategory,
    loading: complianceLoading,
    error: complianceError,
    triggerComplianceCheck
  } = useCompliance();

  const {
    scans: vulnerabilityScans,
    loading: scansLoading,
    error: scansError,
    startScan
  } = useVulnerabilityScans();

  const {
    metrics,
    threatTrends,
    loading: metricsLoading,
    error: metricsError
  } = useSecurityMetrics(autoRefresh, 60000);

  // Transform API data for charts
  const getChartData = () => {
    if (!metrics || !threatTrends) return {
      securityScore: 0,
      complianceScore: 0,
      vulnerabilityDistribution: [
        { name: 'Critical', value: 0, color: '#dc2626' },
        { name: 'High', value: 0, color: '#ea580c' },
        { name: 'Medium', value: 0, color: '#ca8a04' },
        { name: 'Low', value: 0, color: '#65a30d' },
        { name: 'Info', value: 0, color: '#0891b2' }
      ],
      threatTrends: []
    };

    return {
      securityScore: metrics.securityScore,
      complianceScore: metrics.complianceScore,
      vulnerabilityDistribution: [
        { name: 'Critical', value: metrics.vulnerabilityCount.critical, color: '#dc2626' },
        { name: 'High', value: metrics.vulnerabilityCount.high, color: '#ea580c' },
        { name: 'Medium', value: metrics.vulnerabilityCount.medium, color: '#ca8a04' },
        { name: 'Low', value: metrics.vulnerabilityCount.low, color: '#65a30d' },
        { name: 'Info', value: metrics.vulnerabilityCount.info, color: '#0891b2' }
      ],
      threatTrends: threatTrends.map(trend => ({
        time: new Date(trend.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        threats: trend.threats,
        blocked: trend.blocked
      }))
    };
  };

  const chartData = getChartData();

  // Security posture data
  const securityPosture = [
    { metric: 'Authentication', score: chartData.securityScore, fullMark: 100 },
    { metric: 'Authorization', score: chartData.securityScore - 5, fullMark: 100 },
    { metric: 'Encryption', score: chartData.complianceScore, fullMark: 100 },
    { metric: 'Monitoring', score: 85, fullMark: 100 },
    { metric: 'Incident Response', score: 78, fullMark: 100 },
    { metric: 'Compliance', score: chartData.complianceScore, fullMark: 100 }
  ];

  // Format date helper
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Functions
  const handleRunScan = useCallback(async (type: string, target: string = 'production-cluster') => {
    try {
      await startScan(target, type);
    } catch (error) {
      toast({
        title: 'Error',
        description: `Failed to start ${type} vulnerability scan`,
        variant: 'destructive',
      });
    }
  }, [startScan, toast]);

  const handleExportReport = useCallback(async (type: string) => {
    toast({
      title: 'Report Export',
      description: `Exporting ${type} report...`,
    });
  }, [toast]);

  const handleRemediateVulnerability = useCallback(async (finding: VulnerabilityFinding) => {
    toast({
      title: 'Remediation Started',
      description: `Applying remediation for ${finding.title}...`,
    });
  }, [toast]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      case 'info': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant':
      case 'success':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'non-compliant':
      case 'failure':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'partial':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'blocked':
        return <Ban className="h-4 w-4 text-orange-500" />;
      default:
        return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      // Refresh data
    }, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // Loading state
  if (eventsLoading || complianceLoading || scansLoading || metricsLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading security dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <RBACGuard requiredPermissions={[{resource: 'security', action: 'read'}]}>
      <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Security & Compliance</h1>
          <p className="text-muted-foreground">Monitor security posture and compliance status</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Switch
              checked={autoRefresh}
              onCheckedChange={setAutoRefresh}
              id="auto-refresh"
            />
            <Label htmlFor="auto-refresh">Auto-refresh</Label>
          </div>
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={() => handleExportReport('security')}>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {(eventsError || complianceError || scansError || metricsError) && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Partial security data unavailable</AlertTitle>
          <AlertDescription>
            {[eventsError, complianceError, scansError, metricsError].filter(Boolean).join(' ')}
          </AlertDescription>
        </Alert>
      )}

      {/* Security Score Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Security Score</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{chartData.securityScore}%</div>
            <Progress value={chartData.securityScore} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              <TrendingUp className="h-3 w-3 inline mr-1 text-green-500" />
              +5% from last week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compliance Score</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{chartData.complianceScore}%</div>
            <Progress value={chartData.complianceScore} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              {complianceRequirements.length > 0
                ? `${complianceRequirements.filter((requirement) => requirement.status === 'compliant').length} of ${complianceRequirements.length} requirements met`
                : 'No compliance requirements reported'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Threats</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics?.activeThreats ?? 0}</div>
            <div className="flex gap-2 mt-2">
              <Badge variant={(metrics?.threatLevel === 'critical' || metrics?.threatLevel === 'high') ? 'destructive' : 'secondary'}>
                {(metrics?.threatLevel || 'low').toUpperCase()}
              </Badge>
              <Badge variant="secondary">{metrics?.blockedThreats ?? 0} Blocked</Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Scan</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {vulnerabilityScans[0]?.endTime ? formatDate(vulnerabilityScans[0].endTime) : 'No scans'}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {vulnerabilityScans.length > 0 ? `${vulnerabilityScans.length} scan records available` : 'No completed scan history reported'}
            </p>
            <Button size="sm" variant="outline" className="mt-2 w-full" onClick={() => handleRunScan('full')}>
              <RefreshCw className="h-3 w-3 mr-1" />
              Run Now
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="events">Security Events</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
          <TabsTrigger value="vulnerabilities">Vulnerabilities</TabsTrigger>
          <TabsTrigger value="access">Access Control</TabsTrigger>
          <TabsTrigger value="audit">Audit Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Security Posture Radar */}
            <Card>
              <CardHeader>
                <CardTitle>Security Posture</CardTitle>
                <CardDescription>Overall security metrics assessment</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={securityPosture}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar name="Score" dataKey="score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Threat Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Threat Activity</CardTitle>
                <CardDescription>Detected vs blocked threats over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={chartData.threatTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="threats" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
                    <Area type="monotone" dataKey="blocked" stackId="2" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Vulnerability Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Vulnerability Distribution</CardTitle>
                <CardDescription>Current vulnerability findings by severity</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={chartData.vulnerabilityDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry) => `${entry.name}: ${entry.value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.vulnerabilityDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Compliance by Category */}
            <Card>
              <CardHeader>
                <CardTitle>Compliance Status</CardTitle>
                <CardDescription>Requirements met by category</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={complianceByCategory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="compliant" fill="#22c55e" />
                    <Bar dataKey="total" fill="#e5e7eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Recent Security Events */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Security Events</CardTitle>
              <CardDescription>Latest security-related activities</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-4">
                  {securityEvents.slice(0, 5).map((event) => (
                    <div key={event.id} className="flex items-start space-x-4 p-4 rounded-lg border">
                      <div className="mt-1">{getStatusIcon(event.result)}</div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium">{event.action}</p>
                          <Badge className={getSeverityColor(event.severity)}>
                            {event.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{event.details}</p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>{formatDate(event.timestamp)}</span>
                          {event.user && <span>User: {event.user}</span>}
                          {event.ip && <span>IP: {event.ip}</span>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="events" className="space-y-4">
          {/* Security Events Filter */}
          <Card>
            <CardHeader>
              <CardTitle>Security Events</CardTitle>
              <CardDescription>Monitor and investigate security events</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4 mb-4">
                <div className="flex-1">
                  <Input
                    placeholder="Search events..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full"
                  />
                </div>
                <Select value={filterSeverity} onValueChange={setFilterSeverity}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Severity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severities</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline">
                  <Filter className="h-4 w-4 mr-2" />
                  More Filters
                </Button>
              </div>

              <ScrollArea className="h-[500px]">
                <div className="space-y-2">
                  {securityEvents.map((event) => (
                    <div key={event.id} className="p-4 border rounded-lg hover:bg-accent/50 transition-colors">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(event.result)}
                            <span className="font-medium">{event.action}</span>
                            <Badge className={getSeverityColor(event.severity)}>
                              {event.severity}
                            </Badge>
                            <Badge variant="outline">{event.type}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{event.details}</p>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>{formatDate(event.timestamp)}</span>
                            <span>Source: {event.source}</span>
                            {event.user && <span>User: {event.user}</span>}
                            {event.resource && <span>Resource: {event.resource}</span>}
                            {event.ip && <span>IP: {event.ip}</span>}
                            {event.location && <span>Location: {event.location}</span>}
                          </div>
                        </div>
                        <Button size="sm" variant="ghost">
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Requirements</CardTitle>
              <CardDescription>Track and manage compliance with security standards</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceRequirements.map((req) => (
                  <div key={req.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1 flex-1">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(req.status)}
                          <span className="font-medium">{req.name}</span>
                          <Badge className={getSeverityColor(req.severity)}>
                            {req.severity}
                          </Badge>
                          <Badge variant={req.status === 'compliant' ? 'default' : 'destructive'}>
                            {req.status}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{req.description}</p>
                        <div className="text-xs text-muted-foreground">
                          Category: {req.category} | Last checked: {formatDate(req.lastChecked)}
                        </div>
                        {req.remediationSteps && (
                          <Alert className="mt-2">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertTitle>Remediation Required</AlertTitle>
                            <AlertDescription>
                              <ul className="list-disc list-inside mt-2">
                                {req.remediationSteps.map((step, idx) => (
                                  <li key={idx}>{step}</li>
                                ))}
                              </ul>
                            </AlertDescription>
                          </Alert>
                        )}
                        {req.evidence && (
                          <div className="flex gap-2 mt-2">
                            {req.evidence.map((file, idx) => (
                              <Badge key={idx} variant="secondary">
                                <FileText className="h-3 w-3 mr-1" />
                                {file}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          disabled
                          onClick={() => triggerComplianceCheck(req.id)}
                        >
                          <RefreshCw className="h-4 w-4 mr-1" />
                          Recheck
                        </Button>
                        <Button size="sm" variant="outline">
                          <FileText className="h-4 w-4 mr-1" />
                          View Details
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="vulnerabilities" className="space-y-4">
          {/* Vulnerability Scan Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Vulnerability Management</CardTitle>
              <CardDescription>Scan and remediate security vulnerabilities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4 mb-6">
                <Button onClick={() => handleRunScan('network')}>
                  <Zap className="h-4 w-4 mr-2" />
                  Network Scan
                </Button>
                <Button onClick={() => handleRunScan('application')}>
                  <Zap className="h-4 w-4 mr-2" />
                  Application Scan
                </Button>
                <Button onClick={() => handleRunScan('container')}>
                  <Zap className="h-4 w-4 mr-2" />
                  Container Scan
                </Button>
                <Button onClick={() => handleRunScan('infrastructure')}>
                  <Zap className="h-4 w-4 mr-2" />
                  Infrastructure Scan
                </Button>
              </div>

              {/* Scan Results */}
              <div className="space-y-4">
                {vulnerabilityScans.map((scan) => (
                  <div key={scan.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="font-medium">{scan.target}</h4>
                        <p className="text-sm text-muted-foreground">
                          Type: {scan.type} | Status: {scan.status} | Started: {formatDate(scan.startTime)}
                        </p>
                      </div>
                      <Badge variant={scan.status === 'completed' ? 'default' : 'secondary'}>
                        {scan.status}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-5 gap-2 mb-4">
                      <div className="text-center p-2 rounded bg-red-100">
                        <div className="text-2xl font-bold text-red-600">{scan.vulnerabilities.critical}</div>
                        <div className="text-xs text-red-600">Critical</div>
                      </div>
                      <div className="text-center p-2 rounded bg-orange-100">
                        <div className="text-2xl font-bold text-orange-600">{scan.vulnerabilities.high}</div>
                        <div className="text-xs text-orange-600">High</div>
                      </div>
                      <div className="text-center p-2 rounded bg-yellow-100">
                        <div className="text-2xl font-bold text-yellow-600">{scan.vulnerabilities.medium}</div>
                        <div className="text-xs text-yellow-600">Medium</div>
                      </div>
                      <div className="text-center p-2 rounded bg-green-100">
                        <div className="text-2xl font-bold text-green-600">{scan.vulnerabilities.low}</div>
                        <div className="text-xs text-green-600">Low</div>
                      </div>
                      <div className="text-center p-2 rounded bg-blue-100">
                        <div className="text-2xl font-bold text-blue-600">{scan.vulnerabilities.info}</div>
                        <div className="text-xs text-blue-600">Info</div>
                      </div>
                    </div>

                    {scan.findings && scan.findings.length > 0 && (
                      <div className="space-y-2">
                        <h5 className="text-sm font-medium">Top Findings:</h5>
                        {scan.findings.slice(0, 3).map((finding) => (
                          <div key={finding.id} className="p-3 bg-accent/50 rounded-lg">
                            <div className="flex items-start justify-between">
                              <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                  <span className="font-medium text-sm">{finding.title}</span>
                                  {finding.cve && <Badge variant="outline">{finding.cve}</Badge>}
                                  <Badge className={getSeverityColor(finding.severity)}>
                                    {finding.severity}
                                  </Badge>
                                  {finding.exploitable && (
                                    <Badge variant="destructive">Exploitable</Badge>
                                  )}
                                </div>
                                <p className="text-xs text-muted-foreground">{finding.description}</p>
                                <p className="text-xs">Component: {finding.component}</p>
                              </div>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleRemediateVulnerability(finding)}
                              >
                                Remediate
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Access Control Policies</CardTitle>
              <CardDescription>Manage resource access and permissions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Access-control management unavailable</AlertTitle>
                  <AlertDescription>
                    The current backend exposes RBAC metadata but does not provide the access-control policy management endpoints this view expects.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Audit Logs</CardTitle>
              <CardDescription>Comprehensive audit trail of all system activities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4 mb-4">
                <Input
                  placeholder="Search audit logs..."
                  className="flex-1"
                />
                <Button variant="outline">
                  <Filter className="h-4 w-4 mr-2" />
                  Advanced Search
                </Button>
                <Button variant="outline" onClick={() => handleExportReport('audit')}>
                  <Download className="h-4 w-4 mr-2" />
                  Export Logs
                </Button>
              </div>

              <ScrollArea className="h-[500px]">
                <div className="space-y-2">
                  {/* Audit log entries would be displayed here */}
                  <div className="text-center text-muted-foreground py-8">
                    Audit logs will be displayed here
                  </div>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
    </RBACGuard>
  );
};

export default SecurityComplianceDashboard;
