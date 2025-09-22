"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api/client";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { FadeIn } from "@/lib/animations";
import { 
  Shield, 
  AlertTriangle, 
  Lock, 
  Unlock,
  Eye,
  EyeOff,
  Ban,
  CheckCircle,
  XCircle,
  Clock,
  Globe,
  User,
  Key,
  Activity,
  TrendingUp,
  TrendingDown,
  MapPin,
  Smartphone,
  Monitor,
  Wifi,
  FileText,
  Download
} from "lucide-react";
import { cn } from "@/lib/utils";

// Security data types
interface SecurityMetrics {
  threatLevel: string;
  totalAlerts: number;
  activeThreats: number;
  blockedAttacks: number;
  securityScore: number;
  vulnerabilities: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

// Mock data arrays removed - now fetched from API



export function SecurityDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAlert, setSelectedAlert] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics>({
    threatLevel: "low",
    totalAlerts: 0,
    activeThreats: 0,
    blockedAttacks: 0,
    securityScore: 100,
    vulnerabilities: {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0
    }
  });
  const [securityAlerts, setSecurityAlerts] = useState<any[]>([]);
  const [activeSessions, setActiveSessions] = useState<any[]>([]);
  const [blockedIPs, setBlockedIPs] = useState<any[]>([]);

  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/api/security/events/stream');

  // Fetch security data
  const fetchSecurityData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all security data in parallel
      const [threatsResponse, vulnsResponse, complianceResponse, incidentsResponse, auditStatsResponse] = await Promise.all([
        apiClient.get('/api/security/threats'),
        apiClient.get('/api/security/vulnerabilities'),
        apiClient.get('/api/security/compliance'),
        apiClient.get('/api/security/incidents'),
        apiClient.get('/api/security/audit/statistics')
      ]);

      // Destructure the actual response shapes from backend
      const { threats = [] } = threatsResponse.data || {};
      const { vulnerabilities = [], summary: vulnsSummary = {} } = vulnsResponse.data || {};
      const compliance = complianceResponse.data || {};
      const { incidents = [] } = incidentsResponse.data || {};
      const auditStats = auditStatsResponse.data || {};

      // Update metrics using the correct response structure
      setSecurityMetrics({
        threatLevel: threats.length > 10 ? "high" : threats.length > 5 ? "medium" : "low",
        totalAlerts: threats.length + incidents.length,
        activeThreats: threats.filter((t: any) => t.status === 'active').length,
        blockedAttacks: threats.filter((t: any) => t.status === 'blocked').length,
        securityScore: compliance.score || auditStats.overallScore || 85,
        vulnerabilities: {
          critical: vulnsSummary.critical || vulnerabilities.filter((v: any) => v.severity === 'critical').length,
          high: vulnsSummary.high || vulnerabilities.filter((v: any) => v.severity === 'high').length,
          medium: vulnsSummary.medium || vulnerabilities.filter((v: any) => v.severity === 'medium').length,
          low: vulnsSummary.low || vulnerabilities.filter((v: any) => v.severity === 'low').length
        }
      });

      // Update alerts
      setSecurityAlerts([...threats, ...incidents].slice(0, 10));

      // Fetch session and IP data
      const [eventsResponse] = await Promise.all([
        apiClient.get('/api/security/events')
      ]);

      const { events = [] } = eventsResponse.data || {};

      // Extract active sessions from events (assuming events contain session info)
      const sessionEvents = events.filter((e: any) => e.type === 'session' || e.type === 'login');
      setActiveSessions(sessionEvents.map((e: any) => ({
        id: e.id,
        user: e.user || e.actor || 'Unknown',
        ip: e.source_ip || e.ip || 'Unknown',
        location: e.location || 'Unknown',
        device: e.user_agent || e.device || 'Unknown',
        started: e.timestamp || new Date().toISOString(),
        lastActivity: e.timestamp || new Date().toISOString(),
        risk: e.risk_level || 'low'
      })).slice(0, 10));

      // Extract blocked IPs from threats/incidents
      const blockedEvents = [...threats, ...incidents].filter((e: any) => e.status === 'blocked');
      setBlockedIPs(blockedEvents.map((e: any) => ({
        ip: e.source_ip || e.source || 'Unknown',
        reason: e.description || e.title || 'Security violation',
        blockedAt: e.timestamp || new Date().toISOString(),
        attempts: e.attempt_count || 1,
        status: e.block_type || 'temporary'
      })).slice(0, 10));

    } catch (err) {
      console.error('Failed to fetch security data:', err);
      setError('Failed to load security data');

      // Set fallback empty data
      setActiveSessions([]);
      setBlockedIPs([]);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchSecurityData();
  }, []);

  // Handle WebSocket updates
  useEffect(() => {
    if (!wsData) return;

    // The backend sends summary objects, not individual events with type field
    // Check if it's a summary update
    if (wsData.threats !== undefined || wsData.incidents !== undefined) {
      // This is a summary update from StreamSecurityEvents
      fetchSecurityData(); // Refresh all data
    } else if (wsData.type) {
      // Handle specific event types if backend is updated to send them
      if (wsData.type === 'threat_detected') {
        setSecurityMetrics(prev => ({
          ...prev,
          totalAlerts: prev.totalAlerts + 1,
          activeThreats: prev.activeThreats + 1
        }));

        // Add to alerts
        setSecurityAlerts(prev => [wsData, ...prev].slice(0, 10));
      }

      if (wsData.type === 'threat_resolved') {
        setSecurityMetrics(prev => ({
          ...prev,
          activeThreats: Math.max(0, prev.activeThreats - 1),
          blockedAttacks: prev.blockedAttacks + 1
        }));
      }

      if (wsData.type === 'vulnerability_found') {
        const severity = wsData.severity || 'medium';
        setSecurityMetrics(prev => ({
          ...prev,
          vulnerabilities: {
            ...prev.vulnerabilities,
            [severity]: (prev.vulnerabilities[severity as keyof typeof prev.vulnerabilities] || 0) + 1
          }
        }));
      }
    }
  }, [wsData]);

  // Export audit data
  const exportAuditData = async () => {
    try {
      const response = await apiClient.get('/api/security/audit/export', {
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `audit-export-${new Date().toISOString()}.json`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error('Failed to export audit data:', err);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "bg-red-500";
      case "high": return "bg-orange-500";
      case "medium": return "bg-yellow-500";
      case "low": return "bg-blue-500";
      default: return "bg-gray-500";
    }
  };

  const getSeverityVariant = (severity: string) => {
    switch (severity) {
      case "critical": return "destructive";
      case "high": return "destructive";
      case "medium": return "secondary";
      case "low": return "outline";
      default: return "outline";
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high": return "text-red-600";
      case "medium": return "text-yellow-600";
      case "low": return "text-green-600";
      default: return "text-gray-600";
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getTimeAgo = (dateString: string) => {
    const now = new Date();
    const past = new Date(dateString);
    const diff = Math.floor((now.getTime() - past.getTime()) / 1000 / 60); // minutes
    
    if (diff < 60) return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
    return `${Math.floor(diff / 1440)}d ago`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="h-6 w-6" />
            Security Dashboard
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor security events, threats, and system protection
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge 
            variant={securityMetrics.threatLevel === "high" ? "destructive" : "secondary"}
            className="capitalize"
          >
            {securityMetrics.threatLevel} Threat Level
          </Badge>
          <Button variant="outline">
            <FileText className="h-4 w-4 mr-2" />
            Security Report
          </Button>
        </div>
      </div>

      {/* Security Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FadeIn delay={0.1}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Security Score</p>
                  <p className="text-2xl font-bold text-green-600">{securityMetrics.securityScore}%</p>
                </div>
                <Shield className="h-8 w-8 text-green-600" />
              </div>
              <Progress value={securityMetrics.securityScore} className="mt-3" />
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Threats</p>
                  <p className="text-2xl font-bold text-red-600">{securityMetrics.activeThreats}</p>
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
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Blocked Attacks</p>
                  <p className="text-2xl font-bold text-blue-600">{securityMetrics.blockedAttacks}</p>
                </div>
                <Ban className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Alerts</p>
                  <p className="text-2xl font-bold text-orange-600">{securityMetrics.totalAlerts}</p>
                </div>
                <Activity className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Security Alerts */}
        <div className="lg:col-span-2">
          <FadeIn delay={0.5}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Security Alerts
                </CardTitle>
                <CardDescription>Recent security events requiring attention</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {securityAlerts.map((alert) => (
                    <div 
                      key={alert.id} 
                      className="flex items-start gap-4 p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
                      onClick={() => setSelectedAlert(alert)}
                    >
                      <div className={cn("h-3 w-3 rounded-full mt-2", getSeverityColor(alert.severity))} />
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <h4 className="font-medium text-sm">{alert.title}</h4>
                          <div className="flex items-center gap-2 ml-4">
                            <Badge variant={getSeverityVariant(alert.severity) as any}>
                              {alert.severity}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {getTimeAgo(alert.timestamp)}
                            </span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {alert.description}
                        </p>
                        
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>Source: {alert.source}</span>
                          <span>User: {alert.user}</span>
                          <Badge 
                            variant="outline" 
                            className={cn(
                              "text-xs",
                              alert.status === "active" ? "text-red-600" :
                              alert.status === "blocked" ? "text-blue-600" :
                              alert.status === "resolved" ? "text-green-600" : "text-gray-600"
                            )}
                          >
                            {alert.status}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </FadeIn>
        </div>

        {/* Vulnerability Summary */}
        <div>
          <FadeIn delay={0.6}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <XCircle className="h-5 w-5" />
                  Vulnerabilities
                </CardTitle>
                <CardDescription>Security vulnerabilities by severity</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 bg-red-500 rounded-full" />
                    <span className="text-sm">Critical</span>
                  </div>
                  <Badge variant="destructive">{securityMetrics.vulnerabilities.critical}</Badge>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 bg-orange-500 rounded-full" />
                    <span className="text-sm">High</span>
                  </div>
                  <Badge variant="destructive">{securityMetrics.vulnerabilities.high}</Badge>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 bg-yellow-500 rounded-full" />
                    <span className="text-sm">Medium</span>
                  </div>
                  <Badge variant="secondary">{securityMetrics.vulnerabilities.medium}</Badge>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 bg-blue-500 rounded-full" />
                    <span className="text-sm">Low</span>
                  </div>
                  <Badge variant="outline">{securityMetrics.vulnerabilities.low}</Badge>
                </div>

                <Button className="w-full mt-4" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Vulnerability Report
                </Button>
              </CardContent>
            </Card>
          </FadeIn>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Sessions */}
        <FadeIn delay={0.7}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                Active Sessions
              </CardTitle>
              <CardDescription>Currently active user sessions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {activeSessions.map((session) => (
                  <div key={session.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{session.user}</span>
                        <Badge variant="outline" className={getRiskColor(session.risk)}>
                          {session.risk} risk
                        </Badge>
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        <div className="flex items-center gap-1">
                          <Globe className="h-3 w-3" />
                          {session.ip} • {session.location}
                        </div>
                        <div className="flex items-center gap-1 mt-1">
                          <Monitor className="h-3 w-3" />
                          {session.device}
                        </div>
                        <div className="flex items-center gap-1 mt-1">
                          <Clock className="h-3 w-3" />
                          Active: {getTimeAgo(session.lastActivity)}
                        </div>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="ml-2">
                      <Ban className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        {/* Blocked IPs */}
        <FadeIn delay={0.8}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Ban className="h-5 w-5" />
                Blocked IP Addresses
              </CardTitle>
              <CardDescription>Recently blocked suspicious IP addresses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {blockedIPs.map((blocked, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm">{blocked.ip}</span>
                        <Badge 
                          variant={blocked.status === "permanent" ? "destructive" : "secondary"}
                        >
                          {blocked.status}
                        </Badge>
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        <div>Reason: {blocked.reason}</div>
                        <div>Attempts: {blocked.attempts} • Blocked: {getTimeAgo(blocked.blockedAt)}</div>
                      </div>
                    </div>
                    <Button size="sm" variant="ghost" className="ml-2">
                      <Unlock className="h-3 w-3" />
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