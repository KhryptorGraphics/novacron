"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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

// Mock security data
const securityMetrics = {
  threatLevel: "medium",
  totalAlerts: 47,
  activeThreats: 3,
  blockedAttacks: 156,
  securityScore: 85,
  vulnerabilities: {
    critical: 2,
    high: 5,
    medium: 12,
    low: 8
  }
};

const securityAlerts = [
  {
    id: 1,
    type: "authentication",
    severity: "high",
    title: "Multiple Failed Login Attempts",
    description: "User admin@company.com has 15 failed login attempts from IP 203.0.113.42",
    timestamp: "2024-08-24T14:30:00Z",
    status: "active",
    source: "203.0.113.42",
    user: "admin@company.com"
  },
  {
    id: 2,
    type: "access",
    severity: "medium",
    title: "Unusual API Access Pattern",
    description: "High-frequency API requests detected from new location",
    timestamp: "2024-08-24T13:45:00Z",
    status: "investigating",
    source: "198.51.100.15",
    user: "api-client-001"
  },
  {
    id: 3,
    type: "system",
    severity: "critical",
    title: "Unauthorized Admin Access Attempt",
    description: "Attempt to access admin endpoints without proper authorization",
    timestamp: "2024-08-24T12:15:00Z",
    status: "blocked",
    source: "192.0.2.100",
    user: "unknown"
  },
  {
    id: 4,
    type: "data",
    severity: "low",
    title: "Large Data Export",
    description: "User exported unusually large dataset",
    timestamp: "2024-08-24T11:20:00Z",
    status: "resolved",
    source: "10.0.0.45",
    user: "data-analyst@company.com"
  }
];

const activeSessions = [
  {
    id: 1,
    user: "admin@novacron.io",
    ip: "192.168.1.100",
    location: "San Francisco, CA",
    device: "Chrome on Windows",
    started: "2024-08-24T09:00:00Z",
    lastActivity: "2024-08-24T14:25:00Z",
    risk: "low"
  },
  {
    id: 2,
    user: "manager@company.com",
    ip: "203.0.113.42",
    location: "New York, NY",
    device: "Safari on macOS",
    started: "2024-08-24T10:30:00Z",
    lastActivity: "2024-08-24T14:20:00Z",
    risk: "medium"
  },
  {
    id: 3,
    user: "operator@startup.io",
    ip: "198.51.100.75",
    location: "London, UK",
    device: "Firefox on Linux",
    started: "2024-08-24T08:15:00Z",
    lastActivity: "2024-08-24T14:10:00Z",
    risk: "high"
  }
];

const blockedIPs = [
  {
    ip: "192.0.2.100",
    reason: "Brute force attack",
    blockedAt: "2024-08-24T12:15:00Z",
    attempts: 25,
    status: "permanent"
  },
  {
    ip: "203.0.113.99",
    reason: "Suspicious activity",
    blockedAt: "2024-08-24T11:45:00Z",
    attempts: 8,
    status: "temporary"
  },
  {
    ip: "198.51.100.200",
    reason: "Rate limit exceeded",
    blockedAt: "2024-08-24T10:30:00Z",
    attempts: 1000,
    status: "temporary"
  }
];

export function SecurityDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAlert, setSelectedAlert] = useState<any>(null);

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