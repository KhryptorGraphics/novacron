'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Activity, Server, HardDrive, Network, Shield, AlertCircle,
  Monitor, Database, Lock, Settings, TrendingUp, Users,
  Cpu, MemoryStick, HardDriveIcon, WifiIcon, Clock, CheckCircle,
  BarChart3, Brain, Globe, Smartphone, Zap
} from 'lucide-react';

// Dynamically import components to avoid SSR issues
const VMOperationsDashboard = dynamic(
  () => import('@/components/vm/VMOperationsDashboard'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading VM Operations...</p>
        </div>
      </div>
    )
  }
);

const NetworkTopology = dynamic(
  () => import('@/components/visualizations/NetworkTopology').then(mod => ({ default: mod.NetworkTopology })),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Network Topology...</p>
        </div>
      </div>
    )
  }
);

const RealTimeMonitoringDashboard = dynamic(
  () => import('@/components/monitoring/RealTimeMonitoringDashboard'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Monitoring Dashboard...</p>
        </div>
      </div>
    )
  }
);

const StorageManagementUI = dynamic(
  () => import('@/components/storage/StorageManagementUI'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Storage Management...</p>
        </div>
      </div>
    )
  }
);

const NetworkConfigurationPanel = dynamic(
  () => import('@/components/network/NetworkConfigurationPanel'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Network Configuration...</p>
        </div>
      </div>
    )
  }
);

const SecurityComplianceDashboard = dynamic(
  () => import('@/components/security/SecurityComplianceDashboard'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Security Dashboard...</p>
        </div>
      </div>
    )
  }
);

// New distributed monitoring components
const BandwidthMonitoringDashboard = dynamic(
  () => import('@/components/monitoring/BandwidthMonitoringDashboard'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Bandwidth Monitoring...</p>
        </div>
      </div>
    )
  }
);

const PerformancePredictionDashboard = dynamic(
  () => import('@/components/monitoring/PerformancePredictionDashboard'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Performance Predictions...</p>
        </div>
      </div>
    )
  }
);

const SupercomputeFabricDashboard = dynamic(
  () => import('@/components/monitoring/SupercomputeFabricDashboard'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Supercompute Fabric...</p>
        </div>
      </div>
    )
  }
);

const MobileResponsiveAdmin = dynamic(
  () => import('@/components/mobile/MobileResponsiveAdmin'),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <p>Loading Mobile Interface...</p>
        </div>
      </div>
    )
  }
);

export default function UnifiedDashboard() {
  const [activeView, setActiveView] = useState('overview');

  // Mock data for overview - would come from API
  const systemMetrics = {
    totalVMs: 47,
    runningVMs: 42,
    stoppedVMs: 5,
    cpuUsage: 68,
    memoryUsage: 72,
    storageUsage: 54,
    networkThroughput: 8.4,
    activeAlerts: 3,
    criticalAlerts: 0,
    securityScore: 87,
    complianceScore: 92,
    backupStatus: 'healthy',
    lastBackup: '2 hours ago',
    // Distributed system metrics
    connectedClusters: 5,
    federationHealth: 'excellent',
    globalBandwidth: 45.2,
    activeMigrations: 2,
    computeJobs: 18,
    predictionAccuracy: 94,
    fabricUtilization: 73
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">NovaCron Control Center</h1>
              <p className="text-muted-foreground">Unified VM Management & Operations Platform</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="px-3 py-1">
                <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
                System Healthy
              </Badge>
              <Badge variant="outline" className="px-3 py-1">
                <Clock className="h-3 w-3 mr-1" />
                {new Date().toLocaleTimeString()}
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="border-b bg-card">
        <div className="container mx-auto px-6 py-3">
          <div className="flex gap-2 overflow-x-auto">
            <Button 
              variant={activeView === 'overview' ? 'default' : 'ghost'}
              onClick={() => setActiveView('overview')}
              className="flex items-center gap-2"
            >
              <Monitor className="h-4 w-4" />
              Overview
            </Button>
            <Button 
              variant={activeView === 'vms' ? 'default' : 'ghost'}
              onClick={() => setActiveView('vms')}
              className="flex items-center gap-2"
            >
              <Server className="h-4 w-4" />
              Virtual Machines
            </Button>
            <Button 
              variant={activeView === 'monitoring' ? 'default' : 'ghost'}
              onClick={() => setActiveView('monitoring')}
              className="flex items-center gap-2"
            >
              <Activity className="h-4 w-4" />
              Monitoring
            </Button>
            <Button 
              variant={activeView === 'storage' ? 'default' : 'ghost'}
              onClick={() => setActiveView('storage')}
              className="flex items-center gap-2"
            >
              <HardDrive className="h-4 w-4" />
              Storage
            </Button>
            <Button 
              variant={activeView === 'network' ? 'default' : 'ghost'}
              onClick={() => setActiveView('network')}
              className="flex items-center gap-2"
            >
              <Network className="h-4 w-4" />
              Network
            </Button>
            <Button
              variant={activeView === 'security' ? 'default' : 'ghost'}
              onClick={() => setActiveView('security')}
              className="flex items-center gap-2"
            >
              <Shield className="h-4 w-4" />
              Security
            </Button>
            <Button
              variant={activeView === 'bandwidth' ? 'default' : 'ghost'}
              onClick={() => setActiveView('bandwidth')}
              className="flex items-center gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              Bandwidth
            </Button>
            <Button
              variant={activeView === 'predictions' ? 'default' : 'ghost'}
              onClick={() => setActiveView('predictions')}
              className="flex items-center gap-2"
            >
              <Brain className="h-4 w-4" />
              AI Predictions
            </Button>
            <Button
              variant={activeView === 'fabric' ? 'default' : 'ghost'}
              onClick={() => setActiveView('fabric')}
              className="flex items-center gap-2"
            >
              <Globe className="h-4 w-4" />
              Fabric
            </Button>
            <Button
              variant={activeView === 'topology' ? 'default' : 'ghost'}
              onClick={() => setActiveView('topology')}
              className="flex items-center gap-2"
            >
              <Network className="h-4 w-4" />
              Topology
            </Button>
            <Button
              variant={activeView === 'mobile' ? 'default' : 'ghost'}
              onClick={() => setActiveView('mobile')}
              className="flex items-center gap-2"
            >
              <Smartphone className="h-4 w-4" />
              Mobile
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-6">
        {activeView === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total VMs</CardTitle>
                  <Server className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.totalVMs}</div>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="secondary" className="text-xs">
                      {systemMetrics.runningVMs} Running
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {systemMetrics.stoppedVMs} Stopped
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.cpuUsage}%</div>
                  <Progress value={systemMetrics.cpuUsage} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
                  <MemoryStick className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.memoryUsage}%</div>
                  <Progress value={systemMetrics.memoryUsage} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Storage Usage</CardTitle>
                  <HardDriveIcon className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.storageUsage}%</div>
                  <Progress value={systemMetrics.storageUsage} className="mt-2" />
                </CardContent>
              </Card>
            </div>

            {/* Distributed System Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Connected Clusters</CardTitle>
                  <Globe className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.connectedClusters}</div>
                  <Badge variant="secondary" className="text-xs mt-2">
                    {systemMetrics.federationHealth}
                  </Badge>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Global Bandwidth</CardTitle>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.globalBandwidth} Gbps</div>
                  <Progress value={systemMetrics.fabricUtilization} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Migrations</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.activeMigrations}</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Cross-cluster operations
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Compute Jobs</CardTitle>
                  <Server className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.computeJobs}</div>
                  <Badge variant="outline" className="text-xs mt-2">
                    Running
                  </Badge>
                </CardContent>
              </Card>
            </div>

            {/* Secondary Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Security Score</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.securityScore}%</div>
                  <Progress value={systemMetrics.securityScore} className="mt-2" />
                  <p className="text-xs text-muted-foreground mt-2">
                    <TrendingUp className="h-3 w-3 inline mr-1 text-green-500" />
                    +5% from last week
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
                  <AlertCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.activeAlerts}</div>
                  <div className="flex gap-2 mt-2">
                    <Badge variant={systemMetrics.criticalAlerts > 0 ? 'destructive' : 'secondary'}>
                      {systemMetrics.criticalAlerts} Critical
                    </Badge>
                    <Badge variant="outline">
                      {systemMetrics.activeAlerts - systemMetrics.criticalAlerts} Warning
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">AI Prediction Accuracy</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemMetrics.predictionAccuracy}%</div>
                  <Progress value={systemMetrics.predictionAccuracy} className="mt-2" />
                  <p className="text-xs text-muted-foreground mt-2">
                    <TrendingUp className="h-3 w-3 inline mr-1 text-green-500" />
                    Model performance
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Common operations and shortcuts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Button variant="outline" className="h-auto py-4 flex flex-col gap-2">
                    <Server className="h-5 w-5" />
                    <span className="text-xs">Create VM</span>
                  </Button>
                  <Button variant="outline" className="h-auto py-4 flex flex-col gap-2">
                    <Database className="h-5 w-5" />
                    <span className="text-xs">Backup All</span>
                  </Button>
                  <Button variant="outline" className="h-auto py-4 flex flex-col gap-2">
                    <Shield className="h-5 w-5" />
                    <span className="text-xs">Security Scan</span>
                  </Button>
                  <Button variant="outline" className="h-auto py-4 flex flex-col gap-2">
                    <Activity className="h-5 w-5" />
                    <span className="text-xs">Health Check</span>
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>Latest system events and operations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-4 p-3 rounded-lg bg-accent/50">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">VM "web-server-01" successfully migrated</p>
                      <p className="text-xs text-muted-foreground">5 minutes ago</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 p-3 rounded-lg bg-accent/50">
                    <Database className="h-4 w-4 text-blue-500" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">Backup completed for storage pool "primary"</p>
                      <p className="text-xs text-muted-foreground">2 hours ago</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 p-3 rounded-lg bg-accent/50">
                    <Shield className="h-4 w-4 text-purple-500" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">Security scan completed - no vulnerabilities found</p>
                      <p className="text-xs text-muted-foreground">4 hours ago</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeView === 'vms' && <VMOperationsDashboard />}
        {activeView === 'monitoring' && <RealTimeMonitoringDashboard />}
        {activeView === 'storage' && <StorageManagementUI />}
        {activeView === 'network' && <NetworkConfigurationPanel />}
        {activeView === 'security' && <SecurityComplianceDashboard />}
        {activeView === 'bandwidth' && <BandwidthMonitoringDashboard />}
        {activeView === 'predictions' && <PerformancePredictionDashboard />}
        {activeView === 'fabric' && <SupercomputeFabricDashboard />}
        {activeView === 'topology' && (
          <NetworkTopology
            title="Distributed Network Topology"
            description="Real-time visualization of the distributed system topology"
            height={600}
            showDistributed={true}
            showBandwidth={true}
            showPerformanceMetrics={true}
            autoRefresh={true}
          />
        )}
        {activeView === 'mobile' && <MobileResponsiveAdmin />}
      </div>
    </div>
  );
}