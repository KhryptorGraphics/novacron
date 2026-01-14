"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Activity,
  Brain,
  Settings,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  MemoryStick,
  Network,
  HardDrive,
  Zap,
} from 'lucide-react';

import { PlacementDecisionChart } from './PlacementDecisionChart';
import { ScalingMetricsChart } from './ScalingMetricsChart';
import { PolicyManagementPanel } from './PolicyManagementPanel';
import { MLModelPerformancePanel } from './MLModelPerformancePanel';
import { RealTimeMetricsPanel } from './RealTimeMetricsPanel';

// Types for orchestration data
interface EngineStatus {
  state: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
  startTime: string;
  activePolicies: number;
  eventsProcessed: number;
  metrics: Record<string, any>;
}

interface OrchestrationDecision {
  id: string;
  decisionType: 'placement' | 'scaling' | 'healing' | 'migration' | 'optimization';
  recommendation: string;
  score: number;
  confidence: number;
  explanation: string;
  timestamp: string;
  status: 'pending' | 'executed' | 'failed' | 'cancelled';
}

interface OrchestrationPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: number;
  rules: PolicyRule[];
  createdAt: string;
  updatedAt: string;
}

interface PolicyRule {
  type: 'placement' | 'autoscaling' | 'healing' | 'loadbalance' | 'security' | 'compliance';
  enabled: boolean;
  priority: number;
  parameters: Record<string, any>;
}

interface MLModelMetrics {
  modelType: string;
  accuracy: number;
  throughput: number;
  latency: number;
  lastTraining: string;
  version: string;
}

export function OrchestrationDashboard() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [recentDecisions, setRecentDecisions] = useState<OrchestrationDecision[]>([]);
  const [activePolicies, setActivePolicies] = useState<OrchestrationPolicy[]>([]);
  const [mlModels, setMLModels] = useState<MLModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOrchestrationData = async () => {
      try {
        const [statusRes, decisionsRes, policiesRes, modelsRes] = await Promise.all([
          fetch('/api/orchestration/status'),
          fetch('/api/orchestration/decisions?limit=10'),
          fetch('/api/orchestration/policies'),
          fetch('/api/orchestration/ml-models'),
        ]);

        if (statusRes.ok) {
          const status = await statusRes.json();
          setEngineStatus(status);
        }

        if (decisionsRes.ok) {
          const decisions = await decisionsRes.json();
          setRecentDecisions(decisions);
        }

        if (policiesRes.ok) {
          const policies = await policiesRes.json();
          setActivePolicies(policies.filter((p: OrchestrationPolicy) => p.enabled));
        }

        if (modelsRes.ok) {
          const models = await modelsRes.json();
          setMLModels(models);
        }
      } catch (err) {
        setError('Failed to load orchestration data');
        console.error('Orchestration data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchOrchestrationData();
    
    // Set up polling for real-time updates
    const interval = setInterval(fetchOrchestrationData, 30000); // Every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'running': return 'success';
      case 'starting': return 'warning';
      case 'error': return 'destructive';
      default: return 'secondary';
    }
  };

  const getStatusIcon = (state: string) => {
    switch (state) {
      case 'running': return <CheckCircle className="h-4 w-4" />;
      case 'starting': return <Clock className="h-4 w-4" />;
      case 'error': return <AlertCircle className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading orchestration dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Engine Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Engine Status</CardTitle>
            {engineStatus && getStatusIcon(engineStatus.state)}
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <Badge variant={engineStatus ? getStatusColor(engineStatus.state) : 'secondary'}>
                {engineStatus?.state || 'Unknown'}
              </Badge>
            </div>
            {engineStatus?.startTime && (
              <p className="text-xs text-muted-foreground mt-1">
                Running since {new Date(engineStatus.startTime).toLocaleString()}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Policies</CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activePolicies.length}</div>
            <p className="text-xs text-muted-foreground">
              {engineStatus?.activePolicies || 0} total policies
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Events Processed</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{engineStatus?.eventsProcessed || 0}</div>
            <p className="text-xs text-muted-foreground">
              Last hour: {Math.floor((engineStatus?.eventsProcessed || 0) / 24)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Models</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mlModels.length}</div>
            <p className="text-xs text-muted-foreground">
              {mlModels.filter(m => m.accuracy > 0.85).length} performing well
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="decisions">Decisions</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="ml-models">ML Models</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PlacementDecisionChart decisions={recentDecisions} />
            <ScalingMetricsChart />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Recent Decisions</CardTitle>
                <CardDescription>Latest orchestration decisions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {recentDecisions.slice(0, 5).map((decision) => (
                    <div key={decision.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{decision.decisionType}</Badge>
                          <Badge variant={decision.status === 'executed' ? 'success' : 'warning'}>
                            {decision.status}
                          </Badge>
                        </div>
                        <p className="text-sm mt-1">{decision.recommendation}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                          <span>Score: {(decision.score * 100).toFixed(1)}%</span>
                          <span>Confidence: {(decision.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Policy Status</CardTitle>
                <CardDescription>Active orchestration policies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {activePolicies.slice(0, 5).map((policy) => (
                    <div key={policy.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{policy.name}</span>
                          <Badge variant="success">Active</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{policy.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                          <span>Priority: {policy.priority}</span>
                          <span>Rules: {policy.rules.length}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Performance</CardTitle>
                <CardDescription>ML model accuracy metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mlModels.map((model) => (
                    <div key={model.modelType} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{model.modelType}</span>
                        <Badge variant={model.accuracy > 0.85 ? 'success' : 'warning'}>
                          {(model.accuracy * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <Progress value={model.accuracy * 100} className="h-2" />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Throughput: {model.throughput.toFixed(0)} RPS</span>
                        <span>Latency: {model.latency.toFixed(0)}ms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Decisions Tab */}
        <TabsContent value="decisions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Orchestration Decisions</CardTitle>
              <CardDescription>Detailed view of all orchestration decisions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentDecisions.map((decision) => (
                  <div key={decision.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{decision.decisionType}</Badge>
                        <Badge variant={decision.status === 'executed' ? 'success' : decision.status === 'failed' ? 'destructive' : 'warning'}>
                          {decision.status}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {new Date(decision.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-sm">
                        <span>Score: <strong>{(decision.score * 100).toFixed(1)}%</strong></span>
                        <span>Confidence: <strong>{(decision.confidence * 100).toFixed(1)}%</strong></span>
                      </div>
                    </div>
                    
                    <h4 className="font-medium mb-2">{decision.recommendation}</h4>
                    <p className="text-sm text-muted-foreground mb-3">{decision.explanation}</p>
                    
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-muted-foreground">
                        Decision ID: {decision.id}
                      </div>
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Policies Tab */}
        <TabsContent value="policies">
          <PolicyManagementPanel policies={activePolicies} />
        </TabsContent>

        {/* ML Models Tab */}
        <TabsContent value="ml-models">
          <MLModelPerformancePanel models={mlModels} />
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics">
          <RealTimeMetricsPanel engineStatus={engineStatus} />
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Orchestration Settings</CardTitle>
              <CardDescription>Configure orchestration engine parameters</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Orchestration settings will be available in the next update.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}