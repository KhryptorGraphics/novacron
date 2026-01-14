import React, { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import {
  TrendingUp, TrendingDown, Brain, Target, Zap, AlertCircle,
  BarChart3, PieChart, Activity, Cpu, MemoryStick, HardDrive,
  Network, Clock, CheckCircle, XCircle, Lightbulb, ArrowRight
} from 'lucide-react';
import { PredictiveChart } from '@/components/visualizations/PredictiveChart';
import { usePerformancePredictionWebSocket } from '@/hooks/useWebSocket';
import type {
  ResourcePrediction,
  WorkloadPattern,
  MigrationPrediction,
  ScalingRecommendation,
  PerformanceOptimization,
  ModelMetrics
} from '@/lib/api/types';

interface PerformancePredictionDashboardProps {
  clusterId?: string;
  timeHorizon?: '1hr' | '6hr' | '24hr' | '7days' | '30days';
  confidenceThreshold?: number;
}

export const PerformancePredictionDashboard: React.FC<PerformancePredictionDashboardProps> = ({
  clusterId,
  timeHorizon: initialTimeHorizon = '24hr',
  confidenceThreshold: initialConfidenceThreshold = 80,
}) => {
  const [timeHorizon, setTimeHorizon] = useState(initialTimeHorizon);
  const [confidenceThreshold, setConfidenceThreshold] = useState(initialConfidenceThreshold);
  const [selectedScenario, setSelectedScenario] = useState<string>('current');
  const [whatIfParams, setWhatIfParams] = useState({
    cpuIncrease: 0,
    memoryIncrease: 0,
    workloadIncrease: 0,
    description: '',
  });

  // Real-time prediction data from WebSocket with typed payload
  const { data: predictionData, isConnected } = usePerformancePredictionWebSocket();

  // Mock prediction data
  const mockResourcePredictions: ResourcePrediction[] = [
    {
      resourceType: 'cpu',
      currentUsage: 68,
      predictedUsage: 82,
      confidence: 92,
      trend: 'increasing',
      timeToCapacity: '6 hours',
      recommendation: 'Consider scaling CPU resources or load balancing',
      factors: ['Increased workload', 'Peak hours approaching', 'Seasonal pattern'],
    },
    {
      resourceType: 'memory',
      currentUsage: 74,
      predictedUsage: 79,
      confidence: 87,
      trend: 'stable',
      timeToCapacity: '18 hours',
      recommendation: 'Memory usage stable, monitor for memory leaks',
      factors: ['Normal allocation pattern', 'GC efficiency stable'],
    },
    {
      resourceType: 'storage',
      currentUsage: 56,
      predictedUsage: 61,
      confidence: 94,
      trend: 'increasing',
      timeToCapacity: '3 days',
      recommendation: 'Storage growth is predictable, plan expansion',
      factors: ['Log accumulation', 'Data retention policy', 'Backup schedule'],
    },
    {
      resourceType: 'network',
      currentUsage: 45,
      predictedUsage: 52,
      confidence: 89,
      trend: 'increasing',
      timeToCapacity: '2 days',
      recommendation: 'Network bandwidth sufficient, monitor peak usage',
      factors: ['Data synchronization', 'Backup traffic', 'User activity'],
    },
  ];

  const mockWorkloadPatterns: WorkloadPattern[] = [
    {
      id: '1',
      name: 'Daily Peak Pattern',
      type: 'daily',
      confidence: 94,
      description: 'Workload peaks between 9 AM - 5 PM with 40% increase',
      peakHours: ['09:00', '17:00'],
      baselineMultiplier: 1.4,
      nextOccurrence: new Date(Date.now() + 86400000).toISOString(),
      impact: {
        cpu: 35,
        memory: 28,
        network: 45,
        storage: 12,
      },
    },
    {
      id: '2',
      name: 'Weekly Batch Processing',
      type: 'weekly',
      confidence: 87,
      description: 'High CPU usage every Sunday for batch processing',
      peakHours: ['02:00', '06:00'],
      baselineMultiplier: 2.1,
      nextOccurrence: new Date(Date.now() + 432000000).toISOString(),
      impact: {
        cpu: 110,
        memory: 65,
        network: 30,
        storage: 45,
      },
    },
    {
      id: '3',
      name: 'Month-End Reporting',
      type: 'monthly',
      confidence: 91,
      description: 'Intensive data processing for monthly reports',
      peakHours: ['20:00', '02:00'],
      baselineMultiplier: 1.8,
      nextOccurrence: new Date(2024, new Date().getMonth() + 1, 0).toISOString(),
      impact: {
        cpu: 80,
        memory: 90,
        network: 55,
        storage: 120,
      },
    },
  ];

  const mockMigrationPredictions: MigrationPrediction[] = [
    {
      vmId: 'vm-web-001',
      vmName: 'Web Server 01',
      sourceNode: 'node-01',
      targetNode: 'node-03',
      successProbability: 94,
      estimatedDuration: 240, // seconds
      expectedDowntime: 15, // seconds
      resourceImpact: {
        cpu: 12,
        memory: 8,
        network: 25,
      },
      recommendation: 'Optimal migration window: 2 AM - 4 AM',
      risks: ['Network congestion possible', 'Source node memory pressure'],
      optimalTime: new Date(Date.now() + 3600000 * 6).toISOString(),
    },
    {
      vmId: 'vm-db-002',
      vmName: 'Database Server 02',
      sourceNode: 'node-02',
      targetNode: 'node-04',
      successProbability: 87,
      estimatedDuration: 480,
      expectedDowntime: 45,
      resourceImpact: {
        cpu: 18,
        memory: 35,
        network: 40,
      },
      recommendation: 'Schedule during maintenance window',
      risks: ['Large memory footprint', 'Active database connections'],
      optimalTime: new Date(Date.now() + 3600000 * 18).toISOString(),
    },
  ];

  const mockScalingRecommendations: ScalingRecommendation[] = [
    {
      type: 'horizontal',
      resourceType: 'cpu',
      currentCapacity: 32,
      recommendedCapacity: 40,
      confidence: 91,
      costImpact: 250, // USD per month
      performanceGain: 25, // percentage
      timeline: '2 hours',
      reasoning: 'CPU utilization trending upward, additional cores needed for peak capacity',
      implementation: ['Add 2 new compute nodes', 'Migrate 3 VMs to distribute load'],
    },
    {
      type: 'vertical',
      resourceType: 'memory',
      currentCapacity: 128,
      recommendedCapacity: 192,
      confidence: 88,
      costImpact: 180,
      performanceGain: 35,
      timeline: '4 hours',
      reasoning: 'Memory pressure during peak hours, upgrade prevents swapping',
      implementation: ['Upgrade RAM on 4 critical nodes', 'Redistribute memory-intensive VMs'],
    },
    {
      type: 'storage',
      resourceType: 'storage',
      currentCapacity: 2000,
      recommendedCapacity: 3000,
      confidence: 95,
      costImpact: 150,
      performanceGain: 20,
      timeline: '1 day',
      reasoning: 'Storage capacity will reach 85% in 3 days based on current growth',
      implementation: ['Add SSD storage pool', 'Implement tiered storage'],
    },
  ];

  const mockOptimizations: PerformanceOptimization[] = [
    {
      id: '1',
      category: 'resource_allocation',
      title: 'Optimize VM Resource Distribution',
      description: 'Rebalance VMs across nodes to improve resource utilization',
      impact: 'High',
      effort: 'Medium',
      savings: 320, // USD per month
      performanceGain: 18,
      implementation: [
        'Move VM-WEB-03 from node-01 to node-04',
        'Adjust memory allocation for VM-DB-01',
        'Enable CPU affinity for critical workloads',
      ],
      estimatedTime: '2 hours',
      prerequisites: ['Maintenance window', 'Load balancer reconfiguration'],
    },
    {
      id: '2',
      category: 'workload_scheduling',
      title: 'Implement Intelligent Workload Scheduling',
      description: 'Schedule batch jobs during low-usage periods to optimize resource usage',
      impact: 'Medium',
      effort: 'Low',
      savings: 180,
      performanceGain: 12,
      implementation: [
        'Configure cron jobs for 2-4 AM window',
        'Set up resource quotas for batch workloads',
        'Enable workload prioritization',
      ],
      estimatedTime: '30 minutes',
      prerequisites: ['Job scheduler configuration'],
    },
    {
      id: '3',
      category: 'network_optimization',
      title: 'Optimize Network Traffic Patterns',
      description: 'Implement QoS and traffic shaping to reduce network congestion',
      impact: 'Medium',
      effort: 'High',
      savings: 240,
      performanceGain: 22,
      implementation: [
        'Configure QoS policies on network switches',
        'Implement traffic shaping for backup traffic',
        'Optimize inter-cluster communication paths',
      ],
      estimatedTime: '4 hours',
      prerequisites: ['Network maintenance window', 'Switch configuration access'],
    },
  ];

  const mockModelMetrics: ModelMetrics = {
    accuracy: 91.5,
    precision: 89.2,
    recall: 93.8,
    f1Score: 91.4,
    lastTrained: new Date(Date.now() - 3600000 * 12).toISOString(),
    trainingDataSize: 50000,
    predictionCount: 15623,
    modelDrift: 2.3,
    confidenceDistribution: {
      high: 68, // percentage of predictions with >90% confidence
      medium: 24, // 70-90% confidence
      low: 8, // <70% confidence
    },
  };

  // Use WebSocket data when available, fallback to mocks
  const resourcePredictions = useMemo(() =>
    predictionData?.resourcePredictions ?? mockResourcePredictions,
    [predictionData]
  );

  const workloadPatterns = useMemo(() =>
    predictionData?.workloadPatterns ?? mockWorkloadPatterns,
    [predictionData]
  );

  const migrationPredictions = useMemo(() =>
    predictionData?.migrationPredictions ?? mockMigrationPredictions,
    [predictionData]
  );

  const scalingRecommendations = useMemo(() =>
    predictionData?.scalingRecommendations ?? mockScalingRecommendations,
    [predictionData]
  );

  const performanceOptimizations = useMemo(() =>
    predictionData?.performanceOptimizations ?? mockOptimizations,
    [predictionData]
  );

  // Generate time series data for predictions
  const generatePredictionData = (resourceType: string) => {
    const points = 100;
    const now = Date.now();
    const interval = {
      '1hr': 36000, // 36 seconds
      '6hr': 216000, // 3.6 minutes
      '24hr': 864000, // 14.4 minutes
      '7days': 6048000, // 1.68 hours
      '30days': 25920000, // 7.2 hours
    }[timeHorizon];

    const baseValue = resourcePredictions.find(r => r.resourceType === resourceType)?.currentUsage || 50;

    return Array.from({ length: points }, (_, i) => {
      const trend = i * 0.2 + Math.sin(i * 0.1) * 10;
      const noise = Math.random() * 10 - 5;
      const actual = i < 60 ? baseValue + trend + noise : undefined;
      const predicted = baseValue + trend + noise + Math.random() * 5;

      return {
        timestamp: new Date(now + i * interval).toISOString(),
        actual,
        predicted: Math.max(0, Math.min(100, predicted)),
        confidence: {
          lower: Math.max(0, predicted - 10),
          upper: Math.min(100, predicted + 10),
        },
      };
    });
  };

  const handleWhatIfAnalysis = () => {
    // This would trigger a what-if analysis with the API
    console.log('Running what-if analysis with params:', whatIfParams);
  };

  const getImpactIcon = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high': return <TrendingUp className="h-4 w-4 text-red-500" />;
      case 'medium': return <TrendingUp className="h-4 w-4 text-yellow-500" />;
      case 'low': return <TrendingUp className="h-4 w-4 text-green-500" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Performance Prediction</h2>
          <p className="text-muted-foreground">
            AI-driven performance insights and optimization recommendations
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={timeHorizon} onValueChange={(v: any) => setTimeHorizon(v)}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1hr">1 Hour</SelectItem>
              <SelectItem value="6hr">6 Hours</SelectItem>
              <SelectItem value="24hr">24 Hours</SelectItem>
              <SelectItem value="7days">7 Days</SelectItem>
              <SelectItem value="30days">30 Days</SelectItem>
            </SelectContent>
          </Select>
          {isConnected ? (
            <Badge variant="outline">
              <Brain className="h-3 w-3 mr-1 text-green-500" />
              AI Active
            </Badge>
          ) : (
            <Badge variant="outline">
              <XCircle className="h-3 w-3 mr-1 text-red-500" />
              AI Offline
            </Badge>
          )}
        </div>
      </div>

      {/* Model Performance */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>AI Model Performance</CardTitle>
              <CardDescription>Real-time model accuracy and prediction metrics</CardDescription>
            </div>
            <Badge variant={mockModelMetrics.accuracy > 90 ? 'default' : 'secondary'}>
              {mockModelMetrics.accuracy.toFixed(1)}% Accurate
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm font-medium">Precision</p>
              <p className="text-2xl font-bold">{mockModelMetrics.precision.toFixed(1)}%</p>
              <Progress value={mockModelMetrics.precision} className="mt-1" />
            </div>
            <div>
              <p className="text-sm font-medium">Recall</p>
              <p className="text-2xl font-bold">{mockModelMetrics.recall.toFixed(1)}%</p>
              <Progress value={mockModelMetrics.recall} className="mt-1" />
            </div>
            <div>
              <p className="text-sm font-medium">F1 Score</p>
              <p className="text-2xl font-bold">{mockModelMetrics.f1Score.toFixed(1)}%</p>
              <Progress value={mockModelMetrics.f1Score} className="mt-1" />
            </div>
            <div>
              <p className="text-sm font-medium">Model Drift</p>
              <p className="text-2xl font-bold">{mockModelMetrics.modelDrift.toFixed(1)}%</p>
              <Progress value={mockModelMetrics.modelDrift} className="mt-1" />
            </div>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Training Data:</span>
              <span className="ml-1 font-medium">{mockModelMetrics.trainingDataSize.toLocaleString()} samples</span>
            </div>
            <div>
              <span className="text-muted-foreground">Predictions Made:</span>
              <span className="ml-1 font-medium">{mockModelMetrics.predictionCount.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Last Trained:</span>
              <span className="ml-1 font-medium">12 hours ago</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Resource Predictions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {resourcePredictions.map((prediction) => {
          const Icon = {
            cpu: Cpu,
            memory: MemoryStick,
            storage: HardDrive,
            network: Network,
          }[prediction.resourceType];

          return (
            <Card key={prediction.resourceType}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium capitalize">
                  {prediction.resourceType} Usage
                </CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Current</span>
                    <span>{prediction.currentUsage}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Predicted</span>
                    <span className={getConfidenceColor(prediction.confidence)}>
                      {prediction.predictedUsage}%
                    </span>
                  </div>
                  <Progress value={prediction.predictedUsage} className="h-2" />
                  <div className="flex justify-between items-center text-xs">
                    <Badge variant={prediction.trend === 'increasing' ? 'destructive' :
                                  prediction.trend === 'decreasing' ? 'default' : 'secondary'}>
                      {prediction.trend}
                    </Badge>
                    <span className="text-muted-foreground">
                      {prediction.confidence}% confidence
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Capacity in: {prediction.timeToCapacity}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="predictions" className="space-y-4">
        <TabsList className="grid grid-cols-5 w-full">
          <TabsTrigger value="predictions">Resource Predictions</TabsTrigger>
          <TabsTrigger value="patterns">Workload Patterns</TabsTrigger>
          <TabsTrigger value="migration">Migration Planning</TabsTrigger>
          <TabsTrigger value="scaling">Scaling Recommendations</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {resourcePredictions.map((prediction) => (
              <Card key={prediction.resourceType}>
                <CardHeader>
                  <CardTitle className="capitalize">{prediction.resourceType} Forecast</CardTitle>
                  <CardDescription>AI-powered usage prediction over {timeHorizon}</CardDescription>
                </CardHeader>
                <CardContent>
                  <PredictiveChart
                    data={generatePredictionData(prediction.resourceType)}
                    title=""
                    yAxisLabel="Usage (%)"
                    showConfidence={true}
                    height={200}
                  />

                  <div className="mt-4">
                    <h4 className="text-sm font-medium mb-2">Key Factors</h4>
                    <div className="space-y-1">
                      {prediction.factors.map((factor, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <div className="w-1 h-1 rounded-full bg-primary" />
                          <span className="text-muted-foreground">{factor}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Alert className="mt-4">
                    <Lightbulb className="h-4 w-4" />
                    <AlertTitle>Recommendation</AlertTitle>
                    <AlertDescription className="text-sm">
                      {prediction.recommendation}
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <div className="space-y-4">
            {workloadPatterns.map((pattern) => (
              <Card key={pattern.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {pattern.name}
                        <Badge variant={pattern.confidence > 90 ? 'default' : 'secondary'}>
                          {pattern.confidence}% confidence
                        </Badge>
                      </CardTitle>
                      <CardDescription>{pattern.description}</CardDescription>
                    </div>
                    <Badge variant="outline" className="capitalize">
                      {pattern.type}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-sm font-medium mb-3">Pattern Details</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Peak Hours:</span>
                          <span>{pattern.peakHours[0]} - {pattern.peakHours[1]}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Multiplier:</span>
                          <span>{pattern.baselineMultiplier}x baseline</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Next Occurrence:</span>
                          <span>{new Date(pattern.nextOccurrence).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-3">Resource Impact</h4>
                      <div className="space-y-2">
                        {Object.entries(pattern.impact).map(([resource, impact]) => (
                          <div key={resource} className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="text-sm capitalize">{resource}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Progress value={Math.min(100, impact)} className="w-20" />
                              <span className="text-sm w-12 text-right">
                                +{impact}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="migration" className="space-y-4">
          <div className="space-y-4">
            {migrationPredictions.map((migration) => (
              <Card key={migration.vmId}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{migration.vmName}</CardTitle>
                      <CardDescription>
                        {migration.sourceNode} → {migration.targetNode}
                      </CardDescription>
                    </div>
                    <div className="text-right">
                      <Badge variant={migration.successProbability > 90 ? 'default' : 'secondary'}>
                        {migration.successProbability}% Success Rate
                      </Badge>
                      <p className="text-sm text-muted-foreground mt-1">
                        Est. Duration: {formatDuration(migration.estimatedDuration)}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h4 className="text-sm font-medium mb-3">Migration Metrics</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Expected Downtime:</span>
                          <span>{formatDuration(migration.expectedDowntime)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Optimal Time:</span>
                          <span>{new Date(migration.optimalTime).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-3">Resource Impact</h4>
                      <div className="space-y-1">
                        {Object.entries(migration.resourceImpact).map(([resource, impact]) => (
                          <div key={resource} className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground capitalize">{resource}:</span>
                            <span>+{impact}%</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-3">Risk Factors</h4>
                      <div className="space-y-1">
                        {migration.risks.map((risk, idx) => (
                          <div key={idx} className="flex items-start gap-2 text-sm">
                            <AlertCircle className="h-3 w-3 text-yellow-500 mt-0.5 flex-shrink-0" />
                            <span className="text-muted-foreground">{risk}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <Alert className="mt-4">
                    <Target className="h-4 w-4" />
                    <AlertTitle>Recommendation</AlertTitle>
                    <AlertDescription>{migration.recommendation}</AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="scaling" className="space-y-4">
          <div className="space-y-4">
            {scalingRecommendations.map((scaling, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="capitalize flex items-center gap-2">
                        {scaling.type} Scaling - {scaling.resourceType}
                        {getImpactIcon('medium')}
                      </CardTitle>
                      <CardDescription>
                        {scaling.currentCapacity} → {scaling.recommendedCapacity} units
                      </CardDescription>
                    </div>
                    <div className="text-right">
                      <Badge variant={scaling.confidence > 90 ? 'default' : 'secondary'}>
                        {scaling.confidence}% Confidence
                      </Badge>
                      <p className="text-sm text-muted-foreground mt-1">
                        Timeline: {scaling.timeline}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-sm font-medium mb-3">Impact Analysis</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Cost Impact:</span>
                          <span className="font-medium">${scaling.costImpact}/month</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Performance Gain:</span>
                          <span className="font-medium text-green-600">+{scaling.performanceGain}%</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Reasoning:</span>
                          <p className="mt-1">{scaling.reasoning}</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium mb-3">Implementation Steps</h4>
                      <div className="space-y-2">
                        {scaling.implementation.map((step, stepIdx) => (
                          <div key={stepIdx} className="flex items-start gap-2 text-sm">
                            <div className="w-5 h-5 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center mt-0.5 flex-shrink-0">
                              {stepIdx + 1}
                            </div>
                            <span>{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Optimization Recommendations */}
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Optimization Opportunities</CardTitle>
                  <CardDescription>AI-identified performance improvements</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {performanceOptimizations.map((opt) => (
                      <div key={opt.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="text-sm font-medium">{opt.title}</h4>
                          <div className="flex items-center gap-2">
                            {getImpactIcon(opt.impact)}
                            <Badge variant="outline">{opt.impact} Impact</Badge>
                          </div>
                        </div>

                        <p className="text-sm text-muted-foreground mb-3">{opt.description}</p>

                        <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                          <div>
                            <span className="text-muted-foreground">Savings:</span>
                            <p className="font-medium text-green-600">${opt.savings}/month</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Performance:</span>
                            <p className="font-medium">+{opt.performanceGain}%</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Time:</span>
                            <p className="font-medium">{opt.estimatedTime}</p>
                          </div>
                        </div>

                        <details className="mt-3">
                          <summary className="cursor-pointer text-sm font-medium text-primary">
                            Implementation Details
                          </summary>
                          <div className="mt-2 space-y-2">
                            <div>
                              <h5 className="text-xs font-medium text-muted-foreground uppercase">Steps</h5>
                              <ul className="text-sm space-y-1 ml-4 list-disc">
                                {opt.implementation.map((step, idx) => (
                                  <li key={idx}>{step}</li>
                                ))}
                              </ul>
                            </div>
                            {opt.prerequisites.length > 0 && (
                              <div>
                                <h5 className="text-xs font-medium text-muted-foreground uppercase">Prerequisites</h5>
                                <ul className="text-sm space-y-1 ml-4 list-disc">
                                  {opt.prerequisites.map((prereq, idx) => (
                                    <li key={idx}>{prereq}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </details>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* What-If Analysis */}
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>What-If Analysis</CardTitle>
                  <CardDescription>Simulate different scenarios and their impact</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">CPU Increase (%)</label>
                    <Slider
                      value={[whatIfParams.cpuIncrease]}
                      onValueChange={([value]) => setWhatIfParams(prev => ({ ...prev, cpuIncrease: value }))}
                      min={-50}
                      max={200}
                      step={10}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1">
                      Current: {whatIfParams.cpuIncrease > 0 ? '+' : ''}{whatIfParams.cpuIncrease}%
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Memory Increase (%)</label>
                    <Slider
                      value={[whatIfParams.memoryIncrease]}
                      onValueChange={([value]) => setWhatIfParams(prev => ({ ...prev, memoryIncrease: value }))}
                      min={-50}
                      max={200}
                      step={10}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1">
                      Current: {whatIfParams.memoryIncrease > 0 ? '+' : ''}{whatIfParams.memoryIncrease}%
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Workload Increase (%)</label>
                    <Slider
                      value={[whatIfParams.workloadIncrease]}
                      onValueChange={([value]) => setWhatIfParams(prev => ({ ...prev, workloadIncrease: value }))}
                      min={-50}
                      max={300}
                      step={10}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1">
                      Current: {whatIfParams.workloadIncrease > 0 ? '+' : ''}{whatIfParams.workloadIncrease}%
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Scenario Description</label>
                    <Textarea
                      placeholder="Describe the scenario you want to analyze..."
                      value={whatIfParams.description}
                      onChange={(e) => setWhatIfParams(prev => ({ ...prev, description: e.target.value }))}
                      className="mt-2"
                      rows={3}
                    />
                  </div>

                  <Button onClick={handleWhatIfAnalysis} className="w-full">
                    <Zap className="h-4 w-4 mr-2" />
                    Run Analysis
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Confidence Threshold</CardTitle>
                  <CardDescription>Filter predictions by confidence level</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Minimum Confidence</span>
                        <span>{confidenceThreshold}%</span>
                      </div>
                      <Slider
                        value={[confidenceThreshold]}
                        onValueChange={([value]) => setConfidenceThreshold(value)}
                        min={50}
                        max={99}
                        step={5}
                      />
                    </div>

                    <div className="text-sm space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">High Confidence (≥90%):</span>
                        <span>{mockModelMetrics.confidenceDistribution.high}% of predictions</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Medium Confidence (70-90%):</span>
                        <span>{mockModelMetrics.confidenceDistribution.medium}% of predictions</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Low Confidence (&lt;70%):</span>
                        <span>{mockModelMetrics.confidenceDistribution.low}% of predictions</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerformancePredictionDashboard;