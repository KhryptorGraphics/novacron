'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { 
  TrendingUp,
  TrendingDown,
  Zap,
  CPU,
  MemoryStick,
  HardDrive,
  Network,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Settings,
  Play,
  Pause,
  BarChart3,
  LineChart,
  Activity,
  Gauge,
  Wrench,
  Lightbulb,
  RefreshCw,
  ChevronRight,
  Eye,
  ArrowUp,
  ArrowDown
} from 'lucide-react';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useToast } from '@/components/ui/use-toast';

interface PerformanceMetric {
  id: string;
  name: string;
  category: 'cpu' | 'memory' | 'disk' | 'network';
  current: number;
  baseline: number;
  threshold: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  impact: 'high' | 'medium' | 'low';
  history: { time: string; value: number }[];
}

interface BottleneckAnalysis {
  id: string;
  vmId: string;
  vmName: string;
  type: 'cpu_bottleneck' | 'memory_leak' | 'disk_io' | 'network_congestion' | 'resource_contention';
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  impact: string;
  recommendations: OptimizationRecommendation[];
  detectedAt: Date;
  affectedMetrics: string[];
}

interface OptimizationRecommendation {
  id: string;
  title: string;
  description: string;
  category: 'resource_allocation' | 'configuration' | 'infrastructure' | 'application';
  priority: 'high' | 'medium' | 'low';
  estimatedImprovement: string;
  effort: 'low' | 'medium' | 'high';
  autoApplicable: boolean;
  implementation: string[];
  cost: 'free' | 'low' | 'medium' | 'high';
  riskLevel: 'low' | 'medium' | 'high';
}

interface OptimizationJob {
  id: string;
  vmId: string;
  vmName: string;
  recommendations: string[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startedAt: Date;
  estimatedCompletion?: Date;
  results?: {
    appliedOptimizations: number;
    performanceGain: number;
    beforeAfter: {
      cpu: { before: number; after: number };
      memory: { before: number; after: number };
      disk: { before: number; after: number };
      network: { before: number; after: number };
    };
  };
}

const mockMetrics: PerformanceMetric[] = [
  {
    id: 'cpu-usage',
    name: 'CPU Usage',
    category: 'cpu',
    current: 85.4,
    baseline: 65.0,
    threshold: 80.0,
    unit: '%',
    trend: 'up',
    impact: 'high',
    history: Array.from({ length: 24 }, (_, i) => ({
      time: `${23-i}:00`,
      value: 60 + Math.random() * 30 + (i > 18 ? 15 : 0)
    }))
  },
  {
    id: 'memory-usage',
    name: 'Memory Usage',
    category: 'memory',
    current: 78.2,
    baseline: 55.0,
    threshold: 85.0,
    unit: '%',
    trend: 'up',
    impact: 'medium',
    history: Array.from({ length: 24 }, (_, i) => ({
      time: `${23-i}:00`,
      value: 50 + Math.random() * 25 + (i > 20 ? 10 : 0)
    }))
  },
  {
    id: 'disk-io',
    name: 'Disk I/O',
    category: 'disk',
    current: 156.8,
    baseline: 85.0,
    threshold: 120.0,
    unit: 'MB/s',
    trend: 'up',
    impact: 'high',
    history: Array.from({ length: 24 }, (_, i) => ({
      time: `${23-i}:00`,
      value: 70 + Math.random() * 40 + (i > 16 ? 30 : 0)
    }))
  },
  {
    id: 'network-throughput',
    name: 'Network Throughput',
    category: 'network',
    current: 420.5,
    baseline: 280.0,
    threshold: 500.0,
    unit: 'Mbps',
    trend: 'stable',
    impact: 'low',
    history: Array.from({ length: 24 }, (_, i) => ({
      time: `${23-i}:00`,
      value: 250 + Math.random() * 100
    }))
  }
];

const mockBottlenecks: BottleneckAnalysis[] = [
  {
    id: 'bottleneck-1',
    vmId: 'vm-001',
    vmName: 'web-server-01',
    type: 'cpu_bottleneck',
    severity: 'high',
    description: 'CPU utilization consistently above 85% with frequent spikes to 100%',
    impact: 'Response times increased by 40%, potential for request timeouts',
    recommendations: [],
    detectedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    affectedMetrics: ['cpu-usage', 'response-time']
  },
  {
    id: 'bottleneck-2',
    vmId: 'vm-002',
    vmName: 'app-server-02',
    type: 'memory_leak',
    severity: 'medium',
    description: 'Memory usage gradually increasing over time, suggesting a memory leak',
    impact: 'System stability at risk, potential for OOM crashes',
    recommendations: [],
    detectedAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
    affectedMetrics: ['memory-usage']
  },
  {
    id: 'bottleneck-3',
    vmId: 'vm-003',
    vmName: 'db-server-01',
    type: 'disk_io',
    severity: 'critical',
    description: 'Disk I/O operations are severely bottlenecked, causing query delays',
    impact: 'Database performance degraded by 60%, affecting all applications',
    recommendations: [],
    detectedAt: new Date(Date.now() - 30 * 60 * 1000),
    affectedMetrics: ['disk-io', 'query-time']
  }
];

const mockRecommendations: OptimizationRecommendation[] = [
  {
    id: 'rec-1',
    title: 'Increase CPU Cores',
    description: 'Allocate additional CPU cores to handle increased load',
    category: 'resource_allocation',
    priority: 'high',
    estimatedImprovement: '25-35% performance increase',
    effort: 'low',
    autoApplicable: true,
    implementation: [
      'Shutdown VM gracefully',
      'Increase CPU allocation from 4 to 6 cores',
      'Start VM and verify performance'
    ],
    cost: 'medium',
    riskLevel: 'low'
  },
  {
    id: 'rec-2',
    title: 'Enable CPU Caching',
    description: 'Configure advanced CPU caching to improve instruction throughput',
    category: 'configuration',
    priority: 'medium',
    estimatedImprovement: '10-15% CPU efficiency',
    effort: 'low',
    autoApplicable: true,
    implementation: [
      'Enable CPU cache optimization',
      'Configure cache policies',
      'Monitor performance impact'
    ],
    cost: 'free',
    riskLevel: 'low'
  },
  {
    id: 'rec-3',
    title: 'Upgrade to Premium SSD',
    description: 'Migrate to high-performance NVMe storage for better I/O',
    category: 'infrastructure',
    priority: 'high',
    estimatedImprovement: '50-70% disk I/O improvement',
    effort: 'high',
    autoApplicable: false,
    implementation: [
      'Schedule maintenance window',
      'Create storage snapshot',
      'Migrate to premium SSD tier',
      'Verify data integrity'
    ],
    cost: 'high',
    riskLevel: 'medium'
  },
  {
    id: 'rec-4',
    title: 'Application Memory Optimization',
    description: 'Investigate and fix potential memory leaks in applications',
    category: 'application',
    priority: 'medium',
    estimatedImprovement: '20-30% memory efficiency',
    effort: 'high',
    autoApplicable: false,
    implementation: [
      'Analyze memory usage patterns',
      'Identify memory leak sources',
      'Apply code fixes or configuration changes',
      'Monitor memory stability'
    ],
    cost: 'free',
    riskLevel: 'low'
  }
];

export function PerformanceOptimizationFlow() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedVM, setSelectedVM] = useState('vm-001');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedRecommendations, setSelectedRecommendations] = useState<string[]>([]);
  const [optimizationJob, setOptimizationJob] = useState<OptimizationJob | null>(null);
  const [continuousMonitoring, setContinuousMonitoring] = useState(true);
  const [alertThresholds, setAlertThresholds] = useState({
    cpu: 85,
    memory: 90,
    disk: 120,
    network: 800
  });

  const { toast } = useToast();

  // Simulate analysis progress
  useEffect(() => {
    if (isAnalyzing) {
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 100) {
            setIsAnalyzing(false);
            toast({
              title: "Analysis Complete",
              description: "Performance bottlenecks have been identified with optimization recommendations.",
            });
            return 100;
          }
          return prev + Math.random() * 8;
        });
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isAnalyzing, toast]);

  // Simulate optimization job progress
  useEffect(() => {
    if (optimizationJob?.status === 'running') {
      const interval = setInterval(() => {
        setOptimizationJob(prev => {
          if (!prev) return null;
          
          const newProgress = Math.min(prev.progress + Math.random() * 5, 100);
          
          if (newProgress >= 100) {
            const results = {
              appliedOptimizations: selectedRecommendations.length,
              performanceGain: 28.5,
              beforeAfter: {
                cpu: { before: 85.4, after: 62.1 },
                memory: { before: 78.2, after: 68.5 },
                disk: { before: 156.8, after: 98.4 },
                network: { before: 420.5, after: 445.2 }
              }
            };
            
            toast({
              title: "Optimization Complete",
              description: `Applied ${results.appliedOptimizations} optimizations with ${results.performanceGain}% performance improvement.`,
            });
            
            return {
              ...prev,
              status: 'completed',
              progress: 100,
              results
            };
          }
          
          return { ...prev, progress: newProgress };
        });
      }, 800);
      
      return () => clearInterval(interval);
    }
  }, [optimizationJob?.status, selectedRecommendations.length, toast]);

  const startAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    toast({
      title: "Analysis Started",
      description: "Running comprehensive performance analysis...",
    });
  };

  const applyOptimizations = () => {
    if (selectedRecommendations.length === 0) {
      toast({
        title: "No Recommendations Selected",
        description: "Please select at least one optimization to apply.",
        variant: "destructive"
      });
      return;
    }

    const job: OptimizationJob = {
      id: `opt-${Date.now()}`,
      vmId: selectedVM,
      vmName: 'web-server-01',
      recommendations: selectedRecommendations,
      status: 'running',
      progress: 0,
      startedAt: new Date()
    };

    setOptimizationJob(job);
    toast({
      title: "Optimization Started",
      description: `Applying ${selectedRecommendations.length} optimizations...`,
    });
  };

  const getMetricIcon = (category: string) => {
    switch (category) {
      case 'cpu': return CPU;
      case 'memory': return MemoryStick;
      case 'disk': return HardDrive;
      case 'network': return Network;
      default: return Activity;
    }
  };

  const getBottleneckSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold">Performance Optimization</h2>
          <p className="text-gray-600 mt-1">Analyze bottlenecks and optimize VM performance with AI-powered recommendations</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={startAnalysis} disabled={isAnalyzing}>
            {isAnalyzing ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <BarChart3 className="mr-2 h-4 w-4" />
                Analyze Performance
              </>
            )}
          </Button>
          <Button>
            <Zap className="mr-2 h-4 w-4" />
            Quick Optimize
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="bottlenecks">Bottlenecks</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          {/* Real-time Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {mockMetrics.map(metric => {
              const Icon = getMetricIcon(metric.category);
              const isAboveThreshold = metric.current > metric.threshold;
              
              return (
                <Card key={metric.id} className={`${isAboveThreshold ? 'ring-2 ring-red-500' : ''}`}>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-2 rounded ${
                        isAboveThreshold ? 'bg-red-100' : 'bg-blue-100'
                      }`}>
                        <Icon className={`h-6 w-6 ${
                          isAboveThreshold ? 'text-red-600' : 'text-blue-600'
                        }`} />
                      </div>
                      <div className="flex items-center space-x-1">
                        {metric.trend === 'up' ? (
                          <TrendingUp className="h-4 w-4 text-red-500" />
                        ) : metric.trend === 'down' ? (
                          <TrendingDown className="h-4 w-4 text-green-500" />
                        ) : (
                          <Activity className="h-4 w-4 text-gray-500" />
                        )}
                        <Badge variant={isAboveThreshold ? 'destructive' : 'secondary'}>
                          {metric.impact}
                        </Badge>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg">{metric.name}</h3>
                      <div className="flex items-baseline space-x-2">
                        <span className={`text-3xl font-bold ${
                          isAboveThreshold ? 'text-red-600' : 'text-gray-900'
                        }`}>
                          {metric.current.toFixed(1)}
                        </span>
                        <span className="text-gray-500">{metric.unit}</span>
                      </div>
                      <div className="mt-2 text-sm text-gray-600">
                        Baseline: {metric.baseline.toFixed(1)}{metric.unit} | 
                        Threshold: {metric.threshold.toFixed(1)}{metric.unit}
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <Progress 
                        value={(metric.current / (metric.threshold * 1.2)) * 100} 
                        className="h-2" 
                      />
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Performance Trends */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Trends (24 Hours)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockMetrics[0].history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {mockMetrics.map((metric, index) => (
                      <Area
                        key={metric.id}
                        type="monotone"
                        dataKey="value"
                        stroke={['#8884d8', '#82ca9d', '#ffc658', '#ff7300'][index]}
                        fill={['#8884d8', '#82ca9d', '#ffc658', '#ff7300'][index]}
                        fillOpacity={0.3}
                        name={metric.name}
                        data={metric.history}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Progress */}
          {isAnalyzing && (
            <Card>
              <CardHeader>
                <CardTitle>Performance Analysis in Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span>Analyzing system performance...</span>
                    <span className="font-medium">{Math.round(analysisProgress)}%</span>
                  </div>
                  <Progress value={analysisProgress} className="w-full" />
                  <div className="text-sm text-gray-600">
                    {analysisProgress < 25 && "Collecting performance metrics..."}
                    {analysisProgress >= 25 && analysisProgress < 50 && "Analyzing resource utilization patterns..."}
                    {analysisProgress >= 50 && analysisProgress < 75 && "Identifying performance bottlenecks..."}
                    {analysisProgress >= 75 && analysisProgress < 100 && "Generating optimization recommendations..."}
                    {analysisProgress >= 100 && "Analysis complete!"}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="bottlenecks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Identified Bottlenecks</CardTitle>
              <CardDescription>Critical performance issues requiring immediate attention</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockBottlenecks.map(bottleneck => (
                  <div
                    key={bottleneck.id}
                    className={`p-4 rounded-lg border ${getBottleneckSeverityColor(bottleneck.severity)}`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className="p-1 rounded bg-white">
                          <AlertTriangle className="h-5 w-5" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-lg">{bottleneck.vmName}</h4>
                          <p className="text-sm opacity-75">
                            Detected {bottleneck.detectedAt.toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline" className="bg-white">
                        {bottleneck.severity} severity
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div>
                        <Label className="text-sm font-medium">Issue Description</Label>
                        <p className="text-sm">{bottleneck.description}</p>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium">Impact</Label>
                        <p className="text-sm">{bottleneck.impact}</p>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium">Affected Metrics</Label>
                        <div className="flex gap-1 mt-1">
                          {bottleneck.affectedMetrics.map(metric => (
                            <Badge key={metric} variant="secondary" className="text-xs">
                              {metric}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-end space-x-2 mt-4">
                      <Button variant="outline" size="sm">
                        <Eye className="mr-2 h-4 w-4" />
                        View Details
                      </Button>
                      <Button size="sm">
                        <Wrench className="mr-2 h-4 w-4" />
                        Generate Fix
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Optimization Recommendations</CardTitle>
              <CardDescription>AI-powered suggestions to improve system performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockRecommendations.map(rec => (
                  <Card key={rec.id} className="border-l-4 border-l-blue-500">
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-start space-x-3">
                          <input
                            type="checkbox"
                            checked={selectedRecommendations.includes(rec.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedRecommendations(prev => [...prev, rec.id]);
                              } else {
                                setSelectedRecommendations(prev => prev.filter(id => id !== rec.id));
                              }
                            }}
                            className="mt-1"
                          />
                          <div>
                            <h4 className="font-semibold text-lg">{rec.title}</h4>
                            <p className="text-gray-600 mt-1">{rec.description}</p>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Badge variant={getPriorityColor(rec.priority)}>
                            {rec.priority} priority
                          </Badge>
                          {rec.autoApplicable && (
                            <Badge variant="outline">
                              <Zap className="mr-1 h-3 w-3" />
                              Auto-apply
                            </Badge>
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <Label className="text-sm font-medium">Category</Label>
                          <p className="text-sm mt-1 capitalize">{rec.category.replace('_', ' ')}</p>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Estimated Improvement</Label>
                          <p className="text-sm mt-1 text-green-600 font-medium">{rec.estimatedImprovement}</p>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Effort Level</Label>
                          <div className="flex items-center mt-1">
                            <Badge variant="outline" className={
                              rec.effort === 'low' ? 'text-green-600' :
                              rec.effort === 'medium' ? 'text-yellow-600' : 'text-red-600'
                            }>
                              {rec.effort}
                            </Badge>
                          </div>
                        </div>
                        <div>
                          <Label className="text-sm font-medium">Cost & Risk</Label>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {rec.cost} cost
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {rec.riskLevel} risk
                            </Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium">Implementation Steps</Label>
                        <ol className="list-decimal list-inside mt-1 space-y-1">
                          {rec.implementation.map((step, index) => (
                            <li key={index} className="text-sm text-gray-600">{step}</li>
                          ))}
                        </ol>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
              
              {selectedRecommendations.length > 0 && (
                <div className="flex justify-end space-x-2 mt-6 pt-4 border-t">
                  <Button variant="outline">
                    Preview Changes
                  </Button>
                  <Button onClick={applyOptimizations}>
                    <Play className="mr-2 h-4 w-4" />
                    Apply {selectedRecommendations.length} Optimizations
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          {optimizationJob ? (
            <Card>
              <CardHeader>
                <CardTitle>
                  {optimizationJob.status === 'running' ? 'Optimization in Progress' : 
                   optimizationJob.status === 'completed' ? 'Optimization Completed' : 
                   'Optimization Status'}
                </CardTitle>
                <CardDescription>
                  Applying performance optimizations to {optimizationJob.vmName}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Overall Progress</span>
                    <span className="font-medium">{Math.round(optimizationJob.progress)}%</span>
                  </div>
                  <Progress value={optimizationJob.progress} className="w-full" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Recommendations:</span>
                    <span className="ml-2 font-medium">{optimizationJob.recommendations.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Started:</span>
                    <span className="ml-2 font-medium">{optimizationJob.startedAt.toLocaleTimeString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Status:</span>
                    <Badge className="ml-2" variant={optimizationJob.status === 'completed' ? 'default' : 'secondary'}>
                      {optimizationJob.status}
                    </Badge>
                  </div>
                  {optimizationJob.status === 'running' && (
                    <div>
                      <span className="text-gray-600">ETA:</span>
                      <span className="ml-2 font-medium">
                        {Math.round((100 - optimizationJob.progress) / 10)} minutes
                      </span>
                    </div>
                  )}
                </div>

                {optimizationJob.status === 'completed' && optimizationJob.results && (
                  <div className="space-y-4">
                    <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                      <div className="flex items-center space-x-2 mb-2">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <h4 className="font-semibold text-green-800">Optimization Successful!</h4>
                      </div>
                      <p className="text-green-700">
                        Applied {optimizationJob.results.appliedOptimizations} optimizations with 
                        {optimizationJob.results.performanceGain}% overall performance improvement.
                      </p>
                    </div>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Performance Comparison</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {Object.entries(optimizationJob.results.beforeAfter).map(([metric, values]) => (
                            <div key={metric} className="flex items-center justify-between p-3 rounded-lg border">
                              <div className="flex items-center space-x-3">
                                <span className="font-medium capitalize">{metric}</span>
                              </div>
                              <div className="flex items-center space-x-4">
                                <div className="text-right">
                                  <div className="text-sm text-gray-600">Before</div>
                                  <div className="font-medium">{values.before.toFixed(1)}%</div>
                                </div>
                                <ArrowRight className="h-4 w-4 text-gray-400" />
                                <div className="text-right">
                                  <div className="text-sm text-gray-600">After</div>
                                  <div className="font-medium text-green-600">{values.after.toFixed(1)}%</div>
                                </div>
                                <div className="flex items-center text-green-600">
                                  {values.after < values.before ? (
                                    <ArrowDown className="h-4 w-4 mr-1" />
                                  ) : (
                                    <ArrowUp className="h-4 w-4 mr-1" />
                                  )}
                                  <span className="text-sm font-medium">
                                    {Math.abs(((values.after - values.before) / values.before) * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>No Active Optimization</CardTitle>
                <CardDescription>Select recommendations from the previous tab to start optimization</CardDescription>
              </CardHeader>
              <CardContent className="text-center py-8">
                <Lightbulb className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-gray-500 mb-4">
                  Choose optimization recommendations to improve your system performance
                </p>
                <Button onClick={() => setActiveTab('recommendations')}>
                  <ChevronRight className="mr-2 h-4 w-4" />
                  View Recommendations
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>Monitoring Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center justify-between">
                    <Label>Continuous Monitoring</Label>
                    <Switch
                      checked={continuousMonitoring}
                      onCheckedChange={setContinuousMonitoring}
                    />
                  </div>
                  
                  <div className="space-y-4">
                    <h4 className="font-medium">Alert Thresholds</h4>
                    
                    {Object.entries(alertThresholds).map(([metric, threshold]) => (
                      <div key={metric} className="space-y-2">
                        <Label className="capitalize">{metric} Usage (%)</Label>
                        <div className="flex items-center space-x-2">
                          <Input
                            type="number"
                            value={threshold}
                            onChange={(e) => setAlertThresholds(prev => ({
                              ...prev,
                              [metric]: parseInt(e.target.value) || 0
                            }))}
                            className="w-20"
                          />
                          <span className="text-sm text-gray-500">%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <Button className="w-full">
                    <Settings className="mr-2 h-4 w-4" />
                    Save Configuration
                  </Button>
                </CardContent>
              </Card>
            </div>
            
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Real-time Performance Monitor</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsLineChart data={mockMetrics[0].history}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#8884d8"
                          strokeWidth={2}
                          name="CPU Usage"
                        />
                      </RechartsLineChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="mt-4 text-sm text-gray-600">
                    <p>Monitoring {continuousMonitoring ? 'enabled' : 'disabled'}. 
                    {continuousMonitoring && ' Alerts will be triggered when thresholds are exceeded.'}</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}