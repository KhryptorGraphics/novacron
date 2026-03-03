"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, 
  TrendingUp, 
  Zap, 
  RefreshCw, 
  Download,
  Play,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

interface MLModelMetrics {
  modelType: string;
  accuracy: number;
  throughput: number;
  latency: number;
  lastTraining: string;
  version: string;
  status: 'training' | 'deployed' | 'evaluating' | 'error';
  benchmarkResults?: {
    precision: number;
    recall: number;
    f1Score: number;
    auc: number;
  };
  trainingMetrics?: {
    epochs: number;
    trainingLoss: number;
    validationLoss: number;
    trainingTime: string;
  };
}

interface MLModelPerformancePanelProps {
  models: MLModelMetrics[];
}

export function MLModelPerformancePanel({ models: initialModels }: MLModelPerformancePanelProps) {
  const [models, setModels] = useState(initialModels);
  const [isTraining, setIsTraining] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'success';
      case 'training': return 'warning';
      case 'evaluating': return 'secondary';
      case 'error': return 'destructive';
      default: return 'secondary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return <CheckCircle className="h-4 w-4" />;
      case 'training': return <RefreshCw className="h-4 w-4 animate-spin" />;
      case 'evaluating': return <Brain className="h-4 w-4" />;
      case 'error': return <AlertCircle className="h-4 w-4" />;
      default: return <Brain className="h-4 w-4" />;
    }
  };

  const retrainModel = async (modelType: string) => {
    setIsTraining(modelType);
    try {
      const response = await fetch(`/api/orchestration/ml-models/${modelType}/retrain`, {
        method: 'POST',
      });
      
      if (response.ok) {
        // Update model status
        setModels(models.map(model => 
          model.modelType === modelType 
            ? { ...model, status: 'training' as const }
            : model
        ));
        
        // Poll for training completion (simplified)
        setTimeout(() => {
          setModels(models.map(model => 
            model.modelType === modelType 
              ? { ...model, status: 'deployed' as const, lastTraining: new Date().toISOString() }
              : model
          ));
          setIsTraining(null);
        }, 5000);
      }
    } catch (error) {
      console.error('Failed to retrain model:', error);
      setIsTraining(null);
    }
  };

  const downloadModel = async (modelType: string) => {
    try {
      const response = await fetch(`/api/orchestration/ml-models/${modelType}/download`);
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${modelType}-model.pkl`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>ML Model Performance</span>
          </CardTitle>
          <CardDescription>
            Monitor and manage machine learning model performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{models.length}</div>
              <p className="text-sm text-muted-foreground">Active Models</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {models.filter(m => m.accuracy > 0.85).length}
              </div>
              <p className="text-sm text-muted-foreground">High Accuracy</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(models.reduce((sum, m) => sum + m.throughput, 0) / models.length).toFixed(0)}
              </div>
              <p className="text-sm text-muted-foreground">Avg Throughput/s</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {models.filter(m => m.status === 'training').length}
              </div>
              <p className="text-sm text-muted-foreground">Training</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Details */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {models.map((model) => (
              <Card key={model.modelType}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <CardTitle className="capitalize">{model.modelType.replace('_', ' ')}</CardTitle>
                      <Badge variant={getStatusColor(model.status)}>
                        {getStatusIcon(model.status)}
                        <span className="ml-1">{model.status}</span>
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => retrainModel(model.modelType)}
                        disabled={isTraining === model.modelType || model.status === 'training'}
                      >
                        {isTraining === model.modelType ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <Play className="h-4 w-4" />
                        )}
                        <span className="ml-1">Retrain</span>
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadModel(model.modelType)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  <CardDescription>
                    Version {model.version} • Last trained {new Date(model.lastTraining).toLocaleDateString()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Accuracy */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Accuracy</span>
                        <span className="text-sm font-bold">{(model.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={model.accuracy * 100} className="h-2" />
                    </div>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Throughput</div>
                        <div className="font-medium">{model.throughput.toFixed(0)} req/s</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Latency</div>
                        <div className="font-medium">{model.latency.toFixed(0)} ms</div>
                      </div>
                    </div>

                    {/* Benchmark Results */}
                    {model.benchmarkResults && (
                      <div className="border-t pt-3">
                        <div className="text-sm font-medium mb-2">Benchmark Results</div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Precision:</span>
                            <span>{(model.benchmarkResults.precision * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Recall:</span>
                            <span>{(model.benchmarkResults.recall * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">F1 Score:</span>
                            <span>{(model.benchmarkResults.f1Score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">AUC:</span>
                            <span>{(model.benchmarkResults.auc * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          {/* Performance comparison table */}
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Comparison</CardTitle>
              <CardDescription>
                Compare key metrics across all models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Model Type</th>
                      <th className="text-left p-2">Accuracy</th>
                      <th className="text-left p-2">Throughput</th>
                      <th className="text-left p-2">Latency</th>
                      <th className="text-left p-2">Status</th>
                      <th className="text-left p-2">Version</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((model) => (
                      <tr key={model.modelType} className="border-b">
                        <td className="p-2 font-medium capitalize">
                          {model.modelType.replace('_', ' ')}
                        </td>
                        <td className="p-2">
                          <div className="flex items-center space-x-2">
                            <Progress value={model.accuracy * 100} className="w-16 h-2" />
                            <span>{(model.accuracy * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="p-2">{model.throughput.toFixed(0)} req/s</td>
                        <td className="p-2">{model.latency.toFixed(0)} ms</td>
                        <td className="p-2">
                          <Badge variant={getStatusColor(model.status)} className="text-xs">
                            {model.status}
                          </Badge>
                        </td>
                        <td className="p-2 text-muted-foreground">{model.version}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="training" className="space-y-4">
          {/* Training details */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {models.map((model) => (
              <Card key={model.modelType}>
                <CardHeader>
                  <CardTitle className="capitalize">
                    {model.modelType.replace('_', ' ')} Training
                  </CardTitle>
                  <CardDescription>
                    Training configuration and history
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {model.trainingMetrics ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">Epochs</div>
                          <div className="font-medium">{model.trainingMetrics.epochs}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Training Time</div>
                          <div className="font-medium">{model.trainingMetrics.trainingTime}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Training Loss</div>
                          <div className="font-medium">{model.trainingMetrics.trainingLoss.toFixed(4)}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Validation Loss</div>
                          <div className="font-medium">{model.trainingMetrics.validationLoss.toFixed(4)}</div>
                        </div>
                      </div>

                      <div className="border-t pt-3">
                        <div className="text-sm font-medium mb-2">Training Progress</div>
                        <Progress 
                          value={model.status === 'training' ? 45 : 100} 
                          className="h-2" 
                        />
                        <div className="text-xs text-muted-foreground mt-1">
                          {model.status === 'training' ? 'Training in progress...' : 'Training completed'}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground py-4">
                      <Brain className="h-8 w-8 mx-auto mb-2" />
                      <p>No training data available</p>
                      <Button
                        className="mt-2"
                        size="sm"
                        onClick={() => retrainModel(model.modelType)}
                        disabled={isTraining === model.modelType || model.status === 'training'}
                      >
                        Start Training
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {models.length === 0 && (
        <Card>
          <CardContent className="text-center py-8">
            <Brain className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No ML models found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Models will appear here once the ML training system is activated
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}