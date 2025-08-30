"use client";

import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import { TrendingUp, Target, Activity, Clock } from 'lucide-react';

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

interface PlacementDecisionChartProps {
  decisions: OrchestrationDecision[];
}

const COLORS = {
  placement: '#8884d8',
  scaling: '#82ca9d',
  healing: '#ffc658',
  migration: '#ff7c7c',
  optimization: '#8dd1e1',
};

const STATUS_COLORS = {
  executed: '#22c55e',
  pending: '#eab308',
  failed: '#ef4444',
  cancelled: '#6b7280',
};

export function PlacementDecisionChart({ decisions }: PlacementDecisionChartProps) {
  // Process data for different chart types
  const timeSeriesData = useMemo(() => {
    const sortedDecisions = [...decisions]
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map((decision, index) => ({
        time: new Date(decision.timestamp).toLocaleTimeString(),
        score: Math.round(decision.score * 100),
        confidence: Math.round(decision.confidence * 100),
        type: decision.decisionType,
        status: decision.status,
        index,
      }));
    
    return sortedDecisions;
  }, [decisions]);

  const decisionTypeData = useMemo(() => {
    const typeCounts = decisions.reduce((acc, decision) => {
      acc[decision.decisionType] = (acc[decision.decisionType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(typeCounts).map(([type, count]) => ({
      name: type,
      value: count,
      color: COLORS[type as keyof typeof COLORS] || '#8884d8',
    }));
  }, [decisions]);

  const statusData = useMemo(() => {
    const statusCounts = decisions.reduce((acc, decision) => {
      acc[decision.status] = (acc[decision.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(statusCounts).map(([status, count]) => ({
      status,
      count,
      color: STATUS_COLORS[status as keyof typeof STATUS_COLORS] || '#6b7280',
    }));
  }, [decisions]);

  const scatterData = useMemo(() => {
    return decisions.map((decision) => ({
      score: decision.score * 100,
      confidence: decision.confidence * 100,
      type: decision.decisionType,
      status: decision.status,
      id: decision.id,
    }));
  }, [decisions]);

  const averageMetrics = useMemo(() => {
    if (decisions.length === 0) return { avgScore: 0, avgConfidence: 0, successRate: 0 };
    
    const avgScore = decisions.reduce((sum, d) => sum + d.score, 0) / decisions.length;
    const avgConfidence = decisions.reduce((sum, d) => sum + d.confidence, 0) / decisions.length;
    const successRate = decisions.filter(d => d.status === 'executed').length / decisions.length;
    
    return {
      avgScore: avgScore * 100,
      avgConfidence: avgConfidence * 100,
      successRate: successRate * 100,
    };
  }, [decisions]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3">
          <p className="font-medium">{label}</p>
          <p className="text-sm text-muted-foreground">Type: {data.type}</p>
          <p className="text-sm">Score: {payload[0].value}%</p>
          <p className="text-sm">Confidence: {payload[1]?.value || data.confidence}%</p>
          <Badge variant={data.status === 'executed' ? 'success' : 'warning'}>
            {data.status}
          </Badge>
        </div>
      );
    }
    return null;
  };

  const ScatterTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3">
          <p className="font-medium">Decision Analysis</p>
          <p className="text-sm">Type: {data.type}</p>
          <p className="text-sm">Score: {data.score.toFixed(1)}%</p>
          <p className="text-sm">Confidence: {data.confidence.toFixed(1)}%</p>
          <p className="text-sm">ID: {data.id}</p>
          <Badge variant={data.status === 'executed' ? 'success' : 'warning'}>
            {data.status}
          </Badge>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Overview Metrics */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Placement Decision Overview</span>
          </CardTitle>
          <CardDescription>
            Performance metrics for orchestration decisions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {decisions.length}
              </div>
              <p className="text-sm text-muted-foreground">Total Decisions</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {averageMetrics.avgScore.toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">Average Score</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {averageMetrics.avgConfidence.toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">Average Confidence</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {averageMetrics.successRate.toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">Success Rate</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Time Series Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Decision Trends</span>
          </CardTitle>
          <CardDescription>
            Score and confidence over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis domain={[0, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#8884d8"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Score"
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Confidence"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Scatter Plot */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Score vs Confidence</span>
          </CardTitle>
          <CardDescription>
            Decision quality analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={scatterData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="score" 
                  domain={[0, 100]}
                  label={{ value: 'Score (%)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="confidence" 
                  domain={[0, 100]}
                  label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<ScatterTooltip />} />
                <Scatter
                  name="Decisions"
                  data={scatterData}
                  fill="#8884d8"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Decision Types Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Decision Types</CardTitle>
          <CardDescription>
            Distribution of decision types
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={decisionTypeData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {decisionTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Status Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Clock className="h-5 w-5" />
            <span>Decision Status</span>
          </CardTitle>
          <CardDescription>
            Execution status breakdown
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statusData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="status" type="category" />
                <Tooltip />
                <Bar dataKey="count" fill="#8884d8">
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}