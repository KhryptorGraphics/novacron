import React, { useMemo, useCallback } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import {
  AlertCircle,
  AlertTriangle,
  Clock,
  Info,
} from 'lucide-react';
import { Alert, VMMetrics } from './MonitoringDashboard';
import { format } from 'date-fns';

// Optimized MetricCard with React.memo and memoized chart data
export const OptimizedMetricCard = React.memo<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  change?: number;
  sparklineData?: number[];
}>(({ title, value, icon, change, sparklineData }) => {
  // Memoize chart configuration to prevent recreation on every render
  const chartData = useMemo(() => {
    if (!sparklineData) return null;
    
    return {
      labels: sparklineData.map((_, i) => ''),
      datasets: [
        {
          data: sparklineData,
          borderColor: 'rgba(59, 130, 246, 0.8)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: true,
        },
      ],
    };
  }, [sparklineData]);

  const chartOptions = useMemo(() => {
    if (!sparklineData) return null;
    
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false, // Disable animations for better performance
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: { display: false },
        y: {
          display: false,
          min: Math.min(...sparklineData) * 0.8,
          max: Math.max(...sparklineData) * 1.2,
        },
      },
    };
  }, [sparklineData]);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <div className="h-4 w-4 text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && (
          <p className={`text-xs ${change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {change > 0 ? '+' : ''}{change}% from last period
          </p>
        )}
        {chartData && chartOptions && (
          <div className="h-10 mt-2">
            <Line data={chartData} options={chartOptions} />
          </div>
        )}
      </CardContent>
    </Card>
  );
});

OptimizedMetricCard.displayName = 'OptimizedMetricCard';

// Optimized AlertItem with memoized severity functions
const getSeverityColor = (severity: Alert['severity']): string => {
  switch (severity) {
    case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    case 'error': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20 dark:text-orange-400';
    case 'warning': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/20 dark:text-amber-400';
    case 'info': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

const getSeverityIcon = (severity: Alert['severity']) => {
  switch (severity) {
    case 'critical': return <AlertCircle className="h-4 w-4" />;
    case 'error': return <AlertCircle className="h-4 w-4" />;
    case 'warning': return <AlertTriangle className="h-4 w-4" />;
    case 'info': return <Info className="h-4 w-4" />;
    default: return <Info className="h-4 w-4" />;
  }
};

const getStatusColor = (status: Alert['status']): string => {
  switch (status) {
    case 'firing': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    case 'acknowledged': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
    case 'resolved': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

export const OptimizedAlertItem = React.memo<{
  alert: Alert;
  onAcknowledge: (id: string) => void;
}>(({ alert, onAcknowledge }) => {
  const handleAcknowledge = useCallback(() => {
    onAcknowledge(alert.id);
  }, [alert.id, onAcknowledge]);

  const severityColorClass = useMemo(() => getSeverityColor(alert.severity), [alert.severity]);
  const severityIcon = useMemo(() => getSeverityIcon(alert.severity), [alert.severity]);
  const statusColorClass = useMemo(() => getStatusColor(alert.status), [alert.status]);
  const formattedTime = useMemo(() => format(new Date(alert.startTime), 'MMM d, HH:mm:ss'), [alert.startTime]);

  return (
    <div className={`p-4 rounded-lg mb-3 ${severityColorClass}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0 mt-0.5">
          {severityIcon}
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium">{alert.name}</h3>
          <div className="mt-1 text-sm">
            <p>{alert.description}</p>
          </div>
          <div className="mt-2 flex justify-between items-center">
            <div className="flex space-x-2 text-xs">
              <Badge variant="outline" className={statusColorClass}>
                {alert.status}
              </Badge>
              <span className="text-xs text-muted-foreground">
                <Clock className="h-3 w-3 inline mr-1" />
                {formattedTime}
              </span>
              <span className="text-xs">{alert.resource}</span>
            </div>
            {alert.status === 'firing' && (
              <Button 
                size="sm" 
                variant="outline"
                onClick={handleAcknowledge}
              >
                Acknowledge
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

OptimizedAlertItem.displayName = 'OptimizedAlertItem';

// Utility functions for formatting
const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

const formatPercentage = (value: number): string => {
  return `${Math.round(value)}%`;
};

const getVmStatusColor = (status: VMMetrics['status']): string => {
  switch (status) {
    case 'running': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
    case 'stopped': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/20 dark:text-amber-400';
    case 'error': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
    default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
  }
};

// Optimized VM row component for virtual scrolling compatibility
export const OptimizedVMRow = React.memo<{
  vm: VMMetrics;
  index: number;
}>(({ vm, index }) => {
  const statusColorClass = useMemo(() => getVmStatusColor(vm.status), [vm.status]);
  const formattedNetworkRx = useMemo(() => formatBytes(vm.networkRx), [vm.networkRx]);
  const formattedNetworkTx = useMemo(() => formatBytes(vm.networkTx), [vm.networkTx]);

  return (
    <tr className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
        {vm.name}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <Badge variant="outline" className={statusColorClass}>
          {vm.status}
        </Badge>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <div className="flex flex-col">
          <span className="text-sm">{formatPercentage(vm.cpuUsage)}</span>
          <Progress value={vm.cpuUsage} className="h-1 w-24" />
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <div className="flex flex-col">
          <span className="text-sm">{formatPercentage(vm.memoryUsage)}</span>
          <Progress value={vm.memoryUsage} className="h-1 w-24" />
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <div className="flex flex-col">
          <span className="text-sm">{formatPercentage(vm.diskUsage)}</span>
          <Progress value={vm.diskUsage} className="h-1 w-24" />
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <div className="text-sm">
          <span className="block">↑ {formattedNetworkTx}/s</span>
          <span className="block">↓ {formattedNetworkRx}/s</span>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {vm.iops} IOPS
      </td>
    </tr>
  );
});

OptimizedVMRow.displayName = 'OptimizedVMRow';

// Memoized chart options for reuse across components
export const memoizedChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    duration: 300, // Reduced animation duration
  },
  plugins: {
    legend: {
      position: 'top' as const,
    },
    tooltip: {
      mode: 'index' as const,
      intersect: false,
    },
  },
  scales: {
    y: {
      beginAtZero: true,
    },
  },
};

// Performance-optimized chart data structure
export interface OptimizedChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    tension?: number;
  }>;
}

// Hook for optimized chart data preparation
export const useOptimizedChartData = (
  timeLabels: string[],
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    tension?: number;
  }>
): OptimizedChartData => {
  return useMemo(() => ({
    labels: timeLabels,
    datasets,
  }), [timeLabels, datasets]);
};

// Performance monitoring utilities
export const usePerformanceMetrics = () => {
  const [metrics, setMetrics] = React.useState({
    renderCount: 0,
    lastRenderTime: 0,
  });

  React.useEffect(() => {
    const startTime = performance.now();
    setMetrics(prev => ({
      renderCount: prev.renderCount + 1,
      lastRenderTime: performance.now() - startTime,
    }));
  });

  return metrics;
};