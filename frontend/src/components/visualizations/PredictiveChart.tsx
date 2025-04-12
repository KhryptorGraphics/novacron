import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Line } from 'react-chartjs-2';
import { format, addHours, addDays } from 'date-fns';
import { Badge } from '@/components/ui/badge';
import { Info } from 'lucide-react';

interface DataPoint {
  timestamp: string;
  value: number;
}

interface PredictionPoint extends DataPoint {
  upperBound?: number;
  lowerBound?: number;
}

interface PredictiveChartProps {
  title: string;
  description?: string;
  historicalData: DataPoint[];
  predictedData: PredictionPoint[];
  metricName: string;
  metricUnit: string;
  anomalies?: {
    timestamp: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
  }[];
  timeGranularity?: 'hourly' | 'daily';
}

export const PredictiveChart: React.FC<PredictiveChartProps> = ({
  title,
  description,
  historicalData,
  predictedData,
  metricName,
  metricUnit,
  anomalies = [],
  timeGranularity = 'hourly',
}) => {
  // Format timestamps based on granularity
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return timeGranularity === 'hourly' 
      ? format(date, 'HH:mm')
      : format(date, 'MMM dd');
  };
  
  // Generate future timestamps for x-axis
  const generateFutureLabels = () => {
    const lastHistoricalPoint = new Date(historicalData[historicalData.length - 1].timestamp);
    const futureLabels = [];
    
    for (let i = 0; i < predictedData.length; i++) {
      const futureDate = timeGranularity === 'hourly'
        ? addHours(lastHistoricalPoint, i + 1)
        : addDays(lastHistoricalPoint, i + 1);
      
      futureLabels.push(formatTimestamp(futureDate.toISOString()));
    }
    
    return futureLabels;
  };
  
  // Prepare chart data
  const chartData = {
    labels: [
      ...historicalData.map(point => formatTimestamp(point.timestamp)),
      ...generateFutureLabels(),
    ],
    datasets: [
      // Historical data
      {
        label: `Historical ${metricName}`,
        data: [...historicalData.map(point => point.value), ...Array(predictedData.length).fill(null)],
        borderColor: 'rgba(59, 130, 246, 1)', // Blue
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        tension: 0.4,
        fill: false,
      },
      // Predicted data
      {
        label: `Predicted ${metricName}`,
        data: [...Array(historicalData.length).fill(null), ...predictedData.map(point => point.value)],
        borderColor: 'rgba(139, 92, 246, 1)', // Purple
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 3,
        pointBackgroundColor: 'rgba(139, 92, 246, 1)',
        tension: 0.4,
        fill: false,
      },
      // Upper confidence bound
      {
        label: 'Upper Bound',
        data: [
          ...Array(historicalData.length).fill(null),
          ...predictedData.map(point => point.upperBound || null),
        ],
        borderColor: 'rgba(139, 92, 246, 0.3)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        fill: '+1', // Fill between this dataset and the next one
        tension: 0.4,
      },
      // Lower confidence bound
      {
        label: 'Lower Bound',
        data: [
          ...Array(historicalData.length).fill(null),
          ...predictedData.map(point => point.lowerBound || null),
        ],
        borderColor: 'rgba(139, 92, 246, 0.3)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0.4,
      },
      // Anomalies
      {
        label: 'Anomalies',
        data: anomalies.map(anomaly => {
          const anomalyIndex = [...historicalData, ...predictedData].findIndex(
            point => point.timestamp === anomaly.timestamp
          );
          return anomalyIndex >= 0 
            ? [...historicalData, ...predictedData][anomalyIndex].value 
            : null;
        }),
        borderColor: 'rgba(239, 68, 68, 1)', // Red
        backgroundColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 0,
        pointRadius: 6,
        pointStyle: 'triangle',
        pointRotation: 180,
        showLine: false,
      },
    ],
  };
  
  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 12,
          usePointStyle: true,
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: function(context: any) {
            const label = context.dataset.label || '';
            const value = context.parsed.y !== null 
              ? `${context.parsed.y.toFixed(2)} ${metricUnit}` 
              : 'No data';
            return `${label}: ${value}`;
          },
          afterBody: function(tooltipItems: any[]) {
            const index = tooltipItems[0].dataIndex;
            const label = tooltipItems[0].label;
            
            // Check if there's an anomaly at this point
            const anomaly = anomalies.find(a => {
              const anomalyLabel = formatTimestamp(a.timestamp);
              return anomalyLabel === label;
            });
            
            if (anomaly) {
              return [`Anomaly: ${anomaly.message}`];
            }
            
            return [];
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
        },
      },
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: metricUnit,
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.1)',
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  };
  
  // Get the prediction accuracy (simple metric based on confidence interval width)
  const getPredictionAccuracy = () => {
    if (!predictedData.some(point => point.upperBound && point.lowerBound)) {
      return null;
    }
    
    const confidenceWidths = predictedData
      .filter(point => point.upperBound && point.lowerBound)
      .map(point => (point.upperBound! - point.lowerBound!) / point.value);
    
    const avgWidth = confidenceWidths.reduce((sum, width) => sum + width, 0) / confidenceWidths.length;
    
    // Convert to a percentage accuracy (narrower confidence interval = higher accuracy)
    const accuracyScore = Math.max(0, Math.min(100, 100 - (avgWidth * 50)));
    
    return Math.round(accuracyScore);
  };
  
  const accuracy = getPredictionAccuracy();
  
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div>
          <CardTitle>{title}</CardTitle>
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
        {accuracy !== null && (
          <Badge variant="outline" className="flex items-center gap-1">
            <Info className="h-3 w-3" />
            <span>Prediction Accuracy: {accuracy}%</span>
          </Badge>
        )}
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <Line data={chartData} options={chartOptions} />
        </div>
        <div className="mt-4">
          <h4 className="text-sm font-medium mb-2">Insights</h4>
          <div className="space-y-2">
            {predictedData.length > 0 && (
              <div className="text-sm">
                <span className="font-medium">Forecast: </span>
                {predictedData[predictedData.length - 1].value > historicalData[historicalData.length - 1].value
                  ? `${metricName} is predicted to increase by ${((predictedData[predictedData.length - 1].value / historicalData[historicalData.length - 1].value - 1) * 100).toFixed(1)}% in the next ${timeGranularity === 'hourly' ? 'hours' : 'days'}.`
                  : `${metricName} is predicted to decrease by ${((1 - predictedData[predictedData.length - 1].value / historicalData[historicalData.length - 1].value) * 100).toFixed(1)}% in the next ${timeGranularity === 'hourly' ? 'hours' : 'days'}.`
                }
              </div>
            )}
            {anomalies.length > 0 && (
              <div className="text-sm">
                <span className="font-medium">Anomalies: </span>
                {`Detected ${anomalies.length} anomalies in the data.`}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};