import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { format } from 'date-fns';

interface HeatmapProps {
  data: {
    timestamp: string;
    resourceId: string;
    value: number;
  }[];
  title: string;
  description?: string;
  colorScale?: string[];
  xAxisLabel?: string;
  yAxisLabel?: string;
}

const defaultColorScale = [
  '#f7fbff', // Very low - almost white
  '#deebf7', // Low
  '#c6dbef', // Low-medium
  '#9ecae1', // Medium
  '#6baed6', // Medium-high
  '#4292c6', // High
  '#2171b5', // Very high
  '#084594', // Extremely high
];

export const HeatmapChart: React.FC<HeatmapProps> = ({
  data,
  title,
  description,
  colorScale = defaultColorScale,
  xAxisLabel = 'Time',
  yAxisLabel = 'Resource',
}) => {
  // Process data to get unique resources and timestamps
  const resources = Array.from(new Set(data.map(d => d.resourceId))).sort();
  const timestamps = Array.from(new Set(data.map(d => d.timestamp))).sort();
  
  // Find min and max values for color scaling
  const values = data.map(d => d.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  
  // Function to get color based on value
  const getColor = (value: number) => {
    const normalizedValue = (value - minValue) / (maxValue - minValue);
    const colorIndex = Math.min(
      Math.floor(normalizedValue * colorScale.length),
      colorScale.length - 1
    );
    return colorScale[colorIndex];
  };
  
  // Function to format timestamp for display
  const formatTime = (timestamp: string) => {
    return format(new Date(timestamp), 'HH:mm');
  };
  
  // Calculate cell dimensions
  const cellWidth = 100 / timestamps.length;
  const cellHeight = 30; // pixels
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="min-w-max">
            {/* Y-axis label */}
            <div className="flex mb-2">
              <div className="w-24 flex-shrink-0"></div>
              <div className="flex-grow text-center text-sm font-medium text-muted-foreground">
                {xAxisLabel}
              </div>
            </div>
            
            {/* X-axis labels (timestamps) */}
            <div className="flex mb-2">
              <div className="w-24 flex-shrink-0 text-right pr-2 text-sm font-medium text-muted-foreground">
                {yAxisLabel}
              </div>
              <div className="flex-grow flex">
                {timestamps.map((timestamp, i) => (
                  <div 
                    key={`timestamp-${i}`}
                    className="text-xs text-center"
                    style={{ width: `${cellWidth}%` }}
                  >
                    {formatTime(timestamp)}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Heatmap grid */}
            <div>
              {resources.map((resource, resourceIndex) => (
                <div key={`resource-${resourceIndex}`} className="flex mb-1">
                  <div className="w-24 flex-shrink-0 text-right pr-2 text-xs truncate" title={resource}>
                    {resource}
                  </div>
                  <div className="flex-grow flex">
                    {timestamps.map((timestamp, timeIndex) => {
                      const dataPoint = data.find(
                        d => d.resourceId === resource && d.timestamp === timestamp
                      );
                      const value = dataPoint ? dataPoint.value : 0;
                      
                      return (
                        <div
                          key={`cell-${resourceIndex}-${timeIndex}`}
                          className="border border-gray-100 dark:border-gray-800 rounded-sm"
                          style={{
                            width: `${cellWidth}%`,
                            height: `${cellHeight}px`,
                            backgroundColor: getColor(value),
                            cursor: 'pointer',
                          }}
                          title={`${resource}: ${value.toFixed(2)} at ${formatTime(timestamp)}`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
            
            {/* Legend */}
            <div className="mt-4 flex items-center">
              <span className="text-xs mr-2">Low</span>
              <div className="flex h-2">
                {colorScale.map((color, i) => (
                  <div
                    key={`legend-${i}`}
                    style={{
                      backgroundColor: color,
                      width: '20px',
                      height: '100%',
                    }}
                  />
                ))}
              </div>
              <span className="text-xs ml-2">High</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};