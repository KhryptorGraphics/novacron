"use client";

import React from "react";

// This is a simplified example. In a production app, you'd use a more robust charting library
// like Chart.js or D3.js for more sophisticated visualizations
interface ChartData {
  x: string;
  y: number;
}

interface LineChartProps {
  data: ChartData[];
  xLabel: string;
  yLabel: string;
}

export function LineChart({ data, xLabel, yLabel }: LineChartProps) {
  if (!data || data.length === 0) {
    return <div className="flex items-center justify-center h-full">No data available</div>;
  }

  // Find min and max values for scaling
  const minY = Math.min(...data.map(d => d.y));
  const maxY = Math.max(...data.map(d => d.y));
  const range = maxY - minY || 1; // Avoid division by zero

  // Simple SVG chart implementation
  return (
    <div className="w-full h-full">
      <svg width="100%" height="100%" viewBox="0 0 800 400" preserveAspectRatio="none">
        {/* Y-axis label */}
        <text
          x="10"
          y="200"
          transform="rotate(-90, 10, 200)"
          textAnchor="middle"
          className="fill-gray-500 text-xs"
        >
          {yLabel}
        </text>

        {/* X-axis label */}
        <text
          x="400"
          y="390"
          textAnchor="middle"
          className="fill-gray-500 text-xs"
        >
          {xLabel}
        </text>

        {/* Y-axis */}
        <line x1="60" y1="350" x2="60" y2="50" stroke="currentColor" strokeOpacity="0.2" />
        
        {/* X-axis */}
        <line x1="60" y1="350" x2="740" y2="350" stroke="currentColor" strokeOpacity="0.2" />
        
        {/* Y-axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((tick, i) => {
          const y = 350 - tick * 300;
          const value = minY + tick * range;
          return (
            <React.Fragment key={i}>
              <line x1="55" y1={y} x2="60" y2={y} stroke="currentColor" strokeOpacity="0.2" />
              <text x="50" y={y + 5} textAnchor="end" className="fill-gray-500 text-xs">
                {value.toFixed(1)}
              </text>
              <line x1="60" y1={y} x2="740" y2={y} stroke="currentColor" strokeOpacity="0.1" strokeDasharray="4 4" />
            </React.Fragment>
          );
        })}
        
        {/* X-axis ticks - show only a subset for clarity */}
        {data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 5)) === 0).map((d, i, filtered) => {
          const x = 60 + (i / (filtered.length - 1 || 1)) * 680;
          return (
            <React.Fragment key={i}>
              <line x1={x} y1="350" x2={x} y2="355" stroke="currentColor" strokeOpacity="0.2" />
              <text x={x} y="370" textAnchor="middle" className="fill-gray-500 text-xs">
                {d.x}
              </text>
              <line x1={x} y1="50" x2={x} y2="350" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="4 4" />
            </React.Fragment>
          );
        })}
        
        {/* Line */}
        <polyline
          points={data
            .map((d, i) => {
              const x = 60 + (i / (data.length - 1 || 1)) * 680;
              const y = 350 - ((d.y - minY) / range) * 300;
              return `${x},${y}`;
            })
            .join(" ")}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
        />
        
        {/* Data points */}
        {data.map((d, i) => {
          const x = 60 + (i / (data.length - 1 || 1)) * 680;
          const y = 350 - ((d.y - minY) / range) * 300;
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="3"
              fill="#3b82f6"
            />
          );
        })}
      </svg>
    </div>
  );
}

interface PieChartDataPoint {
  name: string;
  value: number;
}

interface PieChartProps {
  data: PieChartDataPoint[];
  label: string;
}

export function PieChart({ data, label }: PieChartProps) {
  if (!data || data.length === 0) {
    return <div className="flex items-center justify-center h-full">No data available</div>;
  }

  const total = data.reduce((sum, d) => sum + d.value, 0);
  
  // Colors for the pie chart
  const colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
  
  // Calculate pie segments
  let cumulativeAngle = 0;
  const segments = data.map((d, i) => {
    const startAngle = cumulativeAngle;
    const angle = (d.value / total) * 360;
    cumulativeAngle += angle;
    
    return {
      ...d,
      startAngle,
      angle,
      color: colors[i % colors.length],
    };
  });
  
  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <div className="relative w-64 h-64">
        <svg width="100%" height="100%" viewBox="0 0 100 100">
          {segments.map((segment, i) => {
            const startAngleRad = (segment.startAngle * Math.PI) / 180;
            const endAngleRad = ((segment.startAngle + segment.angle) * Math.PI) / 180;
            
            // Calculate arc path
            const x1 = 50 + 40 * Math.cos(startAngleRad);
            const y1 = 50 + 40 * Math.sin(startAngleRad);
            const x2 = 50 + 40 * Math.cos(endAngleRad);
            const y2 = 50 + 40 * Math.sin(endAngleRad);
            
            const largeArcFlag = segment.angle > 180 ? 1 : 0;
            
            const pathData = `
              M 50 50
              L ${x1} ${y1}
              A 40 40 0 ${largeArcFlag} 1 ${x2} ${y2}
              Z
            `;
            
            return (
              <path
                key={i}
                d={pathData}
                fill={segment.color}
                stroke="#fff"
                strokeWidth="1"
              />
            );
          })}
        </svg>
      </div>
      
      <div className="mt-4">
        <div className="text-center font-medium mb-2">{label}</div>
        <div className="flex flex-wrap justify-center gap-4">
          {segments.map((segment, i) => (
            <div key={i} className="flex items-center">
              <div
                className="w-3 h-3 mr-1"
                style={{ backgroundColor: segment.color }}
              ></div>
              <div className="text-sm">
                {segment.name}: {((segment.value / total) * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
