# Advanced Visualization Components

This directory contains a set of advanced visualization components for the monitoring dashboard. These components provide rich, interactive visualizations for system monitoring, analytics, and troubleshooting.

## Components

### HeatmapChart

A heatmap visualization for displaying resource usage patterns over time across multiple resources.

```tsx
import { HeatmapChart } from '@/components/visualizations';

<HeatmapChart
  title="CPU Usage Patterns"
  description="Heatmap showing CPU usage patterns across VMs over time"
  data={[
    { timestamp: '2025-04-11T00:00:00Z', resourceId: 'VM-1', value: 45 },
    // More data points...
  ]}
/>
```

### NetworkTopology

An interactive network topology visualization that displays the relationships between different system components.

```tsx
import { NetworkTopology } from '@/components/visualizations';

<NetworkTopology
  title="System Topology"
  description="Interactive visualization of system components and their relationships"
  data={{
    nodes: [
      { id: 'host-1', name: 'Host-1', type: 'host', status: 'healthy' },
      // More nodes...
    ],
    edges: [
      { source: 'host-1', target: 'vm-1', type: 'dependency' },
      // More edges...
    ],
  }}
  height={350}
  onNodeClick={(node) => console.log('Node clicked:', node)}
/>
```

### PredictiveChart

A time-series chart with forecasting capabilities, showing historical data, predictions, confidence intervals, and anomalies.

```tsx
import { PredictiveChart } from '@/components/visualizations';

<PredictiveChart
  title="CPU Usage Forecast"
  description="Predicted CPU usage based on historical patterns"
  metricName="CPU Usage"
  metricUnit="%"
  historicalData={[
    { timestamp: '2025-04-10T00:00:00Z', value: 45 },
    // More historical data points...
  ]}
  predictedData={[
    { timestamp: '2025-04-11T00:00:00Z', value: 48, upperBound: 55, lowerBound: 41 },
    // More prediction data points...
  ]}
  anomalies={[
    { timestamp: '2025-04-11T05:00:00Z', severity: 'medium', message: 'Unusual spike in CPU usage' }
  ]}
/>
```

### ResourceTreemap

A treemap visualization for hierarchical resource utilization data, allowing users to quickly identify which resources are consuming the most capacity.

```tsx
import { ResourceTreemap } from '@/components/visualizations';

<ResourceTreemap
  title="Resource Utilization"
  description="Hierarchical view of resource usage across the system"
  data={{
    id: 'root',
    name: 'System',
    value: 100,
    children: [
      {
        id: 'compute',
        name: 'Compute',
        value: 65,
        children: [
          { id: 'vm-1', name: 'VM-1', value: 78 },
          // More children...
        ]
      },
      // More top-level categories...
    ]
  }}
  height={350}
  onNodeClick={(node) => console.log('Node clicked:', node)}
/>
```

### AlertCorrelation

A visualization for understanding the relationships between different alerts and their potential root causes.

```tsx
import { AlertCorrelation } from '@/components/visualizations';

<AlertCorrelation
  alerts={[
    {
      id: 'alert-1',
      name: 'High CPU Usage',
      description: 'VM-5 CPU usage exceeds 90%',
      severity: 'warning',
      status: 'firing',
      startTime: new Date().toISOString(),
      labels: { host: 'Host-1', vm: 'VM-5', metric: 'cpu' },
      value: 95,
      resource: 'VM-5',
    },
    // More alerts...
  ]}
  relations={[
    { source: 'alert-3', target: 'alert-1', type: 'causes', confidence: 0.85 },
    // More relations...
  ]}
  onAcknowledge={(id) => console.log('Alert acknowledged:', id)}
  onAlertClick={(alert) => console.log('Alert clicked:', alert)}
/>
```

## Usage Guidelines

1. **Data Formatting**: Ensure that data passed to these components follows the expected format as defined in the component interfaces.

2. **Responsive Design**: All components are designed to be responsive and will adapt to their container size. You can also specify a custom height for most components.

3. **Interactivity**: Many components support interactive features like clicking on elements. You can provide callback functions to handle these interactions.

4. **Theming**: Components automatically adapt to light/dark mode based on the application theme.

5. **Performance**: For large datasets, consider implementing pagination or data filtering to maintain performance.

## Best Practices

1. **Real-time Updates**: When using these components with real-time data, consider implementing debouncing or throttling to prevent excessive re-renders.

2. **Error Handling**: Implement proper error handling for data fetching and processing to ensure the visualizations degrade gracefully when data is missing or malformed.

3. **Accessibility**: While these components provide visual insights, ensure that critical information is also available in text form for screen readers.

4. **Mobile Considerations**: Some visualizations (like NetworkTopology) may be complex on small screens. Consider providing alternative views for mobile users.