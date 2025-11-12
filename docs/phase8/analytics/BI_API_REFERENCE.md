# DWCP v3 BI API Reference

## GraphQL API

### Base URL
```
https://api.dwcp.io/graphql
```

### Authentication
```http
Authorization: Bearer <token>
X-API-Key: <api-key>
```

## Query Reference

### Cost Intelligence Queries

#### getCostMetrics

Get cost metrics with flexible filtering and aggregation.

```graphql
query GetCostMetrics {
  getCostMetrics(
    startDate: "2024-01-01T00:00:00Z"
    endDate: "2024-01-31T23:59:59Z"
    providers: [AWS, AZURE, GCP]
    services: ["EC2", "S3", "Lambda"]
    groupBy: [SERVICE, REGION]
    aggregation: SUM
  ) {
    metrics {
      timestamp
      provider
      service
      resourceId
      cost
      usage
      unit
      tags
    }
    summary {
      totalCost
      averageDailyCost
      minCost
      maxCost
      trend
    }
    groupedData
  }
}
```

**Parameters:**
- `startDate` (Time!): Start of date range
- `endDate` (Time!): End of date range
- `providers` ([CloudProvider!]): Cloud providers to include
- `services` ([String!]): Services to include
- `groupBy` ([GroupByField!]): Grouping dimensions
- `aggregation` (AggregationType): Aggregation method

**Response:**
```json
{
  "data": {
    "getCostMetrics": {
      "metrics": [
        {
          "timestamp": "2024-01-01T00:00:00Z",
          "provider": "AWS",
          "service": "EC2",
          "resourceId": "i-1234567890",
          "cost": 123.45,
          "usage": 720.0,
          "unit": "instance-hours",
          "tags": {
            "Department": "Engineering",
            "Environment": "production"
          }
        }
      ],
      "summary": {
        "totalCost": 45623.78,
        "averageDailyCost": 1471.41,
        "minCost": 1234.56,
        "maxCost": 1678.90,
        "trend": "INCREASING"
      },
      "groupedData": {
        "by_service": {
          "EC2": 25000.00,
          "S3": 15000.00,
          "Lambda": 5623.78
        }
      }
    }
  }
}
```

#### getCostForecast

Generate cost forecasts for specified periods.

```graphql
query GetCostForecast {
  getCostForecast(
    provider: AWS
    service: "EC2"
    periods: [DAY_30, DAY_90, DAY_180]
  ) {
    provider
    service
    period
    currentCost
    predictedCost
    confidenceLower
    confidenceUpper
    trend
    accuracy
    recommendations
  }
}
```

**Response:**
```json
{
  "data": {
    "getCostForecast": [
      {
        "provider": "AWS",
        "service": "EC2",
        "period": "DAY_30",
        "currentCost": 25000.00,
        "predictedCost": 29000.00,
        "confidenceLower": 27000.00,
        "confidenceUpper": 31000.00,
        "trend": "INCREASING",
        "accuracy": 0.94,
        "recommendations": [
          "Consider reserved instances for steady-state workloads",
          "Review instance right-sizing opportunities"
        ]
      }
    ]
  }
}
```

#### getCostAnomalies

Detect cost anomalies with configurable thresholds.

```graphql
query GetCostAnomalies {
  getCostAnomalies(
    provider: AWS
    threshold: 0.05
    severity: HIGH
  ) {
    id
    timestamp
    provider
    service
    expectedCost
    actualCost
    deviationPercentage
    severity
    probableCauses
    recommendedActions
  }
}
```

#### getOptimizationRecommendations

Get AI-powered cost optimization recommendations.

```graphql
query GetOptimizationRecommendations {
  getOptimizationRecommendations(
    provider: AWS
    minSavings: 1000.0
    maxRisk: MEDIUM
  ) {
    id
    provider
    service
    resourceId
    currentCost
    optimizedCost
    annualSavings
    roiPercentage
    effort
    risk
    actions {
      type
      description
      estimatedTime
      prerequisites
    }
    impact
  }
}
```

### Capacity Planning Queries

#### getCapacityForecast

Get capacity forecasts for specified resources.

```graphql
query GetCapacityForecast {
  getCapacityForecast(
    resourceType: CPU
    periods: [DAY_7, DAY_30, DAY_90]
  ) {
    resourceType
    period
    currentCapacity
    forecastedDemand
    recommendedCapacity
    exhaustionDate
    confidence
    scalingRecommendations {
      action
      targetCapacity
      estimatedCost
      urgency
      timeline
    }
  }
}
```

#### getResourceUtilization

Get resource utilization statistics.

```graphql
query GetResourceUtilization {
  getResourceUtilization(
    resourceType: MEMORY
    startDate: "2024-01-01T00:00:00Z"
    endDate: "2024-01-31T23:59:59Z"
    aggregation: AVG
  ) {
    utilization {
      timestamp
      usage
      capacity
      utilizationPercentage
    }
    statistics {
      average
      median
      p95
      p99
      max
      min
    }
  }
}
```

#### getGrowthTrends

Analyze resource growth trends.

```graphql
query GetGrowthTrends {
  getGrowthTrends(
    resourceType: STORAGE
    lookbackDays: 180
  ) {
    trendType
    growthRate
    acceleration
    seasonality {
      pattern
      strength
      peakPeriods
    }
    forecastConfidence
  }
}
```

### Real-time Analytics Queries

#### getRealtimeMetrics

Get real-time metric values.

```graphql
query GetRealtimeMetrics {
  getRealtimeMetrics(
    metricNames: ["cpu_usage", "memory_usage", "request_rate"]
    windowSize: "5m"
  ) {
    name
    value
    timestamp
    tags
    trend {
      direction
      magnitude
    }
  }
}
```

#### getDashboard

Get complete dashboard data.

```graphql
query GetDashboard {
  getDashboard(
    dashboardId: "executive-overview"
    timeRange: {
      from: "now-24h"
      to: "now"
    }
  ) {
    id
    title
    description
    panels {
      id
      type
      title
      data
      config
    }
    timeRange {
      from
      to
    }
    refreshInterval
  }
}
```

### Executive Analytics Queries

#### getExecutiveSummary

Get executive summary with key metrics.

```graphql
query GetExecutiveSummary {
  getExecutiveSummary(period: MONTHLY) {
    period
    totalCost
    costTrend
    savingsRealized
    systemAvailability
    slaCompliance
    activeResources
    incidents
    keyMetrics {
      name
      value
      target
      status
      trend
      sparkline
    }
    recommendations
  }
}
```

**Response:**
```json
{
  "data": {
    "getExecutiveSummary": {
      "period": "MONTHLY",
      "totalCost": 87543.21,
      "costTrend": "STABLE",
      "savingsRealized": 24312.50,
      "systemAvailability": 99.995,
      "slaCompliance": 99.7,
      "activeResources": 12543,
      "incidents": 3,
      "keyMetrics": [
        {
          "name": "Cost Efficiency",
          "value": 82.5,
          "target": 85.0,
          "status": "ON_TRACK",
          "trend": "IMPROVING",
          "sparkline": [78, 79, 81, 82, 82.5]
        }
      ],
      "recommendations": [
        "Implement reserved instances to save $15K annually",
        "Right-size 23 underutilized instances"
      ]
    }
  }
}
```

#### getKPIMetrics

Get all KPI metrics.

```graphql
query GetKPIMetrics {
  getKPIMetrics {
    name
    value
    target
    status
    trend
    sparkline
  }
}
```

#### getSLACompliance

Get SLA compliance report.

```graphql
query GetSLACompliance {
  getSLACompliance(services: ["api", "web", "database"]) {
    service
    availability
    latency {
      p50
      p95
      p99
    }
    errorRate
    compliance
    breaches {
      timestamp
      metric
      threshold
      actual
      duration
    }
  }
}
```

## Mutation Reference

### Cost Management Mutations

#### implementOptimization

Implement a cost optimization recommendation.

```graphql
mutation ImplementOptimization {
  implementOptimization(
    recommendationId: "rec-123"
    schedule: "2024-02-01T00:00:00Z"
  ) {
    success
    message
    estimatedSavings
    scheduledFor
    implementationSteps {
      step
      status
      completedAt
    }
  }
}
```

### Capacity Planning Mutations

#### createScenario

Create a what-if scenario for capacity planning.

```graphql
mutation CreateScenario {
  createScenario(
    input: {
      name: "Aggressive Growth"
      growthRate: 150.0
      seasonalityFactor: 1.2
      spikeProbability: 0.3
      externalFactors: {
        marketingCampaign: 0.5
        productLaunch: 0.8
      }
    }
  ) {
    id
    name
    capacityRequirements
    costImplications
    feasibilityScore
  }
}
```

#### runWhatIfAnalysis

Run what-if analysis on scenarios.

```graphql
mutation RunWhatIfAnalysis {
  runWhatIfAnalysis(scenarioIds: ["aggressive", "conservative"]) {
    scenarios {
      id
      name
      capacityRequirements
      costImplications
      riskAssessment
    }
    recommendation
    comparisonChart
  }
}
```

### Dashboard Management Mutations

#### createDashboard

Create a custom dashboard.

```graphql
mutation CreateDashboard {
  createDashboard(
    input: {
      title: "Cost Analysis Dashboard"
      description: "Detailed cost breakdown"
      panels: [
        {
          type: PIE_CHART
          title: "Cost by Service"
          query: "sum by (service) (dwcp_cost_by_service)"
          gridPos: { x: 0, y: 0, w: 6, h: 8 }
        }
      ]
      timeRange: { from: "now-30d", to: "now" }
      refreshInterval: "1m"
    }
  ) {
    id
    title
    url
    createdAt
  }
}
```

#### updateDashboard

Update an existing dashboard.

```graphql
mutation UpdateDashboard {
  updateDashboard(
    id: "dash-123"
    input: {
      title: "Updated Dashboard"
      refreshInterval: "30s"
    }
  ) {
    id
    title
    updatedAt
  }
}
```

### Alert Management Mutations

#### createAlert

Create a new alert rule.

```graphql
mutation CreateAlert {
  createAlert(
    input: {
      name: "High Cost Alert"
      condition: "sum(dwcp_cost_hourly) > 1000"
      duration: "5m"
      severity: WARNING
      notifications: [SLACK, EMAIL]
      annotations: {
        description: "Hourly cost exceeded $1000"
        runbook: "https://docs.dwcp.io/runbooks/high-cost"
      }
    }
  ) {
    id
    name
    status
    createdAt
  }
}
```

#### acknowledgeAlert

Acknowledge an active alert.

```graphql
mutation AcknowledgeAlert {
  acknowledgeAlert(alertId: "alert-456") {
    id
    status
    acknowledgedBy
    acknowledgedAt
  }
}
```

## Subscription Reference

### Real-time Subscriptions

#### streamMetrics

Stream real-time metric updates.

```graphql
subscription StreamMetrics {
  streamMetrics(metricNames: ["cpu_usage", "memory_usage"]) {
    name
    value
    timestamp
    trend {
      direction
      magnitude
    }
  }
}
```

**WebSocket Connection:**
```javascript
const ws = new WebSocket('wss://api.dwcp.io/graphql');

ws.send(JSON.stringify({
  type: 'connection_init',
  payload: {
    headers: {
      Authorization: 'Bearer <token>'
    }
  }
}));

ws.send(JSON.stringify({
  type: 'start',
  id: '1',
  payload: {
    query: `
      subscription {
        streamMetrics(metricNames: ["cpu_usage"]) {
          name
          value
          timestamp
        }
      }
    `
  }
}));
```

#### costAlerts

Subscribe to cost alert notifications.

```graphql
subscription CostAlerts {
  costAlerts(provider: AWS) {
    id
    timestamp
    provider
    service
    expectedCost
    actualCost
    severity
    message
  }
}
```

#### capacityAlerts

Subscribe to capacity alert notifications.

```graphql
subscription CapacityAlerts {
  capacityAlerts(resourceType: CPU) {
    id
    timestamp
    resourceType
    currentUsage
    capacity
    utilizationPercentage
    daysToExhaustion
    severity
    recommendations
  }
}
```

## Type Reference

### Enumerations

#### CloudProvider
```graphql
enum CloudProvider {
  AWS
  AZURE
  GCP
  ORACLE
  IBM
  ALIBABA
  ON_PREMISE
}
```

#### ResourceType
```graphql
enum ResourceType {
  CPU
  MEMORY
  STORAGE
  NETWORK
  GPU
}
```

#### ForecastPeriod
```graphql
enum ForecastPeriod {
  DAY_7
  DAY_30
  DAY_90
  DAY_180
  YEAR_1
}
```

#### AggregationType
```graphql
enum AggregationType {
  SUM
  AVG
  MIN
  MAX
  P50
  P95
  P99
}
```

#### Severity
```graphql
enum Severity {
  LOW
  MEDIUM
  HIGH
  CRITICAL
}
```

#### TrendType
```graphql
enum TrendType {
  INCREASING
  DECREASING
  STABLE
  VOLATILE
}
```

### Scalar Types

#### Time
ISO 8601 datetime string
```
"2024-01-01T00:00:00Z"
```

#### Duration
Duration string
```
"5m"  # 5 minutes
"1h"  # 1 hour
"30s" # 30 seconds
```

#### JSON
Arbitrary JSON data
```json
{
  "key": "value",
  "nested": {
    "data": [1, 2, 3]
  }
}
```

## REST API Endpoints

### Export Endpoints

#### Export to Tableau
```http
POST /api/v1/export/tableau
Content-Type: application/json

{
  "query": "SELECT * FROM cost_metrics WHERE date >= '2024-01-01'",
  "format": "hyper"
}

Response:
{
  "downloadUrl": "https://api.dwcp.io/downloads/tableau-export-123.hyper",
  "expiresAt": "2024-01-01T12:00:00Z"
}
```

#### Export to PowerBI
```http
POST /api/v1/export/powerbi
Content-Type: application/json

{
  "dataset": "DWCP Analytics",
  "tables": ["cost_metrics", "capacity_forecasts"]
}

Response:
{
  "datasetId": "dataset-456",
  "status": "published",
  "refreshSchedule": ["00:00", "06:00", "12:00", "18:00"]
}
```

#### Export to Looker
```http
POST /api/v1/export/looker
Content-Type: application/json

{
  "modelName": "dwcp_analytics",
  "explores": ["costs", "capacity", "performance"]
}

Response:
{
  "modelId": "model-789",
  "status": "deployed",
  "exploreUrls": {
    "costs": "https://looker.company.com/explore/dwcp_analytics/costs",
    "capacity": "https://looker.company.com/explore/dwcp_analytics/capacity"
  }
}
```

## Rate Limits

| Plan | Queries/min | Mutations/min | Subscriptions |
|------|-------------|---------------|---------------|
| Free | 60 | 10 | 5 |
| Pro | 600 | 100 | 50 |
| Enterprise | 6000 | 1000 | 500 |

**Headers:**
```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 543
X-RateLimit-Reset: 1704067200
```

## Error Handling

### Error Format
```json
{
  "errors": [
    {
      "message": "Cost data not found",
      "locations": [{"line": 2, "column": 3}],
      "path": ["getCostMetrics"],
      "extensions": {
        "code": "NOT_FOUND",
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| UNAUTHENTICATED | Missing or invalid authentication | 401 |
| FORBIDDEN | Insufficient permissions | 403 |
| NOT_FOUND | Resource not found | 404 |
| BAD_REQUEST | Invalid input | 400 |
| INTERNAL_ERROR | Server error | 500 |
| RATE_LIMIT_EXCEEDED | Too many requests | 429 |

## SDK Examples

### Python SDK
```python
from dwcp_analytics import DWCPAnalyticsClient

client = DWCPAnalyticsClient(api_key='your-api-key')

# Get cost metrics
metrics = client.query.get_cost_metrics(
    start_date='2024-01-01',
    end_date='2024-01-31',
    providers=['AWS', 'AZURE'],
    group_by=['SERVICE']
)

# Get forecasts
forecasts = client.query.get_cost_forecast(
    provider='AWS',
    periods=['30d', '90d']
)

# Stream metrics
for metric in client.subscribe.stream_metrics(['cpu_usage']):
    print(f"{metric.name}: {metric.value}")
```

### JavaScript SDK
```javascript
import { DWCPAnalyticsClient } from '@dwcp/analytics';

const client = new DWCPAnalyticsClient({ apiKey: 'your-api-key' });

// Get cost metrics
const metrics = await client.query.getCostMetrics({
  startDate: '2024-01-01',
  endDate: '2024-01-31',
  providers: ['AWS', 'AZURE'],
  groupBy: ['SERVICE']
});

// Subscribe to metrics
client.subscribe.streamMetrics(['cpu_usage'], (metric) => {
  console.log(`${metric.name}: ${metric.value}`);
});
```

### Go SDK
```go
import "github.com/dwcp/analytics-go"

client := analytics.NewClient("your-api-key")

// Get cost metrics
metrics, err := client.Query.GetCostMetrics(&analytics.CostMetricsParams{
    StartDate: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
    EndDate:   time.Date(2024, 1, 31, 23, 59, 59, 0, time.UTC),
    Providers: []string{"AWS", "AZURE"},
    GroupBy:   []string{"SERVICE"},
})

// Stream metrics
ch, err := client.Subscribe.StreamMetrics([]string{"cpu_usage"})
for metric := range ch {
    fmt.Printf("%s: %f\n", metric.Name, metric.Value)
}
```

## Support

- **API Documentation**: https://api.dwcp.io/docs
- **GraphQL Playground**: https://api.dwcp.io/graphql/playground
- **Status Page**: https://status.dwcp.io
- **Support**: support@dwcp.io