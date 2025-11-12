// Package api provides GraphQL API for BI analytics and data warehouse integration
// Enables custom queries and integration with Tableau, PowerBI, and Looker
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/99designs/gqlgen/graphql"
	"github.com/99designs/gqlgen/graphql/handler"
	"github.com/99designs/gqlgen/graphql/playground"
	"github.com/vektah/gqlparser/v2/gqlerror"
	"go.uber.org/zap"
)

// GraphQL Schema Definition
const schema = `
type Query {
    # Cost Intelligence Queries
    getCostMetrics(
        startDate: Time!
        endDate: Time!
        providers: [CloudProvider!]
        services: [String!]
        groupBy: [GroupByField!]
        aggregation: AggregationType
    ): CostMetricsResponse!

    getCostForecast(
        provider: CloudProvider!
        service: String
        periods: [ForecastPeriod!]!
    ): [CostForecast!]!

    getCostAnomalies(
        provider: CloudProvider
        threshold: Float
        severity: Severity
    ): [CostAnomaly!]!

    getOptimizationRecommendations(
        provider: CloudProvider
        minSavings: Float
        maxRisk: RiskLevel
    ): [OptimizationRecommendation!]!

    # Capacity Planning Queries
    getCapacityForecast(
        resourceType: ResourceType!
        periods: [ForecastPeriod!]!
    ): [CapacityForecast!]!

    getResourceUtilization(
        resourceType: ResourceType
        startDate: Time!
        endDate: Time!
        aggregation: AggregationType
    ): ResourceUtilizationResponse!

    getGrowthTrends(
        resourceType: ResourceType!
        lookbackDays: Int!
    ): GrowthTrendsResponse!

    # Real-time Analytics Queries
    getRealtimeMetrics(
        metricNames: [String!]!
        windowSize: Duration
    ): [RealtimeMetric!]!

    getDashboard(
        dashboardId: ID!
        timeRange: TimeRange
    ): Dashboard!

    # Executive Analytics
    getExecutiveSummary(
        period: Period!
    ): ExecutiveSummary!

    getKPIMetrics: [KPIMetric!]!

    getSLACompliance(
        services: [String!]
    ): SLAComplianceReport!

    # Advanced Analytics
    runCustomQuery(
        query: String!
        parameters: JSON
    ): CustomQueryResult!

    getAlerts(
        severity: Severity
        status: AlertStatus
    ): [Alert!]!
}

type Mutation {
    # Cost Management
    implementOptimization(
        recommendationId: ID!
        schedule: Time
    ): OptimizationResult!

    # Capacity Planning
    createScenario(
        input: ScenarioInput!
    ): Scenario!

    runWhatIfAnalysis(
        scenarioIds: [ID!]!
    ): WhatIfAnalysisResult!

    # Dashboard Management
    createDashboard(
        input: DashboardInput!
    ): Dashboard!

    updateDashboard(
        id: ID!
        input: DashboardUpdateInput!
    ): Dashboard!

    # Alert Configuration
    createAlert(
        input: AlertInput!
    ): Alert!

    acknowledgeAlert(
        alertId: ID!
    ): Alert!
}

type Subscription {
    # Real-time metrics streaming
    streamMetrics(
        metricNames: [String!]!
    ): MetricUpdate!

    # Cost alerts
    costAlerts(
        provider: CloudProvider
    ): CostAlert!

    # Capacity alerts
    capacityAlerts(
        resourceType: ResourceType
    ): CapacityAlert!
}

# Cost Types
type CostMetricsResponse {
    metrics: [CostMetric!]!
    summary: CostSummary!
    groupedData: JSON
}

type CostMetric {
    timestamp: Time!
    provider: CloudProvider!
    service: String!
    resourceId: String!
    cost: Float!
    usage: Float!
    unit: String!
    tags: JSON
}

type CostForecast {
    provider: CloudProvider!
    service: String!
    period: ForecastPeriod!
    currentCost: Float!
    predictedCost: Float!
    confidenceLower: Float!
    confidenceUpper: Float!
    trend: TrendType!
    accuracy: Float!
    recommendations: [String!]!
}

type CostAnomaly {
    id: ID!
    timestamp: Time!
    provider: CloudProvider!
    service: String!
    expectedCost: Float!
    actualCost: Float!
    deviationPercentage: Float!
    severity: Severity!
    probableCauses: [String!]!
    recommendedActions: [String!]!
}

type OptimizationRecommendation {
    id: ID!
    provider: CloudProvider!
    service: String!
    resourceId: String!
    currentCost: Float!
    optimizedCost: Float!
    annualSavings: Float!
    roiPercentage: Float!
    effort: EffortLevel!
    risk: RiskLevel!
    actions: [OptimizationAction!]!
    impact: JSON
}

# Capacity Types
type CapacityForecast {
    resourceType: ResourceType!
    period: ForecastPeriod!
    currentCapacity: Float!
    forecastedDemand: Float!
    recommendedCapacity: Float!
    exhaustionDate: Time
    confidence: Float!
    scalingRecommendations: [ScalingRecommendation!]!
}

type ResourceUtilizationResponse {
    utilization: [ResourceUtilization!]!
    statistics: UtilizationStats!
}

# Real-time Types
type RealtimeMetric {
    name: String!
    value: Float!
    timestamp: Time!
    tags: JSON
    trend: TrendIndicator
}

type Dashboard {
    id: ID!
    title: String!
    description: String
    panels: [Panel!]!
    timeRange: TimeRange!
    refreshInterval: Duration
    variables: [Variable!]
    annotations: [Annotation!]
}

# Executive Types
type ExecutiveSummary {
    period: Period!
    totalCost: Float!
    costTrend: TrendType!
    savingsRealized: Float!
    systemAvailability: Float!
    slaCompliance: Float!
    activeResources: Int!
    incidents: Int!
    keyMetrics: [KeyMetric!]!
    recommendations: [String!]!
}

type KPIMetric {
    name: String!
    value: Float!
    target: Float!
    status: KPIStatus!
    trend: TrendType!
    sparkline: [Float!]
}

# Enums
enum CloudProvider {
    AWS
    AZURE
    GCP
    ORACLE
    IBM
    ALIBABA
    ON_PREMISE
}

enum ResourceType {
    CPU
    MEMORY
    STORAGE
    NETWORK
    GPU
}

enum ForecastPeriod {
    DAY_7
    DAY_30
    DAY_90
    DAY_180
    YEAR_1
}

enum AggregationType {
    SUM
    AVG
    MIN
    MAX
    P50
    P95
    P99
}

enum Severity {
    LOW
    MEDIUM
    HIGH
    CRITICAL
}

enum TrendType {
    INCREASING
    DECREASING
    STABLE
    VOLATILE
}

scalar Time
scalar Duration
scalar JSON
`

// BIGraphQLAPI provides GraphQL API for BI analytics
type BIGraphQLAPI struct {
	analyticsEngine *AnalyticsEngine
	costPlatform    *CostIntelligencePlatform
	capacityAI      *CapacityPlanningAI
	dataWarehouse   *DataWarehouseConnector
	logger          *zap.Logger
}

// NewBIGraphQLAPI creates a new BI GraphQL API instance
func NewBIGraphQLAPI(config *Config, logger *zap.Logger) *BIGraphQLAPI {
	return &BIGraphQLAPI{
		analyticsEngine: NewAnalyticsEngine(config.Analytics),
		costPlatform:    NewCostIntelligencePlatform(config.Cost),
		capacityAI:      NewCapacityPlanningAI(config.Capacity),
		dataWarehouse:   NewDataWarehouseConnector(config.Warehouse),
		logger:          logger,
	}
}

// Query resolver
type QueryResolver struct {
	api *BIGraphQLAPI
}

// GetCostMetrics resolves cost metrics query
func (r *QueryResolver) GetCostMetrics(
	ctx context.Context,
	startDate, endDate time.Time,
	providers []string,
	services []string,
	groupBy []string,
	aggregation *string,
) (*CostMetricsResponse, error) {
	// Build query parameters
	params := &CostQueryParams{
		StartDate:   startDate,
		EndDate:     endDate,
		Providers:   providers,
		Services:    services,
		GroupBy:     groupBy,
		Aggregation: aggregation,
	}

	// Fetch metrics from cost platform
	metrics, err := r.api.costPlatform.QueryMetrics(ctx, params)
	if err != nil {
		return nil, gqlerror.Errorf("Failed to fetch cost metrics: %v", err)
	}

	// Calculate summary
	summary := r.api.calculateCostSummary(metrics)

	// Group data if requested
	var groupedData interface{}
	if len(groupBy) > 0 {
		groupedData = r.api.groupCostData(metrics, groupBy)
	}

	return &CostMetricsResponse{
		Metrics:     metrics,
		Summary:     summary,
		GroupedData: groupedData,
	}, nil
}

// GetCostForecast resolves cost forecast query
func (r *QueryResolver) GetCostForecast(
	ctx context.Context,
	provider string,
	service *string,
	periods []string,
) ([]*CostForecast, error) {
	forecasts, err := r.api.costPlatform.GenerateForecasts(
		ctx,
		provider,
		service,
		periods,
	)
	if err != nil {
		return nil, gqlerror.Errorf("Failed to generate cost forecast: %v", err)
	}

	return forecasts, nil
}

// GetCapacityForecast resolves capacity forecast query
func (r *QueryResolver) GetCapacityForecast(
	ctx context.Context,
	resourceType string,
	periods []string,
) ([]*CapacityForecast, error) {
	// Get historical data
	historicalData, err := r.api.analyticsEngine.GetHistoricalData(
		ctx,
		resourceType,
		time.Now().AddDate(0, -6, 0), // 6 months back
		time.Now(),
	)
	if err != nil {
		return nil, err
	}

	// Generate forecasts
	forecasts, err := r.api.capacityAI.ForecastCapacity(
		ctx,
		resourceType,
		historicalData,
		periods,
	)
	if err != nil {
		return nil, gqlerror.Errorf("Failed to generate capacity forecast: %v", err)
	}

	return forecasts, nil
}

// GetExecutiveSummary resolves executive summary query
func (r *QueryResolver) GetExecutiveSummary(
	ctx context.Context,
	period string,
) (*ExecutiveSummary, error) {
	// Calculate time range based on period
	endDate := time.Now()
	var startDate time.Time

	switch period {
	case "DAILY":
		startDate = endDate.AddDate(0, 0, -1)
	case "WEEKLY":
		startDate = endDate.AddDate(0, 0, -7)
	case "MONTHLY":
		startDate = endDate.AddDate(0, -1, 0)
	case "QUARTERLY":
		startDate = endDate.AddDate(0, -3, 0)
	case "YEARLY":
		startDate = endDate.AddDate(-1, 0, 0)
	default:
		return nil, gqlerror.Errorf("Invalid period: %s", period)
	}

	// Fetch all metrics in parallel
	type result struct {
		costs      *CostSummary
		savings    float64
		sla        *SLAMetrics
		resources  int
		incidents  int
		err        error
	}

	resultChan := make(chan result, 1)

	go func() {
		var res result

		// Cost metrics
		costs, err := r.api.costPlatform.GetCostSummary(ctx, startDate, endDate)
		if err != nil {
			res.err = err
			resultChan <- res
			return
		}
		res.costs = costs

		// Savings realized
		res.savings, _ = r.api.costPlatform.GetSavingsRealized(ctx, startDate, endDate)

		// SLA compliance
		res.sla, _ = r.api.analyticsEngine.GetSLACompliance(ctx, startDate, endDate)

		// Active resources
		res.resources, _ = r.api.analyticsEngine.GetActiveResourceCount(ctx)

		// Incident count
		res.incidents, _ = r.api.analyticsEngine.GetIncidentCount(ctx, startDate, endDate)

		resultChan <- res
	}()

	// Wait for results with timeout
	select {
	case res := <-resultChan:
		if res.err != nil {
			return nil, res.err
		}

		return &ExecutiveSummary{
			Period:             period,
			TotalCost:          res.costs.Total,
			CostTrend:          res.costs.Trend,
			SavingsRealized:    res.savings,
			SystemAvailability: res.sla.Availability,
			SLACompliance:      res.sla.Compliance,
			ActiveResources:    res.resources,
			Incidents:          res.incidents,
			KeyMetrics:         r.api.getKeyMetrics(ctx),
			Recommendations:    r.api.getTopRecommendations(ctx, 5),
		}, nil

	case <-time.After(30 * time.Second):
		return nil, gqlerror.Errorf("Query timeout")
	}
}

// Mutation resolver
type MutationResolver struct {
	api *BIGraphQLAPI
}

// ImplementOptimization implements a cost optimization recommendation
func (r *MutationResolver) ImplementOptimization(
	ctx context.Context,
	recommendationID string,
	schedule *time.Time,
) (*OptimizationResult, error) {
	// Validate recommendation exists
	recommendation, err := r.api.costPlatform.GetRecommendation(ctx, recommendationID)
	if err != nil {
		return nil, gqlerror.Errorf("Recommendation not found: %v", err)
	}

	// Schedule or execute immediately
	if schedule != nil {
		err = r.api.costPlatform.ScheduleOptimization(ctx, recommendation, *schedule)
	} else {
		err = r.api.costPlatform.ExecuteOptimization(ctx, recommendation)
	}

	if err != nil {
		return nil, gqlerror.Errorf("Failed to implement optimization: %v", err)
	}

	return &OptimizationResult{
		Success:      true,
		Message:      "Optimization implemented successfully",
		EstimatedSavings: recommendation.AnnualSavings,
		ScheduledFor: schedule,
	}, nil
}

// DataWarehouseConnector handles integration with data warehouses
type DataWarehouseConnector struct {
	snowflakeConn *SnowflakeConnection
	bigqueryConn  *BigQueryConnection
	redshiftConn  *RedshiftConnection
	logger        *zap.Logger
}

// ExportToTableau exports data for Tableau integration
func (c *DataWarehouseConnector) ExportToTableau(
	ctx context.Context,
	query string,
	format string,
) (string, error) {
	// Execute query
	data, err := c.executeQuery(ctx, query)
	if err != nil {
		return "", err
	}

	// Convert to Tableau format
	switch format {
	case "tde":
		return c.exportToTDE(data)
	case "hyper":
		return c.exportToHyper(data)
	default:
		return c.exportToCSV(data)
	}
}

// ExportToPowerBI exports data for PowerBI integration
func (c *DataWarehouseConnector) ExportToPowerBI(
	ctx context.Context,
	dataset string,
	tables []string,
) (*PowerBIDataset, error) {
	pbDataset := &PowerBIDataset{
		Name:   dataset,
		Tables: make([]*PowerBITable, 0, len(tables)),
	}

	for _, table := range tables {
		// Fetch table data
		data, err := c.fetchTableData(ctx, table)
		if err != nil {
			c.logger.Error("Failed to fetch table data", zap.String("table", table), zap.Error(err))
			continue
		}

		// Convert to PowerBI table format
		pbTable := c.convertToPowerBITable(table, data)
		pbDataset.Tables = append(pbDataset.Tables, pbTable)
	}

	// Push to PowerBI service
	if err := c.pushToPowerBI(ctx, pbDataset); err != nil {
		return nil, fmt.Errorf("failed to push to PowerBI: %w", err)
	}

	return pbDataset, nil
}

// ExportToLooker exports data for Looker integration
func (c *DataWarehouseConnector) ExportToLooker(
	ctx context.Context,
	modelName string,
	explores []string,
) (*LookerModel, error) {
	model := &LookerModel{
		Name:     modelName,
		Explores: make([]*LookerExplore, 0, len(explores)),
	}

	for _, explore := range explores {
		// Generate LookML for explore
		lookml, err := c.generateLookML(ctx, explore)
		if err != nil {
			c.logger.Error("Failed to generate LookML", zap.String("explore", explore), zap.Error(err))
			continue
		}

		model.Explores = append(model.Explores, &LookerExplore{
			Name:   explore,
			LookML: lookml,
		})
	}

	// Deploy to Looker instance
	if err := c.deployToLooker(ctx, model); err != nil {
		return nil, fmt.Errorf("failed to deploy to Looker: %w", err)
	}

	return model, nil
}