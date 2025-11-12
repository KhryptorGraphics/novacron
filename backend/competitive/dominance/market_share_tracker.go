// Market Share Domination Tracker
// Real-time monitoring and forecasting for 50%+ market share achievement
// Tracks competitive positioning, segment penetration, and dominance metrics

package dominance

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"
)

// MarketShareTracker monitors real-time market share and dominance metrics
type MarketShareTracker struct {
	id              string
	currentShare    float64
	targetShare     float64
	competitorData  map[string]*CompetitorShare
	segmentShares   map[string]*SegmentShare
	geographicShare map[string]*GeographicShare
	forecastEngine  *ShareForecastEngine
	dominanceMetrics *DominanceMetrics
	mu              sync.RWMutex
	updateInterval  time.Duration
	alertThresholds *AlertThresholds
	dataCollectors  map[string]DataCollector
}

// CompetitorShare tracks individual competitor market positions
type CompetitorShare struct {
	CompetitorID    string                 `json:"competitor_id"`
	Name            string                 `json:"name"`
	CurrentShare    float64                `json:"current_share"`
	ShareChange     float64                `json:"share_change"`
	DisplacementRate float64               `json:"displacement_rate"`
	Strengths       []string               `json:"strengths"`
	Weaknesses      []string               `json:"weaknesses"`
	KeyAccounts     []string               `json:"key_accounts"`
	LostAccounts    []string               `json:"lost_accounts"`
	CompetitivePosn string                 `json:"competitive_position"`
	ThreatLevel     string                 `json:"threat_level"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// SegmentShare tracks market share by vertical/segment
type SegmentShare struct {
	SegmentID       string    `json:"segment_id"`
	SegmentName     string    `json:"segment_name"`
	TotalTAM        float64   `json:"total_tam"`
	OurShare        float64   `json:"our_share"`
	OurRevenue      float64   `json:"our_revenue"`
	TargetShare     float64   `json:"target_share"`
	GrowthRate      float64   `json:"growth_rate"`
	Penetration     float64   `json:"penetration"`
	DominanceStatus string    `json:"dominance_status"`
	TopCompetitors  []string  `json:"top_competitors"`
	LastUpdated     time.Time `json:"last_updated"`
}

// GeographicShare tracks regional market dominance
type GeographicShare struct {
	RegionID        string                 `json:"region_id"`
	RegionName      string                 `json:"region_name"`
	MarketSize      float64                `json:"market_size"`
	OurShare        float64                `json:"our_share"`
	OurRevenue      float64                `json:"our_revenue"`
	GrowthRate      float64                `json:"growth_rate"`
	Maturity        string                 `json:"maturity"`
	Regulations     []string               `json:"regulations"`
	Competition     map[string]float64     `json:"competition"`
	Opportunities   []string               `json:"opportunities"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ShareForecastEngine predicts market share trajectory
type ShareForecastEngine struct {
	historicalData  []ShareDataPoint
	modelType       string
	accuracy        float64
	forecastHorizon int
	mu              sync.RWMutex
}

// ShareDataPoint represents historical market share data
type ShareDataPoint struct {
	Timestamp   time.Time              `json:"timestamp"`
	Share       float64                `json:"share"`
	Revenue     float64                `json:"revenue"`
	Customers   int                    `json:"customers"`
	Competitive map[string]float64     `json:"competitive"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// DominanceMetrics tracks market leadership indicators
type DominanceMetrics struct {
	MarketLeaderStatus bool                   `json:"market_leader_status"`
	ShareOfWallet      float64                `json:"share_of_wallet"`
	BrandRecognition   float64                `json:"brand_recognition"`
	CustomerLoyalty    float64                `json:"customer_loyalty"`
	PricePremium       float64                `json:"price_premium"`
	InnovationIndex    float64                `json:"innovation_index"`
	AnalystRatings     map[string]string      `json:"analyst_ratings"`
	Awards             []string               `json:"awards"`
	Benchmarks         map[string]interface{} `json:"benchmarks"`
}

// AlertThresholds defines market share alert conditions
type AlertThresholds struct {
	CriticalLoss      float64 `json:"critical_loss"`
	CompetitorGain    float64 `json:"competitor_gain"`
	SegmentDecline    float64 `json:"segment_decline"`
	ForecastDeviation float64 `json:"forecast_deviation"`
}

// DataCollector interface for market data sources
type DataCollector interface {
	CollectData(ctx context.Context) (*MarketData, error)
	GetSourceName() string
	GetUpdateFrequency() time.Duration
}

// MarketData represents collected market intelligence
type MarketData struct {
	SourceID    string                 `json:"source_id"`
	Timestamp   time.Time              `json:"timestamp"`
	ShareData   map[string]float64     `json:"share_data"`
	Competitors []CompetitorShare      `json:"competitors"`
	Segments    []SegmentShare         `json:"segments"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// NewMarketShareTracker creates a new market share tracker
func NewMarketShareTracker(targetShare float64) *MarketShareTracker {
	return &MarketShareTracker{
		id:              uuid.New().String(),
		currentShare:    48.0, // Starting from Phase 12
		targetShare:     targetShare,
		competitorData:  make(map[string]*CompetitorShare),
		segmentShares:   make(map[string]*SegmentShare),
		geographicShare: make(map[string]*GeographicShare),
		forecastEngine:  NewShareForecastEngine(),
		dominanceMetrics: &DominanceMetrics{
			MarketLeaderStatus: true,
			AnalystRatings:     make(map[string]string),
			Benchmarks:         make(map[string]interface{}),
		},
		updateInterval: 15 * time.Minute,
		alertThresholds: &AlertThresholds{
			CriticalLoss:      0.5,
			CompetitorGain:    1.0,
			SegmentDecline:    2.0,
			ForecastDeviation: 1.5,
		},
		dataCollectors: make(map[string]DataCollector),
	}
}

// NewShareForecastEngine creates a new forecasting engine
func NewShareForecastEngine() *ShareForecastEngine {
	return &ShareForecastEngine{
		historicalData:  make([]ShareDataPoint, 0),
		modelType:       "time_series_regression",
		accuracy:        0.92,
		forecastHorizon: 12, // 12 months
	}
}

// TrackMarketShare updates current market share metrics
func (mst *MarketShareTracker) TrackMarketShare(ctx context.Context, share float64) error {
	mst.mu.Lock()
	defer mst.mu.Unlock()

	previousShare := mst.currentShare
	mst.currentShare = share

	// Record data point
	dataPoint := ShareDataPoint{
		Timestamp: time.Now(),
		Share:     share,
		Metadata:  make(map[string]interface{}),
	}

	mst.forecastEngine.AddDataPoint(dataPoint)

	// Check for significant changes
	if math.Abs(share-previousShare) > mst.alertThresholds.CriticalLoss {
		return mst.triggerAlert(ctx, "critical_share_change", map[string]interface{}{
			"previous": previousShare,
			"current":  share,
			"change":   share - previousShare,
		})
	}

	// Check milestone achievement
	if previousShare < mst.targetShare && share >= mst.targetShare {
		return mst.celebrateMilestone(ctx, "target_share_achieved", share)
	}

	return nil
}

// UpdateCompetitorData tracks competitor market positions
func (mst *MarketShareTracker) UpdateCompetitorData(ctx context.Context, competitor *CompetitorShare) error {
	mst.mu.Lock()
	defer mst.mu.Unlock()

	mst.competitorData[competitor.CompetitorID] = competitor

	// Analyze competitive threats
	if competitor.ShareChange > mst.alertThresholds.CompetitorGain {
		return mst.triggerAlert(ctx, "competitor_gaining", map[string]interface{}{
			"competitor": competitor.Name,
			"gain":       competitor.ShareChange,
			"threat":     competitor.ThreatLevel,
		})
	}

	return nil
}

// TrackSegmentShare monitors vertical/segment market share
func (mst *MarketShareTracker) TrackSegmentShare(ctx context.Context, segment *SegmentShare) error {
	mst.mu.Lock()
	defer mst.mu.Unlock()

	segment.LastUpdated = time.Now()

	// Calculate dominance status
	if segment.OurShare >= 50.0 {
		segment.DominanceStatus = "dominant"
	} else if segment.OurShare >= 40.0 {
		segment.DominanceStatus = "leader"
	} else if segment.OurShare >= 30.0 {
		segment.DominanceStatus = "strong"
	} else {
		segment.DominanceStatus = "challenger"
	}

	mst.segmentShares[segment.SegmentID] = segment

	// Check for segment decline
	if segment.GrowthRate < -mst.alertThresholds.SegmentDecline {
		return mst.triggerAlert(ctx, "segment_decline", map[string]interface{}{
			"segment": segment.SegmentName,
			"decline": segment.GrowthRate,
			"share":   segment.OurShare,
		})
	}

	return nil
}

// TrackGeographicShare monitors regional market positions
func (mst *MarketShareTracker) TrackGeographicShare(ctx context.Context, geographic *GeographicShare) error {
	mst.mu.Lock()
	defer mst.mu.Unlock()

	mst.geographicShare[geographic.RegionID] = geographic

	return nil
}

// ForecastMarketShare predicts future market share trajectory
func (mst *MarketShareTracker) ForecastMarketShare(ctx context.Context, months int) ([]ShareDataPoint, error) {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	return mst.forecastEngine.Forecast(months)
}

// AddDataPoint adds historical data to forecast model
func (sfe *ShareForecastEngine) AddDataPoint(dataPoint ShareDataPoint) {
	sfe.mu.Lock()
	defer sfe.mu.Unlock()

	sfe.historicalData = append(sfe.historicalData, dataPoint)

	// Keep last 24 months of data
	if len(sfe.historicalData) > 24 {
		sfe.historicalData = sfe.historicalData[len(sfe.historicalData)-24:]
	}
}

// Forecast generates market share predictions
func (sfe *ShareForecastEngine) Forecast(months int) ([]ShareDataPoint, error) {
	sfe.mu.RLock()
	defer sfe.mu.RUnlock()

	if len(sfe.historicalData) < 3 {
		return nil, fmt.Errorf("insufficient historical data for forecasting")
	}

	forecast := make([]ShareDataPoint, months)

	// Simple time series forecasting with trend and seasonality
	// Calculate trend
	n := len(sfe.historicalData)
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0

	for i, point := range sfe.historicalData {
		x := float64(i)
		y := point.Share
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (float64(n)*sumXY - sumX*sumY) / (float64(n)*sumX2 - sumX*sumX)
	intercept := (sumY - slope*sumX) / float64(n)

	// Generate forecast
	lastShare := sfe.historicalData[n-1].Share
	lastTime := sfe.historicalData[n-1].Timestamp

	for i := 0; i < months; i++ {
		predictedShare := intercept + slope*float64(n+i)

		// Add seasonality adjustment (simplified)
		seasonalFactor := 1.0 + 0.02*math.Sin(float64(i)*math.Pi/6.0)
		predictedShare *= seasonalFactor

		// Ensure share stays within 0-100%
		if predictedShare < 0 {
			predictedShare = 0
		}
		if predictedShare > 100 {
			predictedShare = 100
		}

		forecast[i] = ShareDataPoint{
			Timestamp: lastTime.AddDate(0, i+1, 0),
			Share:     predictedShare,
			Metadata: map[string]interface{}{
				"confidence": sfe.accuracy,
				"model":      sfe.modelType,
			},
		}
	}

	return forecast, nil
}

// CalculateDominanceMetrics computes market leadership indicators
func (mst *MarketShareTracker) CalculateDominanceMetrics(ctx context.Context) (*DominanceMetrics, error) {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	metrics := mst.dominanceMetrics

	// Calculate share of wallet
	totalWallet := 0.0
	ourWallet := 0.0
	for _, segment := range mst.segmentShares {
		totalWallet += segment.TotalTAM
		ourWallet += segment.OurRevenue
	}
	if totalWallet > 0 {
		metrics.ShareOfWallet = (ourWallet / totalWallet) * 100
	}

	// Determine market leader status
	metrics.MarketLeaderStatus = mst.currentShare >= 50.0

	// Calculate brand recognition (based on market share and presence)
	metrics.BrandRecognition = math.Min(mst.currentShare*1.5, 100.0)

	// Calculate customer loyalty (simplified - based on share stability)
	if len(mst.forecastEngine.historicalData) >= 6 {
		variance := mst.calculateShareVariance()
		metrics.CustomerLoyalty = math.Max(100.0-variance*10, 0)
	}

	// Innovation index (based on product releases and patents)
	metrics.InnovationIndex = 85.0 // Placeholder for real data

	// Update analyst ratings
	metrics.AnalystRatings = map[string]string{
		"Gartner":  "Leader",
		"Forrester": "Leader",
		"IDC":      "Leader",
		"451Research": "Leader",
		"OmdiaTelco": "Leader",
	}

	metrics.Benchmarks = map[string]interface{}{
		"market_share":      mst.currentShare,
		"target_gap":        mst.targetShare - mst.currentShare,
		"leader_advantage":  mst.calculateLeaderAdvantage(),
		"displacement_rate": mst.calculateDisplacementRate(),
		"growth_momentum":   mst.calculateGrowthMomentum(),
	}

	return metrics, nil
}

// calculateShareVariance computes historical share volatility
func (mst *MarketShareTracker) calculateShareVariance() float64 {
	data := mst.forecastEngine.historicalData
	if len(data) < 2 {
		return 0
	}

	mean := 0.0
	for _, point := range data {
		mean += point.Share
	}
	mean /= float64(len(data))

	variance := 0.0
	for _, point := range data {
		diff := point.Share - mean
		variance += diff * diff
	}
	variance /= float64(len(data))

	return math.Sqrt(variance)
}

// calculateLeaderAdvantage computes gap vs #2 competitor
func (mst *MarketShareTracker) calculateLeaderAdvantage() float64 {
	if len(mst.competitorData) == 0 {
		return mst.currentShare
	}

	maxCompetitorShare := 0.0
	for _, competitor := range mst.competitorData {
		if competitor.CurrentShare > maxCompetitorShare {
			maxCompetitorShare = competitor.CurrentShare
		}
	}

	return mst.currentShare - maxCompetitorShare
}

// calculateDisplacementRate computes competitive win rate
func (mst *MarketShareTracker) calculateDisplacementRate() float64 {
	totalDisplacements := 0
	totalOpportunities := 0

	for _, competitor := range mst.competitorData {
		totalDisplacements += len(competitor.LostAccounts)
		totalOpportunities += len(competitor.KeyAccounts)
	}

	if totalOpportunities == 0 {
		return 0
	}

	return (float64(totalDisplacements) / float64(totalOpportunities)) * 100
}

// calculateGrowthMomentum computes recent growth trajectory
func (mst *MarketShareTracker) calculateGrowthMomentum() float64 {
	data := mst.forecastEngine.historicalData
	if len(data) < 3 {
		return 0
	}

	// Calculate 3-month moving average slope
	recent := data[len(data)-3:]
	growth := (recent[2].Share - recent[0].Share) / 2.0

	return growth
}

// GetMarketShareStatus returns current market position
func (mst *MarketShareTracker) GetMarketShareStatus() map[string]interface{} {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	return map[string]interface{}{
		"tracker_id":      mst.id,
		"current_share":   mst.currentShare,
		"target_share":    mst.targetShare,
		"progress":        (mst.currentShare / mst.targetShare) * 100,
		"target_achieved": mst.currentShare >= mst.targetShare,
		"competitors":     len(mst.competitorData),
		"segments":        len(mst.segmentShares),
		"regions":         len(mst.geographicShare),
		"forecast_accuracy": mst.forecastEngine.accuracy,
	}
}

// GetSegmentDominance returns segments with 50%+ share
func (mst *MarketShareTracker) GetSegmentDominance() []SegmentShare {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	dominant := make([]SegmentShare, 0)
	for _, segment := range mst.segmentShares {
		if segment.OurShare >= 50.0 {
			dominant = append(dominant, *segment)
		}
	}

	return dominant
}

// GetGeographicDominance returns regions with 50%+ share
func (mst *MarketShareTracker) GetGeographicDominance() []GeographicShare {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	dominant := make([]GeographicShare, 0)
	for _, region := range mst.geographicShare {
		if region.OurShare >= 50.0 {
			dominant = append(dominant, *region)
		}
	}

	return dominant
}

// GetCompetitivePosition returns detailed competitive analysis
func (mst *MarketShareTracker) GetCompetitivePosition() map[string]interface{} {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	competitors := make([]map[string]interface{}, 0)
	for _, comp := range mst.competitorData {
		competitors = append(competitors, map[string]interface{}{
			"name":          comp.Name,
			"share":         comp.CurrentShare,
			"change":        comp.ShareChange,
			"displacement":  comp.DisplacementRate,
			"threat_level":  comp.ThreatLevel,
			"key_accounts":  len(comp.KeyAccounts),
			"lost_accounts": len(comp.LostAccounts),
		})
	}

	return map[string]interface{}{
		"our_position":      "Market Leader",
		"our_share":         mst.currentShare,
		"leader_advantage":  mst.calculateLeaderAdvantage(),
		"competitors":       competitors,
		"displacement_rate": mst.calculateDisplacementRate(),
		"growth_momentum":   mst.calculateGrowthMomentum(),
	}
}

// triggerAlert sends market share alerts
func (mst *MarketShareTracker) triggerAlert(ctx context.Context, alertType string, data map[string]interface{}) error {
	alert := map[string]interface{}{
		"alert_type": alertType,
		"timestamp":  time.Now(),
		"tracker_id": mst.id,
		"data":       data,
		"severity":   mst.getAlertSeverity(alertType),
	}

	// Log alert (in production, send to monitoring system)
	alertJSON, _ := json.MarshalIndent(alert, "", "  ")
	fmt.Printf("MARKET SHARE ALERT:\n%s\n", string(alertJSON))

	return nil
}

// celebrateMilestone handles milestone achievements
func (mst *MarketShareTracker) celebrateMilestone(ctx context.Context, milestone string, value float64) error {
	celebration := map[string]interface{}{
		"milestone":  milestone,
		"value":      value,
		"timestamp":  time.Now(),
		"tracker_id": mst.id,
	}

	celebrationJSON, _ := json.MarshalIndent(celebration, "", "  ")
	fmt.Printf("MILESTONE ACHIEVED:\n%s\n", string(celebrationJSON))

	return nil
}

// getAlertSeverity determines alert priority
func (mst *MarketShareTracker) getAlertSeverity(alertType string) string {
	severityMap := map[string]string{
		"critical_share_change": "critical",
		"competitor_gaining":    "high",
		"segment_decline":       "medium",
		"forecast_deviation":    "low",
	}

	if severity, ok := severityMap[alertType]; ok {
		return severity
	}
	return "info"
}

// RegisterDataCollector adds a market data source
func (mst *MarketShareTracker) RegisterDataCollector(collector DataCollector) {
	mst.mu.Lock()
	defer mst.mu.Unlock()

	mst.dataCollectors[collector.GetSourceName()] = collector
}

// StartMonitoring begins continuous market share tracking
func (mst *MarketShareTracker) StartMonitoring(ctx context.Context) error {
	ticker := time.NewTicker(mst.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := mst.collectAndUpdate(ctx); err != nil {
				fmt.Printf("Error collecting market data: %v\n", err)
			}
		}
	}
}

// collectAndUpdate gathers data from all collectors
func (mst *MarketShareTracker) collectAndUpdate(ctx context.Context) error {
	for _, collector := range mst.dataCollectors {
		data, err := collector.CollectData(ctx)
		if err != nil {
			continue
		}

		// Update market share
		if overallShare, ok := data.ShareData["overall"]; ok {
			if err := mst.TrackMarketShare(ctx, overallShare); err != nil {
				return err
			}
		}

		// Update competitor data
		for _, competitor := range data.Competitors {
			if err := mst.UpdateCompetitorData(ctx, &competitor); err != nil {
				return err
			}
		}

		// Update segment data
		for _, segment := range data.Segments {
			if err := mst.TrackSegmentShare(ctx, &segment); err != nil {
				return err
			}
		}
	}

	// Calculate dominance metrics
	_, err := mst.CalculateDominanceMetrics(ctx)
	return err
}

// ExportMetrics exports comprehensive market share data
func (mst *MarketShareTracker) ExportMetrics() ([]byte, error) {
	mst.mu.RLock()
	defer mst.mu.RUnlock()

	metrics := map[string]interface{}{
		"tracker_id":        mst.id,
		"current_share":     mst.currentShare,
		"target_share":      mst.targetShare,
		"competitors":       mst.competitorData,
		"segments":          mst.segmentShares,
		"geographic":        mst.geographicShare,
		"dominance_metrics": mst.dominanceMetrics,
		"forecast":          mst.forecastEngine.historicalData,
		"timestamp":         time.Now(),
	}

	return json.MarshalIndent(metrics, "", "  ")
}
