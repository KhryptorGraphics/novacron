// Package acquisition provides new logo acquisition functionality
package acquisition

import (
	"context"
	"sync"
	"time"
)

// NewLogoEngine handles new customer acquisition
type NewLogoEngine struct {
	mu           sync.RWMutex
	config       AcquisitionConfig
	metrics      *EngineMetrics
	opportunities map[string]*Opportunity
}

// AcquisitionConfig configures the acquisition engine
type AcquisitionConfig struct {
	TargetFortune500 int     `json:"target_fortune_500"`
	TargetNewARR     float64 `json:"target_new_arr"`
	MinDealSize      float64 `json:"min_deal_size"`
	TargetSalesCycle int     `json:"target_sales_cycle"`
	EnableAutomation bool    `json:"enable_automation"`
	EnableAIScoring  bool    `json:"enable_ai_scoring"`
}

// EngineMetrics tracks engine performance
type EngineMetrics struct {
	TotalOpportunities int64   `json:"total_opportunities"`
	TotalARR           float64 `json:"total_arr"`
	AvgDealSize        float64 `json:"avg_deal_size"`
	ConversionRate     float64 `json:"conversion_rate"`
}

// Opportunity represents a sales opportunity
type Opportunity struct {
	ID           string    `json:"id"`
	AccountName  string    `json:"account_name"`
	Value        float64   `json:"value"`
	Stage        string    `json:"stage"`
	Probability  float64   `json:"probability"`
	ExpectedClose time.Time `json:"expected_close"`
	CreatedAt    time.Time `json:"created_at"`
}

// NewNewLogoEngine creates a new acquisition engine
func NewNewLogoEngine(config AcquisitionConfig) *NewLogoEngine {
	return &NewLogoEngine{
		config:        config,
		metrics:       &EngineMetrics{},
		opportunities: make(map[string]*Opportunity),
	}
}

// ExportMetrics exports engine metrics
func (e *NewLogoEngine) ExportMetrics() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return map[string]interface{}{
		"total_opportunities": e.metrics.TotalOpportunities,
		"total_arr":           e.metrics.TotalARR,
		"avg_deal_size":       e.metrics.AvgDealSize,
		"conversion_rate":     e.metrics.ConversionRate,
	}
}

// AddOpportunity adds a new opportunity
func (e *NewLogoEngine) AddOpportunity(ctx context.Context, opp *Opportunity) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.opportunities[opp.ID] = opp
	e.metrics.TotalOpportunities++
	e.metrics.TotalARR += opp.Value

	return nil
}
