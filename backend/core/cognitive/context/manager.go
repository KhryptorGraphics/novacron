// Package context provides multi-dimensional context management
package context

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cognitive"
)

// ContextManager manages multi-dimensional context for AI reasoning
type ContextManager struct {
	userContexts     map[string]*UserContext
	systemContext    *SystemContext
	businessContext  *BusinessContext
	temporalContext  *TemporalContext
	geospatialContext *GeospatialContext
	lock             sync.RWMutex
	metricsLock      sync.RWMutex
	metrics          ContextMetrics
}

// UserContext contains user-specific context
type UserContext struct {
	UserID       string
	Role         string                 // admin, developer, operator
	Preferences  map[string]interface{} // UI preferences, notification settings
	History      []UserAction           // Recent actions
	Expertise    map[string]float64     // Domain expertise scores
	LastActivity time.Time
}

// UserAction represents a user action
type UserAction struct {
	Action    string
	Timestamp time.Time
	Metadata  map[string]interface{}
}

// SystemContext contains current system state
type SystemContext struct {
	TotalVMs         int
	TotalLoad        float64
	AvailableCapacity float64
	ActiveIncidents   int
	RecentAlerts     []Alert
	HealthScore      float64
	LastUpdated      time.Time
}

// Alert represents a system alert
type Alert struct {
	ID       string
	Severity string
	Message  string
	Time     time.Time
}

// BusinessContext contains business constraints
type BusinessContext struct {
	ActiveSLAs       []SLA
	MonthlyBudget    float64
	CurrentSpend     float64
	ComplianceReqs   []string // GDPR, HIPAA, etc.
	CostCenters      map[string]float64
	LastUpdated      time.Time
}

// SLA represents a service level agreement
type SLA struct {
	ID           string
	Service      string
	Target       float64 // e.g., 99.9% uptime
	Current      float64
	Breached     bool
}

// TemporalContext contains time-based patterns
type TemporalContext struct {
	CurrentTime     time.Time
	TimeOfDay       string // morning, afternoon, evening, night
	DayOfWeek       string
	IsBusinessHours bool
	LoadPatterns    map[string]LoadPattern
}

// LoadPattern represents historical load patterns
type LoadPattern struct {
	Hour        int
	AvgLoad     float64
	PeakLoad    float64
	Confidence  float64
}

// GeospatialContext contains location-based context
type GeospatialContext struct {
	ActiveRegions    []string
	RegionLatency    map[string]float64 // Region -> avg latency
	Regulations      map[string][]string // Region -> regulations
	DataResidency    map[string]string   // Service -> required region
	LastUpdated      time.Time
}

// ContextMetrics tracks context management performance
type ContextMetrics struct {
	TotalContextSwitches  int64
	AvgSwitchLatency      float64
	ContextHits           int64
	ContextMisses         int64
}

// NewContextManager creates a new context manager
func NewContextManager() *ContextManager {
	return &ContextManager{
		userContexts:      make(map[string]*UserContext),
		systemContext:     NewSystemContext(),
		businessContext:   NewBusinessContext(),
		temporalContext:   NewTemporalContext(),
		geospatialContext: NewGeospatialContext(),
	}
}

// NewSystemContext creates default system context
func NewSystemContext() *SystemContext {
	return &SystemContext{
		RecentAlerts: []Alert{},
		HealthScore:  1.0,
		LastUpdated:  time.Now(),
	}
}

// NewBusinessContext creates default business context
func NewBusinessContext() *BusinessContext {
	return &BusinessContext{
		ActiveSLAs:     []SLA{},
		CostCenters:    make(map[string]float64),
		ComplianceReqs: []string{},
		LastUpdated:    time.Now(),
	}
}

// NewTemporalContext creates temporal context
func NewTemporalContext() *TemporalContext {
	now := time.Now()
	hour := now.Hour()

	var timeOfDay string
	switch {
	case hour >= 5 && hour < 12:
		timeOfDay = "morning"
	case hour >= 12 && hour < 17:
		timeOfDay = "afternoon"
	case hour >= 17 && hour < 21:
		timeOfDay = "evening"
	default:
		timeOfDay = "night"
	}

	isBusinessHours := hour >= 9 && hour < 17 && now.Weekday() >= time.Monday && now.Weekday() <= time.Friday

	return &TemporalContext{
		CurrentTime:     now,
		TimeOfDay:       timeOfDay,
		DayOfWeek:       now.Weekday().String(),
		IsBusinessHours: isBusinessHours,
		LoadPatterns:    make(map[string]LoadPattern),
	}
}

// NewGeospatialContext creates geospatial context
func NewGeospatialContext() *GeospatialContext {
	return &GeospatialContext{
		ActiveRegions: []string{},
		RegionLatency: make(map[string]float64),
		Regulations:   make(map[string][]string),
		DataResidency: make(map[string]string),
		LastUpdated:   time.Now(),
	}
}

// GetUserContext retrieves user context
func (cm *ContextManager) GetUserContext(userID string) *UserContext {
	cm.lock.RLock()
	defer cm.lock.RUnlock()

	ctx, exists := cm.userContexts[userID]
	if !exists {
		return nil
	}
	return ctx
}

// UpdateUserContext updates user context
func (cm *ContextManager) UpdateUserContext(userID string, update func(*UserContext)) {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	ctx, exists := cm.userContexts[userID]
	if !exists {
		ctx = &UserContext{
			UserID:      userID,
			Preferences: make(map[string]interface{}),
			History:     []UserAction{},
			Expertise:   make(map[string]float64),
		}
		cm.userContexts[userID] = ctx
	}

	update(ctx)
	ctx.LastActivity = time.Now()
}

// RecordUserAction records a user action
func (cm *ContextManager) RecordUserAction(userID, action string, metadata map[string]interface{}) {
	cm.UpdateUserContext(userID, func(ctx *UserContext) {
		act := UserAction{
			Action:    action,
			Timestamp: time.Now(),
			Metadata:  metadata,
		}
		ctx.History = append(ctx.History, act)

		// Keep only last 100 actions
		if len(ctx.History) > 100 {
			ctx.History = ctx.History[len(ctx.History)-100:]
		}
	})
}

// GetSystemContext returns current system context
func (cm *ContextManager) GetSystemContext() *SystemContext {
	cm.lock.RLock()
	defer cm.lock.RUnlock()
	return cm.systemContext
}

// UpdateSystemContext updates system context
func (cm *ContextManager) UpdateSystemContext(update func(*SystemContext)) {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	update(cm.systemContext)
	cm.systemContext.LastUpdated = time.Now()
}

// GetBusinessContext returns business context
func (cm *ContextManager) GetBusinessContext() *BusinessContext {
	cm.lock.RLock()
	defer cm.lock.RUnlock()
	return cm.businessContext
}

// UpdateBusinessContext updates business context
func (cm *ContextManager) UpdateBusinessContext(update func(*BusinessContext)) {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	update(cm.businessContext)
	cm.businessContext.LastUpdated = time.Now()
}

// GetTemporalContext returns temporal context
func (cm *ContextManager) GetTemporalContext() *TemporalContext {
	cm.lock.Lock()
	defer cm.lock.Unlock()

	// Refresh temporal context
	cm.temporalContext = NewTemporalContext()
	return cm.temporalContext
}

// GetGeospatialContext returns geospatial context
func (cm *ContextManager) GetGeospatialContext() *GeospatialContext {
	cm.lock.RLock()
	defer cm.lock.RUnlock()
	return cm.geospatialContext
}

// SwitchContext switches context with latency tracking
func (cm *ContextManager) SwitchContext(ctx context.Context, fromUserID, toUserID string) error {
	startTime := time.Now()

	// Get contexts
	fromContext := cm.GetUserContext(fromUserID)
	toContext := cm.GetUserContext(toUserID)

	if fromContext == nil || toContext == nil {
		return fmt.Errorf("context not found")
	}

	// Record metrics
	latency := time.Since(startTime)
	cm.recordContextSwitch(latency)

	return nil
}

// GetFullContext returns all context dimensions
func (cm *ContextManager) GetFullContext(userID string) *FullContext {
	cm.lock.RLock()
	defer cm.lock.RUnlock()

	return &FullContext{
		User:        cm.userContexts[userID],
		System:      cm.systemContext,
		Business:    cm.businessContext,
		Temporal:    cm.temporalContext,
		Geospatial:  cm.geospatialContext,
	}
}

// FullContext contains all context dimensions
type FullContext struct {
	User        *UserContext
	System      *SystemContext
	Business    *BusinessContext
	Temporal    *TemporalContext
	Geospatial  *GeospatialContext
}

// ShouldRecommendAction determines if action should be recommended based on context
func (cm *ContextManager) ShouldRecommendAction(action string, fullContext *FullContext) bool {
	// Check temporal context
	if action == "maintenance" && fullContext.Temporal.IsBusinessHours {
		return false // Don't recommend maintenance during business hours
	}

	// Check business context
	if action == "scale_up" && fullContext.Business.CurrentSpend >= fullContext.Business.MonthlyBudget {
		return false // Don't recommend scaling if over budget
	}

	// Check system context
	if action == "deploy" && fullContext.System.ActiveIncidents > 0 {
		return false // Don't recommend deploys during incidents
	}

	return true
}

// recordContextSwitch records context switch metrics
func (cm *ContextManager) recordContextSwitch(latency time.Duration) {
	cm.metricsLock.Lock()
	defer cm.metricsLock.Unlock()

	cm.metrics.TotalContextSwitches++
	alpha := 0.1
	cm.metrics.AvgSwitchLatency = alpha*float64(latency.Milliseconds()) + (1-alpha)*cm.metrics.AvgSwitchLatency
}

// GetMetrics returns context metrics
func (cm *ContextManager) GetMetrics() ContextMetrics {
	cm.metricsLock.RLock()
	defer cm.metricsLock.RUnlock()
	return cm.metrics
}
