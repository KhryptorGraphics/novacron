package tiering

import (
	"context"
	"log"
	"sync"
	"time"
)

// AdvancedTieringPolicy is a context-aware tiering policy
type AdvancedTieringPolicy struct {
	Name         string
	Priority     int
	EvaluateFunc func(*PolicyContext) (bool, TierLevel, error)
}

// PolicyEngine manages and evaluates tiering policies
type PolicyEngine struct {
	policies []AdvancedTieringPolicy
	mu       sync.RWMutex
	metrics  *PolicyMetrics
	ctx      context.Context
	cancel   context.CancelFunc
}

// PolicyMetrics tracks policy execution statistics
type PolicyMetrics struct {
	PolicyExecutions map[string]int64     `json:"policy_executions"`
	PolicyLatency    map[string]time.Duration `json:"policy_latency"`
	VolumesMoved     map[string]int64     `json:"volumes_moved"`
	LastExecution    time.Time            `json:"last_execution"`
	TotalEvaluations int64                `json:"total_evaluations"`
	mu               sync.RWMutex
}

// PolicyContext provides context for policy evaluation
type PolicyContext struct {
	VolumeStats      *VolumeStats
	TierCapacities   map[TierLevel]StorageCapacity
	SystemLoad       SystemLoadInfo
	CostBudget       CostBudget
	TimeOfDay        time.Time
	DayOfWeek        time.Weekday
	MaintenanceMode  bool
}

// StorageCapacity represents capacity information for a tier
type StorageCapacity struct {
	Total     int64   `json:"total_gb"`
	Used      int64   `json:"used_gb"`
	Available int64   `json:"available_gb"`
	Usage     float64 `json:"usage_percent"`
}

// SystemLoadInfo provides system load context for policy decisions
type SystemLoadInfo struct {
	CPUUsage        float64 `json:"cpu_usage"`
	MemoryUsage     float64 `json:"memory_usage"`
	NetworkBandwidth float64 `json:"network_bandwidth"`
	DiskIOPS        float64 `json:"disk_iops"`
}

// CostBudget defines cost constraints for tiering decisions
type CostBudget struct {
	MonthlyBudget  float64 `json:"monthly_budget"`
	CurrentSpend   float64 `json:"current_spend"`
	RemainingBudget float64 `json:"remaining_budget"`
	DaysRemaining  int     `json:"days_remaining"`
}

// NewPolicyEngine creates a new policy engine
func NewPolicyEngine() *PolicyEngine {
	ctx, cancel := context.WithCancel(context.Background())
	return &PolicyEngine{
		policies: make([]AdvancedTieringPolicy, 0),
		metrics: &PolicyMetrics{
			PolicyExecutions: make(map[string]int64),
			PolicyLatency:    make(map[string]time.Duration),
			VolumesMoved:     make(map[string]int64),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// AddAdvancedPolicy adds a context-aware tiering policy
func (pe *PolicyEngine) AddAdvancedPolicy(name string, priority int, evaluator func(*PolicyContext) (bool, TierLevel, error)) error {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Create enhanced policy
	policy := AdvancedTieringPolicy{
		Name:         name,
		Priority:     priority,
		EvaluateFunc: evaluator,
	}

	pe.policies = append(pe.policies, policy)

	// Sort by priority (higher first)
	for i := 0; i < len(pe.policies)-1; i++ {
		for j := i + 1; j < len(pe.policies); j++ {
			if pe.policies[i].Priority < pe.policies[j].Priority {
				pe.policies[i], pe.policies[j] = pe.policies[j], pe.policies[i]
			}
		}
	}

	return nil
}

// EvaluatePolicies evaluates all policies for a volume with metrics
func (pe *PolicyEngine) EvaluatePolicies(stats *VolumeStats, context *PolicyContext) (bool, TierLevel, error) {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	pe.metrics.mu.Lock()
	pe.metrics.TotalEvaluations++
	pe.metrics.LastExecution = time.Now()
	pe.metrics.mu.Unlock()

	for _, policy := range pe.policies {
		start := time.Now()

		// Use the context if provided, otherwise create a basic one
		if context == nil {
			context = &PolicyContext{
				VolumeStats: stats,
				TimeOfDay:   time.Now(),
				DayOfWeek:   time.Now().Weekday(),
			}
		}

		shouldMove, targetTier, err := policy.EvaluateFunc(context)
		if err != nil {
			log.Printf("Policy %s evaluation error: %v", policy.Name, err)
			continue
		}
		
		duration := time.Since(start)
		
		// Update metrics
		pe.metrics.mu.Lock()
		pe.metrics.PolicyExecutions[policy.Name]++
		pe.metrics.PolicyLatency[policy.Name] = duration
		pe.metrics.mu.Unlock()

		if shouldMove && targetTier != stats.CurrentTier {
			pe.metrics.mu.Lock()
			pe.metrics.VolumesMoved[policy.Name]++
			pe.metrics.mu.Unlock()
			
			log.Printf("Policy %s recommends moving volume %s from tier %d to tier %d",
				policy.Name, stats.Name, stats.CurrentTier, targetTier)
			return true, targetTier, nil
		}
	}

	return false, stats.CurrentTier, nil
}

// GetMetrics returns current policy engine metrics
func (pe *PolicyEngine) GetMetrics() PolicyMetrics {
	pe.metrics.mu.RLock()
	defer pe.metrics.mu.RUnlock()

	// Create a copy to avoid data races
	metrics := PolicyMetrics{
		PolicyExecutions: make(map[string]int64),
		PolicyLatency:    make(map[string]time.Duration),
		VolumesMoved:     make(map[string]int64),
		LastExecution:    pe.metrics.LastExecution,
		TotalEvaluations: pe.metrics.TotalEvaluations,
	}

	for k, v := range pe.metrics.PolicyExecutions {
		metrics.PolicyExecutions[k] = v
	}
	for k, v := range pe.metrics.PolicyLatency {
		metrics.PolicyLatency[k] = v
	}
	for k, v := range pe.metrics.VolumesMoved {
		metrics.VolumesMoved[k] = v
	}

	return metrics
}

// CreateTimeBasedPolicy creates a policy that considers time patterns
func (pe *PolicyEngine) CreateTimeBasedPolicy() error {
	return pe.AddAdvancedPolicy("TimeBased", 200, func(ctx *PolicyContext) (bool, TierLevel, error) {
		stats := ctx.VolumeStats
		hour := ctx.TimeOfDay.Hour()
		weekday := ctx.DayOfWeek

		// Business hours (weekdays 9-17): prefer hot tier for active volumes
		if weekday >= time.Monday && weekday <= time.Friday && hour >= 9 && hour <= 17 {
			if stats.AccessFrequency > 0.5 && stats.CurrentTier != TierHot {
				return true, TierHot, nil
			}
		}

		// Off-hours and weekends: move less active data to warm tier
		if (weekday == time.Saturday || weekday == time.Sunday) || hour < 6 || hour > 22 {
			if stats.AccessFrequency < 0.2 && stats.CurrentTier == TierHot {
				return true, TierWarm, nil
			}
		}
		return false, stats.CurrentTier, nil
	})
}

// CreateCapacityBasedPolicy creates a policy that manages tier capacity
func (pe *PolicyEngine) CreateCapacityBasedPolicy() error {
	return pe.AddAdvancedPolicy("CapacityBased", 300, func(ctx *PolicyContext) (bool, TierLevel, error) {
		stats := ctx.VolumeStats

		// Check hot tier capacity
		if hotCapacity, exists := ctx.TierCapacities[TierHot]; exists {
			// If hot tier is >90% full, move less critical data to warm tier
			if hotCapacity.Usage > 90.0 && stats.CurrentTier == TierHot && stats.AccessFrequency < 1.0 {
				return true, TierWarm, nil
			}
		}

		// Check warm tier capacity
		if warmCapacity, exists := ctx.TierCapacities[TierWarm]; exists {
			// If warm tier is >85% full, move oldest data to cold tier
			if warmCapacity.Usage > 85.0 && stats.CurrentTier == TierWarm {
				daysSinceAccess := time.Since(stats.LastAccessed).Hours() / 24.0
				if daysSinceAccess > 30 { // Not accessed in 30 days
					return true, TierCold, nil
				}
			}
		}

		// If cold tier has capacity and data hasn't been accessed in 90 days, archive it
		if coldCapacity, exists := ctx.TierCapacities[TierCold]; exists && coldCapacity.Usage < 95.0 {
			daysSinceAccess := time.Since(stats.LastAccessed).Hours() / 24.0
			if daysSinceAccess > 90 && stats.CurrentTier != TierCold {
				return true, TierCold, nil
			}
		}

		return false, stats.CurrentTier, nil
	})
}

// CreatePerformanceBasedPolicy creates a policy based on system performance
func (pe *PolicyEngine) CreatePerformanceBasedPolicy() error {
	return pe.AddAdvancedPolicy("PerformanceBased", 250, func(ctx *PolicyContext) (bool, TierLevel, error) {
		stats := ctx.VolumeStats
		load := ctx.SystemLoad

		// If system is under high load, move less critical data away from hot tier
		if load.CPUUsage > 80.0 || load.MemoryUsage > 80.0 {
			if stats.CurrentTier == TierHot && stats.AccessFrequency < 2.0 {
				return true, TierWarm, nil
			}
		}

		// If disk IOPS is high, spread the load
		if load.DiskIOPS > 1000 && stats.CurrentTier == TierHot && stats.AccessFrequency < 0.5 {
			return true, TierWarm, nil
		}

		// If system load is low, we can promote frequently accessed data
		if load.CPUUsage < 50.0 && load.MemoryUsage < 50.0 {
			if stats.AccessFrequency > 3.0 && stats.CurrentTier != TierHot {
				return true, TierHot, nil
			}
		}

		return false, stats.CurrentTier, nil
	})
}

// CreateCostOptimizationPolicy creates an advanced cost-based policy
func (pe *PolicyEngine) CreateCostOptimizationPolicy() error {
	return pe.AddAdvancedPolicy("CostOptimization", 150, func(ctx *PolicyContext) (bool, TierLevel, error) {
		stats := ctx.VolumeStats
		budget := ctx.CostBudget

		// If we're approaching budget limits, be more aggressive about moving to cheaper tiers
		budgetPressure := budget.CurrentSpend / budget.MonthlyBudget

		if budgetPressure > 0.8 { // Over 80% of budget used
			// Move large, infrequently accessed volumes to cold tier
			if stats.SizeGB > 50 && stats.AccessFrequency < 0.3 && stats.CurrentTier != TierCold {
				return true, TierCold, nil
			}

			// Move medium volumes to warm tier
			if stats.SizeGB > 10 && stats.AccessFrequency < 1.0 && stats.CurrentTier == TierHot {
				return true, TierWarm, nil
			}
		}

		// If budget is healthy and we're not in maintenance mode, optimize for performance
		if budgetPressure < 0.6 && !ctx.MaintenanceMode {
			if stats.AccessFrequency > 5.0 && stats.CurrentTier != TierHot {
				return true, TierHot, nil
			}
		}

		return false, stats.CurrentTier, nil
	})
}

// CreateMaintenancePolicy creates a policy for maintenance windows
func (pe *PolicyEngine) CreateMaintenancePolicy() error {
	return pe.AddAdvancedPolicy("Maintenance", 400, func(ctx *PolicyContext) (bool, TierLevel, error) {
		if !ctx.MaintenanceMode {
			return false, ctx.VolumeStats.CurrentTier, nil
		}

		stats := ctx.VolumeStats

		// During maintenance, we can do more aggressive tiering
		// Move everything that hasn't been accessed in 7 days to lower tiers
		daysSinceAccess := time.Since(stats.LastAccessed).Hours() / 24.0

		if daysSinceAccess > 7 {
			if stats.CurrentTier == TierHot {
				return true, TierWarm, nil
			}
			if daysSinceAccess > 30 && stats.CurrentTier == TierWarm {
				return true, TierCold, nil
			}
		}

		return false, stats.CurrentTier, nil
	})
}

// Stop stops the policy engine
func (pe *PolicyEngine) Stop() {
	pe.cancel()
}

// GetPolicyCount returns the number of registered policies
func (pe *PolicyEngine) GetPolicyCount() int {
	pe.mu.RLock()
	defer pe.mu.RUnlock()
	return len(pe.policies)
}

// ListPolicies returns a list of all registered policies
func (pe *PolicyEngine) ListPolicies() []string {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	names := make([]string, len(pe.policies))
	for i, policy := range pe.policies {
		names[i] = policy.Name
	}
	return names
}