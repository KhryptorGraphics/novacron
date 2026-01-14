package tiering

import (
	"testing"
	"time"
)

func TestPolicyEngine_AddAdvancedPolicy(t *testing.T) {
	pe := NewPolicyEngine()

	err := pe.AddAdvancedPolicy("TestPolicy", 100, func(ctx *PolicyContext) (bool, TierLevel, error) {
		return true, TierWarm, nil
	})

	if err != nil {
		t.Errorf("AddAdvancedPolicy failed: %v", err)
	}

	if pe.GetPolicyCount() != 1 {
		t.Errorf("Expected 1 policy, got %d", pe.GetPolicyCount())
	}
}

func TestPolicyEngine_EvaluatePolicies(t *testing.T) {
	pe := NewPolicyEngine()

	// Add a test policy that always moves to warm tier
	err := pe.AddAdvancedPolicy("TestPolicy", 100, func(ctx *PolicyContext) (bool, TierLevel, error) {
		if ctx.VolumeStats.CurrentTier == TierHot {
			return true, TierWarm, nil
		}
		return false, ctx.VolumeStats.CurrentTier, nil
	})

	if err != nil {
		t.Errorf("AddAdvancedPolicy failed: %v", err)
	}

	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierHot,
		LastAccessed:    time.Now().Add(-1 * time.Hour),
		AccessFrequency: 0.5,
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		TimeOfDay:   time.Now(),
		DayOfWeek:   time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove {
		t.Error("Expected policy to recommend move")
	}

	if targetTier != TierWarm {
		t.Errorf("Expected target tier to be Warm, got %d", targetTier)
	}

	// Check metrics
	metrics := pe.GetMetrics()
	if metrics.TotalEvaluations != 1 {
		t.Errorf("Expected 1 evaluation, got %d", metrics.TotalEvaluations)
	}
}

func TestPolicyEngine_TimeBasedPolicy(t *testing.T) {
	pe := NewPolicyEngine()
	err := pe.CreateTimeBasedPolicy()
	if err != nil {
		t.Errorf("CreateTimeBasedPolicy failed: %v", err)
	}

	// Test business hours promotion
	businessHour := time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC) // Monday 10 AM
	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierWarm,
		LastAccessed:    time.Now(),
		AccessFrequency: 1.0, // Frequently accessed
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		TimeOfDay:   businessHour,
		DayOfWeek:   businessHour.Weekday(),
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove || targetTier != TierHot {
		t.Errorf("Expected to move frequently accessed volume to hot tier during business hours. shouldMove: %t, targetTier: %d, currentTier: %d, hour: %d, weekday: %s, accessFreq: %f", 
			shouldMove, targetTier, stats.CurrentTier, businessHour.Hour(), businessHour.Weekday(), stats.AccessFrequency)
	}
}

func TestPolicyEngine_CapacityBasedPolicy(t *testing.T) {
	pe := NewPolicyEngine()
	err := pe.CreateCapacityBasedPolicy()
	if err != nil {
		t.Errorf("CreateCapacityBasedPolicy failed: %v", err)
	}

	// Test hot tier full scenario
	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierHot,
		LastAccessed:    time.Now(),
		AccessFrequency: 0.5, // Less than 1.0, so eligible for demotion
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		TierCapacities: map[TierLevel]StorageCapacity{
			TierHot: {
				Total:     1000,
				Used:      950, // 95% full
				Available: 50,
				Usage:     95.0,
			},
		},
		TimeOfDay: time.Now(),
		DayOfWeek: time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove || targetTier != TierWarm {
		t.Error("Expected to move volume from full hot tier to warm tier")
	}
}

func TestPolicyEngine_PerformanceBasedPolicy(t *testing.T) {
	pe := NewPolicyEngine()
	err := pe.CreatePerformanceBasedPolicy()
	if err != nil {
		t.Errorf("CreatePerformanceBasedPolicy failed: %v", err)
	}

	// Test high CPU load scenario
	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierHot,
		LastAccessed:    time.Now(),
		AccessFrequency: 1.5, // Less than 2.0, eligible for demotion under load
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		SystemLoad: SystemLoadInfo{
			CPUUsage:         85.0, // High CPU load
			MemoryUsage:      75.0,
			NetworkBandwidth: 100.0,
			DiskIOPS:         500.0,
		},
		TimeOfDay: time.Now(),
		DayOfWeek: time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove || targetTier != TierWarm {
		t.Error("Expected to move volume from hot tier under high system load")
	}
}

func TestPolicyEngine_CostOptimizationPolicy(t *testing.T) {
	pe := NewPolicyEngine()
	err := pe.CreateCostOptimizationPolicy()
	if err != nil {
		t.Errorf("CreateCostOptimizationPolicy failed: %v", err)
	}

	// Test budget pressure scenario
	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierWarm,
		LastAccessed:    time.Now().Add(-48 * time.Hour),
		AccessFrequency: 0.2, // Low frequency
		SizeGB:          100,  // Large volume
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		CostBudget: CostBudget{
			MonthlyBudget:   1000.0,
			CurrentSpend:    850.0, // 85% of budget used
			RemainingBudget: 150.0,
			DaysRemaining:   10,
		},
		TimeOfDay: time.Now(),
		DayOfWeek: time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove || targetTier != TierCold {
		t.Error("Expected to move large, infrequent volume to cold tier under budget pressure")
	}
}

func TestPolicyEngine_MaintenancePolicy(t *testing.T) {
	pe := NewPolicyEngine()
	err := pe.CreateMaintenancePolicy()
	if err != nil {
		t.Errorf("CreateMaintenancePolicy failed: %v", err)
	}

	// Test maintenance mode scenario
	stats := &VolumeStats{
		Name:         "test-volume",
		CurrentTier:  TierHot,
		LastAccessed: time.Now().Add(-10 * 24 * time.Hour), // 10 days ago
		AccessFrequency: 0.1,
		SizeGB:       10,
		Pinned:       false,
	}

	context := &PolicyContext{
		VolumeStats:     stats,
		MaintenanceMode: true, // In maintenance mode
		TimeOfDay:       time.Now(),
		DayOfWeek:       time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	if !shouldMove || targetTier != TierWarm {
		t.Error("Expected to move old volume during maintenance mode")
	}
}

func TestPolicyEngine_PolicyPriority(t *testing.T) {
	pe := NewPolicyEngine()

	// Add policies with different priorities
	err := pe.AddAdvancedPolicy("LowPriority", 100, func(ctx *PolicyContext) (bool, TierLevel, error) {
		return true, TierCold, nil // Low priority wants cold
	})
	if err != nil {
		t.Errorf("AddAdvancedPolicy failed: %v", err)
	}

	err = pe.AddAdvancedPolicy("HighPriority", 200, func(ctx *PolicyContext) (bool, TierLevel, error) {
		return true, TierHot, nil // High priority wants hot
	})
	if err != nil {
		t.Errorf("AddAdvancedPolicy failed: %v", err)
	}

	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierWarm,
		LastAccessed:    time.Now(),
		AccessFrequency: 1.0,
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		TimeOfDay:   time.Now(),
		DayOfWeek:   time.Wednesday,
	}

	shouldMove, targetTier, err := pe.EvaluatePolicies(stats, context)
	if err != nil {
		t.Errorf("EvaluatePolicies failed: %v", err)
	}

	// High priority policy should win
	if !shouldMove || targetTier != TierHot {
		t.Error("Expected high priority policy to take precedence")
	}
}

func TestPolicyEngine_Metrics(t *testing.T) {
	pe := NewPolicyEngine()

	err := pe.AddAdvancedPolicy("TestPolicy", 100, func(ctx *PolicyContext) (bool, TierLevel, error) {
		time.Sleep(1 * time.Millisecond) // Small delay to measure latency
		return true, TierWarm, nil
	})
	if err != nil {
		t.Errorf("AddAdvancedPolicy failed: %v", err)
	}

	stats := &VolumeStats{
		Name:            "test-volume",
		CurrentTier:     TierHot,
		LastAccessed:    time.Now(),
		AccessFrequency: 1.0,
		SizeGB:          10,
		Pinned:          false,
	}

	context := &PolicyContext{
		VolumeStats: stats,
		TimeOfDay:   time.Now(),
		DayOfWeek:   time.Wednesday,
	}

	// Execute policy multiple times
	for i := 0; i < 3; i++ {
		_, _, err := pe.EvaluatePolicies(stats, context)
		if err != nil {
			t.Errorf("EvaluatePolicies failed: %v", err)
		}
	}

	metrics := pe.GetMetrics()

	if metrics.TotalEvaluations != 3 {
		t.Errorf("Expected 3 evaluations, got %d", metrics.TotalEvaluations)
	}

	if execCount, exists := metrics.PolicyExecutions["TestPolicy"]; !exists || execCount != 3 {
		t.Errorf("Expected 3 policy executions, got %d", execCount)
	}

	if moveCount, exists := metrics.VolumesMoved["TestPolicy"]; !exists || moveCount != 3 {
		t.Errorf("Expected 3 volumes moved, got %d", moveCount)
	}

	if _, exists := metrics.PolicyLatency["TestPolicy"]; !exists {
		t.Error("Expected policy latency to be recorded")
	}
}

func TestPolicyEngine_ListPolicies(t *testing.T) {
	pe := NewPolicyEngine()

	// Add some policies
	policies := []string{"Policy1", "Policy2", "Policy3"}
	for i, name := range policies {
		err := pe.AddAdvancedPolicy(name, 100+i, func(ctx *PolicyContext) (bool, TierLevel, error) {
			return false, ctx.VolumeStats.CurrentTier, nil
		})
		if err != nil {
			t.Errorf("AddAdvancedPolicy failed: %v", err)
		}
	}

	listedPolicies := pe.ListPolicies()
	if len(listedPolicies) != len(policies) {
		t.Errorf("Expected %d policies, got %d", len(policies), len(listedPolicies))
	}

	// Check if all policies are listed (order might be different due to priority sorting)
	policySet := make(map[string]bool)
	for _, policy := range listedPolicies {
		policySet[policy] = true
	}

	for _, expectedPolicy := range policies {
		if !policySet[expectedPolicy] {
			t.Errorf("Policy %s not found in listed policies", expectedPolicy)
		}
	}
}