package policy

import (
	"context"
	"testing"
	"time"
)

func TestOPAEngine(t *testing.T) {
	ctx := context.Background()

	t.Run("Add Policy", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:        "Test Policy",
			Category:    CategoryAccessControl,
			Description: "Test access control policy",
			Rego:        "package test\nallow { true }",
			Priority:    100,
		}

		err := oe.AddPolicy(ctx, policy)
		if err != nil {
			t.Fatalf("Failed to add policy: %v", err)
		}

		if policy.ID == "" {
			t.Error("Expected policy ID to be generated")
		}
	})

	t.Run("Evaluate Access Control Policy", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:     "Admin Access",
			Category: CategoryAccessControl,
			Rego:     "package authz\nallow { input.user.role == \"admin\" }",
		}

		oe.AddPolicy(ctx, policy)

		request := &PolicyEvaluationRequest{
			PolicyID: policy.ID,
			Input: map[string]interface{}{
				"required_role": "admin",
			},
			Context: PolicyContext{
				UserID:    "user-123",
				Roles:     []string{"admin", "operator"},
				Timestamp: time.Now(),
			},
		}

		result, err := oe.EvaluatePolicy(ctx, request)
		if err != nil {
			t.Fatalf("Failed to evaluate policy: %v", err)
		}

		if !result.Allowed {
			t.Error("Expected access to be allowed")
		}

		if result.EvaluationTime == 0 {
			t.Error("Expected evaluation time to be tracked")
		}
	})

	t.Run("Evaluate Quota Policy", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:     "CPU Quota",
			Category: CategoryQuota,
		}

		oe.AddPolicy(ctx, policy)

		request := &PolicyEvaluationRequest{
			PolicyID: policy.ID,
			Input: map[string]interface{}{
				"current_usage": 80.0,
				"quota_limit":   100.0,
			},
			Context: PolicyContext{
				TenantID: "tenant-1",
			},
		}

		result, err := oe.EvaluatePolicy(ctx, request)
		if err != nil {
			t.Fatalf("Failed to evaluate policy: %v", err)
		}

		if !result.Allowed {
			t.Error("Expected quota to be within limits")
		}
	})

	t.Run("Policy Cache", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:     "Cached Policy",
			Category: CategoryAccessControl,
		}

		oe.AddPolicy(ctx, policy)

		request := &PolicyEvaluationRequest{
			PolicyID: policy.ID,
			Input:    map[string]interface{}{},
			Context:  PolicyContext{},
		}

		// First evaluation
		result1, _ := oe.EvaluatePolicy(ctx, request)

		// Second evaluation (should be cached)
		result2, _ := oe.EvaluatePolicy(ctx, request)

		if result1.PolicyID != result2.PolicyID {
			t.Error("Expected cached result")
		}

		metrics := oe.GetMetrics()
		if metrics.CacheHitRate == 0 {
			t.Error("Expected cache hit rate to be tracked")
		}
	})

	t.Run("Policy Versioning", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:     "Versioned Policy",
			Category: CategoryCompliance,
			Version:  "1.0.0",
		}

		oe.AddPolicy(ctx, policy)

		// Update policy
		updates := &Policy{
			Name:        "Updated Policy",
			Category:    CategoryCompliance,
			Description: "Updated description",
		}

		err := oe.UpdatePolicy(ctx, policy.ID, updates)
		if err != nil {
			t.Fatalf("Failed to update policy: %v", err)
		}

		// Check version was incremented
		if updates.Version == policy.Version {
			t.Error("Expected version to be incremented")
		}
	})

	t.Run("Policy Rollback", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:    "Rollback Test",
			Category: CategorySecurity,
			Version: "1.0.0",
		}

		oe.AddPolicy(ctx, policy)
		originalVersion := policy.Version

		// Update policy
		updates := &Policy{
			Name:    "Updated",
			Category: CategorySecurity,
		}
		oe.UpdatePolicy(ctx, policy.ID, updates)

		// Rollback to original version
		err := oe.RollbackPolicy(ctx, policy.ID, originalVersion)
		if err != nil {
			t.Fatalf("Failed to rollback policy: %v", err)
		}
	})

	t.Run("Performance Target", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		policy := &Policy{
			Name:     "Performance Test",
			Category: CategoryAccessControl,
		}

		oe.AddPolicy(ctx, policy)

		request := &PolicyEvaluationRequest{
			PolicyID: policy.ID,
			Input:    map[string]interface{}{},
			Context:  PolicyContext{},
		}

		result, _ := oe.EvaluatePolicy(ctx, request)

		if result.EvaluationTime > 10*time.Millisecond {
			t.Errorf("Policy evaluation took too long: %v", result.EvaluationTime)
		}
	})

	t.Run("Load Policy Bundle", func(t *testing.T) {
		oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, true, 5*time.Minute)

		err := oe.LoadPolicyBundle(ctx, "http://policy-bundle-url")
		if err != nil {
			t.Fatalf("Failed to load policy bundle: %v", err)
		}

		// Should have default policies loaded
		policies := oe.GetPoliciesByCategory(CategoryAccessControl)
		if len(policies) == 0 {
			t.Error("Expected policies to be loaded")
		}
	})
}

func BenchmarkPolicyEvaluation(b *testing.B) {
	ctx := context.Background()
	oe := NewOPAEngine(10*time.Millisecond, 5*time.Millisecond, false, 5*time.Minute) // Cache disabled for benchmark

	policy := &Policy{
		Name:     "Benchmark Policy",
		Category: CategoryAccessControl,
	}

	oe.AddPolicy(ctx, policy)

	request := &PolicyEvaluationRequest{
		PolicyID: policy.ID,
		Input: map[string]interface{}{
			"required_role": "admin",
		},
		Context: PolicyContext{
			Roles: []string{"admin"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		oe.EvaluatePolicy(ctx, request)
	}
}
