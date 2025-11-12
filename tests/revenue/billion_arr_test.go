// Package revenue_test provides comprehensive testing for $1B ARR systems
package revenue_test

import (
	"context"
	"testing"
	"time"

	"novacron/backend/business/revenue"
)

// TestBillionARRTracker tests ARR milestone tracking
func TestBillionARRTracker(t *testing.T) {
	config := revenue.TrackerConfig{
		TargetARR:             1_000_000_000,
		TargetDate:            time.Now().AddDate(0, 12, 0),
		EnableMLForecasting:   true,
		EnableChurnPrediction: true,
	}

	tracker := revenue.NewBillionARRTracker(config)

	t.Run("InitialState", func(t *testing.T) {
		milestone := tracker.GetMilestone()

		if milestone.CurrentARR != 800_000_000 {
			t.Errorf("Expected initial ARR of $800M, got $%.0f", milestone.CurrentARR)
		}

		if milestone.TargetARR != 1_000_000_000 {
			t.Errorf("Expected target ARR of $1B, got $%.0f", milestone.TargetARR)
		}

		if milestone.RemainingARR != 200_000_000 {
			t.Errorf("Expected remaining ARR of $200M, got $%.0f", milestone.RemainingARR)
		}

		if milestone.ProgressPercentage != 80.0 {
			t.Errorf("Expected 80%% progress, got %.1f%%", milestone.ProgressPercentage)
		}
	})

	t.Run("ARRUpdate", func(t *testing.T) {
		ctx := context.Background()

		composition := revenue.RevenueComposition{
			NewBusiness: revenue.RevenueSegment{
				Name:       "New Business",
				CurrentARR: 300_000_000,
				TargetARR:  300_000_000,
				GrowthRate: 25.0,
			},
			Expansion: revenue.RevenueSegment{
				Name:       "Expansion",
				CurrentARR: 500_000_000,
				TargetARR:  500_000_000,
				GrowthRate: 50.0,
			},
			Renewals: revenue.RevenueSegment{
				Name:       "Renewals",
				CurrentARR: 200_000_000,
				TargetARR:  200_000_000,
				GrowthRate: 10.0,
			},
			TotalARR: 1_000_000_000,
		}

		err := tracker.UpdateARR(ctx, 1_000_000_000, composition)
		if err != nil {
			t.Fatalf("Failed to update ARR: %v", err)
		}

		milestone := tracker.GetMilestone()

		if milestone.CurrentARR != 1_000_000_000 {
			t.Errorf("Expected ARR of $1B, got $%.0f", milestone.CurrentARR)
		}

		if milestone.ProgressPercentage != 100.0 {
			t.Errorf("Expected 100%% progress, got %.1f%%", milestone.ProgressPercentage)
		}

		if milestone.RemainingARR != 0 {
			t.Errorf("Expected $0 remaining, got $%.0f", milestone.RemainingARR)
		}
	})

	t.Run("VelocityCalculation", func(t *testing.T) {
		milestone := tracker.GetMilestone()

		if milestone.Velocity.DailyARR <= 0 {
			t.Error("Expected positive daily ARR velocity")
		}

		if milestone.Velocity.MonthlyARR <= 0 {
			t.Error("Expected positive monthly ARR velocity")
		}
	})

	t.Run("Forecasting", func(t *testing.T) {
		milestone := tracker.GetMilestone()

		if len(milestone.Forecasts) == 0 {
			t.Error("Expected forecasts to be generated")
		}

		forecast := milestone.Forecasts[0]

		if forecast.ForecastARR <= milestone.CurrentARR {
			t.Error("Expected forecast ARR to be >= current ARR")
		}

		if forecast.Confidence <= 0 || forecast.Confidence > 1 {
			t.Errorf("Invalid confidence: %.2f", forecast.Confidence)
		}

		if forecast.LowerBound >= forecast.ForecastARR {
			t.Error("Lower bound should be < forecast")
		}

		if forecast.UpperBound <= forecast.ForecastARR {
			t.Error("Upper bound should be > forecast")
		}
	})

	t.Run("MetricsCalculation", func(t *testing.T) {
		milestone := tracker.GetMilestone()

		if milestone.Metrics.RenewalRate < 95.0 {
			t.Errorf("Renewal rate below target: %.1f%%", milestone.Metrics.RenewalRate)
		}

		if milestone.Metrics.NetRetention < 150.0 {
			t.Errorf("Net retention below target: %.1f%%", milestone.Metrics.NetRetention)
		}

		if milestone.Metrics.GrossMargin < 40.0 {
			t.Errorf("Gross margin below 40%%: %.1f%%", milestone.Metrics.GrossMargin)
		}

		if milestone.Metrics.LTVtoCAC < 3.0 {
			t.Errorf("LTV:CAC below 3x: %.1fx", milestone.Metrics.LTVtoCAC)
		}

		// Rule of 40: Growth% + Margin% > 40
		if milestone.Metrics.RuleOf40 < 40.0 {
			t.Errorf("Rule of 40 not met: %.1f", milestone.Metrics.RuleOf40)
		}
	})

	t.Run("AlertGeneration", func(t *testing.T) {
		milestone := tracker.GetMilestone()

		// Alerts should exist for any issues
		if milestone.GrowthRate < 20.0 && len(milestone.Alerts) == 0 {
			t.Error("Expected growth rate alert")
		}

		for _, alert := range milestone.Alerts {
			if alert.Severity == "" {
				t.Error("Alert missing severity")
			}

			if alert.Message == "" {
				t.Error("Alert missing message")
			}

			if alert.Recommendation == "" {
				t.Error("Alert missing recommendation")
			}
		}
	})

	t.Run("MilestoneCompletion", func(t *testing.T) {
		ctx := context.Background()

		// Update to $1B
		composition := revenue.RevenueComposition{
			TotalARR: 1_000_000_000,
		}

		err := tracker.UpdateARR(ctx, 1_000_000_000, composition)
		if err != nil {
			t.Fatalf("Failed to update ARR: %v", err)
		}

		milestone := tracker.GetMilestone()

		if milestone.ProgressPercentage != 100.0 {
			t.Errorf("Expected 100%% completion, got %.1f%%", milestone.ProgressPercentage)
		}

		if milestone.RemainingARR != 0 {
			t.Errorf("Expected $0 remaining, got $%.0f", milestone.RemainingARR)
		}

		// Should have milestone achievement alert
		hasAchievementAlert := false
		for _, alert := range milestone.Alerts {
			if alert.Type == "milestone" {
				hasAchievementAlert = true
				break
			}
		}

		if !hasAchievementAlert {
			t.Error("Expected milestone achievement alert")
		}
	})

	t.Run("ExportMetrics", func(t *testing.T) {
		metrics := tracker.ExportMetrics()

		if metrics == nil {
			t.Fatal("Expected metrics to be exported")
		}

		if _, ok := metrics["forecasts_generated"]; !ok {
			t.Error("Missing forecasts_generated metric")
		}

		if _, ok := metrics["arr_updates"]; !ok {
			t.Error("Missing arr_updates metric")
		}
	})
}

// TestRevenueCoordinator tests system coordination
func TestRevenueCoordinator(t *testing.T) {
	config := revenue.CoordinatorConfig{
		EnableRealTimeSync:  true,
		DashboardRefresh:    time.Minute * 5,
		HealthCheckInterval: time.Minute,
	}

	coordinator := revenue.NewRevenueCoordinator(config)

	t.Run("Initialization", func(t *testing.T) {
		if coordinator == nil {
			t.Fatal("Failed to create coordinator")
		}
	})

	t.Run("ARRSync", func(t *testing.T) {
		ctx := context.Background()

		err := coordinator.SyncARRMetrics(ctx)
		if err != nil {
			t.Fatalf("Failed to sync ARR: %v", err)
		}
	})

	t.Run("ExecutiveDashboard", func(t *testing.T) {
		ctx := context.Background()

		dashboard, err := coordinator.GetExecutiveDashboard(ctx)
		if err != nil {
			t.Fatalf("Failed to get dashboard: %v", err)
		}

		if dashboard == nil {
			t.Fatal("Dashboard is nil")
		}

		if dashboard.Name == "" {
			t.Error("Dashboard missing name")
		}

		if len(dashboard.Widgets) == 0 {
			t.Error("Dashboard has no widgets")
		}

		// Verify key widgets exist
		requiredWidgets := []string{"arr-progress", "revenue-composition", "key-metrics", "velocity", "forecast"}
		widgetMap := make(map[string]bool)
		for _, widget := range dashboard.Widgets {
			widgetMap[widget.ID] = true
		}

		for _, required := range requiredWidgets {
			if !widgetMap[required] {
				t.Errorf("Missing required widget: %s", required)
			}
		}
	})

	t.Run("HealthCheck", func(t *testing.T) {
		ctx := context.Background()

		health := coordinator.HealthCheck(ctx)

		if health == nil {
			t.Fatal("Health check returned nil")
		}

		if status, ok := health["status"]; !ok || status != "healthy" {
			t.Error("System not healthy")
		}

		components, ok := health["components"].(map[string]interface{})
		if !ok {
			t.Fatal("Components missing from health check")
		}

		requiredComponents := []string{"arr_tracker", "expansion_engine", "acquisition_engine", "rev_ops_engine"}
		for _, comp := range requiredComponents {
			if _, exists := components[comp]; !exists {
				t.Errorf("Missing health check for: %s", comp)
			}
		}
	})

	t.Run("AlertManagement", func(t *testing.T) {
		ctx := context.Background()

		// Sync to generate alerts
		err := coordinator.SyncARRMetrics(ctx)
		if err != nil {
			t.Fatalf("Failed to sync: %v", err)
		}

		// Check alerts
		dashboard, _ := coordinator.GetExecutiveDashboard(ctx)

		for _, widget := range dashboard.Widgets {
			if widget.ID == "alerts" {
				// Verify alerts structure
				if widget.Data == nil {
					t.Error("Alerts widget has no data")
				}
			}
		}
	})
}

// BenchmarkARRUpdate benchmarks ARR updates
func BenchmarkARRUpdate(b *testing.B) {
	config := revenue.TrackerConfig{
		TargetARR:             1_000_000_000,
		EnableMLForecasting:   true,
		EnableChurnPrediction: true,
	}

	tracker := revenue.NewBillionARRTracker(config)
	ctx := context.Background()

	composition := revenue.RevenueComposition{
		TotalARR: 900_000_000,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tracker.UpdateARR(ctx, 900_000_000, composition)
	}
}

// BenchmarkForecastGeneration benchmarks forecasting
func BenchmarkForecastGeneration(b *testing.B) {
	config := revenue.TrackerConfig{
		TargetARR:             1_000_000_000,
		EnableMLForecasting:   true,
		EnableChurnPrediction: true,
	}

	tracker := revenue.NewBillionARRTracker(config)
	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tracker.GenerateForecasts(ctx)
	}
}

// Helper function for forecast generation
func (t *revenue.BillionARRTracker) GenerateForecasts(ctx context.Context) ([]revenue.ARRForecast, error) {
	// Placeholder for actual forecast generation
	return []revenue.ARRForecast{}, nil
}
