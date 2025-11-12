package ma_test

import (
	"context"
	"testing"
	"time"

	"novacron/backend/corporate/ma"
)

func TestEvaluationEngine(t *testing.T) {
	engine := ma.NewEvaluationEngine()
	ctx := context.Background()

	t.Run("ScreenTarget", func(t *testing.T) {
		target := &ma.AcquisitionTarget{
			ID:       "target-1",
			Name:     "StorageTech Inc",
			Category: ma.CategoryStorage,
			Stage:    ma.StageScreening,
			ValuationRange: ma.ValuationRange{
				Low:      200.0,
				Mid:      300.0,
				High:     400.0,
				Currency: "USD",
				Basis:    "Revenue multiple",
				Multiple: 6.0,
			},
			Revenue:   50.0,
			Growth:    0.35,
			Customers: 1200,
			Employees: 350,
			Technology: ma.TechnologyAssets{
				Patents:          75,
				Developers:       150,
				CodebaseSize:     5000000,
				TechStack:        []string{"Go", "Kubernetes", "Ceph"},
				Infrastructure:   []string{"AWS", "GCP"},
				SecurityCerts:    []string{"SOC2", "ISO27001"},
				ComplianceFrames: []string{"GDPR", "HIPAA"},
			},
		}

		err := engine.ScreenTarget(ctx, target)
		if err != nil {
			t.Fatalf("ScreenTarget failed: %v", err)
		}

		// Verify strategic fit was calculated
		if target.StrategicFit.Overall == 0 {
			t.Error("Strategic fit score not calculated")
		}

		if target.StrategicFit.Overall < 0 || target.StrategicFit.Overall > 100 {
			t.Errorf("Strategic fit score out of range: %f", target.StrategicFit.Overall)
		}

		// Verify synergies were identified
		if len(target.StrategicFit.Synergies) == 0 {
			t.Error("No synergies identified")
		}

		// Verify target was stored
		retrieved, err := engine.GetTarget(target.ID)
		if err != nil {
			t.Fatalf("GetTarget failed: %v", err)
		}

		if retrieved.ID != target.ID {
			t.Errorf("Retrieved target ID mismatch: got %s, want %s", retrieved.ID, target.ID)
		}
	})

	t.Run("PerformDueDiligence", func(t *testing.T) {
		target := &ma.AcquisitionTarget{
			ID:       "target-2",
			Name:     "NetworkSDN Corp",
			Category: ma.CategoryNetworking,
			Revenue:  80.0,
			Growth:   0.40,
		}

		// Screen target first
		err := engine.ScreenTarget(ctx, target)
		if err != nil {
			t.Fatalf("ScreenTarget failed: %v", err)
		}

		// Perform due diligence
		err = engine.PerformDueDiligence(ctx, target.ID)
		if err != nil {
			t.Fatalf("PerformDueDiligence failed: %v", err)
		}

		// Verify due diligence was completed
		retrieved, _ := engine.GetTarget(target.ID)
		if retrieved.DueDiligence.Status != "complete" {
			t.Errorf("Due diligence status: got %s, want complete", retrieved.DueDiligence.Status)
		}

		// Verify financial diligence
		if retrieved.DueDiligence.Financial.Status != "complete" {
			t.Error("Financial diligence not completed")
		}

		// Verify recommendations were generated
		if len(retrieved.DueDiligence.Recommendations) == 0 {
			t.Error("No recommendations generated")
		}
	})

	t.Run("BuildFinancialModel", func(t *testing.T) {
		target := &ma.AcquisitionTarget{
			ID:       "target-3",
			Name:     "SecurityCloud Inc",
			Category: ma.CategorySecurity,
			Revenue:  120.0,
			Growth:   0.45,
			ValuationRange: ma.ValuationRange{
				Mid: 600.0,
			},
		}

		engine.ScreenTarget(ctx, target)

		err := engine.BuildFinancialModel(ctx, target.ID)
		if err != nil {
			t.Fatalf("BuildFinancialModel failed: %v", err)
		}

		// Verify financial model was built
		retrieved, _ := engine.GetTarget(target.ID)
		if len(retrieved.FinancialModel.BaseCase.Revenue) == 0 {
			t.Error("No revenue projections in base case")
		}

		// Verify synergy model
		if len(retrieved.FinancialModel.Synergies.TotalSynergies) == 0 {
			t.Error("No synergy projections")
		}

		// Verify ROI metrics
		if retrieved.FinancialModel.ROIAnalysis.NPV == 0 {
			t.Error("NPV not calculated")
		}
	})

	t.Run("CreateIntegrationPlan", func(t *testing.T) {
		target := &ma.AcquisitionTarget{
			ID:       "target-4",
			Name:     "AIMLPlatform Inc",
			Category: ma.CategoryAIML,
			Revenue:  100.0,
			ValuationRange: ma.ValuationRange{
				Mid: 500.0,
			},
		}

		engine.ScreenTarget(ctx, target)

		err := engine.CreateIntegrationPlan(ctx, target.ID)
		if err != nil {
			t.Fatalf("CreateIntegrationPlan failed: %v", err)
		}

		// Verify integration plan was created
		retrieved, _ := engine.GetTarget(target.ID)
		if retrieved.IntegrationPlan.Status != "draft" {
			t.Errorf("Integration plan status: got %s, want draft", retrieved.IntegrationPlan.Status)
		}

		// Verify phases were defined
		if len(retrieved.IntegrationPlan.Phases) == 0 {
			t.Error("No integration phases defined")
		}

		// Verify workstreams were defined
		if len(retrieved.IntegrationPlan.Workstreams) == 0 {
			t.Error("No integration workstreams defined")
		}
	})

	t.Run("CloseDeal", func(t *testing.T) {
		target := &ma.AcquisitionTarget{
			ID:       "target-5",
			Name:     "QuantumSoft Inc",
			Category: ma.CategoryQuantum,
		}

		engine.ScreenTarget(ctx, target)

		closeDate := time.Now().AddDate(0, 3, 0) // 3 months from now
		err := engine.CloseDeal(ctx, target.ID, closeDate)
		if err != nil {
			t.Fatalf("CloseDeal failed: %v", err)
		}

		// Verify deal was closed
		retrieved, _ := engine.GetTarget(target.ID)
		if retrieved.Stage != ma.StageClosed {
			t.Errorf("Deal stage: got %s, want %s", retrieved.Stage, ma.StageClosed)
		}
	})

	t.Run("ListTargets", func(t *testing.T) {
		// List all storage targets
		targets := engine.ListTargets("", ma.CategoryStorage)
		if len(targets) == 0 {
			t.Error("No storage targets found")
		}

		// List all targets in screening stage
		screeningTargets := engine.ListTargets(ma.StageScreening, "")
		if len(screeningTargets) == 0 {
			t.Error("No screening stage targets found")
		}
	})

	t.Run("GetMetrics", func(t *testing.T) {
		metrics := engine.GetMetrics()

		if metrics.TotalTargets == 0 {
			t.Error("No targets tracked in metrics")
		}

		if len(metrics.ByStage) == 0 {
			t.Error("No stage breakdown in metrics")
		}

		if len(metrics.ByCategory) == 0 {
			t.Error("No category breakdown in metrics")
		}
	})

	t.Run("ExportToJSON", func(t *testing.T) {
		data, err := engine.ExportToJSON()
		if err != nil {
			t.Fatalf("ExportToJSON failed: %v", err)
		}

		if len(data) == 0 {
			t.Error("Exported JSON is empty")
		}
	})
}

func TestStrategicFitCalculation(t *testing.T) {
	engine := ma.NewEvaluationEngine()
	ctx := context.Background()

	testCases := []struct {
		name           string
		target         *ma.AcquisitionTarget
		expectedScore  float64
		expectSynergies bool
	}{
		{
			name: "High Technology Fit",
			target: &ma.AcquisitionTarget{
				ID:       "test-1",
				Name:     "HighTech Inc",
				Category: ma.CategorySecurity,
				Revenue:  60.0,
				Growth:   0.40,
				Customers: 1500,
				Technology: ma.TechnologyAssets{
					Patents:       100,
					Developers:    200,
					SecurityCerts: []string{"SOC2", "ISO27001", "FedRAMP"},
					AIModels:      10,
				},
			},
			expectedScore:  75.0,
			expectSynergies: true,
		},
		{
			name: "High Market Fit",
			target: &ma.AcquisitionTarget{
				ID:        "test-2",
				Name:      "MarketLeader Inc",
				Category:  ma.CategoryNetworking,
				Revenue:   100.0,
				Growth:    0.50,
				Customers: 2000,
			},
			expectedScore:  70.0,
			expectSynergies: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := engine.ScreenTarget(ctx, tc.target)
			if err != nil {
				t.Fatalf("ScreenTarget failed: %v", err)
			}

			retrieved, _ := engine.GetTarget(tc.target.ID)

			// Verify strategic fit score is reasonable
			if retrieved.StrategicFit.Overall < tc.expectedScore-20 || retrieved.StrategicFit.Overall > tc.expectedScore+20 {
				t.Errorf("Strategic fit score %f outside expected range %f±20", retrieved.StrategicFit.Overall, tc.expectedScore)
			}

			// Verify synergies
			if tc.expectSynergies && len(retrieved.StrategicFit.Synergies) == 0 {
				t.Error("Expected synergies but none were identified")
			}
		})
	}
}

func TestValuationCalculations(t *testing.T) {
	engine := ma.NewEvaluationEngine()
	ctx := context.Background()

	target := &ma.AcquisitionTarget{
		ID:       "valuation-test",
		Name:     "ValuationTest Inc",
		Category: ma.CategoryAIML,
		Revenue:  100.0,
		Growth:   0.35,
		ValuationRange: ma.ValuationRange{
			Low:      500.0,
			Mid:      700.0,
			High:     900.0,
			Multiple: 7.0,
		},
	}

	engine.ScreenTarget(ctx, target)
	err := engine.BuildFinancialModel(ctx, target.ID)
	if err != nil {
		t.Fatalf("BuildFinancialModel failed: %v", err)
	}

	retrieved, _ := engine.GetTarget(target.ID)
	model := retrieved.FinancialModel

	t.Run("Projection Scenarios", func(t *testing.T) {
		// Verify base case
		if len(model.BaseCase.Revenue) != 5 {
			t.Errorf("Expected 5 years of projections, got %d", len(model.BaseCase.Revenue))
		}

		// Verify revenue growth
		for i := 1; i < len(model.BaseCase.Revenue); i++ {
			if model.BaseCase.Revenue[i] <= model.BaseCase.Revenue[i-1] {
				t.Error("Revenue should be growing year over year")
			}
		}

		// Verify bull case is higher than base case
		if model.BullCase.Revenue[0] <= model.BaseCase.Revenue[0] {
			t.Error("Bull case should be higher than base case")
		}

		// Verify bear case is lower than base case
		if model.BearCase.Revenue[0] >= model.BaseCase.Revenue[0] {
			t.Error("Bear case should be lower than base case")
		}
	})

	t.Run("Synergy Projections", func(t *testing.T) {
		// Verify synergies ramp up over time
		for i := 1; i < len(model.Synergies.TotalSynergies); i++ {
			if model.Synergies.TotalSynergies[i] < model.Synergies.TotalSynergies[i-1] {
				t.Error("Synergies should ramp up over time")
			}
		}
	})

	t.Run("Valuation Methods", func(t *testing.T) {
		// Verify DCF valuation was calculated
		if model.Valuation.DCFValuation == 0 {
			t.Error("DCF valuation not calculated")
		}

		// Verify comps valuation
		expectedComps := target.Revenue * 8.0 // 8x revenue multiple
		if model.Valuation.CompsValuation != expectedComps {
			t.Errorf("Comps valuation: got %f, want %f", model.Valuation.CompsValuation, expectedComps)
		}
	})

	t.Run("ROI Metrics", func(t *testing.T) {
		// Verify NPV is positive (for a good acquisition)
		if model.ROIAnalysis.NPV < 0 {
			t.Error("NPV should be positive for good acquisition")
		}

		// Verify IRR is reasonable
		if model.ROIAnalysis.IRR < 0 || model.ROIAnalysis.IRR > 100 {
			t.Errorf("IRR %f is unreasonable", model.ROIAnalysis.IRR)
		}

		// Verify payback period is reasonable
		if model.ROIAnalysis.PaybackPeriod < 0 || model.ROIAnalysis.PaybackPeriod > 10 {
			t.Errorf("Payback period %f years is unreasonable", model.ROIAnalysis.PaybackPeriod)
		}
	})
}

func TestIntegrationPlanning(t *testing.T) {
	engine := ma.NewEvaluationEngine()
	ctx := context.Background()

	target := &ma.AcquisitionTarget{
		ID:       "integration-test",
		Name:     "IntegrationTest Inc",
		Category: ma.CategoryStorage,
		Revenue:  75.0,
		ValuationRange: ma.ValuationRange{
			Mid: 400.0,
		},
	}

	engine.ScreenTarget(ctx, target)
	err := engine.CreateIntegrationPlan(ctx, target.ID)
	if err != nil {
		t.Fatalf("CreateIntegrationPlan failed: %v", err)
	}

	retrieved, _ := engine.GetTarget(target.ID)
	plan := retrieved.IntegrationPlan

	t.Run("Integration Phases", func(t *testing.T) {
		if len(plan.Phases) < 3 {
			t.Errorf("Expected at least 3 phases, got %d", len(plan.Phases))
		}

		// Verify phases are sequential
		for i := 1; i < len(plan.Phases); i++ {
			if !plan.Phases[i].StartDate.After(plan.Phases[i-1].EndDate) {
				t.Error("Phases should be sequential")
			}
		}
	})

	t.Run("Integration Workstreams", func(t *testing.T) {
		if len(plan.Workstreams) < 5 {
			t.Errorf("Expected at least 5 workstreams, got %d", len(plan.Workstreams))
		}

		// Verify each workstream has owner
		for _, ws := range plan.Workstreams {
			if ws.Owner == "" {
				t.Error("Workstream missing owner")
			}
		}
	})

	t.Run("Milestones", func(t *testing.T) {
		if len(plan.Milestones) == 0 {
			t.Error("No milestones defined")
		}

		// Verify critical milestones exist
		hasDayOne := false
		for _, m := range plan.Milestones {
			if m.Name == "Legal Close" {
				hasDayOne = true
			}
		}
		if !hasDayOne {
			t.Error("Missing Day 1 milestone")
		}
	})

	t.Run("Resource Planning", func(t *testing.T) {
		if plan.Resources.Budget == 0 {
			t.Error("Integration budget not set")
		}

		// Verify budget is reasonable (5-10% of deal value)
		expectedBudget := target.ValuationRange.Mid * 0.075
		if plan.Resources.Budget != expectedBudget {
			t.Errorf("Budget: got %f, want %f", plan.Resources.Budget, expectedBudget)
		}
	})
}

func BenchmarkScreenTarget(b *testing.B) {
	engine := ma.NewEvaluationEngine()
	ctx := context.Background()

	target := &ma.AcquisitionTarget{
		ID:       "benchmark-target",
		Name:     "Benchmark Inc",
		Category: ma.CategorySecurity,
		Revenue:  100.0,
		Growth:   0.40,
		Customers: 1500,
		Technology: ma.TechnologyAssets{
			Patents:    80,
			Developers: 150,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.ID = fmt.Sprintf("target-%d", i)
		engine.ScreenTarget(ctx, target)
	}
}
