package rightsizing

import (
	"context"
	"testing"
	"time"
)

func TestRightSizingEngine(t *testing.T) {
	config := RightSizingConfig{
		CPUTargetMin:        0.60,
		CPUTargetMax:        0.80,
		MemoryTargetMin:     0.70,
		MemoryTargetMax:     0.85,
		ObservationPeriod:   1 * time.Hour,
		ConfidenceThreshold: 0.90,
		CostSavingsMin:      0.10,
	}

	engine := NewEngine(config)

	// Simulate VM with low utilization
	vmSize := VMSize{
		Name:       "m5.large",
		VCPUs:      2,
		MemoryGB:   8,
		HourlyCost: 0.10,
	}

	vmID := "test-vm-1"

	// Record observations showing underutilization
	for i := 0; i < 200; i++ {
		engine.ObserveVM(vmID, 0.25, 0.35, 1000, 50, vmSize)
		time.Sleep(1 * time.Millisecond)
	}

	// Analyze and get recommendations
	recommendations, err := engine.AnalyzeAndRecommend(context.Background())
	if err != nil {
		t.Fatalf("Analysis failed: %v", err)
	}

	if len(recommendations) == 0 {
		t.Fatal("Expected downsize recommendation")
	}

	rec := recommendations[0]
	if rec.Action != "downsize" {
		t.Errorf("Expected downsize, got %s", rec.Action)
	}

	if rec.Confidence < config.ConfidenceThreshold {
		t.Errorf("Confidence %.2f below threshold %.2f",
			rec.Confidence, config.ConfidenceThreshold)
	}

	t.Logf("Recommendation: %s, Savings: $%.2f/month, Confidence: %.2f",
		rec.Action, rec.EstimatedSavings, rec.Confidence)
}

func TestOverutilizedVM(t *testing.T) {
	config := RightSizingConfig{
		CPUTargetMin:    0.60,
		CPUTargetMax:    0.80,
		MemoryTargetMin: 0.70,
		MemoryTargetMax: 0.85,
	}

	engine := NewEngine(config)

	vmSize := VMSize{
		Name:       "t3.micro",
		VCPUs:      1,
		MemoryGB:   1,
		HourlyCost: 0.01,
	}

	vmID := "test-vm-2"

	// Record observations showing overutilization
	for i := 0; i < 200; i++ {
		engine.ObserveVM(vmID, 0.95, 0.90, 5000, 200, vmSize)
	}

	recommendations, _ := engine.AnalyzeAndRecommend(context.Background())

	if len(recommendations) > 0 {
		if recommendations[0].Action != "upsize" {
			t.Errorf("Expected upsize for overutilized VM, got %s",
				recommendations[0].Action)
		}
	}
}
