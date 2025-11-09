package autotuning

import (
	"context"
	"testing"
	"time"
)

func TestOrchestrator(t *testing.T) {
	config := OrchestratorConfig{
		TuningInterval:       5 * time.Second,
		ConvergenceTarget:    30 * time.Second,
		MaxConcurrentTuning:  2,
		SafeTuning:           true,
		GradualChanges:       true,
		AutoRollback:         true,
		ValidationPeriod:     5 * time.Second,
		DegradationThreshold: 0.1,
	}

	orchestrator := NewOrchestrator(config)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Initialize
	if err := orchestrator.Initialize(ctx); err != nil {
		t.Fatalf("Initialization failed: %v", err)
	}

	// Start orchestration
	if err := orchestrator.Start(ctx); err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Let it run
	time.Sleep(12 * time.Second)

	// Stop
	orchestrator.Stop()

	// Check tuning history
	history := orchestrator.GetTuningHistory()
	t.Logf("Tuning history: %d events", len(history))

	// Check convergence
	if orchestrator.convergenceDetector.HasConverged() {
		t.Log("Auto-tuning converged successfully")
	}
}

func TestConvergenceDetector(t *testing.T) {
	detector := NewConvergenceDetector(5, 0.01)

	// Simulate improving performance
	improvements := []float64{0.10, 0.05, 0.02, 0.01, 0.005}

	for _, imp := range improvements {
		detector.Record(imp)
	}

	if !detector.HasConverged() {
		t.Error("Expected convergence")
	}

	// Add another significant improvement
	detector.Record(0.15)

	if detector.HasConverged() {
		t.Error("Should not be converged after significant improvement")
	}
}

func TestTuningEventRecording(t *testing.T) {
	config := OrchestratorConfig{
		TuningInterval: 1 * time.Second,
	}

	orchestrator := NewOrchestrator(config)

	// Record some tuning events
	for i := 0; i < 10; i++ {
		event := TuningEvent{
			Timestamp: time.Now(),
			Component: "cpu",
			Action:    "numa",
			Success:   true,
			Impact:    0.05,
		}
		orchestrator.recordTuningEvent(event)
	}

	history := orchestrator.GetTuningHistory()
	if len(history) != 10 {
		t.Errorf("Expected 10 events, got %d", len(history))
	}
}

func BenchmarkOrchestrator(b *testing.B) {
	config := OrchestratorConfig{
		TuningInterval:      10 * time.Minute,
		MaxConcurrentTuning: 5,
	}

	orchestrator := NewOrchestrator(config)
	ctx := context.Background()
	orchestrator.Initialize(ctx)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		orchestrator.runTuningCycle(ctx)
	}
}
