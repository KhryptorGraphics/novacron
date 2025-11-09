package conflict

import (
	"context"
	"testing"
	"time"
)

func TestVectorClockCompare(t *testing.T) {
	vc1 := NewVectorClock()
	vc1.Increment("node1")
	vc1.Increment("node1")

	vc2 := NewVectorClock()
	vc2.Increment("node1")

	relation := vc1.Compare(vc2)
	if relation != RelationAfter {
		t.Errorf("Expected RelationAfter, got %v", relation)
	}

	relation = vc2.Compare(vc1)
	if relation != RelationBefore {
		t.Errorf("Expected RelationBefore, got %v", relation)
	}

	vc3 := NewVectorClock()
	vc3.Increment("node1")
	vc3.Increment("node1")

	relation = vc1.Compare(vc3)
	if relation != RelationEqual {
		t.Errorf("Expected RelationEqual, got %v", relation)
	}
}

func TestVectorClockConcurrent(t *testing.T) {
	vc1 := NewVectorClock()
	vc1.Increment("node1")

	vc2 := NewVectorClock()
	vc2.Increment("node2")

	relation := vc1.Compare(vc2)
	if relation != RelationConcurrent {
		t.Errorf("Expected RelationConcurrent, got %v", relation)
	}
}

func TestConflictDetection(t *testing.T) {
	config := DefaultDetectorConfig()
	detector := NewConflictDetector(config)

	// Create concurrent versions
	vc1 := NewVectorClock()
	vc1.Increment("node1")

	vc2 := NewVectorClock()
	vc2.Increment("node2")

	local := &Version{
		VectorClock: vc1,
		Timestamp:   time.Now(),
		NodeID:      "node1",
		Data:        "value1",
	}

	remote := &Version{
		VectorClock: vc2,
		Timestamp:   time.Now(),
		NodeID:      "node2",
		Data:        "value2",
	}

	ctx := context.Background()
	conflict, err := detector.DetectConflict(ctx, "resource1", local, remote)

	if err != nil {
		t.Fatalf("Failed to detect conflict: %v", err)
	}

	if conflict == nil {
		t.Fatal("Expected conflict to be detected")
	}

	if conflict.Type != ConflictTypeConcurrentUpdate {
		t.Errorf("Expected ConcurrentUpdate, got %v", conflict.Type)
	}
}

func TestNoConflictDetection(t *testing.T) {
	config := DefaultDetectorConfig()
	detector := NewConflictDetector(config)

	// Create causally ordered versions
	vc1 := NewVectorClock()
	vc1.Increment("node1")

	vc2 := NewVectorClock()
	vc2.Update(vc1)
	vc2.Increment("node2")

	local := &Version{
		VectorClock: vc1,
		Timestamp:   time.Now(),
		NodeID:      "node1",
		Data:        "value1",
	}

	remote := &Version{
		VectorClock: vc2,
		Timestamp:   time.Now().Add(1 * time.Second),
		NodeID:      "node2",
		Data:        "value2",
	}

	ctx := context.Background()
	conflict, err := detector.DetectConflict(ctx, "resource1", local, remote)

	if err != nil {
		t.Fatalf("Failed to detect conflict: %v", err)
	}

	if conflict != nil {
		t.Fatal("Expected no conflict")
	}
}

func TestConflictComplexityCalculation(t *testing.T) {
	calc := &DefaultComplexityCalculator{}

	conflict := &Conflict{
		Type:           ConflictTypeConcurrentUpdate,
		Severity:       SeverityLow,
		AffectedFields: []string{"field1"},
	}

	score := calc.Calculate(conflict)
	if score < 0 || score > 1 {
		t.Errorf("Invalid complexity score: %f", score)
	}

	conflict.Severity = SeverityCritical
	conflict.AffectedFields = []string{"f1", "f2", "f3", "f4"}

	score = calc.Calculate(conflict)
	if score < 0.8 {
		t.Errorf("Expected high complexity score for critical conflict: %f", score)
	}
}

func TestConflictCleanup(t *testing.T) {
	config := DefaultDetectorConfig()
	config.MaxConflictAge = 100 * time.Millisecond
	detector := NewConflictDetector(config)

	// Create old conflict
	vc := NewVectorClock()
	local := &Version{VectorClock: vc, NodeID: "n1", Timestamp: time.Now()}
	remote := &Version{VectorClock: vc, NodeID: "n2", Timestamp: time.Now()}

	ctx := context.Background()
	conflict, _ := detector.DetectConflict(ctx, "res1", local, remote)

	if conflict == nil {
		t.Fatal("Expected conflict")
	}

	// Wait for age limit
	time.Sleep(150 * time.Millisecond)

	detector.CleanupOldConflicts()

	_, exists := detector.GetConflict(conflict.ID)
	if exists {
		t.Error("Old conflict should have been cleaned up")
	}
}

func BenchmarkConflictDetection(b *testing.B) {
	config := DefaultDetectorConfig()
	detector := NewConflictDetector(config)

	vc1 := NewVectorClock()
	vc1.Increment("node1")
	vc2 := NewVectorClock()
	vc2.Increment("node2")

	local := &Version{VectorClock: vc1, NodeID: "n1", Timestamp: time.Now()}
	remote := &Version{VectorClock: vc2, NodeID: "n2", Timestamp: time.Now()}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.DetectConflict(ctx, "resource", local, remote)
	}
}

func BenchmarkVectorClockCompare(b *testing.B) {
	vc1 := NewVectorClock()
	vc1.Increment("node1")
	vc1.Increment("node2")
	vc1.Increment("node3")

	vc2 := NewVectorClock()
	vc2.Increment("node1")
	vc2.Increment("node2")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vc1.Compare(vc2)
	}
}
