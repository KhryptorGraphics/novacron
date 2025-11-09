package conflict

import (
	"context"
	"testing"
	"time"
)

func TestLastWriteWinsStrategy(t *testing.T) {
	strategy := &LastWriteWinsStrategy{}

	now := time.Now()
	local := &Version{
		Timestamp: now,
		NodeID:    "node1",
		Data:      "value1",
	}

	remote := &Version{
		Timestamp: now.Add(1 * time.Second),
		NodeID:    "node2",
		Data:      "value2",
	}

	conflict := &Conflict{
		LocalVersion:  local,
		RemoteVersion: remote,
	}

	ctx := context.Background()
	result, err := strategy.Resolve(ctx, conflict)

	if err != nil {
		t.Fatalf("Resolution failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected successful resolution")
	}

	if result.ResolvedData != "value2" {
		t.Errorf("Expected value2, got %v", result.ResolvedData)
	}
}

func TestMultiValueRegisterStrategy(t *testing.T) {
	strategy := &MultiValueRegisterStrategy{}

	local := &Version{NodeID: "node1", Data: "value1"}
	remote := &Version{NodeID: "node2", Data: "value2"}

	conflict := &Conflict{
		LocalVersion:  local,
		RemoteVersion: remote,
	}

	ctx := context.Background()
	result, err := strategy.Resolve(ctx, conflict)

	if err != nil {
		t.Fatalf("Resolution failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected successful resolution")
	}

	multiValue, ok := result.ResolvedData.(map[string]interface{})
	if !ok {
		t.Fatal("Expected multi-value result")
	}

	if multiValue["local"] != "value1" || multiValue["remote"] != "value2" {
		t.Error("Multi-value register should contain both values")
	}
}

func TestAutomaticRollbackStrategy(t *testing.T) {
	strategy := &AutomaticRollbackStrategy{}

	now := time.Now()
	local := &Version{
		Timestamp: now,
		NodeID:    "node1",
		Data:      "old_value",
	}

	remote := &Version{
		Timestamp: now.Add(1 * time.Second),
		NodeID:    "node2",
		Data:      "new_value",
	}

	conflict := &Conflict{
		LocalVersion:  local,
		RemoteVersion: remote,
		Severity:      SeverityHigh,
	}

	ctx := context.Background()
	result, err := strategy.Resolve(ctx, conflict)

	if err != nil {
		t.Fatalf("Resolution failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected successful resolution")
	}

	// Should rollback to older version
	if result.ResolvedData != "old_value" {
		t.Errorf("Expected rollback to old_value, got %v", result.ResolvedData)
	}
}

func TestConsensusVoteStrategy(t *testing.T) {
	strategy := NewConsensusVoteStrategy()

	local := &Version{NodeID: "node1", Data: "value1", Checksum: "cs1"}
	remote := &Version{NodeID: "node2", Data: "value2", Checksum: "cs2"}

	conflict := &Conflict{
		ID:            "conflict1",
		LocalVersion:  local,
		RemoteVersion: remote,
	}

	// Cast votes
	strategy.Vote("conflict1", "cs1")
	strategy.Vote("conflict1", "cs1")
	strategy.Vote("conflict1", "cs2")

	ctx := context.Background()
	result, err := strategy.Resolve(ctx, conflict)

	if err != nil {
		t.Fatalf("Resolution failed: %v", err)
	}

	if !result.Success {
		t.Error("Expected successful resolution")
	}

	// cs1 has more votes
	if result.ResolvedData != "value1" {
		t.Errorf("Expected value1 to win consensus, got %v", result.ResolvedData)
	}
}

func TestStrategyRegistry(t *testing.T) {
	registry := NewStrategyRegistry()

	strategy, exists := registry.GetStrategy(StrategyLastWriteWins)
	if !exists {
		t.Error("LastWriteWins strategy should be registered")
	}

	if strategy.Type() != StrategyLastWriteWins {
		t.Error("Strategy type mismatch")
	}

	// Test custom strategy registration
	customStrategy := &LastWriteWinsStrategy{}
	registry.RegisterCustom("custom_lww", customStrategy)

	custom, exists := registry.GetCustomStrategy("custom_lww")
	if !exists {
		t.Error("Custom strategy should be registered")
	}

	if custom.Name() != customStrategy.Name() {
		t.Error("Custom strategy name mismatch")
	}
}

func BenchmarkLastWriteWins(b *testing.B) {
	strategy := &LastWriteWinsStrategy{}

	local := &Version{Timestamp: time.Now(), NodeID: "n1", Data: "v1"}
	remote := &Version{Timestamp: time.Now().Add(1 * time.Second), NodeID: "n2", Data: "v2"}
	conflict := &Conflict{LocalVersion: local, RemoteVersion: remote}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		strategy.Resolve(ctx, conflict)
	}
}

func BenchmarkMultiValueRegister(b *testing.B) {
	strategy := &MultiValueRegisterStrategy{}

	local := &Version{NodeID: "n1", Data: "v1"}
	remote := &Version{NodeID: "n2", Data: "v2"}
	conflict := &Conflict{LocalVersion: local, RemoteVersion: remote}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		strategy.Resolve(ctx, conflict)
	}
}
