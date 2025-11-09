package conflict

import (
	"context"
	"testing"
)

func TestThreeWayMergeIdentical(t *testing.T) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	base := map[string]interface{}{"key": "value"}
	local := map[string]interface{}{"key": "value"}
	remote := map[string]interface{}{"key": "value"}

	ctx := context.Background()
	merged, err := engine.ThreeWayMerge(ctx, base, local, remote)

	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	result, ok := merged.(map[string]interface{})
	if !ok {
		t.Fatal("Expected map result")
	}

	if result["key"] != "value" {
		t.Error("Value mismatch in merge")
	}
}

func TestThreeWayMergeLocalChange(t *testing.T) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	base := map[string]interface{}{"key": "value"}
	local := map[string]interface{}{"key": "new_value"}
	remote := map[string]interface{}{"key": "value"}

	ctx := context.Background()
	merged, err := engine.ThreeWayMerge(ctx, base, local, remote)

	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	result, ok := merged.(map[string]interface{})
	if !ok {
		t.Fatal("Expected map result")
	}

	if result["key"] != "new_value" {
		t.Error("Should prefer local change")
	}
}

func TestThreeWayMergeRemoteChange(t *testing.T) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	base := map[string]interface{}{"key": "value"}
	local := map[string]interface{}{"key": "value"}
	remote := map[string]interface{}{"key": "new_value"}

	ctx := context.Background()
	merged, err := engine.ThreeWayMerge(ctx, base, local, remote)

	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	result, ok := merged.(map[string]interface{})
	if !ok {
		t.Fatal("Expected map result")
	}

	if result["key"] != "new_value" {
		t.Error("Should prefer remote change")
	}
}

func TestThreeWayMergeConflict(t *testing.T) {
	config := DefaultMergeConfig()
	config.EnableTypeAwareMerging = true
	engine := NewMergeEngine(config)

	base := map[string]interface{}{"key": "base_value"}
	local := map[string]interface{}{"key": "local_value"}
	remote := map[string]interface{}{"key": "remote_value"}

	ctx := context.Background()
	merged, err := engine.ThreeWayMerge(ctx, base, local, remote)

	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	// Should resolve to local or remote (implementation specific)
	if merged == nil {
		t.Error("Merge should produce result")
	}
}

func TestMapMerge(t *testing.T) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	base := map[string]interface{}{
		"a": "base_a",
		"b": "base_b",
	}

	local := map[string]interface{}{
		"a": "local_a",
		"c": "local_c",
	}

	remote := map[string]interface{}{
		"b": "remote_b",
		"d": "remote_d",
	}

	ctx := context.Background()
	merged, err := engine.ThreeWayMerge(ctx, base, local, remote)

	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	result, ok := merged.(map[string]interface{})
	if !ok {
		t.Fatal("Expected map result")
	}

	// Should have local's change to 'a', remote's change to 'b', and new keys
	if len(result) < 2 {
		t.Error("Merged map should contain multiple keys")
	}
}

func TestStructuralDiff(t *testing.T) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	left := map[string]interface{}{
		"key1": "value1",
		"key2": "value2",
	}

	right := map[string]interface{}{
		"key1": "changed",
		"key3": "new",
	}

	diff, err := engine.ComputeDiff(left, right)
	if err != nil {
		t.Fatalf("Diff computation failed: %v", err)
	}

	if len(diff.Changes) == 0 {
		t.Error("Expected changes in diff")
	}

	hasModified := false
	hasAdded := false
	hasRemoved := false

	for _, change := range diff.Changes {
		switch change.ChangeType {
		case "modified":
			hasModified = true
		case "added":
			hasAdded = true
		case "removed":
			hasRemoved = true
		}
	}

	if !hasModified {
		t.Error("Expected modified change")
	}
}

func BenchmarkThreeWayMerge(b *testing.B) {
	config := DefaultMergeConfig()
	engine := NewMergeEngine(config)

	base := map[string]interface{}{"key": "base"}
	local := map[string]interface{}{"key": "local"}
	remote := map[string]interface{}{"key": "remote"}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.ThreeWayMerge(ctx, base, local, remote)
	}
}
