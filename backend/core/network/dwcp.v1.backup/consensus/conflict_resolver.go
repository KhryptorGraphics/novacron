package consensus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sync"
)

// ConflictResolver handles conflict resolution for concurrent writes
type ConflictResolver struct {
	mu       sync.RWMutex
	strategy ResolutionStrategy

	// Custom resolver function
	customResolver func([]ConflictingWrite) []byte

	// Conflict history for learning
	conflictHistory []ConflictResolution
	maxHistorySize  int
}

// ConflictResolution records a conflict resolution event
type ConflictResolution struct {
	Conflicts      []ConflictingWrite
	ResolvedValue  []byte
	Strategy       ResolutionStrategy
	Timestamp      Timestamp
	ResolutionTime int64 // nanoseconds
}

// NewConflictResolver creates a new conflict resolver
func NewConflictResolver(strategy ResolutionStrategy) *ConflictResolver {
	return &ConflictResolver{
		strategy:        strategy,
		conflictHistory: make([]ConflictResolution, 0),
		maxHistorySize:  1000,
	}
}

// Resolve resolves conflicts between concurrent writes
func (cr *ConflictResolver) Resolve(conflicts []ConflictingWrite) []byte {
	if len(conflicts) == 0 {
		return nil
	}

	if len(conflicts) == 1 {
		return conflicts[0].Value
	}

	startTime := NewTimestamp()

	var resolvedValue []byte

	cr.mu.RLock()
	strategy := cr.strategy
	cr.mu.RUnlock()

	switch strategy {
	case StrategyLWW:
		resolvedValue = cr.lastWriteWins(conflicts)
	case StrategyMV:
		resolvedValue = cr.multiValue(conflicts)
	case StrategyCustom:
		resolvedValue = cr.custom(conflicts)
	default:
		resolvedValue = cr.lastWriteWins(conflicts)
	}

	// Record resolution
	endTime := NewTimestamp()
	resolutionTime := endTime.Wall - startTime.Wall

	cr.mu.Lock()
	cr.conflictHistory = append(cr.conflictHistory, ConflictResolution{
		Conflicts:      conflicts,
		ResolvedValue:  resolvedValue,
		Strategy:       strategy,
		Timestamp:      endTime,
		ResolutionTime: resolutionTime,
	})

	if len(cr.conflictHistory) > cr.maxHistorySize {
		cr.conflictHistory = cr.conflictHistory[1:]
	}
	cr.mu.Unlock()

	return resolvedValue
}

// lastWriteWins resolves conflicts by choosing the write with latest timestamp
func (cr *ConflictResolver) lastWriteWins(conflicts []ConflictingWrite) []byte {
	if len(conflicts) == 0 {
		return nil
	}

	latest := conflicts[0]
	for _, conflict := range conflicts[1:] {
		if conflict.Timestamp.After(latest.Timestamp) {
			latest = conflict
		}
	}

	return latest.Value
}

// multiValue preserves all conflicting values
func (cr *ConflictResolver) multiValue(conflicts []ConflictingWrite) []byte {
	// Create a multi-value structure
	type MultiValue struct {
		Values []struct {
			Value     []byte    `json:"value"`
			Timestamp Timestamp `json:"timestamp"`
			NodeID    string    `json:"node_id"`
			Version   uint64    `json:"version"`
		} `json:"values"`
	}

	mv := MultiValue{
		Values: make([]struct {
			Value     []byte    `json:"value"`
			Timestamp Timestamp `json:"timestamp"`
			NodeID    string    `json:"node_id"`
			Version   uint64    `json:"version"`
		}, len(conflicts)),
	}

	for i, conflict := range conflicts {
		mv.Values[i].Value = conflict.Value
		mv.Values[i].Timestamp = conflict.Timestamp
		mv.Values[i].NodeID = conflict.NodeID
		mv.Values[i].Version = conflict.Version
	}

	// Serialize to JSON
	data, err := json.Marshal(mv)
	if err != nil {
		// Fallback to LWW
		return cr.lastWriteWins(conflicts)
	}

	return data
}

// custom uses custom resolver function
func (cr *ConflictResolver) custom(conflicts []ConflictingWrite) []byte {
	cr.mu.RLock()
	resolver := cr.customResolver
	cr.mu.RUnlock()

	if resolver == nil {
		// Fallback to LWW
		return cr.lastWriteWins(conflicts)
	}

	return resolver(conflicts)
}

// SetCustomResolver sets a custom conflict resolution function
func (cr *ConflictResolver) SetCustomResolver(resolver func([]ConflictingWrite) []byte) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	cr.customResolver = resolver
}

// SetStrategy changes the resolution strategy
func (cr *ConflictResolver) SetStrategy(strategy ResolutionStrategy) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	cr.strategy = strategy
}

// GetStrategy returns the current resolution strategy
func (cr *ConflictResolver) GetStrategy() ResolutionStrategy {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	return cr.strategy
}

// DetectConflicts detects conflicts in a set of writes
func (cr *ConflictResolver) DetectConflicts(writes []ConflictingWrite) [][]ConflictingWrite {
	if len(writes) == 0 {
		return nil
	}

	// Group writes by key
	keyWrites := make(map[string][]ConflictingWrite)
	for _, write := range writes {
		keyWrites[write.Key] = append(keyWrites[write.Key], write)
	}

	// Find keys with conflicts (multiple concurrent writes)
	conflicts := make([][]ConflictingWrite, 0)
	for _, writes := range keyWrites {
		if len(writes) > 1 {
			// Check if writes are concurrent (neither happens-before the other)
			if cr.areConcurrent(writes) {
				conflicts = append(conflicts, writes)
			}
		}
	}

	return conflicts
}

// areConcurrent checks if writes are concurrent
func (cr *ConflictResolver) areConcurrent(writes []ConflictingWrite) bool {
	if len(writes) < 2 {
		return false
	}

	// Simplified: consider writes concurrent if timestamps are very close
	// In real implementation, would use vector clocks
	const concurrencyWindow = int64(1000000000) // 1 second in nanoseconds

	for i := 0; i < len(writes)-1; i++ {
		for j := i + 1; j < len(writes); j++ {
			timeDiff := writes[i].Timestamp.Wall - writes[j].Timestamp.Wall
			if timeDiff < 0 {
				timeDiff = -timeDiff
			}

			if timeDiff < concurrencyWindow {
				return true
			}
		}
	}

	return false
}

// GetConflictStats returns statistics about conflict resolution
func (cr *ConflictResolver) GetConflictStats() ConflictStats {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	stats := ConflictStats{
		TotalConflicts:     len(cr.conflictHistory),
		StrategyUsage:      make(map[string]int),
		AvgResolutionTime:  0,
		MaxConflictingKeys: 0,
	}

	if len(cr.conflictHistory) == 0 {
		return stats
	}

	var totalTime int64
	for _, resolution := range cr.conflictHistory {
		stats.StrategyUsage[resolution.Strategy.String()]++
		totalTime += resolution.ResolutionTime

		if len(resolution.Conflicts) > stats.MaxConflictingKeys {
			stats.MaxConflictingKeys = len(resolution.Conflicts)
		}
	}

	stats.AvgResolutionTime = totalTime / int64(len(cr.conflictHistory))

	return stats
}

// ConflictStats contains conflict resolution statistics
type ConflictStats struct {
	TotalConflicts     int
	StrategyUsage      map[string]int
	AvgResolutionTime  int64
	MaxConflictingKeys int
}

// MergeResolver provides semantic merge for specific data types
type MergeResolver struct {
	mergers map[string]MergeFunction
}

// MergeFunction defines a semantic merge function
type MergeFunction func(base, left, right []byte) ([]byte, error)

// NewMergeResolver creates a new merge resolver
func NewMergeResolver() *MergeResolver {
	return &MergeResolver{
		mergers: make(map[string]MergeFunction),
	}
}

// RegisterMerger registers a merge function for a data type
func (mr *MergeResolver) RegisterMerger(dataType string, merger MergeFunction) {
	mr.mergers[dataType] = merger
}

// Merge performs semantic merge for a data type
func (mr *MergeResolver) Merge(dataType string, base, left, right []byte) ([]byte, error) {
	merger, exists := mr.mergers[dataType]
	if !exists {
		return nil, fmt.Errorf("no merger registered for type: %s", dataType)
	}

	return merger(base, left, right)
}

// JSONMerger implements semantic merge for JSON objects
func JSONMerger(base, left, right []byte) ([]byte, error) {
	var baseObj, leftObj, rightObj map[string]interface{}

	if err := json.Unmarshal(base, &baseObj); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(left, &leftObj); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(right, &rightObj); err != nil {
		return nil, err
	}

	// Three-way merge
	merged := make(map[string]interface{})

	// Start with base
	for k, v := range baseObj {
		merged[k] = v
	}

	// Apply left changes
	for k, v := range leftObj {
		if baseVal, exists := baseObj[k]; !exists || !deepEqual(baseVal, v) {
			merged[k] = v
		}
	}

	// Apply right changes (may conflict)
	for k, v := range rightObj {
		if baseVal, exists := baseObj[k]; !exists || !deepEqual(baseVal, v) {
			// Check for conflict
			if leftVal, leftExists := leftObj[k]; leftExists && !deepEqual(leftVal, v) {
				// Conflict: keep both as array
				merged[k] = []interface{}{leftVal, v}
			} else {
				merged[k] = v
			}
		}
	}

	return json.Marshal(merged)
}

// deepEqual checks deep equality (simplified)
func deepEqual(a, b interface{}) bool {
	aJSON, _ := json.Marshal(a)
	bJSON, _ := json.Marshal(b)
	return bytes.Equal(aJSON, bJSON)
}
