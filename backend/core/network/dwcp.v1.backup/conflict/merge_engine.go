package conflict

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	mergesPerformed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_merges_performed_total",
		Help: "Total number of merges performed",
	}, []string{"type", "result"})

	mergeLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dwcp_merge_latency_ms",
		Help:    "Merge operation latency in milliseconds",
		Buckets: []float64{1, 5, 10, 50, 100, 500},
	})
)

// MergeEngine performs three-way merges
type MergeEngine struct {
	mu              sync.RWMutex
	mergers         map[reflect.Type]TypeMerger
	invariantChecks []InvariantCheck
	config          MergeConfig
}

// MergeConfig configures merge behavior
type MergeConfig struct {
	EnableStructuralDiff   bool
	EnableInvariantChecks  bool
	EnableTypeAwareMerging bool
	MaxMergeDepth          int
	ConflictOnAmbiguity    bool
}

// DefaultMergeConfig returns default merge configuration
func DefaultMergeConfig() MergeConfig {
	return MergeConfig{
		EnableStructuralDiff:   true,
		EnableInvariantChecks:  true,
		EnableTypeAwareMerging: true,
		MaxMergeDepth:          10,
		ConflictOnAmbiguity:    false,
	}
}

// TypeMerger handles merging for specific types
type TypeMerger interface {
	CanMerge(base, local, remote interface{}) bool
	Merge(base, local, remote interface{}) (interface{}, error)
}

// InvariantCheck validates invariants after merge
type InvariantCheck func(merged interface{}) error

// NewMergeEngine creates a new merge engine
func NewMergeEngine(config MergeConfig) *MergeEngine {
	me := &MergeEngine{
		mergers:         make(map[reflect.Type]TypeMerger),
		invariantChecks: make([]InvariantCheck, 0),
		config:          config,
	}

	// Register default type mergers
	me.RegisterMerger(reflect.TypeOf(map[string]interface{}{}), &MapMerger{})
	me.RegisterMerger(reflect.TypeOf([]interface{}{}), &SliceMerger{})
	me.RegisterMerger(reflect.TypeOf(""), &StringMerger{})
	me.RegisterMerger(reflect.TypeOf(0), &IntMerger{})

	return me
}

// RegisterMerger registers a type-specific merger
func (me *MergeEngine) RegisterMerger(t reflect.Type, merger TypeMerger) {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.mergers[t] = merger
}

// RegisterInvariantCheck registers an invariant check
func (me *MergeEngine) RegisterInvariantCheck(check InvariantCheck) {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.invariantChecks = append(me.invariantChecks, check)
}

// ThreeWayMerge performs a three-way merge
func (me *MergeEngine) ThreeWayMerge(ctx context.Context, base, local, remote interface{}) (interface{}, error) {
	start := time.Now()
	defer func() {
		mergeLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	// If base equals local, use remote
	if reflect.DeepEqual(base, local) {
		mergesPerformed.WithLabelValues("base_equals_local", "success").Inc()
		return remote, nil
	}

	// If base equals remote, use local
	if reflect.DeepEqual(base, remote) {
		mergesPerformed.WithLabelValues("base_equals_remote", "success").Inc()
		return local, nil
	}

	// If local equals remote, no conflict
	if reflect.DeepEqual(local, remote) {
		mergesPerformed.WithLabelValues("local_equals_remote", "success").Inc()
		return local, nil
	}

	// Perform type-aware merge
	if me.config.EnableTypeAwareMerging {
		merged, err := me.typeAwareMerge(base, local, remote, 0)
		if err != nil {
			mergesPerformed.WithLabelValues("type_aware", "failure").Inc()
			return nil, err
		}

		// Validate invariants
		if me.config.EnableInvariantChecks {
			if err := me.validateInvariants(merged); err != nil {
				mergesPerformed.WithLabelValues("invariant_check", "failure").Inc()
				return nil, fmt.Errorf("invariant violation: %w", err)
			}
		}

		mergesPerformed.WithLabelValues("type_aware", "success").Inc()
		return merged, nil
	}

	// Default: return local (could be configurable)
	mergesPerformed.WithLabelValues("default", "success").Inc()
	return local, nil
}

// typeAwareMerge performs type-specific merging
func (me *MergeEngine) typeAwareMerge(base, local, remote interface{}, depth int) (interface{}, error) {
	if depth > me.config.MaxMergeDepth {
		return nil, fmt.Errorf("max merge depth exceeded")
	}

	if base == nil || local == nil || remote == nil {
		// Handle nil values
		if local != nil {
			return local, nil
		}
		return remote, nil
	}

	baseType := reflect.TypeOf(base)
	localType := reflect.TypeOf(local)
	remoteType := reflect.TypeOf(remote)

	// Types must match
	if baseType != localType || baseType != remoteType {
		return nil, fmt.Errorf("type mismatch in merge")
	}

	me.mu.RLock()
	merger, exists := me.mergers[baseType]
	me.mu.RUnlock()

	if exists && merger.CanMerge(base, local, remote) {
		return merger.Merge(base, local, remote)
	}

	// Default structural merge
	return me.structuralMerge(base, local, remote, depth)
}

// structuralMerge performs structural merging for complex types
func (me *MergeEngine) structuralMerge(base, local, remote interface{}, depth int) (interface{}, error) {
	baseVal := reflect.ValueOf(base)
	localVal := reflect.ValueOf(local)
	remoteVal := reflect.ValueOf(remote)

	switch baseVal.Kind() {
	case reflect.Map:
		return me.mergeMap(baseVal, localVal, remoteVal, depth)
	case reflect.Slice:
		return me.mergeSlice(baseVal, localVal, remoteVal, depth)
	case reflect.Struct:
		return me.mergeStruct(baseVal, localVal, remoteVal, depth)
	default:
		// For primitive types, prefer local
		return local, nil
	}
}

// mergeMap merges map values
func (me *MergeEngine) mergeMap(base, local, remote reflect.Value, depth int) (interface{}, error) {
	result := reflect.MakeMap(base.Type())

	// Collect all keys
	allKeys := make(map[interface{}]bool)
	for _, key := range base.MapKeys() {
		allKeys[key.Interface()] = true
	}
	for _, key := range local.MapKeys() {
		allKeys[key.Interface()] = true
	}
	for _, key := range remote.MapKeys() {
		allKeys[key.Interface()] = true
	}

	// Merge each key
	for keyIface := range allKeys {
		key := reflect.ValueOf(keyIface)

		baseVal := base.MapIndex(key)
		localVal := local.MapIndex(key)
		remoteVal := remote.MapIndex(key)

		var mergedVal interface{}
		var err error

		if !baseVal.IsValid() {
			// Key added in local or remote
			if localVal.IsValid() && remoteVal.IsValid() {
				// Added in both, merge values
				mergedVal, err = me.typeAwareMerge(nil, localVal.Interface(), remoteVal.Interface(), depth+1)
			} else if localVal.IsValid() {
				mergedVal = localVal.Interface()
			} else {
				mergedVal = remoteVal.Interface()
			}
		} else if !localVal.IsValid() && !remoteVal.IsValid() {
			// Key deleted in both, skip
			continue
		} else if !localVal.IsValid() {
			// Deleted in local
			continue
		} else if !remoteVal.IsValid() {
			// Deleted in remote
			continue
		} else {
			// Key exists in all, merge values
			mergedVal, err = me.typeAwareMerge(baseVal.Interface(), localVal.Interface(), remoteVal.Interface(), depth+1)
		}

		if err != nil {
			return nil, err
		}

		if mergedVal != nil {
			result.SetMapIndex(key, reflect.ValueOf(mergedVal))
		}
	}

	return result.Interface(), nil
}

// mergeSlice merges slice values
func (me *MergeEngine) mergeSlice(base, local, remote reflect.Value, depth int) (interface{}, error) {
	// Simple implementation: prefer longer slice
	if local.Len() >= remote.Len() {
		return local.Interface(), nil
	}
	return remote.Interface(), nil
}

// mergeStruct merges struct values
func (me *MergeEngine) mergeStruct(base, local, remote reflect.Value, depth int) (interface{}, error) {
	result := reflect.New(base.Type()).Elem()

	for i := 0; i < base.NumField(); i++ {
		baseField := base.Field(i)
		localField := local.Field(i)
		remoteField := remote.Field(i)

		if !baseField.CanInterface() {
			continue
		}

		mergedField, err := me.typeAwareMerge(baseField.Interface(), localField.Interface(), remoteField.Interface(), depth+1)
		if err != nil {
			return nil, err
		}

		if result.Field(i).CanSet() {
			result.Field(i).Set(reflect.ValueOf(mergedField))
		}
	}

	return result.Interface(), nil
}

// validateInvariants validates all registered invariants
func (me *MergeEngine) validateInvariants(merged interface{}) error {
	me.mu.RLock()
	defer me.mu.RUnlock()

	for _, check := range me.invariantChecks {
		if err := check(merged); err != nil {
			return err
		}
	}
	return nil
}

// ComputeDiff computes structural diff between two values
func (me *MergeEngine) ComputeDiff(left, right interface{}) (*StructuralDiff, error) {
	diff := &StructuralDiff{
		Changes: make([]DiffChange, 0),
	}

	me.computeDiffRecursive(left, right, "", diff)
	return diff, nil
}

// StructuralDiff represents differences between two structures
type StructuralDiff struct {
	Changes []DiffChange
}

// DiffChange represents a single change
type DiffChange struct {
	Path      string
	ChangeType string // "added", "removed", "modified"
	OldValue  interface{}
	NewValue  interface{}
}

func (me *MergeEngine) computeDiffRecursive(left, right interface{}, path string, diff *StructuralDiff) {
	if reflect.DeepEqual(left, right) {
		return
	}

	if left == nil && right != nil {
		diff.Changes = append(diff.Changes, DiffChange{
			Path:       path,
			ChangeType: "added",
			NewValue:   right,
		})
		return
	}

	if left != nil && right == nil {
		diff.Changes = append(diff.Changes, DiffChange{
			Path:       path,
			ChangeType: "removed",
			OldValue:   left,
		})
		return
	}

	if left == nil || right == nil {
		return
	}

	leftVal := reflect.ValueOf(left)
	rightVal := reflect.ValueOf(right)

	if leftVal.Type() != rightVal.Type() {
		diff.Changes = append(diff.Changes, DiffChange{
			Path:       path,
			ChangeType: "modified",
			OldValue:   left,
			NewValue:   right,
		})
		return
	}

	switch leftVal.Kind() {
	case reflect.Map:
		me.diffMap(leftVal, rightVal, path, diff)
	case reflect.Slice:
		me.diffSlice(leftVal, rightVal, path, diff)
	case reflect.Struct:
		me.diffStruct(leftVal, rightVal, path, diff)
	default:
		if !reflect.DeepEqual(left, right) {
			diff.Changes = append(diff.Changes, DiffChange{
				Path:       path,
				ChangeType: "modified",
				OldValue:   left,
				NewValue:   right,
			})
		}
	}
}

func (me *MergeEngine) diffMap(left, right reflect.Value, path string, diff *StructuralDiff) {
	allKeys := make(map[interface{}]bool)
	for _, key := range left.MapKeys() {
		allKeys[key.Interface()] = true
	}
	for _, key := range right.MapKeys() {
		allKeys[key.Interface()] = true
	}

	for keyIface := range allKeys {
		key := reflect.ValueOf(keyIface)
		keyPath := fmt.Sprintf("%s[%v]", path, keyIface)

		leftVal := left.MapIndex(key)
		rightVal := right.MapIndex(key)

		if !leftVal.IsValid() {
			diff.Changes = append(diff.Changes, DiffChange{
				Path:       keyPath,
				ChangeType: "added",
				NewValue:   rightVal.Interface(),
			})
		} else if !rightVal.IsValid() {
			diff.Changes = append(diff.Changes, DiffChange{
				Path:       keyPath,
				ChangeType: "removed",
				OldValue:   leftVal.Interface(),
			})
		} else {
			me.computeDiffRecursive(leftVal.Interface(), rightVal.Interface(), keyPath, diff)
		}
	}
}

func (me *MergeEngine) diffSlice(left, right reflect.Value, path string, diff *StructuralDiff) {
	minLen := left.Len()
	if right.Len() < minLen {
		minLen = right.Len()
	}

	for i := 0; i < minLen; i++ {
		itemPath := fmt.Sprintf("%s[%d]", path, i)
		me.computeDiffRecursive(left.Index(i).Interface(), right.Index(i).Interface(), itemPath, diff)
	}

	if left.Len() > right.Len() {
		for i := minLen; i < left.Len(); i++ {
			diff.Changes = append(diff.Changes, DiffChange{
				Path:       fmt.Sprintf("%s[%d]", path, i),
				ChangeType: "removed",
				OldValue:   left.Index(i).Interface(),
			})
		}
	} else if right.Len() > left.Len() {
		for i := minLen; i < right.Len(); i++ {
			diff.Changes = append(diff.Changes, DiffChange{
				Path:       fmt.Sprintf("%s[%d]", path, i),
				ChangeType: "added",
				NewValue:   right.Index(i).Interface(),
			})
		}
	}
}

func (me *MergeEngine) diffStruct(left, right reflect.Value, path string, diff *StructuralDiff) {
	for i := 0; i < left.NumField(); i++ {
		if !left.Field(i).CanInterface() {
			continue
		}

		fieldName := left.Type().Field(i).Name
		fieldPath := path + "." + fieldName

		me.computeDiffRecursive(left.Field(i).Interface(), right.Field(i).Interface(), fieldPath, diff)
	}
}

// MapMerger merges map types
type MapMerger struct{}

func (m *MapMerger) CanMerge(base, local, remote interface{}) bool {
	return reflect.TypeOf(base).Kind() == reflect.Map
}

func (m *MapMerger) Merge(base, local, remote interface{}) (interface{}, error) {
	// Use MergeEngine's map merge logic
	return local, nil // Simplified
}

// SliceMerger merges slice types
type SliceMerger struct{}

func (m *SliceMerger) CanMerge(base, local, remote interface{}) bool {
	return reflect.TypeOf(base).Kind() == reflect.Slice
}

func (m *SliceMerger) Merge(base, local, remote interface{}) (interface{}, error) {
	// Prefer longer slice
	localSlice := reflect.ValueOf(local)
	remoteSlice := reflect.ValueOf(remote)

	if localSlice.Len() >= remoteSlice.Len() {
		return local, nil
	}
	return remote, nil
}

// StringMerger merges string types
type StringMerger struct{}

func (m *StringMerger) CanMerge(base, local, remote interface{}) bool {
	return reflect.TypeOf(base).Kind() == reflect.String
}

func (m *StringMerger) Merge(base, local, remote interface{}) (interface{}, error) {
	// If strings are different, concatenate
	baseStr := base.(string)
	localStr := local.(string)
	remoteStr := remote.(string)

	if baseStr == localStr {
		return remoteStr, nil
	}
	if baseStr == remoteStr {
		return localStr, nil
	}

	// Both changed, prefer local
	return localStr, nil
}

// IntMerger merges integer types
type IntMerger struct{}

func (m *IntMerger) CanMerge(base, local, remote interface{}) bool {
	return reflect.TypeOf(base).Kind() == reflect.Int
}

func (m *IntMerger) Merge(base, local, remote interface{}) (interface{}, error) {
	baseInt := base.(int)
	localInt := local.(int)
	remoteInt := remote.(int)

	if baseInt == localInt {
		return remoteInt, nil
	}
	if baseInt == remoteInt {
		return localInt, nil
	}

	// Both changed, use max
	if localInt > remoteInt {
		return localInt, nil
	}
	return remoteInt, nil
}
