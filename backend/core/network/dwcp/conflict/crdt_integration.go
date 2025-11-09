package conflict

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	crdtMerges = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_crdt_merges_total",
		Help: "Total number of CRDT merge operations",
	}, []string{"crdt_type", "result"})

	crdtConflicts = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_crdt_conflicts_total",
		Help: "Total number of CRDT-specific conflicts",
	}, []string{"crdt_type", "conflict_type"})
)

// CRDTType represents different CRDT types
type CRDTType int

const (
	CRDTTypeGCounter CRDTType = iota
	CRDTTypePNCounter
	CRDTTypeGSet
	CRDTTypeORSet
	CRDTTypeLWWRegister
	CRDTTypeMVRegister
	CRDTTypeRGAList
)

func (ct CRDTType) String() string {
	return [...]string{
		"GCounter",
		"PNCounter",
		"GSet",
		"ORSet",
		"LWWRegister",
		"MVRegister",
		"RGAList",
	}[ct]
}

// CRDTIntegration bridges CRDT convergence and conflict resolution
type CRDTIntegration struct {
	mu              sync.RWMutex
	mergeEngine     *MergeEngine
	detector        *ConflictDetector
	crdtHandlers    map[CRDTType]CRDTHandler
	tombstoneGC     *TombstoneGarbageCollector
	counterOverflow *CounterOverflowHandler
}

// CRDTHandler handles CRDT-specific operations
type CRDTHandler interface {
	Type() CRDTType
	Merge(local, remote interface{}) (interface{}, error)
	DetectConflict(local, remote interface{}) (*CRDTConflict, error)
	Validate(state interface{}) error
}

// CRDTConflict represents a CRDT-specific conflict
type CRDTConflict struct {
	Type        CRDTType
	Description string
	LocalState  interface{}
	RemoteState interface{}
	CanAutoResolve bool
	Resolution  interface{}
}

// NewCRDTIntegration creates a new CRDT integration
func NewCRDTIntegration(mergeEngine *MergeEngine, detector *ConflictDetector) *CRDTIntegration {
	ci := &CRDTIntegration{
		mergeEngine:     mergeEngine,
		detector:        detector,
		crdtHandlers:    make(map[CRDTType]CRDTHandler),
		tombstoneGC:     NewTombstoneGarbageCollector(),
		counterOverflow: NewCounterOverflowHandler(),
	}

	// Register CRDT handlers
	ci.RegisterHandler(&GCounterHandler{})
	ci.RegisterHandler(&PNCounterHandler{})
	ci.RegisterHandler(&GSetHandler{})
	ci.RegisterHandler(&ORSetHandler{ci.tombstoneGC})
	ci.RegisterHandler(&LWWRegisterHandler{})
	ci.RegisterHandler(&MVRegisterHandler{})

	return ci
}

// RegisterHandler registers a CRDT handler
func (ci *CRDTIntegration) RegisterHandler(handler CRDTHandler) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.crdtHandlers[handler.Type()] = handler
}

// MergeCRDT merges CRDT states
func (ci *CRDTIntegration) MergeCRDT(ctx context.Context, crdtType CRDTType, local, remote interface{}) (interface{}, error) {
	ci.mu.RLock()
	handler, exists := ci.crdtHandlers[crdtType]
	ci.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no handler for CRDT type %s", crdtType)
	}

	// Detect CRDT-specific conflicts
	conflict, err := handler.DetectConflict(local, remote)
	if err != nil {
		crdtMerges.WithLabelValues(crdtType.String(), "error").Inc()
		return nil, err
	}

	if conflict != nil {
		crdtConflicts.WithLabelValues(crdtType.String(), conflict.Description).Inc()

		if conflict.CanAutoResolve {
			crdtMerges.WithLabelValues(crdtType.String(), "auto_resolved").Inc()
			return conflict.Resolution, nil
		}
	}

	// Perform merge
	merged, err := handler.Merge(local, remote)
	if err != nil {
		crdtMerges.WithLabelValues(crdtType.String(), "failure").Inc()
		return nil, err
	}

	// Validate merged state
	if err := handler.Validate(merged); err != nil {
		crdtMerges.WithLabelValues(crdtType.String(), "validation_failure").Inc()
		return nil, err
	}

	crdtMerges.WithLabelValues(crdtType.String(), "success").Inc()
	return merged, nil
}

// GCounterHandler handles G-Counter (Grow-only Counter)
type GCounterHandler struct{}

func (h *GCounterHandler) Type() CRDTType {
	return CRDTTypeGCounter
}

func (h *GCounterHandler) Merge(local, remote interface{}) (interface{}, error) {
	localCounter := local.(map[string]uint64)
	remoteCounter := remote.(map[string]uint64)

	merged := make(map[string]uint64)

	// Take maximum for each node
	for nodeID, count := range localCounter {
		merged[nodeID] = count
	}

	for nodeID, count := range remoteCounter {
		if merged[nodeID] < count {
			merged[nodeID] = count
		}
	}

	return merged, nil
}

func (h *GCounterHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	// G-Counter is conflict-free by design
	return nil, nil
}

func (h *GCounterHandler) Validate(state interface{}) error {
	counter, ok := state.(map[string]uint64)
	if !ok {
		return fmt.Errorf("invalid G-Counter state type")
	}

	for nodeID, count := range counter {
		if nodeID == "" {
			return fmt.Errorf("empty node ID in G-Counter")
		}
		if count < 0 {
			return fmt.Errorf("negative count in G-Counter")
		}
	}

	return nil
}

// PNCounterHandler handles PN-Counter (Positive-Negative Counter)
type PNCounterHandler struct{}

type PNCounterState struct {
	Positive map[string]uint64
	Negative map[string]uint64
}

func (h *PNCounterHandler) Type() CRDTType {
	return CRDTTypePNCounter
}

func (h *PNCounterHandler) Merge(local, remote interface{}) (interface{}, error) {
	localState := local.(PNCounterState)
	remoteState := remote.(PNCounterState)

	merged := PNCounterState{
		Positive: make(map[string]uint64),
		Negative: make(map[string]uint64),
	}

	// Merge positive counters
	for nodeID, count := range localState.Positive {
		merged.Positive[nodeID] = count
	}
	for nodeID, count := range remoteState.Positive {
		if merged.Positive[nodeID] < count {
			merged.Positive[nodeID] = count
		}
	}

	// Merge negative counters
	for nodeID, count := range localState.Negative {
		merged.Negative[nodeID] = count
	}
	for nodeID, count := range remoteState.Negative {
		if merged.Negative[nodeID] < count {
			merged.Negative[nodeID] = count
		}
	}

	return merged, nil
}

func (h *PNCounterHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	// Check for overflow
	localState := local.(PNCounterState)
	remoteState := remote.(PNCounterState)

	var totalPos, totalNeg uint64
	for _, count := range localState.Positive {
		totalPos += count
	}
	for _, count := range remoteState.Positive {
		totalPos += count
	}
	for _, count := range localState.Negative {
		totalNeg += count
	}
	for _, count := range remoteState.Negative {
		totalNeg += count
	}

	// Check for potential overflow
	const maxUint64 = ^uint64(0)
	if totalPos > maxUint64/2 || totalNeg > maxUint64/2 {
		return &CRDTConflict{
			Type:           CRDTTypePNCounter,
			Description:    "counter_overflow",
			CanAutoResolve: false,
		}, nil
	}

	return nil, nil
}

func (h *PNCounterHandler) Validate(state interface{}) error {
	_, ok := state.(PNCounterState)
	if !ok {
		return fmt.Errorf("invalid PN-Counter state type")
	}
	return nil
}

// GSetHandler handles G-Set (Grow-only Set)
type GSetHandler struct{}

func (h *GSetHandler) Type() CRDTType {
	return CRDTTypeGSet
}

func (h *GSetHandler) Merge(local, remote interface{}) (interface{}, error) {
	localSet := local.(map[string]bool)
	remoteSet := remote.(map[string]bool)

	merged := make(map[string]bool)

	// Union of sets
	for item := range localSet {
		merged[item] = true
	}
	for item := range remoteSet {
		merged[item] = true
	}

	return merged, nil
}

func (h *GSetHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	// G-Set is conflict-free
	return nil, nil
}

func (h *GSetHandler) Validate(state interface{}) error {
	_, ok := state.(map[string]bool)
	if !ok {
		return fmt.Errorf("invalid G-Set state type")
	}
	return nil
}

// ORSetHandler handles OR-Set (Observed-Remove Set)
type ORSetHandler struct {
	tombstoneGC *TombstoneGarbageCollector
}

type ORSetState struct {
	Elements   map[string]map[string]bool // element -> {unique_id: true}
	Tombstones map[string]map[string]bool // element -> {unique_id: true}
}

func (h *ORSetHandler) Type() CRDTType {
	return CRDTTypeORSet
}

func (h *ORSetHandler) Merge(local, remote interface{}) (interface{}, error) {
	localState := local.(ORSetState)
	remoteState := remote.(ORSetState)

	merged := ORSetState{
		Elements:   make(map[string]map[string]bool),
		Tombstones: make(map[string]map[string]bool),
	}

	// Merge elements
	for elem, ids := range localState.Elements {
		if merged.Elements[elem] == nil {
			merged.Elements[elem] = make(map[string]bool)
		}
		for id := range ids {
			merged.Elements[elem][id] = true
		}
	}
	for elem, ids := range remoteState.Elements {
		if merged.Elements[elem] == nil {
			merged.Elements[elem] = make(map[string]bool)
		}
		for id := range ids {
			merged.Elements[elem][id] = true
		}
	}

	// Merge tombstones
	for elem, ids := range localState.Tombstones {
		if merged.Tombstones[elem] == nil {
			merged.Tombstones[elem] = make(map[string]bool)
		}
		for id := range ids {
			merged.Tombstones[elem][id] = true
		}
	}
	for elem, ids := range remoteState.Tombstones {
		if merged.Tombstones[elem] == nil {
			merged.Tombstones[elem] = make(map[string]bool)
		}
		for id := range ids {
			merged.Tombstones[elem][id] = true
		}
	}

	// Remove tombstoned elements
	for elem, tombIds := range merged.Tombstones {
		if elemIds, exists := merged.Elements[elem]; exists {
			for id := range tombIds {
				delete(elemIds, id)
			}
			if len(elemIds) == 0 {
				delete(merged.Elements, elem)
			}
		}
	}

	// Schedule tombstone cleanup
	h.tombstoneGC.ScheduleCleanup(&merged)

	return merged, nil
}

func (h *ORSetHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	localState := local.(ORSetState)
	remoteState := remote.(ORSetState)

	// Check for excessive tombstones
	localTombCount := 0
	for _, ids := range localState.Tombstones {
		localTombCount += len(ids)
	}

	remoteTombCount := 0
	for _, ids := range remoteState.Tombstones {
		remoteTombCount += len(ids)
	}

	if localTombCount > 10000 || remoteTombCount > 10000 {
		return &CRDTConflict{
			Type:           CRDTTypeORSet,
			Description:    "excessive_tombstones",
			CanAutoResolve: true,
		}, nil
	}

	return nil, nil
}

func (h *ORSetHandler) Validate(state interface{}) error {
	_, ok := state.(ORSetState)
	if !ok {
		return fmt.Errorf("invalid OR-Set state type")
	}
	return nil
}

// LWWRegisterHandler handles LWW-Register
type LWWRegisterHandler struct{}

type LWWRegisterState struct {
	Value     interface{}
	Timestamp time.Time
	NodeID    string
}

func (h *LWWRegisterHandler) Type() CRDTType {
	return CRDTTypeLWWRegister
}

func (h *LWWRegisterHandler) Merge(local, remote interface{}) (interface{}, error) {
	localState := local.(LWWRegisterState)
	remoteState := remote.(LWWRegisterState)

	if remoteState.Timestamp.After(localState.Timestamp) {
		return remoteState, nil
	} else if localState.Timestamp.After(remoteState.Timestamp) {
		return localState, nil
	} else {
		// Timestamps equal, use node ID as tiebreaker
		if remoteState.NodeID > localState.NodeID {
			return remoteState, nil
		}
		return localState, nil
	}
}

func (h *LWWRegisterHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	// LWW-Register is conflict-free by design
	return nil, nil
}

func (h *LWWRegisterHandler) Validate(state interface{}) error {
	_, ok := state.(LWWRegisterState)
	if !ok {
		return fmt.Errorf("invalid LWW-Register state type")
	}
	return nil
}

// MVRegisterHandler handles MV-Register (Multi-Value Register)
type MVRegisterHandler struct{}

type MVRegisterState struct {
	Values map[string]ValueVersion // nodeID -> value+version
}

type ValueVersion struct {
	Value     interface{}
	Version   uint64
	Timestamp time.Time
}

func (h *MVRegisterHandler) Type() CRDTType {
	return CRDTTypeMVRegister
}

func (h *MVRegisterHandler) Merge(local, remote interface{}) (interface{}, error) {
	localState := local.(MVRegisterState)
	remoteState := remote.(MVRegisterState)

	merged := MVRegisterState{
		Values: make(map[string]ValueVersion),
	}

	// Keep all concurrent values
	for nodeID, vv := range localState.Values {
		merged.Values[nodeID] = vv
	}

	for nodeID, vv := range remoteState.Values {
		if existingVV, exists := merged.Values[nodeID]; exists {
			if vv.Version > existingVV.Version {
				merged.Values[nodeID] = vv
			}
		} else {
			merged.Values[nodeID] = vv
		}
	}

	return merged, nil
}

func (h *MVRegisterHandler) DetectConflict(local, remote interface{}) (*CRDTConflict, error) {
	localState := local.(MVRegisterState)
	remoteState := remote.(MVRegisterState)

	// Check for too many concurrent values
	totalValues := len(localState.Values) + len(remoteState.Values)
	if totalValues > 100 {
		return &CRDTConflict{
			Type:           CRDTTypeMVRegister,
			Description:    "excessive_concurrent_values",
			CanAutoResolve: false,
		}, nil
	}

	return nil, nil
}

func (h *MVRegisterHandler) Validate(state interface{}) error {
	_, ok := state.(MVRegisterState)
	if !ok {
		return fmt.Errorf("invalid MV-Register state type")
	}
	return nil
}

// TombstoneGarbageCollector manages tombstone cleanup
type TombstoneGarbageCollector struct {
	mu              sync.RWMutex
	cleanupInterval time.Duration
	retentionPeriod time.Duration
}

func NewTombstoneGarbageCollector() *TombstoneGarbageCollector {
	return &TombstoneGarbageCollector{
		cleanupInterval: 1 * time.Hour,
		retentionPeriod: 24 * time.Hour,
	}
}

func (tgc *TombstoneGarbageCollector) ScheduleCleanup(state *ORSetState) {
	// Schedule asynchronous cleanup
	go tgc.cleanup(state)
}

func (tgc *TombstoneGarbageCollector) cleanup(state *ORSetState) {
	tgc.mu.Lock()
	defer tgc.mu.Unlock()

	// Remove old tombstones (simplified - would need timestamp tracking)
	for elem, ids := range state.Tombstones {
		if len(ids) > 1000 {
			// Keep only recent tombstones
			state.Tombstones[elem] = make(map[string]bool)
		}
	}
}

// CounterOverflowHandler handles counter overflow
type CounterOverflowHandler struct {
	mu             sync.RWMutex
	overflowPolicy string // "saturate", "reset", "error"
}

func NewCounterOverflowHandler() *CounterOverflowHandler {
	return &CounterOverflowHandler{
		overflowPolicy: "saturate",
	}
}

func (coh *CounterOverflowHandler) HandleOverflow(crdtType CRDTType, state interface{}) (interface{}, error) {
	switch coh.overflowPolicy {
	case "saturate":
		return coh.saturate(crdtType, state)
	case "reset":
		return coh.reset(crdtType, state)
	case "error":
		return nil, fmt.Errorf("counter overflow detected")
	default:
		return state, nil
	}
}

func (coh *CounterOverflowHandler) saturate(crdtType CRDTType, state interface{}) (interface{}, error) {
	// Cap at maximum value
	const maxValue = ^uint64(0) - 1000

	switch crdtType {
	case CRDTTypeGCounter:
		counter := state.(map[string]uint64)
		for nodeID, count := range counter {
			if count > maxValue {
				counter[nodeID] = maxValue
			}
		}
		return counter, nil

	case CRDTTypePNCounter:
		pnCounter := state.(PNCounterState)
		for nodeID, count := range pnCounter.Positive {
			if count > maxValue {
				pnCounter.Positive[nodeID] = maxValue
			}
		}
		for nodeID, count := range pnCounter.Negative {
			if count > maxValue {
				pnCounter.Negative[nodeID] = maxValue
			}
		}
		return pnCounter, nil

	default:
		return state, nil
	}
}

func (coh *CounterOverflowHandler) reset(crdtType CRDTType, state interface{}) (interface{}, error) {
	// Reset counter (losing precision but avoiding overflow)
	switch crdtType {
	case CRDTTypeGCounter:
		return make(map[string]uint64), nil
	case CRDTTypePNCounter:
		return PNCounterState{
			Positive: make(map[string]uint64),
			Negative: make(map[string]uint64),
		}, nil
	default:
		return state, nil
	}
}
