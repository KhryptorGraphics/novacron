// Package conflict provides advanced conflict detection and resolution for DWCP
package conflict

import (
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	conflictsDetected = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_conflicts_detected_total",
		Help: "Total number of conflicts detected",
	}, []string{"type", "severity"})

	conflictDetectionLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dwcp_conflict_detection_latency_ms",
		Help:    "Conflict detection latency in milliseconds",
		Buckets: []float64{0.1, 0.5, 1, 5, 10, 50, 100},
	})
)

// ConflictType represents the category of conflict
type ConflictType int

const (
	ConflictTypeConcurrentUpdate ConflictType = iota
	ConflictTypeCausalViolation
	ConflictTypeSemanticConflict
	ConflictTypeInvariantViolation
	ConflictTypeResourceContention
)

func (ct ConflictType) String() string {
	return [...]string{
		"ConcurrentUpdate",
		"CausalViolation",
		"SemanticConflict",
		"InvariantViolation",
		"ResourceContention",
	}[ct]
}

// ConflictSeverity indicates how critical a conflict is
type ConflictSeverity int

const (
	SeverityLow ConflictSeverity = iota
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

func (cs ConflictSeverity) String() string {
	return [...]string{"Low", "Medium", "High", "Critical"}[cs]
}

// VectorClock represents a vector clock for causal ordering
type VectorClock struct {
	Clock map[string]uint64
	mu    sync.RWMutex
}

// NewVectorClock creates a new vector clock
func NewVectorClock() *VectorClock {
	return &VectorClock{
		Clock: make(map[string]uint64),
	}
}

// Increment increments the clock for a node
func (vc *VectorClock) Increment(nodeID string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.Clock[nodeID]++
}

// Update updates the clock with another clock
func (vc *VectorClock) Update(other *VectorClock) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	other.mu.RLock()
	defer other.mu.RUnlock()

	for nodeID, timestamp := range other.Clock {
		if vc.Clock[nodeID] < timestamp {
			vc.Clock[nodeID] = timestamp
		}
	}
}

// Compare compares this clock with another
func (vc *VectorClock) Compare(other *VectorClock) ClockRelation {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	other.mu.RLock()
	defer other.mu.RUnlock()

	var lessThan, greaterThan bool

	allNodes := make(map[string]bool)
	for nodeID := range vc.Clock {
		allNodes[nodeID] = true
	}
	for nodeID := range other.Clock {
		allNodes[nodeID] = true
	}

	for nodeID := range allNodes {
		myTime := vc.Clock[nodeID]
		otherTime := other.Clock[nodeID]

		if myTime < otherTime {
			lessThan = true
		} else if myTime > otherTime {
			greaterThan = true
		}
	}

	if lessThan && !greaterThan {
		return RelationBefore
	} else if !lessThan && greaterThan {
		return RelationAfter
	} else if !lessThan && !greaterThan {
		return RelationEqual
	}
	return RelationConcurrent
}

// ClockRelation describes the relationship between two vector clocks
type ClockRelation int

const (
	RelationBefore ClockRelation = iota
	RelationAfter
	RelationEqual
	RelationConcurrent
)

func (cr ClockRelation) String() string {
	return [...]string{"Before", "After", "Equal", "Concurrent"}[cr]
}

// Conflict represents a detected conflict
type Conflict struct {
	ID                string
	Type              ConflictType
	Severity          ConflictSeverity
	DetectedAt        time.Time
	ResourceID        string
	LocalVersion      *Version
	RemoteVersion     *Version
	CausalRelation    ClockRelation
	ComplexityScore   float64
	AffectedFields    []string
	RequiresManual    bool
	Context           map[string]interface{}
}

// Version represents a version of data with vector clock
type Version struct {
	VectorClock *VectorClock
	Timestamp   time.Time
	NodeID      string
	Data        interface{}
	Checksum    string
}

// ConflictDetector detects conflicts between versions
type ConflictDetector struct {
	mu                sync.RWMutex
	detectedConflicts map[string]*Conflict
	classifiers       []ConflictClassifier
	complexityCalc    ComplexityCalculator
	config            DetectorConfig
}

// DetectorConfig configures the conflict detector
type DetectorConfig struct {
	EnableCausalTracking   bool
	EnableSemanticAnalysis bool
	ComplexityThreshold    float64
	MaxConflictAge         time.Duration
	AutoCleanup            bool
}

// DefaultDetectorConfig returns default configuration
func DefaultDetectorConfig() DetectorConfig {
	return DetectorConfig{
		EnableCausalTracking:   true,
		EnableSemanticAnalysis: true,
		ComplexityThreshold:    0.7,
		MaxConflictAge:         24 * time.Hour,
		AutoCleanup:            true,
	}
}

// ConflictClassifier determines the type and severity of conflicts
type ConflictClassifier interface {
	Classify(local, remote *Version) (ConflictType, ConflictSeverity, []string)
	Priority() int
}

// ComplexityCalculator calculates conflict complexity
type ComplexityCalculator interface {
	Calculate(conflict *Conflict) float64
}

// NewConflictDetector creates a new conflict detector
func NewConflictDetector(config DetectorConfig) *ConflictDetector {
	cd := &ConflictDetector{
		detectedConflicts: make(map[string]*Conflict),
		classifiers:       make([]ConflictClassifier, 0),
		config:            config,
	}

	// Register default classifiers
	cd.RegisterClassifier(&ConcurrentUpdateClassifier{})
	cd.RegisterClassifier(&CausalViolationClassifier{})
	cd.RegisterClassifier(&SemanticConflictClassifier{})
	cd.RegisterClassifier(&InvariantViolationClassifier{})

	// Set default complexity calculator
	cd.complexityCalc = &DefaultComplexityCalculator{}

	return cd
}

// RegisterClassifier registers a conflict classifier
func (cd *ConflictDetector) RegisterClassifier(classifier ConflictClassifier) {
	cd.mu.Lock()
	defer cd.mu.Unlock()
	cd.classifiers = append(cd.classifiers, classifier)
}

// DetectConflict detects conflicts between two versions
func (cd *ConflictDetector) DetectConflict(ctx context.Context, resourceID string, local, remote *Version) (*Conflict, error) {
	start := time.Now()
	defer func() {
		conflictDetectionLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Compare vector clocks
	causalRelation := RelationConcurrent
	if cd.config.EnableCausalTracking && local.VectorClock != nil && remote.VectorClock != nil {
		causalRelation = local.VectorClock.Compare(remote.VectorClock)
	}

	// If not concurrent, no conflict
	if causalRelation != RelationConcurrent {
		return nil, nil
	}

	// Classify conflict
	var maxSeverity ConflictSeverity
	var conflictType ConflictType
	var affectedFields []string

	for _, classifier := range cd.classifiers {
		cType, severity, fields := classifier.Classify(local, remote)
		if severity > maxSeverity {
			maxSeverity = severity
			conflictType = cType
			affectedFields = fields
		}
	}

	conflict := &Conflict{
		ID:             generateConflictID(resourceID, local.NodeID, remote.NodeID),
		Type:           conflictType,
		Severity:       maxSeverity,
		DetectedAt:     time.Now(),
		ResourceID:     resourceID,
		LocalVersion:   local,
		RemoteVersion:  remote,
		CausalRelation: causalRelation,
		AffectedFields: affectedFields,
		Context:        make(map[string]interface{}),
	}

	// Calculate complexity
	conflict.ComplexityScore = cd.complexityCalc.Calculate(conflict)
	conflict.RequiresManual = conflict.ComplexityScore > cd.config.ComplexityThreshold

	// Record conflict
	cd.mu.Lock()
	cd.detectedConflicts[conflict.ID] = conflict
	cd.mu.Unlock()

	conflictsDetected.WithLabelValues(conflict.Type.String(), conflict.Severity.String()).Inc()

	return conflict, nil
}

// GetConflict retrieves a conflict by ID
func (cd *ConflictDetector) GetConflict(conflictID string) (*Conflict, bool) {
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	conflict, exists := cd.detectedConflicts[conflictID]
	return conflict, exists
}

// ResolveConflict marks a conflict as resolved
func (cd *ConflictDetector) ResolveConflict(conflictID string) {
	cd.mu.Lock()
	defer cd.mu.Unlock()
	delete(cd.detectedConflicts, conflictID)
}

// GetPendingConflicts returns all pending conflicts
func (cd *ConflictDetector) GetPendingConflicts() []*Conflict {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	conflicts := make([]*Conflict, 0, len(cd.detectedConflicts))
	for _, conflict := range cd.detectedConflicts {
		conflicts = append(conflicts, conflict)
	}
	return conflicts
}

// ConcurrentUpdateClassifier detects concurrent updates
type ConcurrentUpdateClassifier struct{}

func (c *ConcurrentUpdateClassifier) Classify(local, remote *Version) (ConflictType, ConflictSeverity, []string) {
	return ConflictTypeConcurrentUpdate, SeverityMedium, []string{}
}

func (c *ConcurrentUpdateClassifier) Priority() int {
	return 10
}

// CausalViolationClassifier detects causal ordering violations
type CausalViolationClassifier struct{}

func (c *CausalViolationClassifier) Classify(local, remote *Version) (ConflictType, ConflictSeverity, []string) {
	if local.VectorClock != nil && remote.VectorClock != nil {
		relation := local.VectorClock.Compare(remote.VectorClock)
		if relation == RelationConcurrent {
			return ConflictTypeCausalViolation, SeverityHigh, []string{}
		}
	}
	return ConflictTypeConcurrentUpdate, SeverityLow, []string{}
}

func (c *CausalViolationClassifier) Priority() int {
	return 20
}

// SemanticConflictClassifier detects semantic conflicts
type SemanticConflictClassifier struct{}

func (c *SemanticConflictClassifier) Classify(local, remote *Version) (ConflictType, ConflictSeverity, []string) {
	// Semantic analysis would go here
	return ConflictTypeSemanticConflict, SeverityMedium, []string{}
}

func (c *SemanticConflictClassifier) Priority() int {
	return 15
}

// InvariantViolationClassifier detects invariant violations
type InvariantViolationClassifier struct{}

func (c *InvariantViolationClassifier) Classify(local, remote *Version) (ConflictType, ConflictSeverity, []string) {
	// Invariant checking would go here
	return ConflictTypeInvariantViolation, SeverityCritical, []string{}
}

func (c *InvariantViolationClassifier) Priority() int {
	return 30
}

// DefaultComplexityCalculator calculates default complexity
type DefaultComplexityCalculator struct{}

func (c *DefaultComplexityCalculator) Calculate(conflict *Conflict) float64 {
	score := 0.0

	// Base score from severity
	switch conflict.Severity {
	case SeverityLow:
		score += 0.2
	case SeverityMedium:
		score += 0.4
	case SeverityHigh:
		score += 0.6
	case SeverityCritical:
		score += 0.8
	}

	// Increase for multiple affected fields
	if len(conflict.AffectedFields) > 3 {
		score += 0.2
	}

	// Increase for causal violations
	if conflict.Type == ConflictTypeCausalViolation {
		score += 0.3
	}

	if score > 1.0 {
		score = 1.0
	}

	return score
}

// generateConflictID generates a unique conflict ID
func generateConflictID(resourceID, node1, node2 string) string {
	return resourceID + "-" + node1 + "-" + node2 + "-" + time.Now().Format("20060102150405")
}

// CleanupOldConflicts removes old conflicts
func (cd *ConflictDetector) CleanupOldConflicts() {
	if !cd.config.AutoCleanup {
		return
	}

	cd.mu.Lock()
	defer cd.mu.Unlock()

	cutoff := time.Now().Add(-cd.config.MaxConflictAge)
	for id, conflict := range cd.detectedConflicts {
		if conflict.DetectedAt.Before(cutoff) {
			delete(cd.detectedConflicts, id)
		}
	}
}
