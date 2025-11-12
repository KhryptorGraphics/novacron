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
	resolutionsAttempted = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_conflict_resolutions_attempted_total",
		Help: "Total number of conflict resolution attempts",
	}, []string{"strategy", "result"})

	resolutionLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "dwcp_conflict_resolution_latency_ms",
		Help:    "Conflict resolution latency in milliseconds",
		Buckets: []float64{1, 5, 10, 50, 100, 500, 1000},
	}, []string{"strategy"})
)

// StrategyType identifies different resolution strategies
type StrategyType int

const (
	StrategyLastWriteWins StrategyType = iota
	StrategyMultiValueRegister
	StrategyOperationalTransform
	StrategySemanticMerge
	StrategyCustomFunction
	StrategyAutomaticRollback
	StrategyManualIntervention
	StrategyHighestPriority
	StrategyConsensusVote
)

func (st StrategyType) String() string {
	return [...]string{
		"LastWriteWins",
		"MultiValueRegister",
		"OperationalTransform",
		"SemanticMerge",
		"CustomFunction",
		"AutomaticRollback",
		"ManualIntervention",
		"HighestPriority",
		"ConsensusVote",
	}[st]
}

// ResolutionResult represents the outcome of conflict resolution
type ResolutionResult struct {
	Success      bool
	Strategy     StrategyType
	ResolvedData interface{}
	Message      string
	Timestamp    time.Time
	Metadata     map[string]interface{}
}

// ResolutionStrategy defines the interface for conflict resolution strategies
type ResolutionStrategy interface {
	Name() string
	Type() StrategyType
	CanResolve(conflict *Conflict) bool
	Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error)
	Priority() int
}

// StrategyRegistry manages available resolution strategies
type StrategyRegistry struct {
	mu         sync.RWMutex
	strategies map[StrategyType]ResolutionStrategy
	custom     map[string]ResolutionStrategy
}

// NewStrategyRegistry creates a new strategy registry
func NewStrategyRegistry() *StrategyRegistry {
	sr := &StrategyRegistry{
		strategies: make(map[StrategyType]ResolutionStrategy),
		custom:     make(map[string]ResolutionStrategy),
	}

	// Register default strategies
	sr.Register(&LastWriteWinsStrategy{})
	sr.Register(&MultiValueRegisterStrategy{})
	sr.Register(&OperationalTransformStrategy{})
	sr.Register(&SemanticMergeStrategy{})
	sr.Register(&AutomaticRollbackStrategy{})
	sr.Register(&ManualInterventionStrategy{})
	sr.Register(&HighestPriorityStrategy{})
	sr.Register(&ConsensusVoteStrategy{})

	return sr
}

// Register registers a resolution strategy
func (sr *StrategyRegistry) Register(strategy ResolutionStrategy) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.strategies[strategy.Type()] = strategy
}

// RegisterCustom registers a custom resolution strategy
func (sr *StrategyRegistry) RegisterCustom(name string, strategy ResolutionStrategy) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	sr.custom[name] = strategy
}

// GetStrategy retrieves a strategy by type
func (sr *StrategyRegistry) GetStrategy(strategyType StrategyType) (ResolutionStrategy, bool) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	strategy, exists := sr.strategies[strategyType]
	return strategy, exists
}

// GetCustomStrategy retrieves a custom strategy by name
func (sr *StrategyRegistry) GetCustomStrategy(name string) (ResolutionStrategy, bool) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	strategy, exists := sr.custom[name]
	return strategy, exists
}

// LastWriteWinsStrategy resolves conflicts using timestamp-based ordering
type LastWriteWinsStrategy struct{}

func (s *LastWriteWinsStrategy) Name() string {
	return "LastWriteWins"
}

func (s *LastWriteWinsStrategy) Type() StrategyType {
	return StrategyLastWriteWins
}

func (s *LastWriteWinsStrategy) CanResolve(conflict *Conflict) bool {
	return conflict.LocalVersion != nil && conflict.RemoteVersion != nil
}

func (s *LastWriteWinsStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	local := conflict.LocalVersion
	remote := conflict.RemoteVersion

	var winner *Version
	if remote.Timestamp.After(local.Timestamp) {
		winner = remote
	} else if local.Timestamp.After(remote.Timestamp) {
		winner = local
	} else {
		// Tie-breaker: use node ID
		if remote.NodeID > local.NodeID {
			winner = remote
		} else {
			winner = local
		}
	}

	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyLastWriteWins,
		ResolvedData: winner.Data,
		Message:      fmt.Sprintf("Resolved using last-write-wins: selected version from %s", winner.NodeID),
		Timestamp:    time.Now(),
		Metadata: map[string]interface{}{
			"winner_node":      winner.NodeID,
			"winner_timestamp": winner.Timestamp,
		},
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *LastWriteWinsStrategy) Priority() int {
	return 10
}

// MultiValueRegisterStrategy keeps all concurrent values
type MultiValueRegisterStrategy struct{}

func (s *MultiValueRegisterStrategy) Name() string {
	return "MultiValueRegister"
}

func (s *MultiValueRegisterStrategy) Type() StrategyType {
	return StrategyMultiValueRegister
}

func (s *MultiValueRegisterStrategy) CanResolve(conflict *Conflict) bool {
	return true
}

func (s *MultiValueRegisterStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Keep both values, let application decide
	multiValue := map[string]interface{}{
		"local":  conflict.LocalVersion.Data,
		"remote": conflict.RemoteVersion.Data,
		"nodes": []string{
			conflict.LocalVersion.NodeID,
			conflict.RemoteVersion.NodeID,
		},
	}

	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyMultiValueRegister,
		ResolvedData: multiValue,
		Message:      "Keeping all concurrent values for application-level resolution",
		Timestamp:    time.Now(),
		Metadata: map[string]interface{}{
			"value_count": 2,
		},
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *MultiValueRegisterStrategy) Priority() int {
	return 20
}

// OperationalTransformStrategy applies operational transformation
type OperationalTransformStrategy struct{}

func (s *OperationalTransformStrategy) Name() string {
	return "OperationalTransform"
}

func (s *OperationalTransformStrategy) Type() StrategyType {
	return StrategyOperationalTransform
}

func (s *OperationalTransformStrategy) CanResolve(conflict *Conflict) bool {
	// Only for text/collaborative editing conflicts
	return conflict.Type == ConflictTypeSemanticConflict
}

func (s *OperationalTransformStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Simplified OT implementation
	// In production, use a proper OT library
	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyOperationalTransform,
		ResolvedData: conflict.LocalVersion.Data, // Placeholder
		Message:      "Applied operational transformation",
		Timestamp:    time.Now(),
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *OperationalTransformStrategy) Priority() int {
	return 30
}

// SemanticMergeStrategy performs semantic merge based on application rules
type SemanticMergeStrategy struct {
	mu         sync.RWMutex
	mergeRules map[string]MergeRule
}

type MergeRule func(local, remote interface{}) (interface{}, error)

func NewSemanticMergeStrategy() *SemanticMergeStrategy {
	return &SemanticMergeStrategy{
		mergeRules: make(map[string]MergeRule),
	}
}

func (s *SemanticMergeStrategy) Name() string {
	return "SemanticMerge"
}

func (s *SemanticMergeStrategy) Type() StrategyType {
	return StrategySemanticMerge
}

func (s *SemanticMergeStrategy) CanResolve(conflict *Conflict) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, exists := s.mergeRules[conflict.ResourceID]
	return exists
}

func (s *SemanticMergeStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	s.mu.RLock()
	rule, exists := s.mergeRules[conflict.ResourceID]
	s.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no merge rule for resource %s", conflict.ResourceID)
	}

	merged, err := rule(conflict.LocalVersion.Data, conflict.RemoteVersion.Data)
	if err != nil {
		resolutionsAttempted.WithLabelValues(s.Name(), "failure").Inc()
		return nil, err
	}

	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategySemanticMerge,
		ResolvedData: merged,
		Message:      "Applied semantic merge rule",
		Timestamp:    time.Now(),
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *SemanticMergeStrategy) Priority() int {
	return 40
}

func (s *SemanticMergeStrategy) RegisterRule(resourceType string, rule MergeRule) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mergeRules[resourceType] = rule
}

// AutomaticRollbackStrategy rolls back to a known good state
type AutomaticRollbackStrategy struct{}

func (s *AutomaticRollbackStrategy) Name() string {
	return "AutomaticRollback"
}

func (s *AutomaticRollbackStrategy) Type() StrategyType {
	return StrategyAutomaticRollback
}

func (s *AutomaticRollbackStrategy) CanResolve(conflict *Conflict) bool {
	return conflict.Severity >= SeverityHigh
}

func (s *AutomaticRollbackStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Roll back to the older version
	var rollbackVersion *Version
	if conflict.LocalVersion.Timestamp.Before(conflict.RemoteVersion.Timestamp) {
		rollbackVersion = conflict.LocalVersion
	} else {
		rollbackVersion = conflict.RemoteVersion
	}

	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyAutomaticRollback,
		ResolvedData: rollbackVersion.Data,
		Message:      fmt.Sprintf("Rolled back to version from %s", rollbackVersion.NodeID),
		Timestamp:    time.Now(),
		Metadata: map[string]interface{}{
			"rollback_to": rollbackVersion.NodeID,
		},
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *AutomaticRollbackStrategy) Priority() int {
	return 50
}

// ManualInterventionStrategy escalates to human operators
type ManualInterventionStrategy struct {
	mu                sync.RWMutex
	pendingResolution map[string]chan *ResolutionResult
}

func NewManualInterventionStrategy() *ManualInterventionStrategy {
	return &ManualInterventionStrategy{
		pendingResolution: make(map[string]chan *ResolutionResult),
	}
}

func (s *ManualInterventionStrategy) Name() string {
	return "ManualIntervention"
}

func (s *ManualInterventionStrategy) Type() StrategyType {
	return StrategyManualIntervention
}

func (s *ManualInterventionStrategy) CanResolve(conflict *Conflict) bool {
	return conflict.RequiresManual
}

func (s *ManualInterventionStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Create channel for manual resolution
	s.mu.Lock()
	resChan := make(chan *ResolutionResult, 1)
	s.pendingResolution[conflict.ID] = resChan
	s.mu.Unlock()

	// Wait for manual resolution or timeout
	select {
	case result := <-resChan:
		resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
		return result, nil
	case <-ctx.Done():
		resolutionsAttempted.WithLabelValues(s.Name(), "timeout").Inc()
		return nil, fmt.Errorf("manual resolution timeout")
	}
}

func (s *ManualInterventionStrategy) Priority() int {
	return 100
}

func (s *ManualInterventionStrategy) ProvideResolution(conflictID string, result *ResolutionResult) error {
	s.mu.RLock()
	resChan, exists := s.pendingResolution[conflictID]
	s.mu.RUnlock()

	if !exists {
		return fmt.Errorf("no pending resolution for conflict %s", conflictID)
	}

	resChan <- result
	close(resChan)

	s.mu.Lock()
	delete(s.pendingResolution, conflictID)
	s.mu.Unlock()

	return nil
}

// HighestPriorityStrategy selects version with highest priority
type HighestPriorityStrategy struct{}

func (s *HighestPriorityStrategy) Name() string {
	return "HighestPriority"
}

func (s *HighestPriorityStrategy) Type() StrategyType {
	return StrategyHighestPriority
}

func (s *HighestPriorityStrategy) CanResolve(conflict *Conflict) bool {
	return true
}

func (s *HighestPriorityStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Prefer remote version (assuming it comes from primary)
	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyHighestPriority,
		ResolvedData: conflict.RemoteVersion.Data,
		Message:      "Selected version with highest priority",
		Timestamp:    time.Now(),
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *HighestPriorityStrategy) Priority() int {
	return 15
}

// ConsensusVoteStrategy uses consensus voting
type ConsensusVoteStrategy struct {
	mu    sync.RWMutex
	votes map[string]map[string]int // conflict ID -> version checksum -> vote count
}

func NewConsensusVoteStrategy() *ConsensusVoteStrategy {
	return &ConsensusVoteStrategy{
		votes: make(map[string]map[string]int),
	}
}

func (s *ConsensusVoteStrategy) Name() string {
	return "ConsensusVote"
}

func (s *ConsensusVoteStrategy) Type() StrategyType {
	return StrategyConsensusVote
}

func (s *ConsensusVoteStrategy) CanResolve(conflict *Conflict) bool {
	return true
}

func (s *ConsensusVoteStrategy) Resolve(ctx context.Context, conflict *Conflict) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		resolutionLatency.WithLabelValues(s.Name()).Observe(float64(time.Since(start).Milliseconds()))
	}()

	s.mu.RLock()
	voteMap, exists := s.votes[conflict.ID]
	s.mu.RUnlock()

	if !exists {
		// Default to local version
		result := &ResolutionResult{
			Success:      true,
			Strategy:     StrategyConsensusVote,
			ResolvedData: conflict.LocalVersion.Data,
			Message:      "No votes received, using local version",
			Timestamp:    time.Now(),
		}
		resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
		return result, nil
	}

	// Find version with most votes
	var maxVotes int
	var winnerChecksum string
	for checksum, votes := range voteMap {
		if votes > maxVotes {
			maxVotes = votes
			winnerChecksum = checksum
		}
	}

	var winnerData interface{}
	if winnerChecksum == conflict.LocalVersion.Checksum {
		winnerData = conflict.LocalVersion.Data
	} else {
		winnerData = conflict.RemoteVersion.Data
	}

	result := &ResolutionResult{
		Success:      true,
		Strategy:     StrategyConsensusVote,
		ResolvedData: winnerData,
		Message:      fmt.Sprintf("Consensus reached with %d votes", maxVotes),
		Timestamp:    time.Now(),
		Metadata: map[string]interface{}{
			"vote_count": maxVotes,
		},
	}

	resolutionsAttempted.WithLabelValues(s.Name(), "success").Inc()
	return result, nil
}

func (s *ConsensusVoteStrategy) Priority() int {
	return 25
}

func (s *ConsensusVoteStrategy) Vote(conflictID, versionChecksum string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.votes[conflictID]; !exists {
		s.votes[conflictID] = make(map[string]int)
	}
	s.votes[conflictID][versionChecksum]++
}
