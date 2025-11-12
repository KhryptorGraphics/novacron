package security

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ReputationSystem tracks node reputation scores and manages trust
// Implements dynamic scoring with decay, boost, and quarantine mechanisms
type ReputationSystem struct {
	mu sync.RWMutex

	nodeID string
	logger *zap.Logger

	// Reputation data
	reputations map[string]*NodeReputation
	quarantined map[string]*QuarantineRecord

	// Configuration
	config ReputationConfig

	ctx    context.Context
	cancel context.CancelFunc
}

// NodeReputation tracks a node's reputation score
type NodeReputation struct {
	NodeID string
	Score  float64 // 0-100, starts at 50

	// Tracking
	TotalInteractions   int64
	SuccessfulActions   int64
	FailedActions       int64
	ByzantineViolations int64
	ConsensusCorrect    int64
	ConsensusIncorrect  int64

	// Time tracking
	FirstSeen   time.Time
	LastUpdated time.Time
	LastActive  time.Time

	// Status
	IsQuarantined bool
	QuarantinedAt time.Time
}

// QuarantineRecord tracks quarantined nodes
type QuarantineRecord struct {
	NodeID         string
	QuarantinedAt  time.Time
	Reason         string
	Evidence       interface{}
	CanRecover     bool
	RecoveryAt     time.Time
	ViolationCount int
}

// ReputationConfig configures reputation system behavior
type ReputationConfig struct {
	// Initial values
	InitialScore       float64
	NewNodeGracePeriod time.Duration

	// Scoring
	ConsensusCorrectBoost    float64
	ConsensusIncorrectPenalty float64
	ByzantinePenaltyMultiplier float64
	MessageSuccessBoost      float64
	MessageFailurePenalty    float64

	// Decay
	DecayEnabled     bool
	DecayRate        float64 // Points per hour of inactivity
	DecayInterval    time.Duration
	MinimumScore     float64 // Floor for decay

	// Thresholds
	QuarantineThreshold  float64 // Score below this = quarantine
	TrustedThreshold     float64 // Score above this = trusted
	SuspiciousThreshold  float64 // Score below this = suspicious

	// Quarantine
	QuarantineEnabled   bool
	QuarantineDuration  time.Duration
	AllowRecovery       bool
	RecoveryThreshold   float64 // Score needed to recover
	MaxQuarantineCount  int     // Max times a node can be quarantined

	// Cleanup
	CleanupInterval     time.Duration
	RemoveAfterInactive time.Duration
}

// ReputationLevel represents trust level
type ReputationLevel int

const (
	ReputationQuarantined ReputationLevel = iota
	ReputationUntrusted
	ReputationNeutral
	ReputationSuspicious
	ReputationTrusted
	ReputationHighlyTrusted
)

// NewReputationSystem creates a new reputation system
func NewReputationSystem(nodeID string, logger *zap.Logger) *ReputationSystem {
	ctx, cancel := context.WithCancel(context.Background())

	rs := &ReputationSystem{
		nodeID:      nodeID,
		logger:      logger,
		reputations: make(map[string]*NodeReputation),
		quarantined: make(map[string]*QuarantineRecord),
		config:      DefaultReputationConfig(),
		ctx:         ctx,
		cancel:      cancel,
	}

	go rs.decayLoop()
	go rs.cleanupLoop()

	return rs
}

// DefaultReputationConfig returns default reputation configuration
func DefaultReputationConfig() ReputationConfig {
	return ReputationConfig{
		InitialScore:       50.0,
		NewNodeGracePeriod: 5 * time.Minute,

		ConsensusCorrectBoost:      2.0,
		ConsensusIncorrectPenalty:   5.0,
		ByzantinePenaltyMultiplier: 3.0,
		MessageSuccessBoost:        0.5,
		MessageFailurePenalty:      1.0,

		DecayEnabled:  true,
		DecayRate:     0.5, // 0.5 points per hour
		DecayInterval: 1 * time.Hour,
		MinimumScore:  20.0,

		QuarantineThreshold: 15.0,
		TrustedThreshold:    75.0,
		SuspiciousThreshold: 40.0,

		QuarantineEnabled:  true,
		QuarantineDuration: 30 * time.Minute,
		AllowRecovery:      true,
		RecoveryThreshold:  50.0,
		MaxQuarantineCount: 3,

		CleanupInterval:     15 * time.Minute,
		RemoveAfterInactive: 24 * time.Hour,
	}
}

// GetReputation returns a node's reputation
func (rs *ReputationSystem) GetReputation(nodeID string) *NodeReputation {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	rep, exists := rs.reputations[nodeID]
	if !exists {
		return nil
	}
	return rep
}

// GetScore returns a node's reputation score
func (rs *ReputationSystem) GetScore(nodeID string) float64 {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	rep, exists := rs.reputations[nodeID]
	if !exists {
		return rs.config.InitialScore
	}
	return rep.Score
}

// GetLevel returns a node's reputation level
func (rs *ReputationSystem) GetLevel(nodeID string) ReputationLevel {
	score := rs.GetScore(nodeID)

	if rs.IsQuarantined(nodeID) {
		return ReputationQuarantined
	}

	switch {
	case score >= 90.0:
		return ReputationHighlyTrusted
	case score >= rs.config.TrustedThreshold:
		return ReputationTrusted
	case score >= rs.config.SuspiciousThreshold:
		return ReputationNeutral
	case score >= rs.config.QuarantineThreshold:
		return ReputationSuspicious
	default:
		return ReputationUntrusted
	}
}

// IsQuarantined checks if a node is quarantined
func (rs *ReputationSystem) IsQuarantined(nodeID string) bool {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	record, exists := rs.quarantined[nodeID]
	if !exists {
		return false
	}

	// Check if quarantine expired
	if rs.config.AllowRecovery && time.Since(record.QuarantinedAt) > rs.config.QuarantineDuration {
		return false
	}

	return true
}

// IsTrusted checks if a node is trusted
func (rs *ReputationSystem) IsTrusted(nodeID string) bool {
	return rs.GetScore(nodeID) >= rs.config.TrustedThreshold
}

// IsSuspicious checks if a node is suspicious
func (rs *ReputationSystem) IsSuspicious(nodeID string) bool {
	score := rs.GetScore(nodeID)
	return score < rs.config.SuspiciousThreshold && score >= rs.config.QuarantineThreshold
}

// RecordConsensusParticipation records consensus participation
func (rs *ReputationSystem) RecordConsensusParticipation(nodeID string, correct bool) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rep := rs.getOrCreateReputation(nodeID)
	rep.TotalInteractions++
	rep.LastActive = time.Now()
	rep.LastUpdated = time.Now()

	if correct {
		rep.ConsensusCorrect++
		rep.SuccessfulActions++
		rs.adjustScore(rep, rs.config.ConsensusCorrectBoost, "consensus_correct")
	} else {
		rep.ConsensusIncorrect++
		rep.FailedActions++
		rs.adjustScore(rep, -rs.config.ConsensusIncorrectPenalty, "consensus_incorrect")
	}

	rs.checkQuarantine(rep)
}

// RecordByzantineBehavior records Byzantine behavior with penalty
func (rs *ReputationSystem) RecordByzantineBehavior(nodeID string, severity float64) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rep := rs.getOrCreateReputation(nodeID)
	rep.ByzantineViolations++
	rep.FailedActions++
	rep.LastActive = time.Now()
	rep.LastUpdated = time.Now()

	// Heavy penalty for Byzantine behavior
	penalty := severity * rs.config.ByzantinePenaltyMultiplier
	rs.adjustScore(rep, -penalty, "byzantine_behavior")

	rs.logger.Warn("Byzantine behavior recorded",
		zap.String("node_id", nodeID),
		zap.Float64("severity", severity),
		zap.Float64("penalty", penalty),
		zap.Float64("new_score", rep.Score),
	)

	rs.checkQuarantine(rep)
}

// RecordMessageSuccess records successful message handling
func (rs *ReputationSystem) RecordMessageSuccess(nodeID string) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rep := rs.getOrCreateReputation(nodeID)
	rep.TotalInteractions++
	rep.SuccessfulActions++
	rep.LastActive = time.Now()
	rep.LastUpdated = time.Now()

	rs.adjustScore(rep, rs.config.MessageSuccessBoost, "message_success")
	rs.checkQuarantine(rep)
}

// RecordMessageFailure records failed message handling
func (rs *ReputationSystem) RecordMessageFailure(nodeID string) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rep := rs.getOrCreateReputation(nodeID)
	rep.TotalInteractions++
	rep.FailedActions++
	rep.LastActive = time.Now()
	rep.LastUpdated = time.Now()

	rs.adjustScore(rep, -rs.config.MessageFailurePenalty, "message_failure")
	rs.checkQuarantine(rep)
}

// QuarantineNode quarantines a node
func (rs *ReputationSystem) QuarantineNode(nodeID string, reason string) error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if !rs.config.QuarantineEnabled {
		return fmt.Errorf("quarantine disabled")
	}

	rep := rs.getOrCreateReputation(nodeID)

	// Check max quarantine count
	if record, exists := rs.quarantined[nodeID]; exists {
		if record.ViolationCount >= rs.config.MaxQuarantineCount {
			rs.logger.Error("Node exceeded max quarantine count",
				zap.String("node_id", nodeID),
				zap.Int("count", record.ViolationCount),
			)
			return fmt.Errorf("node exceeded max quarantine count")
		}
		record.ViolationCount++
		record.QuarantinedAt = time.Now()
		record.Reason = reason
	} else {
		rs.quarantined[nodeID] = &QuarantineRecord{
			NodeID:         nodeID,
			QuarantinedAt:  time.Now(),
			Reason:         reason,
			CanRecover:     rs.config.AllowRecovery,
			RecoveryAt:     time.Now().Add(rs.config.QuarantineDuration),
			ViolationCount: 1,
		}
	}

	rep.IsQuarantined = true
	rep.QuarantinedAt = time.Now()

	rs.logger.Error("Node quarantined",
		zap.String("node_id", nodeID),
		zap.String("reason", reason),
		zap.Float64("score", rep.Score),
	)

	return nil
}

// ReleaseQuarantine releases a node from quarantine
func (rs *ReputationSystem) ReleaseQuarantine(nodeID string) error {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	record, exists := rs.quarantined[nodeID]
	if !exists {
		return fmt.Errorf("node not quarantined: %s", nodeID)
	}

	rep := rs.getOrCreateReputation(nodeID)

	// Check if meets recovery threshold
	if rs.config.AllowRecovery && rep.Score < rs.config.RecoveryThreshold {
		return fmt.Errorf("score too low for recovery: %.2f < %.2f", rep.Score, rs.config.RecoveryThreshold)
	}

	delete(rs.quarantined, nodeID)
	rep.IsQuarantined = false

	rs.logger.Info("Node released from quarantine",
		zap.String("node_id", nodeID),
		zap.Float64("score", rep.Score),
		zap.Int("violations", record.ViolationCount),
	)

	return nil
}

// GetQuarantinedNodes returns all quarantined nodes
func (rs *ReputationSystem) GetQuarantinedNodes() map[string]*QuarantineRecord {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	result := make(map[string]*QuarantineRecord)
	for id, record := range rs.quarantined {
		result[id] = record
	}
	return result
}

// GetTrustedNodes returns all trusted nodes
func (rs *ReputationSystem) GetTrustedNodes() map[string]*NodeReputation {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	result := make(map[string]*NodeReputation)
	for id, rep := range rs.reputations {
		if rep.Score >= rs.config.TrustedThreshold && !rep.IsQuarantined {
			result[id] = rep
		}
	}
	return result
}

// GetSuspiciousNodes returns all suspicious nodes
func (rs *ReputationSystem) GetSuspiciousNodes() map[string]*NodeReputation {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	result := make(map[string]*NodeReputation)
	for id, rep := range rs.reputations {
		score := rep.Score
		if score < rs.config.SuspiciousThreshold && score >= rs.config.QuarantineThreshold {
			result[id] = rep
		}
	}
	return result
}

// adjustScore adjusts a node's reputation score
func (rs *ReputationSystem) adjustScore(rep *NodeReputation, delta float64, reason string) {
	oldScore := rep.Score
	rep.Score += delta

	// Clamp to [0, 100]
	if rep.Score < 0 {
		rep.Score = 0
	}
	if rep.Score > 100 {
		rep.Score = 100
	}

	rs.logger.Debug("Score adjusted",
		zap.String("node_id", rep.NodeID),
		zap.String("reason", reason),
		zap.Float64("delta", delta),
		zap.Float64("old_score", oldScore),
		zap.Float64("new_score", rep.Score),
	)
}

// checkQuarantine checks if a node should be quarantined
func (rs *ReputationSystem) checkQuarantine(rep *NodeReputation) {
	if !rs.config.QuarantineEnabled {
		return
	}

	if rep.Score <= rs.config.QuarantineThreshold && !rep.IsQuarantined {
		rs.QuarantineNode(rep.NodeID, fmt.Sprintf("Score fell below threshold: %.2f", rep.Score))
	}
}

// getOrCreateReputation gets or creates node reputation
func (rs *ReputationSystem) getOrCreateReputation(nodeID string) *NodeReputation {
	if rep, exists := rs.reputations[nodeID]; exists {
		return rep
	}

	rep := &NodeReputation{
		NodeID:      nodeID,
		Score:       rs.config.InitialScore,
		FirstSeen:   time.Now(),
		LastUpdated: time.Now(),
		LastActive:  time.Now(),
	}
	rs.reputations[nodeID] = rep

	rs.logger.Info("New node reputation created",
		zap.String("node_id", nodeID),
		zap.Float64("initial_score", rs.config.InitialScore),
	)

	return rep
}

// decayLoop periodically decays inactive node scores
func (rs *ReputationSystem) decayLoop() {
	if !rs.config.DecayEnabled {
		return
	}

	ticker := time.NewTicker(rs.config.DecayInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rs.ctx.Done():
			return
		case <-ticker.C:
			rs.applyDecay()
		}
	}
}

// applyDecay applies decay to inactive node scores
func (rs *ReputationSystem) applyDecay() {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	now := time.Now()
	for _, rep := range rs.reputations {
		// Skip grace period for new nodes
		if now.Sub(rep.FirstSeen) < rs.config.NewNodeGracePeriod {
			continue
		}

		// Calculate inactivity
		inactivity := now.Sub(rep.LastActive)
		if inactivity > rs.config.DecayInterval {
			hours := inactivity.Hours()
			decay := rs.config.DecayRate * hours

			// Only decay if above minimum
			if rep.Score > rs.config.MinimumScore {
				oldScore := rep.Score
				rep.Score -= decay
				if rep.Score < rs.config.MinimumScore {
					rep.Score = rs.config.MinimumScore
				}

				if oldScore != rep.Score {
					rs.logger.Debug("Score decayed",
						zap.String("node_id", rep.NodeID),
						zap.Float64("hours_inactive", hours),
						zap.Float64("decay", decay),
						zap.Float64("new_score", rep.Score),
					)
				}
			}
		}
	}
}

// cleanupLoop periodically cleans up old reputation data
func (rs *ReputationSystem) cleanupLoop() {
	ticker := time.NewTicker(rs.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rs.ctx.Done():
			return
		case <-ticker.C:
			rs.cleanup()
		}
	}
}

// cleanup removes old reputation data
func (rs *ReputationSystem) cleanup() {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rs.config.RemoveAfterInactive)

	// Remove very old inactive nodes
	for id, rep := range rs.reputations {
		if rep.LastActive.Before(cutoff) && !rep.IsQuarantined {
			delete(rs.reputations, id)
			rs.logger.Debug("Removed inactive node reputation",
				zap.String("node_id", id),
				zap.Time("last_active", rep.LastActive),
			)
		}
	}

	// Check quarantine expirations
	for id, record := range rs.quarantined {
		if rs.config.AllowRecovery && time.Since(record.QuarantinedAt) > rs.config.QuarantineDuration {
			rep := rs.reputations[id]
			if rep != nil && rep.Score >= rs.config.RecoveryThreshold {
				rs.ReleaseQuarantine(id)
			}
		}
	}
}

// Stop stops the reputation system
func (rs *ReputationSystem) Stop() {
	rs.cancel()
}

// GetStats returns reputation statistics
func (rs *ReputationSystem) GetStats() map[string]interface{} {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	var totalScore float64
	var trusted, suspicious, untrusted int

	for _, rep := range rs.reputations {
		totalScore += rep.Score
		if rep.Score >= rs.config.TrustedThreshold {
			trusted++
		} else if rep.Score < rs.config.SuspiciousThreshold {
			if rep.Score >= rs.config.QuarantineThreshold {
				suspicious++
			} else {
				untrusted++
			}
		}
	}

	avgScore := 0.0
	if len(rs.reputations) > 0 {
		avgScore = totalScore / float64(len(rs.reputations))
	}

	return map[string]interface{}{
		"total_nodes":      len(rs.reputations),
		"trusted_nodes":    trusted,
		"suspicious_nodes": suspicious,
		"untrusted_nodes":  untrusted,
		"quarantined":      len(rs.quarantined),
		"average_score":    avgScore,
	}
}
