package security

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ByzantineDetector identifies malicious nodes based on behavior patterns
// Detects various attack types: message tampering, equivocation, timing attacks, etc.
type ByzantineDetector struct {
	mu sync.RWMutex

	nodeID string
	logger *zap.Logger

	// Behavior tracking
	nodeBehavior   map[string]*NodeBehavior
	messageHistory map[string]*MessageRecord
	consensusVotes map[string]*VoteTracker

	// Detection thresholds
	config DetectorConfig

	// Detection results
	suspiciousNodes map[string]*SuspicionRecord
	confirmedBad    map[string]*ByzantineEvidence

	// Reputation integration
	reputationSystem *ReputationSystem

	ctx    context.Context
	cancel context.CancelFunc
}

// NodeBehavior tracks a node's behavior patterns
type NodeBehavior struct {
	NodeID            string
	TotalMessages     int64
	InvalidSignatures int64
	ConflictingVotes  int64
	TimeoutViolations int64
	MalformedMessages int64
	ResponseTimes     []time.Duration
	LastSeen          time.Time

	// Equivocation detection
	PrePreparesSent  map[int64][]string // sequence -> digests
	PreparesSent     map[int64][]string
	CommitsSent      map[int64][]string
	ViewChangeSent   map[int64][]int64 // sequence -> views
}

// MessageRecord stores message history for validation
type MessageRecord struct {
	MessageID   string
	SenderID    string
	MessageType string
	Digest      string
	Timestamp   time.Time
	Signature   string
	Verified    bool
}

// VoteTracker tracks consensus votes to detect conflicts
type VoteTracker struct {
	Sequence   int64
	View       int64
	Digest     string
	PrePrepare map[string]string // nodeID -> digest
	Prepare    map[string]string
	Commit     map[string]string
}

// SuspicionRecord tracks suspicious behavior
type SuspicionRecord struct {
	NodeID         string
	SuspicionScore float64 // 0-100
	Violations     []*Violation
	FirstDetected  time.Time
	LastUpdated    time.Time
}

// Violation represents a detected misbehavior
type Violation struct {
	Type        ViolationType
	Severity    float64 // 0-100
	Description string
	Evidence    interface{}
	Timestamp   time.Time
}

// ViolationType defines types of Byzantine behavior
type ViolationType int

const (
	ViolationInvalidSignature ViolationType = iota
	ViolationEquivocation                    // Sending conflicting messages
	ViolationTimingAnomaly
	ViolationMalformedMessage
	ViolationConflictingVote
	ViolationSilentNode // Not participating
	ViolationFloodAttack
	ViolationViewChangeAbuse
	ViolationCheckpointManipulation
)

// ByzantineEvidence contains proof of Byzantine behavior
type ByzantineEvidence struct {
	NodeID      string
	AttackType  AttackType
	Evidence    interface{}
	Confidence  float64 // 0-1
	DetectedAt  time.Time
	Violations  []*Violation
}

// AttackType defines categories of Byzantine attacks
type AttackType int

const (
	AttackEquivocation AttackType = iota
	AttackMessageTampering
	AttackTimingManipulation
	AttackDenialOfService
	AttackViewChangeAttack
	AttackSelfishMining
	AttackCollusion
)

// DetectorConfig configures Byzantine detection sensitivity
type DetectorConfig struct {
	// Thresholds
	InvalidSignatureThreshold float64       // % of messages
	EquivocationThreshold     int           // Number of conflicting messages
	TimeoutViolationThreshold int           // Number of timeouts
	ResponseTimeWindow        time.Duration // Window for timing analysis
	ResponseTimeStdDev        float64       // Max standard deviations

	// Detection windows
	BehaviorWindow     time.Duration // Time window for behavior analysis
	SuspicionDecay     time.Duration // How fast suspicion decays
	ConfirmationWindow time.Duration // Time to confirm Byzantine behavior

	// Scoring
	SuspicionThreshold       float64 // Threshold to mark as suspicious
	ByzantineThreshold       float64 // Threshold to confirm Byzantine
	ViolationWeights         map[ViolationType]float64
	RequireMultipleViolation bool // Require multiple violation types
}

// NewByzantineDetector creates a new Byzantine detector
func NewByzantineDetector(nodeID string, reputation *ReputationSystem, logger *zap.Logger) *ByzantineDetector {
	ctx, cancel := context.WithCancel(context.Background())

	detector := &ByzantineDetector{
		nodeID:           nodeID,
		logger:           logger,
		nodeBehavior:     make(map[string]*NodeBehavior),
		messageHistory:   make(map[string]*MessageRecord),
		consensusVotes:   make(map[string]*VoteTracker),
		suspiciousNodes:  make(map[string]*SuspicionRecord),
		confirmedBad:     make(map[string]*ByzantineEvidence),
		reputationSystem: reputation,
		ctx:              ctx,
		cancel:           cancel,
		config:           DefaultDetectorConfig(),
	}

	go detector.monitoringLoop()

	return detector
}

// DefaultDetectorConfig returns default detection configuration
func DefaultDetectorConfig() DetectorConfig {
	return DetectorConfig{
		InvalidSignatureThreshold: 0.05, // 5% invalid signatures
		EquivocationThreshold:     2,    // 2 conflicting messages
		TimeoutViolationThreshold: 3,
		ResponseTimeWindow:        10 * time.Second,
		ResponseTimeStdDev:        3.0,

		BehaviorWindow:     5 * time.Minute,
		SuspicionDecay:     10 * time.Minute,
		ConfirmationWindow: 30 * time.Second,

		SuspicionThreshold: 40.0,
		ByzantineThreshold: 70.0,

		ViolationWeights: map[ViolationType]float64{
			ViolationInvalidSignature:       15.0,
			ViolationEquivocation:            40.0,
			ViolationTimingAnomaly:           10.0,
			ViolationMalformedMessage:        8.0,
			ViolationConflictingVote:         35.0,
			ViolationSilentNode:              5.0,
			ViolationFloodAttack:             25.0,
			ViolationViewChangeAbuse:         30.0,
			ViolationCheckpointManipulation:  45.0,
		},

		RequireMultipleViolation: true,
	}
}

// RecordMessage records a message for validation
func (bd *ByzantineDetector) RecordMessage(senderID, messageType string, message interface{}, signature string) error {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	// Generate message digest
	digest := bd.computeDigest(message)
	messageID := fmt.Sprintf("%s-%s-%d", senderID, messageType, time.Now().UnixNano())

	record := &MessageRecord{
		MessageID:   messageID,
		SenderID:    senderID,
		MessageType: messageType,
		Digest:      digest,
		Timestamp:   time.Now(),
		Signature:   signature,
		Verified:    false,
	}

	bd.messageHistory[messageID] = record

	// Update node behavior
	behavior := bd.getOrCreateBehavior(senderID)
	behavior.TotalMessages++
	behavior.LastSeen = time.Now()

	return nil
}

// ValidateSignature validates a message signature and records result
func (bd *ByzantineDetector) ValidateSignature(messageID string, valid bool) error {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	record, exists := bd.messageHistory[messageID]
	if !exists {
		return fmt.Errorf("message not found: %s", messageID)
	}

	record.Verified = valid

	if !valid {
		behavior := bd.nodeBehavior[record.SenderID]
		behavior.InvalidSignatures++

		// Record violation
		bd.recordViolation(record.SenderID, &Violation{
			Type:        ViolationInvalidSignature,
			Severity:    bd.config.ViolationWeights[ViolationInvalidSignature],
			Description: fmt.Sprintf("Invalid signature on %s message", record.MessageType),
			Evidence:    record,
			Timestamp:   time.Now(),
		})
	}

	return nil
}

// RecordConsensusVote records a consensus vote for equivocation detection
func (bd *ByzantineDetector) RecordConsensusVote(nodeID string, view, sequence int64, digest string, phase string) error {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	key := fmt.Sprintf("%d-%d", view, sequence)
	tracker := bd.getOrCreateVoteTracker(key, view, sequence)

	var voteMap map[string]string
	behavior := bd.getOrCreateBehavior(nodeID)

	switch phase {
	case "pre-prepare":
		voteMap = tracker.PrePrepare
		if behavior.PrePreparesSent == nil {
			behavior.PrePreparesSent = make(map[int64][]string)
		}
		behavior.PrePreparesSent[sequence] = append(behavior.PrePreparesSent[sequence], digest)

	case "prepare":
		voteMap = tracker.Prepare
		if behavior.PreparesSent == nil {
			behavior.PreparesSent = make(map[int64][]string)
		}
		behavior.PreparesSent[sequence] = append(behavior.PreparesSent[sequence], digest)

	case "commit":
		voteMap = tracker.Commit
		if behavior.CommitsSent == nil {
			behavior.CommitsSent = make(map[int64][]string)
		}
		behavior.CommitsSent[sequence] = append(behavior.CommitsSent[sequence], digest)
	}

	// Check for equivocation
	if existingDigest, exists := voteMap[nodeID]; exists {
		if existingDigest != digest {
			// Equivocation detected!
			bd.recordViolation(nodeID, &Violation{
				Type:        ViolationEquivocation,
				Severity:    bd.config.ViolationWeights[ViolationEquivocation],
				Description: fmt.Sprintf("Equivocation in %s: sent different digests for seq %d", phase, sequence),
				Evidence: map[string]interface{}{
					"phase":          phase,
					"sequence":       sequence,
					"first_digest":   existingDigest,
					"second_digest":  digest,
				},
				Timestamp: time.Now(),
			})

			behavior.ConflictingVotes++
		}
	} else {
		voteMap[nodeID] = digest
	}

	return nil
}

// RecordResponseTime records message response time for timing analysis
func (bd *ByzantineDetector) RecordResponseTime(nodeID string, responseTime time.Duration) {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	behavior := bd.getOrCreateBehavior(nodeID)
	behavior.ResponseTimes = append(behavior.ResponseTimes, responseTime)

	// Keep only recent response times
	if len(behavior.ResponseTimes) > 100 {
		behavior.ResponseTimes = behavior.ResponseTimes[len(behavior.ResponseTimes)-100:]
	}

	// Analyze for timing anomalies
	if len(behavior.ResponseTimes) > 10 {
		mean, stdDev := bd.computeStats(behavior.ResponseTimes)
		if responseTime > mean+time.Duration(float64(stdDev)*bd.config.ResponseTimeStdDev) {
			bd.recordViolation(nodeID, &Violation{
				Type:        ViolationTimingAnomaly,
				Severity:    bd.config.ViolationWeights[ViolationTimingAnomaly],
				Description: fmt.Sprintf("Response time anomaly: %v (mean: %v, stddev: %v)", responseTime, mean, stdDev),
				Evidence: map[string]interface{}{
					"response_time": responseTime,
					"mean":          mean,
					"std_dev":       stdDev,
				},
				Timestamp: time.Now(),
			})
		}
	}
}

// RecordMalformedMessage records a malformed message
func (bd *ByzantineDetector) RecordMalformedMessage(nodeID string, messageType string, reason string) {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	behavior := bd.getOrCreateBehavior(nodeID)
	behavior.MalformedMessages++

	bd.recordViolation(nodeID, &Violation{
		Type:        ViolationMalformedMessage,
		Severity:    bd.config.ViolationWeights[ViolationMalformedMessage],
		Description: fmt.Sprintf("Malformed %s message: %s", messageType, reason),
		Evidence: map[string]interface{}{
			"message_type": messageType,
			"reason":       reason,
		},
		Timestamp: time.Now(),
	})
}

// IsSuspicious checks if a node is suspicious
func (bd *ByzantineDetector) IsSuspicious(nodeID string) bool {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	if record, exists := bd.suspiciousNodes[nodeID]; exists {
		return record.SuspicionScore >= bd.config.SuspicionThreshold
	}
	return false
}

// IsByzantine checks if a node is confirmed Byzantine
func (bd *ByzantineDetector) IsByzantine(nodeID string) bool {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	_, exists := bd.confirmedBad[nodeID]
	return exists
}

// GetByzantineEvidence returns evidence for a Byzantine node
func (bd *ByzantineDetector) GetByzantineEvidence(nodeID string) (*ByzantineEvidence, bool) {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	evidence, exists := bd.confirmedBad[nodeID]
	return evidence, exists
}

// GetSuspiciousNodes returns all suspicious nodes
func (bd *ByzantineDetector) GetSuspiciousNodes() map[string]*SuspicionRecord {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	result := make(map[string]*SuspicionRecord)
	for id, record := range bd.suspiciousNodes {
		if record.SuspicionScore >= bd.config.SuspicionThreshold {
			result[id] = record
		}
	}
	return result
}

// GetByzantineNodes returns all confirmed Byzantine nodes
func (bd *ByzantineDetector) GetByzantineNodes() map[string]*ByzantineEvidence {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	result := make(map[string]*ByzantineEvidence)
	for id, evidence := range bd.confirmedBad {
		result[id] = evidence
	}
	return result
}

// recordViolation records a violation and updates suspicion score
func (bd *ByzantineDetector) recordViolation(nodeID string, violation *Violation) {
	record, exists := bd.suspiciousNodes[nodeID]
	if !exists {
		record = &SuspicionRecord{
			NodeID:        nodeID,
			Violations:    make([]*Violation, 0),
			FirstDetected: time.Now(),
		}
		bd.suspiciousNodes[nodeID] = record
	}

	record.Violations = append(record.Violations, violation)
	record.LastUpdated = time.Now()

	// Update suspicion score
	record.SuspicionScore = bd.calculateSuspicionScore(record)

	bd.logger.Warn("Violation recorded",
		zap.String("node_id", nodeID),
		zap.String("type", fmt.Sprintf("%v", violation.Type)),
		zap.Float64("severity", violation.Severity),
		zap.Float64("suspicion_score", record.SuspicionScore),
	)

	// Check if node should be marked as Byzantine
	if record.SuspicionScore >= bd.config.ByzantineThreshold {
		bd.confirmByzantine(nodeID, record)
	}

	// Update reputation system
	if bd.reputationSystem != nil {
		bd.reputationSystem.RecordByzantineBehavior(nodeID, violation.Severity)
	}
}

// calculateSuspicionScore calculates a node's suspicion score
func (bd *ByzantineDetector) calculateSuspicionScore(record *SuspicionRecord) float64 {
	if len(record.Violations) == 0 {
		return 0
	}

	var totalScore float64
	violationTypes := make(map[ViolationType]bool)

	// Recent violations weighted more heavily
	now := time.Now()
	for _, violation := range record.Violations {
		age := now.Sub(violation.Timestamp)
		if age < bd.config.BehaviorWindow {
			// Decay factor based on age
			decay := 1.0 - (float64(age) / float64(bd.config.BehaviorWindow))
			totalScore += violation.Severity * decay
			violationTypes[violation.Type] = true
		}
	}

	// Bonus for multiple violation types (indicates systematic misbehavior)
	if bd.config.RequireMultipleViolation && len(violationTypes) > 1 {
		totalScore *= 1.5
	}

	// Cap at 100
	if totalScore > 100 {
		totalScore = 100
	}

	return totalScore
}

// confirmByzantine marks a node as confirmed Byzantine
func (bd *ByzantineDetector) confirmByzantine(nodeID string, record *SuspicionRecord) {
	if _, exists := bd.confirmedBad[nodeID]; exists {
		return // Already confirmed
	}

	// Determine attack type
	attackType := bd.determineAttackType(record)

	evidence := &ByzantineEvidence{
		NodeID:     nodeID,
		AttackType: attackType,
		Evidence:   record.Violations,
		Confidence: record.SuspicionScore / 100.0,
		DetectedAt: time.Now(),
		Violations: record.Violations,
	}

	bd.confirmedBad[nodeID] = evidence

	bd.logger.Error("Byzantine node confirmed",
		zap.String("node_id", nodeID),
		zap.String("attack_type", fmt.Sprintf("%v", attackType)),
		zap.Float64("confidence", evidence.Confidence),
		zap.Int("violations", len(record.Violations)),
	)

	// Update reputation system to quarantine
	if bd.reputationSystem != nil {
		bd.reputationSystem.QuarantineNode(nodeID, fmt.Sprintf("Byzantine behavior detected: %v", attackType))
	}
}

// determineAttackType determines the primary attack type from violations
func (bd *ByzantineDetector) determineAttackType(record *SuspicionRecord) AttackType {
	typeCounts := make(map[ViolationType]int)
	for _, violation := range record.Violations {
		typeCounts[violation.Type]++
	}

	// Map violation types to attack types
	if typeCounts[ViolationEquivocation] > 0 || typeCounts[ViolationConflictingVote] > 0 {
		return AttackEquivocation
	}
	if typeCounts[ViolationInvalidSignature] > 0 {
		return AttackMessageTampering
	}
	if typeCounts[ViolationTimingAnomaly] > 0 {
		return AttackTimingManipulation
	}
	if typeCounts[ViolationFloodAttack] > 0 {
		return AttackDenialOfService
	}
	if typeCounts[ViolationViewChangeAbuse] > 0 {
		return AttackViewChangeAttack
	}

	return AttackEquivocation // Default
}

// getOrCreateBehavior gets or creates node behavior tracker
func (bd *ByzantineDetector) getOrCreateBehavior(nodeID string) *NodeBehavior {
	if behavior, exists := bd.nodeBehavior[nodeID]; exists {
		return behavior
	}

	behavior := &NodeBehavior{
		NodeID:        nodeID,
		ResponseTimes: make([]time.Duration, 0),
		LastSeen:      time.Now(),
	}
	bd.nodeBehavior[nodeID] = behavior
	return behavior
}

// getOrCreateVoteTracker gets or creates vote tracker
func (bd *ByzantineDetector) getOrCreateVoteTracker(key string, view, sequence int64) *VoteTracker {
	if tracker, exists := bd.consensusVotes[key]; exists {
		return tracker
	}

	tracker := &VoteTracker{
		Sequence:   sequence,
		View:       view,
		PrePrepare: make(map[string]string),
		Prepare:    make(map[string]string),
		Commit:     make(map[string]string),
	}
	bd.consensusVotes[key] = tracker
	return tracker
}

// computeDigest computes SHA-256 digest of a message
func (bd *ByzantineDetector) computeDigest(message interface{}) string {
	data := fmt.Sprintf("%v", message)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// computeStats computes mean and standard deviation of durations
func (bd *ByzantineDetector) computeStats(durations []time.Duration) (time.Duration, time.Duration) {
	if len(durations) == 0 {
		return 0, 0
	}

	var sum time.Duration
	for _, d := range durations {
		sum += d
	}
	mean := sum / time.Duration(len(durations))

	var variance float64
	for _, d := range durations {
		diff := float64(d - mean)
		variance += diff * diff
	}
	variance /= float64(len(durations))
	stdDev := time.Duration(variance)

	return mean, stdDev
}

// monitoringLoop periodically updates suspicion scores and cleans up old data
func (bd *ByzantineDetector) monitoringLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-bd.ctx.Done():
			return
		case <-ticker.C:
			bd.updateSuspicionScores()
			bd.cleanupOldData()
		}
	}
}

// updateSuspicionScores recalculates all suspicion scores
func (bd *ByzantineDetector) updateSuspicionScores() {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	for nodeID, record := range bd.suspiciousNodes {
		newScore := bd.calculateSuspicionScore(record)
		if newScore != record.SuspicionScore {
			record.SuspicionScore = newScore
			record.LastUpdated = time.Now()

			// Check if should be confirmed Byzantine
			if newScore >= bd.config.ByzantineThreshold {
				bd.confirmByzantine(nodeID, record)
			}
		}
	}
}

// cleanupOldData removes old records to prevent memory bloat
func (bd *ByzantineDetector) cleanupOldData() {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-bd.config.BehaviorWindow * 2)

	// Clean message history
	for id, record := range bd.messageHistory {
		if record.Timestamp.Before(cutoff) {
			delete(bd.messageHistory, id)
		}
	}

	// Clean consensus votes
	for key := range bd.consensusVotes {
		// Keep only recent sequences (arbitrary threshold)
		if time.Since(now).Hours() > 1 {
			delete(bd.consensusVotes, key)
		}
	}

	// Clean suspicion records with low scores
	for id, record := range bd.suspiciousNodes {
		if record.SuspicionScore < 10.0 && time.Since(record.LastUpdated) > bd.config.SuspicionDecay {
			delete(bd.suspiciousNodes, id)
		}
	}
}

// Stop stops the Byzantine detector
func (bd *ByzantineDetector) Stop() {
	bd.cancel()
}

// GetStats returns detection statistics
func (bd *ByzantineDetector) GetStats() map[string]interface{} {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	return map[string]interface{}{
		"total_nodes":       len(bd.nodeBehavior),
		"suspicious_nodes":  len(bd.suspiciousNodes),
		"byzantine_nodes":   len(bd.confirmedBad),
		"messages_tracked":  len(bd.messageHistory),
		"votes_tracked":     len(bd.consensusVotes),
	}
}
