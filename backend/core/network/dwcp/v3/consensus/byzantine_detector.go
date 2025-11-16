package consensus

import (
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ByzantineDetector detects and tracks Byzantine node behavior
type ByzantineDetector struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Node behavior tracking
	nodeStats  map[string]*NodeBehaviorStats
	suspicions map[string]int64 // Node ID -> suspicion count

	// Detection thresholds
	failureThreshold       int64   // Failures before marking Byzantine
	inconsistencyThreshold float64 // Inconsistency ratio threshold
	responseTimeThreshold  time.Duration

	// Quarantine management
	quarantinedNodes   map[string]time.Time // Node ID -> quarantine time
	quarantineDuration time.Duration

	// Metrics
	byzantineDetected int64
	falsePositives    int64
	detectionAccuracy float64
}

// NodeBehaviorStats tracks behavior metrics for a node
type NodeBehaviorStats struct {
	NodeID              string
	TotalMessages       int64
	FailedMessages      int64
	InconsistentVotes   int64
	AverageResponseTime time.Duration
	LastSeen            time.Time
	SuspicionLevel      int64 // 0-100
	IsQuarantined       bool
}

// ByzantineBehavior represents detected Byzantine behavior
type ByzantineBehavior struct {
	NodeID    string
	Type      string // "equivocation", "inconsistency", "timeout", "malformed"
	Severity  int    // 1-10
	Evidence  string
	Timestamp time.Time
}

// NewByzantineDetector creates a new Byzantine detector
func NewByzantineDetector(logger *zap.Logger) *ByzantineDetector {
	return &ByzantineDetector{
		logger:                 logger,
		nodeStats:              make(map[string]*NodeBehaviorStats),
		suspicions:             make(map[string]int64),
		quarantinedNodes:       make(map[string]time.Time),
		failureThreshold:       5,
		inconsistencyThreshold: 0.3, // 30% inconsistency
		responseTimeThreshold:  5 * time.Second,
		quarantineDuration:     5 * time.Minute,
	}
}

// RecordMessageFailure records a failed message from a node
func (bd *ByzantineDetector) RecordMessageFailure(nodeID string) {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	stats := bd.getOrCreateStats(nodeID)
	stats.TotalMessages++
	stats.FailedMessages++
	stats.LastSeen = time.Now()

	// Update suspicion level
	failureRatio := float64(stats.FailedMessages) / float64(stats.TotalMessages)
	if failureRatio > 0.5 {
		bd.suspicions[nodeID]++
	}

	// Check if should be quarantined
	if bd.suspicions[nodeID] >= bd.failureThreshold {
		bd.quarantineNode(nodeID)
	}
}

// RecordInconsistentVote records an inconsistent vote from a node
func (bd *ByzantineDetector) RecordInconsistentVote(nodeID string) {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	stats := bd.getOrCreateStats(nodeID)
	stats.InconsistentVotes++
	stats.LastSeen = time.Now()

	// Update suspicion level
	inconsistencyRatio := float64(stats.InconsistentVotes) / float64(stats.TotalMessages)
	if inconsistencyRatio > bd.inconsistencyThreshold {
		bd.suspicions[nodeID]++
	}

	// Check if should be quarantined
	if bd.suspicions[nodeID] >= bd.failureThreshold {
		bd.quarantineNode(nodeID)
	}
}

// RecordResponseTime records response time for a node
func (bd *ByzantineDetector) RecordResponseTime(nodeID string, duration time.Duration) {
	bd.mu.Lock()
	defer bd.mu.Unlock()

	stats := bd.getOrCreateStats(nodeID)
	stats.AverageResponseTime = (stats.AverageResponseTime + duration) / 2
	stats.LastSeen = time.Now()

	// Check for timeout behavior
	if duration > bd.responseTimeThreshold {
		bd.suspicions[nodeID]++
	}

	// Check if should be quarantined
	if bd.suspicions[nodeID] >= bd.failureThreshold {
		bd.quarantineNode(nodeID)
	}
}

// quarantineNode marks a node as quarantined
func (bd *ByzantineDetector) quarantineNode(nodeID string) {
	bd.quarantinedNodes[nodeID] = time.Now().Add(bd.quarantineDuration)
	stats := bd.getOrCreateStats(nodeID)
	stats.IsQuarantined = true
	bd.byzantineDetected++

	bd.logger.Warn("Node quarantined for Byzantine behavior",
		zap.String("node_id", nodeID),
		zap.Int64("suspicion_level", bd.suspicions[nodeID]),
		zap.Duration("quarantine_duration", bd.quarantineDuration))
}

// IsQuarantined checks if a node is currently quarantined
func (bd *ByzantineDetector) IsQuarantined(nodeID string) bool {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	quarantineTime, exists := bd.quarantinedNodes[nodeID]
	if !exists {
		return false
	}

	// Check if quarantine has expired
	if time.Now().After(quarantineTime) {
		return false
	}

	return true
}

// GetNodeStats returns behavior statistics for a node
func (bd *ByzantineDetector) GetNodeStats(nodeID string) *NodeBehaviorStats {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	if stats, exists := bd.nodeStats[nodeID]; exists {
		return stats
	}
	return nil
}

// GetSuspicionLevel returns suspicion level for a node (0-100)
func (bd *ByzantineDetector) GetSuspicionLevel(nodeID string) int {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	suspicion := bd.suspicions[nodeID]
	level := int((suspicion * 100) / bd.failureThreshold)
	if level > 100 {
		level = 100
	}
	return level
}

// getOrCreateStats gets or creates stats for a node
func (bd *ByzantineDetector) getOrCreateStats(nodeID string) *NodeBehaviorStats {
	if stats, exists := bd.nodeStats[nodeID]; exists {
		return stats
	}

	stats := &NodeBehaviorStats{
		NodeID:   nodeID,
		LastSeen: time.Now(),
	}
	bd.nodeStats[nodeID] = stats
	return stats
}

// GetMetrics returns detection metrics
func (bd *ByzantineDetector) GetMetrics() map[string]interface{} {
	bd.mu.RLock()
	defer bd.mu.RUnlock()

	totalNodes := int64(len(bd.nodeStats))
	quarantinedCount := int64(len(bd.quarantinedNodes))

	return map[string]interface{}{
		"total_nodes":        totalNodes,
		"byzantine_detected": atomic.LoadInt64(&bd.byzantineDetected),
		"quarantined_nodes":  quarantinedCount,
		"false_positives":    atomic.LoadInt64(&bd.falsePositives),
		"detection_accuracy": bd.detectionAccuracy,
	}
}
