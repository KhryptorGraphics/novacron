// Package tpbft implements trust management for T-PBFT
package tpbft

import (
	"sync"
	"time"
)

// InteractionType represents the type of node interaction
type InteractionType int

const (
	CorrectVote InteractionType = iota
	IncorrectVote
	TimelyResponse
	LateResponse
	ValidMessage
	InvalidMessage
	ByzantineBehavior
)

// TrustManager coordinates trust operations for T-PBFT
type TrustManager struct {
	mu              sync.RWMutex
	eigenTrust      *EigenTrust
	interactionLog  []Interaction
	updateInterval  time.Duration
	lastUpdate      time.Time
	autoUpdate      bool
	stopCh          chan struct{}

	// Statistics
	totalInteractions int
	byzantineDetected int
}

// Interaction represents a recorded interaction between nodes
type Interaction struct {
	From      string
	To        string
	Type      InteractionType
	Score     float64
	Timestamp time.Time
	Details   string
}

// NewTrustManager creates a new trust manager
func NewTrustManager() *TrustManager {
	return &TrustManager{
		eigenTrust:     NewEigenTrust(),
		interactionLog: make([]Interaction, 0),
		updateInterval: 30 * time.Second, // Recompute trust every 30s
		lastUpdate:     time.Now(),
		autoUpdate:     true,
		stopCh:         make(chan struct{}),
	}
}

// Start begins automatic trust score updates
func (tm *TrustManager) Start() {
	if !tm.autoUpdate {
		return
	}

	go tm.updateLoop()
}

// Stop halts automatic trust updates
func (tm *TrustManager) Stop() {
	close(tm.stopCh)
}

// updateLoop periodically recomputes global trust scores
func (tm *TrustManager) updateLoop() {
	ticker := time.NewTicker(tm.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			tm.RecomputeTrust()
		case <-tm.stopCh:
			return
		}
	}
}

// RecordInteraction logs an interaction and updates local trust
func (tm *TrustManager) RecordInteraction(from, to string, iType InteractionType, details string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Calculate trust score based on interaction type
	score := tm.calculateScore(iType)

	// Record interaction
	interaction := Interaction{
		From:      from,
		To:        to,
		Type:      iType,
		Score:     score,
		Timestamp: time.Now(),
		Details:   details,
	}
	tm.interactionLog = append(tm.interactionLog, interaction)
	tm.totalInteractions++

	// Update EigenTrust
	tm.eigenTrust.UpdateLocalTrust(from, to, score)

	// Detect Byzantine behavior
	if iType == ByzantineBehavior {
		tm.byzantineDetected++
		tm.handleByzantine(to)
	}
}

// calculateScore converts interaction type to trust score
func (tm *TrustManager) calculateScore(iType InteractionType) float64 {
	switch iType {
	case CorrectVote:
		return 1.0
	case IncorrectVote:
		return 0.3
	case TimelyResponse:
		return 0.9
	case LateResponse:
		return 0.5
	case ValidMessage:
		return 0.8
	case InvalidMessage:
		return 0.2
	case ByzantineBehavior:
		return 0.0
	default:
		return 0.5
	}
}

// handleByzantine takes action when Byzantine behavior is detected
func (tm *TrustManager) handleByzantine(nodeID string) {
	// Immediately set trust to zero
	tm.eigenTrust.UpdateLocalTrust("system", nodeID, 0.0)

	// Mark in interaction log
	tm.interactionLog = append(tm.interactionLog, Interaction{
		From:      "system",
		To:        nodeID,
		Type:      ByzantineBehavior,
		Score:     0.0,
		Timestamp: time.Now(),
		Details:   "Byzantine behavior detected",
	})
}

// RecomputeTrust triggers global trust score recomputation
func (tm *TrustManager) RecomputeTrust() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.eigenTrust.ComputeGlobalTrust()
	tm.lastUpdate = time.Now()
}

// GetTrustScore returns the current trust score for a node
func (tm *TrustManager) GetTrustScore(nodeID string) float64 {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.eigenTrust.GetTrustScore(nodeID)
}

// GetTopTrustedNodes returns the N most trusted nodes
func (tm *TrustManager) GetTopTrustedNodes(n int) []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.eigenTrust.GetTopNodes(n)
}

// SetPreTrustedNode sets a node as pre-trusted (for bootstrapping)
func (tm *TrustManager) SetPreTrustedNode(nodeID string, trust float64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.eigenTrust.SetPreTrust(nodeID, trust)
}

// GetInteractionHistory returns recent interactions for a node
func (tm *TrustManager) GetInteractionHistory(nodeID string, limit int) []Interaction {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	var history []Interaction
	count := 0

	// Reverse iteration to get most recent first
	for i := len(tm.interactionLog) - 1; i >= 0 && count < limit; i-- {
		if tm.interactionLog[i].From == nodeID || tm.interactionLog[i].To == nodeID {
			history = append(history, tm.interactionLog[i])
			count++
		}
	}

	return history
}

// GetStats returns trust management statistics
func (tm *TrustManager) GetStats() map[string]interface{} {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	eigenStats := tm.eigenTrust.GetStats()

	return map[string]interface{}{
		"total_interactions":  tm.totalInteractions,
		"byzantine_detected":  tm.byzantineDetected,
		"last_update":         tm.lastUpdate,
		"update_interval_sec": tm.updateInterval.Seconds(),
		"eigen_trust":         eigenStats,
	}
}

// PruneOldInteractions removes interactions older than specified duration
func (tm *TrustManager) PruneOldInteractions(maxAge time.Duration) int {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)
	pruned := 0
	newLog := make([]Interaction, 0)

	for _, interaction := range tm.interactionLog {
		if interaction.Timestamp.After(cutoff) {
			newLog = append(newLog, interaction)
		} else {
			pruned++
		}
	}

	tm.interactionLog = newLog
	return pruned
}

// ExportTrustData exports trust scores for analysis
func (tm *TrustManager) ExportTrustData() map[string]float64 {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	scores := make(map[string]float64)
	nodes := tm.eigenTrust.getAllNodes()

	for _, node := range nodes {
		scores[node] = tm.eigenTrust.GetTrustScore(node)
	}

	return scores
}

// ValidateCommittee checks if proposed committee members have sufficient trust
func (tm *TrustManager) ValidateCommittee(members []string, minTrust float64) (bool, []string) {
	// Compute trust first (needs write access)
	tm.RecomputeTrust()

	// Now validate with read lock
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	invalid := make([]string, 0)

	for _, member := range members {
		trust := tm.eigenTrust.GetTrustScore(member)
		if trust < minTrust {
			invalid = append(invalid, member)
		}
	}

	return len(invalid) == 0, invalid
}

// AdaptiveTrustThreshold calculates dynamic trust threshold based on network conditions
func (tm *TrustManager) AdaptiveTrustThreshold() float64 {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	stats := tm.eigenTrust.GetStats()
	avgTrust := stats["avg_trust"].(float64)

	// Adaptive threshold: 80% of average trust, minimum 0.5
	threshold := avgTrust * 0.8
	if threshold < 0.5 {
		threshold = 0.5
	}

	return threshold
}

// ReputationDecay applies time-based decay to inactive nodes
func (tm *TrustManager) ReputationDecay(decayRate float64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Get all nodes
	nodes := tm.eigenTrust.getAllNodes()

	// Find inactive nodes (no recent interactions)
	cutoff := time.Now().Add(-5 * time.Minute)
	activeNodes := make(map[string]bool)

	for _, interaction := range tm.interactionLog {
		if interaction.Timestamp.After(cutoff) {
			activeNodes[interaction.From] = true
			activeNodes[interaction.To] = true
		}
	}

	// Apply decay to inactive nodes
	for _, node := range nodes {
		if !activeNodes[node] {
			currentTrust := tm.eigenTrust.GetTrustScore(node)
			newTrust := currentTrust * (1.0 - decayRate)
			tm.eigenTrust.UpdateLocalTrust("system", node, newTrust)
		}
	}
}
