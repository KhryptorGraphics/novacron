package consensus

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ThreatLevel represents the Byzantine threat level
type ThreatLevel int

const (
	ThreatLevelLow ThreatLevel = iota
	ThreatLevelMedium
	ThreatLevelHigh
	ThreatLevelCritical
)

// String returns the string representation of threat level
func (tl ThreatLevel) String() string {
	switch tl {
	case ThreatLevelLow:
		return "LOW"
	case ThreatLevelMedium:
		return "MEDIUM"
	case ThreatLevelHigh:
		return "HIGH"
	case ThreatLevelCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// ThreatAnalyzer analyzes Byzantine threat level and recommends protocols
type ThreatAnalyzer struct {
	mu sync.RWMutex

	logger *zap.Logger

	// Threat assessment
	currentThreatLevel ThreatLevel
	byzantineNodeCount int64
	totalNodeCount     int64

	// Protocol recommendations
	recommendedProtocol string
	protocolHistory     []ProtocolRecommendation

	// Thresholds
	lowThreatThreshold      float64 // 0-10% Byzantine
	mediumThreatThreshold   float64 // 10-20% Byzantine
	highThreatThreshold     float64 // 20-33% Byzantine
	criticalThreatThreshold float64 // >33% Byzantine

	// Metrics
	threatAssessments int64
	protocolSwitches  int64
}

// ProtocolRecommendation represents a protocol recommendation
type ProtocolRecommendation struct {
	Timestamp       time.Time
	ThreatLevel     ThreatLevel
	RecommendedProtocol string
	Reason          string
	ByzantineRatio  float64
}

// NewThreatAnalyzer creates a new threat analyzer
func NewThreatAnalyzer(logger *zap.Logger) *ThreatAnalyzer {
	return &ThreatAnalyzer{
		logger:                  logger,
		currentThreatLevel:      ThreatLevelLow,
		recommendedProtocol:     "raft", // Default to Raft
		protocolHistory:         make([]ProtocolRecommendation, 0),
		lowThreatThreshold:      0.10,   // 10%
		mediumThreatThreshold:   0.20,   // 20%
		highThreatThreshold:     0.33,   // 33%
		criticalThreatThreshold: 0.40,   // 40% (beyond tolerance)
	}
}

// UpdateThreatAssessment updates threat level based on Byzantine node count
func (ta *ThreatAnalyzer) UpdateThreatAssessment(byzantineCount, totalCount int64) {
	ta.mu.Lock()
	defer ta.mu.Unlock()

	ta.byzantineNodeCount = byzantineCount
	ta.totalNodeCount = totalCount
	ta.threatAssessments++

	// Calculate Byzantine ratio
	var byzantineRatio float64
	if totalCount > 0 {
		byzantineRatio = float64(byzantineCount) / float64(totalCount)
	}

	// Determine threat level
	oldThreatLevel := ta.currentThreatLevel
	ta.currentThreatLevel = ta.calculateThreatLevel(byzantineRatio)

	// Get protocol recommendation
	protocol := ta.getProtocolRecommendation(ta.currentThreatLevel)

	// Check if protocol needs to change
	if protocol != ta.recommendedProtocol {
		ta.protocolSwitches++
		ta.logger.Info("Protocol switch recommended",
			zap.String("old_protocol", ta.recommendedProtocol),
			zap.String("new_protocol", protocol),
			zap.String("threat_level", ta.currentThreatLevel.String()),
			zap.Float64("byzantine_ratio", byzantineRatio))
	}

	ta.recommendedProtocol = protocol

	// Record recommendation
	recommendation := ProtocolRecommendation{
		Timestamp:           time.Now(),
		ThreatLevel:         ta.currentThreatLevel,
		RecommendedProtocol: protocol,
		ByzantineRatio:      byzantineRatio,
		Reason:              ta.getThreatReason(oldThreatLevel, ta.currentThreatLevel, byzantineRatio),
	}
	ta.protocolHistory = append(ta.protocolHistory, recommendation)

	// Keep history limited to last 100 entries
	if len(ta.protocolHistory) > 100 {
		ta.protocolHistory = ta.protocolHistory[1:]
	}
}

// calculateThreatLevel determines threat level based on Byzantine ratio
func (ta *ThreatAnalyzer) calculateThreatLevel(byzantineRatio float64) ThreatLevel {
	switch {
	case byzantineRatio <= ta.lowThreatThreshold:
		return ThreatLevelLow
	case byzantineRatio <= ta.mediumThreatThreshold:
		return ThreatLevelMedium
	case byzantineRatio <= ta.highThreatThreshold:
		return ThreatLevelHigh
	default:
		return ThreatLevelCritical
	}
}

// getProtocolRecommendation returns recommended protocol for threat level
func (ta *ThreatAnalyzer) getProtocolRecommendation(threatLevel ThreatLevel) string {
	switch threatLevel {
	case ThreatLevelLow:
		return "raft" // Fast, trusted environment
	case ThreatLevelMedium:
		return "pbft" // Byzantine tolerance needed
	case ThreatLevelHigh:
		return "probft" // Enhanced Byzantine tolerance
	case ThreatLevelCritical:
		return "tpbft" // Maximum Byzantine tolerance with reputation
	default:
		return "raft"
	}
}

// getThreatReason returns explanation for threat level change
func (ta *ThreatAnalyzer) getThreatReason(oldLevel, newLevel ThreatLevel, ratio float64) string {
	if oldLevel == newLevel {
		return fmt.Sprintf("Threat level stable at %s (%.1f%% Byzantine)", newLevel.String(), ratio*100)
	}
	return fmt.Sprintf("Threat level changed from %s to %s (%.1f%% Byzantine)", oldLevel.String(), newLevel.String(), ratio*100)
}

// GetCurrentThreatLevel returns current threat level
func (ta *ThreatAnalyzer) GetCurrentThreatLevel() ThreatLevel {
	ta.mu.RLock()
	defer ta.mu.RUnlock()
	return ta.currentThreatLevel
}

// GetRecommendedProtocol returns recommended consensus protocol
func (ta *ThreatAnalyzer) GetRecommendedProtocol() string {
	ta.mu.RLock()
	defer ta.mu.RUnlock()
	return ta.recommendedProtocol
}

// GetByzantineRatio returns current Byzantine node ratio
func (ta *ThreatAnalyzer) GetByzantineRatio() float64 {
	ta.mu.RLock()
	defer ta.mu.RUnlock()

	if ta.totalNodeCount == 0 {
		return 0
	}
	return float64(ta.byzantineNodeCount) / float64(ta.totalNodeCount)
}

// GetMetrics returns threat analysis metrics
func (ta *ThreatAnalyzer) GetMetrics() map[string]interface{} {
	ta.mu.RLock()
	defer ta.mu.RUnlock()

	byzantineRatio := 0.0
	if ta.totalNodeCount > 0 {
		byzantineRatio = float64(ta.byzantineNodeCount) / float64(ta.totalNodeCount)
	}

	return map[string]interface{}{
		"current_threat_level":    ta.currentThreatLevel.String(),
		"recommended_protocol":    ta.recommendedProtocol,
		"byzantine_node_count":    ta.byzantineNodeCount,
		"total_node_count":        ta.totalNodeCount,
		"byzantine_ratio":         byzantineRatio,
		"threat_assessments":      ta.threatAssessments,
		"protocol_switches":       ta.protocolSwitches,
		"protocol_history_length": len(ta.protocolHistory),
	}
}

// GetProtocolHistory returns recent protocol recommendations
func (ta *ThreatAnalyzer) GetProtocolHistory(limit int) []ProtocolRecommendation {
	ta.mu.RLock()
	defer ta.mu.RUnlock()

	if limit <= 0 || limit > len(ta.protocolHistory) {
		limit = len(ta.protocolHistory)
	}

	history := make([]ProtocolRecommendation, limit)
	copy(history, ta.protocolHistory[len(ta.protocolHistory)-limit:])
	return history
}

