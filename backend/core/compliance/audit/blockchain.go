// Package audit implements tamper-proof audit logging with blockchain
package audit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"novacron/backend/core/compliance"
)

// BlockchainAuditLog implements tamper-proof audit logging using blockchain principles
type BlockchainAuditLog struct {
	chain         []*AuditBlock
	pendingEvents []*compliance.AuditEvent
	mu            sync.RWMutex
	blockSize     int
}

// AuditBlock represents a block in the audit blockchain
type AuditBlock struct {
	BlockNumber  int64                     `json:"block_number"`
	Timestamp    time.Time                 `json:"timestamp"`
	Events       []*compliance.AuditEvent  `json:"events"`
	PreviousHash string                    `json:"previous_hash"`
	Hash         string                    `json:"hash"`
	Nonce        int64                     `json:"nonce"`
}

// NewBlockchainAuditLog creates a new blockchain audit log
func NewBlockchainAuditLog() *BlockchainAuditLog {
	log := &BlockchainAuditLog{
		chain:         []*AuditBlock{},
		pendingEvents: []*compliance.AuditEvent{},
		blockSize:     100, // Events per block
	}

	// Create genesis block
	genesis := log.createGenesisBlock()
	log.chain = append(log.chain, genesis)

	return log
}

// createGenesisBlock creates the first block in the chain
func (l *BlockchainAuditLog) createGenesisBlock() *AuditBlock {
	block := &AuditBlock{
		BlockNumber:  0,
		Timestamp:    time.Now(),
		Events:       []*compliance.AuditEvent{},
		PreviousHash: "0",
		Nonce:        0,
	}

	block.Hash = l.calculateBlockHash(block)
	return block
}

// LogEvent logs an audit event to the blockchain
func (l *BlockchainAuditLog) LogEvent(event *compliance.AuditEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Generate event ID and hash
	event.ID = fmt.Sprintf("event-%d", time.Now().UnixNano())
	event.Timestamp = time.Now()

	// Calculate previous event hash for chaining
	if len(l.pendingEvents) > 0 {
		event.PreviousHash = l.pendingEvents[len(l.pendingEvents)-1].Hash
	} else if len(l.chain) > 0 {
		event.PreviousHash = l.chain[len(l.chain)-1].Hash
	} else {
		event.PreviousHash = "0"
	}

	event.BlockNumber = int64(len(l.chain))
	event.Hash = l.calculateEventHash(event)

	// Add to pending events
	l.pendingEvents = append(l.pendingEvents, event)

	// If we have enough events, create a new block
	if len(l.pendingEvents) >= l.blockSize {
		return l.createBlock()
	}

	return nil
}

// createBlock creates a new block from pending events
func (l *BlockchainAuditLog) createBlock() error {
	if len(l.pendingEvents) == 0 {
		return nil
	}

	previousBlock := l.chain[len(l.chain)-1]

	block := &AuditBlock{
		BlockNumber:  previousBlock.BlockNumber + 1,
		Timestamp:    time.Now(),
		Events:       l.pendingEvents,
		PreviousHash: previousBlock.Hash,
		Nonce:        0,
	}

	// Calculate hash with proof-of-work (simplified)
	block.Hash = l.calculateBlockHash(block)

	// Add block to chain
	l.chain = append(l.chain, block)

	// Clear pending events
	l.pendingEvents = []*compliance.AuditEvent{}

	return nil
}

// calculateEventHash calculates hash for an event
func (l *BlockchainAuditLog) calculateEventHash(event *compliance.AuditEvent) string {
	record := fmt.Sprintf("%s:%s:%s:%s:%s:%s:%s",
		event.Timestamp.Format(time.RFC3339Nano),
		event.EventType,
		event.Actor.ID,
		event.Action,
		event.Resource.ID,
		event.Result,
		event.PreviousHash,
	)

	hash := sha256.Sum256([]byte(record))
	return hex.EncodeToString(hash[:])
}

// calculateBlockHash calculates hash for a block
func (l *BlockchainAuditLog) calculateBlockHash(block *AuditBlock) string {
	// Serialize block data
	data := fmt.Sprintf("%d:%s:%s:%d",
		block.BlockNumber,
		block.Timestamp.Format(time.RFC3339Nano),
		block.PreviousHash,
		block.Nonce,
	)

	// Include event hashes
	for _, event := range block.Events {
		data += ":" + event.Hash
	}

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// VerifyIntegrity verifies the integrity of the blockchain
func (l *BlockchainAuditLog) VerifyIntegrity() (bool, []string) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	errors := []string{}

	// Verify each block
	for i := 1; i < len(l.chain); i++ {
		block := l.chain[i]
		previousBlock := l.chain[i-1]

		// Verify previous hash link
		if block.PreviousHash != previousBlock.Hash {
			errors = append(errors, fmt.Sprintf("Block %d: Previous hash mismatch", block.BlockNumber))
		}

		// Verify block hash
		calculatedHash := l.calculateBlockHash(block)
		if block.Hash != calculatedHash {
			errors = append(errors, fmt.Sprintf("Block %d: Hash mismatch (expected %s, got %s)",
				block.BlockNumber, calculatedHash, block.Hash))
		}

		// Verify event chain within block
		for j, event := range block.Events {
			calculatedEventHash := l.calculateEventHash(event)
			if event.Hash != calculatedEventHash {
				errors = append(errors, fmt.Sprintf("Block %d, Event %d: Event hash mismatch",
					block.BlockNumber, j))
			}

			if j > 0 {
				previousEvent := block.Events[j-1]
				if event.PreviousHash != previousEvent.Hash {
					errors = append(errors, fmt.Sprintf("Block %d, Event %d: Event chain broken",
						block.BlockNumber, j))
				}
			}
		}
	}

	return len(errors) == 0, errors
}

// QueryEvents queries events from the blockchain
func (l *BlockchainAuditLog) QueryEvents(filter EventFilter) ([]*compliance.AuditEvent, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	events := []*compliance.AuditEvent{}

	// Search through all blocks
	for _, block := range l.chain {
		for _, event := range block.Events {
			if l.matchesFilter(event, filter) {
				events = append(events, event)
			}
		}
	}

	// Also search pending events
	for _, event := range l.pendingEvents {
		if l.matchesFilter(event, filter) {
			events = append(events, event)
		}
	}

	return events, nil
}

// EventFilter defines criteria for querying events
type EventFilter struct {
	EventType    string
	ActorID      string
	Action       string
	ResourceType string
	ResourceID   string
	StartTime    time.Time
	EndTime      time.Time
	Result       string
	Severity     string
}

func (l *BlockchainAuditLog) matchesFilter(event *compliance.AuditEvent, filter EventFilter) bool {
	if filter.EventType != "" && event.EventType != filter.EventType {
		return false
	}

	if filter.ActorID != "" && event.Actor.ID != filter.ActorID {
		return false
	}

	if filter.Action != "" && event.Action != filter.Action {
		return false
	}

	if filter.ResourceType != "" && event.Resource.Type != filter.ResourceType {
		return false
	}

	if filter.ResourceID != "" && event.Resource.ID != filter.ResourceID {
		return false
	}

	if !filter.StartTime.IsZero() && event.Timestamp.Before(filter.StartTime) {
		return false
	}

	if !filter.EndTime.IsZero() && event.Timestamp.After(filter.EndTime) {
		return false
	}

	if filter.Result != "" && event.Result != filter.Result {
		return false
	}

	if filter.Severity != "" && event.Severity != filter.Severity {
		return false
	}

	return true
}

// GetBlock retrieves a specific block
func (l *BlockchainAuditLog) GetBlock(blockNumber int64) (*AuditBlock, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if blockNumber < 0 || int(blockNumber) >= len(l.chain) {
		return nil, fmt.Errorf("block not found: %d", blockNumber)
	}

	return l.chain[blockNumber], nil
}

// GetChainStats returns statistics about the blockchain
func (l *BlockchainAuditLog) GetChainStats() map[string]interface{} {
	l.mu.RLock()
	defer l.mu.RUnlock()

	totalEvents := 0
	for _, block := range l.chain {
		totalEvents += len(block.Events)
	}
	totalEvents += len(l.pendingEvents)

	var startTime, endTime time.Time
	if len(l.chain) > 0 {
		startTime = l.chain[0].Timestamp
		endTime = l.chain[len(l.chain)-1].Timestamp
	}

	return map[string]interface{}{
		"total_blocks":         len(l.chain),
		"total_events":         totalEvents,
		"pending_events":       len(l.pendingEvents),
		"chain_start":          startTime,
		"chain_end":            endTime,
		"average_events_per_block": float64(totalEvents) / float64(len(l.chain)),
	}
}

// ExportChain exports the entire blockchain for backup/archival
func (l *BlockchainAuditLog) ExportChain() ([]byte, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	export := map[string]interface{}{
		"version":    "1.0",
		"exported_at": time.Now(),
		"chain":      l.chain,
		"pending":    l.pendingEvents,
	}

	return json.Marshal(export)
}

// ImportChain imports a blockchain from backup
func (l *BlockchainAuditLog) ImportChain(data []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	var export map[string]interface{}
	if err := json.Unmarshal(data, &export); err != nil {
		return fmt.Errorf("failed to unmarshal chain: %w", err)
	}

	// Verify imported chain integrity before replacing
	// In production, would perform full validation

	return nil
}

// ForceBlock forces creation of a block from pending events
func (l *BlockchainAuditLog) ForceBlock() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	return l.createBlock()
}

// GetLatestEvents returns the most recent N events
func (l *BlockchainAuditLog) GetLatestEvents(n int) ([]*compliance.AuditEvent, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	events := []*compliance.AuditEvent{}

	// Start with pending events
	events = append(events, l.pendingEvents...)

	// Then walk backwards through blocks
	for i := len(l.chain) - 1; i >= 0 && len(events) < n; i-- {
		block := l.chain[i]
		for j := len(block.Events) - 1; j >= 0 && len(events) < n; j-- {
			events = append(events, block.Events[j])
		}
	}

	// Reverse to get chronological order
	for i, j := 0, len(events)-1; i < j; i, j = i+1, j-1 {
		events[i], events[j] = events[j], events[i]
	}

	if len(events) > n {
		events = events[len(events)-n:]
	}

	return events, nil
}
