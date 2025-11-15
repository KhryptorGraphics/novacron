package sync

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"math/big"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
	"go.uber.org/zap"
)

// AntiEntropyService implements the anti-entropy protocol for eventual consistency
type AntiEntropyService struct {
	engine   *ASSEngine
	interval time.Duration
	ctx      context.Context
	cancel   context.CancelFunc
	logger   *zap.Logger
	stats    *AntiEntropyStats
	statsMu  sync.RWMutex
}

// AntiEntropyStats tracks anti-entropy statistics
type AntiEntropyStats struct {
	TotalSyncs      int64         `json:"total_syncs"`
	SuccessfulSyncs int64         `json:"successful_syncs"`
	FailedSyncs     int64         `json:"failed_syncs"`
	LastSyncTime    time.Time     `json:"last_sync_time"`
	AverageSyncTime time.Duration `json:"average_sync_time"`
	TotalSyncTime   time.Duration `json:"total_sync_time"`
	KeysExchanged   int64         `json:"keys_exchanged"`
}

// NewAntiEntropyService creates a new anti-entropy service
func NewAntiEntropyService(engine *ASSEngine, interval time.Duration, logger *zap.Logger) *AntiEntropyService {
	ctx, cancel := context.WithCancel(context.Background())

	return &AntiEntropyService{
		engine:   engine,
		interval: interval,
		ctx:      ctx,
		cancel:   cancel,
		logger:   logger,
		stats:    &AntiEntropyStats{},
	}
}

// Start starts the anti-entropy service
func (aes *AntiEntropyService) Start() {
	go aes.antiEntropyLoop()
	aes.logger.Info("Anti-entropy service started", zap.Duration("interval", aes.interval))
}

// Stop stops the anti-entropy service
func (aes *AntiEntropyService) Stop() {
	aes.cancel()
	aes.logger.Info("Anti-entropy service stopped")
}

func (aes *AntiEntropyService) antiEntropyLoop() {
	ticker := time.NewTicker(aes.interval)
	defer ticker.Stop()

	for {
		select {
		case <-aes.ctx.Done():
			return
		case <-ticker.C:
			aes.performAntiEntropyRound()
		}
	}
}

func (aes *AntiEntropyService) performAntiEntropyRound() {
	// Select a random peer
	peer := aes.selectRandomPeer()
	if peer == nil {
		return
	}

	// Perform three-way handshake
	aes.threeWayHandshake(peer)
}

// ThreeWayHandshake performs a three-way anti-entropy handshake with a peer
func (aes *AntiEntropyService) threeWayHandshake(peer *RegionPeer) {
	startTime := time.Now()

	aes.statsMu.Lock()
	aes.stats.TotalSyncs++
	aes.statsMu.Unlock()

	aes.logger.Debug("Starting anti-entropy handshake",
		zap.String("peer_id", peer.ID),
		zap.String("region", peer.Region))

	// Phase 1: Send local digest and receive remote digest
	localDigest := aes.engine.crdtStore.Digest(aes.engine.nodeID, aes.engine.vectorClock.Get())
	remoteDigest, err := aes.exchangeDigests(peer, localDigest)
	if err != nil {
		aes.recordFailure(err)
		return
	}

	// Phase 2: Compute delta and exchange missing/stale data
	delta := aes.engine.computeDelta(localDigest, remoteDigest)
	keysExchanged := len(delta.Missing) + len(delta.Theirs) + len(delta.Stale)

	if keysExchanged > 0 {
		if err := aes.exchangeDeltas(peer, delta); err != nil {
			aes.recordFailure(err)
			return
		}
	}

	// Phase 3: Acknowledge synchronization
	if err := aes.sendAck(peer); err != nil {
		aes.recordFailure(err)
		return
	}

	// Record success
	syncDuration := time.Since(startTime)
	aes.recordSuccess(syncDuration, int64(keysExchanged))

	aes.logger.Info("Completed anti-entropy handshake",
		zap.String("peer_id", peer.ID),
		zap.Duration("duration", syncDuration),
		zap.Int("keys_exchanged", keysExchanged))
}

func (aes *AntiEntropyService) exchangeDigests(peer *RegionPeer, localDigest *Digest) (*Digest, error) {
	// Serialize local digest
	digestData, err := json.Marshal(localDigest)
	if err != nil {
		return nil, err
	}

	// Send digest request
	message := &Message{
		ID:          generateMessageID(),
		Type:        "ANTI_ENTROPY_DIGEST",
		SenderID:    aes.engine.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: aes.engine.vectorClock.Get(),
		Payload:     digestData,
		Timestamp:   time.Now(),
	}

	if err := aes.engine.transport.Send(peer, message); err != nil {
		return nil, err
	}

	// Receive remote digest
	response, err := aes.engine.transport.Receive()
	if err != nil {
		return nil, err
	}

	var remoteDigest Digest
	if err := json.Unmarshal(response.Payload, &remoteDigest); err != nil {
		return nil, err
	}

	return &remoteDigest, nil
}

func (aes *AntiEntropyService) exchangeDeltas(peer *RegionPeer, delta *Delta) error {
	// Request missing states from peer
	if len(delta.Missing) > 0 || len(delta.Stale) > 0 {
		requestKeys := append(delta.Missing, delta.Stale...)
		remoteStates, err := aes.engine.requestStates(peer, requestKeys)
		if err != nil {
			return err
		}

		if err := aes.engine.mergeStates(remoteStates); err != nil {
			return err
		}
	}

	// Send our states that peer is missing
	if len(delta.Theirs) > 0 {
		localStates := aes.engine.getStates(delta.Theirs)
		if err := aes.engine.sendStates(peer, localStates); err != nil {
			return err
		}
	}

	return nil
}

func (aes *AntiEntropyService) sendAck(peer *RegionPeer) error {
	message := &Message{
		ID:          generateMessageID(),
		Type:        "ANTI_ENTROPY_ACK",
		SenderID:    aes.engine.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: aes.engine.vectorClock.Get(),
		Timestamp:   time.Now(),
	}

	return aes.engine.transport.Send(peer, message)
}

func (aes *AntiEntropyService) selectRandomPeer() *RegionPeer {
	aes.engine.mu.RLock()
	defer aes.engine.mu.RUnlock()

	if len(aes.engine.regions) == 0 {
		return nil
	}

	peers := make([]*RegionPeer, 0, len(aes.engine.regions))
	for _, peer := range aes.engine.regions {
		peers = append(peers, peer)
	}

	// Select random peer
	idx, _ := rand.Int(rand.Reader, big.NewInt(int64(len(peers))))
	return peers[idx.Int64()]
}

func (aes *AntiEntropyService) recordSuccess(syncTime time.Duration, keysExchanged int64) {
	aes.statsMu.Lock()
	defer aes.statsMu.Unlock()

	aes.stats.SuccessfulSyncs++
	aes.stats.LastSyncTime = time.Now()
	aes.stats.TotalSyncTime += syncTime
	aes.stats.KeysExchanged += keysExchanged

	if aes.stats.SuccessfulSyncs > 0 {
		aes.stats.AverageSyncTime = aes.stats.TotalSyncTime / time.Duration(aes.stats.SuccessfulSyncs)
	}
}

func (aes *AntiEntropyService) recordFailure(err error) {
	aes.statsMu.Lock()
	defer aes.statsMu.Unlock()

	aes.stats.FailedSyncs++

	aes.logger.Error("Anti-entropy sync failed",
		zap.Error(err),
		zap.Int64("total_failures", aes.stats.FailedSyncs))
}

// GetStats returns anti-entropy statistics
func (aes *AntiEntropyService) GetStats() AntiEntropyStats {
	aes.statsMu.RLock()
	defer aes.statsMu.RUnlock()

	return *aes.stats
}

// ResetStats resets anti-entropy statistics
func (aes *AntiEntropyService) ResetStats() {
	aes.statsMu.Lock()
	defer aes.statsMu.Unlock()

	aes.stats = &AntiEntropyStats{}
}

// HandleDigestRequest handles an incoming digest request
func (aes *AntiEntropyService) HandleDigestRequest(message *Message) error {
	var remoteDigest Digest
	if err := json.Unmarshal(message.Payload, &remoteDigest); err != nil {
		return err
	}

	// Generate our digest
	localDigest := aes.engine.crdtStore.Digest(aes.engine.nodeID, aes.engine.vectorClock.Get())

	// Send digest response
	digestData, err := json.Marshal(localDigest)
	if err != nil {
		return err
	}

	response := &Message{
		ID:          generateMessageID(),
		Type:        "ANTI_ENTROPY_DIGEST_RESPONSE",
		SenderID:    aes.engine.nodeID,
		ReceiverID:  message.SenderID,
		VectorClock: aes.engine.vectorClock.Get(),
		Payload:     digestData,
		Timestamp:   time.Now(),
	}

	// Get peer
	aes.engine.mu.RLock()
	peer, exists := aes.engine.regions[message.SenderID]
	aes.engine.mu.RUnlock()

	if !exists {
		return &SyncError{Message: "peer not found"}
	}

	return aes.engine.transport.Send(peer, response)
}

// Merkle tree for efficient digest comparison (future optimization)
type MerkleNode struct {
	Hash  string
	Left  *MerkleNode
	Right *MerkleNode
}

// BloomFilter for efficient membership testing (future optimization)
type BloomFilter struct {
	bits    []bool
	hashFns int
}

// Type aliases for CRDT types
type Digest = crdt.Digest
type Delta = crdt.Delta
