package sync

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"math/big"
	"sync"
	"time"

	"go.uber.org/zap"
)

// GossipProtocol implements the gossip protocol for CRDT propagation
type GossipProtocol struct {
	engine       *ASSEngine
	fanout       int           // Number of peers to gossip to
	interval     time.Duration // Gossip interval
	maxHops      int           // Maximum number of hops (TTL)
	seenMessages map[string]struct{}
	seenMu       sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
	logger       *zap.Logger
}

// GossipMessage represents a gossip message
type GossipMessage struct {
	ID          string        `json:"id"`
	Update      *CRDTUpdate   `json:"update"`
	Hops        int           `json:"hops"`
	TTL         int           `json:"ttl"`
	Timestamp   time.Time     `json:"timestamp"`
	Path        []string      `json:"path"` // Track propagation path
}

// NewGossipProtocol creates a new gossip protocol instance
func NewGossipProtocol(engine *ASSEngine, fanout int, interval time.Duration, maxHops int, logger *zap.Logger) *GossipProtocol {
	ctx, cancel := context.WithCancel(context.Background())

	return &GossipProtocol{
		engine:       engine,
		fanout:       fanout,
		interval:     interval,
		maxHops:      maxHops,
		seenMessages: make(map[string]struct{}),
		ctx:          ctx,
		cancel:       cancel,
		logger:       logger,
	}
}

// Start starts the gossip protocol
func (gp *GossipProtocol) Start() {
	go gp.gossipLoop()
	gp.logger.Info("Gossip protocol started",
		zap.Int("fanout", gp.fanout),
		zap.Duration("interval", gp.interval),
		zap.Int("max_hops", gp.maxHops))
}

// Stop stops the gossip protocol
func (gp *GossipProtocol) Stop() {
	gp.cancel()
	gp.logger.Info("Gossip protocol stopped")
}

func (gp *GossipProtocol) gossipLoop() {
	ticker := time.NewTicker(gp.interval)
	defer ticker.Stop()

	for {
		select {
		case <-gp.ctx.Done():
			return
		case <-ticker.C:
			gp.performGossipRound()
		}
	}
}

func (gp *GossipProtocol) performGossipRound() {
	// Periodic cleanup of seen messages
	gp.cleanupSeenMessages()

	// Select random peers and exchange state
	peers := gp.selectRandomPeers(gp.fanout)
	for _, peer := range peers {
		go gp.gossipWithPeer(peer)
	}
}

func (gp *GossipProtocol) gossipWithPeer(peer *RegionPeer) {
	// Get recent updates (simplified - in production track recent changes)
	// For now, just sync with the peer
	if err := gp.engine.SyncWithRegion(peer.ID); err != nil {
		gp.logger.Error("Gossip sync failed",
			zap.String("peer_id", peer.ID),
			zap.Error(err))
	}
}

// Broadcast broadcasts a CRDT update to random peers
func (gp *GossipProtocol) Broadcast(update *CRDTUpdate) {
	peers := gp.selectRandomPeers(gp.fanout)

	message := &GossipMessage{
		ID:        generateMessageID(),
		Update:    update,
		Hops:      0,
		TTL:       gp.maxHops,
		Timestamp: time.Now(),
		Path:      []string{gp.engine.nodeID},
	}

	// Mark as seen
	gp.markSeen(message.ID)

	// Send to selected peers
	for _, peer := range peers {
		go gp.sendMessage(peer, message)
	}

	gp.logger.Debug("Broadcast CRDT update",
		zap.String("message_id", message.ID),
		zap.String("key", update.Key),
		zap.Int("peers", len(peers)))
}

// HandleMessage handles an incoming gossip message
func (gp *GossipProtocol) HandleMessage(msg *GossipMessage) error {
	// Deduplication
	if gp.hasSeen(msg.ID) {
		return nil
	}
	gp.markSeen(msg.ID)

	// Apply update locally
	if err := gp.applyUpdate(msg.Update); err != nil {
		gp.logger.Error("Failed to apply gossip update",
			zap.String("message_id", msg.ID),
			zap.Error(err))
		return err
	}

	gp.logger.Debug("Applied gossip update",
		zap.String("message_id", msg.ID),
		zap.String("key", msg.Update.Key),
		zap.Int("hops", msg.Hops))

	// Forward if TTL remaining
	if msg.TTL > 0 {
		msg.Hops++
		msg.TTL--
		msg.Path = append(msg.Path, gp.engine.nodeID)

		peers := gp.selectRandomPeersExcluding(gp.fanout, msg.Path)
		for _, peer := range peers {
			go gp.sendMessage(peer, msg)
		}
	}

	return nil
}

func (gp *GossipProtocol) applyUpdate(update *CRDTUpdate) error {
	// Deserialize CRDT
	crdtValue := gp.engine.deserializeCRDT(update.Key, update.Data)
	if crdtValue == nil {
		// Create new CRDT if doesn't exist
		crdtValue = gp.createCRDT(update.Type, update.Data)
		if crdtValue == nil {
			return &SyncError{Message: "failed to create CRDT"}
		}
	}

	// Get existing value or create new
	if existing, exists := gp.engine.Get(update.Key); exists {
		// Merge with existing
		if err := existing.Merge(crdtValue); err != nil {
			return err
		}
		return gp.engine.Set(update.Key, existing)
	}

	// Store new value
	return gp.engine.Set(update.Key, crdtValue)
}

func (gp *GossipProtocol) createCRDT(crdtType string, data json.RawMessage) CvRDT {
	var value CvRDT

	switch crdtType {
	case "g_counter":
		value = NewGCounter(gp.engine.nodeID)
	case "pn_counter":
		value = NewPNCounter(gp.engine.nodeID)
	case "or_set":
		value = NewORSet(gp.engine.nodeID)
	case "lww_register":
		value = NewLWWRegister(gp.engine.nodeID)
	case "mv_register":
		value = NewMVRegister(gp.engine.nodeID)
	case "or_map":
		value = NewORMap(gp.engine.nodeID)
	case "rga":
		value = NewRGA(gp.engine.nodeID)
	default:
		return nil
	}

	if err := value.Unmarshal(data); err != nil {
		gp.logger.Error("Failed to unmarshal CRDT", zap.String("type", crdtType), zap.Error(err))
		return nil
	}

	return value
}

func (gp *GossipProtocol) sendMessage(peer *RegionPeer, msg *GossipMessage) {
	payload, err := json.Marshal(msg)
	if err != nil {
		gp.logger.Error("Failed to marshal gossip message", zap.Error(err))
		return
	}

	message := &Message{
		ID:          msg.ID,
		Type:        "GOSSIP",
		SenderID:    gp.engine.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: gp.engine.vectorClock.Get(),
		Payload:     payload,
		Timestamp:   time.Now(),
	}

	if err := gp.engine.transport.Send(peer, message); err != nil {
		gp.logger.Error("Failed to send gossip message",
			zap.String("peer_id", peer.ID),
			zap.Error(err))
	}
}

func (gp *GossipProtocol) selectRandomPeers(count int) []*RegionPeer {
	gp.engine.mu.RLock()
	defer gp.engine.mu.RUnlock()

	peers := make([]*RegionPeer, 0, len(gp.engine.regions))
	for _, peer := range gp.engine.regions {
		peers = append(peers, peer)
	}

	if len(peers) <= count {
		return peers
	}

	// Randomly select count peers
	selected := make([]*RegionPeer, 0, count)
	indices := make(map[int]struct{})

	for len(selected) < count {
		idx, _ := rand.Int(rand.Reader, big.NewInt(int64(len(peers))))
		idxInt := int(idx.Int64())

		if _, used := indices[idxInt]; !used {
			indices[idxInt] = struct{}{}
			selected = append(selected, peers[idxInt])
		}
	}

	return selected
}

func (gp *GossipProtocol) selectRandomPeersExcluding(count int, excludeIDs []string) []*RegionPeer {
	gp.engine.mu.RLock()
	defer gp.engine.mu.RUnlock()

	excludeMap := make(map[string]struct{})
	for _, id := range excludeIDs {
		excludeMap[id] = struct{}{}
	}

	peers := make([]*RegionPeer, 0)
	for id, peer := range gp.engine.regions {
		if _, excluded := excludeMap[id]; !excluded {
			peers = append(peers, peer)
		}
	}

	if len(peers) <= count {
		return peers
	}

	// Randomly select count peers
	selected := make([]*RegionPeer, 0, count)
	indices := make(map[int]struct{})

	for len(selected) < count && len(selected) < len(peers) {
		idx, _ := rand.Int(rand.Reader, big.NewInt(int64(len(peers))))
		idxInt := int(idx.Int64())

		if _, used := indices[idxInt]; !used {
			indices[idxInt] = struct{}{}
			selected = append(selected, peers[idxInt])
		}
	}

	return selected
}

func (gp *GossipProtocol) hasSeen(messageID string) bool {
	gp.seenMu.RLock()
	defer gp.seenMu.RUnlock()
	_, seen := gp.seenMessages[messageID]
	return seen
}

func (gp *GossipProtocol) markSeen(messageID string) {
	gp.seenMu.Lock()
	defer gp.seenMu.Unlock()
	gp.seenMessages[messageID] = struct{}{}
}

func (gp *GossipProtocol) cleanupSeenMessages() {
	gp.seenMu.Lock()
	defer gp.seenMu.Unlock()

	// Keep only recent messages (last 1000)
	if len(gp.seenMessages) > 1000 {
		// Clear old messages (simplified - in production use time-based cleanup)
		gp.seenMessages = make(map[string]struct{})
	}
}

// GetStats returns gossip protocol statistics
func (gp *GossipProtocol) GetStats() GossipStats {
	gp.seenMu.RLock()
	defer gp.seenMu.RUnlock()

	return GossipStats{
		SeenMessages: len(gp.seenMessages),
		Fanout:       gp.fanout,
		MaxHops:      gp.maxHops,
		Interval:     gp.interval,
	}
}

// GossipStats represents gossip protocol statistics
type GossipStats struct {
	SeenMessages int           `json:"seen_messages"`
	Fanout       int           `json:"fanout"`
	MaxHops      int           `json:"max_hops"`
	Interval     time.Duration `json:"interval"`
}

// SyncError represents a synchronization error
type SyncError struct {
	Message string
}

func (e *SyncError) Error() string {
	return e.Message
}

// Import CRDT types (these would normally be imported)
type CvRDT interface {
	Merge(other CvRDT) error
	Marshal() ([]byte, error)
	Unmarshal(data []byte) error
	Value() interface{}
}

func NewGCounter(nodeID string) CvRDT     { return nil }
func NewPNCounter(nodeID string) CvRDT    { return nil }
func NewORSet(nodeID string) CvRDT        { return nil }
func NewLWWRegister(nodeID string) CvRDT  { return nil }
func NewMVRegister(nodeID string) CvRDT   { return nil }
func NewORMap(nodeID string) CvRDT        { return nil }
func NewRGA(nodeID string) CvRDT          { return nil }
