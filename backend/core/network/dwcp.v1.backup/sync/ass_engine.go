package sync

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/yourusername/novacron/backend/core/network/dwcp/sync/crdt"
	"go.uber.org/zap"
)

// ASSEngine implements the Async State Synchronization engine using CRDTs
type ASSEngine struct {
	nodeID       string
	regions      map[string]*RegionPeer
	crdtStore    *CRDTStore
	vectorClock  *VectorClockManager
	gossip       *GossipProtocol
	antiEntropy  *AntiEntropyService
	causal       *CausalTracker
	transport    Transport
	logger       *zap.Logger
	ctx          context.Context
	cancel       context.CancelFunc
	mu           sync.RWMutex
}

// RegionPeer represents a peer node in another region
type RegionPeer struct {
	ID           string        `json:"id"`
	Region       string        `json:"region"`
	Endpoint     string        `json:"endpoint"`
	Latency      time.Duration `json:"latency"`
	LastSync     time.Time     `json:"last_sync"`
	VectorClock  crdt.VectorClock `json:"vector_clock"`
	mu           sync.RWMutex
}

// CRDTStore manages all CRDT instances
type CRDTStore struct {
	data     map[string]crdt.CvRDT
	metadata map[string]*CRDTMetadata
	mu       sync.RWMutex
}

// CRDTMetadata stores metadata about a CRDT instance
type CRDTMetadata struct {
	Key         string           `json:"key"`
	Type        string           `json:"type"`
	CreatedAt   time.Time        `json:"created_at"`
	UpdatedAt   time.Time        `json:"updated_at"`
	VectorClock crdt.VectorClock `json:"vector_clock"`
	Checksum    string           `json:"checksum"`
}

// Transport defines the interface for network communication
type Transport interface {
	Send(peer *RegionPeer, message *Message) error
	Receive() (*Message, error)
	Close() error
}

// Message represents a message exchanged between nodes
type Message struct {
	ID          string           `json:"id"`
	Type        string           `json:"type"`
	SenderID    string           `json:"sender_id"`
	ReceiverID  string           `json:"receiver_id"`
	VectorClock crdt.VectorClock `json:"vector_clock"`
	Payload     json.RawMessage  `json:"payload"`
	Timestamp   time.Time        `json:"timestamp"`
}

// NewASSEngine creates a new ASS engine
func NewASSEngine(nodeID string, transport Transport, logger *zap.Logger) *ASSEngine {
	ctx, cancel := context.WithCancel(context.Background())

	engine := &ASSEngine{
		nodeID:      nodeID,
		regions:     make(map[string]*RegionPeer),
		crdtStore:   NewCRDTStore(),
		vectorClock: NewVectorClockManager(nodeID),
		causal:      NewCausalTracker(nodeID),
		transport:   transport,
		logger:      logger,
		ctx:         ctx,
		cancel:      cancel,
	}

	// Initialize gossip protocol
	engine.gossip = NewGossipProtocol(engine, 3, 5*time.Second, 10, logger)

	// Initialize anti-entropy service
	engine.antiEntropy = NewAntiEntropyService(engine, 30*time.Second, logger)

	return engine
}

// NewCRDTStore creates a new CRDT store
func NewCRDTStore() *CRDTStore {
	return &CRDTStore{
		data:     make(map[string]crdt.CvRDT),
		metadata: make(map[string]*CRDTMetadata),
	}
}

// RegisterPeer registers a peer node in another region
func (e *ASSEngine) RegisterPeer(peer *RegionPeer) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.regions[peer.ID] = peer
	e.logger.Info("Registered peer", zap.String("peer_id", peer.ID), zap.String("region", peer.Region))
}

// Set stores a CRDT value
func (e *ASSEngine) Set(key string, value crdt.CvRDT) error {
	e.crdtStore.mu.Lock()
	defer e.crdtStore.mu.Unlock()

	// Update vector clock
	e.vectorClock.Increment()

	// Store CRDT
	e.crdtStore.data[key] = value

	// Update metadata
	data, err := value.Marshal()
	if err != nil {
		return err
	}

	checksum := computeChecksum(data)
	metadata := &CRDTMetadata{
		Key:         key,
		Type:        getCRDTType(value),
		UpdatedAt:   time.Now(),
		VectorClock: e.vectorClock.Get(),
		Checksum:    checksum,
	}

	if existing, exists := e.crdtStore.metadata[key]; exists {
		metadata.CreatedAt = existing.CreatedAt
	} else {
		metadata.CreatedAt = time.Now()
	}

	e.crdtStore.metadata[key] = metadata

	// Broadcast update via gossip
	update := &CRDTUpdate{
		Key:         key,
		Type:        metadata.Type,
		Data:        data,
		VectorClock: metadata.VectorClock,
		Timestamp:   time.Now(),
	}

	e.gossip.Broadcast(update)

	return nil
}

// Get retrieves a CRDT value
func (e *ASSEngine) Get(key string) (crdt.CvRDT, bool) {
	e.crdtStore.mu.RLock()
	defer e.crdtStore.mu.RUnlock()

	value, exists := e.crdtStore.data[key]
	return value, exists
}

// Delete removes a CRDT value
func (e *ASSEngine) Delete(key string) {
	e.crdtStore.mu.Lock()
	defer e.crdtStore.mu.Unlock()

	delete(e.crdtStore.data, key)
	delete(e.crdtStore.metadata, key)
}

// SyncWithRegion synchronizes with a specific region peer
func (e *ASSEngine) SyncWithRegion(regionID string) error {
	e.mu.RLock()
	peer, exists := e.regions[regionID]
	e.mu.RUnlock()

	if !exists {
		return fmt.Errorf("peer not found: %s", regionID)
	}

	// Get local state digest
	localDigest := e.crdtStore.Digest(e.nodeID, e.vectorClock.Get())

	// Request remote digest
	remoteDigest, err := e.requestDigest(peer)
	if err != nil {
		return fmt.Errorf("failed to request digest: %w", err)
	}

	// Compute delta
	delta := e.computeDelta(localDigest, remoteDigest)

	// Exchange delta states
	if len(delta.Missing) > 0 {
		remoteStates, err := e.requestStates(peer, delta.Missing)
		if err != nil {
			return fmt.Errorf("failed to request states: %w", err)
		}
		if err := e.mergeStates(remoteStates); err != nil {
			return fmt.Errorf("failed to merge states: %w", err)
		}
	}

	if len(delta.Theirs) > 0 {
		localStates := e.getStates(delta.Theirs)
		if err := e.sendStates(peer, localStates); err != nil {
			return fmt.Errorf("failed to send states: %w", err)
		}
	}

	// Update peer sync time
	peer.mu.Lock()
	peer.LastSync = time.Now()
	peer.VectorClock = e.vectorClock.Get()
	peer.mu.Unlock()

	e.logger.Info("Synchronized with region",
		zap.String("peer_id", peer.ID),
		zap.String("region", peer.Region),
		zap.Int("missing", len(delta.Missing)),
		zap.Int("theirs", len(delta.Theirs)))

	return nil
}

// Digest generates a digest of the CRDT store
func (cs *CRDTStore) Digest(nodeID string, vectorClock crdt.VectorClock) *crdt.Digest {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	checksums := make(map[string]string)
	for key, metadata := range cs.metadata {
		checksums[key] = metadata.Checksum
	}

	return &crdt.Digest{
		NodeID:      nodeID,
		VectorClock: vectorClock,
		Checksums:   checksums,
		Timestamp:   time.Now(),
	}
}

func (e *ASSEngine) computeDelta(local, remote *crdt.Digest) *crdt.Delta {
	delta := &crdt.Delta{
		Missing: make([]string, 0),
		Theirs:  make([]string, 0),
		Stale:   make([]string, 0),
	}

	// Find keys we're missing
	for key := range remote.Checksums {
		if _, exists := local.Checksums[key]; !exists {
			delta.Missing = append(delta.Missing, key)
		}
	}

	// Find keys they're missing
	for key := range local.Checksums {
		if _, exists := remote.Checksums[key]; !exists {
			delta.Theirs = append(delta.Theirs, key)
		}
	}

	// Find stale keys (different checksums)
	for key, localChecksum := range local.Checksums {
		if remoteChecksum, exists := remote.Checksums[key]; exists {
			if localChecksum != remoteChecksum {
				delta.Stale = append(delta.Stale, key)
			}
		}
	}

	return delta
}

func (e *ASSEngine) requestDigest(peer *RegionPeer) (*crdt.Digest, error) {
	message := &Message{
		ID:          generateMessageID(),
		Type:        "DIGEST_REQUEST",
		SenderID:    e.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: e.vectorClock.Get(),
		Timestamp:   time.Now(),
	}

	if err := e.transport.Send(peer, message); err != nil {
		return nil, err
	}

	// Wait for response (simplified - in production use channels)
	response, err := e.transport.Receive()
	if err != nil {
		return nil, err
	}

	var digest crdt.Digest
	if err := json.Unmarshal(response.Payload, &digest); err != nil {
		return nil, err
	}

	return &digest, nil
}

func (e *ASSEngine) requestStates(peer *RegionPeer, keys []string) (map[string]crdt.CvRDT, error) {
	payload, _ := json.Marshal(keys)
	message := &Message{
		ID:          generateMessageID(),
		Type:        "STATES_REQUEST",
		SenderID:    e.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: e.vectorClock.Get(),
		Payload:     payload,
		Timestamp:   time.Now(),
	}

	if err := e.transport.Send(peer, message); err != nil {
		return nil, err
	}

	// Receive response
	response, err := e.transport.Receive()
	if err != nil {
		return nil, err
	}

	var statesData map[string]json.RawMessage
	if err := json.Unmarshal(response.Payload, &statesData); err != nil {
		return nil, err
	}

	// Deserialize CRDTs
	states := make(map[string]crdt.CvRDT)
	for key, data := range statesData {
		// Determine CRDT type and deserialize
		crdtValue := e.deserializeCRDT(key, data)
		if crdtValue != nil {
			states[key] = crdtValue
		}
	}

	return states, nil
}

func (e *ASSEngine) getStates(keys []string) map[string]crdt.CvRDT {
	e.crdtStore.mu.RLock()
	defer e.crdtStore.mu.RUnlock()

	states := make(map[string]crdt.CvRDT)
	for _, key := range keys {
		if value, exists := e.crdtStore.data[key]; exists {
			states[key] = value
		}
	}
	return states
}

func (e *ASSEngine) sendStates(peer *RegionPeer, states map[string]crdt.CvRDT) error {
	statesData := make(map[string]json.RawMessage)
	for key, value := range states {
		data, err := value.Marshal()
		if err != nil {
			continue
		}
		statesData[key] = data
	}

	payload, _ := json.Marshal(statesData)
	message := &Message{
		ID:          generateMessageID(),
		Type:        "STATES_RESPONSE",
		SenderID:    e.nodeID,
		ReceiverID:  peer.ID,
		VectorClock: e.vectorClock.Get(),
		Payload:     payload,
		Timestamp:   time.Now(),
	}

	return e.transport.Send(peer, message)
}

func (e *ASSEngine) mergeStates(states map[string]crdt.CvRDT) error {
	e.crdtStore.mu.Lock()
	defer e.crdtStore.mu.Unlock()

	for key, remoteValue := range states {
		if localValue, exists := e.crdtStore.data[key]; exists {
			// Merge with existing value
			if err := localValue.Merge(remoteValue); err != nil {
				e.logger.Error("Failed to merge CRDT", zap.String("key", key), zap.Error(err))
				continue
			}
		} else {
			// Store new value
			e.crdtStore.data[key] = remoteValue
		}

		// Update metadata
		data, _ := remoteValue.Marshal()
		e.crdtStore.metadata[key] = &CRDTMetadata{
			Key:         key,
			Type:        getCRDTType(remoteValue),
			UpdatedAt:   time.Now(),
			VectorClock: e.vectorClock.Get(),
			Checksum:    computeChecksum(data),
		}
	}

	return nil
}

func (e *ASSEngine) deserializeCRDT(key string, data json.RawMessage) crdt.CvRDT {
	e.crdtStore.mu.RLock()
	metadata, exists := e.crdtStore.metadata[key]
	e.crdtStore.mu.RUnlock()

	if !exists {
		return nil
	}

	var value crdt.CvRDT
	switch metadata.Type {
	case "g_counter":
		value = crdt.NewGCounter(e.nodeID)
	case "pn_counter":
		value = crdt.NewPNCounter(e.nodeID)
	case "or_set":
		value = crdt.NewORSet(e.nodeID)
	case "lww_register":
		value = crdt.NewLWWRegister(e.nodeID)
	case "mv_register":
		value = crdt.NewMVRegister(e.nodeID)
	case "or_map":
		value = crdt.NewORMap(e.nodeID)
	case "rga":
		value = crdt.NewRGA(e.nodeID)
	default:
		return nil
	}

	if err := value.Unmarshal(data); err != nil {
		e.logger.Error("Failed to deserialize CRDT", zap.String("key", key), zap.Error(err))
		return nil
	}

	return value
}

// Start starts the ASS engine
func (e *ASSEngine) Start() error {
	e.logger.Info("Starting ASS engine", zap.String("node_id", e.nodeID))

	// Start anti-entropy service
	e.antiEntropy.Start()

	// Start gossip protocol
	e.gossip.Start()

	return nil
}

// Stop stops the ASS engine
func (e *ASSEngine) Stop() error {
	e.logger.Info("Stopping ASS engine", zap.String("node_id", e.nodeID))

	e.cancel()

	// Stop anti-entropy service
	e.antiEntropy.Stop()

	// Stop gossip protocol
	e.gossip.Stop()

	return nil
}

// Helper functions

func computeChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func getCRDTType(value crdt.CvRDT) string {
	switch value.(type) {
	case *crdt.GCounter:
		return "g_counter"
	case *crdt.PNCounter:
		return "pn_counter"
	case *crdt.GSet:
		return "g_set"
	case *crdt.TwoPhaseSet:
		return "2p_set"
	case *crdt.ORSet:
		return "or_set"
	case *crdt.LWWRegister:
		return "lww_register"
	case *crdt.MVRegister:
		return "mv_register"
	case *crdt.ORMap:
		return "or_map"
	case *crdt.RGA:
		return "rga"
	default:
		return "unknown"
	}
}

func generateMessageID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// CRDTUpdate represents a CRDT update for gossiping
type CRDTUpdate struct {
	Key         string           `json:"key"`
	Type        string           `json:"type"`
	Data        json.RawMessage  `json:"data"`
	VectorClock crdt.VectorClock `json:"vector_clock"`
	Timestamp   time.Time        `json:"timestamp"`
}
