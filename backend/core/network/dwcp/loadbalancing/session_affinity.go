package loadbalancing

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"sort"
	"sync"
	"time"
)

// SessionAffinityManager manages session affinity and consistent hashing
type SessionAffinityManager struct {
	config   *LoadBalancerConfig
	sessions map[string]*SessionAffinity
	ring     *ConsistentHashRing
	mu       sync.RWMutex
}

// ConsistentHashRing implements consistent hashing with virtual nodes
type ConsistentHashRing struct {
	virtualNodes map[uint32]string // hash -> serverID
	sortedHashes []uint32
	servers      map[string]bool
	vnodeCount   int
	mu           sync.RWMutex
}

// NewSessionAffinityManager creates a new session affinity manager
func NewSessionAffinityManager(config *LoadBalancerConfig) *SessionAffinityManager {
	sam := &SessionAffinityManager{
		config:   config,
		sessions: make(map[string]*SessionAffinity),
		ring:     NewConsistentHashRing(config.VirtualNodesPerServer),
	}

	// Start cleanup routine
	go sam.cleanupExpiredSessions()

	return sam
}

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(vnodeCount int) *ConsistentHashRing {
	return &ConsistentHashRing{
		virtualNodes: make(map[uint32]string),
		sortedHashes: make([]uint32, 0),
		servers:      make(map[string]bool),
		vnodeCount:   vnodeCount,
	}
}

// AddServer adds a server to the consistent hash ring
func (chr *ConsistentHashRing) AddServer(serverID string) {
	chr.mu.Lock()
	defer chr.mu.Unlock()

	if chr.servers[serverID] {
		return // Already exists
	}

	// Add virtual nodes
	for i := 0; i < chr.vnodeCount; i++ {
		hash := chr.hash(fmt.Sprintf("%s-%d", serverID, i))
		chr.virtualNodes[hash] = serverID
		chr.sortedHashes = append(chr.sortedHashes, hash)
	}

	// Sort hashes
	sort.Slice(chr.sortedHashes, func(i, j int) bool {
		return chr.sortedHashes[i] < chr.sortedHashes[j]
	})

	chr.servers[serverID] = true
}

// RemoveServer removes a server from the consistent hash ring
func (chr *ConsistentHashRing) RemoveServer(serverID string) {
	chr.mu.Lock()
	defer chr.mu.Unlock()

	if !chr.servers[serverID] {
		return
	}

	// Remove virtual nodes
	newHashes := make([]uint32, 0)
	for _, hash := range chr.sortedHashes {
		if chr.virtualNodes[hash] != serverID {
			newHashes = append(newHashes, hash)
		} else {
			delete(chr.virtualNodes, hash)
		}
	}

	chr.sortedHashes = newHashes
	delete(chr.servers, serverID)
}

// GetServer returns the server for a given key
func (chr *ConsistentHashRing) GetServer(key string) (string, error) {
	chr.mu.RLock()
	defer chr.mu.RUnlock()

	if len(chr.sortedHashes) == 0 {
		return "", ErrNoHealthyServers
	}

	hash := chr.hash(key)

	// Binary search for the first hash >= key hash
	idx := sort.Search(len(chr.sortedHashes), func(i int) bool {
		return chr.sortedHashes[i] >= hash
	})

	// Wrap around if necessary
	if idx == len(chr.sortedHashes) {
		idx = 0
	}

	return chr.virtualNodes[chr.sortedHashes[idx]], nil
}

// hash generates a hash for a key
func (chr *ConsistentHashRing) hash(key string) uint32 {
	return crc32.ChecksumIEEE([]byte(key))
}

// CreateSession creates a new session affinity
func (sam *SessionAffinityManager) CreateSession(sessionID, serverID string) *SessionAffinity {
	sam.mu.Lock()
	defer sam.mu.Unlock()

	session := &SessionAffinity{
		SessionID:    sessionID,
		ServerID:     serverID,
		CreatedAt:    time.Now(),
		ExpiresAt:    time.Now().Add(sam.config.SessionAffinityTTL),
		RequestCount: 0,
	}

	sam.sessions[sessionID] = session
	return session
}

// GetSession retrieves a session by ID
func (sam *SessionAffinityManager) GetSession(sessionID string) (*SessionAffinity, error) {
	sam.mu.RLock()
	defer sam.mu.RUnlock()

	session, exists := sam.sessions[sessionID]
	if !exists {
		return nil, ErrSessionNotFound
	}

	// Check if expired
	if time.Now().After(session.ExpiresAt) {
		return nil, ErrSessionNotFound
	}

	return session, nil
}

// UpdateSession updates session activity
func (sam *SessionAffinityManager) UpdateSession(sessionID string) error {
	sam.mu.Lock()
	defer sam.mu.Unlock()

	session, exists := sam.sessions[sessionID]
	if !exists {
		return ErrSessionNotFound
	}

	// Extend expiration
	session.ExpiresAt = time.Now().Add(sam.config.SessionAffinityTTL)
	session.RequestCount++

	return nil
}

// DeleteSession removes a session
func (sam *SessionAffinityManager) DeleteSession(sessionID string) {
	sam.mu.Lock()
	defer sam.mu.Unlock()

	delete(sam.sessions, sessionID)
}

// MigrateSession migrates a session to a new server (for failover)
func (sam *SessionAffinityManager) MigrateSession(sessionID, newServerID string) error {
	sam.mu.Lock()
	defer sam.mu.Unlock()

	session, exists := sam.sessions[sessionID]
	if !exists {
		return ErrSessionNotFound
	}

	session.ServerID = newServerID
	session.ExpiresAt = time.Now().Add(sam.config.SessionAffinityTTL)

	return nil
}

// cleanupExpiredSessions removes expired sessions periodically
func (sam *SessionAffinityManager) cleanupExpiredSessions() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		sam.mu.Lock()
		now := time.Now()
		for id, session := range sam.sessions {
			if now.After(session.ExpiresAt) {
				delete(sam.sessions, id)
			}
		}
		sam.mu.Unlock()
	}
}

// GetServerByIP returns the server for an IP address using consistent hashing
func (sam *SessionAffinityManager) GetServerByIP(ipAddress string) (string, error) {
	return sam.ring.GetServer(ipAddress)
}

// GetServerByHash returns the server for a hash key
func (sam *SessionAffinityManager) GetServerByHash(key string) (string, error) {
	return sam.ring.GetServer(key)
}

// AddServerToRing adds a server to the hash ring
func (sam *SessionAffinityManager) AddServerToRing(serverID string) {
	sam.ring.AddServer(serverID)
}

// RemoveServerFromRing removes a server from the hash ring
func (sam *SessionAffinityManager) RemoveServerFromRing(serverID string) {
	sam.ring.RemoveServer(serverID)
}

// GetSessionCount returns the number of active sessions
func (sam *SessionAffinityManager) GetSessionCount() int {
	sam.mu.RLock()
	defer sam.mu.RUnlock()
	return len(sam.sessions)
}

// GenerateSessionID generates a unique session ID from client information
func GenerateSessionID(clientIP string, userAgent string) string {
	data := fmt.Sprintf("%s:%s:%d", clientIP, userAgent, time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash[:16])
}

// HashIPAddress creates a hash from an IP address
func HashIPAddress(ip string) uint32 {
	hash := sha256.Sum256([]byte(ip))
	return binary.BigEndian.Uint32(hash[:4])
}
