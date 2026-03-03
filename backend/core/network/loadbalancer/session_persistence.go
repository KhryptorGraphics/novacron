package loadbalancer

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SessionPersistenceManager manages session persistence and affinity
type SessionPersistenceManager struct {
	// Configuration
	config           SessionPersistenceConfig
	
	// Session stores
	sessions         map[string]*Session
	sessionsMutex    sync.RWMutex
	
	// Affinity mappings
	ipAffinity       map[string]*AffinityMapping
	cookieAffinity   map[string]*AffinityMapping
	headerAffinity   map[string]*AffinityMapping
	customAffinity   map[string]*AffinityMapping
	affinityMutex    sync.RWMutex
	
	// Backend tracking
	backendSessions  map[string][]*Session
	backendMutex     sync.RWMutex
	
	// Consistent hashing for distribution
	hashRing         *ConsistentHashRing
	
	// Session storage backend
	storageBackend   SessionStorageBackend
	
	// Metrics
	metrics          *SessionPersistenceMetrics
	metricsMutex     sync.RWMutex
	
	// Runtime state
	ctx              context.Context
	cancel           context.CancelFunc
	initialized      bool
}

// SessionPersistenceConfig holds session persistence configuration
type SessionPersistenceConfig struct {
	// Global settings
	EnablePersistence    bool              `json:"enable_persistence"`
	DefaultMethod        AffinityMethod    `json:"default_method"`
	SessionTimeout       time.Duration     `json:"session_timeout"`
	MaxSessions          int               `json:"max_sessions"`
	CleanupInterval      time.Duration     `json:"cleanup_interval"`
	
	// Cookie-based affinity
	CookieName           string            `json:"cookie_name"`
	CookiePath           string            `json:"cookie_path"`
	CookieDomain         string            `json:"cookie_domain"`
	CookieSecure         bool              `json:"cookie_secure"`
	CookieHTTPOnly       bool              `json:"cookie_http_only"`
	CookieSameSite       string            `json:"cookie_same_site"`
	CookieMaxAge         int               `json:"cookie_max_age"`
	
	// Header-based affinity
	HeaderName           string            `json:"header_name"`
	HeaderPrefix         string            `json:"header_prefix"`
	
	// IP-based affinity
	IPAffinitySubnet     int               `json:"ip_affinity_subnet"`
	IPAffinityTimeout    time.Duration     `json:"ip_affinity_timeout"`
	EnableStickyIP       bool              `json:"enable_sticky_ip"`
	
	// Consistent hashing
	EnableConsistentHash bool              `json:"enable_consistent_hash"`
	HashAlgorithm        string            `json:"hash_algorithm"`
	VirtualNodes         int               `json:"virtual_nodes"`
	
	// Session replication
	EnableReplication    bool              `json:"enable_replication"`
	ReplicationFactor    int               `json:"replication_factor"`
	ReplicationMode      ReplicationMode   `json:"replication_mode"`
	
	// Storage backend
	StorageType          StorageType       `json:"storage_type"`
	StorageConfig        map[string]interface{} `json:"storage_config"`
	
	// Failover settings
	EnableFailover       bool              `json:"enable_failover"`
	FailoverMode         FailoverMode      `json:"failover_mode"`
	FailoverTimeout      time.Duration     `json:"failover_timeout"`
	
	// Load balancing weights
	AffinityWeight       float64           `json:"affinity_weight"`
	LoadBalanceWeight    float64           `json:"load_balance_weight"`
	
	// Monitoring
	EnableMetrics        bool              `json:"enable_metrics"`
	MetricsInterval      time.Duration     `json:"metrics_interval"`
}

// Session represents a client session
type Session struct {
	ID               string                 `json:"id"`
	ClientID         string                 `json:"client_id"`
	BackendID        string                 `json:"backend_id"`
	AffinityMethod   AffinityMethod         `json:"affinity_method"`
	AffinityKey      string                 `json:"affinity_key"`
	CreatedAt        time.Time              `json:"created_at"`
	LastAccessedAt   time.Time              `json:"last_accessed_at"`
	ExpiresAt        time.Time              `json:"expires_at"`
	RequestCount     int64                  `json:"request_count"`
	BytesTransferred int64                  `json:"bytes_transferred"`
	Metadata         map[string]interface{} `json:"metadata"`
	IsSticky         bool                   `json:"is_sticky"`
	FailoverCount    int                    `json:"failover_count"`
}

// AffinityMapping represents a client-to-backend affinity mapping
type AffinityMapping struct {
	ID             string                 `json:"id"`
	ClientKey      string                 `json:"client_key"`
	BackendID      string                 `json:"backend_id"`
	Method         AffinityMethod         `json:"method"`
	CreatedAt      time.Time              `json:"created_at"`
	LastUsedAt     time.Time              `json:"last_used_at"`
	ExpiresAt      time.Time              `json:"expires_at"`
	Weight         float64                `json:"weight"`
	SessionCount   int                    `json:"session_count"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// ConsistentHashRing implements consistent hashing for backend distribution
type ConsistentHashRing struct {
	nodes        map[uint32]string      // hash -> backend_id mapping
	sortedHashes []uint32               // sorted hash values
	virtualNodes int                    // number of virtual nodes per backend
	mutex        sync.RWMutex
}

// Types and enums
type ReplicationMode string
type StorageType string
type FailoverMode string

const (
	ReplicationModeSync    ReplicationMode = "sync"
	ReplicationModeAsync   ReplicationMode = "async"
	ReplicationModeHybrid  ReplicationMode = "hybrid"
	
	StorageTypeMemory      StorageType = "memory"
	StorageTypeRedis       StorageType = "redis"
	StorageTypeDatabase    StorageType = "database"
	StorageTypeDistributed StorageType = "distributed"
	
	FailoverModeImmediate  FailoverMode = "immediate"
	FailoverModeGraceful   FailoverMode = "graceful"
	FailoverModeSticky     FailoverMode = "sticky"
)

// SessionStorageBackend defines interface for session storage
type SessionStorageBackend interface {
	Store(session *Session) error
	Retrieve(sessionID string) (*Session, error)
	Update(session *Session) error
	Delete(sessionID string) error
	List(limit int, offset int) ([]*Session, error)
	Cleanup(expiredBefore time.Time) error
	Close() error
}

// SessionPersistenceMetrics holds session persistence metrics
type SessionPersistenceMetrics struct {
	TotalSessions        int64                          `json:"total_sessions"`
	ActiveSessions       int64                          `json:"active_sessions"`
	ExpiredSessions      int64                          `json:"expired_sessions"`
	SessionsByMethod     map[AffinityMethod]int64       `json:"sessions_by_method"`
	SessionsByBackend    map[string]int64               `json:"sessions_by_backend"`
	AffinityHitRate      float64                        `json:"affinity_hit_rate"`
	FailoverCount        int64                          `json:"failover_count"`
	AverageSessionDuration time.Duration               `json:"average_session_duration"`
	LastUpdated          time.Time                      `json:"last_updated"`
}

// NewSessionPersistenceManager creates a new session persistence manager
func NewSessionPersistenceManager(config SessionPersistenceConfig) *SessionPersistenceManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &SessionPersistenceManager{
		config:          config,
		sessions:        make(map[string]*Session),
		ipAffinity:      make(map[string]*AffinityMapping),
		cookieAffinity:  make(map[string]*AffinityMapping),
		headerAffinity:  make(map[string]*AffinityMapping),
		customAffinity:  make(map[string]*AffinityMapping),
		backendSessions: make(map[string][]*Session),
		metrics: &SessionPersistenceMetrics{
			SessionsByMethod:  make(map[AffinityMethod]int64),
			SessionsByBackend: make(map[string]int64),
			LastUpdated:       time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start initializes and starts the session persistence manager
func (spm *SessionPersistenceManager) Start() error {
	if spm.initialized {
		return fmt.Errorf("session persistence manager already started")
	}
	
	// Initialize consistent hash ring if enabled
	if spm.config.EnableConsistentHash {
		spm.hashRing = &ConsistentHashRing{
			nodes:        make(map[uint32]string),
			virtualNodes: spm.config.VirtualNodes,
		}
		
		if spm.hashRing.virtualNodes == 0 {
			spm.hashRing.virtualNodes = 100 // Default
		}
	}
	
	// Initialize storage backend
	if err := spm.initializeStorageBackend(); err != nil {
		return fmt.Errorf("failed to initialize storage backend: %w", err)
	}
	
	// Load existing sessions from storage
	if err := spm.loadExistingSessions(); err != nil {
		return fmt.Errorf("failed to load existing sessions: %w", err)
	}
	
	// Start background processes
	go spm.cleanupLoop()
	
	if spm.config.EnableMetrics {
		go spm.metricsCollectionLoop()
	}
	
	if spm.config.EnableReplication {
		go spm.replicationLoop()
	}
	
	spm.initialized = true
	return nil
}

// Stop stops the session persistence manager
func (spm *SessionPersistenceManager) Stop() error {
	spm.cancel()
	
	// Save sessions to storage
	if spm.storageBackend != nil {
		spm.sessionsMutex.RLock()
		for _, session := range spm.sessions {
			spm.storageBackend.Store(session)
		}
		spm.sessionsMutex.RUnlock()
		
		spm.storageBackend.Close()
	}
	
	spm.initialized = false
	return nil
}

// GetBackendForRequest determines the backend for a request based on session affinity
func (spm *SessionPersistenceManager) GetBackendForRequest(req *http.Request, availableBackends []string) (string, *Session, error) {
	if !spm.config.EnablePersistence || len(availableBackends) == 0 {
		return "", nil, fmt.Errorf("persistence disabled or no backends available")
	}
	
	// Try different affinity methods in order of preference
	methods := []AffinityMethod{spm.config.DefaultMethod}
	if spm.config.DefaultMethod != AffinityMethodCookie {
		methods = append(methods, AffinityMethodCookie)
	}
	if spm.config.DefaultMethod != AffinityMethodClientIP {
		methods = append(methods, AffinityMethodClientIP)
	}
	if spm.config.DefaultMethod != AffinityMethodHeader {
		methods = append(methods, AffinityMethodHeader)
	}
	
	for _, method := range methods {
		backend, session, err := spm.getBackendByMethod(req, method, availableBackends)
		if err == nil && backend != "" {
			return backend, session, nil
		}
	}
	
	// Fallback to consistent hashing or random selection
	if spm.config.EnableConsistentHash {
		clientKey := spm.getClientKey(req, AffinityMethodClientIP)
		backend := spm.hashRing.GetBackend(clientKey, availableBackends)
		if backend != "" {
			session := spm.createNewSession(clientKey, backend, AffinityMethodConsistentHash)
			return backend, session, nil
		}
	}
	
	return "", nil, fmt.Errorf("no suitable backend found")
}

// getBackendByMethod gets backend using specific affinity method
func (spm *SessionPersistenceManager) getBackendByMethod(req *http.Request, method AffinityMethod, availableBackends []string) (string, *Session, error) {
	clientKey := spm.getClientKey(req, method)
	if clientKey == "" {
		return "", nil, fmt.Errorf("could not extract client key for method %s", method)
	}
	
	// Check for existing affinity mapping
	var affinityStore map[string]*AffinityMapping
	
	switch method {
	case AffinityMethodClientIP:
		affinityStore = spm.ipAffinity
	case AffinityMethodCookie:
		affinityStore = spm.cookieAffinity
	case AffinityMethodHeader:
		affinityStore = spm.headerAffinity
	default:
		affinityStore = spm.customAffinity
	}
	
	spm.affinityMutex.RLock()
	mapping, exists := affinityStore[clientKey]
	spm.affinityMutex.RUnlock()
	
	if exists && !mapping.ExpiresAt.IsZero() && time.Now().Before(mapping.ExpiresAt) {
		// Check if the backend is still available
		if spm.isBackendAvailable(mapping.BackendID, availableBackends) {
			// Update last used time
			mapping.LastUsedAt = time.Now()
			mapping.SessionCount++
			
			// Get or create session
			session := spm.getOrCreateSession(clientKey, mapping.BackendID, method)
			return mapping.BackendID, session, nil
		} else {
			// Backend no longer available, handle failover
			if spm.config.EnableFailover {
				newBackend, err := spm.handleFailover(mapping, availableBackends)
				if err == nil {
					session := spm.getOrCreateSession(clientKey, newBackend, method)
					return newBackend, session, nil
				}
			}
			
			// Remove stale mapping
			spm.affinityMutex.Lock()
			delete(affinityStore, clientKey)
			spm.affinityMutex.Unlock()
		}
	}
	
	// Create new affinity mapping
	if len(availableBackends) > 0 {
		backend := spm.selectBackendForNewSession(clientKey, availableBackends)
		mapping := &AffinityMapping{
			ID:           uuid.New().String(),
			ClientKey:    clientKey,
			BackendID:    backend,
			Method:       method,
			CreatedAt:    time.Now(),
			LastUsedAt:   time.Now(),
			ExpiresAt:    time.Now().Add(spm.getAffinityTimeout(method)),
			Weight:       1.0,
			SessionCount: 1,
			Metadata:     make(map[string]interface{}),
		}
		
		spm.affinityMutex.Lock()
		affinityStore[clientKey] = mapping
		spm.affinityMutex.Unlock()
		
		session := spm.createNewSession(clientKey, backend, method)
		return backend, session, nil
	}
	
	return "", nil, fmt.Errorf("no available backends")
}

// getClientKey extracts client identification key based on affinity method
func (spm *SessionPersistenceManager) getClientKey(req *http.Request, method AffinityMethod) string {
	switch method {
	case AffinityMethodClientIP:
		return getClientIPFromRequest(req)
		
	case AffinityMethodCookie:
		if cookie, err := req.Cookie(spm.config.CookieName); err == nil {
			return cookie.Value
		}
		return ""
		
	case AffinityMethodHeader:
		headerValue := req.Header.Get(spm.config.HeaderName)
		if spm.config.HeaderPrefix != "" && strings.HasPrefix(headerValue, spm.config.HeaderPrefix) {
			return strings.TrimPrefix(headerValue, spm.config.HeaderPrefix)
		}
		return headerValue
		
	default:
		// Try to generate a unique key based on available information
		return fmt.Sprintf("%s_%s_%s", getClientIPFromRequest(req), req.UserAgent(), req.Header.Get("Accept"))
	}
}

// getAffinityTimeout returns timeout for affinity method
func (spm *SessionPersistenceManager) getAffinityTimeout(method AffinityMethod) time.Duration {
	switch method {
	case AffinityMethodClientIP:
		if spm.config.IPAffinityTimeout > 0 {
			return spm.config.IPAffinityTimeout
		}
		return spm.config.SessionTimeout
	default:
		return spm.config.SessionTimeout
	}
}

// isBackendAvailable checks if backend is in the available list
func (spm *SessionPersistenceManager) isBackendAvailable(backendID string, availableBackends []string) bool {
	for _, backend := range availableBackends {
		if backend == backendID {
			return true
		}
	}
	return false
}

// selectBackendForNewSession selects a backend for a new session
func (spm *SessionPersistenceManager) selectBackendForNewSession(clientKey string, availableBackends []string) string {
	if spm.config.EnableConsistentHash {
		return spm.hashRing.GetBackend(clientKey, availableBackends)
	}
	
	// Use weighted least sessions algorithm
	spm.backendMutex.RLock()
	defer spm.backendMutex.RUnlock()
	
	minSessions := int(^uint(0) >> 1) // Max int
	selectedBackend := availableBackends[0]
	
	for _, backend := range availableBackends {
		sessionCount := len(spm.backendSessions[backend])
		if sessionCount < minSessions {
			minSessions = sessionCount
			selectedBackend = backend
		}
	}
	
	return selectedBackend
}

// getOrCreateSession gets existing session or creates new one
func (spm *SessionPersistenceManager) getOrCreateSession(clientKey, backendID string, method AffinityMethod) *Session {
	// Try to find existing session
	spm.sessionsMutex.RLock()
	for _, session := range spm.sessions {
		if session.ClientID == clientKey && session.BackendID == backendID {
			session.LastAccessedAt = time.Now()
			session.RequestCount++
			spm.sessionsMutex.RUnlock()
			return session
		}
	}
	spm.sessionsMutex.RUnlock()
	
	// Create new session
	return spm.createNewSession(clientKey, backendID, method)
}

// createNewSession creates a new session
func (spm *SessionPersistenceManager) createNewSession(clientKey, backendID string, method AffinityMethod) *Session {
	session := &Session{
		ID:               uuid.New().String(),
		ClientID:         clientKey,
		BackendID:        backendID,
		AffinityMethod:   method,
		AffinityKey:      clientKey,
		CreatedAt:        time.Now(),
		LastAccessedAt:   time.Now(),
		ExpiresAt:        time.Now().Add(spm.config.SessionTimeout),
		RequestCount:     1,
		BytesTransferred: 0,
		Metadata:         make(map[string]interface{}),
		IsSticky:         true,
		FailoverCount:    0,
	}
	
	// Store session
	spm.sessionsMutex.Lock()
	spm.sessions[session.ID] = session
	spm.sessionsMutex.Unlock()
	
	// Update backend sessions
	spm.backendMutex.Lock()
	spm.backendSessions[backendID] = append(spm.backendSessions[backendID], session)
	spm.backendMutex.Unlock()
	
	// Store in backend if configured
	if spm.storageBackend != nil {
		spm.storageBackend.Store(session)
	}
	
	return session
}

// handleFailover handles backend failover for a session
func (spm *SessionPersistenceManager) handleFailover(mapping *AffinityMapping, availableBackends []string) (string, error) {
	if !spm.config.EnableFailover {
		return "", fmt.Errorf("failover disabled")
	}
	
	// Select new backend
	newBackend := spm.selectBackendForNewSession(mapping.ClientKey, availableBackends)
	
	switch spm.config.FailoverMode {
	case FailoverModeImmediate:
		// Immediately switch to new backend
		mapping.BackendID = newBackend
		mapping.LastUsedAt = time.Now()
		mapping.FailoverCount++
		return newBackend, nil
		
	case FailoverModeGraceful:
		// Allow existing connections to drain, new connections go to new backend
		// For simplicity, we'll just switch to new backend
		mapping.BackendID = newBackend
		mapping.LastUsedAt = time.Now()
		mapping.FailoverCount++
		return newBackend, nil
		
	case FailoverModeSticky:
		// Try to maintain stickiness if possible
		if spm.isBackendAvailable(mapping.BackendID, availableBackends) {
			return mapping.BackendID, nil
		}
		// Fallback to new backend
		mapping.BackendID = newBackend
		mapping.LastUsedAt = time.Now()
		mapping.FailoverCount++
		return newBackend, nil
		
	default:
		return newBackend, nil
	}
}

// CreateSessionCookie creates a session cookie for the response
func (spm *SessionPersistenceManager) CreateSessionCookie(session *Session) *http.Cookie {
	if spm.config.DefaultMethod != AffinityMethodCookie {
		return nil
	}
	
	cookie := &http.Cookie{
		Name:     spm.config.CookieName,
		Value:    session.AffinityKey,
		Path:     spm.config.CookiePath,
		Domain:   spm.config.CookieDomain,
		Secure:   spm.config.CookieSecure,
		HttpOnly: spm.config.CookieHTTPOnly,
	}
	
	if spm.config.CookieMaxAge > 0 {
		cookie.MaxAge = spm.config.CookieMaxAge
		cookie.Expires = time.Now().Add(time.Duration(spm.config.CookieMaxAge) * time.Second)
	} else {
		cookie.Expires = session.ExpiresAt
	}
	
	switch spm.config.CookieSameSite {
	case "strict":
		cookie.SameSite = http.SameSiteStrictMode
	case "lax":
		cookie.SameSite = http.SameSiteLaxMode
	case "none":
		cookie.SameSite = http.SameSiteNoneMode
	default:
		cookie.SameSite = http.SameSiteDefaultMode
	}
	
	return cookie
}

// UpdateSessionStats updates session statistics
func (spm *SessionPersistenceManager) UpdateSessionStats(sessionID string, bytesTransferred int64) {
	spm.sessionsMutex.Lock()
	defer spm.sessionsMutex.Unlock()
	
	if session, exists := spm.sessions[sessionID]; exists {
		session.BytesTransferred += bytesTransferred
		session.LastAccessedAt = time.Now()
		
		// Update in storage backend
		if spm.storageBackend != nil {
			spm.storageBackend.Update(session)
		}
	}
}

// Consistent hashing implementation

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		nodes:        make(map[uint32]string),
		virtualNodes: virtualNodes,
	}
}

// AddBackend adds a backend to the hash ring
func (chr *ConsistentHashRing) AddBackend(backendID string) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()
	
	for i := 0; i < chr.virtualNodes; i++ {
		hash := chr.hash(fmt.Sprintf("%s:%d", backendID, i))
		chr.nodes[hash] = backendID
	}
	
	chr.updateSortedHashes()
}

// RemoveBackend removes a backend from the hash ring
func (chr *ConsistentHashRing) RemoveBackend(backendID string) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()
	
	for i := 0; i < chr.virtualNodes; i++ {
		hash := chr.hash(fmt.Sprintf("%s:%d", backendID, i))
		delete(chr.nodes, hash)
	}
	
	chr.updateSortedHashes()
}

// GetBackend gets the backend for a key from available backends
func (chr *ConsistentHashRing) GetBackend(key string, availableBackends []string) string {
	chr.mutex.RLock()
	defer chr.mutex.RUnlock()
	
	if len(chr.sortedHashes) == 0 {
		return ""
	}
	
	keyHash := chr.hash(key)
	
	// Find the first hash >= keyHash
	idx := chr.search(keyHash)
	
	// Try to find an available backend starting from this position
	for i := 0; i < len(chr.sortedHashes); i++ {
		hashIdx := (idx + i) % len(chr.sortedHashes)
		hash := chr.sortedHashes[hashIdx]
		backendID := chr.nodes[hash]
		
		// Check if this backend is available
		for _, available := range availableBackends {
			if available == backendID {
				return backendID
			}
		}
	}
	
	return ""
}

// updateSortedHashes updates the sorted hash slice
func (chr *ConsistentHashRing) updateSortedHashes() {
	chr.sortedHashes = make([]uint32, 0, len(chr.nodes))
	for hash := range chr.nodes {
		chr.sortedHashes = append(chr.sortedHashes, hash)
	}
	
	// Sort hashes
	for i := 0; i < len(chr.sortedHashes)-1; i++ {
		for j := i + 1; j < len(chr.sortedHashes); j++ {
			if chr.sortedHashes[i] > chr.sortedHashes[j] {
				chr.sortedHashes[i], chr.sortedHashes[j] = chr.sortedHashes[j], chr.sortedHashes[i]
			}
		}
	}
}

// search performs binary search on sorted hashes
func (chr *ConsistentHashRing) search(hash uint32) int {
	left, right := 0, len(chr.sortedHashes)
	
	for left < right {
		mid := (left + right) / 2
		if chr.sortedHashes[mid] < hash {
			left = mid + 1
		} else {
			right = mid
		}
	}
	
	if left >= len(chr.sortedHashes) {
		left = 0
	}
	
	return left
}

// hash computes hash for a string
func (chr *ConsistentHashRing) hash(key string) uint32 {
	// Simple hash function - in production use a better one like FNV-1a
	h := uint32(2166136261)
	for _, c := range []byte(key) {
		h = (h ^ uint32(c)) * 16777619
	}
	return h
}

// Storage backend implementations

// MemoryStorageBackend implements in-memory storage
type MemoryStorageBackend struct {
	sessions map[string]*Session
	mutex    sync.RWMutex
}

// NewMemoryStorageBackend creates a new memory storage backend
func NewMemoryStorageBackend() *MemoryStorageBackend {
	return &MemoryStorageBackend{
		sessions: make(map[string]*Session),
	}
}

func (msb *MemoryStorageBackend) Store(session *Session) error {
	msb.mutex.Lock()
	defer msb.mutex.Unlock()
	
	sessionCopy := *session
	msb.sessions[session.ID] = &sessionCopy
	return nil
}

func (msb *MemoryStorageBackend) Retrieve(sessionID string) (*Session, error) {
	msb.mutex.RLock()
	defer msb.mutex.RUnlock()
	
	session, exists := msb.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session %s not found", sessionID)
	}
	
	sessionCopy := *session
	return &sessionCopy, nil
}

func (msb *MemoryStorageBackend) Update(session *Session) error {
	return msb.Store(session) // Same as store for memory backend
}

func (msb *MemoryStorageBackend) Delete(sessionID string) error {
	msb.mutex.Lock()
	defer msb.mutex.Unlock()
	
	delete(msb.sessions, sessionID)
	return nil
}

func (msb *MemoryStorageBackend) List(limit int, offset int) ([]*Session, error) {
	msb.mutex.RLock()
	defer msb.mutex.RUnlock()
	
	sessions := make([]*Session, 0, len(msb.sessions))
	count := 0
	
	for _, session := range msb.sessions {
		if count >= offset {
			sessionCopy := *session
			sessions = append(sessions, &sessionCopy)
			
			if limit > 0 && len(sessions) >= limit {
				break
			}
		}
		count++
	}
	
	return sessions, nil
}

func (msb *MemoryStorageBackend) Cleanup(expiredBefore time.Time) error {
	msb.mutex.Lock()
	defer msb.mutex.Unlock()
	
	for id, session := range msb.sessions {
		if session.ExpiresAt.Before(expiredBefore) {
			delete(msb.sessions, id)
		}
	}
	
	return nil
}

func (msb *MemoryStorageBackend) Close() error {
	msb.mutex.Lock()
	defer msb.mutex.Unlock()
	
	msb.sessions = make(map[string]*Session)
	return nil
}

// Background processes

// initializeStorageBackend initializes the storage backend
func (spm *SessionPersistenceManager) initializeStorageBackend() error {
	switch spm.config.StorageType {
	case StorageTypeMemory:
		spm.storageBackend = NewMemoryStorageBackend()
	case StorageTypeRedis:
		// Would implement Redis backend
		return fmt.Errorf("Redis storage backend not implemented")
	case StorageTypeDatabase:
		// Would implement database backend
		return fmt.Errorf("Database storage backend not implemented")
	case StorageTypeDistributed:
		// Would implement distributed storage backend
		return fmt.Errorf("Distributed storage backend not implemented")
	default:
		spm.storageBackend = NewMemoryStorageBackend()
	}
	
	return nil
}

// loadExistingSessions loads existing sessions from storage
func (spm *SessionPersistenceManager) loadExistingSessions() error {
	if spm.storageBackend == nil {
		return nil
	}
	
	sessions, err := spm.storageBackend.List(0, 0) // Get all sessions
	if err != nil {
		return err
	}
	
	spm.sessionsMutex.Lock()
	defer spm.sessionsMutex.Unlock()
	
	for _, session := range sessions {
		// Only load non-expired sessions
		if time.Now().Before(session.ExpiresAt) {
			spm.sessions[session.ID] = session
			
			// Update backend sessions
			spm.backendMutex.Lock()
			spm.backendSessions[session.BackendID] = append(spm.backendSessions[session.BackendID], session)
			spm.backendMutex.Unlock()
		}
	}
	
	return nil
}

// cleanupLoop periodically cleans up expired sessions
func (spm *SessionPersistenceManager) cleanupLoop() {
	ticker := time.NewTicker(spm.config.CleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-spm.ctx.Done():
			return
		case <-ticker.C:
			spm.cleanupExpiredSessions()
		}
	}
}

// cleanupExpiredSessions removes expired sessions
func (spm *SessionPersistenceManager) cleanupExpiredSessions() {
	now := time.Now()
	expiredSessions := make([]string, 0)
	
	spm.sessionsMutex.RLock()
	for id, session := range spm.sessions {
		if now.After(session.ExpiresAt) {
			expiredSessions = append(expiredSessions, id)
		}
	}
	spm.sessionsMutex.RUnlock()
	
	// Remove expired sessions
	spm.sessionsMutex.Lock()
	for _, id := range expiredSessions {
		if session := spm.sessions[id]; session != nil {
			delete(spm.sessions, id)
			
			// Remove from backend sessions
			spm.backendMutex.Lock()
			sessions := spm.backendSessions[session.BackendID]
			for i, s := range sessions {
				if s.ID == id {
					spm.backendSessions[session.BackendID] = append(sessions[:i], sessions[i+1:]...)
					break
				}
			}
			spm.backendMutex.Unlock()
		}
	}
	spm.sessionsMutex.Unlock()
	
	// Clean up storage backend
	if spm.storageBackend != nil {
		spm.storageBackend.Cleanup(now)
	}
	
	// Clean up expired affinity mappings
	spm.cleanupExpiredAffinityMappings(now)
}

// cleanupExpiredAffinityMappings removes expired affinity mappings
func (spm *SessionPersistenceManager) cleanupExpiredAffinityMappings(now time.Time) {
	spm.affinityMutex.Lock()
	defer spm.affinityMutex.Unlock()
	
	stores := []map[string]*AffinityMapping{
		spm.ipAffinity, spm.cookieAffinity, spm.headerAffinity, spm.customAffinity,
	}
	
	for _, store := range stores {
		for key, mapping := range store {
			if !mapping.ExpiresAt.IsZero() && now.After(mapping.ExpiresAt) {
				delete(store, key)
			}
		}
	}
}

// metricsCollectionLoop collects session persistence metrics
func (spm *SessionPersistenceManager) metricsCollectionLoop() {
	ticker := time.NewTicker(spm.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-spm.ctx.Done():
			return
		case <-ticker.C:
			spm.updateMetrics()
		}
	}
}

// updateMetrics updates session persistence metrics
func (spm *SessionPersistenceManager) updateMetrics() {
	spm.metricsMutex.Lock()
	defer spm.metricsMutex.Unlock()
	
	spm.sessionsMutex.RLock()
	defer spm.sessionsMutex.RUnlock()
	
	// Reset metrics
	spm.metrics.ActiveSessions = int64(len(spm.sessions))
	spm.metrics.SessionsByMethod = make(map[AffinityMethod]int64)
	spm.metrics.SessionsByBackend = make(map[string]int64)
	
	var totalDuration time.Duration
	var sessionCount int64
	
	// Count sessions by method and backend
	for _, session := range spm.sessions {
		spm.metrics.SessionsByMethod[session.AffinityMethod]++
		spm.metrics.SessionsByBackend[session.BackendID]++
		
		// Calculate average session duration
		duration := time.Since(session.CreatedAt)
		totalDuration += duration
		sessionCount++
	}
	
	// Calculate average session duration
	if sessionCount > 0 {
		spm.metrics.AverageSessionDuration = totalDuration / time.Duration(sessionCount)
	}
	
	spm.metrics.LastUpdated = time.Now()
}

// replicationLoop handles session replication
func (spm *SessionPersistenceManager) replicationLoop() {
	if !spm.config.EnableReplication {
		return
	}
	
	ticker := time.NewTicker(30 * time.Second) // Replicate every 30 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-spm.ctx.Done():
			return
		case <-ticker.C:
			spm.replicateSessions()
		}
	}
}

// replicateSessions replicates sessions across nodes
func (spm *SessionPersistenceManager) replicateSessions() {
	// Simplified replication - in practice this would replicate to other nodes
	spm.sessionsMutex.RLock()
	defer spm.sessionsMutex.RUnlock()
	
	for _, session := range spm.sessions {
		if spm.storageBackend != nil {
			spm.storageBackend.Update(session)
		}
	}
}

// generateSessionKey generates a secure random session key
func (spm *SessionPersistenceManager) generateSessionKey() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

// Public API methods

// GetActiveSessions returns all active sessions
func (spm *SessionPersistenceManager) GetActiveSessions() []*Session {
	spm.sessionsMutex.RLock()
	defer spm.sessionsMutex.RUnlock()
	
	sessions := make([]*Session, 0, len(spm.sessions))
	for _, session := range spm.sessions {
		sessionCopy := *session
		sessions = append(sessions, &sessionCopy)
	}
	
	return sessions
}

// GetSessionsByBackend returns sessions for a specific backend
func (spm *SessionPersistenceManager) GetSessionsByBackend(backendID string) []*Session {
	spm.backendMutex.RLock()
	defer spm.backendMutex.RUnlock()
	
	sessions := make([]*Session, len(spm.backendSessions[backendID]))
	for i, session := range spm.backendSessions[backendID] {
		sessionCopy := *session
		sessions[i] = &sessionCopy
	}
	
	return sessions
}

// GetMetrics returns session persistence metrics
func (spm *SessionPersistenceManager) GetMetrics() *SessionPersistenceMetrics {
	spm.metricsMutex.RLock()
	defer spm.metricsMutex.RUnlock()
	
	// Return copy of metrics
	metricsCopy := *spm.metrics
	
	// Copy maps
	metricsCopy.SessionsByMethod = make(map[AffinityMethod]int64)
	for k, v := range spm.metrics.SessionsByMethod {
		metricsCopy.SessionsByMethod[k] = v
	}
	
	metricsCopy.SessionsByBackend = make(map[string]int64)
	for k, v := range spm.metrics.SessionsByBackend {
		metricsCopy.SessionsByBackend[k] = v
	}
	
	return &metricsCopy
}

// AddBackendToHashRing adds a backend to the consistent hash ring
func (spm *SessionPersistenceManager) AddBackendToHashRing(backendID string) {
	if spm.hashRing != nil {
		spm.hashRing.AddBackend(backendID)
	}
}

// RemoveBackendFromHashRing removes a backend from the consistent hash ring
func (spm *SessionPersistenceManager) RemoveBackendFromHashRing(backendID string) {
	if spm.hashRing != nil {
		spm.hashRing.RemoveBackend(backendID)
	}
}

// DefaultSessionPersistenceConfig returns default session persistence configuration
func DefaultSessionPersistenceConfig() SessionPersistenceConfig {
	return SessionPersistenceConfig{
		EnablePersistence:     true,
		DefaultMethod:         AffinityMethodCookie,
		SessionTimeout:        30 * time.Minute,
		MaxSessions:           10000,
		CleanupInterval:       5 * time.Minute,
		CookieName:           "LBSESSION",
		CookiePath:           "/",
		CookieDomain:         "",
		CookieSecure:         false,
		CookieHTTPOnly:       true,
		CookieSameSite:       "lax",
		CookieMaxAge:         1800, // 30 minutes
		HeaderName:           "X-Session-ID",
		HeaderPrefix:         "",
		IPAffinitySubnet:     24,
		IPAffinityTimeout:    60 * time.Minute,
		EnableStickyIP:       true,
		EnableConsistentHash: true,
		HashAlgorithm:        "fnv1a",
		VirtualNodes:         100,
		EnableReplication:    false,
		ReplicationFactor:    2,
		ReplicationMode:      ReplicationModeAsync,
		StorageType:          StorageTypeMemory,
		StorageConfig:        make(map[string]interface{}),
		EnableFailover:       true,
		FailoverMode:         FailoverModeGraceful,
		FailoverTimeout:      30 * time.Second,
		AffinityWeight:       0.8,
		LoadBalanceWeight:    0.2,
		EnableMetrics:        true,
		MetricsInterval:      60 * time.Second,
	}
}