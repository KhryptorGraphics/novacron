package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DistributedLock represents a distributed lock using Raft consensus
type DistributedLock struct {
	raft   *RaftNode
	key    string
	owner  string
	ttl    time.Duration
	mu     sync.RWMutex
	cancel context.CancelFunc
}

// LockManager manages distributed locks across the cluster
type LockManager struct {
	raft   *RaftNode
	locks  map[string]*DistributedLock
	mu     sync.RWMutex
	nodeID string
}

// LockRequest represents a lock operation command
type LockRequest struct {
	Type      string        `json:"type"`       // "acquire", "release", "extend"
	Key       string        `json:"key"`        // Lock key
	Owner     string        `json:"owner"`      // Lock owner ID
	TTL       time.Duration `json:"ttl"`        // Lock TTL
	Timestamp time.Time     `json:"timestamp"`  // Request timestamp
}

// LockResponse represents the result of a lock operation
type LockResponse struct {
	Success   bool          `json:"success"`
	Owner     string        `json:"owner,omitempty"`
	ExpiresAt time.Time     `json:"expires_at,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// NewLockManager creates a new distributed lock manager
func NewLockManager(raft *RaftNode, nodeID string) *LockManager {
	lm := &LockManager{
		raft:   raft,
		locks:  make(map[string]*DistributedLock),
		nodeID: nodeID,
	}
	
	// Start applying lock commands from Raft
	go lm.processCommands()
	
	return lm
}

// AcquireLock attempts to acquire a distributed lock
func (lm *LockManager) AcquireLock(ctx context.Context, key string, ttl time.Duration) (*DistributedLock, error) {
	// Generate unique owner ID for this acquisition attempt
	owner := fmt.Sprintf("%s-%d", lm.nodeID, time.Now().UnixNano())
	
	request := LockRequest{
		Type:      "acquire",
		Key:       key,
		Owner:     owner,
		TTL:       ttl,
		Timestamp: time.Now(),
	}
	
	// Submit command to Raft cluster
	index, term, ok := lm.raft.Submit(request)
	if !ok {
		return nil, fmt.Errorf("failed to submit lock request: not leader")
	}
	
	// Wait for command to be applied
	if !lm.waitForApply(ctx, index) {
		return nil, fmt.Errorf("timeout waiting for lock command to be applied")
	}
	
	// Check if lock was acquired
	lm.mu.RLock()
	lock, exists := lm.locks[key]
	lm.mu.RUnlock()
	
	if !exists || lock.owner != owner {
		return nil, fmt.Errorf("failed to acquire lock: already held by another owner")
	}
	
	// Start TTL expiration timer
	lockCtx, cancel := context.WithCancel(ctx)
	lock.cancel = cancel
	go lm.startTTLTimer(lock, lockCtx)
	
	// Log successful acquisition
	lm.raft.stats.mu.Lock()
	lm.raft.stats.LogEntriesCommitted++
	lm.raft.stats.mu.Unlock()
	
	_ = index // Use index to avoid unused variable warning
	_ = term  // Use term to avoid unused variable warning
	
	return lock, nil
}

// ReleaseLock releases a distributed lock
func (lm *LockManager) ReleaseLock(ctx context.Context, lock *DistributedLock) error {
	request := LockRequest{
		Type:      "release",
		Key:       lock.key,
		Owner:     lock.owner,
		Timestamp: time.Now(),
	}
	
	// Submit command to Raft cluster
	index, term, ok := lm.raft.Submit(request)
	if !ok {
		return fmt.Errorf("failed to submit release request: not leader")
	}
	
	// Wait for command to be applied
	if !lm.waitForApply(ctx, index) {
		return fmt.Errorf("timeout waiting for release command to be applied")
	}
	
	// Cancel TTL timer
	if lock.cancel != nil {
		lock.cancel()
	}
	
	_ = term // Use term to avoid unused variable warning
	
	return nil
}

// ExtendLock extends the TTL of a distributed lock
func (lm *LockManager) ExtendLock(ctx context.Context, lock *DistributedLock, newTTL time.Duration) error {
	request := LockRequest{
		Type:      "extend",
		Key:       lock.key,
		Owner:     lock.owner,
		TTL:       newTTL,
		Timestamp: time.Now(),
	}
	
	// Submit command to Raft cluster
	index, term, ok := lm.raft.Submit(request)
	if !ok {
		return fmt.Errorf("failed to submit extend request: not leader")
	}
	
	// Wait for command to be applied
	if !lm.waitForApply(ctx, index) {
		return fmt.Errorf("timeout waiting for extend command to be applied")
	}
	
	// Update local lock TTL
	lock.mu.Lock()
	lock.ttl = newTTL
	lock.mu.Unlock()
	
	_ = term // Use term to avoid unused variable warning
	
	return nil
}

// ListLocks returns all active locks
func (lm *LockManager) ListLocks() map[string]*DistributedLock {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	
	result := make(map[string]*DistributedLock)
	for k, v := range lm.locks {
		result[k] = v
	}
	
	return result
}

// processCommands processes lock commands from the Raft apply channel
func (lm *LockManager) processCommands() {
	applyCh := lm.raft.GetApplyChan()
	
	for msg := range applyCh {
		if !msg.CommandValid {
			continue
		}
		
		request, ok := msg.Command.(LockRequest)
		if !ok {
			continue // Not a lock request
		}
		
		lm.applyLockCommand(request)
	}
}

// applyLockCommand applies a lock command to the local state
func (lm *LockManager) applyLockCommand(request LockRequest) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	
	switch request.Type {
	case "acquire":
		// Check if lock is already held
		if _, exists := lm.locks[request.Key]; exists {
			// Lock is already held - for now, just reject the request
			// In a full implementation, we would check TTL expiration time
			// stored with the lock creation timestamp
			return
		}
		
		// Create new lock
		lock := &DistributedLock{
			raft:  lm.raft,
			key:   request.Key,
			owner: request.Owner,
			ttl:   request.TTL,
		}
		
		lm.locks[request.Key] = lock
		
	case "release":
		// Check if lock exists and is owned by requester
		if existingLock, exists := lm.locks[request.Key]; exists {
			if existingLock.owner == request.Owner {
				delete(lm.locks, request.Key)
			}
		}
		
	case "extend":
		// Check if lock exists and is owned by requester
		if existingLock, exists := lm.locks[request.Key]; exists {
			if existingLock.owner == request.Owner {
				existingLock.mu.Lock()
				existingLock.ttl = request.TTL
				existingLock.mu.Unlock()
			}
		}
	}
}

// waitForApply waits for a command to be applied by the state machine
func (lm *LockManager) waitForApply(ctx context.Context, index int64) bool {
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return false
		case <-timeout.C:
			return false
		case <-ticker.C:
			// Check if the command has been applied
			stats := lm.raft.GetStats()
			if stats.LogEntriesCommitted >= index {
				return true
			}
		}
	}
}

// startTTLTimer starts a timer to automatically release the lock when TTL expires
func (lm *LockManager) startTTLTimer(lock *DistributedLock, ctx context.Context) {
	timer := time.NewTimer(lock.ttl)
	defer timer.Stop()
	
	select {
	case <-ctx.Done():
		// Lock was released or context cancelled
		return
	case <-timer.C:
		// TTL expired, release the lock
		releaseCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		err := lm.ReleaseLock(releaseCtx, lock)
		if err != nil {
			// Log error but don't panic - the lock will eventually be cleaned up
			fmt.Printf("Failed to auto-release expired lock %s: %v\n", lock.key, err)
		}
	}
}

// GetLock returns information about a specific lock
func (lm *LockManager) GetLock(key string) (*DistributedLock, bool) {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	
	lock, exists := lm.locks[key]
	return lock, exists
}

// IsLocked checks if a key is currently locked
func (lm *LockManager) IsLocked(key string) bool {
	_, exists := lm.GetLock(key)
	return exists
}

// String returns a string representation of the lock
func (dl *DistributedLock) String() string {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	
	return fmt.Sprintf("Lock{key: %s, owner: %s, ttl: %v}", dl.key, dl.owner, dl.ttl)
}

// Key returns the lock key
func (dl *DistributedLock) Key() string {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	return dl.key
}

// Owner returns the lock owner
func (dl *DistributedLock) Owner() string {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	return dl.owner
}

// TTL returns the lock TTL
func (dl *DistributedLock) TTL() time.Duration {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	return dl.ttl
}