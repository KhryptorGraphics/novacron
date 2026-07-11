package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DistributedLock represents a distributed lock using Raft consensus
type DistributedLock struct {
	raft      *RaftNode
	key       string
	owner     string
	ttl       time.Duration
	expiresAt time.Time // wall-clock expiry (request timestamp + ttl); zero means no TTL
	mu        sync.RWMutex
}

// LockManager manages distributed locks across the cluster
type LockManager struct {
	raft   *RaftNode
	locks  map[string]*DistributedLock
	mu     sync.RWMutex
	nodeID string

	// applyCh is this manager's own committed-entry stream, obtained via
	// RaftNode.Subscribe(). Each LockManager needs its own full copy of the
	// stream so that several managers backed by the same node build identical
	// lock state; sharing one channel would split the stream between them and
	// break conflict detection.
	applyCh <-chan ApplyMsg

	// done is closed once processCommands has exited, e.g. after the
	// backing RaftNode is stopped. Exposed via Done() for graceful
	// shutdown sequencing and tests.
	done chan struct{}
}

// LockRequest represents a lock operation command
type LockRequest struct {
	Type      string        `json:"type"`      // "acquire", "release", "extend"
	Key       string        `json:"key"`       // Lock key
	Owner     string        `json:"owner"`     // Lock owner ID
	TTL       time.Duration `json:"ttl"`       // Lock TTL
	Timestamp time.Time     `json:"timestamp"` // Request timestamp
}

// LockResponse represents the result of a lock operation
type LockResponse struct {
	Success   bool      `json:"success"`
	Owner     string    `json:"owner,omitempty"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
	Error     string    `json:"error,omitempty"`
}

// NewLockManager creates a new distributed lock manager
func NewLockManager(raft *RaftNode, nodeID string) *LockManager {
	lm := &LockManager{
		raft:   raft,
		locks:  make(map[string]*DistributedLock),
		nodeID: nodeID,
		done:   make(chan struct{}),
	}

	// Subscribe synchronously (before returning / before any Acquire) so no
	// committed entry is missed between construction and the goroutine start.
	lm.applyCh = raft.Subscribe()

	// Start applying lock commands from Raft
	go lm.processCommands()

	return lm
}

// Done returns a channel that is closed once processCommands has exited
// (e.g. after the backing RaftNode is stopped via Stop()). Callers can use
// this to wait for the lock manager's background goroutine to fully
// terminate during graceful shutdown or in tests.
func (lm *LockManager) Done() <-chan struct{} {
	return lm.done
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
	index, _, ok := lm.raft.Submit(request)
	if !ok {
		return nil, fmt.Errorf("failed to submit lock request: not leader")
	}

	// Wait for command to be applied
	if !lm.waitForApply(ctx, index) {
		return nil, fmt.Errorf("timeout waiting for lock command to be applied")
	}

	// Check if the lock is now held by this acquisition. If another owner won
	// (or still holds a live lock), the applied command did not install us.
	lm.mu.RLock()
	lock, exists := lm.locks[key]
	lm.mu.RUnlock()

	if !exists || lock.owner != owner {
		return nil, fmt.Errorf("failed to acquire lock: already held by another owner")
	}

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
	index, _, ok := lm.raft.Submit(request)
	if !ok {
		return fmt.Errorf("failed to submit release request: not leader")
	}

	// Wait for command to be applied
	if !lm.waitForApply(ctx, index) {
		return fmt.Errorf("timeout waiting for release command to be applied")
	}

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
	index, _, ok := lm.raft.Submit(request)
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

// processCommands processes lock commands from the Raft apply channel.
// It exits promptly when the backing RaftNode is stopped: RaftNode.Stop
// cancels the node's context but deliberately never closes applyCh (the
// sole writer, applyLoop, could still be mid-send), so this loop also
// selects on RaftNode.Done() rather than relying solely on `range applyCh`
// -- otherwise it would block forever after Stop(), leaking the goroutine.
func (lm *LockManager) processCommands() {
	defer close(lm.done)

	applyCh := lm.applyCh
	stopped := lm.raft.Done()

	for {
		select {
		case <-stopped:
			return
		case msg, ok := <-applyCh:
			if !ok {
				return
			}
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
}

// applyLockCommand applies a lock command to the local state
func (lm *LockManager) applyLockCommand(request LockRequest) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	switch request.Type {
	case "acquire":
		// Reject only if a still-live lock is held. TTL expiration is lazy:
		// a lock whose deadline has passed (as of this request's timestamp) is
		// treated as free and may be taken over here, rather than being reaped
		// by a background timer. The request timestamp -- not the local clock --
		// is used so every replica reaches the same decision from the same log.
		if existingLock, exists := lm.locks[request.Key]; exists && !existingLock.isExpired(request.Timestamp) {
			return
		}

		// Create (or take over) the lock.
		lock := &DistributedLock{
			raft:      lm.raft,
			key:       request.Key,
			owner:     request.Owner,
			ttl:       request.TTL,
			expiresAt: expiryFor(request),
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
				existingLock.expiresAt = expiryFor(request)
				existingLock.mu.Unlock()
			}
		}
	}
}

// expiryFor computes a lock's wall-clock expiry from a request. A non-positive
// TTL yields a zero time, meaning the lock never expires.
func expiryFor(request LockRequest) time.Time {
	if request.TTL <= 0 {
		return time.Time{}
	}
	return request.Timestamp.Add(request.TTL)
}

// isExpired reports whether the lock's TTL has elapsed as of now. A lock with
// no TTL (zero expiresAt) never expires.
func (dl *DistributedLock) isExpired(now time.Time) bool {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	return !dl.expiresAt.IsZero() && now.After(dl.expiresAt)
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
