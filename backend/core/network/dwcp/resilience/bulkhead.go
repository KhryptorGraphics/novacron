package resilience

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

var (
	// ErrBulkheadFull is returned when the bulkhead is at capacity
	ErrBulkheadFull = errors.New("bulkhead is full")
	// ErrBulkheadTimeout is returned when waiting for bulkhead times out
	ErrBulkheadTimeout = errors.New("bulkhead wait timeout")
)

// Bulkhead implements the bulkhead pattern for fault isolation
type Bulkhead struct {
	name          string
	maxConcurrent int
	maxWaitTime   time.Duration
	sem           chan struct{}
	waitQueue     chan struct{}
	active        int64
	rejected      int64
	completed     int64
	totalTime     int64 // Total execution time in nanoseconds
	logger        *zap.Logger
	mu            sync.RWMutex
}

// NewBulkhead creates a new bulkhead
func NewBulkhead(name string, maxConcurrent int, maxQueueSize int, maxWaitTime time.Duration, logger *zap.Logger) *Bulkhead {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &Bulkhead{
		name:          name,
		maxConcurrent: maxConcurrent,
		maxWaitTime:   maxWaitTime,
		sem:           make(chan struct{}, maxConcurrent),
		waitQueue:     make(chan struct{}, maxQueueSize),
		logger:        logger,
	}
}

// Execute runs a function with bulkhead protection
func (b *Bulkhead) Execute(fn func() error) error {
	return b.ExecuteWithContext(context.Background(), fn)
}

// ExecuteWithContext runs a function with bulkhead protection and context
func (b *Bulkhead) ExecuteWithContext(ctx context.Context, fn func() error) error {
	// Try to acquire semaphore immediately
	select {
	case b.sem <- struct{}{}:
		return b.executeFunction(fn)
	default:
		// Semaphore full, try to queue
	}

	// Try to enter wait queue
	select {
	case b.waitQueue <- struct{}{}:
		// Successfully queued, now wait for semaphore
		defer func() { <-b.waitQueue }()
	default:
		// Queue is full, reject immediately
		atomic.AddInt64(&b.rejected, 1)
		b.logger.Debug("Bulkhead rejecting request - queue full",
			zap.String("name", b.name),
			zap.Int64("rejected", atomic.LoadInt64(&b.rejected)))
		return ErrBulkheadFull
	}

	// Wait for semaphore with timeout
	waitCtx, cancel := context.WithTimeout(ctx, b.maxWaitTime)
	defer cancel()

	select {
	case b.sem <- struct{}{}:
		return b.executeFunction(fn)
	case <-waitCtx.Done():
		atomic.AddInt64(&b.rejected, 1)
		if errors.Is(waitCtx.Err(), context.DeadlineExceeded) {
			b.logger.Debug("Bulkhead wait timeout",
				zap.String("name", b.name),
				zap.Duration("waitTime", b.maxWaitTime))
			return ErrBulkheadTimeout
		}
		return waitCtx.Err()
	}
}

// executeFunction executes the function and tracks metrics
func (b *Bulkhead) executeFunction(fn func() error) error {
	atomic.AddInt64(&b.active, 1)
	startTime := time.Now()

	defer func() {
		duration := time.Since(startTime)
		atomic.AddInt64(&b.totalTime, int64(duration))
		atomic.AddInt64(&b.active, -1)
		atomic.AddInt64(&b.completed, 1)
		<-b.sem
	}()

	return fn()
}

// TryExecute attempts to execute without waiting
func (b *Bulkhead) TryExecute(fn func() error) error {
	select {
	case b.sem <- struct{}{}:
		return b.executeFunction(fn)
	default:
		atomic.AddInt64(&b.rejected, 1)
		return ErrBulkheadFull
	}
}

// GetMetrics returns bulkhead metrics
func (b *Bulkhead) GetMetrics() BulkheadMetrics {
	active := atomic.LoadInt64(&b.active)
	rejected := atomic.LoadInt64(&b.rejected)
	completed := atomic.LoadInt64(&b.completed)
	totalTime := atomic.LoadInt64(&b.totalTime)

	avgExecutionTime := time.Duration(0)
	if completed > 0 {
		avgExecutionTime = time.Duration(totalTime / completed)
	}

	utilization := float64(active) / float64(b.maxConcurrent)
	queueUtilization := float64(len(b.waitQueue)) / float64(cap(b.waitQueue))

	return BulkheadMetrics{
		Name:             b.name,
		Active:           active,
		Rejected:         rejected,
		Completed:        completed,
		Capacity:         int64(b.maxConcurrent),
		Utilization:      utilization,
		QueueSize:        int64(len(b.waitQueue)),
		QueueCapacity:    int64(cap(b.waitQueue)),
		QueueUtilization: queueUtilization,
		AvgExecutionTime: avgExecutionTime,
	}
}

// Reset resets the bulkhead metrics
func (b *Bulkhead) Reset() {
	atomic.StoreInt64(&b.rejected, 0)
	atomic.StoreInt64(&b.completed, 0)
	atomic.StoreInt64(&b.totalTime, 0)
	b.logger.Info("Bulkhead metrics reset", zap.String("name", b.name))
}

// ThreadPoolBulkhead provides thread pool based isolation
type ThreadPoolBulkhead struct {
	name       string
	pool       chan func()
	workers    int
	queueSize  int
	active     int64
	rejected   int64
	completed  int64
	logger     *zap.Logger
	stopCh     chan struct{}
	wg         sync.WaitGroup
}

// NewThreadPoolBulkhead creates a new thread pool bulkhead
func NewThreadPoolBulkhead(name string, workers, queueSize int, logger *zap.Logger) *ThreadPoolBulkhead {
	if logger == nil {
		logger = zap.NewNop()
	}

	tpb := &ThreadPoolBulkhead{
		name:      name,
		pool:      make(chan func(), queueSize),
		workers:   workers,
		queueSize: queueSize,
		logger:    logger,
		stopCh:    make(chan struct{}),
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		tpb.wg.Add(1)
		go tpb.worker(i)
	}

	return tpb
}

// Submit submits a task to the thread pool
func (tpb *ThreadPoolBulkhead) Submit(task func()) error {
	select {
	case tpb.pool <- task:
		return nil
	default:
		atomic.AddInt64(&tpb.rejected, 1)
		tpb.logger.Debug("Thread pool rejecting task - queue full",
			zap.String("name", tpb.name),
			zap.Int64("rejected", atomic.LoadInt64(&tpb.rejected)))
		return ErrBulkheadFull
	}
}

// SubmitWithTimeout submits a task with a timeout
func (tpb *ThreadPoolBulkhead) SubmitWithTimeout(task func(), timeout time.Duration) error {
	select {
	case tpb.pool <- task:
		return nil
	case <-time.After(timeout):
		atomic.AddInt64(&tpb.rejected, 1)
		return ErrBulkheadTimeout
	}
}

// worker processes tasks from the pool
func (tpb *ThreadPoolBulkhead) worker(id int) {
	defer tpb.wg.Done()

	for {
		select {
		case task := <-tpb.pool:
			if task != nil {
				atomic.AddInt64(&tpb.active, 1)
				task()
				atomic.AddInt64(&tpb.active, -1)
				atomic.AddInt64(&tpb.completed, 1)
			}
		case <-tpb.stopCh:
			return
		}
	}
}

// Stop stops the thread pool
func (tpb *ThreadPoolBulkhead) Stop() {
	close(tpb.stopCh)
	tpb.wg.Wait()
	close(tpb.pool)
	tpb.logger.Info("Thread pool bulkhead stopped", zap.String("name", tpb.name))
}

// GetMetrics returns thread pool metrics
func (tpb *ThreadPoolBulkhead) GetMetrics() ThreadPoolMetrics {
	return ThreadPoolMetrics{
		Name:         tpb.name,
		Workers:      tpb.workers,
		Active:       atomic.LoadInt64(&tpb.active),
		Queued:       int64(len(tpb.pool)),
		QueueSize:    int64(tpb.queueSize),
		Rejected:     atomic.LoadInt64(&tpb.rejected),
		Completed:    atomic.LoadInt64(&tpb.completed),
		Utilization:  float64(atomic.LoadInt64(&tpb.active)) / float64(tpb.workers),
	}
}

// SemaphoreBulkhead provides semaphore-based resource limiting
type SemaphoreBulkhead struct {
	name      string
	semaphore chan struct{}
	capacity  int
	active    int64
	rejected  int64
	completed int64
	logger    *zap.Logger
}

// NewSemaphoreBulkhead creates a new semaphore bulkhead
func NewSemaphoreBulkhead(name string, capacity int, logger *zap.Logger) *SemaphoreBulkhead {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &SemaphoreBulkhead{
		name:      name,
		semaphore: make(chan struct{}, capacity),
		capacity:  capacity,
		logger:    logger,
	}
}

// Acquire tries to acquire a semaphore permit
func (sb *SemaphoreBulkhead) Acquire() bool {
	select {
	case sb.semaphore <- struct{}{}:
		atomic.AddInt64(&sb.active, 1)
		return true
	default:
		atomic.AddInt64(&sb.rejected, 1)
		return false
	}
}

// AcquireWithTimeout tries to acquire with timeout
func (sb *SemaphoreBulkhead) AcquireWithTimeout(timeout time.Duration) bool {
	select {
	case sb.semaphore <- struct{}{}:
		atomic.AddInt64(&sb.active, 1)
		return true
	case <-time.After(timeout):
		atomic.AddInt64(&sb.rejected, 1)
		return false
	}
}

// Release releases a semaphore permit
func (sb *SemaphoreBulkhead) Release() {
	select {
	case <-sb.semaphore:
		atomic.AddInt64(&sb.active, -1)
		atomic.AddInt64(&sb.completed, 1)
	default:
		sb.logger.Warn("Attempting to release non-acquired semaphore",
			zap.String("name", sb.name))
	}
}

// GetMetrics returns semaphore metrics
func (sb *SemaphoreBulkhead) GetMetrics() SemaphoreMetrics {
	active := atomic.LoadInt64(&sb.active)
	return SemaphoreMetrics{
		Name:        sb.name,
		Capacity:    int64(sb.capacity),
		Active:      active,
		Available:   int64(sb.capacity) - active,
		Rejected:    atomic.LoadInt64(&sb.rejected),
		Completed:   atomic.LoadInt64(&sb.completed),
		Utilization: float64(active) / float64(sb.capacity),
	}
}

// Metrics types

// BulkheadMetrics contains bulkhead metrics
type BulkheadMetrics struct {
	Name             string
	Active           int64
	Rejected         int64
	Completed        int64
	Capacity         int64
	Utilization      float64
	QueueSize        int64
	QueueCapacity    int64
	QueueUtilization float64
	AvgExecutionTime time.Duration
}

// ThreadPoolMetrics contains thread pool metrics
type ThreadPoolMetrics struct {
	Name        string
	Workers     int
	Active      int64
	Queued      int64
	QueueSize   int64
	Rejected    int64
	Completed   int64
	Utilization float64
}

// SemaphoreMetrics contains semaphore metrics
type SemaphoreMetrics struct {
	Name        string
	Capacity    int64
	Active      int64
	Available   int64
	Rejected    int64
	Completed   int64
	Utilization float64
}