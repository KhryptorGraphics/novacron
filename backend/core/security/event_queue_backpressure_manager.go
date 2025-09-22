package security

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	queueSizeGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_security_event_queue_size",
			Help: "Current size of security event queues",
		},
		[]string{"queue_type", "priority"},
	)

	queueUtilizationGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_security_event_queue_utilization",
			Help: "Queue utilization percentage (0-1)",
		},
		[]string{"queue_type", "priority"},
	)

	backpressureEventsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_backpressure_events_total",
			Help: "Total number of backpressure events by strategy",
		},
		[]string{"strategy", "queue_type"},
	)

	eventProcessingLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "novacron_security_event_processing_latency_seconds",
			Help: "Time taken to process security events",
		},
		[]string{"event_type", "priority"},
	)

	throttleRate = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_security_event_throttle_rate",
			Help: "Current throttle rate for event processing",
		},
		[]string{"queue_type"},
	)
)

// EventPriority defines priority levels for events
type EventPriority int

const (
	PriorityLow EventPriority = iota
	PriorityMedium
	PriorityHigh
	PriorityUrgent
	PriorityCritical
)

func (p EventPriority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityMedium:
		return "medium"
	case PriorityHigh:
		return "high"
	case PriorityUrgent:
		return "urgent"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// PrioritySecurityEvent wraps SecurityEvent with priority
type PrioritySecurityEvent struct {
	Event     SecurityEvent
	Priority  EventPriority
	Timestamp time.Time
	Retries   int32
}

// BackpressureStrategy defines how to handle backpressure
type BackpressureStrategy int

const (
	StrategyDrop BackpressureStrategy = iota
	StrategyThrottle
	StrategySpill
	StrategyShedLoad
	StrategyAdaptive
)

// BackpressureConfig configures backpressure handling
type BackpressureConfig struct {
	// Queue sizes for each priority
	QueueSizes map[EventPriority]int

	// High water mark percentages for backpressure activation (0.0-1.0)
	HighWaterMark float64
	LowWaterMark  float64

	// Strategy to use when backpressure is detected
	Strategy BackpressureStrategy

	// Throttling configuration
	BaseThrottleRate   time.Duration // Base processing interval
	MaxThrottleRate    time.Duration // Maximum throttling delay
	ThrottleMultiplier float64       // Multiplier when throttling

	// Spill configuration
	SpillToFile    bool
	SpillDirectory string
	MaxSpillFiles  int

	// Load shedding configuration
	LoadSheddingRatio float64 // Percentage of low priority events to drop (0.0-1.0)

	// Adaptive configuration
	AdaptiveWindow         time.Duration // Window for calculating event rates
	AdaptiveThreshold      int           // Events per window to trigger adaptation
	AdaptiveRecoveryFactor float64       // Factor for backing off throttling

	// Retry configuration
	MaxRetries    int
	RetryInterval time.Duration
	RetryBackoff  float64 // Exponential backoff multiplier

	// Metrics configuration
	MetricsInterval time.Duration
}

// DefaultBackpressureConfig returns default configuration
func DefaultBackpressureConfig() BackpressureConfig {
	return BackpressureConfig{
		QueueSizes: map[EventPriority]int{
			PriorityLow:      1000,
			PriorityMedium:   2000,
			PriorityHigh:     5000,
			PriorityUrgent:   8000,
			PriorityCritical: 10000,
		},
		HighWaterMark:          0.80,
		LowWaterMark:           0.60,
		Strategy:               StrategyAdaptive,
		BaseThrottleRate:       10 * time.Millisecond,
		MaxThrottleRate:        1 * time.Second,
		ThrottleMultiplier:     1.5,
		SpillToFile:            true,
		SpillDirectory:         "/tmp/security_event_spill",
		MaxSpillFiles:          100,
		LoadSheddingRatio:      0.30,
		AdaptiveWindow:         30 * time.Second,
		AdaptiveThreshold:      1000,
		AdaptiveRecoveryFactor: 0.95,
		MaxRetries:             3,
		RetryInterval:          1 * time.Second,
		RetryBackoff:           2.0,
		MetricsInterval:        10 * time.Second,
	}
}

// PriorityQueue represents a priority-based queue
type PriorityQueue struct {
	queues   map[EventPriority]chan *PrioritySecurityEvent
	config   BackpressureConfig
	mu       sync.RWMutex
	closed   bool
	spillMgr *SpillManager
}

// NewPriorityQueue creates a new priority queue
func NewPriorityQueue(config BackpressureConfig) *PriorityQueue {
	pq := &PriorityQueue{
		queues:   make(map[EventPriority]chan *PrioritySecurityEvent),
		config:   config,
		spillMgr: NewSpillManager(config.SpillDirectory, config.MaxSpillFiles),
	}

	for priority, size := range config.QueueSizes {
		pq.queues[priority] = make(chan *PrioritySecurityEvent, size)
	}

	return pq
}

// Enqueue adds an event to the appropriate priority queue
func (pq *PriorityQueue) Enqueue(event *PrioritySecurityEvent) error {
	pq.mu.RLock()
	if pq.closed {
		pq.mu.RUnlock()
		return fmt.Errorf("queue is closed")
	}

	queue, exists := pq.queues[event.Priority]
	if !exists {
		pq.mu.RUnlock()
		return fmt.Errorf("unknown priority: %v", event.Priority)
	}
	pq.mu.RUnlock()

	select {
	case queue <- event:
		queueSizeGauge.WithLabelValues("security", event.Priority.String()).Set(float64(len(queue)))
		return nil
	default:
		// Queue is full, apply backpressure strategy
		return pq.handleBackpressure(event, queue)
	}
}

// Dequeue removes and returns the highest priority event
func (pq *PriorityQueue) Dequeue(ctx context.Context) (*PrioritySecurityEvent, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if pq.closed {
		return nil, fmt.Errorf("queue is closed")
	}

	// Try to dequeue from highest to lowest priority
	priorities := []EventPriority{PriorityCritical, PriorityUrgent, PriorityHigh, PriorityMedium, PriorityLow}

	for _, priority := range priorities {
		if queue, exists := pq.queues[priority]; exists {
			select {
			case event := <-queue:
				queueSizeGauge.WithLabelValues("security", priority.String()).Set(float64(len(queue)))
				return event, nil
			default:
				continue
			}
		}
	}

	// No events available, wait for one
	return pq.waitForEvent(ctx)
}

// waitForEvent waits for any event to become available
func (pq *PriorityQueue) waitForEvent(ctx context.Context) (*PrioritySecurityEvent, error) {
	priorities := []EventPriority{PriorityCritical, PriorityUrgent, PriorityHigh, PriorityMedium, PriorityLow}

	// Create a slice of cases for select
	cases := make([]reflect.SelectCase, 0, len(priorities)+1)
	queueMap := make(map[int]EventPriority)

	// Add context cancellation case
	cases = append(cases, reflect.SelectCase{
		Dir:  reflect.SelectRecv,
		Chan: reflect.ValueOf(ctx.Done()),
	})

	// Add all priority queues
	for i, priority := range priorities {
		if queue, exists := pq.queues[priority]; exists {
			cases = append(cases, reflect.SelectCase{
				Dir:  reflect.SelectRecv,
				Chan: reflect.ValueOf(queue),
			})
			queueMap[i+1] = priority
		}
	}

	chosen, value, ok := reflect.Select(cases)

	if chosen == 0 {
		// Context cancelled
		return nil, ctx.Err()
	}

	if !ok {
		return nil, fmt.Errorf("queue closed")
	}

	event := value.Interface().(*PrioritySecurityEvent)
	priority := queueMap[chosen]
	queueSizeGauge.WithLabelValues("security", priority.String()).Set(float64(len(pq.queues[priority])))

	return event, nil
}

// handleBackpressure applies the configured backpressure strategy
func (pq *PriorityQueue) handleBackpressure(event *PrioritySecurityEvent, queue chan *PrioritySecurityEvent) error {
	utilization := float64(len(queue)) / float64(cap(queue))
	queueUtilizationGauge.WithLabelValues("security", event.Priority.String()).Set(utilization)

	switch pq.config.Strategy {
	case StrategyDrop:
		return pq.dropEvent(event)
	case StrategyThrottle:
		return pq.throttleEvent(event, queue)
	case StrategySpill:
		return pq.spillEvent(event)
	case StrategyShedLoad:
		return pq.shedLoad(event, queue)
	case StrategyAdaptive:
		return pq.adaptiveStrategy(event, queue, utilization)
	default:
		return pq.dropEvent(event)
	}
}

// dropEvent drops the event and logs it
func (pq *PriorityQueue) dropEvent(event *PrioritySecurityEvent) error {
	backpressureEventsTotal.WithLabelValues("drop", "security").Inc()
	return fmt.Errorf("event dropped due to backpressure: %s", event.Event.ID)
}

// throttleEvent implements throttling strategy
func (pq *PriorityQueue) throttleEvent(event *PrioritySecurityEvent, queue chan *PrioritySecurityEvent) error {
	backpressureEventsTotal.WithLabelValues("throttle", "security").Inc()

	throttleDelay := pq.config.BaseThrottleRate
	utilization := float64(len(queue)) / float64(cap(queue))

	if utilization > pq.config.HighWaterMark {
		multiplier := (utilization - pq.config.HighWaterMark) * 10
		throttleDelay = time.Duration(float64(throttleDelay) * (1 + multiplier))

		if throttleDelay > pq.config.MaxThrottleRate {
			throttleDelay = pq.config.MaxThrottleRate
		}
	}

	throttleRate.WithLabelValues("security").Set(throttleDelay.Seconds())

	timer := time.NewTimer(throttleDelay)
	defer timer.Stop()

	select {
	case <-timer.C:
		select {
		case queue <- event:
			return nil
		default:
			return pq.spillEvent(event)
		}
	}
}

// spillEvent writes the event to disk
func (pq *PriorityQueue) spillEvent(event *PrioritySecurityEvent) error {
	backpressureEventsTotal.WithLabelValues("spill", "security").Inc()

	if pq.spillMgr != nil {
		return pq.spillMgr.SpillEvent(event)
	}

	return pq.dropEvent(event)
}

// shedLoad drops lower priority events to make room
func (pq *PriorityQueue) shedLoad(event *PrioritySecurityEvent, queue chan *PrioritySecurityEvent) error {
	backpressureEventsTotal.WithLabelValues("shed_load", "security").Inc()

	// Only shed load for high priority and above events
	if event.Priority >= PriorityHigh {
		// Try to drop some low priority events
		lowQueue := pq.queues[PriorityLow]
		mediumQueue := pq.queues[PriorityMedium]

		dropCount := int(float64(len(lowQueue)) * pq.config.LoadSheddingRatio)
		for i := 0; i < dropCount; i++ {
			select {
			case <-lowQueue:
				continue
			default:
				break
			}
		}

		// Try to drop some medium priority events if still needed
		if len(queue) >= cap(queue) {
			dropCount = int(float64(len(mediumQueue)) * pq.config.LoadSheddingRatio * 0.5)
			for i := 0; i < dropCount; i++ {
				select {
				case <-mediumQueue:
					continue
				default:
					break
				}
			}
		}

		// Try to enqueue again
		select {
		case queue <- event:
			return nil
		default:
			return pq.spillEvent(event)
		}
	}

	return pq.dropEvent(event)
}

// adaptiveStrategy combines multiple strategies based on conditions
func (pq *PriorityQueue) adaptiveStrategy(event *PrioritySecurityEvent, queue chan *PrioritySecurityEvent, utilization float64) error {
	backpressureEventsTotal.WithLabelValues("adaptive", "security").Inc()

	// Different strategies based on event priority and queue utilization
	switch {
	case event.Priority >= PriorityCritical:
		// Critical events: try load shedding first, then spill
		if err := pq.shedLoad(event, queue); err == nil {
			return nil
		}
		return pq.spillEvent(event)

	case event.Priority >= PriorityHigh && utilization < 0.95:
		// High priority: try throttling
		return pq.throttleEvent(event, queue)

	case event.Priority >= PriorityMedium && utilization < 0.90:
		// Medium priority: throttle with longer delay
		throttleDelay := time.Duration(float64(pq.config.BaseThrottleRate) * 2)
		time.Sleep(throttleDelay)
		select {
		case queue <- event:
			return nil
		default:
			return pq.spillEvent(event)
		}

	default:
		// Low priority: drop or spill
		if pq.config.SpillToFile {
			return pq.spillEvent(event)
		}
		return pq.dropEvent(event)
	}
}

// Close closes all queues
func (pq *PriorityQueue) Close() error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if pq.closed {
		return nil
	}

	pq.closed = true

	for _, queue := range pq.queues {
		close(queue)
	}

	if pq.spillMgr != nil {
		pq.spillMgr.Close()
	}

	return nil
}

// GetMetrics returns current queue metrics
func (pq *PriorityQueue) GetMetrics() map[string]interface{} {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	metrics := make(map[string]interface{})

	for priority, queue := range pq.queues {
		metrics[fmt.Sprintf("%s_queue_size", priority.String())] = len(queue)
		metrics[fmt.Sprintf("%s_queue_capacity", priority.String())] = cap(queue)
		metrics[fmt.Sprintf("%s_utilization", priority.String())] = float64(len(queue)) / float64(cap(queue))
	}

	return metrics
}

// EventQueueBackpressureManager manages backpressure for security event queues
type EventQueueBackpressureManager struct {
	priorityQueue    *PriorityQueue
	config           BackpressureConfig
	logger           *slog.Logger
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	running          int32
	processingRate   int64
	lastMetricsTime  time.Time
	metricsCollector *MetricsCollector
}

// NewEventQueueBackpressureManager creates a new backpressure manager
func NewEventQueueBackpressureManager(config BackpressureConfig, logger *slog.Logger) *EventQueueBackpressureManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &EventQueueBackpressureManager{
		priorityQueue:    NewPriorityQueue(config),
		config:           config,
		logger:           logger,
		ctx:              ctx,
		cancel:           cancel,
		lastMetricsTime:  time.Now(),
		metricsCollector: NewMetricsCollector(config.MetricsInterval),
	}
}

// Start starts the backpressure manager
func (bm *EventQueueBackpressureManager) Start() error {
	if !atomic.CompareAndSwapInt32(&bm.running, 0, 1) {
		return fmt.Errorf("backpressure manager already running")
	}

	bm.wg.Add(3)
	go bm.processEvents()
	go bm.collectMetrics()
	go bm.spillRecovery()

	bm.logger.Info("Event queue backpressure manager started")
	return nil
}

// Stop stops the backpressure manager
func (bm *EventQueueBackpressureManager) Stop() error {
	if !atomic.CompareAndSwapInt32(&bm.running, 1, 0) {
		return nil
	}

	bm.cancel()
	bm.priorityQueue.Close()
	bm.wg.Wait()

	bm.logger.Info("Event queue backpressure manager stopped")
	return nil
}

// EnqueueEvent adds an event to the queue with appropriate priority
func (bm *EventQueueBackpressureManager) EnqueueEvent(event SecurityEvent) error {
	priority := bm.calculatePriority(event)

	priorityEvent := &PrioritySecurityEvent{
		Event:     event,
		Priority:  priority,
		Timestamp: time.Now(),
		Retries:   0,
	}

	return bm.priorityQueue.Enqueue(priorityEvent)
}

// calculatePriority calculates event priority based on event characteristics
func (bm *EventQueueBackpressureManager) calculatePriority(event SecurityEvent) EventPriority {
	switch event.Severity {
	case SeverityCritical:
		return PriorityCritical
	case SeverityHigh:
		return PriorityUrgent
	case SeverityMedium:
		return PriorityHigh
	case SeverityLow:
		return PriorityMedium
	default:
		return PriorityLow
	}
}

// processEvents processes events from the priority queue
func (bm *EventQueueBackpressureManager) processEvents() {
	defer bm.wg.Done()

	for {
		select {
		case <-bm.ctx.Done():
			return
		default:
			event, err := bm.priorityQueue.Dequeue(bm.ctx)
			if err != nil {
				if err != context.Canceled {
					bm.logger.Error("Failed to dequeue event", "error", err)
				}
				continue
			}

			start := time.Now()
			if err := bm.handleEvent(event); err != nil {
				bm.logger.Error("Failed to handle event", "error", err, "event_id", event.Event.ID)
				bm.retryEvent(event)
			}

			duration := time.Since(start)
			eventProcessingLatency.WithLabelValues(string(event.Event.Type), event.Priority.String()).Observe(duration.Seconds())
			atomic.AddInt64(&bm.processingRate, 1)
		}
	}
}

// handleEvent processes a single security event
func (bm *EventQueueBackpressureManager) handleEvent(event *PrioritySecurityEvent) error {
	// This would integrate with the actual security event processing logic
	// For now, we'll simulate processing
	bm.logger.Info("Processing security event",
		"id", event.Event.ID,
		"type", event.Event.Type,
		"priority", event.Priority,
		"severity", event.Event.Severity)

	// Simulate processing time based on priority
	switch event.Priority {
	case PriorityCritical:
		time.Sleep(1 * time.Millisecond)
	case PriorityUrgent:
		time.Sleep(5 * time.Millisecond)
	case PriorityHigh:
		time.Sleep(10 * time.Millisecond)
	case PriorityMedium:
		time.Sleep(20 * time.Millisecond)
	default:
		time.Sleep(50 * time.Millisecond)
	}

	return nil
}

// retryEvent attempts to retry a failed event
func (bm *EventQueueBackpressureManager) retryEvent(event *PrioritySecurityEvent) {
	atomic.AddInt32(&event.Retries, 1)

	if int(event.Retries) >= bm.config.MaxRetries {
		bm.logger.Error("Event exceeded max retries, dropping",
			"event_id", event.Event.ID,
			"retries", event.Retries)
		return
	}

	// Exponential backoff
	delay := time.Duration(float64(bm.config.RetryInterval) *
		(bm.config.RetryBackoff * float64(event.Retries)))

	time.AfterFunc(delay, func() {
		if err := bm.priorityQueue.Enqueue(event); err != nil {
			bm.logger.Error("Failed to re-enqueue event for retry",
				"error", err,
				"event_id", event.Event.ID)
		}
	})
}

// collectMetrics periodically collects and reports metrics
func (bm *EventQueueBackpressureManager) collectMetrics() {
	defer bm.wg.Done()

	ticker := time.NewTicker(bm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-bm.ctx.Done():
			return
		case <-ticker.C:
			bm.updateMetrics()
		}
	}
}

// updateMetrics updates prometheus metrics
func (bm *EventQueueBackpressureManager) updateMetrics() {
	metrics := bm.priorityQueue.GetMetrics()

	for key, value := range metrics {
		if size, ok := value.(int); ok {
			// Update queue size metrics
			// This would be more sophisticated in a real implementation
			_ = size
		}
	}

	// Update processing rate
	currentRate := atomic.SwapInt64(&bm.processingRate, 0)
	elapsed := time.Since(bm.lastMetricsTime).Seconds()
	if elapsed > 0 {
		rate := float64(currentRate) / elapsed
		bm.logger.Debug("Processing rate", "events_per_second", rate)
	}
	bm.lastMetricsTime = time.Now()
}

// spillRecovery recovers spilled events when queues have capacity
func (bm *EventQueueBackpressureManager) spillRecovery() {
	defer bm.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-bm.ctx.Done():
			return
		case <-ticker.C:
			if bm.priorityQueue.spillMgr != nil {
				bm.priorityQueue.spillMgr.RecoverSpilledEvents(bm.priorityQueue)
			}
		}
	}
}

// GetStatus returns the current status of the backpressure manager
func (bm *EventQueueBackpressureManager) GetStatus() map[string]interface{} {
	status := make(map[string]interface{})

	status["running"] = atomic.LoadInt32(&bm.running) == 1
	status["queue_metrics"] = bm.priorityQueue.GetMetrics()
	status["processing_rate"] = atomic.LoadInt64(&bm.processingRate)

	return status
}