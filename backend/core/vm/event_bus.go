package vm

import (
	"context"
	"log"
	"sync"
	"time"
)

// EventType represents different types of lifecycle events
type EventType int

const (
	EventVMCreated EventType = iota
	EventVMStarted
	EventVMStopped
	EventVMPaused
	EventVMResumed
	EventVMFailed
	EventVMTerminated
	EventCheckpointCreated
	EventCheckpointRestored
	EventSnapshotCreated
	EventSnapshotRestored
	EventMigrationStarted
	EventMigrationCompleted
	EventMigrationFailed
	EventHealthCheckFailed
	EventResourceLimitExceeded
)

func (et EventType) String() string {
	events := []string{
		"VMCreated", "VMStarted", "VMStopped", "VMPaused", "VMResumed",
		"VMFailed", "VMTerminated", "CheckpointCreated", "CheckpointRestored",
		"SnapshotCreated", "SnapshotRestored", "MigrationStarted", 
		"MigrationCompleted", "MigrationFailed", "HealthCheckFailed",
		"ResourceLimitExceeded",
	}
	if int(et) < len(events) {
		return events[et]
	}
	return "Unknown"
}

// LifecycleEvent represents a VM lifecycle event
type LifecycleEvent struct {
	Type        EventType              `json:"type"`
	VMID        string                 `json:"vm_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
	Source      string                 `json:"source"`
	Severity    EventSeverity          `json:"severity"`
	Description string                 `json:"description"`
}

// EventSeverity represents event severity levels
type EventSeverity int

const (
	SeverityInfo EventSeverity = iota
	SeverityWarning
	SeverityError
	SeverityCritical
)

func (es EventSeverity) String() string {
	severities := []string{"Info", "Warning", "Error", "Critical"}
	if int(es) < len(severities) {
		return severities[es]
	}
	return "Unknown"
}

// EventHandler defines the interface for event handlers
type EventHandler interface {
	HandleEvent(event *LifecycleEvent) error
	GetEventTypes() []EventType
	GetName() string
}

// LifecycleEventBus manages event distribution and handling
type LifecycleEventBus struct {
	handlers      map[EventType][]EventHandler
	eventBuffer   chan *LifecycleEvent
	eventHistory  []*LifecycleEvent
	filters       []EventFilter
	mu            sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	maxHistory    int
	workerCount   int
}

// EventFilter defines event filtering criteria
type EventFilter struct {
	Name      string
	Types     []EventType
	VMIDs     []string
	Severity  EventSeverity
	Enabled   bool
	Predicate func(*LifecycleEvent) bool
}

// NewLifecycleEventBus creates a new event bus
func NewLifecycleEventBus() *LifecycleEventBus {
	ctx, cancel := context.WithCancel(context.Background())
	
	bus := &LifecycleEventBus{
		handlers:     make(map[EventType][]EventHandler),
		eventBuffer:  make(chan *LifecycleEvent, 1000),
		eventHistory: make([]*LifecycleEvent, 0),
		filters:      make([]EventFilter, 0),
		ctx:          ctx,
		cancel:       cancel,
		maxHistory:   10000,
		workerCount:  5,
	}
	
	// Start worker goroutines
	for i := 0; i < bus.workerCount; i++ {
		go bus.eventWorker()
	}
	
	return bus
}

// RegisterHandler registers an event handler for specific event types
func (bus *LifecycleEventBus) RegisterHandler(handler EventHandler) {
	bus.mu.Lock()
	defer bus.mu.Unlock()
	
	eventTypes := handler.GetEventTypes()
	for _, eventType := range eventTypes {
		if bus.handlers[eventType] == nil {
			bus.handlers[eventType] = make([]EventHandler, 0)
		}
		bus.handlers[eventType] = append(bus.handlers[eventType], handler)
	}
	
	log.Printf("Registered event handler %s for types %v", handler.GetName(), eventTypes)
}

// UnregisterHandler removes an event handler
func (bus *LifecycleEventBus) UnregisterHandler(handler EventHandler) {
	bus.mu.Lock()
	defer bus.mu.Unlock()
	
	eventTypes := handler.GetEventTypes()
	for _, eventType := range eventTypes {
		handlers := bus.handlers[eventType]
		for i, h := range handlers {
			if h == handler {
				bus.handlers[eventType] = append(handlers[:i], handlers[i+1:]...)
				break
			}
		}
	}
	
	log.Printf("Unregistered event handler %s", handler.GetName())
}

// AddFilter adds an event filter
func (bus *LifecycleEventBus) AddFilter(filter EventFilter) {
	bus.mu.Lock()
	defer bus.mu.Unlock()
	
	bus.filters = append(bus.filters, filter)
	log.Printf("Added event filter: %s", filter.Name)
}

// RemoveFilter removes an event filter
func (bus *LifecycleEventBus) RemoveFilter(filterName string) {
	bus.mu.Lock()
	defer bus.mu.Unlock()
	
	for i, filter := range bus.filters {
		if filter.Name == filterName {
			bus.filters = append(bus.filters[:i], bus.filters[i+1:]...)
			break
		}
	}
	
	log.Printf("Removed event filter: %s", filterName)
}

// Emit emits an event to the bus
func (bus *LifecycleEventBus) Emit(event *LifecycleEvent) {
	// Set default values if not provided
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	if event.Source == "" {
		event.Source = "lifecycle_manager"
	}
	if event.Severity == 0 {
		event.Severity = SeverityInfo
	}
	if event.Description == "" {
		event.Description = event.Type.String()
	}
	
	// Try to send to buffer, drop if full
	select {
	case bus.eventBuffer <- event:
		// Event queued successfully
	default:
		log.Printf("Event buffer full, dropping event: %s", event.Type)
	}
}

// GetEventHistory returns the event history with optional filtering
func (bus *LifecycleEventBus) GetEventHistory(vmID string, eventTypes []EventType, limit int) []*LifecycleEvent {
	bus.mu.RLock()
	defer bus.mu.RUnlock()
	
	result := make([]*LifecycleEvent, 0)
	
	// Iterate through history in reverse order (newest first)
	for i := len(bus.eventHistory) - 1; i >= 0 && len(result) < limit; i-- {
		event := bus.eventHistory[i]
		
		// Apply filters
		if vmID != "" && event.VMID != vmID {
			continue
		}
		
		if len(eventTypes) > 0 {
			matched := false
			for _, et := range eventTypes {
				if event.Type == et {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		
		result = append(result, event)
	}
	
	return result
}

// GetEventCount returns the count of events by type
func (bus *LifecycleEventBus) GetEventCount(vmID string, eventType EventType) int {
	bus.mu.RLock()
	defer bus.mu.RUnlock()
	
	count := 0
	for _, event := range bus.eventHistory {
		if (vmID == "" || event.VMID == vmID) && event.Type == eventType {
			count++
		}
	}
	
	return count
}

// Stop stops the event bus
func (bus *LifecycleEventBus) Stop() {
	bus.cancel()
}

// Event worker processes events from the buffer
func (bus *LifecycleEventBus) eventWorker() {
	for {
		select {
		case event := <-bus.eventBuffer:
			bus.processEvent(event)
		case <-bus.ctx.Done():
			return
		}
	}
}

// processEvent processes a single event
func (bus *LifecycleEventBus) processEvent(event *LifecycleEvent) {
	// Apply filters
	if !bus.shouldProcessEvent(event) {
		return
	}
	
	// Add to history
	bus.mu.Lock()
	bus.eventHistory = append(bus.eventHistory, event)
	
	// Trim history if too long
	if len(bus.eventHistory) > bus.maxHistory {
		bus.eventHistory = bus.eventHistory[len(bus.eventHistory)-bus.maxHistory:]
	}
	bus.mu.Unlock()
	
	// Get handlers for this event type
	bus.mu.RLock()
	handlers := bus.handlers[event.Type]
	handlersCopy := make([]EventHandler, len(handlers))
	copy(handlersCopy, handlers)
	bus.mu.RUnlock()
	
	// Process handlers
	for _, handler := range handlersCopy {
		go func(h EventHandler) {
			if err := h.HandleEvent(event); err != nil {
				log.Printf("Event handler %s failed: %v", h.GetName(), err)
			}
		}(handler)
	}
	
	log.Printf("Processed event: %s for VM %s", event.Type, event.VMID)
}

// shouldProcessEvent checks if an event should be processed based on filters
func (bus *LifecycleEventBus) shouldProcessEvent(event *LifecycleEvent) bool {
	bus.mu.RLock()
	defer bus.mu.RUnlock()
	
	for _, filter := range bus.filters {
		if !filter.Enabled {
			continue
		}
		
		// Check event types
		if len(filter.Types) > 0 {
			matched := false
			for _, et := range filter.Types {
				if event.Type == et {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		
		// Check VM IDs
		if len(filter.VMIDs) > 0 {
			matched := false
			for _, vmid := range filter.VMIDs {
				if event.VMID == vmid {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		
		// Check severity
		if event.Severity < filter.Severity {
			continue
		}
		
		// Check custom predicate
		if filter.Predicate != nil && !filter.Predicate(event) {
			continue
		}
		
		// If we reach here, the filter matched - return true
		return true
	}
	
	// If no filters matched, process the event (default behavior)
	return len(bus.filters) == 0
}

// Built-in event handlers

// LifecycleLoggingHandler logs all lifecycle events  
type LifecycleLoggingHandler struct {
	name string
}

// NewLifecycleLoggingHandler creates a new lifecycle logging event handler
func NewLifecycleLoggingHandler() *LifecycleLoggingHandler {
	return &LifecycleLoggingHandler{name: "lifecycle_logging_handler"}
}

// HandleEvent handles the event by logging it
func (h *LifecycleLoggingHandler) HandleEvent(event *LifecycleEvent) error {
	log.Printf("[%s] %s - VM %s: %s (severity: %s)",
		event.Timestamp.Format(time.RFC3339),
		event.Type,
		event.VMID,
		event.Description,
		event.Severity)
	return nil
}

// GetEventTypes returns the event types this handler processes
func (h *LifecycleLoggingHandler) GetEventTypes() []EventType {
	// Handle all event types
	return []EventType{
		EventVMCreated, EventVMStarted, EventVMStopped, EventVMPaused,
		EventVMResumed, EventVMFailed, EventVMTerminated,
		EventCheckpointCreated, EventCheckpointRestored,
		EventSnapshotCreated, EventSnapshotRestored,
		EventMigrationStarted, EventMigrationCompleted, EventMigrationFailed,
		EventHealthCheckFailed, EventResourceLimitExceeded,
	}
}

// GetName returns the handler name
func (h *LifecycleLoggingHandler) GetName() string {
	return h.name
}

// MetricsEventHandler updates metrics based on events
type MetricsEventHandler struct {
	name    string
	metrics *LifecycleMetrics
}

// NewMetricsEventHandler creates a new metrics event handler
func NewMetricsEventHandler(metrics *LifecycleMetrics) *MetricsEventHandler {
	return &MetricsEventHandler{
		name:    "metrics_handler",
		metrics: metrics,
	}
}

// HandleEvent handles the event by updating metrics
func (h *MetricsEventHandler) HandleEvent(event *LifecycleEvent) error {
	if h.metrics == nil {
		return nil
	}
	
	h.metrics.mu.Lock()
	defer h.metrics.mu.Unlock()
	
	eventName := event.Type.String()
	h.metrics.StateTransitions[eventName]++
	
	// Update failure count for error events
	if event.Severity == SeverityError || event.Severity == SeverityCritical {
		h.metrics.FailureCount++
	}
	
	return nil
}

// GetEventTypes returns the event types this handler processes
func (h *MetricsEventHandler) GetEventTypes() []EventType {
	return []EventType{
		EventVMCreated, EventVMStarted, EventVMStopped, EventVMPaused,
		EventVMResumed, EventVMFailed, EventVMTerminated,
		EventCheckpointCreated, EventCheckpointRestored,
		EventSnapshotCreated, EventSnapshotRestored,
		EventMigrationStarted, EventMigrationCompleted, EventMigrationFailed,
		EventHealthCheckFailed, EventResourceLimitExceeded,
	}
}

// GetName returns the handler name
func (h *MetricsEventHandler) GetName() string {
	return h.name
}