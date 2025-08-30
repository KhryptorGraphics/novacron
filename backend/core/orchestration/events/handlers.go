package events

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// EventHandlerFunc is a function type that implements EventHandler
type EventHandlerFunc struct {
	id   string
	fn   func(ctx context.Context, event *OrchestrationEvent) error
	name string
}

// NewEventHandlerFunc creates a new function-based event handler
func NewEventHandlerFunc(id, name string, fn func(ctx context.Context, event *OrchestrationEvent) error) *EventHandlerFunc {
	return &EventHandlerFunc{
		id:   id,
		fn:   fn,
		name: name,
	}
}

// HandleEvent implements EventHandler interface
func (h *EventHandlerFunc) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	return h.fn(ctx, event)
}

// GetHandlerID implements EventHandler interface
func (h *EventHandlerFunc) GetHandlerID() string {
	return h.id
}

// CompositeEventHandler handles events by delegating to multiple handlers
type CompositeEventHandler struct {
	id       string
	handlers map[EventType][]EventHandler
	mu       sync.RWMutex
	logger   *logrus.Logger
}

// NewCompositeEventHandler creates a new composite event handler
func NewCompositeEventHandler(id string, logger *logrus.Logger) *CompositeEventHandler {
	return &CompositeEventHandler{
		id:       id,
		handlers: make(map[EventType][]EventHandler),
		logger:   logger,
	}
}

// AddHandler adds a handler for specific event types
func (c *CompositeEventHandler) AddHandler(eventTypes []EventType, handler EventHandler) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, eventType := range eventTypes {
		c.handlers[eventType] = append(c.handlers[eventType], handler)
	}
}

// RemoveHandler removes a handler for specific event types
func (c *CompositeEventHandler) RemoveHandler(eventTypes []EventType, handlerID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, eventType := range eventTypes {
		handlers := c.handlers[eventType]
		for i, handler := range handlers {
			if handler.GetHandlerID() == handlerID {
				// Remove handler from slice
				c.handlers[eventType] = append(handlers[:i], handlers[i+1:]...)
				break
			}
		}
	}
}

// HandleEvent implements EventHandler interface
func (c *CompositeEventHandler) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	c.mu.RLock()
	handlers := c.handlers[event.Type]
	c.mu.RUnlock()

	if len(handlers) == 0 {
		c.logger.WithField("event_type", event.Type).Debug("No handlers registered for event type")
		return nil
	}

	var errors []error
	for _, handler := range handlers {
		if err := handler.HandleEvent(ctx, event); err != nil {
			c.logger.WithError(err).WithFields(logrus.Fields{
				"event_id":   event.ID,
				"event_type": event.Type,
				"handler_id": handler.GetHandlerID(),
			}).Error("Handler failed to process event")
			errors = append(errors, fmt.Errorf("handler %s: %w", handler.GetHandlerID(), err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("handler errors: %v", errors)
	}

	return nil
}

// GetHandlerID implements EventHandler interface
func (c *CompositeEventHandler) GetHandlerID() string {
	return c.id
}

// FilteredEventHandler filters events based on criteria before handling
type FilteredEventHandler struct {
	id       string
	filter   EventFilter
	handler  EventHandler
	logger   *logrus.Logger
}

// EventFilter defines criteria for filtering events
type EventFilter struct {
	Sources    []string                        `json:"sources,omitempty"`
	Priorities []EventPriority   `json:"priorities,omitempty"`
	Tags       map[string]string              `json:"tags,omitempty"`
	Predicate  func(*OrchestrationEvent) bool `json:"-"`
}

// NewFilteredEventHandler creates a new filtered event handler
func NewFilteredEventHandler(id string, filter EventFilter, handler EventHandler, logger *logrus.Logger) *FilteredEventHandler {
	return &FilteredEventHandler{
		id:      id,
		filter:  filter,
		handler: handler,
		logger:  logger,
	}
}

// HandleEvent implements EventHandler interface
func (f *FilteredEventHandler) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	if !f.shouldHandle(event) {
		f.logger.WithFields(logrus.Fields{
			"event_id":   event.ID,
			"event_type": event.Type,
			"source":     event.Source,
		}).Debug("Event filtered out")
		return nil
	}

	return f.handler.HandleEvent(ctx, event)
}

// GetHandlerID implements EventHandler interface
func (f *FilteredEventHandler) GetHandlerID() string {
	return f.id
}

// shouldHandle determines if the event passes the filter
func (f *FilteredEventHandler) shouldHandle(event *OrchestrationEvent) bool {
	// Check sources filter
	if len(f.filter.Sources) > 0 {
		found := false
		for _, source := range f.filter.Sources {
			if event.Source == source {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check priorities filter
	if len(f.filter.Priorities) > 0 {
		found := false
		for _, priority := range f.filter.Priorities {
			if event.Priority == priority {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check tags filter
	if len(f.filter.Tags) > 0 {
		eventTags, ok := event.Metadata["tags"].(map[string]string)
		if !ok {
			return false
		}

		for key, value := range f.filter.Tags {
			if eventTags[key] != value {
				return false
			}
		}
	}

	// Check custom predicate
	if f.filter.Predicate != nil {
		return f.filter.Predicate(event)
	}

	return true
}

// RetryEventHandler wraps another handler with retry logic
type RetryEventHandler struct {
	id          string
	handler     EventHandler
	maxRetries  int
	backoffFunc func(attempt int) time.Duration
	logger      *logrus.Logger
}

// NewRetryEventHandler creates a new retry event handler
func NewRetryEventHandler(id string, handler EventHandler, maxRetries int, logger *logrus.Logger) *RetryEventHandler {
	return &RetryEventHandler{
		id:         id,
		handler:    handler,
		maxRetries: maxRetries,
		backoffFunc: func(attempt int) time.Duration {
			// Exponential backoff: 100ms * 2^attempt
			return time.Duration(100*math.Pow(2, float64(attempt))) * time.Millisecond
		},
		logger: logger,
	}
}

// HandleEvent implements EventHandler interface with retry logic
func (r *RetryEventHandler) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	var lastErr error

	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := r.backoffFunc(attempt - 1)
			r.logger.WithFields(logrus.Fields{
				"event_id": event.ID,
				"attempt":  attempt,
				"backoff":  backoff,
			}).Info("Retrying event handling")

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		if err := r.handler.HandleEvent(ctx, event); err != nil {
			lastErr = err
			r.logger.WithError(err).WithFields(logrus.Fields{
				"event_id": event.ID,
				"attempt":  attempt + 1,
				"max_retries": r.maxRetries,
			}).Warn("Event handling failed")
			continue
		}

		// Success
		if attempt > 0 {
			r.logger.WithFields(logrus.Fields{
				"event_id": event.ID,
				"attempt":  attempt + 1,
			}).Info("Event handling succeeded after retry")
		}
		return nil
	}

	return fmt.Errorf("event handling failed after %d attempts: %w", r.maxRetries+1, lastErr)
}

// GetHandlerID implements EventHandler interface
func (r *RetryEventHandler) GetHandlerID() string {
	return r.id
}

// AsyncEventHandler processes events asynchronously in a worker pool
type AsyncEventHandler struct {
	id          string
	handler     EventHandler
	workerCount int
	bufferSize  int
	workers     []chan *eventTask
	logger      *logrus.Logger
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

type eventTask struct {
	ctx   context.Context
	event *OrchestrationEvent
	done  chan error
}

// NewAsyncEventHandler creates a new async event handler
func NewAsyncEventHandler(id string, handler EventHandler, workerCount, bufferSize int, logger *logrus.Logger) *AsyncEventHandler {
	ctx, cancel := context.WithCancel(context.Background())
	
	ah := &AsyncEventHandler{
		id:          id,
		handler:     handler,
		workerCount: workerCount,
		bufferSize:  bufferSize,
		workers:     make([]chan *eventTask, workerCount),
		logger:      logger,
		ctx:         ctx,
		cancel:      cancel,
	}

	// Start workers
	for i := 0; i < workerCount; i++ {
		worker := make(chan *eventTask, bufferSize)
		ah.workers[i] = worker
		ah.wg.Add(1)
		
		go func(workerID int, taskChan chan *eventTask) {
			defer ah.wg.Done()
			
			for {
				select {
				case <-ctx.Done():
					return
				case task := <-taskChan:
					if task == nil {
						return
					}
					
					err := ah.handler.HandleEvent(task.ctx, task.event)
					select {
					case task.done <- err:
					case <-task.ctx.Done():
						// Context cancelled, ignore
					}
				}
			}
		}(i, worker)
	}

	return ah
}

// HandleEvent implements EventHandler interface
func (a *AsyncEventHandler) HandleEvent(ctx context.Context, event *OrchestrationEvent) error {
	// Select worker based on event hash for consistency
	workerIndex := int(event.Priority) % a.workerCount
	
	task := &eventTask{
		ctx:   ctx,
		event: event,
		done:  make(chan error, 1),
	}

	select {
	case a.workers[workerIndex] <- task:
		// Task queued successfully
	case <-ctx.Done():
		return ctx.Err()
	default:
		return fmt.Errorf("worker queue full")
	}

	// Wait for completion
	select {
	case err := <-task.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GetHandlerID implements EventHandler interface
func (a *AsyncEventHandler) GetHandlerID() string {
	return a.id
}

// Shutdown gracefully shuts down the async handler
func (a *AsyncEventHandler) Shutdown() {
	a.cancel()
	
	// Close all worker channels
	for _, worker := range a.workers {
		close(worker)
	}
	
	// Wait for workers to finish
	a.wg.Wait()
}