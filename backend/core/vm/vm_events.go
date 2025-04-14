package vm

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// VMEventType represents VM event types
type VMEventType string

const (
	// VMEventCreated is emitted when a VM is created
	VMEventCreated VMEventType = "created"
	
	// VMEventStarted is emitted when a VM is started
	VMEventStarted VMEventType = "started"
	
	// VMEventStopped is emitted when a VM is stopped
	VMEventStopped VMEventType = "stopped"
	
	// VMEventRestarted is emitted when a VM is restarted
	VMEventRestarted VMEventType = "restarted"
	
	// VMEventDeleted is emitted when a VM is deleted
	VMEventDeleted VMEventType = "deleted"
	
	// VMEventPaused is emitted when a VM is paused
	VMEventPaused VMEventType = "paused"
	
	// VMEventResumed is emitted when a VM is resumed
	VMEventResumed VMEventType = "resumed"
	
	// VMEventMigrating is emitted when a VM is being migrated
	VMEventMigrating VMEventType = "migrating"
	
	// VMEventMigrated is emitted when a VM has been migrated
	VMEventMigrated VMEventType = "migrated"
	
	// VMEventSnapshot is emitted when a VM snapshot is created
	VMEventSnapshot VMEventType = "snapshot"
	
	// VMEventBackup is emitted when a VM backup is created
	VMEventBackup VMEventType = "backup"
	
	// VMEventRestore is emitted when a VM is restored from a snapshot or backup
	VMEventRestore VMEventType = "restore"
	
	// VMEventUpdated is emitted when a VM is updated
	VMEventUpdated VMEventType = "updated"
	
	// VMEventError is emitted on VM errors
	VMEventError VMEventType = "error"
)

// VMEvent represents an event related to a VM
type VMEvent struct {
	Type      VMEventType `json:"type"`
	VM        VM          `json:"vm"`
	Timestamp time.Time   `json:"timestamp"`
	NodeID    string      `json:"node_id"`
	Message   string      `json:"message,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// VMEventHandler is a function that handles VM events
type VMEventHandler func(event VMEvent)

// VMEventManager manages VM events
type VMEventManager struct {
	handlers     map[string][]VMEventHandler
	handlersMutex sync.RWMutex
	eventHistory []VMEvent
	historyMutex sync.RWMutex
	maxHistory   int
}

// NewVMEventManager creates a new VM event manager
func NewVMEventManager(maxHistory int) *VMEventManager {
	return &VMEventManager{
		handlers:     make(map[string][]VMEventHandler),
		eventHistory: make([]VMEvent, 0, maxHistory),
		maxHistory:   maxHistory,
	}
}

// RegisterHandler registers a handler for VM events
func (m *VMEventManager) RegisterHandler(eventType string, handler VMEventHandler) {
	m.handlersMutex.Lock()
	defer m.handlersMutex.Unlock()
	
	if _, exists := m.handlers[eventType]; !exists {
		m.handlers[eventType] = make([]VMEventHandler, 0)
	}
	
	m.handlers[eventType] = append(m.handlers[eventType], handler)
}

// RegisterHandlerForAll registers a handler for all VM events
func (m *VMEventManager) RegisterHandlerForAll(handler VMEventHandler) {
	m.handlersMutex.Lock()
	defer m.handlersMutex.Unlock()
	
	if _, exists := m.handlers["*"]; !exists {
		m.handlers["*"] = make([]VMEventHandler, 0)
	}
	
	m.handlers["*"] = append(m.handlers["*"], handler)
}

// UnregisterHandler unregisters a handler for VM events
func (m *VMEventManager) UnregisterHandler(eventType string, handler VMEventHandler) {
	m.handlersMutex.Lock()
	defer m.handlersMutex.Unlock()
	
	if handlers, exists := m.handlers[eventType]; exists {
		for i, h := range handlers {
			if fmt.Sprintf("%p", h) == fmt.Sprintf("%p", handler) {
				m.handlers[eventType] = append(handlers[:i], handlers[i+1:]...)
				break
			}
		}
	}
}

// EmitEvent emits a VM event
func (m *VMEventManager) EmitEvent(event VMEvent) {
	// Add event to history
	m.historyMutex.Lock()
	m.eventHistory = append(m.eventHistory, event)
	if len(m.eventHistory) > m.maxHistory {
		m.eventHistory = m.eventHistory[len(m.eventHistory)-m.maxHistory:]
	}
	m.historyMutex.Unlock()
	
	// Call handlers
	m.handlersMutex.RLock()
	defer m.handlersMutex.RUnlock()
	
	// Call specific handlers
	if handlers, exists := m.handlers[string(event.Type)]; exists {
		for _, handler := range handlers {
			go handler(event)
		}
	}
	
	// Call wildcard handlers
	if handlers, exists := m.handlers["*"]; exists {
		for _, handler := range handlers {
			go handler(event)
		}
	}
}

// GetEventHistory returns the event history
func (m *VMEventManager) GetEventHistory() []VMEvent {
	m.historyMutex.RLock()
	defer m.historyMutex.RUnlock()
	
	history := make([]VMEvent, len(m.eventHistory))
	copy(history, m.eventHistory)
	
	return history
}

// GetEventHistoryForVM returns the event history for a VM
func (m *VMEventManager) GetEventHistoryForVM(vmID string) []VMEvent {
	m.historyMutex.RLock()
	defer m.historyMutex.RUnlock()
	
	history := make([]VMEvent, 0)
	for _, event := range m.eventHistory {
		if event.VM.ID() == vmID {
			history = append(history, event)
		}
	}
	
	return history
}

// LoggingEventHandler is a handler that logs VM events
func LoggingEventHandler(event VMEvent) {
	log.Printf("[VM Event] %s: %s - %s", event.Type, event.VM.ID(), event.Message)
}

// JSONEventHandler is a handler that outputs VM events as JSON
func JSONEventHandler(event VMEvent) {
	eventJSON, err := json.Marshal(event)
	if err != nil {
		log.Printf("Error marshaling event: %v", err)
		return
	}
	
	log.Println(string(eventJSON))
}
