package events

import (
	"time"
)

// OrchestrationEvent represents an event in the orchestration system
type OrchestrationEvent struct {
	ID          string                 `json:"id"`
	Type        EventType              `json:"type"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Priority    EventPriority          `json:"priority"`
	TTL         time.Duration          `json:"ttl,omitempty"`
}

// EventType defines the type of orchestration event
type EventType string

const (
	EventTypeVMCreated         EventType = "vm.created"
	EventTypeVMUpdated         EventType = "vm.updated"
	EventTypeVMStarted         EventType = "vm.started"
	EventTypeVMStopped         EventType = "vm.stopped"
	EventTypeVMDeleted         EventType = "vm.deleted"
	EventTypeVMMetrics         EventType = "vm.metrics"
	EventTypeNodeAdded         EventType = "node.added"
	EventTypeNodeRemoved       EventType = "node.removed"
	EventTypeNodeUpdated       EventType = "node.updated"
	EventTypeNodeMetrics       EventType = "node.metrics"
	EventTypeNodeFailure       EventType = "node.failure"
	EventTypeNodeRecovered     EventType = "node.recovered"
	EventTypeStorageCreated    EventType = "storage.created"
	EventTypeStorageDeleted    EventType = "storage.deleted"
	EventTypeStorageUpdated    EventType = "storage.updated"
	EventTypeNetworkCreated    EventType = "network.created"
	EventTypeNetworkDeleted    EventType = "network.deleted"
	EventTypeNetworkUpdated    EventType = "network.updated"
	EventTypeSystemAlert       EventType = "system.alert"
	EventTypeSystemHealthUpdate EventType = "system.health.update"
	EventTypeSystemError       EventType = "system.error"
	EventTypeScalingTriggered  EventType = "scaling.triggered"
	EventTypeHealingTriggered  EventType = "healing.triggered"
	EventTypePolicyUpdated     EventType = "policy.updated"
	EventTypeOrchestrationLog  EventType = "orchestration.log"
)

// EventPriority defines the priority level of an event
type EventPriority int

const (
	PriorityLow      EventPriority = 1
	PriorityNormal   EventPriority = 3
	PriorityHigh     EventPriority = 5
	PriorityCritical EventPriority = 10
)