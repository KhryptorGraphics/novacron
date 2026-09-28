package vm

import (
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

	// VMEventUpdated is emitted when a VM is updated
	VMEventUpdated VMEventType = "updated"

	// VMEventError is emitted on VM errors
	VMEventError VMEventType = "error"
)

// VMEvent represents an event related to a VM
type VMEvent struct {
	Type      VMEventType            `json:"type"`
	VM        *VM                    `json:"vm"`
	Timestamp time.Time              `json:"timestamp"`
	NodeID    string                 `json:"node_id"`
	Message   string                 `json:"message,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
}
