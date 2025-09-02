package events

// EventType represents the type of event
type EventType string

const (
	// VM Events
	EventTypeVMCreated EventType = "vm.created"
	EventTypeVMUpdated EventType = "vm.updated"
	EventTypeVMDeleted EventType = "vm.deleted"
	EventTypeVMStarted EventType = "vm.started"
	EventTypeVMStopped EventType = "vm.stopped"
	
	// Node Events
	EventTypeNodeAdded   EventType = "node.added"
	EventTypeNodeRemoved EventType = "node.removed"
	EventTypeNodeUpdated EventType = "node.updated"
	
	// Storage Events
	EventTypeStorageCreated EventType = "storage.created"
	EventTypeStorageDeleted EventType = "storage.deleted"
	EventTypeStorageUpdated EventType = "storage.updated"
	
	// Network Events
	EventTypeNetworkCreated EventType = "network.created"
	EventTypeNetworkDeleted EventType = "network.deleted"
	EventTypeNetworkUpdated EventType = "network.updated"
	
	// System Events
	EventTypeSystemAlert        EventType = "system.alert"
	EventTypeSystemHealthUpdate EventType = "system.health.update"
	EventTypeSystemError        EventType = "system.error"
)
