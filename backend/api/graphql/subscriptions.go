package graphql

import (
	"sync"
)

// SubscriptionManager manages GraphQL subscriptions
type SubscriptionManager struct {
	mu sync.RWMutex

	// VM subscriptions
	vmStateListeners   []func(*VM)
	vmMetricsListeners map[string][]func(*VMMetrics)

	// Migration subscriptions
	migrationListeners map[string][]func(*Migration)

	// Alert subscriptions
	alertListeners []func(*Alert)

	// Event subscriptions
	eventListeners map[string][]func(*Event)

	// Metrics subscriptions
	systemMetricsListeners []func(*SystemMetrics)
	nodeMetricsListeners   map[string][]func(*NodeMetrics)
}

// NewSubscriptionManager creates a new subscription manager
func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		vmMetricsListeners:   make(map[string][]func(*VMMetrics)),
		migrationListeners:   make(map[string][]func(*Migration)),
		eventListeners:       make(map[string][]func(*Event)),
		nodeMetricsListeners: make(map[string][]func(*NodeMetrics)),
	}
}

// VM Subscriptions

// SubscribeVMStateChange subscribes to VM state changes
func (sm *SubscriptionManager) SubscribeVMStateChange(listener func(*VM)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.vmStateListeners = append(sm.vmStateListeners, listener)
}

// PublishVMStateChange publishes a VM state change
func (sm *SubscriptionManager) PublishVMStateChange(vm *VM) {
	sm.mu.RLock()
	listeners := make([]func(*VM), len(sm.vmStateListeners))
	copy(listeners, sm.vmStateListeners)
	sm.mu.RUnlock()

	for _, listener := range listeners {
		go listener(vm)
	}
}

// SubscribeVMMetrics subscribes to VM metrics updates
func (sm *SubscriptionManager) SubscribeVMMetrics(vmID string, listener func(*VMMetrics)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.vmMetricsListeners[vmID] == nil {
		sm.vmMetricsListeners[vmID] = make([]func(*VMMetrics), 0)
	}
	sm.vmMetricsListeners[vmID] = append(sm.vmMetricsListeners[vmID], listener)
}

// PublishVMMetrics publishes VM metrics
func (sm *SubscriptionManager) PublishVMMetrics(vmID string, metrics *VMMetrics) {
	sm.mu.RLock()
	listeners := make([]func(*VMMetrics), len(sm.vmMetricsListeners[vmID]))
	copy(listeners, sm.vmMetricsListeners[vmID])
	sm.mu.RUnlock()

	for _, listener := range listeners {
		go listener(metrics)
	}
}

// Migration Subscriptions

// SubscribeMigrationProgress subscribes to migration progress updates
func (sm *SubscriptionManager) SubscribeMigrationProgress(migrationID string, listener func(*Migration)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.migrationListeners[migrationID] == nil {
		sm.migrationListeners[migrationID] = make([]func(*Migration), 0)
	}
	sm.migrationListeners[migrationID] = append(sm.migrationListeners[migrationID], listener)
}

// PublishMigrationProgress publishes migration progress
func (sm *SubscriptionManager) PublishMigrationProgress(migration *Migration) {
	sm.mu.RLock()
	listeners := make([]func(*Migration), len(sm.migrationListeners[migration.ID]))
	copy(listeners, sm.migrationListeners[migration.ID])
	sm.mu.RUnlock()

	for _, listener := range listeners {
		go listener(migration)
	}
}

// Alert Subscriptions

// SubscribeNewAlert subscribes to new alerts
func (sm *SubscriptionManager) SubscribeNewAlert(listener func(*Alert)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.alertListeners = append(sm.alertListeners, listener)
}

// PublishNewAlert publishes a new alert
func (sm *SubscriptionManager) PublishNewAlert(alert *Alert) {
	sm.mu.RLock()
	listeners := make([]func(*Alert), len(sm.alertListeners))
	copy(listeners, sm.alertListeners)
	sm.mu.RUnlock()

	for _, listener := range listeners {
		go listener(alert)
	}
}

// Event Subscriptions

// SubscribeSystemEvent subscribes to system events
func (sm *SubscriptionManager) SubscribeSystemEvent(eventType string, listener func(*Event)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if eventType == "" {
		eventType = "all"
	}

	if sm.eventListeners[eventType] == nil {
		sm.eventListeners[eventType] = make([]func(*Event), 0)
	}
	sm.eventListeners[eventType] = append(sm.eventListeners[eventType], listener)
}

// PublishSystemEvent publishes a system event
func (sm *SubscriptionManager) PublishSystemEvent(event *Event) {
	sm.mu.RLock()

	// Get listeners for specific event type
	specificListeners := make([]func(*Event), len(sm.eventListeners[event.Type]))
	copy(specificListeners, sm.eventListeners[event.Type])

	// Get listeners for all events
	allListeners := make([]func(*Event), len(sm.eventListeners["all"]))
	copy(allListeners, sm.eventListeners["all"])

	sm.mu.RUnlock()

	// Notify specific listeners
	for _, listener := range specificListeners {
		go listener(event)
	}

	// Notify all-event listeners
	for _, listener := range allListeners {
		go listener(event)
	}
}

// Metrics Subscriptions

// SubscribeSystemMetrics subscribes to system metrics updates
func (sm *SubscriptionManager) SubscribeSystemMetrics(listener func(*SystemMetrics)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.systemMetricsListeners = append(sm.systemMetricsListeners, listener)
}

// PublishSystemMetrics publishes system metrics
func (sm *SubscriptionManager) PublishSystemMetrics(metrics *SystemMetrics) {
	sm.mu.RLock()
	listeners := make([]func(*SystemMetrics), len(sm.systemMetricsListeners))
	copy(listeners, sm.systemMetricsListeners)
	sm.mu.RUnlock()

	for _, listener := range listeners {
		go listener(metrics)
	}
}

// SubscribeNodeMetrics subscribes to node metrics updates
func (sm *SubscriptionManager) SubscribeNodeMetrics(nodeID string, listener func(*NodeMetrics)) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if nodeID == "" {
		nodeID = "all"
	}

	if sm.nodeMetricsListeners[nodeID] == nil {
		sm.nodeMetricsListeners[nodeID] = make([]func(*NodeMetrics), 0)
	}
	sm.nodeMetricsListeners[nodeID] = append(sm.nodeMetricsListeners[nodeID], listener)
}

// PublishNodeMetrics publishes node metrics
func (sm *SubscriptionManager) PublishNodeMetrics(nodeID string, metrics *NodeMetrics) {
	sm.mu.RLock()

	// Get listeners for specific node
	specificListeners := make([]func(*NodeMetrics), len(sm.nodeMetricsListeners[nodeID]))
	copy(specificListeners, sm.nodeMetricsListeners[nodeID])

	// Get listeners for all nodes
	allListeners := make([]func(*NodeMetrics), len(sm.nodeMetricsListeners["all"]))
	copy(allListeners, sm.nodeMetricsListeners["all"])

	sm.mu.RUnlock()

	// Notify specific listeners
	for _, listener := range specificListeners {
		go listener(metrics)
	}

	// Notify all-node listeners
	for _, listener := range allListeners {
		go listener(metrics)
	}
}

// Cleanup methods

// UnsubscribeVMStateChange removes all VM state change listeners
func (sm *SubscriptionManager) UnsubscribeVMStateChange() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.vmStateListeners = []func(*VM){}
}

// UnsubscribeVMMetrics removes VM metrics listeners for a specific VM
func (sm *SubscriptionManager) UnsubscribeVMMetrics(vmID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	delete(sm.vmMetricsListeners, vmID)
}

// UnsubscribeMigration removes migration listeners for a specific migration
func (sm *SubscriptionManager) UnsubscribeMigration(migrationID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	delete(sm.migrationListeners, migrationID)
}

// UnsubscribeAllAlerts removes all alert listeners
func (sm *SubscriptionManager) UnsubscribeAllAlerts() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.alertListeners = []func(*Alert){}
}

// UnsubscribeSystemEvents removes event listeners for a specific type
func (sm *SubscriptionManager) UnsubscribeSystemEvents(eventType string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if eventType == "" {
		eventType = "all"
	}

	delete(sm.eventListeners, eventType)
}

// UnsubscribeSystemMetrics removes all system metrics listeners
func (sm *SubscriptionManager) UnsubscribeSystemMetrics() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.systemMetricsListeners = []func(*SystemMetrics){}
}

// UnsubscribeNodeMetrics removes node metrics listeners for a specific node
func (sm *SubscriptionManager) UnsubscribeNodeMetrics(nodeID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if nodeID == "" {
		nodeID = "all"
	}

	delete(sm.nodeMetricsListeners, nodeID)
}
