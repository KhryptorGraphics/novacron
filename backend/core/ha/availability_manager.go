package ha

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// ServiceState represents the current state of a service
type ServiceState string

const (
	// ServiceStateStarting indicates the service is starting up
	ServiceStateStarting ServiceState = "starting"

	// ServiceStateRunning indicates the service is running normally
	ServiceStateRunning ServiceState = "running"

	// ServiceStateDegraded indicates the service is running but in a degraded state
	ServiceStateDegraded ServiceState = "degraded"

	// ServiceStateFailing indicates the service is failing but still partially available
	ServiceStateFailing ServiceState = "failing"

	// ServiceStateDown indicates the service is completely unavailable
	ServiceStateDown ServiceState = "down"

	// ServiceStateStopping indicates the service is in the process of stopping
	ServiceStateStopping ServiceState = "stopping"

	// ServiceStateStopped indicates the service has been stopped
	ServiceStateStopped ServiceState = "stopped"

	// ServiceStateRecovering indicates the service is recovering from a failure
	ServiceStateRecovering ServiceState = "recovering"

	// ServiceStateUnknown indicates the state of the service is unknown
	ServiceStateUnknown ServiceState = "unknown"
)

// ServiceRole defines the role of a service instance in an HA setup
type ServiceRole string

const (
	// ServiceRolePrimary indicates a primary (leader) service instance
	ServiceRolePrimary ServiceRole = "primary"

	// ServiceRoleSecondary indicates a secondary (follower) service instance
	ServiceRoleSecondary ServiceRole = "secondary"

	// ServiceRoleWitness indicates a witness service instance (for quorum)
	ServiceRoleWitness ServiceRole = "witness"

	// ServiceRoleStandby indicates a standby service instance
	ServiceRoleStandby ServiceRole = "standby"

	// ServiceRoleActive indicates an active service instance
	ServiceRoleActive ServiceRole = "active"

	// ServiceRolePassive indicates a passive service instance
	ServiceRolePassive ServiceRole = "passive"
)

// ServiceHealthCheck represents a health check for a service
type ServiceHealthCheck struct {
	// Name is the name of the health check
	Name string

	// Type is the type of health check (e.g., "http", "tcp", "exec")
	Type string

	// Endpoint is the endpoint to check (URL, port, command)
	Endpoint string

	// Interval is how often to perform the check
	Interval time.Duration

	// Timeout is the timeout for the check
	Timeout time.Duration

	// HealthyThreshold is the number of consecutive successful checks required
	// to consider the service healthy
	HealthyThreshold int

	// UnhealthyThreshold is the number of consecutive failed checks required
	// to consider the service unhealthy
	UnhealthyThreshold int

	// SuccessStatus contains success status codes (for HTTP checks)
	SuccessStatus []int

	// AdditionalHeaders contains additional headers (for HTTP checks)
	AdditionalHeaders map[string]string

	// ExpectedResponse contains the expected response (for TCP/HTTP checks)
	ExpectedResponse string

	// Script is the script to execute (for exec checks)
	Script string

	// Args are the arguments for the script (for exec checks)
	Args []string
}

// ServiceInstance represents an instance of a service
type ServiceInstance struct {
	// ID is the unique identifier for this service instance
	ID string

	// Name is the human-readable name of the service
	Name string

	// Type is the type of service
	Type string

	// NodeID is the ID of the node hosting this service instance
	NodeID string

	// Address is the network address of the service
	Address string

	// Port is the port the service is listening on
	Port int

	// Role is the role of this service instance
	Role ServiceRole

	// State is the current state of the service
	State ServiceState

	// Priority is the priority of this instance (higher is more preferred)
	Priority int

	// LastStateChange is when the state last changed
	LastStateChange time.Time

	// HealthChecks are the health checks for this service
	HealthChecks []ServiceHealthCheck

	// Metadata contains additional metadata for the service
	Metadata map[string]string

	// Dependencies are other services this service depends on
	Dependencies []string
}

// AvailabilityGroup represents a group of services that form an HA unit
type AvailabilityGroup struct {
	// ID is the unique identifier for this availability group
	ID string

	// Name is the human-readable name of the group
	Name string

	// Services are the services in this group
	Services []*ServiceInstance

	// DesiredInstances is the desired number of instances
	DesiredInstances int

	// MinInstances is the minimum number of instances needed for operation
	MinInstances int

	// ActiveInstances is the current number of active instances
	ActiveInstances int

	// State is the overall state of the group
	State ServiceState

	// Strategy is the HA strategy for this group
	Strategy AvailabilityStrategy

	// AutoFailback indicates if automatic failback is enabled
	AutoFailback bool

	// CreatedAt is when this group was created
	CreatedAt time.Time

	// UpdatedAt is when this group was last updated
	UpdatedAt time.Time
}

// AvailabilityStrategyType defines the type of HA strategy
type AvailabilityStrategyType string

const (
	// StrategyActiveActive indicates all instances are active
	StrategyActiveActive AvailabilityStrategyType = "active-active"

	// StrategyActivePassive indicates one instance is active, others are passive
	StrategyActivePassive AvailabilityStrategyType = "active-passive"

	// StrategyNPlus1 indicates N+1 redundancy
	StrategyNPlus1 AvailabilityStrategyType = "n+1"

	// StrategyNPlus2 indicates N+2 redundancy
	StrategyNPlus2 AvailabilityStrategyType = "n+2"

	// StrategyLeaderFollower indicates a leader/follower topology
	StrategyLeaderFollower AvailabilityStrategyType = "leader-follower"

	// StrategyQuorum indicates a quorum-based topology
	StrategyQuorum AvailabilityStrategyType = "quorum"
)

// AvailabilityStrategy defines how a service achieves high availability
type AvailabilityStrategy struct {
	// Type is the type of strategy
	Type AvailabilityStrategyType

	// FailoverTimeout is the timeout for failover operations
	FailoverTimeout time.Duration

	// FailbackTimeout is the timeout for failback operations
	FailbackTimeout time.Duration

	// QuorumSize is the number of instances required for quorum
	QuorumSize int

	// MaxFailovers is the maximum number of failovers allowed in a time period
	MaxFailovers int

	// FailoverPeriod is the time period for MaxFailovers
	FailoverPeriod time.Duration

	// SplitBrainDetection indicates if split-brain detection is enabled
	SplitBrainDetection bool

	// SplitBrainResolution indicates the strategy for resolving split-brain
	SplitBrainResolution string

	// Parameters contains additional strategy-specific parameters
	Parameters map[string]interface{}
}

// FailoverEvent represents a failover event
type FailoverEvent struct {
	// ID is the unique identifier for this event
	ID string

	// GroupID is the ID of the availability group
	GroupID string

	// OldPrimaryID is the ID of the old primary service
	OldPrimaryID string

	// NewPrimaryID is the ID of the new primary service
	NewPrimaryID string

	// Reason is the reason for the failover
	Reason string

	// StartTime is when the failover started
	StartTime time.Time

	// EndTime is when the failover completed
	EndTime time.Time

	// Duration is the duration of the failover
	Duration time.Duration

	// Success indicates if the failover was successful
	Success bool

	// Error is the error message if the failover failed
	Error string

	// Manual indicates if this was a manual failover
	Manual bool

	// InitiatedBy is who initiated the failover
	InitiatedBy string
}

// AvailabilityManagerConfig contains configuration for the availability manager
type AvailabilityManagerConfig struct {
	// HealthCheckInterval is how often to perform health checks
	HealthCheckInterval time.Duration

	// StateRefreshInterval is how often to refresh state information
	StateRefreshInterval time.Duration

	// AutoFailover indicates if automatic failover is enabled
	AutoFailover bool

	// AutoFailoverDelay is the delay before automatic failover
	AutoFailoverDelay time.Duration

	// HeartbeatInterval is how often to send heartbeats
	HeartbeatInterval time.Duration

	// HeartbeatTimeout is the timeout for heartbeats
	HeartbeatTimeout time.Duration

	// MaxFailoverRetries is the maximum number of failover retries
	MaxFailoverRetries int

	// DefaultStrategy is the default HA strategy
	DefaultStrategy AvailabilityStrategy
}

// DefaultAvailabilityManagerConfig returns a default configuration
func DefaultAvailabilityManagerConfig() AvailabilityManagerConfig {
	return AvailabilityManagerConfig{
		HealthCheckInterval:  10 * time.Second,
		StateRefreshInterval: 30 * time.Second,
		AutoFailover:         true,
		AutoFailoverDelay:    30 * time.Second,
		HeartbeatInterval:    5 * time.Second,
		HeartbeatTimeout:     15 * time.Second,
		MaxFailoverRetries:   3,
		DefaultStrategy: AvailabilityStrategy{
			Type:                 StrategyActivePassive,
			FailoverTimeout:      2 * time.Minute,
			FailbackTimeout:      5 * time.Minute,
			MaxFailovers:         3,
			FailoverPeriod:       1 * time.Hour,
			SplitBrainDetection:  true,
			SplitBrainResolution: "newest_timestamp",
		},
	}
}

// AvailabilityManager manages the high availability of services
type AvailabilityManager struct {
	config AvailabilityManagerConfig

	// groups maps group IDs to availability groups
	groups     map[string]*AvailabilityGroup
	groupMutex sync.RWMutex

	// services maps service IDs to service instances
	services     map[string]*ServiceInstance
	serviceMutex sync.RWMutex

	// failoverEvents stores recent failover events
	failoverEvents     []*FailoverEvent
	failoverEventMutex sync.RWMutex

	// serviceStateCallbacks maps service IDs to callbacks for state changes
	serviceStateCallbacks     map[string][]func(*ServiceInstance, ServiceState)
	serviceStateCallbackMutex sync.RWMutex

	ctx    context.Context
	cancel context.CancelFunc
}

// NewAvailabilityManager creates a new availability manager
func NewAvailabilityManager(config AvailabilityManagerConfig) *AvailabilityManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &AvailabilityManager{
		config:                config,
		groups:                make(map[string]*AvailabilityGroup),
		services:              make(map[string]*ServiceInstance),
		failoverEvents:        make([]*FailoverEvent, 0),
		serviceStateCallbacks: make(map[string][]func(*ServiceInstance, ServiceState)),
		ctx:                   ctx,
		cancel:                cancel,
	}
}

// Start starts the availability manager
func (m *AvailabilityManager) Start() error {
	log.Println("Starting availability manager")

	// Start the health check loop
	go m.healthCheckLoop()

	// Start the state refresh loop
	go m.stateRefreshLoop()

	// Start the heartbeat loop
	go m.heartbeatLoop()

	return nil
}

// Stop stops the availability manager
func (m *AvailabilityManager) Stop() error {
	log.Println("Stopping availability manager")

	m.cancel()

	return nil
}

// healthCheckLoop periodically performs health checks on services
func (m *AvailabilityManager) healthCheckLoop() {
	ticker := time.NewTicker(m.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.performHealthChecks()
		}
	}
}

// stateRefreshLoop periodically refreshes the state of services
func (m *AvailabilityManager) stateRefreshLoop() {
	ticker := time.NewTicker(m.config.StateRefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.refreshServiceStates()
		}
	}
}

// heartbeatLoop periodically sends heartbeats to track service liveness
func (m *AvailabilityManager) heartbeatLoop() {
	ticker := time.NewTicker(m.config.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.sendHeartbeats()
		}
	}
}

// performHealthChecks performs health checks on all services
func (m *AvailabilityManager) performHealthChecks() {
	m.serviceMutex.RLock()
	serviceIDs := make([]string, 0, len(m.services))
	for id := range m.services {
		serviceIDs = append(serviceIDs, id)
	}
	m.serviceMutex.RUnlock()

	for _, id := range serviceIDs {
		m.checkServiceHealth(id)
	}
}

// checkServiceHealth checks the health of a service
func (m *AvailabilityManager) checkServiceHealth(serviceID string) {
	m.serviceMutex.RLock()
	service, exists := m.services[serviceID]
	m.serviceMutex.RUnlock()

	if !exists {
		return
	}

	// Skip services without health checks
	if len(service.HealthChecks) == 0 {
		return
	}

	// In a real implementation, this would perform actual health checks
	// For now, we'll just simulate a healthy service
	healthy := true

	// Update service state based on health check results
	if healthy && service.State != ServiceStateRunning {
		m.updateServiceState(serviceID, ServiceStateRunning)
	} else if !healthy && service.State == ServiceStateRunning {
		m.updateServiceState(serviceID, ServiceStateDegraded)
	}
}

// updateServiceState updates the state of a service
func (m *AvailabilityManager) updateServiceState(serviceID string, newState ServiceState) {
	m.serviceMutex.Lock()
	service, exists := m.services[serviceID]
	if !exists {
		m.serviceMutex.Unlock()
		return
	}

	oldState := service.State
	service.State = newState
	service.LastStateChange = time.Now()
	m.serviceMutex.Unlock()

	log.Printf("Service %s state changed: %s -> %s", serviceID, oldState, newState)

	// Notify callbacks about state change
	m.notifyStateChange(service, newState)

	// Check if we need to perform failover
	if m.config.AutoFailover && oldState == ServiceStateRunning &&
		(newState == ServiceStateDown || newState == ServiceStateFailing) {
		go m.handlePotentialFailover(serviceID)
	}
}

// notifyStateChange notifies registered callbacks about a state change
func (m *AvailabilityManager) notifyStateChange(service *ServiceInstance, newState ServiceState) {
	m.serviceStateCallbackMutex.RLock()
	callbacks, exists := m.serviceStateCallbacks[service.ID]
	m.serviceStateCallbackMutex.RUnlock()

	if !exists {
		return
	}

	for _, callback := range callbacks {
		go callback(service, newState)
	}
}

// handlePotentialFailover checks if failover is needed and performs it if necessary
func (m *AvailabilityManager) handlePotentialFailover(serviceID string) {
	// In a real implementation, this would check if the service is a primary
	// and if failover is needed

	// For now, just log that we would failover
	log.Printf("Would perform failover for service %s", serviceID)
}

// refreshServiceStates refreshes the state of all services
func (m *AvailabilityManager) refreshServiceStates() {
	m.serviceMutex.RLock()
	serviceIDs := make([]string, 0, len(m.services))
	for id := range m.services {
		serviceIDs = append(serviceIDs, id)
	}
	m.serviceMutex.RUnlock()

	for _, id := range serviceIDs {
		// In a real implementation, this would query the actual service state
		// For now, just check the health
		m.checkServiceHealth(id)
	}

	// Update group states based on service states
	m.updateGroupStates()
}

// updateGroupStates updates the state of all groups based on service states
func (m *AvailabilityManager) updateGroupStates() {
	m.groupMutex.Lock()
	defer m.groupMutex.Unlock()

	for _, group := range m.groups {
		activeCount := 0
		anyDegraded := false
		anyFailing := false

		for _, service := range group.Services {
			m.serviceMutex.RLock()
			state := service.State
			m.serviceMutex.RUnlock()

			if state == ServiceStateRunning || state == ServiceStateDegraded {
				activeCount++
			}

			if state == ServiceStateDegraded {
				anyDegraded = true
			} else if state == ServiceStateFailing {
				anyFailing = true
			}
		}

		group.ActiveInstances = activeCount

		// Update group state based on service states
		if activeCount >= group.DesiredInstances {
			group.State = ServiceStateRunning
		} else if activeCount >= group.MinInstances {
			if anyDegraded {
				group.State = ServiceStateDegraded
			} else if anyFailing {
				group.State = ServiceStateFailing
			} else {
				group.State = ServiceStateRunning
			}
		} else if activeCount > 0 {
			group.State = ServiceStateDegraded
		} else {
			group.State = ServiceStateDown
		}

		group.UpdatedAt = time.Now()
	}
}

// sendHeartbeats sends heartbeats for all services
func (m *AvailabilityManager) sendHeartbeats() {
	// In a real implementation, this would send heartbeats to a coordination service
	// For now, just log that we would send heartbeats
	log.Printf("Would send heartbeats for %d services", len(m.services))
}

// RegisterService registers a service with the availability manager
func (m *AvailabilityManager) RegisterService(service *ServiceInstance) error {
	if service == nil || service.ID == "" {
		return fmt.Errorf("service cannot be nil and must have an ID")
	}

	m.serviceMutex.Lock()
	defer m.serviceMutex.Unlock()

	m.services[service.ID] = service
	log.Printf("Registered service %s", service.ID)

	return nil
}

// UnregisterService unregisters a service from the availability manager
func (m *AvailabilityManager) UnregisterService(serviceID string) error {
	m.serviceMutex.Lock()
	defer m.serviceMutex.Unlock()

	if _, exists := m.services[serviceID]; !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	delete(m.services, serviceID)
	log.Printf("Unregistered service %s", serviceID)

	return nil
}

// GetService gets a service by ID
func (m *AvailabilityManager) GetService(serviceID string) (*ServiceInstance, error) {
	m.serviceMutex.RLock()
	defer m.serviceMutex.RUnlock()

	service, exists := m.services[serviceID]
	if !exists {
		return nil, fmt.Errorf("service not found: %s", serviceID)
	}

	return service, nil
}

// CreateAvailabilityGroup creates a new availability group
func (m *AvailabilityManager) CreateAvailabilityGroup(group *AvailabilityGroup) error {
	if group == nil || group.ID == "" {
		return fmt.Errorf("group cannot be nil and must have an ID")
	}

	m.groupMutex.Lock()
	defer m.groupMutex.Unlock()

	if _, exists := m.groups[group.ID]; exists {
		return fmt.Errorf("group already exists: %s", group.ID)
	}

	// Set creation time if not set
	if group.CreatedAt.IsZero() {
		group.CreatedAt = time.Now()
	}
	group.UpdatedAt = time.Now()

	m.groups[group.ID] = group
	log.Printf("Created availability group %s", group.ID)

	return nil
}

// DeleteAvailabilityGroup deletes an availability group
func (m *AvailabilityManager) DeleteAvailabilityGroup(groupID string) error {
	m.groupMutex.Lock()
	defer m.groupMutex.Unlock()

	if _, exists := m.groups[groupID]; !exists {
		return fmt.Errorf("group not found: %s", groupID)
	}

	delete(m.groups, groupID)
	log.Printf("Deleted availability group %s", groupID)

	return nil
}

// GetAvailabilityGroup gets an availability group by ID
func (m *AvailabilityManager) GetAvailabilityGroup(groupID string) (*AvailabilityGroup, error) {
	m.groupMutex.RLock()
	defer m.groupMutex.RUnlock()

	group, exists := m.groups[groupID]
	if !exists {
		return nil, fmt.Errorf("group not found: %s", groupID)
	}

	return group, nil
}

// AddServiceToGroup adds a service to an availability group
func (m *AvailabilityManager) AddServiceToGroup(groupID, serviceID string) error {
	m.groupMutex.Lock()
	defer m.groupMutex.Unlock()

	group, exists := m.groups[groupID]
	if !exists {
		return fmt.Errorf("group not found: %s", groupID)
	}

	m.serviceMutex.RLock()
	service, exists := m.services[serviceID]
	m.serviceMutex.RUnlock()

	if !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	// Check if service is already in the group
	for _, s := range group.Services {
		if s.ID == serviceID {
			return fmt.Errorf("service already in group: %s", serviceID)
		}
	}

	group.Services = append(group.Services, service)
	group.UpdatedAt = time.Now()

	log.Printf("Added service %s to group %s", serviceID, groupID)

	return nil
}

// RemoveServiceFromGroup removes a service from an availability group
func (m *AvailabilityManager) RemoveServiceFromGroup(groupID, serviceID string) error {
	m.groupMutex.Lock()
	defer m.groupMutex.Unlock()

	group, exists := m.groups[groupID]
	if !exists {
		return fmt.Errorf("group not found: %s", groupID)
	}

	// Find and remove the service
	for i, s := range group.Services {
		if s.ID == serviceID {
			group.Services = append(group.Services[:i], group.Services[i+1:]...)
			group.UpdatedAt = time.Now()
			log.Printf("Removed service %s from group %s", serviceID, groupID)
			return nil
		}
	}

	return fmt.Errorf("service not in group: %s", serviceID)
}

// PerformManualFailover performs a manual failover
func (m *AvailabilityManager) PerformManualFailover(groupID, newPrimaryID, reason string) error {
	m.groupMutex.RLock()
	group, exists := m.groups[groupID]
	m.groupMutex.RUnlock()

	if !exists {
		return fmt.Errorf("group not found: %s", groupID)
	}

	// Find the current primary and the new primary
	var oldPrimary, newPrimary *ServiceInstance
	for _, s := range group.Services {
		if s.Role == ServiceRolePrimary {
			oldPrimary = s
		}
		if s.ID == newPrimaryID {
			newPrimary = s
		}
	}

	if oldPrimary == nil {
		return fmt.Errorf("no primary service found in group: %s", groupID)
	}

	if newPrimary == nil {
		return fmt.Errorf("new primary service not found in group: %s", newPrimaryID)
	}

	if oldPrimary.ID == newPrimaryID {
		return fmt.Errorf("new primary is already the primary: %s", newPrimaryID)
	}

	// Create failover event
	event := &FailoverEvent{
		ID:           fmt.Sprintf("failover-%d", time.Now().UnixNano()),
		GroupID:      groupID,
		OldPrimaryID: oldPrimary.ID,
		NewPrimaryID: newPrimaryID,
		Reason:       reason,
		StartTime:    time.Now(),
		Manual:       true,
		InitiatedBy:  "admin", // In a real implementation, this would be the user
	}

	// In a real implementation, this would perform the actual failover
	// For now, just update the roles and log
	m.serviceMutex.Lock()
	if oldPrimary, exists := m.services[oldPrimary.ID]; exists {
		oldPrimary.Role = ServiceRoleSecondary
	}
	if newPrimary, exists := m.services[newPrimaryID]; exists {
		newPrimary.Role = ServiceRolePrimary
	}
	m.serviceMutex.Unlock()

	// Update the event
	event.EndTime = time.Now()
	event.Duration = event.EndTime.Sub(event.StartTime)
	event.Success = true

	// Store the event
	m.failoverEventMutex.Lock()
	m.failoverEvents = append(m.failoverEvents, event)
	m.failoverEventMutex.Unlock()

	log.Printf("Manual failover completed: %s -> %s", oldPrimary.ID, newPrimaryID)

	return nil
}

// RegisterStateCallback registers a callback for service state changes
func (m *AvailabilityManager) RegisterStateCallback(serviceID string, callback func(*ServiceInstance, ServiceState)) {
	m.serviceStateCallbackMutex.Lock()
	defer m.serviceStateCallbackMutex.Unlock()

	if callbacks, exists := m.serviceStateCallbacks[serviceID]; exists {
		m.serviceStateCallbacks[serviceID] = append(callbacks, callback)
	} else {
		m.serviceStateCallbacks[serviceID] = []func(*ServiceInstance, ServiceState){callback}
	}
}

// GetFailoverEvents gets recent failover events
func (m *AvailabilityManager) GetFailoverEvents() []*FailoverEvent {
	m.failoverEventMutex.RLock()
	defer m.failoverEventMutex.RUnlock()

	// Return a copy of the events
	events := make([]*FailoverEvent, len(m.failoverEvents))
	copy(events, m.failoverEvents)

	return events
}

// GetServicesByRole gets services with a specific role
func (m *AvailabilityManager) GetServicesByRole(role ServiceRole) []*ServiceInstance {
	m.serviceMutex.RLock()
	defer m.serviceMutex.RUnlock()

	services := make([]*ServiceInstance, 0)
	for _, service := range m.services {
		if service.Role == role {
			services = append(services, service)
		}
	}

	return services
}

// GetServicesByState gets services in a specific state
func (m *AvailabilityManager) GetServicesByState(state ServiceState) []*ServiceInstance {
	m.serviceMutex.RLock()
	defer m.serviceMutex.RUnlock()

	services := make([]*ServiceInstance, 0)
	for _, service := range m.services {
		if service.State == state {
			services = append(services, service)
		}
	}

	return services
}

// GetAllGroups gets all availability groups
func (m *AvailabilityManager) GetAllGroups() []*AvailabilityGroup {
	m.groupMutex.RLock()
	defer m.groupMutex.RUnlock()

	groups := make([]*AvailabilityGroup, 0, len(m.groups))
	for _, group := range m.groups {
		groups = append(groups, group)
	}

	return groups
}

// GetAllServices gets all services
func (m *AvailabilityManager) GetAllServices() []*ServiceInstance {
	m.serviceMutex.RLock()
	defer m.serviceMutex.RUnlock()

	services := make([]*ServiceInstance, 0, len(m.services))
	for _, service := range m.services {
		services = append(services, service)
	}

	return services
}
