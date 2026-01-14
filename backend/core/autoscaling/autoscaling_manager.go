package autoscaling

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ScalingMode represents the mode of scaling operation
type ScalingMode string

const (
	// ScalingModeHorizontal represents horizontal scaling (add/remove instances)
	ScalingModeHorizontal ScalingMode = "horizontal"
	// ScalingModeVertical represents vertical scaling (resize instances)
	ScalingModeVertical ScalingMode = "vertical"
	// ScalingModeMixed represents a combination of horizontal and vertical scaling
	ScalingModeMixed ScalingMode = "mixed"
)

// ScalingAction represents the type of scaling action to perform
type ScalingAction string

const (
	// ScalingActionScaleOut increases capacity (add instances)
	ScalingActionScaleOut ScalingAction = "scale_out"
	// ScalingActionScaleIn decreases capacity (remove instances)
	ScalingActionScaleIn ScalingAction = "scale_in"
	// ScalingActionScaleUp increases resources per instance
	ScalingActionScaleUp ScalingAction = "scale_up"
	// ScalingActionScaleDown decreases resources per instance
	ScalingActionScaleDown ScalingAction = "scale_down"
	// ScalingActionNone indicates no scaling is needed
	ScalingActionNone ScalingAction = "none"
)

// MetricType represents the type of metric used for scaling decisions
type MetricType string

const (
	// MetricCPUUtilization represents CPU utilization percentage
	MetricCPUUtilization MetricType = "cpu_utilization"
	// MetricMemoryUtilization represents memory utilization percentage
	MetricMemoryUtilization MetricType = "memory_utilization"
	// MetricRequestCount represents number of requests per time unit
	MetricRequestCount MetricType = "request_count"
	// MetricQueueLength represents queue length
	MetricQueueLength MetricType = "queue_length"
	// MetricResponseTime represents average response time
	MetricResponseTime MetricType = "response_time"
	// MetricCustom represents a custom metric
	MetricCustom MetricType = "custom"
)

// ResourceType represents the type of resource being scaled
type ResourceType string

const (
	// ResourceVM represents a virtual machine
	ResourceVM ResourceType = "vm"
	// ResourceContainer represents a container
	ResourceContainer ResourceType = "container"
	// ResourcePod represents a Kubernetes pod
	ResourcePod ResourceType = "pod"
	// ResourceService represents a service (multiple instances)
	ResourceService ResourceType = "service"
	// ResourceStorageNode represents a storage node
	ResourceStorageNode ResourceType = "storage_node"
	// ResourceComputeNode represents a compute node
	ResourceComputeNode ResourceType = "compute_node"
)

// ScalingEvent represents a scaling event that occurred
type ScalingEvent struct {
	// ID is a unique identifier for this event
	ID string
	// Time when the event occurred
	Timestamp time.Time
	// ScalingGroupID is the ID of the scaling group
	ScalingGroupID string
	// Action taken
	Action ScalingAction
	// Previous capacity
	PreviousCapacity int
	// New capacity
	NewCapacity int
	// Reason for scaling
	Reason string
	// Status of the scaling event
	Status string
	// Details about the scaling event
	Details map[string]interface{}
}

// ScalingRule defines when and how to scale a resource
type ScalingRule struct {
	// ID is a unique identifier for this rule
	ID string
	// Name of the rule
	Name string
	// MetricType is the type of metric to monitor
	MetricType MetricType
	// MetricSource is where to get the metric from
	MetricSource string
	// ScaleOutThreshold is the threshold to trigger scale out
	ScaleOutThreshold float64
	// ScaleInThreshold is the threshold to trigger scale in
	ScaleInThreshold float64
	// ScaleOutIncrement is how many instances to add when scaling out
	ScaleOutIncrement int
	// ScaleInDecrement is how many instances to remove when scaling in
	ScaleInDecrement int
	// CooldownPeriod is the time to wait after scaling before another scale action
	CooldownPeriod time.Duration
	// EvaluationPeriods is number of periods to evaluate before scaling
	EvaluationPeriods int
	// Enabled indicates if this rule is active
	Enabled bool
	// CustomEvaluationFunc is a custom function for evaluation (for MetricCustom)
	CustomEvaluationFunc func(context.Context, map[string]interface{}) (ScalingAction, string, error)
}

// ScalingGroup represents a group of resources that scale together
type ScalingGroup struct {
	// ID is a unique identifier for this scaling group
	ID string
	// Name of the scaling group
	Name string
	// ResourceType is the type of resource being scaled
	ResourceType ResourceType
	// ResourceIDs are the IDs of resources in this group
	ResourceIDs []string
	// ScalingMode defines how this group scales
	ScalingMode ScalingMode
	// MinCapacity is the minimum number of resources
	MinCapacity int
	// MaxCapacity is the maximum number of resources
	MaxCapacity int
	// DesiredCapacity is the current desired number of resources
	DesiredCapacity int
	// CurrentCapacity is the current actual number of resources
	CurrentCapacity int
	// VerticalScalingLimits defines limits for vertical scaling
	VerticalScalingLimits map[string]map[string]interface{}
	// Rules for scaling this group
	Rules []*ScalingRule
	// Tags for organization and filtering
	Tags map[string]string
	// LastScalingAction is when the last scaling action occurred
	LastScalingAction time.Time
	// Status of the scaling group
	Status string
	// LaunchTemplate is a template for creating new resources
	LaunchTemplate map[string]interface{}
}

// PredictiveScalingConfig configures predictive scaling
type PredictiveScalingConfig struct {
	// Enabled indicates if predictive scaling is enabled
	Enabled bool
	// ForecastHorizon is how far ahead to forecast
	ForecastHorizon time.Duration
	// HistoryWindow is how much history to use for forecasting
	HistoryWindow time.Duration
	// MinConfidence is the minimum confidence level for predictions
	MinConfidence float64
	// Schedule contains any fixed scaling schedules
	Schedule map[string]interface{}
	// Algorithm to use for prediction
	Algorithm string
	// AdditionalParams for the algorithm
	AdditionalParams map[string]interface{}
}

// MetricsProvider defines the interface for metrics providers
type MetricsProvider interface {
	// GetMetric retrieves a metric for a resource or resource group
	GetMetric(ctx context.Context, metricType MetricType, resourceID string, timeRange time.Duration) (float64, error)
	// GetMetricHistory retrieves historical metrics
	GetMetricHistory(ctx context.Context, metricType MetricType, resourceID string, start, end time.Time) (map[time.Time]float64, error)
	// RegisterCallback registers a callback for metric changes
	RegisterCallback(ctx context.Context, metricType MetricType, resourceID string, threshold float64, callback func(float64))
}

// ResourceController defines the interface for controlling resources
type ResourceController interface {
	// ScaleResourceGroup scales a resource group
	ScaleResourceGroup(ctx context.Context, groupID string, desiredCapacity int) error
	// GetResourceGroupCapacity gets the current capacity of a resource group
	GetResourceGroupCapacity(ctx context.Context, groupID string) (int, error)
	// GetResourceUtilization gets the current utilization of a resource
	GetResourceUtilization(ctx context.Context, resourceID string, metricType MetricType) (float64, error)
	// CreateResource creates a new resource from a template
	CreateResource(ctx context.Context, template map[string]interface{}) (string, error)
	// DeleteResource deletes a resource
	DeleteResource(ctx context.Context, resourceID string) error
	// ResizeResource changes the size of a resource (vertical scaling)
	ResizeResource(ctx context.Context, resourceID string, newSize map[string]interface{}) error
}

// AutoScalingManager manages auto-scaling operations
type AutoScalingManager struct {
	// Mutex for protecting concurrent access
	mu sync.RWMutex
	// Map of scaling group ID to scaling group
	scalingGroups map[string]*ScalingGroup
	// Map of rule ID to rule
	scalingRules map[string]*ScalingRule
	// Metrics provider
	metricsProvider MetricsProvider
	// Resource controller
	resourceController ResourceController
	// Predictive scaling configuration
	predictiveConfig PredictiveScalingConfig
	// Is the manager initialized
	initialized bool
	// Event history
	eventHistory []*ScalingEvent
	// Maximum number of events to keep in history
	maxEventHistory int
	// Context for the scaling loop
	ctx    context.Context
	cancel context.CancelFunc
}

// NewAutoScalingManager creates a new auto-scaling manager
func NewAutoScalingManager(metricsProvider MetricsProvider, resourceController ResourceController) *AutoScalingManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &AutoScalingManager{
		scalingGroups:      make(map[string]*ScalingGroup),
		scalingRules:       make(map[string]*ScalingRule),
		metricsProvider:    metricsProvider,
		resourceController: resourceController,
		predictiveConfig: PredictiveScalingConfig{
			Enabled: false,
		},
		initialized:     false,
		eventHistory:    make([]*ScalingEvent, 0),
		maxEventHistory: 1000, // Default to storing the last 1000 events
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Initialize initializes the auto-scaling manager
func (m *AutoScalingManager) Initialize(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		return fmt.Errorf("auto-scaling manager already initialized")
	}

	if m.metricsProvider == nil {
		return fmt.Errorf("metrics provider is required")
	}

	if m.resourceController == nil {
		return fmt.Errorf("resource controller is required")
	}

	m.initialized = true
	return nil
}

// RegisterScalingGroup registers a scaling group
func (m *AutoScalingManager) RegisterScalingGroup(group *ScalingGroup) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	if group.ID == "" {
		return fmt.Errorf("scaling group ID cannot be empty")
	}

	if _, exists := m.scalingGroups[group.ID]; exists {
		return fmt.Errorf("scaling group with ID %s already exists", group.ID)
	}

	// Validate min/max capacities
	if group.MinCapacity < 0 {
		return fmt.Errorf("minimum capacity cannot be negative")
	}

	if group.MaxCapacity < group.MinCapacity {
		return fmt.Errorf("maximum capacity cannot be less than minimum capacity")
	}

	if group.DesiredCapacity < group.MinCapacity || group.DesiredCapacity > group.MaxCapacity {
		return fmt.Errorf("desired capacity must be between minimum and maximum capacity")
	}

	// Add rules to rule map
	for _, rule := range group.Rules {
		if rule.ID == "" {
			return fmt.Errorf("scaling rule ID cannot be empty")
		}

		if _, exists := m.scalingRules[rule.ID]; exists {
			return fmt.Errorf("scaling rule with ID %s already exists", rule.ID)
		}

		m.scalingRules[rule.ID] = rule
	}

	// Store the scaling group
	m.scalingGroups[group.ID] = group

	return nil
}

// UpdateScalingGroup updates a scaling group
func (m *AutoScalingManager) UpdateScalingGroup(group *ScalingGroup) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	existingGroup, exists := m.scalingGroups[group.ID]
	if !exists {
		return fmt.Errorf("scaling group %s not found", group.ID)
	}

	// Validate min/max capacities
	if group.MinCapacity < 0 {
		return fmt.Errorf("minimum capacity cannot be negative")
	}

	if group.MaxCapacity < group.MinCapacity {
		return fmt.Errorf("maximum capacity cannot be less than minimum capacity")
	}

	// If desired capacity is outside the new min/max range, adjust it
	newDesired := group.DesiredCapacity
	if newDesired < group.MinCapacity {
		newDesired = group.MinCapacity
	} else if newDesired > group.MaxCapacity {
		newDesired = group.MaxCapacity
	}

	// Remove existing rules
	for _, rule := range existingGroup.Rules {
		delete(m.scalingRules, rule.ID)
	}

	// Add new rules
	for i, rule := range group.Rules {
		if rule.ID == "" {
			return fmt.Errorf("scaling rule ID cannot be empty")
		}

		if _, exists := m.scalingRules[rule.ID]; exists && rule.ID != fmt.Sprintf("%s-rule-%d", group.ID, i) {
			return fmt.Errorf("scaling rule with ID %s already exists", rule.ID)
		}

		m.scalingRules[rule.ID] = rule
	}

	// Update the scaling group
	group.DesiredCapacity = newDesired
	m.scalingGroups[group.ID] = group

	return nil
}

// UnregisterScalingGroup unregisters a scaling group
func (m *AutoScalingManager) UnregisterScalingGroup(groupID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	group, exists := m.scalingGroups[groupID]
	if !exists {
		return fmt.Errorf("scaling group %s not found", groupID)
	}

	// Remove rules
	for _, rule := range group.Rules {
		delete(m.scalingRules, rule.ID)
	}

	// Remove group
	delete(m.scalingGroups, groupID)

	return nil
}

// GetScalingGroup returns a scaling group by ID
func (m *AutoScalingManager) GetScalingGroup(groupID string) (*ScalingGroup, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("auto-scaling manager not initialized")
	}

	group, exists := m.scalingGroups[groupID]
	if !exists {
		return nil, fmt.Errorf("scaling group %s not found", groupID)
	}

	return group, nil
}

// ListScalingGroups returns a list of all scaling groups
func (m *AutoScalingManager) ListScalingGroups() []*ScalingGroup {
	m.mu.RLock()
	defer m.mu.RUnlock()

	groups := make([]*ScalingGroup, 0, len(m.scalingGroups))
	for _, group := range m.scalingGroups {
		groups = append(groups, group)
	}

	return groups
}

// SetGroupCapacity sets the desired capacity for a scaling group
func (m *AutoScalingManager) SetGroupCapacity(ctx context.Context, groupID string, capacity int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	group, exists := m.scalingGroups[groupID]
	if !exists {
		return fmt.Errorf("scaling group %s not found", groupID)
	}

	// Ensure capacity is within allowed range
	if capacity < group.MinCapacity {
		capacity = group.MinCapacity
	}

	if capacity > group.MaxCapacity {
		capacity = group.MaxCapacity
	}

	// Don't do anything if capacity is already at desired level
	if capacity == group.CurrentCapacity {
		return nil
	}

	// Scale the group
	if err := m.resourceController.ScaleResourceGroup(ctx, groupID, capacity); err != nil {
		return fmt.Errorf("failed to scale resource group: %v", err)
	}

	// Update group state
	group.DesiredCapacity = capacity
	group.LastScalingAction = time.Now()

	// Record the event
	event := &ScalingEvent{
		ID:               fmt.Sprintf("evt-%s-%d", groupID, time.Now().Unix()),
		Timestamp:        time.Now(),
		ScalingGroupID:   groupID,
		PreviousCapacity: group.CurrentCapacity,
		NewCapacity:      capacity,
		Status:           "completed",
	}

	if capacity > group.CurrentCapacity {
		event.Action = ScalingActionScaleOut
		event.Reason = "Manual scaling request to increase capacity"
	} else {
		event.Action = ScalingActionScaleIn
		event.Reason = "Manual scaling request to decrease capacity"
	}

	m.addEvent(event)

	// Update current capacity
	group.CurrentCapacity = capacity

	return nil
}

// EvaluateScalingRules evaluates all scaling rules and applies actions
func (m *AutoScalingManager) EvaluateScalingRules(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	for groupID, group := range m.scalingGroups {
		// Skip if the group was scaled recently (cooldown period)
		if !m.canScale(group) {
			continue
		}

		// Evaluate each rule
		for _, rule := range group.Rules {
			if !rule.Enabled {
				continue
			}

			action, reason, err := m.evaluateRule(ctx, group, rule)
			if err != nil {
				return fmt.Errorf("failed to evaluate rule %s: %v", rule.ID, err)
			}

			// Apply the action if needed
			if action != ScalingActionNone {
				if err := m.applyScalingAction(ctx, group, action, reason); err != nil {
					return fmt.Errorf("failed to apply scaling action for group %s: %v", groupID, err)
				}
				// Once a rule triggers scaling, skip other rules for this group
				break
			}
		}
	}

	return nil
}

// StartScalingLoop starts the background scaling loop
func (m *AutoScalingManager) StartScalingLoop(interval time.Duration) error {
	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if err := m.EvaluateScalingRules(m.ctx); err != nil {
					// Log the error, but continue
					fmt.Printf("Error evaluating scaling rules: %v\n", err)
				}
			case <-m.ctx.Done():
				return
			}
		}
	}()

	return nil
}

// StopScalingLoop stops the background scaling loop
func (m *AutoScalingManager) StopScalingLoop() {
	if m.cancel != nil {
		m.cancel()
	}
}

// EnablePredictiveScaling enables predictive scaling with the given configuration
func (m *AutoScalingManager) EnablePredictiveScaling(config PredictiveScalingConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("auto-scaling manager not initialized")
	}

	m.predictiveConfig = config
	m.predictiveConfig.Enabled = true

	return nil
}

// DisablePredictiveScaling disables predictive scaling
func (m *AutoScalingManager) DisablePredictiveScaling() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.predictiveConfig.Enabled = false
}

// GetScalingEvents returns recent scaling events
func (m *AutoScalingManager) GetScalingEvents(limit int) []*ScalingEvent {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if limit <= 0 || limit > len(m.eventHistory) {
		limit = len(m.eventHistory)
	}

	// Return the most recent events
	start := len(m.eventHistory) - limit
	if start < 0 {
		start = 0
	}

	return m.eventHistory[start:]
}

// Shutdown shuts down the auto-scaling manager
func (m *AutoScalingManager) Shutdown() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return nil
	}

	// Stop background loop
	if m.cancel != nil {
		m.cancel()
	}

	m.initialized = false
	return nil
}

// addEvent adds a scaling event to the history
func (m *AutoScalingManager) addEvent(event *ScalingEvent) {
	m.eventHistory = append(m.eventHistory, event)

	// Truncate history if needed
	if len(m.eventHistory) > m.maxEventHistory {
		m.eventHistory = m.eventHistory[len(m.eventHistory)-m.maxEventHistory:]
	}
}

// canScale checks if a group can be scaled (not in cooldown period)
func (m *AutoScalingManager) canScale(group *ScalingGroup) bool {
	// If no previous scaling action, can scale
	if group.LastScalingAction.IsZero() {
		return true
	}

	// If no rules, can't determine cooldown period
	if len(group.Rules) == 0 {
		return true
	}

	// Find the shortest cooldown period among all rules
	shortestCooldown := group.Rules[0].CooldownPeriod
	for _, rule := range group.Rules {
		if rule.CooldownPeriod < shortestCooldown {
			shortestCooldown = rule.CooldownPeriod
		}
	}

	// Check if we're past the cooldown period
	return time.Since(group.LastScalingAction) > shortestCooldown
}

// evaluateRule evaluates a scaling rule and returns the action to take
func (m *AutoScalingManager) evaluateRule(ctx context.Context, group *ScalingGroup, rule *ScalingRule) (ScalingAction, string, error) {
	// For custom metrics, use the custom evaluation function
	if rule.MetricType == MetricCustom && rule.CustomEvaluationFunc != nil {
		return rule.CustomEvaluationFunc(ctx, map[string]interface{}{
			"scaling_group": group,
			"rule":          rule,
		})
	}

	// For regular metrics, get the current value
	var metricValue float64
	var err error

	// Check if this is a resource-specific metric or a group-wide metric
	if rule.MetricSource == "" || rule.MetricSource == group.ID {
		// Group-wide metric
		// Typically would aggregate metrics across resources, but simplified here
		values := make([]float64, 0, len(group.ResourceIDs))
		for _, resourceID := range group.ResourceIDs {
			value, err := m.resourceController.GetResourceUtilization(ctx, resourceID, rule.MetricType)
			if err != nil {
				return ScalingActionNone, "", fmt.Errorf("failed to get utilization for resource %s: %v", resourceID, err)
			}
			values = append(values, value)
		}

		// Use the average as the metric value
		if len(values) > 0 {
			sum := 0.0
			for _, value := range values {
				sum += value
			}
			metricValue = sum / float64(len(values))
		}
	} else {
		// Specific resource metric
		metricValue, err = m.metricsProvider.GetMetric(ctx, rule.MetricType, rule.MetricSource, 5*time.Minute)
		if err != nil {
			return ScalingActionNone, "", fmt.Errorf("failed to get metric: %v", err)
		}
	}

	// Determine action based on thresholds
	if metricValue >= rule.ScaleOutThreshold {
		return ScalingActionScaleOut, fmt.Sprintf("Metric %s value %.2f exceeds scale-out threshold %.2f", rule.MetricType, metricValue, rule.ScaleOutThreshold), nil
	} else if metricValue <= rule.ScaleInThreshold {
		return ScalingActionScaleIn, fmt.Sprintf("Metric %s value %.2f below scale-in threshold %.2f", rule.MetricType, metricValue, rule.ScaleInThreshold), nil
	}

	return ScalingActionNone, "", nil
}

// applyScalingAction applies a scaling action to a group
func (m *AutoScalingManager) applyScalingAction(ctx context.Context, group *ScalingGroup, action ScalingAction, reason string) error {
	var newCapacity int

	switch action {
	case ScalingActionScaleOut:
		// Find the rule that would have the highest increment
		increment := 1
		for _, rule := range group.Rules {
			if rule.ScaleOutIncrement > increment {
				increment = rule.ScaleOutIncrement
			}
		}
		newCapacity = group.CurrentCapacity + increment
		if newCapacity > group.MaxCapacity {
			newCapacity = group.MaxCapacity
		}
	case ScalingActionScaleIn:
		// Find the rule that would have the lowest decrement
		decrement := 1
		for _, rule := range group.Rules {
			if rule.ScaleInDecrement > 0 && rule.ScaleInDecrement < decrement {
				decrement = rule.ScaleInDecrement
			}
		}
		newCapacity = group.CurrentCapacity - decrement
		if newCapacity < group.MinCapacity {
			newCapacity = group.MinCapacity
		}
	default:
		return fmt.Errorf("unsupported scaling action: %s", action)
	}

	// Don't do anything if already at target capacity
	if newCapacity == group.CurrentCapacity {
		return nil
	}

	// Create event before scaling (in case scaling fails)
	event := &ScalingEvent{
		ID:               fmt.Sprintf("evt-%s-%d", group.ID, time.Now().Unix()),
		Timestamp:        time.Now(),
		ScalingGroupID:   group.ID,
		Action:           action,
		PreviousCapacity: group.CurrentCapacity,
		NewCapacity:      newCapacity,
		Reason:           reason,
		Status:           "pending",
	}
	m.addEvent(event)

	// Scale the group
	if err := m.resourceController.ScaleResourceGroup(ctx, group.ID, newCapacity); err != nil {
		event.Status = "failed"
		event.Details = map[string]interface{}{
			"error": err.Error(),
		}
		return fmt.Errorf("failed to scale resource group: %v", err)
	}

	// Update group state
	group.DesiredCapacity = newCapacity
	group.CurrentCapacity = newCapacity
	group.LastScalingAction = time.Now()

	// Update event status
	event.Status = "completed"

	return nil
}
