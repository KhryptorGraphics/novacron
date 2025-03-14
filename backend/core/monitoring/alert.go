package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// AlertSeverity represents the severity of an alert
type AlertSeverity string

const (
	// AlertSeverityCritical is a critical alert
	AlertSeverityCritical AlertSeverity = "critical"

	// AlertSeverityHigh is a high severity alert
	AlertSeverityHigh AlertSeverity = "high"

	// AlertSeverityMedium is a medium severity alert
	AlertSeverityMedium AlertSeverity = "medium"

	// AlertSeverityLow is a low severity alert
	AlertSeverityLow AlertSeverity = "low"

	// AlertSeverityInfo is an informational alert
	AlertSeverityInfo AlertSeverity = "info"
)

// AlertState represents the state of an alert
type AlertState string

const (
	// AlertStateFiring indicates the alert is active
	AlertStateFiring AlertState = "firing"

	// AlertStateResolved indicates the alert has been resolved
	AlertStateResolved AlertState = "resolved"

	// AlertStateSuppressed indicates the alert is suppressed
	AlertStateSuppressed AlertState = "suppressed"

	// AlertStatePending indicates the alert is in a pending state
	AlertStatePending AlertState = "pending"
)

// AlertConditionType represents the type of an alert condition
type AlertConditionType string

const (
	// ThresholdCondition is a condition based on a threshold
	ThresholdCondition AlertConditionType = "threshold"

	// RateOfChangeCondition is a condition based on rate of change
	RateOfChangeCondition AlertConditionType = "rate_of_change"

	// AbsenceCondition is a condition based on absence of data
	AbsenceCondition AlertConditionType = "absence"

	// OutlierCondition is a condition based on statistical outliers
	OutlierCondition AlertConditionType = "outlier"

	// CompositeCondition is a condition based on multiple other conditions
	CompositeCondition AlertConditionType = "composite"
)

// ComparisonOperator represents a comparison operator
type ComparisonOperator string

const (
	// GreaterThan is the greater than operator
	GreaterThan ComparisonOperator = ">"

	// GreaterThanOrEqual is the greater than or equal operator
	GreaterThanOrEqual ComparisonOperator = ">="

	// LessThan is the less than operator
	LessThan ComparisonOperator = "<"

	// LessThanOrEqual is the less than or equal operator
	LessThanOrEqual ComparisonOperator = "<="

	// Equal is the equal operator
	Equal ComparisonOperator = "=="

	// NotEqual is the not equal operator
	NotEqual ComparisonOperator = "!="
)

// LogicalOperator represents a logical operator
type LogicalOperator string

const (
	// And is the logical AND operator
	And LogicalOperator = "and"

	// Or is the logical OR operator
	Or LogicalOperator = "or"
)

// AlertCondition represents a condition for an alert
type AlertCondition struct {
	// Type is the type of condition
	Type AlertConditionType `json:"type"`

	// MetricID is the ID of the metric
	MetricID string `json:"metricId"`

	// Operator is the comparison operator
	Operator ComparisonOperator `json:"operator,omitempty"`

	// Threshold is the threshold value
	Threshold *float64 `json:"threshold,omitempty"`

	// Period is the period over which to evaluate
	Period *time.Duration `json:"period,omitempty"`

	// EvaluationWindow is the number of evaluations for triggering
	EvaluationWindow *int `json:"evaluationWindow,omitempty"`

	// Conditions are sub-conditions for composite conditions
	Conditions []AlertCondition `json:"conditions,omitempty"`

	// LogicalOperator is the logical operator for composite conditions
	LogicalOperator LogicalOperator `json:"logicalOperator,omitempty"`
}

// Alert represents an alert
type Alert struct {
	// ID is the unique identifier for the alert
	ID string `json:"id"`

	// Name is the human-readable name of the alert
	Name string `json:"name"`

	// Description is a description of the alert
	Description string `json:"description"`

	// Severity is the severity of the alert
	Severity AlertSeverity `json:"severity"`

	// State is the state of the alert
	State AlertState `json:"state"`

	// Condition is the condition for the alert
	Condition AlertCondition `json:"condition"`

	// Labels are the labels associated with the alert
	Labels map[string]string `json:"labels"`

	// TenantID is the ID of the tenant this alert belongs to
	TenantID string `json:"tenantId,omitempty"`

	// ResourceID is the ID of the resource this alert is related to
	ResourceID string `json:"resourceId,omitempty"`

	// ResourceType is the type of the resource this alert is related to
	ResourceType string `json:"resourceType,omitempty"`

	// Tags are additional tags for the alert
	Tags []string `json:"tags,omitempty"`

	// FirstTriggeredAt is when the alert was first triggered
	FirstTriggeredAt *time.Time `json:"firstTriggeredAt,omitempty"`

	// LastTriggeredAt is when the alert was last triggered
	LastTriggeredAt *time.Time `json:"lastTriggeredAt,omitempty"`

	// LastUpdatedAt is when the alert was last updated
	LastUpdatedAt time.Time `json:"lastUpdatedAt"`

	// ResolvedAt is when the alert was resolved
	ResolvedAt *time.Time `json:"resolvedAt,omitempty"`

	// SuppressedAt is when the alert was suppressed
	SuppressedAt *time.Time `json:"suppressedAt,omitempty"`

	// SuppressedUntil is when the suppression expires
	SuppressedUntil *time.Time `json:"suppressedUntil,omitempty"`

	// SuppressedBy is who suppressed the alert
	SuppressedBy string `json:"suppressedBy,omitempty"`

	// TriggerCount is the number of times the alert was triggered
	TriggerCount int `json:"triggerCount"`

	// LastNotifiedAt is when the alert was last notified
	LastNotifiedAt *time.Time `json:"lastNotifiedAt,omitempty"`

	// NotificationSent indicates if a notification has been sent
	NotificationSent bool `json:"notificationSent"`

	// AutoResolve indicates if the alert should auto-resolve
	AutoResolve bool `json:"autoResolve"`

	// AutoResolveAfter is the duration after which to auto-resolve
	AutoResolveAfter *time.Duration `json:"autoResolveAfter,omitempty"`

	// runbook is a link to the runbook for this alert
	Runbook string `json:"runbook,omitempty"`

	// CurrentValue is the current value of the metric
	CurrentValue *float64 `json:"currentValue,omitempty"`

	// NotificationChannels are the channels to notify
	NotificationChannels []string `json:"notificationChannels,omitempty"`
}

// AlertRegistry manages alerts
type AlertRegistry struct {
	alerts      map[string]*Alert
	alertsMutex sync.RWMutex
}

// NewAlertRegistry creates a new alert registry
func NewAlertRegistry() *AlertRegistry {
	return &AlertRegistry{
		alerts: make(map[string]*Alert),
	}
}

// RegisterAlert registers an alert
func (r *AlertRegistry) RegisterAlert(alert *Alert) error {
	r.alertsMutex.Lock()
	defer r.alertsMutex.Unlock()

	if _, exists := r.alerts[alert.ID]; exists {
		return fmt.Errorf("alert already exists: %s", alert.ID)
	}

	alert.LastUpdatedAt = time.Now()
	r.alerts[alert.ID] = alert
	return nil
}

// GetAlert gets an alert by ID
func (r *AlertRegistry) GetAlert(id string) (*Alert, error) {
	r.alertsMutex.RLock()
	defer r.alertsMutex.RUnlock()

	alert, exists := r.alerts[id]
	if !exists {
		return nil, fmt.Errorf("alert not found: %s", id)
	}

	return alert, nil
}

// ListAlerts lists all alerts
func (r *AlertRegistry) ListAlerts() []*Alert {
	r.alertsMutex.RLock()
	defer r.alertsMutex.RUnlock()

	alerts := make([]*Alert, 0, len(r.alerts))
	for _, alert := range r.alerts {
		alerts = append(alerts, alert)
	}

	return alerts
}

// RemoveAlert removes an alert
func (r *AlertRegistry) RemoveAlert(id string) error {
	r.alertsMutex.Lock()
	defer r.alertsMutex.Unlock()

	if _, exists := r.alerts[id]; !exists {
		return fmt.Errorf("alert not found: %s", id)
	}

	delete(r.alerts, id)
	return nil
}

// UpdateAlertState updates the state of an alert
func (r *AlertRegistry) UpdateAlertState(id string, state AlertState, value *float64) error {
	r.alertsMutex.Lock()
	defer r.alertsMutex.Unlock()

	alert, exists := r.alerts[id]
	if !exists {
		return fmt.Errorf("alert not found: %s", id)
	}

	now := time.Now()
	alert.LastUpdatedAt = now
	alert.CurrentValue = value

	// State transition logic
	switch state {
	case AlertStateFiring:
		if alert.State != AlertStateFiring {
			if alert.FirstTriggeredAt == nil {
				alert.FirstTriggeredAt = &now
			}
			alert.LastTriggeredAt = &now
			alert.TriggerCount++
			alert.NotificationSent = false
		}
	case AlertStateResolved:
		alert.ResolvedAt = &now
	case AlertStateSuppressed:
		alert.SuppressedAt = &now
	}

	alert.State = state

	return nil
}

// FilterAlerts filters alerts by criteria
func (r *AlertRegistry) FilterAlerts(filter func(*Alert) bool) []*Alert {
	r.alertsMutex.RLock()
	defer r.alertsMutex.RUnlock()

	filtered := make([]*Alert, 0)
	for _, alert := range r.alerts {
		if filter(alert) {
			filtered = append(filtered, alert)
		}
	}

	return filtered
}

// SupressAlert suppresses an alert
func (r *AlertRegistry) SuppressAlert(id string, duration time.Duration, suppressedBy string) error {
	r.alertsMutex.Lock()
	defer r.alertsMutex.Unlock()

	alert, exists := r.alerts[id]
	if !exists {
		return fmt.Errorf("alert not found: %s", id)
	}

	now := time.Now()
	suppressUntil := now.Add(duration)

	alert.SuppressedAt = &now
	alert.SuppressedUntil = &suppressUntil
	alert.SuppressedBy = suppressedBy
	alert.State = AlertStateSuppressed
	alert.LastUpdatedAt = now

	return nil
}

// UpdateNotificationStatus updates the notification status of an alert
func (r *AlertRegistry) UpdateNotificationStatus(id string, sent bool) error {
	r.alertsMutex.Lock()
	defer r.alertsMutex.Unlock()

	alert, exists := r.alerts[id]
	if !exists {
		return fmt.Errorf("alert not found: %s", id)
	}

	alert.NotificationSent = sent
	if sent {
		now := time.Now()
		alert.LastNotifiedAt = &now
	}

	return nil
}

// NewAlert creates a new alert
func NewAlert(id, name, description string, severity AlertSeverity, condition AlertCondition) *Alert {
	return &Alert{
		ID:            id,
		Name:          name,
		Description:   description,
		Severity:      severity,
		Condition:     condition,
		State:         AlertStatePending,
		Labels:        make(map[string]string),
		TriggerCount:  0,
		LastUpdatedAt: time.Now(),
		AutoResolve:   false,
	}
}

// AlertManager manages alerting
type AlertManager struct {
	registry           *AlertRegistry
	metricRegistry     *MetricRegistry
	evaluationInterval time.Duration
	notifiers          []AlertNotifier
	stopChan           chan struct{}
	wg                 sync.WaitGroup
}

// NewAlertManager creates a new alert manager
func NewAlertManager(registry *AlertRegistry, metricRegistry *MetricRegistry, interval time.Duration) *AlertManager {
	return &AlertManager{
		registry:           registry,
		metricRegistry:     metricRegistry,
		evaluationInterval: interval,
		notifiers:          make([]AlertNotifier, 0),
		stopChan:           make(chan struct{}),
	}
}

// Start starts the alert manager
func (m *AlertManager) Start() error {
	m.wg.Add(1)
	go m.run()
	return nil
}

// Stop stops the alert manager
func (m *AlertManager) Stop() error {
	close(m.stopChan)
	m.wg.Wait()
	return nil
}

// run is the main loop of the alert manager
func (m *AlertManager) run() {
	defer m.wg.Done()

	ticker := time.NewTicker(m.evaluationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.evaluateAlerts()
		}
	}
}

// evaluateAlerts evaluates all alerts
func (m *AlertManager) evaluateAlerts() {
	alerts := m.registry.ListAlerts()
	for _, alert := range alerts {
		go m.evaluateAlert(alert)
	}
}

// evaluateAlert evaluates an alert
func (m *AlertManager) evaluateAlert(alert *Alert) {
	// Skip suppressed alerts
	if alert.State == AlertStateSuppressed {
		if alert.SuppressedUntil != nil && time.Now().After(*alert.SuppressedUntil) {
			// Suppression expired
			m.registry.UpdateAlertState(alert.ID, AlertStatePending, nil)
		} else {
			return
		}
	}

	// Evaluate the condition
	triggered, value, err := m.evaluateCondition(alert.Condition)
	if err != nil {
		// Log error but don't change alert state
		fmt.Printf("Error evaluating alert %s: %v\n", alert.ID, err)
		return
	}

	if triggered {
		if alert.State != AlertStateFiring {
			m.registry.UpdateAlertState(alert.ID, AlertStateFiring, value)
			if !alert.NotificationSent {
				m.sendNotification(alert)
			}
		}
	} else {
		if alert.State == AlertStateFiring {
			m.registry.UpdateAlertState(alert.ID, AlertStateResolved, value)
		} else if alert.State == AlertStatePending {
			// Still pending, no change
		}
	}
}

// evaluateCondition evaluates an alert condition
func (m *AlertManager) evaluateCondition(condition AlertCondition) (bool, *float64, error) {
	switch condition.Type {
	case ThresholdCondition:
		return m.evaluateThresholdCondition(condition)
	case CompositeCondition:
		return m.evaluateCompositeCondition(condition)
	case AbsenceCondition:
		return m.evaluateAbsenceCondition(condition)
	case RateOfChangeCondition:
		return m.evaluateRateOfChangeCondition(condition)
	default:
		return false, nil, fmt.Errorf("unsupported condition type: %s", condition.Type)
	}
}

// evaluateThresholdCondition evaluates a threshold condition
func (m *AlertManager) evaluateThresholdCondition(condition AlertCondition) (bool, *float64, error) {
	if condition.Threshold == nil {
		return false, nil, fmt.Errorf("threshold not set")
	}

	metric, err := m.metricRegistry.GetMetric(condition.MetricID)
	if err != nil {
		return false, nil, err
	}

	lastValue := metric.GetLastValue()
	if lastValue == nil {
		return false, nil, fmt.Errorf("no values for metric")
	}

	var period time.Duration
	if condition.Period != nil {
		period = *condition.Period
	} else {
		period = 5 * time.Minute // default period
	}

	var values []MetricValue
	if period > 0 {
		end := time.Now()
		start := end.Add(-period)
		values = metric.GetValues(start, end)
	} else {
		// Just use the most recent value
		values = []MetricValue{*lastValue}
	}

	if len(values) == 0 {
		return false, nil, fmt.Errorf("no values in period")
	}

	// Use the most recent value for the current value
	currentValue := values[len(values)-1].Value
	currentValuePtr := &currentValue

	// For threshold conditions with evaluation windows, we need to check if the condition
	// is met for a number of consecutive evaluations
	evaluationWindow := 1
	if condition.EvaluationWindow != nil {
		evaluationWindow = *condition.EvaluationWindow
	}

	// If we don't have enough values, can't evaluate
	if len(values) < evaluationWindow {
		return false, currentValuePtr, nil
	}

	// Check if the condition is met for the evaluation window
	for i := len(values) - evaluationWindow; i < len(values); i++ {
		value := values[i].Value
		triggered := compareValues(value, *condition.Threshold, condition.Operator)
		if !triggered {
			return false, currentValuePtr, nil
		}
	}

	return true, currentValuePtr, nil
}

// evaluateCompositeCondition evaluates a composite condition
func (m *AlertManager) evaluateCompositeCondition(condition AlertCondition) (bool, *float64, error) {
	if len(condition.Conditions) == 0 {
		return false, nil, fmt.Errorf("no sub-conditions")
	}

	// Evaluate each sub-condition
	results := make([]bool, len(condition.Conditions))
	for i, subCondition := range condition.Conditions {
		triggered, _, err := m.evaluateCondition(subCondition)
		if err != nil {
			return false, nil, err
		}
		results[i] = triggered
	}

	// Combine results based on logical operator
	var result bool
	switch condition.LogicalOperator {
	case And:
		result = true
		for _, r := range results {
			if !r {
				result = false
				break
			}
		}
	case Or:
		result = false
		for _, r := range results {
			if r {
				result = true
				break
			}
		}
	default:
		return false, nil, fmt.Errorf("unsupported logical operator: %s", condition.LogicalOperator)
	}

	return result, nil, nil
}

// evaluateAbsenceCondition evaluates an absence condition
func (m *AlertManager) evaluateAbsenceCondition(condition AlertCondition) (bool, *float64, error) {
	metric, err := m.metricRegistry.GetMetric(condition.MetricID)
	if err != nil {
		return false, nil, err
	}

	lastValue := metric.GetLastValue()
	if lastValue == nil {
		return true, nil, nil // No values = absence condition met
	}

	var period time.Duration
	if condition.Period != nil {
		period = *condition.Period
	} else {
		period = 5 * time.Minute // default period
	}

	// Check if the last value is older than the period
	if time.Since(lastValue.Timestamp) > period {
		return true, nil, nil
	}

	return false, nil, nil
}

// evaluateRateOfChangeCondition evaluates a rate of change condition
func (m *AlertManager) evaluateRateOfChangeCondition(condition AlertCondition) (bool, *float64, error) {
	if condition.Threshold == nil {
		return false, nil, fmt.Errorf("threshold not set")
	}

	metric, err := m.metricRegistry.GetMetric(condition.MetricID)
	if err != nil {
		return false, nil, err
	}

	var period time.Duration
	if condition.Period != nil {
		period = *condition.Period
	} else {
		period = 5 * time.Minute // default period
	}

	end := time.Now()
	start := end.Add(-period)
	values := metric.GetValues(start, end)

	if len(values) < 2 {
		return false, nil, fmt.Errorf("not enough values to calculate rate")
	}

	// Calculate rate of change
	first := values[0]
	last := values[len(values)-1]
	timeDiff := last.Timestamp.Sub(first.Timestamp).Seconds()
	if timeDiff == 0 {
		return false, nil, fmt.Errorf("values have same timestamp")
	}

	rateOfChange := (last.Value - first.Value) / timeDiff
	ratePtr := &rateOfChange

	// Compare to threshold
	return compareValues(rateOfChange, *condition.Threshold, condition.Operator), ratePtr, nil
}

// compareValues compares two values using an operator
func compareValues(value, threshold float64, operator ComparisonOperator) bool {
	switch operator {
	case GreaterThan:
		return value > threshold
	case GreaterThanOrEqual:
		return value >= threshold
	case LessThan:
		return value < threshold
	case LessThanOrEqual:
		return value <= threshold
	case Equal:
		return value == threshold
	case NotEqual:
		return value != threshold
	default:
		return false
	}
}

// AddNotifier adds a notifier
func (m *AlertManager) AddNotifier(notifier AlertNotifier) {
	m.notifiers = append(m.notifiers, notifier)
}

// sendNotification sends a notification for an alert
func (m *AlertManager) sendNotification(alert *Alert) {
	for _, notifier := range m.notifiers {
		err := notifier.Notify(alert)
		if err != nil {
			fmt.Printf("Error sending notification: %v\n", err)
		}
	}

	// Update notification status
	m.registry.UpdateNotificationStatus(alert.ID, true)
}

// AlertNotifier notifies about alerts
type AlertNotifier interface {
	// Notify notifies about an alert
	Notify(alert *Alert) error
}
