package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AlertSeverity represents the severity of an alert
type AlertSeverity string

const (
	// AlertSeverityInfo represents an informational alert
	AlertSeverityInfo AlertSeverity = "info"

	// AlertSeverityWarning represents a warning alert
	AlertSeverityWarning AlertSeverity = "warning"

	// AlertSeverityError represents an error alert
	AlertSeverityError AlertSeverity = "error"

	// AlertSeverityCritical represents a critical alert
	AlertSeverityCritical AlertSeverity = "critical"
)

// AlertStatus represents the status of an alert
type AlertStatus string

const (
	// AlertStatusFiring indicates the alert is currently firing
	AlertStatusFiring AlertStatus = "firing"

	// AlertStatusResolved indicates the alert has been resolved
	AlertStatusResolved AlertStatus = "resolved"

	// AlertStatusAcknowledged indicates the alert has been acknowledged
	AlertStatusAcknowledged AlertStatus = "acknowledged"

	// AlertStatusSuppressed indicates the alert is suppressed
	AlertStatusSuppressed AlertStatus = "suppressed"
)

// AlertType represents the type of alert
type AlertType string

const (
	// AlertTypeThreshold represents a threshold-based alert
	AlertTypeThreshold AlertType = "threshold"

	// AlertTypeRateOfChange represents a rate-of-change alert
	AlertTypeRateOfChange AlertType = "rate_of_change"

	// AlertTypeAnomaly represents an anomaly detection alert
	AlertTypeAnomaly AlertType = "anomaly"

	// AlertTypeEvent represents an event-based alert
	AlertTypeEvent AlertType = "event"
)

// AlertConditionOperator represents comparison operators for alert conditions
type AlertConditionOperator string

const (
	// AlertConditionOperatorEqual represents the equal operator
	AlertConditionOperatorEqual AlertConditionOperator = "eq"

	// AlertConditionOperatorNotEqual represents the not equal operator
	AlertConditionOperatorNotEqual AlertConditionOperator = "ne"

	// AlertConditionOperatorGreaterThan represents the greater than operator
	AlertConditionOperatorGreaterThan AlertConditionOperator = "gt"

	// AlertConditionOperatorGreaterThanOrEqual represents the greater than or equal operator
	AlertConditionOperatorGreaterThanOrEqual AlertConditionOperator = "gte"

	// AlertConditionOperatorLessThan represents the less than operator
	AlertConditionOperatorLessThan AlertConditionOperator = "lt"

	// AlertConditionOperatorLessThanOrEqual represents the less than or equal operator
	AlertConditionOperatorLessThanOrEqual AlertConditionOperator = "lte"
)

// AlertCondition represents a condition for triggering an alert
type AlertCondition struct {
	// MetricName is the name of the metric to check
	MetricName string `json:"metric_name"`

	// Operator is the comparison operator
	Operator AlertConditionOperator `json:"operator"`

	// Threshold is the threshold value
	Threshold float64 `json:"threshold"`

	// Duration is the duration the condition must be true for
	Duration time.Duration `json:"duration"`

	// Tags to filter metrics by
	Tags map[string]string `json:"tags,omitempty"`
}

// Alert represents an alert definition
type Alert struct {
	// ID is a unique identifier for the alert
	ID string `json:"id"`

	// Name is a human-readable name for the alert
	Name string `json:"name"`

	// Description describes the alert
	Description string `json:"description"`

	// Severity indicates the severity of the alert
	Severity AlertSeverity `json:"severity"`

	// Type indicates the type of alert
	Type AlertType `json:"type"`

	// Condition defines when the alert should trigger
	Condition AlertCondition `json:"condition"`

	// Labels are additional metadata for the alert
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations are additional information for the alert
	Annotations map[string]string `json:"annotations,omitempty"`

	// NotificationChannels are the channels to notify when the alert fires
	NotificationChannels []string `json:"notification_channels,omitempty"`

	// Enabled indicates whether the alert is enabled
	Enabled bool `json:"enabled"`

	// Status is the current status of the alert
	Status AlertStatus `json:"status"`
}

// AlertInstance represents an instance of a triggered alert
type AlertInstance struct {
	// Alert is the alert definition
	Alert *Alert `json:"alert"`

	// Value is the value that triggered the alert
	Value float64 `json:"value"`

	// StartTime is when the alert started firing
	StartTime time.Time `json:"start_time"`

	// EndTime is when the alert stopped firing (if resolved)
	EndTime time.Time `json:"end_time,omitempty"`

	// Status is the current status of the alert instance
	Status AlertStatus `json:"status"`

	// AcknowledgedBy is who acknowledged the alert
	AcknowledgedBy string `json:"acknowledged_by,omitempty"`

	// AcknowledgedTime is when the alert was acknowledged
	AcknowledgedTime time.Time `json:"acknowledged_time,omitempty"`

	// AcknowledgementComment is a comment about the acknowledgement
	AcknowledgementComment string `json:"acknowledgement_comment,omitempty"`
}

// AlertManager manages alerts
type AlertManager struct {
	// Map of alert IDs to alert definitions
	alerts      map[string]*Alert
	alertsMutex sync.RWMutex

	// Map of alert instance IDs to alert instances
	instances      map[string]*AlertInstance
	instancesMutex sync.RWMutex

	// Notification manager for sending alerts
	notificationManager *NotificationManager

	// Parent metric collector for accessing metrics
	metricCollector *DistributedMetricCollector

	// Evaluation interval
	evaluationInterval time.Duration

	// Control channels
	stopChan chan struct{}
	running  bool
	mutex    sync.Mutex
}

// NewAlertManager creates a new alert manager
func NewAlertManager(metricCollector *DistributedMetricCollector) *AlertManager {
	return &AlertManager{
		alerts:              make(map[string]*Alert),
		instances:           make(map[string]*AlertInstance),
		notificationManager: NewNotificationManager(),
		metricCollector:     metricCollector,
		evaluationInterval:  30 * time.Second,
		stopChan:            make(chan struct{}),
		running:             false,
	}
}

// Start starts the alert manager
func (m *AlertManager) Start() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.running {
		return fmt.Errorf("alert manager already running")
	}

	m.running = true
	m.stopChan = make(chan struct{})

	// Start the evaluation loop
	go m.evaluationLoop()

	return nil
}

// Stop stops the alert manager
func (m *AlertManager) Stop() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.running {
		return nil
	}

	m.running = false
	close(m.stopChan)

	return nil
}

// RegisterAlert registers an alert
func (m *AlertManager) RegisterAlert(alert *Alert) error {
	if alert.ID == "" {
		return fmt.Errorf("alert ID cannot be empty")
	}

	m.alertsMutex.Lock()
	defer m.alertsMutex.Unlock()

	if _, exists := m.alerts[alert.ID]; exists {
		return fmt.Errorf("alert with ID %s already exists", alert.ID)
	}

	m.alerts[alert.ID] = alert
	return nil
}

// DeregisterAlert deregisters an alert
func (m *AlertManager) DeregisterAlert(alertID string) bool {
	m.alertsMutex.Lock()
	defer m.alertsMutex.Unlock()

	if _, exists := m.alerts[alertID]; !exists {
		return false
	}

	delete(m.alerts, alertID)
	return true
}

// GetAlert gets an alert by ID
func (m *AlertManager) GetAlert(alertID string) (*Alert, error) {
	m.alertsMutex.RLock()
	defer m.alertsMutex.RUnlock()

	alert, exists := m.alerts[alertID]
	if !exists {
		return nil, fmt.Errorf("alert with ID %s not found", alertID)
	}

	return alert, nil
}

// ListAlerts lists all alerts
func (m *AlertManager) ListAlerts() []*Alert {
	m.alertsMutex.RLock()
	defer m.alertsMutex.RUnlock()

	alerts := make([]*Alert, 0, len(m.alerts))
	for _, alert := range m.alerts {
		alerts = append(alerts, alert)
	}

	return alerts
}

// ListAlertInstances lists all alert instances
func (m *AlertManager) ListAlertInstances() []*AlertInstance {
	m.instancesMutex.RLock()
	defer m.instancesMutex.RUnlock()

	instances := make([]*AlertInstance, 0, len(m.instances))
	for _, instance := range m.instances {
		instances = append(instances, instance)
	}

	return instances
}

// AcknowledgeAlert acknowledges an alert instance
func (m *AlertManager) AcknowledgeAlert(instanceID, acknowledgedBy, comment string) error {
	m.instancesMutex.Lock()
	defer m.instancesMutex.Unlock()

	instance, exists := m.instances[instanceID]
	if !exists {
		return fmt.Errorf("alert instance with ID %s not found", instanceID)
	}

	instance.Status = AlertStatusAcknowledged
	instance.AcknowledgedBy = acknowledgedBy
	instance.AcknowledgedTime = time.Now()
	instance.AcknowledgementComment = comment

	return nil
}

// evaluationLoop evaluates alerts at regular intervals
func (m *AlertManager) evaluationLoop() {
	ticker := time.NewTicker(m.evaluationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.evaluateAlerts()
		case <-m.stopChan:
			return
		}
	}
}

// evaluateAlerts evaluates all alerts
func (m *AlertManager) evaluateAlerts() {
	// Get a snapshot of the alerts to evaluate
	m.alertsMutex.RLock()
	alerts := make([]*Alert, 0, len(m.alerts))
	for _, alert := range m.alerts {
		if alert.Enabled {
			alerts = append(alerts, alert)
		}
	}
	m.alertsMutex.RUnlock()

	// Evaluate each alert
	for _, alert := range alerts {
		m.evaluateAlert(alert)
	}
}

// evaluateAlert evaluates a single alert
func (m *AlertManager) evaluateAlert(alert *Alert) {
	// Get the metric data for the alert
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Define the time range based on the condition duration
	end := time.Now()
	start := end.Add(-alert.Condition.Duration * 2) // Get enough data to evaluate the condition

	series, err := m.metricCollector.GetMetric(ctx, alert.Condition.MetricName, alert.Condition.Tags, start, end)
	if err != nil {
		fmt.Printf("failed to get metric data for alert %s: %v\n", alert.ID, err)
		return
	}

	// If no data, skip
	if series == nil || len(series.Metrics) == 0 {
		return
	}

	// Check if condition is met
	conditionMet := false
	latestValue := 0.0

	// For threshold alerts, check all values in the duration window
	durationStart := end.Add(-alert.Condition.Duration)
	matchingValues := 0
	totalValues := 0

	// Find the latest value and count matching values in the duration window
	for _, metric := range series.Metrics {
		if metric.Timestamp.After(durationStart) {
			totalValues++
			if metric.Timestamp.After(time.Unix(0, 0)) && (latestValue == 0 || metric.Timestamp.After(time.Unix(0, 0))) {
				latestValue = metric.Value
			}

			// Check if this data point meets the condition
			if meetsCondition(metric.Value, alert.Condition.Threshold, alert.Condition.Operator) {
				matchingValues++
			}
		}
	}

	// Determine if the condition is met over the required duration
	if totalValues > 0 && matchingValues == totalValues {
		conditionMet = true
	}

	// Handle the alert status change
	instanceID := getAlertInstanceID(alert.ID, alert.Condition.MetricName, alert.Condition.Tags)

	m.instancesMutex.Lock()
	defer m.instancesMutex.Unlock()

	instance, exists := m.instances[instanceID]

	if conditionMet {
		// Alert is firing
		if !exists {
			// New alert firing
			instance = &AlertInstance{
				Alert:     alert,
				Value:     latestValue,
				StartTime: time.Now(),
				Status:    AlertStatusFiring,
			}
			m.instances[instanceID] = instance

			// Send notification for new alert
			m.sendAlertNotification(instance, true)
		} else if instance.Status == AlertStatusResolved {
			// Alert was resolved but is firing again
			instance.Status = AlertStatusFiring
			instance.StartTime = time.Now()
			instance.EndTime = time.Time{}

			// Send notification for re-firing alert
			m.sendAlertNotification(instance, true)
		}
		// If already firing, do nothing
	} else {
		// Alert is not firing
		if exists && instance.Status == AlertStatusFiring {
			// Alert was firing but is now resolved
			instance.Status = AlertStatusResolved
			instance.EndTime = time.Now()

			// Send notification for resolved alert
			m.sendAlertNotification(instance, false)
		}
	}
}

// sendAlertNotification sends a notification for an alert
func (m *AlertManager) sendAlertNotification(instance *AlertInstance, firing bool) {
	// If no notification channels configured, skip
	if len(instance.Alert.NotificationChannels) == 0 {
		return
	}

	// Create notification
	title := instance.Alert.Name
	message := instance.Alert.Description
	if message == "" {
		message = fmt.Sprintf("Alert %s is %s",
			instance.Alert.Name,
			instance.Status)
	}

	// Add details
	details := make(map[string]interface{})
	details["alertId"] = instance.Alert.ID
	details["severity"] = instance.Alert.Severity
	details["metric"] = instance.Alert.Condition.MetricName
	details["value"] = instance.Value
	details["threshold"] = instance.Alert.Condition.Threshold
	details["operator"] = instance.Alert.Condition.Operator
	details["status"] = instance.Status
	details["startTime"] = instance.StartTime
	if !instance.EndTime.IsZero() {
		details["endTime"] = instance.EndTime
	}

	// Send to each configured channel
	for _, channelID := range instance.Alert.NotificationChannels {
		notification := NewNotification(
			NotificationTypeAlert,
			title,
			message,
			instance.Alert.Severity.String(),
			details,
		)

		m.notificationManager.SendNotification(channelID, notification)
	}
}

// CheckMetric checks if a metric triggers any alerts
func (m *AlertManager) CheckMetric(metric *Metric) {
	// This is an optimization to check alerts immediately when a metric arrives
	// instead of waiting for the next evaluation cycle

	// Get all alerts for this metric
	m.alertsMutex.RLock()
	var matchingAlerts []*Alert
	for _, alert := range m.alerts {
		if alert.Enabled && alert.Condition.MetricName == metric.Name {
			// Check if tags match
			tagsMatch := true
			for k, v := range alert.Condition.Tags {
				if metric.Tags[k] != v {
					tagsMatch = false
					break
				}
			}
			if tagsMatch {
				matchingAlerts = append(matchingAlerts, alert)
			}
		}
	}
	m.alertsMutex.RUnlock()

	// For each matching alert, check if this single metric value triggers the condition
	for _, alert := range matchingAlerts {
		// For threshold alerts, we can check immediately
		if alert.Type == AlertTypeThreshold {
			if meetsCondition(metric.Value, alert.Condition.Threshold, alert.Condition.Operator) {
				// But we still need to check the duration, so we'll defer to the next evaluation cycle
				// This just gives a hint that an evaluation should happen soon

				// In a production system, we might implement a more sophisticated approach here,
				// like scheduling an immediate evaluation for this specific alert
			}
		}
	}
}

// meetsCondition checks if a value meets an alert condition
func meetsCondition(value, threshold float64, operator AlertConditionOperator) bool {
	switch operator {
	case AlertConditionOperatorEqual:
		return value == threshold
	case AlertConditionOperatorNotEqual:
		return value != threshold
	case AlertConditionOperatorGreaterThan:
		return value > threshold
	case AlertConditionOperatorGreaterThanOrEqual:
		return value >= threshold
	case AlertConditionOperatorLessThan:
		return value < threshold
	case AlertConditionOperatorLessThanOrEqual:
		return value <= threshold
	default:
		return false
	}
}

// getAlertInstanceID generates a unique ID for an alert instance
func getAlertInstanceID(alertID, metricName string, tags map[string]string) string {
	return fmt.Sprintf("%s:%s:%s", alertID, metricName, formatTags(tags))
}

// String returns the string representation of an AlertSeverity
func (s AlertSeverity) String() string {
	return string(s)
}
