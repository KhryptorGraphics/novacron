package monitoring

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// mockNotificationChannel captures notifications for testing
type mockNotificationChannel struct {
	id            string
	enabled       bool
	notifications []*Notification
}

func (c *mockNotificationChannel) ID() string { return c.id }
func (c *mockNotificationChannel) Send(n *Notification) error {
	c.notifications = append(c.notifications, n)
	return nil
}
func (c *mockNotificationChannel) IsEnabled() bool                { return c.enabled }
func (c *mockNotificationChannel) Type() string                   { return "mock" }
func (c *mockNotificationChannel) Notifications() []*Notification { return c.notifications }

func TestMonitoringAlertTypesAndEdgeCases(t *testing.T) {
	now := time.Now()
	memStorage := storage.NewInMemoryStorage()
	collectorConfig := DefaultDistributedMetricCollectorConfig()
	collector := NewDistributedMetricCollector(collectorConfig, memStorage)
	alertManager := NewAlertManager(collector)
	mockChan := &mockNotificationChannel{id: "mock", enabled: true}
	alertManager.notificationManager.RegisterChannel(mockChan)

	// Rate-of-change alert: triggers if CPU increases by >50 in 1s
	rateAlert := &Alert{
		ID:          "cpu-roc",
		Name:        "CPU Rate Spike",
		Description: "CPU usage increased rapidly",
		Severity:    AlertSeverityWarning,
		Type:        AlertTypeRateOfChange,
		Condition: AlertCondition{
			MetricName: "cpu_usage",
			Operator:   AlertConditionOperatorGreaterThan,
			Threshold:  50.0,
			Duration:   1 * time.Second,
			Tags:       map[string]string{},
		},
		NotificationChannels: []string{"mock"},
		Enabled:              true,
		Status:               AlertStatusResolved,
	}
	alertManager.RegisterAlert(rateAlert)

	// Anomaly alert: triggers on outlier value
	anomalyAlert := &Alert{
		ID:          "cpu-anomaly",
		Name:        "CPU Anomaly",
		Description: "CPU anomaly detected",
		Severity:    AlertSeverityCritical,
		Type:        AlertTypeAnomaly,
		Condition: AlertCondition{
			MetricName: "cpu_usage",
			Operator:   AlertConditionOperatorGreaterThan,
			Threshold:  200.0,
			Duration:   1 * time.Second,
			Tags:       map[string]string{},
		},
		NotificationChannels: []string{"mock"},
		Enabled:              true,
		Status:               AlertStatusResolved,
	}
	alertManager.RegisterAlert(anomalyAlert)

	// Event-based alert: triggers on event metric
	eventAlert := &Alert{
		ID:          "disk-event",
		Name:        "Disk Failure Event",
		Description: "Disk failure event detected",
		Severity:    AlertSeverityError,
		Type:        AlertTypeEvent,
		Condition: AlertCondition{
			MetricName: "disk_failure",
			Operator:   AlertConditionOperatorEqual,
			Threshold:  1.0,
			Duration:   0,
			Tags:       map[string]string{},
		},
		NotificationChannels: []string{"mock"},
		Enabled:              true,
		Status:               AlertStatusResolved,
	}
	alertManager.RegisterAlert(eventAlert)

	// Edge case: disabled alert should not fire
	disabledAlert := &Alert{
		ID:          "disabled",
		Name:        "Disabled Alert",
		Description: "Should not fire",
		Severity:    AlertSeverityInfo,
		Type:        AlertTypeThreshold,
		Condition: AlertCondition{
			MetricName: "cpu_usage",
			Operator:   AlertConditionOperatorGreaterThan,
			Threshold:  0.0,
			Duration:   1 * time.Second,
			Tags:       map[string]string{},
		},
		NotificationChannels: []string{"mock"},
		Enabled:              false,
		Status:               AlertStatusResolved,
	}
	alertManager.RegisterAlert(disabledAlert)

	// Store metrics for rate-of-change and anomaly
	ctx := context.Background()
	metrics := []*Metric{
		{Name: "cpu_usage", Type: "gauge", Value: 10, Timestamp: now.Add(-2 * time.Second), Tags: map[string]string{"vm_id": "vm-1"}},
		{Name: "cpu_usage", Type: "gauge", Value: 70, Timestamp: now.Add(-1 * time.Second), Tags: map[string]string{"vm_id": "vm-1"}}, // triggers rate-of-change
		{Name: "cpu_usage", Type: "gauge", Value: 250, Timestamp: now, Tags: map[string]string{"vm_id": "vm-1"}},                      // triggers anomaly
		{Name: "disk_failure", Type: "event", Value: 1, Timestamp: now, Tags: map[string]string{"vm_id": "vm-1"}},                     // triggers event
	}

	for _, m := range metrics {
		if err := collector.StoreMetric(ctx, m); err != nil {
			t.Fatalf("Failed to store metric: %v", err)
		}
	}

	// Manually trigger alert evaluation
	alertManager.evaluateAlerts()

	// Check that the correct alerts fired
	instances := alertManager.ListAlertInstances()
	alertIDs := map[string]bool{}
	for _, inst := range instances {
		alertIDs[inst.Alert.ID] = true
	}

	if !alertIDs["cpu-roc"] {
		t.Error("Expected rate-of-change alert to fire")
	}
	if !alertIDs["cpu-anomaly"] {
		t.Error("Expected anomaly alert to fire")
	}
	if !alertIDs["disk-event"] {
		t.Error("Expected event alert to fire")
	}
	if alertIDs["disabled"] {
		t.Error("Disabled alert should not fire")
	}

	// Edge case: no metrics for a metric name
	noMetricAlert := &Alert{
		ID:          "no-metric",
		Name:        "No Metric",
		Description: "Should not fire",
		Severity:    AlertSeverityInfo,
		Type:        AlertTypeThreshold,
		Condition: AlertCondition{
			MetricName: "nonexistent_metric",
			Operator:   AlertConditionOperatorGreaterThan,
			Threshold:  0.0,
			Duration:   1 * time.Second,
			Tags:       map[string]string{},
		},
		NotificationChannels: []string{"mock"},
		Enabled:              true,
		Status:               AlertStatusResolved,
	}
	alertManager.RegisterAlert(noMetricAlert)
	alertManager.evaluateAlerts()
	instances = alertManager.ListAlertInstances()
	for _, inst := range instances {
		if inst.Alert.ID == "no-metric" && inst.Status == AlertStatusFiring {
			t.Error("Alert for nonexistent metric should not fire")
		}
	}
}
