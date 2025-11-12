package monitoring

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
)

// AlertManager manages alert dispatch to multiple channels
type AlertManager struct {
	config *AlertConfig
	logger *zap.Logger

	// Alert throttling
	alertHistory map[string]time.Time
	historyMu    sync.RWMutex

	httpClient *http.Client
}

// AlertConfig holds configuration for alert manager
type AlertConfig struct {
	// Webhook
	WebhookURL     string
	WebhookEnabled bool

	// Slack
	SlackWebhookURL string
	SlackChannel    string
	SlackEnabled    bool

	// PagerDuty
	PagerDutyKey     string
	PagerDutyEnabled bool

	// Email
	EmailRecipients []string
	SMTPServer      string
	SMTPPort        int
	SMTPUsername    string
	SMTPPassword    string
	EmailEnabled    bool

	// Alert throttling
	ThrottleDuration time.Duration
}

// DefaultAlertConfig returns default alert configuration
func DefaultAlertConfig() *AlertConfig {
	return &AlertConfig{
		WebhookEnabled:   false,
		SlackEnabled:     false,
		PagerDutyEnabled: false,
		EmailEnabled:     false,
		ThrottleDuration: 5 * time.Minute,
	}
}

// NewAlertManager creates a new alert manager
func NewAlertManager(config *AlertConfig, logger *zap.Logger) *AlertManager {
	if config == nil {
		config = DefaultAlertConfig()
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	return &AlertManager{
		config:       config,
		logger:       logger,
		alertHistory: make(map[string]time.Time),
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// SendCriticalAlert sends a critical severity alert
func (am *AlertManager) SendCriticalAlert(anomaly *Anomaly) error {
	// Check throttling
	if am.shouldThrottle(anomaly) {
		am.logger.Debug("Alert throttled",
			zap.String("metric", anomaly.MetricName),
			zap.String("severity", "critical"))
		return nil
	}

	message := am.formatCriticalMessage(anomaly)

	var errors []error

	// Send to all enabled channels
	if am.config.SlackEnabled {
		if err := am.sendSlackAlert(message, "critical"); err != nil {
			errors = append(errors, fmt.Errorf("slack alert failed: %w", err))
		}
	}

	if am.config.PagerDutyEnabled {
		if err := am.sendPagerDutyAlert(message, anomaly); err != nil {
			errors = append(errors, fmt.Errorf("pagerduty alert failed: %w", err))
		}
	}

	if am.config.WebhookEnabled {
		if err := am.sendWebhookAlert(anomaly, "critical"); err != nil {
			errors = append(errors, fmt.Errorf("webhook alert failed: %w", err))
		}
	}

	if am.config.EmailEnabled {
		if err := am.sendEmailAlert(message, anomaly, "critical"); err != nil {
			errors = append(errors, fmt.Errorf("email alert failed: %w", err))
		}
	}

	// Record alert
	am.recordAlert(anomaly)

	if len(errors) > 0 {
		return fmt.Errorf("some alerts failed: %v", errors)
	}

	am.logger.Info("Critical alert sent",
		zap.String("metric", anomaly.MetricName),
		zap.Float64("confidence", anomaly.Confidence))

	return nil
}

// SendWarningAlert sends a warning severity alert
func (am *AlertManager) SendWarningAlert(anomaly *Anomaly) error {
	// Check throttling
	if am.shouldThrottle(anomaly) {
		am.logger.Debug("Alert throttled",
			zap.String("metric", anomaly.MetricName),
			zap.String("severity", "warning"))
		return nil
	}

	message := am.formatWarningMessage(anomaly)

	var errors []error

	// Send to Slack and Webhook only (not PagerDuty for warnings)
	if am.config.SlackEnabled {
		if err := am.sendSlackAlert(message, "warning"); err != nil {
			errors = append(errors, fmt.Errorf("slack alert failed: %w", err))
		}
	}

	if am.config.WebhookEnabled {
		if err := am.sendWebhookAlert(anomaly, "warning"); err != nil {
			errors = append(errors, fmt.Errorf("webhook alert failed: %w", err))
		}
	}

	// Record alert
	am.recordAlert(anomaly)

	if len(errors) > 0 {
		return fmt.Errorf("some alerts failed: %v", errors)
	}

	am.logger.Info("Warning alert sent",
		zap.String("metric", anomaly.MetricName),
		zap.Float64("confidence", anomaly.Confidence))

	return nil
}

// formatCriticalMessage formats a critical alert message
func (am *AlertManager) formatCriticalMessage(anomaly *Anomaly) string {
	return fmt.Sprintf(
		"🚨 CRITICAL ANOMALY DETECTED\n\n"+
			"Metric: %s\n"+
			"Current Value: %.2f\n"+
			"Expected Value: %.2f\n"+
			"Deviation: %.2f (%.1f%%)\n"+
			"Confidence: %.1f%%\n"+
			"Model: %s\n"+
			"Time: %s\n\n"+
			"Description: %s",
		anomaly.MetricName,
		anomaly.Value,
		anomaly.Expected,
		anomaly.Deviation,
		(anomaly.Deviation/anomaly.Expected)*100,
		anomaly.Confidence*100,
		anomaly.ModelType,
		anomaly.Timestamp.Format(time.RFC3339),
		anomaly.Description,
	)
}

// formatWarningMessage formats a warning alert message
func (am *AlertManager) formatWarningMessage(anomaly *Anomaly) string {
	return fmt.Sprintf(
		"⚠️  WARNING: Anomaly Detected\n\n"+
			"Metric: %s\n"+
			"Current Value: %.2f\n"+
			"Expected Value: %.2f\n"+
			"Deviation: %.2f (%.1f%%)\n"+
			"Confidence: %.1f%%\n"+
			"Model: %s\n"+
			"Time: %s\n\n"+
			"Description: %s",
		anomaly.MetricName,
		anomaly.Value,
		anomaly.Expected,
		anomaly.Deviation,
		(anomaly.Deviation/anomaly.Expected)*100,
		anomaly.Confidence*100,
		anomaly.ModelType,
		anomaly.Timestamp.Format(time.RFC3339),
		anomaly.Description,
	)
}

// sendSlackAlert sends alert to Slack
func (am *AlertManager) sendSlackAlert(message, severity string) error {
	color := "warning"
	if severity == "critical" {
		color = "danger"
	}

	payload := map[string]interface{}{
		"channel": am.config.SlackChannel,
		"attachments": []map[string]interface{}{
			{
				"color": color,
				"text":  message,
				"footer": "DWCP Anomaly Detection",
				"ts":    time.Now().Unix(),
			},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := am.httpClient.Post(
		am.config.SlackWebhookURL,
		"application/json",
		bytes.NewBuffer(body),
	)

	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("slack returned status %d", resp.StatusCode)
	}

	return nil
}

// sendPagerDutyAlert sends alert to PagerDuty
func (am *AlertManager) sendPagerDutyAlert(message string, anomaly *Anomaly) error {
	payload := map[string]interface{}{
		"routing_key":  am.config.PagerDutyKey,
		"event_action": "trigger",
		"payload": map[string]interface{}{
			"summary":   fmt.Sprintf("Critical anomaly in %s", anomaly.MetricName),
			"severity":  "critical",
			"source":    "dwcp-monitoring",
			"timestamp": anomaly.Timestamp.Format(time.RFC3339),
			"custom_details": map[string]interface{}{
				"metric":     anomaly.MetricName,
				"value":      anomaly.Value,
				"expected":   anomaly.Expected,
				"confidence": anomaly.Confidence,
				"model":      anomaly.ModelType,
			},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := am.httpClient.Post(
		"https://events.pagerduty.com/v2/enqueue",
		"application/json",
		bytes.NewBuffer(body),
	)

	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		return fmt.Errorf("pagerduty returned status %d", resp.StatusCode)
	}

	return nil
}

// sendWebhookAlert sends alert to generic webhook
func (am *AlertManager) sendWebhookAlert(anomaly *Anomaly, severity string) error {
	payload := map[string]interface{}{
		"severity":    severity,
		"metric":      anomaly.MetricName,
		"value":       anomaly.Value,
		"expected":    anomaly.Expected,
		"deviation":   anomaly.Deviation,
		"confidence":  anomaly.Confidence,
		"model":       anomaly.ModelType,
		"timestamp":   anomaly.Timestamp,
		"description": anomaly.Description,
		"context":     anomaly.Context,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := am.httpClient.Post(
		am.config.WebhookURL,
		"application/json",
		bytes.NewBuffer(body),
	)

	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook returned status %d", resp.StatusCode)
	}

	return nil
}

// sendEmailAlert sends alert via email
func (am *AlertManager) sendEmailAlert(message string, anomaly *Anomaly, severity string) error {
	// Email sending implementation would go here
	// For now, just log
	am.logger.Info("Email alert would be sent",
		zap.String("severity", severity),
		zap.String("metric", anomaly.MetricName),
		zap.Strings("recipients", am.config.EmailRecipients))

	return nil
}

// shouldThrottle checks if an alert should be throttled
func (am *AlertManager) shouldThrottle(anomaly *Anomaly) bool {
	key := fmt.Sprintf("%s:%s", anomaly.MetricName, anomaly.Severity)

	am.historyMu.RLock()
	lastAlert, exists := am.alertHistory[key]
	am.historyMu.RUnlock()

	if !exists {
		return false
	}

	return time.Since(lastAlert) < am.config.ThrottleDuration
}

// recordAlert records that an alert was sent
func (am *AlertManager) recordAlert(anomaly *Anomaly) {
	key := fmt.Sprintf("%s:%s", anomaly.MetricName, anomaly.Severity)

	am.historyMu.Lock()
	am.alertHistory[key] = time.Now()
	am.historyMu.Unlock()
}

// ClearAlertHistory clears the alert history
func (am *AlertManager) ClearAlertHistory() {
	am.historyMu.Lock()
	am.alertHistory = make(map[string]time.Time)
	am.historyMu.Unlock()
}

// GetAlertHistory returns the alert history
func (am *AlertManager) GetAlertHistory() map[string]time.Time {
	am.historyMu.RLock()
	defer am.historyMu.RUnlock()

	history := make(map[string]time.Time)
	for k, v := range am.alertHistory {
		history[k] = v
	}

	return history
}
