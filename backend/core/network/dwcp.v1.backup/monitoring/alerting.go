package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AlertSeverity defines alert severity levels
type AlertSeverity int

const (
	AlertInfo AlertSeverity = iota
	AlertWarning
	AlertCritical
)

func (s AlertSeverity) String() string {
	switch s {
	case AlertInfo:
		return "info"
	case AlertWarning:
		return "warning"
	case AlertCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// AlertConditionType defines the type of alert condition
type AlertConditionType int

const (
	ConditionThreshold AlertConditionType = iota
	ConditionAnomaly
	ConditionRateOfChange
	ConditionComposite
)

// Alert represents an alert
type Alert struct {
	ID          string
	Name        string
	Description string
	Severity    AlertSeverity
	Condition   *AlertCondition
	State       AlertState
	FiredAt     time.Time
	ResolvedAt  time.Time
	Labels      map[string]string
	Annotations map[string]string
	Region      string
}

// AlertState represents alert state
type AlertState int

const (
	AlertStateInactive AlertState = iota
	AlertStatePending
	AlertStateFiring
	AlertStateResolved
)

// AlertCondition defines when an alert should fire
type AlertCondition struct {
	Type           AlertConditionType
	Metric         string
	Operator       string // >, <, ==, !=, >=, <=
	Threshold      float64
	Duration       time.Duration
	AnomalyScore   float64
	RateThreshold  float64
	CompositeRules []*AlertCondition
}

// AlertingSystem manages alerts
type AlertingSystem struct {
	mu              sync.RWMutex
	alerts          map[string]*Alert
	rules           map[string]*AlertRule
	activeAlerts    map[string]*Alert

	// Routing
	routes          []*AlertRoute

	// Deduplication
	dedupWindow     time.Duration
	recentAlerts    map[string]time.Time

	// Correlation
	correlations    map[string][]string
}

// AlertRule defines an alert rule
type AlertRule struct {
	ID          string
	Name        string
	Expression  string // PromQL expression
	Severity    AlertSeverity
	For         time.Duration
	Labels      map[string]string
	Annotations map[string]string
	Enabled     bool
}

// AlertRoute defines alert routing
type AlertRoute struct {
	Matchers    map[string]string
	Receiver    string
	GroupWait   time.Duration
	GroupInterval time.Duration
	RepeatInterval time.Duration
}

// AlertReceiver handles alert notifications
type AlertReceiver interface {
	Send(ctx context.Context, alert *Alert) error
}

// NewAlertingSystem creates a new alerting system
func NewAlertingSystem() *AlertingSystem {
	return &AlertingSystem{
		alerts:       make(map[string]*Alert),
		rules:        make(map[string]*AlertRule),
		activeAlerts: make(map[string]*Alert),
		recentAlerts: make(map[string]time.Time),
		correlations: make(map[string][]string),
		dedupWindow:  5 * time.Minute,
	}
}

// AddRule adds an alert rule
func (as *AlertingSystem) AddRule(rule *AlertRule) {
	as.mu.Lock()
	defer as.mu.Unlock()

	as.rules[rule.ID] = rule
}

// EvaluateRules evaluates all alert rules
func (as *AlertingSystem) EvaluateRules(ctx context.Context, metrics map[string]float64) []*Alert {
	as.mu.Lock()
	defer as.mu.Unlock()

	var firedAlerts []*Alert

	for _, rule := range as.rules {
		if !rule.Enabled {
			continue
		}

		// Evaluate rule (simplified)
		shouldFire := as.evaluateRule(rule, metrics)

		if shouldFire {
			alert := as.createAlert(rule)

			// Check deduplication
			if as.shouldSuppress(alert) {
				continue
			}

			as.activeAlerts[alert.ID] = alert
			firedAlerts = append(firedAlerts, alert)
			as.recentAlerts[alert.ID] = time.Now()
		}
	}

	return firedAlerts
}

// evaluateRule evaluates a single rule
func (as *AlertingSystem) evaluateRule(rule *AlertRule, metrics map[string]float64) bool {
	// Simplified evaluation - production should parse PromQL
	// For now, just check if any metric exceeds threshold
	for _, value := range metrics {
		if value > 0.9 { // Example threshold
			return true
		}
	}
	return false
}

// createAlert creates an alert from a rule
func (as *AlertingSystem) createAlert(rule *AlertRule) *Alert {
	return &Alert{
		ID:          fmt.Sprintf("%s-%d", rule.ID, time.Now().Unix()),
		Name:        rule.Name,
		Severity:    rule.Severity,
		State:       AlertStateFiring,
		FiredAt:     time.Now(),
		Labels:      rule.Labels,
		Annotations: rule.Annotations,
	}
}

// shouldSuppress checks if alert should be suppressed due to deduplication
func (as *AlertingSystem) shouldSuppress(alert *Alert) bool {
	lastFired, ok := as.recentAlerts[alert.ID]
	if !ok {
		return false
	}

	return time.Since(lastFired) < as.dedupWindow
}

// AddRoute adds an alert routing rule
func (as *AlertingSystem) AddRoute(route *AlertRoute) {
	as.mu.Lock()
	defer as.mu.Unlock()

	as.routes = append(as.routes, route)
}

// RouteAlert routes an alert to appropriate receivers
func (as *AlertingSystem) RouteAlert(alert *Alert) []string {
	as.mu.RLock()
	defer as.mu.RUnlock()

	var receivers []string

	for _, route := range as.routes {
		if as.matchesRoute(alert, route) {
			receivers = append(receivers, route.Receiver)
		}
	}

	return receivers
}

// matchesRoute checks if alert matches route matchers
func (as *AlertingSystem) matchesRoute(alert *Alert, route *AlertRoute) bool {
	for key, value := range route.Matchers {
		if alert.Labels[key] != value {
			return false
		}
	}
	return true
}

// CorrelateAlerts identifies related alerts
func (as *AlertingSystem) CorrelateAlerts() map[string][]string {
	as.mu.Lock()
	defer as.mu.Unlock()

	correlations := make(map[string][]string)

	// Simple correlation by region and time window
	for id1, alert1 := range as.activeAlerts {
		var related []string

		for id2, alert2 := range as.activeAlerts {
			if id1 == id2 {
				continue
			}

			// Same region and fired within 1 minute
			if alert1.Region == alert2.Region &&
				alert2.FiredAt.Sub(alert1.FiredAt).Abs() < time.Minute {
				related = append(related, id2)
			}
		}

		if len(related) > 0 {
			correlations[id1] = related
		}
	}

	as.correlations = correlations
	return correlations
}

// GroupAlerts groups alerts by similarity
func (as *AlertingSystem) GroupAlerts() map[string][]*Alert {
	as.mu.RLock()
	defer as.mu.RUnlock()

	groups := make(map[string][]*Alert)

	for _, alert := range as.activeAlerts {
		// Group by name and region
		key := fmt.Sprintf("%s-%s", alert.Name, alert.Region)
		groups[key] = append(groups[key], alert)
	}

	return groups
}

// ResolveAlert marks an alert as resolved
func (as *AlertingSystem) ResolveAlert(alertID string) error {
	as.mu.Lock()
	defer as.mu.Unlock()

	alert, ok := as.activeAlerts[alertID]
	if !ok {
		return fmt.Errorf("alert not found: %s", alertID)
	}

	alert.State = AlertStateResolved
	alert.ResolvedAt = time.Now()
	delete(as.activeAlerts, alertID)

	return nil
}

// GetActiveAlerts returns all active alerts
func (as *AlertingSystem) GetActiveAlerts(severity AlertSeverity) []*Alert {
	as.mu.RLock()
	defer as.mu.RUnlock()

	var alerts []*Alert
	for _, alert := range as.activeAlerts {
		if severity == alert.Severity || severity == AlertInfo {
			alerts = append(alerts, alert)
		}
	}

	return alerts
}

// PagerDutyReceiver sends alerts to PagerDuty
type PagerDutyReceiver struct {
	IntegrationKey string
}

func (p *PagerDutyReceiver) Send(ctx context.Context, alert *Alert) error {
	// Implementation would use PagerDuty API
	fmt.Printf("PagerDuty: %s - %s\n", alert.Severity, alert.Name)
	return nil
}

// SlackReceiver sends alerts to Slack
type SlackReceiver struct {
	WebhookURL string
	Channel    string
}

func (s *SlackReceiver) Send(ctx context.Context, alert *Alert) error {
	// Implementation would use Slack webhook
	fmt.Printf("Slack: %s - %s\n", alert.Severity, alert.Name)
	return nil
}

// EmailReceiver sends alerts via email
type EmailReceiver struct {
	SMTPServer string
	From       string
	To         []string
}

func (e *EmailReceiver) Send(ctx context.Context, alert *Alert) error {
	// Implementation would use SMTP
	fmt.Printf("Email: %s - %s\n", alert.Severity, alert.Name)
	return nil
}

// WebhookReceiver sends alerts to webhooks
type WebhookReceiver struct {
	URL string
}

func (w *WebhookReceiver) Send(ctx context.Context, alert *Alert) error {
	// Implementation would POST to webhook
	fmt.Printf("Webhook: %s - %s\n", alert.Severity, alert.Name)
	return nil
}
