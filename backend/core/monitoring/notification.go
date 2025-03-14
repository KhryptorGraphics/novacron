package monitoring

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/smtp"
	"strings"
	"sync"
	"text/template"
	"time"
)

// NotificationChannel represents a notification channel
type NotificationChannel string

const (
	// EmailChannel is the email notification channel
	EmailChannel NotificationChannel = "email"

	// WebhookChannel is the webhook notification channel
	WebhookChannel NotificationChannel = "webhook"

	// SlackChannel is the Slack notification channel
	SlackChannel NotificationChannel = "slack"

	// TeamsChannel is the Microsoft Teams notification channel
	TeamsChannel NotificationChannel = "teams"

	// SMSChannel is the SMS notification channel
	SMSChannel NotificationChannel = "sms"

	// PagerDutyChannel is the PagerDuty notification channel
	PagerDutyChannel NotificationChannel = "pagerduty"

	// OpsGenieChannel is the OpsGenie notification channel
	OpsGenieChannel NotificationChannel = "opsgenie"
)

// NotificationTemplate represents a notification template
type NotificationTemplate struct {
	// ID is the unique identifier for the template
	ID string `json:"id"`

	// Name is the human-readable name of the template
	Name string `json:"name"`

	// Description is a description of the template
	Description string `json:"description"`

	// TenantID is the ID of the tenant this template belongs to
	TenantID string `json:"tenantId,omitempty"`

	// Channel is the notification channel this template is for
	Channel NotificationChannel `json:"channel"`

	// Subject is the subject of the notification (for email)
	Subject string `json:"subject,omitempty"`

	// Body is the body of the notification
	Body string `json:"body"`

	// Format is the format of the notification (text, html, markdown)
	Format string `json:"format"`

	// Compiled is the compiled template
	compiled *template.Template `json:"-"`
}

// NotificationConfig represents the configuration for a notification channel
type NotificationConfig struct {
	// ID is the unique identifier for the configuration
	ID string `json:"id"`

	// Name is the human-readable name of the configuration
	Name string `json:"name"`

	// Channel is the notification channel
	Channel NotificationChannel `json:"channel"`

	// Enabled indicates if the channel is enabled
	Enabled bool `json:"enabled"`

	// TenantID is the ID of the tenant this configuration belongs to
	TenantID string `json:"tenantId,omitempty"`

	// Settings contains channel-specific settings
	Settings map[string]interface{} `json:"settings"`
}

// EmailNotifierSettings represents settings for the email notifier
type EmailNotifierSettings struct {
	// Server is the SMTP server
	Server string `json:"server"`

	// Port is the SMTP port
	Port int `json:"port"`

	// Username is the SMTP username
	Username string `json:"username"`

	// Password is the SMTP password
	Password string `json:"password"`

	// FromAddress is the sender address
	FromAddress string `json:"fromAddress"`

	// ToAddresses are the recipient addresses
	ToAddresses []string `json:"toAddresses"`

	// UseTLS indicates if TLS should be used
	UseTLS bool `json:"useTLS"`
}

// WebhookNotifierSettings represents settings for the webhook notifier
type WebhookNotifierSettings struct {
	// URL is the webhook URL
	URL string `json:"url"`

	// Method is the HTTP method to use
	Method string `json:"method"`

	// Headers are the HTTP headers to include
	Headers map[string]string `json:"headers"`

	// Timeout is the request timeout
	Timeout time.Duration `json:"timeout"`

	// RetryCount is the number of retries
	RetryCount int `json:"retryCount"`

	// BasicAuthUsername is the username for basic auth
	BasicAuthUsername string `json:"basicAuthUsername,omitempty"`

	// BasicAuthPassword is the password for basic auth
	BasicAuthPassword string `json:"basicAuthPassword,omitempty"`
}

// NotificationManager manages notifications
type NotificationManager struct {
	templates map[string]*NotificationTemplate
	configs   map[string]*NotificationConfig
	notifiers map[NotificationChannel][]AlertNotifier
	mutex     sync.RWMutex
}

// NewNotificationManager creates a new notification manager
func NewNotificationManager() *NotificationManager {
	return &NotificationManager{
		templates: make(map[string]*NotificationTemplate),
		configs:   make(map[string]*NotificationConfig),
		notifiers: make(map[NotificationChannel][]AlertNotifier),
	}
}

// AddTemplate adds a notification template
func (m *NotificationManager) AddTemplate(template *NotificationTemplate) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Compile the template
	compiled, err := compileTemplate(template.Body)
	if err != nil {
		return fmt.Errorf("failed to compile template: %w", err)
	}
	template.compiled = compiled

	m.templates[template.ID] = template
	return nil
}

// GetTemplate gets a notification template by ID
func (m *NotificationManager) GetTemplate(id string) (*NotificationTemplate, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	template, exists := m.templates[id]
	if !exists {
		return nil, fmt.Errorf("template not found: %s", id)
	}

	return template, nil
}

// AddConfig adds a notification configuration
func (m *NotificationManager) AddConfig(config *NotificationConfig) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.configs[config.ID] = config
	return nil
}

// GetConfig gets a notification configuration by ID
func (m *NotificationManager) GetConfig(id string) (*NotificationConfig, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	config, exists := m.configs[id]
	if !exists {
		return nil, fmt.Errorf("configuration not found: %s", id)
	}

	return config, nil
}

// SendNotification sends a notification
func (m *NotificationManager) SendNotification(alert *Alert, templateID string) error {
	// Get the template
	template, err := m.GetTemplate(templateID)
	if err != nil {
		return err
	}

	// Render the template
	data := map[string]interface{}{
		"Alert": alert,
	}
	renderedBody, err := renderTemplate(template.compiled, data)
	if err != nil {
		return fmt.Errorf("failed to render template: %w", err)
	}

	// Get the notifiers for this channel
	m.mutex.RLock()
	notifiers, exists := m.notifiers[template.Channel]
	m.mutex.RUnlock()

	if !exists || len(notifiers) == 0 {
		return fmt.Errorf("no notifiers for channel: %s", template.Channel)
	}

	// Send the notification through each notifier
	var lastErr error
	for _, notifier := range notifiers {
		if customNotifier, ok := notifier.(CustomTemplateNotifier); ok {
			err = customNotifier.NotifyWithTemplate(alert, renderedBody, template)
		} else {
			err = notifier.Notify(alert)
		}

		if err != nil {
			lastErr = err
		}
	}

	return lastErr
}

// RegisterNotifier registers a notifier
func (m *NotificationManager) RegisterNotifier(channel NotificationChannel, notifier AlertNotifier) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if _, exists := m.notifiers[channel]; !exists {
		m.notifiers[channel] = make([]AlertNotifier, 0)
	}

	m.notifiers[channel] = append(m.notifiers[channel], notifier)
}

// CreateEmailNotifier creates an email notifier
func (m *NotificationManager) CreateEmailNotifier(configID string) error {
	// Get the configuration
	config, err := m.GetConfig(configID)
	if err != nil {
		return err
	}

	if config.Channel != EmailChannel {
		return fmt.Errorf("configuration is not for email channel: %s", config.Channel)
	}

	// Parse settings
	settings := &EmailNotifierSettings{}
	settingsJSON, err := json.Marshal(config.Settings)
	if err != nil {
		return fmt.Errorf("failed to marshal settings: %w", err)
	}
	err = json.Unmarshal(settingsJSON, settings)
	if err != nil {
		return fmt.Errorf("failed to unmarshal settings: %w", err)
	}

	// Create and register the notifier
	notifier := NewEmailNotifier(settings)
	m.RegisterNotifier(EmailChannel, notifier)

	return nil
}

// CreateWebhookNotifier creates a webhook notifier
func (m *NotificationManager) CreateWebhookNotifier(configID string) error {
	// Get the configuration
	config, err := m.GetConfig(configID)
	if err != nil {
		return err
	}

	if config.Channel != WebhookChannel {
		return fmt.Errorf("configuration is not for webhook channel: %s", config.Channel)
	}

	// Parse settings
	settings := &WebhookNotifierSettings{}
	settingsJSON, err := json.Marshal(config.Settings)
	if err != nil {
		return fmt.Errorf("failed to marshal settings: %w", err)
	}
	err = json.Unmarshal(settingsJSON, settings)
	if err != nil {
		return fmt.Errorf("failed to unmarshal settings: %w", err)
	}

	// Create and register the notifier
	notifier := NewWebhookNotifier(settings)
	m.RegisterNotifier(WebhookChannel, notifier)

	return nil
}

// CustomTemplateNotifier is a notifier that supports custom templates
type CustomTemplateNotifier interface {
	AlertNotifier
	NotifyWithTemplate(alert *Alert, renderedBody string, template *NotificationTemplate) error
}

// EmailNotifier sends notifications via email
type EmailNotifier struct {
	settings *EmailNotifierSettings
}

// NewEmailNotifier creates a new email notifier
func NewEmailNotifier(settings *EmailNotifierSettings) *EmailNotifier {
	return &EmailNotifier{
		settings: settings,
	}
}

// Notify sends a notification
func (n *EmailNotifier) Notify(alert *Alert) error {
	// Create a default message if no template is provided
	subject := fmt.Sprintf("Alert: %s", alert.Name)
	body := fmt.Sprintf("Alert: %s\nSeverity: %s\nDescription: %s\nState: %s\n",
		alert.Name, alert.Severity, alert.Description, alert.State)

	return n.sendEmail(subject, body)
}

// NotifyWithTemplate sends a notification with a template
func (n *EmailNotifier) NotifyWithTemplate(alert *Alert, renderedBody string, template *NotificationTemplate) error {
	// Use the template subject or a default
	subject := template.Subject
	if subject == "" {
		subject = fmt.Sprintf("Alert: %s", alert.Name)
	}

	return n.sendEmail(subject, renderedBody)
}

// sendEmail sends an email
func (n *EmailNotifier) sendEmail(subject, body string) error {
	// Compose message
	message := fmt.Sprintf("From: %s\r\n"+
		"To: %s\r\n"+
		"Subject: %s\r\n"+
		"\r\n"+
		"%s", n.settings.FromAddress, strings.Join(n.settings.ToAddresses, ","), subject, body)

	// Set up authentication information.
	auth := smtp.PlainAuth("", n.settings.Username, n.settings.Password, n.settings.Server)

	// Connect to the server, authenticate, set the sender and recipient,
	// and send the email
	err := smtp.SendMail(
		fmt.Sprintf("%s:%d", n.settings.Server, n.settings.Port),
		auth,
		n.settings.FromAddress,
		n.settings.ToAddresses,
		[]byte(message),
	)

	if err != nil {
		return fmt.Errorf("failed to send email: %w", err)
	}

	return nil
}

// WebhookNotifier sends notifications via webhook
type WebhookNotifier struct {
	settings *WebhookNotifierSettings
	client   *http.Client
}

// NewWebhookNotifier creates a new webhook notifier
func NewWebhookNotifier(settings *WebhookNotifierSettings) *WebhookNotifier {
	client := &http.Client{
		Timeout: settings.Timeout,
	}

	return &WebhookNotifier{
		settings: settings,
		client:   client,
	}
}

// Notify sends a notification
func (n *WebhookNotifier) Notify(alert *Alert) error {
	// Marshal the alert to JSON
	alertBytes, err := json.Marshal(alert)
	if err != nil {
		return fmt.Errorf("failed to marshal alert: %w", err)
	}

	return n.sendWebhook(alertBytes)
}

// NotifyWithTemplate sends a notification with a template
func (n *WebhookNotifier) NotifyWithTemplate(alert *Alert, renderedBody string, template *NotificationTemplate) error {
	// For webhook, we might need to construct a specific payload format
	// For simplicity, we'll just use the rendered template as is
	return n.sendWebhook([]byte(renderedBody))
}

// sendWebhook sends a webhook
func (n *WebhookNotifier) sendWebhook(payload []byte) error {
	// Create request
	method := n.settings.Method
	if method == "" {
		method = "POST"
	}

	req, err := http.NewRequest(method, n.settings.URL, bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	for k, v := range n.settings.Headers {
		req.Header.Set(k, v)
	}

	// Add basic auth if provided
	if n.settings.BasicAuthUsername != "" && n.settings.BasicAuthPassword != "" {
		req.SetBasicAuth(n.settings.BasicAuthUsername, n.settings.BasicAuthPassword)
	}

	// Make the request
	resp, err := n.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send webhook: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook request failed with status: %s", resp.Status)
	}

	return nil
}

// compileTemplate compiles a template
func compileTemplate(body string) (*template.Template, error) {
	tmpl, err := template.New("notification").Parse(body)
	if err != nil {
		return nil, err
	}
	return tmpl, nil
}

// renderTemplate renders a template
func renderTemplate(tmpl *template.Template, data interface{}) (string, error) {
	var buf bytes.Buffer
	err := tmpl.Execute(&buf, data)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

// DefaultEmailTemplate returns a default email template
func DefaultEmailTemplate() *NotificationTemplate {
	body := `
Alert: {{.Alert.Name}}
Severity: {{.Alert.Severity}}
Status: {{.Alert.State}}
Time: {{.Alert.LastUpdatedAt}}

Description: {{.Alert.Description}}

{{if .Alert.CurrentValue}}Current Value: {{.Alert.CurrentValue}}{{end}}

Resource: {{.Alert.ResourceType}} ({{.Alert.ResourceID}})

{{if .Alert.Runbook}}Runbook: {{.Alert.Runbook}}{{end}}
`

	template := &NotificationTemplate{
		ID:          "default-email-template",
		Name:        "Default Email Template",
		Description: "Default template for email notifications",
		Channel:     EmailChannel,
		Subject:     "Alert: {{.Alert.Name}} - {{.Alert.Severity}}",
		Body:        body,
		Format:      "text",
	}

	// Pre-compile the template
	compiled, _ := compileTemplate(body)
	template.compiled = compiled

	return template
}

// DefaultWebhookTemplate returns a default webhook template
func DefaultWebhookTemplate() *NotificationTemplate {
	body := `{
  "alert": {
    "id": "{{.Alert.ID}}",
    "name": "{{.Alert.Name}}",
    "description": "{{.Alert.Description}}",
    "severity": "{{.Alert.Severity}}",
    "state": "{{.Alert.State}}",
    "resourceType": "{{.Alert.ResourceType}}",
    "resourceId": "{{.Alert.ResourceID}}",
    "timestamp": "{{.Alert.LastUpdatedAt}}"
  }
}`

	template := &NotificationTemplate{
		ID:          "default-webhook-template",
		Name:        "Default Webhook Template",
		Description: "Default template for webhook notifications",
		Channel:     WebhookChannel,
		Body:        body,
		Format:      "json",
	}

	// Pre-compile the template
	compiled, _ := compileTemplate(body)
	template.compiled = compiled

	return template
}
