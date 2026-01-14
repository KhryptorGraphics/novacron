package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// NotificationType represents the type of notification
type NotificationType string

const (
	// NotificationTypeAlert is a notification for an alert
	NotificationTypeAlert NotificationType = "alert"

	// NotificationTypeSystem is a notification for a system event
	NotificationTypeSystem NotificationType = "system"

	// NotificationTypeInfo is an informational notification
	NotificationTypeInfo NotificationType = "info"
)

// NotificationChannel represents a notification channel
type NotificationChannel interface {
	// ID returns the channel ID
	ID() string

	// Send sends a notification
	Send(notification *Notification) error

	// IsEnabled returns whether the channel is enabled
	IsEnabled() bool

	// Type returns the channel type
	Type() string
}

// Notification represents a notification
type Notification struct {
	// Type is the notification type
	Type NotificationType `json:"type"`

	// Title is the notification title
	Title string `json:"title"`

	// Message is the notification message
	Message string `json:"message"`

	// Severity is the notification severity
	Severity string `json:"severity,omitempty"`

	// Timestamp is when the notification was created
	Timestamp time.Time `json:"timestamp"`

	// Details contains additional information
	Details map[string]interface{} `json:"details,omitempty"`
}

// NewNotification creates a new notification
func NewNotification(notificationType NotificationType, title, message, severity string, details map[string]interface{}) *Notification {
	return &Notification{
		Type:      notificationType,
		Title:     title,
		Message:   message,
		Severity:  severity,
		Timestamp: time.Now(),
		Details:   details,
	}
}

// EmailChannel is a notification channel that sends emails
type EmailChannel struct {
	// ID of the channel
	id string

	// Enabled indicates whether the channel is enabled
	enabled bool

	// Recipients are the email recipients
	recipients []string

	// SmtpServer is the SMTP server
	smtpServer string

	// SmtpPort is the SMTP port
	smtpPort int

	// SmtpUsername is the SMTP username
	smtpUsername string

	// SmtpPassword is the SMTP password
	smtpPassword string

	// FromAddress is the sender email address
	fromAddress string
}

// NewEmailChannel creates a new email channel
func NewEmailChannel(id, smtpServer string, smtpPort int, smtpUsername, smtpPassword, fromAddress string, recipients []string) *EmailChannel {
	return &EmailChannel{
		id:           id,
		enabled:      true,
		recipients:   recipients,
		smtpServer:   smtpServer,
		smtpPort:     smtpPort,
		smtpUsername: smtpUsername,
		smtpPassword: smtpPassword,
		fromAddress:  fromAddress,
	}
}

// ID returns the channel ID
func (c *EmailChannel) ID() string {
	return c.id
}

// Send sends a notification via email
func (c *EmailChannel) Send(notification *Notification) error {
	if !c.enabled {
		return fmt.Errorf("email channel disabled")
	}

	// In a real implementation, this would send an email
	// For now, just log it
	fmt.Printf("Email notification to %v: %s - %s\n", c.recipients, notification.Title, notification.Message)

	return nil
}

// IsEnabled returns whether the channel is enabled
func (c *EmailChannel) IsEnabled() bool {
	return c.enabled
}

// Type returns the channel type
func (c *EmailChannel) Type() string {
	return "email"
}

// WebhookChannel is a notification channel that sends webhooks
type WebhookChannel struct {
	// ID of the channel
	id string

	// Enabled indicates whether the channel is enabled
	enabled bool

	// URL is the webhook URL
	url string

	// Headers are HTTP headers to include
	headers map[string]string

	// Authentication is the authentication method
	authentication string

	// AuthenticationToken is the authentication token
	authenticationToken string
}

// NewWebhookChannel creates a new webhook channel
func NewWebhookChannel(id, url string, headers map[string]string, authentication, authenticationToken string) *WebhookChannel {
	return &WebhookChannel{
		id:                  id,
		enabled:             true,
		url:                 url,
		headers:             headers,
		authentication:      authentication,
		authenticationToken: authenticationToken,
	}
}

// ID returns the channel ID
func (c *WebhookChannel) ID() string {
	return c.id
}

// Send sends a notification via webhook
func (c *WebhookChannel) Send(notification *Notification) error {
	if !c.enabled {
		return fmt.Errorf("webhook channel disabled")
	}

	// In a real implementation, this would send a webhook
	// For now, just log it
	fmt.Printf("Webhook notification to %s: %s - %s\n", c.url, notification.Title, notification.Message)

	return nil
}

// IsEnabled returns whether the channel is enabled
func (c *WebhookChannel) IsEnabled() bool {
	return c.enabled
}

// Type returns the channel type
func (c *WebhookChannel) Type() string {
	return "webhook"
}

// ConsoleChannel is a notification channel that logs to the console
type ConsoleChannel struct {
	// ID of the channel
	id string

	// Enabled indicates whether the channel is enabled
	enabled bool
}

// NewConsoleChannel creates a new console channel
func NewConsoleChannel(id string) *ConsoleChannel {
	return &ConsoleChannel{
		id:      id,
		enabled: true,
	}
}

// ID returns the channel ID
func (c *ConsoleChannel) ID() string {
	return c.id
}

// Send sends a notification to the console
func (c *ConsoleChannel) Send(notification *Notification) error {
	if !c.enabled {
		return fmt.Errorf("console channel disabled")
	}

	// Log the notification
	fmt.Printf("[%s] %s [%s]: %s\n",
		notification.Type,
		notification.Timestamp.Format(time.RFC3339),
		notification.Severity,
		notification.Title)
	fmt.Printf("    %s\n", notification.Message)

	return nil
}

// IsEnabled returns whether the channel is enabled
func (c *ConsoleChannel) IsEnabled() bool {
	return c.enabled
}

// Type returns the channel type
func (c *ConsoleChannel) Type() string {
	return "console"
}

// NotificationManager manages notification channels and sending notifications
type NotificationManager struct {
	// Channels is a map of channel IDs to channels
	channels      map[string]NotificationChannel
	channelsMutex sync.RWMutex
}

// NewNotificationManager creates a new notification manager
func NewNotificationManager() *NotificationManager {
	manager := &NotificationManager{
		channels: make(map[string]NotificationChannel),
	}

	// Add a default console channel
	manager.RegisterChannel(NewConsoleChannel("default-console"))

	return manager
}

// RegisterChannel registers a notification channel
func (m *NotificationManager) RegisterChannel(channel NotificationChannel) error {
	m.channelsMutex.Lock()
	defer m.channelsMutex.Unlock()

	if _, exists := m.channels[channel.ID()]; exists {
		return fmt.Errorf("channel with ID %s already exists", channel.ID())
	}

	m.channels[channel.ID()] = channel
	return nil
}

// DeregisterChannel deregisters a notification channel
func (m *NotificationManager) DeregisterChannel(channelID string) bool {
	m.channelsMutex.Lock()
	defer m.channelsMutex.Unlock()

	if _, exists := m.channels[channelID]; !exists {
		return false
	}

	delete(m.channels, channelID)
	return true
}

// GetChannel gets a notification channel by ID
func (m *NotificationManager) GetChannel(channelID string) (NotificationChannel, error) {
	m.channelsMutex.RLock()
	defer m.channelsMutex.RUnlock()

	channel, exists := m.channels[channelID]
	if !exists {
		return nil, fmt.Errorf("channel with ID %s not found", channelID)
	}

	return channel, nil
}

// ListChannels lists all notification channels
func (m *NotificationManager) ListChannels() []NotificationChannel {
	m.channelsMutex.RLock()
	defer m.channelsMutex.RUnlock()

	channels := make([]NotificationChannel, 0, len(m.channels))
	for _, channel := range m.channels {
		channels = append(channels, channel)
	}

	return channels
}

// SendNotification sends a notification to a channel
func (m *NotificationManager) SendNotification(channelID string, notification *Notification) error {
	// Get the channel
	channel, err := m.GetChannel(channelID)
	if err != nil {
		return err
	}

	// Check if the channel is enabled
	if !channel.IsEnabled() {
		return fmt.Errorf("channel %s is disabled", channelID)
	}

	// Send the notification
	return channel.Send(notification)
}

// BroadcastNotification sends a notification to all channels
func (m *NotificationManager) BroadcastNotification(notification *Notification) map[string]error {
	// Get all channels
	channels := m.ListChannels()

	// Send to all channels
	errors := make(map[string]error)
	for _, channel := range channels {
		if channel.IsEnabled() {
			if err := channel.Send(notification); err != nil {
				errors[channel.ID()] = err
			}
		}
	}

	return errors
}
