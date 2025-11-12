// Package partners provides Partner SDK for third-party integrations with NovaCron DWCP v3
package partners

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// SDK represents the NovaCron Partner SDK
type SDK struct {
	baseURL      string
	partnerID    string
	apiKey       string
	apiSecret    string
	client       *http.Client
	webhookMgr   *WebhookManager
	rateLimiter  *RateLimiter
}

// SDKConfig configures the Partner SDK
type SDKConfig struct {
	BaseURL       string
	PartnerID     string
	APIKey        string
	APISecret     string
	Timeout       time.Duration
	RateLimitRPS  int
}

// Partner represents a partner organization
type Partner struct {
	PartnerID     string                 `json:"partner_id"`
	Name          string                 `json:"name"`
	Type          string                 `json:"type"` // technology, reseller, managed_service, oem
	Tier          string                 `json:"tier"` // bronze, silver, gold, platinum
	Status        string                 `json:"status"`
	Entitlements  []string               `json:"entitlements"`
	QuotaLimits   map[string]int64       `json:"quota_limits"`
	CustomConfig  map[string]interface{} `json:"custom_config"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
}

// Integration represents a partner integration
type Integration struct {
	IntegrationID string                 `json:"integration_id"`
	PartnerID     string                 `json:"partner_id"`
	Name          string                 `json:"name"`
	Type          string                 `json:"type"` // api, webhook, oauth, saml
	Status        string                 `json:"status"`
	Config        map[string]interface{} `json:"config"`
	Endpoints     []Endpoint             `json:"endpoints"`
	CreatedAt     time.Time              `json:"created_at"`
}

// Endpoint represents an integration endpoint
type Endpoint struct {
	Name        string            `json:"name"`
	URL         string            `json:"url"`
	Method      string            `json:"method"`
	Headers     map[string]string `json:"headers"`
	AuthType    string            `json:"auth_type"`
	Timeout     int               `json:"timeout"`
}

// WebhookManager manages webhook subscriptions and delivery
type WebhookManager struct {
	subscriptions map[string]*WebhookSubscription
	mu            sync.RWMutex
	client        *http.Client
}

// WebhookSubscription represents a webhook subscription
type WebhookSubscription struct {
	SubscriptionID string   `json:"subscription_id"`
	PartnerID      string   `json:"partner_id"`
	Events         []string `json:"events"`
	URL            string   `json:"url"`
	Secret         string   `json:"secret"`
	Active         bool     `json:"active"`
	CreatedAt      time.Time `json:"created_at"`
}

// WebhookEvent represents a webhook event
type WebhookEvent struct {
	EventID    string                 `json:"event_id"`
	EventType  string                 `json:"event_type"`
	Timestamp  time.Time              `json:"timestamp"`
	PartnerID  string                 `json:"partner_id"`
	Data       map[string]interface{} `json:"data"`
	Signature  string                 `json:"signature"`
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	rps       int
	tokens    int
	maxTokens int
	mu        sync.Mutex
	lastRefill time.Time
}

// OAuthProvider provides OAuth 2.0 authentication
type OAuthProvider struct {
	clientID     string
	clientSecret string
	redirectURL  string
	scopes       []string
	tokens       map[string]*OAuthToken
	mu           sync.RWMutex
}

// OAuthToken represents an OAuth access token
type OAuthToken struct {
	AccessToken  string    `json:"access_token"`
	TokenType    string    `json:"token_type"`
	ExpiresIn    int       `json:"expires_in"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	Scope        string    `json:"scope,omitempty"`
	IssuedAt     time.Time `json:"issued_at"`
}

// NewSDK creates a new Partner SDK instance
func NewSDK(cfg SDKConfig) (*SDK, error) {
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.RateLimitRPS == 0 {
		cfg.RateLimitRPS = 100
	}

	return &SDK{
		baseURL:     cfg.BaseURL,
		partnerID:   cfg.PartnerID,
		apiKey:      cfg.APIKey,
		apiSecret:   cfg.APISecret,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		webhookMgr:  NewWebhookManager(),
		rateLimiter: NewRateLimiter(cfg.RateLimitRPS),
	}, nil
}

// GetPartner retrieves partner information
func (s *SDK) GetPartner(ctx context.Context) (*Partner, error) {
	endpoint := fmt.Sprintf("%s/api/v1/partners/%s", s.baseURL, s.partnerID)

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	s.setAuthHeaders(req)

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var partner Partner
	if err := json.NewDecoder(resp.Body).Decode(&partner); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &partner, nil
}

// CreateIntegration creates a new partner integration
func (s *SDK) CreateIntegration(ctx context.Context, integration Integration) (*Integration, error) {
	endpoint := fmt.Sprintf("%s/api/v1/partners/%s/integrations", s.baseURL, s.partnerID)

	data, err := json.Marshal(integration)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal integration: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	s.setAuthHeaders(req)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var result Integration
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	fmt.Printf("Created integration: %s\n", result.IntegrationID)
	fmt.Printf("Integration data: %s\n", string(data))

	return &result, nil
}

// SubscribeWebhook subscribes to webhook events
func (s *SDK) SubscribeWebhook(ctx context.Context, events []string, url string, secret string) (*WebhookSubscription, error) {
	subscription := &WebhookSubscription{
		SubscriptionID: generateID(),
		PartnerID:      s.partnerID,
		Events:         events,
		URL:            url,
		Secret:         secret,
		Active:         true,
		CreatedAt:      time.Now(),
	}

	s.webhookMgr.AddSubscription(subscription)

	return subscription, nil
}

// SendWebhookEvent sends a webhook event to subscribers
func (s *SDK) SendWebhookEvent(ctx context.Context, eventType string, data map[string]interface{}) error {
	event := &WebhookEvent{
		EventID:   generateID(),
		EventType: eventType,
		Timestamp: time.Now(),
		PartnerID: s.partnerID,
		Data:      data,
	}

	return s.webhookMgr.DeliverEvent(ctx, event)
}

// CheckQuota checks if partner has quota available
func (s *SDK) CheckQuota(ctx context.Context, resource string, amount int64) (bool, error) {
	partner, err := s.GetPartner(ctx)
	if err != nil {
		return false, err
	}

	if limit, ok := partner.QuotaLimits[resource]; ok {
		// In production, would track actual usage
		return amount <= limit, nil
	}

	return false, fmt.Errorf("quota not defined for resource: %s", resource)
}

// setAuthHeaders sets authentication headers
func (s *SDK) setAuthHeaders(req *http.Request) {
	timestamp := time.Now().Unix()
	signature := s.generateSignature(req.Method, req.URL.Path, timestamp)

	req.Header.Set("X-Partner-ID", s.partnerID)
	req.Header.Set("X-API-Key", s.apiKey)
	req.Header.Set("X-Timestamp", fmt.Sprintf("%d", timestamp))
	req.Header.Set("X-Signature", signature)
}

// generateSignature generates HMAC signature
func (s *SDK) generateSignature(method, path string, timestamp int64) string {
	message := fmt.Sprintf("%s:%s:%d", method, path, timestamp)
	h := hmac.New(sha256.New, []byte(s.apiSecret))
	h.Write([]byte(message))
	return hex.EncodeToString(h.Sum(nil))
}

// NewWebhookManager creates a new webhook manager
func NewWebhookManager() *WebhookManager {
	return &WebhookManager{
		subscriptions: make(map[string]*WebhookSubscription),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// AddSubscription adds a webhook subscription
func (w *WebhookManager) AddSubscription(sub *WebhookSubscription) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.subscriptions[sub.SubscriptionID] = sub
}

// DeliverEvent delivers event to all matching subscriptions
func (w *WebhookManager) DeliverEvent(ctx context.Context, event *WebhookEvent) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	for _, sub := range w.subscriptions {
		if !sub.Active {
			continue
		}

		// Check if subscription is interested in this event
		if !containsEvent(sub.Events, event.EventType) {
			continue
		}

		// Generate signature
		event.Signature = w.signEvent(event, sub.Secret)

		// Deliver event asynchronously
		go w.deliverToEndpoint(ctx, sub.URL, event)
	}

	return nil
}

// deliverToEndpoint delivers event to specific endpoint
func (w *WebhookManager) deliverToEndpoint(ctx context.Context, url string, event *WebhookEvent) {
	data, err := json.Marshal(event)
	if err != nil {
		fmt.Printf("Failed to marshal event: %v\n", err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		fmt.Printf("Failed to create request: %v\n", err)
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Signature", event.Signature)

	resp, err := w.client.Do(req)
	if err != nil {
		fmt.Printf("Failed to deliver webhook to %s: %v\n", url, err)
		return
	}
	defer resp.Body.Close()

	fmt.Printf("Delivered webhook to %s: status=%d, data=%s\n", url, resp.StatusCode, string(data))
}

// signEvent signs an event
func (w *WebhookManager) signEvent(event *WebhookEvent, secret string) string {
	data, _ := json.Marshal(event)
	h := hmac.New(sha256.New, []byte(secret))
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(rps int) *RateLimiter {
	return &RateLimiter{
		rps:        rps,
		tokens:     rps,
		maxTokens:  rps,
		lastRefill: time.Now(),
	}
}

// Allow checks if request is allowed
func (r *RateLimiter) Allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Refill tokens
	now := time.Now()
	elapsed := now.Sub(r.lastRefill)
	tokensToAdd := int(elapsed.Seconds()) * r.rps

	if tokensToAdd > 0 {
		r.tokens = min(r.maxTokens, r.tokens+tokensToAdd)
		r.lastRefill = now
	}

	// Check if we have tokens
	if r.tokens > 0 {
		r.tokens--
		return true
	}

	return false
}

// Helper functions
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func containsEvent(events []string, event string) bool {
	for _, e := range events {
		if e == event {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
