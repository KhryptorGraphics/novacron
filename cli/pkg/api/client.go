package api

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
	"github.com/novacron/cli/pkg/auth"
)

// Client is the main API client
type Client struct {
	baseURL    string
	wsURL      string
	httpClient *http.Client
	auth       auth.Authenticator
	token      string
}

// NewClient creates a new API client
func NewClient(baseURL string, options ...Option) (*Client, error) {
	// Parse base URL
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	// Determine WebSocket URL
	wsScheme := "ws"
	if u.Scheme == "https" {
		wsScheme = "wss"
	}
	wsURL := fmt.Sprintf("%s://%s", wsScheme, u.Host)

	// Create HTTP client with defaults
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: false,
			},
		},
	}

	client := &Client{
		baseURL:    baseURL,
		wsURL:      wsURL,
		httpClient: httpClient,
	}

	// Apply options
	for _, opt := range options {
		opt(client)
	}

	return client, nil
}

// Option is a client configuration option
type Option func(*Client)

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// WithAuth sets the authenticator
func WithAuth(auth auth.Authenticator) Option {
	return func(c *Client) {
		c.auth = auth
	}
}

// WithInsecure disables TLS verification
func WithInsecure(insecure bool) Option {
	return func(c *Client) {
		if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
			transport.TLSClientConfig.InsecureSkipVerify = insecure
		}
	}
}

// WithTimeout sets the HTTP client timeout
func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) {
		c.httpClient.Timeout = timeout
	}
}

// SetToken sets the authentication token
func (c *Client) SetToken(token string) {
	c.token = token
}

// Request makes an HTTP request
func (c *Client) Request(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	// Build URL
	url := c.baseURL + path

	// Encode body if present
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	// Add authentication
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	} else if c.auth != nil {
		if err := c.auth.Apply(req); err != nil {
			return nil, fmt.Errorf("failed to apply authentication: %w", err)
		}
	}

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	// Check for errors
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		var apiErr ErrorResponse
		if err := json.NewDecoder(resp.Body).Decode(&apiErr); err != nil {
			return nil, fmt.Errorf("request failed with status %d", resp.StatusCode)
		}
		return nil, &apiErr
	}

	return resp, nil
}

// Get makes a GET request
func (c *Client) Get(ctx context.Context, path string, result interface{}) error {
	resp, err := c.Request(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// Post makes a POST request
func (c *Client) Post(ctx context.Context, path string, body, result interface{}) error {
	resp, err := c.Request(ctx, http.MethodPost, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// Put makes a PUT request
func (c *Client) Put(ctx context.Context, path string, body, result interface{}) error {
	resp, err := c.Request(ctx, http.MethodPut, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// Delete makes a DELETE request
func (c *Client) Delete(ctx context.Context, path string) error {
	resp, err := c.Request(ctx, http.MethodDelete, path, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// WebSocket creates a WebSocket connection
func (c *Client) WebSocket(ctx context.Context, path string) (*WebSocketConn, error) {
	// Build WebSocket URL
	url := c.wsURL + path

	// Create dialer
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	// Set TLS config if needed
	if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
		dialer.TLSClientConfig = transport.TLSClientConfig
	}

	// Prepare headers
	headers := http.Header{}
	if c.token != "" {
		headers.Set("Authorization", "Bearer "+c.token)
	}

	// Connect
	conn, _, err := dialer.DialContext(ctx, url, headers)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to WebSocket: %w", err)
	}

	return &WebSocketConn{conn: conn}, nil
}

// WebSocketConn wraps a WebSocket connection
type WebSocketConn struct {
	conn *websocket.Conn
}

// Send sends a message over the WebSocket
func (w *WebSocketConn) Send(msg interface{}) error {
	return w.conn.WriteJSON(msg)
}

// Receive receives a message from the WebSocket
func (w *WebSocketConn) Receive(msg interface{}) error {
	return w.conn.ReadJSON(msg)
}

// Close closes the WebSocket connection
func (w *WebSocketConn) Close() error {
	return w.conn.Close()
}

// ErrorResponse represents an API error
type ErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// Error implements the error interface
func (e *ErrorResponse) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("%s: %s (%s)", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}