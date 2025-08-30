package helpers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

// APIClient provides utilities for testing HTTP APIs
type APIClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	AuthToken  string
}

// NewAPIClient creates a new API client for testing
func NewAPIClient(baseURL, apiKey string) *APIClient {
	return &APIClient{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// SetAuthToken sets the authentication token for requests
func (c *APIClient) SetAuthToken(token string) {
	c.AuthToken = token
}

// GET performs a GET request
func (c *APIClient) GET(t *testing.T, endpoint string) *http.Response {
	t.Helper()
	return c.Request(t, "GET", endpoint, nil)
}

// POST performs a POST request with JSON body
func (c *APIClient) POST(t *testing.T, endpoint string, body interface{}) *http.Response {
	t.Helper()
	return c.Request(t, "POST", endpoint, body)
}

// PUT performs a PUT request with JSON body
func (c *APIClient) PUT(t *testing.T, endpoint string, body interface{}) *http.Response {
	t.Helper()
	return c.Request(t, "PUT", endpoint, body)
}

// DELETE performs a DELETE request
func (c *APIClient) DELETE(t *testing.T, endpoint string) *http.Response {
	t.Helper()
	return c.Request(t, "DELETE", endpoint, nil)
}

// Request performs an HTTP request
func (c *APIClient) Request(t *testing.T, method, endpoint string, body interface{}) *http.Response {
	t.Helper()
	
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		require.NoError(t, err, "Failed to marshal request body")
		reqBody = bytes.NewBuffer(jsonBody)
	}
	
	url := fmt.Sprintf("%s%s", c.BaseURL, endpoint)
	req, err := http.NewRequest(method, url, reqBody)
	require.NoError(t, err, "Failed to create HTTP request")
	
	// Set headers
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	
	if c.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}
	
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
	
	resp, err := c.HTTPClient.Do(req)
	require.NoError(t, err, "Failed to perform HTTP request")
	
	return resp
}

// Login authenticates and sets the auth token
func (c *APIClient) Login(t *testing.T, email, password string) string {
	t.Helper()
	
	loginData := map[string]string{
		"email":    email,
		"password": password,
	}
	
	resp := c.POST(t, "/api/auth/login", loginData)
	defer resp.Body.Close()
	
	require.Equal(t, http.StatusOK, resp.StatusCode, "Login failed")
	
	var loginResp map[string]interface{}
	err := json.NewDecoder(resp.Body).Decode(&loginResp)
	require.NoError(t, err, "Failed to decode login response")
	
	token, ok := loginResp["token"].(string)
	require.True(t, ok, "No token in login response")
	require.NotEmpty(t, token, "Empty token received")
	
	c.SetAuthToken(token)
	return token
}

// ExpectStatus checks that the response has the expected status code
func (c *APIClient) ExpectStatus(t *testing.T, resp *http.Response, expectedStatus int) {
	t.Helper()
	
	if resp.StatusCode != expectedStatus {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("Expected status %d, got %d. Response body: %s", 
			expectedStatus, resp.StatusCode, string(body))
	}
}

// ParseJSON parses JSON response body
func (c *APIClient) ParseJSON(t *testing.T, resp *http.Response, target interface{}) {
	t.Helper()
	
	defer resp.Body.Close()
	err := json.NewDecoder(resp.Body).Decode(target)
	require.NoError(t, err, "Failed to decode JSON response")
}

// WebSocketClient provides utilities for testing WebSocket connections
type WebSocketClient struct {
	URL  string
	Conn *websocket.Conn
}

// NewWebSocketClient creates a new WebSocket client
func NewWebSocketClient(url string) *WebSocketClient {
	return &WebSocketClient{
		URL: url,
	}
}

// Connect establishes a WebSocket connection
func (w *WebSocketClient) Connect(t *testing.T) {
	t.Helper()
	
	dialer := websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second
	
	conn, _, err := dialer.Dial(w.URL, nil)
	require.NoError(t, err, "Failed to connect to WebSocket")
	
	w.Conn = conn
}

// Close closes the WebSocket connection
func (w *WebSocketClient) Close() error {
	if w.Conn != nil {
		return w.Conn.Close()
	}
	return nil
}

// SendMessage sends a message over WebSocket
func (w *WebSocketClient) SendMessage(t *testing.T, message interface{}) {
	t.Helper()
	
	require.NotNil(t, w.Conn, "WebSocket not connected")
	
	err := w.Conn.WriteJSON(message)
	require.NoError(t, err, "Failed to send WebSocket message")
}

// ReadMessage reads a message from WebSocket
func (w *WebSocketClient) ReadMessage(t *testing.T, target interface{}) {
	t.Helper()
	
	require.NotNil(t, w.Conn, "WebSocket not connected")
	
	err := w.Conn.ReadJSON(target)
	require.NoError(t, err, "Failed to read WebSocket message")
}

// WaitForMessage waits for a message with timeout
func (w *WebSocketClient) WaitForMessage(t *testing.T, target interface{}, timeout time.Duration) bool {
	t.Helper()
	
	require.NotNil(t, w.Conn, "WebSocket not connected")
	
	w.Conn.SetReadDeadline(time.Now().Add(timeout))
	defer w.Conn.SetReadDeadline(time.Time{})
	
	err := w.Conn.ReadJSON(target)
	if err != nil {
		if websocket.IsUnexpectedCloseError(err) || websocket.IsCloseError(err) {
			return false
		}
		if netErr, ok := err.(*websocket.CloseError); ok && netErr.Code == websocket.CloseNormalClosure {
			return false
		}
		// Timeout or other error
		return false
	}
	
	return true
}

// GraphQLClient provides utilities for GraphQL API testing
type GraphQLClient struct {
	*APIClient
}

// NewGraphQLClient creates a new GraphQL client
func NewGraphQLClient(baseURL, apiKey string) *GraphQLClient {
	return &GraphQLClient{
		APIClient: NewAPIClient(baseURL, apiKey),
	}
}

// Query executes a GraphQL query
func (g *GraphQLClient) Query(t *testing.T, query string, variables map[string]interface{}) map[string]interface{} {
	t.Helper()
	
	request := map[string]interface{}{
		"query": query,
	}
	
	if variables != nil {
		request["variables"] = variables
	}
	
	resp := g.POST(t, "/graphql", request)
	defer resp.Body.Close()
	
	g.ExpectStatus(t, resp, http.StatusOK)
	
	var result map[string]interface{}
	g.ParseJSON(t, resp, &result)
	
	return result
}

// Mutate executes a GraphQL mutation
func (g *GraphQLClient) Mutate(t *testing.T, mutation string, variables map[string]interface{}) map[string]interface{} {
	t.Helper()
	return g.Query(t, mutation, variables)
}

// WaitForAPI waits for API server to become available
func WaitForAPI(baseURL string, timeout time.Duration) error {
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		resp, err := client.Get(baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		
		time.Sleep(1 * time.Second)
	}
	
	return fmt.Errorf("API not available after %v", timeout)
}

// RateLimitTester tests API rate limiting
type RateLimitTester struct {
	client    *APIClient
	endpoint  string
	rateLimit int
	window    time.Duration
}

// NewRateLimitTester creates a rate limit tester
func NewRateLimitTester(client *APIClient, endpoint string, rateLimit int, window time.Duration) *RateLimitTester {
	return &RateLimitTester{
		client:    client,
		endpoint:  endpoint,
		rateLimit: rateLimit,
		window:    window,
	}
}

// TestRateLimit tests that rate limiting is working correctly
func (r *RateLimitTester) TestRateLimit(t *testing.T) {
	t.Helper()
	
	// Make requests up to the rate limit
	for i := 0; i < r.rateLimit; i++ {
		resp := r.client.GET(t, r.endpoint)
		resp.Body.Close()
		require.True(t, resp.StatusCode < 400, "Request %d should succeed", i+1)
	}
	
	// Next request should be rate limited
	resp := r.client.GET(t, r.endpoint)
	resp.Body.Close()
	require.Equal(t, http.StatusTooManyRequests, resp.StatusCode, "Request should be rate limited")
	
	// Wait for rate limit window to reset
	time.Sleep(r.window + time.Second)
	
	// Should be able to make requests again
	resp = r.client.GET(t, r.endpoint)
	resp.Body.Close()
	require.True(t, resp.StatusCode < 400, "Request should succeed after rate limit reset")
}