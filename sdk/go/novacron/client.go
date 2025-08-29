// Package novacron provides a Go SDK for the NovaCron VM management platform
package novacron

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// Client represents the NovaCron API client
type Client struct {
	baseURL    string
	httpClient *http.Client
	apiToken   string
	userAgent  string
}

// ClientConfig holds configuration for the NovaCron client
type ClientConfig struct {
	BaseURL   string
	APIToken  string
	Username  string
	Password  string
	Timeout   time.Duration
	UserAgent string
}

// NewClient creates a new NovaCron client
func NewClient(config ClientConfig) (*Client, error) {
	if config.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required")
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	userAgent := config.UserAgent
	if userAgent == "" {
		userAgent = "NovaCron-Go-SDK/1.0.0"
	}

	client := &Client{
		baseURL: config.BaseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		apiToken:  config.APIToken,
		userAgent: userAgent,
	}

	return client, nil
}

// SetAPIToken sets the API token for authentication
func (c *Client) SetAPIToken(token string) {
	c.apiToken = token
}

// request performs an HTTP request
func (c *Client) request(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
	}

	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)

	if c.apiToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiToken)
	}

	// Perform request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse response
	if result != nil {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read response: %w", err)
		}

		if len(bodyBytes) > 0 {
			if err := json.Unmarshal(bodyBytes, result); err != nil {
				return fmt.Errorf("failed to parse response: %w", err)
			}
		}
	}

	return nil
}

// VM Management Methods

// CreateVM creates a new VM
func (c *Client) CreateVM(ctx context.Context, req *CreateVMRequest) (*VM, error) {
	var vm VM
	if err := c.request(ctx, "POST", "/api/vms", req, &vm); err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}
	return &vm, nil
}

// GetVM retrieves a VM by ID
func (c *Client) GetVM(ctx context.Context, vmID string) (*VM, error) {
	var vm VM
	if err := c.request(ctx, "GET", fmt.Sprintf("/api/vms/%s", vmID), nil, &vm); err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}
	return &vm, nil
}

// ListVMs lists VMs with optional filtering
func (c *Client) ListVMs(ctx context.Context, opts *ListVMOptions) ([]*VM, error) {
	path := "/api/vms"
	if opts != nil {
		params := url.Values{}
		if opts.TenantID != "" {
			params.Set("tenant_id", opts.TenantID)
		}
		if opts.State != "" {
			params.Set("state", opts.State)
		}
		if opts.NodeID != "" {
			params.Set("node_id", opts.NodeID)
		}
		if len(params) > 0 {
			path += "?" + params.Encode()
		}
	}

	var vms []*VM
	if err := c.request(ctx, "GET", path, nil, &vms); err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}
	return vms, nil
}

// UpdateVM updates VM configuration
func (c *Client) UpdateVM(ctx context.Context, vmID string, updates *UpdateVMRequest) (*VM, error) {
	var vm VM
	if err := c.request(ctx, "PUT", fmt.Sprintf("/api/vms/%s", vmID), updates, &vm); err != nil {
		return nil, fmt.Errorf("failed to update VM: %w", err)
	}
	return &vm, nil
}

// DeleteVM deletes a VM
func (c *Client) DeleteVM(ctx context.Context, vmID string) error {
	if err := c.request(ctx, "DELETE", fmt.Sprintf("/api/vms/%s", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}
	return nil
}

// VM Lifecycle Methods

// StartVM starts a VM
func (c *Client) StartVM(ctx context.Context, vmID string) error {
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/start", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to start VM: %w", err)
	}
	return nil
}

// StopVM stops a VM
func (c *Client) StopVM(ctx context.Context, vmID string, force bool) error {
	body := map[string]bool{"force": force}
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/stop", vmID), body, nil); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}
	return nil
}

// RestartVM restarts a VM
func (c *Client) RestartVM(ctx context.Context, vmID string) error {
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/restart", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to restart VM: %w", err)
	}
	return nil
}

// PauseVM pauses a VM
func (c *Client) PauseVM(ctx context.Context, vmID string) error {
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/pause", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}
	return nil
}

// ResumeVM resumes a paused VM
func (c *Client) ResumeVM(ctx context.Context, vmID string) error {
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/resume", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to resume VM: %w", err)
	}
	return nil
}

// Metrics and Monitoring

// GetVMMetrics retrieves VM metrics
func (c *Client) GetVMMetrics(ctx context.Context, vmID string, opts *MetricsOptions) (*VMMetrics, error) {
	path := fmt.Sprintf("/api/vms/%s/metrics", vmID)
	if opts != nil {
		params := url.Values{}
		if !opts.StartTime.IsZero() {
			params.Set("start", opts.StartTime.Format(time.RFC3339))
		}
		if !opts.EndTime.IsZero() {
			params.Set("end", opts.EndTime.Format(time.RFC3339))
		}
		if len(params) > 0 {
			path += "?" + params.Encode()
		}
	}

	var metrics VMMetrics
	if err := c.request(ctx, "GET", path, nil, &metrics); err != nil {
		return nil, fmt.Errorf("failed to get VM metrics: %w", err)
	}
	return &metrics, nil
}

// GetSystemMetrics retrieves system-wide metrics
func (c *Client) GetSystemMetrics(ctx context.Context, opts *SystemMetricsOptions) (*SystemMetrics, error) {
	path := "/api/metrics/system"
	if opts != nil {
		params := url.Values{}
		if opts.NodeID != "" {
			params.Set("node_id", opts.NodeID)
		}
		if !opts.StartTime.IsZero() {
			params.Set("start", opts.StartTime.Format(time.RFC3339))
		}
		if !opts.EndTime.IsZero() {
			params.Set("end", opts.EndTime.Format(time.RFC3339))
		}
		if len(params) > 0 {
			path += "?" + params.Encode()
		}
	}

	var metrics SystemMetrics
	if err := c.request(ctx, "GET", path, nil, &metrics); err != nil {
		return nil, fmt.Errorf("failed to get system metrics: %w", err)
	}
	return &metrics, nil
}

// Migration Methods

// MigrateVM migrates a VM to another node
func (c *Client) MigrateVM(ctx context.Context, vmID string, req *MigrationRequest) (*Migration, error) {
	var migration Migration
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/migrate", vmID), req, &migration); err != nil {
		return nil, fmt.Errorf("failed to migrate VM: %w", err)
	}
	return &migration, nil
}

// GetMigration retrieves migration status
func (c *Client) GetMigration(ctx context.Context, migrationID string) (*Migration, error) {
	var migration Migration
	if err := c.request(ctx, "GET", fmt.Sprintf("/api/migrations/%s", migrationID), nil, &migration); err != nil {
		return nil, fmt.Errorf("failed to get migration: %w", err)
	}
	return &migration, nil
}

// ListMigrations lists migrations with optional filtering
func (c *Client) ListMigrations(ctx context.Context, opts *ListMigrationOptions) ([]*Migration, error) {
	path := "/api/migrations"
	if opts != nil {
		params := url.Values{}
		if opts.VMID != "" {
			params.Set("vm_id", opts.VMID)
		}
		if opts.Status != "" {
			params.Set("status", opts.Status)
		}
		if len(params) > 0 {
			path += "?" + params.Encode()
		}
	}

	var migrations []*Migration
	if err := c.request(ctx, "GET", path, nil, &migrations); err != nil {
		return nil, fmt.Errorf("failed to list migrations: %w", err)
	}
	return migrations, nil
}

// CancelMigration cancels an ongoing migration
func (c *Client) CancelMigration(ctx context.Context, migrationID string) error {
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/migrations/%s/cancel", migrationID), nil, nil); err != nil {
		return fmt.Errorf("failed to cancel migration: %w", err)
	}
	return nil
}

// Template Methods

// CreateVMTemplate creates a VM template
func (c *Client) CreateVMTemplate(ctx context.Context, template *VMTemplate) (*VMTemplate, error) {
	var result VMTemplate
	if err := c.request(ctx, "POST", "/api/templates", template, &result); err != nil {
		return nil, fmt.Errorf("failed to create template: %w", err)
	}
	return &result, nil
}

// GetVMTemplate retrieves a VM template
func (c *Client) GetVMTemplate(ctx context.Context, templateID string) (*VMTemplate, error) {
	var template VMTemplate
	if err := c.request(ctx, "GET", fmt.Sprintf("/api/templates/%s", templateID), nil, &template); err != nil {
		return nil, fmt.Errorf("failed to get template: %w", err)
	}
	return &template, nil
}

// ListVMTemplates lists VM templates
func (c *Client) ListVMTemplates(ctx context.Context) ([]*VMTemplate, error) {
	var templates []*VMTemplate
	if err := c.request(ctx, "GET", "/api/templates", nil, &templates); err != nil {
		return nil, fmt.Errorf("failed to list templates: %w", err)
	}
	return templates, nil
}

// UpdateVMTemplate updates a VM template
func (c *Client) UpdateVMTemplate(ctx context.Context, templateID string, template *VMTemplate) (*VMTemplate, error) {
	var result VMTemplate
	if err := c.request(ctx, "PUT", fmt.Sprintf("/api/templates/%s", templateID), template, &result); err != nil {
		return nil, fmt.Errorf("failed to update template: %w", err)
	}
	return &result, nil
}

// DeleteVMTemplate deletes a VM template
func (c *Client) DeleteVMTemplate(ctx context.Context, templateID string) error {
	if err := c.request(ctx, "DELETE", fmt.Sprintf("/api/templates/%s", templateID), nil, nil); err != nil {
		return fmt.Errorf("failed to delete template: %w", err)
	}
	return nil
}

// Node Management

// ListNodes lists cluster nodes
func (c *Client) ListNodes(ctx context.Context) ([]*Node, error) {
	var nodes []*Node
	if err := c.request(ctx, "GET", "/api/nodes", nil, &nodes); err != nil {
		return nil, fmt.Errorf("failed to list nodes: %w", err)
	}
	return nodes, nil
}

// GetNode retrieves node information
func (c *Client) GetNode(ctx context.Context, nodeID string) (*Node, error) {
	var node Node
	if err := c.request(ctx, "GET", fmt.Sprintf("/api/nodes/%s", nodeID), nil, &node); err != nil {
		return nil, fmt.Errorf("failed to get node: %w", err)
	}
	return &node, nil
}

// GetNodeMetrics retrieves node metrics
func (c *Client) GetNodeMetrics(ctx context.Context, nodeID string, opts *MetricsOptions) (*NodeMetrics, error) {
	path := fmt.Sprintf("/api/nodes/%s/metrics", nodeID)
	if opts != nil {
		params := url.Values{}
		if !opts.StartTime.IsZero() {
			params.Set("start", opts.StartTime.Format(time.RFC3339))
		}
		if !opts.EndTime.IsZero() {
			params.Set("end", opts.EndTime.Format(time.RFC3339))
		}
		if len(params) > 0 {
			path += "?" + params.Encode()
		}
	}

	var metrics NodeMetrics
	if err := c.request(ctx, "GET", path, nil, &metrics); err != nil {
		return nil, fmt.Errorf("failed to get node metrics: %w", err)
	}
	return &metrics, nil
}

// Health and Status

// HealthCheck performs a health check
func (c *Client) HealthCheck(ctx context.Context) (*HealthStatus, error) {
	var status HealthStatus
	if err := c.request(ctx, "GET", "/health", nil, &status); err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	return &status, nil
}

// GetVersion retrieves API version information
func (c *Client) GetVersion(ctx context.Context) (*Version, error) {
	var version Version
	if err := c.request(ctx, "GET", "/version", nil, &version); err != nil {
		return nil, fmt.Errorf("failed to get version: %w", err)
	}
	return &version, nil
}

// Authentication

// Authenticate performs authentication and returns JWT token
func (c *Client) Authenticate(ctx context.Context, username, password string) (string, error) {
	authReq := map[string]string{
		"username": username,
		"password": password,
	}

	var authResp map[string]interface{}
	if err := c.request(ctx, "POST", "/api/auth/login", authReq, &authResp); err != nil {
		return "", fmt.Errorf("authentication failed: %w", err)
	}

	token, ok := authResp["token"].(string)
	if !ok {
		return "", fmt.Errorf("invalid token in response")
	}

	c.apiToken = token
	return token, nil
}

// RefreshToken refreshes the JWT token
func (c *Client) RefreshToken(ctx context.Context) (string, error) {
	var authResp map[string]interface{}
	if err := c.request(ctx, "POST", "/api/auth/refresh", nil, &authResp); err != nil {
		return "", fmt.Errorf("token refresh failed: %w", err)
	}

	token, ok := authResp["token"].(string)
	if !ok {
		return "", fmt.Errorf("invalid token in response")
	}

	c.apiToken = token
	return token, nil
}