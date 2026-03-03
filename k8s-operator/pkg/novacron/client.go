package novacron

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client represents a NovaCron API client
type Client struct {
	baseURL    string
	httpClient *http.Client
	token      string
}

// ClientConfig holds configuration for the NovaCron client
type ClientConfig struct {
	BaseURL string
	Token   string
	Timeout time.Duration
}

// VM API types (matching backend service types)
type CreateVMRequest struct {
	Name       string            `json:"name"`
	Command    string            `json:"command,omitempty"`
	Args       []string          `json:"args,omitempty"`
	CPUShares  int               `json:"cpu_shares"`
	MemoryMB   int               `json:"memory_mb"`
	DiskSizeGB int               `json:"disk_size_gb"`
	Tags       map[string]string `json:"tags,omitempty"`
	TenantID   string            `json:"tenant_id,omitempty"`
}

type VMResponse struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	State      string                 `json:"state"`
	NodeID     *string                `json:"node_id,omitempty"`
	OwnerID    *int                   `json:"owner_id,omitempty"`
	TenantID   string                 `json:"tenant_id"`
	Config     map[string]interface{} `json:"config,omitempty"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

type VMMetrics struct {
	VMID         string    `json:"vm_id"`
	CPUUsage     float64   `json:"cpu_usage"`
	MemoryUsage  float64   `json:"memory_usage"`
	DiskUsage    float64   `json:"disk_usage,omitempty"`
	NetworkSent  int64     `json:"network_sent"`
	NetworkRecv  int64     `json:"network_recv"`
	IOPS         int       `json:"iops,omitempty"`
	LastUpdated  time.Time `json:"last_updated"`
}

// NewClient creates a new NovaCron API client
func NewClient(config ClientConfig) (*Client, error) {
	if config.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required")
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &Client{
		baseURL: config.BaseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		token: config.Token,
	}, nil
}

// CreateVM creates a new VM
func (c *Client) CreateVM(ctx context.Context, req *CreateVMRequest) (*VMResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var response VMResponse
	if err := c.doRequest(ctx, "POST", "/api/vms", bytes.NewReader(body), &response); err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}

	return &response, nil
}

// GetVM retrieves a VM by ID
func (c *Client) GetVM(ctx context.Context, vmID string) (*VMResponse, error) {
	var response VMResponse
	if err := c.doRequest(ctx, "GET", fmt.Sprintf("/api/vms/%s", vmID), nil, &response); err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	return &response, nil
}

// ListVMs lists all VMs
func (c *Client) ListVMs(ctx context.Context) ([]*VMResponse, error) {
	var response []*VMResponse
	if err := c.doRequest(ctx, "GET", "/api/vms", nil, &response); err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}

	return response, nil
}

// UpdateVM updates a VM
func (c *Client) UpdateVM(ctx context.Context, vmID string, updates map[string]interface{}) (*VMResponse, error) {
	body, err := json.Marshal(updates)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updates: %w", err)
	}

	var response VMResponse
	if err := c.doRequest(ctx, "PUT", fmt.Sprintf("/api/vms/%s", vmID), bytes.NewReader(body), &response); err != nil {
		return nil, fmt.Errorf("failed to update VM: %w", err)
	}

	return &response, nil
}

// DeleteVM deletes a VM
func (c *Client) DeleteVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "DELETE", fmt.Sprintf("/api/vms/%s", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}

	return nil
}

// StartVM starts a VM
func (c *Client) StartVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/vms/%s/start", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to start VM: %w", err)
	}

	return nil
}

// StopVM stops a VM
func (c *Client) StopVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/vms/%s/stop", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	return nil
}

// RestartVM restarts a VM
func (c *Client) RestartVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/vms/%s/restart", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to restart VM: %w", err)
	}

	return nil
}

// PauseVM pauses a VM
func (c *Client) PauseVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/vms/%s/pause", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}

	return nil
}

// ResumeVM resumes a paused VM
func (c *Client) ResumeVM(ctx context.Context, vmID string) error {
	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/vms/%s/resume", vmID), nil, nil); err != nil {
		return fmt.Errorf("failed to resume VM: %w", err)
	}

	return nil
}

// GetVMMetrics retrieves metrics for a VM
func (c *Client) GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error) {
	var response map[string]interface{}
	if err := c.doRequest(ctx, "GET", fmt.Sprintf("/api/vms/%s/metrics", vmID), nil, &response); err != nil {
		return nil, fmt.Errorf("failed to get VM metrics: %w", err)
	}

	// Convert response to VMMetrics
	metrics := &VMMetrics{
		VMID: vmID,
	}

	if cpu, ok := response["cpu_usage"].(float64); ok {
		metrics.CPUUsage = cpu
	}
	if memory, ok := response["memory_usage"].(float64); ok {
		metrics.MemoryUsage = memory
	}
	if disk, ok := response["disk_usage"].(float64); ok {
		metrics.DiskUsage = disk
	}
	if sent, ok := response["network_sent"].(float64); ok {
		metrics.NetworkSent = int64(sent)
	}
	if recv, ok := response["network_recv"].(float64); ok {
		metrics.NetworkRecv = int64(recv)
	}
	if iops, ok := response["iops"].(float64); ok {
		metrics.IOPS = int(iops)
	}
	if updated, ok := response["last_updated"].(string); ok {
		if t, err := time.Parse(time.RFC3339, updated); err == nil {
			metrics.LastUpdated = t
		}
	}

	return metrics, nil
}

// HealthCheck performs a health check against the API
func (c *Client) HealthCheck(ctx context.Context) error {
	if err := c.doRequest(ctx, "GET", "/health", nil, nil); err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	return nil
}

// doRequest performs an HTTP request
func (c *Client) doRequest(ctx context.Context, method, path string, body io.Reader, result interface{}) error {
	url := c.baseURL + path

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	// Perform request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to perform request: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse response if result is provided
	if result != nil {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read response body: %w", err)
		}

		if len(bodyBytes) > 0 {
			if err := json.Unmarshal(bodyBytes, result); err != nil {
				return fmt.Errorf("failed to unmarshal response: %w", err)
			}
		}
	}

	return nil
}