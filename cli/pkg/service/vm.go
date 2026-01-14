package service

import (
	"context"
	"fmt"
	"io"

	"github.com/novacron/cli/pkg/api"
)

// VMService handles VM operations
type VMService struct {
	client *api.Client
}

// NewVMService creates a new VM service
func NewVMService(client *api.Client) *VMService {
	return &VMService{client: client}
}

// List lists all VMs
func (s *VMService) List(ctx context.Context, namespace string) ([]api.VirtualMachine, error) {
	path := "/api/v1/vms"
	if namespace != "" {
		path = fmt.Sprintf("/api/v1/namespaces/%s/vms", namespace)
	}

	var vms []api.VirtualMachine
	if err := s.client.Get(ctx, path, &vms); err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}

	return vms, nil
}

// Get gets a specific VM
func (s *VMService) Get(ctx context.Context, namespace, name string) (*api.VirtualMachine, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s", namespace, name)

	var vm api.VirtualMachine
	if err := s.client.Get(ctx, path, &vm); err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	return &vm, nil
}

// Create creates a new VM
func (s *VMService) Create(ctx context.Context, namespace string, vm *api.VirtualMachine) (*api.VirtualMachine, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms", namespace)

	var created api.VirtualMachine
	if err := s.client.Post(ctx, path, vm, &created); err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}

	return &created, nil
}

// Update updates a VM
func (s *VMService) Update(ctx context.Context, namespace, name string, vm *api.VirtualMachine) (*api.VirtualMachine, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s", namespace, name)

	var updated api.VirtualMachine
	if err := s.client.Put(ctx, path, vm, &updated); err != nil {
		return nil, fmt.Errorf("failed to update VM: %w", err)
	}

	return &updated, nil
}

// Delete deletes a VM
func (s *VMService) Delete(ctx context.Context, namespace, name string) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s", namespace, name)

	if err := s.client.Delete(ctx, path); err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}

	return nil
}

// Start starts a VM
func (s *VMService) Start(ctx context.Context, namespace, name string) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/start", namespace, name)

	if err := s.client.Post(ctx, path, nil, nil); err != nil {
		return fmt.Errorf("failed to start VM: %w", err)
	}

	return nil
}

// Stop stops a VM
func (s *VMService) Stop(ctx context.Context, namespace, name string, force bool) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/stop", namespace, name)
	if force {
		path += "?force=true"
	}

	if err := s.client.Post(ctx, path, nil, nil); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	return nil
}

// Restart restarts a VM
func (s *VMService) Restart(ctx context.Context, namespace, name string) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/restart", namespace, name)

	if err := s.client.Post(ctx, path, nil, nil); err != nil {
		return fmt.Errorf("failed to restart VM: %w", err)
	}

	return nil
}

// Migrate migrates a VM to another node
func (s *VMService) Migrate(ctx context.Context, namespace, name, targetNode string, live bool) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/migrate", namespace, name)

	request := map[string]interface{}{
		"targetNode": targetNode,
		"live":       live,
	}

	if err := s.client.Post(ctx, path, request, nil); err != nil {
		return fmt.Errorf("failed to migrate VM: %w", err)
	}

	return nil
}

// Resize resizes a VM
func (s *VMService) Resize(ctx context.Context, namespace, name string, cpu int, memory, disk string) error {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/resize", namespace, name)

	request := map[string]interface{}{
		"cpu":    cpu,
		"memory": memory,
		"disk":   disk,
	}

	if err := s.client.Post(ctx, path, request, nil); err != nil {
		return fmt.Errorf("failed to resize VM: %w", err)
	}

	return nil
}

// Console gets console access to a VM
func (s *VMService) Console(ctx context.Context, namespace, name string) (*api.WebSocketConn, error) {
	path := fmt.Sprintf("/ws/v1/namespaces/%s/vms/%s/console", namespace, name)

	conn, err := s.client.WebSocket(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to console: %w", err)
	}

	return conn, nil
}

// Exec executes a command in a VM
func (s *VMService) Exec(ctx context.Context, namespace, name string, command []string, stdin io.Reader, stdout, stderr io.Writer) error {
	path := fmt.Sprintf("/ws/v1/namespaces/%s/vms/%s/exec", namespace, name)

	conn, err := s.client.WebSocket(ctx, path)
	if err != nil {
		return fmt.Errorf("failed to connect for exec: %w", err)
	}
	defer conn.Close()

	// Send command
	if err := conn.Send(map[string]interface{}{
		"command": command,
	}); err != nil {
		return fmt.Errorf("failed to send command: %w", err)
	}

	// Handle streams
	// This is a simplified implementation
	// In production, you'd need proper stream multiplexing
	
	return nil
}

// Logs gets logs from a VM
func (s *VMService) Logs(ctx context.Context, namespace, name string, follow bool, tail int) (io.ReadCloser, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/logs", namespace, name)
	if follow {
		path += "?follow=true"
	}
	if tail > 0 {
		path += fmt.Sprintf("&tail=%d", tail)
	}

	resp, err := s.client.Request(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get logs: %w", err)
	}

	return resp.Body, nil
}

// Events gets events for a VM
func (s *VMService) Events(ctx context.Context, namespace, name string) ([]api.Event, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/events", namespace, name)

	var events []api.Event
	if err := s.client.Get(ctx, path, &events); err != nil {
		return nil, fmt.Errorf("failed to get events: %w", err)
	}

	return events, nil
}

// Metrics gets metrics for a VM
func (s *VMService) Metrics(ctx context.Context, namespace, name string) ([]api.Metric, error) {
	path := fmt.Sprintf("/api/v1/namespaces/%s/vms/%s/metrics", namespace, name)

	var metrics []api.Metric
	if err := s.client.Get(ctx, path, &metrics); err != nil {
		return nil, fmt.Errorf("failed to get metrics: %w", err)
	}

	return metrics, nil
}