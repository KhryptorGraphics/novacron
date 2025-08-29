package services

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/database"
)

// VMService provides VM management functionality
type VMService struct {
	db        *database.DB
	repos     *database.Repositories
	vmManager *vm.VMManager
	kvmManager *hypervisor.KVMManager
}

// CreateVMRequest represents a request to create a VM
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

// VMResponse represents a VM in API responses
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

// NewVMService creates a new VM service
func NewVMService(db *database.DB, vmManager *vm.VMManager, kvmManager *hypervisor.KVMManager) *VMService {
	return &VMService{
		db:         db,
		repos:      database.NewRepositories(db),
		vmManager:  vmManager,
		kvmManager: kvmManager,
	}
}

// CreateVM creates a new VM
func (s *VMService) CreateVM(ctx context.Context, request CreateVMRequest, userID int) (*VMResponse, error) {
	// Generate VM ID
	vmID := uuid.New().String()
	
	// Set defaults
	if request.TenantID == "" {
		request.TenantID = "default"
	}
	if request.CPUShares == 0 {
		request.CPUShares = 1024
	}
	if request.MemoryMB == 0 {
		request.MemoryMB = 512
	}
	if request.DiskSizeGB == 0 {
		request.DiskSizeGB = 10
	}

	// Create VM config
	vmConfig := vm.VMConfig{
		ID:         vmID,
		Name:       request.Name,
		Command:    request.Command,
		Args:       request.Args,
		CPUShares:  request.CPUShares,
		MemoryMB:   request.MemoryMB,
		DiskSizeGB: request.DiskSizeGB,
		Tags:       request.Tags,
	}

	// If we have VM manager, use it to create the VM
	var newVM *vm.VM
	if s.vmManager != nil {
		createRequest := vm.CreateVMRequest{
			Name: request.Name,
			Spec: vmConfig,
			Tags: request.Tags,
		}
		
		var err error
		newVM, err = s.vmManager.CreateVM(ctx, createRequest)
		if err != nil {
			return nil, fmt.Errorf("failed to create VM with manager: %w", err)
		}
	} else {
		// Create VM directly if no manager available
		var err error
		newVM, err = vm.NewVM(vmConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create VM: %w", err)
		}
	}

	// Store in database
	dbVM := &database.VM{
		ID:       vmID,
		Name:     request.Name,
		State:    string(newVM.State()),
		NodeID:   nil, // Will be set when VM is started
		OwnerID:  &userID,
		TenantID: request.TenantID,
		Config: database.JSONB{
			"command":      request.Command,
			"args":         request.Args,
			"cpu_shares":   request.CPUShares,
			"memory_mb":    request.MemoryMB,
			"disk_size_gb": request.DiskSizeGB,
			"tags":         request.Tags,
		},
	}

	if err := s.repos.VMs.Create(ctx, dbVM); err != nil {
		// If DB creation fails, clean up VM if it was created
		if s.vmManager != nil {
			s.vmManager.DeleteVM(ctx, vmID)
		}
		return nil, fmt.Errorf("failed to store VM in database: %w", err)
	}

	log.Printf("Created VM %s (%s) for user %d", request.Name, vmID, userID)

	return s.convertToVMResponse(dbVM), nil
}

// GetVM retrieves a VM by ID
func (s *VMService) GetVM(ctx context.Context, vmID string) (*VMResponse, error) {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return nil, fmt.Errorf("VM not found")
	}

	return s.convertToVMResponse(dbVM), nil
}

// ListVMs lists VMs with optional filtering
func (s *VMService) ListVMs(ctx context.Context, userID int, tenantID string) ([]*VMResponse, error) {
	filters := map[string]interface{}{
		"tenant_id": tenantID,
	}

	// Non-admin users can only see their own VMs
	// In a real system, you'd check user roles here
	if userID != 0 {
		filters["owner_id"] = userID
	}

	dbVMs, err := s.repos.VMs.List(ctx, filters)
	if err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}

	var vmResponses []*VMResponse
	for _, dbVM := range dbVMs {
		vmResponses = append(vmResponses, s.convertToVMResponse(dbVM))
	}

	return vmResponses, nil
}

// UpdateVM updates a VM
func (s *VMService) UpdateVM(ctx context.Context, vmID string, updates map[string]interface{}) (*VMResponse, error) {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return nil, fmt.Errorf("VM not found")
	}

	// Update fields
	if name, ok := updates["name"]; ok {
		if nameStr, ok := name.(string); ok {
			dbVM.Name = nameStr
		}
	}

	// Update config if provided
	if config := dbVM.Config; config != nil {
		if cpuShares, ok := updates["cpu_shares"]; ok {
			config["cpu_shares"] = cpuShares
		}
		if memoryMB, ok := updates["memory_mb"]; ok {
			config["memory_mb"] = memoryMB
		}
		if diskSizeGB, ok := updates["disk_size_gb"]; ok {
			config["disk_size_gb"] = diskSizeGB
		}
		if tags, ok := updates["tags"]; ok {
			config["tags"] = tags
		}
		dbVM.Config = config
	}

	// Update in database
	if err := s.repos.VMs.Update(ctx, dbVM); err != nil {
		return nil, fmt.Errorf("failed to update VM: %w", err)
	}

	// Update in VM manager if available
	if s.vmManager != nil {
		if managedVM, err := s.vmManager.GetVM(vmID); err == nil {
			if name, ok := updates["name"]; ok {
				if nameStr, ok := name.(string); ok {
					managedVM.SetName(nameStr)
				}
			}
			if cpuShares, ok := updates["cpu_shares"]; ok {
				if cpuSharesInt, ok := cpuShares.(int); ok {
					managedVM.SetCPUShares(cpuSharesInt)
				}
			}
			if memoryMB, ok := updates["memory_mb"]; ok {
				if memoryMBInt, ok := memoryMB.(int); ok {
					managedVM.SetMemoryMB(memoryMBInt)
				}
			}
		}
	}

	log.Printf("Updated VM %s", vmID)
	return s.convertToVMResponse(dbVM), nil
}

// DeleteVM deletes a VM
func (s *VMService) DeleteVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	// Stop VM if running
	if dbVM.State == "running" {
		if err := s.StopVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to stop VM before deletion: %w", err)
		}
	}

	// Delete from VM manager if available
	if s.vmManager != nil {
		if err := s.vmManager.DeleteVM(ctx, vmID); err != nil {
			log.Printf("Warning: failed to delete VM from manager: %v", err)
		}
	}

	// Delete from database
	if err := s.repos.VMs.Delete(ctx, vmID); err != nil {
		return fmt.Errorf("failed to delete VM from database: %w", err)
	}

	log.Printf("Deleted VM %s", vmID)
	return nil
}

// StartVM starts a VM
func (s *VMService) StartVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	if dbVM.State == "running" {
		return fmt.Errorf("VM is already running")
	}

	// Start VM using manager if available
	if s.vmManager != nil {
		if err := s.vmManager.StartVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to start VM: %w", err)
		}
	}

	// Update state in database
	dbVM.State = "running"
	nodeID := "default-node-01"
	dbVM.NodeID = &nodeID
	
	if err := s.repos.VMs.Update(ctx, dbVM); err != nil {
		return fmt.Errorf("failed to update VM state: %w", err)
	}

	log.Printf("Started VM %s", vmID)
	return nil
}

// StopVM stops a VM
func (s *VMService) StopVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	if dbVM.State == "stopped" {
		return fmt.Errorf("VM is already stopped")
	}

	// Stop VM using manager if available
	if s.vmManager != nil {
		if err := s.vmManager.StopVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to stop VM: %w", err)
		}
	}

	// Update state in database
	dbVM.State = "stopped"
	dbVM.NodeID = nil
	
	if err := s.repos.VMs.Update(ctx, dbVM); err != nil {
		return fmt.Errorf("failed to update VM state: %w", err)
	}

	log.Printf("Stopped VM %s", vmID)
	return nil
}

// RestartVM restarts a VM
func (s *VMService) RestartVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	// Restart VM using manager if available
	if s.vmManager != nil {
		if err := s.vmManager.RestartVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to restart VM: %w", err)
		}
	} else {
		// Fallback: stop then start
		if dbVM.State == "running" {
			if err := s.StopVM(ctx, vmID); err != nil {
				return fmt.Errorf("failed to stop VM during restart: %w", err)
			}
		}
		if err := s.StartVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to start VM during restart: %w", err)
		}
	}

	log.Printf("Restarted VM %s", vmID)
	return nil
}

// PauseVM pauses a VM
func (s *VMService) PauseVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	if dbVM.State != "running" {
		return fmt.Errorf("VM must be running to pause")
	}

	// Pause VM using manager if available
	if s.vmManager != nil {
		if err := s.vmManager.PauseVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to pause VM: %w", err)
		}
	}

	// Update state in database
	dbVM.State = "paused"
	
	if err := s.repos.VMs.Update(ctx, dbVM); err != nil {
		return fmt.Errorf("failed to update VM state: %w", err)
	}

	log.Printf("Paused VM %s", vmID)
	return nil
}

// ResumeVM resumes a paused VM
func (s *VMService) ResumeVM(ctx context.Context, vmID string) error {
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return fmt.Errorf("VM not found")
	}

	if dbVM.State != "paused" {
		return fmt.Errorf("VM must be paused to resume")
	}

	// Resume VM using manager if available
	if s.vmManager != nil {
		if err := s.vmManager.ResumeVM(ctx, vmID); err != nil {
			return fmt.Errorf("failed to resume VM: %w", err)
		}
	}

	// Update state in database
	dbVM.State = "running"
	
	if err := s.repos.VMs.Update(ctx, dbVM); err != nil {
		return fmt.Errorf("failed to update VM state: %w", err)
	}

	log.Printf("Resumed VM %s", vmID)
	return nil
}

// GetVMMetrics gets metrics for a specific VM
func (s *VMService) GetVMMetrics(ctx context.Context, vmID string) (map[string]interface{}, error) {
	// Check if VM exists
	dbVM, err := s.repos.VMs.GetByID(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	if dbVM == nil {
		return nil, fmt.Errorf("VM not found")
	}

	// Get metrics from VM manager if available
	if s.vmManager != nil {
		if managedVM, err := s.vmManager.GetVM(vmID); err == nil {
			stats := managedVM.GetStats()
			return map[string]interface{}{
				"vm_id":        vmID,
				"cpu_usage":    stats.CPUUsage,
				"memory_usage": stats.MemoryUsage,
				"network_sent": stats.NetworkSent,
				"network_recv": stats.NetworkRecv,
				"last_updated": stats.LastUpdated,
			}, nil
		}
	}

	// Get metrics from database
	end := time.Now()
	start := end.Add(-5 * time.Minute)
	metrics, err := s.repos.Metrics.GetVMMetrics(ctx, vmID, start, end)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM metrics: %w", err)
	}

	if len(metrics) > 0 {
		latest := metrics[0]
		return map[string]interface{}{
			"vm_id":        vmID,
			"cpu_usage":    latest.CPUUsage,
			"memory_usage": latest.MemoryUsage,
			"disk_usage":   latest.DiskUsage,
			"network_sent": latest.NetworkSent,
			"network_recv": latest.NetworkRecv,
			"iops":         latest.IOPS,
			"last_updated": latest.Timestamp,
		}, nil
	}

	// Return empty metrics if no data available
	return map[string]interface{}{
		"vm_id":        vmID,
		"cpu_usage":    0.0,
		"memory_usage": 0.0,
		"disk_usage":   0.0,
		"network_sent": int64(0),
		"network_recv": int64(0),
		"iops":         0,
		"last_updated": time.Now(),
	}, nil
}

// Helper method to convert database VM to response format
func (s *VMService) convertToVMResponse(dbVM *database.VM) *VMResponse {
	config := make(map[string]interface{})
	
	// Convert JSONB to regular map
	if dbVM.Config != nil {
		for k, v := range dbVM.Config {
			config[k] = v
		}
	}

	return &VMResponse{
		ID:        dbVM.ID,
		Name:      dbVM.Name,
		State:     dbVM.State,
		NodeID:    dbVM.NodeID,
		OwnerID:   dbVM.OwnerID,
		TenantID:  dbVM.TenantID,
		Config:    config,
		CreatedAt: dbVM.CreatedAt,
		UpdatedAt: dbVM.UpdatedAt,
	}
}