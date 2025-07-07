package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ClusterRole represents a role in a VM cluster
type ClusterRole string

const (
	// ClusterRoleMaster represents a master node in a cluster
	ClusterRoleMaster ClusterRole = "master"
	
	// ClusterRoleWorker represents a worker node in a cluster
	ClusterRoleWorker ClusterRole = "worker"
	
	// ClusterRoleStorage represents a storage node in a cluster
	ClusterRoleStorage ClusterRole = "storage"
)

// ClusterState represents the state of a VM cluster
type ClusterState string

const (
	// ClusterStateCreating indicates the cluster is being created
	ClusterStateCreating ClusterState = "creating"
	
	// ClusterStateRunning indicates the cluster is running
	ClusterStateRunning ClusterState = "running"
	
	// ClusterStateUpdating indicates the cluster is being updated
	ClusterStateUpdating ClusterState = "updating"
	
	// ClusterStateDegraded indicates the cluster is in a degraded state
	ClusterStateDegraded ClusterState = "degraded"
	
	// ClusterStateStopped indicates the cluster is stopped
	ClusterStateStopped ClusterState = "stopped"
	
	// ClusterStateDeleting indicates the cluster is being deleted
	ClusterStateDeleting ClusterState = "deleting"
)

// VMClusterMember represents a member of a VM cluster
type VMClusterMember struct {
	VMID      string      `json:"vm_id"`
	Role      ClusterRole `json:"role"`
	JoinedAt  time.Time   `json:"joined_at"`
	Status    State       `json:"status"`
	NodeID    string      `json:"node_id"`
	IPAddress string      `json:"ip_address,omitempty"`
	Hostname  string      `json:"hostname,omitempty"`
}

// VMCluster represents a cluster of VMs
type VMCluster struct {
	ID          string       `json:"id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	State       ClusterState `json:"state"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	Members     []*VMClusterMember `json:"members"`
	Tags        []string     `json:"tags,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// VMClusterManager manages VM clusters
type VMClusterManager struct {
	clusters     map[string]*VMCluster
	clustersMutex sync.RWMutex
	vmManager    *VMManager
	scheduler    *VMScheduler
}

// NewVMClusterManager creates a new VM cluster manager
func NewVMClusterManager(vmManager *VMManager, scheduler *VMScheduler) *VMClusterManager {
	return &VMClusterManager{
		clusters:  make(map[string]*VMCluster),
		vmManager: vmManager,
		scheduler: scheduler,
	}
}

// CreateCluster creates a new VM cluster
func (m *VMClusterManager) CreateCluster(ctx context.Context, name, description string, masterCount, workerCount int, vmConfig VMConfig, tags []string, metadata map[string]string) (*VMCluster, error) {
	// Validate counts
	if masterCount <= 0 {
		return nil, fmt.Errorf("master count must be greater than 0")
	}
	
	if workerCount <= 0 {
		return nil, fmt.Errorf("worker count must be greater than 0")
	}
	
	// Generate cluster ID
	clusterID := uuid.New().String()
	
	// Create cluster
	cluster := &VMCluster{
		ID:          clusterID,
		Name:        name,
		Description: description,
		State:       ClusterStateCreating,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Members:     make([]*VMClusterMember, 0),
		Tags:        tags,
		Metadata:    metadata,
	}
	
	// Store cluster
	m.clustersMutex.Lock()
	m.clusters[clusterID] = cluster
	m.clustersMutex.Unlock()
	
	// Create cluster members in a goroutine
	go func() {
		err := m.createClusterMembers(ctx, cluster, masterCount, workerCount, vmConfig)
		if err != nil {
			log.Printf("Error creating cluster members: %v", err)
			
			m.clustersMutex.Lock()
			cluster.State = ClusterStateDegraded
			cluster.UpdatedAt = time.Now()
			cluster.Metadata["error"] = err.Error()
			m.clustersMutex.Unlock()
			
			return
		}
		
		m.clustersMutex.Lock()
		cluster.State = ClusterStateRunning
		cluster.UpdatedAt = time.Now()
		m.clustersMutex.Unlock()
		
		log.Printf("Created cluster %s (%s) with %d masters and %d workers", cluster.Name, cluster.ID, masterCount, workerCount)
	}()
	
	return cluster, nil
}

// GetCluster returns a cluster by ID
func (m *VMClusterManager) GetCluster(clusterID string) (*VMCluster, error) {
	m.clustersMutex.RLock()
	defer m.clustersMutex.RUnlock()
	
	cluster, exists := m.clusters[clusterID]
	if !exists {
		return nil, fmt.Errorf("cluster %s not found", clusterID)
	}
	
	return cluster, nil
}

// ListClusters returns all clusters
func (m *VMClusterManager) ListClusters() []*VMCluster {
	m.clustersMutex.RLock()
	defer m.clustersMutex.RUnlock()
	
	clusters := make([]*VMCluster, 0, len(m.clusters))
	for _, cluster := range m.clusters {
		clusters = append(clusters, cluster)
	}
	
	return clusters
}

// DeleteCluster deletes a cluster
func (m *VMClusterManager) DeleteCluster(ctx context.Context, clusterID string) error {
	// Get the cluster
	m.clustersMutex.Lock()
	cluster, exists := m.clusters[clusterID]
	if !exists {
		m.clustersMutex.Unlock()
		return fmt.Errorf("cluster %s not found", clusterID)
	}
	
	// Update cluster state
	cluster.State = ClusterStateDeleting
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	// Delete cluster members
	for _, member := range cluster.Members {
		// Get the VM first
		vm, err := m.vmManager.GetVM(member.VMID)
		if err != nil {
			log.Printf("Warning: Failed to get cluster member VM %s: %v", member.VMID, err)
			continue
		}
		
		// Get driver for the VM
		driver, err := m.vmManager.getDriver(vm.config)
		if err != nil {
			log.Printf("Warning: Failed to get driver for VM %s: %v", member.VMID, err)
			continue
		}
		
		// Delete the VM
		_, err = m.vmManager.deleteVM(context.Background(), vm, driver)
		if err != nil {
			log.Printf("Warning: Failed to delete cluster member VM %s: %v", member.VMID, err)
		}
	}
	
	// Remove cluster
	m.clustersMutex.Lock()
	delete(m.clusters, clusterID)
	m.clustersMutex.Unlock()
	
	log.Printf("Deleted cluster %s (%s)", cluster.Name, cluster.ID)
	
	return nil
}

// AddClusterMember adds a member to a cluster
func (m *VMClusterManager) AddClusterMember(ctx context.Context, clusterID string, role ClusterRole, vmConfig VMConfig) (*VMClusterMember, error) {
	// Get the cluster
	m.clustersMutex.Lock()
	cluster, exists := m.clusters[clusterID]
	if !exists {
		m.clustersMutex.Unlock()
		return nil, fmt.Errorf("cluster %s not found", clusterID)
	}
	
	// Update cluster state
	cluster.State = ClusterStateUpdating
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	// Create VM
	vm, err := m.createClusterVM(ctx, cluster, role, vmConfig)
	if err != nil {
		m.clustersMutex.Lock()
		cluster.State = ClusterStateDegraded
		cluster.UpdatedAt = time.Now()
		cluster.Metadata["error"] = err.Error()
		m.clustersMutex.Unlock()
		
		return nil, fmt.Errorf("failed to create cluster member VM: %w", err)
	}
	
	// Create member
	member := &VMClusterMember{
		VMID:     vm.ID(),
		Role:     role,
		JoinedAt: time.Now(),
		Status:   vm.State(),
		NodeID:   vm.GetNodeID(),
	}
	
	// Add member to cluster
	m.clustersMutex.Lock()
	cluster.Members = append(cluster.Members, member)
	cluster.State = ClusterStateRunning
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	log.Printf("Added %s member %s to cluster %s", role, vm.ID(), cluster.ID)
	
	return member, nil
}

// RemoveClusterMember removes a member from a cluster
func (m *VMClusterManager) RemoveClusterMember(ctx context.Context, clusterID, vmID string) error {
	// Get the cluster
	m.clustersMutex.Lock()
	cluster, exists := m.clusters[clusterID]
	if !exists {
		m.clustersMutex.Unlock()
		return fmt.Errorf("cluster %s not found", clusterID)
	}
	
	// Find the member
	var memberIndex int = -1
	for i, member := range cluster.Members {
		if member.VMID == vmID {
			memberIndex = i
			break
		}
	}
	
	if memberIndex == -1 {
		m.clustersMutex.Unlock()
		return fmt.Errorf("member VM %s not found in cluster %s", vmID, clusterID)
	}
	
	// Update cluster state
	cluster.State = ClusterStateUpdating
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	// Get the VM first
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		m.clustersMutex.Lock()
		cluster.State = ClusterStateDegraded
		cluster.UpdatedAt = time.Now()
		cluster.Metadata["error"] = err.Error()
		m.clustersMutex.Unlock()
		return fmt.Errorf("failed to get VM %s: %w", vmID, err)
	}
	
	// Get driver for the VM
	driver, err := m.vmManager.getDriver(vm.config)
	if err != nil {
		m.clustersMutex.Lock()
		cluster.State = ClusterStateDegraded
		cluster.UpdatedAt = time.Now()
		cluster.Metadata["error"] = err.Error()
		m.clustersMutex.Unlock()
		return fmt.Errorf("failed to get driver for VM %s: %w", vmID, err)
	}
	
	// Delete the VM
	_, err = m.vmManager.deleteVM(ctx, vm, driver)
	if err != nil {
		m.clustersMutex.Lock()
		cluster.State = ClusterStateDegraded
		cluster.UpdatedAt = time.Now()
		cluster.Metadata["error"] = err.Error()
		m.clustersMutex.Unlock()
		
		return fmt.Errorf("failed to delete cluster member VM: %w", err)
	}
	
	// Remove member from cluster
	m.clustersMutex.Lock()
	cluster.Members = append(cluster.Members[:memberIndex], cluster.Members[memberIndex+1:]...)
	cluster.State = ClusterStateRunning
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	log.Printf("Removed member %s from cluster %s", vmID, cluster.ID)
	
	return nil
}

// StartCluster starts all VMs in a cluster
func (m *VMClusterManager) StartCluster(ctx context.Context, clusterID string) error {
	// Get the cluster
	m.clustersMutex.Lock()
	cluster, exists := m.clusters[clusterID]
	if !exists {
		m.clustersMutex.Unlock()
		return fmt.Errorf("cluster %s not found", clusterID)
	}
	
	// Update cluster state
	cluster.State = ClusterStateUpdating
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	// Start master VMs first
	for _, member := range cluster.Members {
		if member.Role == ClusterRoleMaster {
			// Get the VM first
			vm, err := m.vmManager.GetVM(member.VMID)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get master VM %s: %w", member.VMID, err)
			}
			
			// Get driver for the VM
			driver, err := m.vmManager.getDriver(vm.config)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get driver for VM %s: %w", member.VMID, err)
			}
			
			// Start the VM
			_, err = m.vmManager.startVM(ctx, vm, driver)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				
				return fmt.Errorf("failed to start master VM %s: %w", member.VMID, err)
			}
			
			// Update member status
			member.Status = StateRunning
		}
	}
	
	// Start worker VMs
	for _, member := range cluster.Members {
		if member.Role == ClusterRoleWorker {
			// Get the VM first
			vm, err := m.vmManager.GetVM(member.VMID)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get worker VM %s: %w", member.VMID, err)
			}
			
			// Get driver for the VM
			driver, err := m.vmManager.getDriver(vm.config)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get driver for VM %s: %w", member.VMID, err)
			}
			
			// Start the VM
			_, err = m.vmManager.startVM(ctx, vm, driver)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				
				return fmt.Errorf("failed to start worker VM %s: %w", member.VMID, err)
			}
			
			// Update member status
			member.Status = StateRunning
		}
	}
	
	// Update cluster state
	m.clustersMutex.Lock()
	cluster.State = ClusterStateRunning
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	log.Printf("Started cluster %s (%s)", cluster.Name, cluster.ID)
	
	return nil
}

// StopCluster stops all VMs in a cluster
func (m *VMClusterManager) StopCluster(ctx context.Context, clusterID string) error {
	// Get the cluster
	m.clustersMutex.Lock()
	cluster, exists := m.clusters[clusterID]
	if !exists {
		m.clustersMutex.Unlock()
		return fmt.Errorf("cluster %s not found", clusterID)
	}
	
	// Update cluster state
	cluster.State = ClusterStateUpdating
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	// Stop worker VMs first
	for _, member := range cluster.Members {
		if member.Role == ClusterRoleWorker {
			// Get the VM first
			vm, err := m.vmManager.GetVM(member.VMID)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get worker VM %s: %w", member.VMID, err)
			}
			
			// Get driver for the VM
			driver, err := m.vmManager.getDriver(vm.config)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get driver for VM %s: %w", member.VMID, err)
			}
			
			// Stop the VM
			_, err = m.vmManager.stopVM(ctx, vm, driver)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				
				return fmt.Errorf("failed to stop worker VM %s: %w", member.VMID, err)
			}
			
			// Update member status
			member.Status = StateStopped
		}
	}
	
	// Stop master VMs
	for _, member := range cluster.Members {
		if member.Role == ClusterRoleMaster {
			// Get the VM first
			vm, err := m.vmManager.GetVM(member.VMID)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get master VM %s: %w", member.VMID, err)
			}
			
			// Get driver for the VM
			driver, err := m.vmManager.getDriver(vm.config)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				return fmt.Errorf("failed to get driver for VM %s: %w", member.VMID, err)
			}
			
			// Stop the VM
			_, err = m.vmManager.stopVM(ctx, vm, driver)
			if err != nil {
				m.clustersMutex.Lock()
				cluster.State = ClusterStateDegraded
				cluster.UpdatedAt = time.Now()
				cluster.Metadata["error"] = err.Error()
				m.clustersMutex.Unlock()
				
				return fmt.Errorf("failed to stop master VM %s: %w", member.VMID, err)
			}
			
			// Update member status
			member.Status = StateStopped
		}
	}
	
	// Update cluster state
	m.clustersMutex.Lock()
	cluster.State = ClusterStateStopped
	cluster.UpdatedAt = time.Now()
	m.clustersMutex.Unlock()
	
	log.Printf("Stopped cluster %s (%s)", cluster.Name, cluster.ID)
	
	return nil
}

// createClusterMembers creates VMs for a cluster
func (m *VMClusterManager) createClusterMembers(ctx context.Context, cluster *VMCluster, masterCount, workerCount int, vmConfig VMConfig) error {
	// Create master VMs
	for i := 0; i < masterCount; i++ {
		vm, err := m.createClusterVM(ctx, cluster, ClusterRoleMaster, vmConfig)
		if err != nil {
			return fmt.Errorf("failed to create master VM: %w", err)
		}
		
		// Create member
		member := &VMClusterMember{
			VMID:     vm.ID(),
			Role:     ClusterRoleMaster,
			JoinedAt: time.Now(),
			Status:   vm.State(),
			NodeID:   vm.GetNodeID(),
		}
		
		// Add member to cluster
		m.clustersMutex.Lock()
		cluster.Members = append(cluster.Members, member)
		m.clustersMutex.Unlock()
	}
	
	// Create worker VMs
	for i := 0; i < workerCount; i++ {
		vm, err := m.createClusterVM(ctx, cluster, ClusterRoleWorker, vmConfig)
		if err != nil {
			return fmt.Errorf("failed to create worker VM: %w", err)
		}
		
		// Create member
		member := &VMClusterMember{
			VMID:     vm.ID(),
			Role:     ClusterRoleWorker,
			JoinedAt: time.Now(),
			Status:   vm.State(),
			NodeID:   vm.GetNodeID(),
		}
		
		// Add member to cluster
		m.clustersMutex.Lock()
		cluster.Members = append(cluster.Members, member)
		m.clustersMutex.Unlock()
	}
	
	return nil
}

// createClusterVM creates a VM for a cluster
func (m *VMClusterManager) createClusterVM(ctx context.Context, cluster *VMCluster, role ClusterRole, vmConfig VMConfig) (*VM, error) {
	// Clone the VM config
	config := vmConfig
	
	// Set VM name based on cluster and role
	config.Name = fmt.Sprintf("%s-%s-%s", cluster.Name, role, uuid.New().String()[:8])
	
	// Add cluster metadata to VM
	if config.Tags == nil {
		config.Tags = make(map[string]string)
	}
	
	config.Tags["cluster_id"] = cluster.ID
	config.Tags["cluster_role"] = string(role)
	
	// Create VM - convert VMConfig to CreateVMRequest
	req := CreateVMRequest{
		Name: config.Name,
		Spec: config,
		Tags: config.Tags,
	}
	vm, err := m.vmManager.CreateVM(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}
	
	// Get driver for starting the VM
	driver, err := m.vmManager.getDriver(vm.config)
	if err != nil {
		return nil, fmt.Errorf("failed to get driver for VM %s: %w", vm.ID(), err)
	}
	
	// Start VM
	_, err = m.vmManager.startVM(ctx, vm, driver)
	if err != nil {
		return nil, fmt.Errorf("failed to start VM: %w", err)
	}
	
	return vm, nil
}
