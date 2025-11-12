package dwcp

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// VMClient provides VM management operations
type VMClient struct {
	client *Client
}

// NewVMClient creates a new VM client
func (c *Client) VM() *VMClient {
	return &VMClient{client: c}
}

// VMConfig represents VM configuration
type VMConfig struct {
	Name        string            `json:"name"`
	Memory      uint64            `json:"memory"`       // Memory in bytes
	CPUs        uint32            `json:"cpus"`         // Number of vCPUs
	Disk        uint64            `json:"disk"`         // Disk size in bytes
	Image       string            `json:"image"`        // Base image
	Network     NetworkConfig     `json:"network"`      // Network configuration
	CloudInit   string            `json:"cloud_init"`   // Cloud-init user data
	Labels      map[string]string `json:"labels"`       // Metadata labels
	Annotations map[string]string `json:"annotations"`  // Additional annotations
	Priority    int               `json:"priority"`     // Scheduling priority
	Affinity    *Affinity         `json:"affinity"`     // Node affinity

	// Advanced features
	EnableGPU        bool     `json:"enable_gpu"`
	GPUType          string   `json:"gpu_type"`
	EnableSR_IOV     bool     `json:"enable_sr_iov"`
	EnableTPM        bool     `json:"enable_tpm"`
	EnableSecureBoot bool     `json:"enable_secure_boot"`
	HostDevices      []string `json:"host_devices"`

	// Performance tuning
	CPUPinning       []int  `json:"cpu_pinning"`
	NUMANodes        []int  `json:"numa_nodes"`
	HugePages        bool   `json:"huge_pages"`
	IOThreads        int    `json:"io_threads"`

	// Resource limits
	MemoryMax        uint64 `json:"memory_max"`
	CPUQuota         int    `json:"cpu_quota"`
	DiskIOPSLimit    int    `json:"disk_iops_limit"`
	NetworkBandwidth uint64 `json:"network_bandwidth"`
}

// NetworkConfig represents network configuration
type NetworkConfig struct {
	Mode       string   `json:"mode"`        // bridge, nat, passthrough
	Interfaces []NetIf  `json:"interfaces"`  // Network interfaces
	DNS        []string `json:"dns"`         // DNS servers
	Gateway    string   `json:"gateway"`     // Default gateway
	MTU        int      `json:"mtu"`         // MTU size
}

// NetIf represents a network interface
type NetIf struct {
	Name       string `json:"name"`
	Type       string `json:"type"`        // virtio, e1000, etc.
	MAC        string `json:"mac"`
	Bridge     string `json:"bridge"`
	VLAN       int    `json:"vlan"`
	IPAddress  string `json:"ip_address"`
	Netmask    string `json:"netmask"`
	Bandwidth  uint64 `json:"bandwidth"`   // Bandwidth limit in bps
}

// Affinity represents node affinity rules
type Affinity struct {
	NodeSelector       map[string]string `json:"node_selector"`
	RequiredNodes      []string          `json:"required_nodes"`
	PreferredNodes     []string          `json:"preferred_nodes"`
	AntiAffinityVMs    []string          `json:"anti_affinity_vms"`
	RequireSameHost    []string          `json:"require_same_host"`
}

// VM represents a virtual machine
type VM struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	State       VMState           `json:"state"`
	Config      VMConfig          `json:"config"`
	Node        string            `json:"node"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	StartedAt   *time.Time        `json:"started_at,omitempty"`
	StoppedAt   *time.Time        `json:"stopped_at,omitempty"`
	Metrics     *VMMetrics        `json:"metrics,omitempty"`
	Labels      map[string]string `json:"labels"`
	Annotations map[string]string `json:"annotations"`
}

// VMState represents VM state
type VMState string

const (
	VMStateCreating  VMState = "creating"
	VMStateStarting  VMState = "starting"
	VMStateRunning   VMState = "running"
	VMStateStopping  VMState = "stopping"
	VMStateStopped   VMState = "stopped"
	VMStateMigrating VMState = "migrating"
	VMStateFailed    VMState = "failed"
	VMStateUnknown   VMState = "unknown"
)

// VMMetrics represents VM runtime metrics
type VMMetrics struct {
	CPUUsage        float64   `json:"cpu_usage"`         // CPU usage percentage
	MemoryUsed      uint64    `json:"memory_used"`       // Memory used in bytes
	MemoryAvailable uint64    `json:"memory_available"`  // Memory available in bytes
	DiskRead        uint64    `json:"disk_read"`         // Disk read bytes
	DiskWrite       uint64    `json:"disk_write"`        // Disk write bytes
	NetworkRx       uint64    `json:"network_rx"`        // Network received bytes
	NetworkTx       uint64    `json:"network_tx"`        // Network transmitted bytes
	Timestamp       time.Time `json:"timestamp"`         // Metric timestamp
}

// CreateVMRequest represents a VM creation request
type CreateVMRequest struct {
	Config VMConfig `json:"config"`
}

// CreateVMResponse represents a VM creation response
type CreateVMResponse struct {
	VM      VM     `json:"vm"`
	Message string `json:"message"`
}

// Create creates a new VM
func (v *VMClient) Create(ctx context.Context, config VMConfig) (*VM, error) {
	req := CreateVMRequest{Config: config}

	resp, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpCreate,
		"request":   req,
	})
	if err != nil {
		return nil, err
	}

	var createResp CreateVMResponse
	if err := json.Unmarshal(resp, &createResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &createResp.VM, nil
}

// Start starts a VM
func (v *VMClient) Start(ctx context.Context, vmID string) error {
	_, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpStart,
		"vm_id":     vmID,
	})
	return err
}

// Stop stops a VM
func (v *VMClient) Stop(ctx context.Context, vmID string, force bool) error {
	_, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpStop,
		"vm_id":     vmID,
		"force":     force,
	})
	return err
}

// Destroy destroys a VM
func (v *VMClient) Destroy(ctx context.Context, vmID string) error {
	_, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpDestroy,
		"vm_id":     vmID,
	})
	return err
}

// Get retrieves VM information
func (v *VMClient) Get(ctx context.Context, vmID string) (*VM, error) {
	resp, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpStatus,
		"vm_id":     vmID,
	})
	if err != nil {
		return nil, err
	}

	var vm VM
	if err := json.Unmarshal(resp, &vm); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &vm, nil
}

// List lists all VMs with optional filters
func (v *VMClient) List(ctx context.Context, filters map[string]string) ([]VM, error) {
	resp, err := v.client.sendRequest(ctx, MsgTypeVM, map[string]interface{}{
		"operation": VMOpStatus,
		"filters":   filters,
	})
	if err != nil {
		return nil, err
	}

	var vms []VM
	if err := json.Unmarshal(resp, &vms); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return vms, nil
}

// Watch watches VM state changes
func (v *VMClient) Watch(ctx context.Context, vmID string) (<-chan VMEvent, error) {
	stream, err := v.client.NewStream(ctx)
	if err != nil {
		return nil, err
	}

	// Send watch request
	req := map[string]interface{}{
		"operation": "watch",
		"vm_id":     vmID,
	}

	reqBytes, _ := json.Marshal(req)
	if err := stream.Send(reqBytes); err != nil {
		stream.Close()
		return nil, err
	}

	eventCh := make(chan VMEvent, 10)

	go func() {
		defer close(eventCh)
		defer stream.Close()

		for {
			data, err := stream.Receive()
			if err != nil {
				return
			}

			var event VMEvent
			if err := json.Unmarshal(data, &event); err != nil {
				continue
			}

			select {
			case eventCh <- event:
			case <-ctx.Done():
				return
			}
		}
	}()

	return eventCh, nil
}

// VMEvent represents a VM state change event
type VMEvent struct {
	Type      string    `json:"type"`
	VM        VM        `json:"vm"`
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message"`
}

// Migrate initiates VM migration
func (v *VMClient) Migrate(ctx context.Context, vmID, targetNode string, options MigrationOptions) (*MigrationStatus, error) {
	req := MigrationRequest{
		VMID:       vmID,
		TargetNode: targetNode,
		Options:    options,
	}

	resp, err := v.client.sendRequest(ctx, MsgTypeMigration, req)
	if err != nil {
		return nil, err
	}

	var status MigrationStatus
	if err := json.Unmarshal(resp, &status); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &status, nil
}

// MigrationOptions represents migration options
type MigrationOptions struct {
	Live              bool   `json:"live"`               // Live migration
	Offline           bool   `json:"offline"`            // Offline migration
	MaxDowntime       int    `json:"max_downtime"`       // Max downtime in ms
	Bandwidth         uint64 `json:"bandwidth"`          // Migration bandwidth limit
	Compression       bool   `json:"compression"`        // Enable compression
	AutoConverge      bool   `json:"auto_converge"`      // Enable auto-converge
	PostCopy          bool   `json:"post_copy"`          // Enable post-copy
	Parallel          int    `json:"parallel"`           // Parallel migration threads
	VerifyChecksum    bool   `json:"verify_checksum"`    // Verify data integrity
	EncryptTransport  bool   `json:"encrypt_transport"`  // Encrypt migration data
}

// MigrationRequest represents a migration request
type MigrationRequest struct {
	VMID       string           `json:"vm_id"`
	TargetNode string           `json:"target_node"`
	Options    MigrationOptions `json:"options"`
}

// MigrationStatus represents migration status
type MigrationStatus struct {
	ID            string              `json:"id"`
	VMID          string              `json:"vm_id"`
	SourceNode    string              `json:"source_node"`
	TargetNode    string              `json:"target_node"`
	State         MigrationState      `json:"state"`
	Progress      float64             `json:"progress"`        // 0-100
	BytesTotal    uint64              `json:"bytes_total"`
	BytesSent     uint64              `json:"bytes_sent"`
	Throughput    uint64              `json:"throughput"`      // bytes/sec
	Downtime      int                 `json:"downtime"`        // milliseconds
	StartedAt     time.Time           `json:"started_at"`
	CompletedAt   *time.Time          `json:"completed_at,omitempty"`
	Error         string              `json:"error,omitempty"`
}

// MigrationState represents migration state
type MigrationState string

const (
	MigrationStatePreparing  MigrationState = "preparing"
	MigrationStateRunning    MigrationState = "running"
	MigrationStateCompleting MigrationState = "completing"
	MigrationStateCompleted  MigrationState = "completed"
	MigrationStateFailed     MigrationState = "failed"
	MigrationStateCancelled  MigrationState = "cancelled"
)

// GetMigrationStatus retrieves migration status
func (v *VMClient) GetMigrationStatus(ctx context.Context, migrationID string) (*MigrationStatus, error) {
	resp, err := v.client.sendRequest(ctx, MsgTypeMigration, map[string]interface{}{
		"operation":    "status",
		"migration_id": migrationID,
	})
	if err != nil {
		return nil, err
	}

	var status MigrationStatus
	if err := json.Unmarshal(resp, &status); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &status, nil
}

// Snapshot creates a VM snapshot
func (v *VMClient) Snapshot(ctx context.Context, vmID, snapshotName string, options SnapshotOptions) (*Snapshot, error) {
	req := SnapshotRequest{
		VMID:    vmID,
		Name:    snapshotName,
		Options: options,
	}

	resp, err := v.client.sendRequest(ctx, MsgTypeSnapshot, req)
	if err != nil {
		return nil, err
	}

	var snapshot Snapshot
	if err := json.Unmarshal(resp, &snapshot); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &snapshot, nil
}

// SnapshotOptions represents snapshot options
type SnapshotOptions struct {
	IncludeMemory bool   `json:"include_memory"`  // Include memory state
	Description   string `json:"description"`     // Snapshot description
	Quiesce       bool   `json:"quiesce"`         // Quiesce filesystem
}

// SnapshotRequest represents a snapshot request
type SnapshotRequest struct {
	VMID    string          `json:"vm_id"`
	Name    string          `json:"name"`
	Options SnapshotOptions `json:"options"`
}

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID          string    `json:"id"`
	VMID        string    `json:"vm_id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Size        uint64    `json:"size"`
	CreatedAt   time.Time `json:"created_at"`
	Parent      string    `json:"parent,omitempty"`
	Children    []string  `json:"children,omitempty"`
}

// ListSnapshots lists all snapshots for a VM
func (v *VMClient) ListSnapshots(ctx context.Context, vmID string) ([]Snapshot, error) {
	resp, err := v.client.sendRequest(ctx, MsgTypeSnapshot, map[string]interface{}{
		"operation": "list",
		"vm_id":     vmID,
	})
	if err != nil {
		return nil, err
	}

	var snapshots []Snapshot
	if err := json.Unmarshal(resp, &snapshots); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return snapshots, nil
}

// RestoreSnapshot restores a VM from a snapshot
func (v *VMClient) RestoreSnapshot(ctx context.Context, vmID, snapshotID string) error {
	_, err := v.client.sendRequest(ctx, MsgTypeSnapshot, map[string]interface{}{
		"operation":   VMOpRestore,
		"vm_id":       vmID,
		"snapshot_id": snapshotID,
	})
	return err
}

// DeleteSnapshot deletes a snapshot
func (v *VMClient) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	_, err := v.client.sendRequest(ctx, MsgTypeSnapshot, map[string]interface{}{
		"operation":   "delete",
		"snapshot_id": snapshotID,
	})
	return err
}

// GetMetrics retrieves VM metrics
func (v *VMClient) GetMetrics(ctx context.Context, vmID string, duration time.Duration) (*VMMetrics, error) {
	resp, err := v.client.sendRequest(ctx, MsgTypeMetrics, map[string]interface{}{
		"vm_id":    vmID,
		"duration": duration.String(),
	})
	if err != nil {
		return nil, err
	}

	var metrics VMMetrics
	if err := json.Unmarshal(resp, &metrics); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &metrics, nil
}

// StreamMetrics streams real-time VM metrics
func (v *VMClient) StreamMetrics(ctx context.Context, vmID string, interval time.Duration) (<-chan VMMetrics, error) {
	stream, err := v.client.NewStream(ctx)
	if err != nil {
		return nil, err
	}

	req := map[string]interface{}{
		"operation": "stream_metrics",
		"vm_id":     vmID,
		"interval":  interval.String(),
	}

	reqBytes, _ := json.Marshal(req)
	if err := stream.Send(reqBytes); err != nil {
		stream.Close()
		return nil, err
	}

	metricsCh := make(chan VMMetrics, 10)

	go func() {
		defer close(metricsCh)
		defer stream.Close()

		for {
			data, err := stream.Receive()
			if err != nil {
				return
			}

			var metrics VMMetrics
			if err := json.Unmarshal(data, &metrics); err != nil {
				continue
			}

			select {
			case metricsCh <- metrics:
			case <-ctx.Done():
				return
			}
		}
	}()

	return metricsCh, nil
}
