package helpers

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/google/uuid"
)

// MockDataGenerator provides utilities for generating test data
type MockDataGenerator struct {
	rand *rand.Rand
}

// NewMockDataGenerator creates a new mock data generator
func NewMockDataGenerator() *MockDataGenerator {
	return &MockDataGenerator{
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// VM test data structures
type MockVM struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	State    string            `json:"state"`
	CPU      int               `json:"cpu"`
	Memory   int               `json:"memory"`
	DiskSize int               `json:"disk_size"`
	Image    string            `json:"image"`
	NodeID   string            `json:"node_id"`
	TenantID string            `json:"tenant_id"`
	UserID   int               `json:"user_id"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// User test data structure
type MockUser struct {
	ID       int    `json:"id"`
	Email    string `json:"email"`
	Name     string `json:"name"`
	Role     string `json:"role"`
	TenantID string `json:"tenant_id"`
}

// Storage tier test data
type MockStorageTier struct {
	ID               int     `json:"id"`
	Name             string  `json:"name"`
	StorageClass     string  `json:"storage_class"`
	PerformanceTier  string  `json:"performance_tier"`
	CostPerGBMonth   float64 `json:"cost_per_gb_month"`
	IOPS             int     `json:"iops"`
	ThroughputMBps   int     `json:"throughput_mbps"`
}

// Quota test data
type MockQuota struct {
	ID           int    `json:"id"`
	TenantID     string `json:"tenant_id"`
	ResourceType string `json:"resource_type"`
	LimitValue   int    `json:"limit_value"`
	CurrentUsage int    `json:"current_usage"`
}

// Network test data
type MockNetwork struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	CIDR     string            `json:"cidr"`
	VlanID   int               `json:"vlan_id"`
	TenantID string            `json:"tenant_id"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Backup test data
type MockBackup struct {
	ID         string    `json:"id"`
	VMID       string    `json:"vm_id"`
	Name       string    `json:"name"`
	Type       string    `json:"type"`
	Status     string    `json:"status"`
	Size       int64     `json:"size"`
	CreatedAt  time.Time `json:"created_at"`
	Checksum   string    `json:"checksum"`
	TenantID   string    `json:"tenant_id"`
}

// GenerateVM generates a mock VM
func (m *MockDataGenerator) GenerateVM(tenantID string, userID int) *MockVM {
	vmID := uuid.New().String()
	
	states := []string{"created", "running", "stopped", "paused"}
	images := []string{"ubuntu:20.04", "centos:8", "debian:11", "alpine:3.16"}
	nodeIDs := []string{"node-1", "node-2", "node-3", "node-4"}
	
	cpuOptions := []int{1, 2, 4, 8, 16}
	memoryOptions := []int{512, 1024, 2048, 4096, 8192, 16384}
	diskOptions := []int{10240, 20480, 51200, 102400, 204800}
	
	vm := &MockVM{
		ID:       vmID,
		Name:     fmt.Sprintf("test-vm-%s", vmID[:8]),
		State:    states[m.rand.Intn(len(states))],
		CPU:      cpuOptions[m.rand.Intn(len(cpuOptions))],
		Memory:   memoryOptions[m.rand.Intn(len(memoryOptions))],
		DiskSize: diskOptions[m.rand.Intn(len(diskOptions))],
		Image:    images[m.rand.Intn(len(images))],
		NodeID:   nodeIDs[m.rand.Intn(len(nodeIDs))],
		TenantID: tenantID,
		UserID:   userID,
		Metadata: map[string]string{
			"environment": []string{"development", "staging", "production"}[m.rand.Intn(3)],
			"application": []string{"web", "database", "cache", "worker"}[m.rand.Intn(4)],
		},
	}
	
	return vm
}

// GenerateVMs generates multiple mock VMs
func (m *MockDataGenerator) GenerateVMs(count int, tenantID string, userID int) []*MockVM {
	vms := make([]*MockVM, count)
	for i := 0; i < count; i++ {
		vms[i] = m.GenerateVM(tenantID, userID)
	}
	return vms
}

// GenerateUser generates a mock user
func (m *MockDataGenerator) GenerateUser(tenantID string) *MockUser {
	userID := m.rand.Intn(10000) + 1000
	domains := []string{"test.com", "example.org", "demo.net"}
	roles := []string{"user", "admin", "operator"}
	
	user := &MockUser{
		ID:       userID,
		Email:    fmt.Sprintf("user%d@%s", userID, domains[m.rand.Intn(len(domains))]),
		Name:     fmt.Sprintf("Test User %d", userID),
		Role:     roles[m.rand.Intn(len(roles))],
		TenantID: tenantID,
	}
	
	return user
}

// GenerateStorageTier generates a mock storage tier
func (m *MockDataGenerator) GenerateStorageTier() *MockStorageTier {
	tiers := []struct {
		name, class, performance string
		cost                     float64
		iops, throughput         int
	}{
		{"standard", "HDD", "standard", 0.045, 100, 125},
		{"premium", "SSD", "high", 0.125, 500, 500},
		{"ultra", "NVMe", "ultra", 0.300, 2000, 1000},
		{"archive", "HDD", "low", 0.015, 50, 50},
	}
	
	tier := tiers[m.rand.Intn(len(tiers))]
	
	return &MockStorageTier{
		ID:              m.rand.Intn(1000) + 1,
		Name:            tier.name,
		StorageClass:    tier.class,
		PerformanceTier: tier.performance,
		CostPerGBMonth:  tier.cost,
		IOPS:            tier.iops,
		ThroughputMBps:  tier.throughput,
	}
}

// GenerateQuota generates a mock resource quota
func (m *MockDataGenerator) GenerateQuota(tenantID string) *MockQuota {
	resourceTypes := []string{"cpu", "memory", "storage", "vms", "networks"}
	resourceType := resourceTypes[m.rand.Intn(len(resourceTypes))]
	
	var limitValue, currentUsage int
	
	switch resourceType {
	case "cpu":
		limitValue = []int{50, 100, 200, 500}[m.rand.Intn(4)]
		currentUsage = m.rand.Intn(limitValue/2)
	case "memory":
		limitValue = []int{51200, 102400, 204800, 512000}[m.rand.Intn(4)]
		currentUsage = m.rand.Intn(limitValue/2)
	case "storage":
		limitValue = []int{500000, 1000000, 2000000, 5000000}[m.rand.Intn(4)]
		currentUsage = m.rand.Intn(limitValue/2)
	case "vms":
		limitValue = []int{25, 50, 100, 250}[m.rand.Intn(4)]
		currentUsage = m.rand.Intn(limitValue/2)
	case "networks":
		limitValue = []int{10, 20, 50, 100}[m.rand.Intn(4)]
		currentUsage = m.rand.Intn(limitValue/2)
	}
	
	return &MockQuota{
		ID:           m.rand.Intn(1000) + 1,
		TenantID:     tenantID,
		ResourceType: resourceType,
		LimitValue:   limitValue,
		CurrentUsage: currentUsage,
	}
}

// GenerateNetwork generates a mock network
func (m *MockDataGenerator) GenerateNetwork(tenantID string) *MockNetwork {
	networkID := uuid.New().String()
	
	// Generate CIDR blocks
	cidrs := []string{
		"10.0.0.0/24", "10.1.0.0/24", "10.2.0.0/24",
		"172.16.0.0/24", "172.17.0.0/24", "172.18.0.0/24",
		"192.168.1.0/24", "192.168.2.0/24", "192.168.3.0/24",
	}
	
	network := &MockNetwork{
		ID:       networkID,
		Name:     fmt.Sprintf("test-network-%s", networkID[:8]),
		CIDR:     cidrs[m.rand.Intn(len(cidrs))],
		VlanID:   m.rand.Intn(4096) + 1,
		TenantID: tenantID,
		Metadata: map[string]string{
			"type":        []string{"internal", "external", "dmz"}[m.rand.Intn(3)],
			"environment": []string{"development", "staging", "production"}[m.rand.Intn(3)],
		},
	}
	
	return network
}

// GenerateBackup generates a mock backup
func (m *MockDataGenerator) GenerateBackup(vmID, tenantID string) *MockBackup {
	backupID := uuid.New().String()
	
	statuses := []string{"completed", "in_progress", "failed", "scheduled"}
	types := []string{"full", "incremental", "snapshot"}
	
	// Generate realistic backup sizes (in bytes)
	sizes := []int64{
		1 * 1024 * 1024 * 1024,      // 1 GB
		5 * 1024 * 1024 * 1024,      // 5 GB
		10 * 1024 * 1024 * 1024,     // 10 GB
		50 * 1024 * 1024 * 1024,     // 50 GB
		100 * 1024 * 1024 * 1024,    // 100 GB
	}
	
	backup := &MockBackup{
		ID:        backupID,
		VMID:      vmID,
		Name:      fmt.Sprintf("backup-%s-%d", vmID[:8], time.Now().Unix()),
		Type:      types[m.rand.Intn(len(types))],
		Status:    statuses[m.rand.Intn(len(statuses))],
		Size:      sizes[m.rand.Intn(len(sizes))],
		CreatedAt: time.Now().Add(-time.Duration(m.rand.Intn(30*24)) * time.Hour), // Random time in last 30 days
		Checksum:  fmt.Sprintf("sha256:%x", m.rand.Uint64()),
		TenantID:  tenantID,
	}
	
	return backup
}

// GenerateMetricsData generates mock metrics data
func (m *MockDataGenerator) GenerateMetricsData(vmID string, points int) []map[string]interface{} {
	metrics := make([]map[string]interface{}, points)
	baseTime := time.Now().Add(-time.Duration(points) * time.Minute)
	
	for i := 0; i < points; i++ {
		timestamp := baseTime.Add(time.Duration(i) * time.Minute)
		
		metrics[i] = map[string]interface{}{
			"timestamp": timestamp.Format(time.RFC3339),
			"vm_id":     vmID,
			"cpu_usage": m.rand.Float64() * 100, // 0-100%
			"memory_usage": m.rand.Float64() * 100, // 0-100%
			"disk_usage": m.rand.Float64() * 100, // 0-100%
			"network_in": m.rand.Float64() * 1000 * 1000, // bytes/sec
			"network_out": m.rand.Float64() * 1000 * 1000, // bytes/sec
			"disk_read": m.rand.Float64() * 100 * 1024 * 1024, // bytes/sec
			"disk_write": m.rand.Float64() * 100 * 1024 * 1024, // bytes/sec
		}
	}
	
	return metrics
}

// GenerateTenantID generates a random tenant ID
func (m *MockDataGenerator) GenerateTenantID() string {
	return fmt.Sprintf("tenant-%d", m.rand.Intn(1000)+1)
}

// GenerateAPIKey generates a mock API key
func (m *MockDataGenerator) GenerateAPIKey() string {
	return fmt.Sprintf("nck_%s", uuid.New().String())
}

// GenerateLoadBalancerConfig generates mock load balancer configuration
func (m *MockDataGenerator) GenerateLoadBalancerConfig(tenantID string) map[string]interface{} {
	algorithms := []string{"round_robin", "least_connections", "ip_hash", "weighted_round_robin"}
	healthCheckPaths := []string{"/health", "/status", "/ping", "/ready"}
	
	return map[string]interface{}{
		"id":        uuid.New().String(),
		"name":      fmt.Sprintf("lb-%d", m.rand.Intn(1000)),
		"algorithm": algorithms[m.rand.Intn(len(algorithms))],
		"tenant_id": tenantID,
		"frontend": map[string]interface{}{
			"port":     []int{80, 443, 8080, 8443}[m.rand.Intn(4)],
			"protocol": []string{"http", "https", "tcp"}[m.rand.Intn(3)],
		},
		"backends": []map[string]interface{}{
			{
				"host":   fmt.Sprintf("backend-1.%s", tenantID),
				"port":   8080,
				"weight": m.rand.Intn(10) + 1,
			},
			{
				"host":   fmt.Sprintf("backend-2.%s", tenantID),
				"port":   8080,
				"weight": m.rand.Intn(10) + 1,
			},
		},
		"health_check": map[string]interface{}{
			"enabled":      true,
			"path":         healthCheckPaths[m.rand.Intn(len(healthCheckPaths))],
			"interval":     30,
			"timeout":      5,
			"max_retries":  3,
		},
	}
}