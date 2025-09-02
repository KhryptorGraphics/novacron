package fixtures

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestFixtures provides test data fixtures for integration tests
type TestFixtures struct {
	db          *sql.DB
	authManager *auth.SimpleAuthManager
	vmManager   *core_vm.VMManager
	
	// Created test data
	testUsers []TestUser
	testVMs   []TestVM
}

// TestUser represents a test user fixture
type TestUser struct {
	ID       int    `json:"id"`
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"` // Plain text for testing
	Role     string `json:"role"`
	TenantID string `json:"tenant_id"`
	Token    string `json:"token,omitempty"`
}

// TestVM represents a test VM fixture
type TestVM struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	State    string                 `json:"state"`
	CPUCores int                    `json:"cpu_cores"`
	MemoryMB int                    `json:"memory_mb"`
	DiskGB   int                    `json:"disk_gb"`
	OwnerID  int                    `json:"owner_id"`
	TenantID string                 `json:"tenant_id"`
	Config   map[string]interface{} `json:"config"`
}

// TestVMMetric represents a test VM metric fixture
type TestVMMetric struct {
	VMID         string    `json:"vm_id"`
	CPUUsage     float64   `json:"cpu_usage"`
	MemoryUsage  float64   `json:"memory_usage"`
	DiskUsage    float64   `json:"disk_usage"`
	NetworkSent  int64     `json:"network_sent"`
	NetworkRecv  int64     `json:"network_recv"`
	Timestamp    time.Time `json:"timestamp"`
}

// New creates a new test fixtures instance
func New(db *sql.DB, authManager *auth.SimpleAuthManager, vmManager *core_vm.VMManager) *TestFixtures {
	return &TestFixtures{
		db:          db,
		authManager: authManager,
		vmManager:   vmManager,
		testUsers:   make([]TestUser, 0),
		testVMs:     make([]TestVM, 0),
	}
}

// CreateTestUsers creates standard test users
func (tf *TestFixtures) CreateTestUsers() error {
	testUsers := []TestUser{
		{
			Username: "admin_test",
			Email:    "admin.test@example.com",
			Password: "AdminPassword123!",
			Role:     "admin",
			TenantID: "test-tenant",
		},
		{
			Username: "user_test",
			Email:    "user.test@example.com",
			Password: "UserPassword123!",
			Role:     "user",
			TenantID: "test-tenant",
		},
		{
			Username: "manager_test",
			Email:    "manager.test@example.com",
			Password: "ManagerPassword123!",
			Role:     "manager",
			TenantID: "test-tenant",
		},
		{
			Username: "readonly_test",
			Email:    "readonly.test@example.com",
			Password: "ReadOnlyPassword123!",
			Role:     "readonly",
			TenantID: "test-tenant",
		},
	}

	for _, userData := range testUsers {
		user, err := tf.authManager.CreateUser(
			userData.Username,
			userData.Email,
			userData.Password,
			userData.Role,
			userData.TenantID,
		)
		if err != nil {
			return fmt.Errorf("failed to create test user %s: %w", userData.Username, err)
		}

		// Generate auth token
		_, token, err := tf.authManager.Authenticate(userData.Username, userData.Password)
		if err != nil {
			return fmt.Errorf("failed to authenticate test user %s: %w", userData.Username, err)
		}

		userData.ID = user.ID
		userData.Token = token
		tf.testUsers = append(tf.testUsers, userData)
	}

	return nil
}

// CreateTestVMs creates standard test VMs
func (tf *TestFixtures) CreateTestVMs() error {
	if len(tf.testUsers) == 0 {
		return fmt.Errorf("no test users available, create users first")
	}

	// Find admin user
	var adminUser TestUser
	for _, user := range tf.testUsers {
		if user.Role == "admin" {
			adminUser = user
			break
		}
	}
	if adminUser.ID == 0 {
		adminUser = tf.testUsers[0] // Fallback to first user
	}

	testVMs := []TestVM{
		{
			ID:       "test-vm-001",
			Name:     "web-server-test",
			State:    "running",
			CPUCores: 2,
			MemoryMB: 2048,
			DiskGB:   20,
			OwnerID:  adminUser.ID,
			TenantID: "test-tenant",
			Config: map[string]interface{}{
				"os":          "ubuntu",
				"version":     "20.04",
				"environment": "test",
			},
		},
		{
			ID:       "test-vm-002",
			Name:     "database-test",
			State:    "running",
			CPUCores: 4,
			MemoryMB: 8192,
			DiskGB:   100,
			OwnerID:  adminUser.ID,
			TenantID: "test-tenant",
			Config: map[string]interface{}{
				"os":          "postgresql",
				"version":     "13",
				"environment": "test",
			},
		},
		{
			ID:       "test-vm-003",
			Name:     "app-server-test",
			State:    "stopped",
			CPUCores: 1,
			MemoryMB: 1024,
			DiskGB:   10,
			OwnerID:  adminUser.ID,
			TenantID: "test-tenant",
			Config: map[string]interface{}{
				"os":          "centos",
				"version":     "8",
				"environment": "test",
			},
		},
		{
			ID:       "test-vm-004",
			Name:     "load-balancer-test",
			State:    "running",
			CPUCores: 2,
			MemoryMB: 4096,
			DiskGB:   50,
			OwnerID:  adminUser.ID,
			TenantID: "test-tenant",
			Config: map[string]interface{}{
				"os":          "nginx",
				"version":     "1.20",
				"environment": "test",
			},
		},
	}

	for _, vmData := range testVMs {
		err := tf.createVMInDatabase(vmData)
		if err != nil {
			return fmt.Errorf("failed to create test VM %s: %w", vmData.Name, err)
		}

		tf.testVMs = append(tf.testVMs, vmData)
	}

	return nil
}

// CreateTestVMMetrics creates sample VM metrics
func (tf *TestFixtures) CreateTestVMMetrics() error {
	if len(tf.testVMs) == 0 {
		return fmt.Errorf("no test VMs available, create VMs first")
	}

	baseTime := time.Now().Add(-24 * time.Hour) // Start from 24 hours ago
	interval := 5 * time.Minute                 // 5-minute intervals

	for _, vm := range tf.testVMs {
		// Generate 24 hours of metrics (288 data points)
		for i := 0; i < 288; i++ {
			timestamp := baseTime.Add(time.Duration(i) * interval)
			
			// Generate realistic metrics based on VM state
			var cpuUsage, memoryUsage, diskUsage float64
			var networkSent, networkRecv int64

			if vm.State == "running" {
				// Running VMs have varying usage
				cpuUsage = 20.0 + float64(i%60) + float64(i%3)*10.0
				if cpuUsage > 95.0 {
					cpuUsage = 95.0
				}
				
				memoryUsage = 30.0 + float64(i%40) + float64(i%5)*8.0
				if memoryUsage > 90.0 {
					memoryUsage = 90.0
				}
				
				diskUsage = 25.0 + float64(i%20)*0.1
				if diskUsage > 80.0 {
					diskUsage = 80.0
				}
				
				networkSent = int64(1024 * (100 + i%1000))
				networkRecv = int64(1024 * (200 + i%800))
			} else {
				// Stopped VMs have minimal usage
				cpuUsage = 0.0
				memoryUsage = 0.0
				diskUsage = 25.0 // Static disk usage
				networkSent = 0
				networkRecv = 0
			}

			metric := TestVMMetric{
				VMID:        vm.ID,
				CPUUsage:    cpuUsage,
				MemoryUsage: memoryUsage,
				DiskUsage:   diskUsage,
				NetworkSent: networkSent,
				NetworkRecv: networkRecv,
				Timestamp:   timestamp,
			}

			err := tf.createVMMetricInDatabase(metric)
			if err != nil {
				return fmt.Errorf("failed to create VM metric for %s: %w", vm.ID, err)
			}
		}
	}

	return nil
}

// GetTestUser returns a test user by role or username
func (tf *TestFixtures) GetTestUser(identifier string) (*TestUser, error) {
	for _, user := range tf.testUsers {
		if user.Username == identifier || user.Role == identifier {
			return &user, nil
		}
	}
	return nil, fmt.Errorf("test user with identifier '%s' not found", identifier)
}

// GetTestUsers returns all test users
func (tf *TestFixtures) GetTestUsers() []TestUser {
	return tf.testUsers
}

// GetTestVM returns a test VM by ID or name
func (tf *TestFixtures) GetTestVM(identifier string) (*TestVM, error) {
	for _, vm := range tf.testVMs {
		if vm.ID == identifier || vm.Name == identifier {
			return &vm, nil
		}
	}
	return nil, fmt.Errorf("test VM with identifier '%s' not found", identifier)
}

// GetTestVMs returns all test VMs
func (tf *TestFixtures) GetTestVMs() []TestVM {
	return tf.testVMs
}

// GetTestVMsByState returns test VMs in a specific state
func (tf *TestFixtures) GetTestVMsByState(state string) []TestVM {
	var vms []TestVM
	for _, vm := range tf.testVMs {
		if vm.State == state {
			vms = append(vms, vm)
		}
	}
	return vms
}

// CreateAdditionalUser creates a new test user with custom parameters
func (tf *TestFixtures) CreateAdditionalUser(username, email, password, role, tenantID string) (*TestUser, error) {
	user, err := tf.authManager.CreateUser(username, email, password, role, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to create additional user %s: %w", username, err)
	}

	// Generate auth token
	_, token, err := tf.authManager.Authenticate(username, password)
	if err != nil {
		return nil, fmt.Errorf("failed to authenticate additional user %s: %w", username, err)
	}

	testUser := TestUser{
		ID:       user.ID,
		Username: username,
		Email:    email,
		Password: password,
		Role:     role,
		TenantID: tenantID,
		Token:    token,
	}

	tf.testUsers = append(tf.testUsers, testUser)
	return &testUser, nil
}

// CreateAdditionalVM creates a new test VM with custom parameters
func (tf *TestFixtures) CreateAdditionalVM(id, name, state string, cpuCores, memoryMB, diskGB, ownerID int, tenantID string) (*TestVM, error) {
	vm := TestVM{
		ID:       id,
		Name:     name,
		State:    state,
		CPUCores: cpuCores,
		MemoryMB: memoryMB,
		DiskGB:   diskGB,
		OwnerID:  ownerID,
		TenantID: tenantID,
		Config:   make(map[string]interface{}),
	}

	err := tf.createVMInDatabase(vm)
	if err != nil {
		return nil, fmt.Errorf("failed to create additional VM %s: %w", name, err)
	}

	tf.testVMs = append(tf.testVMs, vm)
	return &vm, nil
}

// Cleanup removes all test data
func (tf *TestFixtures) Cleanup() {
	// Delete VM metrics
	tf.db.Exec("DELETE FROM vm_metrics WHERE vm_id LIKE 'test-vm-%'")
	
	// Delete VMs
	tf.db.Exec("DELETE FROM vms WHERE id LIKE 'test-vm-%' OR name LIKE '%test%'")
	
	// Delete users (be careful with this)
	tf.db.Exec("DELETE FROM users WHERE username LIKE '%test%' OR email LIKE '%test%'")
	
	// Clear internal state
	tf.testUsers = nil
	tf.testVMs = nil
}

// Database helper methods

func (tf *TestFixtures) createVMInDatabase(vm TestVM) error {
	query := `
		INSERT INTO vms (id, name, state, cpu_cores, memory_mb, disk_gb, owner_id, tenant_id, config, created_at, updated_at) 
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
	`
	
	configJSON := "{}"
	if vm.Config != nil && len(vm.Config) > 0 {
		// In a real implementation, we'd properly serialize the config
		configJSON = `{"environment": "test"}`
	}

	_, err := tf.db.Exec(query, vm.ID, vm.Name, vm.State, vm.CPUCores, vm.MemoryMB, vm.DiskGB, vm.OwnerID, vm.TenantID, configJSON)
	return err
}

func (tf *TestFixtures) createVMMetricInDatabase(metric TestVMMetric) error {
	query := `
		INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, disk_usage, network_sent, network_recv, timestamp) 
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`
	
	_, err := tf.db.Exec(query, metric.VMID, metric.CPUUsage, metric.MemoryUsage, metric.DiskUsage, metric.NetworkSent, metric.NetworkRecv, metric.Timestamp)
	return err
}

// Utility methods

// GetRandomTestUser returns a random test user (useful for load testing)
func (tf *TestFixtures) GetRandomTestUser() *TestUser {
	if len(tf.testUsers) == 0 {
		return nil
	}
	
	// Simple pseudo-random selection based on current time
	index := int(time.Now().UnixNano()) % len(tf.testUsers)
	return &tf.testUsers[index]
}

// GetRunningVMsCount returns the count of running test VMs
func (tf *TestFixtures) GetRunningVMsCount() int {
	count := 0
	for _, vm := range tf.testVMs {
		if vm.State == "running" {
			count++
		}
	}
	return count
}

// GetVMsForUser returns VMs owned by a specific user
func (tf *TestFixtures) GetVMsForUser(userID int) []TestVM {
	var vms []TestVM
	for _, vm := range tf.testVMs {
		if vm.OwnerID == userID {
			vms = append(vms, vm)
		}
	}
	return vms
}

// ValidateFixtures validates that all fixtures are properly created
func (tf *TestFixtures) ValidateFixtures() error {
	// Check users
	if len(tf.testUsers) == 0 {
		return fmt.Errorf("no test users created")
	}

	for _, user := range tf.testUsers {
		if user.ID == 0 || user.Username == "" || user.Token == "" {
			return fmt.Errorf("invalid test user: %+v", user)
		}
	}

	// Check VMs
	if len(tf.testVMs) == 0 {
		return fmt.Errorf("no test VMs created")
	}

	for _, vm := range tf.testVMs {
		if vm.ID == "" || vm.Name == "" || vm.OwnerID == 0 {
			return fmt.Errorf("invalid test VM: %+v", vm)
		}
	}

	// Validate database consistency
	var userCount, vmCount int
	tf.db.QueryRow("SELECT COUNT(*) FROM users WHERE username LIKE '%test%'").Scan(&userCount)
	tf.db.QueryRow("SELECT COUNT(*) FROM vms WHERE id LIKE 'test-vm-%'").Scan(&vmCount)

	if userCount != len(tf.testUsers) {
		return fmt.Errorf("user count mismatch: expected %d, got %d", len(tf.testUsers), userCount)
	}

	if vmCount != len(tf.testVMs) {
		return fmt.Errorf("VM count mismatch: expected %d, got %d", len(tf.testVMs), vmCount)
	}

	return nil
}

// GetStatistics returns statistics about the test fixtures
func (tf *TestFixtures) GetStatistics() map[string]interface{} {
	stats := make(map[string]interface{})

	stats["total_users"] = len(tf.testUsers)
	stats["total_vms"] = len(tf.testVMs)
	stats["running_vms"] = tf.GetRunningVMsCount()

	// User role breakdown
	roleCount := make(map[string]int)
	for _, user := range tf.testUsers {
		roleCount[user.Role]++
	}
	stats["user_roles"] = roleCount

	// VM state breakdown
	stateCount := make(map[string]int)
	for _, vm := range tf.testVMs {
		stateCount[vm.State]++
	}
	stats["vm_states"] = stateCount

	return stats
}