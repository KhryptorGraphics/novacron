// End-to-End Comprehensive Workflow Tests
package e2e

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// E2E test configuration
type E2ETestConfig struct {
	BaseURL           string        `json:"base_url"`
	WebUIURL          string        `json:"web_ui_url"`
	APIKey            string        `json:"api_key"`
	TestTimeout       time.Duration `json:"test_timeout"`
	EnableUITests     bool          `json:"enable_ui_tests"`
	EnableLoadTests   bool          `json:"enable_load_tests"`
	CleanupOnFailure  bool          `json:"cleanup_on_failure"`
	ParallelExecution bool          `json:"parallel_execution"`
}

// Test workflow components
type WorkflowTestSuite struct {
	config     *E2ETestConfig
	httpClient *http.Client
	webDriver  WebDriver  // Placeholder for web automation
	apiClient  APIClient
	testData   *TestDataManager
	metrics    *WorkflowMetrics
	mu         sync.RWMutex
}

type WebDriver interface {
	NavigateTo(url string) error
	FindElement(selector string) (WebElement, error)
	WaitForElement(selector string, timeout time.Duration) (WebElement, error)
	TakeScreenshot() ([]byte, error)
	ExecuteScript(script string) (interface{}, error)
	Close() error
}

type WebElement interface {
	Click() error
	SendKeys(text string) error
	GetText() (string, error)
	GetAttribute(name string) (string, error)
	IsDisplayed() (bool, error)
}

type APIClient interface {
	CreateVM(ctx context.Context, req *CreateVMRequest) (*VM, error)
	GetVM(ctx context.Context, id string) (*VM, error)
	ListVMs(ctx context.Context) ([]*VM, error)
	DeleteVM(ctx context.Context, id string) error
	MigrateVM(ctx context.Context, req *MigrateVMRequest) (*MigrationJob, error)
	GetMigrationStatus(ctx context.Context, jobID string) (*MigrationJob, error)
	GetMetrics(ctx context.Context, vmID string) (*VMMetrics, error)
}

type TestDataManager struct {
	CreatedVMs      []string          `json:"created_vms"`
	MigrationJobs   []string          `json:"migration_jobs"`
	TestFiles       []string          `json:"test_files"`
	Snapshots       []string          `json:"snapshots"`
	CreatedNetworks []string          `json:"created_networks"`
	TestData        map[string]interface{} `json:"test_data"`
	mu              sync.RWMutex
}

type WorkflowMetrics struct {
	TotalWorkflows     int                    `json:"total_workflows"`
	SuccessfulWorkflows int                   `json:"successful_workflows"`
	FailedWorkflows    int                    `json:"failed_workflows"`
	WorkflowDurations  map[string]time.Duration `json:"workflow_durations"`
	ErrorCounts        map[string]int         `json:"error_counts"`
	PerformanceMetrics map[string]interface{} `json:"performance_metrics"`
}

// Data structures
type VM struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Status      string                 `json:"status"`
	NodeID      string                 `json:"node_id"`
	Resources   VMResources            `json:"resources"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

type VMResources struct {
	CPU    int `json:"cpu"`
	Memory int `json:"memory"` // MB
	Disk   int `json:"disk"`   // GB
}

type CreateVMRequest struct {
	Name      string                 `json:"name"`
	Template  string                 `json:"template"`
	Resources VMResources            `json:"resources"`
	NodeID    string                 `json:"node_id,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type MigrateVMRequest struct {
	VMID       string `json:"vm_id"`
	TargetNode string `json:"target_node"`
	MigrationType string `json:"migration_type"` // cold, warm, live
}

type MigrationJob struct {
	ID         string                 `json:"id"`
	VMID       string                 `json:"vm_id"`
	SourceNode string                 `json:"source_node"`
	TargetNode string                 `json:"target_node"`
	Status     string                 `json:"status"`
	Progress   float64                `json:"progress"`
	StartedAt  time.Time              `json:"started_at"`
	CompletedAt *time.Time            `json:"completed_at,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

type VMMetrics struct {
	VMID        string    `json:"vm_id"`
	CPUUsage    float64   `json:"cpu_usage"`
	MemoryUsage float64   `json:"memory_usage"`
	DiskIO      int64     `json:"disk_io"`
	NetworkIO   int64     `json:"network_io"`
	Timestamp   time.Time `json:"timestamp"`
}

func NewWorkflowTestSuite(config *E2ETestConfig) *WorkflowTestSuite {
	return &WorkflowTestSuite{
		config: config,
		httpClient: &http.Client{
			Timeout: config.TestTimeout,
		},
		testData: &TestDataManager{
			CreatedVMs:      make([]string, 0),
			MigrationJobs:   make([]string, 0),
			TestFiles:       make([]string, 0),
			Snapshots:       make([]string, 0),
			CreatedNetworks: make([]string, 0),
			TestData:        make(map[string]interface{}),
		},
		metrics: &WorkflowMetrics{
			WorkflowDurations:  make(map[string]time.Duration),
			ErrorCounts:        make(map[string]int),
			PerformanceMetrics: make(map[string]interface{}),
		},
	}
}

// Main E2E workflow tests
func TestComprehensiveWorkflows(t *testing.T) {
	config := getE2ETestConfig()
	suite := NewWorkflowTestSuite(config)
	
	// Setup and teardown
	defer suite.cleanup()

	t.Run("BasicVMLifecycle", func(t *testing.T) {
		suite.testBasicVMLifecycle(t)
	})

	t.Run("VMMigrationWorkflow", func(t *testing.T) {
		suite.testVMMigrationWorkflow(t)
	})

	t.Run("MultiVMOrchestration", func(t *testing.T) {
		suite.testMultiVMOrchestration(t)
	})

	t.Run("MonitoringAndAlerting", func(t *testing.T) {
		suite.testMonitoringAndAlerting(t)
	})

	t.Run("APIToUIWorkflow", func(t *testing.T) {
		if config.EnableUITests {
			suite.testAPIToUIWorkflow(t)
		} else {
			t.Skip("UI tests disabled")
		}
	})

	t.Run("LoadBalancingWorkflow", func(t *testing.T) {
		if config.EnableLoadTests {
			suite.testLoadBalancingWorkflow(t)
		} else {
			t.Skip("Load tests disabled")
		}
	})

	t.Run("FailureRecoveryWorkflow", func(t *testing.T) {
		suite.testFailureRecoveryWorkflow(t)
	})

	// Generate final report
	suite.generateWorkflowReport(t)
}

// Individual workflow tests
func (suite *WorkflowTestSuite) testBasicVMLifecycle(t *testing.T) {
	start := time.Now()
	workflowName := "BasicVMLifecycle"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()

	t.Log("Step 1: Create VM via API")
	createReq := &CreateVMRequest{
		Name:     "e2e-test-vm-basic",
		Template: "ubuntu-20.04",
		Resources: VMResources{
			CPU:    2,
			Memory: 4096,
			Disk:   50,
		},
		Metadata: map[string]interface{}{
			"test_type": "e2e",
			"workflow":  workflowName,
		},
	}

	vm, err := suite.apiClient.CreateVM(ctx, createReq)
	require.NoError(t, err, "VM creation should succeed")
	require.NotNil(t, vm)
	suite.testData.addCreatedVM(vm.ID)

	assert.Equal(t, createReq.Name, vm.Name)
	assert.Equal(t, createReq.Resources.CPU, vm.Resources.CPU)
	assert.NotEmpty(t, vm.ID)
	t.Logf("Created VM: %s", vm.ID)

	t.Log("Step 2: Wait for VM to be running")
	err = suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
	assert.NoError(t, err, "VM should reach running state")

	t.Log("Step 3: Verify VM is accessible via API")
	retrievedVM, err := suite.apiClient.GetVM(ctx, vm.ID)
	assert.NoError(t, err, "Should be able to retrieve VM")
	assert.Equal(t, vm.ID, retrievedVM.ID)
	assert.Equal(t, "running", retrievedVM.Status)

	t.Log("Step 4: Check VM appears in list")
	vms, err := suite.apiClient.ListVMs(ctx)
	assert.NoError(t, err, "Should be able to list VMs")
	
	found := false
	for _, listedVM := range vms {
		if listedVM.ID == vm.ID {
			found = true
			break
		}
	}
	assert.True(t, found, "VM should appear in list")

	t.Log("Step 5: Collect VM metrics")
	metrics, err := suite.apiClient.GetMetrics(ctx, vm.ID)
	assert.NoError(t, err, "Should be able to get VM metrics")
	assert.Equal(t, vm.ID, metrics.VMID)
	assert.GreaterOrEqual(t, metrics.CPUUsage, 0.0)
	assert.LessOrEqual(t, metrics.CPUUsage, 100.0)

	t.Log("Step 6: Delete VM")
	err = suite.apiClient.DeleteVM(ctx, vm.ID)
	assert.NoError(t, err, "VM deletion should succeed")

	t.Log("Step 7: Verify VM is deleted")
	err = suite.waitForVMStatus(ctx, vm.ID, "deleted", 2*time.Minute)
	assert.NoError(t, err, "VM should be deleted")

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ Basic VM lifecycle completed successfully")
}

func (suite *WorkflowTestSuite) testVMMigrationWorkflow(t *testing.T) {
	start := time.Now()
	workflowName := "VMMigrationWorkflow"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()

	t.Log("Step 1: Create VM for migration")
	createReq := &CreateVMRequest{
		Name:     "e2e-migration-vm",
		Template: "ubuntu-20.04",
		Resources: VMResources{
			CPU:    1,
			Memory: 2048,
			Disk:   30,
		},
		Metadata: map[string]interface{}{
			"test_type": "migration",
			"workflow":  workflowName,
		},
	}

	vm, err := suite.apiClient.CreateVM(ctx, createReq)
	require.NoError(t, err, "VM creation should succeed")
	suite.testData.addCreatedVM(vm.ID)

	t.Log("Step 2: Wait for VM to be running")
	err = suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
	require.NoError(t, err, "VM should reach running state")

	t.Log("Step 3: Get VM details before migration")
	preVM, err := suite.apiClient.GetVM(ctx, vm.ID)
	require.NoError(t, err, "Should get VM before migration")
	sourceNode := preVM.NodeID
	require.NotEmpty(t, sourceNode, "VM should have source node")

	t.Log("Step 4: Start VM migration")
	migrateReq := &MigrateVMRequest{
		VMID:          vm.ID,
		TargetNode:    suite.getAlternativeNode(sourceNode),
		MigrationType: "cold", // Start with cold migration for reliability
	}

	migrationJob, err := suite.apiClient.MigrateVM(ctx, migrateReq)
	require.NoError(t, err, "Migration should start successfully")
	require.NotEmpty(t, migrationJob.ID)
	suite.testData.addMigrationJob(migrationJob.ID)

	t.Logf("Started migration job: %s", migrationJob.ID)

	t.Log("Step 5: Monitor migration progress")
	err = suite.waitForMigrationCompletion(ctx, migrationJob.ID, 10*time.Minute)
	assert.NoError(t, err, "Migration should complete successfully")

	t.Log("Step 6: Verify VM after migration")
	postVM, err := suite.apiClient.GetVM(ctx, vm.ID)
	assert.NoError(t, err, "Should get VM after migration")
	assert.Equal(t, "running", postVM.Status, "VM should be running after migration")
	assert.NotEqual(t, sourceNode, postVM.NodeID, "VM should be on different node")
	assert.Equal(t, migrateReq.TargetNode, postVM.NodeID, "VM should be on target node")

	t.Log("Step 7: Verify VM functionality after migration")
	postMetrics, err := suite.apiClient.GetMetrics(ctx, vm.ID)
	assert.NoError(t, err, "Should get metrics after migration")
	assert.Equal(t, vm.ID, postMetrics.VMID)

	t.Log("Step 8: Cleanup migrated VM")
	err = suite.apiClient.DeleteVM(ctx, vm.ID)
	assert.NoError(t, err, "Should delete migrated VM")

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ VM migration workflow completed successfully")
}

func (suite *WorkflowTestSuite) testMultiVMOrchestration(t *testing.T) {
	start := time.Now()
	workflowName := "MultiVMOrchestration"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()
	vmCount := 5

	t.Logf("Step 1: Create %d VMs concurrently", vmCount)
	var wg sync.WaitGroup
	var mu sync.Mutex
	createdVMs := make([]*VM, 0, vmCount)
	errors := make([]error, 0)

	for i := 0; i < vmCount; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			createReq := &CreateVMRequest{
				Name:     fmt.Sprintf("e2e-multi-vm-%d", index),
				Template: "ubuntu-20.04",
				Resources: VMResources{
					CPU:    1,
					Memory: 1024,
					Disk:   20,
				},
				Metadata: map[string]interface{}{
					"test_type": "multi_vm",
					"workflow":  workflowName,
					"index":     index,
				},
			}

			vm, err := suite.apiClient.CreateVM(ctx, createReq)
			
			mu.Lock()
			if err != nil {
				errors = append(errors, err)
			} else {
				createdVMs = append(createdVMs, vm)
				suite.testData.addCreatedVM(vm.ID)
			}
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	
	require.Empty(t, errors, "All VM creations should succeed")
	require.Len(t, createdVMs, vmCount, "Should create all requested VMs")

	t.Log("Step 2: Wait for all VMs to be running")
	for _, vm := range createdVMs {
		err := suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
		assert.NoError(t, err, "VM %s should reach running state", vm.ID)
	}

	t.Log("Step 3: Verify all VMs are listed")
	listedVMs, err := suite.apiClient.ListVMs(ctx)
	require.NoError(t, err, "Should list all VMs")

	createdVMIDs := make(map[string]bool)
	for _, vm := range createdVMs {
		createdVMIDs[vm.ID] = false
	}

	for _, listedVM := range listedVMs {
		if _, exists := createdVMIDs[listedVM.ID]; exists {
			createdVMIDs[listedVM.ID] = true
		}
	}

	for vmID, found := range createdVMIDs {
		assert.True(t, found, "VM %s should appear in list", vmID)
	}

	t.Log("Step 4: Collect metrics from all VMs")
	metricsErrors := 0
	for _, vm := range createdVMs {
		_, err := suite.apiClient.GetMetrics(ctx, vm.ID)
		if err != nil {
			metricsErrors++
			t.Logf("Failed to get metrics for VM %s: %v", vm.ID, err)
		}
	}

	// Allow some metrics collection failures (up to 20%)
	allowedFailures := vmCount / 5
	assert.LessOrEqual(t, metricsErrors, allowedFailures,
		"Metrics collection failures should be minimal (%d/%d)", metricsErrors, vmCount)

	t.Log("Step 5: Delete all VMs concurrently")
	deletionErrors := 0
	var deleteWg sync.WaitGroup

	for _, vm := range createdVMs {
		deleteWg.Add(1)
		go func(vmID string) {
			defer deleteWg.Done()
			err := suite.apiClient.DeleteVM(ctx, vmID)
			if err != nil {
				mu.Lock()
				deletionErrors++
				mu.Unlock()
				t.Logf("Failed to delete VM %s: %v", vmID, err)
			}
		}(vm.ID)
	}

	deleteWg.Wait()
	assert.Equal(t, 0, deletionErrors, "All VM deletions should succeed")

	t.Log("Step 6: Verify all VMs are deleted")
	time.Sleep(5 * time.Second) // Allow time for deletions to propagate
	
	finalVMs, err := suite.apiClient.ListVMs(ctx)
	require.NoError(t, err, "Should list VMs after deletion")

	remainingTestVMs := 0
	for _, vm := range finalVMs {
		if strings.Contains(vm.Name, "e2e-multi-vm-") {
			remainingTestVMs++
		}
	}

	assert.Equal(t, 0, remainingTestVMs, "All test VMs should be deleted")

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ Multi-VM orchestration completed successfully")
}

func (suite *WorkflowTestSuite) testMonitoringAndAlerting(t *testing.T) {
	start := time.Now()
	workflowName := "MonitoringAndAlerting"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()

	t.Log("Step 1: Create VM for monitoring")
	createReq := &CreateVMRequest{
		Name:     "e2e-monitoring-vm",
		Template: "ubuntu-20.04",
		Resources: VMResources{
			CPU:    2,
			Memory: 2048,
			Disk:   30,
		},
		Metadata: map[string]interface{}{
			"monitoring_enabled": true,
			"test_type":          "monitoring",
			"workflow":           workflowName,
		},
	}

	vm, err := suite.apiClient.CreateVM(ctx, createReq)
	require.NoError(t, err, "VM creation should succeed")
	suite.testData.addCreatedVM(vm.ID)

	err = suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
	require.NoError(t, err, "VM should reach running state")

	t.Log("Step 2: Verify initial metrics collection")
	var initialMetrics *VMMetrics
	for attempts := 0; attempts < 10; attempts++ {
		initialMetrics, err = suite.apiClient.GetMetrics(ctx, vm.ID)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Second)
	}
	require.NoError(t, err, "Should collect initial metrics")
	require.NotNil(t, initialMetrics)

	t.Log("Step 3: Monitor metrics over time")
	metricsHistory := make([]*VMMetrics, 0)
	
	for i := 0; i < 6; i++ { // Collect metrics for 1 minute
		metrics, err := suite.apiClient.GetMetrics(ctx, vm.ID)
		if err == nil {
			metricsHistory = append(metricsHistory, metrics)
		}
		time.Sleep(10 * time.Second)
	}

	require.GreaterOrEqual(t, len(metricsHistory), 3, "Should collect multiple metric samples")

	t.Log("Step 4: Verify metrics consistency")
	for i, metrics := range metricsHistory {
		assert.Equal(t, vm.ID, metrics.VMID, "Metrics should be for correct VM")
		assert.GreaterOrEqual(t, metrics.CPUUsage, 0.0, "CPU usage should be non-negative")
		assert.LessOrEqual(t, metrics.CPUUsage, 100.0, "CPU usage should not exceed 100%")
		assert.GreaterOrEqual(t, metrics.MemoryUsage, 0.0, "Memory usage should be non-negative")
		assert.LessOrEqual(t, metrics.MemoryUsage, 100.0, "Memory usage should not exceed 100%")
		
		if i > 0 {
			timeDiff := metrics.Timestamp.Sub(metricsHistory[i-1].Timestamp)
			assert.Greater(t, timeDiff, time.Duration(0), "Metrics timestamps should increase")
		}

		t.Logf("Metrics %d: CPU=%.1f%%, Memory=%.1f%%, DiskIO=%d, NetworkIO=%d",
			i, metrics.CPUUsage, metrics.MemoryUsage, metrics.DiskIO, metrics.NetworkIO)
	}

	t.Log("Step 5: Simulate high resource usage (if supported)")
	// This would ideally trigger some load on the VM
	// For now, we just verify that the monitoring system can handle the VM

	t.Log("Step 6: Verify alerting capabilities (placeholder)")
	// In a real implementation, this would test alert thresholds and notifications
	// For now, we verify that we can detect resource usage patterns

	avgCPU := 0.0
	for _, metrics := range metricsHistory {
		avgCPU += metrics.CPUUsage
	}
	avgCPU /= float64(len(metricsHistory))

	t.Logf("Average CPU usage over monitoring period: %.1f%%", avgCPU)
	
	// Basic alerting logic simulation
	if avgCPU > 80.0 {
		t.Log("🚨 High CPU usage detected - alert would be triggered")
	} else {
		t.Log("✅ CPU usage within normal range")
	}

	t.Log("Step 7: Cleanup monitoring VM")
	err = suite.apiClient.DeleteVM(ctx, vm.ID)
	assert.NoError(t, err, "Should delete monitoring VM")

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ Monitoring and alerting workflow completed successfully")
}

func (suite *WorkflowTestSuite) testAPIToUIWorkflow(t *testing.T) {
	start := time.Now()
	workflowName := "APIToUIWorkflow"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	if suite.webDriver == nil {
		t.Skip("Web driver not available")
	}

	ctx := context.Background()

	t.Log("Step 1: Create VM via API")
	createReq := &CreateVMRequest{
		Name:     "e2e-ui-vm",
		Template: "ubuntu-20.04",
		Resources: VMResources{
			CPU:    1,
			Memory: 1024,
			Disk:   20,
		},
		Metadata: map[string]interface{}{
			"test_type": "ui_integration",
			"workflow":  workflowName,
		},
	}

	vm, err := suite.apiClient.CreateVM(ctx, createReq)
	require.NoError(t, err, "VM creation should succeed")
	suite.testData.addCreatedVM(vm.ID)

	err = suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
	require.NoError(t, err, "VM should reach running state")

	t.Log("Step 2: Navigate to web UI")
	err = suite.webDriver.NavigateTo(suite.config.WebUIURL)
	require.NoError(t, err, "Should navigate to web UI")

	t.Log("Step 3: Verify VM appears in UI")
	vmListElement, err := suite.webDriver.WaitForElement("#vm-list", 10*time.Second)
	require.NoError(t, err, "Should find VM list in UI")

	vmRowSelector := fmt.Sprintf("[data-vm-id='%s']", vm.ID)
	vmRow, err := suite.webDriver.WaitForElement(vmRowSelector, 30*time.Second)
	assert.NoError(t, err, "VM should appear in UI list")

	if err == nil {
		vmName, err := vmRow.GetText()
		assert.NoError(t, err, "Should get VM name from UI")
		assert.Contains(t, vmName, vm.Name, "UI should display correct VM name")
	}

	t.Log("Step 4: Click on VM in UI")
	if vmRow != nil {
		err = vmRow.Click()
		assert.NoError(t, err, "Should be able to click VM in UI")

		// Wait for VM details page
		detailsElement, err := suite.webDriver.WaitForElement("#vm-details", 10*time.Second)
		assert.NoError(t, err, "Should navigate to VM details page")

		if detailsElement != nil {
			// Verify VM details are displayed
			vmIDElement, err := suite.webDriver.FindElement("#vm-id")
			if err == nil {
				displayedID, _ := vmIDElement.GetText()
				assert.Equal(t, vm.ID, displayedID, "UI should display correct VM ID")
			}
		}
	}

	t.Log("Step 5: Test UI operations")
	// Test basic UI operations like refresh, filter, etc.
	refreshButton, err := suite.webDriver.FindElement("#refresh-button")
	if err == nil {
		err = refreshButton.Click()
		assert.NoError(t, err, "Should be able to refresh UI")
	}

	t.Log("Step 6: Take screenshot for verification")
	screenshot, err := suite.webDriver.TakeScreenshot()
	if err == nil {
		suite.saveScreenshot(screenshot, fmt.Sprintf("ui_workflow_%s.png", vm.ID))
	}

	t.Log("Step 7: Delete VM via API and verify UI update")
	err = suite.apiClient.DeleteVM(ctx, vm.ID)
	assert.NoError(t, err, "Should delete VM via API")

	// Navigate back to VM list
	err = suite.webDriver.NavigateTo(suite.config.WebUIURL + "/vms")
	if err == nil {
		time.Sleep(5 * time.Second) // Allow UI to update

		// Verify VM is no longer in UI
		_, err = suite.webDriver.FindElement(vmRowSelector)
		assert.Error(t, err, "VM should no longer appear in UI after deletion")
	}

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ API to UI workflow completed successfully")
}

func (suite *WorkflowTestSuite) testLoadBalancingWorkflow(t *testing.T) {
	start := time.Now()
	workflowName := "LoadBalancingWorkflow"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()
	vmCount := 10

	t.Logf("Step 1: Create %d VMs to test load balancing", vmCount)
	var wg sync.WaitGroup
	var mu sync.Mutex
	createdVMs := make([]*VM, 0, vmCount)
	nodeDistribution := make(map[string]int)

	for i := 0; i < vmCount; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			createReq := &CreateVMRequest{
				Name:     fmt.Sprintf("e2e-load-vm-%d", index),
				Template: "ubuntu-20.04",
				Resources: VMResources{
					CPU:    1,
					Memory: 1024,
					Disk:   20,
				},
				// Don't specify NodeID - let scheduler decide
			}

			vm, err := suite.apiClient.CreateVM(ctx, createReq)
			if err != nil {
				t.Logf("Failed to create VM %d: %v", index, err)
				return
			}

			mu.Lock()
			createdVMs = append(createdVMs, vm)
			suite.testData.addCreatedVM(vm.ID)
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	require.GreaterOrEqual(t, len(createdVMs), vmCount/2, "At least half of VMs should be created")

	t.Log("Step 2: Wait for VMs and analyze distribution")
	for _, vm := range createdVMs {
		err := suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
		if err != nil {
			t.Logf("VM %s failed to start: %v", vm.ID, err)
			continue
		}

		// Get updated VM info with node assignment
		updatedVM, err := suite.apiClient.GetVM(ctx, vm.ID)
		if err == nil && updatedVM.NodeID != "" {
			nodeDistribution[updatedVM.NodeID]++
		}
	}

	t.Log("Step 3: Analyze load distribution")
	t.Logf("Node distribution: %+v", nodeDistribution)
	
	if len(nodeDistribution) > 1 {
		// Calculate distribution balance
		totalVMs := 0
		for _, count := range nodeDistribution {
			totalVMs += count
		}
		
		expectedPerNode := float64(totalVMs) / float64(len(nodeDistribution))
		maxDeviation := 0.0
		
		for nodeID, count := range nodeDistribution {
			deviation := float64(count) - expectedPerNode
			if deviation < 0 {
				deviation = -deviation
			}
			deviationPercent := deviation / expectedPerNode
			if deviationPercent > maxDeviation {
				maxDeviation = deviationPercent
			}
			
			t.Logf("Node %s: %d VMs (%.1f%% deviation from average)", 
				nodeID, count, deviationPercent*100)
		}
		
		// Load should be reasonably balanced (allow up to 50% deviation)
		assert.Less(t, maxDeviation, 0.5, "Load distribution should be reasonably balanced")
	} else {
		t.Log("Single node detected - load balancing test not applicable")
	}

	t.Log("Step 4: Test scheduler behavior under load")
	// Create additional VMs rapidly to test scheduler behavior
	rapidCreateCount := 5
	var rapidWg sync.WaitGroup
	
	for i := 0; i < rapidCreateCount; i++ {
		rapidWg.Add(1)
		go func(index int) {
			defer rapidWg.Done()

			createReq := &CreateVMRequest{
				Name:     fmt.Sprintf("e2e-rapid-vm-%d", index),
				Template: "ubuntu-20.04",
				Resources: VMResources{
					CPU:    2, // Larger resources
					Memory: 2048,
					Disk:   30,
				},
			}

			vm, err := suite.apiClient.CreateVM(ctx, createReq)
			if err == nil {
				mu.Lock()
				suite.testData.addCreatedVM(vm.ID)
				mu.Unlock()
			}
		}(i)
	}

	rapidWg.Wait()

	t.Log("Step 5: Cleanup all VMs")
	suite.testData.mu.RLock()
	allVMIDs := make([]string, len(suite.testData.CreatedVMs))
	copy(allVMIDs, suite.testData.CreatedVMs)
	suite.testData.mu.RUnlock()

	var cleanupWg sync.WaitGroup
	for _, vmID := range allVMIDs {
		cleanupWg.Add(1)
		go func(id string) {
			defer cleanupWg.Done()
			suite.apiClient.DeleteVM(ctx, id)
		}(vmID)
	}

	cleanupWg.Wait()

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ Load balancing workflow completed successfully")
}

func (suite *WorkflowTestSuite) testFailureRecoveryWorkflow(t *testing.T) {
	start := time.Now()
	workflowName := "FailureRecoveryWorkflow"
	
	defer func() {
		suite.mu.Lock()
		suite.metrics.WorkflowDurations[workflowName] = time.Since(start)
		suite.mu.Unlock()
	}()

	ctx := context.Background()

	t.Log("Step 1: Create VM for failure testing")
	createReq := &CreateVMRequest{
		Name:     "e2e-failure-vm",
		Template: "ubuntu-20.04",
		Resources: VMResources{
			CPU:    1,
			Memory: 1024,
			Disk:   20,
		},
		Metadata: map[string]interface{}{
			"test_type": "failure_recovery",
			"workflow":  workflowName,
		},
	}

	vm, err := suite.apiClient.CreateVM(ctx, createReq)
	require.NoError(t, err, "VM creation should succeed")
	suite.testData.addCreatedVM(vm.ID)

	err = suite.waitForVMStatus(ctx, vm.ID, "running", 5*time.Minute)
	require.NoError(t, err, "VM should reach running state")

	t.Log("Step 2: Test API resilience with invalid requests")
	// Test invalid VM operations
	invalidVM := "invalid-vm-id-12345"
	_, err = suite.apiClient.GetVM(ctx, invalidVM)
	assert.Error(t, err, "Should fail to get invalid VM")

	err = suite.apiClient.DeleteVM(ctx, invalidVM)
	assert.Error(t, err, "Should fail to delete invalid VM")

	// Test that valid operations still work after invalid ones
	validVM, err := suite.apiClient.GetVM(ctx, vm.ID)
	assert.NoError(t, err, "Should still get valid VM after invalid request")
	assert.Equal(t, vm.ID, validVM.ID)

	t.Log("Step 3: Test partial system failures")
	// Test with network timeouts (if supported)
	shortCtx, cancel := context.WithTimeout(ctx, 1*time.Millisecond)
	_, err = suite.apiClient.GetVM(shortCtx, vm.ID)
	cancel()
	assert.Error(t, err, "Should timeout with very short context")

	// Verify system recovers
	normalVM, err := suite.apiClient.GetVM(ctx, vm.ID)
	assert.NoError(t, err, "Should work with normal timeout")
	assert.Equal(t, vm.ID, normalVM.ID)

	t.Log("Step 4: Test concurrent failure scenarios")
	var concurrentWg sync.WaitGroup
	var mu sync.Mutex
	failureResults := make(map[string]int)

	// Simulate concurrent operations with some expected to fail
	for i := 0; i < 10; i++ {
		concurrentWg.Add(1)
		go func(index int) {
			defer concurrentWg.Done()

			if index%3 == 0 {
				// Try invalid operation
				_, err := suite.apiClient.GetVM(ctx, fmt.Sprintf("invalid-%d", index))
				mu.Lock()
				if err != nil {
					failureResults["expected_failures"]++
				}
				mu.Unlock()
			} else {
				// Valid operation
				_, err := suite.apiClient.GetVM(ctx, vm.ID)
				mu.Lock()
				if err != nil {
					failureResults["unexpected_failures"]++
				} else {
					failureResults["successes"]++
				}
				mu.Unlock()
			}
		}(i)
	}

	concurrentWg.Wait()

	t.Logf("Concurrent operation results: %+v", failureResults)
	assert.GreaterOrEqual(t, failureResults["successes"], 6, "Most valid operations should succeed")
	assert.GreaterOrEqual(t, failureResults["expected_failures"], 3, "Invalid operations should fail as expected")
	assert.LessOrEqual(t, failureResults["unexpected_failures"], 1, "Valid operations should rarely fail")

	t.Log("Step 5: Test system stability after failures")
	// Verify the system is stable after the failure scenarios
	for i := 0; i < 5; i++ {
		_, err := suite.apiClient.GetVM(ctx, vm.ID)
		assert.NoError(t, err, "System should be stable after failures (attempt %d)", i+1)
		
		metrics, err := suite.apiClient.GetMetrics(ctx, vm.ID)
		assert.NoError(t, err, "Metrics should be available after failures (attempt %d)", i+1)
		assert.Equal(t, vm.ID, metrics.VMID)
		
		time.Sleep(2 * time.Second)
	}

	t.Log("Step 6: Cleanup failure test VM")
	err = suite.apiClient.DeleteVM(ctx, vm.ID)
	assert.NoError(t, err, "Should delete failure test VM")

	suite.metrics.SuccessfulWorkflows++
	t.Log("✅ Failure recovery workflow completed successfully")
}

// Helper methods
func (suite *WorkflowTestSuite) waitForVMStatus(ctx context.Context, vmID, expectedStatus string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for VM %s to reach status %s", vmID, expectedStatus)
		case <-ticker.C:
			vm, err := suite.apiClient.GetVM(ctx, vmID)
			if err != nil {
				if expectedStatus == "deleted" && strings.Contains(err.Error(), "not found") {
					return nil // VM is deleted
				}
				continue
			}

			if vm.Status == expectedStatus {
				return nil
			}

			// Handle special cases
			if vm.Status == "error" || vm.Status == "failed" {
				return fmt.Errorf("VM %s entered error state: %s", vmID, vm.Status)
			}
		}
	}
}

func (suite *WorkflowTestSuite) waitForMigrationCompletion(ctx context.Context, jobID string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for migration %s to complete", jobID)
		case <-ticker.C:
			job, err := suite.apiClient.GetMigrationStatus(ctx, jobID)
			if err != nil {
				continue
			}

			switch job.Status {
			case "completed":
				return nil
			case "failed", "error":
				return fmt.Errorf("migration %s failed: %s", jobID, job.Error)
			default:
				// Still in progress
			}
		}
	}
}

func (suite *WorkflowTestSuite) getAlternativeNode(sourceNode string) string {
	// In a real implementation, this would query available nodes
	// and return a different node than the source
	if sourceNode == "node-1" {
		return "node-2"
	}
	return "node-1"
}

func (suite *WorkflowTestSuite) saveScreenshot(data []byte, filename string) {
	// Save screenshot to test results directory
	// Implementation would save the screenshot file
}

func (suite *WorkflowTestSuite) generateWorkflowReport(t *testing.T) {
	suite.mu.RLock()
	defer suite.mu.RUnlock()

	t.Log("\n" + strings.Repeat("=", 60))
	t.Log("📊 E2E Workflow Test Summary")
	t.Log(strings.Repeat("=", 60))
	
	t.Logf("Total Workflows: %d", suite.metrics.TotalWorkflows)
	t.Logf("Successful: %d", suite.metrics.SuccessfulWorkflows)
	t.Logf("Failed: %d", suite.metrics.FailedWorkflows)
	
	if suite.metrics.TotalWorkflows > 0 {
		successRate := float64(suite.metrics.SuccessfulWorkflows) / float64(suite.metrics.TotalWorkflows) * 100
		t.Logf("Success Rate: %.1f%%", successRate)
	}

	t.Log("\nWorkflow Durations:")
	for workflow, duration := range suite.metrics.WorkflowDurations {
		t.Logf("  %s: %v", workflow, duration)
	}

	if len(suite.metrics.ErrorCounts) > 0 {
		t.Log("\nError Counts:")
		for errorType, count := range suite.metrics.ErrorCounts {
			t.Logf("  %s: %d", errorType, count)
		}
	}

	// Test data cleanup summary
	suite.testData.mu.RLock()
	t.Logf("\nTest Data Created:")
	t.Logf("  VMs: %d", len(suite.testData.CreatedVMs))
	t.Logf("  Migration Jobs: %d", len(suite.testData.MigrationJobs))
	t.Logf("  Networks: %d", len(suite.testData.CreatedNetworks))
	suite.testData.mu.RUnlock()

	t.Log(strings.Repeat("=", 60))
}

func (suite *WorkflowTestSuite) cleanup() {
	// Cleanup any remaining test resources
	ctx := context.Background()
	
	suite.testData.mu.RLock()
	vmsToCleanup := make([]string, len(suite.testData.CreatedVMs))
	copy(vmsToCleanup, suite.testData.CreatedVMs)
	suite.testData.mu.RUnlock()

	for _, vmID := range vmsToCleanup {
		suite.apiClient.DeleteVM(ctx, vmID)
	}

	if suite.webDriver != nil {
		suite.webDriver.Close()
	}
}

// TestDataManager methods
func (tdm *TestDataManager) addCreatedVM(vmID string) {
	tdm.mu.Lock()
	defer tdm.mu.Unlock()
	tdm.CreatedVMs = append(tdm.CreatedVMs, vmID)
}

func (tdm *TestDataManager) addMigrationJob(jobID string) {
	tdm.mu.Lock()
	defer tdm.mu.Unlock()
	tdm.MigrationJobs = append(tdm.MigrationJobs, jobID)
}

// Configuration helper
func getE2ETestConfig() *E2ETestConfig {
	return &E2ETestConfig{
		BaseURL:           getEnvOrDefault("NOVACRON_API_URL", "http://localhost:8090"),
		WebUIURL:          getEnvOrDefault("NOVACRON_UI_URL", "http://localhost:8092"),
		APIKey:            getEnvOrDefault("NOVACRON_API_KEY", "test-api-key"),
		TestTimeout:       5 * time.Minute,
		EnableUITests:     getEnvOrDefault("ENABLE_UI_TESTS", "false") == "true",
		EnableLoadTests:   getEnvOrDefault("ENABLE_LOAD_TESTS", "false") == "true",
		CleanupOnFailure:  true,
		ParallelExecution: getEnvOrDefault("PARALLEL_EXECUTION", "true") == "true",
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}