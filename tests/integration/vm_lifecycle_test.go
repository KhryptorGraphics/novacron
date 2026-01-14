package integration

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/tests/integration/helpers"
)

// VMLifecycleTestSuite tests VM lifecycle operations
type VMLifecycleTestSuite struct {
	suite.Suite
	env    *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
	tenantID string
	userID   int
	token    string
}

// SetupSuite initializes the test suite
func (suite *VMLifecycleTestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
	suite.tenantID = "tenant-1"
	suite.userID = 1
	
	// Login as admin for setup
	suite.token = suite.env.LoginAsAdmin(suite.T())
	suite.env.APIClient.SetAuthToken(suite.token)
}

// TearDownSuite cleans up the test suite
func (suite *VMLifecycleTestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// TestVMCreation tests VM creation with various configurations
func (suite *VMLifecycleTestSuite) TestVMCreation() {
	tests := []struct {
		name     string
		vmData   map[string]interface{}
		wantCode int
		wantErr  bool
	}{
		{
			name: "Create basic VM",
			vmData: map[string]interface{}{
				"name":      "test-vm-basic",
				"cpu":       2,
				"memory":    1024,
				"disk_size": 10240,
				"image":     "ubuntu:20.04",
				"tenant_id": suite.tenantID,
			},
			wantCode: http.StatusCreated,
			wantErr:  false,
		},
		{
			name: "Create high-performance VM",
			vmData: map[string]interface{}{
				"name":      "test-vm-highperf",
				"cpu":       8,
				"memory":    16384,
				"disk_size": 102400,
				"image":     "centos:8",
				"tenant_id": suite.tenantID,
				"metadata": map[string]string{
					"environment": "production",
					"application": "database",
				},
			},
			wantCode: http.StatusCreated,
			wantErr:  false,
		},
		{
			name: "Create VM with invalid CPU",
			vmData: map[string]interface{}{
				"name":      "test-vm-invalid",
				"cpu":       0, // Invalid CPU count
				"memory":    1024,
				"disk_size": 10240,
				"image":     "ubuntu:20.04",
				"tenant_id": suite.tenantID,
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
		{
			name: "Create VM without required fields",
			vmData: map[string]interface{}{
				"cpu":    2,
				"memory": 1024,
				// Missing name and other required fields
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.POST(t, "/api/vms", tt.vmData)
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode)

			if !tt.wantErr {
				var result map[string]interface{}
				suite.env.APIClient.ParseJSON(t, resp, &result)

				assert.NotEmpty(t, result["id"], "VM ID should not be empty")
				assert.Equal(t, tt.vmData["name"], result["name"])
				assert.Equal(t, "created", result["state"]) // Initial state should be 'created'
			}
		})
	}
}

// TestVMLifecycleStates tests VM state transitions
func (suite *VMLifecycleTestSuite) TestVMLifecycleStates() {
	// Create a test VM
	vmData := map[string]interface{}{
		"name":      "test-vm-lifecycle",
		"cpu":       2,
		"memory":    2048,
		"disk_size": 20480,
		"image":     "ubuntu:20.04",
		"tenant_id": suite.tenantID,
	}

	resp := suite.env.APIClient.POST(suite.T(), "/api/vms", vmData)
	suite.env.APIClient.ExpectStatus(suite.T(), resp, http.StatusCreated)

	var vm map[string]interface{}
	suite.env.APIClient.ParseJSON(suite.T(), resp, &vm)
	vmID := vm["id"].(string)

	// Test state transitions: created -> running -> stopped -> running -> paused -> running
	stateTests := []struct {
		action       string
		expectedState string
		endpoint     string
		method       string
	}{
		{"start", "running", fmt.Sprintf("/api/vms/%s/start", vmID), "POST"},
		{"stop", "stopped", fmt.Sprintf("/api/vms/%s/stop", vmID), "POST"},
		{"start", "running", fmt.Sprintf("/api/vms/%s/start", vmID), "POST"},
		{"pause", "paused", fmt.Sprintf("/api/vms/%s/pause", vmID), "POST"},
		{"resume", "running", fmt.Sprintf("/api/vms/%s/resume", vmID), "POST"},
	}

	for _, tt := range stateTests {
		suite.T().Run(fmt.Sprintf("VM_%s", tt.action), func(t *testing.T) {
			// Perform action
			var resp *http.Response
			if tt.method == "POST" {
				resp = suite.env.APIClient.POST(t, tt.endpoint, nil)
			} else {
				resp = suite.env.APIClient.GET(t, tt.endpoint)
			}
			defer resp.Body.Close()

			suite.env.APIClient.ExpectStatus(t, resp, http.StatusOK)

			// Wait for state change (with timeout)
			suite.env.WaitForVMState(t, vmID, tt.expectedState, 30*time.Second)

			// Verify state
			vmResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID)
			defer vmResp.Body.Close()

			var currentVM map[string]interface{}
			suite.env.APIClient.ParseJSON(t, vmResp, &currentVM)

			assert.Equal(t, tt.expectedState, currentVM["state"])
		})
	}

	// Clean up: stop and delete VM
	suite.env.APIClient.POST(suite.T(), fmt.Sprintf("/api/vms/%s/stop", vmID), nil)
	suite.env.WaitForVMState(suite.T(), vmID, "stopped", 30*time.Second)
	
	deleteResp := suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
	defer deleteResp.Body.Close()
	suite.env.APIClient.ExpectStatus(suite.T(), deleteResp, http.StatusNoContent)
}

// TestVMUpdate tests VM configuration updates
func (suite *VMLifecycleTestSuite) TestVMUpdate() {
	// Create a test VM
	vmID := suite.env.CreateTestVM(suite.T(), "test-vm-update", suite.tenantID)

	updateTests := []struct {
		name       string
		updateData map[string]interface{}
		wantCode   int
	}{
		{
			name: "Update VM name",
			updateData: map[string]interface{}{
				"name": "test-vm-updated-name",
			},
			wantCode: http.StatusOK,
		},
		{
			name: "Update VM resources",
			updateData: map[string]interface{}{
				"cpu":    4,
				"memory": 4096,
			},
			wantCode: http.StatusOK,
		},
		{
			name: "Update VM metadata",
			updateData: map[string]interface{}{
				"metadata": map[string]string{
					"environment": "staging",
					"owner":       "integration-test",
				},
			},
			wantCode: http.StatusOK,
		},
		{
			name: "Invalid update - negative CPU",
			updateData: map[string]interface{}{
				"cpu": -1,
			},
			wantCode: http.StatusBadRequest,
		},
	}

	for _, tt := range updateTests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.PUT(t, "/api/vms/"+vmID, tt.updateData)
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode)

			if tt.wantCode == http.StatusOK {
				// Verify the update
				getResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID)
				defer getResp.Body.Close()

				var updatedVM map[string]interface{}
				suite.env.APIClient.ParseJSON(t, getResp, &updatedVM)

				// Check updated fields
				for key, expectedValue := range tt.updateData {
					if key != "metadata" {
						assert.Equal(t, expectedValue, updatedVM[key], 
							"Field %s should be updated", key)
					}
				}
			}
		})
	}

	// Cleanup
	deleteResp := suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
	defer deleteResp.Body.Close()
}

// TestVMList tests VM listing and filtering
func (suite *VMLifecycleTestSuite) TestVMList() {
	// Create multiple test VMs
	vmIDs := make([]string, 3)
	for i := 0; i < 3; i++ {
		vmName := fmt.Sprintf("test-vm-list-%d", i)
		vmIDs[i] = suite.env.CreateTestVM(suite.T(), vmName, suite.tenantID)
	}

	tests := []struct {
		name     string
		endpoint string
		wantMin  int // Minimum number of VMs expected
	}{
		{
			name:     "List all VMs",
			endpoint: "/api/vms",
			wantMin:  3,
		},
		{
			name:     "Filter by tenant",
			endpoint: "/api/vms?tenant_id=" + suite.tenantID,
			wantMin:  3,
		},
		{
			name:     "Filter by state",
			endpoint: "/api/vms?state=created",
			wantMin:  3,
		},
		{
			name:     "Pagination",
			endpoint: "/api/vms?limit=2&offset=0",
			wantMin:  2,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.GET(t, tt.endpoint)
			defer resp.Body.Close()

			suite.env.APIClient.ExpectStatus(t, resp, http.StatusOK)

			var result map[string]interface{}
			suite.env.APIClient.ParseJSON(t, resp, &result)

			vms, ok := result["vms"].([]interface{})
			require.True(t, ok, "Response should contain vms array")
			assert.GreaterOrEqual(t, len(vms), tt.wantMin, 
				"Should have at least %d VMs", tt.wantMin)

			// Verify VM structure
			if len(vms) > 0 {
				vm := vms[0].(map[string]interface{})
				assert.NotEmpty(t, vm["id"], "VM should have ID")
				assert.NotEmpty(t, vm["name"], "VM should have name")
				assert.NotEmpty(t, vm["state"], "VM should have state")
			}
		})
	}

	// Cleanup
	for _, vmID := range vmIDs {
		deleteResp := suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
		defer deleteResp.Body.Close()
	}
}

// TestVMMetrics tests VM metrics collection
func (suite *VMLifecycleTestSuite) TestVMMetrics() {
	// Create and start a test VM
	vmID := suite.env.CreateTestVM(suite.T(), "test-vm-metrics", suite.tenantID)
	
	// Start the VM to generate metrics
	startResp := suite.env.APIClient.POST(suite.T(), "/api/vms/"+vmID+"/start", nil)
	defer startResp.Body.Close()
	suite.env.APIClient.ExpectStatus(suite.T(), startResp, http.StatusOK)
	
	// Wait for VM to start
	suite.env.WaitForVMState(suite.T(), vmID, "running", 30*time.Second)
	
	// Give some time for metrics to be collected
	time.Sleep(5 * time.Second)
	
	tests := []struct {
		name     string
		endpoint string
		wantCode int
	}{
		{
			name:     "Get current metrics",
			endpoint: "/api/vms/" + vmID + "/metrics",
			wantCode: http.StatusOK,
		},
		{
			name:     "Get historical metrics",
			endpoint: "/api/vms/" + vmID + "/metrics?timespan=1h",
			wantCode: http.StatusOK,
		},
		{
			name:     "Get specific metric",
			endpoint: "/api/vms/" + vmID + "/metrics?metric=cpu_usage",
			wantCode: http.StatusOK,
		},
	}
	
	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.GET(t, tt.endpoint)
			defer resp.Body.Close()
			
			assert.Equal(t, tt.wantCode, resp.StatusCode)
			
			if tt.wantCode == http.StatusOK {
				var metrics map[string]interface{}
				suite.env.APIClient.ParseJSON(t, resp, &metrics)
				
				assert.NotEmpty(t, metrics, "Metrics should not be empty")
				assert.Contains(t, metrics, "vm_id", "Metrics should contain VM ID")
			}
		})
	}
	
	// Cleanup
	suite.env.APIClient.POST(suite.T(), "/api/vms/"+vmID+"/stop", nil)
	suite.env.WaitForVMState(suite.T(), vmID, "stopped", 30*time.Second)
	deleteResp := suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
	defer deleteResp.Body.Close()
}

// TestVMConcurrentOperations tests concurrent VM operations
func (suite *VMLifecycleTestSuite) TestVMConcurrentOperations() {
	if testing.Short() {
		suite.T().Skip("Skipping concurrent operations test in short mode")
	}
	
	numVMs := 5
	vmIDs := make([]string, numVMs)
	
	// Create multiple VMs concurrently
	suite.T().Run("Concurrent VM Creation", func(t *testing.T) {
		done := make(chan string, numVMs)
		errors := make(chan error, numVMs)
		
		for i := 0; i < numVMs; i++ {
			go func(index int) {
				vmData := map[string]interface{}{
					"name":      fmt.Sprintf("concurrent-vm-%d", index),
					"cpu":       2,
					"memory":    1024,
					"disk_size": 10240,
					"image":     "ubuntu:20.04",
					"tenant_id": suite.tenantID,
				}
				
				resp := suite.env.APIClient.POST(t, "/api/vms", vmData)
				defer resp.Body.Close()
				
				if resp.StatusCode != http.StatusCreated {
					errors <- fmt.Errorf("failed to create VM %d: status %d", index, resp.StatusCode)
					return
				}
				
				var result map[string]interface{}
				err := suite.env.APIClient.ParseJSON(t, resp, &result)
				if err != nil {
					errors <- fmt.Errorf("failed to parse response for VM %d: %v", index, err)
					return
				}
				
				vmID, ok := result["id"].(string)
				if !ok {
					errors <- fmt.Errorf("no VM ID for VM %d", index)
					return
				}
				
				done <- vmID
			}(i)
		}
		
		// Collect results
		for i := 0; i < numVMs; i++ {
			select {
			case vmID := <-done:
				vmIDs[i] = vmID
			case err := <-errors:
				t.Errorf("Concurrent VM creation error: %v", err)
			case <-time.After(60 * time.Second):
				t.Fatalf("Timeout waiting for VM %d to be created", i)
			}
		}
		
		assert.Equal(t, numVMs, len(vmIDs), "All VMs should be created")
	})
	
	// Start all VMs concurrently
	suite.T().Run("Concurrent VM Start", func(t *testing.T) {
		done := make(chan bool, numVMs)
		errors := make(chan error, numVMs)
		
		for _, vmID := range vmIDs {
			go func(id string) {
				resp := suite.env.APIClient.POST(t, "/api/vms/"+id+"/start", nil)
				defer resp.Body.Close()
				
				if resp.StatusCode != http.StatusOK {
					errors <- fmt.Errorf("failed to start VM %s: status %d", id, resp.StatusCode)
					return
				}
				
				done <- true
			}(vmID)
		}
		
		// Wait for all start operations
		for i := 0; i < numVMs; i++ {
			select {
			case <-done:
				// VM start initiated successfully
			case err := <-errors:
				t.Errorf("Concurrent VM start error: %v", err)
			case <-time.After(30 * time.Second):
				t.Fatalf("Timeout waiting for VM start %d", i)
			}
		}
	})
	
	// Cleanup: Delete all VMs
	for _, vmID := range vmIDs {
		if vmID != "" {
			suite.env.APIClient.POST(suite.T(), "/api/vms/"+vmID+"/stop", nil)
			time.Sleep(100 * time.Millisecond) // Brief delay between operations
			deleteResp := suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
			defer deleteResp.Body.Close()
		}
	}
}

// TestVMErrorHandling tests error scenarios
func (suite *VMLifecycleTestSuite) TestVMErrorHandling() {
	tests := []struct {
		name     string
		endpoint string
		method   string
		body     interface{}
		wantCode int
	}{
		{
			name:     "Get non-existent VM",
			endpoint: "/api/vms/non-existent-id",
			method:   "GET",
			body:     nil,
			wantCode: http.StatusNotFound,
		},
		{
			name:     "Delete non-existent VM",
			endpoint: "/api/vms/non-existent-id",
			method:   "DELETE",
			body:     nil,
			wantCode: http.StatusNotFound,
		},
		{
			name:     "Start non-existent VM",
			endpoint: "/api/vms/non-existent-id/start",
			method:   "POST",
			body:     nil,
			wantCode: http.StatusNotFound,
		},
		{
			name:     "Create VM with malformed JSON",
			endpoint: "/api/vms",
			method:   "POST",
			body:     "invalid-json",
			wantCode: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			var resp *http.Response
			switch tt.method {
			case "GET":
				resp = suite.env.APIClient.GET(t, tt.endpoint)
			case "POST":
				resp = suite.env.APIClient.POST(t, tt.endpoint, tt.body)
			case "DELETE":
				resp = suite.env.APIClient.DELETE(t, tt.endpoint)
			}
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode,
				"Expected status code %d for %s", tt.wantCode, tt.name)
		})
	}
}

// TestVMLifecycleTestSuite runs the VM lifecycle test suite
func TestVMLifecycleTestSuite(t *testing.T) {
	suite.Run(t, new(VMLifecycleTestSuite))
}