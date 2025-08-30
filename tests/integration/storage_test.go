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

// StorageTestSuite tests storage operations and tiering functionality
type StorageTestSuite struct {
	suite.Suite
	env     *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
	token   string
	tenantID string
}

// SetupSuite initializes the test suite
func (suite *StorageTestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
	suite.tenantID = "tenant-1"
	
	// Login as admin
	suite.token = suite.env.LoginAsAdmin(suite.T())
	suite.env.APIClient.SetAuthToken(suite.token)
}

// TearDownSuite cleans up the test suite
func (suite *StorageTestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// TestStorageTierManagement tests storage tier CRUD operations
func (suite *StorageTestSuite) TestStorageTierManagement() {
	// Test creating storage tiers
	suite.T().Run("Create Storage Tiers", func(t *testing.T) {
		tiers := []map[string]interface{}{
			{
				"name":               "test-standard",
				"storage_class":      "HDD",
				"performance_tier":   "standard",
				"cost_per_gb_month":  0.045,
				"iops":               100,
				"throughput_mbps":    125,
				"description":        "Standard HDD storage for general workloads",
			},
			{
				"name":               "test-premium",
				"storage_class":      "SSD",
				"performance_tier":   "high",
				"cost_per_gb_month":  0.125,
				"iops":               500,
				"throughput_mbps":    500,
				"description":        "Premium SSD storage for high-performance workloads",
			},
			{
				"name":               "test-ultra",
				"storage_class":      "NVMe",
				"performance_tier":   "ultra",
				"cost_per_gb_month":  0.300,
				"iops":               2000,
				"throughput_mbps":    1000,
				"description":        "Ultra-high performance NVMe storage",
			},
		}

		var createdTierIDs []string
		
		for _, tierData := range tiers {
			resp := suite.env.APIClient.POST(t, "/api/storage/tiers", tierData)
			defer resp.Body.Close()
			
			suite.env.APIClient.ExpectStatus(t, resp, http.StatusCreated)
			
			var result map[string]interface{}
			suite.env.APIClient.ParseJSON(t, resp, &result)
			
			assert.NotEmpty(t, result["id"], "Storage tier ID should not be empty")
			assert.Equal(t, tierData["name"], result["name"])
			assert.Equal(t, tierData["storage_class"], result["storage_class"])
			
			createdTierIDs = append(createdTierIDs, fmt.Sprintf("%.0f", result["id"].(float64)))
		}

		// Test listing storage tiers
		listResp := suite.env.APIClient.GET(t, "/api/storage/tiers")
		defer listResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, listResp, http.StatusOK)
		
		var listResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, listResp, &listResult)
		
		tiersList, ok := listResult["tiers"].([]interface{})
		require.True(t, ok, "Response should contain tiers array")
		assert.GreaterOrEqual(t, len(tiersList), 3, "Should have at least 3 storage tiers")
		
		// Cleanup - delete created tiers
		for _, tierID := range createdTierIDs {
			deleteResp := suite.env.APIClient.DELETE(t, "/api/storage/tiers/"+tierID)
			defer deleteResp.Body.Close()
		}
	})
}

// TestStorageVolumeOperations tests storage volume lifecycle
func (suite *StorageTestSuite) TestStorageVolumeOperations() {
	// First, create a storage tier for testing
	tierData := map[string]interface{}{
		"name":               "test-volume-tier",
		"storage_class":      "SSD",
		"performance_tier":   "high",
		"cost_per_gb_month":  0.125,
		"iops":               500,
		"throughput_mbps":    500,
	}
	
	tierResp := suite.env.APIClient.POST(suite.T(), "/api/storage/tiers", tierData)
	defer tierResp.Body.Close()
	suite.env.APIClient.ExpectStatus(suite.T(), tierResp, http.StatusCreated)
	
	var tierResult map[string]interface{}
	suite.env.APIClient.ParseJSON(suite.T(), tierResp, &tierResult)
	tierID := fmt.Sprintf("%.0f", tierResult["id"].(float64))
	
	suite.T().Run("Volume Lifecycle", func(t *testing.T) {
		// Create a storage volume
		volumeData := map[string]interface{}{
			"name":        "test-storage-volume",
			"size_gb":     100,
			"tier_id":     tierID,
			"tenant_id":   suite.tenantID,
			"description": "Test storage volume for integration tests",
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/storage/volumes", volumeData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var volumeResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &volumeResult)
		
		volumeID := volumeResult["id"].(string)
		assert.NotEmpty(t, volumeID, "Volume ID should not be empty")
		assert.Equal(t, volumeData["name"], volumeResult["name"])
		assert.Equal(t, float64(volumeData["size_gb"].(int)), volumeResult["size_gb"])
		
		// Test volume retrieval
		getResp := suite.env.APIClient.GET(t, "/api/storage/volumes/"+volumeID)
		defer getResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, getResp, http.StatusOK)
		
		var getResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, getResp, &getResult)
		assert.Equal(t, volumeID, getResult["id"])
		
		// Test volume resize
		resizeData := map[string]interface{}{
			"size_gb": 200,
		}
		
		resizeResp := suite.env.APIClient.PUT(t, "/api/storage/volumes/"+volumeID+"/resize", resizeData)
		defer resizeResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, resizeResp, http.StatusOK)
		
		// Verify resize
		verifyResp := suite.env.APIClient.GET(t, "/api/storage/volumes/"+volumeID)
		defer verifyResp.Body.Close()
		
		var verifyResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, verifyResp, &verifyResult)
		assert.Equal(t, float64(200), verifyResult["size_gb"], "Volume should be resized to 200GB")
		
		// Test volume attachment to VM
		vmID := suite.env.CreateTestVM(t, "test-vm-for-storage", suite.tenantID)
		
		attachData := map[string]interface{}{
			"vm_id": vmID,
			"mount_point": "/mnt/test-volume",
		}
		
		attachResp := suite.env.APIClient.POST(t, "/api/storage/volumes/"+volumeID+"/attach", attachData)
		defer attachResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, attachResp, http.StatusOK)
		
		// Verify attachment
		attachedResp := suite.env.APIClient.GET(t, "/api/storage/volumes/"+volumeID)
		defer attachedResp.Body.Close()
		
		var attachedResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, attachedResp, &attachedResult)
		assert.Equal(t, vmID, attachedResult["attached_vm_id"])
		
		// Test volume detachment
		detachResp := suite.env.APIClient.POST(t, "/api/storage/volumes/"+volumeID+"/detach", nil)
		defer detachResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, detachResp, http.StatusOK)
		
		// Cleanup
		suite.env.APIClient.DELETE(t, "/api/storage/volumes/"+volumeID)
		suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
	})
	
	// Cleanup tier
	suite.env.APIClient.DELETE(suite.T(), "/api/storage/tiers/"+tierID)
}

// TestStorageSnapshots tests snapshot functionality
func (suite *StorageTestSuite) TestStorageSnapshots() {
	// Create a VM and volume for snapshot testing
	vmID := suite.env.CreateTestVM(suite.T(), "test-vm-snapshots", suite.tenantID)
	
	suite.T().Run("Snapshot Operations", func(t *testing.T) {
		// Create VM snapshot
		snapshotData := map[string]interface{}{
			"name":        "test-snapshot-1",
			"description": "Test snapshot for integration tests",
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/snapshots", snapshotData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var snapshotResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &snapshotResult)
		
		snapshotID := snapshotResult["id"].(string)
		assert.NotEmpty(t, snapshotID, "Snapshot ID should not be empty")
		assert.Equal(t, snapshotData["name"], snapshotResult["name"])
		assert.Equal(t, vmID, snapshotResult["vm_id"])
		
		// Wait for snapshot to complete
		timeout := time.Now().Add(60 * time.Second)
		for time.Now().Before(timeout) {
			statusResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID+"/snapshots/"+snapshotID)
			defer statusResp.Body.Close()
			
			var statusResult map[string]interface{}
			suite.env.APIClient.ParseJSON(t, statusResp, &statusResult)
			
			if statusResult["status"] == "completed" {
				break
			}
			
			time.Sleep(2 * time.Second)
		}
		
		// List snapshots
		listResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID+"/snapshots")
		defer listResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, listResp, http.StatusOK)
		
		var listResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, listResp, &listResult)
		
		snapshots, ok := listResult["snapshots"].([]interface{})
		require.True(t, ok, "Response should contain snapshots array")
		assert.GreaterOrEqual(t, len(snapshots), 1, "Should have at least 1 snapshot")
		
		// Test snapshot restoration
		restoreData := map[string]interface{}{
			"snapshot_id": snapshotID,
		}
		
		restoreResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/restore", restoreData)
		defer restoreResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, restoreResp, http.StatusOK)
		
		// Delete snapshot
		deleteResp := suite.env.APIClient.DELETE(t, "/api/vms/"+vmID+"/snapshots/"+snapshotID)
		defer deleteResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, deleteResp, http.StatusNoContent)
	})
	
	// Cleanup VM
	suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
}

// TestStorageMetrics tests storage performance metrics
func (suite *StorageTestSuite) TestStorageMetrics() {
	// Create a VM for metrics testing
	vmID := suite.env.CreateTestVM(suite.T(), "test-vm-storage-metrics", suite.tenantID)
	
	// Start the VM to generate storage metrics
	startResp := suite.env.APIClient.POST(suite.T(), "/api/vms/"+vmID+"/start", nil)
	defer startResp.Body.Close()
	suite.env.APIClient.ExpectStatus(suite.T(), startResp, http.StatusOK)
	
	suite.env.WaitForVMState(suite.T(), vmID, "running", 30*time.Second)
	
	// Wait for metrics to be collected
	time.Sleep(10 * time.Second)
	
	suite.T().Run("Storage Performance Metrics", func(t *testing.T) {
		tests := []struct {
			name     string
			endpoint string
			wantCode int
		}{
			{
				name:     "VM storage metrics",
				endpoint: "/api/vms/" + vmID + "/metrics/storage",
				wantCode: http.StatusOK,
			},
			{
				name:     "Storage I/O metrics",
				endpoint: "/api/vms/" + vmID + "/metrics/storage?metric=iops",
				wantCode: http.StatusOK,
			},
			{
				name:     "Storage throughput metrics",
				endpoint: "/api/vms/" + vmID + "/metrics/storage?metric=throughput",
				wantCode: http.StatusOK,
			},
			{
				name:     "Storage latency metrics",
				endpoint: "/api/vms/" + vmID + "/metrics/storage?metric=latency",
				wantCode: http.StatusOK,
			},
		}
		
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				resp := suite.env.APIClient.GET(t, tt.endpoint)
				defer resp.Body.Close()
				
				assert.Equal(t, tt.wantCode, resp.StatusCode)
				
				if tt.wantCode == http.StatusOK {
					var metrics map[string]interface{}
					suite.env.APIClient.ParseJSON(t, resp, &metrics)
					
					assert.NotEmpty(t, metrics, "Metrics should not be empty")
					assert.Contains(t, metrics, "vm_id", "Metrics should contain VM ID")
					assert.Equal(t, vmID, metrics["vm_id"], "Metrics should be for correct VM")
				}
			})
		}
	})
	
	// Cleanup
	suite.env.APIClient.POST(suite.T(), "/api/vms/"+vmID+"/stop", nil)
	suite.env.WaitForVMState(suite.T(), vmID, "stopped", 30*time.Second)
	suite.env.APIClient.DELETE(suite.T(), "/api/vms/"+vmID)
}

// TestStorageTieringPolicies tests automated storage tiering
func (suite *StorageTestSuite) TestStorageTieringPolicies() {
	suite.T().Run("Storage Tiering Policies", func(t *testing.T) {
		// Create tiering policy
		policyData := map[string]interface{}{
			"name":        "test-tiering-policy",
			"description": "Test automated storage tiering policy",
			"rules": []map[string]interface{}{
				{
					"condition": "age > 30d AND access_frequency < 0.1",
					"action":    "move_to_tier",
					"target_tier": "archive",
				},
				{
					"condition": "iops > 1000 AND latency > 10ms",
					"action":    "move_to_tier", 
					"target_tier": "ultra",
				},
			},
			"tenant_id": suite.tenantID,
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/storage/tiering/policies", policyData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var policyResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &policyResult)
		
		policyID := policyResult["id"].(string)
		assert.NotEmpty(t, policyID, "Policy ID should not be empty")
		assert.Equal(t, policyData["name"], policyResult["name"])
		
		// Test policy retrieval
		getResp := suite.env.APIClient.GET(t, "/api/storage/tiering/policies/"+policyID)
		defer getResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, getResp, http.StatusOK)
		
		// Test policy execution (simulation)
		executeResp := suite.env.APIClient.POST(t, "/api/storage/tiering/policies/"+policyID+"/execute", nil)
		defer executeResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, executeResp, http.StatusOK)
		
		var executeResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, executeResp, &executeResult)
		
		assert.Contains(t, executeResult, "job_id", "Should return job ID for policy execution")
		
		// Cleanup
		deleteResp := suite.env.APIClient.DELETE(t, "/api/storage/tiering/policies/"+policyID)
		defer deleteResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, deleteResp, http.StatusNoContent)
	})
}

// TestStorageQuotaManagement tests storage quota enforcement
func (suite *StorageTestSuite) TestStorageQuotaManagement() {
	suite.T().Run("Storage Quota Management", func(t *testing.T) {
		// Set storage quota for tenant
		quotaData := map[string]interface{}{
			"tenant_id":     suite.tenantID,
			"storage_limit": 1000, // 1000 GB limit
		}
		
		setQuotaResp := suite.env.APIClient.POST(t, "/api/quotas/storage", quotaData)
		defer setQuotaResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, setQuotaResp, http.StatusOK)
		
		// Get current quota usage
		usageResp := suite.env.APIClient.GET(t, "/api/quotas/storage/"+suite.tenantID)
		defer usageResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, usageResp, http.StatusOK)
		
		var usageResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, usageResp, &usageResult)
		
		assert.Contains(t, usageResult, "limit", "Usage should contain limit")
		assert.Contains(t, usageResult, "used", "Usage should contain used amount")
		assert.Contains(t, usageResult, "available", "Usage should contain available amount")
		
		initialUsed := usageResult["used"].(float64)
		
		// Create a large volume to test quota enforcement
		largeVolumeData := map[string]interface{}{
			"name":      "quota-test-volume",
			"size_gb":   500, // Large volume
			"tenant_id": suite.tenantID,
		}
		
		volumeResp := suite.env.APIClient.POST(t, "/api/storage/volumes", largeVolumeData)
		defer volumeResp.Body.Close()
		
		// Should succeed if under quota
		if volumeResp.StatusCode == http.StatusCreated {
			var volumeResult map[string]interface{}
			suite.env.APIClient.ParseJSON(t, volumeResp, &volumeResult)
			volumeID := volumeResult["id"].(string)
			
			// Verify quota usage increased
			newUsageResp := suite.env.APIClient.GET(t, "/api/quotas/storage/"+suite.tenantID)
			defer newUsageResp.Body.Close()
			
			var newUsageResult map[string]interface{}
			suite.env.APIClient.ParseJSON(t, newUsageResp, &newUsageResult)
			
			newUsed := newUsageResult["used"].(float64)
			assert.Greater(t, newUsed, initialUsed, "Storage usage should increase after volume creation")
			
			// Cleanup volume
			suite.env.APIClient.DELETE(t, "/api/storage/volumes/"+volumeID)
		}
		
		// Test quota enforcement by trying to exceed limit
		oversizedVolumeData := map[string]interface{}{
			"name":      "oversized-volume",
			"size_gb":   2000, // Exceeds quota limit
			"tenant_id": suite.tenantID,
		}
		
		oversizedResp := suite.env.APIClient.POST(t, "/api/storage/volumes", oversizedVolumeData)
		defer oversizedResp.Body.Close()
		
		// Should fail due to quota limit
		assert.Equal(t, http.StatusBadRequest, oversizedResp.StatusCode, 
			"Should reject volume creation that exceeds quota")
	})
}

// TestStorageCompression tests storage compression features
func (suite *StorageTestSuite) TestStorageCompression() {
	suite.T().Run("Storage Compression", func(t *testing.T) {
		// Create volume with compression enabled
		compressedVolumeData := map[string]interface{}{
			"name":        "compressed-test-volume",
			"size_gb":     50,
			"tenant_id":   suite.tenantID,
			"compression": true,
			"compression_algorithm": "lz4",
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/storage/volumes", compressedVolumeData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var volumeResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &volumeResult)
		
		volumeID := volumeResult["id"].(string)
		assert.Equal(t, true, volumeResult["compression"], "Volume should have compression enabled")
		assert.Equal(t, "lz4", volumeResult["compression_algorithm"], "Should use lz4 compression")
		
		// Get compression statistics
		statsResp := suite.env.APIClient.GET(t, "/api/storage/volumes/"+volumeID+"/compression")
		defer statsResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, statsResp, http.StatusOK)
		
		var statsResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, statsResp, &statsResult)
		
		assert.Contains(t, statsResult, "compression_ratio", "Should contain compression ratio")
		assert.Contains(t, statsResult, "space_saved", "Should contain space saved")
		
		// Cleanup
		suite.env.APIClient.DELETE(t, "/api/storage/volumes/"+volumeID)
	})
}

// TestStorageEncryption tests storage encryption features
func (suite *StorageTestSuite) TestStorageEncryption() {
	suite.T().Run("Storage Encryption", func(t *testing.T) {
		// Create encrypted volume
		encryptedVolumeData := map[string]interface{}{
			"name":       "encrypted-test-volume", 
			"size_gb":    25,
			"tenant_id":  suite.tenantID,
			"encryption": true,
			"encryption_algorithm": "aes-256",
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/storage/volumes", encryptedVolumeData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var volumeResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &volumeResult)
		
		volumeID := volumeResult["id"].(string)
		assert.Equal(t, true, volumeResult["encryption"], "Volume should have encryption enabled")
		assert.Equal(t, "aes-256", volumeResult["encryption_algorithm"], "Should use AES-256 encryption")
		
		// Test key rotation
		rotateResp := suite.env.APIClient.POST(t, "/api/storage/volumes/"+volumeID+"/rotate-key", nil)
		defer rotateResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, rotateResp, http.StatusOK)
		
		var rotateResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, rotateResp, &rotateResult)
		
		assert.Contains(t, rotateResult, "key_version", "Should return new key version")
		
		// Cleanup
		suite.env.APIClient.DELETE(t, "/api/storage/volumes/"+volumeID)
	})
}

// TestStorageTestSuite runs the storage integration test suite
func TestStorageTestSuite(t *testing.T) {
	suite.Run(t, new(StorageTestSuite))
}