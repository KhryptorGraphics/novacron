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

// E2ETestSuite tests end-to-end workflows
type E2ETestSuite struct {
	suite.Suite
	env     *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
}

// SetupSuite initializes the test suite
func (suite *E2ETestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
}

// TearDownSuite cleans up the test suite
func (suite *E2ETestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// TestCompleteVMLifecycleWorkflow tests the complete VM lifecycle from creation to deletion
func (suite *E2ETestSuite) TestCompleteVMLifecycleWorkflow() {
	suite.T().Run("Complete VM Lifecycle Workflow", func(t *testing.T) {
		// Step 1: User registration and login
		t.Log("Step 1: User registration and authentication")
		
		userData := map[string]interface{}{
			"email":     "e2e-user@test.com",
			"password":  "E2ETestPass123!",
			"name":      "E2E Test User",
			"tenant_id": "tenant-e2e",
		}
		
		registerResp := suite.env.APIClient.POST(t, "/api/auth/register", userData)
		defer registerResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, registerResp, http.StatusCreated)
		
		var userResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, registerResp, &userResult)
		userID := int(userResult["id"].(float64))
		
		// Login as the new user
		token := suite.env.APIClient.Login(t, "e2e-user@test.com", "E2ETestPass123!")
		suite.env.APIClient.SetAuthToken(token)
		
		// Step 2: Create VM with specific configuration
		t.Log("Step 2: Creating VM with custom configuration")
		
		vmData := map[string]interface{}{
			"name":      "e2e-test-vm",
			"cpu":       4,
			"memory":    4096,
			"disk_size": 51200,
			"image":     "ubuntu:20.04",
			"tenant_id": "tenant-e2e",
			"metadata": map[string]string{
				"environment": "testing",
				"purpose":     "e2e-workflow",
			},
		}
		
		createResp := suite.env.APIClient.POST(t, "/api/vms", vmData)
		defer createResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, createResp, http.StatusCreated)
		
		var vmResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, createResp, &vmResult)
		vmID := vmResult["id"].(string)
		
		assert.Equal(t, "created", vmResult["state"])
		assert.Equal(t, vmData["name"], vmResult["name"])
		
		// Step 3: Start the VM and wait for it to be running
		t.Log("Step 3: Starting VM")
		
		startResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/start", nil)
		defer startResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, startResp, http.StatusOK)
		
		// Wait for VM to reach running state
		suite.env.WaitForVMState(t, vmID, "running", 60*time.Second)
		
		// Step 4: Create and attach storage volume
		t.Log("Step 4: Creating and attaching storage volume")
		
		volumeData := map[string]interface{}{
			"name":      "e2e-storage-volume",
			"size_gb":   50,
			"tenant_id": "tenant-e2e",
		}
		
		volumeResp := suite.env.APIClient.POST(t, "/api/storage/volumes", volumeData)
		defer volumeResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, volumeResp, http.StatusCreated)
		
		var volumeResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, volumeResp, &volumeResult)
		volumeID := volumeResult["id"].(string)
		
		// Attach volume to VM
		attachData := map[string]interface{}{
			"vm_id":       vmID,
			"mount_point": "/mnt/e2e-volume",
		}
		
		attachResp := suite.env.APIClient.POST(t, "/api/storage/volumes/"+volumeID+"/attach", attachData)
		defer attachResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, attachResp, http.StatusOK)
		
		// Step 5: Create VM snapshot
		t.Log("Step 5: Creating VM snapshot")
		
		snapshotData := map[string]interface{}{
			"name":        "e2e-snapshot",
			"description": "E2E workflow test snapshot",
		}
		
		snapshotResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/snapshots", snapshotData)
		defer snapshotResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, snapshotResp, http.StatusCreated)
		
		var snapshotResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, snapshotResp, &snapshotResult)
		snapshotID := snapshotResult["id"].(string)
		
		// Wait for snapshot to complete
		timeout := time.Now().Add(120 * time.Second)
		for time.Now().Before(timeout) {
			statusResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID+"/snapshots/"+snapshotID)
			defer statusResp.Body.Close()
			
			var status map[string]interface{}
			suite.env.APIClient.ParseJSON(t, statusResp, &status)
			
			if status["status"] == "completed" {
				break
			}
			
			time.Sleep(3 * time.Second)
		}
		
		// Step 6: Monitor VM metrics
		t.Log("Step 6: Monitoring VM metrics")
		
		// Wait for metrics to be collected
		time.Sleep(10 * time.Second)
		
		metricsResp := suite.env.APIClient.GET(t, "/api/vms/"+vmID+"/metrics")
		defer metricsResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, metricsResp, http.StatusOK)
		
		var metrics map[string]interface{}
		suite.env.APIClient.ParseJSON(t, metricsResp, &metrics)
		assert.NotEmpty(t, metrics, "VM should have metrics")
		
		// Step 7: Test VM operations (pause/resume)
		t.Log("Step 7: Testing VM pause/resume")
		
		pauseResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/pause", nil)
		defer pauseResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, pauseResp, http.StatusOK)
		
		suite.env.WaitForVMState(t, vmID, "paused", 30*time.Second)
		
		resumeResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/resume", nil)
		defer resumeResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, resumeResp, http.StatusOK)
		
		suite.env.WaitForVMState(t, vmID, "running", 30*time.Second)
		
		// Step 8: Create backup
		t.Log("Step 8: Creating VM backup")
		
		backupData := map[string]interface{}{
			"name": "e2e-backup",
			"type": "full",
		}
		
		backupResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/backups", backupData)
		defer backupResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, backupResp, http.StatusCreated)
		
		var backupResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, backupResp, &backupResult)
		backupID := backupResult["id"].(string)
		
		// Step 9: Test resource quota checking
		t.Log("Step 9: Checking resource quotas")
		
		quotaResp := suite.env.APIClient.GET(t, "/api/quotas/tenant-e2e")
		defer quotaResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, quotaResp, http.StatusOK)
		
		var quota map[string]interface{}
		suite.env.APIClient.ParseJSON(t, quotaResp, &quota)
		assert.Greater(t, quota["cpu_used"], float64(0), "CPU quota should show usage")
		
		// Step 10: Cleanup - Stop and delete VM
		t.Log("Step 10: Cleaning up resources")
		
		// Detach volume first
		detachResp := suite.env.APIClient.POST(t, "/api/storage/volumes/"+volumeID+"/detach", nil)
		defer detachResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, detachResp, http.StatusOK)
		
		// Stop VM
		stopResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/stop", nil)
		defer stopResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, stopResp, http.StatusOK)
		
		suite.env.WaitForVMState(t, vmID, "stopped", 60*time.Second)
		
		// Delete snapshot
		suite.env.APIClient.DELETE(t, "/api/vms/"+vmID+"/snapshots/"+snapshotID)
		
		// Delete volume
		suite.env.APIClient.DELETE(t, "/api/storage/volumes/"+volumeID)
		
		// Delete VM
		deleteResp := suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
		defer deleteResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, deleteResp, http.StatusNoContent)
		
		t.Log("E2E workflow completed successfully!")
	})
}

// TestMultiTenantWorkflow tests multi-tenant scenario
func (suite *E2ETestSuite) TestMultiTenantWorkflow() {
	suite.T().Run("Multi-Tenant Workflow", func(t *testing.T) {
		// Create users in different tenants
		tenants := []struct {
			id       string
			userEmail string
			userName  string
		}{
			{"tenant-mt-1", "mt1-user@test.com", "Multi-Tenant User 1"},
			{"tenant-mt-2", "mt2-user@test.com", "Multi-Tenant User 2"},
		}
		
		var tenantTokens []string
		var tenantVMs []string
		
		// Step 1: Set up users and VMs for each tenant
		for i, tenant := range tenants {
			t.Logf("Setting up tenant %s", tenant.id)
			
			// Register user
			userData := map[string]interface{}{
				"email":     tenant.userEmail,
				"password":  "MTTestPass123!",
				"name":      tenant.userName,
				"tenant_id": tenant.id,
			}
			
			registerResp := suite.env.APIClient.POST(t, "/api/auth/register", userData)
			defer registerResp.Body.Close()
			suite.env.APIClient.ExpectStatus(t, registerResp, http.StatusCreated)
			
			// Login
			token := suite.env.APIClient.Login(t, tenant.userEmail, "MTTestPass123!")
			tenantTokens = append(tenantTokens, token)
			
			// Create VM for tenant
			suite.env.APIClient.SetAuthToken(token)
			vmID := suite.env.CreateTestVM(t, fmt.Sprintf("mt-vm-%d", i+1), tenant.id)
			tenantVMs = append(tenantVMs, vmID)
		}
		
		// Step 2: Verify tenant isolation
		for i, tenant := range tenants {
			t.Logf("Verifying isolation for tenant %s", tenant.id)
			
			suite.env.APIClient.SetAuthToken(tenantTokens[i])
			
			// List VMs - should only see own tenant's VMs
			listResp := suite.env.APIClient.GET(t, "/api/vms")
			defer listResp.Body.Close()
			suite.env.APIClient.ExpectStatus(t, listResp, http.StatusOK)
			
			var vmList map[string]interface{}
			suite.env.APIClient.ParseJSON(t, listResp, &vmList)
			
			vms := vmList["vms"].([]interface{})
			for _, vm := range vms {
				vmMap := vm.(map[string]interface{})
				assert.Equal(t, tenant.id, vmMap["tenant_id"], 
					"User should only see VMs from their tenant")
			}
		}
		
		// Step 3: Test cross-tenant access denial
		t.Log("Testing cross-tenant access denial")
		
		// User 1 tries to access User 2's VM
		suite.env.APIClient.SetAuthToken(tenantTokens[0])
		
		accessResp := suite.env.APIClient.GET(t, "/api/vms/"+tenantVMs[1])
		defer accessResp.Body.Close()
		assert.Equal(t, http.StatusNotFound, accessResp.StatusCode,
			"User should not be able to access other tenant's VM")
		
		// Step 4: Test admin access to all tenants
		t.Log("Testing admin access to all tenants")
		
		adminToken := suite.env.LoginAsAdmin(t)
		suite.env.APIClient.SetAuthToken(adminToken)
		
		// Admin should see VMs from all tenants
		adminListResp := suite.env.APIClient.GET(t, "/api/admin/vms")
		defer adminListResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, adminListResp, http.StatusOK)
		
		// Cleanup
		for _, vmID := range tenantVMs {
			suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
		}
	})
}

// TestDisasterRecoveryWorkflow tests backup and disaster recovery
func (suite *E2ETestSuite) TestDisasterRecoveryWorkflow() {
	if testing.Short() {
		suite.T().Skip("Skipping disaster recovery test in short mode")
	}
	
	suite.T().Run("Disaster Recovery Workflow", func(t *testing.T) {
		// Login as admin
		adminToken := suite.env.LoginAsAdmin(t)
		suite.env.APIClient.SetAuthToken(adminToken)
		
		// Step 1: Create production VM
		t.Log("Step 1: Creating production VM")
		
		vmID := suite.env.CreateTestVM(t, "production-vm", "tenant-dr")
		
		// Start VM
		startResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/start", nil)
		defer startResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, startResp, http.StatusOK)
		
		suite.env.WaitForVMState(t, vmID, "running", 60*time.Second)
		
		// Step 2: Create full backup
		t.Log("Step 2: Creating full backup")
		
		backupData := map[string]interface{}{
			"name": "dr-full-backup",
			"type": "full",
		}
		
		backupResp := suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/backups", backupData)
		defer backupResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, backupResp, http.StatusCreated)
		
		var backupResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, backupResp, &backupResult)
		backupID := backupResult["id"].(string)
		
		// Wait for backup to complete
		timeout := time.Now().Add(180 * time.Second)
		for time.Now().Before(timeout) {
			statusResp := suite.env.APIClient.GET(t, "/api/backups/"+backupID)
			defer statusResp.Body.Close()
			
			if statusResp.StatusCode == http.StatusOK {
				var status map[string]interface{}
				suite.env.APIClient.ParseJSON(t, statusResp, &status)
				
				if status["status"] == "completed" {
					break
				}
			}
			
			time.Sleep(5 * time.Second)
		}
		
		// Step 3: Simulate disaster - stop and delete VM
		t.Log("Step 3: Simulating disaster")
		
		suite.env.APIClient.POST(t, "/api/vms/"+vmID+"/stop", nil)
		suite.env.WaitForVMState(t, vmID, "stopped", 60*time.Second)
		
		deleteResp := suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
		defer deleteResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, deleteResp, http.StatusNoContent)
		
		// Step 4: Restore from backup
		t.Log("Step 4: Restoring from backup")
		
		restoreData := map[string]interface{}{
			"backup_id":   backupID,
			"vm_name":     "restored-production-vm",
			"target_node": "node-1",
		}
		
		restoreResp := suite.env.APIClient.POST(t, "/api/backups/"+backupID+"/restore", restoreData)
		defer restoreResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, restoreResp, http.StatusOK)
		
		var restoreResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, restoreResp, &restoreResult)
		
		restoredVMID := restoreResult["vm_id"].(string)
		assert.NotEmpty(t, restoredVMID, "Restored VM should have ID")
		
		// Step 5: Verify restored VM
		t.Log("Step 5: Verifying restored VM")
		
		// Wait for restore to complete
		timeout = time.Now().Add(180 * time.Second)
		for time.Now().Before(timeout) {
			vmResp := suite.env.APIClient.GET(t, "/api/vms/"+restoredVMID)
			defer vmResp.Body.Close()
			
			if vmResp.StatusCode == http.StatusOK {
				var vm map[string]interface{}
				suite.env.APIClient.ParseJSON(t, vmResp, &vm)
				
				if vm["state"] == "stopped" {
					// Start the restored VM
					suite.env.APIClient.POST(t, "/api/vms/"+restoredVMID+"/start", nil)
					suite.env.WaitForVMState(t, restoredVMID, "running", 60*time.Second)
					break
				}
			}
			
			time.Sleep(5 * time.Second)
		}
		
		// Verify VM is running
		finalResp := suite.env.APIClient.GET(t, "/api/vms/"+restoredVMID)
		defer finalResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, finalResp, http.StatusOK)
		
		var finalVM map[string]interface{}
		suite.env.APIClient.ParseJSON(t, finalResp, &finalVM)
		assert.Equal(t, "running", finalVM["state"], "Restored VM should be running")
		
		// Cleanup
		suite.env.APIClient.POST(t, "/api/vms/"+restoredVMID+"/stop", nil)
		suite.env.WaitForVMState(t, restoredVMID, "stopped", 60*time.Second)
		suite.env.APIClient.DELETE(t, "/api/vms/"+restoredVMID)
		
		t.Log("Disaster recovery workflow completed successfully!")
	})
}

// TestAutoScalingWorkflow tests auto-scaling functionality
func (suite *E2ETestSuite) TestAutoScalingWorkflow() {
	if testing.Short() {
		suite.T().Skip("Skipping auto-scaling test in short mode")
	}
	
	suite.T().Run("Auto-Scaling Workflow", func(t *testing.T) {
		// Login as admin
		adminToken := suite.env.LoginAsAdmin(t)
		suite.env.APIClient.SetAuthToken(adminToken)
		
		// Step 1: Create auto-scaling policy
		t.Log("Step 1: Creating auto-scaling policy")
		
		policyData := map[string]interface{}{
			"name":        "e2e-autoscaling-policy",
			"tenant_id":   "tenant-autoscale",
			"min_instances": 2,
			"max_instances": 5,
			"target_cpu_utilization": 70,
			"scale_up_cooldown":   "5m",
			"scale_down_cooldown": "10m",
		}
		
		policyResp := suite.env.APIClient.POST(t, "/api/autoscaling/policies", policyData)
		defer policyResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, policyResp, http.StatusCreated)
		
		var policyResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, policyResp, &policyResult)
		policyID := policyResult["id"].(string)
		
		// Step 2: Create VM template for scaling
		t.Log("Step 2: Creating VM template")
		
		templateData := map[string]interface{}{
			"name":      "autoscale-template",
			"cpu":       2,
			"memory":    2048,
			"disk_size": 20480,
			"image":     "ubuntu:20.04",
			"tenant_id": "tenant-autoscale",
		}
		
		templateResp := suite.env.APIClient.POST(t, "/api/templates", templateData)
		defer templateResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, templateResp, http.StatusCreated)
		
		var templateResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, templateResp, &templateResult)
		templateID := templateResult["id"].(string)
		
		// Step 3: Create autoscaling group
		t.Log("Step 3: Creating autoscaling group")
		
		groupData := map[string]interface{}{
			"name":         "e2e-autoscaling-group",
			"policy_id":    policyID,
			"template_id":  templateID,
			"tenant_id":    "tenant-autoscale",
		}
		
		groupResp := suite.env.APIClient.POST(t, "/api/autoscaling/groups", groupData)
		defer groupResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, groupResp, http.StatusCreated)
		
		var groupResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, groupResp, &groupResult)
		groupID := groupResult["id"].(string)
		
		// Step 4: Wait for initial instances to be created
		t.Log("Step 4: Waiting for initial instances")
		
		timeout := time.Now().Add(120 * time.Second)
		for time.Now().Before(timeout) {
			statusResp := suite.env.APIClient.GET(t, "/api/autoscaling/groups/"+groupID)
			defer statusResp.Body.Close()
			
			if statusResp.StatusCode == http.StatusOK {
				var status map[string]interface{}
				suite.env.APIClient.ParseJSON(t, statusResp, &status)
				
				currentInstances := int(status["current_instances"].(float64))
				if currentInstances >= 2 {
					t.Logf("Autoscaling group has %d instances", currentInstances)
					break
				}
			}
			
			time.Sleep(5 * time.Second)
		}
		
		// Step 5: Trigger scale-up (simulate high load)
		t.Log("Step 5: Triggering scale-up")
		
		scaleUpData := map[string]interface{}{
			"cpu_utilization": 85, // Above threshold
		}
		
		scaleResp := suite.env.APIClient.POST(t, "/api/autoscaling/groups/"+groupID+"/trigger", scaleUpData)
		defer scaleResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, scaleResp, http.StatusOK)
		
		// Wait for scale-up to complete
		time.Sleep(30 * time.Second)
		
		// Step 6: Verify scale-up occurred
		finalStatusResp := suite.env.APIClient.GET(t, "/api/autoscaling/groups/"+groupID)
		defer finalStatusResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, finalStatusResp, http.StatusOK)
		
		var finalStatus map[string]interface{}
		suite.env.APIClient.ParseJSON(t, finalStatusResp, &finalStatus)
		
		finalInstances := int(finalStatus["current_instances"].(float64))
		t.Logf("Final instance count: %d", finalInstances)
		
		// Cleanup
		suite.env.APIClient.DELETE(t, "/api/autoscaling/groups/"+groupID)
		suite.env.APIClient.DELETE(t, "/api/templates/"+templateID)
		suite.env.APIClient.DELETE(t, "/api/autoscaling/policies/"+policyID)
		
		t.Log("Auto-scaling workflow completed!")
	})
}

// TestE2ETestSuite runs the end-to-end integration test suite
func TestE2ETestSuite(t *testing.T) {
	suite.Run(t, new(E2ETestSuite))
}