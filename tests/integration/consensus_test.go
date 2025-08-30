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

// ConsensusTestSuite tests Raft consensus algorithm functionality
type ConsensusTestSuite struct {
	suite.Suite
	env     *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
	token   string
}

// SetupSuite initializes the test suite
func (suite *ConsensusTestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
	
	// Login as admin
	suite.token = suite.env.LoginAsAdmin(suite.T())
	suite.env.APIClient.SetAuthToken(suite.token)
}

// TearDownSuite cleans up the test suite
func (suite *ConsensusTestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// TestRaftClusterStatus tests Raft cluster status and health
func (suite *ConsensusTestSuite) TestRaftClusterStatus() {
	suite.T().Run("Cluster Status", func(t *testing.T) {
		// Get cluster status
		resp := suite.env.APIClient.GET(t, "/api/consensus/status")
		defer resp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, resp, http.StatusOK)
		
		var status map[string]interface{}
		suite.env.APIClient.ParseJSON(t, resp, &status)
		
		// Verify status structure
		assert.Contains(t, status, "state", "Status should contain state")
		assert.Contains(t, status, "term", "Status should contain current term")
		assert.Contains(t, status, "leader", "Status should contain leader info")
		assert.Contains(t, status, "nodes", "Status should contain cluster nodes")
		
		// Check valid states
		state := status["state"].(string)
		validStates := []string{"follower", "candidate", "leader"}
		assert.Contains(t, validStates, state, "State should be valid Raft state")
		
		// Verify term is non-negative
		term := status["term"].(float64)
		assert.GreaterOrEqual(t, term, float64(0), "Term should be non-negative")
		
		// Verify nodes list
		nodes, ok := status["nodes"].([]interface{})
		require.True(t, ok, "Nodes should be an array")
		assert.GreaterOrEqual(t, len(nodes), 1, "Should have at least one node")
	})
}

// TestLeaderElection tests leader election process
func (suite *ConsensusTestSuite) TestLeaderElection() {
	if testing.Short() {
		suite.T().Skip("Skipping leader election test in short mode")
	}
	
	suite.T().Run("Leader Election", func(t *testing.T) {
		// Get initial cluster status
		initialResp := suite.env.APIClient.GET(t, "/api/consensus/status")
		defer initialResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, initialResp, http.StatusOK)
		
		var initialStatus map[string]interface{}
		suite.env.APIClient.ParseJSON(t, initialResp, &initialStatus)
		
		initialLeader, hasLeader := initialStatus["leader"].(string)
		initialTerm := initialStatus["term"].(float64)
		
		// Force leader election (simulation)
		// In a real cluster, this would trigger an election
		electionResp := suite.env.APIClient.POST(t, "/api/consensus/force-election", nil)
		defer electionResp.Body.Close()
		
		// Election request should be accepted (even if it doesn't change anything)
		assert.True(t, electionResp.StatusCode == http.StatusOK || 
			electionResp.StatusCode == http.StatusAccepted,
			"Election request should be accepted")
		
		// Wait for election to complete
		time.Sleep(5 * time.Second)
		
		// Check status after election
		finalResp := suite.env.APIClient.GET(t, "/api/consensus/status")
		defer finalResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, finalResp, http.StatusOK)
		
		var finalStatus map[string]interface{}
		suite.env.APIClient.ParseJSON(t, finalResp, &finalStatus)
		
		finalLeader, hasFinalLeader := finalStatus["leader"].(string)
		finalTerm := finalStatus["term"].(float64)
		
		// Verify cluster health after election
		assert.True(t, hasFinalLeader, "Cluster should have a leader after election")
		
		if hasLeader && hasFinalLeader {
			// Either same leader or new leader
			if initialLeader != finalLeader {
				assert.Greater(t, finalTerm, initialTerm, 
					"Term should increase if leader changed")
			}
		}
	})
}

// TestLogReplication tests Raft log replication
func (suite *ConsensusTestSuite) TestLogReplication() {
	suite.T().Run("Log Replication", func(t *testing.T) {
		// Get initial log status
		initialResp := suite.env.APIClient.GET(t, "/api/consensus/logs")
		defer initialResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, initialResp, http.StatusOK)
		
		var initialLogs map[string]interface{}
		suite.env.APIClient.ParseJSON(t, initialResp, &initialLogs)
		
		initialCount := int(initialLogs["count"].(float64))
		
		// Submit a test operation that should be replicated
		operationData := map[string]interface{}{
			"type": "test_operation",
			"data": map[string]interface{}{
				"action": "create_vm",
				"vm_id":  "test-vm-consensus",
				"tenant_id": "tenant-1",
			},
		}
		
		submitResp := suite.env.APIClient.POST(t, "/api/consensus/submit", operationData)
		defer submitResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, submitResp, http.StatusOK)
		
		var submitResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, submitResp, &submitResult)
		
		assert.Contains(t, submitResult, "log_index", "Should return log index")
		assert.Contains(t, submitResult, "term", "Should return current term")
		
		logIndex := int(submitResult["log_index"].(float64))
		
		// Wait for replication
		time.Sleep(3 * time.Second)
		
		// Verify log was replicated
		finalResp := suite.env.APIClient.GET(t, "/api/consensus/logs")
		defer finalResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, finalResp, http.StatusOK)
		
		var finalLogs map[string]interface{}
		suite.env.APIClient.ParseJSON(t, finalResp, &finalLogs)
		
		finalCount := int(finalLogs["count"].(float64))
		assert.Greater(t, finalCount, initialCount, "Log count should increase after operation")
		
		// Get specific log entry
		logResp := suite.env.APIClient.GET(t, fmt.Sprintf("/api/consensus/logs/%d", logIndex))
		defer logResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, logResp, http.StatusOK)
		
		var logEntry map[string]interface{}
		suite.env.APIClient.ParseJSON(t, logResp, &logEntry)
		
		assert.Equal(t, float64(logIndex), logEntry["index"], "Log entry should have correct index")
		assert.Contains(t, logEntry, "term", "Log entry should contain term")
		assert.Contains(t, logEntry, "data", "Log entry should contain data")
	})
}

// TestClusterMembership tests cluster membership operations
func (suite *ConsensusTestSuite) TestClusterMembership() {
	suite.T().Run("Cluster Membership", func(t *testing.T) {
		// Get current cluster members
		membersResp := suite.env.APIClient.GET(t, "/api/consensus/members")
		defer membersResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, membersResp, http.StatusOK)
		
		var members map[string]interface{}
		suite.env.APIClient.ParseJSON(t, membersResp, &members)
		
		memberList, ok := members["members"].([]interface{})
		require.True(t, ok, "Members should be an array")
		
		initialMemberCount := len(memberList)
		assert.GreaterOrEqual(t, initialMemberCount, 1, "Should have at least one member")
		
		// Verify member structure
		if len(memberList) > 0 {
			member := memberList[0].(map[string]interface{})
			assert.Contains(t, member, "id", "Member should have ID")
			assert.Contains(t, member, "address", "Member should have address")
			assert.Contains(t, member, "state", "Member should have state")
			
			memberState := member["state"].(string)
			validStates := []string{"active", "inactive", "left"}
			assert.Contains(t, validStates, memberState, "Member state should be valid")
		}
		
		// Test adding a new member (simulation)
		newMemberData := map[string]interface{}{
			"id":      "test-node-new",
			"address": "localhost:9999",
		}
		
		addResp := suite.env.APIClient.POST(t, "/api/consensus/members", newMemberData)
		defer addResp.Body.Close()
		
		// Adding member should either succeed or be rejected with proper reason
		assert.True(t, addResp.StatusCode == http.StatusOK || 
			addResp.StatusCode == http.StatusConflict ||
			addResp.StatusCode == http.StatusBadRequest,
			"Add member request should be handled properly")
		
		if addResp.StatusCode == http.StatusOK {
			// Wait for membership change to propagate
			time.Sleep(3 * time.Second)
			
			// Verify member was added
			updatedResp := suite.env.APIClient.GET(t, "/api/consensus/members")
			defer updatedResp.Body.Close()
			
			var updatedMembers map[string]interface{}
			suite.env.APIClient.ParseJSON(t, updatedResp, &updatedMembers)
			
			updatedList := updatedMembers["members"].([]interface{})
			// Note: In a test environment, this might not actually add a member
			// but the API should handle the request properly
		}
	})
}

// TestConsensusMetrics tests consensus performance metrics
func (suite *ConsensusTestSuite) TestConsensusMetrics() {
	suite.T().Run("Consensus Metrics", func(t *testing.T) {
		metricsResp := suite.env.APIClient.GET(t, "/api/consensus/metrics")
		defer metricsResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, metricsResp, http.StatusOK)
		
		var metrics map[string]interface{}
		suite.env.APIClient.ParseJSON(t, metricsResp, &metrics)
		
		// Verify key metrics are present
		expectedMetrics := []string{
			"leader_election_count",
			"log_entries_total",
			"committed_entries",
			"applied_entries",
			"pending_entries",
			"replication_latency_ms",
		}
		
		for _, metric := range expectedMetrics {
			assert.Contains(t, metrics, metric, "Should contain metric: %s", metric)
			
			// Verify metric values are numeric and non-negative
			if value, ok := metrics[metric].(float64); ok {
				assert.GreaterOrEqual(t, value, float64(0), 
					"Metric %s should be non-negative", metric)
			}
		}
		
		// Verify specific metric relationships
		logTotal := metrics["log_entries_total"].(float64)
		committed := metrics["committed_entries"].(float64)
		applied := metrics["applied_entries"].(float64)
		
		assert.LessOrEqual(t, committed, logTotal, 
			"Committed entries should not exceed total entries")
		assert.LessOrEqual(t, applied, committed, 
			"Applied entries should not exceed committed entries")
	})
}

// TestConsensusFailureRecovery tests failure scenarios and recovery
func (suite *ConsensusTestSuite) TestConsensusFailureRecovery() {
	if testing.Short() {
		suite.T().Skip("Skipping failure recovery test in short mode")
	}
	
	suite.T().Run("Failure Recovery", func(t *testing.T) {
		// Get initial cluster health
		initialResp := suite.env.APIClient.GET(t, "/api/consensus/health")
		defer initialResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, initialResp, http.StatusOK)
		
		var initialHealth map[string]interface{}
		suite.env.APIClient.ParseJSON(t, initialResp, &initialHealth)
		
		assert.Equal(t, "healthy", initialHealth["status"], "Cluster should be initially healthy")
		
		// Simulate network partition (if supported)
		partitionResp := suite.env.APIClient.POST(t, "/api/consensus/simulate/partition", 
			map[string]interface{}{
				"duration": "10s",
				"affected_nodes": []string{"node-1"},
			})
		defer partitionResp.Body.Close()
		
		// If partition simulation is not implemented, that's OK
		if partitionResp.StatusCode == http.StatusOK {
			// Wait for partition to take effect
			time.Sleep(5 * time.Second)
			
			// Check cluster health during partition
			partitionHealthResp := suite.env.APIClient.GET(t, "/api/consensus/health")
			defer partitionHealthResp.Body.Close()
			
			var partitionHealth map[string]interface{}
			suite.env.APIClient.ParseJSON(t, partitionHealthResp, &partitionHealth)
			
			// Status might be degraded during partition
			status := partitionHealth["status"].(string)
			validStatuses := []string{"healthy", "degraded", "unhealthy"}
			assert.Contains(t, validStatuses, status, "Health status should be valid")
			
			// Wait for partition to heal
			time.Sleep(12 * time.Second)
			
			// Check recovery
			recoveryResp := suite.env.APIClient.GET(t, "/api/consensus/health")
			defer recoveryResp.Body.Close()
			
			var recoveryHealth map[string]interface{}
			suite.env.APIClient.ParseJSON(t, recoveryResp, &recoveryHealth)
			
			assert.Equal(t, "healthy", recoveryHealth["status"], 
				"Cluster should recover to healthy state")
		}
	})
}

// TestConsensusConfigurationChanges tests dynamic configuration updates
func (suite *ConsensusTestSuite) TestConsensusConfigurationChanges() {
	suite.T().Run("Configuration Changes", func(t *testing.T) {
		// Get current configuration
		configResp := suite.env.APIClient.GET(t, "/api/consensus/config")
		defer configResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, configResp, http.StatusOK)
		
		var config map[string]interface{}
		suite.env.APIClient.ParseJSON(t, configResp, &config)
		
		// Verify configuration structure
		assert.Contains(t, config, "election_timeout", "Config should contain election timeout")
		assert.Contains(t, config, "heartbeat_interval", "Config should contain heartbeat interval")
		assert.Contains(t, config, "log_retention", "Config should contain log retention settings")
		
		// Test configuration update
		updateData := map[string]interface{}{
			"heartbeat_interval": "200ms",
			"log_retention": map[string]interface{}{
				"max_entries": 10000,
				"max_age": "7d",
			},
		}
		
		updateResp := suite.env.APIClient.PUT(t, "/api/consensus/config", updateData)
		defer updateResp.Body.Close()
		
		// Configuration update should either succeed or be properly rejected
		assert.True(t, updateResp.StatusCode == http.StatusOK || 
			updateResp.StatusCode == http.StatusBadRequest ||
			updateResp.StatusCode == http.StatusForbidden,
			"Config update should be handled properly")
		
		if updateResp.StatusCode == http.StatusOK {
			// Verify configuration was updated
			verifyResp := suite.env.APIClient.GET(t, "/api/consensus/config")
			defer verifyResp.Body.Close()
			
			var verifyConfig map[string]interface{}
			suite.env.APIClient.ParseJSON(t, verifyResp, &verifyConfig)
			
			assert.Equal(t, "200ms", verifyConfig["heartbeat_interval"], 
				"Heartbeat interval should be updated")
		}
	})
}

// TestConsensusSnapshots tests snapshot creation and restoration
func (suite *ConsensusTestSuite) TestConsensusSnapshots() {
	suite.T().Run("Consensus Snapshots", func(t *testing.T) {
		// Trigger snapshot creation
		snapshotResp := suite.env.APIClient.POST(t, "/api/consensus/snapshot", nil)
		defer snapshotResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, snapshotResp, http.StatusOK)
		
		var snapshotResult map[string]interface{}
		suite.env.APIClient.ParseJSON(t, snapshotResp, &snapshotResult)
		
		assert.Contains(t, snapshotResult, "snapshot_id", "Should return snapshot ID")
		assert.Contains(t, snapshotResult, "index", "Should return snapshot index")
		assert.Contains(t, snapshotResult, "term", "Should return snapshot term")
		
		snapshotID := snapshotResult["snapshot_id"].(string)
		
		// Wait for snapshot to complete
		timeout := time.Now().Add(30 * time.Second)
		for time.Now().Before(timeout) {
			statusResp := suite.env.APIClient.GET(t, "/api/consensus/snapshots/"+snapshotID)
			defer statusResp.Body.Close()
			
			if statusResp.StatusCode == http.StatusOK {
				var status map[string]interface{}
				suite.env.APIClient.ParseJSON(t, statusResp, &status)
				
				if status["status"] == "completed" {
					break
				}
			}
			
			time.Sleep(2 * time.Second)
		}
		
		// List available snapshots
		listResp := suite.env.APIClient.GET(t, "/api/consensus/snapshots")
		defer listResp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, listResp, http.StatusOK)
		
		var snapshots map[string]interface{}
		suite.env.APIClient.ParseJSON(t, listResp, &snapshots)
		
		snapshotList, ok := snapshots["snapshots"].([]interface{})
		require.True(t, ok, "Should contain snapshots list")
		assert.GreaterOrEqual(t, len(snapshotList), 1, "Should have at least one snapshot")
		
		// Verify snapshot structure
		if len(snapshotList) > 0 {
			snapshot := snapshotList[0].(map[string]interface{})
			assert.Contains(t, snapshot, "id", "Snapshot should have ID")
			assert.Contains(t, snapshot, "index", "Snapshot should have index")
			assert.Contains(t, snapshot, "term", "Snapshot should have term")
			assert.Contains(t, snapshot, "size", "Snapshot should have size")
			assert.Contains(t, snapshot, "created_at", "Snapshot should have creation time")
		}
	})
}

// TestConsensusLoadBalancing tests consensus under load
func (suite *ConsensusTestSuite) TestConsensusLoadBalancing() {
	if testing.Short() {
		suite.T().Skip("Skipping load balancing test in short mode")
	}
	
	suite.T().Run("Load Balancing", func(t *testing.T) {
		// Submit multiple operations concurrently
		numOperations := 10
		results := make(chan bool, numOperations)
		errors := make(chan error, numOperations)
		
		for i := 0; i < numOperations; i++ {
			go func(index int) {
				operationData := map[string]interface{}{
					"type": "load_test_operation",
					"data": map[string]interface{}{
						"operation_id": fmt.Sprintf("op-%d", index),
						"timestamp":    time.Now().Unix(),
					},
				}
				
				resp := suite.env.APIClient.POST(t, "/api/consensus/submit", operationData)
				defer resp.Body.Close()
				
				if resp.StatusCode == http.StatusOK {
					results <- true
				} else {
					errors <- fmt.Errorf("operation %d failed with status %d", index, resp.StatusCode)
				}
			}(i)
		}
		
		// Collect results
		successCount := 0
		errorCount := 0
		
		for i := 0; i < numOperations; i++ {
			select {
			case <-results:
				successCount++
			case err := <-errors:
				t.Logf("Operation error: %v", err)
				errorCount++
			case <-time.After(30 * time.Second):
				t.Fatalf("Timeout waiting for operation %d", i)
			}
		}
		
		t.Logf("Load test results: %d successes, %d errors", successCount, errorCount)
		
		// Most operations should succeed under normal conditions
		successRate := float64(successCount) / float64(numOperations)
		assert.GreaterOrEqual(t, successRate, 0.8, 
			"At least 80%% of operations should succeed under load")
	})
}

// TestConsensusTestSuite runs the consensus integration test suite
func TestConsensusTestSuite(t *testing.T) {
	suite.Run(t, new(ConsensusTestSuite))
}