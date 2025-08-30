package controller

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// SDNControllerTestSuite provides comprehensive testing for SDN controller
type SDNControllerTestSuite struct {
	suite.Suite
	controller       *SDNController
	mockAIEngine     *MockAIOptimizer
	networkTopology  *network.NetworkTopology
	testContext      context.Context
	testCancel       context.CancelFunc
}

// MockAIOptimizer implements AIOptimizer interface for testing
type MockAIOptimizer struct {
	mock.Mock
}

func (m *MockAIOptimizer) OptimizeIntent(intent *Intent, topology *network.NetworkTopology) ([]FlowRule, error) {
	args := m.Called(intent, topology)
	return args.Get(0).([]FlowRule), args.Error(1)
}

func (m *MockAIOptimizer) PredictTrafficPatterns(nodes []string, timeWindow time.Duration) (map[string]float64, error) {
	args := m.Called(nodes, timeWindow)
	return args.Get(0).(map[string]float64), args.Error(1)
}

func (m *MockAIOptimizer) OptimizeSliceAllocation(slices []*NetworkSlice) (map[string]SliceResources, error) {
	args := m.Called(slices)
	return args.Get(0).(map[string]SliceResources), args.Error(1)
}

// FlowRuleTestCase represents a test case for flow rule operations
type FlowRuleTestCase struct {
	Name        string
	Rule        *FlowRule
	ExpectError bool
	Validate    func(t *testing.T, rule *FlowRule)
}

// IntentTestCase represents a test case for network intent processing
type IntentTestCase struct {
	Name        string
	Intent      *Intent
	ExpectError bool
	MockSetup   func(m *MockAIOptimizer)
	Validate    func(t *testing.T, intent *Intent)
}

// NetworkSliceTestCase represents a test case for network slice operations
type NetworkSliceTestCase struct {
	Name        string
	Slice       *NetworkSlice
	ExpectError bool
	MockSetup   func(m *MockAIOptimizer)
	Validate    func(t *testing.T, slice *NetworkSlice)
}

// SetupSuite initializes the test suite
func (suite *SDNControllerTestSuite) SetupSuite() {
	suite.testContext, suite.testCancel = context.WithCancel(context.Background())
	
	// Create network topology for testing
	suite.networkTopology = network.NewNetworkTopology()
	suite.setupTestTopology()
	
	// Create mock AI engine
	suite.mockAIEngine = &MockAIOptimizer{}
	
	// Create SDN controller with test configuration
	config := SDNControllerConfig{
		IntentEvaluationInterval: 100 * time.Millisecond, // Faster for tests
		FlowRuleTimeout:         1 * time.Second,
		MaxConcurrentOperations: 10,
		EnableAIOptimization:    true,
		EnableP4Support:         false,
		ControllerPort:          16653, // Use different port for tests
	}
	
	suite.controller = NewSDNController(config, suite.networkTopology, suite.mockAIEngine)
	
	// Start the controller
	err := suite.controller.Start()
	suite.Require().NoError(err)
}

// TearDownSuite cleans up after all tests
func (suite *SDNControllerTestSuite) TearDownSuite() {
	if suite.controller != nil {
		suite.controller.Stop()
	}
	if suite.testCancel != nil {
		suite.testCancel()
	}
}

// setupTestTopology creates a test network topology
func (suite *SDNControllerTestSuite) setupTestTopology() {
	// Add test nodes
	nodes := []struct {
		ID   string
		Type string
	}{
		{"switch-1", "switch"},
		{"switch-2", "switch"},
		{"switch-3", "switch"},
		{"host-1", "host"},
		{"host-2", "host"},
		{"host-3", "host"},
	}
	
	location := network.NetworkLocation{
		Datacenter: "test-dc",
		Zone:       "test-zone",
		Rack:       "test-rack",
	}
	
	for _, n := range nodes {
		node := &network.NetworkNode{
			ID:       n.ID,
			Type:     n.Type,
			Location: location,
		}
		suite.networkTopology.AddNode(node)
	}
	
	// Add test links
	links := []struct {
		Source, Dest string
		Bandwidth    int64
		Latency      time.Duration
	}{
		{"switch-1", "switch-2", 10000, 1 * time.Millisecond},
		{"switch-2", "switch-3", 10000, 1 * time.Millisecond},
		{"switch-1", "switch-3", 5000, 2 * time.Millisecond},
		{"host-1", "switch-1", 1000, 500 * time.Microsecond},
		{"host-2", "switch-2", 1000, 500 * time.Microsecond},
		{"host-3", "switch-3", 1000, 500 * time.Microsecond},
	}
	
	for _, l := range links {
		link := &network.NetworkLink{
			SourceID:      l.Source,
			DestinationID: l.Dest,
			Bandwidth:     l.Bandwidth,
			Latency:       l.Latency,
			Type:          network.LinkTypeSameDatacenter,
		}
		suite.networkTopology.AddLink(link)
	}
}

// TestSDNControllerCreation tests SDN controller creation
func (suite *SDNControllerTestSuite) TestSDNControllerCreation() {
	assert.NotNil(suite.T(), suite.controller)
	assert.NotNil(suite.T(), suite.controller.networkTopology)
	assert.NotNil(suite.T(), suite.controller.aiEngine)
	assert.NotNil(suite.T(), suite.controller.metrics)
}

// TestFlowRuleManagement tests flow rule creation, installation, and cleanup
func (suite *SDNControllerTestSuite) TestFlowRuleManagement() {
	testCases := []FlowRuleTestCase{
		{
			Name: "BasicFlowRule",
			Rule: &FlowRule{
				ID:          uuid.New().String(),
				Priority:    1000,
				TableID:     0,
				IdleTimeout: 60,
				HardTimeout: 300,
				Match: FlowMatch{
					EthType: "0x0800", // IPv4
					IPSrc:   "192.168.1.0/24",
					IPDst:   "10.0.0.0/16",
				},
				Actions: []FlowAction{
					{
						Type: ActionOutput,
						Params: map[string]interface{}{
							"port": "2",
						},
					},
				},
			},
			ExpectError: false,
			Validate: func(t *testing.T, rule *FlowRule) {
				assert.Equal(t, 1000, rule.Priority)
				assert.Equal(t, "0x0800", rule.Match.EthType)
				assert.Len(t, rule.Actions, 1)
			},
		},
		{
			Name: "QoSFlowRule",
			Rule: &FlowRule{
				ID:       uuid.New().String(),
				Priority: 5000,
				TableID:  0,
				Match: FlowMatch{
					IPProto: 6, // TCP
					TCPDst:  80,
				},
				Actions: []FlowAction{
					{
						Type: ActionSetQueue,
						Params: map[string]interface{}{
							"queue_id": 1,
						},
					},
					{
						Type: ActionOutput,
						Params: map[string]interface{}{
							"port": "normal",
						},
					},
				},
			},
			ExpectError: false,
			Validate: func(t *testing.T, rule *FlowRule) {
				assert.Equal(t, 5000, rule.Priority)
				assert.Equal(t, 6, rule.Match.IPProto)
				assert.Equal(t, 80, rule.Match.TCPDst)
				assert.Len(t, rule.Actions, 2)
			},
		},
		{
			Name: "VLANTaggingRule",
			Rule: &FlowRule{
				ID:       uuid.New().String(),
				Priority: 3000,
				TableID:  0,
				Match: FlowMatch{
					InPort: "1",
				},
				Actions: []FlowAction{
					{
						Type: ActionPushVlan,
						Params: map[string]interface{}{
							"vlan_id": 100,
						},
					},
					{
						Type: ActionOutput,
						Params: map[string]interface{}{
							"port": "2",
						},
					},
				},
			},
			ExpectError: false,
			Validate: func(t *testing.T, rule *FlowRule) {
				assert.Equal(t, "1", rule.Match.InPort)
				assert.Equal(t, ActionPushVlan, rule.Actions[0].Type)
			},
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.Name, func(t *testing.T) {
			err := suite.controller.installFlowRule(tc.Rule)
			
			if tc.ExpectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				
				// Validate rule was installed
				suite.controller.flowRulesMutex.RLock()
				installedRule, exists := suite.controller.flowRules[tc.Rule.ID]
				suite.controller.flowRulesMutex.RUnlock()
				
				assert.True(t, exists, "Flow rule should be installed")
				assert.Equal(t, tc.Rule.ID, installedRule.ID)
				
				if tc.Validate != nil {
					tc.Validate(t, installedRule)
				}
			}
		})
	}
}

// TestNetworkIntentProcessing tests network intent creation and processing
func (suite *SDNControllerTestSuite) TestNetworkIntentProcessing() {
	testCases := []IntentTestCase{
		{
			Name: "LowLatencyIntent",
			Intent: &Intent{
				Name:        "low-latency-web-services",
				Description: "Minimize latency for web service traffic",
				Priority:    10,
				Constraints: []Constraint{
					{
						Type:      ConstraintTypeLatency,
						Params:    map[string]interface{}{"max_latency_ms": 5.0},
						Mandatory: true,
					},
				},
				Goals: []Goal{
					{
						Type:     GoalTypeMinimize,
						Target:   5.0,
						Operator: GoalOperatorLessThan,
						Params:   map[string]interface{}{"metric": "latency"},
					},
				},
				Scope: IntentScope{
					Nodes:   []string{"switch-1", "switch-2"},
					Global:  false,
				},
			},
			ExpectError: false,
			MockSetup: func(m *MockAIOptimizer) {
				m.On("OptimizeIntent", mock.AnythingOfType("*controller.Intent"), mock.AnythingOfType("*network.NetworkTopology")).
					Return([]FlowRule{
						{
							ID:       uuid.New().String(),
							Priority: 9000,
							Actions: []FlowAction{
								{Type: ActionSetQueue, Params: map[string]interface{}{"queue_id": 0}},
								{Type: ActionOutput, Params: map[string]interface{}{"port": "normal"}},
							},
						},
					}, nil)
			},
			Validate: func(t *testing.T, intent *Intent) {
				assert.Equal(t, "low-latency-web-services", intent.Name)
				assert.Equal(t, 10, intent.Priority)
				assert.Len(t, intent.Constraints, 1)
				assert.Len(t, intent.Goals, 1)
			},
		},
		{
			Name: "HighThroughputIntent",
			Intent: &Intent{
				Name:        "high-throughput-data-transfer",
				Description: "Maximize throughput for bulk data transfers",
				Priority:    5,
				Constraints: []Constraint{
					{
						Type:      ConstraintTypeBandwidth,
						Params:    map[string]interface{}{"min_bandwidth_mbps": 1000.0},
						Mandatory: true,
					},
				},
				Goals: []Goal{
					{
						Type:     GoalTypeMaximize,
						Target:   10000.0,
						Operator: GoalOperatorGreaterThan,
						Params:   map[string]interface{}{"metric": "throughput"},
					},
				},
				Scope: IntentScope{
					Networks: []string{"data-network"},
					Global:   false,
				},
			},
			ExpectError: false,
			MockSetup: func(m *MockAIOptimizer) {
				m.On("OptimizeIntent", mock.AnythingOfType("*controller.Intent"), mock.AnythingOfType("*network.NetworkTopology")).
					Return([]FlowRule{
						{
							ID:       uuid.New().String(),
							Priority: 8000,
							Actions: []FlowAction{
								{Type: ActionSetQueue, Params: map[string]interface{}{"queue_id": 1}},
								{Type: ActionOutput, Params: map[string]interface{}{"port": "normal"}},
							},
						},
					}, nil)
			},
			Validate: func(t *testing.T, intent *Intent) {
				assert.Equal(t, "high-throughput-data-transfer", intent.Name)
				assert.Equal(t, 5, intent.Priority)
			},
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.Name, func(t *testing.T) {
			// Setup mock expectations
			if tc.MockSetup != nil {
				tc.MockSetup(suite.mockAIEngine)
			}
			
			// Create the intent
			err := suite.controller.CreateIntent(tc.Intent)
			if tc.ExpectError {
				assert.Error(t, err)
				return
			}
			
			assert.NoError(t, err)
			assert.NotEmpty(t, tc.Intent.ID)
			assert.Equal(t, IntentStatusPending, tc.Intent.Status)
			
			// Wait for intent processing
			suite.waitForIntentStatus(tc.Intent.ID, IntentStatusActive, 2*time.Second)
			
			// Validate processed intent
			processedIntent, err := suite.controller.GetIntent(tc.Intent.ID)
			assert.NoError(t, err)
			assert.Equal(t, IntentStatusActive, processedIntent.Status)
			
			if tc.Validate != nil {
				tc.Validate(t, processedIntent)
			}
		})
	}
}

// TestNetworkSliceManagement tests network slice creation and management
func (suite *SDNControllerTestSuite) TestNetworkSliceManagement() {
	testCases := []NetworkSliceTestCase{
		{
			Name: "UltraReliableSlice",
			Slice: &NetworkSlice{
				Name:        "critical-control-slice",
				Description: "Ultra-reliable slice for critical control traffic",
				Type:        SliceTypeUltraReliable,
				QoSProfile: QoSProfile{
					MaxLatency:    1 * time.Millisecond,
					MinBandwidth:  1000000, // 1 Mbps
					MaxJitter:     100 * time.Microsecond,
					MaxPacketLoss: 0.00001, // 0.001%
					Availability:  0.999999,
					Priority:      10,
					DSCP:          46, // EF (Expedited Forwarding)
				},
				Resources: SliceResources{
					BandwidthMbps: 10,
					ComputeNodes:  []string{"host-1", "host-2"},
				},
				Endpoints: []SliceEndpoint{
					{
						ID:      "endpoint-1",
						NodeID:  "host-1",
						Type:    "control",
						Address: "192.168.1.10",
						Port:    8080,
					},
					{
						ID:      "endpoint-2",
						NodeID:  "host-2",
						Type:    "control",
						Address: "192.168.1.20",
						Port:    8080,
					},
				},
				Policies: []SlicePolicy{
					{
						ID:   "firewall-policy-1",
						Type: PolicyTypeFirewall,
						Rules: []PolicyRule{
							{
								ID:     "allow-control-traffic",
								Match:  map[string]interface{}{"port": 8080, "protocol": "tcp"},
								Action: "allow",
								Priority: 100,
							},
						},
					},
				},
			},
			ExpectError: false,
			MockSetup: func(m *MockAIOptimizer) {
				m.On("OptimizeSliceAllocation", mock.AnythingOfType("[]*controller.NetworkSlice")).
					Return(map[string]SliceResources{
						"": {
							BandwidthMbps: 15, // Optimized bandwidth
							ComputeNodes:  []string{"host-1", "host-2"},
						},
					}, nil)
			},
			Validate: func(t *testing.T, slice *NetworkSlice) {
				assert.Equal(t, "critical-control-slice", slice.Name)
				assert.Equal(t, SliceTypeUltraReliable, slice.Type)
				assert.Equal(t, int64(1000000), slice.QoSProfile.MinBandwidth)
				assert.Len(t, slice.Endpoints, 2)
				assert.Len(t, slice.Policies, 1)
			},
		},
		{
			Name: "HighThroughputSlice",
			Slice: &NetworkSlice{
				Name:        "bulk-data-slice",
				Description: "High throughput slice for bulk data transfers",
				Type:        SliceTypeHighThroughput,
				QoSProfile: QoSProfile{
					MaxLatency:    100 * time.Millisecond,
					MinBandwidth:  100000000, // 100 Mbps
					MaxJitter:     10 * time.Millisecond,
					MaxPacketLoss: 0.001, // 0.1%
					Availability:  0.99,
					Priority:      3,
					DSCP:          34, // AF41
				},
				Resources: SliceResources{
					BandwidthMbps: 1000,
					ComputeNodes:  []string{"host-2", "host-3"},
				},
				Endpoints: []SliceEndpoint{
					{
						ID:      "data-endpoint-1",
						NodeID:  "host-2",
						Type:    "data",
						Address: "10.0.1.10",
						Port:    9000,
					},
					{
						ID:      "data-endpoint-2",
						NodeID:  "host-3",
						Type:    "data",
						Address: "10.0.1.20",
						Port:    9000,
					},
				},
			},
			ExpectError: false,
			MockSetup: func(m *MockAIOptimizer) {
				m.On("OptimizeSliceAllocation", mock.AnythingOfType("[]*controller.NetworkSlice")).
					Return(map[string]SliceResources{}, nil) // No optimization
			},
			Validate: func(t *testing.T, slice *NetworkSlice) {
				assert.Equal(t, "bulk-data-slice", slice.Name)
				assert.Equal(t, SliceTypeHighThroughput, slice.Type)
				assert.Equal(t, int64(100000000), slice.QoSProfile.MinBandwidth)
			},
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.Name, func(t *testing.T) {
			// Setup mock expectations
			if tc.MockSetup != nil {
				tc.MockSetup(suite.mockAIEngine)
			}
			
			// Create the network slice
			err := suite.controller.CreateNetworkSlice(tc.Slice)
			if tc.ExpectError {
				assert.Error(t, err)
				return
			}
			
			assert.NoError(t, err)
			assert.NotEmpty(t, tc.Slice.ID)
			assert.Equal(t, SliceStatusDeploying, tc.Slice.Status)
			
			// Wait for slice deployment
			suite.waitForSliceStatus(tc.Slice.ID, SliceStatusActive, 3*time.Second)
			
			// Validate deployed slice
			suite.controller.slicesMutex.RLock()
			deployedSlice, exists := suite.controller.networkSlices[tc.Slice.ID]
			suite.controller.slicesMutex.RUnlock()
			
			assert.True(t, exists)
			assert.Equal(t, SliceStatusActive, deployedSlice.Status)
			
			if tc.Validate != nil {
				tc.Validate(t, deployedSlice)
			}
		})
	}
}

// TestConcurrentFlowProcessing tests the controller's ability to handle concurrent operations
func (suite *SDNControllerTestSuite) TestConcurrentFlowProcessing() {
	numWorkers := 10
	numRulesPerWorker := 5
	var wg sync.WaitGroup
	var successCount int32
	var errorCount int32
	var mu sync.Mutex
	
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for j := 0; j < numRulesPerWorker; j++ {
				rule := &FlowRule{
					ID:          uuid.New().String(),
					Priority:    1000 + workerID*100 + j,
					TableID:     0,
					IdleTimeout: 60,
					HardTimeout: 300,
					Match: FlowMatch{
						EthType: "0x0800",
						IPSrc:   fmt.Sprintf("192.168.%d.0/24", workerID),
					},
					Actions: []FlowAction{
						{
							Type: ActionOutput,
							Params: map[string]interface{}{
								"port": fmt.Sprintf("%d", j+1),
							},
						},
					},
				}
				
				err := suite.controller.installFlowRule(rule)
				
				mu.Lock()
				if err != nil {
					errorCount++
				} else {
					successCount++
				}
				mu.Unlock()
			}
		}(i)
	}
	
	wg.Wait()
	
	mu.Lock()
	totalExpected := int32(numWorkers * numRulesPerWorker)
	mu.Unlock()
	
	assert.Equal(suite.T(), totalExpected, successCount, "All flow rules should be installed successfully")
	assert.Equal(suite.T(), int32(0), errorCount, "No errors should occur during concurrent processing")
	
	// Verify all rules are stored
	suite.controller.flowRulesMutex.RLock()
	actualRuleCount := len(suite.controller.flowRules)
	suite.controller.flowRulesMutex.RUnlock()
	
	assert.Equal(suite.T(), int(totalExpected), actualRuleCount, "All rules should be stored in controller")
}

// TestMetricsCollection tests SDN controller metrics collection
func (suite *SDNControllerTestSuite) TestMetricsCollection() {
	// Create some test data
	intent := &Intent{
		Name:        "test-metrics-intent",
		Description: "Intent for testing metrics",
		Priority:    5,
		Status:      IntentStatusActive,
	}
	suite.controller.CreateIntent(intent)
	
	slice := &NetworkSlice{
		Name:        "test-metrics-slice",
		Description: "Slice for testing metrics",
		Type:        SliceTypeBestEffort,
		Status:      SliceStatusActive,
		QoSProfile: QoSProfile{
			MaxLatency:   10 * time.Millisecond,
			MinBandwidth: 1000000,
		},
		Endpoints: []SliceEndpoint{
			{ID: "test-endpoint", NodeID: "host-1", Type: "test"},
		},
	}
	suite.controller.CreateNetworkSlice(slice)
	
	// Wait for metrics to be updated
	time.Sleep(1 * time.Second)
	
	// Get metrics
	metrics := suite.controller.GetMetrics()
	
	assert.NotNil(suite.T(), metrics)
	assert.True(suite.T(), metrics.ActiveIntents > 0, "Should have active intents")
	assert.True(suite.T(), metrics.ActiveSlices > 0, "Should have active slices")
	assert.NotZero(suite.T(), metrics.LastUpdated, "LastUpdated should be set")
}

// TestFlowRuleValidation tests flow rule validation
func (suite *SDNControllerTestSuite) TestFlowRuleValidation() {
	testCases := []struct {
		name        string
		rule        *FlowRule
		expectError bool
		errorMsg    string
	}{
		{
			name: "ValidRule",
			rule: &FlowRule{
				ID:       uuid.New().String(),
				Priority: 1000,
				Actions: []FlowAction{
					{Type: ActionOutput, Params: map[string]interface{}{"port": "1"}},
				},
			},
			expectError: false,
		},
		{
			name: "InvalidPriority",
			rule: &FlowRule{
				ID:       uuid.New().String(),
				Priority: -1,
				Actions: []FlowAction{
					{Type: ActionOutput, Params: map[string]interface{}{"port": "1"}},
				},
			},
			expectError: true,
		},
		{
			name: "NoActions",
			rule: &FlowRule{
				ID:       uuid.New().String(),
				Priority: 1000,
				Actions:  []FlowAction{},
			},
			expectError: true,
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			err := suite.controller.installFlowRule(tc.rule)
			
			if tc.expectError {
				assert.Error(t, err)
				if tc.errorMsg != "" {
					assert.Contains(t, err.Error(), tc.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// Helper methods

// waitForIntentStatus waits for an intent to reach a specific status
func (suite *SDNControllerTestSuite) waitForIntentStatus(intentID string, expectedStatus IntentStatus, timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-timer.C:
			suite.T().Fatalf("Timeout waiting for intent %s to reach status %s", intentID, expectedStatus)
		case <-ticker.C:
			suite.controller.intentsMutex.RLock()
			if intent, exists := suite.controller.intents[intentID]; exists {
				if intent.Status == expectedStatus {
					suite.controller.intentsMutex.RUnlock()
					return
				}
			}
			suite.controller.intentsMutex.RUnlock()
		}
	}
}

// waitForSliceStatus waits for a network slice to reach a specific status
func (suite *SDNControllerTestSuite) waitForSliceStatus(sliceID string, expectedStatus SliceStatus, timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-timer.C:
			suite.T().Fatalf("Timeout waiting for slice %s to reach status %s", sliceID, expectedStatus)
		case <-ticker.C:
			suite.controller.slicesMutex.RLock()
			if slice, exists := suite.controller.networkSlices[sliceID]; exists {
				if slice.Status == expectedStatus {
					suite.controller.slicesMutex.RUnlock()
					return
				}
			}
			suite.controller.slicesMutex.RUnlock()
		}
	}
}

// TestSDNControllerSuite runs the complete SDN controller test suite
func TestSDNControllerSuite(t *testing.T) {
	suite.Run(t, new(SDNControllerTestSuite))
}

// TestSDNControllerCreationStandalone tests basic controller creation
func TestSDNControllerCreationStandalone(t *testing.T) {
	config := DefaultSDNControllerConfig()
	topology := network.NewNetworkTopology()
	
	controller := NewSDNController(config, topology, nil)
	
	assert.NotNil(t, controller)
	assert.Equal(t, config, controller.config)
	assert.Equal(t, topology, controller.networkTopology)
}

// TestFlowRuleGeneration tests flow rule generation for different slice types
func TestFlowRuleGeneration(t *testing.T) {
	config := DefaultSDNControllerConfig()
	topology := network.NewNetworkTopology()
	controller := NewSDNController(config, topology, nil)
	
	testCases := []struct {
		name      string
		sliceType SliceType
		validate  func(t *testing.T, rules []*FlowRule)
	}{
		{
			name:      "LowLatencyRules",
			sliceType: SliceTypeLowLatency,
			validate: func(t *testing.T, rules []*FlowRule) {
				assert.NotEmpty(t, rules)
				// Should have high priority rules
				for _, rule := range rules {
					assert.True(t, rule.Priority >= 8000, "Low latency rules should have high priority")
				}
			},
		},
		{
			name:      "HighThroughputRules",
			sliceType: SliceTypeHighThroughput,
			validate: func(t *testing.T, rules []*FlowRule) {
				assert.NotEmpty(t, rules)
				// Should have medium priority rules optimized for throughput
				for _, rule := range rules {
					assert.True(t, rule.Priority >= 5000, "High throughput rules should have reasonable priority")
				}
			},
		},
		{
			name:      "BestEffortRules",
			sliceType: SliceTypeBestEffort,
			validate: func(t *testing.T, rules []*FlowRule) {
				assert.NotEmpty(t, rules)
				// Should have lower priority rules
				for _, rule := range rules {
					assert.True(t, rule.Priority < 5000, "Best effort rules should have lower priority")
				}
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			slice := &NetworkSlice{
				ID:   uuid.New().String(),
				Name: fmt.Sprintf("test-slice-%s", tc.name),
				Type: tc.sliceType,
				Endpoints: []SliceEndpoint{
					{ID: "ep1", NodeID: "node1"},
				},
			}
			
			rules, err := controller.generateSliceFlowRules(slice)
			assert.NoError(t, err)
			
			if tc.validate != nil {
				tc.validate(t, rules)
			}
		})
	}
}