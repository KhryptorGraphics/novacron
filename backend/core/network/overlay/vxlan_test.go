package overlay

import (
	"context"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/network/overlay/drivers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// VXLANTestSuite provides comprehensive testing for VXLAN overlay networking
type VXLANTestSuite struct {
	suite.Suite
	vxlanDriver      *drivers.VXLANDriver
	overlayManager   *NetworkOverlayManager
	testNetworks     []OverlayNetwork
	testEndpoints    []EndpointConfig
	testContext      context.Context
	testCancel       context.CancelFunc
	vxlanSimulator   *VXLANSimulator
	performanceTest  *VXLANPerformanceTest
}

// VXLANSimulator simulates VXLAN network conditions for testing
type VXLANSimulator struct {
	vteps           map[string]*VTEP
	vxlanNetworks   map[uint32]*VXLANNetwork
	tunnels         map[string]*VXLANTunnel
	macTable        map[string]*MACEntry
	arpTable        map[string]*ARPEntry
	multicastGroups map[string][]string
	mutex           sync.RWMutex
}

// VTEP represents a VXLAN Tunnel Endpoint
type VTEP struct {
	ID              string
	IP              net.IP
	UDPPort         int
	SupportedVNIs   []uint32
	TunnelEndpoint  string
	State           VTEPState
	Stats           *VTEPStats
	HeartbeatTime   time.Time
	LastSeen        time.Time
}

// VXLANNetwork represents a VXLAN network segment
type VXLANNetwork struct {
	VNI             uint32
	MulticastGroup  net.IP
	FloodMode       FloodMode
	ConnectedVTEPs  map[string]*VTEP
	BridgeDomain    string
	VLANMapping     uint16
	Stats           *VXLANNetworkStats
	Created         time.Time
}

// VXLANTunnel represents a VXLAN tunnel between VTEPs
type VXLANTunnel struct {
	ID              string
	SourceVTEP      string
	DestinationVTEP string
	VNI             uint32
	State           TunnelState
	MTU             int
	EncapOverhead   int
	Stats           *VXLANTunnelStats
	LastPacketTime  time.Time
}

// MACEntry represents a MAC address table entry
type MACEntry struct {
	MAC         net.HardwareAddr
	VNI         uint32
	VTEP        string
	Port        string
	LearnedTime time.Time
	Static      bool
	TTL         time.Duration
}

// ARPEntry represents an ARP table entry
type ARPEntry struct {
	IP          net.IP
	MAC         net.HardwareAddr
	VNI         uint32
	VTEP        string
	LearnedTime time.Time
	Static      bool
	TTL         time.Duration
}

// VXLANPerformanceTest handles VXLAN performance testing
type VXLANPerformanceTest struct {
	testScenarios   map[string]*PerformanceScenario
	benchmarkData   *BenchmarkData
	networkTopology *NetworkTopology
	trafficGen      *TrafficGenerator
}

// Performance and state enums
type VTEPState string
type TunnelState string
type FloodMode string

const (
	VTEPStateUp      VTEPState = "up"
	VTEPStateDown    VTEPState = "down"
	VTEPStateLearning VTEPState = "learning"
	
	TunnelStateUp       TunnelState = "up"
	TunnelStateDown     TunnelState = "down"
	TunnelStateBlocked  TunnelState = "blocked"
	
	FloodModeMulticast FloodMode = "multicast"
	FloodModeUnicast   FloodMode = "unicast"
	FloodModeDisabled  FloodMode = "disabled"
)

// Statistics structures
type VTEPStats struct {
	PacketsIn      uint64
	PacketsOut     uint64
	BytesIn        uint64
	BytesOut       uint64
	TunnelsActive  int
	MACEntries     int
	ARPEntries     int
	Errors         uint64
	LastUpdate     time.Time
}

type VXLANNetworkStats struct {
	ActiveVTEPs    int
	ActiveTunnels  int
	BroadcastsSent uint64
	FloodedPackets uint64
	LearnedMACs    int
	DroppedPackets uint64
	LastUpdate     time.Time
}

type VXLANTunnelStats struct {
	PacketsIn       uint64
	PacketsOut      uint64
	BytesIn         uint64
	BytesOut        uint64
	EncapPackets    uint64
	DecapPackets    uint64
	ErrorsIn        uint64
	ErrorsOut       uint64
	LastUpdate      time.Time
}

// Performance testing structures
type PerformanceScenario struct {
	Name            string
	VTEPCount       int
	VNICount        int
	PacketSize      int
	PacketsPerSecond int
	Duration        time.Duration
	TestType        PerformanceTestType
	ExpectedResults *ExpectedPerformance
}

type PerformanceTestType string

const (
	PerformanceTestThroughput PerformanceTestType = "throughput"
	PerformanceTestLatency    PerformanceTestType = "latency"
	PerformanceTestScale      PerformanceTestType = "scale"
	PerformanceTestResilience PerformanceTestType = "resilience"
)

type ExpectedPerformance struct {
	MinThroughputMbps  float64
	MaxLatencyMs       float64
	MaxPacketLossRate  float64
	MinScaleVTEPs      int
	RecoveryTimeMs     float64
}

type BenchmarkData struct {
	TestResults     map[string]*TestResult
	ThroughputData  []ThroughputMeasurement
	LatencyData     []LatencyMeasurement
	ScaleData       []ScaleMeasurement
	ResilienceData  []ResilienceMeasurement
}

type TestResult struct {
	ScenarioName    string
	StartTime       time.Time
	EndTime         time.Time
	Success         bool
	ThroughputMbps  float64
	LatencyMs       float64
	PacketLossRate  float64
	ErrorCount      int
	Details         map[string]interface{}
}

type ThroughputMeasurement struct {
	Timestamp   time.Time
	Throughput  float64
	PacketRate  uint64
	ByteRate    uint64
}

type LatencyMeasurement struct {
	Timestamp   time.Time
	Latency     time.Duration
	MinLatency  time.Duration
	MaxLatency  time.Duration
	Jitter      time.Duration
}

type ScaleMeasurement struct {
	Timestamp   time.Time
	VTEPCount   int
	VNICount    int
	TunnelCount int
	MACEntries  int
	Performance float64
}

type ResilienceMeasurement struct {
	Timestamp    time.Time
	EventType    string
	RecoveryTime time.Duration
	PacketsLost  uint64
	ServiceImpact float64
}

type NetworkTopology struct {
	VTEPs    map[string]*VTEP
	Links    map[string]*NetworkLink
	Subnets  map[string]*NetworkSubnet
}

type NetworkLink struct {
	SourceVTEP    string
	DestVTEP      string
	Bandwidth     int64
	Latency       time.Duration
	PacketLoss    float64
	State         string
}

type NetworkSubnet struct {
	CIDR     string
	VNI      uint32
	Gateway  net.IP
	DHCP     bool
}

type TrafficGenerator struct {
	patterns    map[string]*TrafficPattern
	generators  map[string]*Generator
}

type TrafficPattern struct {
	Name         string
	PacketSize   int
	Rate         int
	Duration     time.Duration
	Protocol     string
	SourceMAC    net.HardwareAddr
	DestMAC      net.HardwareAddr
	SourceIP     net.IP
	DestIP       net.IP
}

type Generator struct {
	ID        string
	VTEP      string
	Active    bool
	Pattern   *TrafficPattern
	Stats     *GeneratorStats
}

type GeneratorStats struct {
	PacketsSent   uint64
	BytesSent     uint64
	PacketsRecv   uint64
	BytesRecv     uint64
	Errors        uint64
	LastUpdate    time.Time
}

// SetupSuite initializes the test suite
func (suite *VXLANTestSuite) SetupSuite() {
	suite.testContext, suite.testCancel = context.WithCancel(context.Background())
	
	// Initialize VXLAN driver
	config := drivers.DefaultVXLANConfig()
	config.MinMTU = 1400
	config.DefaultMulticastGroup = "239.1.1.100"
	config.UDPPort = 4789
	
	suite.vxlanDriver = drivers.NewVXLANDriver(config)
	err := suite.vxlanDriver.Initialize(suite.testContext)
	suite.Require().NoError(err, "VXLAN driver initialization should succeed")
	
	// Initialize overlay manager
	suite.overlayManager = NewNetworkOverlayManager()
	suite.overlayManager.RegisterDriver(suite.vxlanDriver)
	
	// Initialize VXLAN simulator
	suite.vxlanSimulator = NewVXLANSimulator()
	
	// Initialize performance test framework
	suite.performanceTest = NewVXLANPerformanceTest()
	
	// Create test networks and endpoints
	suite.createTestVXLANNetworks()
}

// TearDownSuite cleans up after all tests
func (suite *VXLANTestSuite) TearDownSuite() {
	// Cleanup test networks
	suite.cleanupTestNetworks()
	
	// Shutdown driver
	if suite.vxlanDriver != nil {
		suite.vxlanDriver.Shutdown(suite.testContext)
	}
	
	// Cleanup simulator
	if suite.vxlanSimulator != nil {
		suite.vxlanSimulator.Cleanup()
	}
	
	if suite.testCancel != nil {
		suite.testCancel()
	}
}

// NewVXLANSimulator creates a new VXLAN simulator
func NewVXLANSimulator() *VXLANSimulator {
	return &VXLANSimulator{
		vteps:           make(map[string]*VTEP),
		vxlanNetworks:   make(map[uint32]*VXLANNetwork),
		tunnels:         make(map[string]*VXLANTunnel),
		macTable:        make(map[string]*MACEntry),
		arpTable:        make(map[string]*ARPEntry),
		multicastGroups: make(map[string][]string),
	}
}

// NewVXLANPerformanceTest creates a new VXLAN performance test framework
func NewVXLANPerformanceTest() *VXLANPerformanceTest {
	return &VXLANPerformanceTest{
		testScenarios: make(map[string]*PerformanceScenario),
		benchmarkData: &BenchmarkData{
			TestResults:    make(map[string]*TestResult),
			ThroughputData: []ThroughputMeasurement{},
			LatencyData:    []LatencyMeasurement{},
			ScaleData:      []ScaleMeasurement{},
			ResilienceData: []ResilienceMeasurement{},
		},
		networkTopology: &NetworkTopology{
			VTEPs:   make(map[string]*VTEP),
			Links:   make(map[string]*NetworkLink),
			Subnets: make(map[string]*NetworkSubnet),
		},
		trafficGen: &TrafficGenerator{
			patterns:   make(map[string]*TrafficPattern),
			generators: make(map[string]*Generator),
		},
	}
}

// createTestVXLANNetworks creates standard test VXLAN networks
func (suite *VXLANTestSuite) createTestVXLANNetworks() {
	// Create basic VXLAN network
	basicNetwork := OverlayNetwork{
		ID:   "vxlan-basic-" + uuid.New().String(),
		Name: "basic-vxlan-test",
		Type: VXLAN,
		Subnets: []OverlaySubnet{
			{
				CIDR:    "192.168.100.0/24",
				Gateway: "192.168.100.1",
			},
		},
		MTU: 1450,
		Options: map[string]string{
			"vni":             "1000",
			"multicast_group": "239.1.1.100",
			"udp_port":        "4789",
		},
	}
	
	err := suite.vxlanDriver.CreateNetwork(suite.testContext, basicNetwork)
	suite.Require().NoError(err)
	suite.testNetworks = append(suite.testNetworks, basicNetwork)
	
	// Create multi-tenant VXLAN network
	multiTenantNetwork := OverlayNetwork{
		ID:   "vxlan-multitenant-" + uuid.New().String(),
		Name: "multitenant-vxlan-test",
		Type: VXLAN,
		Subnets: []OverlaySubnet{
			{
				CIDR:    "10.100.0.0/16",
				Gateway: "10.100.0.1",
			},
		},
		MTU: 1450,
		Options: map[string]string{
			"vni":             "2000",
			"multicast_group": "239.1.1.200",
			"udp_port":        "4789",
		},
		Metadata: map[string]string{
			"tenant":    "test-tenant-1",
			"isolation": "strict",
		},
	}
	
	err = suite.vxlanDriver.CreateNetwork(suite.testContext, multiTenantNetwork)
	suite.Require().NoError(err)
	suite.testNetworks = append(suite.testNetworks, multiTenantNetwork)
	
	// Create high-performance VXLAN network
	highPerfNetwork := OverlayNetwork{
		ID:   "vxlan-highperf-" + uuid.New().String(),
		Name: "highperf-vxlan-test",
		Type: VXLAN,
		Subnets: []OverlaySubnet{
			{
				CIDR:    "172.16.0.0/16",
				Gateway: "172.16.0.1",
			},
		},
		MTU: 9000, // Jumbo frames
		Options: map[string]string{
			"vni":                "3000",
			"multicast_group":    "239.1.1.300",
			"udp_port":           "4789",
			"hardware_offload":   "true",
			"checksum_offload":   "true",
		},
		Metadata: map[string]string{
			"performance": "high",
			"offload":     "enabled",
		},
	}
	
	err = suite.vxlanDriver.CreateNetwork(suite.testContext, highPerfNetwork)
	suite.Require().NoError(err)
	suite.testNetworks = append(suite.testNetworks, highPerfNetwork)
}

// cleanupTestNetworks removes all test networks
func (suite *VXLANTestSuite) cleanupTestNetworks() {
	for _, network := range suite.testNetworks {
		err := suite.vxlanDriver.DeleteNetwork(suite.testContext, network.ID)
		if err != nil {
			suite.T().Logf("Warning: Failed to cleanup test network %s: %v", network.ID, err)
		}
	}
	suite.testNetworks = nil
}

// TestVXLANNetworkCreation tests VXLAN network creation and configuration
func (suite *VXLANTestSuite) TestVXLANNetworkCreation() {
	testCases := []struct {
		name        string
		network     OverlayNetwork
		expectError bool
		validate    func(t *testing.T, network OverlayNetwork)
	}{
		{
			name: "ValidVXLANNetwork",
			network: OverlayNetwork{
				ID:   "test-valid-" + uuid.New().String(),
				Name: "valid-vxlan",
				Type: VXLAN,
				Subnets: []OverlaySubnet{
					{CIDR: "192.168.1.0/24", Gateway: "192.168.1.1"},
				},
				MTU: 1450,
				Options: map[string]string{
					"vni":             "5000",
					"multicast_group": "239.1.1.50",
				},
			},
			expectError: false,
			validate: func(t *testing.T, network OverlayNetwork) {
				assert.Equal(t, VXLAN, network.Type)
				assert.Equal(t, "5000", network.Options["vni"])
				assert.Equal(t, 1450, network.MTU)
			},
		},
		{
			name: "InvalidMTU",
			network: OverlayNetwork{
				ID:   "test-invalid-mtu-" + uuid.New().String(),
				Name: "invalid-mtu-vxlan",
				Type: VXLAN,
				MTU:  500, // Too small
				Options: map[string]string{
					"vni": "6000",
				},
			},
			expectError: true,
		},
		{
			name: "InvalidVNI",
			network: OverlayNetwork{
				ID:   "test-invalid-vni-" + uuid.New().String(),
				Name: "invalid-vni-vxlan",
				Type: VXLAN,
				MTU:  1450,
				Options: map[string]string{
					"vni": "16777216", // VNI too large (max is 16777215)
				},
			},
			expectError: true,
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			err := suite.vxlanDriver.CreateNetwork(suite.testContext, tc.network)
			
			if tc.expectError {
				assert.Error(t, err, "Network creation should fail")
			} else {
				assert.NoError(t, err, "Network creation should succeed")
				
				// Verify network exists
				retrievedNet, err := suite.vxlanDriver.GetNetwork(suite.testContext, tc.network.ID)
				assert.NoError(t, err)
				assert.Equal(t, tc.network.ID, retrievedNet.ID)
				
				if tc.validate != nil {
					tc.validate(t, retrievedNet)
				}
				
				// Cleanup
				suite.vxlanDriver.DeleteNetwork(suite.testContext, tc.network.ID)
			}
		})
	}
}

// TestVXLANEndpointManagement tests VXLAN endpoint operations
func (suite *VXLANTestSuite) TestVXLANEndpointManagement() {
	// Use first test network
	require.NotEmpty(suite.T(), suite.testNetworks, "Should have test networks")
	testNetwork := suite.testNetworks[0]
	
	testCases := []struct {
		name     string
		endpoint EndpointConfig
		validate func(t *testing.T, endpoint EndpointConfig)
	}{
		{
			name: "BasicVXLANEndpoint",
			endpoint: EndpointConfig{
				NetworkID: testNetwork.ID,
				Name:      "basic-endpoint-" + uuid.New().String()[:8],
				Type:      "vm",
				IPAddress: "192.168.100.10",
				MACAddress: "02:00:00:00:00:01",
				Metadata: map[string]string{
					"vm_id": "test-vm-1",
					"vtep":  "192.168.1.10",
				},
			},
			validate: func(t *testing.T, endpoint EndpointConfig) {
				assert.Equal(t, "vm", endpoint.Type)
				assert.Equal(t, "192.168.100.10", endpoint.IPAddress)
				assert.NotEmpty(t, endpoint.MACAddress)
			},
		},
		{
			name: "ContainerVXLANEndpoint",
			endpoint: EndpointConfig{
				NetworkID: testNetwork.ID,
				Name:      "container-endpoint-" + uuid.New().String()[:8],
				Type:      "container",
				IPAddress: "192.168.100.20",
				MACAddress: "02:00:00:00:00:02",
				Metadata: map[string]string{
					"container_id": "test-container-1",
					"vtep":         "192.168.1.20",
					"namespace":    "default",
				},
			},
			validate: func(t *testing.T, endpoint EndpointConfig) {
				assert.Equal(t, "container", endpoint.Type)
				assert.Equal(t, "default", endpoint.Metadata["namespace"])
			},
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Create endpoint
			err := suite.vxlanDriver.CreateEndpoint(suite.testContext, tc.endpoint)
			require.NoError(t, err, "Endpoint creation should succeed")
			
			// Verify endpoint exists
			retrievedEndpoint, err := suite.vxlanDriver.GetEndpoint(suite.testContext, tc.endpoint.NetworkID, tc.endpoint.Name)
			assert.NoError(t, err)
			assert.Equal(t, tc.endpoint.Name, retrievedEndpoint.Name)
			
			if tc.validate != nil {
				tc.validate(t, retrievedEndpoint)
			}
			
			// Test endpoint listing
			endpoints, err := suite.vxlanDriver.ListEndpoints(suite.testContext, tc.endpoint.NetworkID)
			assert.NoError(t, err)
			assert.NotEmpty(t, endpoints)
			
			// Cleanup endpoint
			err = suite.vxlanDriver.DeleteEndpoint(suite.testContext, tc.endpoint.NetworkID, tc.endpoint.Name)
			assert.NoError(t, err, "Endpoint deletion should succeed")
		})
	}
}

// TestVXLANConnectivity tests VXLAN connectivity between endpoints
func (suite *VXLANTestSuite) TestVXLANConnectivity() {
	// Setup simulation environment
	err := suite.setupVXLANConnectivityTest()
	require.NoError(suite.T(), err)
	
	// Test scenarios
	testCases := []struct {
		name           string
		sourceVTEP     string
		destVTEP       string
		vni            uint32
		expectedResult bool
		description    string
	}{
		{
			name:           "SameVNIConnectivity",
			sourceVTEP:     "vtep-1",
			destVTEP:       "vtep-2",
			vni:            1000,
			expectedResult: true,
			description:    "Endpoints in same VNI should be able to communicate",
		},
		{
			name:           "DifferentVNIIsolation",
			sourceVTEP:     "vtep-1",
			destVTEP:       "vtep-3",
			vni:            1000, // vtep-3 is in VNI 2000
			expectedResult: false,
			description:    "Endpoints in different VNIs should be isolated",
		},
		{
			name:           "VTEPReachability",
			sourceVTEP:     "vtep-2",
			destVTEP:       "vtep-4",
			vni:            1000,
			expectedResult: true,
			description:    "All VTEPs in same VNI should reach each other",
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Test connectivity using simulator
			canConnect, err := suite.vxlanSimulator.TestConnectivity(tc.sourceVTEP, tc.destVTEP, tc.vni)
			assert.NoError(t, err)
			assert.Equal(t, tc.expectedResult, canConnect, tc.description)
			
			if canConnect {
				// Verify tunnel establishment
				tunnel, err := suite.vxlanSimulator.GetTunnel(tc.sourceVTEP, tc.destVTEP)
				assert.NoError(t, err)
				assert.Equal(t, TunnelStateUp, tunnel.State)
			}
		})
	}
}

// TestVXLANPerformance tests VXLAN performance characteristics
func (suite *VXLANTestSuite) TestVXLANPerformance() {
	// Setup performance test scenarios
	scenarios := []*PerformanceScenario{
		{
			Name:            "BasicThroughput",
			VTEPCount:       4,
			VNICount:        1,
			PacketSize:      1450,
			PacketsPerSecond: 10000,
			Duration:        10 * time.Second,
			TestType:        PerformanceTestThroughput,
			ExpectedResults: &ExpectedPerformance{
				MinThroughputMbps: 100.0,
				MaxLatencyMs:      10.0,
				MaxPacketLossRate: 0.01,
			},
		},
		{
			Name:            "LowLatency",
			VTEPCount:       2,
			VNICount:        1,
			PacketSize:      64,
			PacketsPerSecond: 1000,
			Duration:        5 * time.Second,
			TestType:        PerformanceTestLatency,
			ExpectedResults: &ExpectedPerformance{
				MaxLatencyMs:      1.0,
				MaxPacketLossRate: 0.001,
			},
		},
		{
			Name:            "ScaleTest",
			VTEPCount:       100,
			VNICount:        10,
			PacketSize:      1450,
			PacketsPerSecond: 1000,
			Duration:        30 * time.Second,
			TestType:        PerformanceTestScale,
			ExpectedResults: &ExpectedPerformance{
				MinScaleVTEPs:     50,
				MaxPacketLossRate: 0.05,
			},
		},
	}
	
	for _, scenario := range scenarios {
		suite.T().Run(scenario.Name, func(t *testing.T) {
			result, err := suite.runPerformanceTest(scenario)
			assert.NoError(t, err, "Performance test should complete without error")
			assert.True(t, result.Success, "Performance test should meet expectations")
			
			// Validate specific metrics based on test type
			switch scenario.TestType {
			case PerformanceTestThroughput:
				assert.GreaterOrEqual(t, result.ThroughputMbps, scenario.ExpectedResults.MinThroughputMbps,
					"Throughput should meet minimum requirement")
			case PerformanceTestLatency:
				assert.LessOrEqual(t, result.LatencyMs, scenario.ExpectedResults.MaxLatencyMs,
					"Latency should be within acceptable range")
			case PerformanceTestScale:
				scaleMetric, ok := result.Details["scale_metric"].(float64)
				assert.True(t, ok, "Scale metric should be available")
				assert.GreaterOrEqual(t, scaleMetric, float64(scenario.ExpectedResults.MinScaleVTEPs),
					"Scale test should handle minimum VTEP count")
			}
		})
	}
}

// TestVXLANMTUHandling tests VXLAN MTU discovery and fragmentation
func (suite *VXLANTestSuite) TestVXLANMTUHandling() {
	testCases := []struct {
		name        string
		tunnelMTU   int
		payloadSize int
		expectFrag  bool
		expectError bool
	}{
		{
			name:        "NormalPacket",
			tunnelMTU:   1500,
			payloadSize: 1400,
			expectFrag:  false,
			expectError: false,
		},
		{
			name:        "RequiresFragmentation",
			tunnelMTU:   1500,
			payloadSize: 1600,
			expectFrag:  true,
			expectError: false,
		},
		{
			name:        "JumboFrame",
			tunnelMTU:   9000,
			payloadSize: 8500,
			expectFrag:  false,
			expectError: false,
		},
		{
			name:        "ExcessiveSize",
			tunnelMTU:   1500,
			payloadSize: 10000,
			expectFrag:  true,
			expectError: true, // May fail if fragmentation not supported
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Simulate packet transmission
			result, err := suite.vxlanSimulator.SimulatePacketTransmission(
				"vtep-1", "vtep-2", tc.payloadSize, tc.tunnelMTU)
			
			if tc.expectError {
				assert.Error(t, err, "Should fail for excessive packet sizes")
			} else {
				assert.NoError(t, err, "Packet transmission should succeed")
				assert.Equal(t, tc.expectFrag, result.Fragmented,
					"Fragmentation behavior should match expectation")
			}
		})
	}
}

// TestVXLANEncapsulation tests VXLAN encapsulation and decapsulation
func (suite *VXLANTestSuite) TestVXLANEncapsulation() {
	testPackets := []struct {
		name           string
		originalPacket *EthernetFrame
		vni            uint32
		sourceVTEP     net.IP
		destVTEP       net.IP
		validate       func(t *testing.T, encapped *VXLANPacket, original *EthernetFrame)
	}{
		{
			name: "BasicEthernetFrame",
			originalPacket: &EthernetFrame{
				SrcMAC:  net.HardwareAddr{0x02, 0x00, 0x00, 0x00, 0x00, 0x01},
				DstMAC:  net.HardwareAddr{0x02, 0x00, 0x00, 0x00, 0x00, 0x02},
				EthType: 0x0800,
				Payload: []byte("Hello VXLAN World!"),
			},
			vni:        1000,
			sourceVTEP: net.IPv4(192, 168, 1, 10),
			destVTEP:   net.IPv4(192, 168, 1, 20),
			validate: func(t *testing.T, encapped *VXLANPacket, original *EthernetFrame) {
				assert.Equal(t, uint32(1000), encapped.VNI)
				assert.Equal(t, original.SrcMAC, encapped.InnerFrame.SrcMAC)
				assert.Equal(t, original.DstMAC, encapped.InnerFrame.DstMAC)
				assert.Equal(t, 8, len(encapped.VXLANHeader)) // 8-byte VXLAN header
			},
		},
	}
	
	for _, tc := range testPackets {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Test encapsulation
			encapped, err := suite.vxlanSimulator.EncapsulatePacket(
				tc.originalPacket, tc.vni, tc.sourceVTEP, tc.destVTEP)
			require.NoError(t, err, "Encapsulation should succeed")
			
			if tc.validate != nil {
				tc.validate(t, encapped, tc.originalPacket)
			}
			
			// Test decapsulation
			decapped, err := suite.vxlanSimulator.DecapsulatePacket(encapped)
			require.NoError(t, err, "Decapsulation should succeed")
			
			// Verify round-trip integrity
			assert.Equal(t, tc.originalPacket.SrcMAC, decapped.SrcMAC)
			assert.Equal(t, tc.originalPacket.DstMAC, decapped.DstMAC)
			assert.Equal(t, tc.originalPacket.EthType, decapped.EthType)
			assert.Equal(t, tc.originalPacket.Payload, decapped.Payload)
		})
	}
}

// TestVXLANMulticast tests VXLAN multicast operations
func (suite *VXLANTestSuite) TestVXLANMulticast() {
	multicastGroup := "239.1.1.100"
	vni := uint32(1000)
	
	// Setup multicast group
	err := suite.vxlanSimulator.SetupMulticastGroup(multicastGroup, vni)
	require.NoError(suite.T(), err)
	
	// Add VTEPs to multicast group
	vteps := []string{"vtep-1", "vtep-2", "vtep-3", "vtep-4"}
	for _, vtep := range vteps {
		err := suite.vxlanSimulator.JoinMulticastGroup(vtep, multicastGroup)
		assert.NoError(suite.T(), err, "VTEP should be able to join multicast group")
	}
	
	// Test broadcast/flood behavior
	result, err := suite.vxlanSimulator.SimulateFlood("vtep-1", vni, &EthernetFrame{
		SrcMAC:  net.HardwareAddr{0x02, 0x00, 0x00, 0x00, 0x00, 0x01},
		DstMAC:  net.HardwareAddr{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}, // Broadcast
		EthType: 0x0800,
		Payload: []byte("Broadcast message"),
	})
	
	require.NoError(suite.T(), err)
	assert.Len(suite.T(), result.ReceivedBy, 3, "Broadcast should reach 3 other VTEPs")
	
	// Verify all VTEPs except source received the frame
	expectedVTEPs := []string{"vtep-2", "vtep-3", "vtep-4"}
	for _, vtep := range expectedVTEPs {
		assert.Contains(suite.T(), result.ReceivedBy, vtep, "VTEP %s should receive broadcast", vtep)
	}
}

// TestVXLANResilience tests VXLAN resilience and failure recovery
func (suite *VXLANTestSuite) TestVXLANResilience() {
	// Setup test environment with multiple VTEPs
	err := suite.setupResilienceTest()
	require.NoError(suite.T(), err)
	
	resilienceTests := []struct {
		name        string
		faultType   string
		target      string
		expectation string
	}{
		{
			name:        "VTEPFailure",
			faultType:   "vtep_down",
			target:      "vtep-2",
			expectation: "traffic_rerouted",
		},
		{
			name:        "TunnelFailure", 
			faultType:   "tunnel_down",
			target:      "vtep-1-vtep-2",
			expectation: "alternate_path_used",
		},
		{
			name:        "NetworkPartition",
			faultType:   "partition",
			target:      "vtep-1,vtep-2",
			expectation: "split_brain_handled",
		},
	}
	
	for _, test := range resilienceTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Measure baseline performance
			baseline, err := suite.measureBaselinePerformance()
			require.NoError(t, err)
			
			// Inject fault
			err = suite.vxlanSimulator.InjectFault(test.faultType, test.target)
			require.NoError(t, err)
			
			// Measure recovery time and service impact
			recovery, err := suite.measureRecoveryMetrics(test.expectation)
			assert.NoError(t, err)
			
			// Validate resilience expectations
			assert.LessOrEqual(t, recovery.RecoveryTime, 5*time.Second,
				"Recovery should complete within 5 seconds")
			assert.LessOrEqual(t, recovery.ServiceImpact, 0.1,
				"Service impact should be less than 10%")
			
			// Clear fault and verify full recovery
			err = suite.vxlanSimulator.ClearFault(test.faultType, test.target)
			assert.NoError(t, err)
			
			// Verify performance returns to baseline
			final, err := suite.measureBaselinePerformance()
			assert.NoError(t, err)
			assert.InDelta(t, baseline.ThroughputMbps, final.ThroughputMbps, 0.1,
				"Performance should return to baseline after recovery")
		})
	}
}

// Helper methods for simulator operations

// TestConnectivity tests connectivity between two VTEPs
func (sim *VXLANSimulator) TestConnectivity(sourceVTEP, destVTEP string, vni uint32) (bool, error) {
	sim.mutex.RLock()
	defer sim.mutex.RUnlock()
	
	// Check if both VTEPs exist and are up
	srcVTEP, srcExists := sim.vteps[sourceVTEP]
	if !srcExists || srcVTEP.State != VTEPStateUp {
		return false, fmt.Errorf("source VTEP %s not available", sourceVTEP)
	}
	
	dstVTEP, dstExists := sim.vteps[destVTEP]
	if !dstExists || dstVTEP.State != VTEPStateUp {
		return false, fmt.Errorf("destination VTEP %s not available", destVTEP)
	}
	
	// Check if both VTEPs support the VNI
	srcSupportsVNI := false
	dstSupportsVNI := false
	
	for _, supportedVNI := range srcVTEP.SupportedVNIs {
		if supportedVNI == vni {
			srcSupportsVNI = true
			break
		}
	}
	
	for _, supportedVNI := range dstVTEP.SupportedVNIs {
		if supportedVNI == vni {
			dstSupportsVNI = true
			break
		}
	}
	
	return srcSupportsVNI && dstSupportsVNI, nil
}

// GetTunnel retrieves tunnel information between VTEPs
func (sim *VXLANSimulator) GetTunnel(sourceVTEP, destVTEP string) (*VXLANTunnel, error) {
	sim.mutex.RLock()
	defer sim.mutex.RUnlock()
	
	tunnelID := fmt.Sprintf("%s-%s", sourceVTEP, destVTEP)
	tunnel, exists := sim.tunnels[tunnelID]
	if !exists {
		// Try reverse direction
		tunnelID = fmt.Sprintf("%s-%s", destVTEP, sourceVTEP)
		tunnel, exists = sim.tunnels[tunnelID]
		if !exists {
			return nil, fmt.Errorf("tunnel not found between %s and %s", sourceVTEP, destVTEP)
		}
	}
	
	return tunnel, nil
}

// Additional helper methods would be implemented here for:
// - setupVXLANConnectivityTest()
// - setupResilienceTest() 
// - runPerformanceTest()
// - measureBaselinePerformance()
// - measureRecoveryMetrics()
// - SimulatePacketTransmission()
// - EncapsulatePacket() / DecapsulatePacket()
// - SetupMulticastGroup() / JoinMulticastGroup() / SimulateFlood()
// - InjectFault() / ClearFault()

// Cleanup method
func (sim *VXLANSimulator) Cleanup() {
	sim.mutex.Lock()
	defer sim.mutex.Unlock()
	
	sim.vteps = make(map[string]*VTEP)
	sim.vxlanNetworks = make(map[uint32]*VXLANNetwork)
	sim.tunnels = make(map[string]*VXLANTunnel)
	sim.macTable = make(map[string]*MACEntry)
	sim.arpTable = make(map[string]*ARPEntry)
	sim.multicastGroups = make(map[string][]string)
}

// Placeholder structures for compilation
type EthernetFrame struct {
	SrcMAC  net.HardwareAddr
	DstMAC  net.HardwareAddr
	EthType uint16
	Payload []byte
}

type VXLANPacket struct {
	VXLANHeader []byte
	VNI         uint32
	InnerFrame  *EthernetFrame
}

type FloodResult struct {
	ReceivedBy []string
}

type PacketTransmissionResult struct {
	Fragmented bool
}

type RecoveryMetrics struct {
	RecoveryTime  time.Duration
	ServiceImpact float64
}

type PerformanceMetrics struct {
	ThroughputMbps float64
}

// Test suite runner
func TestVXLANTestSuite(t *testing.T) {
	suite.Run(t, new(VXLANTestSuite))
}

// Standalone tests for basic VXLAN functionality
func TestVXLANDriverCreation(t *testing.T) {
	config := drivers.DefaultVXLANConfig()
	driver := drivers.NewVXLANDriver(config)
	
	assert.NotNil(t, driver)
	assert.Equal(t, "vxlan", driver.Name())
	
	capabilities := driver.Capabilities()
	assert.Contains(t, capabilities.SupportedTypes, VXLAN)
	assert.True(t, capabilities.SupportsL2Extension)
	assert.True(t, capabilities.SupportsNetworkPolicies)
}

func TestVXLANBasicOperations(t *testing.T) {
	config := drivers.DefaultVXLANConfig()
	driver := drivers.NewVXLANDriver(config)
	ctx := context.Background()
	
	// Test initialization
	err := driver.Initialize(ctx)
	assert.NoError(t, err)
	defer driver.Shutdown(ctx)
	
	// Test network creation
	network := OverlayNetwork{
		ID:   "test-basic-" + uuid.New().String(),
		Name: "basic-test",
		Type: VXLAN,
		MTU:  1450,
		Options: map[string]string{
			"vni": "9999",
		},
	}
	
	err = driver.CreateNetwork(ctx, network)
	assert.NoError(t, err)
	
	// Test network retrieval
	retrieved, err := driver.GetNetwork(ctx, network.ID)
	assert.NoError(t, err)
	assert.Equal(t, network.ID, retrieved.ID)
	assert.Equal(t, network.Name, retrieved.Name)
	
	// Test network listing
	networks, err := driver.ListNetworks(ctx)
	assert.NoError(t, err)
	assert.NotEmpty(t, networks)
	
	// Cleanup
	err = driver.DeleteNetwork(ctx, network.ID)
	assert.NoError(t, err)
}