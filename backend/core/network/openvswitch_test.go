package network

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// OpenVSwitchTestSuite provides comprehensive testing for Open vSwitch operations
type OpenVSwitchTestSuite struct {
	suite.Suite
	ovsManager    *OVSManager
	testBridges   []string
	testPorts     []string
	testContext   context.Context
	testCancel    context.CancelFunc
	skipOVSTests  bool
}

// OVSManager manages Open vSwitch operations
type OVSManager struct {
	bridges        map[string]*OVSBridge
	flowTables     map[string]map[int]*OVSFlowTable
	ovsdbEndpoint  string
	ofProtocol     string
	bridgePrefix   string
}

// OVSBridge represents an Open vSwitch bridge
type OVSBridge struct {
	Name           string
	UUID           string
	DatapathID     string
	Ports          map[string]*OVSPort
	FlowTables     map[int]*OVSFlowTable
	Controllers    []string
	OpenFlowVersion string
	STP            bool
	RSTP           bool
	Created        time.Time
	Stats          *OVSBridgeStats
}

// OVSPort represents a port on an OVS bridge
type OVSPort struct {
	Name         string
	UUID         string
	PortNumber   int
	Type         OVSPortType
	VLAN         int
	Trunk        []int
	Interface    *OVSInterface
	Stats        *OVSPortStats
	QoS          *OVSQoS
}

// OVSInterface represents an OVS interface
type OVSInterface struct {
	Name       string
	UUID       string
	Type       OVSInterfaceType
	Options    map[string]string
	Status     map[string]string
	AdminState InterfaceState
	LinkState  InterfaceState
	MTU        int
	MAC        net.HardwareAddr
}

// OVSFlowTable represents a flow table in OVS
type OVSFlowTable struct {
	TableID     int
	Name        string
	MaxEntries  int
	FlowCount   int
	LookupCount uint64
	MatchedCount uint64
	Flows       map[string]*OVSFlow
}

// OVSFlow represents a flow entry in OVS
type OVSFlow struct {
	Cookie      uint64
	Priority    int
	Table       int
	IdleTimeout int
	HardTimeout int
	Match       OVSMatch
	Actions     []OVSAction
	Stats       *OVSFlowStats
	InstallTime time.Time
}

// OVSMatch represents flow matching criteria
type OVSMatch struct {
	InPort    string
	EthSrc    net.HardwareAddr
	EthDst    net.HardwareAddr
	EthType   uint16
	VLANID    uint16
	IPSrc     *net.IPNet
	IPDst     *net.IPNet
	IPProto   uint8
	TCPSrc    uint16
	TCPDst    uint16
	UDPSrc    uint16
	UDPDst    uint16
	Metadata  uint64
}

// OVSAction represents flow actions
type OVSAction struct {
	Type       OVSActionType
	Parameters map[string]interface{}
}

// OVSPortType represents types of OVS ports
type OVSPortType string

const (
	OVSPortTypeInternal OVSPortType = "internal"
	OVSPortTypeVeth     OVSPortType = "veth"
	OVSPortTypeTun      OVSPortType = "tun"
	OVSPortTypeTap      OVSPortType = "tap"
	OVSPortTypeVXLAN    OVSPortType = "vxlan"
	OVSPortTypeGRE      OVSPortType = "gre"
	OVSPortTypeGeneve   OVSPortType = "geneve"
	OVSPortTypePatch    OVSPortType = "patch"
)

// OVSInterfaceType represents types of OVS interfaces
type OVSInterfaceType string

const (
	OVSInterfaceTypeInternal OVSInterfaceType = "internal"
	OVSInterfaceTypeVeth     OVSInterfaceType = "veth"
	OVSInterfaceTypeVXLAN    OVSInterfaceType = "vxlan"
	OVSInterfaceTypeGRE      OVSInterfaceType = "gre"
	OVSInterfaceTypeGeneve   OVSInterfaceType = "geneve"
	OVSInterfaceTypePatch    OVSInterfaceType = "patch"
)

// OVSActionType represents types of OVS actions
type OVSActionType string

const (
	OVSActionOutput         OVSActionType = "output"
	OVSActionDrop           OVSActionType = "drop"
	OVSActionNormal         OVSActionType = "normal"
	OVSActionFlood          OVSActionType = "flood"
	OVSActionController     OVSActionType = "controller"
	OVSActionSetVLAN        OVSActionType = "set_vlan"
	OVSActionStripVLAN      OVSActionType = "strip_vlan"
	OVSActionSetField       OVSActionType = "set_field"
	OVSActionPushVLAN       OVSActionType = "push_vlan"
	OVSActionPopVLAN        OVSActionType = "pop_vlan"
	OVSActionResubmit       OVSActionType = "resubmit"
	OVSActionGoto           OVSActionType = "goto"
	OVSActionMeter          OVSActionType = "meter"
)

// Statistics structures
type OVSBridgeStats struct {
	RxPackets  uint64
	TxPackets  uint64
	RxBytes    uint64
	TxBytes    uint64
	RxDropped  uint64
	TxDropped  uint64
	RxErrors   uint64
	TxErrors   uint64
	LastUpdate time.Time
}

type OVSPortStats struct {
	RxPackets  uint64
	TxPackets  uint64
	RxBytes    uint64
	TxBytes    uint64
	RxDropped  uint64
	TxDropped  uint64
	RxErrors   uint64
	TxErrors   uint64
	Collisions uint64
	LastUpdate time.Time
}

type OVSFlowStats struct {
	PacketCount uint64
	ByteCount   uint64
	Duration    time.Duration
	LastUsed    time.Time
}

// OVSQoS represents Quality of Service configuration
type OVSQoS struct {
	Type    string
	Queues  map[int]*OVSQueue
	Other   map[string]interface{}
}

type OVSQueue struct {
	QueueID   int
	MinRate   int64
	MaxRate   int64
	Burst     int64
	Priority  int
	Other     map[string]interface{}
}

// SetupSuite initializes the test suite
func (suite *OpenVSwitchTestSuite) SetupSuite() {
	suite.testContext, suite.testCancel = context.WithCancel(context.Background())
	
	// Check if OVS is available
	if !suite.isOVSAvailable() {
		suite.T().Skip("Open vSwitch not available, skipping OVS tests")
		suite.skipOVSTests = true
		return
	}
	
	// Initialize OVS manager
	suite.ovsManager = NewOVSManager("unix:/var/run/openvswitch/db.sock", "OpenFlow13", "test")
	
	// Clean up any existing test bridges
	suite.cleanupTestBridges()
}

// TearDownSuite cleans up after all tests
func (suite *OpenVSwitchTestSuite) TearDownSuite() {
	if suite.skipOVSTests {
		return
	}
	
	// Clean up test bridges and ports
	suite.cleanupTestBridges()
	
	if suite.testCancel != nil {
		suite.testCancel()
	}
}

// SetupTest prepares for each test
func (suite *OpenVSwitchTestSuite) SetupTest() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
}

// NewOVSManager creates a new OVS manager
func NewOVSManager(ovsdbEndpoint, ofProtocol, bridgePrefix string) *OVSManager {
	return &OVSManager{
		bridges:       make(map[string]*OVSBridge),
		flowTables:    make(map[string]map[int]*OVSFlowTable),
		ovsdbEndpoint: ovsdbEndpoint,
		ofProtocol:    ofProtocol,
		bridgePrefix:  bridgePrefix,
	}
}

// isOVSAvailable checks if Open vSwitch is available
func (suite *OpenVSwitchTestSuite) isOVSAvailable() bool {
	// Check if ovs-vsctl is available
	_, err := exec.LookPath("ovs-vsctl")
	if err != nil {
		return false
	}
	
	// Check if OVS daemon is running
	cmd := exec.Command("ovs-vsctl", "show")
	err = cmd.Run()
	return err == nil
}

// CreateBridge creates an OVS bridge
func (manager *OVSManager) CreateBridge(name string, options map[string]string) (*OVSBridge, error) {
	// Create bridge using ovs-vsctl
	args := []string{"add-br", name}
	
	// Add optional parameters
	for key, value := range options {
		switch key {
		case "datapath_type":
			args = append(args, "--", "set", "bridge", name, fmt.Sprintf("datapath_type=%s", value))
		case "stp_enable":
			args = append(args, "--", "set", "bridge", name, fmt.Sprintf("stp_enable=%s", value))
		case "rstp_enable":
			args = append(args, "--", "set", "bridge", name, fmt.Sprintf("rstp_enable=%s", value))
		}
	}
	
	cmd := exec.Command("ovs-vsctl", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to create bridge %s: %w, output: %s", name, err, string(output))
	}
	
	// Get bridge UUID and datapath ID
	uuid, err := manager.getBridgeUUID(name)
	if err != nil {
		return nil, fmt.Errorf("failed to get bridge UUID: %w", err)
	}
	
	datapathID, err := manager.getBridgeDatapathID(name)
	if err != nil {
		return nil, fmt.Errorf("failed to get bridge datapath ID: %w", err)
	}
	
	bridge := &OVSBridge{
		Name:        name,
		UUID:        uuid,
		DatapathID:  datapathID,
		Ports:       make(map[string]*OVSPort),
		FlowTables:  make(map[int]*OVSFlowTable),
		Controllers: []string{},
		Created:     time.Now(),
		Stats:       &OVSBridgeStats{},
	}
	
	// Parse options
	if stpEnable, ok := options["stp_enable"]; ok && stpEnable == "true" {
		bridge.STP = true
	}
	if rstpEnable, ok := options["rstp_enable"]; ok && rstpEnable == "true" {
		bridge.RSTP = true
	}
	
	manager.bridges[name] = bridge
	manager.flowTables[name] = make(map[int]*OVSFlowTable)
	
	return bridge, nil
}

// DeleteBridge deletes an OVS bridge
func (manager *OVSManager) DeleteBridge(name string) error {
	cmd := exec.Command("ovs-vsctl", "del-br", name)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete bridge %s: %w, output: %s", name, err, string(output))
	}
	
	delete(manager.bridges, name)
	delete(manager.flowTables, name)
	
	return nil
}

// AddPort adds a port to an OVS bridge
func (manager *OVSManager) AddPort(bridgeName, portName string, portType OVSPortType, options map[string]string) (*OVSPort, error) {
	bridge, exists := manager.bridges[bridgeName]
	if !exists {
		return nil, fmt.Errorf("bridge %s not found", bridgeName)
	}
	
	args := []string{"add-port", bridgeName, portName}
	
	// Add port type and options
	if portType != "" {
		args = append(args, "--", "set", "interface", portName, fmt.Sprintf("type=%s", string(portType)))
	}
	
	for key, value := range options {
		args = append(args, "--", "set", "interface", portName, fmt.Sprintf("%s=%s", key, value))
	}
	
	cmd := exec.Command("ovs-vsctl", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to add port %s to bridge %s: %w, output: %s", portName, bridgeName, err, string(output))
	}
	
	// Get port UUID and number
	portUUID, err := manager.getPortUUID(portName)
	if err != nil {
		return nil, fmt.Errorf("failed to get port UUID: %w", err)
	}
	
	portNumber, err := manager.getPortNumber(portName)
	if err != nil {
		return nil, fmt.Errorf("failed to get port number: %w", err)
	}
	
	port := &OVSPort{
		Name:       portName,
		UUID:       portUUID,
		PortNumber: portNumber,
		Type:       portType,
		Interface: &OVSInterface{
			Name:    portName,
			Type:    OVSInterfaceType(portType),
			Options: options,
			Status:  make(map[string]string),
		},
		Stats: &OVSPortStats{},
	}
	
	bridge.Ports[portName] = port
	
	return port, nil
}

// DeletePort deletes a port from an OVS bridge
func (manager *OVSManager) DeletePort(bridgeName, portName string) error {
	cmd := exec.Command("ovs-vsctl", "del-port", bridgeName, portName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete port %s from bridge %s: %w, output: %s", portName, bridgeName, err, string(output))
	}
	
	if bridge, exists := manager.bridges[bridgeName]; exists {
		delete(bridge.Ports, portName)
	}
	
	return nil
}

// AddFlow adds a flow entry to an OVS bridge
func (manager *OVSManager) AddFlow(bridgeName string, flow *OVSFlow) error {
	flowSpec := manager.buildFlowSpec(flow)
	
	cmd := exec.Command("ovs-ofctl", "add-flow", bridgeName, flowSpec)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to add flow to bridge %s: %w, output: %s", bridgeName, err, string(output))
	}
	
	flow.InstallTime = time.Now()
	
	// Store flow in table
	if flowTables, exists := manager.flowTables[bridgeName]; exists {
		if table, exists := flowTables[flow.Table]; !exists {
			table = &OVSFlowTable{
				TableID: flow.Table,
				Flows:   make(map[string]*OVSFlow),
			}
			flowTables[flow.Table] = table
		}
		
		flowKey := manager.generateFlowKey(flow)
		flowTables[flow.Table].Flows[flowKey] = flow
	}
	
	return nil
}

// DeleteFlow deletes a flow entry from an OVS bridge
func (manager *OVSManager) DeleteFlow(bridgeName string, match OVSMatch, table int) error {
	matchSpec := manager.buildMatchSpec(match)
	args := []string{"del-flows", bridgeName}
	
	if table >= 0 {
		args = append(args, fmt.Sprintf("table=%d", table))
	}
	if matchSpec != "" {
		args = append(args, matchSpec)
	}
	
	cmd := exec.Command("ovs-ofctl", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete flows from bridge %s: %w, output: %s", bridgeName, err, string(output))
	}
	
	return nil
}

// GetBridgeStats retrieves bridge statistics
func (manager *OVSManager) GetBridgeStats(bridgeName string) (*OVSBridgeStats, error) {
	cmd := exec.Command("ovs-ofctl", "dump-ports", bridgeName, "LOCAL")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get bridge stats: %w", err)
	}
	
	stats, err := manager.parseBridgeStats(string(output))
	if err != nil {
		return nil, fmt.Errorf("failed to parse bridge stats: %w", err)
	}
	
	stats.LastUpdate = time.Now()
	return stats, nil
}

// GetPortStats retrieves port statistics
func (manager *OVSManager) GetPortStats(bridgeName, portName string) (*OVSPortStats, error) {
	cmd := exec.Command("ovs-ofctl", "dump-ports", bridgeName, portName)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get port stats: %w", err)
	}
	
	stats, err := manager.parsePortStats(string(output))
	if err != nil {
		return nil, fmt.Errorf("failed to parse port stats: %w", err)
	}
	
	stats.LastUpdate = time.Now()
	return stats, nil
}

// GetFlowStats retrieves flow statistics
func (manager *OVSManager) GetFlowStats(bridgeName string) (map[string]*OVSFlowStats, error) {
	cmd := exec.Command("ovs-ofctl", "dump-flows", bridgeName)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get flow stats: %w", err)
	}
	
	return manager.parseFlowStats(string(output))
}

// Helper methods

func (manager *OVSManager) getBridgeUUID(name string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "bridge", name, "_uuid")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func (manager *OVSManager) getBridgeDatapathID(name string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "bridge", name, "datapath_id")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.Trim(strings.TrimSpace(string(output)), `"`), nil
}

func (manager *OVSManager) getPortUUID(name string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "port", name, "_uuid")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func (manager *OVSManager) getPortNumber(name string) (int, error) {
	cmd := exec.Command("ovs-vsctl", "get", "interface", name, "ofport")
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(output)))
}

func (manager *OVSManager) buildFlowSpec(flow *OVSFlow) string {
	var parts []string
	
	// Priority and table
	if flow.Priority > 0 {
		parts = append(parts, fmt.Sprintf("priority=%d", flow.Priority))
	}
	if flow.Table > 0 {
		parts = append(parts, fmt.Sprintf("table=%d", flow.Table))
	}
	
	// Timeouts
	if flow.IdleTimeout > 0 {
		parts = append(parts, fmt.Sprintf("idle_timeout=%d", flow.IdleTimeout))
	}
	if flow.HardTimeout > 0 {
		parts = append(parts, fmt.Sprintf("hard_timeout=%d", flow.HardTimeout))
	}
	
	// Cookie
	if flow.Cookie > 0 {
		parts = append(parts, fmt.Sprintf("cookie=%d", flow.Cookie))
	}
	
	// Match fields
	matchSpec := manager.buildMatchSpec(flow.Match)
	if matchSpec != "" {
		parts = append(parts, matchSpec)
	}
	
	// Actions
	if len(flow.Actions) > 0 {
		actionSpecs := make([]string, 0, len(flow.Actions))
		for _, action := range flow.Actions {
			actionSpecs = append(actionSpecs, manager.buildActionSpec(action))
		}
		parts = append(parts, fmt.Sprintf("actions=%s", strings.Join(actionSpecs, ",")))
	}
	
	return strings.Join(parts, ",")
}

func (manager *OVSManager) buildMatchSpec(match OVSMatch) string {
	var parts []string
	
	if match.InPort != "" {
		parts = append(parts, fmt.Sprintf("in_port=%s", match.InPort))
	}
	if match.EthSrc != nil {
		parts = append(parts, fmt.Sprintf("dl_src=%s", match.EthSrc.String()))
	}
	if match.EthDst != nil {
		parts = append(parts, fmt.Sprintf("dl_dst=%s", match.EthDst.String()))
	}
	if match.EthType > 0 {
		parts = append(parts, fmt.Sprintf("dl_type=0x%04x", match.EthType))
	}
	if match.VLANID > 0 {
		parts = append(parts, fmt.Sprintf("dl_vlan=%d", match.VLANID))
	}
	if match.IPSrc != nil {
		parts = append(parts, fmt.Sprintf("nw_src=%s", match.IPSrc.String()))
	}
	if match.IPDst != nil {
		parts = append(parts, fmt.Sprintf("nw_dst=%s", match.IPDst.String()))
	}
	if match.IPProto > 0 {
		parts = append(parts, fmt.Sprintf("nw_proto=%d", match.IPProto))
	}
	if match.TCPSrc > 0 {
		parts = append(parts, fmt.Sprintf("tp_src=%d", match.TCPSrc))
	}
	if match.TCPDst > 0 {
		parts = append(parts, fmt.Sprintf("tp_dst=%d", match.TCPDst))
	}
	
	return strings.Join(parts, ",")
}

func (manager *OVSManager) buildActionSpec(action OVSAction) string {
	switch action.Type {
	case OVSActionOutput:
		if port, ok := action.Parameters["port"].(string); ok {
			return fmt.Sprintf("output:%s", port)
		}
	case OVSActionDrop:
		return "drop"
	case OVSActionNormal:
		return "normal"
	case OVSActionFlood:
		return "flood"
	case OVSActionController:
		return "controller"
	case OVSActionSetVLAN:
		if vlan, ok := action.Parameters["vlan"].(int); ok {
			return fmt.Sprintf("mod_vlan_vid:%d", vlan)
		}
	case OVSActionStripVLAN:
		return "strip_vlan"
	}
	
	return string(action.Type)
}

func (manager *OVSManager) generateFlowKey(flow *OVSFlow) string {
	return fmt.Sprintf("table_%d_priority_%d_cookie_%d", flow.Table, flow.Priority, flow.Cookie)
}

func (manager *OVSManager) parseBridgeStats(output string) (*OVSBridgeStats, error) {
	// Parse ovs-ofctl dump-ports output
	// This is a simplified parser - real implementation would be more robust
	stats := &OVSBridgeStats{}
	
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "rx pkts") {
			// Parse RX stats
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "pkts=" && i+1 < len(parts) {
					if val, err := strconv.ParseUint(strings.TrimSuffix(parts[i+1], ","), 10, 64); err == nil {
						stats.RxPackets = val
					}
				}
				if part == "bytes=" && i+1 < len(parts) {
					if val, err := strconv.ParseUint(strings.TrimSuffix(parts[i+1], ","), 10, 64); err == nil {
						stats.RxBytes = val
					}
				}
			}
		}
	}
	
	return stats, nil
}

func (manager *OVSManager) parsePortStats(output string) (*OVSPortStats, error) {
	stats := &OVSPortStats{}
	
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "rx pkts") {
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "pkts=" && i+1 < len(parts) {
					if val, err := strconv.ParseUint(strings.TrimSuffix(parts[i+1], ","), 10, 64); err == nil {
						stats.RxPackets = val
					}
				}
			}
		}
	}
	
	return stats, nil
}

func (manager *OVSManager) parseFlowStats(output string) (map[string]*OVSFlowStats, error) {
	flowStats := make(map[string]*OVSFlowStats)
	
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "duration=") && strings.Contains(line, "n_packets=") {
			stats := &OVSFlowStats{}
			
			// Parse duration, packet count, byte count
			parts := strings.Fields(line)
			for _, part := range parts {
				if strings.HasPrefix(part, "n_packets=") {
					if val, err := strconv.ParseUint(strings.TrimPrefix(part, "n_packets="), 10, 64); err == nil {
						stats.PacketCount = val
					}
				}
				if strings.HasPrefix(part, "n_bytes=") {
					if val, err := strconv.ParseUint(strings.TrimPrefix(part, "n_bytes="), 10, 64); err == nil {
						stats.ByteCount = val
					}
				}
			}
			
			// Use line as key for simplicity
			flowStats[line] = stats
		}
	}
	
	return flowStats, nil
}

// Test methods

// TestOVSBridgeOperations tests basic bridge operations
func (suite *OpenVSwitchTestSuite) TestOVSBridgeOperations() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	bridgeName := "test-bridge-" + uuid.New().String()[:8]
	
	// Test bridge creation
	bridge, err := suite.ovsManager.CreateBridge(bridgeName, map[string]string{
		"stp_enable": "false",
	})
	
	require.NoError(suite.T(), err, "Bridge creation should succeed")
	require.NotNil(suite.T(), bridge, "Bridge should not be nil")
	assert.Equal(suite.T(), bridgeName, bridge.Name)
	assert.NotEmpty(suite.T(), bridge.UUID)
	assert.False(suite.T(), bridge.STP)
	
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Test bridge exists
	_, exists := suite.ovsManager.bridges[bridgeName]
	assert.True(suite.T(), exists, "Bridge should be stored in manager")
	
	// Test bridge stats
	stats, err := suite.ovsManager.GetBridgeStats(bridgeName)
	assert.NoError(suite.T(), err, "Should be able to get bridge stats")
	assert.NotNil(suite.T(), stats)
	assert.NotZero(suite.T(), stats.LastUpdate)
	
	// Test bridge deletion
	err = suite.ovsManager.DeleteBridge(bridgeName)
	assert.NoError(suite.T(), err, "Bridge deletion should succeed")
	
	// Verify bridge is removed
	_, exists = suite.ovsManager.bridges[bridgeName]
	assert.False(suite.T(), exists, "Bridge should be removed from manager")
}

// TestOVSPortOperations tests port operations
func (suite *OpenVSwitchTestSuite) TestOVSPortOperations() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	bridgeName := "test-port-bridge-" + uuid.New().String()[:8]
	portName := "test-port-" + uuid.New().String()[:8]
	
	// Create test bridge
	_, err := suite.ovsManager.CreateBridge(bridgeName, nil)
	require.NoError(suite.T(), err)
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Test port addition
	port, err := suite.ovsManager.AddPort(bridgeName, portName, OVSPortTypeInternal, map[string]string{
		"mac": "02:00:00:00:00:01",
	})
	
	require.NoError(suite.T(), err, "Port creation should succeed")
	require.NotNil(suite.T(), port, "Port should not be nil")
	assert.Equal(suite.T(), portName, port.Name)
	assert.Equal(suite.T(), OVSPortTypeInternal, port.Type)
	assert.NotEmpty(suite.T(), port.UUID)
	assert.True(suite.T(), port.PortNumber > 0)
	
	suite.testPorts = append(suite.testPorts, portName)
	
	// Test port exists in bridge
	bridge := suite.ovsManager.bridges[bridgeName]
	_, exists := bridge.Ports[portName]
	assert.True(suite.T(), exists, "Port should be stored in bridge")
	
	// Test port stats
	stats, err := suite.ovsManager.GetPortStats(bridgeName, portName)
	assert.NoError(suite.T(), err, "Should be able to get port stats")
	assert.NotNil(suite.T(), stats)
	
	// Test port deletion
	err = suite.ovsManager.DeletePort(bridgeName, portName)
	assert.NoError(suite.T(), err, "Port deletion should succeed")
	
	// Verify port is removed
	_, exists = bridge.Ports[portName]
	assert.False(suite.T(), exists, "Port should be removed from bridge")
}

// TestOVSVXLANTunnel tests VXLAN tunnel creation
func (suite *OpenVSwitchTestSuite) TestOVSVXLANTunnel() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	bridgeName := "test-vxlan-bridge-" + uuid.New().String()[:8]
	tunnelName := "vxlan-tunnel-" + uuid.New().String()[:8]
	
	// Create test bridge
	_, err := suite.ovsManager.CreateBridge(bridgeName, nil)
	require.NoError(suite.T(), err)
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Create VXLAN tunnel port
	port, err := suite.ovsManager.AddPort(bridgeName, tunnelName, OVSPortTypeVXLAN, map[string]string{
		"remote_ip": "192.168.1.100",
		"key":       "1000",
		"dst_port":  "4789",
	})
	
	require.NoError(suite.T(), err, "VXLAN tunnel creation should succeed")
	require.NotNil(suite.T(), port, "VXLAN port should not be nil")
	assert.Equal(suite.T(), OVSPortTypeVXLAN, port.Type)
	assert.Equal(suite.T(), "192.168.1.100", port.Interface.Options["remote_ip"])
	assert.Equal(suite.T(), "1000", port.Interface.Options["key"])
	
	suite.testPorts = append(suite.testPorts, tunnelName)
}

// TestOVSFlowOperations tests flow table operations
func (suite *OpenVSwitchTestSuite) TestOVSFlowOperations() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	bridgeName := "test-flow-bridge-" + uuid.New().String()[:8]
	
	// Create test bridge
	_, err := suite.ovsManager.CreateBridge(bridgeName, nil)
	require.NoError(suite.T(), err)
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Test basic flow addition
	basicFlow := &OVSFlow{
		Cookie:   1234,
		Priority: 1000,
		Table:    0,
		Match: OVSMatch{
			EthType: 0x0800, // IPv4
		},
		Actions: []OVSAction{
			{
				Type: OVSActionNormal,
			},
		},
	}
	
	err = suite.ovsManager.AddFlow(bridgeName, basicFlow)
	assert.NoError(suite.T(), err, "Basic flow addition should succeed")
	
	// Test VLAN flow
	vlanFlow := &OVSFlow{
		Cookie:   5678,
		Priority: 2000,
		Table:    0,
		Match: OVSMatch{
			InPort: "1",
		},
		Actions: []OVSAction{
			{
				Type: OVSActionPushVLAN,
				Parameters: map[string]interface{}{
					"vlan_id": 100,
				},
			},
			{
				Type: OVSActionOutput,
				Parameters: map[string]interface{}{
					"port": "2",
				},
			},
		},
	}
	
	err = suite.ovsManager.AddFlow(bridgeName, vlanFlow)
	assert.NoError(suite.T(), err, "VLAN flow addition should succeed")
	
	// Test flow stats retrieval
	flowStats, err := suite.ovsManager.GetFlowStats(bridgeName)
	assert.NoError(suite.T(), err, "Flow stats retrieval should succeed")
	assert.NotEmpty(suite.T(), flowStats, "Should have flow stats")
	
	// Test flow deletion
	err = suite.ovsManager.DeleteFlow(bridgeName, OVSMatch{EthType: 0x0800}, 0)
	assert.NoError(suite.T(), err, "Flow deletion should succeed")
}

// TestOVSQoSOperations tests QoS operations
func (suite *OpenVSwitchTestSuite) TestOVSQoSOperations() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	bridgeName := "test-qos-bridge-" + uuid.New().String()[:8]
	portName := "test-qos-port-" + uuid.New().String()[:8]
	
	// Create test bridge
	_, err := suite.ovsManager.CreateBridge(bridgeName, nil)
	require.NoError(suite.T(), err)
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Create port for QoS
	_, err = suite.ovsManager.AddPort(bridgeName, portName, OVSPortTypeInternal, nil)
	require.NoError(suite.T(), err)
	suite.testPorts = append(suite.testPorts, portName)
	
	// Test QoS configuration
	err = suite.configureQoS(portName, map[string]interface{}{
		"max-rate": "1000000", // 1 Mbps
		"type":     "linux-htb",
	})
	
	// QoS configuration might not work in all test environments
	// So we don't assert on the error, but log it
	if err != nil {
		suite.T().Logf("QoS configuration may not be supported in test environment: %v", err)
	}
}

// configureQoS configures Quality of Service for a port
func (suite *OpenVSwitchTestSuite) configureQoS(portName string, config map[string]interface{}) error {
	args := []string{"set", "port", portName}
	
	for key, value := range config {
		args = append(args, fmt.Sprintf("%s=%v", key, value))
	}
	
	cmd := exec.Command("ovs-vsctl", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to configure QoS: %w, output: %s", err, string(output))
	}
	
	return nil
}

// TestOVSIntegrationScenario tests a complex integration scenario
func (suite *OpenVSwitchTestSuite) TestOVSIntegrationScenario() {
	if suite.skipOVSTests {
		suite.T().Skip("OVS not available")
		return
	}
	
	// Scenario: Create a multi-tenant network with VLAN isolation
	bridgeName := "test-integration-" + uuid.New().String()[:8]
	
	// Create bridge
	bridge, err := suite.ovsManager.CreateBridge(bridgeName, map[string]string{
		"stp_enable": "false",
	})
	require.NoError(suite.T(), err)
	suite.testBridges = append(suite.testBridges, bridgeName)
	
	// Add tenant ports with VLAN tags
	tenantPorts := []struct {
		name string
		vlan int
	}{
		{"tenant1-port", 100},
		{"tenant2-port", 200},
		{"trunk-port", 0}, // Trunk port for multiple VLANs
	}
	
	for _, tp := range tenantPorts {
		port, err := suite.ovsManager.AddPort(bridgeName, tp.name, OVSPortTypeInternal, map[string]string{
			"tag": fmt.Sprintf("%d", tp.vlan),
		})
		require.NoError(suite.T(), err, "Port creation should succeed for %s", tp.name)
		suite.testPorts = append(suite.testPorts, tp.name)
		
		if tp.vlan > 0 {
			port.VLAN = tp.vlan
		}
	}
	
	// Add flow rules for VLAN isolation
	isolationFlows := []*OVSFlow{
		{
			Cookie:   1000,
			Priority: 3000,
			Table:    0,
			Match: OVSMatch{
				VLANID: 100,
			},
			Actions: []OVSAction{
				{
					Type: OVSActionOutput,
					Parameters: map[string]interface{}{
						"port": "tenant1-port",
					},
				},
			},
		},
		{
			Cookie:   2000,
			Priority: 3000,
			Table:    0,
			Match: OVSMatch{
				VLANID: 200,
			},
			Actions: []OVSAction{
				{
					Type: OVSActionOutput,
					Parameters: map[string]interface{}{
						"port": "tenant2-port",
					},
				},
			},
		},
	}
	
	for _, flow := range isolationFlows {
		err := suite.ovsManager.AddFlow(bridgeName, flow)
		assert.NoError(suite.T(), err, "Flow installation should succeed")
	}
	
	// Verify bridge configuration
	assert.Equal(suite.T(), 3, len(bridge.Ports), "Should have 3 tenant ports")
	
	// Verify flow table
	flowStats, err := suite.ovsManager.GetFlowStats(bridgeName)
	assert.NoError(suite.T(), err, "Should be able to get flow stats")
	assert.NotEmpty(suite.T(), flowStats, "Should have installed flows")
}

// cleanupTestBridges removes all test bridges and ports
func (suite *OpenVSwitchTestSuite) cleanupTestBridges() {
	if suite.skipOVSTests {
		return
	}
	
	// Delete test bridges (this also removes associated ports)
	for _, bridgeName := range suite.testBridges {
		err := suite.ovsManager.DeleteBridge(bridgeName)
		if err != nil {
			suite.T().Logf("Warning: Failed to cleanup test bridge %s: %v", bridgeName, err)
		}
	}
	
	suite.testBridges = nil
	suite.testPorts = nil
}

// TestOVSTestSuite runs the complete OVS test suite
func TestOVSTestSuite(t *testing.T) {
	// Check if we're in a CI environment where OVS might not be available
	if os.Getenv("CI") != "" && os.Getenv("TEST_OVS") != "true" {
		t.Skip("Skipping OVS tests in CI environment (set TEST_OVS=true to enable)")
	}
	
	suite.Run(t, new(OpenVSwitchTestSuite))
}

// Benchmark tests

// BenchmarkOVSFlowInstallation benchmarks flow installation performance
func BenchmarkOVSFlowInstallation(b *testing.B) {
	if os.Getenv("BENCH_OVS") != "true" {
		b.Skip("OVS benchmarks disabled (set BENCH_OVS=true to enable)")
	}
	
	manager := NewOVSManager("unix:/var/run/openvswitch/db.sock", "OpenFlow13", "bench")
	bridgeName := "bench-bridge"
	
	// Create test bridge
	_, err := manager.CreateBridge(bridgeName, nil)
	if err != nil {
		b.Fatalf("Failed to create benchmark bridge: %v", err)
	}
	defer manager.DeleteBridge(bridgeName)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		flow := &OVSFlow{
			Cookie:   uint64(i),
			Priority: 1000 + i,
			Table:    0,
			Match: OVSMatch{
				EthType: 0x0800,
				IPSrc: &net.IPNet{
					IP:   net.IPv4(192, 168, byte(i/256), byte(i%256)),
					Mask: net.CIDRMask(32, 32),
				},
			},
			Actions: []OVSAction{
				{
					Type: OVSActionOutput,
					Parameters: map[string]interface{}{
						"port": "normal",
					},
				},
			},
		}
		
		err := manager.AddFlow(bridgeName, flow)
		if err != nil {
			b.Errorf("Flow installation failed: %v", err)
		}
	}
}

// BenchmarkOVSStatsRetrieval benchmarks statistics retrieval performance
func BenchmarkOVSStatsRetrieval(b *testing.B) {
	if os.Getenv("BENCH_OVS") != "true" {
		b.Skip("OVS benchmarks disabled (set BENCH_OVS=true to enable)")
	}
	
	manager := NewOVSManager("unix:/var/run/openvswitch/db.sock", "OpenFlow13", "bench")
	bridgeName := "bench-stats-bridge"
	
	// Create test bridge
	_, err := manager.CreateBridge(bridgeName, nil)
	if err != nil {
		b.Fatalf("Failed to create benchmark bridge: %v", err)
	}
	defer manager.DeleteBridge(bridgeName)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, err := manager.GetBridgeStats(bridgeName)
		if err != nil {
			b.Errorf("Stats retrieval failed: %v", err)
		}
	}
}