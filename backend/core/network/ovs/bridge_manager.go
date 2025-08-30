package ovs

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// BridgeType represents different types of OVS bridges
type BridgeType string

const (
	BridgeTypeNormal     BridgeType = "normal"
	BridgeTypeVXLAN      BridgeType = "vxlan"
	BridgeTypeTunnel     BridgeType = "tunnel"
	BridgeTypeAccess     BridgeType = "access"
)

// Bridge represents an Open vSwitch bridge
type Bridge struct {
	Name        string            `json:"name"`
	UUID        string            `json:"uuid"`
	Type        BridgeType        `json:"type"`
	DPID        string            `json:"datapath_id"`
	Controller  []string          `json:"controller,omitempty"`
	Protocols   []string          `json:"protocols,omitempty"`
	Ports       []Port            `json:"ports"`
	FlowRules   []FlowRule        `json:"flow_rules"`
	QoSPolicies []QoSPolicy       `json:"qos_policies"`
	Options     map[string]string `json:"options,omitempty"`
	Status      BridgeStatus      `json:"status"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// BridgeStatus represents the operational status of a bridge
type BridgeStatus struct {
	State         string    `json:"state"`
	Active        bool      `json:"active"`
	FlowCount     int       `json:"flow_count"`
	PacketCount   int64     `json:"packet_count"`
	ByteCount     int64     `json:"byte_count"`
	LastSeen      time.Time `json:"last_seen"`
	ErrorCount    int       `json:"error_count"`
}

// Port represents a port on an OVS bridge
type Port struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	UUID      string            `json:"uuid"`
	OfPort    int               `json:"ofport"`
	Type      PortType          `json:"type"`
	Interface string            `json:"interface"`
	VLAN      int               `json:"vlan,omitempty"`
	Options   map[string]string `json:"options,omitempty"`
	Status    PortStatus        `json:"status"`
}

// PortType represents different types of OVS ports
type PortType string

const (
	PortTypeInternal PortType = "internal"
	PortTypeVETH     PortType = "veth"
	PortTypeVXLAN    PortType = "vxlan"
	PortTypeGRE      PortType = "gre"
	PortTypeGeneve   PortType = "geneve"
	PortTypePatch    PortType = "patch"
	PortTypePhysical PortType = "physical"
)

// PortStatus represents the operational status of a port
type PortStatus struct {
	State       string    `json:"state"`
	Link        bool      `json:"link"`
	Speed       int64     `json:"speed_mbps"`
	Duplex      string    `json:"duplex"`
	RxPackets   int64     `json:"rx_packets"`
	TxPackets   int64     `json:"tx_packets"`
	RxBytes     int64     `json:"rx_bytes"`
	TxBytes     int64     `json:"tx_bytes"`
	RxErrors    int64     `json:"rx_errors"`
	TxErrors    int64     `json:"tx_errors"`
	LastUpdated time.Time `json:"last_updated"`
}

// FlowRule represents an OpenFlow rule in OVS
type FlowRule struct {
	ID          string            `json:"id"`
	Priority    int               `json:"priority"`
	Table       int               `json:"table"`
	Cookie      uint64            `json:"cookie"`
	Duration    int               `json:"duration_sec"`
	Packets     int64             `json:"n_packets"`
	Bytes       int64             `json:"n_bytes"`
	IdleTimeout int               `json:"idle_timeout"`
	HardTimeout int               `json:"hard_timeout"`
	Match       FlowMatch         `json:"match"`
	Actions     []FlowAction      `json:"actions"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
}

// FlowMatch represents OpenFlow matching criteria
type FlowMatch struct {
	InPort      string `json:"in_port,omitempty"`
	EthSrc      string `json:"dl_src,omitempty"`
	EthDst      string `json:"dl_dst,omitempty"`
	EthType     string `json:"dl_type,omitempty"`
	VlanID      int    `json:"dl_vlan,omitempty"`
	VlanPCP     int    `json:"dl_vlan_pcp,omitempty"`
	IPSrc       string `json:"nw_src,omitempty"`
	IPDst       string `json:"nw_dst,omitempty"`
	IPProto     int    `json:"nw_proto,omitempty"`
	TCPSrc      int    `json:"tp_src,omitempty"`
	TCPDst      int    `json:"tp_dst,omitempty"`
	TunnelID    uint64 `json:"tun_id,omitempty"`
	Metadata    uint64 `json:"metadata,omitempty"`
}

// FlowAction represents OpenFlow actions
type FlowAction struct {
	Type   string            `json:"type"`
	Params map[string]string `json:"params,omitempty"`
}

// QoSPolicy represents Quality of Service policies
type QoSPolicy struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	Type          QoSType           `json:"type"`
	Queues        []QoSQueue        `json:"queues"`
	Rules         []QoSRule         `json:"rules"`
	Options       map[string]string `json:"options,omitempty"`
	CreatedAt     time.Time         `json:"created_at"`
}

// QoSType represents different types of QoS policies
type QoSType string

const (
	QoSTypeHTB    QoSType = "linux-htb"
	QoSTypeCBQ    QoSType = "linux-cbq"
	QoSTypeFQ     QoSType = "linux-fq_codel"
	QoSTypeSFQ    QoSType = "linux-sfq"
)

// QoSQueue represents a QoS queue configuration
type QoSQueue struct {
	ID         int    `json:"id"`
	MinRate    int64  `json:"min_rate_bps"`
	MaxRate    int64  `json:"max_rate_bps"`
	Burst      int64  `json:"burst_bytes"`
	Priority   int    `json:"priority"`
	DSCP       int    `json:"dscp,omitempty"`
}

// QoSRule represents a QoS classification rule
type QoSRule struct {
	ID       string    `json:"id"`
	Match    FlowMatch `json:"match"`
	QueueID  int       `json:"queue_id"`
	Priority int       `json:"priority"`
}

// BridgeManager manages Open vSwitch bridges
type BridgeManager struct {
	bridges      map[string]*Bridge
	bridgesMutex sync.RWMutex
	ovsdbPath    string
	ovsctlPath   string
	ofctlPath    string
	ctx          context.Context
	cancel       context.CancelFunc
	config       BridgeManagerConfig
}

// BridgeManagerConfig holds configuration for the bridge manager
type BridgeManagerConfig struct {
	OVSDBPath           string        `json:"ovsdb_path"`
	OVSctlPath          string        `json:"ovsctl_path"`
	OFctlPath           string        `json:"ofctl_path"`
	DefaultController   []string      `json:"default_controller"`
	DefaultProtocols    []string      `json:"default_protocols"`
	MonitoringInterval  time.Duration `json:"monitoring_interval"`
	FlowTableSize       int           `json:"flow_table_size"`
	EnableSTP           bool          `json:"enable_stp"`
	EnableRSTP          bool          `json:"enable_rstp"`
	FailMode            string        `json:"fail_mode"`
}

// DefaultBridgeManagerConfig returns default configuration
func DefaultBridgeManagerConfig() BridgeManagerConfig {
	return BridgeManagerConfig{
		OVSDBPath:          "unix:/var/run/openvswitch/db.sock",
		OVSctlPath:         "/usr/bin/ovs-vsctl",
		OFctlPath:          "/usr/bin/ovs-ofctl",
		DefaultController:  []string{"tcp:127.0.0.1:6653"},
		DefaultProtocols:   []string{"OpenFlow13", "OpenFlow14", "OpenFlow15"},
		MonitoringInterval: 30 * time.Second,
		FlowTableSize:      100000,
		EnableSTP:          false,
		EnableRSTP:         true,
		FailMode:           "secure",
	}
}

// NewBridgeManager creates a new bridge manager
func NewBridgeManager(config BridgeManagerConfig) *BridgeManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &BridgeManager{
		bridges:    make(map[string]*Bridge),
		ovsdbPath:  config.OVSDBPath,
		ovsctlPath: config.OVSctlPath,
		ofctlPath:  config.OFctlPath,
		ctx:        ctx,
		cancel:     cancel,
		config:     config,
	}
}

// Start starts the bridge manager
func (bm *BridgeManager) Start() error {
	log.Println("Starting OVS Bridge Manager")
	
	// Check if OVS is installed and running
	if err := bm.checkOVSAvailability(); err != nil {
		return fmt.Errorf("OVS not available: %w", err)
	}
	
	// Load existing bridges
	if err := bm.loadExistingBridges(); err != nil {
		log.Printf("Warning: Failed to load existing bridges: %v", err)
	}
	
	// Start monitoring loop
	go bm.monitoringLoop()
	
	log.Println("OVS Bridge Manager started successfully")
	return nil
}

// Stop stops the bridge manager
func (bm *BridgeManager) Stop() error {
	log.Println("Stopping OVS Bridge Manager")
	bm.cancel()
	return nil
}

// checkOVSAvailability checks if Open vSwitch is available
func (bm *BridgeManager) checkOVSAvailability() error {
	// Check ovs-vsctl
	if _, err := exec.LookPath("ovs-vsctl"); err != nil {
		return fmt.Errorf("ovs-vsctl not found: %w", err)
	}
	
	// Check ovs-ofctl
	if _, err := exec.LookPath("ovs-ofctl"); err != nil {
		return fmt.Errorf("ovs-ofctl not found: %w", err)
	}
	
	// Test ovs-vsctl connectivity
	cmd := exec.Command("ovs-vsctl", "show")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("cannot connect to OVS database: %w", err)
	}
	
	return nil
}

// CreateBridge creates a new OVS bridge
func (bm *BridgeManager) CreateBridge(ctx context.Context, name string, bridgeType BridgeType, options map[string]string) (*Bridge, error) {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge already exists
	if _, exists := bm.bridges[name]; exists {
		return nil, fmt.Errorf("bridge %s already exists", name)
	}
	
	// Create the bridge using ovs-vsctl
	cmd := exec.Command("ovs-vsctl", "add-br", name)
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to create bridge %s: %w", name, err)
	}
	
	// Get bridge UUID
	uuid, err := bm.getBridgeUUID(name)
	if err != nil {
		return nil, fmt.Errorf("failed to get bridge UUID: %w", err)
	}
	
	// Generate datapath ID
	dpid, err := bm.getBridgeDatapathID(name)
	if err != nil {
		log.Printf("Warning: Could not get datapath ID for bridge %s: %v", name, err)
		dpid = generateDatapathID()
	}
	
	// Create bridge object
	bridge := &Bridge{
		Name:       name,
		UUID:       uuid,
		Type:       bridgeType,
		DPID:       dpid,
		Controller: bm.config.DefaultController,
		Protocols:  bm.config.DefaultProtocols,
		Ports:      []Port{},
		FlowRules:  []FlowRule{},
		Options:    options,
		Status: BridgeStatus{
			State:  "active",
			Active: true,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	// Configure bridge settings
	if err := bm.configureBridge(bridge); err != nil {
		// Clean up on failure
		bm.deleteBridgeFromOVS(name)
		return nil, fmt.Errorf("failed to configure bridge: %w", err)
	}
	
	// Store the bridge
	bm.bridges[name] = bridge
	
	log.Printf("Created OVS bridge: %s (UUID: %s, DPID: %s)", name, uuid, dpid)
	return bridge, nil
}

// configureBridge applies configuration to a bridge
func (bm *BridgeManager) configureBridge(bridge *Bridge) error {
	name := bridge.Name
	
	// Set controller
	if len(bridge.Controller) > 0 {
		args := []string{"set-controller", name}
		args = append(args, bridge.Controller...)
		cmd := exec.Command("ovs-vsctl", args...)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to set controller: %w", err)
		}
	}
	
	// Set protocols
	if len(bridge.Protocols) > 0 {
		protocols := strings.Join(bridge.Protocols, ",")
		cmd := exec.Command("ovs-vsctl", "set", "bridge", name, 
			fmt.Sprintf("protocols=%s", protocols))
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to set protocols: %w", err)
		}
	}
	
	// Set fail mode
	if bm.config.FailMode != "" {
		cmd := exec.Command("ovs-vsctl", "set-fail-mode", name, bm.config.FailMode)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to set fail mode: %w", err)
		}
	}
	
	// Enable/disable STP/RSTP
	if bm.config.EnableSTP {
		cmd := exec.Command("ovs-vsctl", "set", "bridge", name, "stp_enable=true")
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to enable STP on bridge %s: %v", name, err)
		}
	}
	
	if bm.config.EnableRSTP {
		cmd := exec.Command("ovs-vsctl", "set", "bridge", name, "rstp_enable=true")
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to enable RSTP on bridge %s: %v", name, err)
		}
	}
	
	// Apply custom options
	for key, value := range bridge.Options {
		cmd := exec.Command("ovs-vsctl", "set", "bridge", name, fmt.Sprintf("%s=%s", key, value))
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to set option %s=%s on bridge %s: %v", key, value, name, err)
		}
	}
	
	return nil
}

// DeleteBridge deletes an OVS bridge
func (bm *BridgeManager) DeleteBridge(ctx context.Context, name string) error {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge exists
	bridge, exists := bm.bridges[name]
	if !exists {
		return fmt.Errorf("bridge %s not found", name)
	}
	
	// Delete flow rules first
	if err := bm.deleteAllFlowRules(name); err != nil {
		log.Printf("Warning: Failed to delete flow rules for bridge %s: %v", name, err)
	}
	
	// Delete the bridge from OVS
	if err := bm.deleteBridgeFromOVS(name); err != nil {
		return fmt.Errorf("failed to delete bridge from OVS: %w", err)
	}
	
	// Remove from our map
	delete(bm.bridges, name)
	
	log.Printf("Deleted OVS bridge: %s (UUID: %s)", name, bridge.UUID)
	return nil
}

// deleteBridgeFromOVS removes bridge from OVS
func (bm *BridgeManager) deleteBridgeFromOVS(name string) error {
	cmd := exec.Command("ovs-vsctl", "--if-exists", "del-br", name)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete bridge: %w", err)
	}
	return nil
}

// AddPort adds a port to a bridge
func (bm *BridgeManager) AddPort(ctx context.Context, bridgeName, portName string, portType PortType, options map[string]string) (*Port, error) {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge exists
	bridge, exists := bm.bridges[bridgeName]
	if !exists {
		return nil, fmt.Errorf("bridge %s not found", bridgeName)
	}
	
	// Check if port already exists
	for _, port := range bridge.Ports {
		if port.Name == portName {
			return nil, fmt.Errorf("port %s already exists on bridge %s", portName, bridgeName)
		}
	}
	
	// Add port to OVS
	if err := bm.addPortToOVS(bridgeName, portName, portType, options); err != nil {
		return nil, fmt.Errorf("failed to add port to OVS: %w", err)
	}
	
	// Get port details
	ofPort, err := bm.getPortOfPort(bridgeName, portName)
	if err != nil {
		log.Printf("Warning: Could not get OpenFlow port number for %s: %v", portName, err)
		ofPort = 0
	}
	
	uuid, err := bm.getPortUUID(portName)
	if err != nil {
		log.Printf("Warning: Could not get UUID for port %s: %v", portName, err)
		uuid = uuid.New().String()
	}
	
	// Create port object
	port := &Port{
		ID:        uuid,
		Name:      portName,
		UUID:      uuid,
		OfPort:    ofPort,
		Type:      portType,
		Interface: portName,
		Options:   options,
		Status: PortStatus{
			State: "active",
			Link:  true,
		},
	}
	
	// Add to bridge ports
	bridge.Ports = append(bridge.Ports, *port)
	bridge.UpdatedAt = time.Now()
	
	log.Printf("Added port %s to bridge %s (OF port: %d)", portName, bridgeName, ofPort)
	return port, nil
}

// addPortToOVS adds a port to OVS bridge
func (bm *BridgeManager) addPortToOVS(bridgeName, portName string, portType PortType, options map[string]string) error {
	args := []string{"add-port", bridgeName, portName}
	
	// Add port type and options
	if portType != PortTypePhysical {
		args = append(args, "--", "set", "interface", portName, fmt.Sprintf("type=%s", portType))
		
		// Add type-specific options
		for key, value := range options {
			args = append(args, fmt.Sprintf("options:%s=%s", key, value))
		}
	}
	
	cmd := exec.Command("ovs-vsctl", args...)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add port: %w", err)
	}
	
	return nil
}

// DeletePort removes a port from a bridge
func (bm *BridgeManager) DeletePort(ctx context.Context, bridgeName, portName string) error {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge exists
	bridge, exists := bm.bridges[bridgeName]
	if !exists {
		return fmt.Errorf("bridge %s not found", bridgeName)
	}
	
	// Find and remove port from bridge
	portFound := false
	for i, port := range bridge.Ports {
		if port.Name == portName {
			bridge.Ports = append(bridge.Ports[:i], bridge.Ports[i+1:]...)
			portFound = true
			break
		}
	}
	
	if !portFound {
		return fmt.Errorf("port %s not found on bridge %s", portName, bridgeName)
	}
	
	// Remove from OVS
	cmd := exec.Command("ovs-vsctl", "del-port", bridgeName, portName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete port from OVS: %w", err)
	}
	
	bridge.UpdatedAt = time.Now()
	
	log.Printf("Deleted port %s from bridge %s", portName, bridgeName)
	return nil
}

// AddFlowRule adds a flow rule to a bridge
func (bm *BridgeManager) AddFlowRule(ctx context.Context, bridgeName string, rule FlowRule) error {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge exists
	bridge, exists := bm.bridges[bridgeName]
	if !exists {
		return fmt.Errorf("bridge %s not found", bridgeName)
	}
	
	// Generate rule ID if not provided
	if rule.ID == "" {
		rule.ID = uuid.New().String()
	}
	
	// Set creation time
	rule.CreatedAt = time.Now()
	
	// Convert rule to ovs-ofctl format
	ruleStr := bm.flowRuleToString(rule)
	
	// Add rule to OVS
	cmd := exec.Command("ovs-ofctl", "add-flow", bridgeName, ruleStr)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add flow rule: %w", err)
	}
	
	// Add to bridge rules
	bridge.FlowRules = append(bridge.FlowRules, rule)
	bridge.UpdatedAt = time.Now()
	
	log.Printf("Added flow rule %s to bridge %s", rule.ID, bridgeName)
	return nil
}

// flowRuleToString converts a FlowRule to ovs-ofctl string format
func (bm *BridgeManager) flowRuleToString(rule FlowRule) string {
	var parts []string
	
	// Add table if specified
	if rule.Table > 0 {
		parts = append(parts, fmt.Sprintf("table=%d", rule.Table))
	}
	
	// Add priority
	parts = append(parts, fmt.Sprintf("priority=%d", rule.Priority))
	
	// Add cookie
	if rule.Cookie > 0 {
		parts = append(parts, fmt.Sprintf("cookie=%d", rule.Cookie))
	}
	
	// Add timeouts
	if rule.IdleTimeout > 0 {
		parts = append(parts, fmt.Sprintf("idle_timeout=%d", rule.IdleTimeout))
	}
	if rule.HardTimeout > 0 {
		parts = append(parts, fmt.Sprintf("hard_timeout=%d", rule.HardTimeout))
	}
	
	// Add match criteria
	matchParts := bm.flowMatchToString(rule.Match)
	if matchParts != "" {
		parts = append(parts, matchParts)
	}
	
	// Add actions
	actionParts := bm.flowActionsToString(rule.Actions)
	if actionParts != "" {
		parts = append(parts, fmt.Sprintf("actions=%s", actionParts))
	}
	
	return strings.Join(parts, ",")
}

// flowMatchToString converts FlowMatch to string format
func (bm *BridgeManager) flowMatchToString(match FlowMatch) string {
	var parts []string
	
	if match.InPort != "" {
		parts = append(parts, fmt.Sprintf("in_port=%s", match.InPort))
	}
	if match.EthSrc != "" {
		parts = append(parts, fmt.Sprintf("dl_src=%s", match.EthSrc))
	}
	if match.EthDst != "" {
		parts = append(parts, fmt.Sprintf("dl_dst=%s", match.EthDst))
	}
	if match.EthType != "" {
		parts = append(parts, fmt.Sprintf("dl_type=%s", match.EthType))
	}
	if match.VlanID > 0 {
		parts = append(parts, fmt.Sprintf("dl_vlan=%d", match.VlanID))
	}
	if match.VlanPCP > 0 {
		parts = append(parts, fmt.Sprintf("dl_vlan_pcp=%d", match.VlanPCP))
	}
	if match.IPSrc != "" {
		parts = append(parts, fmt.Sprintf("nw_src=%s", match.IPSrc))
	}
	if match.IPDst != "" {
		parts = append(parts, fmt.Sprintf("nw_dst=%s", match.IPDst))
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
	if match.TunnelID > 0 {
		parts = append(parts, fmt.Sprintf("tun_id=%d", match.TunnelID))
	}
	if match.Metadata > 0 {
		parts = append(parts, fmt.Sprintf("metadata=%d", match.Metadata))
	}
	
	return strings.Join(parts, ",")
}

// flowActionsToString converts FlowActions to string format
func (bm *BridgeManager) flowActionsToString(actions []FlowAction) string {
	var parts []string
	
	for _, action := range actions {
		switch action.Type {
		case "output":
			if port, exists := action.Params["port"]; exists {
				parts = append(parts, fmt.Sprintf("output:%s", port))
			}
		case "drop":
			parts = append(parts, "drop")
		case "set_field":
			if field, exists := action.Params["field"]; exists {
				if value, exists := action.Params["value"]; exists {
					parts = append(parts, fmt.Sprintf("set_field:%s->%s", value, field))
				}
			}
		case "push_vlan":
			parts = append(parts, "push_vlan:0x8100")
		case "pop_vlan":
			parts = append(parts, "pop_vlan")
		case "set_queue":
			if queue, exists := action.Params["queue_id"]; exists {
				parts = append(parts, fmt.Sprintf("set_queue:%s", queue))
			}
		case "group":
			if group, exists := action.Params["group_id"]; exists {
				parts = append(parts, fmt.Sprintf("group:%s", group))
			}
		case "controller":
			parts = append(parts, "controller")
		default:
			// Custom action - convert params to key=value format
			var actionParts []string
			for key, value := range action.Params {
				actionParts = append(actionParts, fmt.Sprintf("%s=%s", key, value))
			}
			if len(actionParts) > 0 {
				parts = append(parts, fmt.Sprintf("%s:%s", action.Type, strings.Join(actionParts, ",")))
			}
		}
	}
	
	return strings.Join(parts, ",")
}

// DeleteFlowRule removes a flow rule from a bridge
func (bm *BridgeManager) DeleteFlowRule(ctx context.Context, bridgeName, ruleID string) error {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	// Check if bridge exists
	bridge, exists := bm.bridges[bridgeName]
	if !exists {
		return fmt.Errorf("bridge %s not found", bridgeName)
	}
	
	// Find and remove rule
	ruleFound := false
	var ruleToDelete FlowRule
	for i, rule := range bridge.FlowRules {
		if rule.ID == ruleID {
			ruleToDelete = rule
			bridge.FlowRules = append(bridge.FlowRules[:i], bridge.FlowRules[i+1:]...)
			ruleFound = true
			break
		}
	}
	
	if !ruleFound {
		return fmt.Errorf("flow rule %s not found on bridge %s", ruleID, bridgeName)
	}
	
	// Delete from OVS (use cookie to identify the rule)
	var cmd *exec.Cmd
	if ruleToDelete.Cookie > 0 {
		cmd = exec.Command("ovs-ofctl", "del-flows", bridgeName, 
			fmt.Sprintf("cookie=%d/-1", ruleToDelete.Cookie))
	} else {
		// Delete by match criteria if no cookie
		matchStr := bm.flowMatchToString(ruleToDelete.Match)
		if matchStr != "" {
			cmd = exec.Command("ovs-ofctl", "del-flows", bridgeName, matchStr)
		} else {
			log.Printf("Warning: Cannot identify flow rule %s for deletion", ruleID)
			return nil
		}
	}
	
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete flow rule from OVS: %w", err)
	}
	
	bridge.UpdatedAt = time.Now()
	
	log.Printf("Deleted flow rule %s from bridge %s", ruleID, bridgeName)
	return nil
}

// deleteAllFlowRules removes all flow rules from a bridge
func (bm *BridgeManager) deleteAllFlowRules(bridgeName string) error {
	cmd := exec.Command("ovs-ofctl", "del-flows", bridgeName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete all flow rules: %w", err)
	}
	
	// Clear from bridge object
	if bridge, exists := bm.bridges[bridgeName]; exists {
		bridge.FlowRules = []FlowRule{}
		bridge.UpdatedAt = time.Now()
	}
	
	return nil
}

// GetBridge returns a bridge by name
func (bm *BridgeManager) GetBridge(name string) (*Bridge, error) {
	bm.bridgesMutex.RLock()
	defer bm.bridgesMutex.RUnlock()
	
	bridge, exists := bm.bridges[name]
	if !exists {
		return nil, fmt.Errorf("bridge %s not found", name)
	}
	
	return bridge, nil
}

// ListBridges returns all bridges
func (bm *BridgeManager) ListBridges() []*Bridge {
	bm.bridgesMutex.RLock()
	defer bm.bridgesMutex.RUnlock()
	
	bridges := make([]*Bridge, 0, len(bm.bridges))
	for _, bridge := range bm.bridges {
		bridges = append(bridges, bridge)
	}
	
	return bridges
}

// loadExistingBridges loads bridges that already exist in OVS
func (bm *BridgeManager) loadExistingBridges() error {
	// Get list of bridges from OVS
	cmd := exec.Command("ovs-vsctl", "list-br")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to list bridges: %w", err)
	}
	
	bridgeNames := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, name := range bridgeNames {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		
		// Get bridge details
		bridge, err := bm.loadBridgeDetails(name)
		if err != nil {
			log.Printf("Warning: Failed to load bridge %s: %v", name, err)
			continue
		}
		
		bm.bridges[name] = bridge
		log.Printf("Loaded existing bridge: %s", name)
	}
	
	return nil
}

// loadBridgeDetails loads detailed information about a bridge
func (bm *BridgeManager) loadBridgeDetails(name string) (*Bridge, error) {
	// Get bridge UUID
	uuid, err := bm.getBridgeUUID(name)
	if err != nil {
		return nil, err
	}
	
	// Get datapath ID
	dpid, err := bm.getBridgeDatapathID(name)
	if err != nil {
		dpid = generateDatapathID()
	}
	
	// Get controller info
	controllers, err := bm.getBridgeControllers(name)
	if err != nil {
		controllers = []string{}
	}
	
	// Get protocols
	protocols, err := bm.getBridgeProtocols(name)
	if err != nil {
		protocols = bm.config.DefaultProtocols
	}
	
	// Load ports
	ports, err := bm.loadBridgePorts(name)
	if err != nil {
		log.Printf("Warning: Failed to load ports for bridge %s: %v", name, err)
		ports = []Port{}
	}
	
	// Load flow rules
	flowRules, err := bm.loadBridgeFlowRules(name)
	if err != nil {
		log.Printf("Warning: Failed to load flow rules for bridge %s: %v", name, err)
		flowRules = []FlowRule{}
	}
	
	bridge := &Bridge{
		Name:       name,
		UUID:       uuid,
		Type:       BridgeTypeNormal,
		DPID:       dpid,
		Controller: controllers,
		Protocols:  protocols,
		Ports:      ports,
		FlowRules:  flowRules,
		Options:    make(map[string]string),
		Status: BridgeStatus{
			State:     "active",
			Active:    true,
			FlowCount: len(flowRules),
		},
		CreatedAt: time.Now(), // We don't know the real creation time
		UpdatedAt: time.Now(),
	}
	
	return bridge, nil
}

// Helper functions to get bridge information from OVS

func (bm *BridgeManager) getBridgeUUID(name string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "bridge", name, "_uuid")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func (bm *BridgeManager) getBridgeDatapathID(name string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "bridge", name, "datapath_id")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.Trim(strings.TrimSpace(string(output)), "\""), nil
}

func (bm *BridgeManager) getBridgeControllers(name string) ([]string, error) {
	cmd := exec.Command("ovs-vsctl", "get-controller", name)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	
	controllerStr := strings.TrimSpace(string(output))
	if controllerStr == "" {
		return []string{}, nil
	}
	
	return strings.Split(controllerStr, " "), nil
}

func (bm *BridgeManager) getBridgeProtocols(name string) ([]string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "bridge", name, "protocols")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	
	protocolsStr := strings.Trim(strings.TrimSpace(string(output)), "[]\"")
	if protocolsStr == "" {
		return []string{}, nil
	}
	
	protocols := strings.Split(protocolsStr, ",")
	for i, p := range protocols {
		protocols[i] = strings.Trim(strings.TrimSpace(p), "\"")
	}
	
	return protocols, nil
}

func (bm *BridgeManager) loadBridgePorts(name string) ([]Port, error) {
	// Get list of ports
	cmd := exec.Command("ovs-vsctl", "list-ports", name)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	
	var ports []Port
	portNames := strings.Split(strings.TrimSpace(string(output)), "\n")
	
	for _, portName := range portNames {
		portName = strings.TrimSpace(portName)
		if portName == "" {
			continue
		}
		
		port, err := bm.loadPortDetails(name, portName)
		if err != nil {
			log.Printf("Warning: Failed to load port %s: %v", portName, err)
			continue
		}
		
		ports = append(ports, *port)
	}
	
	return ports, nil
}

func (bm *BridgeManager) loadPortDetails(bridgeName, portName string) (*Port, error) {
	// Get port UUID
	uuid, err := bm.getPortUUID(portName)
	if err != nil {
		uuid = uuid.New().String()
	}
	
	// Get OpenFlow port number
	ofPort, err := bm.getPortOfPort(bridgeName, portName)
	if err != nil {
		ofPort = 0
	}
	
	// Get port type
	portType, err := bm.getPortType(portName)
	if err != nil {
		portType = PortTypePhysical
	}
	
	port := &Port{
		ID:        uuid,
		Name:      portName,
		UUID:      uuid,
		OfPort:    ofPort,
		Type:      portType,
		Interface: portName,
		Options:   make(map[string]string),
		Status: PortStatus{
			State: "active",
			Link:  true,
		},
	}
	
	return port, nil
}

func (bm *BridgeManager) getPortUUID(portName string) (string, error) {
	cmd := exec.Command("ovs-vsctl", "get", "port", portName, "_uuid")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func (bm *BridgeManager) getPortOfPort(bridgeName, portName string) (int, error) {
	cmd := exec.Command("ovs-vsctl", "get", "interface", portName, "ofport")
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	
	ofPortStr := strings.TrimSpace(string(output))
	ofPort, err := strconv.Atoi(ofPortStr)
	if err != nil {
		return 0, err
	}
	
	return ofPort, nil
}

func (bm *BridgeManager) getPortType(portName string) (PortType, error) {
	cmd := exec.Command("ovs-vsctl", "get", "interface", portName, "type")
	output, err := cmd.Output()
	if err != nil {
		return PortTypePhysical, nil
	}
	
	typeStr := strings.Trim(strings.TrimSpace(string(output)), "\"")
	switch typeStr {
	case "internal":
		return PortTypeInternal, nil
	case "vxlan":
		return PortTypeVXLAN, nil
	case "gre":
		return PortTypeGRE, nil
	case "geneve":
		return PortTypeGeneve, nil
	case "patch":
		return PortTypePatch, nil
	default:
		return PortTypePhysical, nil
	}
}

func (bm *BridgeManager) loadBridgeFlowRules(name string) ([]FlowRule, error) {
	// For now, return empty slice as parsing ovs-ofctl output is complex
	// In a production implementation, this would parse the flow table
	return []FlowRule{}, nil
}

// monitoringLoop periodically updates bridge status and statistics
func (bm *BridgeManager) monitoringLoop() {
	ticker := time.NewTicker(bm.config.MonitoringInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-bm.ctx.Done():
			return
		case <-ticker.C:
			bm.updateBridgeStatistics()
		}
	}
}

// updateBridgeStatistics updates statistics for all bridges
func (bm *BridgeManager) updateBridgeStatistics() {
	bm.bridgesMutex.Lock()
	defer bm.bridgesMutex.Unlock()
	
	for name, bridge := range bm.bridges {
		// Update flow count
		flowCount, err := bm.getBridgeFlowCount(name)
		if err == nil {
			bridge.Status.FlowCount = flowCount
		}
		
		// Update packet/byte counts (simplified - would aggregate from ports)
		bridge.Status.LastSeen = time.Now()
		bridge.UpdatedAt = time.Now()
	}
}

func (bm *BridgeManager) getBridgeFlowCount(name string) (int, error) {
	cmd := exec.Command("ovs-ofctl", "dump-flows", name, "-O", "OpenFlow13")
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	
	lines := strings.Split(string(output), "\n")
	count := 0
	for _, line := range lines {
		if strings.Contains(line, "cookie=") {
			count++
		}
	}
	
	return count, nil
}

// generateDatapathID generates a random datapath ID
func generateDatapathID() string {
	// Generate a random 64-bit datapath ID
	id := uuid.New()
	return fmt.Sprintf("%016x", id.ID())
}