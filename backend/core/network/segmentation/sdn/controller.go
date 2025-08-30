package sdn

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
)

// OpenFlowVersion represents supported OpenFlow versions
type OpenFlowVersion uint8

const (
	OpenFlow13 OpenFlowVersion = 0x04
	OpenFlow14 OpenFlowVersion = 0x05
	OpenFlow15 OpenFlowVersion = 0x06
)

// FlowTableID represents OpenFlow table IDs
type FlowTableID uint8

const (
	TableClassification FlowTableID = 0
	TableAcl            FlowTableID = 1
	TableQoS            FlowTableID = 2
	TableForwarding     FlowTableID = 3
	TableEgress         FlowTableID = 4
)

// ControllerState represents the state of the SDN controller
type ControllerState string

const (
	StateStarting ControllerState = "starting"
	StateRunning  ControllerState = "running"
	StateStopping ControllerState = "stopping"
	StateStopped  ControllerState = "stopped"
	StateError    ControllerState = "error"
)

// Switch represents a managed OpenFlow switch
type Switch struct {
	ID               string            `json:"id"`
	DPID             uint64            `json:"dpid"`
	Name             string            `json:"name"`
	IPAddress        string            `json:"ip_address"`
	Port             int               `json:"port"`
	OpenFlowVersion  OpenFlowVersion   `json:"openflow_version"`
	ConnectionState  string            `json:"connection_state"`
	LastHeartbeat    time.Time         `json:"last_heartbeat"`
	Features         SwitchFeatures    `json:"features"`
	FlowTables       map[FlowTableID]*FlowTable `json:"flow_tables"`
	PortStats        map[uint32]PortStatistics  `json:"port_stats"`
	FlowStats        []FlowStatistics  `json:"flow_stats"`
	Capabilities     []string          `json:"capabilities"`
	Actions          []string          `json:"actions"`
	TenantID         string            `json:"tenant_id,omitempty"`
	CreatedAt        time.Time         `json:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at"`
}

// SwitchFeatures represents OpenFlow switch features
type SwitchFeatures struct {
	NTables    uint8             `json:"n_tables"`
	NPorts     uint32            `json:"n_ports"`
	NBuffers   uint32            `json:"n_buffers"`
	Auxiliary  bool              `json:"auxiliary"`
	Reserved   uint32            `json:"reserved"`
	Ports      []PhysicalPort    `json:"ports"`
}

// PhysicalPort represents a physical port on the switch
type PhysicalPort struct {
	PortNo     uint32    `json:"port_no"`
	HWAddr     [6]byte   `json:"hw_addr"`
	Name       string    `json:"name"`
	Config     uint32    `json:"config"`
	State      uint32    `json:"state"`
	Current    uint32    `json:"current"`
	Advertised uint32    `json:"advertised"`
	Supported  uint32    `json:"supported"`
	Peer       uint32    `json:"peer"`
	CurrSpeed  uint32    `json:"curr_speed"`
	MaxSpeed   uint32    `json:"max_speed"`
}

// FlowTable represents an OpenFlow flow table
type FlowTable struct {
	TableID      FlowTableID `json:"table_id"`
	Name         string      `json:"name"`
	MaxEntries   uint32      `json:"max_entries"`
	ActiveCount  uint32      `json:"active_count"`
	LookupCount  uint64      `json:"lookup_count"`
	MatchedCount uint64      `json:"matched_count"`
	Flows        []FlowEntry `json:"flows"`
}

// FlowEntry represents an OpenFlow flow entry
type FlowEntry struct {
	ID           string         `json:"id"`
	TableID      FlowTableID    `json:"table_id"`
	Priority     uint16         `json:"priority"`
	IdleTimeout  uint16         `json:"idle_timeout"`
	HardTimeout  uint16         `json:"hard_timeout"`
	Cookie       uint64         `json:"cookie"`
	Match        FlowMatch      `json:"match"`
	Instructions []Instruction  `json:"instructions"`
	Stats        FlowStatistics `json:"stats"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
}

// FlowMatch represents OpenFlow flow match criteria
type FlowMatch struct {
	InPort       *uint32    `json:"in_port,omitempty"`
	EthSrc       *[6]byte   `json:"eth_src,omitempty"`
	EthDst       *[6]byte   `json:"eth_dst,omitempty"`
	EthType      *uint16    `json:"eth_type,omitempty"`
	VlanVID      *uint16    `json:"vlan_vid,omitempty"`
	VlanPCP      *uint8     `json:"vlan_pcp,omitempty"`
	IPProto      *uint8     `json:"ip_proto,omitempty"`
	IPv4Src      *net.IPNet `json:"ipv4_src,omitempty"`
	IPv4Dst      *net.IPNet `json:"ipv4_dst,omitempty"`
	IPv6Src      *net.IPNet `json:"ipv6_src,omitempty"`
	IPv6Dst      *net.IPNet `json:"ipv6_dst,omitempty"`
	TCPSrc       *uint16    `json:"tcp_src,omitempty"`
	TCPDst       *uint16    `json:"tcp_dst,omitempty"`
	UDPSrc       *uint16    `json:"udp_src,omitempty"`
	UDPDst       *uint16    `json:"udp_dst,omitempty"`
	ICMPType     *uint8     `json:"icmp_type,omitempty"`
	ICMPCode     *uint8     `json:"icmp_code,omitempty"`
	TunnelID     *uint64    `json:"tunnel_id,omitempty"`
	Metadata     *uint64    `json:"metadata,omitempty"`
}

// InstructionType represents OpenFlow instruction types
type InstructionType string

const (
	InstGotoTable   InstructionType = "goto_table"
	InstWriteMetadata InstructionType = "write_metadata"
	InstWriteActions InstructionType = "write_actions"
	InstApplyActions InstructionType = "apply_actions"
	InstClearActions InstructionType = "clear_actions"
	InstMeter       InstructionType = "meter"
)

// Instruction represents an OpenFlow instruction
type Instruction struct {
	Type     InstructionType `json:"type"`
	TableID  *FlowTableID    `json:"table_id,omitempty"`
	Metadata *uint64         `json:"metadata,omitempty"`
	Actions  []Action        `json:"actions,omitempty"`
	MeterID  *uint32         `json:"meter_id,omitempty"`
}

// ActionType represents OpenFlow action types
type ActionType string

const (
	ActionOutput       ActionType = "output"
	ActionSetField     ActionType = "set_field"
	ActionPushVlan     ActionType = "push_vlan"
	ActionPopVlan      ActionType = "pop_vlan"
	ActionSetQueue     ActionType = "set_queue"
	ActionGroup        ActionType = "group"
	ActionSetNwTTL     ActionType = "set_nw_ttl"
	ActionDecNwTTL     ActionType = "dec_nw_ttl"
	ActionCopyTTLOut   ActionType = "copy_ttl_out"
	ActionCopyTTLIn    ActionType = "copy_ttl_in"
	ActionPushMPLS     ActionType = "push_mpls"
	ActionPopMPLS      ActionType = "pop_mpls"
)

// Action represents an OpenFlow action
type Action struct {
	Type     ActionType      `json:"type"`
	Port     *uint32         `json:"port,omitempty"`
	Field    *FieldType      `json:"field,omitempty"`
	Value    interface{}     `json:"value,omitempty"`
	QueueID  *uint32         `json:"queue_id,omitempty"`
	GroupID  *uint32         `json:"group_id,omitempty"`
	EthType  *uint16         `json:"eth_type,omitempty"`
}

// FieldType represents OpenFlow OXM field types
type FieldType string

const (
	FieldInPort     FieldType = "in_port"
	FieldEthSrc     FieldType = "eth_src"
	FieldEthDst     FieldType = "eth_dst"
	FieldEthType    FieldType = "eth_type"
	FieldVlanVID    FieldType = "vlan_vid"
	FieldVlanPCP    FieldType = "vlan_pcp"
	FieldIPProto    FieldType = "ip_proto"
	FieldIPv4Src    FieldType = "ipv4_src"
	FieldIPv4Dst    FieldType = "ipv4_dst"
	FieldIPv6Src    FieldType = "ipv6_src"
	FieldIPv6Dst    FieldType = "ipv6_dst"
	FieldTCPSrc     FieldType = "tcp_src"
	FieldTCPDst     FieldType = "tcp_dst"
	FieldUDPSrc     FieldType = "udp_src"
	FieldUDPDst     FieldType = "udp_dst"
	FieldTunnelID   FieldType = "tunnel_id"
	FieldMetadata   FieldType = "metadata"
)

// Statistics structures
type PortStatistics struct {
	PortNo       uint32 `json:"port_no"`
	RxPackets    uint64 `json:"rx_packets"`
	TxPackets    uint64 `json:"tx_packets"`
	RxBytes      uint64 `json:"rx_bytes"`
	TxBytes      uint64 `json:"tx_bytes"`
	RxDropped    uint64 `json:"rx_dropped"`
	TxDropped    uint64 `json:"tx_dropped"`
	RxErrors     uint64 `json:"rx_errors"`
	TxErrors     uint64 `json:"tx_errors"`
	RxFrameErr   uint64 `json:"rx_frame_err"`
	RxOverErr    uint64 `json:"rx_over_err"`
	RxCrcErr     uint64 `json:"rx_crc_err"`
	Collisions   uint64 `json:"collisions"`
	DurationSec  uint32 `json:"duration_sec"`
	DurationNsec uint32 `json:"duration_nsec"`
}

type FlowStatistics struct {
	TableID      FlowTableID `json:"table_id"`
	Priority     uint16      `json:"priority"`
	IdleTimeout  uint16      `json:"idle_timeout"`
	HardTimeout  uint16      `json:"hard_timeout"`
	Cookie       uint64      `json:"cookie"`
	PacketCount  uint64      `json:"packet_count"`
	ByteCount    uint64      `json:"byte_count"`
	DurationSec  uint32      `json:"duration_sec"`
	DurationNsec uint32      `json:"duration_nsec"`
}

// PolicyRule represents a high-level network policy rule
type PolicyRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	TenantID    string            `json:"tenant_id"`
	Priority    int               `json:"priority"`
	Direction   string            `json:"direction"` // ingress, egress, both
	Protocol    string            `json:"protocol"`
	SrcSelector string            `json:"src_selector"`
	DstSelector string            `json:"dst_selector"`
	SrcPorts    []string          `json:"src_ports,omitempty"`
	DstPorts    []string          `json:"dst_ports,omitempty"`
	Action      string            `json:"action"` // allow, deny, redirect, qos
	QoSClass    string            `json:"qos_class,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	Enabled     bool              `json:"enabled"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// ControllerEvent represents events from the SDN controller
type ControllerEvent struct {
	Type      string      `json:"type"`
	SwitchID  string      `json:"switch_id,omitempty"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

// EventListener is a callback for controller events
type EventListener func(event ControllerEvent)

// SDNController represents the main SDN controller
type SDNController struct {
	ID               string                    `json:"id"`
	Name             string                    `json:"name"`
	ListenAddress    string                   `json:"listen_address"`
	ListenPort       int                      `json:"listen_port"`
	State            ControllerState          `json:"state"`
	OpenFlowVersion  OpenFlowVersion          `json:"openflow_version"`
	Switches         map[string]*Switch       `json:"switches"`
	PolicyRules      map[string]*PolicyRule   `json:"policy_rules"`
	FlowRuleCache    map[string][]FlowEntry   `json:"flow_rule_cache"`
	EventListeners   []EventListener          `json:"-"`
	
	// Connection management
	listener         net.Listener
	connections      map[string]net.Conn
	connectionsMutex sync.RWMutex
	
	// Policy compilation
	policyCompiler   *PolicyCompiler
	
	// Statistics collection
	statsInterval    time.Duration
	
	// Synchronization
	mutex            sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	
	// Metrics
	metrics          *ControllerMetrics
	
	// Configuration
	config           *ControllerConfig
}

// ControllerConfig holds SDN controller configuration
type ControllerConfig struct {
	ListenAddress       string        `json:"listen_address"`
	ListenPort          int           `json:"listen_port"`
	OpenFlowVersion     OpenFlowVersion `json:"openflow_version"`
	HeartbeatInterval   time.Duration `json:"heartbeat_interval"`
	StatsInterval       time.Duration `json:"stats_interval"`
	FlowTableSize       uint32        `json:"flow_table_size"`
	EnableMetrics       bool          `json:"enable_metrics"`
	EnableEventLogging  bool          `json:"enable_event_logging"`
	TLSConfig           *TLSConfig    `json:"tls_config,omitempty"`
}

// TLSConfig holds TLS configuration for secure connections
type TLSConfig struct {
	Enabled    bool   `json:"enabled"`
	CertFile   string `json:"cert_file"`
	KeyFile    string `json:"key_file"`
	CAFile     string `json:"ca_file,omitempty"`
	VerifyPeer bool   `json:"verify_peer"`
}

// ControllerMetrics holds controller performance metrics
type ControllerMetrics struct {
	SwitchesConnected    int64     `json:"switches_connected"`
	FlowRulesInstalled   int64     `json:"flow_rules_installed"`
	PacketsProcessed     int64     `json:"packets_processed"`
	BytesProcessed       int64     `json:"bytes_processed"`
	PolicyRulesActive    int64     `json:"policy_rules_active"`
	EventsGenerated      int64     `json:"events_generated"`
	LastStatsUpdate      time.Time `json:"last_stats_update"`
	StartTime            time.Time `json:"start_time"`
}

// NewSDNController creates a new SDN controller
func NewSDNController(config *ControllerConfig) *SDNController {
	ctx, cancel := context.WithCancel(context.Background())
	
	controller := &SDNController{
		ID:               uuid.New().String(),
		Name:             fmt.Sprintf("sdn-controller-%s", uuid.New().String()[:8]),
		ListenAddress:    config.ListenAddress,
		ListenPort:       config.ListenPort,
		State:            StateStopped,
		OpenFlowVersion:  config.OpenFlowVersion,
		Switches:         make(map[string]*Switch),
		PolicyRules:      make(map[string]*PolicyRule),
		FlowRuleCache:    make(map[string][]FlowEntry),
		EventListeners:   make([]EventListener, 0),
		connections:      make(map[string]net.Conn),
		statsInterval:    config.StatsInterval,
		ctx:              ctx,
		cancel:           cancel,
		metrics: &ControllerMetrics{
			StartTime: time.Now(),
		},
		config: config,
	}
	
	controller.policyCompiler = NewPolicyCompiler(controller)
	
	return controller
}

// Start starts the SDN controller
func (c *SDNController) Start() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if c.State != StateStopped {
		return fmt.Errorf("controller is not in stopped state: %s", c.State)
	}
	
	c.State = StateStarting
	log.Printf("Starting SDN controller %s on %s:%d", c.Name, c.ListenAddress, c.ListenPort)
	
	// Start listening for OpenFlow connections
	listenAddr := fmt.Sprintf("%s:%d", c.ListenAddress, c.ListenPort)
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		c.State = StateError
		return fmt.Errorf("failed to start listener: %w", err)
	}
	
	c.listener = listener
	c.State = StateRunning
	
	// Start connection acceptor
	c.wg.Add(1)
	go c.acceptConnections()
	
	// Start statistics collector
	if c.config.EnableMetrics {
		c.wg.Add(1)
		go c.collectStatistics()
	}
	
	c.emitEvent(ControllerEvent{
		Type:      "controller_started",
		Data:      map[string]interface{}{"controller_id": c.ID},
		Timestamp: time.Now(),
	})
	
	log.Printf("SDN controller %s started successfully", c.Name)
	return nil
}

// Stop stops the SDN controller
func (c *SDNController) Stop() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if c.State != StateRunning {
		return fmt.Errorf("controller is not running")
	}
	
	c.State = StateStopping
	log.Printf("Stopping SDN controller %s", c.Name)
	
	// Cancel context to stop goroutines
	c.cancel()
	
	// Close listener
	if c.listener != nil {
		c.listener.Close()
	}
	
	// Close all connections
	c.connectionsMutex.Lock()
	for switchID, conn := range c.connections {
		log.Printf("Closing connection to switch %s", switchID)
		conn.Close()
	}
	c.connectionsMutex.Unlock()
	
	// Wait for goroutines to finish
	c.wg.Wait()
	
	c.State = StateStopped
	
	c.emitEvent(ControllerEvent{
		Type:      "controller_stopped",
		Data:      map[string]interface{}{"controller_id": c.ID},
		Timestamp: time.Now(),
	})
	
	log.Printf("SDN controller %s stopped", c.Name)
	return nil
}

// AddSwitch adds a switch to the controller
func (c *SDNController) AddSwitch(dpid uint64, name string, tenantID string) (*Switch, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	switchID := fmt.Sprintf("switch-%016x", dpid)
	
	if _, exists := c.Switches[switchID]; exists {
		return nil, fmt.Errorf("switch with DPID %016x already exists", dpid)
	}
	
	switch_ := &Switch{
		ID:               switchID,
		DPID:             dpid,
		Name:             name,
		OpenFlowVersion:  c.OpenFlowVersion,
		ConnectionState:  "disconnected",
		FlowTables:       make(map[FlowTableID]*FlowTable),
		PortStats:        make(map[uint32]PortStatistics),
		FlowStats:        make([]FlowStatistics, 0),
		TenantID:         tenantID,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}
	
	// Initialize flow tables
	c.initializeFlowTables(switch_)
	
	c.Switches[switchID] = switch_
	c.metrics.SwitchesConnected++
	
	c.emitEvent(ControllerEvent{
		Type:      "switch_added",
		SwitchID:  switchID,
		Data:      switch_,
		Timestamp: time.Now(),
	})
	
	log.Printf("Added switch %s (DPID: %016x) for tenant %s", name, dpid, tenantID)
	return switch_, nil
}

// RemoveSwitch removes a switch from the controller
func (c *SDNController) RemoveSwitch(switchID string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	switch_, exists := c.Switches[switchID]
	if !exists {
		return fmt.Errorf("switch %s not found", switchID)
	}
	
	// Close connection if active
	c.connectionsMutex.Lock()
	if conn, exists := c.connections[switchID]; exists {
		conn.Close()
		delete(c.connections, switchID)
	}
	c.connectionsMutex.Unlock()
	
	delete(c.Switches, switchID)
	c.metrics.SwitchesConnected--
	
	c.emitEvent(ControllerEvent{
		Type:      "switch_removed",
		SwitchID:  switchID,
		Data:      switch_,
		Timestamp: time.Now(),
	})
	
	log.Printf("Removed switch %s", switchID)
	return nil
}

// AddPolicyRule adds a high-level policy rule
func (c *SDNController) AddPolicyRule(rule *PolicyRule) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if rule.ID == "" {
		rule.ID = uuid.New().String()
	}
	
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	
	c.PolicyRules[rule.ID] = rule
	c.metrics.PolicyRulesActive++
	
	// Compile policy rule to flow entries
	if err := c.policyCompiler.CompileRule(rule); err != nil {
		return fmt.Errorf("failed to compile policy rule: %w", err)
	}
	
	c.emitEvent(ControllerEvent{
		Type:      "policy_rule_added",
		Data:      rule,
		Timestamp: time.Now(),
	})
	
	log.Printf("Added policy rule %s: %s", rule.ID, rule.Name)
	return nil
}

// RemovePolicyRule removes a policy rule
func (c *SDNController) RemovePolicyRule(ruleID string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	rule, exists := c.PolicyRules[ruleID]
	if !exists {
		return fmt.Errorf("policy rule %s not found", ruleID)
	}
	
	// Remove compiled flow entries
	if err := c.policyCompiler.RemoveRule(rule); err != nil {
		log.Printf("Warning: failed to remove flow entries for rule %s: %v", ruleID, err)
	}
	
	delete(c.PolicyRules, ruleID)
	c.metrics.PolicyRulesActive--
	
	c.emitEvent(ControllerEvent{
		Type:      "policy_rule_removed",
		Data:      rule,
		Timestamp: time.Now(),
	})
	
	log.Printf("Removed policy rule %s", ruleID)
	return nil
}

// InstallFlowEntry installs a flow entry on a switch
func (c *SDNController) InstallFlowEntry(switchID string, flowEntry FlowEntry) error {
	c.mutex.RLock()
	switch_, exists := c.Switches[switchID]
	c.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("switch %s not found", switchID)
	}
	
	if switch_.ConnectionState != "connected" {
		return fmt.Errorf("switch %s is not connected", switchID)
	}
	
	// Add flow entry to switch's flow table
	table, exists := switch_.FlowTables[flowEntry.TableID]
	if !exists {
		return fmt.Errorf("flow table %d not found on switch %s", flowEntry.TableID, switchID)
	}
	
	if flowEntry.ID == "" {
		flowEntry.ID = uuid.New().String()
	}
	flowEntry.CreatedAt = time.Now()
	flowEntry.UpdatedAt = time.Now()
	
	table.Flows = append(table.Flows, flowEntry)
	table.ActiveCount++
	c.metrics.FlowRulesInstalled++
	
	// Send OpenFlow flow mod message to switch
	if err := c.sendFlowMod(switchID, flowEntry, "add"); err != nil {
		return fmt.Errorf("failed to send flow mod to switch: %w", err)
	}
	
	c.emitEvent(ControllerEvent{
		Type:      "flow_entry_installed",
		SwitchID:  switchID,
		Data:      flowEntry,
		Timestamp: time.Now(),
	})
	
	return nil
}

// RemoveFlowEntry removes a flow entry from a switch
func (c *SDNController) RemoveFlowEntry(switchID, flowID string) error {
	c.mutex.RLock()
	switch_, exists := c.Switches[switchID]
	c.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("switch %s not found", switchID)
	}
	
	if switch_.ConnectionState != "connected" {
		return fmt.Errorf("switch %s is not connected", switchID)
	}
	
	// Find and remove flow entry
	var removedFlow FlowEntry
	var found bool
	
	for tableID, table := range switch_.FlowTables {
		for i, flow := range table.Flows {
			if flow.ID == flowID {
				removedFlow = flow
				table.Flows = append(table.Flows[:i], table.Flows[i+1:]...)
				table.ActiveCount--
				found = true
				break
			}
		}
		if found {
			break
		}
		_ = tableID // Use tableID to avoid unused variable warning
	}
	
	if !found {
		return fmt.Errorf("flow entry %s not found on switch %s", flowID, switchID)
	}
	
	// Send OpenFlow flow mod message to switch
	if err := c.sendFlowMod(switchID, removedFlow, "delete"); err != nil {
		return fmt.Errorf("failed to send flow mod to switch: %w", err)
	}
	
	c.emitEvent(ControllerEvent{
		Type:      "flow_entry_removed",
		SwitchID:  switchID,
		Data:      removedFlow,
		Timestamp: time.Now(),
	})
	
	return nil
}

// GetSwitch returns information about a switch
func (c *SDNController) GetSwitch(switchID string) (*Switch, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	switch_, exists := c.Switches[switchID]
	if !exists {
		return nil, fmt.Errorf("switch %s not found", switchID)
	}
	
	return switch_, nil
}

// ListSwitches returns all switches
func (c *SDNController) ListSwitches() []*Switch {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	switches := make([]*Switch, 0, len(c.Switches))
	for _, switch_ := range c.Switches {
		switches = append(switches, switch_)
	}
	
	return switches
}

// GetMetrics returns controller metrics
func (c *SDNController) GetMetrics() *ControllerMetrics {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	// Create a copy to avoid race conditions
	metrics := *c.metrics
	return &metrics
}

// AddEventListener adds an event listener
func (c *SDNController) AddEventListener(listener EventListener) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	c.EventListeners = append(c.EventListeners, listener)
}

// Private methods

func (c *SDNController) initializeFlowTables(switch_ *Switch) {
	// Initialize standard flow tables
	tables := []struct {
		id   FlowTableID
		name string
	}{
		{TableClassification, "Classification"},
		{TableAcl, "ACL"},
		{TableQoS, "QoS"},
		{TableForwarding, "Forwarding"},
		{TableEgress, "Egress"},
	}
	
	for _, table := range tables {
		switch_.FlowTables[table.id] = &FlowTable{
			TableID:      table.id,
			Name:         table.name,
			MaxEntries:   c.config.FlowTableSize,
			ActiveCount:  0,
			LookupCount:  0,
			MatchedCount: 0,
			Flows:        make([]FlowEntry, 0),
		}
	}
}

func (c *SDNController) acceptConnections() {
	defer c.wg.Done()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
			conn, err := c.listener.Accept()
			if err != nil {
				if c.ctx.Err() != nil {
					return // Context cancelled
				}
				log.Printf("Failed to accept connection: %v", err)
				continue
			}
			
			// Handle connection in a new goroutine
			c.wg.Add(1)
			go c.handleConnection(conn)
		}
	}
}

func (c *SDNController) handleConnection(conn net.Conn) {
	defer c.wg.Done()
	defer conn.Close()
	
	remoteAddr := conn.RemoteAddr().String()
	log.Printf("New connection from %s", remoteAddr)
	
	// Perform OpenFlow handshake
	switchID, err := c.performHandshake(conn)
	if err != nil {
		log.Printf("Handshake failed with %s: %v", remoteAddr, err)
		return
	}
	
	// Store connection
	c.connectionsMutex.Lock()
	c.connections[switchID] = conn
	c.connectionsMutex.Unlock()
	
	// Update switch state
	c.mutex.Lock()
	if switch_, exists := c.Switches[switchID]; exists {
		switch_.ConnectionState = "connected"
		switch_.IPAddress = conn.RemoteAddr().(*net.TCPAddr).IP.String()
		switch_.Port = conn.RemoteAddr().(*net.TCPAddr).Port
		switch_.LastHeartbeat = time.Now()
		switch_.UpdatedAt = time.Now()
	}
	c.mutex.Unlock()
	
	c.emitEvent(ControllerEvent{
		Type:      "switch_connected",
		SwitchID:  switchID,
		Data:      map[string]interface{}{"remote_addr": remoteAddr},
		Timestamp: time.Now(),
	})
	
	// Handle messages
	c.handleMessages(conn, switchID)
	
	// Cleanup on disconnect
	c.connectionsMutex.Lock()
	delete(c.connections, switchID)
	c.connectionsMutex.Unlock()
	
	c.mutex.Lock()
	if switch_, exists := c.Switches[switchID]; exists {
		switch_.ConnectionState = "disconnected"
		switch_.UpdatedAt = time.Now()
	}
	c.mutex.Unlock()
	
	c.emitEvent(ControllerEvent{
		Type:      "switch_disconnected",
		SwitchID:  switchID,
		Data:      map[string]interface{}{"remote_addr": remoteAddr},
		Timestamp: time.Now(),
	})
	
	log.Printf("Connection closed with switch %s (%s)", switchID, remoteAddr)
}

func (c *SDNController) performHandshake(conn net.Conn) (string, error) {
	// Simplified handshake - in a real implementation, this would involve
	// proper OpenFlow protocol negotiation
	
	// For now, we'll simulate the handshake and generate a switch ID
	// based on the connection details
	remoteAddr := conn.RemoteAddr().String()
	dpid := c.generateDPIDFromAddr(remoteAddr)
	switchID := fmt.Sprintf("switch-%016x", dpid)
	
	log.Printf("Handshake completed with switch %s (DPID: %016x)", switchID, dpid)
	return switchID, nil
}

func (c *SDNController) generateDPIDFromAddr(addr string) uint64 {
	// Generate a deterministic DPID from the address
	hash := uint64(0)
	for _, b := range []byte(addr) {
		hash = hash*31 + uint64(b)
	}
	return hash
}

func (c *SDNController) handleMessages(conn net.Conn, switchID string) {
	buffer := make([]byte, 1024)
	
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
			conn.SetReadDeadline(time.Now().Add(30 * time.Second))
			n, err := conn.Read(buffer)
			if err != nil {
				if c.ctx.Err() != nil {
					return // Context cancelled
				}
				log.Printf("Failed to read from switch %s: %v", switchID, err)
				return
			}
			
			if n > 0 {
				// Process OpenFlow message
				c.processMessage(switchID, buffer[:n])
				
				// Update heartbeat
				c.mutex.Lock()
				if switch_, exists := c.Switches[switchID]; exists {
					switch_.LastHeartbeat = time.Now()
					switch_.UpdatedAt = time.Now()
				}
				c.mutex.Unlock()
			}
		}
	}
}

func (c *SDNController) processMessage(switchID string, data []byte) {
	// Simplified message processing - in a real implementation,
	// this would parse OpenFlow messages and handle different message types
	
	if len(data) < 8 {
		return // Invalid OpenFlow message
	}
	
	// Extract basic OpenFlow header fields
	version := data[0]
	msgType := data[1]
	length := binary.BigEndian.Uint16(data[2:4])
	xid := binary.BigEndian.Uint32(data[4:8])
	
	log.Printf("Received OpenFlow message from switch %s: version=%d, type=%d, length=%d, xid=%d",
		switchID, version, msgType, length, xid)
	
	// Update metrics
	c.metrics.PacketsProcessed++
	c.metrics.BytesProcessed += int64(len(data))
	
	c.emitEvent(ControllerEvent{
		Type:      "message_received",
		SwitchID:  switchID,
		Data: map[string]interface{}{
			"version": version,
			"type":    msgType,
			"length":  length,
			"xid":     xid,
		},
		Timestamp: time.Now(),
	})
}

func (c *SDNController) sendFlowMod(switchID string, flowEntry FlowEntry, command string) error {
	// Simplified flow mod sending - in a real implementation,
	// this would construct proper OpenFlow FLOW_MOD messages
	
	c.connectionsMutex.RLock()
	conn, exists := c.connections[switchID]
	c.connectionsMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("no connection to switch %s", switchID)
	}
	
	// Simulate sending OpenFlow FLOW_MOD message
	message := fmt.Sprintf("FLOW_MOD: %s flow %s on table %d", command, flowEntry.ID, flowEntry.TableID)
	_, err := conn.Write([]byte(message))
	if err != nil {
		return fmt.Errorf("failed to send flow mod: %w", err)
	}
	
	log.Printf("Sent flow mod to switch %s: %s", switchID, message)
	return nil
}

func (c *SDNController) collectStatistics() {
	defer c.wg.Done()
	
	ticker := time.NewTicker(c.statsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.updateStatistics()
		}
	}
}

func (c *SDNController) updateStatistics() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	// Update controller metrics
	c.metrics.LastStatsUpdate = time.Now()
	c.metrics.SwitchesConnected = int64(len(c.Switches))
	c.metrics.PolicyRulesActive = int64(len(c.PolicyRules))
	
	// Calculate total flow rules
	totalFlowRules := int64(0)
	for _, switch_ := range c.Switches {
		for _, table := range switch_.FlowTables {
			totalFlowRules += int64(table.ActiveCount)
		}
	}
	c.metrics.FlowRulesInstalled = totalFlowRules
	
	// Collect statistics from switches
	for switchID := range c.Switches {
		c.collectSwitchStatistics(switchID)
	}
}

func (c *SDNController) collectSwitchStatistics(switchID string) {
	// Simplified statistics collection - in a real implementation,
	// this would send STATS_REQUEST messages to switches
	
	switch_, exists := c.Switches[switchID]
	if !exists || switch_.ConnectionState != "connected" {
		return
	}
	
	// Update switch statistics (simulated)
	for portNo := range switch_.PortStats {
		stats := switch_.PortStats[portNo]
		stats.RxPackets += uint64(100 + portNo*10) // Simulate traffic
		stats.TxPackets += uint64(90 + portNo*8)
		stats.RxBytes += uint64(1500 * (100 + portNo*10))
		stats.TxBytes += uint64(1500 * (90 + portNo*8))
		switch_.PortStats[portNo] = stats
	}
	
	switch_.UpdatedAt = time.Now()
}

func (c *SDNController) emitEvent(event ControllerEvent) {
	c.mutex.RLock()
	listeners := make([]EventListener, len(c.EventListeners))
	copy(listeners, c.EventListeners)
	c.mutex.RUnlock()
	
	for _, listener := range listeners {
		go func(l EventListener, e ControllerEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Event listener panic: %v", r)
				}
			}()
			l(e)
		}(listener, event)
	}
	
	c.metrics.EventsGenerated++
	
	if c.config.EnableEventLogging {
		log.Printf("Event: %s", event.Type)
	}
}