package firewall

import (
	"context"
	"fmt"
	"log"
	"net"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ConnectionState represents the state of a network connection
type ConnectionState string

const (
	StateNew        ConnectionState = "NEW"
	StateEstablished ConnectionState = "ESTABLISHED"
	StateRelated    ConnectionState = "RELATED"
	StateInvalid    ConnectionState = "INVALID"
	StateClosing    ConnectionState = "CLOSING"
	StateClosed     ConnectionState = "CLOSED"
	StateTimeWait   ConnectionState = "TIME_WAIT"
)

// FirewallAction represents the action to take on a packet
type FirewallAction string

const (
	ActionAccept   FirewallAction = "ACCEPT"
	ActionDrop     FirewallAction = "DROP"
	ActionReject   FirewallAction = "REJECT"
	ActionLog      FirewallAction = "LOG"
	ActionQueue    FirewallAction = "QUEUE"
	ActionRedirect FirewallAction = "REDIRECT"
	ActionMark     FirewallAction = "MARK"
	ActionLimit    FirewallAction = "LIMIT"
)

// ProtocolType represents network protocol types
type ProtocolType string

const (
	ProtocolTCP    ProtocolType = "tcp"
	ProtocolUDP    ProtocolType = "udp"
	ProtocolICMP   ProtocolType = "icmp"
	ProtocolICMPv6 ProtocolType = "icmpv6"
	ProtocolAH     ProtocolType = "ah"
	ProtocolESP    ProtocolType = "esp"
	ProtocolGRE    ProtocolType = "gre"
	ProtocolSCTP   ProtocolType = "sctp"
	ProtocolAny    ProtocolType = "any"
)

// ApplicationProtocol represents Layer 7 application protocols
type ApplicationProtocol string

const (
	AppHTTP      ApplicationProtocol = "http"
	AppHTTPS     ApplicationProtocol = "https"
	AppSSH       ApplicationProtocol = "ssh"
	AppFTP       ApplicationProtocol = "ftp"
	AppSMTP      ApplicationProtocol = "smtp"
	AppDNS       ApplicationProtocol = "dns"
	AppDHCP      ApplicationProtocol = "dhcp"
	AppSNMP      ApplicationProtocol = "snmp"
	AppMySQL     ApplicationProtocol = "mysql"
	AppPostgres  ApplicationProtocol = "postgres"
	AppRedis     ApplicationProtocol = "redis"
	AppMongoDB   ApplicationProtocol = "mongodb"
	AppUnknown   ApplicationProtocol = "unknown"
)

// ThreatLevel represents the severity of detected threats
type ThreatLevel string

const (
	ThreatLow      ThreatLevel = "low"
	ThreatMedium   ThreatLevel = "medium"
	ThreatHigh     ThreatLevel = "high"
	ThreatCritical ThreatLevel = "critical"
)

// Packet represents a network packet with metadata
type Packet struct {
	ID            string              `json:"id"`
	Timestamp     time.Time           `json:"timestamp"`
	SrcIP         net.IP              `json:"src_ip"`
	DstIP         net.IP              `json:"dst_ip"`
	SrcPort       uint16              `json:"src_port"`
	DstPort       uint16              `json:"dst_port"`
	Protocol      ProtocolType        `json:"protocol"`
	Length        int                 `json:"length"`
	TTL           uint8               `json:"ttl"`
	Flags         []string            `json:"flags"`
	Payload       []byte              `json:"payload,omitempty"`
	Headers       map[string]string   `json:"headers"`
	AppProtocol   ApplicationProtocol `json:"app_protocol"`
	TenantID      string              `json:"tenant_id"`
	NetworkID     string              `json:"network_id"`
	InterfaceIn   string              `json:"interface_in"`
	InterfaceOut  string              `json:"interface_out"`
	ConnectionID  string              `json:"connection_id"`
	State         ConnectionState     `json:"state"`
}

// Connection represents a network connection being tracked
type Connection struct {
	ID                string          `json:"id"`
	SrcIP             net.IP          `json:"src_ip"`
	DstIP             net.IP          `json:"dst_ip"`
	SrcPort           uint16          `json:"src_port"`
	DstPort           uint16          `json:"dst_port"`
	Protocol          ProtocolType    `json:"protocol"`
	State             ConnectionState `json:"state"`
	StartTime         time.Time       `json:"start_time"`
	LastActivity      time.Time       `json:"last_activity"`
	BytesSent         uint64          `json:"bytes_sent"`
	BytesReceived     uint64          `json:"bytes_received"`
	PacketsSent       uint64          `json:"packets_sent"`
	PacketsReceived   uint64          `json:"packets_received"`
	TenantID          string          `json:"tenant_id"`
	NetworkID         string          `json:"network_id"`
	ApplicationProtocol ApplicationProtocol `json:"app_protocol"`
	ThreatLevel       ThreatLevel     `json:"threat_level"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// FirewallRule represents a firewall rule with match criteria and actions
type FirewallRule struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	TenantID          string                 `json:"tenant_id"`
	Priority          int                    `json:"priority"`
	Enabled           bool                   `json:"enabled"`
	Direction         string                 `json:"direction"` // ingress, egress, both
	SrcIP             *net.IPNet             `json:"src_ip,omitempty"`
	DstIP             *net.IPNet             `json:"dst_ip,omitempty"`
	SrcPorts          []PortRange            `json:"src_ports,omitempty"`
	DstPorts          []PortRange            `json:"dst_ports,omitempty"`
	Protocols         []ProtocolType         `json:"protocols,omitempty"`
	AppProtocols      []ApplicationProtocol  `json:"app_protocols,omitempty"`
	ConnectionStates  []ConnectionState      `json:"connection_states,omitempty"`
	TimeRange         *TimeRange             `json:"time_range,omitempty"`
	RateLimit         *RateLimit             `json:"rate_limit,omitempty"`
	GeoLocation       *GeoMatch              `json:"geo_location,omitempty"`
	ThreatIntel       bool                   `json:"threat_intel"`
	DPIRules          []DPIRule              `json:"dpi_rules,omitempty"`
	Action            FirewallAction         `json:"action"`
	ActionParams      map[string]interface{} `json:"action_params,omitempty"`
	LogLevel          string                 `json:"log_level,omitempty"`
	Description       string                 `json:"description"`
	Tags              []string               `json:"tags,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	CreatedBy         string                 `json:"created_by"`
	HitCount          uint64                 `json:"hit_count"`
	LastHit           time.Time              `json:"last_hit"`
}

// PortRange represents a range of ports
type PortRange struct {
	Start uint16 `json:"start"`
	End   uint16 `json:"end"`
}

// TimeRange represents a time-based match condition
type TimeRange struct {
	StartTime string   `json:"start_time"` // HH:MM format
	EndTime   string   `json:"end_time"`   // HH:MM format
	Days      []string `json:"days"`       // mon, tue, wed, thu, fri, sat, sun
	Timezone  string   `json:"timezone"`
}

// RateLimit represents rate limiting configuration
type RateLimit struct {
	Rate     uint64        `json:"rate"`     // packets per interval
	Interval time.Duration `json:"interval"` // time interval
	Burst    uint64        `json:"burst"`    // burst allowance
}

// GeoMatch represents geographic matching
type GeoMatch struct {
	Countries []string `json:"countries"`
	Regions   []string `json:"regions,omitempty"`
	Cities    []string `json:"cities,omitempty"`
	Exclude   bool     `json:"exclude"` // true to exclude these locations
}

// DPIRule represents a Deep Packet Inspection rule
type DPIRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Pattern     string            `json:"pattern"`     // regex pattern
	Offset      int               `json:"offset"`      // byte offset in payload
	Length      int               `json:"length"`      // number of bytes to examine
	CaseSensitive bool            `json:"case_sensitive"`
	Protocol    ApplicationProtocol `json:"protocol"`
	Direction   string            `json:"direction"`   // request, response, both
	Action      FirewallAction    `json:"action"`
	ThreatLevel ThreatLevel       `json:"threat_level"`
}

// ThreatSignature represents a threat detection signature
type ThreatSignature struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Pattern     string      `json:"pattern"`
	Level       ThreatLevel `json:"level"`
	Category    string      `json:"category"`
	Description string      `json:"description"`
	CVE         string      `json:"cve,omitempty"`
	References  []string    `json:"references,omitempty"`
}

// ConnectionTracker tracks network connections and their state
type ConnectionTracker struct {
	connections     map[string]*Connection
	connectionsByIP map[string][]string // IP -> connection IDs
	mutex           sync.RWMutex
	gcInterval      time.Duration
	tcpTimeout      time.Duration
	udpTimeout      time.Duration
	icmpTimeout     time.Duration
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
}

// DPIEngine performs deep packet inspection
type DPIEngine struct {
	rules           map[string]*DPIRule
	signatures      map[string]*ThreatSignature
	patterns        map[string]*regexp.Regexp
	appDetectors    map[ApplicationProtocol]func([]byte) bool
	mutex           sync.RWMutex
}

// RateLimiter implements rate limiting functionality
type RateLimiter struct {
	limits     map[string]*TokenBucket // IP -> TokenBucket
	mutex      sync.RWMutex
	gcInterval time.Duration
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// TokenBucket implements token bucket rate limiting
type TokenBucket struct {
	capacity     uint64
	tokens       uint64
	refillRate   uint64
	lastRefill   time.Time
	mutex        sync.Mutex
}

// MicrosegmentationFirewall implements comprehensive network segmentation
type MicrosegmentationFirewall struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	TenantID          string                 `json:"tenant_id"`
	Rules             map[string]*FirewallRule `json:"rules"`
	DefaultAction     FirewallAction         `json:"default_action"`
	ConnectionTracker *ConnectionTracker     `json:"-"`
	DPIEngine         *DPIEngine             `json:"-"`
	RateLimiter       *RateLimiter           `json:"-"`
	ThreatIntel       *ThreatIntelligence    `json:"-"`
	PacketProcessor   *PacketProcessor       `json:"-"`
	Metrics           *FirewallMetrics       `json:"metrics"`
	Config            *FirewallConfig        `json:"config"`
	EventListeners    []FirewallEventListener `json:"-"`
	mutex             sync.RWMutex
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
}

// ThreatIntelligence provides threat intelligence capabilities
type ThreatIntelligence struct {
	blacklistedIPs    map[string]ThreatLevel
	maliciousDomains  map[string]ThreatLevel
	signatures        map[string]*ThreatSignature
	geoIPDB           map[string]string // IP -> Country
	mutex             sync.RWMutex
	updateInterval    time.Duration
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
}

// PacketProcessor handles packet processing and filtering
type PacketProcessor struct {
	firewall    *MicrosegmentationFirewall
	workers     int
	packetQueue chan *Packet
	resultQueue chan *PacketResult
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

// PacketResult represents the result of packet processing
type PacketResult struct {
	Packet    *Packet        `json:"packet"`
	Action    FirewallAction `json:"action"`
	Rule      *FirewallRule  `json:"rule,omitempty"`
	Reason    string         `json:"reason"`
	Threat    *ThreatInfo    `json:"threat,omitempty"`
	Timestamp time.Time      `json:"timestamp"`
}

// ThreatInfo contains information about detected threats
type ThreatInfo struct {
	Level       ThreatLevel `json:"level"`
	Category    string      `json:"category"`
	Description string      `json:"description"`
	Signature   string      `json:"signature,omitempty"`
	IOCs        []string    `json:"iocs,omitempty"` // Indicators of Compromise
}

// FirewallMetrics tracks firewall performance and activity
type FirewallMetrics struct {
	PacketsProcessed    uint64            `json:"packets_processed"`
	PacketsAccepted     uint64            `json:"packets_accepted"`
	PacketsDropped      uint64            `json:"packets_dropped"`
	PacketsRejected     uint64            `json:"packets_rejected"`
	ConnectionsTracked  uint64            `json:"connections_tracked"`
	ConnectionsActive   uint64            `json:"connections_active"`
	ThreatsDetected     uint64            `json:"threats_detected"`
	ThreatsBlocked      uint64            `json:"threats_blocked"`
	RateLimitHits       uint64            `json:"rate_limit_hits"`
	DPIInspections      uint64            `json:"dpi_inspections"`
	RuleHits            map[string]uint64 `json:"rule_hits"`
	ThreatsByLevel      map[ThreatLevel]uint64 `json:"threats_by_level"`
	ProcessingLatency   time.Duration     `json:"processing_latency"`
	LastUpdated         time.Time         `json:"last_updated"`
}

// FirewallConfig holds firewall configuration
type FirewallConfig struct {
	DefaultAction         FirewallAction `json:"default_action"`
	EnableConnectionTracking bool        `json:"enable_connection_tracking"`
	EnableDPI             bool           `json:"enable_dpi"`
	EnableThreatIntel     bool           `json:"enable_threat_intel"`
	EnableRateLimit       bool           `json:"enable_rate_limit"`
	EnableGeoBlocking     bool           `json:"enable_geo_blocking"`
	EnableLogging         bool           `json:"enable_logging"`
	LogLevel              string         `json:"log_level"`
	MaxConnections        uint64         `json:"max_connections"`
	ConnectionTimeout     time.Duration  `json:"connection_timeout"`
	PacketBufferSize      int            `json:"packet_buffer_size"`
	WorkerCount           int            `json:"worker_count"`
	MetricsInterval       time.Duration  `json:"metrics_interval"`
}

// FirewallEvent represents firewall events
type FirewallEvent struct {
	Type        string         `json:"type"`
	TenantID    string         `json:"tenant_id"`
	FirewallID  string         `json:"firewall_id"`
	RuleID      string         `json:"rule_id,omitempty"`
	PacketID    string         `json:"packet_id,omitempty"`
	Action      FirewallAction `json:"action"`
	Data        interface{}    `json:"data"`
	Timestamp   time.Time      `json:"timestamp"`
}

// FirewallEventListener is a callback for firewall events
type FirewallEventListener func(event FirewallEvent)

// NewMicrosegmentationFirewall creates a new microsegmentation firewall
func NewMicrosegmentationFirewall(tenantID, name string, config *FirewallConfig) *MicrosegmentationFirewall {
	ctx, cancel := context.WithCancel(context.Background())
	
	fw := &MicrosegmentationFirewall{
		ID:            uuid.New().String(),
		Name:          name,
		TenantID:      tenantID,
		Rules:         make(map[string]*FirewallRule),
		DefaultAction: config.DefaultAction,
		Config:        config,
		EventListeners: make([]FirewallEventListener, 0),
		ctx:           ctx,
		cancel:        cancel,
		Metrics: &FirewallMetrics{
			RuleHits:        make(map[string]uint64),
			ThreatsByLevel:  make(map[ThreatLevel]uint64),
			LastUpdated:     time.Now(),
		},
	}
	
	// Initialize components
	if config.EnableConnectionTracking {
		fw.ConnectionTracker = NewConnectionTracker()
	}
	
	if config.EnableDPI {
		fw.DPIEngine = NewDPIEngine()
	}
	
	if config.EnableRateLimit {
		fw.RateLimiter = NewRateLimiter()
	}
	
	if config.EnableThreatIntel {
		fw.ThreatIntel = NewThreatIntelligence()
	}
	
	fw.PacketProcessor = NewPacketProcessor(fw, config.WorkerCount)
	
	return fw
}

// Start starts the microsegmentation firewall
func (fw *MicrosegmentationFirewall) Start() error {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	log.Printf("Starting microsegmentation firewall %s for tenant %s", fw.Name, fw.TenantID)
	
	// Start components
	if fw.ConnectionTracker != nil {
		if err := fw.ConnectionTracker.Start(); err != nil {
			return fmt.Errorf("failed to start connection tracker: %w", err)
		}
	}
	
	if fw.DPIEngine != nil {
		if err := fw.DPIEngine.Start(); err != nil {
			return fmt.Errorf("failed to start DPI engine: %w", err)
		}
	}
	
	if fw.RateLimiter != nil {
		if err := fw.RateLimiter.Start(); err != nil {
			return fmt.Errorf("failed to start rate limiter: %w", err)
		}
	}
	
	if fw.ThreatIntel != nil {
		if err := fw.ThreatIntel.Start(); err != nil {
			return fmt.Errorf("failed to start threat intelligence: %w", err)
		}
	}
	
	if err := fw.PacketProcessor.Start(); err != nil {
		return fmt.Errorf("failed to start packet processor: %w", err)
	}
	
	// Start metrics collection
	if fw.Config.MetricsInterval > 0 {
		fw.wg.Add(1)
		go fw.collectMetrics()
	}
	
	fw.emitEvent(FirewallEvent{
		Type:       "firewall_started",
		TenantID:   fw.TenantID,
		FirewallID: fw.ID,
		Timestamp:  time.Now(),
	})
	
	log.Printf("Microsegmentation firewall %s started successfully", fw.Name)
	return nil
}

// Stop stops the microsegmentation firewall
func (fw *MicrosegmentationFirewall) Stop() error {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	log.Printf("Stopping microsegmentation firewall %s", fw.Name)
	
	fw.cancel()
	fw.wg.Wait()
	
	// Stop components
	if fw.PacketProcessor != nil {
		fw.PacketProcessor.Stop()
	}
	
	if fw.ThreatIntel != nil {
		fw.ThreatIntel.Stop()
	}
	
	if fw.RateLimiter != nil {
		fw.RateLimiter.Stop()
	}
	
	if fw.ConnectionTracker != nil {
		fw.ConnectionTracker.Stop()
	}
	
	fw.emitEvent(FirewallEvent{
		Type:       "firewall_stopped",
		TenantID:   fw.TenantID,
		FirewallID: fw.ID,
		Timestamp:  time.Now(),
	})
	
	log.Printf("Microsegmentation firewall %s stopped", fw.Name)
	return nil
}

// ProcessPacket processes a network packet through the firewall
func (fw *MicrosegmentationFirewall) ProcessPacket(packet *Packet) (*PacketResult, error) {
	startTime := time.Now()
	
	// Update metrics
	fw.Metrics.PacketsProcessed++
	
	// Check if packet processing is enabled
	if fw.PacketProcessor == nil {
		return &PacketResult{
			Packet:    packet,
			Action:    fw.DefaultAction,
			Reason:    "packet processor not available",
			Timestamp: time.Now(),
		}, nil
	}
	
	// Process packet through the pipeline
	result := fw.processPacketInternal(packet)
	
	// Update processing latency
	fw.Metrics.ProcessingLatency = time.Since(startTime)
	
	// Update action metrics
	switch result.Action {
	case ActionAccept:
		fw.Metrics.PacketsAccepted++
	case ActionDrop:
		fw.Metrics.PacketsDropped++
	case ActionReject:
		fw.Metrics.PacketsRejected++
	}
	
	// Update rule hit count
	if result.Rule != nil {
		fw.mutex.Lock()
		fw.Metrics.RuleHits[result.Rule.ID]++
		result.Rule.HitCount++
		result.Rule.LastHit = time.Now()
		fw.mutex.Unlock()
	}
	
	// Emit event for significant actions
	if result.Action != ActionAccept || result.Threat != nil {
		fw.emitEvent(FirewallEvent{
			Type:       "packet_processed",
			TenantID:   fw.TenantID,
			FirewallID: fw.ID,
			RuleID:     func() string { if result.Rule != nil { return result.Rule.ID }; return "" }(),
			PacketID:   packet.ID,
			Action:     result.Action,
			Data:       result,
			Timestamp:  time.Now(),
		})
	}
	
	return result, nil
}

// AddRule adds a firewall rule
func (fw *MicrosegmentationFirewall) AddRule(rule *FirewallRule) error {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	if rule.ID == "" {
		rule.ID = uuid.New().String()
	}
	
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	
	fw.Rules[rule.ID] = rule
	
	fw.emitEvent(FirewallEvent{
		Type:       "rule_added",
		TenantID:   fw.TenantID,
		FirewallID: fw.ID,
		RuleID:     rule.ID,
		Data:       rule,
		Timestamp:  time.Now(),
	})
	
	log.Printf("Added firewall rule %s: %s", rule.ID, rule.Name)
	return nil
}

// RemoveRule removes a firewall rule
func (fw *MicrosegmentationFirewall) RemoveRule(ruleID string) error {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	rule, exists := fw.Rules[ruleID]
	if !exists {
		return fmt.Errorf("rule %s not found", ruleID)
	}
	
	delete(fw.Rules, ruleID)
	delete(fw.Metrics.RuleHits, ruleID)
	
	fw.emitEvent(FirewallEvent{
		Type:       "rule_removed",
		TenantID:   fw.TenantID,
		FirewallID: fw.ID,
		RuleID:     ruleID,
		Data:       rule,
		Timestamp:  time.Now(),
	})
	
	log.Printf("Removed firewall rule %s", ruleID)
	return nil
}

// GetRule retrieves a firewall rule
func (fw *MicrosegmentationFirewall) GetRule(ruleID string) (*FirewallRule, error) {
	fw.mutex.RLock()
	defer fw.mutex.RUnlock()
	
	rule, exists := fw.Rules[ruleID]
	if !exists {
		return nil, fmt.Errorf("rule %s not found", ruleID)
	}
	
	return rule, nil
}

// ListRules returns all firewall rules
func (fw *MicrosegmentationFirewall) ListRules() []*FirewallRule {
	fw.mutex.RLock()
	defer fw.mutex.RUnlock()
	
	rules := make([]*FirewallRule, 0, len(fw.Rules))
	for _, rule := range fw.Rules {
		rules = append(rules, rule)
	}
	
	return rules
}

// processPacketInternal processes a packet through the firewall pipeline
func (fw *MicrosegmentationFirewall) processPacketInternal(packet *Packet) *PacketResult {
	// 1. Connection tracking
	if fw.ConnectionTracker != nil {
		conn := fw.ConnectionTracker.TrackPacket(packet)
		if conn != nil {
			packet.ConnectionID = conn.ID
			packet.State = conn.State
		}
	}
	
	// 2. Rate limiting
	if fw.RateLimiter != nil {
		if !fw.RateLimiter.Allow(packet.SrcIP.String()) {
			fw.Metrics.RateLimitHits++
			return &PacketResult{
				Packet:    packet,
				Action:    ActionDrop,
				Reason:    "rate limit exceeded",
				Timestamp: time.Now(),
			}
		}
	}
	
	// 3. Threat intelligence
	if fw.ThreatIntel != nil {
		if threat := fw.ThreatIntel.CheckThreat(packet); threat != nil {
			fw.Metrics.ThreatsDetected++
			fw.Metrics.ThreatsBlocked++
			fw.Metrics.ThreatsByLevel[threat.Level]++
			
			return &PacketResult{
				Packet:    packet,
				Action:    ActionDrop,
				Reason:    "threat detected",
				Threat:    threat,
				Timestamp: time.Now(),
			}
		}
	}
	
	// 4. Rule matching
	matchedRule := fw.findMatchingRule(packet)
	if matchedRule != nil {
		// 5. Deep packet inspection (if required by rule)
		if fw.DPIEngine != nil && len(matchedRule.DPIRules) > 0 {
			fw.Metrics.DPIInspections++
			if threat := fw.DPIEngine.InspectPacket(packet, matchedRule.DPIRules); threat != nil {
				fw.Metrics.ThreatsDetected++
				fw.Metrics.ThreatsBlocked++
				fw.Metrics.ThreatsByLevel[threat.Level]++
				
				return &PacketResult{
					Packet:    packet,
					Action:    ActionDrop,
					Rule:      matchedRule,
					Reason:    "DPI threat detected",
					Threat:    threat,
					Timestamp: time.Now(),
				}
			}
		}
		
		return &PacketResult{
			Packet:    packet,
			Action:    matchedRule.Action,
			Rule:      matchedRule,
			Reason:    fmt.Sprintf("matched rule: %s", matchedRule.Name),
			Timestamp: time.Now(),
		}
	}
	
	// 6. Default action
	return &PacketResult{
		Packet:    packet,
		Action:    fw.DefaultAction,
		Reason:    "no matching rule",
		Timestamp: time.Now(),
	}
}

// findMatchingRule finds the first matching firewall rule for a packet
func (fw *MicrosegmentationFirewall) findMatchingRule(packet *Packet) *FirewallRule {
	fw.mutex.RLock()
	defer fw.mutex.RUnlock()
	
	// Sort rules by priority (higher priority first)
	rules := make([]*FirewallRule, 0, len(fw.Rules))
	for _, rule := range fw.Rules {
		if rule.Enabled {
			rules = append(rules, rule)
		}
	}
	
	// Simple sort by priority (in a real implementation, use sort.Slice)
	for i := 0; i < len(rules)-1; i++ {
		for j := i + 1; j < len(rules); j++ {
			if rules[i].Priority < rules[j].Priority {
				rules[i], rules[j] = rules[j], rules[i]
			}
		}
	}
	
	// Check each rule for a match
	for _, rule := range rules {
		if fw.ruleMatches(rule, packet) {
			return rule
		}
	}
	
	return nil
}

// ruleMatches checks if a firewall rule matches a packet
func (fw *MicrosegmentationFirewall) ruleMatches(rule *FirewallRule, packet *Packet) bool {
	// Check IP addresses
	if rule.SrcIP != nil && !rule.SrcIP.Contains(packet.SrcIP) {
		return false
	}
	
	if rule.DstIP != nil && !rule.DstIP.Contains(packet.DstIP) {
		return false
	}
	
	// Check protocols
	if len(rule.Protocols) > 0 {
		protocolMatch := false
		for _, proto := range rule.Protocols {
			if proto == ProtocolAny || proto == packet.Protocol {
				protocolMatch = true
				break
			}
		}
		if !protocolMatch {
			return false
		}
	}
	
	// Check application protocols
	if len(rule.AppProtocols) > 0 {
		appMatch := false
		for _, appProto := range rule.AppProtocols {
			if appProto == packet.AppProtocol {
				appMatch = true
				break
			}
		}
		if !appMatch {
			return false
		}
	}
	
	// Check ports
	if len(rule.SrcPorts) > 0 {
		portMatch := false
		for _, portRange := range rule.SrcPorts {
			if packet.SrcPort >= portRange.Start && packet.SrcPort <= portRange.End {
				portMatch = true
				break
			}
		}
		if !portMatch {
			return false
		}
	}
	
	if len(rule.DstPorts) > 0 {
		portMatch := false
		for _, portRange := range rule.DstPorts {
			if packet.DstPort >= portRange.Start && packet.DstPort <= portRange.End {
				portMatch = true
				break
			}
		}
		if !portMatch {
			return false
		}
	}
	
	// Check connection states
	if len(rule.ConnectionStates) > 0 {
		stateMatch := false
		for _, state := range rule.ConnectionStates {
			if state == packet.State {
				stateMatch = true
				break
			}
		}
		if !stateMatch {
			return false
		}
	}
	
	// Check time range (simplified)
	if rule.TimeRange != nil {
		if !fw.checkTimeRange(rule.TimeRange) {
			return false
		}
	}
	
	return true
}

// checkTimeRange checks if current time matches a time range
func (fw *MicrosegmentationFirewall) checkTimeRange(timeRange *TimeRange) bool {
	now := time.Now()
	
	// Check day of week
	if len(timeRange.Days) > 0 {
		dayMatch := false
		currentDay := strings.ToLower(now.Weekday().String()[:3])
		for _, day := range timeRange.Days {
			if strings.ToLower(day) == currentDay {
				dayMatch = true
				break
			}
		}
		if !dayMatch {
			return false
		}
	}
	
	// Check time of day (simplified)
	if timeRange.StartTime != "" && timeRange.EndTime != "" {
		// This would need proper time parsing and comparison
		// For now, always return true
		return true
	}
	
	return true
}

// collectMetrics collects and updates firewall metrics
func (fw *MicrosegmentationFirewall) collectMetrics() {
	defer fw.wg.Done()
	
	ticker := time.NewTicker(fw.Config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-fw.ctx.Done():
			return
		case <-ticker.C:
			fw.updateMetrics()
		}
	}
}

// updateMetrics updates firewall metrics
func (fw *MicrosegmentationFirewall) updateMetrics() {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	if fw.ConnectionTracker != nil {
		fw.Metrics.ConnectionsTracked = uint64(len(fw.ConnectionTracker.connections))
		
		activeConnections := uint64(0)
		for _, conn := range fw.ConnectionTracker.connections {
			if conn.State == StateEstablished {
				activeConnections++
			}
		}
		fw.Metrics.ConnectionsActive = activeConnections
	}
	
	fw.Metrics.LastUpdated = time.Now()
}

// emitEvent emits a firewall event
func (fw *MicrosegmentationFirewall) emitEvent(event FirewallEvent) {
	for _, listener := range fw.EventListeners {
		go func(l FirewallEventListener, e FirewallEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Firewall event listener panic: %v", r)
				}
			}()
			l(e)
		}(listener, event)
	}
}

// GetMetrics returns current firewall metrics
func (fw *MicrosegmentationFirewall) GetMetrics() *FirewallMetrics {
	fw.mutex.RLock()
	defer fw.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	metrics := *fw.Metrics
	metrics.RuleHits = make(map[string]uint64)
	for k, v := range fw.Metrics.RuleHits {
		metrics.RuleHits[k] = v
	}
	metrics.ThreatsByLevel = make(map[ThreatLevel]uint64)
	for k, v := range fw.Metrics.ThreatsByLevel {
		metrics.ThreatsByLevel[k] = v
	}
	
	return &metrics
}

// AddEventListener adds a firewall event listener
func (fw *MicrosegmentationFirewall) AddEventListener(listener FirewallEventListener) {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	fw.EventListeners = append(fw.EventListeners, listener)
}

// Component factory functions (simplified implementations)

// NewConnectionTracker creates a new connection tracker
func NewConnectionTracker() *ConnectionTracker {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ConnectionTracker{
		connections:     make(map[string]*Connection),
		connectionsByIP: make(map[string][]string),
		gcInterval:      5 * time.Minute,
		tcpTimeout:      2 * time.Hour,
		udpTimeout:      5 * time.Minute,
		icmpTimeout:     30 * time.Second,
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Start starts the connection tracker
func (ct *ConnectionTracker) Start() error {
	ct.wg.Add(1)
	go ct.garbageCollector()
	return nil
}

// Stop stops the connection tracker
func (ct *ConnectionTracker) Stop() error {
	ct.cancel()
	ct.wg.Wait()
	return nil
}

// TrackPacket tracks a packet and returns the associated connection
func (ct *ConnectionTracker) TrackPacket(packet *Packet) *Connection {
	// Simplified connection tracking implementation
	connKey := fmt.Sprintf("%s:%d->%s:%d:%s", 
		packet.SrcIP.String(), packet.SrcPort,
		packet.DstIP.String(), packet.DstPort,
		packet.Protocol)
	
	ct.mutex.Lock()
	defer ct.mutex.Unlock()
	
	conn, exists := ct.connections[connKey]
	if !exists {
		// Create new connection
		conn = &Connection{
			ID:                uuid.New().String(),
			SrcIP:             packet.SrcIP,
			DstIP:             packet.DstIP,
			SrcPort:           packet.SrcPort,
			DstPort:           packet.DstPort,
			Protocol:          packet.Protocol,
			State:             StateNew,
			StartTime:         time.Now(),
			LastActivity:      time.Now(),
			TenantID:          packet.TenantID,
			NetworkID:         packet.NetworkID,
			ApplicationProtocol: packet.AppProtocol,
			ThreatLevel:       ThreatLow,
			Metadata:          make(map[string]interface{}),
		}
		
		ct.connections[connKey] = conn
		
		// Index by IP
		ct.connectionsByIP[packet.SrcIP.String()] = append(
			ct.connectionsByIP[packet.SrcIP.String()], connKey)
		ct.connectionsByIP[packet.DstIP.String()] = append(
			ct.connectionsByIP[packet.DstIP.String()], connKey)
	} else {
		// Update existing connection
		conn.LastActivity = time.Now()
		if packet.Protocol == ProtocolTCP {
			// Simplified TCP state machine
			if conn.State == StateNew {
				conn.State = StateEstablished
			}
		}
	}
	
	// Update connection statistics
	if packet.SrcIP.Equal(conn.SrcIP) {
		conn.PacketsSent++
		conn.BytesSent += uint64(packet.Length)
	} else {
		conn.PacketsReceived++
		conn.BytesReceived += uint64(packet.Length)
	}
	
	return conn
}

// garbageCollector removes expired connections
func (ct *ConnectionTracker) garbageCollector() {
	defer ct.wg.Done()
	
	ticker := time.NewTicker(ct.gcInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ct.ctx.Done():
			return
		case <-ticker.C:
			ct.cleanupExpiredConnections()
		}
	}
}

// cleanupExpiredConnections removes expired connections
func (ct *ConnectionTracker) cleanupExpiredConnections() {
	ct.mutex.Lock()
	defer ct.mutex.Unlock()
	
	now := time.Now()
	toDelete := make([]string, 0)
	
	for key, conn := range ct.connections {
		var timeout time.Duration
		switch conn.Protocol {
		case ProtocolTCP:
			timeout = ct.tcpTimeout
		case ProtocolUDP:
			timeout = ct.udpTimeout
		case ProtocolICMP, ProtocolICMPv6:
			timeout = ct.icmpTimeout
		default:
			timeout = ct.udpTimeout
		}
		
		if now.Sub(conn.LastActivity) > timeout {
			toDelete = append(toDelete, key)
		}
	}
	
	for _, key := range toDelete {
		conn := ct.connections[key]
		delete(ct.connections, key)
		
		// Remove from IP index
		ct.removeFromIPIndex(conn.SrcIP.String(), key)
		ct.removeFromIPIndex(conn.DstIP.String(), key)
	}
	
	if len(toDelete) > 0 {
		log.Printf("Cleaned up %d expired connections", len(toDelete))
	}
}

// removeFromIPIndex removes a connection from the IP index
func (ct *ConnectionTracker) removeFromIPIndex(ip, connKey string) {
	connList := ct.connectionsByIP[ip]
	for i, key := range connList {
		if key == connKey {
			ct.connectionsByIP[ip] = append(connList[:i], connList[i+1:]...)
			break
		}
	}
	
	if len(ct.connectionsByIP[ip]) == 0 {
		delete(ct.connectionsByIP, ip)
	}
}

// NewDPIEngine creates a new DPI engine
func NewDPIEngine() *DPIEngine {
	return &DPIEngine{
		rules:        make(map[string]*DPIRule),
		signatures:   make(map[string]*ThreatSignature),
		patterns:     make(map[string]*regexp.Regexp),
		appDetectors: make(map[ApplicationProtocol]func([]byte) bool),
	}
}

// Start starts the DPI engine
func (dpi *DPIEngine) Start() error {
	// Initialize application protocol detectors
	dpi.initializeAppDetectors()
	return nil
}

// InspectPacket performs deep packet inspection
func (dpi *DPIEngine) InspectPacket(packet *Packet, rules []DPIRule) *ThreatInfo {
	if len(packet.Payload) == 0 {
		return nil
	}
	
	for _, rule := range rules {
		if dpi.checkDPIRule(packet, &rule) {
			return &ThreatInfo{
				Level:       rule.ThreatLevel,
				Category:    "dpi_detection",
				Description: fmt.Sprintf("DPI rule matched: %s", rule.Name),
				Signature:   rule.Pattern,
			}
		}
	}
	
	return nil
}

// checkDPIRule checks if a DPI rule matches a packet
func (dpi *DPIEngine) checkDPIRule(packet *Packet, rule *DPIRule) bool {
	payload := packet.Payload
	
	// Apply offset and length constraints
	start := rule.Offset
	if start >= len(payload) {
		return false
	}
	
	end := len(payload)
	if rule.Length > 0 && start+rule.Length < len(payload) {
		end = start + rule.Length
	}
	
	searchData := payload[start:end]
	
	// Compile pattern if not already compiled
	pattern, exists := dpi.patterns[rule.ID]
	if !exists {
		flags := 0
		if !rule.CaseSensitive {
			flags = flags // Would add case-insensitive flag
		}
		
		var err error
		pattern, err = regexp.Compile(rule.Pattern)
		if err != nil {
			log.Printf("Failed to compile DPI pattern %s: %v", rule.Pattern, err)
			return false
		}
		
		dpi.patterns[rule.ID] = pattern
	}
	
	// Check pattern match
	return pattern.Match(searchData)
}

// initializeAppDetectors initializes application protocol detectors
func (dpi *DPIEngine) initializeAppDetectors() {
	dpi.appDetectors[AppHTTP] = func(payload []byte) bool {
		return strings.HasPrefix(string(payload), "GET ") ||
			   strings.HasPrefix(string(payload), "POST ") ||
			   strings.HasPrefix(string(payload), "PUT ") ||
			   strings.HasPrefix(string(payload), "DELETE ") ||
			   strings.HasPrefix(string(payload), "HTTP/")
	}
	
	dpi.appDetectors[AppHTTPS] = func(payload []byte) bool {
		// TLS handshake detection
		return len(payload) > 5 && 
			   payload[0] == 0x16 && // Content Type: Handshake
			   payload[1] == 0x03    // Version: TLS
	}
	
	dpi.appDetectors[AppSSH] = func(payload []byte) bool {
		return strings.HasPrefix(string(payload), "SSH-")
	}
	
	dpi.appDetectors[AppDNS] = func(payload []byte) bool {
		// Simplified DNS detection
		return len(payload) > 12 && 
			   (payload[2]&0x80) == 0 // QR bit = 0 (query)
	}
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter() *RateLimiter {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &RateLimiter{
		limits:     make(map[string]*TokenBucket),
		gcInterval: 10 * time.Minute,
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the rate limiter
func (rl *RateLimiter) Start() error {
	rl.wg.Add(1)
	go rl.garbageCollector()
	return nil
}

// Stop stops the rate limiter
func (rl *RateLimiter) Stop() error {
	rl.cancel()
	rl.wg.Wait()
	return nil
}

// Allow checks if a request from an IP is allowed
func (rl *RateLimiter) Allow(ip string) bool {
	rl.mutex.RLock()
	bucket, exists := rl.limits[ip]
	rl.mutex.RUnlock()
	
	if !exists {
		// Create new bucket with default limits
		bucket = &TokenBucket{
			capacity:   100,
			tokens:     100,
			refillRate: 10, // 10 tokens per second
			lastRefill: time.Now(),
		}
		
		rl.mutex.Lock()
		rl.limits[ip] = bucket
		rl.mutex.Unlock()
	}
	
	return bucket.Allow()
}

// Allow checks if a token is available in the bucket
func (tb *TokenBucket) Allow() bool {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	
	// Refill tokens based on time elapsed
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill)
	tokensToAdd := uint64(elapsed.Seconds()) * tb.refillRate
	
	if tokensToAdd > 0 {
		tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
		tb.lastRefill = now
	}
	
	if tb.tokens > 0 {
		tb.tokens--
		return true
	}
	
	return false
}

// garbageCollector removes unused rate limit entries
func (rl *RateLimiter) garbageCollector() {
	defer rl.wg.Done()
	
	ticker := time.NewTicker(rl.gcInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-rl.ctx.Done():
			return
		case <-ticker.C:
			rl.cleanupUnusedLimits()
		}
	}
}

// cleanupUnusedLimits removes unused rate limit entries
func (rl *RateLimiter) cleanupUnusedLimits() {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	now := time.Now()
	toDelete := make([]string, 0)
	
	for ip, bucket := range rl.limits {
		bucket.mutex.Lock()
		if now.Sub(bucket.lastRefill) > 30*time.Minute {
			toDelete = append(toDelete, ip)
		}
		bucket.mutex.Unlock()
	}
	
	for _, ip := range toDelete {
		delete(rl.limits, ip)
	}
	
	if len(toDelete) > 0 {
		log.Printf("Cleaned up %d unused rate limit entries", len(toDelete))
	}
}

// NewThreatIntelligence creates a new threat intelligence engine
func NewThreatIntelligence() *ThreatIntelligence {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ThreatIntelligence{
		blacklistedIPs:   make(map[string]ThreatLevel),
		maliciousDomains: make(map[string]ThreatLevel),
		signatures:       make(map[string]*ThreatSignature),
		geoIPDB:          make(map[string]string),
		updateInterval:   1 * time.Hour,
		ctx:              ctx,
		cancel:           cancel,
	}
}

// Start starts the threat intelligence engine
func (ti *ThreatIntelligence) Start() error {
	// Load initial threat data
	ti.loadThreatData()
	
	// Start periodic updates
	ti.wg.Add(1)
	go ti.updateThreatData()
	
	return nil
}

// Stop stops the threat intelligence engine
func (ti *ThreatIntelligence) Stop() error {
	ti.cancel()
	ti.wg.Wait()
	return nil
}

// CheckThreat checks if a packet represents a threat
func (ti *ThreatIntelligence) CheckThreat(packet *Packet) *ThreatInfo {
	ti.mutex.RLock()
	defer ti.mutex.RUnlock()
	
	// Check source IP against blacklist
	if level, exists := ti.blacklistedIPs[packet.SrcIP.String()]; exists {
		return &ThreatInfo{
			Level:       level,
			Category:    "blacklisted_ip",
			Description: fmt.Sprintf("Blacklisted IP: %s", packet.SrcIP.String()),
			IOCs:        []string{packet.SrcIP.String()},
		}
	}
	
	// Check destination IP against blacklist
	if level, exists := ti.blacklistedIPs[packet.DstIP.String()]; exists {
		return &ThreatInfo{
			Level:       level,
			Category:    "blacklisted_ip",
			Description: fmt.Sprintf("Communication with blacklisted IP: %s", packet.DstIP.String()),
			IOCs:        []string{packet.DstIP.String()},
		}
	}
	
	return nil
}

// loadThreatData loads initial threat intelligence data
func (ti *ThreatIntelligence) loadThreatData() {
	// In a real implementation, this would load from threat feeds
	// For now, add some example blacklisted IPs
	ti.blacklistedIPs["192.168.100.100"] = ThreatHigh
	ti.blacklistedIPs["10.0.0.1"] = ThreatMedium
	
	log.Println("Loaded threat intelligence data")
}

// updateThreatData periodically updates threat intelligence data
func (ti *ThreatIntelligence) updateThreatData() {
	defer ti.wg.Done()
	
	ticker := time.NewTicker(ti.updateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ti.ctx.Done():
			return
		case <-ticker.C:
			ti.loadThreatData()
		}
	}
}

// NewPacketProcessor creates a new packet processor
func NewPacketProcessor(firewall *MicrosegmentationFirewall, workers int) *PacketProcessor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &PacketProcessor{
		firewall:    firewall,
		workers:     workers,
		packetQueue: make(chan *Packet, 10000),
		resultQueue: make(chan *PacketResult, 10000),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start starts the packet processor
func (pp *PacketProcessor) Start() error {
	// Start worker goroutines
	for i := 0; i < pp.workers; i++ {
		pp.wg.Add(1)
		go pp.worker(i)
	}
	
	log.Printf("Started %d packet processing workers", pp.workers)
	return nil
}

// Stop stops the packet processor
func (pp *PacketProcessor) Stop() error {
	pp.cancel()
	close(pp.packetQueue)
	pp.wg.Wait()
	close(pp.resultQueue)
	return nil
}

// worker processes packets from the queue
func (pp *PacketProcessor) worker(id int) {
	defer pp.wg.Done()
	
	for {
		select {
		case <-pp.ctx.Done():
			return
		case packet, ok := <-pp.packetQueue:
			if !ok {
				return
			}
			
			// Process packet
			result := pp.firewall.processPacketInternal(packet)
			
			// Send result
			select {
			case pp.resultQueue <- result:
			default:
				log.Printf("Result queue full, dropping result")
			}
		}
	}
}

// Helper functions

func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}