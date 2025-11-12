// Package edge provides edge network optimization capabilities
package edge

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/net/quic"
)

// TransportProtocol represents transport protocols
type TransportProtocol string

const (
	TransportTCP    TransportProtocol = "tcp"
	TransportUDP    TransportProtocol = "udp"
	TransportQUIC   TransportProtocol = "quic"
	TransportSCTP   TransportProtocol = "sctp"
	TransportCustom TransportProtocol = "custom"
)

// PathSelectionStrategy represents path selection strategies
type PathSelectionStrategy string

const (
	PathStrategyLatency    PathSelectionStrategy = "latency"
	PathStrategyBandwidth  PathSelectionStrategy = "bandwidth"
	PathStrategyReliability PathSelectionStrategy = "reliability"
	PathStrategyHybrid     PathSelectionStrategy = "hybrid"
	PathStrategyAdaptive   PathSelectionStrategy = "adaptive"
)

// EdgeNetworkOptimizer manages network optimization for edge
type EdgeNetworkOptimizer struct {
	transport      *MultiProtocolTransport
	pathManager    *MultiPathManager
	natTraversal   *NATTraversalManager
	p2pManager     *P2PConnectionManager
	qosManager     *QoSManager
	metrics        *NetworkMetrics
	config         *NetworkConfig
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
}

// NetworkConfig contains network configuration
type NetworkConfig struct {
	Protocols           []TransportProtocol
	EnableMultiPath     bool
	MaxPaths            int
	EnableNATTraversal  bool
	EnableP2P           bool
	QoSEnabled          bool
	MaxBandwidthMbps    float64
	LatencyTargetMs     float64
	PacketLossThreshold float64
	RetransmitTimeout   time.Duration
	KeepAliveInterval   time.Duration
}

// MultiProtocolTransport handles multiple transport protocols
type MultiProtocolTransport struct {
	protocols  map[TransportProtocol]ProtocolHandler
	active     map[string]*Connection
	selector   *ProtocolSelector
	mu         sync.RWMutex
}

// ProtocolHandler interface for protocol implementations
type ProtocolHandler interface {
	Connect(address string, options map[string]interface{}) (*Connection, error)
	Listen(address string, options map[string]interface{}) (net.Listener, error)
	Send(conn *Connection, data []byte) error
	Receive(conn *Connection) ([]byte, error)
	Close(conn *Connection) error
}

// Connection represents a network connection
type Connection struct {
	ID           string
	Protocol     TransportProtocol
	LocalAddr    net.Addr
	RemoteAddr   net.Addr
	State        ConnectionState
	Stats        *ConnectionStats
	conn         net.Conn
	quicSession  quic.Session
	streams      sync.Map
	mu           sync.RWMutex
}

// ConnectionState represents connection state
type ConnectionState string

const (
	ConnectionStateConnecting   ConnectionState = "connecting"
	ConnectionStateConnected    ConnectionState = "connected"
	ConnectionStateDisconnected ConnectionState = "disconnected"
	ConnectionStateFailed       ConnectionState = "failed"
)

// ConnectionStats tracks connection statistics
type ConnectionStats struct {
	BytesSent       uint64
	BytesReceived   uint64
	PacketsSent     uint64
	PacketsReceived uint64
	PacketsLost     uint64
	RTT             float64 // milliseconds
	Bandwidth       float64 // Mbps
	LastActivity    time.Time
}

// ProtocolSelector selects optimal protocol
type ProtocolSelector struct {
	strategy SelectionStrategy
	history  map[string]*ProtocolPerformance
	mu       sync.RWMutex
}

// SelectionStrategy for protocol selection
type SelectionStrategy interface {
	Select(protocols []TransportProtocol, endpoint string) TransportProtocol
}

// ProtocolPerformance tracks protocol performance
type ProtocolPerformance struct {
	Protocol     TransportProtocol
	SuccessRate  float64
	AvgLatency   float64
	AvgBandwidth float64
	LastUsed     time.Time
}

// MultiPathManager manages multi-path routing
type MultiPathManager struct {
	paths       map[string][]*NetworkPath
	strategy    PathSelectionStrategy
	scheduler   *PathScheduler
	monitor     *PathMonitor
	maxPaths    int
	mu          sync.RWMutex
}

// NetworkPath represents a network path
type NetworkPath struct {
	ID          string
	Source      string
	Destination string
	Hops        []string
	Latency     float64
	Bandwidth   float64
	PacketLoss  float64
	State       PathState
	Priority    int
	LastProbe   time.Time
}

// PathState represents path state
type PathState string

const (
	PathStateActive   PathState = "active"
	PathStateBackup   PathState = "backup"
	PathStateFailed   PathState = "failed"
	PathStateProbing  PathState = "probing"
)

// PathScheduler schedules packets across paths
type PathScheduler struct {
	algorithm SchedulingAlgorithm
	queues    map[string]*PacketQueue
	weights   map[string]float64
	mu        sync.RWMutex
}

// SchedulingAlgorithm for packet scheduling
type SchedulingAlgorithm interface {
	Schedule(packet *Packet, paths []*NetworkPath) *NetworkPath
}

// Packet represents a network packet
type Packet struct {
	ID        uint64
	Data      []byte
	Priority  int
	Timestamp time.Time
	Retries   int
}

// PacketQueue manages packet queuing
type PacketQueue struct {
	packets  []*Packet
	capacity int
	mu       sync.RWMutex
}

// PathMonitor monitors path health
type PathMonitor struct {
	probes   map[string]*ProbeResult
	interval time.Duration
	mu       sync.RWMutex
}

// ProbeResult represents path probe result
type ProbeResult struct {
	PathID     string
	RTT        float64
	PacketLoss float64
	Jitter     float64
	Timestamp  time.Time
}

// NATTraversalManager handles NAT traversal
type NATTraversalManager struct {
	stunServers  []string
	turnServers  []string
	iceAgents    sync.Map
	mappings     sync.Map
	mu           sync.RWMutex
}

// ICEAgent handles ICE protocol
type ICEAgent struct {
	ID            string
	LocalCandidates  []ICECandidate
	RemoteCandidates []ICECandidate
	SelectedPair     *CandidatePair
	State           ICEState
}

// ICECandidate represents an ICE candidate
type ICECandidate struct {
	Type      CandidateType
	Address   string
	Port      int
	Priority  uint32
	Foundation string
}

// CandidateType represents ICE candidate type
type CandidateType string

const (
	CandidateTypeHost  CandidateType = "host"
	CandidateTypeSrflx CandidateType = "srflx"
	CandidateTypeRelay CandidateType = "relay"
)

// CandidatePair represents an ICE candidate pair
type CandidatePair struct {
	Local    ICECandidate
	Remote   ICECandidate
	Priority uint64
	State    PairState
}

// PairState represents candidate pair state
type PairState string

const (
	PairStateWaiting   PairState = "waiting"
	PairStateInProgress PairState = "in_progress"
	PairStateSucceeded  PairState = "succeeded"
	PairStateFailed     PairState = "failed"
)

// ICEState represents ICE connection state
type ICEState string

const (
	ICEStateNew          ICEState = "new"
	ICEStateGathering    ICEState = "gathering"
	ICEStateConnecting   ICEState = "connecting"
	ICEStateConnected    ICEState = "connected"
	ICEStateDisconnected ICEState = "disconnected"
	ICEStateFailed       ICEState = "failed"
)

// P2PConnectionManager manages peer-to-peer connections
type P2PConnectionManager struct {
	peers       sync.Map
	overlay     *OverlayNetwork
	discovery   *PeerDiscovery
	mu          sync.RWMutex
}

// Peer represents a peer node
type Peer struct {
	ID         string
	Address    string
	PublicKey  []byte
	Connection *Connection
	LastSeen   time.Time
}

// OverlayNetwork manages overlay network
type OverlayNetwork struct {
	topology  NetworkTopology
	routing   RoutingTable
	mu        sync.RWMutex
}

// NetworkTopology represents network topology
type NetworkTopology interface {
	AddNode(nodeID string, metadata map[string]interface{})
	RemoveNode(nodeID string)
	GetNeighbors(nodeID string) []string
	GetPath(source, destination string) []string
}

// RoutingTable manages routing information
type RoutingTable struct {
	routes map[string]*Route
	mu     sync.RWMutex
}

// Route represents a network route
type Route struct {
	Destination string
	NextHop     string
	Metric      int
	Interface   string
}

// PeerDiscovery handles peer discovery
type PeerDiscovery struct {
	methods []DiscoveryMethod
	found   sync.Map
	mu      sync.RWMutex
}

// DiscoveryMethod interface for peer discovery
type DiscoveryMethod interface {
	Discover() ([]*Peer, error)
	Announce(peer *Peer) error
}

// QoSManager manages quality of service
type QoSManager struct {
	classes    map[string]*QoSClass
	limiters   sync.Map
	shapers    sync.Map
	mu         sync.RWMutex
}

// QoSClass represents a QoS class
type QoSClass struct {
	Name         string
	Priority     int
	MinBandwidth float64
	MaxBandwidth float64
	MaxLatency   float64
	MaxJitter    float64
}

// NetworkMetrics tracks network metrics
type NetworkMetrics struct {
	bytesSent        prometheus.Counter
	bytesReceived    prometheus.Counter
	packetsSent      prometheus.Counter
	packetsReceived  prometheus.Counter
	packetsLost      prometheus.Counter
	connectionCount  prometheus.Gauge
	activePathsCount prometheus.Gauge
	p2pPeersCount    prometheus.Gauge
	rttHistogram     prometheus.Histogram
	bandwidthGauge   prometheus.Gauge
}

// NewEdgeNetworkOptimizer creates a new network optimizer
func NewEdgeNetworkOptimizer(config *NetworkConfig) *EdgeNetworkOptimizer {
	ctx, cancel := context.WithCancel(context.Background())

	optimizer := &EdgeNetworkOptimizer{
		transport:    NewMultiProtocolTransport(),
		pathManager:  NewMultiPathManager(config.MaxPaths),
		natTraversal: NewNATTraversalManager(),
		p2pManager:   NewP2PConnectionManager(),
		qosManager:   NewQoSManager(),
		metrics:      NewNetworkMetrics(),
		config:       config,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Start optimization workers
	optimizer.wg.Add(3)
	go optimizer.pathMonitorWorker()
	go optimizer.qosWorker()
	go optimizer.p2pWorker()

	return optimizer
}

// NewMultiProtocolTransport creates a new multi-protocol transport
func NewMultiProtocolTransport() *MultiProtocolTransport {
	return &MultiProtocolTransport{
		protocols: make(map[TransportProtocol]ProtocolHandler),
		active:    make(map[string]*Connection),
		selector:  NewProtocolSelector(),
	}
}

// NewProtocolSelector creates a new protocol selector
func NewProtocolSelector() *ProtocolSelector {
	return &ProtocolSelector{
		strategy: &AdaptiveSelectionStrategy{},
		history:  make(map[string]*ProtocolPerformance),
	}
}

// NewMultiPathManager creates a new multi-path manager
func NewMultiPathManager(maxPaths int) *MultiPathManager {
	return &MultiPathManager{
		paths:     make(map[string][]*NetworkPath),
		strategy:  PathStrategyAdaptive,
		scheduler: NewPathScheduler(),
		monitor:   NewPathMonitor(),
		maxPaths:  maxPaths,
	}
}

// NewPathScheduler creates a new path scheduler
func NewPathScheduler() *PathScheduler {
	return &PathScheduler{
		algorithm: &WeightedRoundRobinScheduler{},
		queues:    make(map[string]*PacketQueue),
		weights:   make(map[string]float64),
	}
}

// NewPathMonitor creates a new path monitor
func NewPathMonitor() *PathMonitor {
	return &PathMonitor{
		probes:   make(map[string]*ProbeResult),
		interval: 5 * time.Second,
	}
}

// NewNATTraversalManager creates a new NAT traversal manager
func NewNATTraversalManager() *NATTraversalManager {
	return &NATTraversalManager{
		stunServers: []string{
			"stun.l.google.com:19302",
			"stun1.l.google.com:19302",
		},
		turnServers: []string{
			"turn.example.com:3478",
		},
	}
}

// NewP2PConnectionManager creates a new P2P connection manager
func NewP2PConnectionManager() *P2PConnectionManager {
	return &P2PConnectionManager{
		overlay:   NewOverlayNetwork(),
		discovery: NewPeerDiscovery(),
	}
}

// NewOverlayNetwork creates a new overlay network
func NewOverlayNetwork() *OverlayNetwork {
	return &OverlayNetwork{
		topology: &MeshTopology{nodes: make(map[string][]string)},
		routing:  RoutingTable{routes: make(map[string]*Route)},
	}
}

// NewPeerDiscovery creates a new peer discovery
func NewPeerDiscovery() *PeerDiscovery {
	return &PeerDiscovery{
		methods: []DiscoveryMethod{},
	}
}

// NewQoSManager creates a new QoS manager
func NewQoSManager() *QoSManager {
	return &QoSManager{
		classes: make(map[string]*QoSClass),
	}
}

// NewNetworkMetrics creates new network metrics
func NewNetworkMetrics() *NetworkMetrics {
	return &NetworkMetrics{
		bytesSent: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_network_bytes_sent_total",
				Help: "Total bytes sent",
			},
		),
		bytesReceived: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_network_bytes_received_total",
				Help: "Total bytes received",
			},
		),
		packetsSent: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_network_packets_sent_total",
				Help: "Total packets sent",
			},
		),
		packetsReceived: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_network_packets_received_total",
				Help: "Total packets received",
			},
		),
		packetsLost: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_network_packets_lost_total",
				Help: "Total packets lost",
			},
		),
		connectionCount: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_network_connections",
				Help: "Number of active connections",
			},
		),
		activePathsCount: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_network_active_paths",
				Help: "Number of active network paths",
			},
		),
		p2pPeersCount: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_network_p2p_peers",
				Help: "Number of P2P peers",
			},
		),
		rttHistogram: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_network_rtt_milliseconds",
				Help:    "Round trip time",
				Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
			},
		),
		bandwidthGauge: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_network_bandwidth_mbps",
				Help: "Current bandwidth usage",
			},
		),
	}
}

// Connect establishes a connection to an edge node
func (o *EdgeNetworkOptimizer) Connect(address string, options map[string]interface{}) (*Connection, error) {
	// Select optimal protocol
	protocol := o.transport.selector.SelectProtocol(address)

	// Get protocol handler
	handler, exists := o.transport.protocols[protocol]
	if !exists {
		return nil, fmt.Errorf("protocol %s not supported", protocol)
	}

	// Establish connection
	conn, err := handler.Connect(address, options)
	if err != nil {
		return nil, err
	}

	// Store connection
	o.transport.mu.Lock()
	o.transport.active[conn.ID] = conn
	o.transport.mu.Unlock()

	// Update metrics
	o.metrics.connectionCount.Set(float64(len(o.transport.active)))

	// Setup multi-path if enabled
	if o.config.EnableMultiPath {
		o.setupMultiPath(conn)
	}

	return conn, nil
}

// Send sends data over optimized connection
func (o *EdgeNetworkOptimizer) Send(conn *Connection, data []byte) error {
	// Apply QoS if enabled
	if o.config.QoSEnabled {
		o.qosManager.ApplyQoS(conn, data)
	}

	// Select path if multi-path enabled
	var path *NetworkPath
	if o.config.EnableMultiPath {
		paths := o.pathManager.GetPaths(conn.RemoteAddr.String())
		if len(paths) > 0 {
			packet := &Packet{
				ID:        uint64(time.Now().UnixNano()),
				Data:      data,
				Timestamp: time.Now(),
			}
			path = o.pathManager.scheduler.algorithm.Schedule(packet, paths)
		}
	}

	// Send data
	handler := o.transport.protocols[conn.Protocol]
	if err := handler.Send(conn, data); err != nil {
		return err
	}

	// Update stats
	atomic.AddUint64(&conn.Stats.BytesSent, uint64(len(data)))
	atomic.AddUint64(&conn.Stats.PacketsSent, 1)
	conn.Stats.LastActivity = time.Now()

	// Update metrics
	o.metrics.bytesSent.Add(float64(len(data)))
	o.metrics.packetsSent.Inc()

	return nil
}

// EstablishP2P establishes a P2P connection
func (o *EdgeNetworkOptimizer) EstablishP2P(peerID string) (*Peer, error) {
	// Check if already connected
	if peer, exists := o.p2pManager.peers.Load(peerID); exists {
		return peer.(*Peer), nil
	}

	// Discover peer
	peers, err := o.p2pManager.discovery.DiscoverPeer(peerID)
	if err != nil {
		return nil, err
	}

	if len(peers) == 0 {
		return nil, fmt.Errorf("peer %s not found", peerID)
	}

	peer := peers[0]

	// Handle NAT traversal if needed
	if o.config.EnableNATTraversal {
		if err := o.natTraversal.EstablishConnection(peer); err != nil {
			return nil, err
		}
	}

	// Connect to peer
	conn, err := o.Connect(peer.Address, nil)
	if err != nil {
		return nil, err
	}

	peer.Connection = conn

	// Store peer
	o.p2pManager.peers.Store(peerID, peer)

	// Update metrics
	count := 0
	o.p2pManager.peers.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	o.metrics.p2pPeersCount.Set(float64(count))

	return peer, nil
}

// setupMultiPath sets up multi-path routing
func (o *EdgeNetworkOptimizer) setupMultiPath(conn *Connection) {
	destination := conn.RemoteAddr.String()

	// Discover paths
	paths := o.discoverPaths(conn.LocalAddr.String(), destination)

	// Store paths
	o.pathManager.mu.Lock()
	o.pathManager.paths[destination] = paths
	o.pathManager.mu.Unlock()

	// Start path monitoring
	for _, path := range paths {
		go o.monitorPath(path)
	}

	// Update metrics
	o.metrics.activePathsCount.Set(float64(len(paths)))
}

// discoverPaths discovers network paths
func (o *EdgeNetworkOptimizer) discoverPaths(source, destination string) []*NetworkPath {
	paths := []*NetworkPath{}

	// Primary path
	primary := &NetworkPath{
		ID:          fmt.Sprintf("path-%s-%s-primary", source, destination),
		Source:      source,
		Destination: destination,
		State:       PathStateActive,
		Priority:    1,
		LastProbe:   time.Now(),
	}
	paths = append(paths, primary)

	// Alternative paths (simplified)
	if o.config.MaxPaths > 1 {
		for i := 1; i < o.config.MaxPaths && i < 3; i++ {
			alt := &NetworkPath{
				ID:          fmt.Sprintf("path-%s-%s-alt%d", source, destination, i),
				Source:      source,
				Destination: destination,
				State:       PathStateBackup,
				Priority:    i + 1,
				LastProbe:   time.Now(),
			}
			paths = append(paths, alt)
		}
	}

	return paths
}

// monitorPath monitors path health
func (o *EdgeNetworkOptimizer) monitorPath(path *NetworkPath) {
	ticker := time.NewTicker(o.pathManager.monitor.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.probePath(path)
		case <-o.ctx.Done():
			return
		}
	}
}

// probePath probes a network path
func (o *EdgeNetworkOptimizer) probePath(path *NetworkPath) {
	start := time.Now()

	// Simulate probe (would send actual probe packets in production)
	path.Latency = float64(time.Since(start).Milliseconds())
	path.LastProbe = time.Now()

	// Store probe result
	result := &ProbeResult{
		PathID:    path.ID,
		RTT:       path.Latency,
		Timestamp: time.Now(),
	}

	o.pathManager.monitor.mu.Lock()
	o.pathManager.monitor.probes[path.ID] = result
	o.pathManager.monitor.mu.Unlock()

	// Update metrics
	o.metrics.rttHistogram.Observe(path.Latency)
}

// NAT Traversal methods

func (nat *NATTraversalManager) EstablishConnection(peer *Peer) error {
	// Create ICE agent
	agent := &ICEAgent{
		ID:    fmt.Sprintf("ice-%s", peer.ID),
		State: ICEStateNew,
	}

	// Gather candidates
	agent.State = ICEStateGathering
	candidates, err := nat.gatherCandidates()
	if err != nil {
		return err
	}
	agent.LocalCandidates = candidates

	// Exchange candidates with peer (simplified)
	// In production, would use signaling server

	// Perform connectivity checks
	agent.State = ICEStateConnecting
	pair, err := nat.performConnectivityChecks(agent)
	if err != nil {
		agent.State = ICEStateFailed
		return err
	}

	agent.SelectedPair = pair
	agent.State = ICEStateConnected

	// Store agent
	nat.iceAgents.Store(peer.ID, agent)

	return nil
}

func (nat *NATTraversalManager) gatherCandidates() ([]ICECandidate, error) {
	candidates := []ICECandidate{}

	// Host candidate
	hostAddr, _ := net.InterfaceAddrs()
	if len(hostAddr) > 0 {
		candidates = append(candidates, ICECandidate{
			Type:     CandidateTypeHost,
			Address:  hostAddr[0].String(),
			Port:     5000,
			Priority: 2130706431, // Host priority
		})
	}

	// STUN candidate (server reflexive)
	for _, stunServer := range nat.stunServers {
		// Would perform actual STUN binding request
		candidates = append(candidates, ICECandidate{
			Type:     CandidateTypeSrflx,
			Address:  stunServer,
			Port:     5001,
			Priority: 1694498815, // Srflx priority
		})
	}

	return candidates, nil
}

func (nat *NATTraversalManager) performConnectivityChecks(agent *ICEAgent) (*CandidatePair, error) {
	// Simplified connectivity check
	// In production, would perform actual STUN binding requests

	if len(agent.LocalCandidates) > 0 && len(agent.RemoteCandidates) > 0 {
		pair := &CandidatePair{
			Local:  agent.LocalCandidates[0],
			Remote: agent.RemoteCandidates[0],
			State:  PairStateSucceeded,
		}
		return pair, nil
	}

	return nil, fmt.Errorf("no valid candidate pairs")
}

// QoS methods

func (qos *QoSManager) ApplyQoS(conn *Connection, data []byte) {
	// Apply rate limiting based on QoS class
	// Simplified implementation
}

// Protocol Selection Strategy

type AdaptiveSelectionStrategy struct{}

func (s *AdaptiveSelectionStrategy) Select(protocols []TransportProtocol, endpoint string) TransportProtocol {
	// Adaptive selection based on conditions
	// Default to QUIC for low latency
	for _, p := range protocols {
		if p == TransportQUIC {
			return TransportQUIC
		}
	}
	return TransportTCP
}

func (ps *ProtocolSelector) SelectProtocol(endpoint string) TransportProtocol {
	// Check history for best performing protocol
	ps.mu.RLock()
	perf, exists := ps.history[endpoint]
	ps.mu.RUnlock()

	if exists && perf.SuccessRate > 0.9 {
		return perf.Protocol
	}

	// Use selection strategy
	availableProtocols := []TransportProtocol{TransportTCP, TransportUDP, TransportQUIC}
	return ps.strategy.Select(availableProtocols, endpoint)
}

// Path Scheduling

type WeightedRoundRobinScheduler struct {
	current uint64
}

func (s *WeightedRoundRobinScheduler) Schedule(packet *Packet, paths []*NetworkPath) *NetworkPath {
	if len(paths) == 0 {
		return nil
	}

	// Simple round-robin for now
	idx := atomic.AddUint64(&s.current, 1) % uint64(len(paths))
	return paths[idx]
}

// Peer Discovery

func (pd *PeerDiscovery) DiscoverPeer(peerID string) ([]*Peer, error) {
	peers := []*Peer{}

	for _, method := range pd.methods {
		discovered, err := method.Discover()
		if err != nil {
			continue
		}

		for _, peer := range discovered {
			if peer.ID == peerID {
				peers = append(peers, peer)
			}
		}
	}

	return peers, nil
}

// Mesh Topology

type MeshTopology struct {
	nodes map[string][]string
	mu    sync.RWMutex
}

func (t *MeshTopology) AddNode(nodeID string, metadata map[string]interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.nodes[nodeID]; !exists {
		t.nodes[nodeID] = []string{}
	}
}

func (t *MeshTopology) RemoveNode(nodeID string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	delete(t.nodes, nodeID)

	// Remove from other nodes' neighbor lists
	for id, neighbors := range t.nodes {
		filtered := []string{}
		for _, n := range neighbors {
			if n != nodeID {
				filtered = append(filtered, n)
			}
		}
		t.nodes[id] = filtered
	}
}

func (t *MeshTopology) GetNeighbors(nodeID string) []string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.nodes[nodeID]
}

func (t *MeshTopology) GetPath(source, destination string) []string {
	// Simplified path finding
	return []string{source, destination}
}

// Worker loops

func (o *EdgeNetworkOptimizer) pathMonitorWorker() {
	defer o.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.evaluatePaths()
		case <-o.ctx.Done():
			return
		}
	}
}

func (o *EdgeNetworkOptimizer) qosWorker() {
	defer o.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.enforceQoS()
		case <-o.ctx.Done():
			return
		}
	}
}

func (o *EdgeNetworkOptimizer) p2pWorker() {
	defer o.wg.Done()

	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.maintainP2PConnections()
		case <-o.ctx.Done():
			return
		}
	}
}

func (o *EdgeNetworkOptimizer) evaluatePaths() {
	o.pathManager.mu.RLock()
	defer o.pathManager.mu.RUnlock()

	for _, paths := range o.pathManager.paths {
		for _, path := range paths {
			// Evaluate path performance
			if path.PacketLoss > o.config.PacketLossThreshold {
				path.State = PathStateFailed
			} else if path.Latency > o.config.LatencyTargetMs {
				path.State = PathStateBackup
			} else {
				path.State = PathStateActive
			}
		}
	}
}

func (o *EdgeNetworkOptimizer) enforceQoS() {
	// Enforce QoS policies
}

func (o *EdgeNetworkOptimizer) maintainP2PConnections() {
	now := time.Now()

	o.p2pManager.peers.Range(func(key, value interface{}) bool {
		peer := value.(*Peer)
		if now.Sub(peer.LastSeen) > 5*time.Minute {
			// Remove stale peer
			o.p2pManager.peers.Delete(key)
		}
		return true
	})
}

// Stop stops the network optimizer
func (o *EdgeNetworkOptimizer) Stop() {
	o.cancel()
	o.wg.Wait()
}