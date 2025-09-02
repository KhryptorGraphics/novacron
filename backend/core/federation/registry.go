package federation

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/hashicorp/mdns"
)

// DiscoveryManager handles node discovery and registration
type DiscoveryManagerImpl struct {
	config         *FederationConfig
	mdnsServer     *mdns.Server
	discoveredNodes map[string]*Node
	nodesMu        sync.RWMutex
	gossipManager  *GossipManager
	tlsConfig      *tls.Config
	logger         Logger
	stopCh         chan struct{}
	isRunning      bool
	runningMu      sync.RWMutex
}

// NewDiscoveryManager creates a new discovery manager
func NewDiscoveryManager(config *FederationConfig, logger Logger) (*DiscoveryManagerImpl, error) {
	dm := &DiscoveryManagerImpl{
		config:          config,
		discoveredNodes: make(map[string]*Node),
		logger:          logger,
		stopCh:          make(chan struct{}),
	}

	// Initialize TLS configuration for secure node communication
	if config.TLSConfig != nil {
		dm.tlsConfig = config.TLSConfig
	} else {
		// Create default TLS config with mutual TLS
		tlsConfig, err := dm.createDefaultTLSConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to create TLS config: %w", err)
		}
		dm.tlsConfig = tlsConfig
	}

	// Initialize gossip manager
	gossipManager, err := NewGossipManager(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create gossip manager: %w", err)
	}
	dm.gossipManager = gossipManager

	return dm, nil
}

// Start starts the discovery manager
func (dm *DiscoveryManagerImpl) Start(ctx context.Context) error {
	dm.runningMu.Lock()
	defer dm.runningMu.Unlock()

	if dm.isRunning {
		return fmt.Errorf("discovery manager already running")
	}

	dm.logger.Info("Starting discovery manager")

	// Start mDNS if enabled
	if dm.config.EnableMDNS {
		if err := dm.startMDNS(); err != nil {
			return fmt.Errorf("failed to start mDNS: %w", err)
		}
	}

	// Start gossip protocol if enabled
	if dm.config.EnableGossip {
		if err := dm.gossipManager.Start(ctx); err != nil {
			return fmt.Errorf("failed to start gossip: %w", err)
		}
	}

	// Start discovery loops
	go dm.discoveryLoop(ctx)
	go dm.verificationLoop(ctx)

	dm.isRunning = true
	dm.logger.Info("Discovery manager started")

	return nil
}

// Stop stops the discovery manager
func (dm *DiscoveryManagerImpl) Stop(ctx context.Context) error {
	dm.runningMu.Lock()
	defer dm.runningMu.Unlock()

	if !dm.isRunning {
		return fmt.Errorf("discovery manager not running")
	}

	dm.logger.Info("Stopping discovery manager")

	// Stop mDNS
	if dm.mdnsServer != nil {
		dm.mdnsServer.Shutdown()
	}

	// Stop gossip
	if dm.gossipManager != nil {
		_ = dm.gossipManager.Stop(ctx)
	}

	close(dm.stopCh)
	dm.isRunning = false

	dm.logger.Info("Discovery manager stopped")

	return nil
}

// DiscoverNodes discovers nodes in the federation
func (dm *DiscoveryManagerImpl) DiscoverNodes(ctx context.Context) ([]*Node, error) {
	var nodes []*Node

	// mDNS discovery
	if dm.config.EnableMDNS {
		mdnsNodes, err := dm.discoverMDNS(ctx)
		if err != nil {
			dm.logger.Warn("mDNS discovery failed", "error", err)
		} else {
			nodes = append(nodes, mdnsNodes...)
		}
	}

	// Gossip discovery
	if dm.config.EnableGossip {
		gossipNodes := dm.gossipManager.GetKnownNodes()
		nodes = append(nodes, gossipNodes...)
	}

	// Seed nodes from configuration
	for _, addr := range dm.config.JoinAddresses {
		node := &Node{
			Address: addr,
			State:   NodeStateUnknown,
		}
		nodes = append(nodes, node)
	}

	// Deduplicate and verify nodes
	verifiedNodes := dm.verifyAndDeduplicate(nodes)

	// Update discovered nodes
	dm.updateDiscoveredNodes(verifiedNodes)

	return verifiedNodes, nil
}

// RegisterNode registers a node in the federation
func (dm *DiscoveryManagerImpl) RegisterNode(ctx context.Context, node *Node) error {
	if node == nil {
		return fmt.Errorf("node is nil")
	}

	dm.logger.Info("Registering node", "node_id", node.ID, "address", node.Address)

	// Verify node trust
	if err := dm.verifyNodeTrust(ctx, node); err != nil {
		return fmt.Errorf("failed to verify node trust: %w", err)
	}

	// Exchange capabilities
	if err := dm.exchangeCapabilities(ctx, node); err != nil {
		return fmt.Errorf("failed to exchange capabilities: %w", err)
	}

	// Add to discovered nodes
	dm.nodesMu.Lock()
	dm.discoveredNodes[node.ID] = node
	dm.nodesMu.Unlock()

	// Broadcast via gossip
	if dm.config.EnableGossip {
		dm.gossipManager.BroadcastNodeJoin(node)
	}

	dm.logger.Info("Node registered successfully", "node_id", node.ID)

	return nil
}

// UnregisterNode unregisters a node from the federation
func (dm *DiscoveryManagerImpl) UnregisterNode(ctx context.Context, nodeID string) error {
	dm.logger.Info("Unregistering node", "node_id", nodeID)

	dm.nodesMu.Lock()
	delete(dm.discoveredNodes, nodeID)
	dm.nodesMu.Unlock()

	// Broadcast via gossip
	if dm.config.EnableGossip {
		dm.gossipManager.BroadcastNodeLeave(nodeID)
	}

	return nil
}

// GetDiscoveredNodes returns all discovered nodes
func (dm *DiscoveryManagerImpl) GetDiscoveredNodes() []*Node {
	dm.nodesMu.RLock()
	defer dm.nodesMu.RUnlock()

	nodes := make([]*Node, 0, len(dm.discoveredNodes))
	for _, node := range dm.discoveredNodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// mDNS discovery

func (dm *DiscoveryManagerImpl) startMDNS() error {
	// Create mDNS service
	info := []string{
		"node_id=" + dm.config.NodeID,
		"cluster_id=" + dm.config.ClusterID,
		"version=1.0.0",
	}

	service, err := mdns.NewMDNSService(
		dm.config.NodeID,
		"_novacron._tcp",
		"",
		"",
		5353,
		nil,
		info,
	)
	if err != nil {
		return fmt.Errorf("failed to create mDNS service: %w", err)
	}

	// Create mDNS server
	server, err := mdns.NewServer(&mdns.Config{Zone: service})
	if err != nil {
		return fmt.Errorf("failed to create mDNS server: %w", err)
	}

	dm.mdnsServer = server
	dm.logger.Info("mDNS discovery started")

	return nil
}

func (dm *DiscoveryManagerImpl) discoverMDNS(ctx context.Context) ([]*Node, error) {
	entries := make(chan *mdns.ServiceEntry, 10)
	var nodes []*Node

	// Start mDNS lookup
	go func() {
		defer close(entries)
		params := mdns.DefaultParams("_novacron._tcp")
		params.Entries = entries
		params.Timeout = 5 * time.Second
		_ = mdns.Query(params)
	}()

	// Collect discovered nodes
	for entry := range entries {
		node := dm.parseMDNSEntry(entry)
		if node != nil && node.ID != dm.config.NodeID {
			nodes = append(nodes, node)
		}
	}

	return nodes, nil
}

func (dm *DiscoveryManagerImpl) parseMDNSEntry(entry *mdns.ServiceEntry) *Node {
	if entry == nil {
		return nil
	}

	node := &Node{
		Name:     entry.Name,
		Address:  fmt.Sprintf("%s:%d", entry.AddrV4, entry.Port),
		State:    NodeStateDiscovering,
		Metadata: make(map[string]string),
	}

	// Parse info fields
	for _, info := range entry.InfoFields {
		if len(info) > 0 {
			// Parse key=value pairs
			for i := 0; i < len(info); i++ {
				if info[i] == '=' && i > 0 && i < len(info)-1 {
					key := string(info[:i])
					value := string(info[i+1:])
					
					switch key {
					case "node_id":
						node.ID = value
					case "cluster_id":
						node.ClusterID = value
					case "version":
						node.Version = value
					default:
						node.Metadata[key] = value
					}
					break
				}
			}
		}
	}

	return node
}

// Node verification and trust

func (dm *DiscoveryManagerImpl) verifyNodeTrust(ctx context.Context, node *Node) error {
	// Connect to node with TLS
	conn, err := tls.Dial("tcp", node.Address, dm.tlsConfig)
	if err != nil {
		return fmt.Errorf("failed to establish secure connection: %w", err)
	}
	defer conn.Close()

	// Verify certificate
	state := conn.ConnectionState()
	if len(state.PeerCertificates) == 0 {
		return fmt.Errorf("no peer certificates provided")
	}

	cert := state.PeerCertificates[0]

	// Verify certificate CN matches node ID
	if cert.Subject.CommonName != node.ID {
		return fmt.Errorf("certificate CN mismatch: expected %s, got %s", node.ID, cert.Subject.CommonName)
	}

	// Store TLS config for future connections
	node.TLSConfig = dm.tlsConfig

	return nil
}

func (dm *DiscoveryManagerImpl) exchangeCapabilities(ctx context.Context, node *Node) error {
	// Connect to node
	conn, err := net.DialTimeout("tcp", node.Address, 5*time.Second)
	if err != nil {
		return fmt.Errorf("failed to connect to node: %w", err)
	}
	defer conn.Close()

	// Send our capabilities
	localCapabilities := dm.getLocalCapabilities()
	encoder := json.NewEncoder(conn)
	if err := encoder.Encode(localCapabilities); err != nil {
		return fmt.Errorf("failed to send capabilities: %w", err)
	}

	// Receive remote capabilities
	decoder := json.NewDecoder(conn)
	var remoteCapabilities NodeCapabilities
	if err := decoder.Decode(&remoteCapabilities); err != nil {
		return fmt.Errorf("failed to receive capabilities: %w", err)
	}

	node.Capabilities = remoteCapabilities

	return nil
}

func (dm *DiscoveryManagerImpl) getLocalCapabilities() NodeCapabilities {
	// In production, would query actual system resources
	return NodeCapabilities{
		CPUCores:             8,
		MemoryGB:             32,
		StorageGB:            500,
		NetworkBandwidthMbps: 1000,
		Features:             []string{"vm", "container", "gpu", "sriov"},
		Resources: ResourceInventory{
			TotalCPU:     8,
			UsedCPU:      2,
			TotalMemory:  32 * 1024 * 1024 * 1024,
			UsedMemory:   8 * 1024 * 1024 * 1024,
			TotalStorage: 500 * 1024 * 1024 * 1024,
			UsedStorage:  100 * 1024 * 1024 * 1024,
			VMs:          5,
			Containers:   10,
			NetworkPools: 3,
		},
	}
}

// Discovery loops

func (dm *DiscoveryManagerImpl) discoveryLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-dm.stopCh:
			return
		case <-ticker.C:
			_, err := dm.DiscoverNodes(ctx)
			if err != nil {
				dm.logger.Error("Discovery failed", "error", err)
			}
		}
	}
}

func (dm *DiscoveryManagerImpl) verificationLoop(ctx context.Context) {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-dm.stopCh:
			return
		case <-ticker.C:
			dm.verifyNodes(ctx)
		}
	}
}

func (dm *DiscoveryManagerImpl) verifyNodes(ctx context.Context) {
	dm.nodesMu.RLock()
	nodes := make([]*Node, 0, len(dm.discoveredNodes))
	for _, node := range dm.discoveredNodes {
		nodes = append(nodes, node)
	}
	dm.nodesMu.RUnlock()

	for _, node := range nodes {
		if err := dm.verifyNodeTrust(ctx, node); err != nil {
			dm.logger.Warn("Node verification failed", "node_id", node.ID, "error", err)
			
			// Remove untrusted node
			dm.nodesMu.Lock()
			delete(dm.discoveredNodes, node.ID)
			dm.nodesMu.Unlock()
		}
	}
}

// Helper methods

func (dm *DiscoveryManagerImpl) updateDiscoveredNodes(nodes []*Node) {
	dm.nodesMu.Lock()
	defer dm.nodesMu.Unlock()

	for _, node := range nodes {
		if node.ID != "" {
			dm.discoveredNodes[node.ID] = node
		}
	}
}

func (dm *DiscoveryManagerImpl) verifyAndDeduplicate(nodes []*Node) []*Node {
	seen := make(map[string]bool)
	verified := make([]*Node, 0)

	for _, node := range nodes {
		// Skip if already seen
		if node.ID != "" && seen[node.ID] {
			continue
		}

		// Skip self
		if node.ID == dm.config.NodeID {
			continue
		}

		// Basic validation
		if node.Address == "" {
			continue
		}

		if node.ID != "" {
			seen[node.ID] = true
		}
		verified = append(verified, node)
	}

	return verified
}

func (dm *DiscoveryManagerImpl) createDefaultTLSConfig() (*tls.Config, error) {
	// In production, would load certificates from files or vault
	// For now, create a basic TLS config
	
	return &tls.Config{
		MinVersion:               tls.VersionTLS13,
		PreferServerCipherSuites: true,
		CipherSuites: []uint16{
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
		ClientAuth: tls.RequireAndVerifyClientCert,
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			// Custom verification logic
			if len(rawCerts) == 0 {
				return fmt.Errorf("no certificates provided")
			}
			
			// In production, would verify against CA and check revocation
			return nil
		},
	}, nil
}

// GossipManager handles gossip protocol for node discovery
type GossipManager struct {
	config      *FederationConfig
	knownNodes  map[string]*Node
	nodesMu     sync.RWMutex
	messageCh   chan *GossipMessage
	logger      Logger
	stopCh      chan struct{}
	isRunning   bool
}

// NewGossipManager creates a new gossip manager
func NewGossipManager(config *FederationConfig, logger Logger) (*GossipManager, error) {
	return &GossipManager{
		config:     config,
		knownNodes: make(map[string]*Node),
		messageCh:  make(chan *GossipMessage, 1000),
		logger:     logger,
		stopCh:     make(chan struct{}),
	}, nil
}

// Start starts the gossip manager
func (gm *GossipManager) Start(ctx context.Context) error {
	if gm.isRunning {
		return fmt.Errorf("gossip manager already running")
	}

	go gm.gossipLoop(ctx)
	
	gm.isRunning = true
	gm.logger.Info("Gossip manager started")

	return nil
}

// Stop stops the gossip manager
func (gm *GossipManager) Stop(ctx context.Context) error {
	if !gm.isRunning {
		return fmt.Errorf("gossip manager not running")
	}

	close(gm.stopCh)
	gm.isRunning = false

	return nil
}

// BroadcastNodeJoin broadcasts a node join event
func (gm *GossipManager) BroadcastNodeJoin(node *Node) {
	msg := &GossipMessage{
		ID:        generateMessageID(),
		Type:      GossipNodeJoin,
		Source:    gm.config.NodeID,
		Data:      node,
		TTL:       3,
		Timestamp: time.Now(),
	}

	gm.broadcast(msg)
}

// BroadcastNodeLeave broadcasts a node leave event
func (gm *GossipManager) BroadcastNodeLeave(nodeID string) {
	msg := &GossipMessage{
		ID:        generateMessageID(),
		Type:      GossipNodeLeave,
		Source:    gm.config.NodeID,
		Data:      nodeID,
		TTL:       3,
		Timestamp: time.Now(),
	}

	gm.broadcast(msg)
}

// GetKnownNodes returns all known nodes from gossip
func (gm *GossipManager) GetKnownNodes() []*Node {
	gm.nodesMu.RLock()
	defer gm.nodesMu.RUnlock()

	nodes := make([]*Node, 0, len(gm.knownNodes))
	for _, node := range gm.knownNodes {
		nodes = append(nodes, node)
	}

	return nodes
}

func (gm *GossipManager) gossipLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gm.stopCh:
			return
		case msg := <-gm.messageCh:
			gm.handleGossipMessage(msg)
		case <-ticker.C:
			gm.performGossipRound()
		}
	}
}

func (gm *GossipManager) handleGossipMessage(msg *GossipMessage) {
	// Decrement TTL
	msg.TTL--
	if msg.TTL <= 0 {
		return
	}

	switch msg.Type {
	case GossipNodeJoin:
		if node, ok := msg.Data.(*Node); ok {
			gm.nodesMu.Lock()
			gm.knownNodes[node.ID] = node
			gm.nodesMu.Unlock()
		}
	case GossipNodeLeave:
		if nodeID, ok := msg.Data.(string); ok {
			gm.nodesMu.Lock()
			delete(gm.knownNodes, nodeID)
			gm.nodesMu.Unlock()
		}
	}

	// Forward to other nodes
	gm.forward(msg)
}

func (gm *GossipManager) performGossipRound() {
	// Select random subset of nodes to gossip with
	// In production, would implement proper gossip protocol
	// For now, just maintain node list
}

func (gm *GossipManager) broadcast(msg *GossipMessage) {
	select {
	case gm.messageCh <- msg:
	default:
		gm.logger.Warn("Gossip message channel full")
	}
}

func (gm *GossipManager) forward(msg *GossipMessage) {
	// In production, would forward to random subset of peers
	// For now, just log
	gm.logger.Debug("Forwarding gossip message", "type", msg.Type, "ttl", msg.TTL)
}