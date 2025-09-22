package discovery

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"bytes"
	"log"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Constants for internet discovery
const (
	// DefaultBootstrapNodes are default nodes to connect to when first joining the network
	DefaultBootstrapNodes = "discovery.novacron.io:7700,discovery2.novacron.io:7700"

	// MaxPeerConnections is the maximum number of direct peer connections
	MaxPeerConnections = 100

	// RoutingTableSize is the size of the DHT routing table buckets
	RoutingTableSize = 20

	// DistanceCalculationBits affects the key space for peer distance calculation
	DistanceCalculationBits = 256 // SHA-256
)

// PeerInfo contains information about a peer in the network
type PeerInfo struct {
	NodeInfo
	PeerID          string                 `json:"peer_id"`           // SHA-256 hash of NodeID for DHT
	Endpoints       []string               `json:"endpoints"`         // List of IP:Port endpoints
	ExternalAddr    *ExternalEndpoint      `json:"external_addr"`     // NAT traversal external address
	PublicKey       string                 `json:"public_key"`        // Optional public key for authentication
	LastPing        time.Time              `json:"last_ping"`
	ConnectionType  string                 `json:"connection_type"`   // direct, nat_traversal, relay
	Quality         *ConnectionQuality     `json:"quality,omitempty"` // Connection quality metrics
	BandwidthCaps   map[string]uint64      `json:"bandwidth_caps"`    // Interface bandwidth capabilities
	NetworkTopology map[string]interface{} `json:"network_topology"`  // Network topology info
}

// RoutingTableBucket represents a k-bucket in the Kademlia DHT
type RoutingTableBucket struct {
	Peers []PeerInfo // Sorted by last ping time
	Mutex sync.RWMutex
}

// InternetDiscoveryConfig extends Config with internet discovery settings
type InternetDiscoveryConfig struct {
	Config
	BootstrapNodes     string        // Comma-separated list of bootstrap nodes
	EnableNATTraversal bool          // Whether to enable NAT traversal
	EnableDHT          bool          // Whether to enable DHT-based discovery
	EnableGossip       bool          // Whether to enable gossip protocol
	PingInterval       time.Duration // Interval between ping messages
	StunServers        []string      // STUN servers for NAT traversal
	ExternalPort       int           // External port to use for NAT traversal
}

// InternetDiscoveryService extends Service with internet discovery capabilities
type InternetDiscoveryService struct {
	*Service
	config           InternetDiscoveryConfig
	peerID           string // SHA-256 hash of NodeID
	routingTable     [DistanceCalculationBits]RoutingTableBucket
	peerEndpoints    map[string][]string // NodeID -> endpoints
	peerMutex        sync.RWMutex
	peerConnections  map[string]*PeerConnection // NodeID -> connection
	connectionsMutex sync.RWMutex
	httpServer       *http.Server
	natTraversal     *NATTraversalManager
	logger           *zap.Logger
}

// DefaultInternetDiscoveryConfig returns a default configuration for internet discovery
func DefaultInternetDiscoveryConfig() InternetDiscoveryConfig {
	return InternetDiscoveryConfig{
		Config:             DefaultConfig(),
		BootstrapNodes:     DefaultBootstrapNodes,
		EnableNATTraversal: true,
		EnableDHT:          true,
		EnableGossip:       true,
		PingInterval:       30 * time.Second,
		StunServers:        []string{"stun.l.google.com:19302", "stun1.l.google.com:19302"},
	}
}

// NewInternetDiscovery creates a new internet discovery service
func NewInternetDiscovery(config InternetDiscoveryConfig, logger *zap.Logger) (*InternetDiscoveryService, error) {
	baseService, err := New(config.Config)
	if err != nil {
		return nil, err
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	// Create peer ID from node ID
	h := sha256.New()
	h.Write([]byte(config.NodeID))
	peerID := hex.EncodeToString(h.Sum(nil))

	service := &InternetDiscoveryService{
		Service:         baseService,
		config:          config,
		peerID:          peerID,
		peerEndpoints:   make(map[string][]string),
		peerConnections: make(map[string]*PeerConnection),
		logger:          logger,
	}

	// Initialize NAT traversal if enabled
	if config.EnableNATTraversal {
		natConfig := &NATTraversalConfig{
			RefreshInterval: 5 * time.Minute,
			LocalPort:       config.Port,
			EnableRelay:     true,
		}
		
		for _, server := range config.StunServers {
			parts := strings.Split(server, ":")
			if len(parts) == 2 {
				port := 19302 // Default STUN port
				if p, err := strconv.Atoi(parts[1]); err == nil {
					port = p
				}
				natConfig.STUNServers = append(natConfig.STUNServers, STUNServer{
					Host: parts[0],
					Port: port,
				})
			}
		}

		natTraversal, err := NewNATTraversalManager(natConfig, logger)
		if err != nil {
			logger.Warn("Failed to create NAT traversal manager", zap.Error(err))
		} else {
			service.natTraversal = natTraversal
		}
	}

	return service, nil
}

// Start starts the internet discovery service
func (s *InternetDiscoveryService) Start() error {
	// Start the base discovery service
	if err := s.Service.Start(); err != nil {
		return err
	}

	// Start NAT traversal if enabled
	if s.natTraversal != nil {
		if err := s.natTraversal.Start(); err != nil {
			s.logger.Warn("Failed to start NAT traversal", zap.Error(err))
		} else {
			s.logger.Info("NAT traversal started successfully")
		}
	}

	// Start the HTTP discovery server for direct connections
	go s.startHTTPServer()

	// Connect to bootstrap nodes
	if s.config.BootstrapNodes != "" {
		go s.connectToBootstrapNodes()
	}

	// Start periodic maintenance tasks
	go s.maintenanceLoop()

	s.logger.Info("Internet discovery service started", zap.String("peer_id", s.peerID))

	return nil
}

// Stop stops the internet discovery service
func (s *InternetDiscoveryService) Stop() error {
	// Close all peer connections
	s.connectionsMutex.Lock()
	for _, peerConn := range s.peerConnections {
		if peerConn.conn != nil {
			peerConn.conn.Close()
		}
	}
	s.connectionsMutex.Unlock()

	// Stop NAT traversal
	if s.natTraversal != nil {
		s.natTraversal.Stop()
	}

	// Stop the HTTP server
	if s.httpServer != nil {
		// Give 5 seconds for graceful shutdown
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		s.httpServer.Shutdown(ctx)
	}

	s.logger.Info("Internet discovery service stopped")

	// Stop the base discovery service
	return s.Service.Stop()
}

// FindPeer looks for a peer with the given ID
func (s *InternetDiscoveryService) FindPeer(id string) (PeerInfo, bool) {
	s.peerMutex.RLock()
	defer s.peerMutex.RUnlock()

	// First, check if we have this peer in our direct connections
	nodeInfo, exists := s.GetNodeByID(id)
	if exists {
		peerInfo := PeerInfo{
			NodeInfo:        nodeInfo,
			PeerID:          calculatePeerID(id),
			Endpoints:       s.peerEndpoints[id],
			BandwidthCaps:   make(map[string]uint64),
			NetworkTopology: make(map[string]interface{}),
		}
		
		// Add connection quality if we have an active connection
		s.connectionsMutex.RLock()
		if peerConn, exists := s.peerConnections[id]; exists {
			peerInfo.Quality = &peerConn.Quality
			peerInfo.ConnectionType = peerConn.ConnectionType
		}
		s.connectionsMutex.RUnlock()
		
		return peerInfo, true
	}

	// Implement DHT lookup for peers we don't directly know
	return s.performDHTLookup(id)
}

// performDHTLookup performs a DHT lookup to find a peer
func (s *InternetDiscoveryService) performDHTLookup(targetID string) (PeerInfo, bool) {
	targetPeerID := calculatePeerID(targetID)
	
	// Find the closest peers we know to the target
	closestPeers := s.findClosestPeers(targetPeerID, 3)
	
	for _, peer := range closestPeers {
		// Query each peer for the target
		if result := s.queryPeerForTarget(peer, targetID); result != nil {
			return *result, true
		}
	}
	
	return PeerInfo{}, false
}

// findClosestPeers finds the k closest peers to a target peer ID
func (s *InternetDiscoveryService) findClosestPeers(targetPeerID string, k int) []PeerInfo {
	var allPeers []PeerInfo
	
	// Collect all peers from routing table
	for i := 0; i < DistanceCalculationBits; i++ {
		bucket := &s.routingTable[i]
		bucket.Mutex.RLock()
		for _, peer := range bucket.Peers {
			allPeers = append(allPeers, peer)
		}
		bucket.Mutex.RUnlock()
	}
	
	// Sort by distance to target
	sortPeersByDistance(allPeers, targetPeerID)
	
	if len(allPeers) > k {
		return allPeers[:k]
	}
	return allPeers
}

// queryPeerForTarget queries a specific peer for information about a target
func (s *InternetDiscoveryService) queryPeerForTarget(peer PeerInfo, targetID string) *PeerInfo {
	// This would send a "FIND_NODE" message to the peer
	// For now, return nil as this is a complex DHT operation
	s.logger.Debug("DHT lookup query", 
		zap.String("peer", peer.ID),
		zap.String("target", targetID))
	return nil
}

// ConnectToPeer establishes a direct connection to a peer
func (s *InternetDiscoveryService) ConnectToPeer(peer PeerInfo) error {
	s.connectionsMutex.Lock()
	_, exists := s.peerConnections[peer.ID]
	s.connectionsMutex.Unlock()

	if exists {
		return nil // Already connected
	}

	var peerConn *PeerConnection

	// Strategy 1: Try direct TCP connection to each endpoint
	for _, endpoint := range peer.Endpoints {
		conn, connErr := net.DialTimeout("tcp", endpoint, 10*time.Second)
		if connErr == nil {
			s.logger.Debug("Direct connection established", 
				zap.String("peer", peer.ID),
				zap.String("endpoint", endpoint))
			
			// Parse endpoint as TCP address for direct connections
			tcpAddr, _ := net.ResolveTCPAddr("tcp", endpoint)

			peerConn = &PeerConnection{
				PeerID:         peer.ID,
				LocalEndpoint:  nil, // Will be filled by connection details
				RemoteEndpoint: nil, // No UDP endpoint for direct TCP
				RemoteTCPAddr:  tcpAddr, // Store TCP address separately
				ConnectionType: "direct",
				Established:    true,
				LastActivity:   time.Now(),
				conn:           conn,  // net.Conn interface already
			}
			break
		}
		s.logger.Debug("Failed to connect to endpoint", 
			zap.String("endpoint", endpoint),
			zap.Error(connErr))
	}

	// Strategy 2: Try NAT traversal if direct connection failed
	if peerConn == nil && s.natTraversal != nil && peer.ExternalAddr != nil {
		s.logger.Debug("Attempting NAT traversal connection", zap.String("peer", peer.ID))
		
		remoteAddr := &net.UDPAddr{
			IP:   peer.ExternalAddr.IP,
			Port: peer.ExternalAddr.Port,
		}
		
		traversalConn, traversalErr := s.natTraversal.EstablishP2PConnection(peer.ID, remoteAddr)
		if traversalErr == nil {
			s.logger.Info("NAT traversal connection established", zap.String("peer", peer.ID))
			peerConn = traversalConn
		} else {
			s.logger.Debug("NAT traversal failed", 
				zap.String("peer", peer.ID),
				zap.Error(traversalErr))
		}
	}

	// Strategy 3: Relay connection (fallback)
	if peerConn == nil {
		s.logger.Debug("All direct connections failed, connection not established", zap.String("peer", peer.ID))
		return fmt.Errorf("failed to connect to peer %s: all connection strategies failed", peer.ID)
	}

	// Register the connection
	s.connectionsMutex.Lock()
	s.peerConnections[peer.ID] = peerConn
	s.connectionsMutex.Unlock()

	// Start handling messages from this peer
	go s.handlePeerConnection(peer.ID, peerConn.conn)

	// Start connection quality monitoring
	go s.monitorConnectionQuality(peer.ID)

	s.logger.Info("Peer connection established", 
		zap.String("peer", peer.ID),
		zap.String("connection_type", peerConn.ConnectionType))

	return nil
}

// parseUDPAddr parses a string address into a UDP address
func parseUDPAddr(addr string) *net.UDPAddr {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil
	}
	return udpAddr
}

// monitorConnectionQuality continuously monitors connection quality
func (s *InternetDiscoveryService) monitorConnectionQuality(peerID string) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		s.connectionsMutex.RLock()
		peerConn, exists := s.peerConnections[peerID]
		s.connectionsMutex.RUnlock()

		if !exists || !peerConn.Established {
			return
		}

		// Measure RTT with standardized ping
		start := time.Now()
		if peerConn.conn != nil {
			pingMsg := fmt.Sprintf("{\"type\":\"PING\",\"ts\":%d}\n", start.UnixNano())
			_, err := peerConn.conn.Write([]byte(pingMsg))
			if err != nil {
				s.logger.Warn("Failed to ping peer", 
					zap.String("peer", peerID),
					zap.Error(err))
				s.handlePeerDisconnect(peerID)
				return
			}

			// Wait for response
			buffer := make([]byte, 1024)
			peerConn.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
			n, err := peerConn.conn.Read(buffer)
			if err != nil {
				s.logger.Warn("Ping timeout or error", 
					zap.String("peer", peerID),
					zap.Error(err))
				continue
			}
			
			response := string(buffer[:n])
			if !strings.Contains(response, "PONG") {
				s.logger.Warn("Invalid ping response", 
					zap.String("peer", peerID),
					zap.String("response", response))
				continue
			}

			rtt := time.Since(start)

			// Update connection quality
			s.connectionsMutex.Lock()
			peerConn.Quality.RTT = rtt
			peerConn.Quality.LastMeasured = time.Now()
			peerConn.LastActivity = time.Now()
			s.connectionsMutex.Unlock()

			s.logger.Debug("Connection quality updated", 
				zap.String("peer", peerID),
				zap.Duration("rtt", rtt))
		}
	}
}

// handlePeerDisconnect handles peer disconnection
func (s *InternetDiscoveryService) handlePeerDisconnect(peerID string) {
	s.connectionsMutex.Lock()
	if peerConn, exists := s.peerConnections[peerID]; exists {
		if peerConn.conn != nil {
			peerConn.conn.Close()
		}
		delete(s.peerConnections, peerID)
	}
	s.connectionsMutex.Unlock()

	// Mark node as unavailable
	s.nodesMutex.Lock()
	if node, exists := s.nodes[peerID]; exists {
		node.Available = false
		s.nodes[peerID] = node

		// Notify listeners
		for _, listener := range s.listeners {
			listener(EventNodeLeft, node)
		}
	}
	s.nodesMutex.Unlock()

	s.logger.Info("Peer disconnected", zap.String("peer", peerID))
}

// GetPeerConnectionInfo returns connection information for a peer
func (s *InternetDiscoveryService) GetPeerConnectionInfo(peerID string) (*PeerConnection, bool) {
	s.connectionsMutex.RLock()
	defer s.connectionsMutex.RUnlock()
	
	peerConn, exists := s.peerConnections[peerID]
	return peerConn, exists
}

// GetNetworkTopologyInfo returns network topology information for scheduling
func (s *InternetDiscoveryService) GetNetworkTopologyInfo() map[string]interface{} {
	topology := make(map[string]interface{})
	
	// Add external address if available
	if s.natTraversal != nil {
		if external := s.natTraversal.GetExternalEndpoint(); external != nil {
			topology["external_ip"] = external.IP.String()
			topology["external_port"] = external.Port
			topology["nat_type"] = s.natTraversal.GetNATType()
		}
	}
	
	// Add peer connection information
	s.connectionsMutex.RLock()
	peerStats := make(map[string]interface{})
	for peerID, peerConn := range s.peerConnections {
		peerStats[peerID] = map[string]interface{}{
			"connection_type": peerConn.ConnectionType,
			"rtt":            peerConn.Quality.RTT.Milliseconds(),
			"established":    peerConn.Established,
			"last_activity":  peerConn.LastActivity,
		}
	}
	s.connectionsMutex.RUnlock()
	
	topology["peer_connections"] = peerStats
	topology["active_connections"] = len(s.peerConnections)
	topology["peer_id"] = s.peerID
	
	return topology
}

// GetBandwidthCapabilities returns bandwidth capabilities for network-aware scheduling
func (s *InternetDiscoveryService) GetBandwidthCapabilities() map[string]uint64 {
	capabilities := make(map[string]uint64)
	
	s.connectionsMutex.RLock()
	defer s.connectionsMutex.RUnlock()
	
	for peerID, peerConn := range s.peerConnections {
		if peerConn.Quality.BandwidthEstimate > 0 {
			capabilities[peerID] = peerConn.Quality.BandwidthEstimate
		} else {
			// Default estimate based on connection type
			switch peerConn.ConnectionType {
			case "direct":
				capabilities[peerID] = 1_000_000_000 // 1 Gbps default
			case "nat_traversal":
				capabilities[peerID] = 100_000_000   // 100 Mbps default
			case "relay":
				capabilities[peerID] = 10_000_000    // 10 Mbps default
			}
		}
	}
	
	return capabilities
}

// startHTTPServer starts the HTTP discovery server
func (s *InternetDiscoveryService) startHTTPServer() {
	mux := http.NewServeMux()

	// Handler for peer discovery
	mux.HandleFunc("/discover", func(w http.ResponseWriter, r *http.Request) {
		remoteAddr := r.RemoteAddr

		// Extract request information (peer announcing itself)
		var peerInfo PeerInfo
		decoder := json.NewDecoder(r.Body)
		if err := decoder.Decode(&peerInfo); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Prefer provided endpoints from the request over remote address
		providedEndpoints := peerInfo.Endpoints
		if len(providedEndpoints) == 0 {
			// Parse remote address to extract IP and use service port
			if host, _, err := net.SplitHostPort(remoteAddr); err == nil {
				// Build endpoint with the service port (not ephemeral port)
				endpoint := net.JoinHostPort(host, strconv.Itoa(s.config.Port))
				if !contains(providedEndpoints, endpoint) {
					peerInfo.Endpoints = append(providedEndpoints, endpoint)
				}
			} else {
				// Fallback: if parsing fails, try to use the address with service port
				s.logger.Warn("Failed to parse remote address",
					zap.String("remote_addr", remoteAddr),
					zap.Error(err))
				// Try to extract just the host and add with service port
				host := remoteAddr
				if idx := strings.LastIndex(remoteAddr, ":"); idx >= 0 {
					host = remoteAddr[:idx]
				}
				endpoint := net.JoinHostPort(host, strconv.Itoa(s.config.Port))
				if !contains(providedEndpoints, endpoint) {
					peerInfo.Endpoints = append(providedEndpoints, endpoint)
				}
			}
		}

		// Update our knowledge of this peer
		s.updatePeer(peerInfo)

		// Respond with our peers (up to a limit)
		peers := s.getRandomPeers(20)

		w.Header().Set("Content-Type", "application/json")
		encoder := json.NewEncoder(w)
		if err := encoder.Encode(peers); err != nil {
			http.Error(w, "Error encoding response", http.StatusInternalServerError)
			return
		}
	})

	// Handler for ping requests
	mux.HandleFunc("/ping", func(w http.ResponseWriter, r *http.Request) {
		// Simple ping-pong for connection testing
		w.Write([]byte("pong"))
	})

	// Start the HTTP server
	serverAddr := fmt.Sprintf(":%d", s.config.Port)
	s.httpServer = &http.Server{
		Addr:    serverAddr,
		Handler: mux,
	}

	log.Printf("Starting HTTP discovery server on %s", serverAddr)
	if err := s.httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("HTTP server error: %v", err)
	}
}

// connectToBootstrapNodes connects to the configured bootstrap nodes
func (s *InternetDiscoveryService) connectToBootstrapNodes() {
	bootstrapNodes := strings.Split(s.config.BootstrapNodes, ",")

	for _, node := range bootstrapNodes {
		node = strings.TrimSpace(node)
		if node == "" {
			continue
		}

		// Connect to bootstrap node over HTTP
		url := fmt.Sprintf("http://%s/discover", node)

		// Prepare our node info to announce
		selfPeer := PeerInfo{
			NodeInfo: NodeInfo{
				ID:        s.config.NodeID,
				Name:      s.config.NodeName,
				Role:      s.config.NodeRole,
				Address:   s.config.Address,
				Port:      s.config.Port,
				Tags:      s.config.Tags,
				JoinedAt:  time.Now(),
				LastSeen:  time.Now(),
				Available: true,
			},
			PeerID:    s.peerID,
			Endpoints: []string{fmt.Sprintf("%s:%d", s.config.Address, s.config.Port)},
		}

		// Perform discovery request
		s.performDiscoveryRequest(url, selfPeer)
	}
}

// performDiscoveryRequest sends a discovery request to a bootstrap node
func (s *InternetDiscoveryService) performDiscoveryRequest(url string, selfPeer PeerInfo) {
	// Propagate external endpoint if NAT traversal is enabled
	if s.natTraversal != nil {
		if externalEndpoint := s.natTraversal.GetExternalEndpoint(); externalEndpoint != nil {
			selfPeer.ExternalAddr = externalEndpoint
			// Also add external endpoint to the list of endpoints
			externalAddr := fmt.Sprintf("%s:%d", externalEndpoint.IP.String(), externalEndpoint.Port)
			if !contains(selfPeer.Endpoints, externalAddr) {
				selfPeer.Endpoints = append(selfPeer.Endpoints, externalAddr)
			}
		}
	}
	
	// Convert our node info to JSON
	body, err := json.Marshal(selfPeer)
	if err != nil {
		log.Printf("Error marshaling node info: %v", err)
		return
	}

	// Send HTTP POST request to bootstrap node
	req, err := http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		log.Printf("Error creating HTTP request: %v", err)
		return
	}

	req.Header.Add("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error connecting to bootstrap node %s: %v", url, err)
		return
	}
	defer resp.Body.Close()

	// Parse response
	var peers []PeerInfo
	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&peers); err != nil {
		log.Printf("Error decoding response from bootstrap node: %v", err)
		return
	}

	// Process discovered peers
	for _, peer := range peers {
		// Skip ourselves
		if peer.ID == s.config.NodeID {
			continue
		}

		// Update our knowledge of this peer
		s.updatePeer(peer)

		// Attempt to connect to the peer
		go s.ConnectToPeer(peer)
	}
}

// updatePeer updates our knowledge of a peer
func (s *InternetDiscoveryService) updatePeer(peer PeerInfo) {
	// Update node info
	nodeInfo := peer.NodeInfo
	nodeInfo.LastSeen = time.Now()
	nodeInfo.Available = true

	// Update routing table
	bucket := s.getBucketForPeer(peer.PeerID)

	bucket.Mutex.Lock()
	defer bucket.Mutex.Unlock()

	// Check if peer is already in bucket
	for i, existingPeer := range bucket.Peers {
		if existingPeer.ID == peer.ID {
			// Update existing peer
			bucket.Peers[i].LastSeen = time.Now()
			bucket.Peers[i].Endpoints = peer.Endpoints

			// Move to the end of the bucket (most recently seen)
			updated := bucket.Peers[i]
			bucket.Peers = append(bucket.Peers[:i], bucket.Peers[i+1:]...)
			bucket.Peers = append(bucket.Peers, updated)

			// Update service node map
			s.nodesMutex.Lock()
			s.nodes[peer.ID] = nodeInfo
			s.nodesMutex.Unlock()

			// Update endpoints
			s.peerMutex.Lock()
			s.peerEndpoints[peer.ID] = peer.Endpoints
			s.peerMutex.Unlock()

			return
		}
	}

	// Add peer to bucket if not full
	if len(bucket.Peers) < RoutingTableSize {
		bucket.Peers = append(bucket.Peers, peer)

		// Update service node map
		s.nodesMutex.Lock()
		s.nodes[peer.ID] = nodeInfo
		s.nodesMutex.Unlock()

		// Update endpoints
		s.peerMutex.Lock()
		s.peerEndpoints[peer.ID] = peer.Endpoints
		s.peerMutex.Unlock()

		// Notify listeners
		for _, listener := range s.listeners {
			listener(EventNodeJoined, nodeInfo)
		}
	} else {
		// Bucket is full, try to ping oldest peer
		oldestPeer := bucket.Peers[0]

		// In a real implementation, we would ping the oldest peer
		// and replace it if it doesn't respond

		// For simplicity, we just replace it here
		bucket.Peers = bucket.Peers[1:]
		bucket.Peers = append(bucket.Peers, peer)

		// Fix: Capture node info BEFORE deletion to avoid race condition
		s.nodesMutex.Lock()
		oldestNodeInfo, oldestExists := s.nodes[oldestPeer.ID]
		if oldestExists {
			delete(s.nodes, oldestPeer.ID)
		}
		s.nodes[peer.ID] = nodeInfo
		s.nodesMutex.Unlock()

		// Update endpoints
		s.peerMutex.Lock()
		delete(s.peerEndpoints, oldestPeer.ID)
		s.peerEndpoints[peer.ID] = peer.Endpoints
		s.peerMutex.Unlock()

		// Notify listeners with the captured node info to avoid race
		for _, listener := range s.listeners {
			if oldestExists {
				listener(EventNodeLeft, oldestNodeInfo)
			}
			listener(EventNodeJoined, nodeInfo)
		}
	}
}

// getBucketForPeer determines the appropriate k-bucket for a peer
func (s *InternetDiscoveryService) getBucketForPeer(peerID string) *RoutingTableBucket {
	// Calculate distance between our peer ID and the target peer ID
	// This is a simple XOR-based distance calculation for the Kademlia DHT

	// Convert hex string to byte array
	peerIdBytes, _ := hex.DecodeString(peerID)
	ourPeerIdBytes, _ := hex.DecodeString(s.peerID)

	// Find the first bit position where the two IDs differ
	bucketIndex := 0
	for i := 0; i < len(peerIdBytes) && i < len(ourPeerIdBytes); i++ {
		xor := peerIdBytes[i] ^ ourPeerIdBytes[i]
		if xor == 0 {
			bucketIndex += 8
			continue
		}

		// Find the position of the highest bit set in xor
		for j := 7; j >= 0; j-- {
			if (xor & (1 << j)) != 0 {
				bucketIndex += j
				break
			}
		}
		break
	}

	if bucketIndex >= DistanceCalculationBits {
		bucketIndex = DistanceCalculationBits - 1
	}

	return &s.routingTable[bucketIndex]
}

// getRandomPeers returns a random selection of peers
func (s *InternetDiscoveryService) getRandomPeers(count int) []PeerInfo {
	var allPeers []PeerInfo

	// Collect peers from all buckets
	for i := 0; i < DistanceCalculationBits; i++ {
		bucket := &s.routingTable[i]
		bucket.Mutex.RLock()

		for _, peer := range bucket.Peers {
			allPeers = append(allPeers, peer)
		}

		bucket.Mutex.RUnlock()
	}

	// If we have fewer peers than requested, return all
	if len(allPeers) <= count {
		return allPeers
	}

	// Shuffle peers and return the requested count
	shufflePeers(allPeers)
	return allPeers[:count]
}

// handlePeerConnection handles messages from a peer connection
func (s *InternetDiscoveryService) handlePeerConnection(id string, conn net.Conn) {
	// Set a read deadline to avoid hanging forever
	conn.SetReadDeadline(time.Now().Add(1 * time.Hour))

	buffer := make([]byte, 4096)
	for {
		n, err := conn.Read(buffer)
		if err != nil {
			log.Printf("Error reading from peer %s: %v", id, err)
			break
		}

		// Process message using standardized protocol
		message := string(buffer[:n])
		lines := strings.Split(message, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			
			// Handle standardized messages
			if strings.Contains(line, "\"type\":\"PING\"") {
				// Extract timestamp and send PONG
				start := strings.Index(line, "\"ts\":")
				if start >= 0 {
					start += 5
					end := strings.Index(line[start:], "}")
					if end > 0 {
						ts := line[start:start+end]
						pongMsg := fmt.Sprintf("{\"type\":\"PONG\",\"ts\":%s}\n", ts)
						conn.Write([]byte(pongMsg))
					}
				}
			} else if strings.Contains(line, "\"type\":\"HANDSHAKE\"") {
				// Reply with HANDSHAKE_ACK
				ackMsg := "{\"type\":\"HANDSHAKE_ACK\"}\n"
				conn.Write([]byte(ackMsg))
			} else {
				// Log other messages
				log.Printf("Received message from peer %s: %s", id, line)
			}
		}
	}

	// Close and clean up connection
	conn.Close()

	s.connectionsMutex.Lock()
	delete(s.peerConnections, id)
	s.connectionsMutex.Unlock()

	log.Printf("Connection to peer %s closed", id)
}

// maintenanceLoop runs periodic maintenance tasks
func (s *InternetDiscoveryService) maintenanceLoop() {
	pingTicker := time.NewTicker(s.config.PingInterval)
	defer pingTicker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-pingTicker.C:
			s.pingAllPeers()
			s.refreshRoutes()
		}
	}
}

// pingAllPeers sends a ping to all connected peers
func (s *InternetDiscoveryService) pingAllPeers() {
	s.connectionsMutex.RLock()
	peersCopy := make(map[string]*PeerConnection, len(s.peerConnections))
	for id, pc := range s.peerConnections {
		peersCopy[id] = pc
	}
	s.connectionsMutex.RUnlock()

	for id, pc := range peersCopy {
		if pc.conn == nil {
			continue
		}
		// Standardized ping message
		pingMsg := fmt.Sprintf("{\"type\":\"PING\",\"ts\":%d}\n", time.Now().UnixNano())
		_, err := pc.conn.Write([]byte(pingMsg))
		if err != nil {
			log.Printf("Error pinging peer %s: %v", id, err)

			// Close and remove the connection
			pc.conn.Close()

			s.connectionsMutex.Lock()
			delete(s.peerConnections, id)
			s.connectionsMutex.Unlock()

			// Mark node as unavailable
			s.nodesMutex.Lock()
			node, exists := s.nodes[id]
			if exists {
				node.Available = false
				s.nodes[id] = node

				// Notify listeners
				for _, listener := range s.listeners {
					listener(EventNodeLeft, node)
				}
			}
			s.nodesMutex.Unlock()
		}
	}
}

// refreshRoutes refreshes the routing table and connections
func (s *InternetDiscoveryService) refreshRoutes() {
	s.logger.Debug("Refreshing routing tables")

	// Implement DHT routing table refresh
	for i := 0; i < DistanceCalculationBits; i++ {
		bucket := &s.routingTable[i]
		bucket.Mutex.RLock()
		peerCount := len(bucket.Peers)
		bucket.Mutex.RUnlock()

		// If bucket is not full, try to find more peers
		if peerCount < RoutingTableSize {
			// Generate a random ID in this bucket's range
			randomID := generateRandomIDInBucketRange(s.peerID, i)
			
			// Perform a find node operation for this ID
			s.performDHTLookup(randomID)
		}
		
		// Ping oldest peers in the bucket to ensure they're still alive
		bucket.Mutex.Lock()
		if len(bucket.Peers) > 0 {
			oldestPeer := bucket.Peers[0]
			
			// Check if we have a connection to this peer
			s.connectionsMutex.RLock()
			_, hasConnection := s.peerConnections[oldestPeer.ID]
			s.connectionsMutex.RUnlock()
			
			if !hasConnection {
				// Try to reconnect to the peer
				go s.ConnectToPeer(oldestPeer)
			}
		}
		bucket.Mutex.Unlock()
	}
}

// generateRandomIDInBucketRange generates a random node ID that would fall into the specified bucket
func generateRandomIDInBucketRange(ourPeerID string, bucketIndex int) string {
	// This is a simplified implementation
	// In practice, you'd need to generate an ID that differs from ourPeerID
	// at exactly the bit position specified by bucketIndex
	
	// For now, just generate a random ID
	randomBytes := make([]byte, 32)
	rand.Read(randomBytes)
	return hex.EncodeToString(randomBytes)
}

// Helper functions

// calculatePeerID generates a peer ID from a node ID
func calculatePeerID(nodeID string) string {
	h := sha256.New()
	h.Write([]byte(nodeID))
	return hex.EncodeToString(h.Sum(nil))
}

// shufflePeers randomly shuffles a slice of peers
func shufflePeers(peers []PeerInfo) {
	// Fisher-Yates shuffle with crypto/rand for better randomness
	for i := len(peers) - 1; i > 0; i-- {
		// Generate random bytes
		b := make([]byte, 8)
		rand.Read(b)

		// Convert to a number and get modulo
		j := int(binary.BigEndian.Uint64(b) % uint64(i+1))

		// Swap elements at indices i and j
		peers[i], peers[j] = peers[j], peers[i]
	}
}

// contains checks if a string is in a slice
func contains(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

// sortPeersByDistance sorts peers by XOR distance to a target peer ID
func sortPeersByDistance(peers []PeerInfo, targetPeerID string) {
	targetBytes, _ := hex.DecodeString(targetPeerID)
	
	// Use sort.Slice for efficient sorting
	sort.Slice(peers, func(i, j int) bool {
		peerBytesI, _ := hex.DecodeString(peers[i].PeerID)
		peerBytesJ, _ := hex.DecodeString(peers[j].PeerID)
		
		distanceI := make([]byte, len(targetBytes))
		distanceJ := make([]byte, len(targetBytes))
		
		for k := 0; k < len(targetBytes); k++ {
			if k < len(peerBytesI) {
				distanceI[k] = targetBytes[k] ^ peerBytesI[k]
			}
			if k < len(peerBytesJ) {
				distanceJ[k] = targetBytes[k] ^ peerBytesJ[k]
			}
		}
		
		return bytes.Compare(distanceI, distanceJ) < 0
	})
}

// compareDistance compares two XOR distances
// Returns -1 if a < b, 0 if a == b, 1 if a > b
func compareDistance(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] < b[i] {
			return -1
		} else if a[i] > b[i] {
			return 1
		}
	}
	
	if len(a) < len(b) {
		return -1
	} else if len(a) > len(b) {
		return 1
	}
	
	return 0
}
