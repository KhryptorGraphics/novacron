package discovery

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
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
	PeerID    string    `json:"peer_id"`    // SHA-256 hash of NodeID for DHT
	Endpoints []string  `json:"endpoints"`  // List of IP:Port endpoints
	PublicKey string    `json:"public_key"` // Optional public key for authentication
	LastPing  time.Time `json:"last_ping"`
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
	peerConnections  map[string]net.Conn // NodeID -> connection
	connectionsMutex sync.RWMutex
	httpServer       *http.Server
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
func NewInternetDiscovery(config InternetDiscoveryConfig) (*InternetDiscoveryService, error) {
	baseService, err := New(config.Config)
	if err != nil {
		return nil, err
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
		peerConnections: make(map[string]net.Conn),
	}

	return service, nil
}

// Start starts the internet discovery service
func (s *InternetDiscoveryService) Start() error {
	// Start the base discovery service
	if err := s.Service.Start(); err != nil {
		return err
	}

	// Determine our external IP and port via STUN if NAT traversal is enabled
	if s.config.EnableNATTraversal {
		go s.setupNATTraversal()
	}

	// Start the HTTP discovery server for direct connections
	go s.startHTTPServer()

	// Connect to bootstrap nodes
	if s.config.BootstrapNodes != "" {
		go s.connectToBootstrapNodes()
	}

	// Start periodic maintenance tasks
	go s.maintenanceLoop()

	log.Printf("Internet discovery service started, peer ID: %s", s.peerID)

	return nil
}

// Stop stops the internet discovery service
func (s *InternetDiscoveryService) Stop() error {
	// Close all peer connections
	s.connectionsMutex.Lock()
	for _, conn := range s.peerConnections {
		conn.Close()
	}
	s.connectionsMutex.Unlock()

	// Stop the HTTP server
	if s.httpServer != nil {
		// Give 5 seconds for graceful shutdown
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		s.httpServer.Shutdown(ctx)
	}

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
		return PeerInfo{
			NodeInfo:  nodeInfo,
			PeerID:    calculatePeerID(id),
			Endpoints: s.peerEndpoints[id],
		}, true
	}

	// TODO: Implement DHT lookup for peers we don't directly know

	return PeerInfo{}, false
}

// ConnectToPeer establishes a direct connection to a peer
func (s *InternetDiscoveryService) ConnectToPeer(peer PeerInfo) error {
	s.connectionsMutex.Lock()
	_, exists := s.peerConnections[peer.ID]
	s.connectionsMutex.Unlock()

	if exists {
		return nil // Already connected
	}

	// Try each endpoint until we succeed
	var conn net.Conn
	var err error

	for _, endpoint := range peer.Endpoints {
		// Attempt to establish a TCP connection
		conn, err = net.DialTimeout("tcp", endpoint, 10*time.Second)
		if err == nil {
			break
		}
		log.Printf("Failed to connect to endpoint %s: %v", endpoint, err)
	}

	if err != nil {
		return fmt.Errorf("failed to connect to peer %s: %v", peer.ID, err)
	}

	// Register the connection
	s.connectionsMutex.Lock()
	s.peerConnections[peer.ID] = conn
	s.connectionsMutex.Unlock()

	// Start handling messages from this peer
	go s.handlePeerConnection(peer.ID, conn)

	return nil
}

// setupNATTraversal uses STUN to determine our external IP and port
func (s *InternetDiscoveryService) setupNATTraversal() {
	// TODO: Implement STUN client to determine external IP and port
	// This is a simplified placeholder

	for _, stunServer := range s.config.StunServers {
		// Attempt to connect to STUN server and get external IP
		log.Printf("Attempting NAT traversal via STUN server: %s", stunServer)

		// In a real implementation, this would use the STUN protocol to discover
		// the external IP address and update the node's information
	}
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

		// Add remote address to peer endpoints if not already there
		endpoint := remoteAddr
		if !contains(peerInfo.Endpoints, endpoint) {
			peerInfo.Endpoints = append(peerInfo.Endpoints, endpoint)
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
func (s *InternetDiscoveryService) performDiscoveryRequest(url string, self PeerInfo) {
	// Convert our node info to JSON
	body, err := json.Marshal(self)
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
			bucket.Peers = append(bucket.Peers[:i], bucket.Peers[i+1:]...)
			bucket.Peers = append(bucket.Peers, existingPeer)

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

		// Update service node map
		s.nodesMutex.Lock()
		delete(s.nodes, oldestPeer.ID)
		s.nodes[peer.ID] = nodeInfo
		s.nodesMutex.Unlock()

		// Update endpoints
		s.peerMutex.Lock()
		delete(s.peerEndpoints, oldestPeer.ID)
		s.peerEndpoints[peer.ID] = peer.Endpoints
		s.peerMutex.Unlock()

		// Notify listeners
		for _, listener := range s.listeners {
			listener(EventNodeLeft, s.nodes[oldestPeer.ID])
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

		// Process message
		// In a real implementation, this would parse and handle different message types
		log.Printf("Received %d bytes from peer %s", n, id)
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
	peers := make(map[string]net.Conn, len(s.peerConnections))
	for id, conn := range s.peerConnections {
		peers[id] = conn
	}
	s.connectionsMutex.RUnlock()

	for id, conn := range peers {
		// Simple ping message (in a real implementation, this would be a structured message)
		_, err := conn.Write([]byte("ping"))
		if err != nil {
			log.Printf("Error pinging peer %s: %v", id, err)

			// Close and remove the connection
			conn.Close()

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
	// TODO: Implement DHT routing table refresh
	log.Printf("Refreshing routing tables")

	// In a real implementation, this would:
	// 1. Find a random ID in each bucket's range
	// 2. Perform a find node operation for each ID
	// 3. Update the routing table with any new nodes found
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
