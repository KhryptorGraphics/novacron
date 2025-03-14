package discovery

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

// generateUUID generates a random UUID
func generateUUID() string {
	u := make([]byte, 16)
	_, err := rand.Read(u)
	if err != nil {
		panic(fmt.Sprintf("Failed to generate UUID: %v", err))
	}

	u[8] = (u[8] | 0x80) & 0xBF // variant bits
	u[6] = (u[6] & 0x0F) | 0x40 // version 4

	return hex.EncodeToString(u)
}

// NodeRole represents the role of a node in the cluster
type NodeRole string

const (
	// RoleManager indicates a node that can manage the cluster
	RoleManager NodeRole = "manager"

	// RoleWorker indicates a node that runs workloads but doesn't manage the cluster
	RoleWorker NodeRole = "worker"

	// RoleStorage indicates a node primarily used for storage
	RoleStorage NodeRole = "storage"

	// RoleCompute indicates a node primarily used for compute
	RoleCompute NodeRole = "compute"

	// RoleEdge indicates an edge node
	RoleEdge NodeRole = "edge"
)

// NodeInfo contains information about a node in the cluster
type NodeInfo struct {
	ID        string            `json:"id"`        // Unique identifier for the node
	Name      string            `json:"name"`      // Human-readable name
	Role      NodeRole          `json:"role"`      // Role in the cluster
	Address   string            `json:"address"`   // IP address or hostname
	Port      int               `json:"port"`      // Port for node communication
	Tags      map[string]string `json:"tags"`      // Arbitrary tags for node classification
	JoinedAt  time.Time         `json:"joined_at"` // When the node joined the cluster
	LastSeen  time.Time         `json:"last_seen"` // Last time the node was seen
	Available bool              `json:"available"` // Whether the node is currently available
	Resources NodeResources     `json:"resources"` // Resource information
}

// NodeResources contains information about a node's resources
type NodeResources struct {
	CPUCores    int    `json:"cpu_cores"`    // Number of CPU cores
	CPUModel    string `json:"cpu_model"`    // CPU model
	MemoryMB    int    `json:"memory_mb"`    // Total memory in MB
	DiskGB      int    `json:"disk_gb"`      // Total disk space in GB
	NetworkMbps int    `json:"network_mbps"` // Network bandwidth in Mbps
}

// EventType represents the type of node event
type EventType string

const (
	// EventNodeJoined indicates a node has joined the cluster
	EventNodeJoined EventType = "node_joined"

	// EventNodeLeft indicates a node has left the cluster
	EventNodeLeft EventType = "node_left"

	// EventNodeUpdated indicates a node's information has been updated
	EventNodeUpdated EventType = "node_updated"

	// EventNodeRoleChanged indicates a node's role has changed
	EventNodeRoleChanged EventType = "node_role_changed"

	// EventClusterFormed indicates a cluster has been formed
	EventClusterFormed EventType = "cluster_formed"

	// EventClusterSplit indicates a cluster has split into multiple clusters
	EventClusterSplit EventType = "cluster_split"

	// EventLeaderElected indicates a new leader has been elected
	EventLeaderElected EventType = "leader_elected"
)

// NodeEventListener is a function that handles node events
type NodeEventListener func(eventType EventType, nodeInfo NodeInfo)

// Config contains configuration for the discovery service
type Config struct {
	NodeID   string            `json:"node_id"`   // Unique identifier for this node
	NodeName string            `json:"node_name"` // Human-readable name for this node
	NodeRole NodeRole          `json:"node_role"` // Role of this node in the cluster
	Address  string            `json:"address"`   // IP address or hostname to bind to
	Port     int               `json:"port"`      // Port to listen on
	Tags     map[string]string `json:"tags"`      // Arbitrary tags for node classification
}

// DefaultConfig returns a default configuration
func DefaultConfig() Config {
	// Generate a random ID for the node
	id := generateUUID()

	// Try to get the machine's hostname
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "unknown"
	}

	return Config{
		NodeID:   id,
		NodeName: fmt.Sprintf("node-%s", hostname),
		NodeRole: RoleWorker,
		Address:  "0.0.0.0",
		Port:     7700,
		Tags:     make(map[string]string),
	}
}

// Service is the base discovery service interface
type Service struct {
	config         Config
	nodes          map[string]NodeInfo // Map of node ID to node info
	listeners      []NodeEventListener // List of event listeners
	nodesMutex     sync.RWMutex
	listenersMutex sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
	running        bool
	runningMutex   sync.RWMutex
}

// New creates a new discovery service
func New(config Config) (*Service, error) {
	ctx, cancel := context.WithCancel(context.Background())

	service := &Service{
		config:    config,
		nodes:     make(map[string]NodeInfo),
		listeners: make([]NodeEventListener, 0),
		ctx:       ctx,
		cancel:    cancel,
		running:   false,
	}

	// Add self to node list
	selfInfo := NodeInfo{
		ID:        config.NodeID,
		Name:      config.NodeName,
		Role:      config.NodeRole,
		Address:   config.Address,
		Port:      config.Port,
		Tags:      config.Tags,
		JoinedAt:  time.Now(),
		LastSeen:  time.Now(),
		Available: true,
	}

	service.nodes[config.NodeID] = selfInfo

	return service, nil
}

// Start starts the discovery service
func (s *Service) Start() error {
	s.runningMutex.Lock()
	defer s.runningMutex.Unlock()

	if s.running {
		return fmt.Errorf("discovery service already running")
	}

	log.Printf("Starting discovery service for node %s (%s)", s.config.NodeName, s.config.NodeID)
	s.running = true

	return nil
}

// Stop stops the discovery service
func (s *Service) Stop() error {
	s.runningMutex.Lock()
	defer s.runningMutex.Unlock()

	if !s.running {
		return nil
	}

	log.Printf("Stopping discovery service for node %s", s.config.NodeName)
	s.cancel()
	s.running = false

	return nil
}

// IsRunning returns whether the service is running
func (s *Service) IsRunning() bool {
	s.runningMutex.RLock()
	defer s.runningMutex.RUnlock()
	return s.running
}

// GetNodeByID returns a node by its ID
func (s *Service) GetNodeByID(id string) (NodeInfo, bool) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	node, exists := s.nodes[id]
	return node, exists
}

// GetNodes returns all known nodes
func (s *Service) GetNodes() []NodeInfo {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	nodes := make([]NodeInfo, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// AddNode adds a node to the list of known nodes
func (s *Service) AddNode(node NodeInfo) {
	s.nodesMutex.Lock()

	// Check if node already exists
	existingNode, exists := s.nodes[node.ID]
	if exists {
		// Update existing node
		existingNode.Name = node.Name
		existingNode.Role = node.Role
		existingNode.Address = node.Address
		existingNode.Port = node.Port
		existingNode.Tags = node.Tags
		existingNode.LastSeen = time.Now()
		existingNode.Available = true

		s.nodes[node.ID] = existingNode
		s.nodesMutex.Unlock()

		// Notify listeners of node update
		s.notifyListeners(EventNodeUpdated, existingNode)
	} else {
		// Add new node
		node.LastSeen = time.Now()
		node.Available = true
		s.nodes[node.ID] = node
		s.nodesMutex.Unlock()

		// Notify listeners of new node
		s.notifyListeners(EventNodeJoined, node)
	}
}

// RemoveNode removes a node from the list of known nodes
func (s *Service) RemoveNode(id string) {
	s.nodesMutex.Lock()

	node, exists := s.nodes[id]
	if exists {
		delete(s.nodes, id)
		s.nodesMutex.Unlock()

		// Notify listeners of node removal
		s.notifyListeners(EventNodeLeft, node)
	} else {
		s.nodesMutex.Unlock()
	}
}

// UpdateNodeStatus updates a node's availability status
func (s *Service) UpdateNodeStatus(id string, available bool) {
	s.nodesMutex.Lock()

	node, exists := s.nodes[id]
	if exists {
		node.Available = available
		node.LastSeen = time.Now()
		s.nodes[id] = node
		s.nodesMutex.Unlock()

		// Notify listeners of node update
		s.notifyListeners(EventNodeUpdated, node)
	} else {
		s.nodesMutex.Unlock()
	}
}

// AddListener adds an event listener
func (s *Service) AddListener(listener NodeEventListener) {
	s.listenersMutex.Lock()
	defer s.listenersMutex.Unlock()

	s.listeners = append(s.listeners, listener)
}

// RemoveListener removes an event listener
func (s *Service) RemoveListener(listener NodeEventListener) {
	s.listenersMutex.Lock()
	defer s.listenersMutex.Unlock()

	for i, l := range s.listeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			s.listeners = append(s.listeners[:i], s.listeners[i+1:]...)
			return
		}
	}
}

// notifyListeners notifies all listeners of an event
func (s *Service) notifyListeners(eventType EventType, node NodeInfo) {
	s.listenersMutex.RLock()
	listeners := make([]NodeEventListener, len(s.listeners))
	copy(listeners, s.listeners)
	s.listenersMutex.RUnlock()

	for _, listener := range listeners {
		listener(eventType, node)
	}
}

// Helpers

// getHostname returns the hostname of the machine
func getHostname() (string, error) {
	return "", fmt.Errorf("not implemented")
	// In real implementation:
	// return os.Hostname()
}
