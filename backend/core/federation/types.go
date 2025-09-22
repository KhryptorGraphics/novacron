package federation

import (
	"context"
	"crypto/tls"
	"net"
	"sync"
	"time"
)

// NodeState represents the state of a federation node
type NodeState string

const (
	NodeStateUnknown     NodeState = "unknown"
	NodeStateDiscovering NodeState = "discovering"
	NodeStateJoining     NodeState = "joining"
	NodeStateActive      NodeState = "active"
	NodeStateUnhealthy   NodeState = "unhealthy"
	NodeStateDraining    NodeState = "draining"
	NodeStateLeaving     NodeState = "leaving"
	NodeStateOffline     NodeState = "offline"
)

// ConsensusRole represents the role of a node in the consensus protocol
type ConsensusRole string

const (
	RoleFollower  ConsensusRole = "follower"
	RoleCandidate ConsensusRole = "candidate"
	RoleLeader    ConsensusRole = "leader"
)

// Node represents a node in the federation
type Node struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	ClusterID   string            `json:"cluster_id"`
	Address     string            `json:"address"`
	State       NodeState         `json:"state"`
	Role        ConsensusRole     `json:"role"`
	Capabilities NodeCapabilities `json:"capabilities"`
	Metadata    map[string]string `json:"metadata"`
	LastSeen    time.Time         `json:"last_seen"`
	JoinedAt    time.Time         `json:"joined_at"`
	Version     string            `json:"version"`
	TLSConfig   *tls.Config       `json:"-"`
	mu          sync.RWMutex
}

// NodeCapabilities represents the resources and features of a node
type NodeCapabilities struct {
	CPUCores       int               `json:"cpu_cores"`
	MemoryGB       float64           `json:"memory_gb"`
	StorageGB      float64           `json:"storage_gb"`
	NetworkBandwidthMbps float64      `json:"network_bandwidth_mbps"`
	Features       []string          `json:"features"`
	Resources      ResourceInventory `json:"resources"`
}

// ResourceInventory tracks available resources
type ResourceInventory struct {
	TotalCPU      float64 `json:"total_cpu"`
	UsedCPU       float64 `json:"used_cpu"`
	TotalMemory   int64   `json:"total_memory"`
	UsedMemory    int64   `json:"used_memory"`
	TotalStorage  int64   `json:"total_storage"`
	UsedStorage   int64   `json:"used_storage"`
	VMs           int     `json:"vms"`
	Containers    int     `json:"containers"`
	NetworkPools  int     `json:"network_pools"`
}

// FederationConfig represents the configuration for the federation
type FederationConfig struct {
	ClusterID            string        `json:"cluster_id"`
	NodeID               string        `json:"node_id"`
	BindAddress          string        `json:"bind_address"`
	AdvertiseAddress     string        `json:"advertise_address"`
	JoinAddresses        []string      `json:"join_addresses"`
	HeartbeatInterval    time.Duration `json:"heartbeat_interval"`
	ElectionTimeout      time.Duration `json:"election_timeout"`
	FailureThreshold     int           `json:"failure_threshold"`
	EnableMDNS           bool          `json:"enable_mdns"`
	EnableGossip         bool          `json:"enable_gossip"`
	TLSConfig            *tls.Config   `json:"-"`
	ResourceSharingPolicy ResourcePolicy `json:"resource_sharing_policy"`
}

// ResourcePolicy defines how resources are shared across the federation
type ResourcePolicy struct {
	EnableSharing      bool              `json:"enable_sharing"`
	MaxSharePercentage float64           `json:"max_share_percentage"`
	Priority           int               `json:"priority"`
	Affinity           []string          `json:"affinity"`
	AntiAffinity       []string          `json:"anti_affinity"`
	Constraints        map[string]string `json:"constraints"`
}

// FederationProvider interface abstracts federation operations for different implementations
type FederationProvider interface {
	// Lifecycle management
	Start(ctx context.Context) error
	Stop(ctx context.Context) error

	// Node management
	JoinFederation(ctx context.Context, joinAddresses []string) error
	LeaveFederation(ctx context.Context) error
	GetNodes(ctx context.Context) ([]*Node, error)
	GetNode(ctx context.Context, nodeID string) (*Node, error)
	GetLocalNodeID() string

	// Leadership and consensus
	GetLeader(ctx context.Context) (*Node, error)
	IsLeader() bool

	// Resource management
	RequestResources(ctx context.Context, request *ResourceRequest) (*ResourceAllocation, error)
	ReleaseResources(ctx context.Context, allocationID string) error
	AllocateResources(ctx context.Context, clusterID string, request *ResourceRequest) error

	// Cluster operations
	ListClusters() []*Cluster
	GetCluster(clusterID string) (*Cluster, error)
	GetFederatedClusters(ctx context.Context) ([]*Cluster, error)
	GetClusterResources(ctx context.Context, clusterID string) (*ClusterResources, error)
	GetClusterEndpoint(ctx context.Context, clusterID string) (string, error)

	// VM operations
	GetVMInfo(ctx context.Context, clusterID, vmID string) (*VMInfo, error)
	ScheduleVMCrossCluster(ctx context.Context, vmSpec *VMSchedulingSpec) (*VMPlacement, error)

	// Health and monitoring
	GetHealth(ctx context.Context) (*HealthCheck, error)
}

// FederationManager is a compatibility alias for FederationProvider
type FederationManager interface {
	FederationProvider
}

// ResourceRequest represents a request for resources from the federation
type ResourceRequest struct {
	ID             string            `json:"id"`
	RequesterID    string            `json:"requester_id"`
	ResourceType   string            `json:"resource_type"`
	CPUCores       float64           `json:"cpu_cores"`
	MemoryGB       float64           `json:"memory_gb"`
	StorageGB      float64           `json:"storage_gb"`
	Duration       time.Duration     `json:"duration"`
	Priority       int               `json:"priority"`
	Constraints    map[string]string `json:"constraints"`
	CreatedAt      time.Time         `json:"created_at"`
}


// HealthCheck represents a health check for a node
type HealthCheck struct {
	NodeID        string        `json:"node_id"`
	Timestamp     time.Time     `json:"timestamp"`
	Latency       time.Duration `json:"latency"`
	CPUUsage      float64       `json:"cpu_usage"`
	MemoryUsage   float64       `json:"memory_usage"`
	DiskUsage     float64       `json:"disk_usage"`
	NetworkLatency map[string]time.Duration `json:"network_latency"`
	Services      map[string]ServiceHealth `json:"services"`
	Healthy       bool          `json:"healthy"`
	Issues        []string      `json:"issues"`
}

// ServiceHealth represents the health of a service on a node
type ServiceHealth struct {
	Name    string `json:"name"`
	Status  string `json:"status"`
	Latency time.Duration `json:"latency"`
	Error   string `json:"error,omitempty"`
}

// RaftMessage represents a message in the Raft consensus protocol
type RaftMessage struct {
	Term         uint64      `json:"term"`
	Type         MessageType `json:"type"`
	From         string      `json:"from"`
	To           string      `json:"to"`
	Entries      []LogEntry  `json:"entries,omitempty"`
	CommitIndex  uint64      `json:"commit_index"`
	PrevLogIndex uint64      `json:"prev_log_index"`
	PrevLogTerm  uint64      `json:"prev_log_term"`
	Success      bool        `json:"success"`
	VoteGranted  bool        `json:"vote_granted"`
}

// MessageType represents the type of Raft message
type MessageType string

const (
	MessageVoteRequest     MessageType = "vote_request"
	MessageVoteResponse    MessageType = "vote_response"
	MessageAppendEntries   MessageType = "append_entries"
	MessageAppendResponse  MessageType = "append_response"
	MessageInstallSnapshot MessageType = "install_snapshot"
)

// LogEntry represents an entry in the Raft log
type LogEntry struct {
	Index     uint64      `json:"index"`
	Term      uint64      `json:"term"`
	Type      EntryType   `json:"type"`
	Data      []byte      `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

// EntryType represents the type of log entry
type EntryType string

const (
	EntryCommand      EntryType = "command"
	EntryConfiguration EntryType = "configuration"
	EntryNoop         EntryType = "noop"
)

// GossipMessage represents a message in the gossip protocol
type GossipMessage struct {
	ID        string            `json:"id"`
	Type      GossipType        `json:"type"`
	Source    string            `json:"source"`
	Data      interface{}       `json:"data"`
	TTL       int               `json:"ttl"`
	Timestamp time.Time         `json:"timestamp"`
	Signature []byte            `json:"signature"`
}

// GossipType represents the type of gossip message
type GossipType string

const (
	GossipNodeJoin     GossipType = "node_join"
	GossipNodeLeave    GossipType = "node_leave"
	GossipNodeUpdate   GossipType = "node_update"
	GossipResourceUpdate GossipType = "resource_update"
	GossipHealthCheck  GossipType = "health_check"
)

// Interfaces

// FederationManager manages the federation of clusters
type FederationManager interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	JoinFederation(ctx context.Context, joinAddresses []string) error
	LeaveFederation(ctx context.Context) error
	GetNodes(ctx context.Context) ([]*Node, error)
	GetNode(ctx context.Context, nodeID string) (*Node, error)
	GetLeader(ctx context.Context) (*Node, error)
	IsLeader() bool
	RequestResources(ctx context.Context, request *ResourceRequest) (*ResourceAllocation, error)
	ReleaseResources(ctx context.Context, allocationID string) error
	GetHealth(ctx context.Context) (*HealthCheck, error)
	ListClusters() []*Cluster
	GetLocalClusterID() string
	GetCluster(clusterID string) (*Cluster, error)
	CreateCrossClusterOperation(operation *CrossClusterOperation) error
	UpdateCrossClusterOperation(operationID string, status string, progress int, error string) error
	UpdateClusterState(clusterID string, state ClusterState) error
}

// ConsensusManager manages the consensus protocol
type ConsensusManager interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	ProposeValue(ctx context.Context, key string, value []byte) error
	GetValue(ctx context.Context, key string) ([]byte, error)
	GetLeader() (string, error)
	IsLeader() bool
	AddNode(ctx context.Context, nodeID string, address string) error
	RemoveNode(ctx context.Context, nodeID string) error
}

// DiscoveryManager manages node discovery
type DiscoveryManager interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	DiscoverNodes(ctx context.Context) ([]*Node, error)
	RegisterNode(ctx context.Context, node *Node) error
	UnregisterNode(ctx context.Context, nodeID string) error
	GetDiscoveredNodes() []*Node
}

// ResourceManager manages resource allocation
type ResourceManager interface {
	AllocateResources(ctx context.Context, request *ResourceRequest) (*ResourceAllocation, error)
	ReleaseResources(ctx context.Context, allocationID string) error
	GetAllocations(ctx context.Context) ([]*ResourceAllocation, error)
	GetAvailableResources(ctx context.Context) (*ResourceInventory, error)
	UpdateResourceInventory(ctx context.Context, nodeID string, inventory *ResourceInventory) error
}

// HealthChecker performs health checks
type HealthChecker interface {
	CheckHealth(ctx context.Context, node *Node) (*HealthCheck, error)
	StartMonitoring(ctx context.Context, interval time.Duration) error
	StopMonitoring(ctx context.Context) error
	GetHealthStatus(nodeID string) (*HealthCheck, error)
	RegisterHealthHandler(handler HealthHandler)
}

// HealthHandler handles health check events
type HealthHandler interface {
	OnNodeHealthy(node *Node)
	OnNodeUnhealthy(node *Node, issues []string)
	OnNodeOffline(node *Node)
}

// Transport handles network communication
type Transport interface {
	Listen(address string) (net.Listener, error)
	Dial(ctx context.Context, address string) (net.Conn, error)
	Send(ctx context.Context, nodeID string, message interface{}) error
	Receive(ctx context.Context) (interface{}, error)
	Close() error
}