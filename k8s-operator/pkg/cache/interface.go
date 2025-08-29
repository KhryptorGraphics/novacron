package cache

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Manager manages cache cluster connections
type Manager interface {
	// GetOrCreateCluster gets or creates a cache cluster connection
	GetOrCreateCluster(clusterID string, config *ClusterConfig) (Cluster, error)

	// GetCluster gets an existing cluster connection
	GetCluster(clusterID string) (Cluster, error)

	// DisconnectCluster disconnects from a cluster
	DisconnectCluster(clusterID string) error

	// ListClusters lists all connected clusters
	ListClusters() []string
}

// Cluster represents a Redis cluster connection
type Cluster interface {
	// Basic operations
	Get(key string) (string, error)
	Set(key string, value string, ttl string) error
	Delete(key string) error
	Exists(key string) (bool, error)

	// Batch operations
	MGet(keys []string) (map[string]string, error)
	MSet(values map[string]string, ttl string) error
	MDelete(keys []string) error

	// Configuration
	ConfigureStrategy(strategy *Strategy) error
	SetTTLPolicy(policy *TTLPolicy) error
	ConfigureEviction(policy *EvictionPolicy) error

	// Cache warming
	WarmCache(strategy *WarmingStrategy) error

	// Monitoring and health
	GetHealth() (*Health, error)
	GetPerformanceMetrics() (*PerformanceMetrics, error)
	GetMemoryUsage() (*MemoryUsage, error)

	// Administrative operations
	FlushDB() error
	FlushAll() error
	Info() (map[string]string, error)

	// Connection management
	Ping() error
	Close() error
}

// ClusterConfig represents cluster configuration
type ClusterConfig struct {
	Endpoints   []string               `json:"endpoints"`
	Credentials map[string]string      `json:"credentials,omitempty"`
	HA          *HAConfig             `json:"ha,omitempty"`
	Security    *SecurityConfig       `json:"security,omitempty"`
	Timeouts    *TimeoutConfig        `json:"timeouts,omitempty"`
	PoolConfig  *ConnectionPoolConfig `json:"poolConfig,omitempty"`
}

// HAConfig represents high availability configuration
type HAConfig struct {
	SentinelEnabled bool     `json:"sentinelEnabled,omitempty"`
	Replicas        int32    `json:"replicas,omitempty"`
	SentinelHosts   []string `json:"sentinelHosts,omitempty"`
	MasterName      string   `json:"masterName,omitempty"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	TLSEnabled  bool   `json:"tlsEnabled,omitempty"`
	AuthEnabled bool   `json:"authEnabled,omitempty"`
	CertSecret  string `json:"certSecret,omitempty"`
	Username    string `json:"username,omitempty"`
	Password    string `json:"password,omitempty"`
}

// TimeoutConfig represents timeout configuration
type TimeoutConfig struct {
	ConnectTimeout string `json:"connectTimeout,omitempty"`
	ReadTimeout    string `json:"readTimeout,omitempty"`
	WriteTimeout   string `json:"writeTimeout,omitempty"`
}

// ConnectionPoolConfig represents connection pool configuration
type ConnectionPoolConfig struct {
	MaxActive     int `json:"maxActive,omitempty"`
	MaxIdle       int `json:"maxIdle,omitempty"`
	MinIdle       int `json:"minIdle,omitempty"`
	MaxWait       int `json:"maxWait,omitempty"`
}

// Strategy represents caching strategy
type Strategy struct {
	Type        string  `json:"type"`
	Levels      []Level `json:"levels,omitempty"`
	Consistency string  `json:"consistency,omitempty"`
}

// Level represents cache level configuration
type Level struct {
	Name     string `json:"name"`
	Size     string `json:"size"`
	TTL      string `json:"ttl,omitempty"`
	Strategy string `json:"strategy,omitempty"`
}

// TTLPolicy represents TTL policy configuration
type TTLPolicy struct {
	Pattern         string `json:"pattern"`
	TTL            string `json:"ttl"`
	RefreshStrategy string `json:"refreshStrategy,omitempty"`
}

// EvictionPolicy represents eviction policy configuration
type EvictionPolicy struct {
	Algorithm        string             `json:"algorithm"`
	MemoryThresholds *MemoryThresholds `json:"memoryThresholds,omitempty"`
}

// MemoryThresholds represents memory threshold configuration
type MemoryThresholds struct {
	Warning  float64 `json:"warning,omitempty"`
	Critical float64 `json:"critical,omitempty"`
	Eviction float64 `json:"eviction,omitempty"`
}

// WarmingStrategy represents cache warming strategy
type WarmingStrategy struct {
	Type     string   `json:"type"`
	Patterns []string `json:"patterns,omitempty"`
	Priority int32    `json:"priority,omitempty"`
	Data     map[string]string `json:"data,omitempty"`
}

// Health represents cluster health information
type Health struct {
	Status      string       `json:"status"`
	Nodes       []NodeStatus `json:"nodes,omitempty"`
	Replication string       `json:"replication,omitempty"`
}

// NodeStatus represents individual node status
type NodeStatus struct {
	NodeID   string           `json:"nodeId"`
	Status   string           `json:"status"`
	Role     string           `json:"role"`
	Memory   string           `json:"memory,omitempty"`
	LastSeen *metav1.Time     `json:"lastSeen,omitempty"`
}

// PerformanceMetrics represents performance metrics
type PerformanceMetrics struct {
	HitRate      float64 `json:"hitRate"`
	MissRate     float64 `json:"missRate"`
	ResponseTime string  `json:"responseTime"`
	Throughput   float64 `json:"throughput"`
	Operations   int64   `json:"operations,omitempty"`
	Errors       int64   `json:"errors,omitempty"`
}

// MemoryUsage represents memory usage information
type MemoryUsage struct {
	TotalMemory     string            `json:"totalMemory"`
	UsedMemory      string            `json:"usedMemory"`
	UsageByLevel    map[string]string `json:"usageByLevel,omitempty"`
	FragmentationRatio float64        `json:"fragmentationRatio,omitempty"`
}

// CacheEntry represents a cache entry
type CacheEntry struct {
	Key        string `json:"key"`
	Value      string `json:"value"`
	TTL        int64  `json:"ttl,omitempty"`
	Expiration int64  `json:"expiration,omitempty"`
}

// CacheStats represents cache statistics
type CacheStats struct {
	TotalKeys        int64   `json:"totalKeys"`
	ExpiredKeys      int64   `json:"expiredKeys"`
	EvictedKeys      int64   `json:"evictedKeys"`
	Hits             int64   `json:"hits"`
	Misses           int64   `json:"misses"`
	HitRate          float64 `json:"hitRate"`
	MemoryUsed       int64   `json:"memoryUsed"`
	MemoryPeak       int64   `json:"memoryPeak"`
	ConnectedClients int32   `json:"connectedClients"`
}

// WarmingResult represents cache warming result
type WarmingResult struct {
	Strategy      string `json:"strategy"`
	KeysWarmed    int64  `json:"keysWarmed"`
	Duration      string `json:"duration"`
	Success       bool   `json:"success"`
	ErrorMessage  string `json:"errorMessage,omitempty"`
}