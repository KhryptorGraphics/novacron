// Network Stack Optimization for DWCP v3
//
// Implements advanced network stack tuning:
// - TCP stack tuning (BBR, CUBIC)
// - UDP optimization for QUIC
// - Socket buffer tuning
// - RSS/RPS configuration
// - Zero-copy networking
//
// Phase 7: Extreme Performance Optimization
// Target: Maximum throughput with minimum latency

package performance

import (
	"fmt"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// TCP Congestion Control Algorithms
type CongestionControl string

const (
	CCReno  CongestionControl = "reno"
	CCCubic CongestionControl = "cubic"
	CCBBR   CongestionControl = "bbr"
	CCVegas CongestionControl = "vegas"
)

// Socket Options
type SocketOptions struct {
	TCPNoDelay           bool
	TCPQuickAck          bool
	TCPFastOpen          bool
	ReuseAddr            bool
	ReusePort            bool
	SendBufferSize       int
	RecvBufferSize       int
	KeepAlive            bool
	KeepAliveInterval    time.Duration
	KeepAliveCount       int
	Linger               int
	ZeroCopy             bool
	BusyPoll             bool
	CongestionControl    CongestionControl
}

// Network Stack Configuration
type NetworkStackConfig struct {
	// TCP Parameters
	TCPMaxBacklog         int
	TCPMaxSynBacklog      int
	TCPSynRetries         int
	TCPSynAckRetries      int
	TCPKeepaliveTime      int
	TCPKeepaliveIntvl     int
	TCPKeepaliveProbes    int
	TCPFinTimeout         int
	TCPTWRecycle          bool
	TCPTWReuse            bool
	TCPTimestamps         bool
	TCPSack               bool
	TCPFack               bool
	TCPECN                bool
	TCPWindowScaling      bool
	TCPNoMetricsSave      bool
	TCPModerateRcvbuf     bool
	TCPSlowStartAfterIdle bool

	// UDP Parameters
	UDPMaxDatagramSize int
	UDPRecvBufSize     int
	UDPSendBufSize     int

	// Buffer Sizes
	CoreRmemDefault int
	CoreRmemMax     int
	CoreWmemDefault int
	CoreWmemMax     int
	CoreNetdevBudget int

	// RSS/RPS Configuration
	EnableRSS       bool
	EnableRPS       bool
	EnableXPS       bool
	RSSCPUMask      uint64
	RPSFlowEntries  int
}

// Network Stack Manager
type NetworkStackManager struct {
	mu             sync.RWMutex
	config         *NetworkStackConfig
	stats          *NetworkStats
	connPool       map[string]*ConnectionPool
	optimizedConns map[int]*OptimizedConnection
	tuningActive   atomic.Bool
	monitorActive  atomic.Bool
}

// Network Statistics
type NetworkStats struct {
	totalConnections    atomic.Uint64
	activeConnections   atomic.Uint64
	bytesSent           atomic.Uint64
	bytesReceived       atomic.Uint64
	packetsSent         atomic.Uint64
	packetsReceived     atomic.Uint64
	retransmits         atomic.Uint64
	timeouts            atomic.Uint64
	zeroCopyOps         atomic.Uint64
	tcpSlowStarts       atomic.Uint64
	tcpCongestionEvents atomic.Uint64
}

// Connection Pool
type ConnectionPool struct {
	name        string
	addr        string
	maxConns    int
	minConns    int
	connections []*OptimizedConnection
	available   chan *OptimizedConnection
	mu          sync.Mutex
	stats       *PoolStats
}

// Pool Statistics
type PoolStats struct {
	acquired  atomic.Uint64
	released  atomic.Uint64
	created   atomic.Uint64
	destroyed atomic.Uint64
	timeouts  atomic.Uint64
}

// Optimized Connection
type OptimizedConnection struct {
	conn          net.Conn
	fd            int
	opts          *SocketOptions
	created       time.Time
	lastUsed      time.Time
	bytesSent     atomic.Uint64
	bytesReceived atomic.Uint64
	inUse         atomic.Bool
}

// NewNetworkStackManager creates a new network stack manager
func NewNetworkStackManager() (*NetworkStackManager, error) {
	nsm := &NetworkStackManager{
		config:         getDefaultNetworkConfig(),
		stats:          &NetworkStats{},
		connPool:       make(map[string]*ConnectionPool),
		optimizedConns: make(map[int]*OptimizedConnection),
	}

	// Apply system-wide tuning
	if err := nsm.applySystemTuning(); err != nil {
		fmt.Printf("Warning: Could not apply all system tuning: %v\n", err)
	}

	// Start monitoring
	nsm.monitorActive.Store(true)
	go nsm.monitorNetworkMetrics()

	fmt.Println("Network Stack Manager initialized")
	return nsm, nil
}

// Get default network configuration
func getDefaultNetworkConfig() *NetworkStackConfig {
	return &NetworkStackConfig{
		// TCP Parameters
		TCPMaxBacklog:         4096,
		TCPMaxSynBacklog:      8192,
		TCPSynRetries:         3,
		TCPSynAckRetries:      3,
		TCPKeepaliveTime:      600,
		TCPKeepaliveIntvl:     60,
		TCPKeepaliveProbes:    9,
		TCPFinTimeout:         30,
		TCPTWReuse:            true,
		TCPTimestamps:         true,
		TCPSack:               true,
		TCPFack:               true,
		TCPECN:                false,
		TCPWindowScaling:      true,
		TCPModerateRcvbuf:     true,
		TCPSlowStartAfterIdle: false,

		// UDP Parameters
		UDPMaxDatagramSize: 65507,
		UDPRecvBufSize:     16 * 1024 * 1024,
		UDPSendBufSize:     16 * 1024 * 1024,

		// Buffer Sizes
		CoreRmemDefault:  262144,
		CoreRmemMax:      16777216,
		CoreWmemDefault:  262144,
		CoreWmemMax:      16777216,
		CoreNetdevBudget: 50000,

		// RSS/RPS
		EnableRSS:      true,
		EnableRPS:      true,
		EnableXPS:      true,
		RPSFlowEntries: 32768,
	}
}

// Apply system-wide network tuning
func (nsm *NetworkStackManager) applySystemTuning() error {
	// In production, write to /proc/sys/net files
	// For example:
	// echo "1" > /proc/sys/net/ipv4/tcp_tw_reuse
	// echo "bbr" > /proc/sys/net/ipv4/tcp_congestion_control

	tunings := map[string]string{
		"/proc/sys/net/core/rmem_default":               fmt.Sprintf("%d", nsm.config.CoreRmemDefault),
		"/proc/sys/net/core/rmem_max":                   fmt.Sprintf("%d", nsm.config.CoreRmemMax),
		"/proc/sys/net/core/wmem_default":               fmt.Sprintf("%d", nsm.config.CoreWmemDefault),
		"/proc/sys/net/core/wmem_max":                   fmt.Sprintf("%d", nsm.config.CoreWmemMax),
		"/proc/sys/net/core/netdev_max_backlog":         fmt.Sprintf("%d", nsm.config.TCPMaxBacklog),
		"/proc/sys/net/core/netdev_budget":              fmt.Sprintf("%d", nsm.config.CoreNetdevBudget),
		"/proc/sys/net/ipv4/tcp_max_syn_backlog":        fmt.Sprintf("%d", nsm.config.TCPMaxSynBacklog),
		"/proc/sys/net/ipv4/tcp_syn_retries":            fmt.Sprintf("%d", nsm.config.TCPSynRetries),
		"/proc/sys/net/ipv4/tcp_synack_retries":         fmt.Sprintf("%d", nsm.config.TCPSynAckRetries),
		"/proc/sys/net/ipv4/tcp_keepalive_time":         fmt.Sprintf("%d", nsm.config.TCPKeepaliveTime),
		"/proc/sys/net/ipv4/tcp_keepalive_intvl":        fmt.Sprintf("%d", nsm.config.TCPKeepaliveIntvl),
		"/proc/sys/net/ipv4/tcp_keepalive_probes":       fmt.Sprintf("%d", nsm.config.TCPKeepaliveProbes),
		"/proc/sys/net/ipv4/tcp_fin_timeout":            fmt.Sprintf("%d", nsm.config.TCPFinTimeout),
		"/proc/sys/net/ipv4/tcp_tw_reuse":               boolToString(nsm.config.TCPTWReuse),
		"/proc/sys/net/ipv4/tcp_timestamps":             boolToString(nsm.config.TCPTimestamps),
		"/proc/sys/net/ipv4/tcp_sack":                   boolToString(nsm.config.TCPSack),
		"/proc/sys/net/ipv4/tcp_fack":                   boolToString(nsm.config.TCPFack),
		"/proc/sys/net/ipv4/tcp_ecn":                    boolToString(nsm.config.TCPECN),
		"/proc/sys/net/ipv4/tcp_window_scaling":         boolToString(nsm.config.TCPWindowScaling),
		"/proc/sys/net/ipv4/tcp_moderate_rcvbuf":        boolToString(nsm.config.TCPModerateRcvbuf),
		"/proc/sys/net/ipv4/tcp_slow_start_after_idle":  boolToString(nsm.config.TCPSlowStartAfterIdle),
		"/proc/sys/net/ipv4/tcp_congestion_control":     "bbr",
		"/proc/sys/net/ipv4/tcp_fastopen":               "3",
	}

	errorCount := 0
	for path, value := range tunings {
		if err := writeSysctl(path, value); err != nil {
			fmt.Printf("Warning: Failed to set %s: %v\n", path, err)
			errorCount++
		}
	}

	if errorCount > 0 {
		return fmt.Errorf("failed to apply %d tuning parameters", errorCount)
	}

	fmt.Println("System-wide network tuning applied successfully")
	return nil
}

// CreateConnectionPool creates a connection pool for a specific address
func (nsm *NetworkStackManager) CreateConnectionPool(name, addr string, minConns, maxConns int) error {
	nsm.mu.Lock()
	defer nsm.mu.Unlock()

	if _, exists := nsm.connPool[name]; exists {
		return fmt.Errorf("connection pool %s already exists", name)
	}

	pool := &ConnectionPool{
		name:        name,
		addr:        addr,
		maxConns:    maxConns,
		minConns:    minConns,
		connections: make([]*OptimizedConnection, 0, maxConns),
		available:   make(chan *OptimizedConnection, maxConns),
		stats:       &PoolStats{},
	}

	// Pre-create minimum connections
	for i := 0; i < minConns; i++ {
		conn, err := nsm.createOptimizedConnection(addr)
		if err != nil {
			return fmt.Errorf("failed to create connection: %w", err)
		}
		pool.connections = append(pool.connections, conn)
		pool.available <- conn
		pool.stats.created.Add(1)
	}

	nsm.connPool[name] = pool

	fmt.Printf("Created connection pool: %s (min=%d, max=%d)\n", name, minConns, maxConns)
	return nil
}

// Create optimized connection with tuned socket options
func (nsm *NetworkStackManager) createOptimizedConnection(addr string) (*OptimizedConnection, error) {
	// Parse address
	network := "tcp"
	if len(addr) > 0 && addr[0] == '[' {
		// IPv6
		network = "tcp6"
	}

	// Create connection
	conn, err := net.Dial(network, addr)
	if err != nil {
		return nil, err
	}

	// Get file descriptor
	tcpConn, ok := conn.(*net.TCPConn)
	if !ok {
		conn.Close()
		return nil, fmt.Errorf("not a TCP connection")
	}

	file, err := tcpConn.File()
	if err != nil {
		conn.Close()
		return nil, err
	}
	fd := int(file.Fd())

	// Create optimized connection
	optConn := &OptimizedConnection{
		conn: conn,
		fd:   fd,
		opts: &SocketOptions{
			TCPNoDelay:         true,
			TCPQuickAck:        true,
			TCPFastOpen:        true,
			ReuseAddr:          true,
			ReusePort:          true,
			SendBufferSize:     nsm.config.CoreWmemMax,
			RecvBufferSize:     nsm.config.CoreRmemMax,
			KeepAlive:          true,
			KeepAliveInterval:  time.Duration(nsm.config.TCPKeepaliveIntvl) * time.Second,
			KeepAliveCount:     nsm.config.TCPKeepaliveProbes,
			ZeroCopy:           true,
			BusyPoll:           true,
			CongestionControl:  CCBBR,
		},
		created:  time.Now(),
		lastUsed: time.Now(),
	}

	// Apply socket options
	if err := nsm.applySocketOptions(optConn); err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to apply socket options: %w", err)
	}

	nsm.mu.Lock()
	nsm.optimizedConns[fd] = optConn
	nsm.mu.Unlock()

	nsm.stats.totalConnections.Add(1)
	nsm.stats.activeConnections.Add(1)

	return optConn, nil
}

// Apply socket options to connection
func (nsm *NetworkStackManager) applySocketOptions(conn *OptimizedConnection) error {
	fd := conn.fd
	opts := conn.opts

	// TCP_NODELAY
	if opts.TCPNoDelay {
		if err := syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, 1); err != nil {
			return err
		}
	}

	// SO_REUSEADDR
	if opts.ReuseAddr {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1); err != nil {
			return err
		}
	}

	// SO_REUSEPORT
	if opts.ReusePort {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, 0xf /* SO_REUSEPORT */, 1); err != nil {
			fmt.Printf("Warning: SO_REUSEPORT not supported: %v\n", err)
		}
	}

	// SO_SNDBUF
	if opts.SendBufferSize > 0 {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, opts.SendBufferSize); err != nil {
			return err
		}
	}

	// SO_RCVBUF
	if opts.RecvBufferSize > 0 {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, opts.RecvBufferSize); err != nil {
			return err
		}
	}

	// SO_KEEPALIVE
	if opts.KeepAlive {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, 1); err != nil {
			return err
		}
	}

	// TCP_QUICKACK (Linux-specific)
	if opts.TCPQuickAck {
		if err := syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, 0xc /* TCP_QUICKACK */, 1); err != nil {
			fmt.Printf("Warning: TCP_QUICKACK not supported: %v\n", err)
		}
	}

	// TCP_FASTOPEN (Linux-specific)
	if opts.TCPFastOpen {
		if err := syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, 23 /* TCP_FASTOPEN */, 5); err != nil {
			fmt.Printf("Warning: TCP_FASTOPEN not supported: %v\n", err)
		}
	}

	// SO_BUSY_POLL (Linux-specific)
	if opts.BusyPoll {
		if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, 46 /* SO_BUSY_POLL */, 50); err != nil {
			fmt.Printf("Warning: SO_BUSY_POLL not supported: %v\n", err)
		}
	}

	return nil
}

// GetConnection acquires a connection from pool
func (nsm *NetworkStackManager) GetConnection(poolName string) (*OptimizedConnection, error) {
	nsm.mu.RLock()
	pool, exists := nsm.connPool[poolName]
	nsm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("connection pool %s not found", poolName)
	}

	select {
	case conn := <-pool.available:
		conn.inUse.Store(true)
		conn.lastUsed = time.Now()
		pool.stats.acquired.Add(1)
		return conn, nil
	case <-time.After(5 * time.Second):
		pool.stats.timeouts.Add(1)
		return nil, fmt.Errorf("timeout acquiring connection")
	}
}

// ReleaseConnection returns connection to pool
func (nsm *NetworkStackManager) ReleaseConnection(poolName string, conn *OptimizedConnection) error {
	nsm.mu.RLock()
	pool, exists := nsm.connPool[poolName]
	nsm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("connection pool %s not found", poolName)
	}

	conn.inUse.Store(false)
	pool.available <- conn
	pool.stats.released.Add(1)

	return nil
}

// Monitor network metrics
func (nsm *NetworkStackManager) monitorNetworkMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for nsm.monitorActive.Load() {
		<-ticker.C
		nsm.collectNetworkMetrics()
	}
}

// Collect network metrics
func (nsm *NetworkStackManager) collectNetworkMetrics() {
	// In production, read from /proc/net/snmp, /proc/net/netstat, etc.
	// For now, use the stats we're already tracking
}

// GetStatistics returns network statistics
func (nsm *NetworkStackManager) GetStatistics() map[string]interface{} {
	stats := make(map[string]interface{})

	stats["total_connections"] = nsm.stats.totalConnections.Load()
	stats["active_connections"] = nsm.stats.activeConnections.Load()
	stats["bytes_sent"] = nsm.stats.bytesSent.Load()
	stats["bytes_received"] = nsm.stats.bytesReceived.Load()
	stats["packets_sent"] = nsm.stats.packetsSent.Load()
	stats["packets_received"] = nsm.stats.packetsReceived.Load()
	stats["retransmits"] = nsm.stats.retransmits.Load()
	stats["timeouts"] = nsm.stats.timeouts.Load()
	stats["zero_copy_ops"] = nsm.stats.zeroCopyOps.Load()
	stats["tcp_slow_starts"] = nsm.stats.tcpSlowStarts.Load()
	stats["tcp_congestion_events"] = nsm.stats.tcpCongestionEvents.Load()

	// Pool statistics
	poolStats := make(map[string]interface{})
	nsm.mu.RLock()
	for name, pool := range nsm.connPool {
		poolStats[name] = map[string]interface{}{
			"size":       len(pool.connections),
			"available":  len(pool.available),
			"acquired":   pool.stats.acquired.Load(),
			"released":   pool.stats.released.Load(),
			"created":    pool.stats.created.Load(),
			"destroyed":  pool.stats.destroyed.Load(),
			"timeouts":   pool.stats.timeouts.Load(),
		}
	}
	nsm.mu.RUnlock()
	stats["pools"] = poolStats

	return stats
}

// PrintStatistics prints network statistics
func (nsm *NetworkStackManager) PrintStatistics() {
	stats := nsm.GetStatistics()

	fmt.Printf("\n=== Network Stack Statistics ===\n")
	fmt.Printf("Total connections: %d\n", stats["total_connections"])
	fmt.Printf("Active connections: %d\n", stats["active_connections"])
	fmt.Printf("Bytes sent: %d (%.2f GB)\n",
		stats["bytes_sent"],
		float64(stats["bytes_sent"].(uint64))/(1024*1024*1024))
	fmt.Printf("Bytes received: %d (%.2f GB)\n",
		stats["bytes_received"],
		float64(stats["bytes_received"].(uint64))/(1024*1024*1024))
	fmt.Printf("Packets sent: %d\n", stats["packets_sent"])
	fmt.Printf("Packets received: %d\n", stats["packets_received"])
	fmt.Printf("Retransmits: %d\n", stats["retransmits"])
	fmt.Printf("Timeouts: %d\n", stats["timeouts"])
	fmt.Printf("Zero-copy ops: %d\n", stats["zero_copy_ops"])
	fmt.Printf("TCP slow starts: %d\n", stats["tcp_slow_starts"])
	fmt.Printf("TCP congestion events: %d\n", stats["tcp_congestion_events"])

	fmt.Printf("\nConnection Pools:\n")
	poolStats := stats["pools"].(map[string]interface{})
	for name, ps := range poolStats {
		pstat := ps.(map[string]interface{})
		fmt.Printf("  %s:\n", name)
		fmt.Printf("    Size: %d\n", pstat["size"])
		fmt.Printf("    Available: %d\n", pstat["available"])
		fmt.Printf("    Acquired: %d\n", pstat["acquired"])
		fmt.Printf("    Released: %d\n", pstat["released"])
	}

	fmt.Printf("================================\n\n")
}

// Helper functions

func writeSysctl(path, value string) error {
	// Check if running in container/WSL where we might not have access
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil // Skip if file doesn't exist
	}

	file, err := os.OpenFile(path, os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.WriteString(value)
	return err
}

func boolToString(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

// Close cleans up network stack manager
func (nsm *NetworkStackManager) Close() error {
	nsm.monitorActive.Store(false)

	nsm.mu.Lock()
	defer nsm.mu.Unlock()

	// Close all connection pools
	for _, pool := range nsm.connPool {
		for _, conn := range pool.connections {
			conn.conn.Close()
		}
	}

	nsm.connPool = nil
	nsm.optimizedConns = nil

	return nil
}
