// Package optimization provides network optimization for DWCP v3.
package optimization

import (
	"context"
	"fmt"
	"net"
	"sync"
	"syscall"
	"time"
)

// NetworkOptimizerConfig defines configuration for network optimization.
type NetworkOptimizerConfig struct {
	// RDMA settings (datacenter mode)
	RDMAEnabled         bool
	RDMAQueueDepth      int
	RDMAMaxSGE          int // Scatter-Gather Elements
	RDMACompletionQueue int
	RDMAInlineSize      int // bytes

	// TCP settings (internet mode)
	TCPEnabled          bool
	TCPCongestion       string // "bbr", "cubic", etc.
	TCPWindowSize       int
	TCPNoDelay          bool
	TCPQuickAck         bool
	TCPKeepAlive        bool
	TCPKeepAliveIdle    time.Duration
	TCPKeepAliveInterval time.Duration

	// Connection pooling
	EnableConnPool      bool
	MaxConnsPerHost     int
	MaxIdleConns        int
	MaxIdleTime         time.Duration
	ConnTimeout         time.Duration

	// Packet optimization
	EnablePacketCoalesce bool
	CoalesceDelay        time.Duration
	CoalesceMaxSize      int

	// Bandwidth optimization
	EnableRateLimiting   bool
	MaxBandwidth         uint64 // bytes/sec
	BurstSize            int

	// Buffer sizes
	SendBufferSize       int
	RecvBufferSize       int
}

// DefaultNetworkOptimizerConfig returns default network optimizer configuration.
func DefaultNetworkOptimizerConfig() *NetworkOptimizerConfig {
	return &NetworkOptimizerConfig{
		RDMAEnabled:          true,
		RDMAQueueDepth:       1024,
		RDMAMaxSGE:           16,
		RDMACompletionQueue:  2048,
		RDMAInlineSize:       256,
		TCPEnabled:           true,
		TCPCongestion:        "bbr",
		TCPWindowSize:        4 * 1024 * 1024, // 4MB
		TCPNoDelay:           true,
		TCPQuickAck:          true,
		TCPKeepAlive:         true,
		TCPKeepAliveIdle:     30 * time.Second,
		TCPKeepAliveInterval: 10 * time.Second,
		EnableConnPool:       true,
		MaxConnsPerHost:      100,
		MaxIdleConns:         50,
		MaxIdleTime:          90 * time.Second,
		ConnTimeout:          10 * time.Second,
		EnablePacketCoalesce: true,
		CoalesceDelay:        1 * time.Millisecond,
		CoalesceMaxSize:      64 * 1024,
		EnableRateLimiting:   false,
		MaxBandwidth:         10 * 1024 * 1024 * 1024, // 10 GB/s
		BurstSize:            1024 * 1024,              // 1MB
		SendBufferSize:       4 * 1024 * 1024,         // 4MB
		RecvBufferSize:       4 * 1024 * 1024,         // 4MB
	}
}

// NetworkOptimizer provides network optimization for DWCP v3.
type NetworkOptimizer struct {
	config *NetworkOptimizerConfig
	mu     sync.RWMutex

	// Connection pools
	connPools map[string]*ConnectionPool

	// RDMA context
	rdmaContext *RDMAContext

	// TCP optimizer
	tcpOptimizer *TCPOptimizer

	// Packet coalescer
	coalescer *PacketCoalescer

	// Rate limiter
	rateLimiter *RateLimiter

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ConnectionPool manages connection pooling.
type ConnectionPool struct {
	host     string
	maxConns int
	maxIdle  int
	timeout  time.Duration

	mu      sync.Mutex
	conns   []net.Conn
	idle    []net.Conn
	created int
}

// RDMAContext manages RDMA resources.
type RDMAContext struct {
	queueDepth      int
	maxSGE          int
	completionQueue int
	inlineSize      int

	mu sync.Mutex
	// RDMA resources would go here
	queuePairs map[string]*RDMAQueuePair
}

// RDMAQueuePair represents an RDMA queue pair.
type RDMAQueuePair struct {
	id         string
	sendQueue  chan []byte
	recvQueue  chan []byte
	completions chan *RDMACompletion
}

// RDMACompletion represents an RDMA operation completion.
type RDMACompletion struct {
	OpID   uint64
	Status int
	Bytes  uint64
	Error  error
}

// TCPOptimizer optimizes TCP connections.
type TCPOptimizer struct {
	congestion       string
	windowSize       int
	noDelay          bool
	quickAck         bool
	keepAlive        bool
	keepAliveIdle    time.Duration
	keepAliveInterval time.Duration
	sendBuffer       int
	recvBuffer       int
}

// PacketCoalescer coalesces small packets.
type PacketCoalescer struct {
	enabled  bool
	delay    time.Duration
	maxSize  int

	mu      sync.Mutex
	buffers map[string]*coalesceBuffer
}

// coalesceBuffer buffers packets for coalescing.
type coalesceBuffer struct {
	data   []byte
	timer  *time.Timer
	mu     sync.Mutex
	sender func([]byte) error
}

// RateLimiter implements token bucket rate limiting.
type RateLimiter struct {
	enabled   bool
	rate      uint64 // bytes/sec
	burst     int
	tokens    float64
	lastCheck time.Time
	mu        sync.Mutex
}

// NewNetworkOptimizer creates a new network optimizer.
func NewNetworkOptimizer(config *NetworkOptimizerConfig) (*NetworkOptimizer, error) {
	if config == nil {
		config = DefaultNetworkOptimizerConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	o := &NetworkOptimizer{
		config:    config,
		connPools: make(map[string]*ConnectionPool),
		ctx:       ctx,
		cancel:    cancel,
	}

	// Initialize RDMA context
	if config.RDMAEnabled {
		o.rdmaContext = &RDMAContext{
			queueDepth:      config.RDMAQueueDepth,
			maxSGE:          config.RDMAMaxSGE,
			completionQueue: config.RDMACompletionQueue,
			inlineSize:      config.RDMAInlineSize,
			queuePairs:      make(map[string]*RDMAQueuePair),
		}
	}

	// Initialize TCP optimizer
	if config.TCPEnabled {
		o.tcpOptimizer = &TCPOptimizer{
			congestion:        config.TCPCongestion,
			windowSize:        config.TCPWindowSize,
			noDelay:           config.TCPNoDelay,
			quickAck:          config.TCPQuickAck,
			keepAlive:         config.TCPKeepAlive,
			keepAliveIdle:     config.TCPKeepAliveIdle,
			keepAliveInterval: config.TCPKeepAliveInterval,
			sendBuffer:        config.SendBufferSize,
			recvBuffer:        config.RecvBufferSize,
		}
	}

	// Initialize packet coalescer
	if config.EnablePacketCoalesce {
		o.coalescer = &PacketCoalescer{
			enabled: true,
			delay:   config.CoalesceDelay,
			maxSize: config.CoalesceMaxSize,
			buffers: make(map[string]*coalesceBuffer),
		}
	}

	// Initialize rate limiter
	if config.EnableRateLimiting {
		o.rateLimiter = &RateLimiter{
			enabled:   true,
			rate:      config.MaxBandwidth,
			burst:     config.BurstSize,
			tokens:    float64(config.BurstSize),
			lastCheck: time.Now(),
		}
	}

	return o, nil
}

// GetConnection gets a connection from the pool.
func (o *NetworkOptimizer) GetConnection(host string) (net.Conn, error) {
	if !o.config.EnableConnPool {
		return net.DialTimeout("tcp", host, o.config.ConnTimeout)
	}

	o.mu.Lock()
	pool, exists := o.connPools[host]
	if !exists {
		pool = &ConnectionPool{
			host:     host,
			maxConns: o.config.MaxConnsPerHost,
			maxIdle:  o.config.MaxIdleConns,
			timeout:  o.config.ConnTimeout,
			conns:    make([]net.Conn, 0),
			idle:     make([]net.Conn, 0),
		}
		o.connPools[host] = pool
	}
	o.mu.Unlock()

	return pool.Get()
}

// PutConnection returns a connection to the pool.
func (o *NetworkOptimizer) PutConnection(host string, conn net.Conn) error {
	if !o.config.EnableConnPool {
		return conn.Close()
	}

	o.mu.RLock()
	pool, exists := o.connPools[host]
	o.mu.RUnlock()

	if !exists {
		return conn.Close()
	}

	return pool.Put(conn)
}

// Get gets a connection from the pool.
func (cp *ConnectionPool) Get() (net.Conn, error) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Try to get idle connection
	if len(cp.idle) > 0 {
		conn := cp.idle[len(cp.idle)-1]
		cp.idle = cp.idle[:len(cp.idle)-1]
		return conn, nil
	}

	// Create new connection if under limit
	if cp.created < cp.maxConns {
		conn, err := net.DialTimeout("tcp", cp.host, cp.timeout)
		if err != nil {
			return nil, err
		}
		cp.created++
		cp.conns = append(cp.conns, conn)
		return conn, nil
	}

	return nil, fmt.Errorf("connection pool exhausted")
}

// Put returns a connection to the pool.
func (cp *ConnectionPool) Put(conn net.Conn) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Check if we can keep it idle
	if len(cp.idle) < cp.maxIdle {
		cp.idle = append(cp.idle, conn)
		return nil
	}

	// Pool full, close connection
	return conn.Close()
}

// OptimizeTCPConnection optimizes a TCP connection.
func (o *NetworkOptimizer) OptimizeTCPConnection(conn net.Conn) error {
	if o.tcpOptimizer == nil {
		return nil
	}

	tcpConn, ok := conn.(*net.TCPConn)
	if !ok {
		return fmt.Errorf("not a TCP connection")
	}

	// Set TCP_NODELAY
	if o.tcpOptimizer.noDelay {
		if err := tcpConn.SetNoDelay(true); err != nil {
			return fmt.Errorf("set TCP_NODELAY: %w", err)
		}
	}

	// Set keepalive
	if o.tcpOptimizer.keepAlive {
		if err := tcpConn.SetKeepAlive(true); err != nil {
			return fmt.Errorf("set keepalive: %w", err)
		}
		if err := tcpConn.SetKeepAlivePeriod(o.tcpOptimizer.keepAliveIdle); err != nil {
			return fmt.Errorf("set keepalive period: %w", err)
		}
	}

	// Set buffer sizes
	rawConn, err := tcpConn.SyscallConn()
	if err != nil {
		return fmt.Errorf("get raw conn: %w", err)
	}

	var setErr error
	err = rawConn.Control(func(fd uintptr) {
		// Set send buffer
		if err := syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_SNDBUF, o.tcpOptimizer.sendBuffer); err != nil {
			setErr = fmt.Errorf("set send buffer: %w", err)
			return
		}

		// Set receive buffer
		if err := syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_RCVBUF, o.tcpOptimizer.recvBuffer); err != nil {
			setErr = fmt.Errorf("set recv buffer: %w", err)
			return
		}

		// Set TCP_QUICKACK (Linux only)
		if o.tcpOptimizer.quickAck {
			syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, 0x0C, 1) // TCP_QUICKACK = 12
		}

		// Set congestion control (Linux only)
		if o.tcpOptimizer.congestion != "" {
			syscall.SetsockoptString(int(fd), syscall.IPPROTO_TCP, 13, o.tcpOptimizer.congestion) // TCP_CONGESTION = 13
		}
	})

	if err != nil {
		return fmt.Errorf("control: %w", err)
	}
	if setErr != nil {
		return setErr
	}

	return nil
}

// CreateRDMAQueuePair creates an RDMA queue pair.
func (o *NetworkOptimizer) CreateRDMAQueuePair(id string) (*RDMAQueuePair, error) {
	if o.rdmaContext == nil {
		return nil, fmt.Errorf("RDMA not enabled")
	}

	o.rdmaContext.mu.Lock()
	defer o.rdmaContext.mu.Unlock()

	if _, exists := o.rdmaContext.queuePairs[id]; exists {
		return nil, fmt.Errorf("queue pair already exists: %s", id)
	}

	qp := &RDMAQueuePair{
		id:          id,
		sendQueue:   make(chan []byte, o.rdmaContext.queueDepth),
		recvQueue:   make(chan []byte, o.rdmaContext.queueDepth),
		completions: make(chan *RDMACompletion, o.rdmaContext.completionQueue),
	}

	o.rdmaContext.queuePairs[id] = qp

	// Start completion handler
	o.wg.Add(1)
	go o.handleRDMACompletions(qp)

	return qp, nil
}

// RDMAWrite performs an RDMA write operation.
func (o *NetworkOptimizer) RDMAWrite(qpID string, data []byte) error {
	if o.rdmaContext == nil {
		return fmt.Errorf("RDMA not enabled")
	}

	o.rdmaContext.mu.Lock()
	qp, exists := o.rdmaContext.queuePairs[qpID]
	o.rdmaContext.mu.Unlock()

	if !exists {
		return fmt.Errorf("queue pair not found: %s", qpID)
	}

	select {
	case qp.sendQueue <- data:
		return nil
	default:
		return fmt.Errorf("send queue full")
	}
}

// RDMARead performs an RDMA read operation.
func (o *NetworkOptimizer) RDMARead(qpID string) ([]byte, error) {
	if o.rdmaContext == nil {
		return nil, fmt.Errorf("RDMA not enabled")
	}

	o.rdmaContext.mu.Lock()
	qp, exists := o.rdmaContext.queuePairs[qpID]
	o.rdmaContext.mu.Unlock()

	if !exists {
		return nil, fmt.Errorf("queue pair not found: %s", qpID)
	}

	select {
	case data := <-qp.recvQueue:
		return data, nil
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("read timeout")
	}
}

// handleRDMACompletions handles RDMA completions.
func (o *NetworkOptimizer) handleRDMACompletions(qp *RDMAQueuePair) {
	defer o.wg.Done()

	for {
		select {
		case <-o.ctx.Done():
			return

		case completion := <-qp.completions:
			// Process completion
			if completion.Error != nil {
				// Handle error
			}
			// In real implementation, would update metrics, etc.
		}
	}
}

// CoalescePacket coalesces a packet.
func (o *NetworkOptimizer) CoalescePacket(dest string, data []byte, sender func([]byte) error) error {
	if o.coalescer == nil || !o.coalescer.enabled {
		return sender(data)
	}

	o.coalescer.mu.Lock()
	buf, exists := o.coalescer.buffers[dest]
	if !exists {
		buf = &coalesceBuffer{
			data:   make([]byte, 0, o.coalescer.maxSize),
			sender: sender,
		}
		o.coalescer.buffers[dest] = buf
	}
	o.coalescer.mu.Unlock()

	buf.mu.Lock()
	defer buf.mu.Unlock()

	// Add data to buffer
	buf.data = append(buf.data, data...)

	// Flush if buffer is full
	if len(buf.data) >= o.coalescer.maxSize {
		return o.flushCoalesceBuffer(dest, buf)
	}

	// Start timer if not already running
	if buf.timer == nil {
		buf.timer = time.AfterFunc(o.coalescer.delay, func() {
			buf.mu.Lock()
			defer buf.mu.Unlock()
			o.flushCoalesceBuffer(dest, buf)
		})
	}

	return nil
}

// flushCoalesceBuffer flushes a coalesce buffer.
func (o *NetworkOptimizer) flushCoalesceBuffer(dest string, buf *coalesceBuffer) error {
	if len(buf.data) == 0 {
		return nil
	}

	data := buf.data
	buf.data = make([]byte, 0, o.coalescer.maxSize)

	if buf.timer != nil {
		buf.timer.Stop()
		buf.timer = nil
	}

	return buf.sender(data)
}

// RateLimit applies rate limiting to data transfer.
func (o *NetworkOptimizer) RateLimit(bytes uint64) error {
	if o.rateLimiter == nil || !o.rateLimiter.enabled {
		return nil
	}

	o.rateLimiter.mu.Lock()
	defer o.rateLimiter.mu.Unlock()

	// Refill tokens based on elapsed time
	now := time.Now()
	elapsed := now.Sub(o.rateLimiter.lastCheck).Seconds()
	o.rateLimiter.tokens += float64(o.rateLimiter.rate) * elapsed

	// Cap tokens at burst size
	if o.rateLimiter.tokens > float64(o.rateLimiter.burst) {
		o.rateLimiter.tokens = float64(o.rateLimiter.burst)
	}

	o.rateLimiter.lastCheck = now

	// Check if we have enough tokens
	if o.rateLimiter.tokens < float64(bytes) {
		// Calculate wait time
		deficit := float64(bytes) - o.rateLimiter.tokens
		waitTime := time.Duration(deficit / float64(o.rateLimiter.rate) * float64(time.Second))

		// Wait for tokens
		time.Sleep(waitTime)

		o.rateLimiter.tokens = 0
	} else {
		o.rateLimiter.tokens -= float64(bytes)
	}

	return nil
}

// GetNetworkStats returns network optimization statistics.
func (o *NetworkOptimizer) GetNetworkStats() map[string]interface{} {
	stats := map[string]interface{}{
		"connection_pools": make(map[string]interface{}),
	}

	o.mu.RLock()
	for host, pool := range o.connPools {
		pool.mu.Lock()
		stats["connection_pools"].(map[string]interface{})[host] = map[string]interface{}{
			"total":   pool.created,
			"idle":    len(pool.idle),
			"max":     pool.maxConns,
			"maxidle": pool.maxIdle,
		}
		pool.mu.Unlock()
	}
	o.mu.RUnlock()

	if o.rdmaContext != nil {
		o.rdmaContext.mu.Lock()
		stats["rdma"] = map[string]interface{}{
			"queue_pairs":   len(o.rdmaContext.queuePairs),
			"queue_depth":   o.rdmaContext.queueDepth,
			"max_sge":       o.rdmaContext.maxSGE,
			"inline_size":   o.rdmaContext.inlineSize,
		}
		o.rdmaContext.mu.Unlock()
	}

	if o.rateLimiter != nil {
		o.rateLimiter.mu.Lock()
		stats["rate_limiter"] = map[string]interface{}{
			"enabled": o.rateLimiter.enabled,
			"rate":    o.rateLimiter.rate,
			"tokens":  o.rateLimiter.tokens,
		}
		o.rateLimiter.mu.Unlock()
	}

	return stats
}

// Close stops the optimizer and cleans up resources.
func (o *NetworkOptimizer) Close() error {
	o.cancel()
	o.wg.Wait()

	// Close all connections
	o.mu.Lock()
	for _, pool := range o.connPools {
		pool.mu.Lock()
		for _, conn := range pool.conns {
			conn.Close()
		}
		pool.mu.Unlock()
	}
	o.mu.Unlock()

	return nil
}
