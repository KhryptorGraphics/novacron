// Package dwcpv4 implements HTTP/3 and QUIC transport for next-generation DWCP
// This prototype explores low-latency, multiplexed communication for distributed VMs
package dwcpv4

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"
)

// QUICTransport provides HTTP/3 and QUIC-based communication for DWCP v4
// Key features:
// - 0-RTT connection establishment for reduced latency
// - Stream multiplexing without head-of-line blocking
// - Built-in congestion control and loss recovery
// - Connection migration for mobile/edge scenarios
type QUICTransport struct {
	mu          sync.RWMutex
	config      *QUICConfig
	listeners   map[string]*QUICListener
	connections map[string]*QUICConnection
	streams     map[string]*QUICStream
	logger      *log.Logger
	metrics     *QUICMetrics
}

// QUICConfig configures QUIC transport parameters
type QUICConfig struct {
	ListenAddr            string        `json:"listen_addr"`
	MaxStreamsPerConn     int           `json:"max_streams_per_conn"`
	MaxConnectionsPerPeer int           `json:"max_connections_per_peer"`
	Enable0RTT            bool          `json:"enable_0rtt"`
	EnableEarlyData       bool          `json:"enable_early_data"`
	IdleTimeout           time.Duration `json:"idle_timeout"`
	MaxIdleTimeout        time.Duration `json:"max_idle_timeout"`
	KeepAlivePeriod       time.Duration `json:"keep_alive_period"`
	InitialStreamWindow   uint64        `json:"initial_stream_window"`
	InitialConnWindow     uint64        `json:"initial_conn_window"`
	MaxStreamWindow       uint64        `json:"max_stream_window"`
	MaxConnWindow         uint64        `json:"max_conn_window"`
	TLSConfig             *tls.Config   `json:"-"`
}

// QUICListener listens for incoming QUIC connections
type QUICListener struct {
	ID        string         `json:"id"`
	Address   string         `json:"address"`
	Transport *QUICTransport `json:"-"`
	Active    bool           `json:"active"`
	CreatedAt time.Time      `json:"created_at"`
}

// QUICConnection represents a QUIC connection to a peer
type QUICConnection struct {
	ID              string                 `json:"id"`
	LocalAddr       string                 `json:"local_addr"`
	RemoteAddr      string                 `json:"remote_addr"`
	State           QUICConnectionState    `json:"state"`
	Streams         map[string]*QUICStream `json:"streams"`
	RTT             time.Duration          `json:"rtt"`
	CongestionWindow uint64                `json:"congestion_window"`
	BytesSent       uint64                 `json:"bytes_sent"`
	BytesReceived   uint64                 `json:"bytes_received"`
	PacketsSent     uint64                 `json:"packets_sent"`
	PacketsReceived uint64                 `json:"packets_received"`
	PacketsLost     uint64                 `json:"packets_lost"`
	Metadata        map[string]interface{} `json:"metadata"`
	EstablishedAt   time.Time              `json:"established_at"`
	ClosedAt        *time.Time             `json:"closed_at,omitempty"`
}

// QUICConnectionState represents connection lifecycle state
type QUICConnectionState string

const (
	ConnStateHandshaking QUICConnectionState = "handshaking"
	ConnStateEstablished QUICConnectionState = "established"
	ConnStateMigrating   QUICConnectionState = "migrating"
	ConnStateClosing     QUICConnectionState = "closing"
	ConnStateClosed      QUICConnectionState = "closed"
)

// QUICStream represents a bidirectional QUIC stream
type QUICStream struct {
	ID           string           `json:"id"`
	ConnectionID string           `json:"connection_id"`
	StreamID     uint64           `json:"stream_id"`
	State        QUICStreamState  `json:"state"`
	BytesSent    uint64           `json:"bytes_sent"`
	BytesReceived uint64          `json:"bytes_received"`
	Priority     int              `json:"priority"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time        `json:"created_at"`
	ClosedAt     *time.Time       `json:"closed_at,omitempty"`
}

// QUICStreamState represents stream lifecycle state
type QUICStreamState string

const (
	StreamStateOpen   QUICStreamState = "open"
	StreamStateActive QUICStreamState = "active"
	StreamStateClosed QUICStreamState = "closed"
)

// QUICMetrics tracks transport performance
type QUICMetrics struct {
	mu                  sync.RWMutex
	TotalConnections    int64         `json:"total_connections"`
	ActiveConnections   int64         `json:"active_connections"`
	TotalStreams        int64         `json:"total_streams"`
	ActiveStreams       int64         `json:"active_streams"`
	BytesSent           uint64        `json:"bytes_sent"`
	BytesReceived       uint64        `json:"bytes_received"`
	PacketsSent         uint64        `json:"packets_sent"`
	PacketsReceived     uint64        `json:"packets_received"`
	PacketsLost         uint64        `json:"packets_lost"`
	AvgRTT              time.Duration `json:"avg_rtt"`
	ConnectionsUsing0RTT int64        `json:"connections_using_0rtt"`
	ConnectionMigrations int64        `json:"connection_migrations"`
}

// NewQUICTransport creates a new QUIC transport
func NewQUICTransport(config *QUICConfig, logger *log.Logger) (*QUICTransport, error) {
	if config == nil {
		config = defaultQUICConfig()
	}

	return &QUICTransport{
		config:      config,
		listeners:   make(map[string]*QUICListener),
		connections: make(map[string]*QUICConnection),
		streams:     make(map[string]*QUICStream),
		logger:      logger,
		metrics:     &QUICMetrics{},
	}, nil
}

// Listen starts listening for QUIC connections
func (qt *QUICTransport) Listen(ctx context.Context, addr string) (*QUICListener, error) {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	listener := &QUICListener{
		ID:        fmt.Sprintf("quic-listener-%d", time.Now().UnixNano()),
		Address:   addr,
		Transport: qt,
		Active:    true,
		CreatedAt: time.Now(),
	}

	qt.listeners[listener.ID] = listener
	qt.logger.Printf("Started QUIC listener on %s", addr)

	// Start accepting connections in background
	go qt.acceptConnections(ctx, listener)

	return listener, nil
}

// Connect establishes a QUIC connection to a remote peer
func (qt *QUICTransport) Connect(ctx context.Context, remoteAddr string, use0RTT bool) (*QUICConnection, error) {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	localAddr := qt.config.ListenAddr
	if localAddr == "" {
		localAddr = "0.0.0.0:0"
	}

	conn := &QUICConnection{
		ID:              fmt.Sprintf("quic-conn-%d", time.Now().UnixNano()),
		LocalAddr:       localAddr,
		RemoteAddr:      remoteAddr,
		State:           ConnStateHandshaking,
		Streams:         make(map[string]*QUICStream),
		RTT:             0,
		CongestionWindow: qt.config.InitialConnWindow,
		Metadata:        make(map[string]interface{}),
		EstablishedAt:   time.Now(),
	}

	// Simulate connection establishment
	if use0RTT && qt.config.Enable0RTT {
		// 0-RTT handshake - no round trip needed
		qt.logger.Printf("Establishing 0-RTT connection to %s", remoteAddr)
		conn.State = ConnStateEstablished
		qt.metrics.mu.Lock()
		qt.metrics.ConnectionsUsing0RTT++
		qt.metrics.mu.Unlock()
	} else {
		// Standard 1-RTT handshake
		qt.logger.Printf("Establishing 1-RTT connection to %s", remoteAddr)
		time.Sleep(10 * time.Millisecond) // Simulate RTT
		conn.State = ConnStateEstablished
		conn.RTT = 10 * time.Millisecond
	}

	qt.connections[conn.ID] = conn

	qt.metrics.mu.Lock()
	qt.metrics.TotalConnections++
	qt.metrics.ActiveConnections++
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Connected to %s (ID: %s, 0-RTT: %v)", remoteAddr, conn.ID, use0RTT)

	return conn, nil
}

// OpenStream opens a new bidirectional stream on a connection
func (qt *QUICTransport) OpenStream(ctx context.Context, connID string, priority int) (*QUICStream, error) {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	conn, exists := qt.connections[connID]
	if !exists {
		return nil, fmt.Errorf("connection not found: %s", connID)
	}

	if conn.State != ConnStateEstablished {
		return nil, fmt.Errorf("connection not established: %s", conn.State)
	}

	if len(conn.Streams) >= qt.config.MaxStreamsPerConn {
		return nil, fmt.Errorf("maximum streams per connection reached: %d", qt.config.MaxStreamsPerConn)
	}

	stream := &QUICStream{
		ID:           fmt.Sprintf("quic-stream-%d", time.Now().UnixNano()),
		ConnectionID: connID,
		StreamID:     uint64(len(conn.Streams)),
		State:        StreamStateOpen,
		Priority:     priority,
		Metadata:     make(map[string]interface{}),
		CreatedAt:    time.Now(),
	}

	conn.Streams[stream.ID] = stream
	qt.streams[stream.ID] = stream

	qt.metrics.mu.Lock()
	qt.metrics.TotalStreams++
	qt.metrics.ActiveStreams++
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Opened stream %d on connection %s (Priority: %d)", stream.StreamID, connID, priority)

	return stream, nil
}

// Send sends data on a QUIC stream
func (qt *QUICTransport) Send(ctx context.Context, streamID string, data []byte) (int, error) {
	qt.mu.RLock()
	stream, exists := qt.streams[streamID]
	qt.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("stream not found: %s", streamID)
	}

	if stream.State == StreamStateClosed {
		return 0, fmt.Errorf("stream closed: %s", streamID)
	}

	qt.mu.RLock()
	conn, exists := qt.connections[stream.ConnectionID]
	qt.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("connection not found: %s", stream.ConnectionID)
	}

	// Simulate sending with congestion control
	dataLen := uint64(len(data))

	qt.mu.Lock()
	stream.BytesSent += dataLen
	stream.State = StreamStateActive
	conn.BytesSent += dataLen
	conn.PacketsSent += (dataLen / 1200) + 1 // Simulate packets
	qt.mu.Unlock()

	qt.metrics.mu.Lock()
	qt.metrics.BytesSent += dataLen
	qt.metrics.PacketsSent += (dataLen / 1200) + 1
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Sent %d bytes on stream %d", len(data), stream.StreamID)

	return len(data), nil
}

// Receive receives data from a QUIC stream
func (qt *QUICTransport) Receive(ctx context.Context, streamID string, buffer []byte) (int, error) {
	qt.mu.RLock()
	stream, exists := qt.streams[streamID]
	qt.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("stream not found: %s", streamID)
	}

	if stream.State == StreamStateClosed {
		return 0, io.EOF
	}

	// Simulate receiving data
	time.Sleep(5 * time.Millisecond)

	n := len(buffer)
	if n > 1024 {
		n = 1024 // Simulate partial read
	}

	qt.mu.Lock()
	stream.BytesReceived += uint64(n)
	qt.mu.Unlock()

	qt.mu.RLock()
	conn, _ := qt.connections[stream.ConnectionID]
	qt.mu.RUnlock()

	qt.mu.Lock()
	if conn != nil {
		conn.BytesReceived += uint64(n)
		conn.PacketsReceived += 1
	}
	qt.mu.Unlock()

	qt.metrics.mu.Lock()
	qt.metrics.BytesReceived += uint64(n)
	qt.metrics.PacketsReceived += 1
	qt.metrics.mu.Unlock()

	return n, nil
}

// CloseStream closes a QUIC stream
func (qt *QUICTransport) CloseStream(streamID string) error {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	stream, exists := qt.streams[streamID]
	if !exists {
		return fmt.Errorf("stream not found: %s", streamID)
	}

	if stream.State == StreamStateClosed {
		return nil // Already closed
	}

	now := time.Now()
	stream.State = StreamStateClosed
	stream.ClosedAt = &now

	qt.metrics.mu.Lock()
	qt.metrics.ActiveStreams--
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Closed stream %d", stream.StreamID)

	return nil
}

// CloseConnection closes a QUIC connection
func (qt *QUICTransport) CloseConnection(connID string) error {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	conn, exists := qt.connections[connID]
	if !exists {
		return fmt.Errorf("connection not found: %s", connID)
	}

	if conn.State == ConnStateClosed {
		return nil // Already closed
	}

	conn.State = ConnStateClosing

	// Close all streams
	for streamID, stream := range conn.Streams {
		if stream.State != StreamStateClosed {
			now := time.Now()
			stream.State = StreamStateClosed
			stream.ClosedAt = &now

			qt.metrics.mu.Lock()
			qt.metrics.ActiveStreams--
			qt.metrics.mu.Unlock()
		}
		delete(qt.streams, streamID)
	}

	now := time.Now()
	conn.State = ConnStateClosed
	conn.ClosedAt = &now

	qt.metrics.mu.Lock()
	qt.metrics.ActiveConnections--
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Closed connection %s", connID)

	return nil
}

// MigrateConnection migrates a connection to a new network path
func (qt *QUICTransport) MigrateConnection(connID string, newLocalAddr string) error {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	conn, exists := qt.connections[connID]
	if !exists {
		return fmt.Errorf("connection not found: %s", connID)
	}

	if conn.State != ConnStateEstablished {
		return fmt.Errorf("connection not established: %s", conn.State)
	}

	oldAddr := conn.LocalAddr
	conn.State = ConnStateMigrating

	// Simulate migration
	time.Sleep(20 * time.Millisecond)

	conn.LocalAddr = newLocalAddr
	conn.State = ConnStateEstablished

	qt.metrics.mu.Lock()
	qt.metrics.ConnectionMigrations++
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Migrated connection %s from %s to %s", connID, oldAddr, newLocalAddr)

	return nil
}

// GetMetrics returns current transport metrics
func (qt *QUICTransport) GetMetrics() *QUICMetrics {
	qt.metrics.mu.RLock()
	defer qt.metrics.mu.RUnlock()

	return &QUICMetrics{
		TotalConnections:    qt.metrics.TotalConnections,
		ActiveConnections:   qt.metrics.ActiveConnections,
		TotalStreams:        qt.metrics.TotalStreams,
		ActiveStreams:       qt.metrics.ActiveStreams,
		BytesSent:           qt.metrics.BytesSent,
		BytesReceived:       qt.metrics.BytesReceived,
		PacketsSent:         qt.metrics.PacketsSent,
		PacketsReceived:     qt.metrics.PacketsReceived,
		PacketsLost:         qt.metrics.PacketsLost,
		AvgRTT:              qt.metrics.AvgRTT,
		ConnectionsUsing0RTT: qt.metrics.ConnectionsUsing0RTT,
		ConnectionMigrations: qt.metrics.ConnectionMigrations,
	}
}

// acceptConnections accepts incoming QUIC connections (simplified)
func (qt *QUICTransport) acceptConnections(ctx context.Context, listener *QUICListener) {
	// Simulated connection acceptance
	// In production, integrate with actual QUIC library like quic-go

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Simulate incoming connection
			qt.simulateIncomingConnection(listener)
		}
	}
}

// simulateIncomingConnection simulates an incoming connection
func (qt *QUICTransport) simulateIncomingConnection(listener *QUICListener) {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	remoteAddr := fmt.Sprintf("192.168.1.%d:12345", 100+len(qt.connections))

	conn := &QUICConnection{
		ID:              fmt.Sprintf("quic-conn-%d", time.Now().UnixNano()),
		LocalAddr:       listener.Address,
		RemoteAddr:      remoteAddr,
		State:           ConnStateEstablished,
		Streams:         make(map[string]*QUICStream),
		RTT:             15 * time.Millisecond,
		CongestionWindow: qt.config.InitialConnWindow,
		Metadata:        make(map[string]interface{}),
		EstablishedAt:   time.Now(),
	}

	qt.connections[conn.ID] = conn

	qt.metrics.mu.Lock()
	qt.metrics.TotalConnections++
	qt.metrics.ActiveConnections++
	qt.metrics.mu.Unlock()

	qt.logger.Printf("Accepted connection from %s", remoteAddr)
}

// defaultQUICConfig returns default QUIC configuration
func defaultQUICConfig() *QUICConfig {
	return &QUICConfig{
		ListenAddr:            "0.0.0.0:4433",
		MaxStreamsPerConn:     100,
		MaxConnectionsPerPeer: 10,
		Enable0RTT:            true,
		EnableEarlyData:       true,
		IdleTimeout:           30 * time.Second,
		MaxIdleTimeout:        120 * time.Second,
		KeepAlivePeriod:       10 * time.Second,
		InitialStreamWindow:   512 * 1024,  // 512KB
		InitialConnWindow:     1024 * 1024, // 1MB
		MaxStreamWindow:       6 * 1024 * 1024, // 6MB
		MaxConnWindow:         15 * 1024 * 1024, // 15MB
	}
}

// HTTP3Handler handles HTTP/3 requests over QUIC
type HTTP3Handler struct {
	transport *QUICTransport
	logger    *log.Logger
}

// NewHTTP3Handler creates an HTTP/3 request handler
func NewHTTP3Handler(transport *QUICTransport, logger *log.Logger) *HTTP3Handler {
	return &HTTP3Handler{
		transport: transport,
		logger:    logger,
	}
}

// HandleRequest handles an HTTP/3 request
func (h *HTTP3Handler) HandleRequest(ctx context.Context, connID string, method string, path string, body []byte) ([]byte, int, error) {
	h.logger.Printf("HTTP/3 %s %s", method, path)

	// Open stream for request
	stream, err := h.transport.OpenStream(ctx, connID, 0)
	if err != nil {
		return nil, 500, fmt.Errorf("failed to open stream: %w", err)
	}
	defer h.transport.CloseStream(stream.ID)

	// Send request
	if len(body) > 0 {
		if _, err := h.transport.Send(ctx, stream.ID, body); err != nil {
			return nil, 500, fmt.Errorf("failed to send request: %w", err)
		}
	}

	// Receive response (simplified)
	responseBuffer := make([]byte, 4096)
	n, err := h.transport.Receive(ctx, stream.ID, responseBuffer)
	if err != nil && err != io.EOF {
		return nil, 500, fmt.Errorf("failed to receive response: %w", err)
	}

	return responseBuffer[:n], 200, nil
}

// GetConnectionStats returns connection statistics
func (qt *QUICTransport) GetConnectionStats(connID string) (*QUICConnection, error) {
	qt.mu.RLock()
	defer qt.mu.RUnlock()

	conn, exists := qt.connections[connID]
	if !exists {
		return nil, fmt.Errorf("connection not found: %s", connID)
	}

	// Return a copy
	connCopy := *conn
	return &connCopy, nil
}

// OptimizeConnection applies congestion control optimizations
func (qt *QUICTransport) OptimizeConnection(connID string) error {
	qt.mu.Lock()
	defer qt.mu.Unlock()

	conn, exists := qt.connections[connID]
	if !exists {
		return fmt.Errorf("connection not found: %s", connID)
	}

	// Simulate BBR congestion control optimization
	if conn.PacketsLost > 0 {
		lossRate := float64(conn.PacketsLost) / float64(conn.PacketsSent)
		if lossRate > 0.01 { // >1% loss
			conn.CongestionWindow = conn.CongestionWindow / 2
			qt.logger.Printf("Reduced congestion window for %s due to loss (%.2f%%)", connID, lossRate*100)
		}
	} else {
		// Increase window if no loss
		conn.CongestionWindow = conn.CongestionWindow * 2
		if conn.CongestionWindow > qt.config.MaxConnWindow {
			conn.CongestionWindow = qt.config.MaxConnWindow
		}
	}

	return nil
}
