package rdma

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// RDMAManager provides high-level RDMA operations with Go semantics
type RDMAManager struct {
	ctx       *Context
	logger    *zap.Logger
	config    *Config

	// Memory management
	sendBuffer     []byte
	recvBuffer     []byte
	sendBufSize    int
	recvBufSize    int

	// Connection state
	localInfo      ConnInfo
	remoteInfo     ConnInfo
	connected      atomic.Bool

	// Work request tracking
	nextWRID       atomic.Uint64
	pendingSends   sync.Map // wrID -> *pendingOp
	pendingRecvs   sync.Map // wrID -> *pendingOp

	// Statistics
	stats          RDMAStats

	// Lifecycle
	shutdownCtx    context.Context
	shutdownCancel context.CancelFunc
	wg             sync.WaitGroup
	mu             sync.RWMutex
}

// Config represents RDMA configuration
type Config struct {
	DeviceName      string        `json:"device_name" yaml:"device_name"`
	Port            int           `json:"port" yaml:"port"`
	GIDIndex        int           `json:"gid_index" yaml:"gid_index"`
	MTU             int           `json:"mtu" yaml:"mtu"`
	MaxInlineData   int           `json:"max_inline_data" yaml:"max_inline_data"`
	MaxSendWR       int           `json:"max_send_wr" yaml:"max_send_wr"`
	MaxRecvWR       int           `json:"max_recv_wr" yaml:"max_recv_wr"`
	MaxSGE          int           `json:"max_sge" yaml:"max_sge"`
	QPType          string        `json:"qp_type" yaml:"qp_type"` // RC, UD, DCT
	UseSRQ          bool          `json:"use_srq" yaml:"use_srq"`
	UseEventChannel bool          `json:"use_event_channel" yaml:"use_event_channel"`
	SendBufferSize  int           `json:"send_buffer_size" yaml:"send_buffer_size"`
	RecvBufferSize  int           `json:"recv_buffer_size" yaml:"recv_buffer_size"`
}

// DefaultConfig returns default RDMA configuration
func DefaultConfig() *Config {
	return &Config{
		DeviceName:      "", // Auto-detect
		Port:            1,
		GIDIndex:        0,
		MTU:             4096,
		MaxInlineData:   256,
		MaxSendWR:       1024,
		MaxRecvWR:       1024,
		MaxSGE:          16,
		QPType:          "RC", // Reliable Connection
		UseSRQ:          false,
		UseEventChannel: false, // Use polling for lowest latency
		SendBufferSize:  4 * 1024 * 1024, // 4MB
		RecvBufferSize:  4 * 1024 * 1024, // 4MB
	}
}

// RDMAStats tracks RDMA operation statistics
type RDMAStats struct {
	SendOperations     atomic.Uint64
	RecvOperations     atomic.Uint64
	WriteOperations    atomic.Uint64
	ReadOperations     atomic.Uint64
	SendCompletions    atomic.Uint64
	RecvCompletions    atomic.Uint64
	SendErrors         atomic.Uint64
	RecvErrors         atomic.Uint64
	BytesSent          atomic.Uint64
	BytesReceived      atomic.Uint64
	AvgSendLatencyNs   atomic.Uint64
	AvgRecvLatencyNs   atomic.Uint64
	MinSendLatencyNs   atomic.Uint64
	MaxSendLatencyNs   atomic.Uint64
}

type pendingOp struct {
	wrID      uint64
	buffer    []byte
	startTime time.Time
	done      chan error
}

// NewRDMAManager creates a new RDMA manager
func NewRDMAManager(config *Config, logger *zap.Logger) (*RDMAManager, error) {
	if config == nil {
		config = DefaultConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	// Check RDMA availability
	if !CheckAvailability() {
		return nil, fmt.Errorf("RDMA not available on this system")
	}

	// Initialize RDMA context
	ctx, err := Initialize(config.DeviceName, config.Port, config.UseEventChannel)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RDMA: %w", err)
	}

	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())

	mgr := &RDMAManager{
		ctx:            ctx,
		logger:         logger,
		config:         config,
		sendBufSize:    config.SendBufferSize,
		recvBufSize:    config.RecvBufferSize,
		shutdownCtx:    shutdownCtx,
		shutdownCancel: shutdownCancel,
	}

	// Allocate buffers
	mgr.sendBuffer = make([]byte, mgr.sendBufSize)
	mgr.recvBuffer = make([]byte, mgr.recvBufSize)

	// Register memory
	if err := ctx.RegisterMemory(mgr.sendBuffer); err != nil {
		ctx.Close()
		return nil, fmt.Errorf("failed to register send buffer: %w", err)
	}

	// Get local connection info
	mgr.localInfo, err = ctx.GetConnInfo()
	if err != nil {
		ctx.Close()
		return nil, fmt.Errorf("failed to get connection info: %w", err)
	}

	logger.Info("RDMA manager created",
		zap.String("device", config.DeviceName),
		zap.Int("port", config.Port),
		zap.Uint32("qp_num", mgr.localInfo.QPNum))

	return mgr, nil
}

// GetLocalConnInfo returns local connection information for exchange
func (m *RDMAManager) GetLocalConnInfo() ConnInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.localInfo
}

// Connect establishes RDMA connection with remote peer
func (m *RDMAManager) Connect(remoteInfo ConnInfo) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.connected.Load() {
		return fmt.Errorf("already connected")
	}

	m.logger.Info("Connecting to remote RDMA peer",
		zap.Uint16("remote_lid", remoteInfo.LID),
		zap.Uint32("remote_qp", remoteInfo.QPNum))

	if err := m.ctx.Connect(remoteInfo); err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	m.remoteInfo = remoteInfo
	m.connected.Store(true)

	// Start completion polling goroutine
	m.wg.Add(2)
	go m.pollSendCompletions()
	go m.pollRecvCompletions()

	m.logger.Info("RDMA connection established")

	return nil
}

// Send sends data via RDMA (two-sided operation)
func (m *RDMAManager) Send(data []byte) error {
	if !m.connected.Load() {
		return fmt.Errorf("not connected")
	}

	if len(data) > m.sendBufSize {
		return fmt.Errorf("data size %d exceeds send buffer size %d", len(data), m.sendBufSize)
	}

	startTime := time.Now()

	// Copy data to send buffer
	m.mu.Lock()
	copy(m.sendBuffer, data)
	m.mu.Unlock()

	// Generate work request ID
	wrID := m.nextWRID.Add(1)

	// Create pending operation
	op := &pendingOp{
		wrID:      wrID,
		buffer:    data,
		startTime: startTime,
		done:      make(chan error, 1),
	}
	m.pendingSends.Store(wrID, op)

	// Post send
	if err := m.ctx.PostSend(m.sendBuffer[:len(data)], wrID); err != nil {
		m.pendingSends.Delete(wrID)
		m.stats.SendErrors.Add(1)
		return fmt.Errorf("failed to post send: %w", err)
	}

	m.stats.SendOperations.Add(1)

	// Wait for completion
	select {
	case err := <-op.done:
		if err != nil {
			return err
		}

		// Update statistics
		latency := time.Since(startTime).Nanoseconds()
		m.updateSendLatency(uint64(latency))
		m.stats.BytesSent.Add(uint64(len(data)))

		return nil

	case <-m.shutdownCtx.Done():
		return fmt.Errorf("shutdown in progress")
	}
}

// Receive receives data via RDMA (two-sided operation)
func (m *RDMAManager) Receive(buf []byte) (int, error) {
	if !m.connected.Load() {
		return 0, fmt.Errorf("not connected")
	}

	if len(buf) > m.recvBufSize {
		return 0, fmt.Errorf("buffer size %d exceeds recv buffer size %d", len(buf), m.recvBufSize)
	}

	startTime := time.Now()

	// Generate work request ID
	wrID := m.nextWRID.Add(1)

	// Create pending operation
	op := &pendingOp{
		wrID:      wrID,
		buffer:    buf,
		startTime: startTime,
		done:      make(chan error, 1),
	}
	m.pendingRecvs.Store(wrID, op)

	// Post receive
	if err := m.ctx.PostRecv(m.recvBuffer[:len(buf)], wrID); err != nil {
		m.pendingRecvs.Delete(wrID)
		m.stats.RecvErrors.Add(1)
		return 0, fmt.Errorf("failed to post recv: %w", err)
	}

	m.stats.RecvOperations.Add(1)

	// Wait for completion
	select {
	case err := <-op.done:
		if err != nil {
			return 0, err
		}

		// Copy received data
		m.mu.RLock()
		n := copy(buf, m.recvBuffer[:len(buf)])
		m.mu.RUnlock()

		// Update statistics
		latency := time.Since(startTime).Nanoseconds()
		m.updateRecvLatency(uint64(latency))
		m.stats.BytesReceived.Add(uint64(n))

		return n, nil

	case <-m.shutdownCtx.Done():
		return 0, fmt.Errorf("shutdown in progress")
	}
}

// Write performs one-sided RDMA write
func (m *RDMAManager) Write(data []byte, remoteAddr uint64, rkey uint32) error {
	if !m.connected.Load() {
		return fmt.Errorf("not connected")
	}

	if len(data) > m.sendBufSize {
		return fmt.Errorf("data size %d exceeds send buffer size %d", len(data), m.sendBufSize)
	}

	// Copy data to send buffer
	m.mu.Lock()
	copy(m.sendBuffer, data)
	m.mu.Unlock()

	// Generate work request ID
	wrID := m.nextWRID.Add(1)

	// Post RDMA write
	if err := m.ctx.PostWrite(m.sendBuffer[:len(data)], remoteAddr, rkey, wrID); err != nil {
		m.stats.SendErrors.Add(1)
		return fmt.Errorf("failed to post write: %w", err)
	}

	m.stats.WriteOperations.Add(1)
	m.stats.BytesSent.Add(uint64(len(data)))

	return nil
}

// Read performs one-sided RDMA read
func (m *RDMAManager) Read(buf []byte, remoteAddr uint64, rkey uint32) error {
	if !m.connected.Load() {
		return fmt.Errorf("not connected")
	}

	if len(buf) > m.recvBufSize {
		return fmt.Errorf("buffer size %d exceeds recv buffer size %d", len(buf), m.recvBufSize)
	}

	// Generate work request ID
	wrID := m.nextWRID.Add(1)

	// Post RDMA read
	if err := m.ctx.PostRead(m.recvBuffer[:len(buf)], remoteAddr, rkey, wrID); err != nil {
		m.stats.RecvErrors.Add(1)
		return fmt.Errorf("failed to post read: %w", err)
	}

	m.stats.ReadOperations.Add(1)
	m.stats.BytesReceived.Add(uint64(len(buf)))

	// Copy data from receive buffer
	m.mu.RLock()
	copy(buf, m.recvBuffer[:len(buf)])
	m.mu.RUnlock()

	return nil
}

// pollSendCompletions polls for send completions
func (m *RDMAManager) pollSendCompletions() {
	defer m.wg.Done()

	for {
		select {
		case <-m.shutdownCtx.Done():
			return
		default:
		}

		completed, wrID, length, err := m.ctx.PollCompletion(true)
		if err != nil {
			m.logger.Error("Send poll error", zap.Error(err))
			m.stats.SendErrors.Add(1)
			continue
		}

		if !completed {
			time.Sleep(10 * time.Microsecond) // Small delay to avoid spinning
			continue
		}

		// Handle completion
		if op, ok := m.pendingSends.LoadAndDelete(wrID); ok {
			pendingOp := op.(*pendingOp)
			m.stats.SendCompletions.Add(1)

			if length != len(pendingOp.buffer) {
				pendingOp.done <- fmt.Errorf("length mismatch: expected %d, got %d", len(pendingOp.buffer), length)
			} else {
				pendingOp.done <- nil
			}
		}
	}
}

// pollRecvCompletions polls for receive completions
func (m *RDMAManager) pollRecvCompletions() {
	defer m.wg.Done()

	for {
		select {
		case <-m.shutdownCtx.Done():
			return
		default:
		}

		completed, wrID, length, err := m.ctx.PollCompletion(false)
		if err != nil {
			m.logger.Error("Recv poll error", zap.Error(err))
			m.stats.RecvErrors.Add(1)
			continue
		}

		if !completed {
			time.Sleep(10 * time.Microsecond) // Small delay to avoid spinning
			continue
		}

		// Handle completion
		if op, ok := m.pendingRecvs.LoadAndDelete(wrID); ok {
			pendingOp := op.(*pendingOp)
			m.stats.RecvCompletions.Add(1)

			if length > len(pendingOp.buffer) {
				pendingOp.done <- fmt.Errorf("received too much data: %d > %d", length, len(pendingOp.buffer))
			} else {
				pendingOp.done <- nil
			}
		}
	}
}

// updateSendLatency updates send latency statistics
func (m *RDMAManager) updateSendLatency(latencyNs uint64) {
	// Update average (simple moving average)
	current := m.stats.AvgSendLatencyNs.Load()
	if current == 0 {
		m.stats.AvgSendLatencyNs.Store(latencyNs)
	} else {
		newAvg := (current + latencyNs) / 2
		m.stats.AvgSendLatencyNs.Store(newAvg)
	}

	// Update min
	for {
		current := m.stats.MinSendLatencyNs.Load()
		if current == 0 || latencyNs < current {
			if m.stats.MinSendLatencyNs.CompareAndSwap(current, latencyNs) {
				break
			}
		} else {
			break
		}
	}

	// Update max
	for {
		current := m.stats.MaxSendLatencyNs.Load()
		if latencyNs > current {
			if m.stats.MaxSendLatencyNs.CompareAndSwap(current, latencyNs) {
				break
			}
		} else {
			break
		}
	}
}

// updateRecvLatency updates receive latency statistics
func (m *RDMAManager) updateRecvLatency(latencyNs uint64) {
	// Update average (simple moving average)
	current := m.stats.AvgRecvLatencyNs.Load()
	if current == 0 {
		m.stats.AvgRecvLatencyNs.Store(latencyNs)
	} else {
		newAvg := (current + latencyNs) / 2
		m.stats.AvgRecvLatencyNs.Store(newAvg)
	}
}

// GetStats returns current RDMA statistics
func (m *RDMAManager) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"send_operations":      m.stats.SendOperations.Load(),
		"recv_operations":      m.stats.RecvOperations.Load(),
		"write_operations":     m.stats.WriteOperations.Load(),
		"read_operations":      m.stats.ReadOperations.Load(),
		"send_completions":     m.stats.SendCompletions.Load(),
		"recv_completions":     m.stats.RecvCompletions.Load(),
		"send_errors":          m.stats.SendErrors.Load(),
		"recv_errors":          m.stats.RecvErrors.Load(),
		"bytes_sent":           m.stats.BytesSent.Load(),
		"bytes_received":       m.stats.BytesReceived.Load(),
		"avg_send_latency_ns":  m.stats.AvgSendLatencyNs.Load(),
		"avg_recv_latency_ns":  m.stats.AvgRecvLatencyNs.Load(),
		"min_send_latency_ns":  m.stats.MinSendLatencyNs.Load(),
		"max_send_latency_ns":  m.stats.MaxSendLatencyNs.Load(),
		"avg_send_latency_us":  float64(m.stats.AvgSendLatencyNs.Load()) / 1000.0,
		"avg_recv_latency_us":  float64(m.stats.AvgRecvLatencyNs.Load()) / 1000.0,
	}
}

// IsConnected returns whether RDMA is connected
func (m *RDMAManager) IsConnected() bool {
	return m.connected.Load() && m.ctx.IsConnected()
}

// Close closes the RDMA manager
func (m *RDMAManager) Close() error {
	m.logger.Info("Closing RDMA manager")

	m.shutdownCancel()
	m.wg.Wait()

	if m.ctx != nil {
		m.ctx.UnregisterMemory()
		m.ctx.Close()
	}

	m.connected.Store(false)

	m.logger.Info("RDMA manager closed")
	return nil
}

// ExchangeConnInfoJSON exchanges connection info with peer via JSON
func ExchangeConnInfoJSON(localInfo ConnInfo) (string, error) {
	data, err := json.Marshal(localInfo)
	if err != nil {
		return "", fmt.Errorf("failed to marshal conn info: %w", err)
	}
	return string(data), nil
}

// ParseConnInfoJSON parses connection info from JSON
func ParseConnInfoJSON(jsonStr string) (ConnInfo, error) {
	var info ConnInfo
	if err := json.Unmarshal([]byte(jsonStr), &info); err != nil {
		return ConnInfo{}, fmt.Errorf("failed to unmarshal conn info: %w", err)
	}
	return info, nil
}
