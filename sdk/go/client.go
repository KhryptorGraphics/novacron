// Package dwcp provides a comprehensive Go SDK for the Distributed Worker Control Protocol (DWCP) v3
package dwcp

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	// Protocol constants
	ProtocolVersion = 3
	DefaultPort     = 9000

	// Message types
	MsgTypeAuth           = 0x01
	MsgTypeVM             = 0x02
	MsgTypeStream         = 0x03
	MsgTypeMigration      = 0x04
	MsgTypeHealth         = 0x05
	MsgTypeMetrics        = 0x06
	MsgTypeConfig         = 0x07
	MsgTypeSnapshot       = 0x08

	// VM operations
	VMOpCreate   = 0x10
	VMOpStart    = 0x11
	VMOpStop     = 0x12
	VMOpDestroy  = 0x13
	VMOpStatus   = 0x14
	VMOpMigrate  = 0x15
	VMOpSnapshot = 0x16
	VMOpRestore  = 0x17

	// Error codes
	ErrCodeAuth          = 0x1000
	ErrCodeInvalidMsg    = 0x1001
	ErrCodeVMNotFound    = 0x1002
	ErrCodeResourceLimit = 0x1003
	ErrCodeMigration     = 0x1004
)

var (
	ErrNotConnected     = errors.New("not connected to DWCP server")
	ErrAuthFailed       = errors.New("authentication failed")
	ErrInvalidResponse  = errors.New("invalid response from server")
	ErrTimeout          = errors.New("operation timeout")
	ErrVMNotFound       = errors.New("VM not found")
	ErrInvalidOperation = errors.New("invalid operation")
)

// Client represents a DWCP client connection
type Client struct {
	conn        net.Conn
	tlsConfig   *tls.Config
	address     string
	apiKey      string

	mu          sync.RWMutex
	connected   bool
	authenticated bool

	// Streaming support
	streams     map[string]*Stream
	streamMu    sync.RWMutex

	// Configuration
	config      ClientConfig

	// Metrics
	metrics     *ClientMetrics

	// Context management
	ctx         context.Context
	cancel      context.CancelFunc

	// Message handling
	msgHandlers map[uint8]MessageHandler
	handlerMu   sync.RWMutex
}

// ClientConfig holds client configuration
type ClientConfig struct {
	Address         string
	Port            int
	APIKey          string
	TLSEnabled      bool
	TLSConfig       *tls.Config
	ConnectTimeout  time.Duration
	RequestTimeout  time.Duration
	RetryAttempts   int
	RetryBackoff    time.Duration
	KeepAlive       bool
	KeepAlivePeriod time.Duration
	MaxStreams      int
	BufferSize      int
}

// DefaultConfig returns a default client configuration
func DefaultConfig() ClientConfig {
	return ClientConfig{
		Port:            DefaultPort,
		TLSEnabled:      true,
		ConnectTimeout:  30 * time.Second,
		RequestTimeout:  60 * time.Second,
		RetryAttempts:   3,
		RetryBackoff:    time.Second,
		KeepAlive:       true,
		KeepAlivePeriod: 30 * time.Second,
		MaxStreams:      100,
		BufferSize:      65536,
	}
}

// NewClient creates a new DWCP client
func NewClient(config ClientConfig) (*Client, error) {
	if config.Address == "" {
		return nil, errors.New("address is required")
	}

	if config.Port == 0 {
		config.Port = DefaultPort
	}

	ctx, cancel := context.WithCancel(context.Background())

	client := &Client{
		address:     fmt.Sprintf("%s:%d", config.Address, config.Port),
		apiKey:      config.APIKey,
		tlsConfig:   config.TLSConfig,
		config:      config,
		streams:     make(map[string]*Stream),
		msgHandlers: make(map[uint8]MessageHandler),
		metrics:     newClientMetrics(),
		ctx:         ctx,
		cancel:      cancel,
	}

	return client, nil
}

// Connect establishes a connection to the DWCP server
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected {
		return nil
	}

	// Apply connection timeout
	connectCtx, cancel := context.WithTimeout(ctx, c.config.ConnectTimeout)
	defer cancel()

	var conn net.Conn
	var err error

	// Retry logic
	for attempt := 0; attempt <= c.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			select {
			case <-connectCtx.Done():
				return ErrTimeout
			case <-time.After(c.config.RetryBackoff * time.Duration(attempt)):
			}
		}

		dialer := &net.Dialer{
			Timeout:   c.config.ConnectTimeout,
			KeepAlive: c.config.KeepAlivePeriod,
		}

		if c.config.TLSEnabled {
			tlsConfig := c.tlsConfig
			if tlsConfig == nil {
				tlsConfig = &tls.Config{
					MinVersion: tls.VersionTLS13,
				}
			}
			conn, err = tls.DialWithDialer(dialer, "tcp", c.address, tlsConfig)
		} else {
			conn, err = dialer.DialContext(connectCtx, "tcp", c.address)
		}

		if err == nil {
			break
		}
	}

	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	c.conn = conn
	c.connected = true

	// Start message reader
	go c.readLoop()

	// Authenticate if API key provided
	if c.apiKey != "" {
		if err := c.authenticate(ctx); err != nil {
			c.conn.Close()
			c.connected = false
			return err
		}
	}

	c.metrics.ConnectionsTotal++
	c.metrics.LastConnected = time.Now()

	return nil
}

// authenticate performs API key authentication
func (c *Client) authenticate(ctx context.Context) error {
	authReq := &AuthRequest{
		APIKey:  c.apiKey,
		Version: ProtocolVersion,
	}

	resp, err := c.sendRequest(ctx, MsgTypeAuth, authReq)
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	var authResp AuthResponse
	if err := json.Unmarshal(resp, &authResp); err != nil {
		return ErrInvalidResponse
	}

	if !authResp.Success {
		return ErrAuthFailed
	}

	c.authenticated = true
	return nil
}

// Disconnect closes the connection to the DWCP server
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected {
		return nil
	}

	// Close all streams
	c.streamMu.Lock()
	for _, stream := range c.streams {
		stream.Close()
	}
	c.streams = make(map[string]*Stream)
	c.streamMu.Unlock()

	// Close connection
	if c.conn != nil {
		c.conn.Close()
	}

	c.connected = false
	c.authenticated = false
	c.cancel()

	return nil
}

// sendRequest sends a request and waits for response
func (c *Client) sendRequest(ctx context.Context, msgType uint8, payload interface{}) ([]byte, error) {
	c.mu.RLock()
	if !c.connected {
		c.mu.RUnlock()
		return nil, ErrNotConnected
	}
	c.mu.RUnlock()

	// Serialize payload
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	// Build message
	msg := &Message{
		Version:   ProtocolVersion,
		Type:      msgType,
		Timestamp: time.Now().Unix(),
		RequestID: uuid.New().String(),
		Payload:   payloadBytes,
	}

	// Serialize message
	msgBytes, err := msg.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal message: %w", err)
	}

	// Send with timeout
	sendCtx, cancel := context.WithTimeout(ctx, c.config.RequestTimeout)
	defer cancel()

	errCh := make(chan error, 1)
	respCh := make(chan []byte, 1)

	// Register response handler
	c.registerResponseHandler(msg.RequestID, respCh)
	defer c.unregisterResponseHandler(msg.RequestID)

	// Send message
	go func() {
		c.mu.RLock()
		defer c.mu.RUnlock()

		if _, err := c.conn.Write(msgBytes); err != nil {
			errCh <- err
			return
		}

		c.metrics.MessagesSent++
		c.metrics.BytesSent += uint64(len(msgBytes))
	}()

	// Wait for response or timeout
	select {
	case err := <-errCh:
		return nil, err
	case resp := <-respCh:
		return resp, nil
	case <-sendCtx.Done():
		return nil, ErrTimeout
	}
}

// readLoop continuously reads messages from the connection
func (c *Client) readLoop() {
	buffer := make([]byte, c.config.BufferSize)

	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		c.mu.RLock()
		conn := c.conn
		c.mu.RUnlock()

		if conn == nil {
			return
		}

		// Read message length
		var msgLen uint32
		if err := binary.Read(conn, binary.BigEndian, &msgLen); err != nil {
			if err != io.EOF {
				c.metrics.ErrorsTotal++
			}
			return
		}

		// Read message data
		msgData := make([]byte, msgLen)
		if _, err := io.ReadFull(conn, msgData); err != nil {
			c.metrics.ErrorsTotal++
			return
		}

		c.metrics.MessagesReceived++
		c.metrics.BytesReceived += uint64(msgLen)

		// Parse message
		msg := &Message{}
		if err := msg.Unmarshal(msgData); err != nil {
			c.metrics.ErrorsTotal++
			continue
		}

		// Handle message
		c.handleMessage(msg)
	}
}

// handleMessage processes incoming messages
func (c *Client) handleMessage(msg *Message) {
	c.handlerMu.RLock()
	handler, exists := c.msgHandlers[msg.Type]
	c.handlerMu.RUnlock()

	if exists {
		go handler(msg)
	}
}

// Message represents a DWCP protocol message
type Message struct {
	Version   uint8
	Type      uint8
	Timestamp int64
	RequestID string
	Payload   []byte
}

// Marshal serializes the message
func (m *Message) Marshal() ([]byte, error) {
	buf := make([]byte, 4) // Reserve space for length

	buf = append(buf, m.Version)
	buf = append(buf, m.Type)

	ts := make([]byte, 8)
	binary.BigEndian.PutUint64(ts, uint64(m.Timestamp))
	buf = append(buf, ts...)

	ridLen := uint16(len(m.RequestID))
	rid := make([]byte, 2)
	binary.BigEndian.PutUint16(rid, ridLen)
	buf = append(buf, rid...)
	buf = append(buf, []byte(m.RequestID)...)

	payloadLen := uint32(len(m.Payload))
	pl := make([]byte, 4)
	binary.BigEndian.PutUint32(pl, payloadLen)
	buf = append(buf, pl...)
	buf = append(buf, m.Payload...)

	// Write total length
	msgLen := uint32(len(buf) - 4)
	binary.BigEndian.PutUint32(buf[:4], msgLen)

	return buf, nil
}

// Unmarshal deserializes the message
func (m *Message) Unmarshal(data []byte) error {
	if len(data) < 15 {
		return errors.New("message too short")
	}

	m.Version = data[0]
	m.Type = data[1]
	m.Timestamp = int64(binary.BigEndian.Uint64(data[2:10]))

	ridLen := binary.BigEndian.Uint16(data[10:12])
	if len(data) < int(12+ridLen+4) {
		return errors.New("invalid message format")
	}

	m.RequestID = string(data[12 : 12+ridLen])

	payloadLen := binary.BigEndian.Uint32(data[12+ridLen : 16+ridLen])
	if len(data) < int(16+ridLen+payloadLen) {
		return errors.New("invalid payload length")
	}

	m.Payload = data[16+ridLen : 16+ridLen+payloadLen]

	return nil
}

// MessageHandler handles incoming messages
type MessageHandler func(*Message)

// registerResponseHandler registers a handler for a specific request ID
func (c *Client) registerResponseHandler(requestID string, ch chan []byte) {
	c.handlerMu.Lock()
	defer c.handlerMu.Unlock()

	c.msgHandlers[MsgTypeVM] = func(msg *Message) {
		if msg.RequestID == requestID {
			ch <- msg.Payload
		}
	}
}

// unregisterResponseHandler removes a response handler
func (c *Client) unregisterResponseHandler(requestID string) {
	c.handlerMu.Lock()
	defer c.handlerMu.Unlock()

	delete(c.msgHandlers, MsgTypeVM)
}

// AuthRequest represents an authentication request
type AuthRequest struct {
	APIKey  string `json:"api_key"`
	Version uint8  `json:"version"`
}

// AuthResponse represents an authentication response
type AuthResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Token   string `json:"token,omitempty"`
}

// ClientMetrics holds client metrics
type ClientMetrics struct {
	ConnectionsTotal  uint64
	MessagesSent      uint64
	MessagesReceived  uint64
	BytesSent         uint64
	BytesReceived     uint64
	ErrorsTotal       uint64
	LastConnected     time.Time

	mu sync.RWMutex
}

// newClientMetrics creates a new metrics instance
func newClientMetrics() *ClientMetrics {
	return &ClientMetrics{}
}

// GetMetrics returns a copy of current metrics
func (c *Client) GetMetrics() ClientMetrics {
	c.metrics.mu.RLock()
	defer c.metrics.mu.RUnlock()

	return *c.metrics
}

// Stream represents a bidirectional streaming connection
type Stream struct {
	id       string
	client   *Client
	ctx      context.Context
	cancel   context.CancelFunc
	dataCh   chan []byte
	errorCh  chan error
	closed   bool
	mu       sync.RWMutex
}

// NewStream creates a new stream
func (c *Client) NewStream(ctx context.Context) (*Stream, error) {
	c.streamMu.Lock()
	defer c.streamMu.Unlock()

	if len(c.streams) >= c.config.MaxStreams {
		return nil, errors.New("max streams reached")
	}

	streamCtx, cancel := context.WithCancel(ctx)

	stream := &Stream{
		id:      uuid.New().String(),
		client:  c,
		ctx:     streamCtx,
		cancel:  cancel,
		dataCh:  make(chan []byte, 100),
		errorCh: make(chan error, 1),
	}

	c.streams[stream.id] = stream

	return stream, nil
}

// Send sends data on the stream
func (s *Stream) Send(data []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return errors.New("stream closed")
	}

	select {
	case s.dataCh <- data:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// Receive receives data from the stream
func (s *Stream) Receive() ([]byte, error) {
	select {
	case data := <-s.dataCh:
		return data, nil
	case err := <-s.errorCh:
		return nil, err
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

// Close closes the stream
func (s *Stream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true
	s.cancel()
	close(s.dataCh)
	close(s.errorCh)

	s.client.streamMu.Lock()
	delete(s.client.streams, s.id)
	s.client.streamMu.Unlock()

	return nil
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.connected
}

// IsAuthenticated returns whether the client is authenticated
func (c *Client) IsAuthenticated() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.authenticated
}
