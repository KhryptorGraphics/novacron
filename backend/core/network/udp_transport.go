package network

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"time"
)

// Constants for UDP transport
const (
	// MaxUDPPacketSize is the maximum size of a UDP packet
	MaxUDPPacketSize = 65507 // Max UDP packet size (65535 - 28 bytes for headers)
	
	// DefaultReadBufferSize is the default size of the read buffer
	DefaultReadBufferSize = 8 * 1024 * 1024 // 8 MB
	
	// DefaultWriteBufferSize is the default size of the write buffer
	DefaultWriteBufferSize = 8 * 1024 * 1024 // 8 MB
	
	// DefaultReceiveQueueSize is the default size of the receive queue
	DefaultReceiveQueueSize = 1000
	
	// DefaultSendQueueSize is the default size of the send queue
	DefaultSendQueueSize = 1000
	
	// AckTimeout is the timeout for acknowledgments
	AckTimeout = 500 * time.Millisecond
	
	// MaxRetries is the maximum number of retries for reliable messages
	MaxRetries = 5
	
	// KeepAliveInterval is the interval for keep-alive messages
	KeepAliveInterval = 15 * time.Second
	
	// ConnectionTimeout is the timeout for inactive connections
	ConnectionTimeout = 30 * time.Second
)

// UDPTransportConfig contains configuration for the UDP transport
type UDPTransportConfig struct {
	ListenAddr         string
	ReadBufferSize     int
	WriteBufferSize    int
	ReceiveQueueSize   int
	SendQueueSize      int
	AckTimeout         time.Duration
	MaxRetries         int
	KeepAliveInterval  time.Duration
	ConnectionTimeout  time.Duration
	EnableFlowControl  bool
	BatchSendThreshold int    // Minimum number of messages to trigger batch sending
	BatchSendInterval  time.Duration // Maximum time to wait before sending a batch
}

// DefaultUDPTransportConfig returns a default UDP transport configuration
func DefaultUDPTransportConfig() UDPTransportConfig {
	return UDPTransportConfig{
		ListenAddr:         ":7700",
		ReadBufferSize:     DefaultReadBufferSize,
		WriteBufferSize:    DefaultWriteBufferSize,
		ReceiveQueueSize:   DefaultReceiveQueueSize,
		SendQueueSize:      DefaultSendQueueSize,
		AckTimeout:         AckTimeout,
		MaxRetries:         MaxRetries,
		KeepAliveInterval:  KeepAliveInterval,
		ConnectionTimeout:  ConnectionTimeout,
		EnableFlowControl:  true,
		BatchSendThreshold: 10,
		BatchSendInterval:  5 * time.Millisecond,
	}
}

// UDPTransport implements a UDP-based transport layer
type UDPTransport struct {
	config     UDPTransportConfig
	conn       *net.UDPConn
	peers      map[string]*UDPPeer
	peersMutex sync.RWMutex
	receiveQueue chan *Message
	closed     bool
	closedMutex sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
}

// UDPPeer represents a remote peer connection over UDP
type UDPPeer struct {
	addr          *net.UDPAddr
	lastReceived  time.Time
	lastSent      time.Time
	sendQueue     chan *Message
	pendingAcks   map[uint32]*pendingMessage
	acksMutex     sync.Mutex
	nextSequenceID uint32
	seqMutex      sync.Mutex
	transport     *UDPTransport
	closed        bool
}

// pendingMessage represents a message waiting for acknowledgment
type pendingMessage struct {
	message *Message
	sentAt  time.Time
	retries int
}

// NewUDPTransport creates a new UDP transport
func NewUDPTransport(config UDPTransportConfig) (*UDPTransport, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	transport := &UDPTransport{
		config:     config,
		peers:      make(map[string]*UDPPeer),
		receiveQueue: make(chan *Message, config.ReceiveQueueSize),
		ctx:        ctx,
		cancel:     cancel,
	}
	
	return transport, nil
}

// Start starts the UDP transport
func (t *UDPTransport) Start() error {
	// Resolve the UDP address
	addr, err := net.ResolveUDPAddr("udp", t.config.ListenAddr)
	if err != nil {
		return fmt.Errorf("failed to resolve UDP address: %w", err)
	}
	
	// Create the UDP connection
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen on UDP: %w", err)
	}
	
	// Set buffer sizes
	if err := conn.SetReadBuffer(t.config.ReadBufferSize); err != nil {
		return fmt.Errorf("failed to set read buffer size: %w", err)
	}
	
	if err := conn.SetWriteBuffer(t.config.WriteBufferSize); err != nil {
		return fmt.Errorf("failed to set write buffer size: %w", err)
	}
	
	t.conn = conn
	
	// Start the receive loop
	go t.receiveLoop()
	
	// Start the maintenance loop
	go t.maintenanceLoop()
	
	log.Printf("UDP transport started on %s", t.config.ListenAddr)
	
	return nil
}

// Stop stops the UDP transport
func (t *UDPTransport) Stop() error {
	t.closedMutex.Lock()
	if t.closed {
		t.closedMutex.Unlock()
		return nil
	}
	t.closed = true
	t.closedMutex.Unlock()
	
	t.cancel()
	
	// Close all peer connections
	t.peersMutex.Lock()
	for _, peer := range t.peers {
		peer.closed = true
		close(peer.sendQueue)
	}
	t.peers = make(map[string]*UDPPeer)
	t.peersMutex.Unlock()
	
	// Close the UDP connection
	if t.conn != nil {
		err := t.conn.Close()
		t.conn = nil
		return err
	}
	
	return nil
}

// Connect connects to a remote peer
func (t *UDPTransport) Connect(addr string) (*UDPPeer, error) {
	t.closedMutex.RLock()
	if t.closed {
		t.closedMutex.RUnlock()
		return nil, errors.New("transport is closed")
	}
	t.closedMutex.RUnlock()
	
	// Resolve the UDP address
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve UDP address: %w", err)
	}
	
	// Check if we already have a connection to this peer
	t.peersMutex.RLock()
	peer, exists := t.peers[udpAddr.String()]
	t.peersMutex.RUnlock()
	
	if exists {
		return peer, nil
	}
	
	// Create a new peer
	peer = &UDPPeer{
		addr:          udpAddr,
		lastReceived:  time.Now(),
		lastSent:      time.Now(),
		sendQueue:     make(chan *Message, t.config.SendQueueSize),
		pendingAcks:   make(map[uint32]*pendingMessage),
		transport:     t,
	}
	
	// Register the peer
	t.peersMutex.Lock()
	t.peers[udpAddr.String()] = peer
	t.peersMutex.Unlock()
	
	// Start the send loop for this peer
	go t.sendLoop(peer)
	
	// Start the retry loop for this peer
	go t.retryLoop(peer)
	
	// Send a handshake message
	handshakeMsg := NewMessage(TypeHandshake, []byte("NOVACRON"), FlagReliable, peer.nextSequenceID())
	peer.SendMessage(handshakeMsg)
	
	return peer, nil
}

// receiveLoop reads messages from the UDP connection
func (t *UDPTransport) receiveLoop() {
	buffer := make([]byte, MaxUDPPacketSize)
	
	for {
		select {
		case <-t.ctx.Done():
			return
		default:
			// Read from the UDP connection
			n, addr, err := t.conn.ReadFromUDP(buffer)
			if err != nil {
				log.Printf("Error reading from UDP: %v", err)
				continue
			}
			
			// Process the message
			t.handlePacket(buffer[:n], addr)
		}
	}
}

// handlePacket processes a received UDP packet
func (t *UDPTransport) handlePacket(data []byte, addr *net.UDPAddr) {
	// Deserialize the message
	msg, err := Deserialize(data)
	if err != nil {
		log.Printf("Error deserializing message from %s: %v", addr.String(), err)
		return
	}
	
	// Get or create peer for this address
	peer := t.getOrCreatePeer(addr)
	
	// Update last received time
	peer.lastReceived = time.Now()
	
	// Handle acknowledgments
	if msg.Header.IsFlag(FlagAck) {
		peer.acksMutex.Lock()
		pendingMsg, exists := peer.pendingAcks[msg.Header.SequenceID]
		if exists {
			delete(peer.pendingAcks, msg.Header.SequenceID)
		}
		peer.acksMutex.Unlock()
		
		// Don't process further for pure ACK messages
		if msg.Header.Type == TypePong && len(msg.Payload) == 0 {
			return
		}
	}
	
	// Send acknowledgment for reliable messages
	if msg.Header.IsFlag(FlagReliable) {
		ackMsg := NewMessage(TypePong, nil, FlagAck, msg.Header.SequenceID)
		t.sendToAddr(ackMsg, addr)
	}
	
	// Process message based on type
	switch msg.Header.Type {
	case TypeHandshake:
		// Respond to handshake
		responseMsg := NewMessage(TypeHandshakeResponse, []byte("NOVACRON_ACK"), FlagReliable, peer.nextSequenceID())
		peer.SendMessage(responseMsg)
		
	case TypePing:
		// Respond to ping
		responseMsg := NewMessage(TypePong, nil, 0, 0)
		peer.SendMessage(responseMsg)
	}
	
	// Deliver message to receiver
	select {
	case t.receiveQueue <- msg:
		// Message delivered to receive queue
	default:
		log.Printf("Receive queue full, dropping message from %s", addr.String())
	}
}

// getOrCreatePeer gets an existing peer or creates a new one
func (t *UDPTransport) getOrCreatePeer(addr *net.UDPAddr) *UDPPeer {
	addrStr := addr.String()
	
	t.peersMutex.RLock()
	peer, exists := t.peers[addrStr]
	t.peersMutex.RUnlock()
	
	if exists {
		return peer
	}
	
	// Create a new peer
	peer = &UDPPeer{
		addr:          addr,
		lastReceived:  time.Now(),
		lastSent:      time.Now(),
		sendQueue:     make(chan *Message, t.config.SendQueueSize),
		pendingAcks:   make(map[uint32]*pendingMessage),
		transport:     t,
	}
	
	// Register the peer
	t.peersMutex.Lock()
	t.peers[addrStr] = peer
	t.peersMutex.Unlock()
	
	// Start the send loop for this peer
	go t.sendLoop(peer)
	
	// Start the retry loop for this peer
	go t.retryLoop(peer)
	
	return peer
}

// sendLoop sends messages from the send queue
func (t *UDPTransport) sendLoop(peer *UDPPeer) {
	batchTimer := time.NewTimer(t.config.BatchSendInterval)
	defer batchTimer.Stop()
	
	var batch []*Message
	
	for {
		if peer.closed {
			return
		}
		
		// Start with an empty batch
		batch = batch[:0]
		
		// First message in the batch (blocking)
		var msg *Message
		select {
		case <-t.ctx.Done():
			return
		case msg = <-peer.sendQueue:
			if msg == nil {
				return // Channel closed
			}
			batch = append(batch, msg)
		}
		
		// Collect more messages if available
	collectLoop:
		for len(batch) < t.config.BatchSendThreshold {
			select {
			case <-t.ctx.Done():
				return
			case msg = <-peer.sendQueue:
				if msg == nil {
					break collectLoop // Channel closed
				}
				batch = append(batch, msg)
			case <-batchTimer.C:
				break collectLoop
			default:
				// No more messages, send what we have
				break collectLoop
			}
		}
		
		// Reset the batch timer
		if !batchTimer.Stop() {
			select {
			case <-batchTimer.C:
			default:
			}
		}
		batchTimer.Reset(t.config.BatchSendInterval)
		
		// Send all messages in the batch
		for _, msg := range batch {
			t.sendToAddr(msg, peer.addr)
			
			// Store reliable messages for potential retries
			if msg.Header.IsFlag(FlagReliable) {
				peer.acksMutex.Lock()
				peer.pendingAcks[msg.Header.SequenceID] = &pendingMessage{
					message: msg,
					sentAt:  time.Now(),
				}
				peer.acksMutex.Unlock()
			}
		}
		
		// Update last sent time
		peer.lastSent = time.Now()
	}
}

// retryLoop retries reliable messages that haven't been acknowledged
func (t *UDPTransport) retryLoop(peer *UDPPeer) {
	ticker := time.NewTicker(t.config.AckTimeout / 2)
	defer ticker.Stop()
	
	for {
		select {
		case <-t.ctx.Done():
			return
		case <-ticker.C:
			if peer.closed {
				return
			}
			
			now := time.Now()
			toRetry := make([]*pendingMessage, 0)
			toRemove := make([]uint32, 0)
			
			// Collect messages that need to be retried or removed
			peer.acksMutex.Lock()
			for seqID, pending := range peer.pendingAcks {
				if now.Sub(pending.sentAt) > t.config.AckTimeout {
					// Check if we've reached max retries
					if pending.retries >= t.config.MaxRetries {
						toRemove = append(toRemove, seqID)
					} else {
						// Increment retry count and add to retry list
						pending.retries++
						toRetry = append(toRetry, pending)
					}
				}
			}
			
			// Remove messages that have exceeded max retries
			for _, seqID := range toRemove {
				delete(peer.pendingAcks, seqID)
			}
			peer.acksMutex.Unlock()
			
			// Retry sending messages
			for _, pending := range toRetry {
				t.sendToAddr(pending.message, peer.addr)
				pending.sentAt = now
			}
			
			// If we've removed messages due to exceeded retries, the peer might be down
			if len(toRemove) > 0 {
				log.Printf("Peer %s may be down, %d messages exceeded max retries", peer.addr.String(), len(toRemove))
				
				// TODO: Consider marking the peer as suspicious or disconnected
				// For now, we'll keep the connection and let the maintenance loop handle it
			}
		}
	}
}

// maintenanceLoop performs periodic maintenance tasks
func (t *UDPTransport) maintenanceLoop() {
	keepAliveTicker := time.NewTicker(t.config.KeepAliveInterval)
	defer keepAliveTicker.Stop()
	
	connectionCheckTicker := time.NewTicker(t.config.ConnectionTimeout / 2)
	defer connectionCheckTicker.Stop()
	
	for {
		select {
		case <-t.ctx.Done():
			return
		case <-keepAliveTicker.C:
			// Send keep-alive messages
			t.peersMutex.RLock()
			for _, peer := range t.peers {
				if time.Since(peer.lastSent) > t.config.KeepAliveInterval {
					pingMsg := NewMessage(TypePing, nil, 0, 0)
					peer.SendMessage(pingMsg)
				}
			}
			t.peersMutex.RUnlock()
			
		case <-connectionCheckTicker.C:
			// Check for timed-out connections
			now := time.Now()
			var timedOutPeers []string
			
			t.peersMutex.RLock()
			for addrStr, peer := range t.peers {
				if now.Sub(peer.lastReceived) > t.config.ConnectionTimeout {
					timedOutPeers = append(timedOutPeers, addrStr)
				}
			}
			t.peersMutex.RUnlock()
			
			// Remove timed-out peers
			if len(timedOutPeers) > 0 {
				t.peersMutex.Lock()
				for _, addrStr := range timedOutPeers {
					peer, exists := t.peers[addrStr]
					if exists {
						peer.closed = true
						close(peer.sendQueue)
						delete(t.peers, addrStr)
						log.Printf("Removed timed-out peer: %s", addrStr)
					}
				}
				t.peersMutex.Unlock()
			}
		}
	}
}

// sendToAddr sends a message to a specific address
func (t *UDPTransport) sendToAddr(msg *Message, addr *net.UDPAddr) error {
	t.closedMutex.RLock()
	if t.closed {
		t.closedMutex.RUnlock()
		return errors.New("transport is closed")
	}
	t.closedMutex.RUnlock()
	
	data := msg.Serialize()
	
	// Check message size
	if len(data) > MaxUDPPacketSize {
		return fmt.Errorf("message too large: %d bytes (max: %d)", len(data), MaxUDPPacketSize)
	}
	
	// Send the message
	_, err := t.conn.WriteToUDP(data, addr)
	return err
}

// Receive receives a message from the transport
func (t *UDPTransport) Receive(ctx context.Context) (*Message, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-t.ctx.Done():
		return nil, errors.New("transport closed")
	case msg := <-t.receiveQueue:
		return msg, nil
	}
}

// SendMessage sends a message to the peer
func (p *UDPPeer) SendMessage(msg *Message) error {
	if p.closed {
		return errors.New("peer connection is closed")
	}
	
	select {
	case p.sendQueue <- msg:
		return nil
	default:
		return errors.New("send queue is full")
	}
}

// nextSequenceID generates the next sequence ID for the peer
func (p *UDPPeer) nextSequenceID() uint32 {
	p.seqMutex.Lock()
	defer p.seqMutex.Unlock()
	
	id := p.nextSequenceID
	p.nextSequenceID++
	return id
}
