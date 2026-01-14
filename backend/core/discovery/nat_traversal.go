package discovery

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

const (
	STUN_MAGIC_COOKIE = 0x2112A442
	STUN_BINDING_REQUEST = 0x0001
	STUN_BINDING_RESPONSE = 0x0101
	STUN_ERROR_RESPONSE = 0x0111

	ATTR_MAPPED_ADDRESS = 0x0001
	ATTR_XOR_MAPPED_ADDRESS = 0x0020
	ATTR_ERROR_CODE = 0x0009
	ATTR_UNKNOWN_ATTRIBUTES = 0x000A

	NAT_TYPE_UNKNOWN = 0
	NAT_TYPE_OPEN_INTERNET = 1
	NAT_TYPE_FULL_CONE = 2
	NAT_TYPE_RESTRICTED_CONE = 3
	NAT_TYPE_PORT_RESTRICTED_CONE = 4
	NAT_TYPE_SYMMETRIC = 5
)

type STUNMessage struct {
	Type         uint16
	Length       uint16
	MagicCookie  uint32
	TransactionID [12]byte
	Attributes   []STUNAttribute
}

type STUNAttribute struct {
	Type   uint16
	Length uint16
	Value  []byte
}

type STUNServer struct {
	Host string `json:"host"`
	Port int    `json:"port"`
}

type ExternalEndpoint struct {
	IP           net.IP    `json:"ip"`
	Port         int       `json:"port"`
	NATType      int       `json:"nat_type"`
	LastUpdated  time.Time `json:"last_updated"`
	ServerUsed   string    `json:"server_used"`
}

type ConnectionQuality struct {
	RTT              time.Duration `json:"rtt"`
	PacketLoss       float64       `json:"packet_loss"`
	BandwidthEstimate uint64       `json:"bandwidth_estimate"`
	Jitter           time.Duration `json:"jitter"`
	LastMeasured     time.Time     `json:"last_measured"`
}

type PeerConnection struct {
	PeerID           string            `json:"peer_id"`
	LocalEndpoint    *net.UDPAddr      `json:"local_endpoint"`
	RemoteEndpoint   *net.UDPAddr      `json:"remote_endpoint"`
	RemoteTCPAddr    *net.TCPAddr      `json:"remote_tcp_addr,omitempty"` // For direct TCP connections
	ConnectionType   string            `json:"connection_type"`
	Quality          ConnectionQuality `json:"quality"`
	Established      bool              `json:"established"`
	LastActivity     time.Time         `json:"last_activity"`
	conn    net.Conn          // Can be *net.UDPConn or *net.TCPConn
}

type STUNClient struct {
	servers       []STUNServer
	localAddr     *net.UDPAddr
	logger        *zap.Logger
	timeout       time.Duration
	retries       int
	mu            sync.RWMutex
}

func NewSTUNClient(servers []STUNServer, logger *zap.Logger) *STUNClient {
	if len(servers) == 0 {
		servers = []STUNServer{
			{Host: "stun.l.google.com", Port: 19302},
			{Host: "stun1.l.google.com", Port: 19302},
			{Host: "stun2.l.google.com", Port: 19302},
			{Host: "stun.stunprotocol.org", Port: 3478},
		}
	}

	return &STUNClient{
		servers: servers,
		logger:  logger,
		timeout: 5 * time.Second,
		retries: 3,
	}
}

func (sc *STUNClient) DiscoverExternalAddress() (*ExternalEndpoint, error) {
	var lastErr error

	for _, server := range sc.servers {
		endpoint, err := sc.querySTUNServer(server)
		if err != nil {
			sc.logger.Warn("STUN server query failed", 
				zap.String("server", fmt.Sprintf("%s:%d", server.Host, server.Port)),
				zap.Error(err))
			lastErr = err
			continue
		}

		sc.logger.Info("Successfully discovered external address", 
			zap.String("server", fmt.Sprintf("%s:%d", server.Host, server.Port)),
			zap.String("external_ip", endpoint.IP.String()),
			zap.Int("external_port", endpoint.Port))

		return endpoint, nil
	}

	return nil, fmt.Errorf("all STUN servers failed, last error: %w", lastErr)
}

func (sc *STUNClient) querySTUNServer(server STUNServer) (*ExternalEndpoint, error) {
	serverAddr := fmt.Sprintf("%s:%d", server.Host, server.Port)
	
	conn, err := net.DialTimeout("udp", serverAddr, sc.timeout)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to STUN server: %w", err)
	}
	defer conn.Close()

	msg := sc.createBindingRequest()
	data := sc.marshalSTUNMessage(msg)

	_, err = conn.Write(data)
	if err != nil {
		return nil, fmt.Errorf("failed to send STUN request: %w", err)
	}

	conn.SetReadDeadline(time.Now().Add(sc.timeout))
	
	buffer := make([]byte, 1024)
	n, err := conn.Read(buffer)
	if err != nil {
		return nil, fmt.Errorf("failed to read STUN response: %w", err)
	}

	response, err := sc.parseSTUNMessage(buffer[:n])
	if err != nil {
		return nil, fmt.Errorf("failed to parse STUN response: %w", err)
	}

	if response.Type == STUN_ERROR_RESPONSE {
		return nil, fmt.Errorf("STUN server returned error")
	}

	if response.Type != STUN_BINDING_RESPONSE {
		return nil, fmt.Errorf("unexpected STUN response type: %d", response.Type)
	}

	endpoint, err := sc.extractExternalAddress(response)
	if err != nil {
		return nil, fmt.Errorf("failed to extract external address: %w", err)
	}

	endpoint.ServerUsed = serverAddr
	endpoint.LastUpdated = time.Now()

	return endpoint, nil
}

func (sc *STUNClient) createBindingRequest() *STUNMessage {
	msg := &STUNMessage{
		Type:        STUN_BINDING_REQUEST,
		Length:      0,
		MagicCookie: STUN_MAGIC_COOKIE,
	}

	rand.Read(msg.TransactionID[:])
	
	return msg
}

func (sc *STUNClient) marshalSTUNMessage(msg *STUNMessage) []byte {
	attributesData := make([]byte, 0)
	for _, attr := range msg.Attributes {
		attrData := make([]byte, 4+len(attr.Value))
		binary.BigEndian.PutUint16(attrData[0:2], attr.Type)
		binary.BigEndian.PutUint16(attrData[2:4], attr.Length)
		copy(attrData[4:], attr.Value)
		
		padding := (4 - (len(attr.Value) % 4)) % 4
		if padding > 0 {
			attrData = append(attrData, make([]byte, padding)...)
		}
		
		attributesData = append(attributesData, attrData...)
	}

	data := make([]byte, 20+len(attributesData))
	binary.BigEndian.PutUint16(data[0:2], msg.Type)
	binary.BigEndian.PutUint16(data[2:4], uint16(len(attributesData)))
	binary.BigEndian.PutUint32(data[4:8], msg.MagicCookie)
	copy(data[8:20], msg.TransactionID[:])
	copy(data[20:], attributesData)

	return data
}

func (sc *STUNClient) parseSTUNMessage(data []byte) (*STUNMessage, error) {
	if len(data) < 20 {
		return nil, fmt.Errorf("STUN message too short")
	}

	msg := &STUNMessage{
		Type:        binary.BigEndian.Uint16(data[0:2]),
		Length:      binary.BigEndian.Uint16(data[2:4]),
		MagicCookie: binary.BigEndian.Uint32(data[4:8]),
	}

	copy(msg.TransactionID[:], data[8:20])

	if msg.MagicCookie != STUN_MAGIC_COOKIE {
		return nil, fmt.Errorf("invalid STUN magic cookie")
	}

	attributesData := data[20:]
	offset := 0

	for offset < len(attributesData) {
		if offset+4 > len(attributesData) {
			break
		}

		attrType := binary.BigEndian.Uint16(attributesData[offset:offset+2])
		attrLength := binary.BigEndian.Uint16(attributesData[offset+2:offset+4])
		
		if offset+4+int(attrLength) > len(attributesData) {
			break
		}

		attr := STUNAttribute{
			Type:   attrType,
			Length: attrLength,
			Value:  make([]byte, attrLength),
		}
		copy(attr.Value, attributesData[offset+4:offset+4+int(attrLength)])

		msg.Attributes = append(msg.Attributes, attr)

		offset += 4 + int(attrLength)
		padding := (4 - (int(attrLength) % 4)) % 4
		offset += padding
	}

	return msg, nil
}

func (sc *STUNClient) extractExternalAddress(msg *STUNMessage) (*ExternalEndpoint, error) {
	for _, attr := range msg.Attributes {
		if attr.Type == ATTR_XOR_MAPPED_ADDRESS || attr.Type == ATTR_MAPPED_ADDRESS {
			return sc.parseAddressAttribute(&attr, attr.Type == ATTR_XOR_MAPPED_ADDRESS, msg.TransactionID)
		}
	}
	return nil, fmt.Errorf("no mapped address found in STUN response")
}

func (sc *STUNClient) parseAddressAttribute(attr *STUNAttribute, isXOR bool, transactionID [12]byte) (*ExternalEndpoint, error) {
	if len(attr.Value) < 8 {
		return nil, fmt.Errorf("address attribute too short")
	}

	family := uint16(attr.Value[1])
	port := binary.BigEndian.Uint16(attr.Value[2:4])
	
	if isXOR {
		port ^= uint16(STUN_MAGIC_COOKIE >> 16)
	}

	var ip net.IP
	if family == 0x01 {
		if len(attr.Value) < 8 {
			return nil, fmt.Errorf("IPv4 address attribute too short")
		}
		
		ipBytes := attr.Value[4:8]
		if isXOR {
			magicBytes := make([]byte, 4)
			binary.BigEndian.PutUint32(magicBytes, STUN_MAGIC_COOKIE)
			for i := 0; i < 4; i++ {
				ipBytes[i] ^= magicBytes[i]
			}
		}
		ip = net.IP(ipBytes)
	} else if family == 0x02 {
		if len(attr.Value) < 20 {
			return nil, fmt.Errorf("IPv6 address attribute too short")
		}
		
		ipBytes := make([]byte, 16)
		copy(ipBytes, attr.Value[4:20])
		if isXOR {
			magicBytes := make([]byte, 4)
			binary.BigEndian.PutUint32(magicBytes, STUN_MAGIC_COOKIE)
			for i := 0; i < 4; i++ {
				ipBytes[i] ^= magicBytes[i]
			}
			for i := 4; i < 16; i++ {
				ipBytes[i] ^= transactionID[i-4]
			}
		}
		ip = net.IP(ipBytes)
	} else {
		return nil, fmt.Errorf("unsupported address family: %d", family)
	}

	return &ExternalEndpoint{
		IP:      ip,
		Port:    int(port),
		NATType: NAT_TYPE_UNKNOWN, // Will be set by caller
	}, nil
}

type NATTypeDetector struct {
	stunClient *STUNClient
	logger     *zap.Logger
}

func NewNATTypeDetector(stunClient *STUNClient, logger *zap.Logger) *NATTypeDetector {
	return &NATTypeDetector{
		stunClient: stunClient,
		logger:     logger,
	}
}

func (ntd *NATTypeDetector) DetectNATType() (int, error) {
	endpoint1, err := ntd.stunClient.DiscoverExternalAddress()
	if err != nil {
		return NAT_TYPE_UNKNOWN, fmt.Errorf("failed to discover external address: %w", err)
	}

	if len(ntd.stunClient.servers) < 2 {
		ntd.logger.Warn("Not enough STUN servers for comprehensive NAT type detection")
		return NAT_TYPE_UNKNOWN, nil
	}

	// Create a copy of servers to avoid mutating shared state (thread-safety fix)
	serversCopy := make([]STUNServer, len(ntd.stunClient.servers))
	copy(serversCopy, ntd.stunClient.servers)
	
	// Query second server directly without modifying the shared client
	endpoint2, err := ntd.querySpecificSTUNServer(serversCopy[1])
	
	if err != nil {
		ntd.logger.Warn("Failed to query second STUN server for NAT type detection", zap.Error(err))
		return NAT_TYPE_UNKNOWN, nil
	}

	if !endpoint1.IP.Equal(endpoint2.IP) || endpoint1.Port != endpoint2.Port {
		ntd.logger.Info("Detected Symmetric NAT", 
			zap.String("endpoint1", fmt.Sprintf("%s:%d", endpoint1.IP, endpoint1.Port)),
			zap.String("endpoint2", fmt.Sprintf("%s:%d", endpoint2.IP, endpoint2.Port)))
		return NAT_TYPE_SYMMETRIC, nil
	}

	ntd.logger.Info("Detected Cone NAT", 
		zap.String("external_endpoint", fmt.Sprintf("%s:%d", endpoint1.IP, endpoint1.Port)))
	
	return NAT_TYPE_FULL_CONE, nil
}

// querySpecificSTUNServer queries a specific STUN server without mutating client state
func (ntd *NATTypeDetector) querySpecificSTUNServer(server STUNServer) (*ExternalEndpoint, error) {
	return ntd.stunClient.querySTUNServer(server)
}

type UDPHolePuncher struct {
	localAddr    *net.UDPAddr
	connections  map[string]*PeerConnection
	logger       *zap.Logger
	mu           sync.RWMutex
	receiver     *net.UDPConn // Single UDP conn for both receiving and sending
	stopReceiver chan struct{}
	stopped      int32 // Atomic flag to track if the hole puncher has been stopped
	pendingPings map[uint64]time.Time // Track pending PINGs for RTT correlation
	pongCh       map[uint64]chan time.Duration // Channels for PONG RTT responses
	pingsMu      sync.Mutex           // Mutex for pendingPings and pongCh maps
}

func NewUDPHolePuncher(localAddr *net.UDPAddr, logger *zap.Logger) (*UDPHolePuncher, error) {
	// Use a single UDP connection for both sending and receiving to avoid port conflicts
	listener, err := net.ListenUDP("udp", localAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to create UDP listener: %w", err)
	}
	
	uhp := &UDPHolePuncher{
		localAddr:    localAddr,
		connections:  make(map[string]*PeerConnection),
		logger:       logger,
		receiver:     listener,
		stopReceiver: make(chan struct{}),
		pendingPings: make(map[uint64]time.Time),
		pongCh:       make(map[uint64]chan time.Duration),
	}
	// Start the receiver goroutine
	go uhp.receiverLoop()
	return uhp, nil
}

func (uhp *UDPHolePuncher) EstablishConnection(peerID string, remoteAddr *net.UDPAddr) (*PeerConnection, error) {
	uhp.mu.Lock()
	defer uhp.mu.Unlock()

	if conn, exists := uhp.connections[peerID]; exists && conn.Established {
		return conn, nil
	}

	// Use the existing receiver connection for sending and receiving
	// This avoids port conflicts and EADDRINUSE errors
	peerConn := &PeerConnection{
		PeerID:         peerID,
		LocalEndpoint:  uhp.localAddr,
		RemoteEndpoint: remoteAddr,
		ConnectionType: "nat_traversal", // Fixed: correctly label as NAT traversal
		Established:    false,
		LastActivity:   time.Now(),
		conn:           uhp.receiver,  // Use the shared receiver connection
	}

	if err := uhp.performHandshake(peerConn); err != nil {
		return nil, fmt.Errorf("handshake failed: %w", err)
	}

	peerConn.Established = true
	uhp.connections[peerID] = peerConn

	uhp.logger.Info("UDP hole punching successful", 
		zap.String("peer_id", peerID),
		zap.String("remote_addr", remoteAddr.String()))

	go uhp.measureConnectionQuality(peerConn)

	return peerConn, nil
}

func (uhp *UDPHolePuncher) performHandshake(conn *PeerConnection) error {
	// Check if stopped before each write
	if atomic.LoadInt32(&uhp.stopped) == 1 {
		return fmt.Errorf("stopped")
	}

	handshakeMsg := []byte("{\"type\":\"HANDSHAKE\",\"peer_id\":\"" + conn.PeerID + "\"}")

	for i := 0; i < 5; i++ {
		// Check if stopped before each write
		if atomic.LoadInt32(&uhp.stopped) == 1 {
			return fmt.Errorf("stopped")
		}

		// Use WriteToUDP for sending via the shared receiver connection
		_, err := uhp.receiver.WriteToUDP(handshakeMsg, conn.RemoteEndpoint)
		if err != nil {
			return fmt.Errorf("failed to send handshake: %w", err)
		}

		// Wait for response (handled by receiverLoop)
		time.Sleep(500 * time.Millisecond)
		
		// Check if connection was established
		if conn.Established {
			return nil
		}
	}

	return fmt.Errorf("handshake timeout")
}

func (uhp *UDPHolePuncher) measureConnectionQuality(conn *PeerConnection) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		quality, err := uhp.measureRTT(conn)
		if err != nil {
			uhp.logger.Warn("Failed to measure connection quality",
				zap.String("peer_id", conn.PeerID),
				zap.Error(err))
			continue
		}

		uhp.mu.Lock()
		// Only update RTT if we got a PONG-based measurement (PacketLoss == 0)
		// or if the new RTT is better than existing
		if quality.PacketLoss == 0 {
			// Got actual PONG response, use accurate RTT
			conn.Quality.RTT = quality.RTT
			conn.Quality.PacketLoss = quality.PacketLoss
		} else if conn.Quality.RTT == 0 || quality.RTT < conn.Quality.RTT {
			// No existing RTT or new heuristic is better, update conservatively
			conn.Quality.RTT = quality.RTT
			conn.Quality.PacketLoss = quality.PacketLoss
		} else {
			// Keep existing RTT if it's better, but update packet loss
			conn.Quality.PacketLoss = quality.PacketLoss
		}
		conn.Quality.LastMeasured = quality.LastMeasured
		conn.LastActivity = time.Now()
		uhp.mu.Unlock()
	}
}

func (uhp *UDPHolePuncher) measureRTT(conn *PeerConnection) (*ConnectionQuality, error) {
	// Check if stopped before measurement
	if atomic.LoadInt32(&uhp.stopped) == 1 {
		return nil, fmt.Errorf("stopped")
	}

	start := time.Now()
	pingID := uint64(start.UnixNano())

	// Create channel for this ping's PONG response
	pongChan := make(chan time.Duration, 1)

	// Store pending ping and channel for RTT correlation
	uhp.pingsMu.Lock()
	uhp.pendingPings[pingID] = start
	uhp.pongCh[pingID] = pongChan
	uhp.pingsMu.Unlock()

	// Clean up on return
	defer func() {
		uhp.pingsMu.Lock()
		delete(uhp.pendingPings, pingID)
		delete(uhp.pongCh, pingID)
		uhp.pingsMu.Unlock()
		close(pongChan)
	}()

	// Send PING message
	pingMsg := []byte(fmt.Sprintf("{\"type\":\"PING\",\"id\":%d}", pingID))
	_, err := uhp.receiver.WriteToUDP(pingMsg, conn.RemoteEndpoint)
	if err != nil {
		return nil, err
	}

	// Get previous RTT for fallback
	uhp.mu.RLock()
	previousRTT := conn.Quality.RTT
	if previousRTT == 0 {
		previousRTT = 100 * time.Millisecond // Initial estimate
	}
	uhp.mu.RUnlock()

	// Wait for PONG response with actual RTT measurement
	select {
	case rtt := <-pongChan:
		// Received actual RTT from PONG
		quality := &ConnectionQuality{
			RTT:          rtt,
			PacketLoss:   0.0,
			LastMeasured: time.Now(),
		}
		return quality, nil

	case <-time.After(2 * time.Second):
		// Timeout - indicate packet loss and use degraded RTT
		quality := &ConnectionQuality{
			RTT:          previousRTT + 50*time.Millisecond, // Increase RTT on timeout
			PacketLoss:   0.1, // Indicate packet loss
			LastMeasured: time.Now(),
		}
		return quality, nil
	}
}

func (uhp *UDPHolePuncher) GetConnection(peerID string) (*PeerConnection, bool) {
	uhp.mu.RLock()
	defer uhp.mu.RUnlock()

	conn, exists := uhp.connections[peerID]
	return conn, exists
}

// GetLocalAddress returns the local UDP address of the hole puncher
func (uhp *UDPHolePuncher) GetLocalAddress() *net.UDPAddr {
	if uhp.receiver != nil {
		return uhp.receiver.LocalAddr().(*net.UDPAddr)
	}
	return uhp.localAddr
}

// GetReceiver returns the UDP connection for testing purposes
func (uhp *UDPHolePuncher) GetReceiver() *net.UDPConn {
	return uhp.receiver
}

// AddConnection adds a connection for testing purposes
func (uhp *UDPHolePuncher) AddConnection(conn *PeerConnection) {
	uhp.mu.Lock()
	defer uhp.mu.Unlock()
	uhp.connections[conn.PeerID] = conn
}

// MeasureRTT measures the RTT to a peer connection (exported for testing)
func (uhp *UDPHolePuncher) MeasureRTT(conn *PeerConnection) (*ConnectionQuality, error) {
	return uhp.measureRTT(conn)
}

// HandleIncomingMessage processes incoming UDP messages (exported for testing)
func (uhp *UDPHolePuncher) HandleIncomingMessage(message string, addr *net.UDPAddr) {
	uhp.handleIncomingMessage(message, addr)
}

func (uhp *UDPHolePuncher) CloseConnection(peerID string) error {
	uhp.mu.Lock()
	defer uhp.mu.Unlock()

	if atomic.LoadInt32(&uhp.stopped) == 1 {
		return fmt.Errorf("hole puncher is stopped")
	}

	_, exists := uhp.connections[peerID]
	if !exists {
		return fmt.Errorf("connection to peer %s not found", peerID)
	}

	// Don't close conn.conn since it's the shared receiver
	delete(uhp.connections, peerID)

	uhp.logger.Info("Connection closed", zap.String("peer_id", peerID))
	return nil
}

type NATTraversalManager struct {
	config            *NATTraversalConfig
	stunClient        *STUNClient
	natTypeDetector   *NATTypeDetector
	holePuncher       *UDPHolePuncher
	currentExternal   *ExternalEndpoint
	natType           int
	logger            *zap.Logger
	mu                sync.RWMutex
	refreshInterval   time.Duration
	ctx               context.Context
	cancel            context.CancelFunc
}

type NATTraversalConfig struct {
	STUNServers      []STUNServer  `json:"stun_servers"`
	RefreshInterval  time.Duration `json:"refresh_interval"`
	LocalPort        int           `json:"local_port"`
	EnableRelay      bool          `json:"enable_relay"`
	RelayServers     []string      `json:"relay_servers"`
}

func NewNATTraversalManager(config *NATTraversalConfig, logger *zap.Logger) (*NATTraversalManager, error) {
	if config.RefreshInterval == 0 {
		config.RefreshInterval = 5 * time.Minute
	}

	stunClient := NewSTUNClient(config.STUNServers, logger)
	natTypeDetector := NewNATTypeDetector(stunClient, logger)

	localAddr := &net.UDPAddr{
		IP:   net.IPv4zero,
		Port: config.LocalPort,
	}

	holePuncher, err := NewUDPHolePuncher(localAddr, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create UDP hole puncher: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	ntm := &NATTraversalManager{
		config:          config,
		stunClient:      stunClient,
		natTypeDetector: natTypeDetector,
		holePuncher:     holePuncher,
		logger:          logger,
		refreshInterval: config.RefreshInterval,
		ctx:             ctx,
		cancel:          cancel,
	}

	return ntm, nil
}

func (ntm *NATTraversalManager) Start() error {
	external, err := ntm.stunClient.DiscoverExternalAddress()
	if err != nil {
		return fmt.Errorf("failed to discover external address: %w", err)
	}

	natType, err := ntm.natTypeDetector.DetectNATType()
	if err != nil {
		ntm.logger.Warn("Failed to detect NAT type", zap.Error(err))
		natType = NAT_TYPE_UNKNOWN
	}

	ntm.mu.Lock()
	external.NATType = natType
	ntm.currentExternal = external
	ntm.natType = natType
	ntm.mu.Unlock()

	go ntm.refreshLoop()

	ntm.logger.Info("NAT traversal manager started", 
		zap.String("external_ip", external.IP.String()),
		zap.Int("external_port", external.Port),
		zap.Int("nat_type", natType))

	return nil
}

func (ntm *NATTraversalManager) Stop() {
	ntm.cancel()
	// Properly stop the UDP hole puncher and its receiver
	if ntm.holePuncher != nil {
		ntm.holePuncher.Stop()
	}
	ntm.logger.Info("NAT traversal manager stopped")
}

func (ntm *NATTraversalManager) refreshLoop() {
	ticker := time.NewTicker(ntm.refreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ntm.ctx.Done():
			return
		case <-ticker.C:
			ntm.refreshExternalAddress()
		}
	}
}

func (ntm *NATTraversalManager) refreshExternalAddress() {
	external, err := ntm.stunClient.DiscoverExternalAddress()
	if err != nil {
		ntm.logger.Warn("Failed to refresh external address", zap.Error(err))
		return
	}

	ntm.mu.Lock()
	oldExternal := ntm.currentExternal
	ntm.currentExternal = external
	ntm.mu.Unlock()

	if oldExternal != nil && (!oldExternal.IP.Equal(external.IP) || oldExternal.Port != external.Port) {
		ntm.logger.Info("External address changed", 
			zap.String("old", fmt.Sprintf("%s:%d", oldExternal.IP, oldExternal.Port)),
			zap.String("new", fmt.Sprintf("%s:%d", external.IP, external.Port)))
	}
}

func (ntm *NATTraversalManager) GetExternalEndpoint() *ExternalEndpoint {
	ntm.mu.RLock()
	defer ntm.mu.RUnlock()
	return ntm.currentExternal
}

func (ntm *NATTraversalManager) GetNATType() int {
	ntm.mu.RLock()
	defer ntm.mu.RUnlock()
	return ntm.natType
}

func (ntm *NATTraversalManager) EstablishP2PConnection(peerID string, remoteAddr *net.UDPAddr) (*PeerConnection, error) {
	// Check NAT type and use relay if necessary
	if ntm.GetNATType() == NAT_TYPE_SYMMETRIC && ntm.config.EnableRelay {
		return ntm.establishRelayConnection(peerID, remoteAddr)
	}
	
	// Try direct hole punching first
	conn, err := ntm.holePuncher.EstablishConnection(peerID, remoteAddr)
	if err != nil && ntm.config.EnableRelay {
		// Fallback to relay on failure
		ntm.logger.Warn("Direct connection failed, falling back to relay",
			zap.String("peer_id", peerID),
			zap.Error(err))
		return ntm.establishRelayConnection(peerID, remoteAddr)
	}
	
	return conn, err
}

// establishRelayConnection establishes a connection via relay server
func (ntm *NATTraversalManager) establishRelayConnection(peerID string, remoteAddr *net.UDPAddr) (*PeerConnection, error) {
	if len(ntm.config.RelayServers) == 0 {
		return nil, fmt.Errorf("no relay servers configured")
	}
	
	// Use the first available relay server
	relayServer := ntm.config.RelayServers[0]
	
	// TODO: Implement actual relay client connection
	// For now, return a placeholder connection marked as relay type
	ntm.logger.Info("Establishing relay connection",
		zap.String("peer_id", peerID),
		zap.String("relay_server", relayServer))
	
	// Create a relay connection placeholder
	peerConn := &PeerConnection{
		PeerID:         peerID,
		LocalEndpoint:  ntm.holePuncher.localAddr,
		RemoteEndpoint: remoteAddr,
		ConnectionType: "relay",
		Established:    true,
		LastActivity:   time.Now(),
		Quality: ConnectionQuality{
			RTT:          200 * time.Millisecond, // Estimated relay latency
			PacketLoss:   0.01, // Estimated relay packet loss
			LastMeasured: time.Now(),
		},
	}
	
	// Register the relay connection
	ntm.holePuncher.mu.Lock()
	ntm.holePuncher.connections[peerID] = peerConn
	ntm.holePuncher.mu.Unlock()
	
	return peerConn, nil
}

func (ntm *NATTraversalManager) GetConnectionQuality(peerID string) (*ConnectionQuality, error) {
	conn, exists := ntm.holePuncher.GetConnection(peerID)
	if !exists {
		return nil, fmt.Errorf("no connection to peer %s", peerID)
	}

	return &conn.Quality, nil
}
// receiverLoop handles incoming UDP messages
func (uhp *UDPHolePuncher) receiverLoop() {
	// Do not defer Close here - let Stop() handle it

	buffer := make([]byte, 4096)
	for {
		select {
		case <-uhp.stopReceiver:
			return
		default:
			uhp.receiver.SetReadDeadline(time.Now().Add(1 * time.Second))
			n, addr, err := uhp.receiver.ReadFromUDP(buffer)
			if err != nil {
				// Check if we're stopped or connection closed
				if atomic.LoadInt32(&uhp.stopped) == 1 || strings.Contains(err.Error(), "use of closed") {
					return
				}
				// Handle timeout by checking stop channel
				if strings.Contains(err.Error(), "timeout") {
					select {
					case <-uhp.stopReceiver:
						return
					default:
						continue
					}
				}
				uhp.logger.Debug("UDP receiver read error", zap.Error(err))
				continue
			}
			
			message := string(buffer[:n])
			uhp.handleIncomingMessage(message, addr)
		}
	}
}

// handleIncomingMessage processes incoming UDP messages
func (uhp *UDPHolePuncher) handleIncomingMessage(message string, addr *net.UDPAddr) {
	// Check if shutting down
	if atomic.LoadInt32(&uhp.stopped) == 1 {
		return
	}

	uhp.logger.Debug("Received UDP message",
		zap.String("message", message),
		zap.String("from", addr.String()))

	// Handle protocol messages (both old and new formats)
	if strings.Contains(message, "HANDSHAKE:") || strings.Contains(message, "\"type\":\"HANDSHAKE\"") {
		// Check if stopped before replying
		if atomic.LoadInt32(&uhp.stopped) == 1 {
			return
		}
		// Reply with HANDSHAKE_ACK
		response := []byte("{\"type\":\"HANDSHAKE_ACK\"}")
		uhp.receiver.WriteToUDP(response, addr)
	} else if strings.Contains(message, "HANDSHAKE_ACK") || strings.Contains(message, "\"type\":\"HANDSHAKE_ACK\"") {
		// Mark connection as established
		uhp.mu.Lock()
		for _, conn := range uhp.connections {
			if conn.RemoteEndpoint != nil && conn.RemoteEndpoint.String() == addr.String() {
				conn.Established = true
				conn.LastActivity = time.Now()
				break
			}
		}
		uhp.mu.Unlock()
	} else if strings.Contains(message, "PING:") || strings.Contains(message, "\"type\":\"PING\"") {
		// Check if stopped before replying
		if atomic.LoadInt32(&uhp.stopped) == 1 {
			return
		}

		// Extract ID and reply with PONG
		var id string
		if strings.Contains(message, "\"id\":") {
			// JSON format with ID
			start := strings.Index(message, "\"id\":")
			if start >= 0 {
				start += 5
				end := strings.Index(message[start:], "}")
				if end > 0 {
					id = message[start:start+end]
				}
			}
		} else if strings.HasPrefix(message, "PING:") {
			// Old format
			id = message[5:]
		}

		response := []byte(fmt.Sprintf("{\"type\":\"PONG\",\"id\":%s}", id))
		uhp.receiver.WriteToUDP(response, addr)
	} else if strings.Contains(message, "PONG:") || strings.Contains(message, "\"type\":\"PONG\"") {
		// Handle PONG response for RTT measurement
		var id uint64
		if strings.Contains(message, "\"id\":") {
			// Extract ping ID from PONG response
			start := strings.Index(message, "\"id\":")
			if start >= 0 {
				start += 5
				end := strings.Index(message[start:], "}")
				if end > 0 {
					idStr := message[start:start+end]
					// Parse ID as uint64
					if _, err := fmt.Sscanf(idStr, "%d", &id); err == nil {
						// Look up pending ping
						uhp.pingsMu.Lock()
						if startTime, ok := uhp.pendingPings[id]; ok {
							rtt := time.Since(startTime)
							delete(uhp.pendingPings, id)

							// Send RTT to waiting channel if it exists
							if ch, ok := uhp.pongCh[id]; ok {
								select {
								case ch <- rtt:
									// Successfully sent RTT to channel
								default:
									// Channel buffer full, ignore (shouldn't happen with buffer size 1)
								}
								// Note: Don't delete pongCh here, measureRTT cleans it up
							}
							uhp.pingsMu.Unlock()

							// Also update connection quality directly for non-measureRTT pings
							uhp.mu.Lock()
							for _, conn := range uhp.connections {
								if conn.RemoteEndpoint != nil && conn.RemoteEndpoint.String() == addr.String() {
									conn.Quality.RTT = rtt
									conn.Quality.LastMeasured = time.Now()
									break
								}
							}
							uhp.mu.Unlock()
						} else {
							uhp.pingsMu.Unlock()
						}
					}
				}
			}
		}
	}
	
	// Update connection activity if we know this peer
	uhp.mu.Lock()
	for _, conn := range uhp.connections {
		if conn.RemoteEndpoint != nil && conn.RemoteEndpoint.String() == addr.String() {
			conn.LastActivity = time.Now()
			break
		}
	}
	uhp.mu.Unlock()
}

// Stop stops the UDP hole puncher and its receiver
func (uhp *UDPHolePuncher) Stop() {
	// Use atomic compare-and-swap to ensure stop is only called once
	if !atomic.CompareAndSwapInt32(&uhp.stopped, 0, 1) {
		return // Already stopped
	}

	uhp.mu.Lock()
	defer uhp.mu.Unlock()
	close(uhp.stopReceiver)
	if uhp.receiver != nil {
		uhp.receiver.Close()
	}

	// Clear all connections
	for peerID := range uhp.connections {
		delete(uhp.connections, peerID)
	}

	// Clean up pending pings and channels
	uhp.pingsMu.Lock()
	for id := range uhp.pendingPings {
		delete(uhp.pendingPings, id)
	}
	// Close all pong channels
	for id, ch := range uhp.pongCh {
		close(ch)
		delete(uhp.pongCh, id)
	}
	uhp.pingsMu.Unlock()

	uhp.logger.Info("UDP hole puncher stopped")
}

func (ntm *NATTraversalManager) GetActiveConnections() map[string]*PeerConnection {
	ntm.holePuncher.mu.RLock()
	defer ntm.holePuncher.mu.RUnlock()

	connections := make(map[string]*PeerConnection)
	for peerID, conn := range ntm.holePuncher.connections {
		if conn.Established {
			connections[peerID] = conn
		}
	}

	return connections
}