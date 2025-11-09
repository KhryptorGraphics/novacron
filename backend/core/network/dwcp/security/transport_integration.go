package security

import (
	"crypto/tls"
	"fmt"
	"net"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SecureTransport wraps transport connections with security features
type SecureTransport struct {
	tlsManager      *TLSManager
	securityAuditor *SecurityAuditor
	dataEncryptor   *DataEncryptor
	logger          *zap.Logger
	connections     map[string]*SecureConnection
	connMu          sync.RWMutex
}

// SecureConnection represents a secured transport connection
type SecureConnection struct {
	tlsConn         *tls.Conn
	rawConn         net.Conn
	remoteAddr      string
	established     time.Time
	lastActivity    time.Time
	bytesSent       uint64
	bytesReceived   uint64
	tlsVersion      uint16
	cipherSuite     uint16
	mu              sync.Mutex
}

// NewSecureTransport creates a new secure transport wrapper
func NewSecureTransport(tlsManager *TLSManager, auditor *SecurityAuditor, encryptor *DataEncryptor, logger *zap.Logger) *SecureTransport {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &SecureTransport{
		tlsManager:      tlsManager,
		securityAuditor: auditor,
		dataEncryptor:   encryptor,
		logger:          logger,
		connections:     make(map[string]*SecureConnection),
	}
}

// WrapConnection wraps a raw connection with TLS
func (st *SecureTransport) WrapConnection(conn net.Conn, isClient bool) (*tls.Conn, error) {
	tlsConfig := st.tlsManager.GetTLSConfig()

	var tlsConn *tls.Conn
	if isClient {
		tlsConn = tls.Client(conn, tlsConfig)
	} else {
		tlsConn = tls.Server(conn, tlsConfig)
	}

	// Perform handshake with timeout
	if err := tlsConn.SetDeadline(time.Now().Add(30 * time.Second)); err != nil {
		return nil, fmt.Errorf("failed to set handshake deadline: %w", err)
	}

	if err := tlsConn.Handshake(); err != nil {
		st.securityAuditor.AuditHandshakeFailure(conn.RemoteAddr().String(), err)
		return nil, fmt.Errorf("TLS handshake failed: %w", err)
	}

	// Clear deadline after handshake
	if err := tlsConn.SetDeadline(time.Time{}); err != nil {
		return nil, fmt.Errorf("failed to clear deadline: %w", err)
	}

	// Audit successful connection
	st.securityAuditor.AuditTLSConnection(tlsConn)

	// Track connection
	secConn := &SecureConnection{
		tlsConn:      tlsConn,
		rawConn:      conn,
		remoteAddr:   conn.RemoteAddr().String(),
		established:  time.Now(),
		lastActivity: time.Now(),
		tlsVersion:   tlsConn.ConnectionState().Version,
		cipherSuite:  tlsConn.ConnectionState().CipherSuite,
	}

	st.connMu.Lock()
	st.connections[conn.RemoteAddr().String()] = secConn
	st.connMu.Unlock()

	st.logger.Info("Secure connection established",
		zap.String("remote_addr", conn.RemoteAddr().String()),
		zap.String("tls_version", tlsVersionName(secConn.tlsVersion)),
		zap.String("cipher_suite", cipherSuiteName(secConn.cipherSuite)))

	return tlsConn, nil
}

// SecureDial establishes a secure client connection
func (st *SecureTransport) SecureDial(network, address string) (*tls.Conn, error) {
	// Dial raw connection
	conn, err := net.DialTimeout(network, address, 30*time.Second)
	if err != nil {
		return nil, fmt.Errorf("dial failed: %w", err)
	}

	// Wrap with TLS
	tlsConn, err := st.WrapConnection(conn, true)
	if err != nil {
		conn.Close()
		return nil, err
	}

	return tlsConn, nil
}

// SecureListen creates a secure listener
func (st *SecureTransport) SecureListen(network, address string) (net.Listener, error) {
	// Create base listener
	listener, err := net.Listen(network, address)
	if err != nil {
		return nil, fmt.Errorf("listen failed: %w", err)
	}

	st.logger.Info("Secure listener created",
		zap.String("network", network),
		zap.String("address", address))

	return &SecureListener{
		listener:        listener,
		secureTransport: st,
		logger:          st.logger,
	}, nil
}

// EncryptData encrypts data for transmission
func (st *SecureTransport) EncryptData(data []byte) ([]byte, error) {
	if st.dataEncryptor == nil {
		return data, nil // No encryption configured
	}

	encrypted, err := st.dataEncryptor.Encrypt(data)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	return encrypted, nil
}

// DecryptData decrypts received data
func (st *SecureTransport) DecryptData(data []byte) ([]byte, error) {
	if st.dataEncryptor == nil {
		return data, nil // No encryption configured
	}

	decrypted, err := st.dataEncryptor.Decrypt(data)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	return decrypted, nil
}

// GetConnection returns a tracked secure connection
func (st *SecureTransport) GetConnection(remoteAddr string) (*SecureConnection, bool) {
	st.connMu.RLock()
	defer st.connMu.RUnlock()
	conn, exists := st.connections[remoteAddr]
	return conn, exists
}

// RemoveConnection removes a tracked connection
func (st *SecureTransport) RemoveConnection(remoteAddr string) {
	st.connMu.Lock()
	defer st.connMu.Unlock()
	delete(st.connections, remoteAddr)
}

// GetActiveConnections returns count of active connections
func (st *SecureTransport) GetActiveConnections() int {
	st.connMu.RLock()
	defer st.connMu.RUnlock()
	return len(st.connections)
}

// GetConnectionStats returns statistics for all connections
func (st *SecureTransport) GetConnectionStats() []map[string]interface{} {
	st.connMu.RLock()
	defer st.connMu.RUnlock()

	stats := make([]map[string]interface{}, 0, len(st.connections))
	for _, conn := range st.connections {
		conn.mu.Lock()
		stats = append(stats, map[string]interface{}{
			"remote_addr":     conn.remoteAddr,
			"established":     conn.established,
			"last_activity":   conn.lastActivity,
			"bytes_sent":      conn.bytesSent,
			"bytes_received":  conn.bytesReceived,
			"tls_version":     tlsVersionName(conn.tlsVersion),
			"cipher_suite":    cipherSuiteName(conn.cipherSuite),
			"duration":        time.Since(conn.established).String(),
		})
		conn.mu.Unlock()
	}

	return stats
}

// SecureListener wraps a listener with TLS
type SecureListener struct {
	listener        net.Listener
	secureTransport *SecureTransport
	logger          *zap.Logger
}

// Accept waits for and returns the next secure connection
func (sl *SecureListener) Accept() (net.Conn, error) {
	conn, err := sl.listener.Accept()
	if err != nil {
		return nil, err
	}

	// Wrap with TLS (server mode)
	tlsConn, err := sl.secureTransport.WrapConnection(conn, false)
	if err != nil {
		conn.Close()
		sl.logger.Error("Failed to wrap connection with TLS",
			zap.String("remote_addr", conn.RemoteAddr().String()),
			zap.Error(err))
		return nil, err
	}

	return tlsConn, nil
}

// Close closes the listener
func (sl *SecureListener) Close() error {
	return sl.listener.Close()
}

// Addr returns the listener's network address
func (sl *SecureListener) Addr() net.Addr {
	return sl.listener.Addr()
}

// UpdateActivity updates connection activity timestamp
func (sc *SecureConnection) UpdateActivity() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.lastActivity = time.Now()
}

// AddBytesSent adds to bytes sent counter
func (sc *SecureConnection) AddBytesSent(bytes uint64) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.bytesSent += bytes
	sc.lastActivity = time.Now()
}

// AddBytesReceived adds to bytes received counter
func (sc *SecureConnection) AddBytesReceived(bytes uint64) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.bytesReceived += bytes
	sc.lastActivity = time.Now()
}

// GetStats returns connection statistics
func (sc *SecureConnection) GetStats() map[string]interface{} {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	return map[string]interface{}{
		"remote_addr":     sc.remoteAddr,
		"established":     sc.established,
		"last_activity":   sc.lastActivity,
		"bytes_sent":      sc.bytesSent,
		"bytes_received":  sc.bytesReceived,
		"tls_version":     tlsVersionName(sc.tlsVersion),
		"cipher_suite":    cipherSuiteName(sc.cipherSuite),
		"duration":        time.Since(sc.established).String(),
		"idle_duration":   time.Since(sc.lastActivity).String(),
	}
}
