package security

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SecurityEvent represents a security-related event
type SecurityEvent struct {
	Timestamp     time.Time              `json:"timestamp"`
	EventType     string                 `json:"event_type"`
	Severity      string                 `json:"severity"`
	Source        string                 `json:"source"`
	Description   string                 `json:"description"`
	Details       map[string]interface{} `json:"details"`
	RemoteAddr    string                 `json:"remote_addr,omitempty"`
	LocalAddr     string                 `json:"local_addr,omitempty"`
	TLSVersion    string                 `json:"tls_version,omitempty"`
	CipherSuite   string                 `json:"cipher_suite,omitempty"`
	PeerCertInfo  string                 `json:"peer_cert_info,omitempty"`
}

// SecurityAuditor handles security auditing and logging
type SecurityAuditor struct {
	logger        *zap.Logger
	events        []SecurityEvent
	eventsMu      sync.RWMutex
	maxEvents     int
	alertHandlers []func(SecurityEvent)
	statsmu       sync.RWMutex
	stats         SecurityStats
}

// SecurityStats tracks security-related statistics
type SecurityStats struct {
	TotalConnections    int64
	TLSConnections      int64
	FailedHandshakes    int64
	ExpiredCerts        int64
	RevokedCerts        int64
	WeakProtocols       int64
	AuthFailures        int64
	LastUpdate          time.Time
}

// NewSecurityAuditor creates a new security auditor
func NewSecurityAuditor(logger *zap.Logger, maxEvents int) *SecurityAuditor {
	if logger == nil {
		logger = zap.NewNop()
	}

	if maxEvents <= 0 {
		maxEvents = 10000
	}

	return &SecurityAuditor{
		logger:    logger,
		events:    make([]SecurityEvent, 0, maxEvents),
		maxEvents: maxEvents,
		stats:     SecurityStats{LastUpdate: time.Now()},
	}
}

// AuditTLSConnection audits a TLS connection
func (sa *SecurityAuditor) AuditTLSConnection(conn *tls.Conn) {
	state := conn.ConnectionState()

	// Extract peer certificate info
	peerCertInfo := ""
	if len(state.PeerCertificates) > 0 {
		cert := state.PeerCertificates[0]
		peerCertInfo = fmt.Sprintf("CN=%s, Issuer=%s, Expiry=%s",
			cert.Subject.CommonName,
			cert.Issuer.CommonName,
			cert.NotAfter.Format(time.RFC3339))
	}

	event := SecurityEvent{
		Timestamp:    time.Now(),
		EventType:    "tls_connection",
		Severity:     "info",
		Source:       "tls_auditor",
		Description:  "TLS connection established",
		RemoteAddr:   conn.RemoteAddr().String(),
		LocalAddr:    conn.LocalAddr().String(),
		TLSVersion:   tlsVersionName(state.Version),
		CipherSuite:  cipherSuiteName(state.CipherSuite),
		PeerCertInfo: peerCertInfo,
		Details: map[string]interface{}{
			"server_name":          state.ServerName,
			"resumed":              state.DidResume,
			"negotiated_protocol":  state.NegotiatedProtocol,
			"peer_certificates":    len(state.PeerCertificates),
			"verified_chains":      len(state.VerifiedChains),
		},
	}

	sa.logger.Info("TLS connection established",
		zap.String("version", event.TLSVersion),
		zap.String("cipher_suite", event.CipherSuite),
		zap.String("remote_addr", event.RemoteAddr),
		zap.String("server_name", state.ServerName),
		zap.Bool("resumed", state.DidResume),
		zap.String("peer_cert", peerCertInfo))

	// Check for weak configurations
	if state.Version < tls.VersionTLS13 {
		sa.AuditWeakProtocol(state.Version, conn.RemoteAddr().String())
	}

	sa.recordEvent(event)
	sa.updateStats(func(stats *SecurityStats) {
		stats.TotalConnections++
		stats.TLSConnections++
		if state.Version < tls.VersionTLS13 {
			stats.WeakProtocols++
		}
	})
}

// AuditHandshakeFailure audits a TLS handshake failure
func (sa *SecurityAuditor) AuditHandshakeFailure(remoteAddr string, err error) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "handshake_failure",
		Severity:    "warning",
		Source:      "tls_auditor",
		Description: "TLS handshake failed",
		RemoteAddr:  remoteAddr,
		Details: map[string]interface{}{
			"error": err.Error(),
		},
	}

	sa.logger.Warn("TLS handshake failed",
		zap.String("remote_addr", remoteAddr),
		zap.Error(err))

	sa.recordEvent(event)
	sa.updateStats(func(stats *SecurityStats) {
		stats.FailedHandshakes++
	})
	sa.triggerAlert(event)
}

// AuditCertificateExpiry audits certificate expiration
func (sa *SecurityAuditor) AuditCertificateExpiry(cert *x509.Certificate, remoteAddr string) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "certificate_expired",
		Severity:    "error",
		Source:      "certificate_auditor",
		Description: "Expired certificate detected",
		RemoteAddr:  remoteAddr,
		Details: map[string]interface{}{
			"subject":    cert.Subject.CommonName,
			"issuer":     cert.Issuer.CommonName,
			"not_after":  cert.NotAfter,
			"serial":     cert.SerialNumber.String(),
		},
	}

	sa.logger.Error("Expired certificate detected",
		zap.String("subject", cert.Subject.CommonName),
		zap.Time("not_after", cert.NotAfter),
		zap.String("remote_addr", remoteAddr))

	sa.recordEvent(event)
	sa.updateStats(func(stats *SecurityStats) {
		stats.ExpiredCerts++
	})
	sa.triggerAlert(event)
}

// AuditCertificateRevocation audits certificate revocation
func (sa *SecurityAuditor) AuditCertificateRevocation(cert *x509.Certificate, remoteAddr string) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "certificate_revoked",
		Severity:    "error",
		Source:      "certificate_auditor",
		Description: "Revoked certificate detected",
		RemoteAddr:  remoteAddr,
		Details: map[string]interface{}{
			"subject": cert.Subject.CommonName,
			"issuer":  cert.Issuer.CommonName,
			"serial":  cert.SerialNumber.String(),
		},
	}

	sa.logger.Error("Revoked certificate detected",
		zap.String("subject", cert.Subject.CommonName),
		zap.String("serial", cert.SerialNumber.String()),
		zap.String("remote_addr", remoteAddr))

	sa.recordEvent(event)
	sa.updateStats(func(stats *SecurityStats) {
		stats.RevokedCerts++
	})
	sa.triggerAlert(event)
}

// AuditWeakProtocol audits usage of weak protocol versions
func (sa *SecurityAuditor) AuditWeakProtocol(version uint16, remoteAddr string) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "weak_protocol",
		Severity:    "warning",
		Source:      "protocol_auditor",
		Description: "Weak TLS protocol version detected",
		RemoteAddr:  remoteAddr,
		TLSVersion:  tlsVersionName(version),
		Details: map[string]interface{}{
			"version_code": version,
		},
	}

	sa.logger.Warn("Weak TLS protocol version detected",
		zap.String("version", tlsVersionName(version)),
		zap.String("remote_addr", remoteAddr))

	sa.recordEvent(event)
	sa.triggerAlert(event)
}

// AuditAuthFailure audits authentication failures
func (sa *SecurityAuditor) AuditAuthFailure(username, remoteAddr string, reason string) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "auth_failure",
		Severity:    "warning",
		Source:      "auth_auditor",
		Description: "Authentication failed",
		RemoteAddr:  remoteAddr,
		Details: map[string]interface{}{
			"username": username,
			"reason":   reason,
		},
	}

	sa.logger.Warn("Authentication failed",
		zap.String("username", username),
		zap.String("remote_addr", remoteAddr),
		zap.String("reason", reason))

	sa.recordEvent(event)
	sa.updateStats(func(stats *SecurityStats) {
		stats.AuthFailures++
	})
	sa.triggerAlert(event)
}

// AuditCustomEvent audits a custom security event
func (sa *SecurityAuditor) AuditCustomEvent(eventType, severity, description string, details map[string]interface{}) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   eventType,
		Severity:    severity,
		Source:      "custom",
		Description: description,
		Details:     details,
	}

	sa.logger.Info("Custom security event",
		zap.String("event_type", eventType),
		zap.String("severity", severity),
		zap.String("description", description))

	sa.recordEvent(event)

	if severity == "error" || severity == "critical" {
		sa.triggerAlert(event)
	}
}

// recordEvent records a security event
func (sa *SecurityAuditor) recordEvent(event SecurityEvent) {
	sa.eventsMu.Lock()
	defer sa.eventsMu.Unlock()

	// Add event
	sa.events = append(sa.events, event)

	// Trim if exceeding max
	if len(sa.events) > sa.maxEvents {
		// Remove oldest 10%
		trimCount := sa.maxEvents / 10
		sa.events = sa.events[trimCount:]
	}
}

// GetEvents returns recent security events
func (sa *SecurityAuditor) GetEvents(count int) []SecurityEvent {
	sa.eventsMu.RLock()
	defer sa.eventsMu.RUnlock()

	if count <= 0 || count > len(sa.events) {
		count = len(sa.events)
	}

	// Return most recent events
	start := len(sa.events) - count
	events := make([]SecurityEvent, count)
	copy(events, sa.events[start:])

	return events
}

// GetEventsByType returns events filtered by type
func (sa *SecurityAuditor) GetEventsByType(eventType string, count int) []SecurityEvent {
	sa.eventsMu.RLock()
	defer sa.eventsMu.RUnlock()

	filtered := []SecurityEvent{}
	for i := len(sa.events) - 1; i >= 0 && len(filtered) < count; i-- {
		if sa.events[i].EventType == eventType {
			filtered = append(filtered, sa.events[i])
		}
	}

	return filtered
}

// GetStats returns current security statistics
func (sa *SecurityAuditor) GetStats() SecurityStats {
	sa.statsmu.RLock()
	defer sa.statsmu.RUnlock()
	return sa.stats
}

// updateStats updates statistics with a function
func (sa *SecurityAuditor) updateStats(fn func(*SecurityStats)) {
	sa.statsmu.Lock()
	defer sa.statsmu.Unlock()
	fn(&sa.stats)
	sa.stats.LastUpdate = time.Now()
}

// RegisterAlertHandler registers a handler for security alerts
func (sa *SecurityAuditor) RegisterAlertHandler(handler func(SecurityEvent)) {
	sa.eventsMu.Lock()
	defer sa.eventsMu.Unlock()
	sa.alertHandlers = append(sa.alertHandlers, handler)
}

// triggerAlert triggers all registered alert handlers
func (sa *SecurityAuditor) triggerAlert(event SecurityEvent) {
	sa.eventsMu.RLock()
	handlers := make([]func(SecurityEvent), len(sa.alertHandlers))
	copy(handlers, sa.alertHandlers)
	sa.eventsMu.RUnlock()

	for _, handler := range handlers {
		go handler(event)
	}
}

// ExportEvents exports events to JSON
func (sa *SecurityAuditor) ExportEvents() ([]byte, error) {
	sa.eventsMu.RLock()
	defer sa.eventsMu.RUnlock()

	data, err := json.MarshalIndent(sa.events, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal events: %w", err)
	}

	return data, nil
}

// ClearOldEvents removes events older than the specified duration
func (sa *SecurityAuditor) ClearOldEvents(maxAge time.Duration) int {
	sa.eventsMu.Lock()
	defer sa.eventsMu.Unlock()

	cutoff := time.Now().Add(-maxAge)
	newEvents := []SecurityEvent{}

	for _, event := range sa.events {
		if event.Timestamp.After(cutoff) {
			newEvents = append(newEvents, event)
		}
	}

	removed := len(sa.events) - len(newEvents)
	sa.events = newEvents

	sa.logger.Info("Cleared old security events",
		zap.Int("removed", removed),
		zap.Int("remaining", len(newEvents)))

	return removed
}
