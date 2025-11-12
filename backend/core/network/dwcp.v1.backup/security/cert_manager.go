package security

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"go.uber.org/zap"
)

// CertificateManager manages certificate lifecycle
type CertificateManager struct {
	certPath      string
	keyPath       string
	caPath        string
	autoRenew     bool
	renewBefore   time.Duration
	watcher       *fsnotify.Watcher
	logger        *zap.Logger
	mu            sync.RWMutex
	current       *tls.Certificate
	next          *tls.Certificate
	rotating      atomic.Bool
	revokedCerts  map[string]time.Time
	renewCallback func() error
	stopChan      chan struct{}
	stopped       atomic.Bool
}

// CertificateManagerConfig holds certificate manager configuration
type CertificateManagerConfig struct {
	CertPath    string
	KeyPath     string
	CAPath      string
	AutoRenew   bool
	RenewBefore time.Duration
}

// NewCertificateManager creates a new certificate manager
func NewCertificateManager(config CertificateManagerConfig, logger *zap.Logger) (*CertificateManager, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	cm := &CertificateManager{
		certPath:     config.CertPath,
		keyPath:      config.KeyPath,
		caPath:       config.CAPath,
		autoRenew:    config.AutoRenew,
		renewBefore:  config.RenewBefore,
		logger:       logger,
		revokedCerts: make(map[string]time.Time),
		stopChan:     make(chan struct{}),
	}

	// Load initial certificate
	if err := cm.loadCertificate(); err != nil {
		return nil, fmt.Errorf("failed to load initial certificate: %w", err)
	}

	// Start file watcher
	if err := cm.startWatcher(); err != nil {
		logger.Warn("Failed to start certificate file watcher", zap.Error(err))
	}

	// Start auto-renewal if enabled
	if config.AutoRenew {
		go cm.autoRenewLoop()
		logger.Info("Auto-renewal enabled",
			zap.Duration("renew_before", config.RenewBefore))
	}

	logger.Info("Certificate manager initialized",
		zap.String("cert_path", config.CertPath),
		zap.Bool("auto_renew", config.AutoRenew))

	return cm, nil
}

// loadCertificate loads certificate from disk
func (cm *CertificateManager) loadCertificate() error {
	cert, err := tls.LoadX509KeyPair(cm.certPath, cm.keyPath)
	if err != nil {
		return fmt.Errorf("failed to load certificate pair: %w", err)
	}

	// Parse certificate to get expiration info
	if len(cert.Certificate) > 0 {
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err == nil {
			cm.logger.Info("Certificate loaded",
				zap.String("subject", x509Cert.Subject.CommonName),
				zap.Time("not_after", x509Cert.NotAfter),
				zap.Duration("valid_for", time.Until(x509Cert.NotAfter)))
		}
	}

	cm.mu.Lock()
	cm.current = &cert
	cm.mu.Unlock()

	return nil
}

// GetCertificate returns the current certificate
func (cm *CertificateManager) GetCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// If rotating and next certificate is available, return it
	if cm.rotating.Load() && cm.next != nil {
		return cm.next, nil
	}

	if cm.current == nil {
		return nil, errors.New("no certificate available")
	}

	return cm.current, nil
}

// autoRenewLoop runs the auto-renewal check loop
func (cm *CertificateManager) autoRenewLoop() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := cm.checkAndRenew(); err != nil {
				cm.logger.Error("Auto-renewal check failed", zap.Error(err))
			}
		case <-cm.stopChan:
			return
		}
	}
}

// checkAndRenew checks certificate expiration and renews if necessary
func (cm *CertificateManager) checkAndRenew() error {
	cm.mu.RLock()
	current := cm.current
	cm.mu.RUnlock()

	if current == nil {
		return errors.New("no current certificate")
	}

	// Parse certificate
	x509Cert, err := x509.ParseCertificate(current.Certificate[0])
	if err != nil {
		return fmt.Errorf("failed to parse certificate: %w", err)
	}

	// Check if renewal is needed
	timeUntilExpiry := time.Until(x509Cert.NotAfter)
	if timeUntilExpiry > cm.renewBefore {
		cm.logger.Debug("Certificate renewal not needed",
			zap.Duration("time_until_expiry", timeUntilExpiry),
			zap.Duration("renew_before", cm.renewBefore))
		return nil
	}

	cm.logger.Info("Certificate renewal needed",
		zap.Duration("time_until_expiry", timeUntilExpiry),
		zap.Time("expires_at", x509Cert.NotAfter))

	// Perform renewal
	return cm.RenewCertificate(x509Cert)
}

// RenewCertificate renews the certificate
func (cm *CertificateManager) RenewCertificate(oldCert *x509.Certificate) error {
	cm.logger.Info("Starting certificate renewal",
		zap.String("subject", oldCert.Subject.CommonName))

	// Call custom renewal callback if set
	if cm.renewCallback != nil {
		if err := cm.renewCallback(); err != nil {
			return fmt.Errorf("renewal callback failed: %w", err)
		}
		cm.logger.Info("Custom renewal callback completed")
	} else {
		// Generate self-signed certificate as fallback
		if err := cm.generateSelfSigned(oldCert.Subject); err != nil {
			return fmt.Errorf("failed to generate self-signed certificate: %w", err)
		}
	}

	// Reload certificate
	if err := cm.loadCertificate(); err != nil {
		return fmt.Errorf("failed to reload renewed certificate: %w", err)
	}

	cm.logger.Info("Certificate renewal completed successfully")
	return nil
}

// generateSelfSigned generates a self-signed certificate
func (cm *CertificateManager) generateSelfSigned(subject pkix.Name) error {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return fmt.Errorf("failed to generate private key: %w", err)
	}

	// Create certificate template
	serialNumber, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject:      subject,
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour), // 1 year
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}

	// Create self-signed certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return fmt.Errorf("failed to create certificate: %w", err)
	}

	// Save certificate
	certOut, err := os.Create(cm.certPath)
	if err != nil {
		return fmt.Errorf("failed to create cert file: %w", err)
	}
	defer certOut.Close()
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: certDER})

	// Save private key
	keyOut, err := os.OpenFile(cm.keyPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("failed to create key file: %w", err)
	}
	defer keyOut.Close()
	pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	cm.logger.Info("Self-signed certificate generated",
		zap.String("subject", subject.CommonName))

	return nil
}

// startWatcher starts watching certificate files for changes
func (cm *CertificateManager) startWatcher() error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create file watcher: %w", err)
	}

	cm.watcher = watcher

	// Add certificate and key files
	if err := watcher.Add(cm.certPath); err != nil {
		cm.logger.Warn("Failed to watch certificate file", zap.Error(err))
	}
	if err := watcher.Add(cm.keyPath); err != nil {
		cm.logger.Warn("Failed to watch key file", zap.Error(err))
	}

	// Start watching goroutine
	go cm.watchLoop()

	return nil
}

// watchLoop processes file system events
func (cm *CertificateManager) watchLoop() {
	for {
		select {
		case event, ok := <-cm.watcher.Events:
			if !ok {
				return
			}
			if event.Op&fsnotify.Write == fsnotify.Write {
				cm.logger.Info("Certificate file changed, reloading",
					zap.String("file", event.Name))
				if err := cm.loadCertificate(); err != nil {
					cm.logger.Error("Failed to reload certificate after file change",
						zap.Error(err))
				}
			}
		case err, ok := <-cm.watcher.Errors:
			if !ok {
				return
			}
			cm.logger.Error("File watcher error", zap.Error(err))
		case <-cm.stopChan:
			return
		}
	}
}

// RotateCertificate performs zero-downtime certificate rotation
func (cm *CertificateManager) RotateCertificate(newCert *tls.Certificate) error {
	if cm.rotating.Load() {
		return errors.New("rotation already in progress")
	}

	cm.rotating.Store(true)
	defer cm.rotating.Store(false)

	cm.logger.Info("Starting certificate rotation")

	cm.mu.Lock()
	cm.next = newCert
	cm.mu.Unlock()

	// Wait for active connections to transition (grace period)
	time.Sleep(30 * time.Second)

	cm.mu.Lock()
	cm.current = cm.next
	cm.next = nil
	cm.mu.Unlock()

	cm.logger.Info("Certificate rotation completed")
	return nil
}

// IsRevoked checks if a certificate is revoked
func (cm *CertificateManager) IsRevoked(cert *x509.Certificate) bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	serial := cert.SerialNumber.String()
	_, revoked := cm.revokedCerts[serial]
	return revoked
}

// RevokeCertificate adds a certificate to the revocation list
func (cm *CertificateManager) RevokeCertificate(cert *x509.Certificate) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	serial := cert.SerialNumber.String()
	cm.revokedCerts[serial] = time.Now()
	cm.logger.Info("Certificate revoked",
		zap.String("serial", serial),
		zap.String("subject", cert.Subject.CommonName))
}

// SetRenewCallback sets a custom renewal callback
func (cm *CertificateManager) SetRenewCallback(callback func() error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.renewCallback = callback
}

// GetExpirationTime returns the current certificate expiration time
func (cm *CertificateManager) GetExpirationTime() (time.Time, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if cm.current == nil {
		return time.Time{}, errors.New("no current certificate")
	}

	if len(cm.current.Certificate) == 0 {
		return time.Time{}, errors.New("invalid certificate")
	}

	x509Cert, err := x509.ParseCertificate(cm.current.Certificate[0])
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to parse certificate: %w", err)
	}

	return x509Cert.NotAfter, nil
}

// Stop stops the certificate manager
func (cm *CertificateManager) Stop() {
	if cm.stopped.Swap(true) {
		return
	}

	close(cm.stopChan)

	if cm.watcher != nil {
		cm.watcher.Close()
	}

	cm.logger.Info("Certificate manager stopped")
}
