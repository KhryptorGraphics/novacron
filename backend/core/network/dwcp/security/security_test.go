package security

import (
	"crypto/tls"
	"crypto/x509"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestTLSManager(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	t.Run("Initialize with defaults", func(t *testing.T) {
		config := SecurityConfig{
			TLSEnabled:    true,
			SessionCacheSize: 128,
		}

		tm, err := NewTLSManager(config, logger)
		if err != nil {
			t.Fatalf("Failed to create TLS manager: %v", err)
		}

		tlsConfig := tm.GetTLSConfig()
		if tlsConfig.MinVersion != tls.VersionTLS13 {
			t.Errorf("Expected TLS 1.3, got version %d", tlsConfig.MinVersion)
		}

		if len(tlsConfig.CipherSuites) == 0 {
			t.Error("No cipher suites configured")
		}
	})

	t.Run("Configure mTLS", func(t *testing.T) {
		config := SecurityConfig{
			TLSEnabled:        true,
			MTLSEnabled:       true,
			RequireClientCert: true,
		}

		tm, err := NewTLSManager(config, logger)
		if err != nil {
			t.Fatalf("Failed to create TLS manager: %v", err)
		}

		tlsConfig := tm.GetTLSConfig()
		if tlsConfig.ClientAuth != tls.RequireAndVerifyClientCert {
			t.Error("mTLS not properly configured")
		}
	})
}

func TestCertificateManager(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	t.Run("Generate self-signed certificate", func(t *testing.T) {
		config := CertificateManagerConfig{
			CertPath:    "/tmp/test-cert.pem",
			KeyPath:     "/tmp/test-key.pem",
			AutoRenew:   false,
			RenewBefore: 30 * 24 * time.Hour,
		}

		cm, err := NewCertificateManager(config, logger)
		if err == nil {
			// Certificate files don't exist yet, so this should fail
			// which is expected in test environment
			defer cm.Stop()
		}

		// Test would require actual certificate files
		// This is a placeholder for integration testing
	})

	t.Run("Certificate revocation check", func(t *testing.T) {
		config := CertificateManagerConfig{
			AutoRenew:   false,
			RenewBefore: 30 * 24 * time.Hour,
		}

		cm := &CertificateManager{
			logger:       logger,
			revokedCerts: make(map[string]time.Time),
			stopChan:     make(chan struct{}),
		}

		// Create a dummy certificate
		cert := &x509.Certificate{
			SerialNumber: big.NewInt(12345),
		}

		// Should not be revoked initially
		if cm.IsRevoked(cert) {
			t.Error("Certificate should not be revoked")
		}

		// Revoke certificate
		cm.RevokeCertificate(cert)

		// Should now be revoked
		if !cm.IsRevoked(cert) {
			t.Error("Certificate should be revoked")
		}
	})
}

func TestDataEncryptor(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	t.Run("Encrypt and decrypt with Argon2", func(t *testing.T) {
		password := "test-password-123"
		salt, err := GenerateSalt()
		if err != nil {
			t.Fatalf("Failed to generate salt: %v", err)
		}

		config := DefaultEncryptionConfig()
		encryptor, err := NewDataEncryptor(password, salt, config, logger)
		if err != nil {
			t.Fatalf("Failed to create encryptor: %v", err)
		}

		plaintext := []byte("This is sensitive data that needs encryption")

		// Encrypt
		ciphertext, err := encryptor.Encrypt(plaintext)
		if err != nil {
			t.Fatalf("Encryption failed: %v", err)
		}

		if len(ciphertext) <= len(plaintext) {
			t.Error("Ciphertext should be longer than plaintext (includes nonce)")
		}

		// Decrypt
		decrypted, err := encryptor.Decrypt(ciphertext)
		if err != nil {
			t.Fatalf("Decryption failed: %v", err)
		}

		if string(decrypted) != string(plaintext) {
			t.Errorf("Decrypted text doesn't match: got %s, want %s", decrypted, plaintext)
		}
	})

	t.Run("Encrypt and decrypt strings", func(t *testing.T) {
		password := "test-password-123"
		salt, _ := GenerateSalt()
		config := DefaultEncryptionConfig()
		encryptor, _ := NewDataEncryptor(password, salt, config, logger)

		plaintext := "Secret message"

		// Encrypt
		encrypted, err := encryptor.EncryptString(plaintext)
		if err != nil {
			t.Fatalf("String encryption failed: %v", err)
		}

		// Decrypt
		decrypted, err := encryptor.DecryptString(encrypted)
		if err != nil {
			t.Fatalf("String decryption failed: %v", err)
		}

		if decrypted != plaintext {
			t.Errorf("Decrypted string doesn't match: got %s, want %s", decrypted, plaintext)
		}
	})

	t.Run("Key rotation", func(t *testing.T) {
		password := "old-password"
		salt, _ := GenerateSalt()
		config := DefaultEncryptionConfig()
		encryptor, _ := NewDataEncryptor(password, salt, config, logger)

		// Rotate to new password
		newPassword := "new-password"
		newSalt, _ := GenerateSalt()
		err := encryptor.RotateKey(newPassword, newSalt)
		if err != nil {
			t.Fatalf("Key rotation failed: %v", err)
		}

		// Should be able to encrypt with new key
		plaintext := []byte("test data")
		ciphertext, err := encryptor.Encrypt(plaintext)
		if err != nil {
			t.Fatalf("Encryption after rotation failed: %v", err)
		}

		// Should be able to decrypt with new key
		decrypted, err := encryptor.Decrypt(ciphertext)
		if err != nil {
			t.Fatalf("Decryption after rotation failed: %v", err)
		}

		if string(decrypted) != string(plaintext) {
			t.Error("Decryption after rotation failed")
		}
	})
}

func TestSecurityAuditor(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	t.Run("Record and retrieve events", func(t *testing.T) {
		auditor := NewSecurityAuditor(logger, 100)

		// Record some events
		auditor.AuditAuthFailure("testuser", "192.168.1.1", "invalid password")
		auditor.AuditCustomEvent("test_event", "info", "test description", nil)

		// Retrieve events
		events := auditor.GetEvents(10)
		if len(events) != 2 {
			t.Errorf("Expected 2 events, got %d", len(events))
		}

		// Check stats
		stats := auditor.GetStats()
		if stats.AuthFailures != 1 {
			t.Errorf("Expected 1 auth failure, got %d", stats.AuthFailures)
		}
	})

	t.Run("Filter events by type", func(t *testing.T) {
		auditor := NewSecurityAuditor(logger, 100)

		auditor.AuditAuthFailure("user1", "192.168.1.1", "reason1")
		auditor.AuditAuthFailure("user2", "192.168.1.2", "reason2")
		auditor.AuditCustomEvent("custom", "info", "description", nil)

		authEvents := auditor.GetEventsByType("auth_failure", 10)
		if len(authEvents) != 2 {
			t.Errorf("Expected 2 auth_failure events, got %d", len(authEvents))
		}
	})

	t.Run("Alert handlers", func(t *testing.T) {
		auditor := NewSecurityAuditor(logger, 100)

		alertCalled := false
		auditor.RegisterAlertHandler(func(event SecurityEvent) {
			alertCalled = true
		})

		// Trigger an error event (should call alert handler)
		auditor.AuditCustomEvent("critical_event", "error", "critical issue", nil)

		// Give handler time to execute (it runs in goroutine)
		time.Sleep(100 * time.Millisecond)

		if !alertCalled {
			t.Error("Alert handler was not called")
		}
	})

	t.Run("Clear old events", func(t *testing.T) {
		auditor := NewSecurityAuditor(logger, 100)

		// Add some events
		for i := 0; i < 5; i++ {
			auditor.AuditCustomEvent("test", "info", "test", nil)
		}

		// Clear events older than 1 hour (should clear none since they're new)
		removed := auditor.ClearOldEvents(1 * time.Hour)
		if removed != 0 {
			t.Errorf("Expected 0 removed, got %d", removed)
		}

		// Clear events older than 0 seconds (should clear all)
		removed = auditor.ClearOldEvents(0)
		if removed != 5 {
			t.Errorf("Expected 5 removed, got %d", removed)
		}
	})
}

func TestVaultIntegration(t *testing.T) {
	// This test requires a running Vault instance
	// Skip in normal test runs
	t.Skip("Vault integration test requires running Vault instance")

	logger, _ := zap.NewDevelopment()

	config := VaultConfig{
		Address:  "http://localhost:8200",
		Token:    "dev-token",
		PKIPath:  "pki",
		Role:     "dwcp-role",
		TokenTTL: 1 * time.Hour,
	}

	vc, err := NewVaultClient(config, logger)
	if err != nil {
		t.Fatalf("Failed to create Vault client: %v", err)
	}

	// Test certificate issuance
	cert, x509Cert, err := vc.IssueCertificate("test.example.com", "24h", nil, nil)
	if err != nil {
		t.Fatalf("Failed to issue certificate: %v", err)
	}

	if cert == nil || x509Cert == nil {
		t.Error("Certificate issuance returned nil")
	}
}

func TestACMEIntegration(t *testing.T) {
	// This test requires DNS and HTTP challenge setup
	// Skip in normal test runs
	t.Skip("ACME integration test requires DNS/HTTP challenge setup")

	logger, _ := zap.NewDevelopment()

	config := ACMEConfig{
		Domains:     []string{"test.example.com"},
		Email:       "admin@example.com",
		CacheDir:    "/tmp/acme-test",
		RenewBefore: 30 * 24 * time.Hour,
		UseStaging:  true, // Use Let's Encrypt staging
	}

	am, err := NewACMEManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create ACME manager: %v", err)
	}
	defer am.Stop()

	// Test certificate obtainment
	err = am.ObtainCertificate("test.example.com")
	if err != nil {
		t.Fatalf("Failed to obtain certificate: %v", err)
	}
}

func BenchmarkEncryption(b *testing.B) {
	password := "benchmark-password"
	salt, _ := GenerateSalt()
	config := DefaultEncryptionConfig()
	encryptor, _ := NewDataEncryptor(password, salt, config, nil)

	data := make([]byte, 1024*1024) // 1MB
	for i := range data {
		data[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = encryptor.Encrypt(data)
	}
}

func BenchmarkDecryption(b *testing.B) {
	password := "benchmark-password"
	salt, _ := GenerateSalt()
	config := DefaultEncryptionConfig()
	encryptor, _ := NewDataEncryptor(password, salt, config, nil)

	data := make([]byte, 1024*1024) // 1MB
	ciphertext, _ := encryptor.Encrypt(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = encryptor.Decrypt(ciphertext)
	}
}
