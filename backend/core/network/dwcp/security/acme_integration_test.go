package security

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestParseCachedCertificateExpiry(t *testing.T) {
	notAfter := time.Now().Add(48 * time.Hour).UTC().Truncate(time.Second)
	cacheData := testAutocertCachePEM(t, notAfter)

	got, err := parseCachedCertificateExpiry(cacheData)
	if err != nil {
		t.Fatalf("parseCachedCertificateExpiry() error = %v", err)
	}
	if !got.Equal(notAfter) {
		t.Fatalf("expiry = %s, want %s", got, notAfter)
	}
}

func TestCheckCertificateExpiryParsesCachedPEM(t *testing.T) {
	cacheDir := t.TempDir()
	domain := "example.test"
	notAfter := time.Now().Add(72 * time.Hour).UTC().Truncate(time.Second)
	if err := os.WriteFile(filepath.Join(cacheDir, domain), testAutocertCachePEM(t, notAfter), 0600); err != nil {
		t.Fatalf("write cache file: %v", err)
	}

	manager, err := NewACMEManager(ACMEConfig{
		Domains:  []string{domain},
		Email:    "ops@example.test",
		CacheDir: cacheDir,
	}, zap.NewNop())
	if err != nil {
		t.Fatalf("NewACMEManager() error = %v", err)
	}
	defer manager.Stop()

	expiry, err := manager.CheckCertificateExpiry()
	if err != nil {
		t.Fatalf("CheckCertificateExpiry() error = %v", err)
	}
	if !expiry[domain].Equal(notAfter) {
		t.Fatalf("expiry[%q] = %s, want %s", domain, expiry[domain], notAfter)
	}
}

func testAutocertCachePEM(t *testing.T, notAfter time.Time) []byte {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "example.test",
		},
		DNSNames:              []string{"example.test"},
		NotBefore:             time.Now().Add(-time.Hour).UTC(),
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create cert: %v", err)
	}

	keyBlock := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	})
	certBlock := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDER,
	})

	return append(keyBlock, certBlock...)
}
