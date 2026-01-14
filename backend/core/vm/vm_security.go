package vm

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SecurityProfile represents a VM security profile
type SecurityProfile struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Description       string            `json:"description"`
	SecureBoot        bool              `json:"secure_boot"`
	TPMEnabled        bool              `json:"tpm_enabled"`
	EncryptionEnabled bool              `json:"encryption_enabled"`
	EncryptionType    string            `json:"encryption_type,omitempty"`
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
	Tags              []string          `json:"tags,omitempty"`
	Metadata          map[string]string `json:"metadata,omitempty"`
}

// Certificate represents a certificate
type Certificate struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Type        string            `json:"type"`
	Fingerprint string            `json:"fingerprint"`
	NotBefore   time.Time         `json:"not_before"`
	NotAfter    time.Time         `json:"not_after"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Path        string            `json:"path"`
	Tags        []string          `json:"tags,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// VMSecurityManager manages VM security
type VMSecurityManager struct {
	profiles      map[string]*SecurityProfile
	profilesMutex sync.RWMutex
	certificates  map[string]*Certificate
	certsMutex    sync.RWMutex
	vmManager     *VMManager
	certsDir      string
}

// NewVMSecurityManager creates a new VM security manager
func NewVMSecurityManager(vmManager *VMManager, certsDir string) *VMSecurityManager {
	return &VMSecurityManager{
		profiles:     make(map[string]*SecurityProfile),
		certificates: make(map[string]*Certificate),
		vmManager:    vmManager,
		certsDir:     certsDir,
	}
}

// CreateSecurityProfile creates a new security profile
func (m *VMSecurityManager) CreateSecurityProfile(ctx context.Context, name, description string, secureBoot, tpmEnabled, encryptionEnabled bool, encryptionType string, tags []string, metadata map[string]string) (*SecurityProfile, error) {
	// Validate encryption type
	if encryptionEnabled && encryptionType == "" {
		return nil, fmt.Errorf("encryption type is required when encryption is enabled")
	}

	// Generate profile ID
	profileID := uuid.New().String()

	// Create profile
	profile := &SecurityProfile{
		ID:                profileID,
		Name:              name,
		Description:       description,
		SecureBoot:        secureBoot,
		TPMEnabled:        tpmEnabled,
		EncryptionEnabled: encryptionEnabled,
		EncryptionType:    encryptionType,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		Tags:              tags,
		Metadata:          metadata,
	}

	// Store profile
	m.profilesMutex.Lock()
	m.profiles[profileID] = profile
	m.profilesMutex.Unlock()

	log.Printf("Created security profile %s (%s)", profile.Name, profile.ID)

	return profile, nil
}

// GetSecurityProfile returns a security profile by ID
func (m *VMSecurityManager) GetSecurityProfile(profileID string) (*SecurityProfile, error) {
	m.profilesMutex.RLock()
	defer m.profilesMutex.RUnlock()

	profile, exists := m.profiles[profileID]
	if !exists {
		return nil, fmt.Errorf("security profile %s not found", profileID)
	}

	return profile, nil
}

// ListSecurityProfiles returns all security profiles
func (m *VMSecurityManager) ListSecurityProfiles() []*SecurityProfile {
	m.profilesMutex.RLock()
	defer m.profilesMutex.RUnlock()

	profiles := make([]*SecurityProfile, 0, len(m.profiles))
	for _, profile := range m.profiles {
		profiles = append(profiles, profile)
	}

	return profiles
}

// UpdateSecurityProfile updates a security profile
func (m *VMSecurityManager) UpdateSecurityProfile(ctx context.Context, profileID, name, description string, secureBoot, tpmEnabled, encryptionEnabled bool, encryptionType string, tags []string, metadata map[string]string) (*SecurityProfile, error) {
	// Get the profile
	m.profilesMutex.Lock()
	defer m.profilesMutex.Unlock()

	profile, exists := m.profiles[profileID]
	if !exists {
		return nil, fmt.Errorf("security profile %s not found", profileID)
	}

	// Validate encryption type
	if encryptionEnabled && encryptionType == "" {
		return nil, fmt.Errorf("encryption type is required when encryption is enabled")
	}

	// Update profile
	if name != "" {
		profile.Name = name
	}

	if description != "" {
		profile.Description = description
	}

	profile.SecureBoot = secureBoot
	profile.TPMEnabled = tpmEnabled
	profile.EncryptionEnabled = encryptionEnabled

	if encryptionEnabled {
		profile.EncryptionType = encryptionType
	} else {
		profile.EncryptionType = ""
	}

	if tags != nil {
		profile.Tags = tags
	}

	if metadata != nil {
		profile.Metadata = metadata
	}

	profile.UpdatedAt = time.Now()

	log.Printf("Updated security profile %s (%s)", profile.Name, profile.ID)

	return profile, nil
}

// DeleteSecurityProfile deletes a security profile
func (m *VMSecurityManager) DeleteSecurityProfile(ctx context.Context, profileID string) error {
	// Get the profile
	m.profilesMutex.Lock()
	defer m.profilesMutex.Unlock()

	profile, exists := m.profiles[profileID]
	if !exists {
		return fmt.Errorf("security profile %s not found", profileID)
	}

	// Delete profile
	delete(m.profiles, profileID)

	log.Printf("Deleted security profile %s (%s)", profile.Name, profile.ID)

	return nil
}

// CreateCertificate creates a new certificate
func (m *VMSecurityManager) CreateCertificate(ctx context.Context, name, description, certType string, validityDays int, tags []string, metadata map[string]string) (*Certificate, error) {
	// Validate certificate type
	switch certType {
	case "ca", "server", "client":
		// Valid certificate type
	default:
		return nil, fmt.Errorf("invalid certificate type: %s", certType)
	}

	// Validate validity days
	if validityDays <= 0 {
		return nil, fmt.Errorf("validity days must be greater than 0")
	}

	// Generate certificate ID
	certID := uuid.New().String()

	// Create certificates directory if it doesn't exist
	if err := os.MkdirAll(m.certsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create certificates directory: %w", err)
	}

	// Generate certificate paths
	certPath := filepath.Join(m.certsDir, certID+".crt")
	keyPath := filepath.Join(m.certsDir, certID+".key")

	// Generate certificate
	notBefore := time.Now()
	notAfter := notBefore.Add(time.Duration(validityDays) * 24 * time.Hour)

	// Generate key pair
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	// Create certificate template
	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %w", err)
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"NovaCron"},
			CommonName:   name,
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	// Set certificate type-specific options
	switch certType {
	case "ca":
		template.IsCA = true
		template.KeyUsage |= x509.KeyUsageCertSign
	case "server":
		template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	case "client":
		template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	}

	// Create certificate
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	// Save certificate
	certFile, err := os.Create(certPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate file: %w", err)
	}
	defer certFile.Close()

	if err := pem.Encode(certFile, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return nil, fmt.Errorf("failed to encode certificate: %w", err)
	}

	// Save private key
	keyFile, err := os.Create(keyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create key file: %w", err)
	}
	defer keyFile.Close()

	if err := pem.Encode(keyFile, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)}); err != nil {
		return nil, fmt.Errorf("failed to encode private key: %w", err)
	}

	// Calculate fingerprint
	fingerprint := fmt.Sprintf("%x", serialNumber)

	// Create certificate object
	cert := &Certificate{
		ID:          certID,
		Name:        name,
		Description: description,
		Type:        certType,
		Fingerprint: fingerprint,
		NotBefore:   notBefore,
		NotAfter:    notAfter,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Path:        certPath,
		Tags:        tags,
		Metadata:    metadata,
	}

	// Store certificate
	m.certsMutex.Lock()
	m.certificates[certID] = cert
	m.certsMutex.Unlock()

	log.Printf("Created certificate %s (%s)", cert.Name, cert.ID)

	return cert, nil
}

// GetCertificate returns a certificate by ID
func (m *VMSecurityManager) GetCertificate(certID string) (*Certificate, error) {
	m.certsMutex.RLock()
	defer m.certsMutex.RUnlock()

	cert, exists := m.certificates[certID]
	if !exists {
		return nil, fmt.Errorf("certificate %s not found", certID)
	}

	return cert, nil
}

// ListCertificates returns all certificates
func (m *VMSecurityManager) ListCertificates() []*Certificate {
	m.certsMutex.RLock()
	defer m.certsMutex.RUnlock()

	certs := make([]*Certificate, 0, len(m.certificates))
	for _, cert := range m.certificates {
		certs = append(certs, cert)
	}

	return certs
}

// DeleteCertificate deletes a certificate
func (m *VMSecurityManager) DeleteCertificate(ctx context.Context, certID string) error {
	// Get the certificate
	m.certsMutex.Lock()
	defer m.certsMutex.Unlock()

	cert, exists := m.certificates[certID]
	if !exists {
		return fmt.Errorf("certificate %s not found", certID)
	}

	// Delete certificate files
	certPath := cert.Path
	keyPath := strings.TrimSuffix(certPath, ".crt") + ".key"

	if err := os.Remove(certPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete certificate file: %w", err)
	}

	if err := os.Remove(keyPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete key file: %w", err)
	}

	// Delete certificate
	delete(m.certificates, certID)

	log.Printf("Deleted certificate %s (%s)", cert.Name, cert.ID)

	return nil
}

// ApplySecurityProfile applies a security profile to a VM
func (m *VMSecurityManager) ApplySecurityProfile(ctx context.Context, vmID, profileID string) error {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	// Get the profile
	m.profilesMutex.RLock()
	profile, exists := m.profiles[profileID]
	if !exists {
		m.profilesMutex.RUnlock()
		return fmt.Errorf("security profile %s not found", profileID)
	}
	m.profilesMutex.RUnlock()

	// Check if VM is running
	if vm.State() == StateRunning {
		return fmt.Errorf("VM must be stopped to apply security profile")
	}

	// Apply security profile
	log.Printf("Applying security profile %s to VM %s", profile.Name, vm.ID())

	// In a real implementation, this would configure the VM's security settings
	// For example, enabling secure boot, TPM, encryption, etc.

	return nil
}

// GetVMSecurityInfo returns security information for a VM
func (m *VMSecurityManager) GetVMSecurityInfo(ctx context.Context, vmID string) (map[string]interface{}, error) {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	// Get VM security information
	info := map[string]interface{}{
		"vm_id":   vm.ID(),
		"vm_name": vm.Name(),
	}

	// In a real implementation, this would get the VM's security settings
	// For example, secure boot status, TPM status, encryption status, etc.

	return info, nil
}
