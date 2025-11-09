// Package attestation implements attestation and verification
package attestation

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// AttestationType represents the type of attestation
type AttestationType string

const (
	AttestationRemote       AttestationType = "remote"
	AttestationMeasuredBoot AttestationType = "measured_boot"
	AttestationRuntime      AttestationType = "runtime"
	AttestationTPM          AttestationType = "tpm"
)

// AttestationStatus represents attestation verification status
type AttestationStatus string

const (
	StatusPending  AttestationStatus = "pending"
	StatusVerified AttestationStatus = "verified"
	StatusFailed   AttestationStatus = "failed"
	StatusExpired  AttestationStatus = "expired"
)

// Quote represents an attestation quote
type Quote struct {
	ID           string
	Type         AttestationType
	Nonce        []byte
	Measurement  []byte
	Signature    []byte
	Timestamp    time.Time
	PCRValues    map[int][]byte // TPM Platform Configuration Registers
	Metadata     map[string]interface{}
}

// Report represents an attestation report
type Report struct {
	ID            string
	EntityID      string
	EntityType    string
	Type          AttestationType
	Quote         *Quote
	Status        AttestationStatus
	TrustLevel    float64
	Verified      bool
	VerifiedAt    time.Time
	ExpiresAt     time.Time
	Violations    []string
	Metadata      map[string]interface{}
}

// Policy represents an attestation policy
type Policy struct {
	ID                  string
	Name                string
	Enabled             bool
	RequiredMeasurements []string
	AllowedPCRValues    map[int][]byte
	MinTrustLevel       float64
	MaxAge              time.Duration
	Metadata            map[string]interface{}
}

// Verifier implements attestation verification
type Verifier struct {
	policies            map[string]*Policy
	reports             map[string]*Report
	quotes              map[string]*Quote
	tpmEnabled          bool
	measuredBoot        bool
	runtimeIntegrity    bool
	attestationInterval time.Duration
	mu                  sync.RWMutex
	totalVerifications  int64
	successfulVerifications int64
	failedVerifications int64
}

// NewVerifier creates a new attestation verifier
func NewVerifier(tpmEnabled, measuredBoot, runtimeIntegrity bool, interval time.Duration) *Verifier {
	return &Verifier{
		policies:            make(map[string]*Policy),
		reports:             make(map[string]*Report),
		quotes:              make(map[string]*Quote),
		tpmEnabled:          tpmEnabled,
		measuredBoot:        measuredBoot,
		runtimeIntegrity:    runtimeIntegrity,
		attestationInterval: interval,
	}
}

// AddPolicy adds an attestation policy
func (v *Verifier) AddPolicy(policy *Policy) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if policy.ID == "" {
		return fmt.Errorf("policy ID is required")
	}

	v.policies[policy.ID] = policy
	return nil
}

// RemovePolicy removes an attestation policy
func (v *Verifier) RemovePolicy(policyID string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	delete(v.policies, policyID)
}

// GenerateQuote generates an attestation quote
func (v *Verifier) GenerateQuote(entityID string, attestationType AttestationType, nonce []byte) (*Quote, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Generate measurement
	measurement := v.generateMeasurement(entityID, attestationType)

	// Generate PCR values if TPM enabled
	var pcrValues map[int][]byte
	if v.tpmEnabled && attestationType == AttestationTPM {
		pcrValues = v.generatePCRValues()
	}

	// Sign the quote
	signature := v.signQuote(measurement, nonce)

	quote := &Quote{
		ID:          generateQuoteID(),
		Type:        attestationType,
		Nonce:       nonce,
		Measurement: measurement,
		Signature:   signature,
		Timestamp:   time.Now(),
		PCRValues:   pcrValues,
		Metadata:    make(map[string]interface{}),
	}

	v.quotes[quote.ID] = quote
	return quote, nil
}

// generateMeasurement generates a measurement hash
func (v *Verifier) generateMeasurement(entityID string, attestationType AttestationType) []byte {
	h := sha256.New()
	h.Write([]byte(entityID))
	h.Write([]byte(attestationType))
	h.Write([]byte(time.Now().String()))

	// Add boot measurements if measured boot enabled
	if v.measuredBoot && attestationType == AttestationMeasuredBoot {
		h.Write([]byte("measured-boot-hash"))
	}

	// Add runtime measurements if runtime integrity enabled
	if v.runtimeIntegrity && attestationType == AttestationRuntime {
		h.Write([]byte("runtime-integrity-hash"))
	}

	return h.Sum(nil)
}

// generatePCRValues generates TPM PCR values
func (v *Verifier) generatePCRValues() map[int][]byte {
	pcrValues := make(map[int][]byte)

	// PCR 0-7: BIOS and firmware
	for i := 0; i < 8; i++ {
		pcr := make([]byte, 32)
		rand.Read(pcr)
		pcrValues[i] = pcr
	}

	// PCR 8-15: OS and bootloader
	for i := 8; i < 16; i++ {
		pcr := make([]byte, 32)
		rand.Read(pcr)
		pcrValues[i] = pcr
	}

	return pcrValues
}

// signQuote signs an attestation quote
func (v *Verifier) signQuote(measurement, nonce []byte) []byte {
	h := sha256.New()
	h.Write(measurement)
	h.Write(nonce)

	// Simulate signing (in production, would use actual signing key)
	signature := make([]byte, 64)
	copy(signature[:32], h.Sum(nil))
	rand.Read(signature[32:])

	return signature
}

// VerifyQuote verifies an attestation quote
func (v *Verifier) VerifyQuote(quote *Quote, entityID, entityType string) (*Report, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.totalVerifications++

	report := &Report{
		ID:         generateReportID(),
		EntityID:   entityID,
		EntityType: entityType,
		Type:       quote.Type,
		Quote:      quote,
		Status:     StatusPending,
		TrustLevel: 0,
		Verified:   false,
		VerifiedAt: time.Now(),
		ExpiresAt:  time.Now().Add(v.attestationInterval),
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
	}

	// Verify quote signature
	if !v.verifySignature(quote) {
		report.Status = StatusFailed
		report.Violations = append(report.Violations, "Invalid signature")
		v.failedVerifications++
		v.reports[report.ID] = report
		return report, fmt.Errorf("signature verification failed")
	}

	// Check quote freshness
	if time.Since(quote.Timestamp) > 5*time.Minute {
		report.Status = StatusFailed
		report.Violations = append(report.Violations, "Quote expired")
		v.failedVerifications++
		v.reports[report.ID] = report
		return report, fmt.Errorf("quote expired")
	}

	// Apply policies
	trustLevel := 1.0
	for _, policy := range v.policies {
		if !policy.Enabled {
			continue
		}

		// Verify measurements
		if len(policy.RequiredMeasurements) > 0 {
			measurementHex := hex.EncodeToString(quote.Measurement)
			found := false
			for _, required := range policy.RequiredMeasurements {
				if measurementHex == required {
					found = true
					break
				}
			}
			if !found {
				trustLevel -= 0.2
				report.Violations = append(report.Violations, fmt.Sprintf("Measurement mismatch (policy: %s)", policy.Name))
			}
		}

		// Verify PCR values if TPM attestation
		if quote.Type == AttestationTPM && len(policy.AllowedPCRValues) > 0 {
			for pcr, expectedValue := range policy.AllowedPCRValues {
				actualValue, exists := quote.PCRValues[pcr]
				if !exists || !bytesEqual(actualValue, expectedValue) {
					trustLevel -= 0.15
					report.Violations = append(report.Violations, fmt.Sprintf("PCR %d mismatch", pcr))
				}
			}
		}

		// Check minimum trust level
		if trustLevel < policy.MinTrustLevel {
			report.Status = StatusFailed
			report.TrustLevel = trustLevel
			v.failedVerifications++
			v.reports[report.ID] = report
			return report, fmt.Errorf("trust level below policy minimum: %.2f < %.2f", trustLevel, policy.MinTrustLevel)
		}
	}

	// Verification successful
	report.Status = StatusVerified
	report.TrustLevel = trustLevel
	report.Verified = true

	if trustLevel >= 0.9 {
		report.Status = StatusVerified
	} else if trustLevel >= 0.7 {
		report.Status = StatusVerified
	} else {
		report.Status = StatusFailed
		v.failedVerifications++
		v.reports[report.ID] = report
		return report, fmt.Errorf("insufficient trust level: %.2f", trustLevel)
	}

	v.successfulVerifications++
	v.reports[report.ID] = report
	return report, nil
}

// verifySignature verifies quote signature
func (v *Verifier) verifySignature(quote *Quote) bool {
	// Simplified verification
	h := sha256.New()
	h.Write(quote.Measurement)
	h.Write(quote.Nonce)
	expectedHash := h.Sum(nil)

	if len(quote.Signature) < 32 {
		return false
	}

	return bytesEqual(quote.Signature[:32], expectedHash)
}

// GetReport retrieves an attestation report
func (v *Verifier) GetReport(reportID string) (*Report, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	report, exists := v.reports[reportID]
	if !exists {
		return nil, fmt.Errorf("report not found: %s", reportID)
	}

	// Check expiration
	if time.Now().After(report.ExpiresAt) {
		report.Status = StatusExpired
	}

	return report, nil
}

// ListReports lists all attestation reports
func (v *Verifier) ListReports() []*Report {
	v.mu.RLock()
	defer v.mu.RUnlock()

	reports := make([]*Report, 0, len(v.reports))
	for _, report := range v.reports {
		// Check expiration
		if time.Now().After(report.ExpiresAt) {
			report.Status = StatusExpired
		}
		reports = append(reports, report)
	}

	return reports
}

// ContinuousAttestation performs continuous attestation
func (v *Verifier) ContinuousAttestation(entityID string, attestationType AttestationType) error {
	// Generate nonce
	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		return err
	}

	// Generate quote
	quote, err := v.GenerateQuote(entityID, attestationType, nonce)
	if err != nil {
		return fmt.Errorf("quote generation failed: %w", err)
	}

	// Verify quote
	_, err = v.VerifyQuote(quote, entityID, "vm")
	if err != nil {
		return fmt.Errorf("quote verification failed: %w", err)
	}

	return nil
}

// GetMetrics returns attestation metrics
func (v *Verifier) GetMetrics() map[string]interface{} {
	v.mu.RLock()
	defer v.mu.RUnlock()

	verifiedReports := 0
	failedReports := 0
	expiredReports := 0

	for _, report := range v.reports {
		switch report.Status {
		case StatusVerified:
			verifiedReports++
		case StatusFailed:
			failedReports++
		case StatusExpired:
			expiredReports++
		}
	}

	successRate := 0.0
	if v.totalVerifications > 0 {
		successRate = float64(v.successfulVerifications) / float64(v.totalVerifications)
	}

	return map[string]interface{}{
		"total_policies":           len(v.policies),
		"total_quotes":             len(v.quotes),
		"total_reports":            len(v.reports),
		"verified_reports":         verifiedReports,
		"failed_reports":           failedReports,
		"expired_reports":          expiredReports,
		"total_verifications":      v.totalVerifications,
		"successful_verifications": v.successfulVerifications,
		"failed_verifications":     v.failedVerifications,
		"success_rate":             successRate,
		"tpm_enabled":              v.tpmEnabled,
		"measured_boot":            v.measuredBoot,
		"runtime_integrity":        v.runtimeIntegrity,
		"attestation_interval_min": v.attestationInterval.Minutes(),
	}
}

// Helper functions

func generateQuoteID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("quote-%s", hex.EncodeToString(b))
}

func generateReportID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("report-%s", hex.EncodeToString(b))
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
