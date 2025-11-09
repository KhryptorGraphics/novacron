// Package security provides advanced security and zero-trust architecture for NovaCron
package security

import (
	"time"
)

// SecurityConfig represents the comprehensive security configuration
type SecurityConfig struct {
	ZeroTrust             ZeroTrustConfig
	AIThreatDetection     AIThreatConfig
	ConfidentialComputing CCConfig
	PostQuantumCrypto     PQCConfig
	HomomorphicEncryption HEConfig
	SMPC                  SMPCConfig
	HSM                   HSMConfig
	Attestation           AttestationConfig
	Policies              PolicyConfig
	ThreatIntelligence    ThreatIntelConfig
	IncidentResponse      IRConfig
	Metrics               MetricsConfig
}

// ZeroTrustConfig for zero-trust architecture
type ZeroTrustConfig struct {
	Enabled                  bool
	ContinuousAuthentication bool
	MicroSegmentation        bool
	LeastPrivilege           bool
	ContextAwarePolicies     bool
	MaxTrustDuration         time.Duration // Re-verify trust
	DeviceVerification       bool
	LocationVerification     bool
	BehaviorAnalysis         bool
}

// AIThreatConfig for AI-powered threat detection
type AIThreatConfig struct {
	Enabled                 bool
	Model                   string  // "isolation_forest", "lstm", "ensemble"
	Threshold               float64 // 0.8 (threat score threshold)
	FalsePositiveTarget     float64 // 0.001 (<0.1%)
	AnomalyDetection        bool
	BehavioralAnalysis      bool
	SignaturelessDetection  bool
	ThreatIntelIntegration  bool
	RealTimeScoring         bool
	DetectionLatencyTarget  time.Duration // <500ms
	ModelUpdateInterval     time.Duration
	TrainingDataRetention   time.Duration
}

// CCConfig for confidential computing
type CCConfig struct {
	Enabled      bool
	TEEType      string // "sgx", "sev", "trustzone"
	IntelSGX     IntelSGXConfig
	AMDSEV       AMDSEVConfig
	ARMTrustZone ARMTrustZoneConfig
	Attestation  bool
	MemoryEncryption bool
}

// IntelSGXConfig for Intel SGX
type IntelSGXConfig struct {
	Enabled          bool
	EnclaveSize      int64
	AttestationURL   string
	QuoteGeneration  bool
	RemoteAttestation bool
}

// AMDSEVConfig for AMD SEV
type AMDSEVConfig struct {
	Enabled       bool
	SEVEnabled    bool
	SEVESEnabled  bool // Encrypted State
	SEVSNPEnabled bool // Secure Nested Paging
	MinFirmware   string
}

// ARMTrustZoneConfig for ARM TrustZone
type ARMTrustZoneConfig struct {
	Enabled       bool
	SecureWorld   bool
	NormalWorld   bool
	TrustedApps   []string
}

// PQCConfig for post-quantum cryptography
type PQCConfig struct {
	Enabled       bool
	Algorithms    []string // ["kyber", "dilithium", "falcon", "sphincs"]
	HybridMode    bool     // Classical + PQC
	KeySize       int
	Kyber         KyberConfig
	Dilithium     DilithiumConfig
	FALCON        FALCONConfig
	SPHINCS       SPHINCSConfig
	TLSEnabled    bool
}

// KyberConfig for CRYSTALS-Kyber (key encapsulation)
type KyberConfig struct {
	Enabled    bool
	SecurityLevel int // 1, 3, or 5 (512, 768, 1024)
}

// DilithiumConfig for CRYSTALS-Dilithium (signatures)
type DilithiumConfig struct {
	Enabled    bool
	SecurityLevel int // 2, 3, or 5
}

// FALCONConfig for FALCON signatures
type FALCONConfig struct {
	Enabled    bool
	KeySize    int // 512 or 1024
}

// SPHINCSConfig for SPHINCS+ hash-based signatures
type SPHINCSConfig struct {
	Enabled    bool
	SecurityLevel int
	Variant    string // "shake256", "sha256"
}

// HEConfig for homomorphic encryption
type HEConfig struct {
	Enabled       bool
	Scheme        string // "phe", "she", "lfhe"
	SecurityLevel int
	KeySize       int
	UseCases      []string // ["encrypted_vm_state", "private_analytics", "mpc"]
}

// SMPCConfig for secure multi-party computation
type SMPCConfig struct {
	Enabled          bool
	Protocol         string // "shamir", "garbled_circuits", "oblivious_transfer"
	Parties          int
	Threshold        int // For Shamir secret sharing
	PrivacyPreserving bool
}

// HSMConfig for hardware security modules
type HSMConfig struct {
	Enabled      bool
	Provider     string // "aws_cloudhsm", "azure_keyvault", "thales"
	FIPSLevel    int    // 140-2 Level (1-4)
	Endpoint     string
	PartitionID  string
	KeyRotation  bool
	RotationInterval time.Duration
}

// AttestationConfig for attestation and verification
type AttestationConfig struct {
	Enabled          bool
	RemoteAttestation bool
	MeasuredBoot     bool
	RuntimeIntegrity bool
	TPM20Enabled     bool
	AttestationInterval time.Duration
	VerificationPolicy  string
}

// PolicyConfig for security policies
type PolicyConfig struct {
	Enabled            bool
	PolicyAsCode       bool
	OPAEnabled         bool
	OPAEndpoint        string
	DataClassification bool
	EncryptionRequired bool
	NetworkSegmentation bool
	ComplianceFrameworks []string // ["gdpr", "hipaa", "pci_dss", "soc2"]
}

// ThreatIntelConfig for threat intelligence
type ThreatIntelConfig struct {
	Enabled       bool
	Feeds         []string // ["misp", "stix", "taxii", "otx"]
	IoCSDetection bool
	ActorTracking bool
	VulnScanning  bool
	CVSSEnabled   bool
	UpdateInterval time.Duration
}

// IRConfig for incident response
type IRConfig struct {
	Enabled              bool
	AutoDetection        bool
	AutoContainment      bool
	PlaybookExecution    bool
	ForensicsCollection  bool
	AlertPrioritization  bool
	MTTDTarget           time.Duration // Mean Time To Detect
	MTTRTarget           time.Duration // Mean Time To Respond
}

// MetricsConfig for security metrics
type MetricsConfig struct {
	Enabled             bool
	ThreatDetectionRate bool
	FalsePositiveRate   bool
	MTTD                bool
	MTTR                bool
	SecurityPosture     bool
	ComplianceStatus    bool
	VulnerabilityCount  bool
}

// DefaultSecurityConfig returns the default security configuration
func DefaultSecurityConfig() *SecurityConfig {
	return &SecurityConfig{
		ZeroTrust: ZeroTrustConfig{
			Enabled:                  true,
			ContinuousAuthentication: true,
			MicroSegmentation:        true,
			LeastPrivilege:           true,
			ContextAwarePolicies:     true,
			MaxTrustDuration:         15 * time.Minute,
			DeviceVerification:       true,
			LocationVerification:     true,
			BehaviorAnalysis:         true,
		},
		AIThreatDetection: AIThreatConfig{
			Enabled:                true,
			Model:                  "ensemble",
			Threshold:              0.8,
			FalsePositiveTarget:    0.001,
			AnomalyDetection:       true,
			BehavioralAnalysis:     true,
			SignaturelessDetection: true,
			ThreatIntelIntegration: true,
			RealTimeScoring:        true,
			DetectionLatencyTarget: 500 * time.Millisecond,
			ModelUpdateInterval:    24 * time.Hour,
			TrainingDataRetention:  90 * 24 * time.Hour,
		},
		ConfidentialComputing: CCConfig{
			Enabled:          true,
			TEEType:          "sgx",
			Attestation:      true,
			MemoryEncryption: true,
			IntelSGX: IntelSGXConfig{
				Enabled:           true,
				EnclaveSize:       128 * 1024 * 1024, // 128MB
				QuoteGeneration:   true,
				RemoteAttestation: true,
			},
			AMDSEV: AMDSEVConfig{
				Enabled:       true,
				SEVEnabled:    true,
				SEVESEnabled:  true,
				SEVSNPEnabled: true,
			},
		},
		PostQuantumCrypto: PQCConfig{
			Enabled:    true,
			Algorithms: []string{"kyber", "dilithium", "falcon"},
			HybridMode: true,
			KeySize:    3072,
			Kyber: KyberConfig{
				Enabled:       true,
				SecurityLevel: 3, // Kyber768
			},
			Dilithium: DilithiumConfig{
				Enabled:       true,
				SecurityLevel: 3,
			},
			FALCON: FALCONConfig{
				Enabled: true,
				KeySize: 1024,
			},
			TLSEnabled: true,
		},
		HomomorphicEncryption: HEConfig{
			Enabled:       true,
			Scheme:        "lfhe",
			SecurityLevel: 128,
			KeySize:       4096,
			UseCases:      []string{"encrypted_vm_state", "private_analytics"},
		},
		SMPC: SMPCConfig{
			Enabled:           true,
			Protocol:          "shamir",
			Parties:           5,
			Threshold:         3,
			PrivacyPreserving: true,
		},
		HSM: HSMConfig{
			Enabled:          true,
			Provider:         "aws_cloudhsm",
			FIPSLevel:        3,
			KeyRotation:      true,
			RotationInterval: 90 * 24 * time.Hour,
		},
		Attestation: AttestationConfig{
			Enabled:             true,
			RemoteAttestation:   true,
			MeasuredBoot:        true,
			RuntimeIntegrity:    true,
			TPM20Enabled:        true,
			AttestationInterval: 5 * time.Minute,
		},
		Policies: PolicyConfig{
			Enabled:             true,
			PolicyAsCode:        true,
			OPAEnabled:          true,
			DataClassification:  true,
			EncryptionRequired:  true,
			NetworkSegmentation: true,
			ComplianceFrameworks: []string{"soc2", "hipaa", "pci_dss", "gdpr"},
		},
		ThreatIntelligence: ThreatIntelConfig{
			Enabled:        true,
			Feeds:          []string{"misp", "stix"},
			IoCSDetection:  true,
			ActorTracking:  true,
			VulnScanning:   true,
			CVSSEnabled:    true,
			UpdateInterval: 1 * time.Hour,
		},
		IncidentResponse: IRConfig{
			Enabled:             true,
			AutoDetection:       true,
			AutoContainment:     true,
			PlaybookExecution:   true,
			ForensicsCollection: true,
			AlertPrioritization: true,
			MTTDTarget:          1 * time.Minute,
			MTTRTarget:          5 * time.Minute,
		},
		Metrics: MetricsConfig{
			Enabled:             true,
			ThreatDetectionRate: true,
			FalsePositiveRate:   true,
			MTTD:                true,
			MTTR:                true,
			SecurityPosture:     true,
			ComplianceStatus:    true,
			VulnerabilityCount:  true,
		},
	}
}
