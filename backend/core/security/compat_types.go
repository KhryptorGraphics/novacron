//go:build !novacron_security_orchestrator && !novacron_security_enterprise && !novacron_dating_app_security

package security

import "time"

// ThreatSeverity remains part of the default security surface even when the
// broader enterprise/orchestrator stacks are not compiled in.
type ThreatSeverity = SecuritySeverity

const (
	SeverityInfo     ThreatSeverity = "info"
	SeverityLow      ThreatSeverity = "low"
	SeverityMedium   ThreatSeverity = "medium"
	SeverityHigh     ThreatSeverity = "high"
	SeverityCritical ThreatSeverity = "critical"
)

// AIThreatConfig keeps the default server's security config shape intact while
// the enterprise AI threat module is disabled.
type AIThreatConfig struct {
	Enabled                bool
	Model                  string
	Threshold              float64
	FalsePositiveTarget    float64
	AnomalyDetection       bool
	BehavioralAnalysis     bool
	SignaturelessDetection bool
	ThreatIntelIntegration bool
	RealTimeScoring        bool
	DetectionLatencyTarget time.Duration
	ModelUpdateInterval    time.Duration
	TrainingDataRetention  time.Duration

	EnableMLDetection        bool
	MinModelAccuracy         float64
	EnableOnlineLearning     bool
	AnomalyThreshold         float64
	BaselineWindow           time.Duration
	EnableBehaviorAnalysis   bool
	EnableThreatIntel        bool
	ThreatFeedURLs           []string
	ThreatFeedUpdateInterval time.Duration
	EnableAutoResponse       bool
	ResponseConfidenceMin    float64
	EscalationThreshold      float64
	MaxConcurrentAnalysis    int
	AnalysisTimeout          time.Duration
	FeatureCacheSize         int
}

// EncryptionConfig preserves the simple root-server constructor shape while the
// nested encryption manager still expects key-management details.
type EncryptionConfig struct {
	Algorithm           string
	KeyRotationEnabled  bool
	KeyRotationInterval time.Duration
	KeyManagement       KeyManagementConfig
}

type KeyManagementConfig struct {
	KeyRotation int
}

type Purpose string

const (
	PurposeMatching  Purpose = "matching"
	PurposeMessaging Purpose = "messaging"
	PurposeLocation  Purpose = "location_services"
	PurposePayment   Purpose = "payment_processing"
	PurposeAnalytics Purpose = "analytics"
	PurposeMarketing Purpose = "marketing"
	PurposeSupport   Purpose = "customer_support"
	PurposeSafety    Purpose = "safety_security"
)

type DataCategory string

const (
	DataIdentity      DataCategory = "identity"
	DataContact       DataCategory = "contact"
	DataLocation      DataCategory = "location"
	DataBehavioral    DataCategory = "behavioral"
	DataBiometric     DataCategory = "biometric"
	DataFinancial     DataCategory = "financial"
	DataCommunication DataCategory = "communication"
	DataMedia         DataCategory = "media"
)
