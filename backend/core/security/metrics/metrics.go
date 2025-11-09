// Package metrics implements security metrics and monitoring
package metrics

import (
	"sync"
	"time"
)

// SecurityMetrics represents comprehensive security metrics
type SecurityMetrics struct {
	// Threat Detection
	ThreatDetectionRate   float64
	FalsePositiveRate     float64
	FalseNegativeRate     float64
	TotalThreatsDetected  int64
	TruePositives         int64
	FalsePositives        int64
	TrueNegatives         int64
	FalseNegatives        int64

	// Incident Response
	MTTD                  time.Duration // Mean Time To Detect
	MTTR                  time.Duration // Mean Time To Respond
	TotalIncidents        int64
	ResolvedIncidents     int64
	ActiveIncidents       int64
	IncidentResolutionRate float64

	// Security Posture
	SecurityPostureScore  float64 // 0-100
	VulnerabilityCount    int64
	CriticalVulnerabilities int64
	HighVulnerabilities   int64
	MediumVulnerabilities int64
	LowVulnerabilities    int64

	// Compliance
	ComplianceStatus      map[string]float64 // Framework -> Compliance %
	PolicyViolations      int64
	ComplianceScore       float64

	// Encryption
	EncryptedDataPercent  float64
	EncryptionOperations  int64
	DecryptionOperations  int64

	// Access Control
	TotalAccessAttempts   int64
	AllowedAccess         int64
	DeniedAccess          int64
	AccessDenialRate      float64

	// Attestation
	TotalAttestations     int64
	SuccessfulAttestations int64
	FailedAttestations    int64
	AttestationSuccessRate float64

	// Timestamp
	Timestamp             time.Time

	mu sync.RWMutex
}

// Collector collects and aggregates security metrics
type Collector struct {
	metrics              *SecurityMetrics
	threatDetector       ThreatDetectorMetrics
	incidentOrchestrator IncidentOrchestratorMetrics
	policyEngine         PolicyEngineMetrics
	hsmManager           HSMMetrics
	attestationVerifier  AttestationMetrics
	mu                   sync.RWMutex
}

// ThreatDetectorMetrics interface for threat detector metrics
type ThreatDetectorMetrics interface {
	GetMetrics() map[string]interface{}
}

// IncidentOrchestratorMetrics interface for incident orchestrator metrics
type IncidentOrchestratorMetrics interface {
	GetMetrics() map[string]interface{}
}

// PolicyEngineMetrics interface for policy engine metrics
type PolicyEngineMetrics interface {
	GetMetrics() map[string]interface{}
}

// HSMMetrics interface for HSM metrics
type HSMMetrics interface {
	GetMetrics() map[string]interface{}
}

// AttestationMetrics interface for attestation metrics
type AttestationMetrics interface{
	GetMetrics() map[string]interface{}
}

// NewCollector creates a new metrics collector
func NewCollector() *Collector {
	return &Collector{
		metrics: &SecurityMetrics{
			ComplianceStatus: make(map[string]float64),
			Timestamp:        time.Now(),
		},
	}
}

// SetThreatDetector sets the threat detector metrics source
func (c *Collector) SetThreatDetector(detector ThreatDetectorMetrics) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.threatDetector = detector
}

// SetIncidentOrchestrator sets the incident orchestrator metrics source
func (c *Collector) SetIncidentOrchestrator(orchestrator IncidentOrchestratorMetrics) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.incidentOrchestrator = orchestrator
}

// SetPolicyEngine sets the policy engine metrics source
func (c *Collector) SetPolicyEngine(engine PolicyEngineMetrics) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.policyEngine = engine
}

// SetHSMManager sets the HSM manager metrics source
func (c *Collector) SetHSMManager(manager HSMMetrics) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.hsmManager = manager
}

// SetAttestationVerifier sets the attestation verifier metrics source
func (c *Collector) SetAttestationVerifier(verifier AttestationMetrics) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.attestationVerifier = verifier
}

// Collect collects metrics from all sources
func (c *Collector) Collect() *SecurityMetrics {
	c.mu.Lock()
	defer c.mu.Unlock()

	metrics := &SecurityMetrics{
		ComplianceStatus: make(map[string]float64),
		Timestamp:        time.Now(),
	}

	// Collect threat detection metrics
	if c.threatDetector != nil {
		tdMetrics := c.threatDetector.GetMetrics()
		if totalDetections, ok := tdMetrics["total_detections"].(int64); ok {
			metrics.TotalThreatsDetected = totalDetections
		}
		if tp, ok := tdMetrics["true_positives"].(int64); ok {
			metrics.TruePositives = tp
		}
		if fp, ok := tdMetrics["false_positives"].(int64); ok {
			metrics.FalsePositives = fp
		}
		if tn, ok := tdMetrics["true_negatives"].(int64); ok {
			metrics.TrueNegatives = tn
		}
		if fn, ok := tdMetrics["false_negatives"].(int64); ok {
			metrics.FalseNegatives = fn
		}
		if fpr, ok := tdMetrics["false_positive_rate"].(float64); ok {
			metrics.FalsePositiveRate = fpr
		}

		// Calculate threat detection rate
		totalPredictions := metrics.TruePositives + metrics.FalsePositives + metrics.TrueNegatives + metrics.FalseNegatives
		if totalPredictions > 0 {
			metrics.ThreatDetectionRate = float64(metrics.TruePositives) / float64(metrics.TruePositives+metrics.FalseNegatives)
		}
	}

	// Collect incident response metrics
	if c.incidentOrchestrator != nil {
		irMetrics := c.incidentOrchestrator.GetMetrics()
		if totalInc, ok := irMetrics["total_incidents"].(int64); ok {
			metrics.TotalIncidents = totalInc
		}
		if resolvedInc, ok := irMetrics["resolved_incidents"].(int64); ok {
			metrics.ResolvedIncidents = resolvedInc
		}
		if mttd, ok := irMetrics["avg_mttd_ms"].(int64); ok {
			metrics.MTTD = time.Duration(mttd) * time.Millisecond
		}
		if mttr, ok := irMetrics["avg_mttr_ms"].(int64); ok {
			metrics.MTTR = time.Duration(mttr) * time.Millisecond
		}

		metrics.ActiveIncidents = metrics.TotalIncidents - metrics.ResolvedIncidents
		if metrics.TotalIncidents > 0 {
			metrics.IncidentResolutionRate = float64(metrics.ResolvedIncidents) / float64(metrics.TotalIncidents)
		}
	}

	// Collect policy engine metrics
	if c.policyEngine != nil {
		peMetrics := c.policyEngine.GetMetrics()
		if totalEval, ok := peMetrics["total_evaluations"].(int64); ok {
			metrics.TotalAccessAttempts = totalEval
		}
		if allowed, ok := peMetrics["allowed_evaluations"].(int64); ok {
			metrics.AllowedAccess = allowed
		}
		if denied, ok := peMetrics["denied_evaluations"].(int64); ok {
			metrics.DeniedAccess = denied
			metrics.PolicyViolations = denied
		}

		if metrics.TotalAccessAttempts > 0 {
			metrics.AccessDenialRate = float64(metrics.DeniedAccess) / float64(metrics.TotalAccessAttempts)
		}

		// Compliance status
		if frameworks, ok := peMetrics["compliance_frameworks"].([]interface{}); ok {
			for _, fw := range frameworks {
				if framework, ok := fw.(string); ok {
					// Simplified compliance calculation
					metrics.ComplianceStatus[framework] = 0.95 // 95% compliant
				}
			}
		}
	}

	// Collect HSM metrics
	if c.hsmManager != nil {
		hsmMetrics := c.hsmManager.GetMetrics()
		if totalOps, ok := hsmMetrics["total_operations"].(int64); ok {
			// Approximate encryption/decryption split
			metrics.EncryptionOperations = totalOps / 2
			metrics.DecryptionOperations = totalOps / 2
		}
	}

	// Collect attestation metrics
	if c.attestationVerifier != nil {
		attMetrics := c.attestationVerifier.GetMetrics()
		if totalAtt, ok := attMetrics["total_verifications"].(int64); ok {
			metrics.TotalAttestations = totalAtt
		}
		if successful, ok := attMetrics["successful_verifications"].(int64); ok {
			metrics.SuccessfulAttestations = successful
		}
		if failed, ok := attMetrics["failed_verifications"].(int64); ok {
			metrics.FailedAttestations = failed
		}

		if metrics.TotalAttestations > 0 {
			metrics.AttestationSuccessRate = float64(metrics.SuccessfulAttestations) / float64(metrics.TotalAttestations)
		}
	}

	// Calculate security posture score (0-100)
	metrics.SecurityPostureScore = c.calculateSecurityPostureScore(metrics)

	// Calculate overall compliance score
	totalCompliance := 0.0
	for _, score := range metrics.ComplianceStatus {
		totalCompliance += score
	}
	if len(metrics.ComplianceStatus) > 0 {
		metrics.ComplianceScore = totalCompliance / float64(len(metrics.ComplianceStatus))
	}

	c.metrics = metrics
	return metrics
}

// calculateSecurityPostureScore calculates overall security posture score
func (c *Collector) calculateSecurityPostureScore(metrics *SecurityMetrics) float64 {
	score := 100.0

	// Threat detection impact (-20 points for poor detection)
	if metrics.ThreatDetectionRate < 0.8 {
		score -= (0.8 - metrics.ThreatDetectionRate) * 25
	}

	// False positive impact (-10 points for high FP rate)
	if metrics.FalsePositiveRate > 0.1 {
		score -= (metrics.FalsePositiveRate - 0.1) * 100
	}

	// Incident resolution impact (-15 points for poor resolution)
	if metrics.IncidentResolutionRate < 0.9 {
		score -= (0.9 - metrics.IncidentResolutionRate) * 15
	}

	// MTTD impact (-10 points if exceeds 1 minute)
	if metrics.MTTD > 1*time.Minute {
		score -= float64(metrics.MTTD.Minutes()) * 2
	}

	// MTTR impact (-15 points if exceeds 5 minutes)
	if metrics.MTTR > 5*time.Minute {
		score -= float64(metrics.MTTR.Minutes()) * 1
	}

	// Compliance impact (-20 points for low compliance)
	if metrics.ComplianceScore < 0.95 {
		score -= (0.95 - metrics.ComplianceScore) * 40
	}

	// Attestation impact (-10 points for low success rate)
	if metrics.AttestationSuccessRate < 0.95 {
		score -= (0.95 - metrics.AttestationSuccessRate) * 20
	}

	// Ensure score is within bounds
	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	return score
}

// GetMetrics returns the current security metrics
func (c *Collector) GetMetrics() *SecurityMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.metrics
}

// GetSummary returns a summary of security metrics
func (c *Collector) GetSummary() map[string]interface{} {
	metrics := c.Collect()

	return map[string]interface{}{
		"security_posture_score":   metrics.SecurityPostureScore,
		"threat_detection_rate":    metrics.ThreatDetectionRate,
		"false_positive_rate":      metrics.FalsePositiveRate,
		"total_threats_detected":   metrics.TotalThreatsDetected,
		"mttd_ms":                  metrics.MTTD.Milliseconds(),
		"mttr_ms":                  metrics.MTTR.Milliseconds(),
		"total_incidents":          metrics.TotalIncidents,
		"active_incidents":         metrics.ActiveIncidents,
		"incident_resolution_rate": metrics.IncidentResolutionRate,
		"compliance_score":         metrics.ComplianceScore,
		"compliance_status":        metrics.ComplianceStatus,
		"policy_violations":        metrics.PolicyViolations,
		"access_denial_rate":       metrics.AccessDenialRate,
		"attestation_success_rate": metrics.AttestationSuccessRate,
		"timestamp":                metrics.Timestamp,
	}
}

// ExportMetrics exports metrics in a specified format
func (c *Collector) ExportMetrics(format string) ([]byte, error) {
	metrics := c.Collect()

	switch format {
	case "json":
		// Would use json.Marshal in production
		return []byte(fmt.Sprintf("{\"security_posture_score\":%.2f}", metrics.SecurityPostureScore)), nil
	case "prometheus":
		// Export in Prometheus format
		return c.exportPrometheus(metrics), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}

// exportPrometheus exports metrics in Prometheus format
func (c *Collector) exportPrometheus(metrics *SecurityMetrics) []byte {
	output := ""

	output += fmt.Sprintf("# HELP security_posture_score Overall security posture score (0-100)\n")
	output += fmt.Sprintf("# TYPE security_posture_score gauge\n")
	output += fmt.Sprintf("security_posture_score %.2f\n\n", metrics.SecurityPostureScore)

	output += fmt.Sprintf("# HELP threat_detection_rate Threat detection rate\n")
	output += fmt.Sprintf("# TYPE threat_detection_rate gauge\n")
	output += fmt.Sprintf("threat_detection_rate %.4f\n\n", metrics.ThreatDetectionRate)

	output += fmt.Sprintf("# HELP false_positive_rate False positive rate\n")
	output += fmt.Sprintf("# TYPE false_positive_rate gauge\n")
	output += fmt.Sprintf("false_positive_rate %.4f\n\n", metrics.FalsePositiveRate)

	output += fmt.Sprintf("# HELP mttd_milliseconds Mean time to detect in milliseconds\n")
	output += fmt.Sprintf("# TYPE mttd_milliseconds gauge\n")
	output += fmt.Sprintf("mttd_milliseconds %d\n\n", metrics.MTTD.Milliseconds())

	output += fmt.Sprintf("# HELP mttr_milliseconds Mean time to respond in milliseconds\n")
	output += fmt.Sprintf("# TYPE mttr_milliseconds gauge\n")
	output += fmt.Sprintf("mttr_milliseconds %d\n\n", metrics.MTTR.Milliseconds())

	output += fmt.Sprintf("# HELP total_incidents Total security incidents\n")
	output += fmt.Sprintf("# TYPE total_incidents counter\n")
	output += fmt.Sprintf("total_incidents %d\n\n", metrics.TotalIncidents)

	output += fmt.Sprintf("# HELP compliance_score Overall compliance score\n")
	output += fmt.Sprintf("# TYPE compliance_score gauge\n")
	output += fmt.Sprintf("compliance_score %.4f\n\n", metrics.ComplianceScore)

	return []byte(output)
}
