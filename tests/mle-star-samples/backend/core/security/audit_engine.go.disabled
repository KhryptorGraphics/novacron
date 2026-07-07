package security

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// AuditLevel defines the security audit level
type AuditLevel string

const (
	AuditLevelBasic       AuditLevel = "basic"
	AuditLevelStandard    AuditLevel = "standard"
	AuditLevelEnhanced    AuditLevel = "enhanced"
	AuditLevelEnterprise  AuditLevel = "enterprise"
)

// SecurityAuditEngine is the core security audit orchestrator
type SecurityAuditEngine struct {
	config           *AuditConfig
	vulnerabilityScanner *VulnerabilityScanner
	complianceMonitor    *ComplianceMonitor
	incidentResponder    *IncidentResponder
	continuousMonitor    *ContinuousMonitor
	mutex               sync.RWMutex
	auditSessions       map[string]*AuditSession
}

// AuditConfig holds configuration for security auditing
type AuditConfig struct {
	Level                 AuditLevel            `json:"level"`
	Environment          string                `json:"environment"`
	EnabledScans         []string              `json:"enabled_scans"`
	ComplianceFrameworks []string              `json:"compliance_frameworks"`
	NotificationChannels []NotificationChannel `json:"notification_channels"`
	RetentionPeriod      time.Duration         `json:"retention_period"`
	ScanInterval         time.Duration         `json:"scan_interval"`
	AlertThresholds      AlertThresholds       `json:"alert_thresholds"`
}

// AlertThresholds defines when to trigger security alerts
type AlertThresholds struct {
	CriticalVulnerabilities int     `json:"critical_vulnerabilities"`
	HighVulnerabilities     int     `json:"high_vulnerabilities"`
	ComplianceScore         float64 `json:"compliance_score"`
	SecurityScore           float64 `json:"security_score"`
	IncidentResponseTime    time.Duration `json:"incident_response_time"`
}

// AuditSession represents a single security audit session
type AuditSession struct {
	ID                string                 `json:"id"`
	StartTime         time.Time             `json:"start_time"`
	EndTime           *time.Time            `json:"end_time,omitempty"`
	Level             AuditLevel            `json:"level"`
	Environment       string                `json:"environment"`
	Status            AuditStatus           `json:"status"`
	Results           *AuditResults         `json:"results,omitempty"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// AuditStatus represents the status of an audit
type AuditStatus string

const (
	AuditStatusPending    AuditStatus = "pending"
	AuditStatusRunning    AuditStatus = "running"
	AuditStatusCompleted  AuditStatus = "completed"
	AuditStatusFailed     AuditStatus = "failed"
	AuditStatusCancelled  AuditStatus = "cancelled"
)

// AuditResults aggregates all security audit findings
type AuditResults struct {
	VulnerabilityResults *VulnerabilityResults `json:"vulnerability_results"`
	ComplianceResults    *ComplianceResults    `json:"compliance_results"`
	IncidentResults      *IncidentResults      `json:"incident_results"`
	ContinuousResults    *ContinuousResults    `json:"continuous_results"`
	OverallScore         float64               `json:"overall_score"`
	RiskLevel           RiskLevel             `json:"risk_level"`
	Summary             string                `json:"summary"`
	Recommendations     []Recommendation      `json:"recommendations"`
	GeneratedAt         time.Time            `json:"generated_at"`
}

// RiskLevel defines the overall security risk
type RiskLevel string

const (
	RiskLevelLow      RiskLevel = "low"
	RiskLevelMedium   RiskLevel = "medium"
	RiskLevelHigh     RiskLevel = "high"
	RiskLevelCritical RiskLevel = "critical"
)

// Recommendation represents a security improvement recommendation
type Recommendation struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Remediation string    `json:"remediation"`
	Priority    int       `json:"priority"`
	EstimatedEffort string `json:"estimated_effort"`
	References  []string  `json:"references"`
}

// NewSecurityAuditEngine creates a new security audit engine
func NewSecurityAuditEngine(config *AuditConfig) *SecurityAuditEngine {
	engine := &SecurityAuditEngine{
		config:              config,
		auditSessions:       make(map[string]*AuditSession),
		vulnerabilityScanner: NewVulnerabilityScanner(config),
		complianceMonitor:    NewComplianceMonitor(config),
		incidentResponder:    NewIncidentResponder(config),
		continuousMonitor:    NewContinuousMonitor(config),
	}
	
	return engine
}

// StartAudit initiates a new security audit session
func (e *SecurityAuditEngine) StartAudit(ctx context.Context, level AuditLevel, environment string) (*AuditSession, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	
	session := &AuditSession{
		ID:          uuid.New().String(),
		StartTime:   time.Now(),
		Level:       level,
		Environment: environment,
		Status:      AuditStatusPending,
		Metadata:    make(map[string]interface{}),
	}
	
	e.auditSessions[session.ID] = session
	
	// Start audit in background
	go e.executeAudit(ctx, session)
	
	return session, nil
}

// executeAudit runs the complete security audit
func (e *SecurityAuditEngine) executeAudit(ctx context.Context, session *AuditSession) {
	e.updateSessionStatus(session.ID, AuditStatusRunning)
	
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Security audit panic: %v", r)
			e.updateSessionStatus(session.ID, AuditStatusFailed)
		}
	}()
	
	results := &AuditResults{
		GeneratedAt: time.Now(),
	}
	
	// Run vulnerability scanning
	vulnResults, err := e.vulnerabilityScanner.Scan(ctx, session.Level, session.Environment)
	if err != nil {
		log.Printf("Vulnerability scan failed: %v", err)
	} else {
		results.VulnerabilityResults = vulnResults
	}
	
	// Run compliance monitoring
	compResults, err := e.complianceMonitor.Assess(ctx, session.Level)
	if err != nil {
		log.Printf("Compliance assessment failed: %v", err)
	} else {
		results.ComplianceResults = compResults
	}
	
	// Run incident analysis
	incidentResults, err := e.incidentResponder.Analyze(ctx, session.Level)
	if err != nil {
		log.Printf("Incident analysis failed: %v", err)
	} else {
		results.IncidentResults = incidentResults
	}
	
	// Run continuous monitoring check
	contResults, err := e.continuousMonitor.GetStatus(ctx)
	if err != nil {
		log.Printf("Continuous monitoring check failed: %v", err)
	} else {
		results.ContinuousResults = contResults
	}
	
	// Calculate overall security score and risk level
	results.OverallScore = e.calculateOverallScore(results)
	results.RiskLevel = e.calculateRiskLevel(results.OverallScore, results)
	results.Summary = e.generateSummary(results)
	results.Recommendations = e.generateRecommendations(results)
	
	// Update session with results
	e.updateSessionResults(session.ID, results)
	e.updateSessionStatus(session.ID, AuditStatusCompleted)
	
	// Trigger alerts if necessary
	e.processAlerts(ctx, results)
}

// calculateOverallScore computes the overall security score
func (e *SecurityAuditEngine) calculateOverallScore(results *AuditResults) float64 {
	var totalScore, weightSum float64
	
	// Vulnerability score (weight: 0.3)
	if results.VulnerabilityResults != nil {
		vulnScore := 100.0 - float64(results.VulnerabilityResults.Critical*10 + 
			results.VulnerabilityResults.High*5 + 
			results.VulnerabilityResults.Medium*2 + 
			results.VulnerabilityResults.Low*1)
		if vulnScore < 0 {
			vulnScore = 0
		}
		totalScore += vulnScore * 0.3
		weightSum += 0.3
	}
	
	// Compliance score (weight: 0.25)
	if results.ComplianceResults != nil {
		totalScore += results.ComplianceResults.OverallScore * 0.25
		weightSum += 0.25
	}
	
	// Incident response score (weight: 0.25)
	if results.IncidentResults != nil {
		incidentScore := 100.0 - float64(results.IncidentResults.OpenIncidents*5 + 
			results.IncidentResults.UnresolvedCritical*10)
		if incidentScore < 0 {
			incidentScore = 0
		}
		totalScore += incidentScore * 0.25
		weightSum += 0.25
	}
	
	// Continuous monitoring score (weight: 0.2)
	if results.ContinuousResults != nil {
		totalScore += results.ContinuousResults.HealthScore * 0.2
		weightSum += 0.2
	}
	
	if weightSum == 0 {
		return 0
	}
	
	return totalScore / weightSum
}

// calculateRiskLevel determines the overall risk level
func (e *SecurityAuditEngine) calculateRiskLevel(score float64, results *AuditResults) RiskLevel {
	// Check for critical conditions first
	if results.VulnerabilityResults != nil && results.VulnerabilityResults.Critical > 0 {
		return RiskLevelCritical
	}
	
	if results.IncidentResults != nil && results.IncidentResults.UnresolvedCritical > 0 {
		return RiskLevelCritical
	}
	
	// Base risk level on score
	switch {
	case score >= 80:
		return RiskLevelLow
	case score >= 60:
		return RiskLevelMedium
	case score >= 40:
		return RiskLevelHigh
	default:
		return RiskLevelCritical
	}
}

// generateSummary creates a human-readable summary
func (e *SecurityAuditEngine) generateSummary(results *AuditResults) string {
	summary := fmt.Sprintf("Security audit completed with overall score: %.1f/100 (Risk: %s)", 
		results.OverallScore, results.RiskLevel)
	
	if results.VulnerabilityResults != nil {
		summary += fmt.Sprintf("\nVulnerabilities: %d Critical, %d High, %d Medium, %d Low",
			results.VulnerabilityResults.Critical,
			results.VulnerabilityResults.High,
			results.VulnerabilityResults.Medium,
			results.VulnerabilityResults.Low)
	}
	
	if results.ComplianceResults != nil {
		summary += fmt.Sprintf("\nCompliance Score: %.1f/100", results.ComplianceResults.OverallScore)
	}
	
	return summary
}

// generateRecommendations creates actionable security recommendations
func (e *SecurityAuditEngine) generateRecommendations(results *AuditResults) []Recommendation {
	var recommendations []Recommendation
	
	// Vulnerability-based recommendations
	if results.VulnerabilityResults != nil {
		if results.VulnerabilityResults.Critical > 0 {
			recommendations = append(recommendations, Recommendation{
				ID:          "vuln-critical",
				Type:        "vulnerability",
				Severity:    "critical",
				Title:       "Address Critical Vulnerabilities",
				Description: fmt.Sprintf("%d critical vulnerabilities require immediate attention", results.VulnerabilityResults.Critical),
				Remediation: "Review and patch all critical vulnerabilities within 24 hours",
				Priority:    1,
				EstimatedEffort: "1-3 days",
				References:  []string{"OWASP Top 10", "CVE Database"},
			})
		}
		
		if results.VulnerabilityResults.DependencyVulns > 0 {
			recommendations = append(recommendations, Recommendation{
				ID:          "vuln-deps",
				Type:        "dependency",
				Severity:    "high",
				Title:       "Update Vulnerable Dependencies",
				Description: fmt.Sprintf("%d vulnerable dependencies detected", results.VulnerabilityResults.DependencyVulns),
				Remediation: "Update all dependencies to latest secure versions",
				Priority:    2,
				EstimatedEffort: "1-2 days",
				References:  []string{"npm audit", "OWASP Dependency Check"},
			})
		}
	}
	
	// Compliance-based recommendations
	if results.ComplianceResults != nil && results.ComplianceResults.OverallScore < 80 {
		recommendations = append(recommendations, Recommendation{
			ID:          "compliance-improve",
			Type:        "compliance",
			Severity:    "medium",
			Title:       "Improve Compliance Posture",
			Description: fmt.Sprintf("Compliance score is %.1f/100, below recommended threshold", results.ComplianceResults.OverallScore),
			Remediation: "Review and implement missing compliance controls",
			Priority:    3,
			EstimatedEffort: "1-2 weeks",
			References:  []string{"SOC 2", "GDPR Guidelines", "NIST Framework"},
		})
	}
	
	return recommendations
}

// updateSessionStatus updates the status of an audit session
func (e *SecurityAuditEngine) updateSessionStatus(sessionID string, status AuditStatus) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	
	if session, exists := e.auditSessions[sessionID]; exists {
		session.Status = status
		if status == AuditStatusCompleted || status == AuditStatusFailed || status == AuditStatusCancelled {
			now := time.Now()
			session.EndTime = &now
		}
	}
}

// updateSessionResults updates the results of an audit session
func (e *SecurityAuditEngine) updateSessionResults(sessionID string, results *AuditResults) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	
	if session, exists := e.auditSessions[sessionID]; exists {
		session.Results = results
	}
}

// GetAuditSession retrieves an audit session by ID
func (e *SecurityAuditEngine) GetAuditSession(sessionID string) (*AuditSession, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	
	session, exists := e.auditSessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("audit session not found: %s", sessionID)
	}
	
	return session, nil
}

// ListAuditSessions returns all audit sessions
func (e *SecurityAuditEngine) ListAuditSessions() []*AuditSession {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	
	sessions := make([]*AuditSession, 0, len(e.auditSessions))
	for _, session := range e.auditSessions {
		sessions = append(sessions, session)
	}
	
	return sessions
}

// processAlerts handles alert processing based on results
func (e *SecurityAuditEngine) processAlerts(ctx context.Context, results *AuditResults) {
	thresholds := e.config.AlertThresholds
	
	// Check critical vulnerability threshold
	if results.VulnerabilityResults != nil && 
		results.VulnerabilityResults.Critical >= thresholds.CriticalVulnerabilities {
		e.sendAlert(ctx, "critical_vulnerabilities", fmt.Sprintf(
			"Critical vulnerability threshold exceeded: %d vulnerabilities found",
			results.VulnerabilityResults.Critical))
	}
	
	// Check overall security score
	if results.OverallScore < thresholds.SecurityScore {
		e.sendAlert(ctx, "low_security_score", fmt.Sprintf(
			"Security score below threshold: %.1f/100", results.OverallScore))
	}
	
	// Check compliance score
	if results.ComplianceResults != nil && 
		results.ComplianceResults.OverallScore < thresholds.ComplianceScore {
		e.sendAlert(ctx, "compliance_failure", fmt.Sprintf(
			"Compliance score below threshold: %.1f/100", 
			results.ComplianceResults.OverallScore))
	}
}

// sendAlert sends security alerts through configured channels
func (e *SecurityAuditEngine) sendAlert(ctx context.Context, alertType, message string) {
	for _, channel := range e.config.NotificationChannels {
		go func(ch NotificationChannel) {
			if err := e.sendNotification(ctx, ch, alertType, message); err != nil {
				log.Printf("Failed to send alert via %s: %v", ch.Type, err)
			}
		}(channel)
	}
}

// sendNotification sends a notification through a specific channel
func (e *SecurityAuditEngine) sendNotification(ctx context.Context, channel NotificationChannel, alertType, message string) error {
	// Implementation would depend on the notification channel type
	// This is a placeholder for the actual notification logic
	log.Printf("Alert [%s] via %s: %s", alertType, channel.Type, message)
	return nil
}

// NotificationChannel defines a notification channel
type NotificationChannel struct {
	Type       string                 `json:"type"`
	Endpoint   string                 `json:"endpoint"`
	Credentials map[string]string     `json:"credentials"`
	Config     map[string]interface{} `json:"config"`
}

// ExportResults exports audit results to various formats
func (e *SecurityAuditEngine) ExportResults(sessionID string, format string) ([]byte, error) {
	session, err := e.GetAuditSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	if session.Results == nil {
		return nil, fmt.Errorf("no results available for session %s", sessionID)
	}
	
	switch format {
	case "json":
		return json.MarshalIndent(session.Results, "", "  ")
	case "summary":
		return []byte(session.Results.Summary), nil
	default:
		return nil, fmt.Errorf("unsupported export format: %s", format)
	}
}