// Package posture implements security posture management
package posture

import (
	"context"
	"fmt"
	"sync"
	"time"

	"novacron/backend/core/compliance"
)

// Engine implements continuous security posture assessment
type Engine struct {
	currentPosture  *compliance.SecurityPosture
	postureHistory  []*compliance.SecurityPosture
	vulnerabilities map[string]*Vulnerability
	scanners        map[string]Scanner
	mu              sync.RWMutex
}

// Vulnerability represents a security vulnerability
type Vulnerability struct {
	ID             string    `json:"id"`
	Severity       string    `json:"severity"` // critical, high, medium, low
	Type           string    `json:"type"`
	Title          string    `json:"title"`
	Description    string    `json:"description"`
	Affected       []string  `json:"affected"`
	CVE            string    `json:"cve,omitempty"`
	CVSS           float64   `json:"cvss,omitempty"`
	Remediation    string    `json:"remediation"`
	DetectedAt     time.Time `json:"detected_at"`
	ResolvedAt     *time.Time `json:"resolved_at,omitempty"`
	Status         string    `json:"status"`
}

// Scanner defines vulnerability scanner interface
type Scanner interface {
	Scan(ctx context.Context) ([]*Vulnerability, error)
	GetType() string
}

// NewEngine creates a new security posture engine
func NewEngine() *Engine {
	engine := &Engine{
		postureHistory:  []*compliance.SecurityPosture{},
		vulnerabilities: make(map[string]*Vulnerability),
		scanners:        make(map[string]Scanner),
	}

	engine.registerDefaultScanners()
	return engine
}

// registerDefaultScanners registers built-in vulnerability scanners
func (e *Engine) registerDefaultScanners() {
	// In production, would register actual scanner implementations
	e.scanners["network"] = &NetworkScanner{}
	e.scanners["host"] = &HostScanner{}
	e.scanners["container"] = &ContainerScanner{}
	e.scanners["config"] = &ConfigScanner{}
	e.scanners["dependency"] = &DependencyScanner{}
}

// AssessPosture performs comprehensive security posture assessment
func (e *Engine) AssessPosture(ctx context.Context) (*compliance.SecurityPosture, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	posture := &compliance.SecurityPosture{
		ID:        fmt.Sprintf("posture-%d", time.Now().Unix()),
		Timestamp: time.Now(),
	}

	// Run all vulnerability scanners
	vulnSummary := compliance.VulnerabilitySummary{}
	for scannerType, scanner := range e.scanners {
		vulns, err := scanner.Scan(ctx)
		if err != nil {
			return nil, fmt.Errorf("scanner %s failed: %w", scannerType, err)
		}

		// Aggregate vulnerabilities
		for _, vuln := range vulns {
			e.vulnerabilities[vuln.ID] = vuln

			switch vuln.Severity {
			case "critical":
				vulnSummary.Critical++
			case "high":
				vulnSummary.High++
			case "medium":
				vulnSummary.Medium++
			case "low":
				vulnSummary.Low++
			}
			vulnSummary.Total++
		}
	}

	vulnSummary.LastScan = time.Now()
	posture.Vulnerabilities = vulnSummary

	// Assess compliance scores
	posture.Compliance = e.assessComplianceScores(ctx)

	// Assess control categories
	posture.Controls = e.assessControlCategories(ctx)

	// Calculate overall security score
	posture.OverallScore = e.calculateSecurityScore(posture)

	// Determine risk level
	posture.RiskLevel = e.determineRiskLevel(posture)

	// Calculate trends
	posture.Trends = e.calculateTrends()

	// Generate recommendations
	posture.Recommendations = e.generateRecommendations(posture)

	// Store in history
	e.currentPosture = posture
	e.postureHistory = append(e.postureHistory, posture)

	return posture, nil
}

// assessComplianceScores assesses compliance with various frameworks
func (e *Engine) assessComplianceScores(ctx context.Context) map[compliance.ComplianceFramework]float64 {
	// In production, would query actual compliance engines
	scores := map[compliance.ComplianceFramework]float64{
		compliance.FrameworkSOC2:     92.5,
		compliance.FrameworkGDPR:     95.0,
		compliance.FrameworkHIPAA:    88.0,
		compliance.FrameworkPCIDSS:   85.0,
		compliance.FrameworkISO27001: 90.0,
		compliance.FrameworkNIST:     87.5,
	}

	return scores
}

// assessControlCategories assesses security controls by category
func (e *Engine) assessControlCategories(ctx context.Context) map[compliance.ControlCategory]float64 {
	// In production, would perform actual control assessments
	controls := map[compliance.ControlCategory]float64{
		compliance.CategoryAccessControl:    95.0,
		compliance.CategoryDataProtection:   92.0,
		compliance.CategoryAuditLogging:     98.0,
		compliance.CategoryIncidentResponse: 85.0,
		compliance.CategoryChangeManagement: 88.0,
		compliance.CategoryNetworkSecurity:  90.0,
		compliance.CategoryEncryption:       96.0,
		compliance.CategoryMonitoring:       93.0,
		compliance.CategoryBackupRecovery:   87.0,
		compliance.CategoryVendorManagement: 82.0,
	}

	return controls
}

// calculateSecurityScore calculates overall security score
func (e *Engine) calculateSecurityScore(posture *compliance.SecurityPosture) float64 {
	// Weighted scoring algorithm
	vulnWeight := 0.4
	complianceWeight := 0.3
	controlWeight := 0.3

	// Vulnerability score (inverse - fewer vulnerabilities = higher score)
	vulnScore := 100.0
	vulnScore -= float64(posture.Vulnerabilities.Critical) * 10.0
	vulnScore -= float64(posture.Vulnerabilities.High) * 5.0
	vulnScore -= float64(posture.Vulnerabilities.Medium) * 2.0
	vulnScore -= float64(posture.Vulnerabilities.Low) * 0.5

	if vulnScore < 0 {
		vulnScore = 0
	}

	// Compliance score (average of all frameworks)
	complianceScore := 0.0
	for _, score := range posture.Compliance {
		complianceScore += score
	}
	if len(posture.Compliance) > 0 {
		complianceScore /= float64(len(posture.Compliance))
	}

	// Control score (average of all categories)
	controlScore := 0.0
	for _, score := range posture.Controls {
		controlScore += score
	}
	if len(posture.Controls) > 0 {
		controlScore /= float64(len(posture.Controls))
	}

	// Calculate weighted score
	score := (vulnScore * vulnWeight) + (complianceScore * complianceWeight) + (controlScore * controlWeight)

	return score
}

// determineRiskLevel determines overall risk level
func (e *Engine) determineRiskLevel(posture *compliance.SecurityPosture) string {
	score := posture.OverallScore

	// Check for critical vulnerabilities
	if posture.Vulnerabilities.Critical > 0 {
		return "critical"
	}

	// Score-based risk levels
	if score >= 90 {
		return "low"
	} else if score >= 75 {
		return "medium"
	} else if score >= 60 {
		return "high"
	}

	return "critical"
}

// calculateTrends calculates security posture trends
func (e *Engine) calculateTrends() compliance.PostureTrends {
	trends := compliance.PostureTrends{
		Direction: "stable",
	}

	if len(e.postureHistory) < 2 {
		return trends
	}

	current := e.postureHistory[len(e.postureHistory)-1]

	// Find posture from 7 days ago
	sevenDaysAgo := time.Now().AddDate(0, 0, -7)
	var posture7d *compliance.SecurityPosture
	for i := len(e.postureHistory) - 1; i >= 0; i-- {
		if e.postureHistory[i].Timestamp.Before(sevenDaysAgo) {
			posture7d = e.postureHistory[i]
			break
		}
	}

	if posture7d != nil {
		trends.ScoreChange7d = current.OverallScore - posture7d.OverallScore
		trends.VulnChange7d = current.Vulnerabilities.Total - posture7d.Vulnerabilities.Total
	}

	// Find posture from 30 days ago
	thirtyDaysAgo := time.Now().AddDate(0, 0, -30)
	var posture30d *compliance.SecurityPosture
	for i := len(e.postureHistory) - 1; i >= 0; i-- {
		if e.postureHistory[i].Timestamp.Before(thirtyDaysAgo) {
			posture30d = e.postureHistory[i]
			break
		}
	}

	if posture30d != nil {
		trends.ScoreChange30d = current.OverallScore - posture30d.OverallScore
		trends.VulnChange30d = current.Vulnerabilities.Total - posture30d.Vulnerabilities.Total
	}

	// Determine overall direction
	if trends.ScoreChange7d > 2.0 {
		trends.Direction = "improving"
	} else if trends.ScoreChange7d < -2.0 {
		trends.Direction = "declining"
	}

	return trends
}

// generateRecommendations generates security recommendations
func (e *Engine) generateRecommendations(posture *compliance.SecurityPosture) []compliance.Recommendation {
	recommendations := []compliance.Recommendation{}

	// Critical vulnerabilities
	if posture.Vulnerabilities.Critical > 0 {
		recommendations = append(recommendations, compliance.Recommendation{
			ID:          fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Priority:    "critical",
			Category:    "vulnerability_management",
			Title:       "Address Critical Vulnerabilities",
			Description: fmt.Sprintf("You have %d critical vulnerabilities that require immediate attention", posture.Vulnerabilities.Critical),
			Impact:      "High risk of security breach",
			Effort:      "Varies by vulnerability",
			References:  []string{"https://nvd.nist.gov/"},
			CreatedAt:   time.Now(),
		})
	}

	// High vulnerabilities
	if posture.Vulnerabilities.High > 5 {
		recommendations = append(recommendations, compliance.Recommendation{
			ID:          fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Priority:    "high",
			Category:    "vulnerability_management",
			Title:       "Reduce High Severity Vulnerabilities",
			Description: fmt.Sprintf("You have %d high severity vulnerabilities", posture.Vulnerabilities.High),
			Impact:      "Elevated security risk",
			Effort:      "Medium",
			References:  []string{},
			CreatedAt:   time.Now(),
		})
	}

	// Low compliance scores
	for framework, score := range posture.Compliance {
		if score < 90.0 {
			recommendations = append(recommendations, compliance.Recommendation{
				ID:          fmt.Sprintf("rec-%d", time.Now().UnixNano()),
				Priority:    "high",
				Category:    "compliance",
				Title:       fmt.Sprintf("Improve %s Compliance", framework),
				Description: fmt.Sprintf("Current compliance score: %.1f%%. Target: 95%%+", score),
				Impact:      "Compliance certification risk",
				Effort:      "Medium to High",
				References:  []string{},
				CreatedAt:   time.Now(),
			})
		}
	}

	// Weak control categories
	for category, score := range posture.Controls {
		if score < 85.0 {
			recommendations = append(recommendations, compliance.Recommendation{
				ID:          fmt.Sprintf("rec-%d", time.Now().UnixNano()),
				Priority:    "medium",
				Category:    "security_controls",
				Title:       fmt.Sprintf("Strengthen %s Controls", category),
				Description: fmt.Sprintf("Current score: %.1f%%. Requires improvement", score),
				Impact:      "Security control gap",
				Effort:      "Medium",
				References:  []string{},
				CreatedAt:   time.Now(),
			})
		}
	}

	// Declining trends
	if posture.Trends.Direction == "declining" {
		recommendations = append(recommendations, compliance.Recommendation{
			ID:          fmt.Sprintf("rec-%d", time.Now().UnixNano()),
			Priority:    "high",
			Category:    "security_posture",
			Title:       "Address Declining Security Posture",
			Description: "Security posture has declined in recent period",
			Impact:      "Increasing security risk",
			Effort:      "High",
			References:  []string{},
			CreatedAt:   time.Now(),
		})
	}

	return recommendations
}

// GetCurrentPosture returns current security posture
func (e *Engine) GetCurrentPosture() (*compliance.SecurityPosture, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.currentPosture == nil {
		return nil, fmt.Errorf("no posture assessment available")
	}

	return e.currentPosture, nil
}

// GetVulnerabilities returns vulnerabilities by severity
func (e *Engine) GetVulnerabilities(severity string) ([]*Vulnerability, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	vulns := []*Vulnerability{}
	for _, vuln := range e.vulnerabilities {
		if severity == "" || vuln.Severity == severity {
			vulns = append(vulns, vuln)
		}
	}

	return vulns, nil
}

// MockScanner implementations

type NetworkScanner struct{}

func (s *NetworkScanner) Scan(ctx context.Context) ([]*Vulnerability, error) {
	// Mock network vulnerabilities
	return []*Vulnerability{
		{
			ID:          "net-001",
			Severity:    "high",
			Type:        "network",
			Title:       "Open port on production server",
			Description: "Port 23 (telnet) open on production server",
			Affected:    []string{"server-prod-01"},
			Remediation: "Close unused ports and disable telnet",
			DetectedAt:  time.Now(),
			Status:      "open",
		},
	}, nil
}

func (s *NetworkScanner) GetType() string {
	return "network"
}

type HostScanner struct{}

func (s *HostScanner) Scan(ctx context.Context) ([]*Vulnerability, error) {
	return []*Vulnerability{}, nil
}

func (s *HostScanner) GetType() string {
	return "host"
}

type ContainerScanner struct{}

func (s *ContainerScanner) Scan(ctx context.Context) ([]*Vulnerability, error) {
	return []*Vulnerability{
		{
			ID:          "cont-001",
			Severity:    "medium",
			Type:        "container",
			Title:       "Container running as root",
			Description: "Container nginx:latest running with root privileges",
			Affected:    []string{"container-web-01"},
			Remediation: "Configure container to run as non-root user",
			DetectedAt:  time.Now(),
			Status:      "open",
		},
	}, nil
}

func (s *ContainerScanner) GetType() string {
	return "container"
}

type ConfigScanner struct{}

func (s *ConfigScanner) Scan(ctx context.Context) ([]*Vulnerability, error) {
	return []*Vulnerability{}, nil
}

func (s *ConfigScanner) GetType() string {
	return "config"
}

type DependencyScanner struct{}

func (s *DependencyScanner) Scan(ctx context.Context) ([]*Vulnerability, error) {
	return []*Vulnerability{
		{
			ID:          "dep-001",
			Severity:    "critical",
			Type:        "dependency",
			Title:       "Critical vulnerability in log4j",
			Description: "Apache Log4j RCE vulnerability (CVE-2021-44228)",
			Affected:    []string{"app-backend"},
			CVE:         "CVE-2021-44228",
			CVSS:        10.0,
			Remediation: "Upgrade log4j to version 2.17.1 or later",
			DetectedAt:  time.Now(),
			Status:      "open",
		},
	}, nil
}

func (s *DependencyScanner) GetType() string {
	return "dependency"
}
