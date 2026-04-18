package handlers

import "time"

type ScanType string

const (
	ScanTypeSecrets      ScanType = "secrets"
	ScanTypeDependencies ScanType = "dependencies"
	ScanTypeFilesystem   ScanType = "filesystem"
)

type FindingSeverity string

const (
	SeverityCritical FindingSeverity = "critical"
	SeverityHigh     FindingSeverity = "high"
	SeverityMedium   FindingSeverity = "medium"
	SeverityLow      FindingSeverity = "low"
	SeverityInfo     FindingSeverity = "info"
)

type SecurityFinding struct {
	ID             string                 `json:"id"`
	Category       string                 `json:"category"`
	Severity       FindingSeverity        `json:"severity"`
	Target         string                 `json:"target"`
	Title          string                 `json:"title"`
	Description    string                 `json:"description"`
	Recommendation string                 `json:"recommendation,omitempty"`
	DiscoveredAt   time.Time              `json:"discovered_at"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

type ScanSummary struct {
	Total    int `json:"total"`
	Critical int `json:"critical"`
	High     int `json:"high"`
	Medium   int `json:"medium"`
	Low      int `json:"low"`
	Info     int `json:"info"`
}

type ScanResults struct {
	ScanID      string            `json:"scan_id"`
	Status      string            `json:"status"`
	Targets     []string          `json:"targets"`
	ScanTypes   []ScanType        `json:"scan_types"`
	Findings    []SecurityFinding `json:"findings"`
	Summary     ScanSummary       `json:"summary"`
	StartedAt   time.Time         `json:"started_at"`
	CompletedAt time.Time         `json:"completed_at"`
}

type SecurityThreat struct {
	ID          string                 `json:"id"`
	Severity    FindingSeverity        `json:"severity"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type SecurityEvent struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Severity  FindingSeverity        `json:"severity"`
	Message   string                 `json:"message"`
	Source    string                 `json:"source"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}
