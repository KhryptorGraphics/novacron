// Package threat_intel implements threat intelligence feed integration
package threat_intel

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// FeedType represents the type of threat intelligence feed
type FeedType string

const (
	FeedMISP  FeedType = "misp"  // Malware Information Sharing Platform
	FeedSTIX  FeedType = "stix"  // Structured Threat Information Expression
	FeedTAXII FeedType = "taxii" // Trusted Automated eXchange of Indicator Information
	FeedOTX   FeedType = "otx"   // Open Threat Exchange
)

// IndicatorType represents the type of indicator of compromise
type IndicatorType string

const (
	IndicatorIP           IndicatorType = "ip"
	IndicatorDomain       IndicatorType = "domain"
	IndicatorURL          IndicatorType = "url"
	IndicatorFileHash     IndicatorType = "filehash"
	IndicatorEmail        IndicatorType = "email"
	IndicatorCVE          IndicatorType = "cve"
)

// ThreatLevel represents the threat severity level
type ThreatLevel string

const (
	ThreatLevelLow      ThreatLevel = "low"
	ThreatLevelMedium   ThreatLevel = "medium"
	ThreatLevelHigh     ThreatLevel = "high"
	ThreatLevelCritical ThreatLevel = "critical"
)

// Indicator represents an indicator of compromise (IoC)
type Indicator struct {
	ID          string
	Type        IndicatorType
	Value       string
	ThreatLevel ThreatLevel
	Confidence  float64
	Source      FeedType
	Description string
	Tags        []string
	FirstSeen   time.Time
	LastSeen    time.Time
	Metadata    map[string]interface{}
}

// ThreatActor represents a threat actor
type ThreatActor struct {
	ID          string
	Name        string
	Aliases     []string
	Country     string
	Motivation  string
	Tactics     []string
	Techniques  []string
	Procedures  []string
	Indicators  []string
	FirstSeen   time.Time
	LastSeen    time.Time
	Metadata    map[string]interface{}
}

// Vulnerability represents a security vulnerability
type Vulnerability struct {
	ID          string
	CVE         string
	CVSS        float64
	Severity    ThreatLevel
	Description string
	Affected    []string
	Published   time.Time
	Updated     time.Time
	Exploited   bool
	Patch       string
	Metadata    map[string]interface{}
}

// Feed manages threat intelligence feeds
type Feed struct {
	feeds               map[FeedType]*FeedConfig
	indicators          map[string]*Indicator
	actors              map[string]*ThreatActor
	vulnerabilities     map[string]*Vulnerability
	updateInterval      time.Duration
	lastUpdate          time.Time
	mu                  sync.RWMutex
	totalIndicators     int64
	matchedIndicators   int64
}

// FeedConfig represents feed configuration
type FeedConfig struct {
	Type       FeedType
	Enabled    bool
	Endpoint   string
	APIKey     string
	UpdateFreq time.Duration
	LastUpdate time.Time
}

// NewFeed creates a new threat intelligence feed
func NewFeed(updateInterval time.Duration) *Feed {
	return &Feed{
		feeds:           make(map[FeedType]*FeedConfig),
		indicators:      make(map[string]*Indicator),
		actors:          make(map[string]*ThreatActor),
		vulnerabilities: make(map[string]*Vulnerability),
		updateInterval:  updateInterval,
		lastUpdate:      time.Now(),
	}
}

// AddFeed adds a threat intelligence feed
func (f *Feed) AddFeed(feedType FeedType, endpoint, apiKey string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	config := &FeedConfig{
		Type:       feedType,
		Enabled:    true,
		Endpoint:   endpoint,
		APIKey:     apiKey,
		UpdateFreq: f.updateInterval,
		LastUpdate: time.Time{},
	}

	f.feeds[feedType] = config
	return nil
}

// RemoveFeed removes a threat intelligence feed
func (f *Feed) RemoveFeed(feedType FeedType) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.feeds, feedType)
}

// UpdateFeed updates threat intelligence from a specific feed
func (f *Feed) UpdateFeed(ctx context.Context, feedType FeedType) error {
	f.mu.RLock()
	config, exists := f.feeds[feedType]
	f.mu.RUnlock()

	if !exists {
		return fmt.Errorf("feed not found: %s", feedType)
	}

	if !config.Enabled {
		return fmt.Errorf("feed disabled: %s", feedType)
	}

	// Simulate feed update based on type
	var indicators []*Indicator
	var err error

	switch feedType {
	case FeedMISP:
		indicators, err = f.updateMISPFeed(ctx, config)
	case FeedSTIX:
		indicators, err = f.updateSTIXFeed(ctx, config)
	case FeedTAXII:
		indicators, err = f.updateTAXIIFeed(ctx, config)
	case FeedOTX:
		indicators, err = f.updateOTXFeed(ctx, config)
	default:
		return fmt.Errorf("unsupported feed type: %s", feedType)
	}

	if err != nil {
		return fmt.Errorf("feed update failed: %w", err)
	}

	// Store indicators
	f.mu.Lock()
	for _, indicator := range indicators {
		f.indicators[indicator.ID] = indicator
		f.totalIndicators++
	}
	config.LastUpdate = time.Now()
	f.lastUpdate = time.Now()
	f.mu.Unlock()

	return nil
}

// updateMISPFeed updates from MISP feed
func (f *Feed) updateMISPFeed(ctx context.Context, config *FeedConfig) ([]*Indicator, error) {
	// Simulate MISP feed update (in production, would use actual MISP API)
	indicators := []*Indicator{
		{
			ID:          generateIndicatorID(),
			Type:        IndicatorIP,
			Value:       "192.0.2.1",
			ThreatLevel: ThreatLevelHigh,
			Confidence:  0.9,
			Source:      FeedMISP,
			Description: "Malicious IP address",
			Tags:        []string{"malware", "botnet"},
			FirstSeen:   time.Now().Add(-24 * time.Hour),
			LastSeen:    time.Now(),
			Metadata:    make(map[string]interface{}),
		},
		{
			ID:          generateIndicatorID(),
			Type:        IndicatorDomain,
			Value:       "malicious.example.com",
			ThreatLevel: ThreatLevelCritical,
			Confidence:  0.95,
			Source:      FeedMISP,
			Description: "C2 domain",
			Tags:        []string{"c2", "apt"},
			FirstSeen:   time.Now().Add(-48 * time.Hour),
			LastSeen:    time.Now(),
			Metadata:    make(map[string]interface{}),
		},
	}

	return indicators, nil
}

// updateSTIXFeed updates from STIX feed
func (f *Feed) updateSTIXFeed(ctx context.Context, config *FeedConfig) ([]*Indicator, error) {
	// Simulate STIX feed update
	indicators := []*Indicator{
		{
			ID:          generateIndicatorID(),
			Type:        IndicatorFileHash,
			Value:       "a1b2c3d4e5f6",
			ThreatLevel: ThreatLevelHigh,
			Confidence:  0.85,
			Source:      FeedSTIX,
			Description: "Malware hash",
			Tags:        []string{"ransomware"},
			FirstSeen:   time.Now().Add(-12 * time.Hour),
			LastSeen:    time.Now(),
			Metadata:    make(map[string]interface{}),
		},
	}

	return indicators, nil
}

// updateTAXIIFeed updates from TAXII feed
func (f *Feed) updateTAXIIFeed(ctx context.Context, config *FeedConfig) ([]*Indicator, error) {
	// Simulate TAXII feed update
	indicators := []*Indicator{}
	return indicators, nil
}

// updateOTXFeed updates from OTX feed
func (f *Feed) updateOTXFeed(ctx context.Context, config *FeedConfig) ([]*Indicator, error) {
	// Simulate OTX feed update
	indicators := []*Indicator{}
	return indicators, nil
}

// CheckIndicator checks if an indicator matches known threats
func (f *Feed) CheckIndicator(ctx context.Context, indicatorValue string) (bool, float64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, indicator := range f.indicators {
		if indicator.Value == indicatorValue {
			f.matchedIndicators++
			return true, indicator.Confidence, nil
		}
	}

	return false, 0, nil
}

// GetThreatScore calculates threat score for a source
func (f *Feed) GetThreatScore(ctx context.Context, source string) (float64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	score := 0.0
	count := 0

	for _, indicator := range f.indicators {
		// Simple matching (in production, would use more sophisticated matching)
		if indicator.Value == source {
			switch indicator.ThreatLevel {
			case ThreatLevelCritical:
				score += 1.0
			case ThreatLevelHigh:
				score += 0.75
			case ThreatLevelMedium:
				score += 0.5
			case ThreatLevelLow:
				score += 0.25
			}
			count++
		}
	}

	if count > 0 {
		return score / float64(count), nil
	}

	return 0, nil
}

// AddVulnerability adds a vulnerability to the feed
func (f *Feed) AddVulnerability(vuln *Vulnerability) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if vuln.ID == "" {
		vuln.ID = generateVulnerabilityID()
	}

	f.vulnerabilities[vuln.ID] = vuln
	return nil
}

// GetVulnerability retrieves a vulnerability by CVE
func (f *Feed) GetVulnerability(cve string) (*Vulnerability, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, vuln := range f.vulnerabilities {
		if vuln.CVE == cve {
			return vuln, nil
		}
	}

	return nil, fmt.Errorf("vulnerability not found: %s", cve)
}

// ScanVulnerabilities scans for vulnerabilities in a target
func (f *Feed) ScanVulnerabilities(target string) ([]*Vulnerability, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	var matches []*Vulnerability

	for _, vuln := range f.vulnerabilities {
		for _, affected := range vuln.Affected {
			if affected == target {
				matches = append(matches, vuln)
			}
		}
	}

	return matches, nil
}

// AddThreatActor adds a threat actor to the feed
func (f *Feed) AddThreatActor(actor *ThreatActor) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if actor.ID == "" {
		actor.ID = generateActorID()
	}

	f.actors[actor.ID] = actor
	return nil
}

// GetThreatActor retrieves a threat actor
func (f *Feed) GetThreatActor(actorID string) (*ThreatActor, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	actor, exists := f.actors[actorID]
	if !exists {
		return nil, fmt.Errorf("threat actor not found: %s", actorID)
	}

	return actor, nil
}

// UpdateAllFeeds updates all enabled feeds
func (f *Feed) UpdateAllFeeds(ctx context.Context) error {
	f.mu.RLock()
	feedTypes := make([]FeedType, 0, len(f.feeds))
	for feedType, config := range f.feeds {
		if config.Enabled {
			feedTypes = append(feedTypes, feedType)
		}
	}
	f.mu.RUnlock()

	for _, feedType := range feedTypes {
		if err := f.UpdateFeed(ctx, feedType); err != nil {
			return fmt.Errorf("failed to update feed %s: %w", feedType, err)
		}
	}

	return nil
}

// AutoUpdate performs automatic periodic updates
func (f *Feed) AutoUpdate(ctx context.Context) {
	ticker := time.NewTicker(f.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			f.UpdateAllFeeds(ctx)
		}
	}
}

// GetMetrics returns threat intelligence metrics
func (f *Feed) GetMetrics() map[string]interface{} {
	f.mu.RLock()
	defer f.mu.RUnlock()

	enabledFeeds := 0
	for _, config := range f.feeds {
		if config.Enabled {
			enabledFeeds++
		}
	}

	indicatorsByType := make(map[IndicatorType]int)
	indicatorsByLevel := make(map[ThreatLevel]int)

	for _, indicator := range f.indicators {
		indicatorsByType[indicator.Type]++
		indicatorsByLevel[indicator.ThreatLevel]++
	}

	vulnerabilityBySeverity := make(map[ThreatLevel]int)
	for _, vuln := range f.vulnerabilities {
		vulnerabilityBySeverity[vuln.Severity]++
	}

	matchRate := 0.0
	if f.totalIndicators > 0 {
		matchRate = float64(f.matchedIndicators) / float64(f.totalIndicators)
	}

	return map[string]interface{}{
		"total_feeds":              len(f.feeds),
		"enabled_feeds":            enabledFeeds,
		"total_indicators":         len(f.indicators),
		"total_actors":             len(f.actors),
		"total_vulnerabilities":    len(f.vulnerabilities),
		"indicators_by_type":       indicatorsByType,
		"indicators_by_level":      indicatorsByLevel,
		"vulnerability_by_severity": vulnerabilityBySeverity,
		"matched_indicators":       f.matchedIndicators,
		"match_rate":               matchRate,
		"last_update":              f.lastUpdate,
		"update_interval_hours":    f.updateInterval.Hours(),
	}
}

// Helper functions

func generateIndicatorID() string {
	b := make([]byte, 16)
	// Generate random bytes
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("ind-%s", hex.EncodeToString(b))
}

func generateVulnerabilityID() string {
	b := make([]byte, 16)
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("vuln-%s", hex.EncodeToString(b))
}

func generateActorID() string {
	b := make([]byte, 16)
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("actor-%s", hex.EncodeToString(b))
}
