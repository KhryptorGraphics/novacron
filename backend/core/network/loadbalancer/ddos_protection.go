package loadbalancer

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// DDoSProtection provides comprehensive DDoS protection and mitigation
type DDoSProtection struct {
	// Configuration
	config           DDoSProtectionConfig
	
	// Rate limiters
	ipRateLimiter    *IPRateLimiter
	geoRateLimiter   *GeoRateLimiter
	asnRateLimiter   *ASNRateLimiter
	
	// Traffic analysis
	trafficAnalyzer  *TrafficAnalyzer
	behaviorAnalyzer *BehaviorAnalyzer
	
	// Attack detection
	attackDetector   *AttackDetector
	
	// Mitigation strategies
	mitigationEngine *MitigationEngine
	
	// Blacklists and whitelists
	ipBlacklist      *IPBlacklist
	ipWhitelist      *IPWhitelist
	geoBlacklist     *GeoBlacklist
	
	// Challenge systems
	challengeSystem  *ChallengeSystem
	
	// Metrics and monitoring
	metrics          *DDoSMetrics
	metricsMutex     sync.RWMutex
	
	// Runtime state
	ctx              context.Context
	cancel           context.CancelFunc
	initialized      bool
	activeMitigations map[string]*ActiveMitigation
	mitigationMutex  sync.RWMutex
}

// DDoSProtectionConfig holds DDoS protection configuration
type DDoSProtectionConfig struct {
	// Basic protection settings
	EnableProtection        bool              `json:"enable_protection"`
	ProtectionMode          ProtectionMode    `json:"protection_mode"`
	SensitivityLevel        SensitivityLevel  `json:"sensitivity_level"`
	
	// Rate limiting configuration
	GlobalRateLimit         int               `json:"global_rate_limit"`
	IPRateLimit             int               `json:"ip_rate_limit"`
	SubnetRateLimit         int               `json:"subnet_rate_limit"`
	CountryRateLimit        int               `json:"country_rate_limit"`
	ASNRateLimit            int               `json:"asn_rate_limit"`
	RateLimitWindow         time.Duration     `json:"rate_limit_window"`
	
	// Connection limits
	MaxConnectionsPerIP     int               `json:"max_connections_per_ip"`
	MaxConnectionsPerSubnet int               `json:"max_connections_per_subnet"`
	ConnectionTimeout       time.Duration     `json:"connection_timeout"`
	
	// SYN flood protection
	EnableSYNProtection     bool              `json:"enable_syn_protection"`
	SYNRateLimit           int               `json:"syn_rate_limit"`
	SYNBurstSize           int               `json:"syn_burst_size"`
	EnableSYNCookies       bool              `json:"enable_syn_cookies"`
	
	// Slowloris protection
	EnableSlowlorisProtection bool            `json:"enable_slowloris_protection"`
	RequestTimeout          time.Duration     `json:"request_timeout"`
	HeaderTimeout           time.Duration     `json:"header_timeout"`
	MaxHeaderSize           int               `json:"max_header_size"`
	
	// HTTP-specific protection
	UserAgentBlacklist      []string          `json:"user_agent_blacklist"`
	RefererBlacklist        []string          `json:"referer_blacklist"`
	MaxRequestSize          int64             `json:"max_request_size"`
	MaxURLLength            int               `json:"max_url_length"`
	
	// Behavioral analysis
	EnableBehaviorAnalysis  bool              `json:"enable_behavior_analysis"`
	BehaviorWindow          time.Duration     `json:"behavior_window"`
	AnomalyThreshold        float64           `json:"anomaly_threshold"`
	
	// Geographical protection
	EnableGeoBlocking       bool              `json:"enable_geo_blocking"`
	BlockedCountries        []string          `json:"blocked_countries"`
	AllowedCountries        []string          `json:"allowed_countries"`
	
	// Challenge systems
	EnableChallenges        bool              `json:"enable_challenges"`
	ChallengeTypes          []ChallengeType   `json:"challenge_types"`
	ChallengeTimeout        time.Duration     `json:"challenge_timeout"`
	ChallengeRetries        int               `json:"challenge_retries"`
	
	// Mitigation settings
	AutoMitigation          bool              `json:"auto_mitigation"`
	MitigationDuration      time.Duration     `json:"mitigation_duration"`
	MitigationCooldown      time.Duration     `json:"mitigation_cooldown"`
	
	// Monitoring and alerting
	EnableMetrics           bool              `json:"enable_metrics"`
	MetricsInterval         time.Duration     `json:"metrics_interval"`
	AlertThresholds         AlertThresholds   `json:"alert_thresholds"`
}

// Types and enums
type ProtectionMode string
type SensitivityLevel string
type ChallengeType string
type AttackType string
type MitigationType string

const (
	ProtectionModeMonitoring ProtectionMode = "monitoring"
	ProtectionModeActive     ProtectionMode = "active"
	ProtectionModeStrict     ProtectionMode = "strict"
	
	SensitivityLow           SensitivityLevel = "low"
	SensitivityMedium        SensitivityLevel = "medium"
	SensitivityHigh          SensitivityLevel = "high"
	
	ChallengeCAPTCHA         ChallengeType = "captcha"
	ChallengeJavaScript      ChallengeType = "javascript"
	ChallengeCookie          ChallengeType = "cookie"
	ChallengeProofOfWork     ChallengeType = "proof_of_work"
	
	AttackTypeVolumetric     AttackType = "volumetric"
	AttackTypeProtocol       AttackType = "protocol"
	AttackTypeApplication    AttackType = "application"
	AttackTypeSYNFlood       AttackType = "syn_flood"
	AttackTypeSlowloris      AttackType = "slowloris"
	AttackTypeHTTPFlood      AttackType = "http_flood"
	AttackTypeDNSAmplification AttackType = "dns_amplification"
	
	MitigationTypeRateLimit  MitigationType = "rate_limit"
	MitigationTypeBlacklist  MitigationType = "blacklist"
	MitigationTypeChallenge  MitigationType = "challenge"
	MitigationTypeTarpit     MitigationType = "tarpit"
	MitigationTypeDrop       MitigationType = "drop"
)

// IPRateLimiter manages per-IP rate limiting
type IPRateLimiter struct {
	limits    map[string]*RateLimitEntry
	mutex     sync.RWMutex
	maxEntries int
	window    time.Duration
}

// RateLimitEntry represents a rate limit entry for an IP/entity
type RateLimitEntry struct {
	Requests      int64     `json:"requests"`
	LastReset     time.Time `json:"last_reset"`
	FirstRequest  time.Time `json:"first_request"`
	Blocked       bool      `json:"blocked"`
	BlockedUntil  time.Time `json:"blocked_until"`
	Violations    int       `json:"violations"`
}

// GeoRateLimiter manages geographical rate limiting
type GeoRateLimiter struct {
	countryLimits map[string]*RateLimitEntry
	mutex         sync.RWMutex
}

// ASNRateLimiter manages Autonomous System Number rate limiting
type ASNRateLimiter struct {
	asnLimits map[string]*RateLimitEntry
	mutex     sync.RWMutex
}

// TrafficAnalyzer analyzes traffic patterns for anomaly detection
type TrafficAnalyzer struct {
	patterns     map[string]*TrafficPattern
	baselines    map[string]*TrafficBaseline
	mutex        sync.RWMutex
	windowSize   time.Duration
}

// TrafficPattern represents traffic patterns for analysis
type TrafficPattern struct {
	RequestsPerSecond    float64           `json:"requests_per_second"`
	UniqueIPs            int               `json:"unique_ips"`
	TopPaths             map[string]int    `json:"top_paths"`
	TopUserAgents        map[string]int    `json:"top_user_agents"`
	TopReferers          map[string]int    `json:"top_referers"`
	ResponseCodes        map[int]int       `json:"response_codes"`
	AverageRequestSize   float64           `json:"average_request_size"`
	GeographicDistribution map[string]int  `json:"geographic_distribution"`
	Timestamp            time.Time         `json:"timestamp"`
}

// TrafficBaseline represents normal traffic baselines
type TrafficBaseline struct {
	BaselineRPS          float64           `json:"baseline_rps"`
	BaselineUniqueIPs    int               `json:"baseline_unique_ips"`
	BaselineGeoDistribution map[string]float64 `json:"baseline_geo_distribution"`
	CalculatedAt         time.Time         `json:"calculated_at"`
	ValidUntil           time.Time         `json:"valid_until"`
}

// BehaviorAnalyzer analyzes client behavior patterns
type BehaviorAnalyzer struct {
	clientProfiles map[string]*ClientProfile
	mutex          sync.RWMutex
}

// ClientProfile represents a client's behavioral profile
type ClientProfile struct {
	IP                   string            `json:"ip"`
	FirstSeen            time.Time         `json:"first_seen"`
	LastSeen             time.Time         `json:"last_seen"`
	TotalRequests        int64             `json:"total_requests"`
	RequestsPerSecond    float64           `json:"requests_per_second"`
	UniqueEndpoints      []string          `json:"unique_endpoints"`
	UserAgents           []string          `json:"user_agents"`
	Countries            []string          `json:"countries"`
	SuspiciousActivity   []SuspiciousActivity `json:"suspicious_activity"`
	ThreatScore          float64           `json:"threat_score"`
}

// SuspiciousActivity represents detected suspicious behavior
type SuspiciousActivity struct {
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Severity    string    `json:"severity"`
	DetectedAt  time.Time `json:"detected_at"`
	Evidence    map[string]interface{} `json:"evidence"`
}

// AttackDetector detects various types of DDoS attacks
type AttackDetector struct {
	detectors map[AttackType]AttackDetectorFunc
	config    *DDoSProtectionConfig
}

// AttackDetectorFunc is a function that detects specific attack types
type AttackDetectorFunc func(pattern *TrafficPattern, baseline *TrafficBaseline) (bool, float64)

// MitigationEngine manages active mitigations
type MitigationEngine struct {
	strategies    map[MitigationType]MitigationStrategy
	activeMitigations map[string]*ActiveMitigation
	mutex         sync.RWMutex
}

// MitigationStrategy defines how to apply a mitigation
type MitigationStrategy interface {
	Apply(target string, config map[string]interface{}) error
	Remove(target string) error
	IsActive(target string) bool
}

// ActiveMitigation represents an active mitigation
type ActiveMitigation struct {
	ID             string                 `json:"id"`
	Type           MitigationType         `json:"type"`
	Target         string                 `json:"target"`
	AttackType     AttackType             `json:"attack_type"`
	Config         map[string]interface{} `json:"config"`
	StartedAt      time.Time              `json:"started_at"`
	ExpiresAt      time.Time              `json:"expires_at"`
	RequestsBlocked int64                 `json:"requests_blocked"`
	Effectiveness  float64                `json:"effectiveness"`
}

// IPBlacklist manages IP-based blacklists
type IPBlacklist struct {
	entries     map[string]*BlacklistEntry
	mutex       sync.RWMutex
	maxEntries  int
}

// IPWhitelist manages IP-based whitelists
type IPWhitelist struct {
	entries map[string]*WhitelistEntry
	mutex   sync.RWMutex
}

// GeoBlacklist manages geography-based blacklists
type GeoBlacklist struct {
	countries map[string]bool
	mutex     sync.RWMutex
}

// BlacklistEntry represents a blacklisted entity
type BlacklistEntry struct {
	IP          string    `json:"ip"`
	Reason      string    `json:"reason"`
	AddedAt     time.Time `json:"added_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	Violations  int       `json:"violations"`
}

// WhitelistEntry represents a whitelisted entity
type WhitelistEntry struct {
	IP          string    `json:"ip"`
	Reason      string    `json:"reason"`
	AddedAt     time.Time `json:"added_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

// ChallengeSystem manages client challenges
type ChallengeSystem struct {
	challenges    map[string]*Challenge
	mutex         sync.RWMutex
	config        *DDoSProtectionConfig
}

// Challenge represents a client challenge
type Challenge struct {
	ID          string        `json:"id"`
	IP          string        `json:"ip"`
	Type        ChallengeType `json:"type"`
	Data        string        `json:"data"`
	Solution    string        `json:"solution"`
	CreatedAt   time.Time     `json:"created_at"`
	ExpiresAt   time.Time     `json:"expires_at"`
	Attempts    int           `json:"attempts"`
	Solved      bool          `json:"solved"`
}

// DDoSMetrics holds DDoS protection metrics
type DDoSMetrics struct {
	TotalRequests           int64                    `json:"total_requests"`
	BlockedRequests         int64                    `json:"blocked_requests"`
	ChallengedRequests      int64                    `json:"challenged_requests"`
	AttacksDetected         int64                    `json:"attacks_detected"`
	AttacksByType           map[AttackType]int64     `json:"attacks_by_type"`
	MitigationsActive       int64                    `json:"mitigations_active"`
	MitigationsByType       map[MitigationType]int64 `json:"mitigations_by_type"`
	TopAttackerIPs          map[string]int64         `json:"top_attacker_ips"`
	TopAttackerCountries    map[string]int64         `json:"top_attacker_countries"`
	AverageRequestsPerSecond float64                 `json:"average_requests_per_second"`
	PeakRequestsPerSecond   float64                  `json:"peak_requests_per_second"`
	LastUpdated             time.Time                `json:"last_updated"`
}

// AlertThresholds defines thresholds for alerting
type AlertThresholds struct {
	RequestsPerSecond    float64 `json:"requests_per_second"`
	BlockedRequestsRatio float64 `json:"blocked_requests_ratio"`
	NewAttackDetected    bool    `json:"new_attack_detected"`
	MitigationFailed     bool    `json:"mitigation_failed"`
}

// NewDDoSProtection creates a new DDoS protection system
func NewDDoSProtection(config DDoSProtectionConfig) *DDoSProtection {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DDoSProtection{
		config: config,
		metrics: &DDoSMetrics{
			AttacksByType:     make(map[AttackType]int64),
			MitigationsByType: make(map[MitigationType]int64),
			TopAttackerIPs:    make(map[string]int64),
			TopAttackerCountries: make(map[string]int64),
			LastUpdated:       time.Now(),
		},
		activeMitigations: make(map[string]*ActiveMitigation),
		ctx:               ctx,
		cancel:            cancel,
	}
}

// Start initializes and starts the DDoS protection system
func (ddos *DDoSProtection) Start() error {
	if ddos.initialized {
		return fmt.Errorf("DDoS protection already started")
	}
	
	// Initialize rate limiters
	ddos.ipRateLimiter = &IPRateLimiter{
		limits:     make(map[string]*RateLimitEntry),
		maxEntries: 100000, // Configurable
		window:     ddos.config.RateLimitWindow,
	}
	
	ddos.geoRateLimiter = &GeoRateLimiter{
		countryLimits: make(map[string]*RateLimitEntry),
	}
	
	ddos.asnRateLimiter = &ASNRateLimiter{
		asnLimits: make(map[string]*RateLimitEntry),
	}
	
	// Initialize traffic analyzer
	ddos.trafficAnalyzer = &TrafficAnalyzer{
		patterns:   make(map[string]*TrafficPattern),
		baselines:  make(map[string]*TrafficBaseline),
		windowSize: ddos.config.BehaviorWindow,
	}
	
	// Initialize behavior analyzer
	ddos.behaviorAnalyzer = &BehaviorAnalyzer{
		clientProfiles: make(map[string]*ClientProfile),
	}
	
	// Initialize attack detector
	ddos.attackDetector = &AttackDetector{
		detectors: make(map[AttackType]AttackDetectorFunc),
		config:    &ddos.config,
	}
	ddos.registerAttackDetectors()
	
	// Initialize mitigation engine
	ddos.mitigationEngine = &MitigationEngine{
		strategies:        make(map[MitigationType]MitigationStrategy),
		activeMitigations: make(map[string]*ActiveMitigation),
	}
	ddos.registerMitigationStrategies()
	
	// Initialize blacklists and whitelists
	ddos.ipBlacklist = &IPBlacklist{
		entries:    make(map[string]*BlacklistEntry),
		maxEntries: 10000, // Configurable
	}
	
	ddos.ipWhitelist = &IPWhitelist{
		entries: make(map[string]*WhitelistEntry),
	}
	
	ddos.geoBlacklist = &GeoBlacklist{
		countries: make(map[string]bool),
	}
	
	// Initialize challenge system
	if ddos.config.EnableChallenges {
		ddos.challengeSystem = &ChallengeSystem{
			challenges: make(map[string]*Challenge),
			config:     &ddos.config,
		}
	}
	
	// Populate geo blacklist
	for _, country := range ddos.config.BlockedCountries {
		ddos.geoBlacklist.countries[country] = true
	}
	
	// Start background tasks
	go ddos.trafficAnalysisLoop()
	go ddos.attackDetectionLoop()
	go ddos.cleanupLoop()
	
	if ddos.config.EnableMetrics {
		go ddos.metricsCollectionLoop()
	}
	
	ddos.initialized = true
	return nil
}

// Stop stops the DDoS protection system
func (ddos *DDoSProtection) Stop() error {
	ddos.cancel()
	
	// Remove all active mitigations
	ddos.mitigationMutex.Lock()
	for id, mitigation := range ddos.activeMitigations {
		ddos.mitigationEngine.removeMitigation(id, mitigation.Type)
	}
	ddos.mitigationMutex.Unlock()
	
	ddos.initialized = false
	return nil
}

// ProcessRequest processes an incoming request through DDoS protection
func (ddos *DDoSProtection) ProcessRequest(req *http.Request) (*DDoSDecision, error) {
	clientIP := getClientIPFromRequest(req)
	
	// Check whitelist first
	if ddos.isWhitelisted(clientIP) {
		return &DDoSDecision{
			Action:    ActionAllow,
			Reason:    "IP is whitelisted",
			Challenge: nil,
		}, nil
	}
	
	// Check blacklist
	if ddos.isBlacklisted(clientIP) {
		atomic.AddInt64(&ddos.metrics.BlockedRequests, 1)
		return &DDoSDecision{
			Action: ActionBlock,
			Reason: "IP is blacklisted",
		}, nil
	}
	
	// Check geographical restrictions
	if ddos.config.EnableGeoBlocking {
		if blocked, reason := ddos.checkGeoBlocking(clientIP); blocked {
			atomic.AddInt64(&ddos.metrics.BlockedRequests, 1)
			return &DDoSDecision{
				Action: ActionBlock,
				Reason: reason,
			}, nil
		}
	}
	
	// Check rate limits
	if decision := ddos.checkRateLimits(clientIP, req); decision.Action != ActionAllow {
		if decision.Action == ActionBlock {
			atomic.AddInt64(&ddos.metrics.BlockedRequests, 1)
		}
		return decision, nil
	}
	
	// Check for suspicious patterns
	if ddos.config.EnableBehaviorAnalysis {
		if decision := ddos.checkBehaviorAnalysis(clientIP, req); decision.Action != ActionAllow {
			if decision.Action == ActionBlock {
				atomic.AddInt64(&ddos.metrics.BlockedRequests, 1)
			} else if decision.Action == ActionChallenge {
				atomic.AddInt64(&ddos.metrics.ChallengedRequests, 1)
			}
			return decision, nil
		}
	}
	
	// Update traffic analysis
	ddos.updateTrafficAnalysis(req)
	
	// Update client profile
	ddos.updateClientProfile(clientIP, req)
	
	atomic.AddInt64(&ddos.metrics.TotalRequests, 1)
	
	return &DDoSDecision{
		Action: ActionAllow,
		Reason: "Request passed all checks",
	}, nil
}

// DDoSDecision represents the decision for a request
type DDoSDecision struct {
	Action    ActionType          `json:"action"`
	Reason    string              `json:"reason"`
	Challenge *Challenge          `json:"challenge,omitempty"`
	Headers   map[string]string   `json:"headers,omitempty"`
	Delay     time.Duration       `json:"delay,omitempty"`
}

// Additional action types for DDoS protection (extends the common ActionType)
const (
	ActionTarpit    ActionType = ActionType(iota + 10) // Offset to avoid conflicts
	ActionRateLimit ActionType = ActionType(iota + 11)
)

// Rate limiting implementation

// checkRateLimits checks various rate limits for a request
func (ddos *DDoSProtection) checkRateLimits(clientIP string, req *http.Request) *DDoSDecision {
	// Check IP-specific rate limit
	if ddos.config.IPRateLimit > 0 {
		if exceeded, resetTime := ddos.ipRateLimiter.checkLimit(clientIP, ddos.config.IPRateLimit); exceeded {
			return &DDoSDecision{
				Action: ActionRateLimit,
				Reason: fmt.Sprintf("IP rate limit exceeded, resets at %v", resetTime),
				Headers: map[string]string{
					"X-RateLimit-Limit":     fmt.Sprintf("%d", ddos.config.IPRateLimit),
					"X-RateLimit-Reset":     fmt.Sprintf("%d", resetTime.Unix()),
					"Retry-After":           fmt.Sprintf("%d", int(time.Until(resetTime).Seconds())),
				},
			}
		}
	}
	
	// Check subnet rate limit
	if ddos.config.SubnetRateLimit > 0 {
		subnet := getSubnet(clientIP, 24) // /24 subnet
		if exceeded, resetTime := ddos.ipRateLimiter.checkLimit(subnet, ddos.config.SubnetRateLimit); exceeded {
			return &DDoSDecision{
				Action: ActionRateLimit,
				Reason: fmt.Sprintf("Subnet rate limit exceeded, resets at %v", resetTime),
			}
		}
	}
	
	return &DDoSDecision{Action: ActionAllow}
}

// checkLimit checks if a rate limit is exceeded
func (rl *IPRateLimiter) checkLimit(key string, limit int) (bool, time.Time) {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	now := time.Now()
	
	entry, exists := rl.limits[key]
	if !exists {
		entry = &RateLimitEntry{
			FirstRequest: now,
			LastReset:    now,
		}
		rl.limits[key] = entry
	}
	
	// Reset counter if window has passed
	if now.Sub(entry.LastReset) >= rl.window {
		entry.Requests = 0
		entry.LastReset = now
		entry.Blocked = false
	}
	
	entry.Requests++
	
	// Check if limit is exceeded
	if int(entry.Requests) > limit {
		entry.Blocked = true
		entry.BlockedUntil = now.Add(rl.window)
		entry.Violations++
		return true, entry.BlockedUntil
	}
	
	return false, time.Time{}
}

// Behavior analysis implementation

// checkBehaviorAnalysis analyzes client behavior for anomalies
func (ddos *DDoSProtection) checkBehaviorAnalysis(clientIP string, req *http.Request) *DDoSDecision {
	profile := ddos.getOrCreateClientProfile(clientIP)
	
	// Update profile with current request
	profile.TotalRequests++
	profile.LastSeen = time.Now()
	
	// Calculate requests per second
	duration := time.Since(profile.FirstSeen).Seconds()
	if duration > 0 {
		profile.RequestsPerSecond = float64(profile.TotalRequests) / duration
	}
	
	// Check for suspicious patterns
	suspiciousActivities := ddos.detectSuspiciousActivity(profile, req)
	profile.SuspiciousActivity = append(profile.SuspiciousActivity, suspiciousActivities...)
	
	// Calculate threat score
	profile.ThreatScore = ddos.calculateThreatScore(profile)
	
	// Make decision based on threat score
	if profile.ThreatScore > 0.8 { // High threat
		return &DDoSDecision{
			Action: ActionBlock,
			Reason: fmt.Sprintf("High threat score: %.2f", profile.ThreatScore),
		}
	} else if profile.ThreatScore > 0.5 { // Medium threat
		challenge := ddos.createChallenge(clientIP)
		return &DDoSDecision{
			Action:    ActionChallenge,
			Reason:    fmt.Sprintf("Medium threat score: %.2f", profile.ThreatScore),
			Challenge: challenge,
		}
	}
	
	return &DDoSDecision{Action: ActionAllow}
}

// getOrCreateClientProfile gets or creates a client profile
func (ddos *DDoSProtection) getOrCreateClientProfile(clientIP string) *ClientProfile {
	ddos.behaviorAnalyzer.mutex.Lock()
	defer ddos.behaviorAnalyzer.mutex.Unlock()
	
	profile, exists := ddos.behaviorAnalyzer.clientProfiles[clientIP]
	if !exists {
		profile = &ClientProfile{
			IP:        clientIP,
			FirstSeen: time.Now(),
			LastSeen:  time.Now(),
		}
		ddos.behaviorAnalyzer.clientProfiles[clientIP] = profile
	}
	
	return profile
}

// detectSuspiciousActivity detects suspicious activity patterns
func (ddos *DDoSProtection) detectSuspiciousActivity(profile *ClientProfile, req *http.Request) []SuspiciousActivity {
	var activities []SuspiciousActivity
	
	// Check for high request rate
	if profile.RequestsPerSecond > 100 { // Configurable threshold
		activities = append(activities, SuspiciousActivity{
			Type:        "high_request_rate",
			Description: fmt.Sprintf("Request rate: %.2f RPS", profile.RequestsPerSecond),
			Severity:    "high",
			DetectedAt:  time.Now(),
			Evidence:    map[string]interface{}{"rps": profile.RequestsPerSecond},
		})
	}
	
	// Check for suspicious user agent
	userAgent := req.Header.Get("User-Agent")
	if ddos.isSuspiciousUserAgent(userAgent) {
		activities = append(activities, SuspiciousActivity{
			Type:        "suspicious_user_agent",
			Description: fmt.Sprintf("Suspicious user agent: %s", userAgent),
			Severity:    "medium",
			DetectedAt:  time.Now(),
			Evidence:    map[string]interface{}{"user_agent": userAgent},
		})
	}
	
	// Check for missing common headers
	if req.Header.Get("Accept") == "" || req.Header.Get("Accept-Language") == "" {
		activities = append(activities, SuspiciousActivity{
			Type:        "missing_headers",
			Description: "Missing common HTTP headers",
			Severity:    "low",
			DetectedAt:  time.Now(),
		})
	}
	
	return activities
}

// calculateThreatScore calculates a threat score for a client
func (ddos *DDoSProtection) calculateThreatScore(profile *ClientProfile) float64 {
	score := 0.0
	
	// Base score on request rate
	if profile.RequestsPerSecond > 50 {
		score += 0.3
	}
	if profile.RequestsPerSecond > 100 {
		score += 0.3
	}
	
	// Factor in suspicious activities
	for _, activity := range profile.SuspiciousActivity {
		switch activity.Severity {
		case "low":
			score += 0.1
		case "medium":
			score += 0.2
		case "high":
			score += 0.4
		}
	}
	
	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}
	
	return score
}

// Attack detection implementation

// registerAttackDetectors registers attack detection functions
func (ddos *DDoSProtection) registerAttackDetectors() {
	ddos.attackDetector.detectors[AttackTypeVolumetric] = ddos.detectVolumetricAttack
	ddos.attackDetector.detectors[AttackTypeHTTPFlood] = ddos.detectHTTPFloodAttack
	ddos.attackDetector.detectors[AttackTypeSYNFlood] = ddos.detectSYNFloodAttack
	ddos.attackDetector.detectors[AttackTypeSlowloris] = ddos.detectSlowlorisAttack
}

// detectVolumetricAttack detects volumetric attacks
func (ddos *DDoSProtection) detectVolumetricAttack(pattern *TrafficPattern, baseline *TrafficBaseline) (bool, float64) {
	if baseline.BaselineRPS == 0 {
		return false, 0 // No baseline yet
	}
	
	// Check if RPS is significantly higher than baseline
	threshold := baseline.BaselineRPS * 3.0 // 3x baseline
	if pattern.RequestsPerSecond > threshold {
		confidence := (pattern.RequestsPerSecond - baseline.BaselineRPS) / baseline.BaselineRPS
		return true, confidence
	}
	
	return false, 0
}

// detectHTTPFloodAttack detects HTTP flood attacks
func (ddos *DDoSProtection) detectHTTPFloodAttack(pattern *TrafficPattern, baseline *TrafficBaseline) (bool, float64) {
	// Check for unusually high request rate with low diversity
	if pattern.RequestsPerSecond > 1000 && pattern.UniqueIPs < 10 {
		confidence := pattern.RequestsPerSecond / float64(pattern.UniqueIPs) / 100.0
		return true, confidence
	}
	
	return false, 0
}

// detectSYNFloodAttack detects SYN flood attacks (simplified)
func (ddos *DDoSProtection) detectSYNFloodAttack(pattern *TrafficPattern, baseline *TrafficBaseline) (bool, float64) {
	// This would typically analyze network-level metrics
	// For HTTP load balancer, we can detect patterns consistent with SYN floods
	return false, 0
}

// detectSlowlorisAttack detects Slowloris attacks
func (ddos *DDoSProtection) detectSlowlorisAttack(pattern *TrafficPattern, baseline *TrafficBaseline) (bool, float64) {
	// Check for many slow connections from few IPs
	// This would need connection-level metrics
	return false, 0
}

// Challenge system implementation

// createChallenge creates a challenge for a client
func (ddos *DDoSProtection) createChallenge(clientIP string) *Challenge {
	if ddos.challengeSystem == nil {
		return nil
	}
	
	challengeType := ddos.selectChallengeType()
	
	challenge := &Challenge{
		ID:        uuid.New().String(),
		IP:        clientIP,
		Type:      challengeType,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(ddos.config.ChallengeTimeout),
	}
	
	// Generate challenge data based on type
	switch challengeType {
	case ChallengeJavaScript:
		challenge.Data = ddos.generateJavaScriptChallenge()
	case ChallengeCookie:
		challenge.Data = ddos.generateCookieChallenge()
	case ChallengeProofOfWork:
		challenge.Data = ddos.generateProofOfWorkChallenge()
	}
	
	ddos.challengeSystem.mutex.Lock()
	ddos.challengeSystem.challenges[challenge.ID] = challenge
	ddos.challengeSystem.mutex.Unlock()
	
	return challenge
}

// selectChallengeType selects an appropriate challenge type
func (ddos *DDoSProtection) selectChallengeType() ChallengeType {
	// Select based on configuration
	if len(ddos.config.ChallengeTypes) > 0 {
		return ddos.config.ChallengeTypes[0] // Simplified selection
	}
	return ChallengeJavaScript
}

// generateJavaScriptChallenge generates a JavaScript challenge
func (ddos *DDoSProtection) generateJavaScriptChallenge() string {
	return fmt.Sprintf(`
		<script>
			function solvePuzzle() {
				var solution = Math.random().toString(36).substring(7);
				document.cookie = "challenge_solution=" + solution + "; path=/";
				return solution;
			}
			solvePuzzle();
		</script>
	`)
}

// generateCookieChallenge generates a cookie-based challenge
func (ddos *DDoSProtection) generateCookieChallenge() string {
	return uuid.New().String()
}

// generateProofOfWorkChallenge generates a proof-of-work challenge
func (ddos *DDoSProtection) generateProofOfWorkChallenge() string {
	return fmt.Sprintf("Find a string that when hashed with SHA256 starts with '0000': %s", uuid.New().String())
}

// Mitigation strategies

// registerMitigationStrategies registers mitigation strategies
func (ddos *DDoSProtection) registerMitigationStrategies() {
	ddos.mitigationEngine.strategies[MitigationTypeRateLimit] = &RateLimitMitigation{}
	ddos.mitigationEngine.strategies[MitigationTypeBlacklist] = &BlacklistMitigation{ddos: ddos}
	ddos.mitigationEngine.strategies[MitigationTypeChallenge] = &ChallengeMitigation{ddos: ddos}
	ddos.mitigationEngine.strategies[MitigationTypeDrop] = &DropMitigation{}
}

// RateLimitMitigation implements rate limiting mitigation
type RateLimitMitigation struct{}

func (rlm *RateLimitMitigation) Apply(target string, config map[string]interface{}) error {
	// Apply rate limiting to target
	return nil
}

func (rlm *RateLimitMitigation) Remove(target string) error {
	// Remove rate limiting from target
	return nil
}

func (rlm *RateLimitMitigation) IsActive(target string) bool {
	// Check if rate limiting is active for target
	return false
}

// BlacklistMitigation implements blacklisting mitigation
type BlacklistMitigation struct {
	ddos *DDoSProtection
}

func (bm *BlacklistMitigation) Apply(target string, config map[string]interface{}) error {
	// Add to blacklist
	entry := &BlacklistEntry{
		IP:        target,
		Reason:    "DDoS mitigation",
		AddedAt:   time.Now(),
		ExpiresAt: time.Now().Add(24 * time.Hour), // Configurable
	}
	
	bm.ddos.ipBlacklist.mutex.Lock()
	bm.ddos.ipBlacklist.entries[target] = entry
	bm.ddos.ipBlacklist.mutex.Unlock()
	
	return nil
}

func (bm *BlacklistMitigation) Remove(target string) error {
	bm.ddos.ipBlacklist.mutex.Lock()
	delete(bm.ddos.ipBlacklist.entries, target)
	bm.ddos.ipBlacklist.mutex.Unlock()
	return nil
}

func (bm *BlacklistMitigation) IsActive(target string) bool {
	bm.ddos.ipBlacklist.mutex.RLock()
	_, exists := bm.ddos.ipBlacklist.entries[target]
	bm.ddos.ipBlacklist.mutex.RUnlock()
	return exists
}

// ChallengeMitigation implements challenge-based mitigation
type ChallengeMitigation struct {
	ddos *DDoSProtection
}

func (cm *ChallengeMitigation) Apply(target string, config map[string]interface{}) error {
	// Challenge will be applied per-request
	return nil
}

func (cm *ChallengeMitigation) Remove(target string) error {
	// Remove challenges for target
	return nil
}

func (cm *ChallengeMitigation) IsActive(target string) bool {
	return false
}

// DropMitigation implements packet dropping mitigation
type DropMitigation struct{}

func (dm *DropMitigation) Apply(target string, config map[string]interface{}) error {
	// Configure packet dropping for target
	return nil
}

func (dm *DropMitigation) Remove(target string) error {
	// Remove packet dropping for target
	return nil
}

func (dm *DropMitigation) IsActive(target string) bool {
	return false
}

// Helper functions

// getClientIPFromRequest extracts client IP from request
func getClientIPFromRequest(req *http.Request) string {
	// Check X-Forwarded-For header first
	if xff := req.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		return strings.TrimSpace(ips[0])
	}
	
	// Check X-Real-IP header
	if xri := req.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	
	// Use remote address
	ip, _, _ := net.SplitHostPort(req.RemoteAddr)
	return ip
}

// getSubnet returns the subnet for an IP address
func getSubnet(ip string, maskSize int) string {
	parsedIP := net.ParseIP(ip)
	if parsedIP == nil {
		return ip
	}
	
	var mask net.IPMask
	if parsedIP.To4() != nil {
		mask = net.CIDRMask(maskSize, 32)
	} else {
		mask = net.CIDRMask(maskSize, 128)
	}
	
	network := parsedIP.Mask(mask)
	return fmt.Sprintf("%s/%d", network.String(), maskSize)
}

// isWhitelisted checks if an IP is whitelisted
func (ddos *DDoSProtection) isWhitelisted(ip string) bool {
	ddos.ipWhitelist.mutex.RLock()
	defer ddos.ipWhitelist.mutex.RUnlock()
	
	entry, exists := ddos.ipWhitelist.entries[ip]
	if !exists {
		return false
	}
	
	// Check if entry has expired
	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		return false
	}
	
	return true
}

// isBlacklisted checks if an IP is blacklisted
func (ddos *DDoSProtection) isBlacklisted(ip string) bool {
	ddos.ipBlacklist.mutex.RLock()
	defer ddos.ipBlacklist.mutex.RUnlock()
	
	entry, exists := ddos.ipBlacklist.entries[ip]
	if !exists {
		return false
	}
	
	// Check if entry has expired
	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		delete(ddos.ipBlacklist.entries, ip) // Clean up expired entry
		return false
	}
	
	return true
}

// checkGeoBlocking checks geographical blocking
func (ddos *DDoSProtection) checkGeoBlocking(ip string) (bool, string) {
	// Simplified geo-blocking check
	// In practice, this would use a GeoIP database
	return false, ""
}

// isSuspiciousUserAgent checks if a user agent is suspicious
func (ddos *DDoSProtection) isSuspiciousUserAgent(userAgent string) bool {
	suspiciousAgents := []string{
		"bot", "crawler", "spider", "scraper",
		"python", "curl", "wget", "httpie",
	}
	
	userAgentLower := strings.ToLower(userAgent)
	for _, suspicious := range suspiciousAgents {
		if strings.Contains(userAgentLower, suspicious) {
			return true
		}
	}
	
	return false
}

// Background loops

// trafficAnalysisLoop analyzes traffic patterns
func (ddos *DDoSProtection) trafficAnalysisLoop() {
	ticker := time.NewTicker(60 * time.Second) // Analyze every minute
	defer ticker.Stop()
	
	for {
		select {
		case <-ddos.ctx.Done():
			return
		case <-ticker.C:
			ddos.analyzeTrafficPatterns()
		}
	}
}

// attackDetectionLoop runs attack detection
func (ddos *DDoSProtection) attackDetectionLoop() {
	ticker := time.NewTicker(30 * time.Second) // Check every 30 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-ddos.ctx.Done():
			return
		case <-ticker.C:
			ddos.runAttackDetection()
		}
	}
}

// cleanupLoop cleans up expired entries
func (ddos *DDoSProtection) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute) // Cleanup every 5 minutes
	defer ticker.Stop()
	
	for {
		select {
		case <-ddos.ctx.Done():
			return
		case <-ticker.C:
			ddos.cleanup()
		}
	}
}

// metricsCollectionLoop collects DDoS metrics
func (ddos *DDoSProtection) metricsCollectionLoop() {
	ticker := time.NewTicker(ddos.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ddos.ctx.Done():
			return
		case <-ticker.C:
			ddos.updateMetrics()
		}
	}
}

// analyzeTrafficPatterns analyzes current traffic patterns
func (ddos *DDoSProtection) analyzeTrafficPatterns() {
	// Implementation would analyze recent traffic and update patterns
}

// runAttackDetection runs attack detection algorithms
func (ddos *DDoSProtection) runAttackDetection() {
	// Implementation would run registered attack detectors
}

// cleanup removes expired entries and data
func (ddos *DDoSProtection) cleanup() {
	now := time.Now()
	
	// Cleanup blacklist entries
	ddos.ipBlacklist.mutex.Lock()
	for ip, entry := range ddos.ipBlacklist.entries {
		if !entry.ExpiresAt.IsZero() && now.After(entry.ExpiresAt) {
			delete(ddos.ipBlacklist.entries, ip)
		}
	}
	ddos.ipBlacklist.mutex.Unlock()
	
	// Cleanup whitelist entries
	ddos.ipWhitelist.mutex.Lock()
	for ip, entry := range ddos.ipWhitelist.entries {
		if !entry.ExpiresAt.IsZero() && now.After(entry.ExpiresAt) {
			delete(ddos.ipWhitelist.entries, ip)
		}
	}
	ddos.ipWhitelist.mutex.Unlock()
	
	// Cleanup challenges
	if ddos.challengeSystem != nil {
		ddos.challengeSystem.mutex.Lock()
		for id, challenge := range ddos.challengeSystem.challenges {
			if now.After(challenge.ExpiresAt) {
				delete(ddos.challengeSystem.challenges, id)
			}
		}
		ddos.challengeSystem.mutex.Unlock()
	}
}

// updateTrafficAnalysis updates traffic analysis with new request
func (ddos *DDoSProtection) updateTrafficAnalysis(req *http.Request) {
	// Implementation would update traffic patterns
}

// updateClientProfile updates client profile with new request
func (ddos *DDoSProtection) updateClientProfile(clientIP string, req *http.Request) {
	// Implementation would update client behavioral profile
}

// updateMetrics updates DDoS protection metrics
func (ddos *DDoSProtection) updateMetrics() {
	ddos.metricsMutex.Lock()
	defer ddos.metricsMutex.Unlock()
	
	ddos.metrics.LastUpdated = time.Now()
	
	// Count active mitigations
	ddos.mitigationMutex.RLock()
	ddos.metrics.MitigationsActive = int64(len(ddos.activeMitigations))
	
	// Reset mitigation counters
	ddos.metrics.MitigationsByType = make(map[MitigationType]int64)
	for _, mitigation := range ddos.activeMitigations {
		ddos.metrics.MitigationsByType[mitigation.Type]++
	}
	ddos.mitigationMutex.RUnlock()
}

// Public API methods

// GetMetrics returns DDoS protection metrics
func (ddos *DDoSProtection) GetMetrics() *DDoSMetrics {
	ddos.metricsMutex.RLock()
	defer ddos.metricsMutex.RUnlock()
	
	// Return copy of metrics
	metricsCopy := *ddos.metrics
	
	// Copy maps
	metricsCopy.AttacksByType = make(map[AttackType]int64)
	for k, v := range ddos.metrics.AttacksByType {
		metricsCopy.AttacksByType[k] = v
	}
	
	metricsCopy.MitigationsByType = make(map[MitigationType]int64)
	for k, v := range ddos.metrics.MitigationsByType {
		metricsCopy.MitigationsByType[k] = v
	}
	
	metricsCopy.TopAttackerIPs = make(map[string]int64)
	for k, v := range ddos.metrics.TopAttackerIPs {
		metricsCopy.TopAttackerIPs[k] = v
	}
	
	metricsCopy.TopAttackerCountries = make(map[string]int64)
	for k, v := range ddos.metrics.TopAttackerCountries {
		metricsCopy.TopAttackerCountries[k] = v
	}
	
	return &metricsCopy
}

// GetActiveMitigations returns currently active mitigations
func (ddos *DDoSProtection) GetActiveMitigations() []*ActiveMitigation {
	ddos.mitigationMutex.RLock()
	defer ddos.mitigationMutex.RUnlock()
	
	mitigations := make([]*ActiveMitigation, 0, len(ddos.activeMitigations))
	for _, mitigation := range ddos.activeMitigations {
		mitigationCopy := *mitigation
		mitigations = append(mitigations, &mitigationCopy)
	}
	
	return mitigations
}

// removeMitigation removes an active mitigation
func (me *MitigationEngine) removeMitigation(id string, mitigationType MitigationType) error {
	me.mutex.Lock()
	defer me.mutex.Unlock()
	
	mitigation, exists := me.activeMitigations[id]
	if !exists {
		return fmt.Errorf("mitigation %s not found", id)
	}
	
	strategy, exists := me.strategies[mitigationType]
	if !exists {
		return fmt.Errorf("mitigation strategy %s not found", mitigationType)
	}
	
	if err := strategy.Remove(mitigation.Target); err != nil {
		return fmt.Errorf("failed to remove mitigation: %w", err)
	}
	
	delete(me.activeMitigations, id)
	return nil
}

// DefaultDDoSProtectionConfig returns default DDoS protection configuration
func DefaultDDoSProtectionConfig() DDoSProtectionConfig {
	return DDoSProtectionConfig{
		EnableProtection:          true,
		ProtectionMode:            ProtectionModeActive,
		SensitivityLevel:          SensitivityMedium,
		GlobalRateLimit:           10000,
		IPRateLimit:               100,
		SubnetRateLimit:           1000,
		CountryRateLimit:          5000,
		ASNRateLimit:              2000,
		RateLimitWindow:           time.Minute,
		MaxConnectionsPerIP:       100,
		MaxConnectionsPerSubnet:   1000,
		ConnectionTimeout:         30 * time.Second,
		EnableSYNProtection:       true,
		SYNRateLimit:             1000,
		SYNBurstSize:             100,
		EnableSYNCookies:         true,
		EnableSlowlorisProtection: true,
		RequestTimeout:            30 * time.Second,
		HeaderTimeout:             10 * time.Second,
		MaxHeaderSize:             8192,
		MaxRequestSize:            10 * 1024 * 1024, // 10MB
		MaxURLLength:              2048,
		EnableBehaviorAnalysis:    true,
		BehaviorWindow:            5 * time.Minute,
		AnomalyThreshold:          0.7,
		EnableGeoBlocking:         false,
		EnableChallenges:          true,
		ChallengeTypes:            []ChallengeType{ChallengeJavaScript, ChallengeCookie},
		ChallengeTimeout:          5 * time.Minute,
		ChallengeRetries:          3,
		AutoMitigation:            true,
		MitigationDuration:        60 * time.Minute,
		MitigationCooldown:        10 * time.Minute,
		EnableMetrics:             true,
		MetricsInterval:           30 * time.Second,
		AlertThresholds: AlertThresholds{
			RequestsPerSecond:    1000,
			BlockedRequestsRatio: 0.1,
			NewAttackDetected:    true,
			MitigationFailed:     true,
		},
	}
}