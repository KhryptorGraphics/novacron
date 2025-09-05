package security

import (
	"fmt"
	"net"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// EnterpriseRateLimiter provides comprehensive rate limiting and DDoS protection
type EnterpriseRateLimiter struct {
	config          RateLimitConfig
	globalLimiter   *rate.Limiter
	userLimiters    map[string]*rate.Limiter
	ipLimiters      map[string]*rate.Limiter
	endpointLimiters map[string]*rate.Limiter
	ddosProtector   *DDoSProtector
	analytics       *RateLimitAnalytics
	mu              sync.RWMutex
}

// DDoSProtector provides DDoS detection and mitigation
type DDoSProtector struct {
	config           DDoSConfig
	blockedIPs       map[string]time.Time
	suspiciousIPs    map[string]*SuspiciousActivity
	whitelist        []*net.IPNet
	requestCounters  map[string]*RequestCounter
	mu               sync.RWMutex
}

// SuspiciousActivity tracks suspicious IP activity
type SuspiciousActivity struct {
	IP               string    `json:"ip"`
	SuspicionScore   float64   `json:"suspicion_score"`
	RequestCount     int       `json:"request_count"`
	FailedRequests   int       `json:"failed_requests"`
	LastActivity     time.Time `json:"last_activity"`
	FirstSeen        time.Time `json:"first_seen"`
	UserAgents       []string  `json:"user_agents"`
	RequestedPaths   []string  `json:"requested_paths"`
	BlockedUntil     time.Time `json:"blocked_until"`
	ThreatIndicators []string  `json:"threat_indicators"`
}

// RequestCounter tracks request patterns for DDoS detection
type RequestCounter struct {
	IP           string    `json:"ip"`
	Count        int       `json:"count"`
	WindowStart  time.Time `json:"window_start"`
	LastRequest  time.Time `json:"last_request"`
	RequestSpike bool      `json:"request_spike"`
}

// RateLimitAnalytics provides rate limiting analytics and reporting
type RateLimitAnalytics struct {
	metrics          map[string]*RateLimitMetrics
	alertThresholds  map[string]float64
	reportingEnabled bool
	mu               sync.RWMutex
}

// RateLimitMetrics tracks rate limiting metrics
type RateLimitMetrics struct {
	Endpoint         string    `json:"endpoint"`
	TotalRequests    int64     `json:"total_requests"`
	RejectedRequests int64     `json:"rejected_requests"`
	RejectionRate    float64   `json:"rejection_rate"`
	PeakRPS          int       `json:"peak_rps"`
	AverageRPS       float64   `json:"average_rps"`
	LastUpdated      time.Time `json:"last_updated"`
	WindowStart      time.Time `json:"window_start"`
}

// RateLimitResult represents the result of a rate limit check
type RateLimitResult struct {
	Allowed      bool          `json:"allowed"`
	Reason       string        `json:"reason,omitempty"`
	Remaining    int           `json:"remaining"`
	ResetTime    time.Time     `json:"reset_time"`
	RetryAfter   time.Duration `json:"retry_after,omitempty"`
	ThreatLevel  string        `json:"threat_level,omitempty"`
}

// NewEnterpriseRateLimiter creates a new enterprise rate limiter
func NewEnterpriseRateLimiter(config RateLimitConfig) *EnterpriseRateLimiter {
	// Create global rate limiter
	globalLimit := rate.Every(time.Second / time.Duration(config.Global.RequestsPerSecond))
	globalLimiter := rate.NewLimiter(globalLimit, config.Global.BurstSize)

	// Create DDoS protector
	ddosProtector := &DDoSProtector{
		config:          config.DDoSConfig,
		blockedIPs:      make(map[string]time.Time),
		suspiciousIPs:   make(map[string]*SuspiciousActivity),
		requestCounters: make(map[string]*RequestCounter),
	}

	// Parse whitelist CIDRs
	ddosProtector.parseWhitelist()

	// Create analytics
	analytics := &RateLimitAnalytics{
		metrics:          make(map[string]*RateLimitMetrics),
		alertThresholds:  make(map[string]float64),
		reportingEnabled: true,
	}

	limiter := &EnterpriseRateLimiter{
		config:           config,
		globalLimiter:    globalLimiter,
		userLimiters:     make(map[string]*rate.Limiter),
		ipLimiters:       make(map[string]*rate.Limiter),
		endpointLimiters: make(map[string]*rate.Limiter),
		ddosProtector:    ddosProtector,
		analytics:        analytics,
	}

	// Start background cleanup and monitoring
	go limiter.cleanup()
	go limiter.monitor()

	return limiter
}

// Allow checks if a request is allowed based on comprehensive rate limiting rules
func (erl *EnterpriseRateLimiter) Allow(secCtx *SecurityContext) bool {
	erl.mu.Lock()
	defer erl.mu.Unlock()

	// Update analytics
	erl.analytics.updateMetrics(secCtx.Path, true, false)

	// DDoS protection check first
	if erl.config.DDoSConfig.Enabled {
		if blocked, reason := erl.ddosProtector.checkDDoS(secCtx); blocked {
			erl.analytics.updateMetrics(secCtx.Path, true, true)
			return false
		}
	}

	// Global rate limit check
	if !erl.globalLimiter.Allow() {
		erl.analytics.updateMetrics(secCtx.Path, true, true)
		return false
	}

	// IP-based rate limiting
	if !erl.allowByIP(secCtx.ClientIP) {
		erl.analytics.updateMetrics(secCtx.Path, true, true)
		return false
	}

	// User-based rate limiting (if user is authenticated)
	if secCtx.UserID != "" && !erl.allowByUser(secCtx.UserID) {
		erl.analytics.updateMetrics(secCtx.Path, true, true)
		return false
	}

	// Endpoint-specific rate limiting
	if !erl.allowByEndpoint(secCtx.Path) {
		erl.analytics.updateMetrics(secCtx.Path, true, true)
		return false
	}

	return true
}

// allowByIP checks IP-based rate limits
func (erl *EnterpriseRateLimiter) allowByIP(ip string) bool {
	limiter, exists := erl.ipLimiters[ip]
	if !exists {
		// Create new limiter for this IP
		ipLimit := rate.Every(time.Second / time.Duration(erl.config.PerIP.RequestsPerSecond))
		limiter = rate.NewLimiter(ipLimit, erl.config.PerIP.BurstSize)
		erl.ipLimiters[ip] = limiter
	}

	return limiter.Allow()
}

// allowByUser checks user-based rate limits
func (erl *EnterpriseRateLimiter) allowByUser(userID string) bool {
	limiter, exists := erl.userLimiters[userID]
	if !exists {
		// Create new limiter for this user
		userLimit := rate.Every(time.Second / time.Duration(erl.config.PerUser.RequestsPerSecond))
		limiter = rate.NewLimiter(userLimit, erl.config.PerUser.BurstSize)
		erl.userLimiters[userID] = limiter
	}

	return limiter.Allow()
}

// allowByEndpoint checks endpoint-specific rate limits
func (erl *EnterpriseRateLimiter) allowByEndpoint(path string) bool {
	// Check if there's a specific limit for this endpoint
	endpointLimit, exists := erl.config.Endpoints[path]
	if !exists {
		return true // No specific limit
	}

	limiter, exists := erl.endpointLimiters[path]
	if !exists {
		// Create new limiter for this endpoint
		pathLimit := rate.Every(time.Second / time.Duration(endpointLimit.RequestsPerSecond))
		limiter = rate.NewLimiter(pathLimit, endpointLimit.BurstSize)
		erl.endpointLimiters[path] = limiter
	}

	return limiter.Allow()
}

// GetRateLimitInfo returns current rate limit information
func (erl *EnterpriseRateLimiter) GetRateLimitInfo(secCtx *SecurityContext) *RateLimitResult {
	erl.mu.RLock()
	defer erl.mu.RUnlock()

	result := &RateLimitResult{
		Allowed:     true,
		ResetTime:   time.Now().Add(time.Minute),
		ThreatLevel: "none",
	}

	// Check DDoS status
	if erl.config.DDoSConfig.Enabled {
		if activity := erl.ddosProtector.getSuspiciousActivity(secCtx.ClientIP); activity != nil {
			result.ThreatLevel = erl.calculateThreatLevel(activity.SuspicionScore)
			if time.Now().Before(activity.BlockedUntil) {
				result.Allowed = false
				result.Reason = "IP temporarily blocked due to suspicious activity"
				result.RetryAfter = activity.BlockedUntil.Sub(time.Now())
			}
		}
	}

	// Get remaining capacity for IP
	if ipLimiter, exists := erl.ipLimiters[secCtx.ClientIP]; exists {
		result.Remaining = int(ipLimiter.Tokens())
	} else {
		result.Remaining = erl.config.PerIP.BurstSize
	}

	return result
}

// DDoS Protection Methods

// checkDDoS performs comprehensive DDoS detection
func (ddos *DDoSProtector) checkDDoS(secCtx *SecurityContext) (bool, string) {
	ddos.mu.Lock()
	defer ddos.mu.Unlock()

	ip := secCtx.ClientIP

	// Check if IP is in whitelist
	if ddos.isWhitelisted(ip) {
		return false, ""
	}

	// Check if IP is currently blocked
	if blockedUntil, exists := ddos.blockedIPs[ip]; exists {
		if time.Now().Before(blockedUntil) {
			return true, "IP temporarily blocked"
		}
		// Block expired, remove it
		delete(ddos.blockedIPs, ip)
	}

	// Update request counter
	ddos.updateRequestCounter(ip, secCtx)

	// Check for request spike
	if counter := ddos.requestCounters[ip]; counter != nil && counter.RequestSpike {
		return ddos.handleRequestSpike(ip, secCtx)
	}

	// Update suspicious activity
	activity := ddos.updateSuspiciousActivity(ip, secCtx)

	// Check suspicion score
	if activity.SuspicionScore >= ddos.config.SuspicionScore {
		return ddos.handleSuspiciousActivity(ip, activity)
	}

	return false, ""
}

// updateRequestCounter updates request tracking for an IP
func (ddos *DDoSProtector) updateRequestCounter(ip string, secCtx *SecurityContext) {
	now := time.Now()
	counter, exists := ddos.requestCounters[ip]

	if !exists {
		counter = &RequestCounter{
			IP:          ip,
			Count:       1,
			WindowStart: now,
			LastRequest: now,
		}
		ddos.requestCounters[ip] = counter
		return
	}

	// Reset window if it's been more than 1 minute
	if now.Sub(counter.WindowStart) > time.Minute {
		counter.Count = 1
		counter.WindowStart = now
		counter.RequestSpike = false
	} else {
		counter.Count++
	}

	counter.LastRequest = now

	// Check for request spike
	if counter.Count > ddos.config.ThresholdRPS/60 { // Per-minute threshold
		counter.RequestSpike = true
	}
}

// updateSuspiciousActivity updates suspicious activity tracking
func (ddos *DDoSProtector) updateSuspiciousActivity(ip string, secCtx *SecurityContext) *SuspiciousActivity {
	now := time.Now()
	activity, exists := ddos.suspiciousIPs[ip]

	if !exists {
		activity = &SuspiciousActivity{
			IP:               ip,
			SuspicionScore:   0.0,
			RequestCount:     1,
			LastActivity:     now,
			FirstSeen:        now,
			UserAgents:       []string{secCtx.UserAgent},
			RequestedPaths:   []string{secCtx.Path},
			ThreatIndicators: []string{},
		}
		ddos.suspiciousIPs[ip] = activity
		return activity
	}

	activity.RequestCount++
	activity.LastActivity = now

	// Add user agent if not seen before
	if !contains(activity.UserAgents, secCtx.UserAgent) {
		activity.UserAgents = append(activity.UserAgents, secCtx.UserAgent)
	}

	// Add requested path if not seen before
	if !contains(activity.RequestedPaths, secCtx.Path) {
		activity.RequestedPaths = append(activity.RequestedPaths, secCtx.Path)
	}

	// Calculate suspicion score based on various factors
	activity.SuspicionScore = ddos.calculateSuspicionScore(activity, secCtx)

	return activity
}

// calculateSuspicionScore calculates suspicion score for an IP
func (ddos *DDoSProtector) calculateSuspicionScore(activity *SuspiciousActivity, secCtx *SecurityContext) float64 {
	score := 0.0

	// High request frequency
	duration := time.Since(activity.FirstSeen)
	if duration > 0 {
		rps := float64(activity.RequestCount) / duration.Seconds()
		if rps > 10 {
			score += 0.3
		}
	}

	// Multiple user agents (possible bot rotation)
	if len(activity.UserAgents) > 5 {
		score += 0.2
	}

	// Suspicious user agent patterns
	for _, ua := range activity.UserAgents {
		if isSuspiciousUserAgent(ua) {
			score += 0.2
			activity.ThreatIndicators = append(activity.ThreatIndicators, "suspicious_user_agent")
			break
		}
	}

	// Scanning behavior (accessing many different paths)
	if len(activity.RequestedPaths) > 20 {
		score += 0.3
		activity.ThreatIndicators = append(activity.ThreatIndicators, "scanning_behavior")
	}

	// Failed requests (from security context)
	if len(secCtx.RiskFactors) > 0 {
		activity.FailedRequests++
		failureRate := float64(activity.FailedRequests) / float64(activity.RequestCount)
		if failureRate > 0.5 {
			score += 0.2
			activity.ThreatIndicators = append(activity.ThreatIndicators, "high_failure_rate")
		}
	}

	return score
}

// handleRequestSpike handles detected request spikes
func (ddos *DDoSProtector) handleRequestSpike(ip string, secCtx *SecurityContext) (bool, string) {
	// Temporary rate limiting for request spikes
	activity := ddos.suspiciousIPs[ip]
	if activity != nil {
		activity.SuspicionScore += 0.1
	}
	
	return false, "request_spike_detected" // Don't block immediately for spikes
}

// handleSuspiciousActivity handles high suspicion score IPs
func (ddos *DDoSProtector) handleSuspiciousActivity(ip string, activity *SuspiciousActivity) (bool, string) {
	// Block the IP temporarily
	blockDuration := ddos.config.BlockDuration
	activity.BlockedUntil = time.Now().Add(blockDuration)
	ddos.blockedIPs[ip] = activity.BlockedUntil

	return true, fmt.Sprintf("IP blocked due to suspicious activity (score: %.2f)", activity.SuspicionScore)
}

// isWhitelisted checks if an IP is in the whitelist
func (ddos *DDoSProtector) isWhitelisted(ip string) bool {
	parsedIP := net.ParseIP(ip)
	if parsedIP == nil {
		return false
	}

	for _, network := range ddos.whitelist {
		if network.Contains(parsedIP) {
			return true
		}
	}

	return false
}

// parseWhitelist parses whitelist CIDR blocks
func (ddos *DDoSProtector) parseWhitelist() {
	ddos.whitelist = []*net.IPNet{}
	for _, cidr := range ddos.config.WhitelistCIDRs {
		_, network, err := net.ParseCIDR(cidr)
		if err == nil {
			ddos.whitelist = append(ddos.whitelist, network)
		}
	}
}

// getSuspiciousActivity gets suspicious activity for an IP
func (ddos *DDoSProtector) getSuspiciousActivity(ip string) *SuspiciousActivity {
	ddos.mu.RLock()
	defer ddos.mu.RUnlock()
	return ddos.suspiciousIPs[ip]
}

// Analytics Methods

// updateMetrics updates rate limiting metrics
func (rla *RateLimitAnalytics) updateMetrics(endpoint string, request, rejected bool) {
	rla.mu.Lock()
	defer rla.mu.Unlock()

	metrics, exists := rla.metrics[endpoint]
	if !exists {
		metrics = &RateLimitMetrics{
			Endpoint:    endpoint,
			WindowStart: time.Now(),
		}
		rla.metrics[endpoint] = metrics
	}

	if request {
		metrics.TotalRequests++
	}
	if rejected {
		metrics.RejectedRequests++
	}

	// Calculate rejection rate
	if metrics.TotalRequests > 0 {
		metrics.RejectionRate = float64(metrics.RejectedRequests) / float64(metrics.TotalRequests)
	}

	// Update average RPS
	duration := time.Since(metrics.WindowStart)
	if duration > 0 {
		metrics.AverageRPS = float64(metrics.TotalRequests) / duration.Seconds()
	}

	metrics.LastUpdated = time.Now()
}

// GetMetrics returns current rate limiting metrics
func (erl *EnterpriseRateLimiter) GetMetrics() map[string]*RateLimitMetrics {
	erl.analytics.mu.RLock()
	defer erl.analytics.mu.RUnlock()

	// Return copy of metrics
	result := make(map[string]*RateLimitMetrics)
	for endpoint, metrics := range erl.analytics.metrics {
		result[endpoint] = &RateLimitMetrics{
			Endpoint:         metrics.Endpoint,
			TotalRequests:    metrics.TotalRequests,
			RejectedRequests: metrics.RejectedRequests,
			RejectionRate:    metrics.RejectionRate,
			PeakRPS:          metrics.PeakRPS,
			AverageRPS:       metrics.AverageRPS,
			LastUpdated:      metrics.LastUpdated,
			WindowStart:      metrics.WindowStart,
		}
	}

	return result
}

// GetSuspiciousIPs returns current suspicious IPs
func (erl *EnterpriseRateLimiter) GetSuspiciousIPs() map[string]*SuspiciousActivity {
	erl.ddosProtector.mu.RLock()
	defer erl.ddosProtector.mu.RUnlock()

	// Return copy of suspicious IPs
	result := make(map[string]*SuspiciousActivity)
	for ip, activity := range erl.ddosProtector.suspiciousIPs {
		result[ip] = &SuspiciousActivity{
			IP:               activity.IP,
			SuspicionScore:   activity.SuspicionScore,
			RequestCount:     activity.RequestCount,
			FailedRequests:   activity.FailedRequests,
			LastActivity:     activity.LastActivity,
			FirstSeen:        activity.FirstSeen,
			UserAgents:       activity.UserAgents,
			RequestedPaths:   activity.RequestedPaths,
			BlockedUntil:     activity.BlockedUntil,
			ThreatIndicators: activity.ThreatIndicators,
		}
	}

	return result
}

// cleanup removes old entries to prevent memory leaks
func (erl *EnterpriseRateLimiter) cleanup() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		erl.mu.Lock()
		now := time.Now()

		// Cleanup old IP limiters (inactive for > 1 hour)
		for ip, limiter := range erl.ipLimiters {
			if limiter.Tokens() == float64(erl.config.PerIP.BurstSize) && 
			   time.Since(now) > time.Hour {
				delete(erl.ipLimiters, ip)
			}
		}

		// Cleanup old user limiters
		for userID, limiter := range erl.userLimiters {
			if limiter.Tokens() == float64(erl.config.PerUser.BurstSize) && 
			   time.Since(now) > time.Hour {
				delete(erl.userLimiters, userID)
			}
		}

		// Cleanup DDoS protection data
		erl.ddosProtector.mu.Lock()
		
		// Remove expired blocked IPs
		for ip, blockedUntil := range erl.ddosProtector.blockedIPs {
			if now.After(blockedUntil) {
				delete(erl.ddosProtector.blockedIPs, ip)
			}
		}

		// Remove old suspicious activities (older than 24 hours)
		for ip, activity := range erl.ddosProtector.suspiciousIPs {
			if now.Sub(activity.LastActivity) > 24*time.Hour {
				delete(erl.ddosProtector.suspiciousIPs, ip)
			}
		}

		// Remove old request counters
		for ip, counter := range erl.ddosProtector.requestCounters {
			if now.Sub(counter.LastRequest) > time.Hour {
				delete(erl.ddosProtector.requestCounters, ip)
			}
		}

		erl.ddosProtector.mu.Unlock()
		erl.mu.Unlock()
	}
}

// monitor performs continuous monitoring and alerting
func (erl *EnterpriseRateLimiter) monitor() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		if !erl.analytics.reportingEnabled {
			continue
		}

		erl.analytics.mu.RLock()
		for endpoint, metrics := range erl.analytics.metrics {
			// Check for high rejection rates
			if metrics.RejectionRate > 0.5 && metrics.TotalRequests > 100 {
				// TODO: Send alert for high rejection rate
				fmt.Printf("Alert: High rejection rate for %s: %.2f%%\n", 
					endpoint, metrics.RejectionRate*100)
			}
		}
		erl.analytics.mu.RUnlock()
	}
}

// calculateThreatLevel calculates threat level based on suspicion score
func (erl *EnterpriseRateLimiter) calculateThreatLevel(score float64) string {
	if score >= 0.8 {
		return "critical"
	} else if score >= 0.6 {
		return "high"
	} else if score >= 0.4 {
		return "medium"
	} else if score >= 0.2 {
		return "low"
	}
	return "none"
}

// Helper functions

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func isSuspiciousUserAgent(userAgent string) bool {
	suspiciousPatterns := []string{
		"bot", "crawler", "spider", "scraper", "scanner",
		"curl", "wget", "python", "go-http", "java",
		"nmap", "sqlmap", "nikto", "w3af", "burp",
	}

	ua := strings.ToLower(userAgent)
	for _, pattern := range suspiciousPatterns {
		if strings.Contains(ua, pattern) {
			return true
		}
	}

	return false
}