// Package routing implements intelligent global routing for DWCP v3 federation
// Provides latency-based routing, anycast, and DDoS mitigation at global scale
package routing

import (
	"context"
	"fmt"
	"math"
	"net"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Global metrics for routing
var (
	routingDecisions = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_routing_decision_latency_ms",
			Help:    "Latency of routing decisions in milliseconds",
			Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 25, 50, 100},
		},
		[]string{"algorithm"},
	)

	routedTraffic = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_routed_traffic_bytes",
			Help: "Total routed traffic in bytes",
		},
		[]string{"source_region", "target_region", "protocol"},
	)

	routingErrors = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_routing_errors_total",
			Help: "Total routing errors",
		},
		[]string{"error_type"},
	)

	measurementLatency = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_measured_latency_ms",
			Help: "Measured network latency between regions in milliseconds",
		},
		[]string{"source_region", "target_region"},
	)

	dDoSDetections = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_ddos_detections_total",
			Help: "Total DDoS attack detections",
		},
		[]string{"region", "attack_type"},
	)

	trafficShapingActions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_traffic_shaping_actions_total",
			Help: "Total traffic shaping actions taken",
		},
		[]string{"action_type"},
	)
)

// IntelligentGlobalRouter implements global traffic routing
type IntelligentGlobalRouter struct {
	mu                  sync.RWMutex
	regions             map[string]*RegionEndpoint
	latencyMatrix       map[string]map[string]float64 // source -> target -> latency(ms)
	costMatrix          map[string]map[string]float64 // source -> target -> cost per GB
	routingAlgorithm    RoutingAlgorithm
	dDoSProtector       *DDoSProtector
	trafficShaper       *TrafficShaper
	anycastGroups       map[string]*AnycastGroup
	healthChecker       *HealthChecker
	metricsCollector    *MetricsCollector
	stopCh              chan struct{}
	wg                  sync.WaitGroup
	measurementInterval time.Duration
}

// RegionEndpoint represents a regional endpoint
type RegionEndpoint struct {
	RegionID      string
	IPAddress     string
	IPv6Address   string
	Port          int
	Protocol      string // "tcp", "udp", "quic"
	Capacity      int64  // Mbps
	CurrentLoad   int64  // Mbps
	HealthStatus  HealthStatus
	Priority      int
	Weight        int
	Metadata      map[string]string
	LastUpdate    time.Time
}

// HealthStatus represents endpoint health
type HealthStatus struct {
	State            string  // "healthy", "degraded", "unhealthy"
	ResponseTime     time.Duration
	PacketLoss       float64 // Percentage
	Availability     float64 // Percentage
	ConsecutiveFailures int
	LastCheck        time.Time
}

// RoutingAlgorithm defines routing strategy
type RoutingAlgorithm int

const (
	RoutingLatencyBased RoutingAlgorithm = iota // Route to lowest latency
	RoutingCostOptimized                        // Route to lowest cost
	RoutingLoadBalanced                         // Distribute load evenly
	RoutingGeoProximity                         // Route to nearest geographic region
	RoutingHybrid                               // Combine latency + cost + load
)

// RoutingRequest represents a routing decision request
type RoutingRequest struct {
	SourceIP        string
	DestinationID   string
	Protocol        string
	PayloadSize     int64
	QoSRequirement  QoSRequirement
	Priority        int
	Metadata        map[string]string
	Timestamp       time.Time
}

// QoSRequirement defines quality of service needs
type QoSRequirement struct {
	MaxLatencyMS     float64
	MinBandwidthMbps int64
	MaxPacketLoss    float64
	RequireEncryption bool
	TrafficClass     string // "interactive", "bulk", "realtime"
}

// RoutingDecision represents the routing decision
type RoutingDecision struct {
	TargetRegion    string
	TargetEndpoint  string
	ExpectedLatency float64
	EstimatedCost   float64
	DecisionTime    time.Duration
	Algorithm       string
	RoutingPath     []string
	QoSGuarantee    bool
	Timestamp       time.Time
}

// AnycastGroup represents an anycast routing group
type AnycastGroup struct {
	GroupID     string
	ServiceName string
	VirtualIP   string
	Endpoints   []string // Region IDs
	Policy      AnycastPolicy
}

// AnycastPolicy defines anycast routing policy
type AnycastPolicy struct {
	SelectionAlgorithm string // "nearest", "load_balanced", "cost_optimized"
	HealthCheckEnabled bool
	FailoverTimeout    time.Duration
	StickySessions     bool
	SessionTimeout     time.Duration
}

// DDoSProtector implements DDoS protection
type DDoSProtector struct {
	mu                  sync.RWMutex
	enabled             bool
	rateLimits          map[string]*RateLimit
	blacklist           map[string]time.Time
	whitelist           map[string]bool
	detectionAlgorithm  string // "rate", "pattern", "ml"
	thresholds          DDoSThresholds
	mitigationActions   []string
	alertWebhooks       []string
}

// RateLimit tracks rate limiting state
type RateLimit struct {
	SourceIP        string
	RequestsPerSec  float64
	BytesPerSec     int64
	LastRequest     time.Time
	RequestCount    int64
	ByteCount       int64
	WindowStart     time.Time
	WindowDuration  time.Duration
	Blocked         bool
}

// DDoSThresholds defines attack detection thresholds
type DDoSThresholds struct {
	MaxRequestsPerSec    int64
	MaxBytesPerSec       int64
	MaxConnections       int64
	SynFloodThreshold    int64
	UdpFloodThreshold    int64
	IcmpFloodThreshold   int64
	DnsAmplificationDetection bool
}

// TrafficShaper implements traffic shaping
type TrafficShaper struct {
	mu              sync.RWMutex
	enabled         bool
	policies        map[string]*ShapingPolicy
	priorityQueues  map[string]*PriorityQueue
	bandwidthLimits map[string]int64
}

// ShapingPolicy defines traffic shaping rules
type ShapingPolicy struct {
	PolicyID        string
	MatchCriteria   MatchCriteria
	Actions         []ShapingAction
	Priority        int
	Enabled         bool
}

// MatchCriteria defines traffic matching rules
type MatchCriteria struct {
	SourceIP        string
	DestinationIP   string
	SourcePort      int
	DestinationPort int
	Protocol        string
	TrafficClass    string
}

// ShapingAction defines actions to take
type ShapingAction struct {
	ActionType      string // "rate_limit", "priority", "drop", "mark_dscp"
	Parameters      map[string]interface{}
}

// PriorityQueue implements traffic prioritization
type PriorityQueue struct {
	QueueID       string
	Priority      int
	MaxSize       int64
	CurrentSize   int64
	Packets       [][]byte
	mu            sync.Mutex
}

// HealthChecker performs endpoint health checks
type HealthChecker struct {
	mu               sync.RWMutex
	checkInterval    time.Duration
	timeout          time.Duration
	healthyThreshold int
	unhealthyThreshold int
}

// MetricsCollector collects routing metrics
type MetricsCollector struct {
	mu                  sync.RWMutex
	routingDecisions    int64
	totalBytesRouted    int64
	averageLatency      time.Duration
	p50Latency          time.Duration
	p95Latency          time.Duration
	p99Latency          time.Duration
	errorRate           float64
	latencyHistory      []time.Duration
	decisionLatencies   []time.Duration
}

// NewIntelligentGlobalRouter creates a new global router
func NewIntelligentGlobalRouter(cfg *RouterConfig) (*IntelligentGlobalRouter, error) {
	if cfg == nil {
		return nil, fmt.Errorf("router config cannot be nil")
	}

	router := &IntelligentGlobalRouter{
		regions:             make(map[string]*RegionEndpoint),
		latencyMatrix:       make(map[string]map[string]float64),
		costMatrix:          make(map[string]map[string]float64),
		routingAlgorithm:    cfg.RoutingAlgorithm,
		anycastGroups:       make(map[string]*AnycastGroup),
		stopCh:              make(chan struct{}),
		measurementInterval: cfg.MeasurementInterval,
		metricsCollector: &MetricsCollector{
			latencyHistory:    make([]time.Duration, 0, 1000),
			decisionLatencies: make([]time.Duration, 0, 1000),
		},
	}

	// Initialize DDoS protector
	if cfg.EnableDDoSProtection {
		router.dDoSProtector = &DDoSProtector{
			enabled:            true,
			rateLimits:         make(map[string]*RateLimit),
			blacklist:          make(map[string]time.Time),
			whitelist:          make(map[string]bool),
			detectionAlgorithm: cfg.DDoSDetectionAlgorithm,
			thresholds:         cfg.DDoSThresholds,
		}
	}

	// Initialize traffic shaper
	if cfg.EnableTrafficShaping {
		router.trafficShaper = &TrafficShaper{
			enabled:         true,
			policies:        make(map[string]*ShapingPolicy),
			priorityQueues:  make(map[string]*PriorityQueue),
			bandwidthLimits: make(map[string]int64),
		}
	}

	// Initialize health checker
	router.healthChecker = &HealthChecker{
		checkInterval:      cfg.HealthCheckInterval,
		timeout:            cfg.HealthCheckTimeout,
		healthyThreshold:   3,
		unhealthyThreshold: 3,
	}

	// Register regions
	for _, endpoint := range cfg.RegionEndpoints {
		if err := router.RegisterRegion(endpoint); err != nil {
			return nil, fmt.Errorf("failed to register region %s: %w", endpoint.RegionID, err)
		}
	}

	return router, nil
}

// RouterConfig defines router configuration
type RouterConfig struct {
	RegionEndpoints          []*RegionEndpoint
	RoutingAlgorithm         RoutingAlgorithm
	EnableDDoSProtection     bool
	DDoSDetectionAlgorithm   string
	DDoSThresholds           DDoSThresholds
	EnableTrafficShaping     bool
	HealthCheckInterval      time.Duration
	HealthCheckTimeout       time.Duration
	MeasurementInterval      time.Duration
}

// Start starts the router
func (igr *IntelligentGlobalRouter) Start(ctx context.Context) error {
	// Start latency measurement
	igr.wg.Add(1)
	go igr.measureLatencies(ctx)

	// Start health checks
	igr.wg.Add(1)
	go igr.performHealthChecks(ctx)

	// Start DDoS protection if enabled
	if igr.dDoSProtector != nil && igr.dDoSProtector.enabled {
		igr.wg.Add(1)
		go igr.runDDoSProtection(ctx)
	}

	// Start metrics collection
	igr.wg.Add(1)
	go igr.collectMetrics(ctx)

	return nil
}

// Stop stops the router
func (igr *IntelligentGlobalRouter) Stop() error {
	close(igr.stopCh)
	igr.wg.Wait()
	return nil
}

// RegisterRegion registers a new region endpoint
func (igr *IntelligentGlobalRouter) RegisterRegion(endpoint *RegionEndpoint) error {
	igr.mu.Lock()
	defer igr.mu.Unlock()

	if endpoint == nil {
		return fmt.Errorf("endpoint cannot be nil")
	}

	if _, exists := igr.regions[endpoint.RegionID]; exists {
		return fmt.Errorf("region %s already registered", endpoint.RegionID)
	}

	endpoint.LastUpdate = time.Now()
	igr.regions[endpoint.RegionID] = endpoint

	// Initialize latency and cost matrices
	if igr.latencyMatrix[endpoint.RegionID] == nil {
		igr.latencyMatrix[endpoint.RegionID] = make(map[string]float64)
	}
	if igr.costMatrix[endpoint.RegionID] == nil {
		igr.costMatrix[endpoint.RegionID] = make(map[string]float64)
	}

	return nil
}

// RouteTraffic makes an intelligent routing decision
// Target: <50ms decision latency, <100ms p99
func (igr *IntelligentGlobalRouter) RouteTraffic(ctx context.Context, req *RoutingRequest) (*RoutingDecision, error) {
	startTime := time.Now()
	defer func() {
		latency := time.Since(startTime)
		routingDecisions.WithLabelValues(igr.algorithmName()).Observe(float64(latency.Milliseconds()))

		igr.metricsCollector.mu.Lock()
		igr.metricsCollector.decisionLatencies = append(igr.metricsCollector.decisionLatencies, latency)
		// Keep only last 1000 measurements
		if len(igr.metricsCollector.decisionLatencies) > 1000 {
			igr.metricsCollector.decisionLatencies = igr.metricsCollector.decisionLatencies[1:]
		}
		igr.metricsCollector.mu.Unlock()
	}()

	// Check DDoS protection
	if igr.dDoSProtector != nil && igr.dDoSProtector.enabled {
		if blocked, reason := igr.checkDDoSProtection(req); blocked {
			routingErrors.WithLabelValues("ddos_blocked").Inc()
			return nil, fmt.Errorf("request blocked by DDoS protection: %s", reason)
		}
	}

	// Get eligible regions
	eligibleRegions := igr.filterEligibleRegions(req)
	if len(eligibleRegions) == 0 {
		routingErrors.WithLabelValues("no_eligible_regions").Inc()
		return nil, fmt.Errorf("no eligible regions for routing")
	}

	// Score regions based on algorithm
	var selectedRegion *RegionEndpoint
	var expectedLatency, estimatedCost float64

	switch igr.routingAlgorithm {
	case RoutingLatencyBased:
		selectedRegion = igr.selectByLatency(eligibleRegions, req)
	case RoutingCostOptimized:
		selectedRegion = igr.selectByCost(eligibleRegions, req)
	case RoutingLoadBalanced:
		selectedRegion = igr.selectByLoad(eligibleRegions, req)
	case RoutingGeoProximity:
		selectedRegion = igr.selectByProximity(eligibleRegions, req)
	case RoutingHybrid:
		selectedRegion = igr.selectHybrid(eligibleRegions, req)
	default:
		selectedRegion = igr.selectByLatency(eligibleRegions, req)
	}

	if selectedRegion == nil {
		routingErrors.WithLabelValues("selection_failed").Inc()
		return nil, fmt.Errorf("failed to select target region")
	}

	// Get expected latency
	sourceRegion := igr.getSourceRegion(req.SourceIP)
	if latencies, ok := igr.latencyMatrix[sourceRegion]; ok {
		expectedLatency = latencies[selectedRegion.RegionID]
	}

	// Calculate estimated cost
	estimatedCost = igr.calculateRoutingCost(sourceRegion, selectedRegion.RegionID, req.PayloadSize)

	// Build routing path
	routingPath := igr.computePath(sourceRegion, selectedRegion.RegionID)

	decision := &RoutingDecision{
		TargetRegion:    selectedRegion.RegionID,
		TargetEndpoint:  fmt.Sprintf("%s:%d", selectedRegion.IPAddress, selectedRegion.Port),
		ExpectedLatency: expectedLatency,
		EstimatedCost:   estimatedCost,
		DecisionTime:    time.Since(startTime),
		Algorithm:       igr.algorithmName(),
		RoutingPath:     routingPath,
		QoSGuarantee:    igr.canGuaranteeQoS(selectedRegion, req.QoSRequirement),
		Timestamp:       time.Now(),
	}

	// Update metrics
	igr.metricsCollector.mu.Lock()
	igr.metricsCollector.routingDecisions++
	igr.metricsCollector.totalBytesRouted += req.PayloadSize
	igr.metricsCollector.mu.Unlock()

	// Track routed traffic
	routedTraffic.WithLabelValues(sourceRegion, selectedRegion.RegionID, req.Protocol).
		Add(float64(req.PayloadSize))

	return decision, nil
}

// filterEligibleRegions filters regions based on request requirements
func (igr *IntelligentGlobalRouter) filterEligibleRegions(req *RoutingRequest) []*RegionEndpoint {
	igr.mu.RLock()
	defer igr.mu.RUnlock()

	var eligible []*RegionEndpoint

	for _, region := range igr.regions {
		// Check health
		if region.HealthStatus.State != "healthy" && region.HealthStatus.State != "degraded" {
			continue
		}

		// Check capacity
		if region.CurrentLoad >= region.Capacity {
			continue
		}

		// Check latency requirement
		if req.QoSRequirement.MaxLatencyMS > 0 {
			sourceRegion := igr.getSourceRegion(req.SourceIP)
			if latencies, ok := igr.latencyMatrix[sourceRegion]; ok {
				if latency, exists := latencies[region.RegionID]; exists {
					if latency > req.QoSRequirement.MaxLatencyMS {
						continue
					}
				}
			}
		}

		// Check protocol support
		if req.Protocol != "" && req.Protocol != region.Protocol {
			continue
		}

		eligible = append(eligible, region)
	}

	return eligible
}

// selectByLatency selects region with lowest latency
func (igr *IntelligentGlobalRouter) selectByLatency(regions []*RegionEndpoint, req *RoutingRequest) *RegionEndpoint {
	sourceRegion := igr.getSourceRegion(req.SourceIP)
	latencies, ok := igr.latencyMatrix[sourceRegion]
	if !ok {
		return regions[0] // Fallback to first region
	}

	var bestRegion *RegionEndpoint
	minLatency := math.MaxFloat64

	for _, region := range regions {
		if latency, exists := latencies[region.RegionID]; exists {
			if latency < minLatency {
				minLatency = latency
				bestRegion = region
			}
		}
	}

	if bestRegion == nil {
		return regions[0]
	}
	return bestRegion
}

// selectByCost selects region with lowest cost
func (igr *IntelligentGlobalRouter) selectByCost(regions []*RegionEndpoint, req *RoutingRequest) *RegionEndpoint {
	sourceRegion := igr.getSourceRegion(req.SourceIP)
	costs, ok := igr.costMatrix[sourceRegion]
	if !ok {
		return regions[0]
	}

	var bestRegion *RegionEndpoint
	minCost := math.MaxFloat64

	for _, region := range regions {
		if cost, exists := costs[region.RegionID]; exists {
			totalCost := cost * float64(req.PayloadSize) / (1024 * 1024 * 1024) // Cost per GB
			if totalCost < minCost {
				minCost = totalCost
				bestRegion = region
			}
		}
	}

	if bestRegion == nil {
		return regions[0]
	}
	return bestRegion
}

// selectByLoad selects region with lowest load
func (igr *IntelligentGlobalRouter) selectByLoad(regions []*RegionEndpoint, req *RoutingRequest) *RegionEndpoint {
	var bestRegion *RegionEndpoint
	minUtilization := math.MaxFloat64

	for _, region := range regions {
		utilization := float64(region.CurrentLoad) / float64(region.Capacity)
		if utilization < minUtilization {
			minUtilization = utilization
			bestRegion = region
		}
	}

	if bestRegion == nil {
		return regions[0]
	}
	return bestRegion
}

// selectByProximity selects geographically nearest region
func (igr *IntelligentGlobalRouter) selectByProximity(regions []*RegionEndpoint, req *RoutingRequest) *RegionEndpoint {
	// For simplicity, use latency as proxy for geographic proximity
	return igr.selectByLatency(regions, req)
}

// selectHybrid selects region using hybrid algorithm (latency + cost + load)
func (igr *IntelligentGlobalRouter) selectHybrid(regions []*RegionEndpoint, req *RoutingRequest) *RegionEndpoint {
	sourceRegion := igr.getSourceRegion(req.SourceIP)

	type regionScore struct {
		region *RegionEndpoint
		score  float64
	}

	scores := make([]regionScore, 0, len(regions))

	for _, region := range regions {
		// Factor 1: Latency (40% weight)
		latencyScore := 1.0
		if latencies, ok := igr.latencyMatrix[sourceRegion]; ok {
			if latency, exists := latencies[region.RegionID]; exists {
				// Normalize: lower latency = higher score
				latencyScore = 1.0 / (1.0 + latency/100.0)
			}
		}

		// Factor 2: Cost (30% weight)
		costScore := 1.0
		if costs, ok := igr.costMatrix[sourceRegion]; ok {
			if cost, exists := costs[region.RegionID]; exists {
				// Normalize: lower cost = higher score
				costScore = 1.0 / (1.0 + cost)
			}
		}

		// Factor 3: Load (30% weight)
		utilization := float64(region.CurrentLoad) / float64(region.Capacity)
		loadScore := 1.0 - utilization

		// Calculate weighted score
		totalScore := (latencyScore * 0.4) + (costScore * 0.3) + (loadScore * 0.3)

		scores = append(scores, regionScore{
			region: region,
			score:  totalScore,
		})
	}

	// Find highest score
	var bestRegion *RegionEndpoint
	maxScore := -math.MaxFloat64

	for _, rs := range scores {
		if rs.score > maxScore {
			maxScore = rs.score
			bestRegion = rs.region
		}
	}

	if bestRegion == nil {
		return regions[0]
	}
	return bestRegion
}

// getSourceRegion determines source region from IP
func (igr *IntelligentGlobalRouter) getSourceRegion(sourceIP string) string {
	// TODO: Implement IP geolocation
	// For now, return "default"
	return "default"
}

// calculateRoutingCost calculates cost for routing
func (igr *IntelligentGlobalRouter) calculateRoutingCost(sourceRegion, targetRegion string, payloadSize int64) float64 {
	igr.mu.RLock()
	defer igr.mu.RUnlock()

	if costs, ok := igr.costMatrix[sourceRegion]; ok {
		if costPerGB, exists := costs[targetRegion]; exists {
			return costPerGB * float64(payloadSize) / (1024 * 1024 * 1024)
		}
	}

	return 0.0 // Default: no cost
}

// computePath computes routing path between regions
func (igr *IntelligentGlobalRouter) computePath(sourceRegion, targetRegion string) []string {
	// Simple: direct path
	// TODO: Implement multi-hop routing for better optimization
	return []string{sourceRegion, targetRegion}
}

// canGuaranteeQoS checks if QoS can be guaranteed
func (igr *IntelligentGlobalRouter) canGuaranteeQoS(region *RegionEndpoint, qos QoSRequirement) bool {
	// Check latency
	if qos.MaxLatencyMS > 0 && region.HealthStatus.ResponseTime.Milliseconds() > int64(qos.MaxLatencyMS) {
		return false
	}

	// Check bandwidth
	availableBandwidth := region.Capacity - region.CurrentLoad
	if qos.MinBandwidthMbps > 0 && availableBandwidth < qos.MinBandwidthMbps {
		return false
	}

	// Check packet loss
	if qos.MaxPacketLoss > 0 && region.HealthStatus.PacketLoss > qos.MaxPacketLoss {
		return false
	}

	return true
}

// checkDDoSProtection checks if request should be blocked
func (igr *IntelligentGlobalRouter) checkDDoSProtection(req *RoutingRequest) (bool, string) {
	igr.dDoSProtector.mu.RLock()
	defer igr.dDoSProtector.mu.RUnlock()

	// Check whitelist
	if igr.dDoSProtector.whitelist[req.SourceIP] {
		return false, ""
	}

	// Check blacklist
	if blockedUntil, exists := igr.dDoSProtector.blacklist[req.SourceIP]; exists {
		if time.Now().Before(blockedUntil) {
			return true, "IP blacklisted"
		}
	}

	// Check rate limit
	if rateLimit, exists := igr.dDoSProtector.rateLimits[req.SourceIP]; exists {
		if rateLimit.Blocked {
			return true, "Rate limit exceeded"
		}

		// Check requests per second
		elapsed := time.Since(rateLimit.WindowStart).Seconds()
		if elapsed > 0 {
			rps := float64(rateLimit.RequestCount) / elapsed
			if int64(rps) > igr.dDoSProtector.thresholds.MaxRequestsPerSec {
				return true, fmt.Sprintf("Requests per second exceeded: %.2f", rps)
			}
		}
	}

	return false, ""
}

// measureLatencies periodically measures inter-region latencies
func (igr *IntelligentGlobalRouter) measureLatencies(ctx context.Context) {
	defer igr.wg.Done()

	ticker := time.NewTicker(igr.measurementInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-igr.stopCh:
			return
		case <-ticker.C:
			igr.performLatencyMeasurements()
		}
	}
}

// performLatencyMeasurements measures latencies between all region pairs
func (igr *IntelligentGlobalRouter) performLatencyMeasurements() {
	igr.mu.RLock()
	regions := make([]*RegionEndpoint, 0, len(igr.regions))
	for _, region := range igr.regions {
		regions = append(regions, region)
	}
	igr.mu.RUnlock()

	for _, source := range regions {
		for _, target := range regions {
			if source.RegionID == target.RegionID {
				continue
			}

			// Perform ping measurement
			latency := igr.measureLatency(source, target)

			igr.mu.Lock()
			if igr.latencyMatrix[source.RegionID] == nil {
				igr.latencyMatrix[source.RegionID] = make(map[string]float64)
			}
			igr.latencyMatrix[source.RegionID][target.RegionID] = latency
			igr.mu.Unlock()

			// Update metric
			measurementLatency.WithLabelValues(source.RegionID, target.RegionID).Set(latency)
		}
	}
}

// measureLatency measures latency between two endpoints
func (igr *IntelligentGlobalRouter) measureLatency(source, target *RegionEndpoint) float64 {
	startTime := time.Now()

	// Perform TCP connection
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", target.IPAddress, target.Port), 5*time.Second)
	if err != nil {
		return 1000.0 // Return high latency on error
	}
	defer conn.Close()

	latency := time.Since(startTime).Seconds() * 1000.0 // Convert to milliseconds
	return latency
}

// performHealthChecks performs periodic health checks
func (igr *IntelligentGlobalRouter) performHealthChecks(ctx context.Context) {
	defer igr.wg.Done()

	ticker := time.NewTicker(igr.healthChecker.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-igr.stopCh:
			return
		case <-ticker.C:
			igr.checkAllRegions()
		}
	}
}

// checkAllRegions checks health of all regions
func (igr *IntelligentGlobalRouter) checkAllRegions() {
	igr.mu.Lock()
	defer igr.mu.Unlock()

	for _, region := range igr.regions {
		healthy := igr.checkRegionHealth(region)

		if healthy {
			region.HealthStatus.ConsecutiveFailures = 0
			if region.HealthStatus.State != "healthy" {
				region.HealthStatus.State = "healthy"
			}
		} else {
			region.HealthStatus.ConsecutiveFailures++
			if region.HealthStatus.ConsecutiveFailures >= igr.healthChecker.unhealthyThreshold {
				region.HealthStatus.State = "unhealthy"
			} else {
				region.HealthStatus.State = "degraded"
			}
		}

		region.HealthStatus.LastCheck = time.Now()
	}
}

// checkRegionHealth checks if a region is healthy
func (igr *IntelligentGlobalRouter) checkRegionHealth(region *RegionEndpoint) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", region.IPAddress, region.Port), igr.healthChecker.timeout)
	if err != nil {
		return false
	}
	defer conn.Close()
	return true
}

// runDDoSProtection runs DDoS protection loop
func (igr *IntelligentGlobalRouter) runDDoSProtection(ctx context.Context) {
	defer igr.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-igr.stopCh:
			return
		case <-ticker.C:
			igr.updateDDoSProtection()
		}
	}
}

// updateDDoSProtection updates DDoS protection state
func (igr *IntelligentGlobalRouter) updateDDoSProtection() {
	igr.dDoSProtector.mu.Lock()
	defer igr.dDoSProtector.mu.Unlock()

	now := time.Now()

	// Clean up expired blacklist entries
	for ip, expiry := range igr.dDoSProtector.blacklist {
		if now.After(expiry) {
			delete(igr.dDoSProtector.blacklist, ip)
		}
	}

	// Reset rate limit windows
	for ip, rateLimit := range igr.dDoSProtector.rateLimits {
		if now.Sub(rateLimit.WindowStart) > rateLimit.WindowDuration {
			rateLimit.WindowStart = now
			rateLimit.RequestCount = 0
			rateLimit.ByteCount = 0
			rateLimit.Blocked = false
		}
	}
}

// collectMetrics collects routing metrics
func (igr *IntelligentGlobalRouter) collectMetrics(ctx context.Context) {
	defer igr.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-igr.stopCh:
			return
		case <-ticker.C:
			igr.calculatePercentiles()
		}
	}
}

// calculatePercentiles calculates latency percentiles
func (igr *IntelligentGlobalRouter) calculatePercentiles() {
	igr.metricsCollector.mu.Lock()
	defer igr.metricsCollector.mu.Unlock()

	if len(igr.metricsCollector.decisionLatencies) == 0 {
		return
	}

	// Calculate p50, p95, p99
	// TODO: Implement proper percentile calculation
	igr.metricsCollector.p99Latency = igr.metricsCollector.decisionLatencies[len(igr.metricsCollector.decisionLatencies)-1]
}

// algorithmName returns the routing algorithm name
func (igr *IntelligentGlobalRouter) algorithmName() string {
	switch igr.routingAlgorithm {
	case RoutingLatencyBased:
		return "latency"
	case RoutingCostOptimized:
		return "cost"
	case RoutingLoadBalanced:
		return "load"
	case RoutingGeoProximity:
		return "proximity"
	case RoutingHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// GetMetrics returns current routing metrics
func (igr *IntelligentGlobalRouter) GetMetrics() *MetricsCollector {
	igr.metricsCollector.mu.RLock()
	defer igr.metricsCollector.mu.RUnlock()

	return &MetricsCollector{
		routingDecisions:  igr.metricsCollector.routingDecisions,
		totalBytesRouted:  igr.metricsCollector.totalBytesRouted,
		averageLatency:    igr.metricsCollector.averageLatency,
		p50Latency:        igr.metricsCollector.p50Latency,
		p95Latency:        igr.metricsCollector.p95Latency,
		p99Latency:        igr.metricsCollector.p99Latency,
		errorRate:         igr.metricsCollector.errorRate,
	}
}
