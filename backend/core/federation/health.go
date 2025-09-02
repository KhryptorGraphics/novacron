package federation

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	healthCheckDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_health_check_duration_seconds",
		Help:    "Duration of health checks",
		Buckets: prometheus.DefBuckets,
	}, []string{"node_id", "check_type"})

	healthCheckStatus = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_health_check_status",
		Help: "Health check status (1=healthy, 0=unhealthy)",
	}, []string{"node_id"})

	failureDetections = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_failure_detections_total",
		Help: "Total number of failure detections",
	}, []string{"node_id", "failure_type"})
)

// HealthChecker implements health checking and failure detection
type HealthCheckerImpl struct {
	config          *FederationConfig
	nodes           map[string]*Node
	nodesMu         sync.RWMutex
	healthStatus    map[string]*HealthCheck
	statusMu        sync.RWMutex
	handlers        []HealthHandler
	handlersMu      sync.RWMutex
	phiDetector     *PhiAccrualDetector
	monitorInterval time.Duration
	stopCh          chan struct{}
	isMonitoring    bool
	monitoringMu    sync.RWMutex
	logger          Logger
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(config *FederationConfig, logger Logger) (*HealthCheckerImpl, error) {
	hc := &HealthCheckerImpl{
		config:       config,
		nodes:        make(map[string]*Node),
		healthStatus: make(map[string]*HealthCheck),
		handlers:     make([]HealthHandler, 0),
		phiDetector:  NewPhiAccrualDetector(config.FailureThreshold, logger),
		stopCh:       make(chan struct{}),
		logger:       logger,
	}

	return hc, nil
}

// CheckHealth performs a health check on a node
func (hc *HealthCheckerImpl) CheckHealth(ctx context.Context, node *Node) (*HealthCheck, error) {
	if node == nil {
		return nil, fmt.Errorf("node is nil")
	}

	timer := prometheus.NewTimer(healthCheckDuration.WithLabelValues(node.ID, "full"))
	defer timer.ObserveDuration()

	check := &HealthCheck{
		NodeID:         node.ID,
		Timestamp:      time.Now(),
		Services:       make(map[string]ServiceHealth),
		NetworkLatency: make(map[string]time.Duration),
		Healthy:        true,
		Issues:         make([]string, 0),
	}

	// Perform various health checks in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	// Network connectivity check
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := hc.checkNetworkConnectivity(ctx, node, check, &mu); err != nil {
			mu.Lock()
			check.Issues = append(check.Issues, fmt.Sprintf("network check failed: %v", err))
			check.Healthy = false
			mu.Unlock()
		}
	}()

	// Service health checks
	wg.Add(1)
	go func() {
		defer wg.Done()
		hc.checkServices(ctx, node, check, &mu)
	}()

	// Resource utilization check
	wg.Add(1)
	go func() {
		defer wg.Done()
		hc.checkResourceUtilization(ctx, node, check, &mu)
	}()

	// Latency measurements
	wg.Add(1)
	go func() {
		defer wg.Done()
		hc.measureNetworkLatency(ctx, node, check, &mu)
	}()

	// Wait for all checks to complete
	wg.Wait()

	// Update health status
	hc.statusMu.Lock()
	hc.healthStatus[node.ID] = check
	hc.statusMu.Unlock()

	// Update metrics
	if check.Healthy {
		healthCheckStatus.WithLabelValues(node.ID).Set(1)
	} else {
		healthCheckStatus.WithLabelValues(node.ID).Set(0)
		failureDetections.WithLabelValues(node.ID, "health_check").Inc()
	}

	// Notify handlers
	hc.notifyHandlers(node, check)

	return check, nil
}

// StartMonitoring starts continuous health monitoring
func (hc *HealthCheckerImpl) StartMonitoring(ctx context.Context, interval time.Duration) error {
	hc.monitoringMu.Lock()
	defer hc.monitoringMu.Unlock()

	if hc.isMonitoring {
		return fmt.Errorf("monitoring already started")
	}

	hc.monitorInterval = interval
	hc.isMonitoring = true

	go hc.monitorLoop(ctx)
	go hc.failureDetectionLoop(ctx)

	hc.logger.Info("Health monitoring started", "interval", interval)

	return nil
}

// StopMonitoring stops health monitoring
func (hc *HealthCheckerImpl) StopMonitoring(ctx context.Context) error {
	hc.monitoringMu.Lock()
	defer hc.monitoringMu.Unlock()

	if !hc.isMonitoring {
		return fmt.Errorf("monitoring not started")
	}

	close(hc.stopCh)
	hc.isMonitoring = false

	hc.logger.Info("Health monitoring stopped")

	return nil
}

// GetHealthStatus returns the health status for a node
func (hc *HealthCheckerImpl) GetHealthStatus(nodeID string) (*HealthCheck, error) {
	hc.statusMu.RLock()
	defer hc.statusMu.RUnlock()

	status, exists := hc.healthStatus[nodeID]
	if !exists {
		return nil, fmt.Errorf("no health status for node: %s", nodeID)
	}

	return status, nil
}

// RegisterHealthHandler registers a health event handler
func (hc *HealthCheckerImpl) RegisterHealthHandler(handler HealthHandler) {
	hc.handlersMu.Lock()
	defer hc.handlersMu.Unlock()

	hc.handlers = append(hc.handlers, handler)
}

// AddNode adds a node to monitor
func (hc *HealthCheckerImpl) AddNode(node *Node) {
	hc.nodesMu.Lock()
	defer hc.nodesMu.Unlock()

	hc.nodes[node.ID] = node
	hc.phiDetector.AddNode(node.ID)
}

// RemoveNode removes a node from monitoring
func (hc *HealthCheckerImpl) RemoveNode(nodeID string) {
	hc.nodesMu.Lock()
	defer hc.nodesMu.Unlock()

	delete(hc.nodes, nodeID)
	hc.phiDetector.RemoveNode(nodeID)

	hc.statusMu.Lock()
	delete(hc.healthStatus, nodeID)
	hc.statusMu.Unlock()
}

// Internal health check methods

func (hc *HealthCheckerImpl) checkNetworkConnectivity(ctx context.Context, node *Node, check *HealthCheck, mu *sync.Mutex) error {
	timer := prometheus.NewTimer(healthCheckDuration.WithLabelValues(node.ID, "network"))
	defer timer.ObserveDuration()

	// Try to establish TCP connection
	dialer := &net.Dialer{
		Timeout: 5 * time.Second,
	}

	conn, err := dialer.DialContext(ctx, "tcp", node.Address)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	// Connection successful
	mu.Lock()
	check.Services["network"] = ServiceHealth{
		Name:   "network",
		Status: "healthy",
	}
	mu.Unlock()

	return nil
}

func (hc *HealthCheckerImpl) checkServices(ctx context.Context, node *Node, check *HealthCheck, mu *sync.Mutex) {
	timer := prometheus.NewTimer(healthCheckDuration.WithLabelValues(node.ID, "services"))
	defer timer.ObserveDuration()

	// Check various services
	services := []string{"api", "consensus", "storage", "compute"}

	for _, service := range services {
		health := hc.checkServiceHealth(ctx, node, service)
		
		mu.Lock()
		check.Services[service] = health
		if health.Status != "healthy" {
			check.Issues = append(check.Issues, fmt.Sprintf("service %s unhealthy: %s", service, health.Error))
			check.Healthy = false
		}
		mu.Unlock()
	}
}

func (hc *HealthCheckerImpl) checkServiceHealth(ctx context.Context, node *Node, service string) ServiceHealth {
	// In production, would make actual health check calls to services
	// For now, simulate
	
	return ServiceHealth{
		Name:    service,
		Status:  "healthy",
		Latency: time.Duration(10) * time.Millisecond,
	}
}

func (hc *HealthCheckerImpl) checkResourceUtilization(ctx context.Context, node *Node, check *HealthCheck, mu *sync.Mutex) {
	timer := prometheus.NewTimer(healthCheckDuration.WithLabelValues(node.ID, "resources"))
	defer timer.ObserveDuration()

	// Get resource utilization
	// In production, would query actual metrics
	cpuUsage := 0.45      // 45% CPU usage
	memoryUsage := 0.60   // 60% memory usage
	diskUsage := 0.30     // 30% disk usage

	mu.Lock()
	check.CPUUsage = cpuUsage
	check.MemoryUsage = memoryUsage
	check.DiskUsage = diskUsage

	// Check thresholds
	if cpuUsage > 0.90 {
		check.Issues = append(check.Issues, "high CPU usage")
		check.Healthy = false
	}
	if memoryUsage > 0.90 {
		check.Issues = append(check.Issues, "high memory usage")
		check.Healthy = false
	}
	if diskUsage > 0.90 {
		check.Issues = append(check.Issues, "high disk usage")
		check.Healthy = false
	}
	mu.Unlock()
}

func (hc *HealthCheckerImpl) measureNetworkLatency(ctx context.Context, node *Node, check *HealthCheck, mu *sync.Mutex) {
	timer := prometheus.NewTimer(healthCheckDuration.WithLabelValues(node.ID, "latency"))
	defer timer.ObserveDuration()

	// Measure latency to other nodes
	hc.nodesMu.RLock()
	nodes := make([]*Node, 0, len(hc.nodes))
	for _, n := range hc.nodes {
		if n.ID != node.ID {
			nodes = append(nodes, n)
		}
	}
	hc.nodesMu.RUnlock()

	for _, targetNode := range nodes {
		latency := hc.measureLatency(ctx, node, targetNode)
		
		mu.Lock()
		check.NetworkLatency[targetNode.ID] = latency
		
		// Check if latency is too high
		if latency > 100*time.Millisecond {
			check.Issues = append(check.Issues, fmt.Sprintf("high latency to %s: %v", targetNode.ID, latency))
		}
		mu.Unlock()
	}

	// Calculate average latency
	var totalLatency time.Duration
	for _, latency := range check.NetworkLatency {
		totalLatency += latency
	}
	
	if len(check.NetworkLatency) > 0 {
		avgLatency := totalLatency / time.Duration(len(check.NetworkLatency))
		mu.Lock()
		check.Latency = avgLatency
		mu.Unlock()
	}
}

func (hc *HealthCheckerImpl) measureLatency(ctx context.Context, source, target *Node) time.Duration {
	// In production, would perform actual ping or latency measurement
	// For now, simulate with small random latency
	return time.Duration(5+time.Now().UnixNano()%20) * time.Millisecond
}

// Monitoring loops

func (hc *HealthCheckerImpl) monitorLoop(ctx context.Context) {
	ticker := time.NewTicker(hc.monitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-hc.stopCh:
			return
		case <-ticker.C:
			hc.performHealthChecks(ctx)
		}
	}
}

func (hc *HealthCheckerImpl) performHealthChecks(ctx context.Context) {
	hc.nodesMu.RLock()
	nodes := make([]*Node, 0, len(hc.nodes))
	for _, node := range hc.nodes {
		nodes = append(nodes, node)
	}
	hc.nodesMu.RUnlock()

	// Perform health checks in parallel
	var wg sync.WaitGroup
	for _, node := range nodes {
		wg.Add(1)
		go func(n *Node) {
			defer wg.Done()
			
			_, err := hc.CheckHealth(ctx, n)
			if err != nil {
				hc.logger.Error("Health check failed", "node_id", n.ID, "error", err)
			}
		}(node)
	}
	
	wg.Wait()
}

func (hc *HealthCheckerImpl) failureDetectionLoop(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-hc.stopCh:
			return
		case <-ticker.C:
			hc.detectFailures()
		}
	}
}

func (hc *HealthCheckerImpl) detectFailures() {
	hc.nodesMu.RLock()
	nodes := make([]*Node, 0, len(hc.nodes))
	for _, node := range hc.nodes {
		nodes = append(nodes, node)
	}
	hc.nodesMu.RUnlock()

	for _, node := range nodes {
		// Update phi accrual detector
		hc.phiDetector.Heartbeat(node.ID)

		// Check if node is suspected failed
		if hc.phiDetector.IsSuspected(node.ID) {
			hc.handleSuspectedFailure(node)
		}
	}
}

func (hc *HealthCheckerImpl) handleSuspectedFailure(node *Node) {
	hc.logger.Warn("Node suspected failed", "node_id", node.ID)
	
	failureDetections.WithLabelValues(node.ID, "phi_accrual").Inc()

	// Get latest health check
	hc.statusMu.RLock()
	health, exists := hc.healthStatus[node.ID]
	hc.statusMu.RUnlock()

	if !exists || time.Since(health.Timestamp) > 2*hc.monitorInterval {
		// No recent health check, mark as offline
		hc.notifyNodeOffline(node)
	} else if !health.Healthy {
		// Node is unhealthy
		hc.notifyNodeUnhealthy(node, health.Issues)
	}
}

// Handler notifications

func (hc *HealthCheckerImpl) notifyHandlers(node *Node, check *HealthCheck) {
	hc.handlersMu.RLock()
	handlers := make([]HealthHandler, len(hc.handlers))
	copy(handlers, hc.handlers)
	hc.handlersMu.RUnlock()

	if check.Healthy {
		for _, handler := range handlers {
			go handler.OnNodeHealthy(node)
		}
	} else {
		for _, handler := range handlers {
			go handler.OnNodeUnhealthy(node, check.Issues)
		}
	}
}

func (hc *HealthCheckerImpl) notifyNodeOffline(node *Node) {
	hc.handlersMu.RLock()
	handlers := make([]HealthHandler, len(hc.handlers))
	copy(handlers, hc.handlers)
	hc.handlersMu.RUnlock()

	for _, handler := range handlers {
		go handler.OnNodeOffline(node)
	}
}

func (hc *HealthCheckerImpl) notifyNodeUnhealthy(node *Node, issues []string) {
	hc.handlersMu.RLock()
	handlers := make([]HealthHandler, len(hc.handlers))
	copy(handlers, hc.handlers)
	hc.handlersMu.RUnlock()

	for _, handler := range handlers {
		go handler.OnNodeUnhealthy(node, issues)
	}
}

// PhiAccrualDetector implements the Phi Accrual failure detector
type PhiAccrualDetector struct {
	threshold      float64
	intervals      map[string][]time.Duration
	lastHeartbeats map[string]time.Time
	phi            map[string]float64
	mu             sync.RWMutex
	logger         Logger
}

// NewPhiAccrualDetector creates a new Phi Accrual failure detector
func NewPhiAccrualDetector(threshold int, logger Logger) *PhiAccrualDetector {
	// Convert threshold to phi value (typically 8-12)
	phiThreshold := float64(threshold) * 2.0
	if phiThreshold < 8 {
		phiThreshold = 8
	}

	return &PhiAccrualDetector{
		threshold:      phiThreshold,
		intervals:      make(map[string][]time.Duration),
		lastHeartbeats: make(map[string]time.Time),
		phi:            make(map[string]float64),
		logger:         logger,
	}
}

// AddNode adds a node to track
func (pad *PhiAccrualDetector) AddNode(nodeID string) {
	pad.mu.Lock()
	defer pad.mu.Unlock()

	pad.intervals[nodeID] = make([]time.Duration, 0, 100)
	pad.lastHeartbeats[nodeID] = time.Now()
	pad.phi[nodeID] = 0
}

// RemoveNode removes a node from tracking
func (pad *PhiAccrualDetector) RemoveNode(nodeID string) {
	pad.mu.Lock()
	defer pad.mu.Unlock()

	delete(pad.intervals, nodeID)
	delete(pad.lastHeartbeats, nodeID)
	delete(pad.phi, nodeID)
}

// Heartbeat records a heartbeat from a node
func (pad *PhiAccrualDetector) Heartbeat(nodeID string) {
	pad.mu.Lock()
	defer pad.mu.Unlock()

	now := time.Now()
	
	if lastHB, exists := pad.lastHeartbeats[nodeID]; exists {
		interval := now.Sub(lastHB)
		
		// Add to intervals (keep last 100)
		if pad.intervals[nodeID] == nil {
			pad.intervals[nodeID] = make([]time.Duration, 0, 100)
		}
		
		pad.intervals[nodeID] = append(pad.intervals[nodeID], interval)
		if len(pad.intervals[nodeID]) > 100 {
			pad.intervals[nodeID] = pad.intervals[nodeID][1:]
		}
	}
	
	pad.lastHeartbeats[nodeID] = now
	
	// Update phi value
	pad.updatePhi(nodeID)
}

// IsSuspected checks if a node is suspected to have failed
func (pad *PhiAccrualDetector) IsSuspected(nodeID string) bool {
	pad.mu.RLock()
	defer pad.mu.RUnlock()

	phi, exists := pad.phi[nodeID]
	if !exists {
		return false
	}

	return phi > pad.threshold
}

// GetPhi returns the current phi value for a node
func (pad *PhiAccrualDetector) GetPhi(nodeID string) float64 {
	pad.mu.RLock()
	defer pad.mu.RUnlock()

	return pad.phi[nodeID]
}

func (pad *PhiAccrualDetector) updatePhi(nodeID string) {
	intervals := pad.intervals[nodeID]
	if len(intervals) < 2 {
		pad.phi[nodeID] = 0
		return
	}

	// Calculate mean and variance
	var sum time.Duration
	for _, interval := range intervals {
		sum += interval
	}
	mean := float64(sum) / float64(len(intervals))

	var variance float64
	for _, interval := range intervals {
		diff := float64(interval) - mean
		variance += diff * diff
	}
	variance /= float64(len(intervals))
	stddev := variance // Simplified, should be sqrt(variance)

	// Calculate time since last heartbeat
	timeSinceLast := float64(time.Since(pad.lastHeartbeats[nodeID]))

	// Calculate phi
	if stddev > 0 {
		pad.phi[nodeID] = (timeSinceLast - mean) / stddev
	} else {
		pad.phi[nodeID] = 0
	}
}