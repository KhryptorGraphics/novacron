package metrics

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"github.com/khryptorgraphics/novacron/backend/core/network"
)

// NetworkMetricsCollector collects comprehensive network metrics
type NetworkMetricsCollector struct {
	logger              *log.Logger
	networkManager      *network.NetworkManager
	discoveryManager    *discovery.InternetDiscoveryManager
	performancePredictor network.PerformancePredictor
	
	// Collection configuration
	collectionInterval  time.Duration
	retentionPeriod     time.Duration
	aggregationWindow   time.Duration
	
	// Collected metrics
	interfaceMetrics    map[string]*InterfaceMetrics
	connectionMetrics   map[string]*ConnectionMetrics
	topologyMetrics     *TopologyMetrics
	qosMetrics          *QoSMetrics
	bandwidthMetrics    *BandwidthMetrics
	
	// Synchronization
	metricsMutex        sync.RWMutex
	collectionMutex     sync.Mutex
	
	// Collection state
	running            bool
	stopChan           chan struct{}
	collectionTicker   *time.Ticker
	
	// Historical data
	historicalMetrics  []NetworkSnapshot
	maxHistorySize     int
}

// InterfaceMetrics represents network interface metrics
type InterfaceMetrics struct {
	InterfaceName       string                 `json:"interface_name"`
	IPAddress          string                 `json:"ip_address"`
	MACAddress         string                 `json:"mac_address"`
	MTU                int                    `json:"mtu"`
	Speed              uint64                 `json:"speed_mbps"`
	IsUp               bool                   `json:"is_up"`
	BytesReceived      uint64                 `json:"bytes_received"`
	BytesSent          uint64                 `json:"bytes_sent"`
	PacketsReceived    uint64                 `json:"packets_received"`
	PacketsSent        uint64                 `json:"packets_sent"`
	ErrorsReceived     uint64                 `json:"errors_received"`
	ErrorsSent         uint64                 `json:"errors_sent"`
	DropsReceived      uint64                 `json:"drops_received"`
	DropsSent          uint64                 `json:"drops_sent"`
	Utilization        float64                `json:"utilization_percent"`
	Timestamp          time.Time              `json:"timestamp"`
	AdditionalMetrics  map[string]interface{} `json:"additional_metrics"`
}

// ConnectionMetrics represents peer connection metrics
type ConnectionMetrics struct {
	PeerID             string                 `json:"peer_id"`
	RemoteAddress      string                 `json:"remote_address"`
	LocalAddress       string                 `json:"local_address"`
	ConnectionType     string                 `json:"connection_type"`
	Protocol           string                 `json:"protocol"`
	State              string                 `json:"state"`
	RTT                time.Duration          `json:"rtt"`
	Bandwidth          uint64                 `json:"bandwidth_mbps"`
	PacketLoss         float64                `json:"packet_loss"`
	Jitter             time.Duration          `json:"jitter"`
	ThroughputIn       uint64                 `json:"throughput_in_mbps"`
	ThroughputOut      uint64                 `json:"throughput_out_mbps"`
	ConnectionQuality  float64                `json:"connection_quality"`
	LastSeen           time.Time              `json:"last_seen"`
	ConnectionDuration time.Duration          `json:"connection_duration"`
	AdditionalMetrics  map[string]interface{} `json:"additional_metrics"`
}

// TopologyMetrics represents network topology metrics
type TopologyMetrics struct {
	TotalPeers         int                    `json:"total_peers"`
	ConnectedPeers     int                    `json:"connected_peers"`
	DisconnectedPeers  int                    `json:"disconnected_peers"`
	AverageHopCount    float64                `json:"average_hop_count"`
	NetworkDiameter    int                    `json:"network_diameter"`
	ClusteringCoeff    float64                `json:"clustering_coefficient"`
	TopologyType       string                 `json:"topology_type"`
	Regions            map[string]int         `json:"regions"`
	ConnectivityMatrix map[string][]string    `json:"connectivity_matrix"`
	LastUpdated        time.Time              `json:"last_updated"`
}

// QoSMetrics represents quality of service metrics
type QoSMetrics struct {
	TrafficClasses     map[string]*TrafficClassMetrics `json:"traffic_classes"`
	TotalPolicies      int                             `json:"total_policies"`
	ActivePolicies     int                             `json:"active_policies"`
	ViolatedPolicies   int                             `json:"violated_policies"`
	AverageLatency     time.Duration                   `json:"average_latency"`
	AverageThroughput  uint64                          `json:"average_throughput"`
	PacketDropRate     float64                         `json:"packet_drop_rate"`
	BandwidthUtil      map[string]float64              `json:"bandwidth_utilization"`
	LastUpdated        time.Time                       `json:"last_updated"`
}

// TrafficClassMetrics represents metrics for a specific traffic class
type TrafficClassMetrics struct {
	ClassName          string        `json:"class_name"`
	Priority           int           `json:"priority"`
	Bandwidth          uint64        `json:"bandwidth_mbps"`
	BandwidthUsed      uint64        `json:"bandwidth_used_mbps"`
	PacketsProcessed   uint64        `json:"packets_processed"`
	PacketsDropped     uint64        `json:"packets_dropped"`
	AverageLatency     time.Duration `json:"average_latency"`
	MaxLatency         time.Duration `json:"max_latency"`
	Utilization        float64       `json:"utilization_percent"`
	PolicyViolations   int           `json:"policy_violations"`
}

// BandwidthMetrics represents bandwidth utilization metrics
type BandwidthMetrics struct {
	TotalCapacity      uint64                 `json:"total_capacity_mbps"`
	UsedCapacity       uint64                 `json:"used_capacity_mbps"`
	AvailableCapacity  uint64                 `json:"available_capacity_mbps"`
	UtilizationPercent float64                `json:"utilization_percent"`
	PeakUtilization    float64                `json:"peak_utilization_percent"`
	AverageUtilization float64                `json:"average_utilization_percent"`
	PerInterfaceUtil   map[string]float64     `json:"per_interface_utilization"`
	PerPeerBandwidth   map[string]uint64      `json:"per_peer_bandwidth"`
	PredictedPeak      *BandwidthPrediction   `json:"predicted_peak"`
	TrendAnalysis      *BandwidthTrend        `json:"trend_analysis"`
	LastUpdated        time.Time              `json:"last_updated"`
}

// BandwidthPrediction represents predicted bandwidth usage
type BandwidthPrediction struct {
	TimeHorizon        time.Duration `json:"time_horizon"`
	PredictedUsage     uint64        `json:"predicted_usage_mbps"`
	ConfidenceLevel    float64       `json:"confidence_level"`
	PredictionModel    string        `json:"prediction_model"`
	GeneratedAt        time.Time     `json:"generated_at"`
}

// BandwidthTrend represents bandwidth usage trends
type BandwidthTrend struct {
	Direction          string        `json:"direction"` // increasing, decreasing, stable
	Rate               float64       `json:"rate_mbps_per_hour"`
	Confidence         float64       `json:"confidence"`
	PeriodAnalyzed     time.Duration `json:"period_analyzed"`
	SeasonalPattern    bool          `json:"seasonal_pattern"`
	PeakHours          []int         `json:"peak_hours"`
	AnalyzedAt         time.Time     `json:"analyzed_at"`
}

// NetworkSnapshot represents a point-in-time snapshot of network metrics
type NetworkSnapshot struct {
	Timestamp         time.Time                      `json:"timestamp"`
	InterfaceMetrics  map[string]*InterfaceMetrics   `json:"interface_metrics"`
	ConnectionMetrics map[string]*ConnectionMetrics  `json:"connection_metrics"`
	TopologyMetrics   *TopologyMetrics               `json:"topology_metrics"`
	QoSMetrics        *QoSMetrics                    `json:"qos_metrics"`
	BandwidthMetrics  *BandwidthMetrics              `json:"bandwidth_metrics"`
}

// NewNetworkMetricsCollector creates a new network metrics collector
func NewNetworkMetricsCollector(
	networkManager *network.NetworkManager,
	discoveryManager *discovery.InternetDiscoveryManager,
	performancePredictor network.PerformancePredictor,
	logger *log.Logger,
) *NetworkMetricsCollector {
	return &NetworkMetricsCollector{
		logger:              logger,
		networkManager:      networkManager,
		discoveryManager:    discoveryManager,
		performancePredictor: performancePredictor,
		collectionInterval:  30 * time.Second,
		retentionPeriod:     24 * time.Hour,
		aggregationWindow:   5 * time.Minute,
		interfaceMetrics:    make(map[string]*InterfaceMetrics),
		connectionMetrics:   make(map[string]*ConnectionMetrics),
		maxHistorySize:      2880, // 24 hours at 30-second intervals
		stopChan:           make(chan struct{}),
	}
}

// Start starts the metrics collection
func (nmc *NetworkMetricsCollector) Start(ctx context.Context) error {
	nmc.collectionMutex.Lock()
	defer nmc.collectionMutex.Unlock()

	if nmc.running {
		return fmt.Errorf("metrics collector is already running")
	}

	nmc.running = true
	nmc.collectionTicker = time.NewTicker(nmc.collectionInterval)

	// Start collection goroutine
	go nmc.collectionLoop(ctx)
	
	// Start aggregation goroutine
	go nmc.aggregationLoop(ctx)
	
	// Start prediction update goroutine
	go nmc.predictionLoop(ctx)

	nmc.logger.Println("Network metrics collector started")
	return nil
}

// Stop stops the metrics collection
func (nmc *NetworkMetricsCollector) Stop() error {
	nmc.collectionMutex.Lock()
	defer nmc.collectionMutex.Unlock()

	if !nmc.running {
		return fmt.Errorf("metrics collector is not running")
	}

	nmc.running = false
	close(nmc.stopChan)
	
	if nmc.collectionTicker != nil {
		nmc.collectionTicker.Stop()
	}

	nmc.logger.Println("Network metrics collector stopped")
	return nil
}

// collectionLoop runs the main collection loop
func (nmc *NetworkMetricsCollector) collectionLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-nmc.stopChan:
			return
		case <-nmc.collectionTicker.C:
			nmc.collectMetrics(ctx)
		}
	}
}

// collectMetrics collects all network metrics
func (nmc *NetworkMetricsCollector) collectMetrics(ctx context.Context) {
	nmc.logger.Printf("Collecting network metrics...")
	
	// Collect interface metrics
	interfaceMetrics := nmc.collectInterfaceMetrics()
	
	// Collect connection metrics
	connectionMetrics := nmc.collectConnectionMetrics(ctx)
	
	// Collect topology metrics
	topologyMetrics := nmc.collectTopologyMetrics(ctx)
	
	// Collect QoS metrics
	qosMetrics := nmc.collectQoSMetrics()
	
	// Collect bandwidth metrics
	bandwidthMetrics := nmc.collectBandwidthMetrics()

	// Update stored metrics
	nmc.metricsMutex.Lock()
	nmc.interfaceMetrics = interfaceMetrics
	nmc.connectionMetrics = connectionMetrics
	nmc.topologyMetrics = topologyMetrics
	nmc.qosMetrics = qosMetrics
	nmc.bandwidthMetrics = bandwidthMetrics
	nmc.metricsMutex.Unlock()

	// Store historical snapshot
	nmc.storeHistoricalSnapshot()

	// Send metrics to performance predictor
	nmc.sendMetricsToPredictor(ctx, connectionMetrics)

	nmc.logger.Printf("Network metrics collection completed")
}

// collectInterfaceMetrics collects network interface metrics
func (nmc *NetworkMetricsCollector) collectInterfaceMetrics() map[string]*InterfaceMetrics {
	metrics := make(map[string]*InterfaceMetrics)

	// Get network interfaces
	interfaces, err := net.Interfaces()
	if err != nil {
		nmc.logger.Printf("Error getting network interfaces: %v", err)
		return metrics
	}

	for _, iface := range interfaces {
		if strings.HasPrefix(iface.Name, "lo") {
			continue // Skip loopback interfaces
		}

		metric := &InterfaceMetrics{
			InterfaceName:     iface.Name,
			MACAddress:       iface.HardwareAddr.String(),
			MTU:              iface.MTU,
			IsUp:             iface.Flags&net.FlagUp != 0,
			Timestamp:        time.Now(),
			AdditionalMetrics: make(map[string]interface{}),
		}

		// Get IP addresses
		addrs, err := iface.Addrs()
		if err == nil && len(addrs) > 0 {
			for _, addr := range addrs {
				if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
					if ipnet.IP.To4() != nil {
						metric.IPAddress = ipnet.IP.String()
						break
					}
				}
			}
		}

		// Get interface statistics from /proc/net/dev
		nmc.updateInterfaceStatistics(metric)

		// Get interface speed and utilization
		nmc.updateInterfaceSpeed(metric)

		metrics[iface.Name] = metric
	}

	return metrics
}

// updateInterfaceStatistics updates interface statistics from /proc/net/dev
func (nmc *NetworkMetricsCollector) updateInterfaceStatistics(metric *InterfaceMetrics) {
	data, err := os.ReadFile("/proc/net/dev")
	if err != nil {
		nmc.logger.Printf("Error reading /proc/net/dev: %v", err)
		return
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.Contains(line, metric.InterfaceName+":") {
			fields := strings.Fields(strings.Replace(line, ":", " ", 1))
			if len(fields) >= 17 {
				metric.BytesReceived, _ = strconv.ParseUint(fields[1], 10, 64)
				metric.PacketsReceived, _ = strconv.ParseUint(fields[2], 10, 64)
				metric.ErrorsReceived, _ = strconv.ParseUint(fields[3], 10, 64)
				metric.DropsReceived, _ = strconv.ParseUint(fields[4], 10, 64)
				metric.BytesSent, _ = strconv.ParseUint(fields[9], 10, 64)
				metric.PacketsSent, _ = strconv.ParseUint(fields[10], 10, 64)
				metric.ErrorsSent, _ = strconv.ParseUint(fields[11], 10, 64)
				metric.DropsSent, _ = strconv.ParseUint(fields[12], 10, 64)
			}
			break
		}
	}
}

// updateInterfaceSpeed updates interface speed and calculates utilization
func (nmc *NetworkMetricsCollector) updateInterfaceSpeed(metric *InterfaceMetrics) {
	// Try to get interface speed from ethtool or sys filesystem
	speedPath := fmt.Sprintf("/sys/class/net/%s/speed", metric.InterfaceName)
	if speedData, err := os.ReadFile(speedPath); err == nil {
		if speed, err := strconv.ParseUint(strings.TrimSpace(string(speedData)), 10, 64); err == nil {
			metric.Speed = speed
		}
	}

	// Calculate utilization based on historical data
	if metric.Speed > 0 {
		// Simple utilization calculation (would need historical data for accurate calculation)
		totalBytes := metric.BytesReceived + metric.BytesSent
		maxBytesPerSecond := metric.Speed * 1000000 / 8 // Convert Mbps to bytes/second
		
		// This is a simplified calculation - in reality, you'd need time-based sampling
		if maxBytesPerSecond > 0 {
			metric.Utilization = float64(totalBytes) / float64(maxBytesPerSecond) * 100
			if metric.Utilization > 100 {
				metric.Utilization = 100
			}
		}
	}
}

// collectConnectionMetrics collects peer connection metrics
func (nmc *NetworkMetricsCollector) collectConnectionMetrics(ctx context.Context) map[string]*ConnectionMetrics {
	metrics := make(map[string]*ConnectionMetrics)

	if nmc.discoveryManager == nil {
		return metrics
	}

	// Get connected peers
	peers, err := nmc.discoveryManager.GetConnectedPeers()
	if err != nil {
		nmc.logger.Printf("Error getting connected peers: %v", err)
		return metrics
	}

	for _, peer := range peers {
		metric := &ConnectionMetrics{
			PeerID:             peer.ID,
			RemoteAddress:      peer.Address,
			ConnectionType:     "p2p",
			Protocol:          "tcp", // Assume TCP for now
			State:             "established",
			RTT:               peer.RTT,
			Bandwidth:         peer.BandwidthCapability,
			PacketLoss:        peer.PacketLoss,
			Jitter:            time.Duration(peer.Jitter) * time.Millisecond,
			ConnectionQuality: peer.ConnectionQuality,
			LastSeen:          peer.LastSeen,
			ConnectionDuration: time.Since(peer.ConnectedAt),
			AdditionalMetrics: make(map[string]interface{}),
		}

		// Calculate throughput (simplified)
		metric.ThroughputIn = uint64(float64(peer.BandwidthCapability) * 0.7) // Assume 70% utilization
		metric.ThroughputOut = uint64(float64(peer.BandwidthCapability) * 0.3)

		// Store additional peer-specific metrics
		metric.AdditionalMetrics["nat_type"] = peer.NATType.String()
		metric.AdditionalMetrics["region"] = peer.Region
		metric.AdditionalMetrics["connection_attempts"] = peer.ConnectionAttempts
		metric.AdditionalMetrics["last_ping"] = peer.LastPing

		metrics[peer.ID] = metric
	}

	return metrics
}

// collectTopologyMetrics collects network topology metrics
func (nmc *NetworkMetricsCollector) collectTopologyMetrics(ctx context.Context) *TopologyMetrics {
	metrics := &TopologyMetrics{
		Regions:            make(map[string]int),
		ConnectivityMatrix: make(map[string][]string),
		LastUpdated:        time.Now(),
	}

	if nmc.discoveryManager == nil {
		return metrics
	}

	// Get all peers
	peers, err := nmc.discoveryManager.GetConnectedPeers()
	if err != nil {
		nmc.logger.Printf("Error getting peers for topology metrics: %v", err)
		return metrics
	}

	metrics.TotalPeers = len(peers)
	
	connectedCount := 0
	totalHops := 0
	
	for _, peer := range peers {
		if peer.ConnectionQuality > 0.5 {
			connectedCount++
		}
		
		// Count hops (simplified - would need actual routing data)
		hops := nmc.estimateHopCount(peer.RTT)
		totalHops += hops
		
		// Group by region
		metrics.Regions[peer.Region]++
		
		// Build connectivity matrix (simplified)
		if _, exists := metrics.ConnectivityMatrix[peer.ID]; !exists {
			metrics.ConnectivityMatrix[peer.ID] = []string{}
		}
	}

	metrics.ConnectedPeers = connectedCount
	metrics.DisconnectedPeers = metrics.TotalPeers - connectedCount

	if metrics.TotalPeers > 0 {
		metrics.AverageHopCount = float64(totalHops) / float64(metrics.TotalPeers)
	}

	// Calculate network diameter (simplified)
	metrics.NetworkDiameter = nmc.calculateNetworkDiameter(peers)
	
	// Calculate clustering coefficient (simplified)
	metrics.ClusteringCoeff = nmc.calculateClusteringCoefficient(peers)
	
	// Determine topology type
	metrics.TopologyType = nmc.determineTopologyType(metrics)

	return metrics
}

// estimateHopCount estimates hop count based on RTT
func (nmc *NetworkMetricsCollector) estimateHopCount(rtt time.Duration) int {
	// Simple heuristic: ~10ms per hop on average
	return int(rtt.Milliseconds() / 10)
}

// calculateNetworkDiameter calculates the network diameter
func (nmc *NetworkMetricsCollector) calculateNetworkDiameter(peers []discovery.PeerInfo) int {
	// Simplified calculation - would need full topology graph
	if len(peers) == 0 {
		return 0
	}
	
	maxRTT := time.Duration(0)
	for _, peer := range peers {
		if peer.RTT > maxRTT {
			maxRTT = peer.RTT
		}
	}
	
	return nmc.estimateHopCount(maxRTT)
}

// calculateClusteringCoefficient calculates clustering coefficient
func (nmc *NetworkMetricsCollector) calculateClusteringCoefficient(peers []discovery.PeerInfo) float64 {
	// Simplified calculation - would need full connection matrix
	if len(peers) < 3 {
		return 0.0
	}
	
	// Assume some clustering based on connection quality
	highQualityConnections := 0
	for _, peer := range peers {
		if peer.ConnectionQuality > 0.8 {
			highQualityConnections++
		}
	}
	
	return float64(highQualityConnections) / float64(len(peers))
}

// determineTopologyType determines the network topology type
func (nmc *NetworkMetricsCollector) determineTopologyType(metrics *TopologyMetrics) string {
	if metrics.TotalPeers == 0 {
		return "isolated"
	}
	
	connectedRatio := float64(metrics.ConnectedPeers) / float64(metrics.TotalPeers)
	
	if connectedRatio > 0.8 && metrics.ClusteringCoeff > 0.6 {
		return "mesh"
	} else if metrics.AverageHopCount > 5 {
		return "hierarchical"
	} else if connectedRatio < 0.5 {
		return "sparse"
	} else {
		return "hybrid"
	}
}

// collectQoSMetrics collects quality of service metrics
func (nmc *NetworkMetricsCollector) collectQoSMetrics() *QoSMetrics {
	metrics := &QoSMetrics{
		TrafficClasses: make(map[string]*TrafficClassMetrics),
		LastUpdated:    time.Now(),
	}

	if nmc.networkManager == nil {
		return metrics
	}

	// Get QoS information from network manager
	qosInfo := nmc.networkManager.GetQoSStatus()
	if qosInfo != nil {
		metrics.TotalPolicies = len(qosInfo.Policies)
		
		activePolicies := 0
		violatedPolicies := 0
		
		for _, policy := range qosInfo.Policies {
			if policy.Active {
				activePolicies++
			}
			if policy.Violations > 0 {
				violatedPolicies++
			}
			
			// Create traffic class metrics
			classMetrics := &TrafficClassMetrics{
				ClassName:        policy.Name,
				Priority:         policy.Priority,
				Bandwidth:        policy.Bandwidth,
				BandwidthUsed:    policy.UsedBandwidth,
				PacketsProcessed: policy.ProcessedPackets,
				PacketsDropped:   policy.DroppedPackets,
				AverageLatency:   policy.AverageLatency,
				MaxLatency:       policy.MaxLatency,
				PolicyViolations: policy.Violations,
			}
			
			if policy.Bandwidth > 0 {
				classMetrics.Utilization = float64(policy.UsedBandwidth) / float64(policy.Bandwidth) * 100
			}
			
			metrics.TrafficClasses[policy.Name] = classMetrics
		}
		
		metrics.ActivePolicies = activePolicies
		metrics.ViolatedPolicies = violatedPolicies
		metrics.AverageLatency = qosInfo.AverageLatency
		metrics.AverageThroughput = qosInfo.TotalThroughput
		metrics.PacketDropRate = qosInfo.PacketDropRate
	}

	return metrics
}

// collectBandwidthMetrics collects bandwidth utilization metrics
func (nmc *NetworkMetricsCollector) collectBandwidthMetrics() *BandwidthMetrics {
	metrics := &BandwidthMetrics{
		PerInterfaceUtil: make(map[string]float64),
		PerPeerBandwidth: make(map[string]uint64),
		LastUpdated:      time.Now(),
	}

	// Calculate from interface metrics
	nmc.metricsMutex.RLock()
	totalCapacity := uint64(0)
	usedCapacity := uint64(0)
	
	for _, ifMetrics := range nmc.interfaceMetrics {
		if ifMetrics.Speed > 0 {
			totalCapacity += ifMetrics.Speed
			used := uint64(float64(ifMetrics.Speed) * ifMetrics.Utilization / 100)
			usedCapacity += used
			metrics.PerInterfaceUtil[ifMetrics.InterfaceName] = ifMetrics.Utilization
		}
	}
	
	// Add peer bandwidth usage
	for peerID, connMetrics := range nmc.connectionMetrics {
		bandwidth := connMetrics.ThroughputIn + connMetrics.ThroughputOut
		metrics.PerPeerBandwidth[peerID] = bandwidth
	}
	nmc.metricsMutex.RUnlock()

	metrics.TotalCapacity = totalCapacity
	metrics.UsedCapacity = usedCapacity
	metrics.AvailableCapacity = totalCapacity - usedCapacity

	if totalCapacity > 0 {
		metrics.UtilizationPercent = float64(usedCapacity) / float64(totalCapacity) * 100
	}

	// Calculate trends and predictions
	metrics.TrendAnalysis = nmc.calculateBandwidthTrend()
	metrics.AverageUtilization = nmc.calculateAverageUtilization()
	metrics.PeakUtilization = nmc.calculatePeakUtilization()

	return metrics
}

// calculateBandwidthTrend calculates bandwidth usage trends
func (nmc *NetworkMetricsCollector) calculateBandwidthTrend() *BandwidthTrend {
	// Analyze historical data for trends
	if len(nmc.historicalMetrics) < 10 {
		return &BandwidthTrend{
			Direction:       "stable",
			Rate:           0,
			Confidence:     0.5,
			PeriodAnalyzed: time.Hour,
			AnalyzedAt:     time.Now(),
		}
	}

	// Simple linear regression on recent data
	recent := nmc.historicalMetrics
	if len(recent) > 100 {
		recent = recent[len(recent)-100:] // Last 100 samples
	}

	// Calculate trend
	n := len(recent)
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i, snapshot := range recent {
		x := float64(i)
		y := snapshot.BandwidthMetrics.UtilizationPercent
		
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (float64(n)*sumXY - sumX*sumY) / (float64(n)*sumX2 - sumX*sumX)
	
	// Convert slope to direction and rate
	direction := "stable"
	if slope > 0.1 {
		direction = "increasing"
	} else if slope < -0.1 {
		direction = "decreasing"
	}

	// Calculate confidence based on R²
	meanY := sumY / float64(n)
	ssRes := 0.0
	ssTot := 0.0
	
	for i, snapshot := range recent {
		x := float64(i)
		y := snapshot.BandwidthMetrics.UtilizationPercent
		predicted := slope*x + (sumY-slope*sumX)/float64(n)
		
		ssRes += (y - predicted) * (y - predicted)
		ssTot += (y - meanY) * (y - meanY)
	}
	
	confidence := 1 - ssRes/ssTot
	if confidence < 0 {
		confidence = 0
	}

	return &BandwidthTrend{
		Direction:       direction,
		Rate:           slope * 3600, // Convert to per-hour rate
		Confidence:     confidence,
		PeriodAnalyzed: time.Duration(len(recent)) * nmc.collectionInterval,
		AnalyzedAt:     time.Now(),
	}
}

// calculateAverageUtilization calculates average bandwidth utilization
func (nmc *NetworkMetricsCollector) calculateAverageUtilization() float64 {
	if len(nmc.historicalMetrics) == 0 {
		return 0.0
	}

	totalUtil := 0.0
	count := 0

	for _, snapshot := range nmc.historicalMetrics {
		if snapshot.BandwidthMetrics != nil {
			totalUtil += snapshot.BandwidthMetrics.UtilizationPercent
			count++
		}
	}

	if count > 0 {
		return totalUtil / float64(count)
	}
	return 0.0
}

// calculatePeakUtilization calculates peak bandwidth utilization
func (nmc *NetworkMetricsCollector) calculatePeakUtilization() float64 {
	peak := 0.0

	for _, snapshot := range nmc.historicalMetrics {
		if snapshot.BandwidthMetrics != nil {
			if snapshot.BandwidthMetrics.UtilizationPercent > peak {
				peak = snapshot.BandwidthMetrics.UtilizationPercent
			}
		}
	}

	return peak
}

// storeHistoricalSnapshot stores a historical snapshot
func (nmc *NetworkMetricsCollector) storeHistoricalSnapshot() {
	snapshot := NetworkSnapshot{
		Timestamp:         time.Now(),
		InterfaceMetrics:  make(map[string]*InterfaceMetrics),
		ConnectionMetrics: make(map[string]*ConnectionMetrics),
		TopologyMetrics:   nmc.topologyMetrics,
		QoSMetrics:        nmc.qosMetrics,
		BandwidthMetrics:  nmc.bandwidthMetrics,
	}

	// Deep copy metrics
	for k, v := range nmc.interfaceMetrics {
		snapshot.InterfaceMetrics[k] = v
	}
	for k, v := range nmc.connectionMetrics {
		snapshot.ConnectionMetrics[k] = v
	}

	// Add to historical data
	nmc.historicalMetrics = append(nmc.historicalMetrics, snapshot)

	// Trim if too large
	if len(nmc.historicalMetrics) > nmc.maxHistorySize {
		nmc.historicalMetrics = nmc.historicalMetrics[len(nmc.historicalMetrics)-nmc.maxHistorySize:]
	}
}

// sendMetricsToPredictor sends metrics to the performance predictor
func (nmc *NetworkMetricsCollector) sendMetricsToPredictor(ctx context.Context, connectionMetrics map[string]*ConnectionMetrics) {
	if nmc.performancePredictor == nil {
		return
	}

	// Convert connection metrics to network metrics format
	for _, connMetric := range connectionMetrics {
		networkMetric := network.NetworkMetrics{
			Timestamp:         time.Now(),
			SourceNode:        "local", // Local node
			TargetNode:        connMetric.PeerID,
			BandwidthMbps:     float64(connMetric.Bandwidth),
			LatencyMs:         float64(connMetric.RTT.Milliseconds()),
			PacketLoss:        connMetric.PacketLoss,
			JitterMs:          float64(connMetric.Jitter.Milliseconds()),
			ThroughputMbps:    float64(connMetric.ThroughputIn + connMetric.ThroughputOut),
			ConnectionQuality: connMetric.ConnectionQuality,
			RouteHops:         nmc.estimateHopCount(connMetric.RTT),
			CongestionLevel:   nmc.estimateCongestionLevel(connMetric),
		}

		if err := nmc.performancePredictor.StoreNetworkMetrics(ctx, networkMetric); err != nil {
			nmc.logger.Printf("Error storing network metrics for predictor: %v", err)
		}
	}
}

// estimateCongestionLevel estimates congestion level from connection metrics
func (nmc *NetworkMetricsCollector) estimateCongestionLevel(metric *ConnectionMetrics) float64 {
	// Simple heuristic based on packet loss and utilization
	congestion := metric.PacketLoss * 10 // Packet loss contributes heavily
	
	// Add jitter factor
	if metric.Jitter > 10*time.Millisecond {
		congestion += 0.3
	}
	
	// Add quality factor
	congestion += (1.0 - metric.ConnectionQuality) * 0.5
	
	// Clamp to 0-1 range
	if congestion > 1.0 {
		congestion = 1.0
	}
	if congestion < 0.0 {
		congestion = 0.0
	}
	
	return congestion
}

// aggregationLoop runs periodic aggregation of metrics
func (nmc *NetworkMetricsCollector) aggregationLoop(ctx context.Context) {
	aggregationTicker := time.NewTicker(nmc.aggregationWindow)
	defer aggregationTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-nmc.stopChan:
			return
		case <-aggregationTicker.C:
			nmc.aggregateMetrics()
		}
	}
}

// aggregateMetrics performs periodic aggregation of collected metrics
func (nmc *NetworkMetricsCollector) aggregateMetrics() {
	// Aggregate historical data for different time windows
	nmc.aggregateByTimeWindow(time.Hour)
	nmc.aggregateByTimeWindow(time.Hour * 6)
	nmc.aggregateByTimeWindow(time.Hour * 24)
}

// aggregateByTimeWindow aggregates metrics by time window
func (nmc *NetworkMetricsCollector) aggregateByTimeWindow(window time.Duration) {
	now := time.Now()
	cutoff := now.Add(-window)

	// Find metrics within the window
	var windowMetrics []NetworkSnapshot
	for _, snapshot := range nmc.historicalMetrics {
		if snapshot.Timestamp.After(cutoff) {
			windowMetrics = append(windowMetrics, snapshot)
		}
	}

	if len(windowMetrics) == 0 {
		return
	}

	// Aggregate bandwidth utilization
	totalUtil := 0.0
	maxUtil := 0.0
	minUtil := 100.0

	for _, snapshot := range windowMetrics {
		if snapshot.BandwidthMetrics != nil {
			util := snapshot.BandwidthMetrics.UtilizationPercent
			totalUtil += util
			if util > maxUtil {
				maxUtil = util
			}
			if util < minUtil {
				minUtil = util
			}
		}
	}

	avgUtil := totalUtil / float64(len(windowMetrics))

	nmc.logger.Printf("Aggregated metrics for %v window: avg=%.2f%%, max=%.2f%%, min=%.2f%%",
		window, avgUtil, maxUtil, minUtil)
}

// predictionLoop runs periodic bandwidth prediction updates
func (nmc *NetworkMetricsCollector) predictionLoop(ctx context.Context) {
	predictionTicker := time.NewTicker(10 * time.Minute)
	defer predictionTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-nmc.stopChan:
			return
		case <-predictionTicker.C:
			nmc.updateBandwidthPredictions(ctx)
		}
	}
}

// updateBandwidthPredictions updates bandwidth predictions
func (nmc *NetworkMetricsCollector) updateBandwidthPredictions(ctx context.Context) {
	if nmc.performancePredictor == nil {
		return
	}

	// Create a general prediction request
	workloadChars := network.WorkloadCharacteristics{
		VMID:                    "aggregate",
		WorkloadType:           "mixed",
		CPUCores:               8,
		MemoryGB:               16,
		StorageGB:              500,
		NetworkIntensive:       true,
		ExpectedConnections:    10,
		DataTransferPattern:    "steady",
		PeakHours:             []int{8, 9, 10, 16, 17, 18},
		HistoricalBandwidth:   100,
	}

	request := network.PredictionRequest{
		SourceNode:          "local",
		TargetNode:          "aggregate",
		WorkloadChars:       workloadChars,
		TimeHorizonHours:    24,
		ConfidenceLevel:     0.95,
		IncludeUncertainty:  true,
	}

	prediction, err := nmc.performancePredictor.PredictBandwidth(ctx, request)
	if err != nil {
		nmc.logger.Printf("Error getting bandwidth prediction: %v", err)
		return
	}

	// Update bandwidth metrics with prediction
	nmc.metricsMutex.Lock()
	if nmc.bandwidthMetrics != nil {
		nmc.bandwidthMetrics.PredictedPeak = &BandwidthPrediction{
			TimeHorizon:     24 * time.Hour,
			PredictedUsage:  uint64(prediction.PredictedBandwidth),
			ConfidenceLevel: prediction.PredictionConfidence,
			PredictionModel: "ai",
			GeneratedAt:     time.Now(),
		}
	}
	nmc.metricsMutex.Unlock()

	nmc.logger.Printf("Updated bandwidth prediction: %.2f Mbps (confidence: %.3f)",
		prediction.PredictedBandwidth, prediction.PredictionConfidence)
}

// GetCurrentMetrics returns current network metrics snapshot
func (nmc *NetworkMetricsCollector) GetCurrentMetrics() *NetworkSnapshot {
	nmc.metricsMutex.RLock()
	defer nmc.metricsMutex.RUnlock()

	return &NetworkSnapshot{
		Timestamp:         time.Now(),
		InterfaceMetrics:  nmc.interfaceMetrics,
		ConnectionMetrics: nmc.connectionMetrics,
		TopologyMetrics:   nmc.topologyMetrics,
		QoSMetrics:        nmc.qosMetrics,
		BandwidthMetrics:  nmc.bandwidthMetrics,
	}
}

// GetHistoricalMetrics returns historical metrics for a time range
func (nmc *NetworkMetricsCollector) GetHistoricalMetrics(since time.Time) []NetworkSnapshot {
	var result []NetworkSnapshot

	for _, snapshot := range nmc.historicalMetrics {
		if snapshot.Timestamp.After(since) {
			result = append(result, snapshot)
		}
	}

	// Sort by timestamp
	sort.Slice(result, func(i, j int) bool {
		return result[i].Timestamp.Before(result[j].Timestamp)
	})

	return result
}

// GetInterfaceMetrics returns current interface metrics
func (nmc *NetworkMetricsCollector) GetInterfaceMetrics() map[string]*InterfaceMetrics {
	nmc.metricsMutex.RLock()
	defer nmc.metricsMutex.RUnlock()

	// Return a copy to avoid race conditions
	result := make(map[string]*InterfaceMetrics)
	for k, v := range nmc.interfaceMetrics {
		result[k] = v
	}
	return result
}

// GetConnectionMetrics returns current connection metrics
func (nmc *NetworkMetricsCollector) GetConnectionMetrics() map[string]*ConnectionMetrics {
	nmc.metricsMutex.RLock()
	defer nmc.metricsMutex.RUnlock()

	// Return a copy to avoid race conditions
	result := make(map[string]*ConnectionMetrics)
	for k, v := range nmc.connectionMetrics {
		result[k] = v
	}
	return result
}

// GetTopologyMetrics returns current topology metrics
func (nmc *NetworkMetricsCollector) GetTopologyMetrics() *TopologyMetrics {
	nmc.metricsMutex.RLock()
	defer nmc.metricsMutex.RUnlock()
	
	return nmc.topologyMetrics
}

// GetBandwidthMetrics returns current bandwidth metrics
func (nmc *NetworkMetricsCollector) GetBandwidthMetrics() *BandwidthMetrics {
	nmc.metricsMutex.RLock()
	defer nmc.metricsMutex.RUnlock()
	
	return nmc.bandwidthMetrics
}