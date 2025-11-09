package testing

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// NetworkSimulator simulates realistic WAN network conditions
type NetworkSimulator struct {
	topology       *NetworkTopology
	latencyModel   *LatencyModel
	lossModel      *PacketLossModel
	bandwidthModel *BandwidthModel
	mu             sync.RWMutex
	active         bool
}

// NetworkTopology represents the datacenter network topology
type NetworkTopology struct {
	Datacenters map[string]*Datacenter
	Links       map[string]*Link
}

// Datacenter represents a datacenter location
type Datacenter struct {
	ID       string
	Region   string
	Location GeoLocation
	Nodes    []*Node
}

// Node represents a compute node in a datacenter
type Node struct {
	ID         string
	Datacenter string
	IPAddress  string
	Capacity   ResourceCapacity
}

// ResourceCapacity defines node resources
type ResourceCapacity struct {
	CPU       int     // cores
	Memory    int64   // bytes
	Bandwidth int     // Mbps
}

// GeoLocation represents geographic coordinates
type GeoLocation struct {
	Latitude  float64
	Longitude float64
}

// Link represents a network link between datacenters
type Link struct {
	Source      string
	Destination string
	Latency     LatencyProfile
	Bandwidth   BandwidthProfile
	PacketLoss  LossProfile
}

// LatencyProfile defines latency characteristics
type LatencyProfile struct {
	BaseLatency  time.Duration
	Jitter       time.Duration
	Distribution DistributionType
}

// BandwidthProfile defines bandwidth characteristics
type BandwidthProfile struct {
	Capacity    int     // Mbps
	Utilization float64 // 0.0 - 1.0
	Burstable   bool
}

// LossProfile defines packet loss characteristics
type LossProfile struct {
	Rate         float64 // 0.0 - 1.0
	BurstLength  int     // packets
	Distribution DistributionType
}

// DistributionType represents statistical distribution
type DistributionType int

const (
	DistributionUniform DistributionType = iota
	DistributionNormal
	DistributionPareto
	DistributionExponential
)

// LatencyModel simulates network latency
type LatencyModel struct {
	profiles map[string]*LatencyProfile
	mu       sync.RWMutex
}

// PacketLossModel simulates packet loss
type PacketLossModel struct {
	profiles   map[string]*LossProfile
	lossState  map[string]int // burst tracking
	mu         sync.RWMutex
}

// BandwidthModel simulates bandwidth constraints
type BandwidthModel struct {
	profiles      map[string]*BandwidthProfile
	utilization   map[string]float64
	lastUpdate    map[string]time.Time
	mu            sync.RWMutex
}

// NewNetworkSimulator creates a new network simulator
func NewNetworkSimulator(topology *NetworkTopology) *NetworkSimulator {
	return &NetworkSimulator{
		topology:       topology,
		latencyModel:   NewLatencyModel(),
		lossModel:      NewPacketLossModel(),
		bandwidthModel: NewBandwidthModel(),
		active:         false,
	}
}

// NewLatencyModel creates a new latency model
func NewLatencyModel() *LatencyModel {
	return &LatencyModel{
		profiles: make(map[string]*LatencyProfile),
	}
}

// NewPacketLossModel creates a new packet loss model
func NewPacketLossModel() *PacketLossModel {
	return &PacketLossModel{
		profiles:  make(map[string]*LossProfile),
		lossState: make(map[string]int),
	}
}

// NewBandwidthModel creates a new bandwidth model
func NewBandwidthModel() *BandwidthModel {
	return &BandwidthModel{
		profiles:    make(map[string]*BandwidthProfile),
		utilization: make(map[string]float64),
		lastUpdate:  make(map[string]time.Time),
	}
}

// ApplyTopology applies the network topology
func (ns *NetworkSimulator) ApplyTopology(topology *NetworkTopology) error {
	ns.mu.Lock()
	defer ns.mu.Unlock()

	ns.topology = topology

	// Initialize models from topology
	for linkID, link := range topology.Links {
		ns.latencyModel.AddProfile(linkID, &link.Latency)
		ns.lossModel.AddProfile(linkID, &link.PacketLoss)
		ns.bandwidthModel.AddProfile(linkID, &link.Bandwidth)
	}

	ns.active = true
	return nil
}

// SimulateLatency simulates network latency for a link
func (ns *NetworkSimulator) SimulateLatency(src, dst string) time.Duration {
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	if !ns.active {
		return 0
	}

	linkID := fmt.Sprintf("%s-%s", src, dst)
	return ns.latencyModel.SimulateLatency(linkID)
}

// SimulatePacketLoss determines if a packet should be lost
func (ns *NetworkSimulator) SimulatePacketLoss(src, dst string) bool {
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	if !ns.active {
		return false
	}

	linkID := fmt.Sprintf("%s-%s", src, dst)
	return ns.lossModel.ShouldDropPacket(linkID)
}

// GetAvailableBandwidth returns available bandwidth for a link
func (ns *NetworkSimulator) GetAvailableBandwidth(src, dst string) int {
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	if !ns.active {
		return 10000 // 10 Gbps default
	}

	linkID := fmt.Sprintf("%s-%s", src, dst)
	return ns.bandwidthModel.GetAvailableBandwidth(linkID)
}

// Reset resets the simulator state
func (ns *NetworkSimulator) Reset() {
	ns.mu.Lock()
	defer ns.mu.Unlock()

	ns.active = false
	ns.latencyModel = NewLatencyModel()
	ns.lossModel = NewPacketLossModel()
	ns.bandwidthModel = NewBandwidthModel()
}

// AddProfile adds a latency profile
func (lm *LatencyModel) AddProfile(linkID string, profile *LatencyProfile) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	lm.profiles[linkID] = profile
}

// SimulateLatency simulates latency for a link
func (lm *LatencyModel) SimulateLatency(linkID string) time.Duration {
	lm.mu.RLock()
	profile, exists := lm.profiles[linkID]
	lm.mu.RUnlock()

	if !exists {
		return 0
	}

	baseLatency := profile.BaseLatency
	jitter := profile.Jitter

	switch profile.Distribution {
	case DistributionNormal:
		return lm.normalDistribution(baseLatency, jitter)
	case DistributionPareto:
		return lm.paretoDistribution(baseLatency, jitter)
	case DistributionExponential:
		return lm.exponentialDistribution(baseLatency)
	default:
		return lm.uniformDistribution(baseLatency, jitter)
	}
}

// normalDistribution generates normally distributed latency
func (lm *LatencyModel) normalDistribution(mean, stddev time.Duration) time.Duration {
	z := rand.NormFloat64()
	latency := float64(mean) + z*float64(stddev)
	if latency < 0 {
		latency = 0
	}
	return time.Duration(latency)
}

// paretoDistribution generates Pareto distributed latency (heavy tail)
func (lm *LatencyModel) paretoDistribution(min, scale time.Duration) time.Duration {
	alpha := 1.5 // shape parameter
	u := rand.Float64()
	latency := float64(min) * math.Pow(1-u, -1/alpha)
	return time.Duration(latency)
}

// exponentialDistribution generates exponentially distributed latency
func (lm *LatencyModel) exponentialDistribution(mean time.Duration) time.Duration {
	lambda := 1.0 / float64(mean)
	u := rand.Float64()
	latency := -math.Log(u) / lambda
	return time.Duration(latency)
}

// uniformDistribution generates uniformly distributed latency
func (lm *LatencyModel) uniformDistribution(base, jitter time.Duration) time.Duration {
	minLatency := float64(base - jitter)
	maxLatency := float64(base + jitter)
	if minLatency < 0 {
		minLatency = 0
	}
	latency := minLatency + rand.Float64()*(maxLatency-minLatency)
	return time.Duration(latency)
}

// GeographicLatency calculates latency based on geographic distance
func (lm *LatencyModel) GeographicLatency(src, dst GeoLocation) time.Duration {
	distance := haversineDistance(src, dst) // km

	// Speed of light in fiber: ~200,000 km/s (2/3 speed of light in vacuum)
	// Add routing overhead (1.5x for non-direct routes)
	baseLatency := distance / 200000.0 * 1.5

	// Add processing delays (0.5ms per hop, estimate 10-15 hops for long distances)
	hops := math.Min(15, distance/1000+5)
	processingDelay := 0.5 * hops

	// Add queuing delay (variable, avg 2ms)
	queuingDelay := 2.0

	totalMs := baseLatency*1000 + processingDelay + queuingDelay
	return time.Duration(totalMs) * time.Millisecond
}

// haversineDistance calculates the great-circle distance between two points
func haversineDistance(loc1, loc2 GeoLocation) float64 {
	const earthRadius = 6371.0 // km

	lat1 := loc1.Latitude * math.Pi / 180
	lat2 := loc2.Latitude * math.Pi / 180
	dLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180
	dLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

// AddProfile adds a packet loss profile
func (plm *PacketLossModel) AddProfile(linkID string, profile *LossProfile) {
	plm.mu.Lock()
	defer plm.mu.Unlock()
	plm.profiles[linkID] = profile
	plm.lossState[linkID] = 0
}

// ShouldDropPacket determines if a packet should be dropped
func (plm *PacketLossModel) ShouldDropPacket(linkID string) bool {
	plm.mu.Lock()
	defer plm.mu.Unlock()

	profile, exists := plm.profiles[linkID]
	if !exists {
		return false
	}

	// Check if we're in a burst
	if plm.lossState[linkID] > 0 {
		plm.lossState[linkID]--
		return true
	}

	// Random loss decision
	if rand.Float64() < profile.Rate {
		// Start a burst if configured
		if profile.BurstLength > 1 {
			plm.lossState[linkID] = profile.BurstLength - 1
		}
		return true
	}

	return false
}

// AddProfile adds a bandwidth profile
func (bm *BandwidthModel) AddProfile(linkID string, profile *BandwidthProfile) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	bm.profiles[linkID] = profile
	bm.utilization[linkID] = profile.Utilization
	bm.lastUpdate[linkID] = time.Now()
}

// GetAvailableBandwidth returns available bandwidth in Mbps
func (bm *BandwidthModel) GetAvailableBandwidth(linkID string) int {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	profile, exists := bm.profiles[linkID]
	if !exists {
		return 10000 // 10 Gbps default
	}

	// Update utilization (simulate time-varying utilization)
	now := time.Now()
	if elapsed := now.Sub(bm.lastUpdate[linkID]); elapsed > time.Second {
		bm.updateUtilization(linkID, profile)
		bm.lastUpdate[linkID] = now
	}

	util := bm.utilization[linkID]
	available := float64(profile.Capacity) * (1.0 - util)

	// Allow bursting if configured
	if profile.Burstable && rand.Float64() < 0.1 {
		available *= 1.5 // 50% burst capacity
	}

	return int(available)
}

// updateUtilization updates bandwidth utilization
func (bm *BandwidthModel) updateUtilization(linkID string, profile *BandwidthProfile) {
	currentUtil := bm.utilization[linkID]
	targetUtil := profile.Utilization

	// Simulate gradual changes in utilization
	delta := (targetUtil - currentUtil) * 0.1
	noise := (rand.Float64() - 0.5) * 0.05 // +/- 2.5% noise

	newUtil := currentUtil + delta + noise
	if newUtil < 0 {
		newUtil = 0
	}
	if newUtil > 1.0 {
		newUtil = 1.0
	}

	bm.utilization[linkID] = newUtil
}

// GetTopologyStats returns statistics about the topology
func (ns *NetworkSimulator) GetTopologyStats() map[string]interface{} {
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	stats := make(map[string]interface{})
	stats["datacenters"] = len(ns.topology.Datacenters)
	stats["links"] = len(ns.topology.Links)

	totalNodes := 0
	for _, dc := range ns.topology.Datacenters {
		totalNodes += len(dc.Nodes)
	}
	stats["total_nodes"] = totalNodes
	stats["active"] = ns.active

	return stats
}
