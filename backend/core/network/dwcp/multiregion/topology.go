package multiregion

import (
	"fmt"
	"sync"
	"time"
)

// LinkID uniquely identifies an inter-region link
type LinkID string

// GeoLocation represents geographical coordinates
type GeoLocation struct {
	Latitude  float64
	Longitude float64
	Country   string
	City      string
}

// NetworkEndpoint represents a network access point
type NetworkEndpoint struct {
	Address   string
	Port      int
	Protocol  string
	PublicIP  string
	PrivateIP string
}

// RegionCapacity represents capacity metrics for a region
type RegionCapacity struct {
	MaxInstances    int
	MaxBandwidthMbps int64
	MaxStorage      int64
	AvailableVCPUs  int
	AvailableRAM    int64
}

// Datacenter represents a physical datacenter
type Datacenter struct {
	ID       string
	Name     string
	Location GeoLocation
	Zones    []string
	Provider string
}

// LinkHealth represents the health status of a link
type LinkHealth int

const (
	HealthUnknown LinkHealth = iota
	HealthUp
	HealthDegraded
	HealthDown
)

// Region represents a geographical region
type Region struct {
	ID          string
	Name        string
	Location    GeoLocation
	Datacenters []*Datacenter
	Endpoints   []NetworkEndpoint
	Capacity    RegionCapacity
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// InterRegionLink represents a connection between two regions
type InterRegionLink struct {
	ID          LinkID
	Source      string
	Destination string
	Latency     time.Duration
	Bandwidth   int64 // Mbps
	Cost        float64
	Utilization float64
	Health      LinkHealth
	LastProbe   time.Time
	Metrics     *LinkMetrics
	mu          sync.RWMutex
}

// LinkMetrics tracks detailed metrics for a link
type LinkMetrics struct {
	PacketsSent     uint64
	PacketsReceived uint64
	PacketsLost     uint64
	BytesSent       uint64
	BytesReceived   uint64
	AvgLatency      time.Duration
	MaxLatency      time.Duration
	MinLatency      time.Duration
	Jitter          time.Duration
}

// RoutingKey uniquely identifies a route
type RoutingKey struct {
	Source      string
	Destination string
}

// RouteMetric represents routing metrics
type RouteMetric struct {
	Latency   time.Duration
	Cost      float64
	Bandwidth int64
	Hops      int
	Reliability float64
}

// Route represents a network route
type Route struct {
	Destination string
	NextHop     string
	Path        []string
	Links       []LinkID
	Metric      RouteMetric
	Priority    int
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// RoutingTable maintains routing information
type RoutingTable struct {
	routes map[RoutingKey]*Route
	mu     sync.RWMutex
}

// GlobalTopology represents the entire multi-region network
type GlobalTopology struct {
	regions      map[string]*Region
	links        map[LinkID]*InterRegionLink
	routingTable *RoutingTable
	tunnels      map[string]*VPNTunnel
	mu           sync.RWMutex
}

// NewGlobalTopology creates a new global topology
func NewGlobalTopology() *GlobalTopology {
	return &GlobalTopology{
		regions: make(map[string]*Region),
		links:   make(map[LinkID]*InterRegionLink),
		routingTable: &RoutingTable{
			routes: make(map[RoutingKey]*Route),
		},
		tunnels: make(map[string]*VPNTunnel),
	}
}

// AddRegion adds a region to the topology
func (gt *GlobalTopology) AddRegion(region *Region) error {
	gt.mu.Lock()
	defer gt.mu.Unlock()

	if _, exists := gt.regions[region.ID]; exists {
		return fmt.Errorf("region %s already exists", region.ID)
	}

	region.CreatedAt = time.Now()
	region.UpdatedAt = time.Now()
	gt.regions[region.ID] = region
	return nil
}

// RemoveRegion removes a region from the topology
func (gt *GlobalTopology) RemoveRegion(regionID string) error {
	gt.mu.Lock()
	defer gt.mu.Unlock()

	if _, exists := gt.regions[regionID]; !exists {
		return fmt.Errorf("region %s not found", regionID)
	}

	// Remove all links involving this region
	for linkID, link := range gt.links {
		if link.Source == regionID || link.Destination == regionID {
			delete(gt.links, linkID)
		}
	}

	delete(gt.regions, regionID)
	return nil
}

// AddLink adds an inter-region link
func (gt *GlobalTopology) AddLink(link *InterRegionLink) error {
	gt.mu.Lock()
	defer gt.mu.Unlock()

	// Validate regions exist
	if _, exists := gt.regions[link.Source]; !exists {
		return fmt.Errorf("source region %s not found", link.Source)
	}
	if _, exists := gt.regions[link.Destination]; !exists {
		return fmt.Errorf("destination region %s not found", link.Destination)
	}

	link.ID = LinkID(fmt.Sprintf("link-%s-%s", link.Source, link.Destination))
	link.Metrics = &LinkMetrics{
		MinLatency: time.Hour, // Initialize to high value
	}
	gt.links[link.ID] = link
	return nil
}

// GetLink retrieves a link by ID
func (gt *GlobalTopology) GetLink(linkID LinkID) (*InterRegionLink, error) {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	link, exists := gt.links[linkID]
	if !exists {
		return nil, fmt.Errorf("link %s not found", linkID)
	}
	return link, nil
}

// GetRegion retrieves a region by ID
func (gt *GlobalTopology) GetRegion(regionID string) (*Region, error) {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	region, exists := gt.regions[regionID]
	if !exists {
		return nil, fmt.Errorf("region %s not found", regionID)
	}
	return region, nil
}

// ListRegions returns all regions
func (gt *GlobalTopology) ListRegions() []*Region {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	regions := make([]*Region, 0, len(gt.regions))
	for _, region := range gt.regions {
		regions = append(regions, region)
	}
	return regions
}

// ListLinks returns all inter-region links
func (gt *GlobalTopology) ListLinks() []*InterRegionLink {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	links := make([]*InterRegionLink, 0, len(gt.links))
	for _, link := range gt.links {
		links = append(links, link)
	}
	return links
}

// UpdateLinkHealth updates the health status of a link
func (gt *GlobalTopology) UpdateLinkHealth(linkID LinkID, health LinkHealth) error {
	gt.mu.RLock()
	link, exists := gt.links[linkID]
	gt.mu.RUnlock()

	if !exists {
		return fmt.Errorf("link %s not found", linkID)
	}

	link.mu.Lock()
	defer link.mu.Unlock()

	link.Health = health
	link.LastProbe = time.Now()
	return nil
}

// UpdateLinkMetrics updates metrics for a link
func (link *InterRegionLink) UpdateMetrics(latency time.Duration, throughput int64, packetLoss float64) {
	link.mu.Lock()
	defer link.mu.Unlock()

	link.Latency = latency
	link.Utilization = float64(throughput) / float64(link.Bandwidth) * 100.0

	// Update detailed metrics
	if link.Metrics != nil {
		link.Metrics.AvgLatency = (link.Metrics.AvgLatency + latency) / 2
		if latency > link.Metrics.MaxLatency {
			link.Metrics.MaxLatency = latency
		}
		if latency < link.Metrics.MinLatency {
			link.Metrics.MinLatency = latency
		}
		// Calculate jitter (variation in latency)
		if link.Metrics.AvgLatency > 0 {
			delta := latency - link.Metrics.AvgLatency
			if delta < 0 {
				delta = -delta
			}
			link.Metrics.Jitter = delta
		}
	}
}

// GetOutgoingLinks returns all links originating from a region
func (gt *GlobalTopology) GetOutgoingLinks(regionID string) []*InterRegionLink {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	links := make([]*InterRegionLink, 0)
	for _, link := range gt.links {
		if link.Source == regionID && link.Health != HealthDown {
			links = append(links, link)
		}
	}
	return links
}

// GetIncomingLinks returns all links terminating at a region
func (gt *GlobalTopology) GetIncomingLinks(regionID string) []*InterRegionLink {
	gt.mu.RLock()
	defer gt.mu.RUnlock()

	links := make([]*InterRegionLink, 0)
	for _, link := range gt.links {
		if link.Destination == regionID && link.Health != HealthDown {
			links = append(links, link)
		}
	}
	return links
}

// NewRoutingTable creates a new routing table
func NewRoutingTable() *RoutingTable {
	return &RoutingTable{
		routes: make(map[RoutingKey]*Route),
	}
}

// Update updates a route in the routing table
func (rt *RoutingTable) Update(source, destination string, route *Route) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	key := RoutingKey{Source: source, Destination: destination}
	route.UpdatedAt = time.Now()
	if _, exists := rt.routes[key]; !exists {
		route.CreatedAt = time.Now()
	}
	rt.routes[key] = route
}

// Get retrieves a route from the routing table
func (rt *RoutingTable) Get(source, destination string) (*Route, bool) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	key := RoutingKey{Source: source, Destination: destination}
	route, exists := rt.routes[key]
	return route, exists
}

// Delete removes a route from the routing table
func (rt *RoutingTable) Delete(source, destination string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	key := RoutingKey{Source: source, Destination: destination}
	delete(rt.routes, key)
}

// List returns all routes
func (rt *RoutingTable) List() []*Route {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	routes := make([]*Route, 0, len(rt.routes))
	for _, route := range rt.routes {
		routes = append(routes, route)
	}
	return routes
}

// String returns a string representation of link health
func (lh LinkHealth) String() string {
	switch lh {
	case HealthUp:
		return "UP"
	case HealthDegraded:
		return "DEGRADED"
	case HealthDown:
		return "DOWN"
	default:
		return "UNKNOWN"
	}
}
