package multiregion

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// RoutingStrategy defines the routing optimization strategy
type RoutingStrategy int

const (
	StrategyLatency   RoutingStrategy = iota // Minimum latency
	StrategyCost                              // Minimum cost
	StrategyBandwidth                         // Maximum bandwidth
	StrategyBalanced                          // Balanced all factors
)

// WeightFunction calculates weight for a link
type WeightFunction func(*InterRegionLink) float64

// RoutingEngine computes optimal routes
type RoutingEngine struct {
	topology *GlobalTopology
	strategy RoutingStrategy
	cache    *routeCache
	mu       sync.RWMutex
}

// routeCache caches computed routes
type routeCache struct {
	routes map[RoutingKey]*cachedRoute
	mu     sync.RWMutex
	ttl    time.Duration
}

// cachedRoute represents a cached route with expiry
type cachedRoute struct {
	route     *Route
	expiresAt time.Time
}

// PriorityQueue implements heap.Interface for Dijkstra's algorithm
type PriorityQueue []*Item

// Item represents an item in the priority queue
type Item struct {
	value    interface{}
	priority float64
	index    int
}

// NewRoutingEngine creates a new routing engine
func NewRoutingEngine(topology *GlobalTopology, strategy RoutingStrategy) *RoutingEngine {
	return &RoutingEngine{
		topology: topology,
		strategy: strategy,
		cache: &routeCache{
			routes: make(map[RoutingKey]*cachedRoute),
			ttl:    5 * time.Minute,
		},
	}
}

// ComputeRoute computes the optimal route between source and destination
func (re *RoutingEngine) ComputeRoute(src, dst string) (*Route, error) {
	// Check cache first
	if route := re.cache.get(src, dst); route != nil {
		return route, nil
	}

	// Validate regions exist
	if _, err := re.topology.GetRegion(src); err != nil {
		return nil, fmt.Errorf("source region: %w", err)
	}
	if _, err := re.topology.GetRegion(dst); err != nil {
		return nil, fmt.Errorf("destination region: %w", err)
	}

	var route *Route
	var err error

	switch re.strategy {
	case StrategyLatency:
		route, err = re.dijkstra(src, dst, latencyWeight)
	case StrategyCost:
		route, err = re.dijkstra(src, dst, costWeight)
	case StrategyBandwidth:
		route, err = re.widestPath(src, dst)
	case StrategyBalanced:
		route, err = re.multiObjective(src, dst)
	default:
		return nil, fmt.Errorf("unknown routing strategy: %d", re.strategy)
	}

	if err != nil {
		return nil, err
	}

	// Cache the computed route
	re.cache.set(src, dst, route)

	return route, nil
}

// dijkstra implements Dijkstra's shortest path algorithm
func (re *RoutingEngine) dijkstra(src, dst string, weightFn WeightFunction) (*Route, error) {
	dist := make(map[string]float64)
	prev := make(map[string]string)
	unvisited := make(map[string]bool)

	// Initialize distances
	for _, region := range re.topology.ListRegions() {
		dist[region.ID] = math.Inf(1)
		unvisited[region.ID] = true
	}
	dist[src] = 0

	// Main loop
	for len(unvisited) > 0 {
		current := re.minDistance(dist, unvisited)
		if current == "" {
			break // No path exists
		}

		if current == dst {
			break
		}

		delete(unvisited, current)

		// Examine neighbors
		for _, link := range re.topology.GetOutgoingLinks(current) {
			neighbor := link.Destination
			if !unvisited[neighbor] {
				continue
			}

			weight := weightFn(link)
			alt := dist[current] + weight

			if alt < dist[neighbor] {
				dist[neighbor] = alt
				prev[neighbor] = current
			}
		}
	}

	// No path found
	if dist[dst] == math.Inf(1) {
		return nil, fmt.Errorf("no path from %s to %s", src, dst)
	}

	return re.reconstructPath(prev, src, dst)
}

// widestPath finds the path with maximum bandwidth (maximum bottleneck bandwidth)
func (re *RoutingEngine) widestPath(src, dst string) (*Route, error) {
	bandwidth := make(map[string]int64)
	prev := make(map[string]string)

	// Initialize
	for _, region := range re.topology.ListRegions() {
		bandwidth[region.ID] = 0
	}
	bandwidth[src] = math.MaxInt64

	// Priority queue (max heap)
	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Item{value: src, priority: float64(bandwidth[src])})

	visited := make(map[string]bool)

	for pq.Len() > 0 {
		current := heap.Pop(pq).(*Item).value.(string)

		if visited[current] {
			continue
		}
		visited[current] = true

		if current == dst {
			break
		}

		for _, link := range re.topology.GetOutgoingLinks(current) {
			neighbor := link.Destination
			if visited[neighbor] {
				continue
			}

			// Available bandwidth is minimum of current path and link bandwidth
			availableBW := link.Bandwidth * (100 - int64(link.Utilization)) / 100
			minBW := min(bandwidth[current], availableBW)

			if minBW > bandwidth[neighbor] {
				bandwidth[neighbor] = minBW
				prev[neighbor] = current
				heap.Push(pq, &Item{value: neighbor, priority: float64(minBW)})
			}
		}
	}

	if bandwidth[dst] == 0 {
		return nil, fmt.Errorf("no path from %s to %s", src, dst)
	}

	return re.reconstructPath(prev, src, dst)
}

// multiObjective finds a balanced route considering multiple metrics
func (re *RoutingEngine) multiObjective(src, dst string) (*Route, error) {
	// Use weighted sum of normalized metrics
	weightFn := func(link *InterRegionLink) float64 {
		// Normalize metrics (assuming max values)
		const (
			maxLatency   = 1000.0 // 1000ms
			maxCost      = 100.0  // $100
			maxUtilization = 100.0
		)

		normalizedLatency := float64(link.Latency.Milliseconds()) / maxLatency
		normalizedCost := link.Cost / maxCost
		normalizedUtil := link.Utilization / maxUtilization

		// Weighted combination (can be tuned)
		return 0.4*normalizedLatency + 0.3*normalizedCost + 0.3*normalizedUtil
	}

	return re.dijkstra(src, dst, weightFn)
}

// minDistance finds the unvisited node with minimum distance
func (re *RoutingEngine) minDistance(dist map[string]float64, unvisited map[string]bool) string {
	minDist := math.Inf(1)
	var minNode string

	for node := range unvisited {
		if dist[node] < minDist {
			minDist = dist[node]
			minNode = node
		}
	}

	return minNode
}

// reconstructPath reconstructs the path from prev map
func (re *RoutingEngine) reconstructPath(prev map[string]string, src, dst string) (*Route, error) {
	path := []string{}
	links := []LinkID{}

	current := dst
	for current != src {
		path = append([]string{current}, path...)
		prevNode, exists := prev[current]
		if !exists {
			return nil, fmt.Errorf("path reconstruction failed at %s", current)
		}

		// Find the link
		linkID := LinkID(fmt.Sprintf("link-%s-%s", prevNode, current))
		links = append([]LinkID{linkID}, links...)

		current = prevNode
	}
	path = append([]string{src}, path...)

	// Calculate route metrics
	metric := re.calculateRouteMetric(links)

	return &Route{
		Destination: dst,
		NextHop:     path[1],
		Path:        path,
		Links:       links,
		Metric:      metric,
		Priority:    1,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}, nil
}

// calculateRouteMetric calculates aggregate metrics for a route
func (re *RoutingEngine) calculateRouteMetric(links []LinkID) RouteMetric {
	var totalLatency time.Duration
	var totalCost float64
	var minBandwidth int64 = math.MaxInt64
	var reliabilityProduct float64 = 1.0

	for _, linkID := range links {
		link, err := re.topology.GetLink(linkID)
		if err != nil {
			continue
		}

		totalLatency += link.Latency
		totalCost += link.Cost

		// Available bandwidth is the bottleneck
		availableBW := link.Bandwidth * (100 - int64(link.Utilization)) / 100
		if availableBW < minBandwidth {
			minBandwidth = availableBW
		}

		// Reliability as (100 - utilization) / 100
		reliability := (100.0 - link.Utilization) / 100.0
		reliabilityProduct *= reliability
	}

	return RouteMetric{
		Latency:     totalLatency,
		Cost:        totalCost,
		Bandwidth:   minBandwidth,
		Hops:        len(links),
		Reliability: reliabilityProduct,
	}
}

// FindEqualCostPaths finds all paths with equal cost (for ECMP)
func (re *RoutingEngine) FindEqualCostPaths(src, dst string) ([]*Route, error) {
	// For simplicity, find the optimal path and then find paths with similar cost
	optimal, err := re.ComputeRoute(src, dst)
	if err != nil {
		return nil, err
	}

	paths := []*Route{optimal}

	// Find alternative paths within 10% cost of optimal
	threshold := optimal.Metric.Latency.Seconds() * 1.1

	// Try alternative routes by removing each link and recomputing
	for _, linkID := range optimal.Links {
		// Temporarily mark link as down
		link, _ := re.topology.GetLink(linkID)
		originalHealth := link.Health
		link.Health = HealthDown

		altRoute, err := re.dijkstra(src, dst, latencyWeight)
		if err == nil && altRoute.Metric.Latency.Seconds() <= threshold {
			paths = append(paths, altRoute)
		}

		// Restore link
		link.Health = originalHealth
	}

	return paths, nil
}

// Weight functions for different optimization strategies
func latencyWeight(link *InterRegionLink) float64 {
	return float64(link.Latency.Milliseconds())
}

func costWeight(link *InterRegionLink) float64 {
	return link.Cost
}

// routeCache methods
func (rc *routeCache) get(src, dst string) *Route {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	key := RoutingKey{Source: src, Destination: dst}
	cached, exists := rc.routes[key]
	if !exists {
		return nil
	}

	if time.Now().After(cached.expiresAt) {
		delete(rc.routes, key)
		return nil
	}

	return cached.route
}

func (rc *routeCache) set(src, dst string, route *Route) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	key := RoutingKey{Source: src, Destination: dst}
	rc.routes[key] = &cachedRoute{
		route:     route,
		expiresAt: time.Now().Add(rc.ttl),
	}
}

func (rc *routeCache) invalidate(src, dst string) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	key := RoutingKey{Source: src, Destination: dst}
	delete(rc.routes, key)
}

// InvalidateCache invalidates cached routes
func (re *RoutingEngine) InvalidateCache(src, dst string) {
	re.cache.invalidate(src, dst)
}

// Priority Queue implementation for heap
func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// Max heap for widest path
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

var (
	ErrNoPath          = errors.New("no path found")
	ErrInvalidStrategy = errors.New("invalid routing strategy")
)
