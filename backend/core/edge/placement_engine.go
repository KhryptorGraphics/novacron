package edge

import (
	"context"
	"math"
	"sort"
	"time"
)

// PlacementEngine handles edge placement decisions
type PlacementEngine struct {
	config    *EdgeConfig
	discovery *EdgeDiscovery
}

// NewPlacementEngine creates a new placement engine
func NewPlacementEngine(config *EdgeConfig, discovery *EdgeDiscovery) *PlacementEngine {
	return &PlacementEngine{
		config:    config,
		discovery: discovery,
	}
}

// PlaceVM determines the optimal edge node for VM placement
func (pe *PlacementEngine) PlaceVM(ctx context.Context, req *PlacementRequest) (*PlacementDecision, error) {
	startTime := time.Now()

	// Get healthy edge nodes
	nodes := pe.discovery.GetHealthyNodes()
	if len(nodes) == 0 {
		return nil, ErrInsufficientEdgeNodes
	}

	// Filter nodes by hard constraints
	candidates := pe.filterByConstraints(nodes, req)
	if len(candidates) == 0 {
		return nil, ErrNoSuitableEdgeNode
	}

	// Score each candidate
	scores := make([]*scoredNode, 0, len(candidates))
	for _, node := range candidates {
		score := pe.calculatePlacementScore(node, req)
		if score != nil {
			scores = append(scores, score)
		}
	}

	if len(scores) == 0 {
		return nil, ErrNoSuitableEdgeNode
	}

	// Sort by total score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].totalScore > scores[j].totalScore
	})

	// Select best node
	best := scores[0]
	decision := &PlacementDecision{
		EdgeNodeID:       best.node.ID,
		Score:            best.totalScore,
		LatencyScore:     best.latencyScore,
		ResourceScore:    best.resourceScore,
		CostScore:        best.costScore,
		ProximityScore:   best.proximityScore,
		EstimatedLatency: best.node.Latency.RTTAvg,
		DecisionTime:     time.Since(startTime),
		CreatedAt:        time.Now(),
		Reason:           best.reason,
	}

	// Check if decision time meets target
	if decision.DecisionTime > pe.config.TargetPlacementTime {
		// Log warning but don't fail
	}

	return decision, nil
}

// filterByConstraints filters nodes by hard constraints
func (pe *PlacementEngine) filterByConstraints(nodes []*EdgeNode, req *PlacementRequest) []*EdgeNode {
	candidates := make([]*EdgeNode, 0)

	for _, node := range nodes {
		// Check resource requirements
		if !pe.hasRequiredResources(node, req.Requirements) {
			continue
		}

		// Check latency constraint
		if req.Constraints.MaxLatency > 0 &&
		   node.Latency.RTTAvg > req.Constraints.MaxLatency {
			continue
		}

		// Check region constraints
		if !pe.matchesRegionConstraints(node, req.Constraints) {
			continue
		}

		// Check edge type constraint
		if req.Constraints.RequiredEdgeType != "" &&
		   node.Type != req.Constraints.RequiredEdgeType {
			continue
		}

		// Check data residency
		if req.Constraints.DataResidency != "" {
			if !pe.matchesDataResidency(node, req.Constraints.DataResidency) {
				continue
			}
		}

		// Check architecture
		if req.Requirements.Architecture != "" {
			if !pe.supportsArchitecture(node, req.Requirements.Architecture) {
				continue
			}
		}

		candidates = append(candidates, node)
	}

	return candidates
}

// hasRequiredResources checks if node has required resources
func (pe *PlacementEngine) hasRequiredResources(node *EdgeNode, req PlacementRequirements) bool {
	availCPU := node.Resources.TotalCPUCores - node.Resources.UsedCPUCores
	availMem := node.Resources.TotalMemoryMB - node.Resources.UsedMemoryMB
	availStorage := node.Resources.TotalStorageGB - node.Resources.UsedStorageGB
	availBW := node.Resources.TotalBandwidthMbps - node.Resources.UsedBandwidthMbps

	if availCPU < req.CPUCores {
		return false
	}

	if availMem < req.MemoryMB {
		return false
	}

	if availStorage < req.StorageGB {
		return false
	}

	if availBW < req.BandwidthMbps {
		return false
	}

	if req.GPURequired && node.Resources.GPUCount == 0 {
		return false
	}

	return true
}

// matchesRegionConstraints checks region constraints
func (pe *PlacementEngine) matchesRegionConstraints(node *EdgeNode, constraints PlacementConstraints) bool {
	// Check required regions
	if len(constraints.RequiredRegions) > 0 {
		found := false
		for _, region := range constraints.RequiredRegions {
			if node.Location.Region == region || node.Location.Country == region {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check excluded regions
	for _, region := range constraints.ExcludedRegions {
		if node.Location.Region == region || node.Location.Country == region {
			return false
		}
	}

	return true
}

// matchesDataResidency checks data residency compliance
func (pe *PlacementEngine) matchesDataResidency(node *EdgeNode, residency string) bool {
	switch residency {
	case "EU":
		// EU countries
		euCountries := []string{"DE", "FR", "IT", "ES", "NL", "BE", "AT", "IE", "FI", "SE"}
		for _, country := range euCountries {
			if node.Location.Country == country {
				return true
			}
		}
		return false
	case "US":
		return node.Location.Country == "US"
	case "APAC":
		apacCountries := []string{"JP", "SG", "AU", "IN", "KR", "CN", "HK"}
		for _, country := range apacCountries {
			if node.Location.Country == country {
				return true
			}
		}
		return false
	default:
		return node.Location.Country == residency
	}
}

// supportsArchitecture checks architecture support
func (pe *PlacementEngine) supportsArchitecture(node *EdgeNode, arch string) bool {
	switch arch {
	case "x86_64", "amd64":
		return node.Capabilities.SupportsX86
	case "arm64", "aarch64":
		return node.Capabilities.SupportsARM64
	default:
		return true
	}
}

// scoredNode represents a scored edge node
type scoredNode struct {
	node           *EdgeNode
	totalScore     float64
	latencyScore   float64
	resourceScore  float64
	costScore      float64
	proximityScore float64
	reason         string
}

// calculatePlacementScore calculates placement score for a node
func (pe *PlacementEngine) calculatePlacementScore(node *EdgeNode, req *PlacementRequest) *scoredNode {
	weights := pe.config.PlacementWeights

	// Calculate latency score (lower is better, normalize to 0-1)
	latencyMs := float64(node.Latency.RTTAvg) / float64(time.Millisecond)
	maxLatencyMs := float64(pe.config.MaxEdgeLatency) / float64(time.Millisecond)
	latencyScore := 1.0 - math.Min(latencyMs/maxLatencyMs, 1.0)

	// Calculate resource score (lower utilization is better)
	resourceScore := 1.0 - (node.Resources.UtilizationPercent / 100.0)

	// Calculate cost score (lower cost is better, normalize)
	// Assume max cost is $1/hour
	costScore := 1.0 - math.Min(node.Cost.CostPerHour/1.0, 1.0)

	// Calculate proximity score if user location provided
	proximityScore := 0.0
	if req.UserLocation != nil && weights.Proximity > 0 {
		distance := pe.calculateDistance(req.UserLocation, &node.Location)
		// Normalize distance (assume max relevant distance is 5000km)
		proximityScore = 1.0 - math.Min(distance/5000.0, 1.0)
	}

	// Calculate total weighted score
	totalScore := weights.Latency*latencyScore +
		weights.Resources*resourceScore +
		weights.Cost*costScore +
		weights.Proximity*proximityScore

	// Bonus for preferred regions
	if len(req.Preferences.PreferredRegions) > 0 {
		for _, region := range req.Preferences.PreferredRegions {
			if node.Location.Region == region {
				totalScore += 0.1 // 10% bonus
				break
			}
		}
	}

	// Penalty for high utilization
	if node.Resources.UtilizationPercent > 80.0 {
		totalScore *= 0.9 // 10% penalty
	}

	// Penalty for recent errors
	if node.Status.ErrorCount > 5 {
		totalScore *= 0.8 // 20% penalty
	}

	reason := pe.generateReason(latencyScore, resourceScore, costScore, proximityScore)

	return &scoredNode{
		node:           node,
		totalScore:     totalScore,
		latencyScore:   latencyScore,
		resourceScore:  resourceScore,
		costScore:      costScore,
		proximityScore: proximityScore,
		reason:         reason,
	}
}

// calculateDistance calculates distance between two geographic locations (Haversine formula)
func (pe *PlacementEngine) calculateDistance(loc1, loc2 *GeoLocation) float64 {
	const earthRadius = 6371.0 // km

	lat1 := loc1.Latitude * math.Pi / 180.0
	lat2 := loc2.Latitude * math.Pi / 180.0
	deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180.0
	deltaLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180.0

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

// generateReason generates a human-readable reason for placement decision
func (pe *PlacementEngine) generateReason(latency, resource, cost, proximity float64) string {
	if latency > 0.8 {
		return "Selected for ultra-low latency"
	}
	if resource > 0.8 {
		return "Selected for abundant resources"
	}
	if cost > 0.8 {
		return "Selected for cost efficiency"
	}
	if proximity > 0.8 {
		return "Selected for geographic proximity"
	}
	return "Selected for balanced optimization"
}

// FindNearestNodes finds N nearest nodes to a location
func (pe *PlacementEngine) FindNearestNodes(location *GeoLocation, count int) []*EdgeNode {
	nodes := pe.discovery.GetHealthyNodes()

	type nodeDistance struct {
		node     *EdgeNode
		distance float64
	}

	distances := make([]nodeDistance, len(nodes))
	for i, node := range nodes {
		distances[i] = nodeDistance{
			node:     node,
			distance: pe.calculateDistance(location, &node.Location),
		}
	}

	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Return top N
	result := make([]*EdgeNode, 0, count)
	for i := 0; i < count && i < len(distances); i++ {
		result = append(result, distances[i].node)
	}

	return result
}

// CalculateUserProximity calculates user proximity to edge nodes
func (pe *PlacementEngine) CalculateUserProximity(location *GeoLocation, count int) (*UserProximity, error) {
	nearestNodes := pe.FindNearestNodes(location, count)

	distances := make([]float64, len(nearestNodes))
	estimatedLatencies := make([]time.Duration, len(nearestNodes))

	for i, node := range nearestNodes {
		distance := pe.calculateDistance(location, &node.Location)
		distances[i] = distance

		// Estimate latency based on distance
		// Rough approximation: 0.1ms per 10km + base latency
		estimatedMs := (distance / 10.0 * 0.1) + float64(node.Latency.RTTAvg)/float64(time.Millisecond)
		estimatedLatencies[i] = time.Duration(estimatedMs) * time.Millisecond
	}

	return &UserProximity{
		UserLocation:       *location,
		NearestNodes:       nearestNodes,
		DistancesKM:        distances,
		EstimatedLatencies: estimatedLatencies,
		CalculatedAt:       time.Now(),
	}, nil
}

// RecommendPlacement provides placement recommendations with alternatives
func (pe *PlacementEngine) RecommendPlacement(ctx context.Context, req *PlacementRequest, topN int) ([]*PlacementDecision, error) {
	nodes := pe.discovery.GetHealthyNodes()
	if len(nodes) == 0 {
		return nil, ErrInsufficientEdgeNodes
	}

	candidates := pe.filterByConstraints(nodes, req)
	if len(candidates) == 0 {
		return nil, ErrNoSuitableEdgeNode
	}

	scores := make([]*scoredNode, 0, len(candidates))
	for _, node := range candidates {
		score := pe.calculatePlacementScore(node, req)
		if score != nil {
			scores = append(scores, score)
		}
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].totalScore > scores[j].totalScore
	})

	// Return top N recommendations
	count := topN
	if count > len(scores) {
		count = len(scores)
	}

	decisions := make([]*PlacementDecision, count)
	for i := 0; i < count; i++ {
		s := scores[i]
		decisions[i] = &PlacementDecision{
			EdgeNodeID:       s.node.ID,
			Score:            s.totalScore,
			LatencyScore:     s.latencyScore,
			ResourceScore:    s.resourceScore,
			CostScore:        s.costScore,
			ProximityScore:   s.proximityScore,
			EstimatedLatency: s.node.Latency.RTTAvg,
			CreatedAt:        time.Now(),
			Reason:           s.reason,
		}
	}

	return decisions, nil
}
