package partition

import (
	"context"
	"errors"
	"math"
	"sort"
	"time"
)

// GeographicOptimizer optimizes VM placement based on geographic constraints
type GeographicOptimizer struct {
	regions map[string]*Region

	// Optimization parameters
	latencyWeight     float64
	reliabilityWeight float64
	costWeight        float64
	sovereigntyWeight float64

	// Cross-region traffic costs
	crossRegionCosts map[string]map[string]float64 // Cost per GB
}

// NewGeographicOptimizer creates a new geographic optimizer
func NewGeographicOptimizer() *GeographicOptimizer {
	return &GeographicOptimizer{
		regions:           make(map[string]*Region),
		latencyWeight:     0.3,
		reliabilityWeight: 0.4,
		costWeight:        0.2,
		sovereigntyWeight: 0.1,
		crossRegionCosts:  make(map[string]map[string]float64),
	}
}

// OptimalPlacement finds the optimal node for VM placement considering geographic factors
func (g *GeographicOptimizer) OptimalPlacement(vm *VM, constraints *Constraints, nodes []*Node) (*Node, error) {
	if len(nodes) == 0 {
		return nil, errors.New("no available nodes")
	}

	// Score each node based on geographic factors
	type nodeScore struct {
		node  *Node
		score float64
	}

	scores := make([]nodeScore, 0, len(nodes))

	for _, node := range nodes {
		score := g.calculateNodeScore(vm, node, constraints)
		scores = append(scores, nodeScore{node: node, score: score})
	}

	// Sort by score (higher is better)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return the best node
	return scores[0].node, nil
}

// calculateNodeScore calculates a placement score for a node
func (g *GeographicOptimizer) calculateNodeScore(vm *VM, node *Node, constraints *Constraints) float64 {
	score := 0.0

	// Geographic proximity score (minimize latency)
	proximityScore := g.calculateProximityScore(vm, node)
	score += proximityScore * g.latencyWeight

	// Reliability score (node uptime and failure rate)
	reliabilityScore := g.calculateReliabilityScore(node)
	score += reliabilityScore * g.reliabilityWeight

	// Cost score (minimize cross-region traffic costs)
	costScore := g.calculateCostScore(vm, node)
	score += costScore * g.costWeight

	// Data sovereignty score (regulatory compliance)
	sovereigntyScore := g.calculateSovereigntyScore(vm, node)
	score += sovereigntyScore * g.sovereigntyWeight

	// Apply constraint penalties
	if constraints != nil {
		// Latency constraint
		if constraints.MaxLatency > 0 {
			region := g.regions[node.Region]
			if region != nil {
				// Check if node can meet latency requirements
				maxLatency := g.getMaxLatencyFromRegion(region)
				if maxLatency > constraints.MaxLatency {
					score *= 0.1 // Heavy penalty for not meeting latency constraint
				}
			}
		}

		// Cost constraint
		if constraints.MaxCostPerHour > 0 && node.CostPerHour > constraints.MaxCostPerHour {
			score *= 0.1 // Heavy penalty for exceeding cost constraint
		}

		// Uptime constraint
		if constraints.RequiredUptime > 0 {
			uptimeRatio := float64(node.Uptime) / float64(time.Hour*24*365)
			if uptimeRatio < constraints.RequiredUptime {
				score *= 0.1 // Heavy penalty for not meeting uptime requirement
			}
		}
	}

	return score
}

// calculateProximityScore calculates score based on geographic proximity
func (g *GeographicOptimizer) calculateProximityScore(vm *VM, node *Node) float64 {
	score := 1.0

	// If VM has required regions, prefer nodes in those regions
	if len(vm.RequiredRegions) > 0 {
		inRequiredRegion := false
		for _, region := range vm.RequiredRegions {
			if node.Region == region {
				inRequiredRegion = true
				break
			}
		}
		if inRequiredRegion {
			score = 1.0
		} else {
			// Calculate distance to nearest required region
			minDistance := g.getDistanceToNearestRegion(node.Region, vm.RequiredRegions)
			// Normalize distance (assuming max distance is Earth's circumference / 2)
			maxDistance := 20000.0 // km
			score = 1.0 - (minDistance / maxDistance)
		}
	}

	// Penalize excluded regions
	for _, region := range vm.ExcludedRegions {
		if node.Region == region {
			score = 0.0
			break
		}
	}

	return score
}

// calculateReliabilityScore calculates score based on node reliability
func (g *GeographicOptimizer) calculateReliabilityScore(node *Node) float64 {
	// Uptime score (normalized to 0-1)
	uptimeScore := float64(node.Uptime) / float64(time.Hour*24*365) // Assume max uptime is 1 year

	// Failure rate score (inverse, lower is better)
	failureScore := 1.0 / (1.0 + node.FailureRate)

	// Node type reliability (cloud > datacenter > edge > volunteer)
	typeScore := 0.0
	switch node.Type {
	case NodeTypeCloud:
		typeScore = 1.0
	case NodeTypeDatacenter:
		typeScore = 0.9
	case NodeTypeEdge:
		typeScore = 0.7
	case NodeTypeVolunteer:
		typeScore = 0.5
	}

	// Combine scores
	return (uptimeScore + failureScore + typeScore) / 3.0
}

// calculateCostScore calculates score based on placement cost
func (g *GeographicOptimizer) calculateCostScore(vm *VM, node *Node) float64 {
	// Base cost score (inverse, lower is better)
	maxCost := 10.0 // Maximum expected cost per hour
	baseCostScore := 1.0 - (node.CostPerHour / maxCost)
	if baseCostScore < 0 {
		baseCostScore = 0
	}

	// Cross-region traffic cost
	crossRegionCost := g.estimateCrossRegionCost(vm, node)
	maxCrossRegionCost := 100.0 // Maximum expected cross-region cost
	crossRegionScore := 1.0 - (crossRegionCost / maxCrossRegionCost)
	if crossRegionScore < 0 {
		crossRegionScore = 0
	}

	// Combine scores
	return (baseCostScore + crossRegionScore) / 2.0
}

// calculateSovereigntyScore calculates score based on data sovereignty requirements
func (g *GeographicOptimizer) calculateSovereigntyScore(vm *VM, node *Node) float64 {
	score := 1.0

	// Check if VM has sovereignty requirements
	if sovereigntyLabel, ok := vm.RequiredLabels["data-sovereignty"]; ok {
		region := g.regions[node.Region]
		if region != nil {
			if !region.DataSovereignty {
				score = 0.0
			} else if region.ComplianceZone != sovereigntyLabel {
				score = 0.5
			}
		}
	}

	return score
}

// getDistanceToNearestRegion calculates distance to nearest required region
func (g *GeographicOptimizer) getDistanceToNearestRegion(currentRegion string, requiredRegions []string) float64 {
	current := g.regions[currentRegion]
	if current == nil {
		return 10000.0 // Default large distance
	}

	minDistance := math.MaxFloat64

	for _, reqRegion := range requiredRegions {
		required := g.regions[reqRegion]
		if required == nil {
			continue
		}

		// Calculate great circle distance
		distance := g.calculateGreatCircleDistance(
			current.Latitude, current.Longitude,
			required.Latitude, required.Longitude,
		)

		if distance < minDistance {
			minDistance = distance
		}
	}

	return minDistance
}

// calculateGreatCircleDistance calculates distance between two points on Earth
func (g *GeographicOptimizer) calculateGreatCircleDistance(lat1, lon1, lat2, lon2 float64) float64 {
	// Convert to radians
	lat1Rad := lat1 * math.Pi / 180
	lon1Rad := lon1 * math.Pi / 180
	lat2Rad := lat2 * math.Pi / 180
	lon2Rad := lon2 * math.Pi / 180

	// Haversine formula
	dlat := lat2Rad - lat1Rad
	dlon := lon2Rad - lon1Rad

	a := math.Sin(dlat/2)*math.Sin(dlat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(dlon/2)*math.Sin(dlon/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	// Earth's radius in km
	earthRadius := 6371.0

	return earthRadius * c
}

// getMaxLatencyFromRegion gets maximum latency from a region to other regions
func (g *GeographicOptimizer) getMaxLatencyFromRegion(region *Region) time.Duration {
	maxLatency := time.Duration(0)

	for _, latency := range region.InternetLatency {
		if latency > maxLatency {
			maxLatency = latency
		}
	}

	return maxLatency
}

// estimateCrossRegionCost estimates cross-region traffic costs
func (g *GeographicOptimizer) estimateCrossRegionCost(vm *VM, node *Node) float64 {
	// Estimate based on VM's expected data transfer patterns
	// This is a simplified model - real implementation would use historical data

	estimatedTransferGB := float64(vm.RequestedMemory) / 1e9 // Rough estimate

	totalCost := 0.0

	// Check if VM will communicate with other VMs in different regions
	if vm.AffinityGroup != "" {
		// Find other VMs in the same affinity group
		// For now, use a default cross-region cost
		if costs, ok := g.crossRegionCosts[node.Region]; ok {
			for _, cost := range costs {
				totalCost += estimatedTransferGB * cost
			}
		} else {
			// Default cost if not specified
			totalCost = estimatedTransferGB * 0.1 // $0.10 per GB
		}
	}

	return totalCost
}

// SetWeights sets the optimization weights
func (g *GeographicOptimizer) SetWeights(latency, reliability, cost, sovereignty float64) {
	sum := latency + reliability + cost + sovereignty
	if sum > 0 {
		g.latencyWeight = latency / sum
		g.reliabilityWeight = reliability / sum
		g.costWeight = cost / sum
		g.sovereigntyWeight = sovereignty / sum
	}
}

// AddRegion adds a region to the optimizer
func (g *GeographicOptimizer) AddRegion(region *Region) {
	g.regions[region.ID] = region
}

// SetCrossRegionCost sets the cost for cross-region traffic
func (g *GeographicOptimizer) SetCrossRegionCost(from, to string, costPerGB float64) {
	if _, ok := g.crossRegionCosts[from]; !ok {
		g.crossRegionCosts[from] = make(map[string]float64)
	}
	g.crossRegionCosts[from][to] = costPerGB
}

// GeographicPlacer implements geographic-aware VM placement
type GeographicPlacer struct {
	optimizer *GeographicOptimizer
}

// NewGeographicPlacer creates a new geographic placer
func NewGeographicPlacer() *GeographicPlacer {
	return &GeographicPlacer{
		optimizer: NewGeographicOptimizer(),
	}
}

// Place places a VM using geographic optimization
func (gp *GeographicPlacer) Place(ctx context.Context, vm *VM, nodes []*Node, constraints *Constraints) (*Node, error) {
	return gp.optimizer.OptimalPlacement(vm, constraints, nodes)
}

// SetRegions sets the regions for the placer
func (gp *GeographicPlacer) SetRegions(regions map[string]*Region) {
	for _, region := range regions {
		gp.optimizer.AddRegion(region)
	}
}

// SetOptimizationWeights configures optimization weights
func (gp *GeographicPlacer) SetOptimizationWeights(latency, reliability, cost, sovereignty float64) {
	gp.optimizer.SetWeights(latency, reliability, cost, sovereignty)
}
