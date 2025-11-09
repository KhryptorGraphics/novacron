package consensus

import (
	"fmt"
)

// QuorumStrategy defines different quorum calculation strategies
type QuorumStrategy interface {
	GetQuorumSize(totalNodes int) int
	IsQuorum(responses int, totalNodes int) bool
	String() string
}

// MajorityQuorum implements simple majority quorum (n/2 + 1)
type MajorityQuorum struct{}

// NewMajorityQuorum creates a new majority quorum strategy
func NewMajorityQuorum() *MajorityQuorum {
	return &MajorityQuorum{}
}

// GetQuorumSize returns the quorum size for majority
func (mq *MajorityQuorum) GetQuorumSize(totalNodes int) int {
	return totalNodes/2 + 1
}

// IsQuorum checks if we have a majority quorum
func (mq *MajorityQuorum) IsQuorum(responses int, totalNodes int) bool {
	return responses >= mq.GetQuorumSize(totalNodes)
}

func (mq *MajorityQuorum) String() string {
	return "MajorityQuorum"
}

// FlexibleQuorum implements flexible quorum (R + W > N)
// Allows trading off read and write quorum sizes
type FlexibleQuorum struct {
	WriteQuorum int
	ReadQuorum  int
	TotalNodes  int
}

// NewFlexibleQuorum creates a new flexible quorum strategy
func NewFlexibleQuorum(totalNodes, writeQuorum, readQuorum int) (*FlexibleQuorum, error) {
	if writeQuorum+readQuorum <= totalNodes {
		return nil, fmt.Errorf("invalid flexible quorum: W + R must be > N")
	}

	return &FlexibleQuorum{
		WriteQuorum: writeQuorum,
		ReadQuorum:  readQuorum,
		TotalNodes:  totalNodes,
	}, nil
}

// GetQuorumSize returns the write quorum size
func (fq *FlexibleQuorum) GetQuorumSize(totalNodes int) int {
	return fq.WriteQuorum
}

// IsQuorum checks if we have write quorum
func (fq *FlexibleQuorum) IsQuorum(responses int, totalNodes int) bool {
	return responses >= fq.WriteQuorum
}

// IsReadQuorum checks if we have read quorum
func (fq *FlexibleQuorum) IsReadQuorum(responses int) bool {
	return responses >= fq.ReadQuorum
}

func (fq *FlexibleQuorum) String() string {
	return fmt.Sprintf("FlexibleQuorum(W=%d,R=%d)", fq.WriteQuorum, fq.ReadQuorum)
}

// GeographicQuorum requires majority in each geographic region
type GeographicQuorum struct {
	regions map[string]int // region -> node count
}

// NewGeographicQuorum creates a new geographic quorum strategy
func NewGeographicQuorum(regions map[string]int) *GeographicQuorum {
	return &GeographicQuorum{
		regions: regions,
	}
}

// GetQuorumSize returns total quorum size across all regions
func (gq *GeographicQuorum) GetQuorumSize(totalNodes int) int {
	quorum := 0
	for _, nodeCount := range gq.regions {
		quorum += nodeCount/2 + 1
	}
	return quorum
}

// IsQuorum checks if we have majority in all regions
func (gq *GeographicQuorum) IsQuorum(responses int, totalNodes int) bool {
	// This is a simplified check
	// Real implementation would track responses per region
	return responses >= gq.GetQuorumSize(totalNodes)
}

// IsRegionQuorum checks if we have quorum for specific regions
func (gq *GeographicQuorum) IsRegionQuorum(regionResponses map[string]int) bool {
	for region, nodeCount := range gq.regions {
		required := nodeCount/2 + 1
		if regionResponses[region] < required {
			return false
		}
	}
	return true
}

// AddRegion adds a new region to the quorum calculation
func (gq *GeographicQuorum) AddRegion(regionID string, nodeCount int) {
	gq.regions[regionID] = nodeCount
}

// RemoveRegion removes a region from the quorum calculation
func (gq *GeographicQuorum) RemoveRegion(regionID string) {
	delete(gq.regions, regionID)
}

func (gq *GeographicQuorum) String() string {
	return fmt.Sprintf("GeographicQuorum(regions=%d)", len(gq.regions))
}

// FastPathQuorum implements EPaxos fast-path quorum
// Requires F + ⌊F/2⌋ + 1 responses (where F = ⌊N/2⌋)
type FastPathQuorum struct {
	totalNodes int
}

// NewFastPathQuorum creates a new fast-path quorum strategy
func NewFastPathQuorum(totalNodes int) *FastPathQuorum {
	return &FastPathQuorum{
		totalNodes: totalNodes,
	}
}

// GetQuorumSize returns the fast-path quorum size
func (fpq *FastPathQuorum) GetQuorumSize(totalNodes int) int {
	f := totalNodes / 2
	return f + f/2 + 1
}

// IsQuorum checks if we have fast-path quorum
func (fpq *FastPathQuorum) IsQuorum(responses int, totalNodes int) bool {
	return responses >= fpq.GetQuorumSize(totalNodes)
}

// GetSlowPathQuorumSize returns the slow-path (classic) quorum size
func (fpq *FastPathQuorum) GetSlowPathQuorumSize(totalNodes int) int {
	return totalNodes/2 + 1
}

// IsSlowPathQuorum checks if we have slow-path quorum
func (fpq *FastPathQuorum) IsSlowPathQuorum(responses int, totalNodes int) bool {
	return responses >= fpq.GetSlowPathQuorumSize(totalNodes)
}

func (fpq *FastPathQuorum) String() string {
	return "FastPathQuorum"
}

// HierarchicalQuorum implements hierarchical quorum for multi-tier systems
type HierarchicalQuorum struct {
	tiers []TierQuorum
}

// TierQuorum represents quorum requirements for a tier
type TierQuorum struct {
	Name        string
	Nodes       int
	QuorumRatio float64 // fraction of nodes required
}

// NewHierarchicalQuorum creates a new hierarchical quorum strategy
func NewHierarchicalQuorum(tiers []TierQuorum) *HierarchicalQuorum {
	return &HierarchicalQuorum{
		tiers: tiers,
	}
}

// GetQuorumSize returns total quorum size across all tiers
func (hq *HierarchicalQuorum) GetQuorumSize(totalNodes int) int {
	quorum := 0
	for _, tier := range hq.tiers {
		tierQuorum := int(float64(tier.Nodes) * tier.QuorumRatio)
		if tierQuorum == 0 {
			tierQuorum = 1 // At least one node per tier
		}
		quorum += tierQuorum
	}
	return quorum
}

// IsQuorum checks if we have quorum across all tiers
func (hq *HierarchicalQuorum) IsQuorum(responses int, totalNodes int) bool {
	// Simplified check
	return responses >= hq.GetQuorumSize(totalNodes)
}

// IsTierQuorum checks if we have quorum for specific tiers
func (hq *HierarchicalQuorum) IsTierQuorum(tierResponses map[string]int) bool {
	for _, tier := range hq.tiers {
		required := int(float64(tier.Nodes) * tier.QuorumRatio)
		if required == 0 {
			required = 1
		}
		if tierResponses[tier.Name] < required {
			return false
		}
	}
	return true
}

func (hq *HierarchicalQuorum) String() string {
	return fmt.Sprintf("HierarchicalQuorum(tiers=%d)", len(hq.tiers))
}

// WeightedQuorum implements weighted quorum where nodes have different weights
type WeightedQuorum struct {
	nodeWeights  map[string]int // node -> weight
	totalWeight  int
	quorumWeight int
}

// NewWeightedQuorum creates a new weighted quorum strategy
func NewWeightedQuorum(nodeWeights map[string]int) *WeightedQuorum {
	totalWeight := 0
	for _, weight := range nodeWeights {
		totalWeight += weight
	}

	return &WeightedQuorum{
		nodeWeights:  nodeWeights,
		totalWeight:  totalWeight,
		quorumWeight: totalWeight/2 + 1,
	}
}

// GetQuorumSize returns the quorum weight required
func (wq *WeightedQuorum) GetQuorumSize(totalNodes int) int {
	// Return number of nodes needed if all had weight 1
	return wq.quorumWeight
}

// IsQuorum checks if we have sufficient weight
func (wq *WeightedQuorum) IsQuorum(responses int, totalNodes int) bool {
	// This is simplified - real implementation would track which nodes responded
	return responses >= wq.quorumWeight
}

// IsWeightedQuorum checks if responding nodes have sufficient weight
func (wq *WeightedQuorum) IsWeightedQuorum(respondingNodes []string) bool {
	weight := 0
	for _, node := range respondingNodes {
		weight += wq.nodeWeights[node]
	}
	return weight >= wq.quorumWeight
}

// SetNodeWeight updates the weight for a node
func (wq *WeightedQuorum) SetNodeWeight(nodeID string, weight int) {
	oldWeight := wq.nodeWeights[nodeID]
	wq.nodeWeights[nodeID] = weight
	wq.totalWeight = wq.totalWeight - oldWeight + weight
	wq.quorumWeight = wq.totalWeight/2 + 1
}

func (wq *WeightedQuorum) String() string {
	return fmt.Sprintf("WeightedQuorum(total=%d,quorum=%d)", wq.totalWeight, wq.quorumWeight)
}

// QuorumFactory creates appropriate quorum strategies
type QuorumFactory struct{}

// CreateQuorum creates a quorum strategy based on type
func (qf *QuorumFactory) CreateQuorum(quorumType QuorumType, config map[string]interface{}) (QuorumStrategy, error) {
	switch quorumType {
	case QuorumMajority:
		return NewMajorityQuorum(), nil

	case QuorumFlexible:
		totalNodes := config["totalNodes"].(int)
		writeQuorum := config["writeQuorum"].(int)
		readQuorum := config["readQuorum"].(int)
		return NewFlexibleQuorum(totalNodes, writeQuorum, readQuorum)

	case QuorumGeographic:
		regions := config["regions"].(map[string]int)
		return NewGeographicQuorum(regions), nil

	case QuorumFastPath:
		totalNodes := config["totalNodes"].(int)
		return NewFastPathQuorum(totalNodes), nil

	default:
		return nil, fmt.Errorf("unknown quorum type: %v", quorumType)
	}
}
