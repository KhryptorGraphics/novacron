package placement

// Constraint represents a constraint for orchestration decisions
type Constraint struct {
	Type        ConstraintType         `json:"type"`
	Enforcement ConstraintEnforcement  `json:"enforcement"`
	Parameters  map[string]interface{} `json:"parameters"`
	Weight      float64                `json:"weight,omitempty"`
}

// ConstraintType defines the type of constraint
type ConstraintType string

const (
	ConstraintTypeAffinity       ConstraintType = "affinity"
	ConstraintTypeAntiAffinity   ConstraintType = "anti_affinity"
	ConstraintTypeResourceLimit  ConstraintType = "resource_limit"
	ConstraintTypeNetworkLatency ConstraintType = "network_latency"
	ConstraintTypeCompliance     ConstraintType = "compliance"
	ConstraintTypeCost           ConstraintType = "cost"
	ConstraintTypeAvailability   ConstraintType = "availability"
)

// ConstraintEnforcement defines how strictly a constraint should be enforced
type ConstraintEnforcement string

const (
	EnforcementHard       ConstraintEnforcement = "hard"       // Must be satisfied
	EnforcementSoft       ConstraintEnforcement = "soft"       // Preferred but can be violated
	EnforcementPreferred  ConstraintEnforcement = "preferred"  // Weighted preference
)