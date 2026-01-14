package policy

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PolicyType defines the category of a scheduling policy
type PolicyType string

const (
	// PlacementPolicy affects VM placement decisions
	PlacementPolicy PolicyType = "placement"
	// MigrationPolicy affects VM migration decisions
	MigrationPolicy PolicyType = "migration"
	// ResourceAllocationPolicy affects resource allocation decisions
	ResourceAllocationPolicy PolicyType = "resource_allocation"
	// MaintenancePolicy affects maintenance scheduling
	MaintenancePolicy PolicyType = "maintenance"
	// CompositePolicY combines multiple policies
	CompositePolicy PolicyType = "composite"
)

// PolicyStatus represents the current status of a policy
type PolicyStatus string

const (
	// PolicyStatusDraft indicates a policy that is not yet active
	PolicyStatusDraft PolicyStatus = "draft"
	// PolicyStatusActive indicates an active policy
	PolicyStatusActive PolicyStatus = "active"
	// PolicyStatusDisabled indicates a disabled policy
	PolicyStatusDisabled PolicyStatus = "disabled"
	// PolicyStatusDeprecated indicates a deprecated policy that should not be used
	PolicyStatusDeprecated PolicyStatus = "deprecated"
)

// PolicyRule represents a single rule within a policy
type PolicyRule struct {
	// ID is a unique identifier for this rule
	ID string

	// Name is a human-readable name for this rule
	Name string

	// Description provides details about the rule's purpose
	Description string

	// Condition is an expression that determines if the rule applies
	Condition Expression

	// Actions are the actions to perform if the condition is true
	Actions []PolicyAction

	// Priority determines the rule's precedence (higher is more important)
	Priority int

	// Weight affects the rule's influence when scoring (0-100)
	Weight int

	// IsHardConstraint indicates if this rule is a hard requirement
	IsHardConstraint bool
}

// PolicyAction represents an action to take as part of a rule
type PolicyAction interface {
	// Execute performs the action in the given context
	Execute(ctx context.Context, evalCtx *EvaluationContext) error

	// GetType returns the type of action
	GetType() string

	// GetParameters returns the action's parameters
	GetParameters() map[string]interface{}
}

// ScoreAction adjusts the score of a placement candidate
type ScoreAction struct {
	// ScoreExpression determines the score adjustment
	ScoreExpression Expression

	// Reason explains the score adjustment
	Reason string
}

// Execute implements PolicyAction.Execute for ScoreAction
func (a *ScoreAction) Execute(ctx context.Context, evalCtx *EvaluationContext) error {
	score, err := a.ScoreExpression.Evaluate(evalCtx)
	if err != nil {
		return fmt.Errorf("failed to evaluate score expression: %v", err)
	}

	// Convert the expression result to a float64 score
	scoreVal, ok := score.(float64)
	if !ok {
		return fmt.Errorf("score expression must evaluate to a number")
	}

	// Update the score in the evaluation context
	evalCtx.AddScore(scoreVal, a.Reason)
	return nil
}

// GetType implements PolicyAction.GetType for ScoreAction
func (a *ScoreAction) GetType() string {
	return "score"
}

// GetParameters implements PolicyAction.GetParameters for ScoreAction
func (a *ScoreAction) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"expression": a.ScoreExpression.String(),
		"reason":     a.Reason,
	}
}

// FilterAction filters out a placement candidate
type FilterAction struct {
	// Reason explains why the candidate was filtered
	Reason string
}

// Execute implements PolicyAction.Execute for FilterAction
func (a *FilterAction) Execute(ctx context.Context, evalCtx *EvaluationContext) error {
	evalCtx.SetFiltered(true, a.Reason)
	return nil
}

// GetType implements PolicyAction.GetType for FilterAction
func (a *FilterAction) GetType() string {
	return "filter"
}

// GetParameters implements PolicyAction.GetParameters for FilterAction
func (a *FilterAction) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"reason": a.Reason,
	}
}

// LogAction logs information during policy evaluation
type LogAction struct {
	// Message is the log message
	Message string

	// Level is the log level
	Level string

	// AdditionalData contains extra data to log
	AdditionalData map[string]interface{}
}

// Execute implements PolicyAction.Execute for LogAction
func (a *LogAction) Execute(ctx context.Context, evalCtx *EvaluationContext) error {
	// In a real implementation, this would log to a proper logging system
	fmt.Printf("[%s] %s: %v\n", a.Level, a.Message, a.AdditionalData)
	return nil
}

// GetType implements PolicyAction.GetType for LogAction
func (a *LogAction) GetType() string {
	return "log"
}

// GetParameters implements PolicyAction.GetParameters for LogAction
func (a *LogAction) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"message":        a.Message,
		"level":          a.Level,
		"additionalData": a.AdditionalData,
	}
}

// SchedulingPolicy represents a complete scheduling policy
type SchedulingPolicy struct {
	// ID is a unique identifier for this policy
	ID string

	// Name is a human-readable name
	Name string

	// Description provides details about the policy's purpose
	Description string

	// Type indicates the policy's category
	Type PolicyType

	// Version is the policy version
	Version string

	// Status indicates if the policy is active, disabled, etc.
	Status PolicyStatus

	// Rules are the policy's rules in priority order
	Rules []*PolicyRule

	// Parameters are configurable settings for the policy
	Parameters map[string]*PolicyParameter

	// TargetSelector determines which VMs/nodes this policy applies to
	TargetSelector Expression

	// Metadata contains additional information about the policy
	Metadata map[string]interface{}

	// CreatedAt is when the policy was created
	CreatedAt time.Time

	// UpdatedAt is when the policy was last updated
	UpdatedAt time.Time

	// CreatedBy is who created the policy
	CreatedBy string

	// UpdatedBy is who last updated the policy
	UpdatedBy string
}

// PolicyParameter represents a configurable parameter for a policy
type PolicyParameter struct {
	// Name is the parameter name
	Name string

	// Description explains the parameter's purpose
	Description string

	// Type is the parameter's data type
	Type string

	// DefaultValue is the default value
	DefaultValue interface{}

	// CurrentValue is the currently configured value
	CurrentValue interface{}

	// Constraints define valid values
	Constraints map[string]interface{}
	
	// MinValue is the minimum allowed value (for numeric types)
	MinValue interface{}
	
	// MaxValue is the maximum allowed value (for numeric types)
	MaxValue interface{}
	
	// AllowedValues is the list of allowed values (for enum types)
	AllowedValues []interface{}
}

// PolicyConfiguration represents a specific configuration of a policy
type PolicyConfiguration struct {
	// PolicyID references the policy being configured
	PolicyID string

	// Priority determines this configuration's precedence
	Priority int

	// ParameterValues contains the configured parameter values
	ParameterValues map[string]interface{}

	// Enabled indicates if this configuration is active
	Enabled bool
}

// PolicyEngine manages and evaluates scheduling policies
type PolicyEngine struct {
	// Mutex for protecting concurrent access
	mu sync.RWMutex

	// Policies is a map of policy ID to policy
	Policies map[string]*SchedulingPolicy

	// ActiveConfigurations is a map of policy ID to configuration
	ActiveConfigurations map[string]*PolicyConfiguration

	// TypeIndex indexes policies by type
	TypeIndex map[PolicyType]map[string]*SchedulingPolicy

	// ExpressionCompiler compiles policy expressions
	ExpressionCompiler ExpressionCompiler

	// VersionManager manages policy versions
	VersionManager *PolicyVersionManager
}

// NewPolicyEngine creates a new policy engine
func NewPolicyEngine() *PolicyEngine {
	return &PolicyEngine{
		Policies:             make(map[string]*SchedulingPolicy),
		ActiveConfigurations: make(map[string]*PolicyConfiguration),
		TypeIndex:            make(map[PolicyType]map[string]*SchedulingPolicy),
		ExpressionCompiler:   &DefaultExpressionCompiler{},
		VersionManager:       NewPolicyVersionManager(),
	}
}

// RegisterPolicy registers a new policy
func (e *PolicyEngine) RegisterPolicy(policy *SchedulingPolicy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if policy.ID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	if _, exists := e.Policies[policy.ID]; exists {
		return fmt.Errorf("policy with ID %s already exists", policy.ID)
	}

	// Initialize type index if needed
	if _, exists := e.TypeIndex[policy.Type]; !exists {
		e.TypeIndex[policy.Type] = make(map[string]*SchedulingPolicy)
	}

	// Store the policy
	e.Policies[policy.ID] = policy
	e.TypeIndex[policy.Type][policy.ID] = policy

	// Register with version manager
	_, err := e.VersionManager.SaveVersion(policy, policy.CreatedBy, "Initial registration")
	if err != nil {
		// Rollback on error
		delete(e.Policies, policy.ID)
		delete(e.TypeIndex[policy.Type], policy.ID)
		return fmt.Errorf("failed to save initial version: %v", err)
	}

	return nil
}

// UnregisterPolicy removes a policy
func (e *PolicyEngine) UnregisterPolicy(policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	policy, exists := e.Policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Remove from active configurations
	delete(e.ActiveConfigurations, policyID)

	// Remove from type index
	delete(e.TypeIndex[policy.Type], policyID)

	// Remove the policy
	delete(e.Policies, policyID)

	return nil
}

// GetPolicy returns a policy by ID
func (e *PolicyEngine) GetPolicy(policyID string) (*SchedulingPolicy, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policy, exists := e.Policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy with ID %s not found", policyID)
	}

	return policy, nil
}

// ListPolicies returns all policies
func (e *PolicyEngine) ListPolicies() []*SchedulingPolicy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policies := make([]*SchedulingPolicy, 0, len(e.Policies))
	for _, policy := range e.Policies {
		policies = append(policies, policy)
	}

	return policies
}

// ListPoliciesByType returns policies of a specific type
func (e *PolicyEngine) ListPoliciesByType(policyType PolicyType) []*SchedulingPolicy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	typeMap, exists := e.TypeIndex[policyType]
	if !exists {
		return []*SchedulingPolicy{}
	}

	policies := make([]*SchedulingPolicy, 0, len(typeMap))
	for _, policy := range typeMap {
		policies = append(policies, policy)
	}

	return policies
}

// UpdatePolicy updates an existing policy
func (e *PolicyEngine) UpdatePolicy(policy *SchedulingPolicy, updatedBy, changeDesc string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.Policies[policy.ID]; !exists {
		return fmt.Errorf("policy with ID %s not found", policy.ID)
	}

	// Store the old type for index update
	oldPolicy := e.Policies[policy.ID]
	oldType := oldPolicy.Type

	// Update type index if type changed
	if oldType != policy.Type {
		delete(e.TypeIndex[oldType], policy.ID)

		if _, exists := e.TypeIndex[policy.Type]; !exists {
			e.TypeIndex[policy.Type] = make(map[string]*SchedulingPolicy)
		}
	}

	// Update timestamps
	policy.UpdatedAt = time.Now()
	policy.UpdatedBy = updatedBy

	// Save version first
	newVersion, err := e.VersionManager.SaveVersion(policy, updatedBy, changeDesc)
	if err != nil {
		return fmt.Errorf("failed to save new version: %v", err)
	}

	// Update the policy
	policy.Version = newVersion
	e.Policies[policy.ID] = policy
	e.TypeIndex[policy.Type][policy.ID] = policy

	return nil
}

// ActivatePolicy activates a policy with a specific configuration
func (e *PolicyEngine) ActivatePolicy(policyID string, config *PolicyConfiguration) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.Policies[policyID]; !exists {
		return fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Update policy status
	policy := e.Policies[policyID]
	policy.Status = PolicyStatusActive

	// Activate configuration
	config.Enabled = true
	e.ActiveConfigurations[policyID] = config

	return nil
}

// DeactivatePolicy deactivates a policy
func (e *PolicyEngine) DeactivatePolicy(policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.Policies[policyID]; !exists {
		return fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Update policy status
	policy := e.Policies[policyID]
	policy.Status = PolicyStatusDisabled

	// Deactivate configuration if it exists
	if config, exists := e.ActiveConfigurations[policyID]; exists {
		config.Enabled = false
	}

	return nil
}

// EvaluatePlacementPolicies evaluates placement policies for a VM
func (e *PolicyEngine) EvaluatePlacementPolicies(
	ctx context.Context,
	vm map[string]interface{},
	candidateNodes []map[string]interface{}) ([]map[string]interface{}, error) {

	e.mu.RLock()
	placementPolicies := e.TypeIndex[PlacementPolicy]
	activePolicyConfigs := make(map[string]*PolicyConfiguration)

	// Copy active configurations to avoid holding lock during evaluation
	for id, config := range e.ActiveConfigurations {
		if policy, exists := placementPolicies[id]; exists && policy.Status == PolicyStatusActive && config.Enabled {
			activePolicyConfigs[id] = config
		}
	}
	e.mu.RUnlock()

	// Filter and score candidate nodes
	var filteredCandidates []map[string]interface{}
	nodeScores := make(map[int]float64)

	for _, node := range candidateNodes {
		// Create evaluation context for this node
		evalCtx := NewEvaluationContext()
		evalCtx.InitializeForVM(vm, nil, node)

		// Evaluate all active placement policies
		for policyID, config := range activePolicyConfigs {
			policy, _ := e.GetPolicy(policyID) // Safe because we checked existence earlier

			// Set policy parameters in context
			for paramName, paramValue := range config.ParameterValues {
				evalCtx.SetVariable(fmt.Sprintf("param.%s", paramName), paramValue)
			}

			// Evaluate policy rules in priority order
			for _, rule := range policy.Rules {
				// Evaluate rule condition
				result, err := rule.Condition.Evaluate(evalCtx)
				if err != nil {
					return nil, fmt.Errorf("error evaluating rule condition: %v", err)
				}

				// Skip rule if condition is false
				conditionMet, ok := result.(bool)
				if !ok || !conditionMet {
					continue
				}

				// Execute rule actions
				for _, action := range rule.Actions {
					if err := action.Execute(ctx, evalCtx); err != nil {
						return nil, fmt.Errorf("error executing rule action: %v", err)
					}

					// Check if node was filtered
					if filtered, _ := evalCtx.IsFiltered(); filtered && rule.IsHardConstraint {
						// Skip to next node if this is a hard constraint
						break
					}
				}

				// Break if node was filtered by a hard constraint
				if filtered, _ := evalCtx.IsFiltered(); filtered && rule.IsHardConstraint {
					break
				}
			}
		}

		// Check if node passed all filters
		filtered, _ := evalCtx.IsFiltered()
		if !filtered {
			filteredCandidates = append(filteredCandidates, node)
			nodeScores[len(filteredCandidates)-1] = evalCtx.GetScore()
		}
	}

	// Sort candidates by score (descending)
	for i := 0; i < len(filteredCandidates)-1; i++ {
		for j := i + 1; j < len(filteredCandidates); j++ {
			if nodeScores[i] < nodeScores[j] {
				// Swap nodes
				filteredCandidates[i], filteredCandidates[j] = filteredCandidates[j], filteredCandidates[i]
				// Swap scores
				nodeScores[i], nodeScores[j] = nodeScores[j], nodeScores[i]
			}
		}
	}

	return filteredCandidates, nil
}

// PolicyVersionManager manages policy versions
type PolicyVersionManager struct {
	// Mutex for protecting concurrent access
	mu sync.RWMutex

	// Versions stores all policy versions
	Versions map[string]map[string]*PolicyVersion
}

// PolicyVersion represents a versioned policy
type PolicyVersion struct {
	// PolicyID is the ID of the policy
	PolicyID string

	// Version is the version identifier
	Version string

	// Policy is the policy at this version
	Policy *SchedulingPolicy

	// CreatedAt is when this version was created
	CreatedAt time.Time

	// CreatedBy is who created this version
	CreatedBy string

	// ChangeDescription explains the changes in this version
	ChangeDescription string
}

// NewPolicyVersionManager creates a new policy version manager
func NewPolicyVersionManager() *PolicyVersionManager {
	return &PolicyVersionManager{
		Versions: make(map[string]map[string]*PolicyVersion),
	}
}

// SaveVersion saves a new policy version
func (m *PolicyVersionManager) SaveVersion(policy *SchedulingPolicy, author, changeDesc string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Initialize version map for this policy if needed
	if _, exists := m.Versions[policy.ID]; !exists {
		m.Versions[policy.ID] = make(map[string]*PolicyVersion)
	}

	// Generate a new version ID (using timestamp for simplicity)
	newVersion := fmt.Sprintf("v%d", time.Now().UnixNano())

	// Create a deep copy of the policy for versioning
	policyCopy := *policy

	// Store the version
	m.Versions[policy.ID][newVersion] = &PolicyVersion{
		PolicyID:          policy.ID,
		Version:           newVersion,
		Policy:            &policyCopy,
		CreatedAt:         time.Now(),
		CreatedBy:         author,
		ChangeDescription: changeDesc,
	}

	return newVersion, nil
}

// GetVersion retrieves a specific policy version
func (m *PolicyVersionManager) GetVersion(policyID, version string) (*PolicyVersion, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check if policy exists
	versionMap, exists := m.Versions[policyID]
	if !exists {
		return nil, fmt.Errorf("no versions found for policy %s", policyID)
	}

	// Check if version exists
	policyVersion, exists := versionMap[version]
	if !exists {
		return nil, fmt.Errorf("version %s not found for policy %s", version, policyID)
	}

	return policyVersion, nil
}

// ListVersions lists all versions of a policy
func (m *PolicyVersionManager) ListVersions(policyID string) ([]*PolicyVersion, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check if policy exists
	versionMap, exists := m.Versions[policyID]
	if !exists {
		return nil, fmt.Errorf("no versions found for policy %s", policyID)
	}

	// Collect all versions
	versions := make([]*PolicyVersion, 0, len(versionMap))
	for _, version := range versionMap {
		versions = append(versions, version)
	}

	return versions, nil
}
