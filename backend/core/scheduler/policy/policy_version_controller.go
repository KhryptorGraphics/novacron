package policy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// PolicyVersionController manages policy versioning, including rollbacks and history
type PolicyVersionController struct {
	// VersionManager is the underlying version manager
	VersionManager *PolicyVersionManager

	// Engine is the policy engine this controller manages
	Engine *PolicyEngine

	// VersionStore handles persistence of policy versions
	VersionStore PolicyVersionStore

	// mutex protects internal state
	mutex sync.RWMutex

	// auditLog tracks changes to policies
	auditLog []*PolicyAuditEntry
}

// PolicyVersionStore defines the interface for storing policy versions
type PolicyVersionStore interface {
	// SaveVersion saves a policy version
	SaveVersion(ctx context.Context, version *PolicyVersion) error

	// GetVersion retrieves a policy version
	GetVersion(ctx context.Context, policyID, versionID string) (*PolicyVersion, error)

	// ListVersions lists all versions for a policy
	ListVersions(ctx context.Context, policyID string) ([]*PolicyVersion, error)

	// DeleteVersion deletes a version
	DeleteVersion(ctx context.Context, policyID, versionID string) error
}

// PolicyAuditEntry represents an entry in the policy change audit log
type PolicyAuditEntry struct {
	// Timestamp is when the change occurred
	Timestamp time.Time `json:"timestamp"`

	// User is who performed the change
	User string `json:"user"`

	// PolicyID is the ID of the affected policy
	PolicyID string `json:"policy_id"`

	// Action is what was done (create, update, rollback, etc.)
	Action string `json:"action"`

	// FromVersion is the version before the change (if applicable)
	FromVersion string `json:"from_version,omitempty"`

	// ToVersion is the version after the change
	ToVersion string `json:"to_version,omitempty"`

	// Description is a human-readable description of the change
	Description string `json:"description"`
}

// NewPolicyVersionController creates a new policy version controller
func NewPolicyVersionController(engine *PolicyEngine, store PolicyVersionStore) *PolicyVersionController {
	return &PolicyVersionController{
		VersionManager: engine.VersionManager,
		Engine:         engine,
		VersionStore:   store,
		auditLog:       make([]*PolicyAuditEntry, 0),
	}
}

// CreatePolicy creates a new policy with versioning
func (c *PolicyVersionController) CreatePolicy(
	ctx context.Context,
	policy *SchedulingPolicy,
	user, description string,
) (string, error) {
	// Register the policy with the engine
	if err := c.Engine.RegisterPolicy(policy); err != nil {
		return "", fmt.Errorf("failed to register policy: %w", err)
	}

	// The policy was already versioned during registration
	// Get the version ID
	versions, err := c.VersionManager.ListVersions(policy.ID)
	if err != nil {
		return "", fmt.Errorf("failed to get versions: %w", err)
	}

	if len(versions) == 0 {
		return "", errors.New("policy was registered but no version was created")
	}

	// Get the latest version
	latestVersion := versions[len(versions)-1]

	// Save to persistent store if one is configured
	if c.VersionStore != nil {
		if err := c.VersionStore.SaveVersion(ctx, latestVersion); err != nil {
			return "", fmt.Errorf("failed to save version: %w", err)
		}
	}

	// Add audit entry
	c.mutex.Lock()
	c.auditLog = append(c.auditLog, &PolicyAuditEntry{
		Timestamp:   time.Now(),
		User:        user,
		PolicyID:    policy.ID,
		Action:      "create",
		ToVersion:   latestVersion.Version,
		Description: description,
	})
	c.mutex.Unlock()

	return latestVersion.Version, nil
}

// UpdatePolicy updates a policy with versioning
func (c *PolicyVersionController) UpdatePolicy(
	ctx context.Context,
	policy *SchedulingPolicy,
	user, description string,
) (string, error) {
	// Get the current version
	currentVersion := policy.Version

	// Update the policy
	if err := c.Engine.UpdatePolicy(policy, user, description); err != nil {
		return "", fmt.Errorf("failed to update policy: %w", err)
	}

	// Get the new version
	versions, err := c.VersionManager.ListVersions(policy.ID)
	if err != nil {
		return "", fmt.Errorf("failed to get versions: %w", err)
	}

	if len(versions) == 0 {
		return "", errors.New("policy was updated but no version was created")
	}

	// Get the latest version
	latestVersion := versions[len(versions)-1]

	// Save to persistent store if one is configured
	if c.VersionStore != nil {
		if err := c.VersionStore.SaveVersion(ctx, latestVersion); err != nil {
			return "", fmt.Errorf("failed to save version: %w", err)
		}
	}

	// Add audit entry
	c.mutex.Lock()
	c.auditLog = append(c.auditLog, &PolicyAuditEntry{
		Timestamp:   time.Now(),
		User:        user,
		PolicyID:    policy.ID,
		Action:      "update",
		FromVersion: currentVersion,
		ToVersion:   latestVersion.Version,
		Description: description,
	})
	c.mutex.Unlock()

	return latestVersion.Version, nil
}

// RollbackPolicy rolls back a policy to a previous version
func (c *PolicyVersionController) RollbackPolicy(
	ctx context.Context,
	policyID, versionID, user, description string,
) error {
	// Get the policy
	policy, err := c.Engine.GetPolicy(policyID)
	if err != nil {
		return fmt.Errorf("failed to get policy: %w", err)
	}

	// Get the target version
	targetVersion, err := c.VersionManager.GetVersion(policyID, versionID)
	if err != nil {
		// Try the persistent store if the version is not in memory
		if c.VersionStore != nil {
			targetVersion, err = c.VersionStore.GetVersion(ctx, policyID, versionID)
			if err != nil {
				return fmt.Errorf("failed to get target version: %w", err)
			}
		} else {
			return fmt.Errorf("failed to get target version: %w", err)
		}
	}

	// The current version before rollback
	currentVersion := policy.Version

	// Use the policy from the target version
	rollbackPolicy := targetVersion.Policy

	// Update with the current timestamp
	rollbackPolicy.UpdatedAt = time.Now()
	rollbackPolicy.UpdatedBy = user

	// Update the policy (which will create a new version with the old content)
	if err := c.Engine.UpdatePolicy(rollbackPolicy, user,
		fmt.Sprintf("Rollback to version %s: %s", versionID, description)); err != nil {
		return fmt.Errorf("failed to rollback policy: %w", err)
	}

	// Get the new version (which is the result of the rollback)
	versions, err := c.VersionManager.ListVersions(policyID)
	if err != nil {
		return fmt.Errorf("failed to get versions: %w", err)
	}

	if len(versions) == 0 {
		return errors.New("policy was rolled back but no version was created")
	}

	// Get the latest version (the rollback result)
	latestVersion := versions[len(versions)-1]

	// Save to persistent store if one is configured
	if c.VersionStore != nil {
		if err := c.VersionStore.SaveVersion(ctx, latestVersion); err != nil {
			return fmt.Errorf("failed to save version: %w", err)
		}
	}

	// Add audit entry
	c.mutex.Lock()
	c.auditLog = append(c.auditLog, &PolicyAuditEntry{
		Timestamp:   time.Now(),
		User:        user,
		PolicyID:    policyID,
		Action:      "rollback",
		FromVersion: currentVersion,
		ToVersion:   latestVersion.Version,
		Description: fmt.Sprintf("Rolled back to previous version %s: %s", versionID, description),
	})
	c.mutex.Unlock()

	return nil
}

// GetVersionHistory gets the version history for a policy
func (c *PolicyVersionController) GetVersionHistory(
	ctx context.Context,
	policyID string,
) ([]*PolicyVersion, error) {
	// Try getting from in-memory version manager first
	versions, err := c.VersionManager.ListVersions(policyID)
	if err != nil {
		return nil, fmt.Errorf("failed to get versions: %w", err)
	}

	// If we have a persistent store, also check there for older versions
	if c.VersionStore != nil {
		// Get versions from the store
		storeVersions, err := c.VersionStore.ListVersions(ctx, policyID)
		if err != nil {
			return nil, fmt.Errorf("failed to get versions from store: %w", err)
		}

		// Create a map of existing versions to avoid duplicates
		existingVersions := make(map[string]bool)
		for _, v := range versions {
			existingVersions[v.Version] = true
		}

		// Add versions from the store that aren't already in memory
		for _, v := range storeVersions {
			if !existingVersions[v.Version] {
				versions = append(versions, v)
			}
		}
	}

	// Sort versions by creation time
	sort.Slice(versions, func(i, j int) bool {
		return versions[i].CreatedAt.Before(versions[j].CreatedAt)
	})

	return versions, nil
}

// GetAuditLog gets the audit log entries for a policy
func (c *PolicyVersionController) GetAuditLog(policyID string) []*PolicyAuditEntry {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if policyID == "" {
		// Return all entries
		return c.auditLog
	}

	// Filter entries for the specified policy
	entries := make([]*PolicyAuditEntry, 0)
	for _, entry := range c.auditLog {
		if entry.PolicyID == policyID {
			entries = append(entries, entry)
		}
	}

	return entries
}

// CompareVersions compares two versions of a policy
func (c *PolicyVersionController) CompareVersions(
	ctx context.Context,
	policyID, version1, version2 string,
) (*PolicyVersionDiff, error) {
	// Get the two versions
	v1, err := c.getVersionFromAnySource(ctx, policyID, version1)
	if err != nil {
		return nil, fmt.Errorf("failed to get version %s: %w", version1, err)
	}

	v2, err := c.getVersionFromAnySource(ctx, policyID, version2)
	if err != nil {
		return nil, fmt.Errorf("failed to get version %s: %w", version2, err)
	}

	// Compare the policies
	diff := &PolicyVersionDiff{
		PolicyID:       policyID,
		Version1:       version1,
		Version2:       version2,
		CreatedAt1:     v1.CreatedAt,
		CreatedAt2:     v2.CreatedAt,
		CreatedBy1:     v1.CreatedBy,
		CreatedBy2:     v2.CreatedBy,
		Description1:   v1.ChangeDescription,
		Description2:   v2.ChangeDescription,
		ChangedRules:   make([]RuleDiff, 0),
		ChangedTargets: make([]TargetDiff, 0),
		ChangedParams:  make([]ParamDiff, 0),
	}

	// Compare policy attributes
	if v1.Policy.Name != v2.Policy.Name {
		diff.NameChanged = true
		diff.OldName = v1.Policy.Name
		diff.NewName = v2.Policy.Name
	}

	if v1.Policy.Description != v2.Policy.Description {
		diff.DescriptionChanged = true
		diff.OldDescription = v1.Policy.Description
		diff.NewDescription = v2.Policy.Description
	}

	if v1.Policy.Type != v2.Policy.Type {
		diff.TypeChanged = true
		diff.OldType = string(v1.Policy.Type)
		diff.NewType = string(v2.Policy.Type)
	}

	// Compare rules
	diff.ChangedRules = compareRules(v1.Policy.Rules, v2.Policy.Rules)

	// Compare target selectors
	oldTarget := "nil"
	newTarget := "nil"
	if v1.Policy.TargetSelector != nil {
		oldTarget = v1.Policy.TargetSelector.String()
	}
	if v2.Policy.TargetSelector != nil {
		newTarget = v2.Policy.TargetSelector.String()
	}
	if oldTarget != newTarget {
		diff.TargetSelectorChanged = true
		diff.OldTargetSelector = oldTarget
		diff.NewTargetSelector = newTarget
	}

	// Compare parameters
	diff.ChangedParams = compareParameters(v1.Policy.Parameters, v2.Policy.Parameters)

	return diff, nil
}

// PolicyVersionDiff represents the differences between two policy versions
type PolicyVersionDiff struct {
	// Basic information
	PolicyID     string    `json:"policy_id"`
	Version1     string    `json:"version1"`
	Version2     string    `json:"version2"`
	CreatedAt1   time.Time `json:"created_at1"`
	CreatedAt2   time.Time `json:"created_at2"`
	CreatedBy1   string    `json:"created_by1"`
	CreatedBy2   string    `json:"created_by2"`
	Description1 string    `json:"description1"`
	Description2 string    `json:"description2"`

	// Basic attribute changes
	NameChanged        bool   `json:"name_changed"`
	OldName            string `json:"old_name,omitempty"`
	NewName            string `json:"new_name,omitempty"`
	DescriptionChanged bool   `json:"description_changed"`
	OldDescription     string `json:"old_description,omitempty"`
	NewDescription     string `json:"new_description,omitempty"`
	TypeChanged        bool   `json:"type_changed"`
	OldType            string `json:"old_type,omitempty"`
	NewType            string `json:"new_type,omitempty"`

	// Target selector changes
	TargetSelectorChanged bool   `json:"target_selector_changed"`
	OldTargetSelector     string `json:"old_target_selector,omitempty"`
	NewTargetSelector     string `json:"new_target_selector,omitempty"`

	// Rule changes
	ChangedRules []RuleDiff `json:"changed_rules"`

	// Target changes
	ChangedTargets []TargetDiff `json:"changed_targets"`

	// Parameter changes
	ChangedParams []ParamDiff `json:"changed_params"`
}

// RuleDiff represents the difference between two rule versions
type RuleDiff struct {
	// RuleID is the ID of the rule
	RuleID string `json:"rule_id"`

	// ChangeType indicates how the rule changed (added, removed, modified)
	ChangeType string `json:"change_type"`

	// OldRule is the old version of the rule
	OldRule *PolicyRule `json:"old_rule,omitempty"`

	// NewRule is the new version of the rule
	NewRule *PolicyRule `json:"new_rule,omitempty"`

	// Specific changes
	NameChanged           bool   `json:"name_changed"`
	OldName               string `json:"old_name,omitempty"`
	NewName               string `json:"new_name,omitempty"`
	DescriptionChanged    bool   `json:"description_changed"`
	OldDescription        string `json:"old_description,omitempty"`
	NewDescription        string `json:"new_description,omitempty"`
	PriorityChanged       bool   `json:"priority_changed"`
	OldPriority           int    `json:"old_priority,omitempty"`
	NewPriority           int    `json:"new_priority,omitempty"`
	WeightChanged         bool   `json:"weight_changed"`
	OldWeight             int    `json:"old_weight,omitempty"`
	NewWeight             int    `json:"new_weight,omitempty"`
	ConditionChanged      bool   `json:"condition_changed"`
	OldCondition          string `json:"old_condition,omitempty"`
	NewCondition          string `json:"new_condition,omitempty"`
	ActionsChanged        bool   `json:"actions_changed"`
	ActionsDiff           string `json:"actions_diff,omitempty"`
	HardConstraintChanged bool   `json:"hard_constraint_changed"`
	OldHardConstraint     bool   `json:"old_hard_constraint,omitempty"`
	NewHardConstraint     bool   `json:"new_hard_constraint,omitempty"`
}

// TargetDiff represents the difference between two target versions
type TargetDiff struct {
	// TargetID is the ID of the target
	TargetID string `json:"target_id"`

	// ChangeType indicates how the target changed (added, removed, modified)
	ChangeType string `json:"change_type"`

	// Changes contains specific changes to the target
	Changes map[string]interface{} `json:"changes,omitempty"`
}

// ParamDiff represents the difference between two parameter versions
type ParamDiff struct {
	// ParamName is the name of the parameter
	ParamName string `json:"param_name"`

	// ChangeType indicates how the parameter changed (added, removed, modified)
	ChangeType string `json:"change_type"`

	// OldParam is the old version of the parameter
	OldParam *PolicyParameter `json:"old_param,omitempty"`

	// NewParam is the new version of the parameter
	NewParam *PolicyParameter `json:"new_param,omitempty"`

	// Specific changes
	DescriptionChanged  bool                   `json:"description_changed"`
	OldDescription      string                 `json:"old_description,omitempty"`
	NewDescription      string                 `json:"new_description,omitempty"`
	TypeChanged         bool                   `json:"type_changed"`
	OldType             string                 `json:"old_type,omitempty"`
	NewType             string                 `json:"new_type,omitempty"`
	DefaultValueChanged bool                   `json:"default_value_changed"`
	OldDefaultValue     interface{}            `json:"old_default_value,omitempty"`
	NewDefaultValue     interface{}            `json:"new_default_value,omitempty"`
	ConstraintsChanged  bool                   `json:"constraints_changed"`
	OldConstraints      map[string]interface{} `json:"old_constraints,omitempty"`
	NewConstraints      map[string]interface{} `json:"new_constraints,omitempty"`
}

// getVersionFromAnySource gets a version from either in-memory or persistent store
func (c *PolicyVersionController) getVersionFromAnySource(
	ctx context.Context,
	policyID, versionID string,
) (*PolicyVersion, error) {
	// Try in-memory first
	version, err := c.VersionManager.GetVersion(policyID, versionID)
	if err == nil {
		return version, nil
	}

	// If not found and we have a persistent store, try there
	if c.VersionStore != nil {
		return c.VersionStore.GetVersion(ctx, policyID, versionID)
	}

	return nil, fmt.Errorf("version not found: %w", err)
}

// compareRules compares two sets of rules
func compareRules(oldRules, newRules []*PolicyRule) []RuleDiff {
	diffs := make([]RuleDiff, 0)

	// Create maps for faster lookups
	oldRuleMap := make(map[string]*PolicyRule)
	for _, rule := range oldRules {
		oldRuleMap[rule.ID] = rule
	}

	newRuleMap := make(map[string]*PolicyRule)
	for _, rule := range newRules {
		newRuleMap[rule.ID] = rule
	}

	// Find removed rules
	for id, oldRule := range oldRuleMap {
		if _, exists := newRuleMap[id]; !exists {
			diffs = append(diffs, RuleDiff{
				RuleID:     id,
				ChangeType: "removed",
				OldRule:    oldRule,
			})
		}
	}

	// Find added rules
	for id, newRule := range newRuleMap {
		if _, exists := oldRuleMap[id]; !exists {
			diffs = append(diffs, RuleDiff{
				RuleID:     id,
				ChangeType: "added",
				NewRule:    newRule,
			})
		}
	}

	// Find modified rules
	for id, oldRule := range oldRuleMap {
		if newRule, exists := newRuleMap[id]; exists {
			diff := RuleDiff{
				RuleID:     id,
				ChangeType: "modified",
				OldRule:    oldRule,
				NewRule:    newRule,
			}

			changed := false

			// Compare rule attributes
			if oldRule.Name != newRule.Name {
				diff.NameChanged = true
				diff.OldName = oldRule.Name
				diff.NewName = newRule.Name
				changed = true
			}

			if oldRule.Description != newRule.Description {
				diff.DescriptionChanged = true
				diff.OldDescription = oldRule.Description
				diff.NewDescription = newRule.Description
				changed = true
			}

			if oldRule.Priority != newRule.Priority {
				diff.PriorityChanged = true
				diff.OldPriority = oldRule.Priority
				diff.NewPriority = newRule.Priority
				changed = true
			}

			if oldRule.Weight != newRule.Weight {
				diff.WeightChanged = true
				diff.OldWeight = oldRule.Weight
				diff.NewWeight = newRule.Weight
				changed = true
			}

			if oldRule.IsHardConstraint != newRule.IsHardConstraint {
				diff.HardConstraintChanged = true
				diff.OldHardConstraint = oldRule.IsHardConstraint
				diff.NewHardConstraint = newRule.IsHardConstraint
				changed = true
			}

			// Compare conditions
			oldCondition := "nil"
			newCondition := "nil"
			if oldRule.Condition != nil {
				oldCondition = oldRule.Condition.String()
			}
			if newRule.Condition != nil {
				newCondition = newRule.Condition.String()
			}
			if oldCondition != newCondition {
				diff.ConditionChanged = true
				diff.OldCondition = oldCondition
				diff.NewCondition = newCondition
				changed = true
			}

			// Compare actions (simplified)
			if len(oldRule.Actions) != len(newRule.Actions) {
				diff.ActionsChanged = true
				diff.ActionsDiff = fmt.Sprintf("Action count changed from %d to %d",
					len(oldRule.Actions), len(newRule.Actions))
				changed = true
			} else {
				// This is a simplified comparison
				// A real implementation would do a deeper comparison of actions
				for i, oldAction := range oldRule.Actions {
					if oldAction.GetType() != newRule.Actions[i].GetType() {
						diff.ActionsChanged = true
						diff.ActionsDiff = fmt.Sprintf("Action at position %d changed from %s to %s",
							i, oldAction.GetType(), newRule.Actions[i].GetType())
						changed = true
						break
					}
				}
			}

			if changed {
				diffs = append(diffs, diff)
			}
		}
	}

	return diffs
}

// compareParameters compares two sets of parameters
func compareParameters(oldParams, newParams map[string]*PolicyParameter) []ParamDiff {
	diffs := make([]ParamDiff, 0)

	// Find removed parameters
	for name, oldParam := range oldParams {
		if _, exists := newParams[name]; !exists {
			diffs = append(diffs, ParamDiff{
				ParamName:  name,
				ChangeType: "removed",
				OldParam:   oldParam,
			})
		}
	}

	// Find added parameters
	for name, newParam := range newParams {
		if _, exists := oldParams[name]; !exists {
			diffs = append(diffs, ParamDiff{
				ParamName:  name,
				ChangeType: "added",
				NewParam:   newParam,
			})
		}
	}

	// Find modified parameters
	for name, oldParam := range oldParams {
		if newParam, exists := newParams[name]; exists {
			diff := ParamDiff{
				ParamName:  name,
				ChangeType: "modified",
				OldParam:   oldParam,
				NewParam:   newParam,
			}

			changed := false

			// Compare parameter attributes
			if oldParam.Description != newParam.Description {
				diff.DescriptionChanged = true
				diff.OldDescription = oldParam.Description
				diff.NewDescription = newParam.Description
				changed = true
			}

			if oldParam.Type != newParam.Type {
				diff.TypeChanged = true
				diff.OldType = oldParam.Type
				diff.NewType = newParam.Type
				changed = true
			}

			// A simple check for default value changes
			// A real implementation would do a deeper comparison
			if fmt.Sprintf("%v", oldParam.DefaultValue) != fmt.Sprintf("%v", newParam.DefaultValue) {
				diff.DefaultValueChanged = true
				diff.OldDefaultValue = oldParam.DefaultValue
				diff.NewDefaultValue = newParam.DefaultValue
				changed = true
			}

			// Check for constraint changes
			// This is a simplified comparison
			if len(oldParam.Constraints) != len(newParam.Constraints) {
				diff.ConstraintsChanged = true
				diff.OldConstraints = oldParam.Constraints
				diff.NewConstraints = newParam.Constraints
				changed = true
			} else {
				for k, v := range oldParam.Constraints {
					if newVal, exists := newParam.Constraints[k]; !exists ||
						fmt.Sprintf("%v", v) != fmt.Sprintf("%v", newVal) {
						diff.ConstraintsChanged = true
						diff.OldConstraints = oldParam.Constraints
						diff.NewConstraints = newParam.Constraints
						changed = true
						break
					}
				}
			}

			if changed {
				diffs = append(diffs, diff)
			}
		}
	}

	return diffs
}

// FileSystemVersionStore is an implementation of PolicyVersionStore that uses the file system
type FileSystemVersionStore struct {
	// BaseDir is the directory where versions are stored
	BaseDir string
}

// NewFileSystemVersionStore creates a new file system version store
func NewFileSystemVersionStore(baseDir string) (*FileSystemVersionStore, error) {
	// Create the base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	return &FileSystemVersionStore{
		BaseDir: baseDir,
	}, nil
}

// SaveVersion saves a policy version to the file system
func (s *FileSystemVersionStore) SaveVersion(ctx context.Context, version *PolicyVersion) error {
	// Create the policy directory if it doesn't exist
	policyDir := filepath.Join(s.BaseDir, version.PolicyID)
	if err := os.MkdirAll(policyDir, 0755); err != nil {
		return fmt.Errorf("failed to create policy directory: %w", err)
	}

	// Marshal the version to JSON
	data, err := json.MarshalIndent(version, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal version: %w", err)
	}

	// Write the version to a file
	versionPath := filepath.Join(policyDir, fmt.Sprintf("%s.json", version.Version))
	if err := ioutil.WriteFile(versionPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}

	return nil
}

// GetVersion retrieves a policy version from the file system
func (s *FileSystemVersionStore) GetVersion(ctx context.Context, policyID, versionID string) (*PolicyVersion, error) {
	// Construct the path to the version file
	versionPath := filepath.Join(s.BaseDir, policyID, fmt.Sprintf("%s.json", versionID))

	// Check if the file exists
	if _, err := os.Stat(versionPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("version not found: %w", err)
	}

	// Read the file
	data, err := ioutil.ReadFile(versionPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read version: %w", err)
	}

	// Unmarshal the JSON
	var version PolicyVersion
	if err := json.Unmarshal(data, &version); err != nil {
		return nil, fmt.Errorf("failed to unmarshal version: %w", err)
	}

	return &version, nil
}

// ListVersions lists all versions for a policy
func (s *FileSystemVersionStore) ListVersions(ctx context.Context, policyID string) ([]*PolicyVersion, error) {
	// Construct the path to the policy directory
	policyDir := filepath.Join(s.BaseDir, policyID)

	// Check if the directory exists
	if _, err := os.Stat(policyDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("policy not found: %w", err)
	}

	// Read the directory
	files, err := ioutil.ReadDir(policyDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read policy directory: %w", err)
	}

	// Filter for JSON files
	versions := make([]*PolicyVersion, 0)
	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		// Extract version ID from filename
		versionID := strings.TrimSuffix(file.Name(), ".json")

		// Get the version
		version, err := s.GetVersion(ctx, policyID, versionID)
		if err != nil {
			continue
		}

		versions = append(versions, version)
	}

	return versions, nil
}

// DeleteVersion deletes a version
func (s *FileSystemVersionStore) DeleteVersion(ctx context.Context, policyID, versionID string) error {
	// Construct the path to the version file
	versionPath := filepath.Join(s.BaseDir, policyID, fmt.Sprintf("%s.json", versionID))

	// Check if the file exists
	if _, err := os.Stat(versionPath); os.IsNotExist(err) {
		return fmt.Errorf("version not found: %w", err)
	}

	// Delete the file
	if err := os.Remove(versionPath); err != nil {
		return fmt.Errorf("failed to delete version: %w", err)
	}

	return nil
}

// InMemoryPolicyVersionStore is an in-memory implementation of PolicyVersionStore
type InMemoryPolicyVersionStore struct {
	versions map[string]map[string]*PolicyVersion // policyID -> versionID -> version
	mutex    sync.RWMutex
}

// NewInMemoryPolicyVersionStore creates a new in-memory policy version store
func NewInMemoryPolicyVersionStore() PolicyVersionStore {
	return &InMemoryPolicyVersionStore{
		versions: make(map[string]map[string]*PolicyVersion),
	}
}

// SaveVersion saves a policy version
func (s *InMemoryPolicyVersionStore) SaveVersion(ctx context.Context, version *PolicyVersion) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.versions[version.PolicyID] == nil {
		s.versions[version.PolicyID] = make(map[string]*PolicyVersion)
	}

	s.versions[version.PolicyID][version.VersionID] = version
	return nil
}

// GetVersion retrieves a policy version
func (s *InMemoryPolicyVersionStore) GetVersion(ctx context.Context, policyID, versionID string) (*PolicyVersion, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	policyVersions, exists := s.versions[policyID]
	if !exists {
		return nil, fmt.Errorf("policy not found: %s", policyID)
	}

	version, exists := policyVersions[versionID]
	if !exists {
		return nil, fmt.Errorf("version not found: %s for policy %s", versionID, policyID)
	}

	return version, nil
}

// ListVersions lists all versions for a policy
func (s *InMemoryPolicyVersionStore) ListVersions(ctx context.Context, policyID string) ([]*PolicyVersion, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	policyVersions, exists := s.versions[policyID]
	if !exists {
		return []*PolicyVersion{}, nil
	}

	versions := make([]*PolicyVersion, 0, len(policyVersions))
	for _, version := range policyVersions {
		versions = append(versions, version)
	}

	return versions, nil
}

// DeleteVersion deletes a version
func (s *InMemoryPolicyVersionStore) DeleteVersion(ctx context.Context, policyID, versionID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	policyVersions, exists := s.versions[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	if _, exists := policyVersions[versionID]; !exists {
		return fmt.Errorf("version not found: %s for policy %s", versionID, policyID)
	}

	delete(policyVersions, versionID)
	
	// Clean up empty policy maps
	if len(policyVersions) == 0 {
		delete(s.versions, policyID)
	}

	return nil
}
