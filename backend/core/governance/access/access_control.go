package access

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Role represents an RBAC role
type Role struct {
	ID          string       `json:"id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Permissions []Permission `json:"permissions"`
	ParentRoles []string     `json:"parent_roles"` // For role hierarchy
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
}

// Permission represents a specific permission
type Permission struct {
	Resource string   `json:"resource"` // vm, network, storage, user
	Actions  []string `json:"actions"`  // create, read, update, delete
	Scope    string   `json:"scope"`    // tenant, project, global
}

// Attribute represents an ABAC attribute
type Attribute struct {
	Name  string      `json:"name"`
	Value interface{} `json:"value"`
	Type  string      `json:"type"` // string, number, boolean, time
}

// AccessRequest represents an access control request
type AccessRequest struct {
	SubjectID   string              `json:"subject_id"`   // User or service ID
	SubjectType string              `json:"subject_type"` // user, service
	Action      string              `json:"action"`
	Resource    string              `json:"resource"`
	ResourceID  string              `json:"resource_id"`
	Context     *AccessContext      `json:"context"`
	Attributes  map[string]Attribute `json:"attributes"`
}

// AccessContext provides contextual information for access decisions
type AccessContext struct {
	Timestamp    time.Time         `json:"timestamp"`
	IPAddress    string            `json:"ip_address"`
	Location     string            `json:"location"`
	DeviceType   string            `json:"device_type"`
	RiskScore    float64           `json:"risk_score"` // 0-100
	TenantID     string            `json:"tenant_id"`
	ProjectID    string            `json:"project_id"`
	Environment  string            `json:"environment"` // dev, staging, prod
	Metadata     map[string]string `json:"metadata"`
}

// AccessDecision represents an access control decision
type AccessDecision struct {
	Allowed    bool          `json:"allowed"`
	Reasons    []string      `json:"reasons"`
	Decision   string        `json:"decision"` // allow, deny, conditional
	Conditions []Condition   `json:"conditions"`
	EvaluatedBy string       `json:"evaluated_by"` // rbac, abac, policy
	Timestamp  time.Time     `json:"timestamp"`
	EvaluationTime time.Duration `json:"evaluation_time"`
}

// Condition represents a conditional access requirement
type Condition struct {
	Type        string `json:"type"` // mfa-required, ip-allowlist, time-based
	Description string `json:"description"`
	Met         bool   `json:"met"`
}

// AccessController implements RBAC and ABAC
type AccessController struct {
	mu                     sync.RWMutex
	roles                  map[string]*Role
	userRoles              map[string][]string // UserID -> RoleIDs
	rbacEnabled            bool
	abacEnabled            bool
	leastPrivilege         bool
	separationOfDuties     bool
	roleHierarchyEnabled   bool
	dynamicAccessDecisions bool
	contextualAttributes   []string
	auditDecisions         bool
	metrics                *AccessMetrics
}

// AccessMetrics tracks access control metrics
type AccessMetrics struct {
	mu                    sync.RWMutex
	TotalRequests         int64
	AllowedRequests       int64
	DeniedRequests        int64
	ConditionalRequests   int64
	RBACDecisions         int64
	ABACDecisions         int64
	AverageDecisionTime   time.Duration
	SeparationViolations  int64
}

// NewAccessController creates a new access controller
func NewAccessController(rbacEnabled, abacEnabled, leastPrivilege, separationOfDuties bool) *AccessController {
	ac := &AccessController{
		roles:                  make(map[string]*Role),
		userRoles:              make(map[string][]string),
		rbacEnabled:            rbacEnabled,
		abacEnabled:            abacEnabled,
		leastPrivilege:         leastPrivilege,
		separationOfDuties:     separationOfDuties,
		roleHierarchyEnabled:   true,
		dynamicAccessDecisions: true,
		contextualAttributes:   []string{"time", "location", "device", "risk-score"},
		auditDecisions:         true,
		metrics: &AccessMetrics{},
	}

	ac.initializeDefaultRoles()
	return ac
}

// initializeDefaultRoles initializes default RBAC roles
func (ac *AccessController) initializeDefaultRoles() {
	defaultRoles := []*Role{
		{
			ID:          "admin",
			Name:        "Administrator",
			Description: "Full system access",
			Permissions: []Permission{
				{Resource: "*", Actions: []string{"*"}, Scope: "global"},
			},
			ParentRoles: []string{},
		},
		{
			ID:          "operator",
			Name:        "Operator",
			Description: "VM and network operations",
			Permissions: []Permission{
				{Resource: "vm", Actions: []string{"create", "read", "update", "delete", "start", "stop"}, Scope: "tenant"},
				{Resource: "network", Actions: []string{"create", "read", "update", "delete"}, Scope: "tenant"},
				{Resource: "storage", Actions: []string{"create", "read", "update"}, Scope: "tenant"},
			},
			ParentRoles: []string{},
		},
		{
			ID:          "developer",
			Name:        "Developer",
			Description: "Development access",
			Permissions: []Permission{
				{Resource: "vm", Actions: []string{"create", "read", "start", "stop"}, Scope: "project"},
				{Resource: "network", Actions: []string{"read"}, Scope: "project"},
				{Resource: "storage", Actions: []string{"read"}, Scope: "project"},
			},
			ParentRoles: []string{},
		},
		{
			ID:          "viewer",
			Name:        "Viewer",
			Description: "Read-only access",
			Permissions: []Permission{
				{Resource: "*", Actions: []string{"read"}, Scope: "tenant"},
			},
			ParentRoles: []string{},
		},
		{
			ID:          "auditor",
			Name:        "Auditor",
			Description: "Audit and compliance access",
			Permissions: []Permission{
				{Resource: "audit", Actions: []string{"read"}, Scope: "global"},
				{Resource: "compliance", Actions: []string{"read"}, Scope: "global"},
				{Resource: "policy", Actions: []string{"read"}, Scope: "global"},
			},
			ParentRoles: []string{},
		},
	}

	for _, role := range defaultRoles {
		role.CreatedAt = time.Now()
		role.UpdatedAt = time.Now()
		ac.roles[role.ID] = role
	}
}

// CreateRole creates a new role
func (ac *AccessController) CreateRole(ctx context.Context, role *Role) error {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	if role.ID == "" {
		role.ID = fmt.Sprintf("role-%d", time.Now().UnixNano())
	}

	role.CreatedAt = time.Now()
	role.UpdatedAt = time.Now()

	ac.roles[role.ID] = role
	return nil
}

// AssignRole assigns a role to a user
func (ac *AccessController) AssignRole(ctx context.Context, userID, roleID string) error {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	if _, exists := ac.roles[roleID]; !exists {
		return fmt.Errorf("role %s not found", roleID)
	}

	if _, exists := ac.userRoles[userID]; !exists {
		ac.userRoles[userID] = make([]string, 0)
	}

	// Check for role conflicts (separation of duties)
	if ac.separationOfDuties {
		if err := ac.checkSeparationOfDuties(userID, roleID); err != nil {
			ac.metrics.mu.Lock()
			ac.metrics.SeparationViolations++
			ac.metrics.mu.Unlock()
			return err
		}
	}

	ac.userRoles[userID] = append(ac.userRoles[userID], roleID)
	return nil
}

// checkSeparationOfDuties checks for conflicting role assignments
func (ac *AccessController) checkSeparationOfDuties(userID, newRoleID string) error {
	// Define conflicting role pairs
	conflicts := map[string][]string{
		"admin":    {"auditor"},      // Admin cannot be auditor
		"auditor":  {"admin", "operator"}, // Auditor cannot be admin or operator
		"developer": {"operator"},    // Developer cannot be operator (example)
	}

	existingRoles := ac.userRoles[userID]
	conflictingRoles, exists := conflicts[newRoleID]

	if !exists {
		return nil
	}

	for _, existingRole := range existingRoles {
		for _, conflictingRole := range conflictingRoles {
			if existingRole == conflictingRole {
				return fmt.Errorf("role conflict: cannot assign %s to user with %s role (separation of duties)", newRoleID, conflictingRole)
			}
		}
	}

	return nil
}

// CheckAccess checks if access should be granted
func (ac *AccessController) CheckAccess(ctx context.Context, request *AccessRequest) (*AccessDecision, error) {
	startTime := time.Now()

	decision := &AccessDecision{
		Reasons:    make([]string, 0),
		Conditions: make([]Condition, 0),
		Timestamp:  time.Now(),
	}

	// Try RBAC first
	if ac.rbacEnabled {
		rbacDecision := ac.checkRBAC(request)
		if rbacDecision.Allowed {
			decision.Allowed = true
			decision.Reasons = append(decision.Reasons, rbacDecision.Reasons...)
			decision.EvaluatedBy = "rbac"
			ac.metrics.mu.Lock()
			ac.metrics.RBACDecisions++
			ac.metrics.mu.Unlock()
		}
	}

	// Try ABAC if RBAC denied or ABAC is primary
	if !decision.Allowed && ac.abacEnabled {
		abacDecision := ac.checkABAC(request)
		if abacDecision.Allowed {
			decision.Allowed = true
			decision.Reasons = append(decision.Reasons, abacDecision.Reasons...)
			decision.EvaluatedBy = "abac"
			ac.metrics.mu.Lock()
			ac.metrics.ABACDecisions++
			ac.metrics.mu.Unlock()
		} else {
			decision.Reasons = append(decision.Reasons, abacDecision.Reasons...)
		}
	}

	// Apply contextual attributes for dynamic decisions
	if ac.dynamicAccessDecisions && decision.Allowed {
		conditions := ac.evaluateContextualConditions(request)
		decision.Conditions = conditions

		// Check if all conditions are met
		allConditionsMet := true
		for _, condition := range conditions {
			if !condition.Met {
				allConditionsMet = false
				break
			}
		}

		if !allConditionsMet {
			decision.Decision = "conditional"
			ac.metrics.mu.Lock()
			ac.metrics.ConditionalRequests++
			ac.metrics.mu.Unlock()
		} else {
			decision.Decision = "allow"
		}
	} else if decision.Allowed {
		decision.Decision = "allow"
	} else {
		decision.Decision = "deny"
		if len(decision.Reasons) == 0 {
			decision.Reasons = append(decision.Reasons, "Access denied by policy")
		}
	}

	evaluationTime := time.Since(startTime)
	decision.EvaluationTime = evaluationTime

	// Update metrics
	ac.updateMetrics(decision, evaluationTime)

	return decision, nil
}

// checkRBAC performs RBAC evaluation
func (ac *AccessController) checkRBAC(request *AccessRequest) *AccessDecision {
	decision := &AccessDecision{
		Reasons: make([]string, 0),
	}

	ac.mu.RLock()
	userRoles := ac.userRoles[request.SubjectID]
	ac.mu.RUnlock()

	if len(userRoles) == 0 {
		decision.Allowed = false
		decision.Reasons = append(decision.Reasons, "No roles assigned to user")
		return decision
	}

	// Check each role for required permission
	for _, roleID := range userRoles {
		ac.mu.RLock()
		role, exists := ac.roles[roleID]
		ac.mu.RUnlock()

		if !exists {
			continue
		}

		// Check role hierarchy (inherit parent permissions)
		if ac.roleHierarchyEnabled {
			if ac.checkRolePermissions(role, request, true) {
				decision.Allowed = true
				decision.Reasons = append(decision.Reasons, fmt.Sprintf("Permission granted by role: %s", role.Name))
				return decision
			}
		} else {
			if ac.checkRolePermissions(role, request, false) {
				decision.Allowed = true
				decision.Reasons = append(decision.Reasons, fmt.Sprintf("Permission granted by role: %s", role.Name))
				return decision
			}
		}
	}

	decision.Allowed = false
	decision.Reasons = append(decision.Reasons, "No matching RBAC permission found")
	return decision
}

// checkRolePermissions checks if role has required permission
func (ac *AccessController) checkRolePermissions(role *Role, request *AccessRequest, includeParents bool) bool {
	// Check direct permissions
	for _, perm := range role.Permissions {
		if ac.matchesPermission(perm, request) {
			return true
		}
	}

	// Check parent roles recursively
	if includeParents && len(role.ParentRoles) > 0 {
		for _, parentRoleID := range role.ParentRoles {
			ac.mu.RLock()
			parentRole, exists := ac.roles[parentRoleID]
			ac.mu.RUnlock()

			if exists && ac.checkRolePermissions(parentRole, request, true) {
				return true
			}
		}
	}

	return false
}

// matchesPermission checks if permission matches request
func (ac *AccessController) matchesPermission(perm Permission, request *AccessRequest) bool {
	// Check resource match
	if perm.Resource != "*" && perm.Resource != request.Resource {
		return false
	}

	// Check action match
	actionMatches := false
	for _, action := range perm.Actions {
		if action == "*" || action == request.Action {
			actionMatches = true
			break
		}
	}

	if !actionMatches {
		return false
	}

	// Check scope (simplified - in production, this would check tenant/project ownership)
	if perm.Scope == "global" {
		return true
	}

	if perm.Scope == "tenant" && request.Context.TenantID != "" {
		return true
	}

	if perm.Scope == "project" && request.Context.ProjectID != "" {
		return true
	}

	return false
}

// checkABAC performs ABAC evaluation
func (ac *AccessController) checkABAC(request *AccessRequest) *AccessDecision {
	decision := &AccessDecision{
		Reasons: make([]string, 0),
	}

	// ABAC rules based on attributes
	score := 0.0
	requiredScore := 3.0 // Need at least 3 points to allow

	// Rule 1: High-privilege resources require low risk score
	if request.Resource == "vm" && request.Action == "delete" {
		if request.Context.RiskScore < 30 {
			score += 2.0
			decision.Reasons = append(decision.Reasons, "Risk score acceptable for VM deletion")
		} else {
			decision.Reasons = append(decision.Reasons, fmt.Sprintf("Risk score too high for VM deletion: %.0f", request.Context.RiskScore))
		}
	}

	// Rule 2: Location-based access
	if request.Context.Location != "" {
		// Simplified location check
		score += 1.0
		decision.Reasons = append(decision.Reasons, "Location verified")
	}

	// Rule 3: Time-based access (business hours)
	hour := request.Context.Timestamp.Hour()
	if hour >= 8 && hour <= 18 {
		score += 1.0
		decision.Reasons = append(decision.Reasons, "Access during business hours")
	}

	// Rule 4: Device type
	if request.Context.DeviceType == "corporate-managed" {
		score += 1.0
		decision.Reasons = append(decision.Reasons, "Corporate-managed device")
	}

	decision.Allowed = score >= requiredScore
	if !decision.Allowed {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("ABAC score insufficient: %.1f/%.1f", score, requiredScore))
	}

	return decision
}

// evaluateContextualConditions evaluates contextual access conditions
func (ac *AccessController) evaluateContextualConditions(request *AccessRequest) []Condition {
	conditions := make([]Condition, 0)

	// Condition 1: MFA for high-risk operations
	if request.Action == "delete" || request.Action == "modify" {
		conditions = append(conditions, Condition{
			Type:        "mfa-required",
			Description: "Multi-factor authentication required for this operation",
			Met:         true, // Simplified - would check actual MFA status
		})
	}

	// Condition 2: IP allowlist for admin actions
	if request.Resource == "policy" || request.Resource == "compliance" {
		conditions = append(conditions, Condition{
			Type:        "ip-allowlist",
			Description: "IP address must be in corporate allowlist",
			Met:         true, // Simplified - would check actual IP
		})
	}

	// Condition 3: Time-based restrictions
	hour := request.Context.Timestamp.Hour()
	if hour < 6 || hour > 22 {
		conditions = append(conditions, Condition{
			Type:        "time-based",
			Description: "Access restricted outside business hours",
			Met:         false,
		})
	}

	// Condition 4: Risk score threshold
	if request.Context.RiskScore > 70 {
		conditions = append(conditions, Condition{
			Type:        "risk-score",
			Description: fmt.Sprintf("Risk score too high: %.0f", request.Context.RiskScore),
			Met:         false,
		})
	}

	return conditions
}

// updateMetrics updates access control metrics
func (ac *AccessController) updateMetrics(decision *AccessDecision, evaluationTime time.Duration) {
	ac.metrics.mu.Lock()
	defer ac.metrics.mu.Unlock()

	ac.metrics.TotalRequests++

	if decision.Allowed {
		ac.metrics.AllowedRequests++
	} else {
		ac.metrics.DeniedRequests++
	}

	// Update average decision time
	if ac.metrics.AverageDecisionTime == 0 {
		ac.metrics.AverageDecisionTime = evaluationTime
	} else {
		alpha := 0.1
		ac.metrics.AverageDecisionTime = time.Duration(
			float64(ac.metrics.AverageDecisionTime)*(1-alpha) +
				float64(evaluationTime)*alpha,
		)
	}
}

// GetUserRoles returns roles assigned to a user
func (ac *AccessController) GetUserRoles(userID string) []string {
	ac.mu.RLock()
	defer ac.mu.RUnlock()

	roles := make([]string, len(ac.userRoles[userID]))
	copy(roles, ac.userRoles[userID])
	return roles
}

// GetMetrics returns access control metrics
func (ac *AccessController) GetMetrics() *AccessMetrics {
	ac.metrics.mu.RLock()
	defer ac.metrics.mu.RUnlock()

	return &AccessMetrics{
		TotalRequests:        ac.metrics.TotalRequests,
		AllowedRequests:      ac.metrics.AllowedRequests,
		DeniedRequests:       ac.metrics.DeniedRequests,
		ConditionalRequests:  ac.metrics.ConditionalRequests,
		RBACDecisions:        ac.metrics.RBACDecisions,
		ABACDecisions:        ac.metrics.ABACDecisions,
		AverageDecisionTime:  ac.metrics.AverageDecisionTime,
		SeparationViolations: ac.metrics.SeparationViolations,
	}
}
