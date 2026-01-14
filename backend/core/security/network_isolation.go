package security

import (
	"errors"
	"fmt"
	"net"
	"sync"
)

// NetworkPolicyType defines the type of network policy
type NetworkPolicyType string

const (
	// AllowPolicy allows traffic that matches the rules
	AllowPolicy NetworkPolicyType = "allow"

	// DenyPolicy denies traffic that matches the rules
	DenyPolicy NetworkPolicyType = "deny"
)

// NetworkDirection defines the direction of network traffic
type NetworkDirection string

const (
	// Ingress represents inbound traffic
	Ingress NetworkDirection = "ingress"

	// Egress represents outbound traffic
	Egress NetworkDirection = "egress"

	// Both represents both inbound and outbound traffic
	Both NetworkDirection = "both"
)

// NetworkRule defines a network rule
type NetworkRule struct {
	// ID is the unique identifier of the rule
	ID string `json:"id"`

	// Name is the human-readable name of the rule
	Name string `json:"name"`

	// Description describes the purpose of the rule
	Description string `json:"description"`

	// Type is the type of rule (allow/deny)
	Type NetworkPolicyType `json:"type"`

	// Direction is the direction of traffic (ingress/egress/both)
	Direction NetworkDirection `json:"direction"`

	// Protocol is the IP protocol (tcp/udp/icmp)
	Protocol string `json:"protocol"`

	// SourceCIDR is the source CIDR for the rule
	SourceCIDR []string `json:"source_cidr,omitempty"`

	// DestinationCIDR is the destination CIDR for the rule
	DestinationCIDR []string `json:"destination_cidr,omitempty"`

	// SourcePortRange is the source port range for the rule
	SourcePortRange []string `json:"source_port_range,omitempty"`

	// DestinationPortRange is the destination port range for the rule
	DestinationPortRange []string `json:"destination_port_range,omitempty"`

	// Priority is the priority of the rule (lower number = higher priority)
	Priority int `json:"priority"`

	// Enabled indicates if the rule is enabled
	Enabled bool `json:"enabled"`
}

// NetworkPolicy defines a network policy
type NetworkPolicy struct {
	// ID is the unique identifier of the policy
	ID string `json:"id"`

	// Name is the human-readable name of the policy
	Name string `json:"name"`

	// Description describes the purpose of the policy
	Description string `json:"description"`

	// TenantID is the ID of the tenant this policy belongs to
	TenantID string `json:"tenant_id"`

	// Rules are the rules for this policy
	Rules []*NetworkRule `json:"rules"`

	// DefaultAction is the default action for this policy
	DefaultAction NetworkPolicyType `json:"default_action"`

	// Enabled indicates if the policy is enabled
	Enabled bool `json:"enabled"`
}

// NetworkIsolationManager manages network isolation between tenants
type NetworkIsolationManager struct {
	// policies is a map of policy ID to policy
	policies map[string]*NetworkPolicy

	// tenantPolicies is a map of tenant ID to policy IDs
	tenantPolicies map[string][]string

	// mutex protects the policies and tenantPolicies maps
	mutex sync.RWMutex
}

// NewNetworkIsolationManager creates a new network isolation manager
func NewNetworkIsolationManager() *NetworkIsolationManager {
	return &NetworkIsolationManager{
		policies:       make(map[string]*NetworkPolicy),
		tenantPolicies: make(map[string][]string),
	}
}

// AddPolicy adds a policy to the manager
func (m *NetworkIsolationManager) AddPolicy(policy *NetworkPolicy) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy ID already exists
	if _, exists := m.policies[policy.ID]; exists {
		return fmt.Errorf("policy with ID %s already exists", policy.ID)
	}

	// Add policy
	m.policies[policy.ID] = policy

	// Add to tenant policies
	m.tenantPolicies[policy.TenantID] = append(m.tenantPolicies[policy.TenantID], policy.ID)

	return nil
}

// UpdatePolicy updates a policy
func (m *NetworkIsolationManager) UpdatePolicy(policy *NetworkPolicy) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy exists
	existingPolicy, exists := m.policies[policy.ID]
	if !exists {
		return fmt.Errorf("policy with ID %s does not exist", policy.ID)
	}

	// Check if tenant ID is being changed
	if existingPolicy.TenantID != policy.TenantID {
		// Remove from old tenant
		tenantPolicies := m.tenantPolicies[existingPolicy.TenantID]
		for i, id := range tenantPolicies {
			if id == policy.ID {
				m.tenantPolicies[existingPolicy.TenantID] = append(tenantPolicies[:i], tenantPolicies[i+1:]...)
				break
			}
		}

		// Add to new tenant
		m.tenantPolicies[policy.TenantID] = append(m.tenantPolicies[policy.TenantID], policy.ID)
	}

	// Update policy
	m.policies[policy.ID] = policy

	return nil
}

// RemovePolicy removes a policy
func (m *NetworkIsolationManager) RemovePolicy(policyID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy exists
	policy, exists := m.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s does not exist", policyID)
	}

	// Remove from tenant policies
	tenantPolicies := m.tenantPolicies[policy.TenantID]
	for i, id := range tenantPolicies {
		if id == policyID {
			m.tenantPolicies[policy.TenantID] = append(tenantPolicies[:i], tenantPolicies[i+1:]...)
			break
		}
	}

	// Remove policy
	delete(m.policies, policyID)

	return nil
}

// GetPolicy gets a policy by ID
func (m *NetworkIsolationManager) GetPolicy(policyID string) (*NetworkPolicy, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if policy exists
	policy, exists := m.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy with ID %s does not exist", policyID)
	}

	return policy, nil
}

// GetPoliciesForTenant gets all policies for a tenant
func (m *NetworkIsolationManager) GetPoliciesForTenant(tenantID string) ([]*NetworkPolicy, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Get policy IDs for tenant
	policyIDs, exists := m.tenantPolicies[tenantID]
	if !exists {
		return []*NetworkPolicy{}, nil
	}

	// Get policies
	policies := make([]*NetworkPolicy, 0, len(policyIDs))
	for _, policyID := range policyIDs {
		if policy, exists := m.policies[policyID]; exists {
			policies = append(policies, policy)
		}
	}

	return policies, nil
}

// AddRule adds a rule to a policy
func (m *NetworkIsolationManager) AddRule(policyID string, rule *NetworkRule) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy exists
	policy, exists := m.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s does not exist", policyID)
	}

	// Check if rule ID already exists
	for _, r := range policy.Rules {
		if r.ID == rule.ID {
			return fmt.Errorf("rule with ID %s already exists in policy %s", rule.ID, policyID)
		}
	}

	// Add rule
	policy.Rules = append(policy.Rules, rule)

	return nil
}

// UpdateRule updates a rule
func (m *NetworkIsolationManager) UpdateRule(policyID string, rule *NetworkRule) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy exists
	policy, exists := m.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s does not exist", policyID)
	}

	// Find and update rule
	for i, r := range policy.Rules {
		if r.ID == rule.ID {
			policy.Rules[i] = rule
			return nil
		}
	}

	return fmt.Errorf("rule with ID %s does not exist in policy %s", rule.ID, policyID)
}

// RemoveRule removes a rule from a policy
func (m *NetworkIsolationManager) RemoveRule(policyID, ruleID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if policy exists
	policy, exists := m.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s does not exist", policyID)
	}

	// Find and remove rule
	for i, r := range policy.Rules {
		if r.ID == ruleID {
			policy.Rules = append(policy.Rules[:i], policy.Rules[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("rule with ID %s does not exist in policy %s", ruleID, policyID)
}

// CheckConnectivity checks if a connection is allowed between two endpoints
func (m *NetworkIsolationManager) CheckConnectivity(sourceIP, destIP net.IP, sourcePort, destPort uint16, protocol string, tenantID string) (bool, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Get policies for tenant
	policyIDs, exists := m.tenantPolicies[tenantID]
	if !exists {
		// No policies for tenant, default to allow
		return true, nil
	}

	// Check policies
	for _, policyID := range policyIDs {
		policy, exists := m.policies[policyID]
		if !exists || !policy.Enabled {
			continue
		}

		// Check rules in priority order
		matchingRules := make([]*NetworkRule, 0, len(policy.Rules))
		for _, rule := range policy.Rules {
			if !rule.Enabled {
				continue
			}

			// Check if rule applies to this connection
			if ruleMatches(rule, sourceIP, destIP, sourcePort, destPort, protocol) {
				matchingRules = append(matchingRules, rule)
			}
		}

		// Sort by priority
		sortRulesByPriority(matchingRules)

		// Apply first matching rule
		if len(matchingRules) > 0 {
			rule := matchingRules[0]
			if rule.Type == AllowPolicy {
				return true, nil
			} else if rule.Type == DenyPolicy {
				return false, nil
			}
		}

		// No matching rules, apply default action
		if policy.DefaultAction == AllowPolicy {
			return true, nil
		} else if policy.DefaultAction == DenyPolicy {
			return false, nil
		}
	}

	// No matching policies, default to allow
	return true, nil
}

// ListNetworkPolicies lists all network policies
func (m *NetworkIsolationManager) ListNetworkPolicies() []*NetworkPolicy {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	policies := make([]*NetworkPolicy, 0, len(m.policies))
	for _, policy := range m.policies {
		policies = append(policies, policy)
	}

	return policies
}

// CreateDefaultPolicy creates a default policy for a tenant
func (m *NetworkIsolationManager) CreateDefaultPolicy(tenantID string) (*NetworkPolicy, error) {
	// Check if tenant already has policies
	m.mutex.RLock()
	if _, exists := m.tenantPolicies[tenantID]; exists && len(m.tenantPolicies[tenantID]) > 0 {
		m.mutex.RUnlock()
		return nil, errors.New("tenant already has policies")
	}
	m.mutex.RUnlock()

	// Create policy
	policy := &NetworkPolicy{
		ID:            fmt.Sprintf("default-%s", tenantID),
		Name:          "Default Tenant Policy",
		Description:   "Default network isolation policy for tenant",
		TenantID:      tenantID,
		Rules:         make([]*NetworkRule, 0),
		DefaultAction: AllowPolicy,
		Enabled:       true,
	}

	// Add intra-tenant allow rule
	intraRule := &NetworkRule{
		ID:              fmt.Sprintf("intra-tenant-%s", tenantID),
		Name:            "Allow Intra-Tenant Traffic",
		Description:     "Allow traffic within the tenant",
		Type:            AllowPolicy,
		Direction:       Both,
		Protocol:        "all",
		SourceCIDR:      []string{"0.0.0.0/0"},
		DestinationCIDR: []string{"0.0.0.0/0"},
		Priority:        1000,
		Enabled:         true,
	}
	policy.Rules = append(policy.Rules, intraRule)

	// Add policy
	err := m.AddPolicy(policy)
	if err != nil {
		return nil, err
	}

	return policy, nil
}

// Helper function to check if a rule matches
func ruleMatches(rule *NetworkRule, sourceIP, destIP net.IP, sourcePort, destPort uint16, protocol string) bool {
	// Check protocol
	if rule.Protocol != "all" && rule.Protocol != protocol {
		return false
	}

	// Check direction
	if rule.Direction != Both {
		// For now, we assume ingress/egress is handled by the caller
		// and matches the expected direction
	}

	// Check source and destination CIDRs
	if !matchesCIDR(sourceIP, rule.SourceCIDR) {
		return false
	}
	if !matchesCIDR(destIP, rule.DestinationCIDR) {
		return false
	}

	// Check source port ranges
	if !matchesPortRange(sourcePort, rule.SourcePortRange) {
		return false
	}

	// Check destination port ranges
	if !matchesPortRange(destPort, rule.DestinationPortRange) {
		return false
	}

	return true
}

// Helper function to check if an IP matches any CIDR
func matchesCIDR(ip net.IP, cidrs []string) bool {
	// If no CIDRs specified, allow all
	if len(cidrs) == 0 {
		return true
	}

	for _, cidrStr := range cidrs {
		_, cidr, err := net.ParseCIDR(cidrStr)
		if err != nil {
			continue
		}
		if cidr.Contains(ip) {
			return true
		}
	}
	return false
}

// Helper function to check if a port matches any port range
func matchesPortRange(port uint16, ranges []string) bool {
	// If no port ranges specified, allow all
	if len(ranges) == 0 {
		return true
	}

	for _, r := range ranges {
		// Parse port range
		var start, end uint16
		n, err := fmt.Sscanf(r, "%d-%d", &start, &end)
		if err != nil || n != 2 {
			// Try single port
			var singlePort uint16
			n, err := fmt.Sscanf(r, "%d", &singlePort)
			if err != nil || n != 1 {
				continue
			}
			if port == singlePort {
				return true
			}
		} else {
			// Check range
			if port >= start && port <= end {
				return true
			}
		}
	}
	return false
}

// Helper function to sort rules by priority
func sortRulesByPriority(rules []*NetworkRule) {
	// Use a simple bubble sort for now
	for i := 0; i < len(rules); i++ {
		for j := i + 1; j < len(rules); j++ {
			if rules[i].Priority > rules[j].Priority {
				rules[i], rules[j] = rules[j], rules[i]
			}
		}
	}
}
