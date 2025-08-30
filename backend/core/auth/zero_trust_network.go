package auth

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"regexp"
	"strings"
	"sync"
	"time"
)

// NetworkPolicy defines a zero-trust network policy
type NetworkPolicy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Enabled     bool                   `json:"enabled"`
	Priority    int                    `json:"priority"` // Higher priority = evaluated first
	Source      NetworkPolicySelector  `json:"source"`
	Destination NetworkPolicySelector  `json:"destination"`
	Action      NetworkPolicyAction    `json:"action"`
	Protocols   []NetworkProtocol      `json:"protocols"`
	Conditions  []NetworkPolicyCondition `json:"conditions,omitempty"`
	Metrics     NetworkPolicyMetrics   `json:"metrics,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	CreatedBy   string                 `json:"created_by,omitempty"`
}

// NetworkPolicySelector defines network policy selectors
type NetworkPolicySelector struct {
	TenantIDs    []string          `json:"tenant_ids,omitempty"`
	UserIDs      []string          `json:"user_ids,omitempty"`
	Roles        []string          `json:"roles,omitempty"`
	Services     []string          `json:"services,omitempty"`
	Namespaces   []string          `json:"namespaces,omitempty"`
	IPRanges     []string          `json:"ip_ranges,omitempty"`
	Domains      []string          `json:"domains,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
	Any          bool              `json:"any,omitempty"`
}

// NetworkPolicyAction defines policy actions
type NetworkPolicyAction string

const (
	NetworkPolicyAllow       NetworkPolicyAction = "allow"
	NetworkPolicyDeny        NetworkPolicyAction = "deny"
	NetworkPolicyLog         NetworkPolicyAction = "log"
	NetworkPolicyRateLimit   NetworkPolicyAction = "rate_limit"
	NetworkPolicyRequireMTLS NetworkPolicyAction = "require_mtls"
	NetworkPolicyQuarantine  NetworkPolicyAction = "quarantine"
)

// NetworkProtocol defines allowed network protocols
type NetworkProtocol struct {
	Protocol string `json:"protocol"` // TCP, UDP, HTTP, HTTPS, gRPC
	Ports    []int  `json:"ports,omitempty"`
	PortRanges []PortRange `json:"port_ranges,omitempty"`
}

// PortRange defines a range of ports
type PortRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// NetworkPolicyCondition defines additional policy conditions
type NetworkPolicyCondition struct {
	Type      string      `json:"type"` // time, geo, device_trust, etc.
	Operator  string      `json:"operator"` // equals, not_equals, in, not_in, etc.
	Value     interface{} `json:"value"`
}

// NetworkPolicyMetrics tracks policy usage
type NetworkPolicyMetrics struct {
	Connections  int64     `json:"connections"`
	BytesTransferred int64 `json:"bytes_transferred"`
	Violations   int64     `json:"violations"`
	LastAccess   time.Time `json:"last_access"`
	LastViolation time.Time `json:"last_violation"`
}

// NetworkConnection represents a network connection
type NetworkConnection struct {
	ID            string            `json:"id"`
	SourceIP      string            `json:"source_ip"`
	DestinationIP string            `json:"destination_ip"`
	SourcePort    int               `json:"source_port"`
	DestPort      int               `json:"dest_port"`
	Protocol      string            `json:"protocol"`
	UserID        string            `json:"user_id,omitempty"`
	TenantID      string            `json:"tenant_id,omitempty"`
	Service       string            `json:"service,omitempty"`
	Namespace     string            `json:"namespace,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
	TLSEnabled    bool              `json:"tls_enabled"`
	MTLSEnabled   bool              `json:"mtls_enabled"`
	EstablishedAt time.Time         `json:"established_at"`
	LastActivity  time.Time         `json:"last_activity"`
	BytesIn       int64             `json:"bytes_in"`
	BytesOut      int64             `json:"bytes_out"`
	TrustScore    int               `json:"trust_score"` // 0-100
}

// DeviceTrust represents device trust information
type DeviceTrust struct {
	DeviceID      string            `json:"device_id"`
	UserID        string            `json:"user_id"`
	TenantID      string            `json:"tenant_id"`
	DeviceType    string            `json:"device_type"`
	OS            string            `json:"os"`
	TrustLevel    int               `json:"trust_level"` // 0-100
	Compliance    DeviceCompliance  `json:"compliance"`
	Certificates  []string          `json:"certificates,omitempty"`
	Fingerprint   string            `json:"fingerprint"`
	LastSeen      time.Time         `json:"last_seen"`
	RegisteredAt  time.Time         `json:"registered_at"`
	Attributes    map[string]string `json:"attributes,omitempty"`
}

// DeviceCompliance represents device compliance status
type DeviceCompliance struct {
	Antivirus         bool      `json:"antivirus"`
	Firewall          bool      `json:"firewall"`
	Encryption        bool      `json:"encryption"`
	OSUpdated         bool      `json:"os_updated"`
	ScreenLock        bool      `json:"screen_lock"`
	Jailbroken        bool      `json:"jailbroken"`
	ComplianceScore   int       `json:"compliance_score"`
	LastCheck         time.Time `json:"last_check"`
}

// Microsegment represents a network microsegment
type Microsegment struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	TenantID    string            `json:"tenant_id"`
	Namespace   string            `json:"namespace,omitempty"`
	IPRanges    []string          `json:"ip_ranges"`
	Services    []string          `json:"services"`
	Labels      map[string]string `json:"labels,omitempty"`
	Isolation   IsolationLevel    `json:"isolation"`
	Policies    []string          `json:"policies"` // Policy IDs
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// IsolationLevel defines the level of isolation
type IsolationLevel string

const (
	IsolationNone     IsolationLevel = "none"
	IsolationPartial  IsolationLevel = "partial"
	IsolationStrict   IsolationLevel = "strict"
	IsolationComplete IsolationLevel = "complete"
)

// ZeroTrustNetworkService manages zero-trust network policies
type ZeroTrustNetworkService struct {
	policies      map[string]*NetworkPolicy
	microsegments map[string]*Microsegment
	connections   map[string]*NetworkConnection
	devices       map[string]*DeviceTrust
	mu            sync.RWMutex
	auditService  AuditService
	encryption    *EncryptionService
	tlsConfig     *tls.Config
}

// NewZeroTrustNetworkService creates a new zero-trust network service
func NewZeroTrustNetworkService(auditService AuditService, encryptionService *EncryptionService) *ZeroTrustNetworkService {
	return &ZeroTrustNetworkService{
		policies:      make(map[string]*NetworkPolicy),
		microsegments: make(map[string]*Microsegment),
		connections:   make(map[string]*NetworkConnection),
		devices:       make(map[string]*DeviceTrust),
		auditService:  auditService,
		encryption:    encryptionService,
	}
}

// CreatePolicy creates a new network policy
func (z *ZeroTrustNetworkService) CreatePolicy(policy *NetworkPolicy) error {
	if policy.ID == "" {
		return fmt.Errorf("policy ID is required")
	}

	// Validate policy
	if err := z.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	z.mu.Lock()
	defer z.mu.Unlock()

	if _, exists := z.policies[policy.ID]; exists {
		return fmt.Errorf("policy already exists: %s", policy.ID)
	}

	now := time.Now()
	policy.CreatedAt = now
	policy.UpdatedAt = now
	if policy.Metrics == (NetworkPolicyMetrics{}) {
		policy.Metrics = NetworkPolicyMetrics{}
	}

	z.policies[policy.ID] = policy

	// Log policy creation
	if z.auditService != nil {
		z.auditService.LogAccess(&AuditEntry{
			UserID:       policy.CreatedBy,
			ResourceType: "network_policy",
			ResourceID:   policy.ID,
			Action:       "create",
			Success:      true,
			Timestamp:    now,
			AdditionalData: map[string]interface{}{
				"policy_name": policy.Name,
				"action":      policy.Action,
			},
		})
	}

	return nil
}

// EvaluateConnection evaluates a network connection against policies
func (z *ZeroTrustNetworkService) EvaluateConnection(ctx context.Context, conn *NetworkConnection) (NetworkPolicyAction, *NetworkPolicy, error) {
	z.mu.RLock()
	defer z.mu.RUnlock()

	// Get applicable policies sorted by priority
	applicablePolicies := z.getApplicablePolicies(conn)

	for _, policy := range applicablePolicies {
		if !policy.Enabled {
			continue
		}

		// Check if policy matches the connection
		if matches, err := z.evaluatePolicyMatch(ctx, policy, conn); err != nil {
			return NetworkPolicyDeny, nil, fmt.Errorf("policy evaluation error: %w", err)
		} else if matches {
			// Update policy metrics
			policy.Metrics.Connections++
			policy.Metrics.LastAccess = time.Now()

			// Log policy application
			if z.auditService != nil {
				z.auditService.LogAccess(&AuditEntry{
					UserID:       conn.UserID,
					TenantID:     conn.TenantID,
					ResourceType: "network_connection",
					ResourceID:   conn.ID,
					Action:       string(policy.Action),
					Success:      policy.Action == NetworkPolicyAllow,
					Timestamp:    time.Now(),
					IPAddress:    conn.SourceIP,
					AdditionalData: map[string]interface{}{
						"policy_id":      policy.ID,
						"policy_name":    policy.Name,
						"destination_ip": conn.DestinationIP,
						"protocol":       conn.Protocol,
						"port":           conn.DestPort,
					},
				})
			}

			// Handle violations
			if policy.Action == NetworkPolicyDeny {
				policy.Metrics.Violations++
				policy.Metrics.LastViolation = time.Now()
			}

			return policy.Action, policy, nil
		}
	}

	// Default deny if no policy matches
	return NetworkPolicyDeny, nil, nil
}

// CreateMicrosegment creates a new microsegment
func (z *ZeroTrustNetworkService) CreateMicrosegment(segment *Microsegment) error {
	if segment.ID == "" {
		return fmt.Errorf("microsegment ID is required")
	}

	z.mu.Lock()
	defer z.mu.Unlock()

	if _, exists := z.microsegments[segment.ID]; exists {
		return fmt.Errorf("microsegment already exists: %s", segment.ID)
	}

	now := time.Now()
	segment.CreatedAt = now
	segment.UpdatedAt = now

	z.microsegments[segment.ID] = segment

	return nil
}

// RegisterDevice registers a device for zero-trust access
func (z *ZeroTrustNetworkService) RegisterDevice(device *DeviceTrust) error {
	if device.DeviceID == "" {
		return fmt.Errorf("device ID is required")
	}

	// Calculate initial trust level based on compliance
	device.TrustLevel = z.calculateDeviceTrustLevel(device)

	z.mu.Lock()
	defer z.mu.Unlock()

	device.RegisteredAt = time.Now()
	device.LastSeen = time.Now()

	z.devices[device.DeviceID] = device

	return nil
}

// ValidateDeviceTrust validates device trust for network access
func (z *ZeroTrustNetworkService) ValidateDeviceTrust(deviceID string, minTrustLevel int) (bool, *DeviceTrust, error) {
	z.mu.RLock()
	defer z.mu.RUnlock()

	device, exists := z.devices[deviceID]
	if !exists {
		return false, nil, fmt.Errorf("device not found: %s", deviceID)
	}

	// Update last seen
	device.LastSeen = time.Now()

	// Re-calculate trust level if compliance was recently checked
	if time.Since(device.Compliance.LastCheck) < time.Hour {
		device.TrustLevel = z.calculateDeviceTrustLevel(device)
	}

	return device.TrustLevel >= minTrustLevel, device, nil
}

// EnforcePolicy enforces network policies on connections
func (z *ZeroTrustNetworkService) EnforcePolicy(ctx context.Context, conn *NetworkConnection) error {
	action, policy, err := z.EvaluateConnection(ctx, conn)
	if err != nil {
		return fmt.Errorf("policy evaluation failed: %w", err)
	}

	switch action {
	case NetworkPolicyAllow:
		// Store allowed connection
		z.mu.Lock()
		z.connections[conn.ID] = conn
		z.mu.Unlock()
		return nil

	case NetworkPolicyDeny:
		return fmt.Errorf("connection denied by policy: %s", policy.Name)

	case NetworkPolicyRequireMTLS:
		if !conn.MTLSEnabled {
			return fmt.Errorf("mutual TLS required by policy: %s", policy.Name)
		}
		z.mu.Lock()
		z.connections[conn.ID] = conn
		z.mu.Unlock()
		return nil

	case NetworkPolicyRateLimit:
		// Implement rate limiting logic here
		return nil

	case NetworkPolicyQuarantine:
		// Move connection to quarantine segment
		return z.quarantineConnection(conn)

	default:
		return fmt.Errorf("unknown policy action: %s", action)
	}
}

// getApplicablePolicies returns policies that might apply to the connection
func (z *ZeroTrustNetworkService) getApplicablePolicies(conn *NetworkConnection) []*NetworkPolicy {
	var policies []*NetworkPolicy

	for _, policy := range z.policies {
		if z.policyMightApply(policy, conn) {
			policies = append(policies, policy)
		}
	}

	// Sort by priority (higher first)
	for i := 0; i < len(policies)-1; i++ {
		for j := i + 1; j < len(policies); j++ {
			if policies[i].Priority < policies[j].Priority {
				policies[i], policies[j] = policies[j], policies[i]
			}
		}
	}

	return policies
}

// policyMightApply does a quick check if policy might apply
func (z *ZeroTrustNetworkService) policyMightApply(policy *NetworkPolicy, conn *NetworkConnection) bool {
	// Quick filters to avoid expensive evaluation
	if !policy.Enabled {
		return false
	}

	// Check tenant match
	if len(policy.Source.TenantIDs) > 0 && conn.TenantID != "" {
		found := false
		for _, tenantID := range policy.Source.TenantIDs {
			if tenantID == conn.TenantID {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// evaluatePolicyMatch evaluates if a policy matches the connection
func (z *ZeroTrustNetworkService) evaluatePolicyMatch(ctx context.Context, policy *NetworkPolicy, conn *NetworkConnection) (bool, error) {
	// Evaluate source selector
	if !z.evaluateSelector(policy.Source, conn, "source") {
		return false, nil
	}

	// Evaluate destination selector
	if !z.evaluateSelector(policy.Destination, conn, "destination") {
		return false, nil
	}

	// Evaluate protocols
	if !z.evaluateProtocols(policy.Protocols, conn) {
		return false, nil
	}

	// Evaluate additional conditions
	for _, condition := range policy.Conditions {
		if matches, err := z.evaluateCondition(ctx, condition, conn); err != nil {
			return false, err
		} else if !matches {
			return false, nil
		}
	}

	return true, nil
}

// evaluateSelector evaluates a policy selector against connection
func (z *ZeroTrustNetworkService) evaluateSelector(selector NetworkPolicySelector, conn *NetworkConnection, side string) bool {
	if selector.Any {
		return true
	}

	// Check tenant IDs
	if len(selector.TenantIDs) > 0 && conn.TenantID != "" {
		found := false
		for _, tenantID := range selector.TenantIDs {
			if tenantID == conn.TenantID {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check user IDs
	if len(selector.UserIDs) > 0 && conn.UserID != "" {
		found := false
		for _, userID := range selector.UserIDs {
			if userID == conn.UserID {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check services
	if len(selector.Services) > 0 && conn.Service != "" {
		found := false
		for _, service := range selector.Services {
			if service == conn.Service {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check IP ranges
	if len(selector.IPRanges) > 0 {
		ip := conn.SourceIP
		if side == "destination" {
			ip = conn.DestinationIP
		}
		if !z.ipInRanges(ip, selector.IPRanges) {
			return false
		}
	}

	// Check labels
	if len(selector.Labels) > 0 {
		for key, value := range selector.Labels {
			if connValue, exists := conn.Labels[key]; !exists || connValue != value {
				return false
			}
		}
	}

	return true
}

// evaluateProtocols evaluates protocol restrictions
func (z *ZeroTrustNetworkService) evaluateProtocols(protocols []NetworkProtocol, conn *NetworkConnection) bool {
	if len(protocols) == 0 {
		return true // No protocol restrictions
	}

	for _, protocol := range protocols {
		if strings.ToLower(protocol.Protocol) != strings.ToLower(conn.Protocol) {
			continue
		}

		// Check specific ports
		if len(protocol.Ports) > 0 {
			found := false
			for _, port := range protocol.Ports {
				if port == conn.DestPort {
					found = true
					break
				}
			}
			if found {
				return true
			}
		}

		// Check port ranges
		if len(protocol.PortRanges) > 0 {
			for _, portRange := range protocol.PortRanges {
				if conn.DestPort >= portRange.Start && conn.DestPort <= portRange.End {
					return true
				}
			}
		}

		// If no port restrictions, protocol match is enough
		if len(protocol.Ports) == 0 && len(protocol.PortRanges) == 0 {
			return true
		}
	}

	return false
}

// evaluateCondition evaluates additional policy conditions
func (z *ZeroTrustNetworkService) evaluateCondition(ctx context.Context, condition NetworkPolicyCondition, conn *NetworkConnection) (bool, error) {
	switch condition.Type {
	case "time":
		return z.evaluateTimeCondition(condition, conn)
	case "device_trust":
		return z.evaluateDeviceTrustCondition(condition, conn)
	case "geo":
		return z.evaluateGeoCondition(condition, conn)
	default:
		return true, nil // Unknown conditions are ignored
	}
}

// evaluateTimeCondition evaluates time-based conditions
func (z *ZeroTrustNetworkService) evaluateTimeCondition(condition NetworkPolicyCondition, conn *NetworkConnection) (bool, error) {
	// Implementation would check business hours, weekends, etc.
	return true, nil
}

// evaluateDeviceTrustCondition evaluates device trust conditions
func (z *ZeroTrustNetworkService) evaluateDeviceTrustCondition(condition NetworkPolicyCondition, conn *NetworkConnection) (bool, error) {
	minTrustLevel, ok := condition.Value.(float64)
	if !ok {
		return false, fmt.Errorf("invalid device trust condition value")
	}

	// Try to find device based on connection attributes
	deviceID := conn.Labels["device_id"]
	if deviceID == "" {
		return false, nil // No device information
	}

	trusted, _, err := z.ValidateDeviceTrust(deviceID, int(minTrustLevel))
	return trusted, err
}

// evaluateGeoCondition evaluates geolocation conditions
func (z *ZeroTrustNetworkService) evaluateGeoCondition(condition NetworkPolicyCondition, conn *NetworkConnection) (bool, error) {
	// Implementation would check IP geolocation
	return true, nil
}

// ipInRanges checks if IP is in any of the given CIDR ranges
func (z *ZeroTrustNetworkService) ipInRanges(ipStr string, ranges []string) bool {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return false
	}

	for _, rangeStr := range ranges {
		if strings.Contains(rangeStr, "/") {
			// CIDR notation
			if _, network, err := net.ParseCIDR(rangeStr); err == nil {
				if network.Contains(ip) {
					return true
				}
			}
		} else {
			// Single IP
			if rangeStr == ipStr {
				return true
			}
		}
	}

	return false
}

// calculateDeviceTrustLevel calculates device trust level based on compliance
func (z *ZeroTrustNetworkService) calculateDeviceTrustLevel(device *DeviceTrust) int {
	baseTrust := 50 // Base trust level

	if device.Compliance.Antivirus {
		baseTrust += 10
	}
	if device.Compliance.Firewall {
		baseTrust += 10
	}
	if device.Compliance.Encryption {
		baseTrust += 15
	}
	if device.Compliance.OSUpdated {
		baseTrust += 10
	}
	if device.Compliance.ScreenLock {
		baseTrust += 5
	}
	if device.Compliance.Jailbroken {
		baseTrust -= 30 // Major trust penalty
	}

	// Cap at 100
	if baseTrust > 100 {
		baseTrust = 100
	}
	if baseTrust < 0 {
		baseTrust = 0
	}

	return baseTrust
}

// quarantineConnection moves a connection to a quarantine segment
func (z *ZeroTrustNetworkService) quarantineConnection(conn *NetworkConnection) error {
	// Implementation would isolate the connection
	conn.Labels["quarantined"] = "true"
	conn.Labels["quarantine_reason"] = "policy_violation"
	return nil
}

// validatePolicy validates a network policy
func (z *ZeroTrustNetworkService) validatePolicy(policy *NetworkPolicy) error {
	if policy.Name == "" {
		return fmt.Errorf("policy name is required")
	}

	if policy.Action == "" {
		return fmt.Errorf("policy action is required")
	}

	// Validate action
	validActions := []NetworkPolicyAction{
		NetworkPolicyAllow,
		NetworkPolicyDeny,
		NetworkPolicyLog,
		NetworkPolicyRateLimit,
		NetworkPolicyRequireMTLS,
		NetworkPolicyQuarantine,
	}

	validAction := false
	for _, validAct := range validActions {
		if policy.Action == validAct {
			validAction = true
			break
		}
	}

	if !validAction {
		return fmt.Errorf("invalid policy action: %s", policy.Action)
	}

	// Validate IP ranges
	for _, ipRange := range policy.Source.IPRanges {
		if err := z.validateIPRange(ipRange); err != nil {
			return fmt.Errorf("invalid source IP range %s: %w", ipRange, err)
		}
	}

	for _, ipRange := range policy.Destination.IPRanges {
		if err := z.validateIPRange(ipRange); err != nil {
			return fmt.Errorf("invalid destination IP range %s: %w", ipRange, err)
		}
	}

	return nil
}

// validateIPRange validates an IP range or CIDR
func (z *ZeroTrustNetworkService) validateIPRange(ipRange string) error {
	if strings.Contains(ipRange, "/") {
		// CIDR notation
		if _, _, err := net.ParseCIDR(ipRange); err != nil {
			return fmt.Errorf("invalid CIDR: %w", err)
		}
	} else {
		// Single IP
		if net.ParseIP(ipRange) == nil {
			return fmt.Errorf("invalid IP address: %s", ipRange)
		}
	}
	return nil
}

// GetPolicies returns all policies
func (z *ZeroTrustNetworkService) GetPolicies() []*NetworkPolicy {
	z.mu.RLock()
	defer z.mu.RUnlock()

	policies := make([]*NetworkPolicy, 0, len(z.policies))
	for _, policy := range z.policies {
		policies = append(policies, policy)
	}

	return policies
}

// GetMicrosegments returns all microsegments
func (z *ZeroTrustNetworkService) GetMicrosegments() []*Microsegment {
	z.mu.RLock()
	defer z.mu.RUnlock()

	segments := make([]*Microsegment, 0, len(z.microsegments))
	for _, segment := range z.microsegments {
		segments = append(segments, segment)
	}

	return segments
}

// GetConnections returns active connections
func (z *ZeroTrustNetworkService) GetConnections() []*NetworkConnection {
	z.mu.RLock()
	defer z.mu.RUnlock()

	connections := make([]*NetworkConnection, 0, len(z.connections))
	for _, conn := range z.connections {
		connections = append(connections, conn)
	}

	return connections
}

// CleanupExpiredConnections removes old connections
func (z *ZeroTrustNetworkService) CleanupExpiredConnections(maxAge time.Duration) {
	z.mu.Lock()
	defer z.mu.Unlock()

	now := time.Now()
	for id, conn := range z.connections {
		if now.Sub(conn.LastActivity) > maxAge {
			delete(z.connections, id)
		}
	}
}