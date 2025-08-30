package sdn

import (
	"fmt"
	"log"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/google/uuid"
)

// PolicyCompiler compiles high-level policy rules into OpenFlow flow entries
type PolicyCompiler struct {
	controller     *SDNController
	selectorCache  map[string][]string // Cache for endpoint selectors
	compiledRules  map[string][]FlowEntry // Rule ID -> Flow entries
	mutex          sync.RWMutex
}

// NewPolicyCompiler creates a new policy compiler
func NewPolicyCompiler(controller *SDNController) *PolicyCompiler {
	return &PolicyCompiler{
		controller:    controller,
		selectorCache: make(map[string][]string),
		compiledRules: make(map[string][]FlowEntry),
	}
}

// CompileRule compiles a high-level policy rule into flow entries
func (pc *PolicyCompiler) CompileRule(rule *PolicyRule) error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()
	
	if !rule.Enabled {
		log.Printf("Skipping disabled policy rule: %s", rule.ID)
		return nil
	}
	
	log.Printf("Compiling policy rule: %s (%s)", rule.ID, rule.Name)
	
	// Resolve selectors to concrete endpoints
	srcEndpoints, err := pc.resolveSelector(rule.SrcSelector, rule.TenantID)
	if err != nil {
		return fmt.Errorf("failed to resolve source selector '%s': %w", rule.SrcSelector, err)
	}
	
	dstEndpoints, err := pc.resolveSelector(rule.DstSelector, rule.TenantID)
	if err != nil {
		return fmt.Errorf("failed to resolve destination selector '%s': %w", rule.DstSelector, err)
	}
	
	// Generate flow entries for all endpoint combinations
	var flowEntries []FlowEntry
	
	for _, srcEndpoint := range srcEndpoints {
		for _, dstEndpoint := range dstEndpoints {
			entries, err := pc.generateFlowEntries(rule, srcEndpoint, dstEndpoint)
			if err != nil {
				return fmt.Errorf("failed to generate flow entries: %w", err)
			}
			flowEntries = append(flowEntries, entries...)
		}
	}
	
	// Store compiled flow entries
	pc.compiledRules[rule.ID] = flowEntries
	
	// Install flow entries on relevant switches
	if err := pc.installFlowEntries(rule.TenantID, flowEntries); err != nil {
		return fmt.Errorf("failed to install flow entries: %w", err)
	}
	
	log.Printf("Successfully compiled and installed %d flow entries for rule %s", 
		len(flowEntries), rule.ID)
	return nil
}

// RemoveRule removes compiled flow entries for a policy rule
func (pc *PolicyCompiler) RemoveRule(rule *PolicyRule) error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()
	
	flowEntries, exists := pc.compiledRules[rule.ID]
	if !exists {
		log.Printf("No compiled flow entries found for rule %s", rule.ID)
		return nil
	}
	
	// Remove flow entries from switches
	if err := pc.removeFlowEntries(rule.TenantID, flowEntries); err != nil {
		return fmt.Errorf("failed to remove flow entries: %w", err)
	}
	
	// Remove from compiled rules cache
	delete(pc.compiledRules, rule.ID)
	
	log.Printf("Successfully removed %d flow entries for rule %s", 
		len(flowEntries), rule.ID)
	return nil
}

// RecompileRule recompiles a policy rule (useful when selectors change)
func (pc *PolicyCompiler) RecompileRule(rule *PolicyRule) error {
	// Remove existing compiled entries
	if err := pc.RemoveRule(rule); err != nil {
		log.Printf("Warning: failed to remove existing rule during recompile: %v", err)
	}
	
	// Compile new entries
	return pc.CompileRule(rule)
}

// GetCompiledFlows returns the compiled flow entries for a rule
func (pc *PolicyCompiler) GetCompiledFlows(ruleID string) ([]FlowEntry, bool) {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()
	
	flows, exists := pc.compiledRules[ruleID]
	return flows, exists
}

// ClearSelectorCache clears the endpoint selector cache
func (pc *PolicyCompiler) ClearSelectorCache() {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()
	
	pc.selectorCache = make(map[string][]string)
}

// resolveSelector resolves an endpoint selector to a list of IP addresses
func (pc *PolicyCompiler) resolveSelector(selector, tenantID string) ([]string, error) {
	cacheKey := fmt.Sprintf("%s:%s", tenantID, selector)
	
	// Check cache first
	if endpoints, exists := pc.selectorCache[cacheKey]; exists {
		return endpoints, nil
	}
	
	var endpoints []string
	
	// Parse selector format
	if selector == "*" || selector == "any" {
		// Match any endpoint - this would need integration with service discovery
		endpoints = []string{"0.0.0.0/0"}
	} else if strings.HasPrefix(selector, "label:") {
		// Label-based selector (e.g., label:app=web)
		label := strings.TrimPrefix(selector, "label:")
		endpoints = pc.resolveLabelSelector(label, tenantID)
	} else if strings.HasPrefix(selector, "namespace:") {
		// Namespace-based selector
		namespace := strings.TrimPrefix(selector, "namespace:")
		endpoints = pc.resolveNamespaceSelector(namespace, tenantID)
	} else if strings.HasPrefix(selector, "service:") {
		// Service-based selector
		service := strings.TrimPrefix(selector, "service:")
		endpoints = pc.resolveServiceSelector(service, tenantID)
	} else if strings.Contains(selector, "/") {
		// CIDR block
		if _, _, err := net.ParseCIDR(selector); err != nil {
			return nil, fmt.Errorf("invalid CIDR: %s", selector)
		}
		endpoints = []string{selector}
	} else if net.ParseIP(selector) != nil {
		// Single IP address
		endpoints = []string{selector + "/32"}
	} else {
		// Hostname or other identifier - would need DNS resolution
		endpoints = pc.resolveHostnameSelector(selector, tenantID)
	}
	
	if len(endpoints) == 0 {
		return nil, fmt.Errorf("no endpoints found for selector: %s", selector)
	}
	
	// Cache the result
	pc.selectorCache[cacheKey] = endpoints
	
	log.Printf("Resolved selector '%s' for tenant %s to %d endpoints", 
		selector, tenantID, len(endpoints))
	return endpoints, nil
}

// resolveLabelSelector resolves a label-based selector
func (pc *PolicyCompiler) resolveLabelSelector(label, tenantID string) []string {
	// This would integrate with a service discovery system or endpoint registry
	// For now, return some example endpoints based on common labels
	
	var endpoints []string
	
	if strings.Contains(label, "app=web") {
		endpoints = []string{"10.0.1.10/32", "10.0.1.11/32", "10.0.1.12/32"}
	} else if strings.Contains(label, "app=db") {
		endpoints = []string{"10.0.2.10/32", "10.0.2.11/32"}
	} else if strings.Contains(label, "tier=frontend") {
		endpoints = []string{"10.0.1.0/24"}
	} else if strings.Contains(label, "tier=backend") {
		endpoints = []string{"10.0.2.0/24"}
	} else {
		// Default fallback - this would query the actual service registry
		endpoints = []string{"10.0.0.0/16"}
	}
	
	log.Printf("Resolved label selector '%s' for tenant %s to endpoints: %v", 
		label, tenantID, endpoints)
	return endpoints
}

// resolveNamespaceSelector resolves a namespace-based selector
func (pc *PolicyCompiler) resolveNamespaceSelector(namespace, tenantID string) []string {
	// This would integrate with Kubernetes or similar orchestration system
	// For now, map namespaces to subnet ranges
	
	var endpoints []string
	
	switch namespace {
	case "default":
		endpoints = []string{"10.0.0.0/24"}
	case "production":
		endpoints = []string{"10.0.10.0/24"}
	case "staging":
		endpoints = []string{"10.0.11.0/24"}
	case "development":
		endpoints = []string{"10.0.12.0/24"}
	default:
		// Dynamic namespace - would query orchestrator
		endpoints = []string{fmt.Sprintf("10.0.100.0/24")} // Example subnet
	}
	
	log.Printf("Resolved namespace selector '%s' for tenant %s to endpoints: %v", 
		namespace, tenantID, endpoints)
	return endpoints
}

// resolveServiceSelector resolves a service-based selector
func (pc *PolicyCompiler) resolveServiceSelector(service, tenantID string) []string {
	// This would integrate with service discovery (DNS, Consul, etc.)
	// For now, provide some example service mappings
	
	var endpoints []string
	
	switch service {
	case "web":
		endpoints = []string{"10.0.1.100/32", "10.0.1.101/32"}
	case "api":
		endpoints = []string{"10.0.2.100/32", "10.0.2.101/32"}
	case "database":
		endpoints = []string{"10.0.3.100/32"}
	case "cache":
		endpoints = []string{"10.0.4.100/32"}
	default:
		// Would perform actual service discovery lookup
		endpoints = []string{"10.0.50.100/32"}
	}
	
	log.Printf("Resolved service selector '%s' for tenant %s to endpoints: %v", 
		service, tenantID, endpoints)
	return endpoints
}

// resolveHostnameSelector resolves a hostname-based selector
func (pc *PolicyCompiler) resolveHostnameSelector(hostname, tenantID string) []string {
	// This would perform DNS resolution
	// For now, provide example mappings
	
	var endpoints []string
	
	if ips, err := net.LookupIP(hostname); err == nil {
		for _, ip := range ips {
			if ip.To4() != nil {
				endpoints = append(endpoints, ip.String()+"/32")
			} else {
				endpoints = append(endpoints, ip.String()+"/128")
			}
		}
	} else {
		// Fallback for unresolvable hostnames
		log.Printf("Warning: failed to resolve hostname %s: %v", hostname, err)
		endpoints = []string{"0.0.0.0/0"} // Allow all as fallback
	}
	
	log.Printf("Resolved hostname selector '%s' for tenant %s to endpoints: %v", 
		hostname, tenantID, endpoints)
	return endpoints
}

// generateFlowEntries generates OpenFlow flow entries for a policy rule
func (pc *PolicyCompiler) generateFlowEntries(rule *PolicyRule, srcEndpoint, dstEndpoint string) ([]FlowEntry, error) {
	var flowEntries []FlowEntry
	
	// Parse source and destination endpoints
	srcIP, srcNet, err := net.ParseCIDR(srcEndpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid source endpoint: %s", srcEndpoint)
	}
	
	dstIP, dstNet, err := net.ParseCIDR(dstEndpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid destination endpoint: %s", dstEndpoint)
	}
	
	// Determine protocol number
	var protoNum *uint8
	if rule.Protocol != "" && rule.Protocol != "any" {
		switch strings.ToLower(rule.Protocol) {
		case "tcp":
			num := uint8(6)
			protoNum = &num
		case "udp":
			num := uint8(17)
			protoNum = &num
		case "icmp":
			num := uint8(1)
			protoNum = &num
		case "icmpv6":
			num := uint8(58)
			protoNum = &num
		default:
			if num, err := strconv.ParseUint(rule.Protocol, 10, 8); err == nil {
				p := uint8(num)
				protoNum = &p
			}
		}
	}
	
	// Generate different flow entries based on direction and action
	switch rule.Direction {
	case "ingress":
		entries := pc.generateIngressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		flowEntries = append(flowEntries, entries...)
	case "egress":
		entries := pc.generateEgressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		flowEntries = append(flowEntries, entries...)
	case "both":
		ingressEntries := pc.generateIngressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		egressEntries := pc.generateEgressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		flowEntries = append(flowEntries, ingressEntries...)
		flowEntries = append(flowEntries, egressEntries...)
	default:
		// Default to both directions
		ingressEntries := pc.generateIngressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		egressEntries := pc.generateEgressFlows(rule, srcIP, srcNet, dstIP, dstNet, protoNum)
		flowEntries = append(flowEntries, ingressEntries...)
		flowEntries = append(flowEntries, egressEntries...)
	}
	
	return flowEntries, nil
}

// generateIngressFlows generates ingress flow entries
func (pc *PolicyCompiler) generateIngressFlows(rule *PolicyRule, srcIP net.IP, srcNet *net.IPNet, dstIP net.IP, dstNet *net.IPNet, protoNum *uint8) []FlowEntry {
	var flowEntries []FlowEntry
	
	// Create basic match for ingress traffic
	match := FlowMatch{
		IPv4Src: srcNet,
		IPv4Dst: dstNet,
	}
	
	if srcIP.To4() == nil && dstIP.To4() == nil {
		// IPv6 addresses
		match.IPv4Src = nil
		match.IPv4Dst = nil
		match.IPv6Src = &net.IPNet{IP: srcIP, Mask: srcNet.Mask}
		match.IPv6Dst = &net.IPNet{IP: dstIP, Mask: dstNet.Mask}
	}
	
	if protoNum != nil {
		match.IPProto = protoNum
	}
	
	// Add port restrictions if specified
	if len(rule.DstPorts) > 0 {
		for _, portSpec := range rule.DstPorts {
			portMatch := match // Copy base match
			if err := pc.addPortMatch(&portMatch, portSpec, "dst", rule.Protocol); err != nil {
				log.Printf("Warning: invalid port specification %s: %v", portSpec, err)
				continue
			}
			
			entry := pc.createFlowEntry(rule, TableAcl, portMatch)
			flowEntries = append(flowEntries, entry)
		}
	} else {
		// No port restrictions
		entry := pc.createFlowEntry(rule, TableAcl, match)
		flowEntries = append(flowEntries, entry)
	}
	
	return flowEntries
}

// generateEgressFlows generates egress flow entries
func (pc *PolicyCompiler) generateEgressFlows(rule *PolicyRule, srcIP net.IP, srcNet *net.IPNet, dstIP net.IP, dstNet *net.IPNet, protoNum *uint8) []FlowEntry {
	var flowEntries []FlowEntry
	
	// Create basic match for egress traffic (swap src/dst)
	match := FlowMatch{
		IPv4Src: dstNet,
		IPv4Dst: srcNet,
	}
	
	if srcIP.To4() == nil && dstIP.To4() == nil {
		// IPv6 addresses
		match.IPv4Src = nil
		match.IPv4Dst = nil
		match.IPv6Src = &net.IPNet{IP: dstIP, Mask: dstNet.Mask}
		match.IPv6Dst = &net.IPNet{IP: srcIP, Mask: srcNet.Mask}
	}
	
	if protoNum != nil {
		match.IPProto = protoNum
	}
	
	// Add port restrictions if specified (swap src/dst ports for egress)
	if len(rule.SrcPorts) > 0 {
		for _, portSpec := range rule.SrcPorts {
			portMatch := match // Copy base match
			if err := pc.addPortMatch(&portMatch, portSpec, "dst", rule.Protocol); err != nil {
				log.Printf("Warning: invalid port specification %s: %v", portSpec, err)
				continue
			}
			
			entry := pc.createFlowEntry(rule, TableEgress, portMatch)
			flowEntries = append(flowEntries, entry)
		}
	} else {
		// No port restrictions
		entry := pc.createFlowEntry(rule, TableEgress, match)
		flowEntries = append(flowEntries, entry)
	}
	
	return flowEntries
}

// addPortMatch adds port matching to a flow match
func (pc *PolicyCompiler) addPortMatch(match *FlowMatch, portSpec, direction, protocol string) error {
	if protocol != "tcp" && protocol != "udp" {
		return nil // Ports only apply to TCP/UDP
	}
	
	// Parse port specification (can be single port, range, or list)
	if strings.Contains(portSpec, "-") {
		// Port range (e.g., "80-90")
		parts := strings.Split(portSpec, "-")
		if len(parts) != 2 {
			return fmt.Errorf("invalid port range: %s", portSpec)
		}
		
		startPort, err := strconv.ParseUint(parts[0], 10, 16)
		if err != nil {
			return fmt.Errorf("invalid start port: %s", parts[0])
		}
		
		_, err = strconv.ParseUint(parts[1], 10, 16)
		if err != nil {
			return fmt.Errorf("invalid end port: %s", parts[1])
		}
		
		// For ranges, we'll create multiple flow entries
		// For now, just use the start port
		port := uint16(startPort)
		pc.setPortInMatch(match, port, direction, protocol)
		
		// TODO: Handle port ranges properly by creating multiple flow entries
		log.Printf("Warning: port range %s simplified to single port %d", portSpec, startPort)
	} else {
		// Single port
		port, err := strconv.ParseUint(portSpec, 10, 16)
		if err != nil {
			return fmt.Errorf("invalid port: %s", portSpec)
		}
		
		pc.setPortInMatch(match, uint16(port), direction, protocol)
	}
	
	return nil
}

// setPortInMatch sets the appropriate port field in a flow match
func (pc *PolicyCompiler) setPortInMatch(match *FlowMatch, port uint16, direction, protocol string) {
	switch strings.ToLower(protocol) {
	case "tcp":
		if direction == "src" {
			match.TCPSrc = &port
		} else {
			match.TCPDst = &port
		}
	case "udp":
		if direction == "src" {
			match.UDPSrc = &port
		} else {
			match.UDPDst = &port
		}
	}
}

// createFlowEntry creates a flow entry from a policy rule and match
func (pc *PolicyCompiler) createFlowEntry(rule *PolicyRule, tableID FlowTableID, match FlowMatch) FlowEntry {
	entry := FlowEntry{
		ID:          uuid.New().String(),
		TableID:     tableID,
		Priority:    uint16(rule.Priority),
		IdleTimeout: 0,  // No timeout for policy rules
		HardTimeout: 0,  // No timeout for policy rules
		Cookie:      pc.generateCookie(rule.ID),
		Match:       match,
		Instructions: pc.createInstructions(rule),
	}
	
	return entry
}

// generateCookie generates a unique cookie for flow entries from a policy rule
func (pc *PolicyCompiler) generateCookie(ruleID string) uint64 {
	// Generate a deterministic cookie based on rule ID
	hash := uint64(0)
	for _, b := range []byte(ruleID) {
		hash = hash*31 + uint64(b)
	}
	return hash
}

// createInstructions creates OpenFlow instructions based on policy rule action
func (pc *PolicyCompiler) createInstructions(rule *PolicyRule) []Instruction {
	var instructions []Instruction
	
	switch strings.ToLower(rule.Action) {
	case "allow", "permit":
		// Allow traffic - continue to next table or output
		instructions = []Instruction{
			{
				Type:    InstGotoTable,
				TableID: &[]FlowTableID{TableForwarding}[0],
			},
		}
		
	case "deny", "drop", "reject":
		// Drop traffic - no instructions (packet dropped)
		// Empty instructions list means drop
		
	case "redirect":
		// Redirect to different output - would need destination info
		// For now, just allow and let forwarding table handle it
		instructions = []Instruction{
			{
				Type:    InstGotoTable,
				TableID: &[]FlowTableID{TableForwarding}[0],
			},
		}
		
	case "qos":
		// Apply QoS - set queue and continue
		if rule.QoSClass != "" {
			queueID := pc.getQueueIDForClass(rule.QoSClass)
			instructions = []Instruction{
				{
					Type: InstApplyActions,
					Actions: []Action{
						{
							Type:    ActionSetQueue,
							QueueID: &queueID,
						},
					},
				},
				{
					Type:    InstGotoTable,
					TableID: &[]FlowTableID{TableForwarding}[0],
				},
			}
		} else {
			// No QoS class specified, just continue
			instructions = []Instruction{
				{
					Type:    InstGotoTable,
					TableID: &[]FlowTableID{TableForwarding}[0],
				},
			}
		}
		
	default:
		// Default action is allow
		instructions = []Instruction{
			{
				Type:    InstGotoTable,
				TableID: &[]FlowTableID{TableForwarding}[0],
			},
		}
	}
	
	return instructions
}

// getQueueIDForClass maps QoS class names to queue IDs
func (pc *PolicyCompiler) getQueueIDForClass(qosClass string) uint32 {
	// Map QoS classes to queue IDs
	switch strings.ToLower(qosClass) {
	case "high", "priority", "realtime":
		return 1
	case "medium", "normal", "default":
		return 2
	case "low", "background", "bulk":
		return 3
	case "best-effort":
		return 0
	default:
		return 0 // Default queue
	}
}

// installFlowEntries installs flow entries on switches for a tenant
func (pc *PolicyCompiler) installFlowEntries(tenantID string, flowEntries []FlowEntry) error {
	// Find switches that belong to this tenant
	switches := pc.controller.ListSwitches()
	var tenantSwitches []*Switch
	
	for _, switch_ := range switches {
		if switch_.TenantID == tenantID || switch_.TenantID == "" {
			tenantSwitches = append(tenantSwitches, switch_)
		}
	}
	
	if len(tenantSwitches) == 0 {
		log.Printf("Warning: no switches found for tenant %s", tenantID)
		return nil
	}
	
	// Install flow entries on all tenant switches
	for _, switch_ := range tenantSwitches {
		for _, flowEntry := range flowEntries {
			if err := pc.controller.InstallFlowEntry(switch_.ID, flowEntry); err != nil {
				log.Printf("Failed to install flow entry %s on switch %s: %v", 
					flowEntry.ID, switch_.ID, err)
				// Continue with other entries rather than failing completely
			}
		}
	}
	
	log.Printf("Installed %d flow entries on %d switches for tenant %s", 
		len(flowEntries), len(tenantSwitches), tenantID)
	return nil
}

// removeFlowEntries removes flow entries from switches for a tenant
func (pc *PolicyCompiler) removeFlowEntries(tenantID string, flowEntries []FlowEntry) error {
	// Find switches that belong to this tenant
	switches := pc.controller.ListSwitches()
	var tenantSwitches []*Switch
	
	for _, switch_ := range switches {
		if switch_.TenantID == tenantID || switch_.TenantID == "" {
			tenantSwitches = append(tenantSwitches, switch_)
		}
	}
	
	// Remove flow entries from all tenant switches
	for _, switch_ := range tenantSwitches {
		for _, flowEntry := range flowEntries {
			if err := pc.controller.RemoveFlowEntry(switch_.ID, flowEntry.ID); err != nil {
				log.Printf("Failed to remove flow entry %s from switch %s: %v", 
					flowEntry.ID, switch_.ID, err)
				// Continue with other entries rather than failing completely
			}
		}
	}
	
	log.Printf("Removed %d flow entries from %d switches for tenant %s", 
		len(flowEntries), len(tenantSwitches), tenantID)
	return nil
}