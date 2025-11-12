# DWCP v3 Network Policy
# Open Policy Agent (OPA) policies for network validation

package dwcp.network

import future.keywords.if
import future.keywords.in

# Default deny
default allow = false

# Policy metadata
metadata := {
    "name": "DWCP v3 Network Policy",
    "version": "3.0.0",
    "description": "Network configuration validation for DWCP v3"
}

# ============================================================================
# NETWORK TOPOLOGY POLICIES
# ============================================================================

# Allow network configuration if topology is valid
allow if {
    input.action == "configure_network"
    valid_topology
    valid_subnets
    valid_routing
    valid_firewall_rules
}

# Validate network topology
valid_topology if {
    # VPC must have valid CIDR
    valid_vpc_cidr

    # Subnets must not overlap
    no_subnet_overlap

    # Availability zones properly distributed
    proper_az_distribution
}

valid_vpc_cidr if {
    # VPC CIDR must be private
    private_cidr(input.vpc.cidr)

    # VPC CIDR size must be appropriate
    cidr_size_valid(input.vpc.cidr)
}

cidr_size_valid(cidr) if {
    # Extract prefix length
    parts := split(cidr, "/")
    prefix := to_number(parts[1])

    # Must be between /16 and /24
    prefix >= 16
    prefix <= 24
}

private_cidr(cidr) if {
    startswith(cidr, "10.")
}

private_cidr(cidr) if {
    regex.match(`^172\.(1[6-9]|2[0-9]|3[0-1])\.`, cidr)
}

private_cidr(cidr) if {
    startswith(cidr, "192.168.")
}

no_subnet_overlap if {
    # Check all subnet pairs for overlap
    every i, subnet1 in input.subnets {
        every j, subnet2 in input.subnets {
            i != j implies not cidrs_overlap(subnet1.cidr, subnet2.cidr)
        }
    }
}

cidrs_overlap(cidr1, cidr2) if {
    # Simplified overlap check (in production, use proper CIDR library)
    cidr1 == cidr2
}

proper_az_distribution if {
    # Must have at least 2 AZs for high availability
    count(input.availability_zones) >= 2

    # Datacenter subnets must span multiple AZs
    datacenter_multi_az

    # Internet subnets must span multiple AZs
    internet_multi_az
}

datacenter_multi_az if {
    datacenter_subnets := [s | s := input.subnets[_]; s.type == "datacenter"]
    azs := {s.availability_zone | s := datacenter_subnets[_]}
    count(azs) >= 2
}

internet_multi_az if {
    internet_subnets := [s | s := input.subnets[_]; s.type == "internet"]
    azs := {s.availability_zone | s := internet_subnets[_]}
    count(azs) >= 2
}

# ============================================================================
# SUBNET POLICIES
# ============================================================================

# Validate subnet configuration
valid_subnets if {
    # All subnets must be within VPC CIDR
    subnets_within_vpc

    # Datacenter subnets must be private
    datacenter_subnets_private

    # Internet subnets can be public
    internet_subnets_configured
}

subnets_within_vpc if {
    every subnet in input.subnets {
        subnet_within_vpc(subnet)
    }
}

subnet_within_vpc(subnet) if {
    # Simplified check (in production, use proper CIDR library)
    startswith(subnet.cidr, split(input.vpc.cidr, "/")[0])
}

datacenter_subnets_private if {
    datacenter_subnets := [s | s := input.subnets[_]; s.type == "datacenter"]
    every subnet in datacenter_subnets {
        subnet.public == false
    }
}

internet_subnets_configured if {
    internet_subnets := [s | s := input.subnets[_]; s.type == "internet"]
    every subnet in internet_subnets {
        subnet.internet_gateway_attached == true
    }
}

# ============================================================================
# ROUTING POLICIES
# ============================================================================

# Validate routing configuration
valid_routing if {
    # Route tables must be defined
    count(input.route_tables) > 0

    # Default routes must be valid
    valid_default_routes

    # NAT gateways for private subnets
    nat_gateways_configured
}

valid_default_routes if {
    every rt in input.route_tables {
        has_valid_default_route(rt)
    }
}

has_valid_default_route(rt) if {
    # Private subnets route to NAT gateway
    rt.subnet_type == "datacenter"
    some route in rt.routes
    route.destination == "0.0.0.0/0"
    route.target_type == "nat_gateway"
}

has_valid_default_route(rt) if {
    # Public subnets route to internet gateway
    rt.subnet_type == "internet"
    some route in rt.routes
    route.destination == "0.0.0.0/0"
    route.target_type == "internet_gateway"
}

nat_gateways_configured if {
    datacenter_subnets := [s | s := input.subnets[_]; s.type == "datacenter"]
    count(datacenter_subnets) > 0 implies count(input.nat_gateways) > 0
}

# ============================================================================
# FIREWALL RULES POLICIES
# ============================================================================

# Validate firewall rules
valid_firewall_rules if {
    # Security groups must be defined
    count(input.security_groups) > 0

    # Datacenter security group rules
    valid_datacenter_firewall

    # Internet security group rules
    valid_internet_firewall

    # No overly permissive rules
    no_permissive_rules
}

valid_datacenter_firewall if {
    datacenter_sg := [sg | sg := input.security_groups[_]; sg.type == "datacenter"]
    every sg in datacenter_sg {
        # Allow RDMA traffic within VPC
        allows_rdma_traffic(sg)

        # Allow control plane traffic
        allows_control_plane(sg)

        # Restrict SSH to VPC
        restricts_ssh_to_vpc(sg)
    }
}

allows_rdma_traffic(sg) if {
    some rule in sg.ingress_rules
    rule.protocol == "-1"  # All protocols
    private_cidr(rule.cidr_blocks[_])
}

allows_control_plane(sg) if {
    some rule in sg.ingress_rules
    rule.port in [8080, 9100]
    private_cidr(rule.cidr_blocks[_])
}

restricts_ssh_to_vpc(sg) if {
    ssh_rules := [r | r := sg.ingress_rules[_]; r.port == 22]
    every rule in ssh_rules {
        private_cidr(rule.cidr_blocks[_])
    }
}

valid_internet_firewall if {
    internet_sg := [sg | sg := input.security_groups[_]; sg.type == "internet"]
    every sg in internet_sg {
        # Allow DWCP traffic
        allows_dwcp_traffic(sg)

        # Restrict management ports
        restricts_management_ports(sg)
    }
}

allows_dwcp_traffic(sg) if {
    some rule in sg.ingress_rules
    rule.port in [443, 8443]
    rule.protocol in ["tcp", "udp"]
}

restricts_management_ports(sg) if {
    mgmt_rules := [r | r := sg.ingress_rules[_]; r.port in [22, 3389, 8080, 9090]]
    every rule in mgmt_rules {
        private_cidr(rule.cidr_blocks[_])
    }
}

no_permissive_rules if {
    not has_permissive_rule
}

has_permissive_rule if {
    some sg in input.security_groups
    some rule in sg.ingress_rules
    rule.protocol == "-1"
    rule.cidr_blocks[_] == "0.0.0.0/0"
}

# ============================================================================
# RDMA POLICIES (DATACENTER)
# ============================================================================

# Validate RDMA configuration
valid_rdma_config if {
    input.mode == "datacenter" implies rdma_requirements_met
}

rdma_requirements_met if {
    # RDMA device specified
    input.config.network.rdma.enabled == true
    input.config.network.rdmaDevice != ""

    # Proper queue configuration
    input.config.network.rdma.maxQueuePairs >= 256
    input.config.network.rdma.maxSendWr >= 1024

    # Hugepages enabled
    input.config.network.rdma.hugepagesEnabled == true

    # NUMA awareness
    input.config.memory.numaAware == true
}

# ============================================================================
# PERFORMANCE POLICIES
# ============================================================================

# Validate performance configuration
valid_performance_config if {
    # MTU appropriate for mode
    valid_mtu

    # Buffer sizes appropriate for mode
    valid_buffer_sizes

    # Connection limits reasonable
    valid_connection_limits
}

valid_mtu if {
    input.mode == "datacenter" implies input.config.network.mtu == 9000
}

valid_mtu if {
    input.mode == "internet" implies input.config.network.mtu == 1500
}

valid_mtu if {
    input.mode == "hybrid"
}

valid_buffer_sizes if {
    input.mode == "datacenter" implies datacenter_buffer_sizes
}

valid_buffer_sizes if {
    input.mode == "internet" implies internet_buffer_sizes
}

valid_buffer_sizes if {
    input.mode == "hybrid"
}

datacenter_buffer_sizes if {
    buffer_size_mb(input.config.performance.sendBufferSize) >= 16
    buffer_size_mb(input.config.performance.recvBufferSize) >= 16
}

internet_buffer_sizes if {
    buffer_size_mb(input.config.performance.sendBufferSize) >= 4
    buffer_size_mb(input.config.performance.recvBufferSize) >= 4
}

buffer_size_mb(size_str) := mb if {
    endswith(size_str, "MB")
    mb := to_number(trim_suffix(size_str, "MB"))
}

valid_connection_limits if {
    input.config.performance.maxConnections > 0
    input.config.performance.maxConnections <= 100000
}

# ============================================================================
# MULTI-REGION POLICIES
# ============================================================================

# Validate multi-region configuration
valid_multi_region if {
    input.deployment_type == "multi-region" implies multi_region_requirements
}

multi_region_requirements if {
    # At least 2 regions
    count(input.regions) >= 2

    # VPC peering configured
    vpc_peering_configured

    # DNS failover configured
    dns_failover_configured
}

vpc_peering_configured if {
    count(input.vpc_peering_connections) >= count(input.regions) - 1
}

dns_failover_configured if {
    input.route53.health_checks_enabled == true
}

# ============================================================================
# VIOLATION REPORTING
# ============================================================================

violations[msg] {
    not valid_vpc_cidr
    msg := "VPC CIDR is invalid or not private"
}

violations[msg] {
    not no_subnet_overlap
    msg := "Subnet CIDR blocks overlap"
}

violations[msg] {
    not proper_az_distribution
    msg := "Subnets must be distributed across at least 2 availability zones"
}

violations[msg] {
    not datacenter_subnets_private
    msg := "Datacenter subnets must be private"
}

violations[msg] {
    not nat_gateways_configured
    msg := "NAT gateways must be configured for private subnets"
}

violations[msg] {
    not valid_datacenter_firewall
    msg := "Datacenter firewall rules are invalid"
}

violations[msg] {
    not valid_internet_firewall
    msg := "Internet firewall rules are invalid"
}

violations[msg] {
    has_permissive_rule
    msg := "Security groups contain overly permissive rules (0.0.0.0/0 with all protocols)"
}

violations[msg] {
    input.mode == "datacenter"
    not rdma_requirements_met
    msg := "RDMA requirements not met for datacenter mode"
}

violations[msg] {
    not valid_mtu
    msg := sprintf("Invalid MTU for mode %s", [input.mode])
}

violations[msg] {
    not valid_connection_limits
    msg := "Connection limits are outside acceptable range (1-100000)"
}

# ============================================================================
# DECISION OUTPUT
# ============================================================================

decision := {
    "allow": allow,
    "violations": violations,
    "metadata": metadata
}
