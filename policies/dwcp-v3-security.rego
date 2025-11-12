# DWCP v3 Security Policy
# Open Policy Agent (OPA) policies for security validation

package dwcp.security

import future.keywords.if
import future.keywords.in

# Default deny
default allow = false

# Policy metadata
metadata := {
    "name": "DWCP v3 Security Policy",
    "version": "3.0.0",
    "description": "Security validation for DWCP v3 infrastructure"
}

# ============================================================================
# RESOURCE PROVISIONING POLICIES
# ============================================================================

# Allow resource creation if security requirements are met
allow if {
    input.action == "create"
    input.resource_type in ["vm", "network", "storage"]
    valid_security_configuration
    no_critical_vulnerabilities
}

# ============================================================================
# NETWORK SECURITY POLICIES
# ============================================================================

# Validate network security configuration
valid_network_security if {
    # Security groups must be defined
    count(input.security_groups) > 0

    # All security groups must have restricted ingress
    every sg in input.security_groups {
        restricted_ingress(sg)
    }

    # Egress rules must be defined
    every sg in input.security_groups {
        has_egress_rules(sg)
    }
}

# Restricted ingress rules
restricted_ingress(sg) if {
    # No unrestricted SSH access
    not unrestricted_ssh(sg)

    # No unrestricted RDP access
    not unrestricted_rdp(sg)

    # Management ports restricted to VPC
    management_ports_restricted(sg)
}

unrestricted_ssh(sg) if {
    some rule in sg.ingress_rules
    rule.port == 22
    rule.cidr_blocks[_] == "0.0.0.0/0"
}

unrestricted_rdp(sg) if {
    some rule in sg.ingress_rules
    rule.port == 3389
    rule.cidr_blocks[_] == "0.0.0.0/0"
}

management_ports_restricted(sg) if {
    management_ports := [22, 3389, 8080, 9090, 3000]
    every rule in sg.ingress_rules {
        rule.port in management_ports implies vpc_only(rule)
    }
}

vpc_only(rule) if {
    every cidr in rule.cidr_blocks {
        private_cidr(cidr)
    }
}

private_cidr(cidr) if {
    startswith(cidr, "10.")
}

private_cidr(cidr) if {
    startswith(cidr, "172.16.")
}

private_cidr(cidr) if {
    startswith(cidr, "192.168.")
}

has_egress_rules(sg) if {
    count(sg.egress_rules) > 0
}

# ============================================================================
# ENCRYPTION POLICIES
# ============================================================================

# Validate encryption configuration
valid_encryption if {
    # Internet mode must have encryption enabled
    input.mode == "internet" implies encryption_enabled

    # Datacenter mode can disable encryption (trusted network)
    input.mode == "datacenter"

    # Hybrid mode must support encryption
    input.mode == "hybrid" implies encryption_supported
}

encryption_enabled if {
    input.config.protocol.encryption.enabled == true
    valid_encryption_algorithm
}

encryption_supported if {
    input.config.protocol.encryption.enabled in [true, "auto"]
    valid_encryption_algorithm
}

valid_encryption_algorithm if {
    input.config.protocol.encryption.algorithm in [
        "aes-256-gcm",
        "aes-256-cbc",
        "chacha20-poly1305"
    ]
}

# ============================================================================
# AUTHENTICATION POLICIES
# ============================================================================

# Validate authentication configuration
valid_authentication if {
    # Internet mode must have authentication
    input.mode == "internet" implies authentication_enabled

    # Hybrid mode must support authentication
    input.mode == "hybrid" implies authentication_supported
}

authentication_enabled if {
    input.config.security.authentication.enabled == true
    valid_authentication_method
}

authentication_supported if {
    input.config.security.authentication.enabled in [true, "auto"]
    valid_authentication_method
}

valid_authentication_method if {
    input.config.security.authentication.method in [
        "jwt",
        "oauth2",
        "mtls",
        "hmac-sha256"
    ]
}

# ============================================================================
# TLS POLICIES
# ============================================================================

# Validate TLS configuration
valid_tls if {
    # Internet mode must have TLS
    input.mode == "internet" implies tls_enabled

    # Hybrid mode must support TLS
    input.mode == "hybrid" implies tls_supported

    # TLS version must be 1.2 or higher
    tls_enabled implies modern_tls_version
}

tls_enabled if {
    input.config.security.tls.enabled == true
}

tls_supported if {
    input.config.security.tls.enabled in [true, "auto"]
}

modern_tls_version if {
    input.config.security.tls.minVersion in ["TLS1.2", "TLS1.3"]
}

# ============================================================================
# BYZANTINE FAULT TOLERANCE POLICIES
# ============================================================================

# Validate Byzantine fault tolerance
valid_byzantine if {
    # Internet mode must have Byzantine tolerance
    input.mode == "internet" implies byzantine_enabled

    # Hybrid mode must support Byzantine tolerance
    input.mode == "hybrid" implies byzantine_supported
}

byzantine_enabled if {
    input.config.byzantine.enabled == true
    valid_byzantine_config
}

byzantine_supported if {
    input.config.byzantine.enabled in [true, "auto"]
    valid_byzantine_config
}

valid_byzantine_config if {
    input.config.byzantine.authentication.method != ""
    input.config.byzantine.validation.checksums == true
    input.config.byzantine.rateLimit.enabled == true
}

# ============================================================================
# ACCESS CONTROL POLICIES
# ============================================================================

# Validate IAM roles and permissions
valid_iam if {
    # Least privilege principle
    least_privilege_roles

    # No overly permissive policies
    no_wildcard_resources

    # MFA required for sensitive operations
    mfa_enforced
}

least_privilege_roles if {
    every role in input.iam_roles {
        scoped_permissions(role)
    }
}

scoped_permissions(role) if {
    # Roles should not have full admin access
    not admin_access(role)

    # Permissions should be scoped to specific resources
    every policy in role.policies {
        not wildcard_actions(policy)
    }
}

admin_access(role) if {
    some policy in role.policies
    policy.actions[_] == "*"
    policy.resources[_] == "*"
}

wildcard_actions(policy) if {
    policy.actions[_] == "*"
}

no_wildcard_resources if {
    every role in input.iam_roles {
        every policy in role.policies {
            not wildcard_resources(policy)
        }
    }
}

wildcard_resources(policy) if {
    policy.resources[_] == "*"
    sensitive_action(policy.actions[_])
}

sensitive_action(action) if {
    action in [
        "iam:*",
        "s3:DeleteBucket",
        "ec2:TerminateInstances",
        "rds:DeleteDBInstance"
    ]
}

mfa_enforced if {
    input.security_config.mfa_required == true
}

# ============================================================================
# COMPLIANCE POLICIES
# ============================================================================

# Validate compliance requirements
valid_compliance if {
    # Logging must be enabled
    logging_enabled

    # Audit trail must be configured
    audit_trail_configured

    # Data retention policies
    data_retention_compliant
}

logging_enabled if {
    input.config.logging.enabled == true
    input.config.logging.level in ["info", "warn", "error"]
}

audit_trail_configured if {
    input.monitoring.enabled == true
    input.monitoring.metrics != []
}

data_retention_compliant if {
    input.config.logging.maxAge >= 30
    input.monitoring.prometheus_retention_days >= 30
}

# ============================================================================
# VULNERABILITY POLICIES
# ============================================================================

# Check for known vulnerabilities
no_critical_vulnerabilities if {
    # No CVEs with CVSS >= 9.0
    not has_critical_cve

    # No known exploitable vulnerabilities
    not has_exploitable_vuln

    # Security patches up to date
    security_patches_current
}

has_critical_cve if {
    some vuln in input.vulnerabilities
    vuln.cvss_score >= 9.0
}

has_exploitable_vuln if {
    some vuln in input.vulnerabilities
    vuln.exploitable == true
}

security_patches_current if {
    input.last_security_patch_date != null
    time.now_ns() - time.parse_rfc3339_ns(input.last_security_patch_date) < 2592000000000000  # 30 days in nanoseconds
}

# ============================================================================
# OVERALL VALIDATION
# ============================================================================

# Comprehensive security validation
valid_security_configuration if {
    valid_network_security
    valid_encryption
    valid_authentication
    valid_tls
    valid_byzantine
    valid_iam
    valid_compliance
}

# ============================================================================
# VIOLATION REPORTING
# ============================================================================

# Collect all violations
violations[msg] {
    input.action == "create"
    not valid_network_security
    msg := "Network security configuration is invalid"
}

violations[msg] {
    input.mode == "internet"
    not encryption_enabled
    msg := "Encryption must be enabled for internet mode"
}

violations[msg] {
    input.mode == "internet"
    not authentication_enabled
    msg := "Authentication must be enabled for internet mode"
}

violations[msg] {
    input.mode == "internet"
    not tls_enabled
    msg := "TLS must be enabled for internet mode"
}

violations[msg] {
    input.mode == "internet"
    not byzantine_enabled
    msg := "Byzantine fault tolerance must be enabled for internet mode"
}

violations[msg] {
    tls_enabled
    not modern_tls_version
    msg := "TLS version must be 1.2 or higher"
}

violations[msg] {
    not least_privilege_roles
    msg := "IAM roles violate least privilege principle"
}

violations[msg] {
    has_critical_cve
    msg := sprintf("Critical CVE detected: %v", [input.vulnerabilities])
}

violations[msg] {
    not logging_enabled
    msg := "Logging must be enabled for compliance"
}

violations[msg] {
    some sg in input.security_groups
    unrestricted_ssh(sg)
    msg := sprintf("Security group %s has unrestricted SSH access", [sg.name])
}

violations[msg] {
    not security_patches_current
    msg := "Security patches are not up to date (must be within 30 days)"
}

# ============================================================================
# DECISION OUTPUT
# ============================================================================

# Final decision with details
decision := {
    "allow": allow,
    "violations": violations,
    "metadata": metadata
}
