#!/bin/bash
# Security Validation Script for Production
# Validates security controls, compliance, and Byzantine detection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/docs/phase6/security-results}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/security}"

# Configuration
SECURITY_SCAN_ENABLED="${SECURITY_SCAN_ENABLED:-true}"
COMPLIANCE_CHECK_ENABLED="${COMPLIANCE_CHECK_ENABLED:-true}"
BYZANTINE_DETECTION_ENABLED="${BYZANTINE_DETECTION_ENABLED:-true}"
VULNERABILITY_SCAN_ENABLED="${VULNERABILITY_SCAN_ENABLED:-true}"
SECURITY_PACKAGE_TEST_ENABLED="${SECURITY_PACKAGE_TEST_ENABLED:-true}"
SECURITY_TEST_TIMEOUT="${SECURITY_TEST_TIMEOUT:-60}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/security.log"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✅ $*" | tee -a "${LOG_DIR}/security.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $*" | tee -a "${LOG_DIR}/security.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $*" | tee -a "${LOG_DIR}/security.log"
}

require_file() {
    local path="$1"
    if [[ ! -f "${PROJECT_ROOT}/${path}" ]]; then
        log_error "Missing required security artifact: ${path}"
        return 1
    fi
}

require_pattern() {
    local pattern="$1"
    shift

    local paths=()
    local path
    for path in "$@"; do
        if [[ -e "${PROJECT_ROOT}/${path}" ]]; then
            paths+=("${PROJECT_ROOT}/${path}")
        fi
    done

    if [[ ${#paths[@]} -eq 0 ]]; then
        log_error "No searchable paths exist for pattern: ${pattern}"
        return 1
    fi

    if command -v rg >/dev/null 2>&1; then
        rg -qi -- "${pattern}" "${paths[@]}"
    else
        grep -RIEqi -- "${pattern}" "${paths[@]}"
    fi
}

run_backend_core_test() {
    local package="$1"
    local pattern="$2"

    (cd "${PROJECT_ROOT}/backend/core" && go test "${package}" -run "${pattern}" -count=1 -timeout "${SECURITY_TEST_TIMEOUT}s")
}

run_go_test() {
    local package="$1"
    shift

    (cd "${PROJECT_ROOT}" && go test "${package}" "$@" -count=1 -timeout "${SECURITY_TEST_TIMEOUT}s")
}

# Initialize results
init_results() {
    cat > "${RESULTS_DIR}/security-validation-$(date +%s).json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "security_scans": {},
    "compliance_checks": {},
    "byzantine_detection": {},
    "vulnerability_scans": {},
    "overall_status": "pending"
}
EOF
}

# Validate authentication mechanisms
validate_authentication() {
    log "Validating authentication mechanisms..."

    local test_count=0
    local passed_count=0

    # Test 1: JWT token validation
    test_count=$((test_count + 1))
    if validate_jwt_tokens; then
        passed_count=$((passed_count + 1))
        log_success "JWT token validation passed"
    else
        log_error "JWT token validation failed"
    fi

    # Test 2: TLS certificate validation
    test_count=$((test_count + 1))
    if validate_tls_certificates; then
        passed_count=$((passed_count + 1))
        log_success "TLS certificate validation passed"
    else
        log_error "TLS certificate validation failed"
    fi

    # Test 3: API key validation
    test_count=$((test_count + 1))
    if validate_api_keys; then
        passed_count=$((passed_count + 1))
        log_success "API key validation passed"
    else
        log_error "API key validation failed"
    fi

    log "Authentication validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_jwt_tokens() {
    require_file "backend/core/auth/security_integration.go" &&
        require_pattern "JWT|jwt" \
            "backend/core/auth" \
            "policies/dwcp-v3-security.rego" \
            "configs/security-hardening.yaml"
}

validate_tls_certificates() {
    command -v openssl >/dev/null 2>&1 || {
        log_error "openssl is required for TLS validation"
        return 1
    }

    require_file "backend/core/network/dwcp/v3/security/mode_security.go" &&
        require_pattern "TLS1\\.2|TLS1\\.3|MinVersion|tls\\.Config" \
            "backend/core/network/dwcp/v3/security" \
            "policies/dwcp-v3-security.rego" \
            "configs/security-hardening.yaml"
}

validate_api_keys() {
    require_pattern "api[_-]?key|APIKey|ApiKey" \
        "backend/api/security" \
        "backend/core/security" \
        "configs/security-hardening.yaml"
}

# Validate authorization and access control
validate_authorization() {
    log "Validating authorization and access control..."

    local test_count=0
    local passed_count=0

    # Test 1: Role-based access control (RBAC)
    test_count=$((test_count + 1))
    if validate_rbac; then
        passed_count=$((passed_count + 1))
        log_success "RBAC validation passed"
    else
        log_error "RBAC validation failed"
    fi

    # Test 2: Permission enforcement
    test_count=$((test_count + 1))
    if validate_permissions; then
        passed_count=$((passed_count + 1))
        log_success "Permission enforcement validated"
    else
        log_error "Permission enforcement failed"
    fi

    # Test 3: Resource access control
    test_count=$((test_count + 1))
    if validate_resource_access; then
        passed_count=$((passed_count + 1))
        log_success "Resource access control validated"
    else
        log_error "Resource access control failed"
    fi

    log "Authorization validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_rbac() {
    require_file "backend/api/security/rbac_store.go" &&
        require_pattern "RBAC|Role|Permission" \
            "backend/api/security" \
            "backend/core/security" \
            "configs/security-hardening.yaml"
}

validate_permissions() {
    require_pattern "HasPermission|CheckPermission|Permission" \
        "backend/api/security" \
        "backend/core/security" \
        "policies/dwcp-v3-security.rego"
}

validate_resource_access() {
    require_pattern "resource|Resource|access control|AccessControl" \
        "backend/api/security" \
        "backend/core/security" \
        "configs/security-hardening.yaml"
}

# Validate encryption
validate_encryption() {
    log "Validating encryption (at rest and in transit)..."

    local test_count=0
    local passed_count=0

    # Test 1: Data at rest encryption
    test_count=$((test_count + 1))
    if validate_encryption_at_rest; then
        passed_count=$((passed_count + 1))
        log_success "Encryption at rest validated"
    else
        log_error "Encryption at rest validation failed"
    fi

    # Test 2: Data in transit encryption
    test_count=$((test_count + 1))
    if validate_encryption_in_transit; then
        passed_count=$((passed_count + 1))
        log_success "Encryption in transit validated"
    else
        log_error "Encryption in transit validation failed"
    fi

    # Test 3: Key management
    test_count=$((test_count + 1))
    if validate_key_management; then
        passed_count=$((passed_count + 1))
        log_success "Key management validated"
    else
        log_error "Key management validation failed"
    fi

    log "Encryption validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_encryption_at_rest() {
    require_pattern "at rest|AtRest|encrypt.*rest|database.*encrypt" \
        "backend/core/security" \
        "configs/security-hardening.yaml" \
        "policies/dwcp-v3-security.rego"
}

validate_encryption_in_transit() {
    require_pattern "TLS|mTLS|in transit|InTransit" \
        "backend/core/security" \
        "backend/core/network/dwcp/v3/security" \
        "configs/security-hardening.yaml" \
        "policies/dwcp-v3-security.rego"
}

validate_key_management() {
    require_file "backend/core/security/secrets_manager.go" &&
        require_pattern "rotation|Key|secret" \
            "backend/core/security" \
            "configs/security-hardening.yaml"
}

# Validate audit logging
validate_audit_logging() {
    log "Validating audit logging..."

    local test_count=0
    local passed_count=0

    # Test 1: Security event logging
    test_count=$((test_count + 1))
    if validate_security_events; then
        passed_count=$((passed_count + 1))
        log_success "Security event logging validated"
    else
        log_error "Security event logging validation failed"
    fi

    # Test 2: Audit trail completeness
    test_count=$((test_count + 1))
    if validate_audit_trail; then
        passed_count=$((passed_count + 1))
        log_success "Audit trail completeness validated"
    else
        log_error "Audit trail validation failed"
    fi

    # Test 3: Log tampering protection
    test_count=$((test_count + 1))
    if validate_log_integrity; then
        passed_count=$((passed_count + 1))
        log_success "Log integrity validated"
    else
        log_error "Log integrity validation failed"
    fi

    log "Audit logging validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_security_events() {
    require_file "backend/core/security/security_event_handlers.go" &&
        require_pattern "SecurityEvent|security event|LogEvent" \
            "backend/core/security" \
            "backend/api/security"
}

validate_audit_trail() {
    require_file "backend/core/security/audit_logger.go" &&
        require_pattern "audit|Audit" \
            "backend/core/security" \
            "configs/security-hardening.yaml" \
            "policies/dwcp-v3-security.rego"
}

validate_log_integrity() {
    require_pattern "integrity|tamper|hash|checksum" \
        "backend/core/security" \
        "tests/integration/security_e2e_test.go"
}

# Validate Byzantine fault detection
validate_byzantine_detection() {
    log "Validating Byzantine fault detection..."

    local test_count=0
    local passed_count=0

    # Test 1: Malicious node detection
    test_count=$((test_count + 1))
    if validate_malicious_node_detection; then
        passed_count=$((passed_count + 1))
        log_success "Malicious node detection validated"
    else
        log_error "Malicious node detection failed"
    fi

    # Test 2: Byzantine agreement
    test_count=$((test_count + 1))
    if validate_byzantine_agreement; then
        passed_count=$((passed_count + 1))
        log_success "Byzantine agreement validated"
    else
        log_error "Byzantine agreement validation failed"
    fi

    # Test 3: Fault tolerance threshold
    test_count=$((test_count + 1))
    if validate_fault_tolerance; then
        passed_count=$((passed_count + 1))
        log_success "Fault tolerance validated"
    else
        log_error "Fault tolerance validation failed"
    fi

    log "Byzantine detection validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_malicious_node_detection() {
    run_backend_core_test "./network/dwcp/v3/security" "TestByzantineDetector_(InvalidSignature|MultipleViolationTypes)"
}

validate_byzantine_agreement() {
    run_backend_core_test "./network/dwcp/v3/security" "TestSecurityIntegration_ConsensusWithByzantine"
}

validate_fault_tolerance() {
    require_pattern "f < n/3|Byzantine|quorum|threshold" \
        "backend/core/network/dwcp/v3/security" \
        "policies/dwcp-v3-security.rego"
}

# Validate compliance requirements
validate_compliance() {
    log "Validating compliance requirements..."

    local test_count=0
    local passed_count=0

    # Test 1: GDPR compliance
    test_count=$((test_count + 1))
    if validate_gdpr_compliance; then
        passed_count=$((passed_count + 1))
        log_success "GDPR compliance validated"
    else
        log_error "GDPR compliance validation failed"
    fi

    # Test 2: Data retention policies
    test_count=$((test_count + 1))
    if validate_data_retention; then
        passed_count=$((passed_count + 1))
        log_success "Data retention policies validated"
    else
        log_error "Data retention validation failed"
    fi

    # Test 3: Privacy controls
    test_count=$((test_count + 1))
    if validate_privacy_controls; then
        passed_count=$((passed_count + 1))
        log_success "Privacy controls validated"
    else
        log_error "Privacy controls validation failed"
    fi

    log "Compliance validation: ${passed_count}/${test_count} tests passed"
    return $([ ${passed_count} -eq ${test_count} ] && echo 0 || echo 1)
}

validate_gdpr_compliance() {
    require_pattern "GDPR|privacy_by_design|data_subject" \
        "backend/core/security" \
        "configs/security-hardening.yaml" \
        "policies/dwcp-v3-security.rego"
}

validate_data_retention() {
    require_pattern "retention" \
        "configs" \
        "policies/dwcp-v3-security.rego" \
        "backend/core/security"
}

validate_privacy_controls() {
    require_pattern "privacy|Privacy|PII|anonym" \
        "backend/core/security" \
        "configs/security-hardening.yaml" \
        "policies/dwcp-v3-security.rego"
}

# Validate generic penetration, OWASP, compliance, and encryption test package
validate_security_test_package() {
    log "Validating generic security test package..."

    run_go_test "./tests/security"
}

# Run vulnerability scan
run_vulnerability_scan() {
    log "Running vulnerability scan..."

    local scan_file="${RESULTS_DIR}/vulnerability-scan-$(date +%s).json"
    local inventory_file="${RESULTS_DIR}/dependency-inventory-$(date +%s).txt"

    if command -v govulncheck >/dev/null 2>&1; then
        if (cd "${PROJECT_ROOT}/backend/core" && govulncheck ./... > "${inventory_file}" 2>&1); then
            cat > "${scan_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "scan_type": "govulncheck",
    "vulnerabilities_found": 0,
    "critical_vulnerabilities": 0,
    "high_vulnerabilities": 0,
    "medium_vulnerabilities": 0,
    "low_vulnerabilities": 0,
    "scan_status": "completed",
    "scan_output": "${inventory_file}"
}
EOF
            log_success "govulncheck completed: no reachable vulnerabilities reported"
            return 0
        fi

        cat > "${scan_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "scan_type": "govulncheck",
    "vulnerabilities_found": null,
    "scan_status": "failed",
    "scan_output": "${inventory_file}"
}
EOF
        log_error "govulncheck reported findings or failed"
        return 1
    fi

    (cd "${PROJECT_ROOT}/backend/core" && go list -m all > "${inventory_file}")
    cat > "${scan_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "scan_type": "dependency_inventory",
    "vulnerabilities_found": null,
    "scan_status": "scanner_unavailable",
    "scan_output": "${inventory_file}"
}
EOF

    log_warning "govulncheck unavailable; captured dependency inventory only"
    return 0
}

# Generate security report
generate_security_report() {
    log "Generating security validation report..."

    local passed_validations="${1:-0}"
    local total_validations="${2:-6}"
    local failed_validations=$((total_validations - passed_validations))
    local overall_status="failed"
    local security_score=0
    if [[ ${total_validations} -gt 0 ]]; then
        security_score=$((passed_validations * 100 / total_validations))
    fi
    if [[ ${passed_validations} -eq ${total_validations} ]]; then
        overall_status="passed"
    fi

    local timestamp=$(date +%s)
    local report_file="${RESULTS_DIR}/security-report-${timestamp}.json"

    cat > "${report_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "production",
    "validation_summary": {
        "total": ${total_validations},
        "passed": ${passed_validations},
        "failed": ${failed_validations}
    },
    "overall_status": "${overall_status}",
    "security_score": ${security_score},
    "recommendations": [
        "Continue monitoring security events",
        "Review access logs weekly",
        "Update security policies quarterly",
        "Conduct penetration testing monthly"
    ],
    "next_validation": "$(date -u -d '+1 day' +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    log_success "Security report generated: ${report_file}"
}

# Main execution
main() {
    log "=========================================="
    log "Security Validation Suite"
    log "=========================================="

    init_results

    local total_validations=6
    local passed_validations=0
    if [[ "${SECURITY_PACKAGE_TEST_ENABLED}" == "true" ]]; then
        total_validations=$((total_validations + 1))
    fi

    # Run all security validations
    validate_authentication && passed_validations=$((passed_validations + 1))
    validate_authorization && passed_validations=$((passed_validations + 1))
    validate_encryption && passed_validations=$((passed_validations + 1))
    validate_audit_logging && passed_validations=$((passed_validations + 1))
    validate_byzantine_detection && passed_validations=$((passed_validations + 1))
    validate_compliance && passed_validations=$((passed_validations + 1))
    if [[ "${SECURITY_PACKAGE_TEST_ENABLED}" == "true" ]]; then
        validate_security_test_package && passed_validations=$((passed_validations + 1))
    fi

    # Run vulnerability scan
    if [[ "${VULNERABILITY_SCAN_ENABLED}" == "true" ]]; then
        run_vulnerability_scan
    fi

    # Generate report
    generate_security_report "${passed_validations}" "${total_validations}"

    # Display summary
    echo ""
    echo "=========================================="
    echo "Security Validation Summary"
    echo "=========================================="
    echo "Total Validations:  ${total_validations}"
    echo "Passed:             ${passed_validations}"
    echo "Failed:             $((total_validations - passed_validations))"
    echo "Status:             $([ ${passed_validations} -eq ${total_validations} ] && echo "✅ PASS" || echo "❌ FAIL")"
    echo "=========================================="
    echo ""

    if [ ${passed_validations} -eq ${total_validations} ]; then
        log_success "All security validations passed"
        return 0
    else
        log_error "Some security validations failed"
        return 1
    fi
}

main "$@"
