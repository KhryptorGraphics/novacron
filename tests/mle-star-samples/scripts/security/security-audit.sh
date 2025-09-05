#!/bin/bash

# NovaCron Security Audit Script
# Comprehensive automated security scanning for different environments and security levels
# Usage: ./security-audit.sh [LEVEL] [ENVIRONMENT] [OPTIONS]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_CONFIG_DIR="$SCRIPT_DIR/config"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$RESULTS_DIR/security-audit-$(date +%Y%m%d-%H%M%S).log"

# Security levels
declare -A SECURITY_LEVELS=(
    ["basic"]="Basic security scanning - code and dependencies only"
    ["standard"]="Standard scanning - includes infrastructure and configuration"
    ["enhanced"]="Enhanced scanning - includes network and database security"
    ["enterprise"]="Enterprise scanning - comprehensive security audit with compliance"
)

# Environment configurations
declare -A ENVIRONMENTS=(
    ["development"]="Development environment - relaxed security policies"
    ["staging"]="Staging environment - production-like security policies"
    ["production"]="Production environment - strict security policies"
    ["cloud"]="Cloud environment - cloud-specific security checks"
)

# Default values
AUDIT_LEVEL="${1:-standard}"
ENVIRONMENT="${2:-development}"
PARALLEL_SCANS=true
GENERATE_REPORT=true
SEND_NOTIFICATIONS=false
EXPORT_FORMATS=("json" "html" "csv")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Usage information
show_usage() {
    echo "NovaCron Security Audit Script"
    echo ""
    echo "Usage: $0 [LEVEL] [ENVIRONMENT] [OPTIONS]"
    echo ""
    echo "Security Levels:"
    for level in "${!SECURITY_LEVELS[@]}"; do
        echo "  $level - ${SECURITY_LEVELS[$level]}"
    done
    echo ""
    echo "Environments:"
    for env in "${!ENVIRONMENTS[@]}"; do
        echo "  $env - ${ENVIRONMENTS[$env]}"
    done
    echo ""
    echo "Options:"
    echo "  --no-parallel    Disable parallel scanning"
    echo "  --no-report      Skip report generation"
    echo "  --notify         Enable notifications"
    echo "  --format FORMAT  Export format (json,html,csv,all)"
    echo "  --config-dir DIR Custom configuration directory"
    echo "  --output-dir DIR Custom output directory"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 basic development"
    echo "  $0 enterprise production --notify --format all"
    echo "  $0 enhanced staging --no-parallel --config-dir ./custom-config"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-parallel)
                PARALLEL_SCANS=false
                shift
                ;;
            --no-report)
                GENERATE_REPORT=false
                shift
                ;;
            --notify)
                SEND_NOTIFICATIONS=true
                shift
                ;;
            --format)
                if [[ "$2" == "all" ]]; then
                    EXPORT_FORMATS=("json" "html" "csv" "xml" "pdf")
                else
                    IFS=',' read -ra EXPORT_FORMATS <<< "$2"
                fi
                shift 2
                ;;
            --config-dir)
                AUDIT_CONFIG_DIR="$2"
                shift 2
                ;;
            --output-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Create necessary directories
setup_directories() {
    log_info "Setting up audit directories..."
    
    mkdir -p "$RESULTS_DIR"/{raw,processed,reports,artifacts}
    mkdir -p "$AUDIT_CONFIG_DIR"
    mkdir -p "$RESULTS_DIR/evidence"
    
    # Create session directory
    SESSION_DIR="$RESULTS_DIR/session-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$SESSION_DIR"
    
    log_success "Audit directories created: $SESSION_DIR"
}

# Check prerequisites and install missing tools
check_prerequisites() {
    log_info "Checking security scanning prerequisites..."
    
    local missing_tools=()
    local required_tools=(
        "git:Git version control"
        "node:Node.js runtime" 
        "npm:Node package manager"
        "docker:Docker container platform"
        "curl:HTTP client"
        "jq:JSON processor"
        "openssl:SSL/TLS toolkit"
        "nmap:Network mapper"
        "sqlmap:SQL injection tester"
    )
    
    for tool_info in "${required_tools[@]}"; do
        IFS=':' read -r tool desc <<< "$tool_info"
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool ($desc)")
            log_warning "Missing tool: $tool"
        fi
    done
    
    # Install missing tools automatically where possible
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_info "Installing missing security tools..."
        install_security_tools "${missing_tools[@]}"
    fi
    
    # Verify Node.js security tools
    check_nodejs_security_tools
    
    log_success "Prerequisites check completed"
}

# Install security scanning tools
install_security_tools() {
    log_info "Installing security scanning tools..."
    
    # Install npm security tools
    if command -v npm &> /dev/null; then
        log_info "Installing Node.js security tools..."
        npm install -g npm-audit-resolver eslint-plugin-security semgrep retire 2>/dev/null || log_warning "Some npm tools failed to install"
    fi
    
    # Install Python security tools if Python is available
    if command -v pip3 &> /dev/null; then
        log_info "Installing Python security tools..."
        pip3 install --user bandit safety pylint 2>/dev/null || log_warning "Some Python tools failed to install"
    fi
    
    # Download and setup additional tools
    setup_additional_tools
}

# Setup additional security tools
setup_additional_tools() {
    local tools_dir="$SCRIPT_DIR/tools"
    mkdir -p "$tools_dir"
    
    # Download Trivy for container scanning
    if [[ ! -f "$tools_dir/trivy" ]]; then
        log_info "Installing Trivy container scanner..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b "$tools_dir" 2>/dev/null || log_warning "Trivy installation failed"
    fi
    
    # Download TruffleHog for secret scanning
    if [[ ! -f "$tools_dir/trufflehog" ]]; then
        log_info "Installing TruffleHog secret scanner..."
        curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b "$tools_dir" 2>/dev/null || log_warning "TruffleHog installation failed"
    fi
    
    # Make tools executable
    chmod +x "$tools_dir"/* 2>/dev/null || true
    
    # Add tools to PATH for this session
    export PATH="$tools_dir:$PATH"
}

# Check Node.js security tools
check_nodejs_security_tools() {
    if [[ -f "$PROJECT_ROOT/package.json" ]]; then
        log_info "Verifying Node.js project security tools..."
        
        cd "$PROJECT_ROOT"
        
        # Install audit tools if not present
        if ! npm list --depth=0 audit &> /dev/null; then
            npm install --save-dev audit 2>/dev/null || true
        fi
        
        cd - > /dev/null
    fi
}

# Load configuration for audit level and environment
load_configuration() {
    log_info "Loading configuration for level: $AUDIT_LEVEL, environment: $ENVIRONMENT"
    
    # Create default config if not exists
    local config_file="$AUDIT_CONFIG_DIR/${AUDIT_LEVEL}-${ENVIRONMENT}.json"
    
    if [[ ! -f "$config_file" ]]; then
        create_default_config "$config_file"
    fi
    
    # Load scan configuration
    if command -v jq &> /dev/null && [[ -f "$config_file" ]]; then
        SCAN_CONFIG=$(cat "$config_file")
        log_success "Configuration loaded from $config_file"
    else
        log_warning "Using default configuration"
        SCAN_CONFIG='{"enabled_scans":["static","dependency","secrets"],"thresholds":{"critical":0,"high":5}}'
    fi
}

# Create default configuration file
create_default_config() {
    local config_file="$1"
    
    log_info "Creating default configuration: $config_file"
    
    case "$AUDIT_LEVEL" in
        "basic")
            cat > "$config_file" << 'EOF'
{
  "level": "basic",
  "environment": "development",
  "enabled_scans": [
    "static_analysis",
    "dependency_check",
    "secrets_scan"
  ],
  "thresholds": {
    "critical": 0,
    "high": 10,
    "medium": 50,
    "low": 100
  },
  "scan_timeout": 600,
  "parallel_execution": true,
  "export_formats": ["json"],
  "notification_channels": [],
  "compliance_frameworks": []
}
EOF
            ;;
        "standard")
            cat > "$config_file" << 'EOF'
{
  "level": "standard",
  "environment": "development", 
  "enabled_scans": [
    "static_analysis",
    "dependency_check",
    "secrets_scan",
    "container_scan",
    "infrastructure_scan"
  ],
  "thresholds": {
    "critical": 0,
    "high": 5,
    "medium": 25,
    "low": 75
  },
  "scan_timeout": 1200,
  "parallel_execution": true,
  "export_formats": ["json", "html"],
  "notification_channels": ["slack"],
  "compliance_frameworks": ["SOC2"]
}
EOF
            ;;
        "enhanced")
            cat > "$config_file" << 'EOF'
{
  "level": "enhanced",
  "environment": "staging",
  "enabled_scans": [
    "static_analysis",
    "dependency_check", 
    "secrets_scan",
    "container_scan",
    "infrastructure_scan",
    "network_scan",
    "database_scan"
  ],
  "thresholds": {
    "critical": 0,
    "high": 2,
    "medium": 10,
    "low": 30
  },
  "scan_timeout": 1800,
  "parallel_execution": true,
  "export_formats": ["json", "html", "csv"],
  "notification_channels": ["slack", "email"],
  "compliance_frameworks": ["SOC2", "GDPR"]
}
EOF
            ;;
        "enterprise")
            cat > "$config_file" << 'EOF'
{
  "level": "enterprise",
  "environment": "production",
  "enabled_scans": [
    "static_analysis",
    "dependency_check",
    "secrets_scan",
    "container_scan",
    "infrastructure_scan", 
    "network_scan",
    "database_scan",
    "compliance_check",
    "penetration_test"
  ],
  "thresholds": {
    "critical": 0,
    "high": 0,
    "medium": 5,
    "low": 15
  },
  "scan_timeout": 3600,
  "parallel_execution": true,
  "export_formats": ["json", "html", "csv", "xml", "pdf"],
  "notification_channels": ["slack", "email", "pagerduty"],
  "compliance_frameworks": ["SOC2", "GDPR", "NIST", "ISO27001", "PCI_DSS"]
}
EOF
            ;;
    esac
    
    log_success "Default configuration created: $config_file"
}

# Execute security scans based on configuration
execute_security_scans() {
    log_info "Starting security scans for level: $AUDIT_LEVEL"
    
    local start_time=$(date +%s)
    local scan_results=()
    local failed_scans=()
    
    # Parse enabled scans from config
    local enabled_scans
    if command -v jq &> /dev/null; then
        enabled_scans=($(echo "$SCAN_CONFIG" | jq -r '.enabled_scans[]' 2>/dev/null || echo "static_analysis dependency_check secrets_scan"))
    else
        enabled_scans=("static_analysis" "dependency_check" "secrets_scan")
    fi
    
    log_info "Enabled scans: ${enabled_scans[*]}"
    
    # Execute scans
    if [[ "$PARALLEL_SCANS" == "true" ]]; then
        execute_parallel_scans "${enabled_scans[@]}"
    else
        execute_sequential_scans "${enabled_scans[@]}"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Security scans completed in ${duration} seconds"
}

# Execute scans in parallel
execute_parallel_scans() {
    local scans=("$@")
    local pids=()
    
    log_info "Executing ${#scans[@]} scans in parallel..."
    
    for scan in "${scans[@]}"; do
        execute_scan "$scan" "$SESSION_DIR/raw/${scan}-results.json" &
        pids+=($!)
        log_info "Started scan: $scan (PID: $!)"
    done
    
    # Wait for all scans to complete
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            log_success "Scan completed successfully (PID: $pid)"
        else
            log_error "Scan failed (PID: $pid)"
        fi
    done
}

# Execute scans sequentially  
execute_sequential_scans() {
    local scans=("$@")
    
    log_info "Executing ${#scans[@]} scans sequentially..."
    
    for scan in "${scans[@]}"; do
        log_info "Starting scan: $scan"
        if execute_scan "$scan" "$SESSION_DIR/raw/${scan}-results.json"; then
            log_success "Scan completed: $scan"
        else
            log_error "Scan failed: $scan"
        fi
    done
}

# Execute individual security scan
execute_scan() {
    local scan_type="$1"
    local output_file="$2"
    
    case "$scan_type" in
        "static_analysis")
            execute_static_analysis_scan "$output_file"
            ;;
        "dependency_check")
            execute_dependency_scan "$output_file"
            ;;
        "secrets_scan")
            execute_secrets_scan "$output_file"
            ;;
        "container_scan")
            execute_container_scan "$output_file"
            ;;
        "infrastructure_scan")
            execute_infrastructure_scan "$output_file"
            ;;
        "network_scan")
            execute_network_scan "$output_file"
            ;;
        "database_scan")
            execute_database_scan "$output_file"
            ;;
        "compliance_check")
            execute_compliance_scan "$output_file"
            ;;
        *)
            log_error "Unknown scan type: $scan_type"
            return 1
            ;;
    esac
}

# Static Analysis Scan
execute_static_analysis_scan() {
    local output_file="$1"
    log_info "Executing static analysis scan..."
    
    cd "$PROJECT_ROOT"
    
    local results='{"scan_type":"static_analysis","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","findings":[]}'
    
    # ESLint security scan (if Node.js project)
    if [[ -f "package.json" ]]; then
        log_info "Running ESLint security analysis..."
        if command -v eslint &> /dev/null; then
            eslint --ext .js,.ts,.jsx,.tsx --format json . 2>/dev/null | jq -r '.' > "$SESSION_DIR/raw/eslint-results.json" || true
        fi
    fi
    
    # Bandit scan (if Python project)  
    if [[ -f "requirements.txt" ]] || [[ -f "setup.py" ]] || find . -name "*.py" -type f | head -1 | grep -q "\.py$"; then
        log_info "Running Bandit security analysis..."
        if command -v bandit &> /dev/null; then
            bandit -r . -f json -o "$SESSION_DIR/raw/bandit-results.json" 2>/dev/null || true
        fi
    fi
    
    # Semgrep scan
    if command -v semgrep &> /dev/null; then
        log_info "Running Semgrep analysis..."
        semgrep --config=auto --json --output="$SESSION_DIR/raw/semgrep-results.json" . 2>/dev/null || true
    fi
    
    # Aggregate results
    aggregate_static_analysis_results "$output_file"
    
    cd - > /dev/null
}

# Aggregate static analysis results
aggregate_static_analysis_results() {
    local output_file="$1"
    local findings=()
    local total_issues=0
    
    # Process ESLint results
    if [[ -f "$SESSION_DIR/raw/eslint-results.json" ]]; then
        local eslint_count=$(jq 'length' "$SESSION_DIR/raw/eslint-results.json" 2>/dev/null || echo 0)
        total_issues=$((total_issues + eslint_count))
    fi
    
    # Process Bandit results  
    if [[ -f "$SESSION_DIR/raw/bandit-results.json" ]]; then
        local bandit_count=$(jq '.results | length' "$SESSION_DIR/raw/bandit-results.json" 2>/dev/null || echo 0)
        total_issues=$((total_issues + bandit_count))
    fi
    
    # Process Semgrep results
    if [[ -f "$SESSION_DIR/raw/semgrep-results.json" ]]; then
        local semgrep_count=$(jq '.results | length' "$SESSION_DIR/raw/semgrep-results.json" 2>/dev/null || echo 0)
        total_issues=$((total_issues + semgrep_count))
    fi
    
    # Create summary
    cat > "$output_file" << EOF
{
  "scan_type": "static_analysis",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {
    "total_issues": $total_issues,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "tools": {
    "eslint": {"enabled": $(command -v eslint &>/dev/null && echo true || echo false)},
    "bandit": {"enabled": $(command -v bandit &>/dev/null && echo true || echo false)},
    "semgrep": {"enabled": $(command -v semgrep &>/dev/null && echo true || echo false)}
  },
  "findings": []
}
EOF
    
    log_success "Static analysis completed: $total_issues issues found"
}

# Dependency Vulnerability Scan
execute_dependency_scan() {
    local output_file="$1"
    log_info "Executing dependency vulnerability scan..."
    
    cd "$PROJECT_ROOT"
    
    local total_vulns=0
    
    # npm audit (if Node.js project)
    if [[ -f "package.json" ]]; then
        log_info "Running npm audit..."
        npm audit --json > "$SESSION_DIR/raw/npm-audit.json" 2>/dev/null || true
        if [[ -f "$SESSION_DIR/raw/npm-audit.json" ]]; then
            total_vulns=$(jq '.metadata.vulnerabilities.total // 0' "$SESSION_DIR/raw/npm-audit.json" 2>/dev/null || echo 0)
        fi
    fi
    
    # Python safety scan
    if [[ -f "requirements.txt" ]]; then
        log_info "Running Safety scan..."
        if command -v safety &> /dev/null; then
            safety check --json > "$SESSION_DIR/raw/safety-results.json" 2>/dev/null || true
        fi
    fi
    
    # Retire.js scan
    if command -v retire &> /dev/null; then
        log_info "Running Retire.js scan..."
        retire --json > "$SESSION_DIR/raw/retire-results.json" 2>/dev/null || true
    fi
    
    # Create summary
    cat > "$output_file" << EOF
{
  "scan_type": "dependency_check",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {
    "total_vulnerabilities": $total_vulns,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "package_managers": {
    "npm": $(test -f "package.json" && echo true || echo false),
    "pip": $(test -f "requirements.txt" && echo true || echo false)
  }
}
EOF
    
    log_success "Dependency scan completed: $total_vulns vulnerabilities found"
    
    cd - > /dev/null
}

# Secrets Scan
execute_secrets_scan() {
    local output_file="$1"
    log_info "Executing secrets scan..."
    
    cd "$PROJECT_ROOT"
    
    local secrets_found=0
    
    # TruffleHog scan
    if command -v trufflehog &> /dev/null; then
        log_info "Running TruffleHog secrets scan..."
        trufflehog filesystem . --json > "$SESSION_DIR/raw/trufflehog-results.json" 2>/dev/null || true
        if [[ -f "$SESSION_DIR/raw/trufflehog-results.json" ]]; then
            secrets_found=$(wc -l < "$SESSION_DIR/raw/trufflehog-results.json" 2>/dev/null || echo 0)
        fi
    fi
    
    # Git secrets scan
    if command -v git-secrets &> /dev/null; then
        log_info "Running git-secrets scan..."
        git secrets --scan > "$SESSION_DIR/raw/git-secrets-results.txt" 2>/dev/null || true
    fi
    
    # Manual pattern search
    log_info "Running pattern-based secrets scan..."
    execute_pattern_based_secrets_scan
    
    # Create summary
    cat > "$output_file" << EOF
{
  "scan_type": "secrets_scan",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {
    "total_secrets": $secrets_found,
    "high_confidence": 0,
    "medium_confidence": 0,
    "low_confidence": 0
  },
  "patterns_checked": [
    "api_keys", "passwords", "tokens", "certificates", "private_keys"
  ]
}
EOF
    
    log_success "Secrets scan completed: $secrets_found potential secrets found"
    
    cd - > /dev/null
}

# Pattern-based secrets scanning
execute_pattern_based_secrets_scan() {
    local patterns_file="$SESSION_DIR/raw/pattern-secrets.txt"
    
    # Common secret patterns
    local patterns=(
        "password\s*[:=]\s*['\"][^'\"]{8,}['\"]"
        "api[_-]?key\s*[:=]\s*['\"][^'\"]{16,}['\"]"
        "secret\s*[:=]\s*['\"][^'\"]{8,}['\"]"
        "token\s*[:=]\s*['\"][^'\"]{16,}['\"]"
        "-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----"
        "-----BEGIN CERTIFICATE-----"
        "sk_live_[0-9a-zA-Z]{24,}"
        "pk_live_[0-9a-zA-Z]{24,}"
    )
    
    for pattern in "${patterns[@]}"; do
        grep -r -i -E "$pattern" . --exclude-dir=.git --exclude-dir=node_modules \
             --exclude-dir=.venv --exclude="*.log" 2>/dev/null >> "$patterns_file" || true
    done
    
    if [[ -f "$patterns_file" ]]; then
        log_info "Pattern-based scan found $(wc -l < "$patterns_file") potential matches"
    fi
}

# Container Security Scan
execute_container_scan() {
    local output_file="$1"
    log_info "Executing container security scan..."
    
    local containers_scanned=0
    local total_vulns=0
    
    # Find Dockerfiles
    local dockerfiles=($(find "$PROJECT_ROOT" -name "Dockerfile*" -type f))
    
    if [[ ${#dockerfiles[@]} -gt 0 ]]; then
        log_info "Found ${#dockerfiles[@]} Dockerfile(s)"
        
        # Trivy scan
        if command -v trivy &> /dev/null; then
            for dockerfile in "${dockerfiles[@]}"; do
                log_info "Scanning Dockerfile: $dockerfile"
                local image_name="novacron-security-scan:$(basename "$dockerfile")"
                
                # Build image for scanning
                if docker build -t "$image_name" -f "$dockerfile" "$PROJECT_ROOT" &>/dev/null; then
                    trivy image --format json --output "$SESSION_DIR/raw/trivy-$(basename "$dockerfile").json" "$image_name" 2>/dev/null || true
                    containers_scanned=$((containers_scanned + 1))
                    
                    # Clean up image
                    docker rmi "$image_name" &>/dev/null || true
                fi
            done
        fi
        
        # Docker bench security
        if [[ -f "/usr/local/bin/docker-bench-security.sh" ]]; then
            log_info "Running Docker Bench Security..."
            /usr/local/bin/docker-bench-security.sh > "$SESSION_DIR/raw/docker-bench.txt" 2>/dev/null || true
        fi
    else
        log_info "No Dockerfiles found, skipping container scan"
    fi
    
    # Create summary
    cat > "$output_file" << EOF
{
  "scan_type": "container_scan",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {
    "containers_scanned": $containers_scanned,
    "total_vulnerabilities": $total_vulns,
    "dockerfiles_found": ${#dockerfiles[@]}
  }
}
EOF
    
    log_success "Container scan completed: $containers_scanned containers scanned"
}

# Infrastructure Security Scan
execute_infrastructure_scan() {
    local output_file="$1"
    log_info "Executing infrastructure security scan..."
    
    local issues_found=0
    
    # Terraform scan (if Terraform files exist)
    if find "$PROJECT_ROOT" -name "*.tf" -type f | head -1 | grep -q "\.tf$"; then
        log_info "Scanning Terraform configurations..."
        execute_terraform_scan
    fi
    
    # Kubernetes scan (if K8s manifests exist)
    if find "$PROJECT_ROOT" -name "*.yaml" -o -name "*.yml" | xargs grep -l "apiVersion.*v1" &>/dev/null; then
        log_info "Scanning Kubernetes manifests..."
        execute_kubernetes_scan
    fi
    
    # Cloud configuration scan
    execute_cloud_config_scan
    
    # Create summary
    cat > "$output_file" << EOF
{
  "scan_type": "infrastructure_scan", 
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {
    "total_issues": $issues_found,
    "terraform_files": $(find "$PROJECT_ROOT" -name "*.tf" | wc -l),
    "kubernetes_manifests": $(find "$PROJECT_ROOT" -name "*.yaml" -o -name "*.yml" | xargs grep -l "apiVersion" 2>/dev/null | wc -l || echo 0)
  }
}
EOF
    
    log_success "Infrastructure scan completed"
}

# Execute other scan types (stubs for now)
execute_network_scan() {
    local output_file="$1"
    log_info "Executing network security scan..."
    
    cat > "$output_file" << EOF
{
  "scan_type": "network_scan",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {"message": "Network scan not implemented in this version"}
}
EOF
}

execute_database_scan() {
    local output_file="$1"
    log_info "Executing database security scan..."
    
    cat > "$output_file" << EOF
{
  "scan_type": "database_scan",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {"message": "Database scan not implemented in this version"}
}
EOF
}

execute_compliance_scan() {
    local output_file="$1"
    log_info "Executing compliance scan..."
    
    cat > "$output_file" << EOF
{
  "scan_type": "compliance_check",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "summary": {"message": "Compliance scan not implemented in this version"}
}
EOF
}

# Placeholder functions for infrastructure scanning
execute_terraform_scan() {
    log_info "Terraform security scanning not implemented in this version"
}

execute_kubernetes_scan() {
    log_info "Kubernetes security scanning not implemented in this version"
}

execute_cloud_config_scan() {
    log_info "Cloud configuration scanning not implemented in this version"
}

# Generate consolidated security report
generate_security_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        log_info "Report generation skipped"
        return 0
    fi
    
    log_info "Generating security audit report..."
    
    local report_data='{"audit_metadata":{},"scan_results":{},"summary":{}}'
    
    # Collect audit metadata
    report_data=$(echo "$report_data" | jq --arg level "$AUDIT_LEVEL" \
                                          --arg env "$ENVIRONMENT" \
                                          --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                                          '.audit_metadata = {
                                            "level": $level,
                                            "environment": $env,
                                            "timestamp": $timestamp,
                                            "session_id": "'$(basename "$SESSION_DIR")'"
                                          }')
    
    # Aggregate scan results
    local total_issues=0
    local critical_issues=0
    local high_issues=0
    
    for result_file in "$SESSION_DIR"/raw/*-results.json; do
        if [[ -f "$result_file" ]]; then
            local scan_type=$(basename "$result_file" | sed 's/-results\.json$//')
            local scan_data=$(cat "$result_file")
            
            report_data=$(echo "$report_data" | jq --argjson scan "$scan_data" \
                                                  --arg type "$scan_type" \
                                                  '.scan_results[$type] = $scan')
            
            # Extract issue counts
            local issues=$(echo "$scan_data" | jq '.summary.total_issues // .summary.total_vulnerabilities // .summary.total_secrets // 0' 2>/dev/null || echo 0)
            total_issues=$((total_issues + issues))
        fi
    done
    
    # Add summary
    report_data=$(echo "$report_data" | jq --arg total "$total_issues" \
                                          --arg critical "$critical_issues" \
                                          --arg high "$high_issues" \
                                          '.summary = {
                                            "total_issues": ($total | tonumber),
                                            "critical_issues": ($critical | tonumber),
                                            "high_issues": ($high | tonumber),
                                            "risk_level": (if ($critical | tonumber) > 0 then "critical" 
                                                          elif ($high | tonumber) > 10 then "high"
                                                          elif ($total | tonumber) > 50 then "medium"
                                                          else "low" end)
                                          }')
    
    # Save consolidated report
    echo "$report_data" | jq '.' > "$SESSION_DIR/security-audit-report.json"
    
    # Generate reports in requested formats
    for format in "${EXPORT_FORMATS[@]}"; do
        generate_report_format "$format" "$report_data"
    done
    
    log_success "Security audit report generated: $SESSION_DIR/security-audit-report.json"
}

# Generate report in specific format
generate_report_format() {
    local format="$1"
    local data="$2"
    
    case "$format" in
        "json")
            # Already saved as JSON
            ;;
        "html")
            generate_html_report "$data"
            ;;
        "csv")
            generate_csv_report "$data"
            ;;
        "xml")
            generate_xml_report "$data"
            ;;
        *)
            log_warning "Unsupported report format: $format"
            ;;
    esac
}

# Generate HTML report
generate_html_report() {
    local data="$1"
    local html_file="$SESSION_DIR/security-audit-report.html"
    
    cat > "$html_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>NovaCron Security Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .critical { color: #dc3545; }
        .high { color: #fd7e14; }
        .medium { color: #ffc107; }
        .low { color: #28a745; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NovaCron Security Audit Report</h1>
        <p><strong>Level:</strong> AUDIT_LEVEL</p>
        <p><strong>Environment:</strong> ENVIRONMENT</p>
        <p><strong>Generated:</strong> TIMESTAMP</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>This automated security audit identified <strong>TOTAL_ISSUES</strong> potential security issues across multiple scan types.</p>
        <p><strong>Risk Level:</strong> <span class="RISK_CLASS">RISK_LEVEL</span></p>
    </div>
    
    <h2>Scan Results</h2>
    <table>
        <thead>
            <tr>
                <th>Scan Type</th>
                <th>Issues Found</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr><td colspan="3">Scan details would be populated here...</td></tr>
        </tbody>
    </table>
</body>
</html>
EOF
    
    # Replace placeholders with actual data
    sed -i "s/AUDIT_LEVEL/$AUDIT_LEVEL/g" "$html_file"
    sed -i "s/ENVIRONMENT/$ENVIRONMENT/g" "$html_file"
    sed -i "s/TIMESTAMP/$(date)/g" "$html_file"
    
    log_success "HTML report generated: $html_file"
}

# Generate CSV report
generate_csv_report() {
    local data="$1"
    local csv_file="$SESSION_DIR/security-audit-report.csv"
    
    cat > "$csv_file" << 'EOF'
Scan Type,Issues Found,Critical,High,Medium,Low,Status
EOF
    
    # Extract scan results and add to CSV
    echo "$data" | jq -r '.scan_results | to_entries[] | "\(.key),\(.value.summary.total_issues // 0),\(.value.summary.critical // 0),\(.value.summary.high // 0),\(.value.summary.medium // 0),\(.value.summary.low // 0),completed"' >> "$csv_file" 2>/dev/null || true
    
    log_success "CSV report generated: $csv_file"
}

# Generate XML report  
generate_xml_report() {
    local data="$1"
    local xml_file="$SESSION_DIR/security-audit-report.xml"
    
    cat > "$xml_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<security_audit>
    <metadata>
        <level>$AUDIT_LEVEL</level>
        <environment>$ENVIRONMENT</environment>
        <timestamp>$(date -u +%Y-%m-%dT%H:%M:%SZ)</timestamp>
    </metadata>
    <summary>
        <total_issues>0</total_issues>
        <risk_level>unknown</risk_level>
    </summary>
    <scans>
        <!-- Scan results would be populated here -->
    </scans>
</security_audit>
EOF
    
    log_success "XML report generated: $xml_file"
}

# Send notifications if enabled
send_notifications() {
    if [[ "$SEND_NOTIFICATIONS" != "true" ]]; then
        log_info "Notifications disabled"
        return 0
    fi
    
    log_info "Sending security audit notifications..."
    
    local report_file="$SESSION_DIR/security-audit-report.json"
    if [[ ! -f "$report_file" ]]; then
        log_error "Report file not found, cannot send notifications"
        return 1
    fi
    
    local total_issues=$(jq '.summary.total_issues // 0' "$report_file" 2>/dev/null || echo 0)
    local risk_level=$(jq -r '.summary.risk_level // "unknown"' "$report_file" 2>/dev/null || echo "unknown")
    
    local message="NovaCron Security Audit Completed
Level: $AUDIT_LEVEL
Environment: $ENVIRONMENT  
Total Issues: $total_issues
Risk Level: $risk_level
Report: $report_file"
    
    # Send to configured channels
    send_slack_notification "$message"
    send_email_notification "$message"
    
    log_success "Notifications sent"
}

# Send Slack notification
send_slack_notification() {
    local message="$1"
    
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_info "Sending Slack notification..."
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"$message\"}" \
             "$SLACK_WEBHOOK_URL" &>/dev/null || log_warning "Slack notification failed"
    fi
}

# Send email notification
send_email_notification() {
    local message="$1"
    
    if [[ -n "${EMAIL_RECIPIENTS:-}" ]] && command -v mail &> /dev/null; then
        log_info "Sending email notification..."
        echo "$message" | mail -s "NovaCron Security Audit Report" "$EMAIL_RECIPIENTS" || log_warning "Email notification failed"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary Docker images
    docker images | grep "novacron-security-scan" | awk '{print $3}' | xargs -r docker rmi &>/dev/null || true
    
    # Compress raw results
    if [[ -d "$SESSION_DIR/raw" ]]; then
        tar -czf "$SESSION_DIR/raw-results.tar.gz" -C "$SESSION_DIR" raw && rm -rf "$SESSION_DIR/raw"
    fi
    
    log_success "Cleanup completed"
}

# Main execution flow
main() {
    echo "===== NovaCron Security Audit Script ====="
    echo "Level: $AUDIT_LEVEL | Environment: $ENVIRONMENT"
    echo "=========================================="
    
    # Parse additional arguments
    parse_arguments "$@"
    
    # Validate inputs
    if [[ ! " ${!SECURITY_LEVELS[@]} " =~ " $AUDIT_LEVEL " ]]; then
        log_error "Invalid audit level: $AUDIT_LEVEL"
        show_usage
        exit 1
    fi
    
    if [[ ! " ${!ENVIRONMENTS[@]} " =~ " $ENVIRONMENT " ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        show_usage  
        exit 1
    fi
    
    # Setup
    setup_directories
    check_prerequisites
    load_configuration
    
    # Execute security audit
    execute_security_scans
    
    # Generate reports
    generate_security_report
    
    # Send notifications
    send_notifications
    
    # Cleanup
    cleanup
    
    # Final summary
    log_success "Security audit completed successfully!"
    log_info "Results available in: $SESSION_DIR"
    
    # Exit with appropriate code based on findings
    if [[ -f "$SESSION_DIR/security-audit-report.json" ]]; then
        local risk_level=$(jq -r '.summary.risk_level // "unknown"' "$SESSION_DIR/security-audit-report.json" 2>/dev/null || echo "unknown")
        case "$risk_level" in
            "critical") exit 3 ;;
            "high") exit 2 ;;
            "medium") exit 1 ;;
            *) exit 0 ;;
        esac
    fi
    
    exit 0
}

# Trap for cleanup on exit
trap cleanup EXIT

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi