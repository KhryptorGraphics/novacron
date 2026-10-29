#!/bin/bash
# NovaCron Integration Validation Script
# Comprehensive testing suite for Ubuntu 24.04 deployment

set -euo pipefail

# Configuration
SCRIPT_VERSION="1.0.0"
TEST_DATE=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/var/log/novacron/validation_${TEST_DATE}.log"
API_BASE="http://localhost:8090"
WS_BASE="ws://localhost:8091"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

log_test() {
    local status="$1"
    local test_name="$2"
    local details="$3"
    
    ((TOTAL_TESTS++))
    
    if [[ "$status" == "PASS" ]]; then
        echo -e "${GREEN}✓${NC} $test_name: $details" | tee -a "${LOG_FILE}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗${NC} $test_name: $details" | tee -a "${LOG_FILE}"
        ((FAILED_TESTS++))
    fi
}

# Print test header
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    NovaCron Integration Validation${NC}"
    echo -e "${BLUE}    Version: ${SCRIPT_VERSION}${NC}"
    echo -e "${BLUE}    Date: ${TEST_DATE}${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Test system requirements
test_system_requirements() {
    log_info "Testing system requirements..."
    
    # Check Ubuntu version
    if lsb_release -rs | grep -q "24.04"; then
        log_test "PASS" "Ubuntu Version" "24.04 LTS detected"
    else
        log_test "FAIL" "Ubuntu Version" "Not Ubuntu 24.04 LTS"
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    if [[ $cpu_cores -ge 8 ]]; then
        log_test "PASS" "CPU Cores" "${cpu_cores} cores available"
    else
        log_test "FAIL" "CPU Cores" "Only ${cpu_cores} cores (minimum 8 recommended)"
    fi
    
    # Check memory
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $memory_gb -ge 16 ]]; then
        log_test "PASS" "Memory" "${memory_gb}GB RAM available"
    else
        log_test "FAIL" "Memory" "Only ${memory_gb}GB RAM (minimum 16GB recommended)"
    fi
    
    # Check KVM support
    if [[ -e /dev/kvm ]] && [[ -r /dev/kvm ]] && [[ -w /dev/kvm ]]; then
        log_test "PASS" "KVM Support" "/dev/kvm is accessible"
    else
        log_test "FAIL" "KVM Support" "/dev/kvm not accessible"
    fi
    
    # Check disk space
    local disk_space=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [[ $disk_space -ge 100 ]]; then
        log_test "PASS" "Disk Space" "${disk_space}GB free space"
    else
        log_test "FAIL" "Disk Space" "Only ${disk_space}GB free (minimum 100GB recommended)"
    fi
}

# Test package installations
test_package_installations() {
    log_info "Testing package installations..."
    
    local packages=(
        "qemu-kvm"
        "libvirt-daemon-system" 
        "apparmor"
        "zfsutils-linux"
        "prometheus"
        "grafana"
        "docker.io"
        "golang-1.22"
        "nodejs"
    )
    
    for package in "${packages[@]}"; do
        if dpkg -l | grep -q "^ii  $package "; then
            log_test "PASS" "Package Install" "$package is installed"
        else
            log_test "FAIL" "Package Install" "$package is not installed"
        fi
    done
}

# Test user and group setup
test_users_groups() {
    log_info "Testing user and group setup..."
    
    # Check novacron user
    if getent passwd novacron &>/dev/null; then
        log_test "PASS" "User Setup" "novacron user exists"
    else
        log_test "FAIL" "User Setup" "novacron user does not exist"
    fi
    
    # Check novacron group
    if getent group novacron &>/dev/null; then
        log_test "PASS" "Group Setup" "novacron group exists"
    else
        log_test "FAIL" "Group Setup" "novacron group does not exist"
    fi
    
    # Check group memberships
    local user_groups=$(groups novacron 2>/dev/null || echo "")
    for required_group in kvm libvirt storage docker; do
        if echo "$user_groups" | grep -q "$required_group"; then
            log_test "PASS" "Group Membership" "novacron user is in $required_group group"
        else
            log_test "FAIL" "Group Membership" "novacron user not in $required_group group"
        fi
    done
}

# Test directory structure
test_directories() {
    log_info "Testing directory structure..."
    
    local directories=(
        "/opt/novacron"
        "/opt/novacron/bin"
        "/etc/novacron"
        "/var/lib/novacron"
        "/var/lib/novacron/vms"
        "/var/lib/novacron/models"
        "/var/log/novacron"
        "/var/cache/novacron"
    )
    
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            log_test "PASS" "Directory" "$dir exists"
        else
            log_test "FAIL" "Directory" "$dir does not exist"
        fi
    done
    
    # Check permissions
    if [[ -d "/etc/novacron" ]]; then
        local perms=$(stat -c "%a" /etc/novacron)
        if [[ "$perms" == "750" ]] || [[ "$perms" == "755" ]]; then
            log_test "PASS" "Directory Permissions" "/etc/novacron has correct permissions ($perms)"
        else
            log_test "FAIL" "Directory Permissions" "/etc/novacron has incorrect permissions ($perms)"
        fi
    fi
}

# Test systemd services
test_systemd_services() {
    log_info "Testing systemd services..."
    
    local services=(
        "novacron-storage.service"
        "novacron-network-manager.service"
        "novacron-api.service"
        "novacron-hypervisor.service"
        "novacron-llm-engine.service"
        "novacron.target"
    )
    
    for service in "${services[@]}"; do
        # Check if service file exists
        if [[ -f "/etc/systemd/system/$service" ]]; then
            log_test "PASS" "Service File" "$service file exists"
        else
            log_test "FAIL" "Service File" "$service file does not exist"
            continue
        fi
        
        # Check if service is enabled
        if systemctl is-enabled "$service" &>/dev/null; then
            log_test "PASS" "Service Enabled" "$service is enabled"
        else
            log_test "FAIL" "Service Enabled" "$service is not enabled"
        fi
        
        # Check service status (for running services)
        if systemctl is-active "$service" &>/dev/null; then
            log_test "PASS" "Service Active" "$service is running"
        else
            log_test "WARN" "Service Active" "$service is not running (may be expected)"
        fi
    done
}

# Test AppArmor profiles
test_apparmor() {
    log_info "Testing AppArmor profiles..."
    
    # Check if AppArmor is enabled
    if aa-status &>/dev/null; then
        log_test "PASS" "AppArmor" "AppArmor is running"
    else
        log_test "FAIL" "AppArmor" "AppArmor is not running"
        return
    fi
    
    local profiles=(
        "novacron-api"
        "novacron-hypervisor" 
        "novacron-llm-engine"
    )
    
    for profile in "${profiles[@]}"; do
        if [[ -f "/etc/apparmor.d/$profile" ]]; then
            log_test "PASS" "AppArmor Profile" "$profile profile file exists"
        else
            log_test "FAIL" "AppArmor Profile" "$profile profile file does not exist"
        fi
        
        # Check if profile is loaded
        if aa-status | grep -q "$profile"; then
            log_test "PASS" "AppArmor Loaded" "$profile profile is loaded"
        else
            log_test "FAIL" "AppArmor Loaded" "$profile profile is not loaded"
        fi
    done
}

# Test network configuration
test_network() {
    log_info "Testing network configuration..."
    
    # Check IP forwarding
    if [[ "$(sysctl -n net.ipv4.ip_forward)" == "1" ]]; then
        log_test "PASS" "IP Forwarding" "IPv4 forwarding is enabled"
    else
        log_test "FAIL" "IP Forwarding" "IPv4 forwarding is disabled"
    fi
    
    # Check bridge interface
    if ip link show novacron-br0 &>/dev/null; then
        log_test "PASS" "Bridge Interface" "novacron-br0 bridge exists"
    else
        log_test "FAIL" "Bridge Interface" "novacron-br0 bridge does not exist"
    fi
    
    # Check iptables/netfilter
    if command -v iptables &>/dev/null; then
        log_test "PASS" "Firewall Tools" "iptables is available"
    else
        log_test "FAIL" "Firewall Tools" "iptables is not available"
    fi
}

# Test storage configuration
test_storage() {
    log_info "Testing storage configuration..."
    
    # Check ZFS
    if command -v zfs &>/dev/null; then
        log_test "PASS" "ZFS Tools" "ZFS utilities are available"
        
        # Check if novacron pool exists
        if zpool status novacron &>/dev/null; then
            log_test "PASS" "ZFS Pool" "novacron pool exists"
        else
            log_test "WARN" "ZFS Pool" "novacron pool does not exist (may be intended)"
        fi
    else
        log_test "FAIL" "ZFS Tools" "ZFS utilities are not available"
    fi
    
    # Check libvirt storage
    if [[ -d "/var/lib/libvirt/images" ]]; then
        log_test "PASS" "Libvirt Storage" "libvirt images directory exists"
    else
        log_test "FAIL" "Libvirt Storage" "libvirt images directory does not exist"
    fi
}

# Test binary files
test_binaries() {
    log_info "Testing NovaCron binaries..."
    
    local binaries=(
        "novacron-api"
        "novacron-hypervisor"
    )
    
    for binary in "${binaries[@]}"; do
        local binary_path="/opt/novacron/bin/$binary"
        
        if [[ -f "$binary_path" ]]; then
            log_test "PASS" "Binary File" "$binary exists"
            
            # Check if executable
            if [[ -x "$binary_path" ]]; then
                log_test "PASS" "Binary Executable" "$binary is executable"
            else
                log_test "FAIL" "Binary Executable" "$binary is not executable"
            fi
            
            # Check ownership
            local owner=$(stat -c "%U:%G" "$binary_path")
            if [[ "$owner" == "novacron:novacron" ]]; then
                log_test "PASS" "Binary Ownership" "$binary has correct ownership"
            else
                log_test "FAIL" "Binary Ownership" "$binary has incorrect ownership ($owner)"
            fi
        else
            log_test "FAIL" "Binary File" "$binary does not exist"
        fi
    done
}

# Test configuration files
test_configuration() {
    log_info "Testing configuration files..."
    
    local config_files=(
        "api.conf"
        "hypervisor.conf"
        "network-topology.yaml"
        "security-hardening.yaml"
    )
    
    for config in "${config_files[@]}"; do
        local config_path="/etc/novacron/$config"
        
        if [[ -f "$config_path" ]]; then
            log_test "PASS" "Config File" "$config exists"
            
            # Check syntax for YAML files
            if [[ "$config" == *.yaml ]]; then
                if command -v python3 &>/dev/null; then
                    if python3 -c "import yaml; yaml.safe_load(open('$config_path'))" 2>/dev/null; then
                        log_test "PASS" "YAML Syntax" "$config has valid YAML syntax"
                    else
                        log_test "FAIL" "YAML Syntax" "$config has invalid YAML syntax"
                    fi
                fi
            fi
        else
            log_test "FAIL" "Config File" "$config does not exist"
        fi
    done
}

# Test API endpoints
test_api_endpoints() {
    log_info "Testing API endpoints..."
    
    # Test health endpoint
    if curl -s --max-time 5 "$API_BASE/health" | grep -q "healthy"; then
        log_test "PASS" "Health Endpoint" "API health endpoint responds correctly"
    else
        log_test "FAIL" "Health Endpoint" "API health endpoint is not responding"
    fi
    
    # Test API info endpoint
    if curl -s --max-time 5 "$API_BASE/api/info" | grep -q "NovaCron"; then
        log_test "PASS" "Info Endpoint" "API info endpoint responds correctly"
    else
        log_test "FAIL" "Info Endpoint" "API info endpoint is not responding"
    fi
    
    # Test monitoring endpoints
    local endpoints=(
        "/api/monitoring/metrics"
        "/api/monitoring/vms"
        "/api/monitoring/alerts"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s --max-time 5 "$API_BASE$endpoint" | grep -q "{"; then
            log_test "PASS" "API Endpoint" "$endpoint responds with JSON"
        else
            log_test "FAIL" "API Endpoint" "$endpoint does not respond correctly"
        fi
    done
}

# Test GPU support (if available)
test_gpu_support() {
    log_info "Testing GPU support..."
    
    # Check for NVIDIA GPUs
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            log_test "PASS" "NVIDIA GPU" "$gpu_count NVIDIA GPU(s) detected"
        else
            log_test "FAIL" "NVIDIA GPU" "nvidia-smi failed"
        fi
    else
        log_test "WARN" "NVIDIA GPU" "nvidia-smi not available (no NVIDIA GPUs expected)"
    fi
    
    # Check for AMD GPUs
    if [[ -d "/sys/class/drm" ]]; then
        local amd_cards=$(ls /sys/class/drm/ | grep -c "card" || echo "0")
        if [[ $amd_cards -gt 0 ]]; then
            log_test "PASS" "Graphics Cards" "$amd_cards graphics card(s) detected"
        else
            log_test "WARN" "Graphics Cards" "No graphics cards detected"
        fi
    fi
    
    # Check for CUDA support
    if [[ -d "/usr/local/cuda" ]] || [[ -f "/usr/bin/nvcc" ]]; then
        log_test "PASS" "CUDA Support" "CUDA toolkit is installed"
    else
        log_test "WARN" "CUDA Support" "CUDA toolkit not found"
    fi
}

# Test libvirt functionality
test_libvirt() {
    log_info "Testing libvirt functionality..."
    
    # Check libvirt daemon
    if systemctl is-active libvirtd &>/dev/null; then
        log_test "PASS" "Libvirt Daemon" "libvirtd is running"
    else
        log_test "FAIL" "Libvirt Daemon" "libvirtd is not running"
        return
    fi
    
    # Check virsh connectivity
    if sudo -u novacron virsh list &>/dev/null; then
        log_test "PASS" "Virsh Access" "virsh connects successfully"
    else
        log_test "FAIL" "Virsh Access" "virsh connection failed"
    fi
    
    # Check default network
    if sudo -u novacron virsh net-list | grep -q "default"; then
        log_test "PASS" "Default Network" "libvirt default network exists"
    else
        log_test "FAIL" "Default Network" "libvirt default network does not exist"
    fi
}

# Test monitoring services
test_monitoring() {
    log_info "Testing monitoring services..."
    
    # Test Prometheus
    if systemctl is-active prometheus &>/dev/null; then
        log_test "PASS" "Prometheus" "Prometheus service is running"
        
        # Test Prometheus endpoint
        if curl -s --max-time 5 "http://localhost:9090/-/healthy" | grep -q "Prometheus"; then
            log_test "PASS" "Prometheus API" "Prometheus API is responding"
        else
            log_test "FAIL" "Prometheus API" "Prometheus API is not responding"
        fi
    else
        log_test "FAIL" "Prometheus" "Prometheus service is not running"
    fi
    
    # Test Grafana
    if systemctl is-active grafana-server &>/dev/null; then
        log_test "PASS" "Grafana" "Grafana service is running"
    else
        log_test "FAIL" "Grafana" "Grafana service is not running"
    fi
}

# Test security configurations
test_security() {
    log_info "Testing security configurations..."
    
    # Test fail2ban
    if systemctl is-active fail2ban &>/dev/null; then
        log_test "PASS" "Fail2ban" "fail2ban is running"
    else
        log_test "WARN" "Fail2ban" "fail2ban is not running"
    fi
    
    # Test auditd
    if systemctl is-active auditd &>/dev/null; then
        log_test "PASS" "Auditd" "auditd is running"
    else
        log_test "WARN" "Auditd" "auditd is not running"
    fi
    
    # Check for sensitive file permissions
    if [[ -f "/etc/novacron/api.conf" ]]; then
        local perms=$(stat -c "%a" /etc/novacron/api.conf)
        if [[ "$perms" == "644" ]] || [[ "$perms" == "640" ]]; then
            log_test "PASS" "File Permissions" "api.conf has secure permissions ($perms)"
        else
            log_test "WARN" "File Permissions" "api.conf may have insecure permissions ($perms)"
        fi
    fi
}

# Performance benchmarks
test_performance() {
    log_info "Testing performance benchmarks..."
    
    # Test disk I/O
    if command -v dd &>/dev/null; then
        local io_start=$(date +%s%N)
        dd if=/dev/zero of=/tmp/novacron_test_file bs=1M count=100 conv=fdatasync &>/dev/null
        local io_end=$(date +%s%N)
        local io_duration=$(( (io_end - io_start) / 1000000 )) # Convert to milliseconds
        rm -f /tmp/novacron_test_file
        
        if [[ $io_duration -lt 5000 ]]; then # Less than 5 seconds
            log_test "PASS" "Disk Performance" "Disk I/O: ${io_duration}ms (good)"
        else
            log_test "WARN" "Disk Performance" "Disk I/O: ${io_duration}ms (slow)"
        fi
    fi
    
    # Test memory allocation
    local mem_available=$(free -m | awk '/^Mem:/{print $7}')
    if [[ $mem_available -gt 8192 ]]; then # More than 8GB available
        log_test "PASS" "Memory Availability" "${mem_available}MB available"
    else
        log_test "WARN" "Memory Availability" "Only ${mem_available}MB available"
    fi
}

# Generate test report
generate_report() {
    local report_file="/var/log/novacron/validation_report_${TEST_DATE}.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>NovaCron Validation Report - ${TEST_DATE}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .warn { color: orange; font-weight: bold; }
        .summary { background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NovaCron Integration Validation Report</h1>
        <p>Generated: ${TEST_DATE}</p>
        <p>Version: ${SCRIPT_VERSION}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Total Tests: <strong>${TOTAL_TESTS}</strong></p>
        <p>Passed: <span class="pass">${PASSED_TESTS}</span></p>
        <p>Failed: <span class="fail">${FAILED_TESTS}</span></p>
        <p>Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%</p>
    </div>
    
    <h2>Detailed Results</h2>
    <p>See log file for detailed results: <code>${LOG_FILE}</code></p>
    
    <footer>
        <p><em>Report generated by NovaCron validation script v${SCRIPT_VERSION}</em></p>
    </footer>
</body>
</html>
EOF
    
    log_info "HTML report generated: $report_file"
}

# Display summary
display_summary() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}    Validation Summary${NC}"  
    echo -e "${BLUE}============================================${NC}"
    echo -e "Total Tests: ${TOTAL_TESTS}"
    echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
    echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
    
    local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    echo -e "Success Rate: ${success_rate}%"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "\n${GREEN}✓ All tests passed! NovaCron is ready for production.${NC}"
    elif [[ $success_rate -ge 80 ]]; then
        echo -e "\n${YELLOW}⚠ Most tests passed, but some issues need attention.${NC}"
    else
        echo -e "\n${RED}✗ Many tests failed. Deployment needs significant fixes.${NC}"
    fi
    
    echo -e "\nLog file: ${LOG_FILE}"
    echo -e "${BLUE}============================================${NC}"
}

# Main validation function
main() {
    print_header
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run test suites
    test_system_requirements
    test_package_installations
    test_users_groups
    test_directories
    test_systemd_services
    test_apparmor
    test_network
    test_storage
    test_binaries
    test_configuration
    test_libvirt
    test_monitoring
    test_security
    test_performance
    
    # Test API endpoints if services are running
    if systemctl is-active novacron-api.service &>/dev/null; then
        test_api_endpoints
    else
        log_warn "API service not running, skipping API tests"
    fi
    
    # Test GPU support if available
    test_gpu_support
    
    # Generate reports
    generate_report
    display_summary
    
    # Exit with appropriate code
    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Handle script interruption
trap 'log_error "Validation interrupted"; exit 1' INT TERM

# Run main function
main "$@"