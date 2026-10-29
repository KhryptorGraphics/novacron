#!/bin/bash
# NovaCron Ubuntu 24.04 Core Deployment Automation Script
# Comprehensive deployment with security hardening and optimization

set -euo pipefail

# Script metadata
SCRIPT_VERSION="2.0.0"
DEPLOYMENT_DATE=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/var/log/novacron/deployment_${DEPLOYMENT_DATE}.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Info logging with color
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" | tee -a "${LOG_FILE}"
}

# Warning logging with color  
log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "${LOG_FILE}"
}

# Error logging with color
log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

# Header function
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    NovaCron Ubuntu 24.04 Deployment${NC}"
    echo -e "${BLUE}    Version: ${SCRIPT_VERSION}${NC}"
    echo -e "${BLUE}    Date: ${DEPLOYMENT_DATE}${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Validate Ubuntu version
validate_ubuntu_version() {
    log_info "Validating Ubuntu version..."
    
    if ! command -v lsb_release &> /dev/null; then
        log_error "lsb_release not found. Cannot determine Ubuntu version."
        exit 1
    fi
    
    local version=$(lsb_release -rs)
    local codename=$(lsb_release -cs)
    
    if [[ "$version" != "24.04" ]] && [[ "$codename" != "noble" ]]; then
        log_warn "Not running Ubuntu 24.04 LTS. Current version: $version ($codename)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_info "Ubuntu version validated: $version ($codename)"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check CPU
    local cpu_cores=$(nproc)
    if [[ $cpu_cores -lt 8 ]]; then
        log_warn "Minimum 8 CPU cores recommended. Found: $cpu_cores"
    fi
    
    # Check memory (in KB, convert to GB)
    local memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local memory_gb=$((memory_kb / 1024 / 1024))
    if [[ $memory_gb -lt 16 ]]; then
        log_warn "Minimum 16GB RAM recommended. Found: ${memory_gb}GB"
    fi
    
    # Check disk space (in GB)
    local disk_space=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [[ $disk_space -lt 100 ]]; then
        log_warn "Minimum 100GB free space recommended. Found: ${disk_space}GB"
    fi
    
    # Check KVM support
    if [[ ! -e /dev/kvm ]]; then
        log_warn "/dev/kvm not found. Hardware virtualization may not be available."
    fi
    
    log_info "System requirements check completed"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get upgrade -y -q
    apt-get autoremove -y -q
    apt-get autoclean -q
    
    log_info "System packages updated successfully"
}

# Install required packages
install_packages() {
    log_info "Installing required packages..."
    
    local packages=(
        # Core system packages
        "curl" "wget" "git" "unzip" "jq" "htop" "iotop"
        
        # Build tools
        "build-essential" "pkg-config" "cmake"
        
        # Virtualization
        "qemu-kvm" "libvirt-daemon-system" "libvirt-clients" "virtinst"
        "virt-manager" "bridge-utils" "virt-viewer" "ovmf"
        
        # Networking
        "openvswitch-switch" "wireguard" "iptables-persistent"
        
        # Storage
        "zfsutils-linux" "cryptsetup" "lvm2"
        
        # Security
        "apparmor" "apparmor-utils" "apparmor-profiles" "fail2ban"
        "auditd" "rkhunter" "chkrootkit"
        
        # Monitoring
        "prometheus" "grafana" "telegraf"
        
        # Containers
        "docker.io" "docker-compose"
        
        # Development
        "golang-1.22" "nodejs" "npm" "python3" "python3-pip"
        
        # GPU support (optional)
        "nvidia-driver-535" "nvidia-cuda-toolkit"
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            log_info "Installing $package..."
            apt-get install -y -q "$package" || log_warn "Failed to install $package"
        fi
    done
    
    log_info "Package installation completed"
}

# Setup users and groups
setup_users() {
    log_info "Setting up users and groups..."
    
    # Create novacron group
    if ! getent group novacron &>/dev/null; then
        groupadd -r novacron
        log_info "Created novacron group"
    fi
    
    # Create novacron user
    if ! getent passwd novacron &>/dev/null; then
        useradd -r -g novacron -d /opt/novacron -s /bin/bash novacron
        log_info "Created novacron user"
    fi
    
    # Create storage group
    if ! getent group storage &>/dev/null; then
        groupadd -r storage
        log_info "Created storage group"
    fi
    
    # Add novacron user to required groups
    usermod -a -G kvm,libvirt,storage,docker novacron
    
    log_info "User and group setup completed"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local directories=(
        "/opt/novacron"
        "/opt/novacron/bin"
        "/opt/novacron/config"
        "/etc/novacron"
        "/var/lib/novacron"
        "/var/lib/novacron/api"
        "/var/lib/novacron/vms"
        "/var/lib/novacron/models"
        "/var/lib/novacron/sessions"
        "/var/lib/novacron/network"
        "/var/lib/novacron/migration"
        "/var/log/novacron"
        "/var/cache/novacron"
        "/mnt/novacron-storage"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chown novacron:novacron "$dir"
        chmod 755 "$dir"
    done
    
    # Secure sensitive directories
    chmod 750 /etc/novacron
    chmod 750 /var/lib/novacron/sessions
    
    log_info "Directory structure created"
}

# Setup systemd services
setup_systemd_services() {
    log_info "Setting up systemd services..."
    
    # Copy systemd service files
    if [[ -d "./systemd" ]]; then
        cp ./systemd/*.service /etc/systemd/system/
        cp ./systemd/*.target /etc/systemd/system/
        systemctl daemon-reload
        log_info "Systemd service files installed"
    else
        log_warn "Systemd service files not found in ./systemd/"
    fi
    
    # Enable services (but don't start yet)
    systemctl enable novacron.target
    systemctl enable novacron-storage.service
    systemctl enable novacron-network-manager.service
    systemctl enable novacron-api.service
    systemctl enable novacron-hypervisor.service
    systemctl enable novacron-llm-engine.service
    
    log_info "Systemd services enabled"
}

# Setup AppArmor profiles
setup_apparmor() {
    log_info "Setting up AppArmor profiles..."
    
    if [[ -d "./apparmor" ]]; then
        # Copy profiles
        cp ./apparmor/* /etc/apparmor.d/
        
        # Load profiles
        apparmor_parser -r /etc/apparmor.d/novacron-api
        apparmor_parser -r /etc/apparmor.d/novacron-hypervisor
        apparmor_parser -r /etc/apparmor.d/novacron-llm-engine
        
        # Verify profiles are loaded
        aa-status | grep novacron || log_warn "Some AppArmor profiles may not be loaded"
        
        log_info "AppArmor profiles configured"
    else
        log_warn "AppArmor profiles not found in ./apparmor/"
    fi
}

# Configure network
configure_network() {
    log_info "Configuring network..."
    
    # Enable IP forwarding
    echo 'net.ipv4.ip_forward=1' > /etc/sysctl.d/99-novacron.conf
    echo 'net.ipv6.conf.all.forwarding=1' >> /etc/sysctl.d/99-novacron.conf
    
    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/99-novacron.conf
    
    # Configure bridge for VM networking
    if ! ip link show novacron-br0 &>/dev/null; then
        ip link add name novacron-br0 type bridge
        ip addr add 172.16.0.1/24 dev novacron-br0
        ip link set novacron-br0 up
        log_info "Created novacron-br0 bridge"
    fi
    
    log_info "Network configuration completed"
}

# Setup ZFS storage
setup_zfs_storage() {
    log_info "Setting up ZFS storage..."
    
    # Check if ZFS is available
    if ! command -v zfs &> /dev/null; then
        log_warn "ZFS not available, skipping ZFS setup"
        return
    fi
    
    # Create ZFS pool (using file-based vdev for testing)
    if ! zpool status novacron &>/dev/null; then
        # Create a file-based pool for development/testing
        mkdir -p /var/lib/novacron/zfs
        truncate -s 50G /var/lib/novacron/zfs/vdev0
        zpool create novacron /var/lib/novacron/zfs/vdev0
        
        # Set ZFS properties
        zfs set compression=lz4 novacron
        zfs set atime=off novacron
        zfs set xattr=sa novacron
        
        # Create datasets
        zfs create novacron/vms
        zfs create novacron/models
        zfs create novacron/backups
        
        log_info "ZFS storage pool created"
    else
        log_info "ZFS pool 'novacron' already exists"
    fi
}

# Setup security hardening
setup_security() {
    log_info "Applying security hardening..."
    
    # Configure fail2ban
    if command -v fail2ban-client &> /dev/null; then
        cat > /etc/fail2ban/jail.d/novacron.conf << 'EOF'
[novacron-api]
enabled = true
port = 8090
filter = novacron-api
logpath = /var/log/novacron/api.log
maxretry = 5
bantime = 3600

[sshd]
enabled = true
bantime = 3600
maxretry = 3
EOF
        
        systemctl enable fail2ban
        systemctl restart fail2ban
        log_info "fail2ban configured"
    fi
    
    # Configure audit daemon
    if command -v auditd &> /dev/null; then
        systemctl enable auditd
        systemctl start auditd
        log_info "auditd enabled"
    fi
    
    # Set file permissions
    chmod 644 /etc/novacron/*.conf 2>/dev/null || true
    chmod 600 /etc/novacron/*key* 2>/dev/null || true
    
    log_info "Security hardening applied"
}

# Configure monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Configure Prometheus
    if command -v prometheus &> /dev/null; then
        # Copy configuration if available
        if [[ -f "./configs/prometheus.yml" ]]; then
            cp ./configs/prometheus.yml /etc/prometheus/
            chown prometheus:prometheus /etc/prometheus/prometheus.yml
        fi
        
        systemctl enable prometheus
        log_info "Prometheus configured"
    fi
    
    # Configure Grafana
    if command -v grafana-server &> /dev/null; then
        systemctl enable grafana-server
        log_info "Grafana configured"
    fi
    
    log_info "Monitoring setup completed"
}

# Build NovaCron binaries
build_novacron() {
    log_info "Building NovaCron binaries..."
    
    if [[ ! -d "./backend" ]]; then
        log_warn "Backend source code not found, skipping build"
        return
    fi
    
    cd backend
    
    # Set Go environment
    export GOPATH="/opt/novacron/go"
    export PATH="$PATH:/usr/lib/go-1.22/bin"
    
    # Build API server
    go build -o /opt/novacron/bin/novacron-api ./cmd/api-server/main.go
    
    # Build other components
    go build -o /opt/novacron/bin/novacron-hypervisor ./core/cmd/novacron/main.go
    
    # Set permissions
    chown novacron:novacron /opt/novacron/bin/*
    chmod 755 /opt/novacron/bin/*
    
    cd ..
    
    log_info "NovaCron binaries built successfully"
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Copy configuration files if they exist
    if [[ -d "./configs" ]]; then
        cp ./configs/*.yaml /etc/novacron/ 2>/dev/null || true
        cp ./configs/*.conf /etc/novacron/ 2>/dev/null || true
        chown -R novacron:novacron /etc/novacron/
    fi
    
    # Generate API configuration
    cat > /etc/novacron/api.conf << EOF
# NovaCron API Configuration
api_port = 8090
ws_port = 8091
db_url = "postgresql://postgres:postgres@localhost:5432/novacron"
log_level = "info"
auth_secret = "$(openssl rand -hex 32)"
EOF
    
    # Generate hypervisor configuration
    cat > /etc/novacron/hypervisor.conf << EOF
# NovaCron Hypervisor Configuration
node_id = "$(hostname)"
storage_path = "/var/lib/novacron/vms"
cluster_addr = "localhost:8090"
libvirt_uri = "qemu:///system"
EOF
    
    chmod 644 /etc/novacron/*.conf
    chown novacron:novacron /etc/novacron/*.conf
    
    log_info "Configuration files created"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    local errors=0
    
    # Check if binaries exist
    for binary in novacron-api novacron-hypervisor; do
        if [[ ! -f "/opt/novacron/bin/$binary" ]]; then
            log_error "Binary not found: $binary"
            ((errors++))
        fi
    done
    
    # Check if services are enabled
    for service in novacron-storage novacron-api novacron-hypervisor; do
        if ! systemctl is-enabled "$service.service" &>/dev/null; then
            log_error "Service not enabled: $service"
            ((errors++))
        fi
    done
    
    # Check if configuration files exist
    for config in api.conf hypervisor.conf; do
        if [[ ! -f "/etc/novacron/$config" ]]; then
            log_error "Configuration file not found: $config"
            ((errors++))
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        log_info "Deployment validation passed"
        return 0
    else
        log_error "Deployment validation failed with $errors errors"
        return 1
    fi
}

# Start services
start_services() {
    log_info "Starting NovaCron services..."
    
    # Start target (which will start all dependencies)
    systemctl start novacron.target
    
    # Wait for services to start
    sleep 10
    
    # Check service status
    systemctl --no-pager status novacron.target
    
    log_info "Service startup completed"
}

# Display summary
display_summary() {
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    NovaCron Deployment Summary${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "Deployment Date: ${DEPLOYMENT_DATE}"
    echo -e "Log File: ${LOG_FILE}"
    echo -e ""
    echo -e "Services Status:"
    systemctl --no-pager --quiet is-active novacron.target && echo -e "  ${GREEN}✓${NC} NovaCron services are running" || echo -e "  ${RED}✗${NC} Some services may not be running"
    echo -e ""
    echo -e "Access URLs:"
    echo -e "  API Server: http://localhost:8090"
    echo -e "  WebSocket: ws://localhost:8091"
    echo -e "  Prometheus: http://localhost:9090"
    echo -e "  Grafana: http://localhost:3000"
    echo -e ""
    echo -e "Next Steps:"
    echo -e "  1. Configure PostgreSQL database"
    echo -e "  2. Upload LLM models to /var/lib/novacron/models/"
    echo -e "  3. Configure network topology"
    echo -e "  4. Test VM creation and migration"
    echo -e "${GREEN}============================================${NC}"
}

# Main deployment function
main() {
    print_header
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Execute deployment steps
    check_root
    validate_ubuntu_version
    check_system_requirements
    update_system
    install_packages
    setup_users
    create_directories
    configure_network
    setup_zfs_storage
    create_config_files
    build_novacron
    setup_systemd_services
    setup_apparmor
    setup_security
    setup_monitoring
    
    # Validate before starting
    if validate_deployment; then
        start_services
        display_summary
        log_info "NovaCron deployment completed successfully"
        exit 0
    else
        log_error "Deployment validation failed"
        exit 1
    fi
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"