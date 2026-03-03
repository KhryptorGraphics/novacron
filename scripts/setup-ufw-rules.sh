#!/bin/bash

# NovaCron UFW Rules Setup Script
# Configures comprehensive firewall rules for secure NovaCron deployment
# Supports development, staging, and production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Environment detection
ENVIRONMENT="${NOVACRON_ENV:-development}"
ALLOW_ALL_SSH="${ALLOW_ALL_SSH:-false}"
RESTRICT_MONITORING="${RESTRICT_MONITORING:-true}"
ENABLE_VNC_ACCESS="${ENABLE_VNC_ACCESS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        log_info "Usage: sudo $0 [environment]"
        log_info "Environments: development, staging, production"
        exit 1
    fi
}

# Check UFW status and profiles
check_prerequisites() {
    if ! command -v ufw >/dev/null 2>&1; then
        log_error "UFW is not installed. Please install UFW first."
        exit 1
    fi
    
    if ! ufw app list | grep -q "NovaCron"; then
        log_error "NovaCron UFW profiles not found"
        log_info "Please run: sudo $PROJECT_ROOT/scripts/install-ufw-profiles.sh"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Reset UFW to clean state
reset_ufw() {
    log_info "Resetting UFW to clean state..."
    ufw --force reset >/dev/null
    log_success "UFW reset completed"
}

# Set default policies
set_default_policies() {
    log_info "Setting default UFW policies..."
    
    # Deny all incoming by default
    ufw default deny incoming
    
    # Allow all outgoing by default
    ufw default allow outgoing
    
    log_success "Default policies configured (deny incoming, allow outgoing)"
}

# Configure SSH access
configure_ssh() {
    log_info "Configuring SSH access..."
    
    if [[ "$ALLOW_ALL_SSH" == "true" ]]; then
        ufw allow "NovaCron SSH"
        log_success "SSH access allowed from anywhere"
    else
        # More restrictive SSH access
        case "$ENVIRONMENT" in
            "production")
                log_info "Production environment: SSH access limited to management networks"
                # Add specific IP ranges for production management
                ufw allow from 10.0.0.0/8 to any port 22
                ufw allow from 172.16.0.0/12 to any port 22
                ufw allow from 192.168.0.0/16 to any port 22
                log_success "SSH access limited to RFC1918 private networks"
                ;;
            "staging")
                log_info "Staging environment: SSH access from private networks"
                ufw allow from 10.0.0.0/8 to any port 22
                ufw allow from 172.16.0.0/12 to any port 22
                ufw allow from 192.168.0.0/16 to any port 22
                log_success "SSH access allowed from private networks"
                ;;
            "development")
                ufw allow "NovaCron SSH"
                log_success "SSH access allowed from anywhere (development)"
                ;;
        esac
    fi
}

# Configure web services
configure_web_services() {
    log_info "Configuring web services..."
    
    case "$ENVIRONMENT" in
        "production")
            # HTTPS only for production
            ufw allow "NovaCron HTTPS"
            ufw allow "NovaCron HTTP"  # For redirect to HTTPS
            log_success "Web services: HTTPS (443) and HTTP redirect (80)"
            ;;
        "staging"|"development")
            # Allow direct access to all web services
            ufw allow "NovaCron Core Services"
            ufw allow "NovaCron HTTPS"
            ufw allow "NovaCron HTTP"
            log_success "Web services: HTTP/HTTPS and direct service access"
            ;;
    esac
}

# Configure API and core services
configure_core_services() {
    log_info "Configuring NovaCron core services..."
    
    # Always allow core services
    ufw allow "NovaCron API"
    ufw allow "NovaCron WebSocket"
    ufw allow "NovaCron Frontend"
    ufw allow "NovaCron AI Engine"
    
    log_success "Core services configured (API: 8090, WebSocket: 8091, Frontend: 8092, AI: 8093)"
}

# Configure monitoring services
configure_monitoring() {
    log_info "Configuring monitoring services..."
    
    if [[ "$RESTRICT_MONITORING" == "true" ]]; then
        case "$ENVIRONMENT" in
            "production")
                # Restrict monitoring access to management networks
                ufw allow from 10.0.0.0/8 to any port 9090
                ufw allow from 10.0.0.0/8 to any port 3001
                ufw allow from 10.0.0.0/8 to any port 9100
                ufw allow from 172.16.0.0/12 to any port 9090
                ufw allow from 172.16.0.0/12 to any port 3001
                ufw allow from 172.16.0.0/12 to any port 9100
                ufw allow from 192.168.0.0/16 to any port 9090
                ufw allow from 192.168.0.0/16 to any port 3001
                ufw allow from 192.168.0.0/16 to any port 9100
                log_success "Monitoring access restricted to private networks"
                ;;
            *)
                ufw allow "NovaCron Monitoring Stack"
                log_success "Monitoring services accessible (Prometheus: 9090, Grafana: 3001)"
                ;;
        esac
    else
        ufw allow "NovaCron Monitoring Stack"
        log_success "Monitoring services open access"
    fi
}

# Configure hypervisor and VM services
configure_hypervisor() {
    log_info "Configuring hypervisor services..."
    
    # Allow hypervisor communication
    ufw allow "NovaCron Hypervisor"
    
    # VNC access (conditional)
    if [[ "$ENABLE_VNC_ACCESS" == "true" ]]; then
        case "$ENVIRONMENT" in
            "production")
                # Restrict VNC to management networks
                ufw allow from 10.0.0.0/8 to any port 5900:5999
                ufw allow from 172.16.0.0/12 to any port 5900:5999
                ufw allow from 192.168.0.0/16 to any port 5900:5999
                log_success "VNC console access restricted to private networks"
                ;;
            *)
                ufw allow "NovaCron VNC"
                log_success "VNC console access enabled"
                ;;
        esac
    else
        log_info "VNC access disabled (set ENABLE_VNC_ACCESS=true to enable)"
    fi
    
    # Migration ports for cluster communication
    ufw allow "NovaCron Migration"
    ufw allow "NovaCron Cluster"
    
    log_success "Hypervisor services configured"
}

# Configure database and cache
configure_data_services() {
    log_info "Configuring data services..."
    
    case "$ENVIRONMENT" in
        "production")
            # Restrict database access to localhost and private networks
            ufw allow from 127.0.0.1 to any port 11432
            ufw allow from 10.0.0.0/8 to any port 11432
            ufw allow from 172.16.0.0/12 to any port 11432
            ufw allow from 192.168.0.0/16 to any port 11432
            
            # Redis access similarly restricted
            ufw allow from 127.0.0.1 to any port 6379
            ufw allow from 10.0.0.0/8 to any port 6379
            ufw allow from 172.16.0.0/12 to any port 6379
            ufw allow from 192.168.0.0/16 to any port 6379
            
            log_success "Data services restricted to localhost and private networks"
            ;;
        *)
            ufw allow "NovaCron Data Layer"
            log_success "Data services accessible (PostgreSQL: 11432, Redis: 6379)"
            ;;
    esac
}

# Configure logging and SNMP
configure_system_services() {
    log_info "Configuring system services..."
    
    # DNS (if needed)
    # ufw allow "NovaCron DNS"
    
    # SNMP monitoring (restricted)
    case "$ENVIRONMENT" in
        "production"|"staging")
            ufw allow from 10.0.0.0/8 to any port 161
            ufw allow from 172.16.0.0/12 to any port 161
            ufw allow from 192.168.0.0/16 to any port 161
            log_success "SNMP access restricted to private networks"
            ;;
        "development")
            ufw allow "NovaCron SNMP"
            log_success "SNMP monitoring enabled"
            ;;
    esac
    
    # Syslog (usually internal only)
    ufw allow from 127.0.0.1 to any port 514
    log_success "Syslog configured for localhost"
}

# Rate limiting for security
configure_rate_limiting() {
    log_info "Configuring rate limiting..."
    
    # Limit SSH connection attempts
    ufw limit ssh
    
    # Limit HTTP connections
    ufw limit 80/tcp
    ufw limit 443/tcp
    
    log_success "Rate limiting configured for SSH and HTTP(S)"
}

# Show UFW status and rules
show_status() {
    log_info "Current UFW configuration:"
    echo ""
    ufw status verbose
    echo ""
    
    log_info "NovaCron-specific rules:"
    ufw status | grep -i novacron || log_info "No NovaCron-specific rules found"
    echo ""
}

# Environment-specific configurations
configure_environment() {
    log_info "Configuring for environment: $ENVIRONMENT"
    
    case "$ENVIRONMENT" in
        "production")
            log_info "Production security profile: Maximum security"
            RESTRICT_MONITORING="true"
            ENABLE_VNC_ACCESS="false"
            ;;
        "staging")
            log_info "Staging security profile: Balanced security"
            RESTRICT_MONITORING="true"
            ENABLE_VNC_ACCESS="false"
            ;;
        "development")
            log_info "Development security profile: Accessible for testing"
            RESTRICT_MONITORING="false"
            ENABLE_VNC_ACCESS="true"
            ;;
        *)
            log_warning "Unknown environment: $ENVIRONMENT, using development profile"
            ENVIRONMENT="development"
            ;;
    esac
}

# Main setup function
main() {
    log_info "Starting NovaCron UFW rules configuration..."
    log_info "Environment: $ENVIRONMENT"
    echo ""
    
    check_root
    check_prerequisites
    configure_environment
    
    # Confirm before proceeding
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "This will configure PRODUCTION firewall rules"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborted by user"
            exit 0
        fi
    fi
    
    reset_ufw
    set_default_policies
    configure_ssh
    configure_web_services
    configure_core_services
    configure_monitoring
    configure_hypervisor
    configure_data_services
    configure_system_services
    configure_rate_limiting
    
    echo ""
    log_success "NovaCron UFW rules configuration completed!"
    log_info "To enable the firewall: sudo ufw enable"
    log_warning "IMPORTANT: Test SSH connectivity before enabling!"
    
    echo ""
    show_status
    
    echo ""
    log_info "Quick reference commands:"
    log_info "  View status: sudo ufw status verbose"
    log_info "  Enable firewall: sudo ufw enable"
    log_info "  Disable firewall: sudo ufw disable"
    log_info "  Reset rules: sudo ufw --force reset"
    log_info "  List NovaCron profiles: ufw app list | grep NovaCron"
}

# Handle command line arguments
if [[ $# -gt 0 ]]; then
    ENVIRONMENT="$1"
fi

# Run main function
main "$@"