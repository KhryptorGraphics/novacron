#!/bin/bash

# NovaCron UFW Helper Script
# Quick reference and management commands for NovaCron firewall configuration

set -euo pipefail

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

# Show help
show_help() {
    echo "NovaCron UFW Helper Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  install       Install UFW profiles"
    echo "  configure     Configure UFW rules for environment"
    echo "  status        Show UFW status and NovaCron rules"
    echo "  enable        Enable UFW firewall"
    echo "  disable       Disable UFW firewall"
    echo "  reset         Reset UFW to clean state"
    echo "  profiles      List available NovaCron profiles"
    echo "  logs          Show recent UFW logs"
    echo "  test          Test service connectivity"
    echo "  backup        Backup current UFW configuration"
    echo "  restore       Restore UFW configuration from backup"
    echo "  help          Show this help message"
    echo ""
    echo "Environment Configuration:"
    echo "  $0 configure development    # Development setup"
    echo "  $0 configure staging        # Staging setup"
    echo "  $0 configure production     # Production setup"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 configure production"
    echo "  $0 status"
    echo "  $0 test api"
    echo ""
}

# Check if running as root when needed
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This operation requires root privileges"
        log_info "Please run with sudo: sudo $0 $*"
        exit 1
    fi
}

# Install UFW profiles
install_profiles() {
    check_root
    
    log_info "Installing NovaCron UFW profiles..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [[ -x "$SCRIPT_DIR/install-ufw-profiles.sh" ]]; then
        "$SCRIPT_DIR/install-ufw-profiles.sh"
    else
        log_error "UFW profile installer not found"
        exit 1
    fi
}

# Configure UFW rules
configure_rules() {
    check_root
    
    local environment="${1:-development}"
    
    log_info "Configuring UFW rules for environment: $environment"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [[ -x "$SCRIPT_DIR/setup-ufw-rules.sh" ]]; then
        "$SCRIPT_DIR/setup-ufw-rules.sh" "$environment"
    else
        log_error "UFW rules setup script not found"
        exit 1
    fi
}

# Show UFW status
show_status() {
    log_info "UFW Status:"
    ufw status verbose
    
    echo ""
    log_info "NovaCron-specific rules:"
    ufw status | grep -i novacron || log_warning "No NovaCron-specific rules found"
    
    echo ""
    log_info "Active connections to NovaCron services:"
    netstat -tlnp 2>/dev/null | grep -E ":(8090|8091|8092|8093|9090|3001|6379|11432)" || log_warning "No active connections found"
}

# List available profiles
list_profiles() {
    log_info "Available NovaCron UFW profiles:"
    ufw app list | grep -i novacron | sed 's/^/  /'
    
    echo ""
    log_info "Profile details (use 'ufw app info \"Profile Name\"' for more info):"
    
    local profiles=(
        "NovaCron API"
        "NovaCron WebSocket"
        "NovaCron Frontend"
        "NovaCron Core Services"
        "NovaCron Monitoring Stack"
        "NovaCron Full Stack"
    )
    
    for profile in "${profiles[@]}"; do
        if ufw app info "$profile" >/dev/null 2>&1; then
            echo "  ✅ $profile"
        else
            echo "  ❌ $profile (not available)"
        fi
    done
}

# Show UFW logs
show_logs() {
    local lines="${1:-20}"
    
    log_info "Recent UFW log entries (last $lines lines):"
    
    if [[ -f /var/log/ufw.log ]]; then
        tail -n "$lines" /var/log/ufw.log | grep --color=always -E "BLOCK|DENY|ALLOW|$"
    else
        log_warning "UFW log file not found"
        log_info "UFW logs may be in syslog:"
        journalctl -u ufw -n "$lines" --no-pager
    fi
}

# Test service connectivity
test_connectivity() {
    local service="${1:-all}"
    
    log_info "Testing NovaCron service connectivity..."
    
    case "$service" in
        "api"|"all")
            log_info "Testing API server (port 8090)..."
            if curl -s -m 5 http://localhost:8090/health >/dev/null 2>&1; then
                log_success "API server accessible"
            else
                log_error "API server not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "websocket"|"ws"|"all")
            log_info "Testing WebSocket server (port 8091)..."
            if nc -z localhost 8091 2>/dev/null; then
                log_success "WebSocket server accessible"
            else
                log_error "WebSocket server not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "frontend"|"all")
            log_info "Testing Frontend (port 8092)..."
            if nc -z localhost 8092 2>/dev/null; then
                log_success "Frontend accessible"
            else
                log_error "Frontend not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "prometheus"|"metrics"|"all")
            log_info "Testing Prometheus (port 9090)..."
            if curl -s -m 5 http://localhost:9090/-/healthy >/dev/null 2>&1; then
                log_success "Prometheus accessible"
            else
                log_error "Prometheus not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "grafana"|"all")
            log_info "Testing Grafana (port 3001)..."
            if nc -z localhost 3001 2>/dev/null; then
                log_success "Grafana accessible"
            else
                log_error "Grafana not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "database"|"db"|"all")
            log_info "Testing Database (port 11432)..."
            if nc -z localhost 11432 2>/dev/null; then
                log_success "Database accessible"
            else
                log_error "Database not accessible"
            fi
            ;;
    esac
    
    case "$service" in
        "redis"|"cache"|"all")
            log_info "Testing Redis (port 6379)..."
            if redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
                log_success "Redis accessible"
            else
                log_error "Redis not accessible"
            fi
            ;;
    esac
}

# Enable UFW
enable_ufw() {
    check_root
    
    log_warning "This will enable UFW firewall"
    log_warning "Make sure SSH access is configured before proceeding!"
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
    
    log_info "Enabling UFW..."
    ufw --force enable
    
    log_success "UFW enabled"
    show_status
}

# Disable UFW
disable_ufw() {
    check_root
    
    log_info "Disabling UFW..."
    ufw disable
    
    log_success "UFW disabled"
}

# Reset UFW
reset_ufw() {
    check_root
    
    log_warning "This will reset UFW to default state and remove all rules"
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
    
    log_info "Resetting UFW..."
    ufw --force reset
    
    log_success "UFW reset to default state"
}

# Backup UFW configuration
backup_ufw() {
    check_root
    
    local backup_dir="/root/ufw-backups"
    local backup_file="ufw-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log_info "Creating UFW configuration backup..."
    
    mkdir -p "$backup_dir"
    
    tar -czf "$backup_dir/$backup_file" \
        /etc/ufw \
        /lib/ufw \
        /etc/default/ufw \
        /var/lib/ufw 2>/dev/null || true
    
    log_success "UFW configuration backed up to: $backup_dir/$backup_file"
}

# Restore UFW configuration
restore_ufw() {
    check_root
    
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        log_error "Please specify backup file to restore"
        log_info "Usage: $0 restore /path/to/backup.tar.gz"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_warning "This will restore UFW configuration from backup"
    log_warning "Current configuration will be overwritten"
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
    
    log_info "Restoring UFW configuration from: $backup_file"
    
    # Stop UFW
    ufw disable
    
    # Restore files
    tar -xzf "$backup_file" -C /
    
    # Reload UFW
    ufw --force enable
    
    log_success "UFW configuration restored"
}

# Main function
main() {
    local command="${1:-help}"
    
    case "$command" in
        "install")
            install_profiles
            ;;
        "configure")
            configure_rules "${2:-development}"
            ;;
        "status")
            show_status
            ;;
        "enable")
            enable_ufw
            ;;
        "disable")
            disable_ufw
            ;;
        "reset")
            reset_ufw
            ;;
        "profiles")
            list_profiles
            ;;
        "logs")
            show_logs "${2:-20}"
            ;;
        "test")
            test_connectivity "${2:-all}"
            ;;
        "backup")
            backup_ufw
            ;;
        "restore")
            restore_ufw "$2"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"