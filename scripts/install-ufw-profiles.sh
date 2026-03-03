#!/bin/bash

# NovaCron UFW Profile Installation Script
# This script installs UFW application profiles for NovaCron services
# and sets up basic firewall rules for secure operation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UFW_PROFILES_SOURCE="$PROJECT_ROOT/configs/ufw/applications.d"
UFW_PROFILES_DEST="/etc/ufw/applications.d"

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
        log_info "Usage: sudo $0"
        exit 1
    fi
}

# Check if UFW is installed
check_ufw_installed() {
    if ! command -v ufw >/dev/null 2>&1; then
        log_error "UFW is not installed"
        log_info "Please install UFW first:"
        log_info "  Ubuntu/Debian: sudo apt-get install ufw"
        log_info "  CentOS/RHEL: sudo yum install ufw"
        exit 1
    fi
    
    log_success "UFW is installed"
}

# Create UFW applications directory if it doesn't exist
ensure_applications_dir() {
    if [[ ! -d "$UFW_PROFILES_DEST" ]]; then
        log_info "Creating UFW applications directory: $UFW_PROFILES_DEST"
        mkdir -p "$UFW_PROFILES_DEST"
    fi
    log_success "UFW applications directory exists"
}

# Install UFW profiles
install_profiles() {
    log_info "Installing NovaCron UFW profiles..."
    
    if [[ ! -f "$UFW_PROFILES_SOURCE/novacron" ]]; then
        log_error "NovaCron UFW profile not found at $UFW_PROFILES_SOURCE/novacron"
        exit 1
    fi
    
    # Copy the profile
    cp "$UFW_PROFILES_SOURCE/novacron" "$UFW_PROFILES_DEST/"
    chmod 644 "$UFW_PROFILES_DEST/novacron"
    
    log_success "Installed NovaCron UFW profile"
    
    # Reload UFW application profiles
    ufw app update all >/dev/null 2>&1 || true
    
    log_success "Reloaded UFW application profiles"
}

# Verify profiles are available
verify_profiles() {
    log_info "Verifying installed profiles..."
    
    local profiles=(
        "NovaCron API"
        "NovaCron WebSocket"
        "NovaCron Frontend"
        "NovaCron AI Engine"
        "NovaCron Prometheus"
        "NovaCron Grafana"
        "NovaCron Core Services"
        "NovaCron Monitoring Stack"
        "NovaCron Full Stack"
    )
    
    local missing_profiles=()
    
    for profile in "${profiles[@]}"; do
        if ufw app info "$profile" >/dev/null 2>&1; then
            log_success "Profile available: $profile"
        else
            log_warning "Profile not found: $profile"
            missing_profiles+=("$profile")
        fi
    done
    
    if [[ ${#missing_profiles[@]} -gt 0 ]]; then
        log_error "Some profiles are missing or invalid"
        log_info "Run 'sudo ufw app list | grep NovaCron' to see available profiles"
    else
        log_success "All core profiles are available"
    fi
}

# Show available profiles
show_profiles() {
    log_info "Available NovaCron UFW profiles:"
    echo ""
    ufw app list | grep -i novacron | sed 's/^/  /'
    echo ""
    log_info "Use 'ufw app info \"Profile Name\"' to see profile details"
}

# Main installation function
main() {
    log_info "Starting NovaCron UFW profile installation..."
    echo ""
    
    check_root
    check_ufw_installed
    ensure_applications_dir
    install_profiles
    verify_profiles
    
    echo ""
    show_profiles
    
    echo ""
    log_success "NovaCron UFW profiles installation completed!"
    log_info "Next steps:"
    log_info "1. Configure basic UFW rules with: sudo $PROJECT_ROOT/scripts/setup-ufw-rules.sh"
    log_info "2. Enable UFW with: sudo ufw enable"
    log_info "3. Check status with: sudo ufw status verbose"
}

# Run main function
main "$@"