#!/bin/bash

# NovaCron Ubuntu 24.04 Core Deployment Script
# Handles systemd services, AppArmor profiles, and snap packaging

set -euo pipefail

# Configuration
NOVACRON_USER="novacron"
NOVACRON_GROUP="novacron"
INSTALL_DIR="/opt/novacron"
DATA_DIR="/var/lib/novacron"
LOG_DIR="/var/log/novacron"
CONFIG_DIR="/etc/novacron"
APPARMOR_DIR="/etc/apparmor.d"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root"
        exit 1
    fi
}

check_ubuntu_version() {
    if ! grep -q "Ubuntu 24.04" /etc/os-release 2>/dev/null; then
        warn "This script is designed for Ubuntu 24.04, but will continue anyway"
    fi
}

create_user() {
    log "Creating NovaCron user and group..."
    if ! getent group "$NOVACRON_GROUP" > /dev/null; then
        groupadd --system "$NOVACRON_GROUP"
    fi
    
    if ! getent passwd "$NOVACRON_USER" > /dev/null; then
        useradd --system --gid "$NOVACRON_GROUP" \
                --home-dir "$DATA_DIR" \
                --shell /usr/sbin/nologin \
                --comment "NovaCron service user" \
                "$NOVACRON_USER"
    fi
}

create_directories() {
    log "Creating directories..."
    
    # Create main directories
    mkdir -p "$INSTALL_DIR"/{bin,lib}
    mkdir -p "$DATA_DIR"/{vms,db,backups}
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Set permissions
    chown -R "$NOVACRON_USER:$NOVACRON_GROUP" "$DATA_DIR" "$LOG_DIR"
    chown root:root "$CONFIG_DIR"
    chmod 755 "$INSTALL_DIR" "$CONFIG_DIR"
    chmod 750 "$DATA_DIR" "$LOG_DIR"
}

install_binary() {
    log "Installing NovaCron API server binary..."
    
    if [ -f "./api-server-production" ]; then
        cp "./api-server-production" "$INSTALL_DIR/bin/"
        chmod 755 "$INSTALL_DIR/bin/api-server-production"
        chown root:root "$INSTALL_DIR/bin/api-server-production"
    else
        error "Binary ./api-server-production not found. Please build it first."
        exit 1
    fi
}

setup_apparmor() {
    log "Setting up AppArmor profile..."
    
    if [ -f "./apparmor/novacron-api" ]; then
        cp "./apparmor/novacron-api" "$APPARMOR_DIR/"
        chmod 644 "$APPARMOR_DIR/novacron-api"
        
        # Load and enforce AppArmor profile
        if command -v apparmor_parser > /dev/null; then
            apparmor_parser -r "$APPARMOR_DIR/novacron-api"
            log "AppArmor profile loaded and enforced"
        else
            warn "AppArmor not available, skipping profile enforcement"
        fi
    else
        warn "AppArmor profile not found at ./apparmor/novacron-api"
    fi
}

setup_systemd() {
    log "Setting up systemd service..."
    
    if [ -f "./systemd/novacron-api.service" ]; then
        cp "./systemd/novacron-api.service" "$SYSTEMD_DIR/"
        chmod 644 "$SYSTEMD_DIR/novacron-api.service"
        
        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable novacron-api.service
        
        log "Systemd service installed and enabled"
    else
        error "Systemd service file not found at ./systemd/novacron-api.service"
        exit 1
    fi
}

generate_auth_secret() {
    log "Generating authentication secret..."
    
    AUTH_SECRET_FILE="$CONFIG_DIR/auth.secret"
    if [ ! -f "$AUTH_SECRET_FILE" ]; then
        openssl rand -hex 32 > "$AUTH_SECRET_FILE"
        chmod 600 "$AUTH_SECRET_FILE"
        chown "$NOVACRON_USER:$NOVACRON_GROUP" "$AUTH_SECRET_FILE"
        log "Authentication secret generated at $AUTH_SECRET_FILE"
    else
        log "Authentication secret already exists"
    fi
}

setup_database() {
    log "Setting up PostgreSQL database..."
    
    # Install PostgreSQL if not present
    if ! command -v psql > /dev/null; then
        apt update
        apt install -y postgresql postgresql-contrib
    fi
    
    # Start PostgreSQL service
    systemctl enable postgresql
    systemctl start postgresql
    
    # Create database and user
    sudo -u postgres psql -c "CREATE USER novacron WITH PASSWORD 'novacron';" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE DATABASE novacron OWNER novacron;" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE novacron TO novacron;" 2>/dev/null || true
    
    log "PostgreSQL database setup completed"
}

setup_netplan() {
    log "Configuring Netplan for Ubuntu Core..."
    
    # Create a basic netplan configuration for the API service
    NETPLAN_CONFIG="/etc/netplan/90-novacron.yaml"
    
    if [ ! -f "$NETPLAN_CONFIG" ]; then
        cat > "$NETPLAN_CONFIG" << EOF
# NovaCron network configuration
network:
  version: 2
  ethernets:
    novacron-api:
      match:
        name: lo
      optional: true
      addresses:
        - 127.0.0.1/8
EOF
        
        chmod 600 "$NETPLAN_CONFIG"
        log "Netplan configuration created"
    fi
}

create_environment_file() {
    log "Creating environment configuration..."
    
    ENV_FILE="$CONFIG_DIR/environment"
    cat > "$ENV_FILE" << EOF
# NovaCron Environment Configuration
DB_URL=postgresql://novacron:novacron@localhost:5432/novacron
AUTH_SECRET_FILE=$CONFIG_DIR/auth.secret
LOG_LEVEL=info
LOG_FORMAT=json
LOG_OUTPUT=$LOG_DIR/api.log
STORAGE_PATH=$DATA_DIR/vms
API_PORT=8090
WS_PORT=8091
CORS_ALLOWED_ORIGINS=http://localhost:8092,http://localhost:3001
EOF
    
    chmod 644 "$ENV_FILE"
    chown root:root "$ENV_FILE"
}

build_snap() {
    log "Building snap package..."
    
    if command -v snapcraft > /dev/null; then
        if [ -f "./snap/snapcraft.yaml" ]; then
            snapcraft --verbosity=brief
            log "Snap package built successfully"
        else
            warn "Snap configuration not found, skipping snap build"
        fi
    else
        warn "Snapcraft not installed, skipping snap build"
        log "To install snapcraft: sudo snap install snapcraft --classic"
    fi
}

start_services() {
    log "Starting NovaCron services..."
    
    # Start the API service
    if systemctl start novacron-api.service; then
        log "NovaCron API service started successfully"
    else
        error "Failed to start NovaCron API service"
        systemctl status novacron-api.service
        return 1
    fi
    
    # Check if service is running
    if systemctl is-active --quiet novacron-api.service; then
        log "NovaCron API service is running"
    else
        error "NovaCron API service is not running"
        return 1
    fi
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check if API server is responding
    sleep 5  # Give the service time to start
    
    if curl -f -s http://localhost:8090/health > /dev/null; then
        log "✓ API server is responding on port 8090"
    else
        warn "✗ API server is not responding on port 8090"
    fi
    
    # Check if database connection works
    if curl -f -s http://localhost:8090/health | grep -q '"database":"ok"'; then
        log "✓ Database connection is working"
    else
        warn "✗ Database connection may have issues"
    fi
    
    # Show service status
    log "Service status:"
    systemctl status novacron-api.service --no-pager
}

print_summary() {
    log "Deployment Summary"
    echo "=================="
    echo "Installation Directory: $INSTALL_DIR"
    echo "Data Directory: $DATA_DIR"
    echo "Log Directory: $LOG_DIR"
    echo "Configuration Directory: $CONFIG_DIR"
    echo ""
    echo "Services:"
    echo "  - NovaCron API: http://localhost:8090"
    echo "  - Health Check: http://localhost:8090/health"
    echo "  - API Info: http://localhost:8090/api/info"
    echo ""
    echo "Management Commands:"
    echo "  - Start:   sudo systemctl start novacron-api"
    echo "  - Stop:    sudo systemctl stop novacron-api"
    echo "  - Restart: sudo systemctl restart novacron-api"
    echo "  - Status:  sudo systemctl status novacron-api"
    echo "  - Logs:    sudo journalctl -u novacron-api -f"
    echo ""
    echo "Configuration files:"
    echo "  - Environment: $CONFIG_DIR/environment"
    echo "  - Auth Secret: $CONFIG_DIR/auth.secret"
    echo ""
}

main() {
    log "Starting NovaCron Ubuntu 24.04 Core deployment..."
    
    check_root
    check_ubuntu_version
    create_user
    create_directories
    install_binary
    setup_apparmor
    setup_systemd
    generate_auth_secret
    setup_database
    setup_netplan
    create_environment_file
    
    # Build snap if requested
    if [ "${BUILD_SNAP:-}" = "true" ]; then
        build_snap
    fi
    
    start_services
    verify_deployment
    print_summary
    
    log "NovaCron deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    --build-snap)
        BUILD_SNAP=true
        shift
        ;;
    --help|-h)
        echo "Usage: $0 [--build-snap] [--help]"
        echo ""
        echo "Options:"
        echo "  --build-snap    Build snap package after deployment"
        echo "  --help, -h      Show this help message"
        exit 0
        ;;
esac

main "$@"