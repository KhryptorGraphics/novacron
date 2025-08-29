#!/bin/bash
set -euo pipefail

# NovaCron Deployment Script for Ubuntu 24.04
# This script deploys the complete NovaCron backend system with real implementations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_PREFIX="/opt/novacron"
CONFIG_DIR="/etc/novacron"
DATA_DIR="/var/lib/novacron"
LOG_DIR="/var/log/novacron"
USER="novacron"
GROUP="novacron"

echo "🚀 NovaCron Real Backend Deployment"
echo "===================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "❌ This script must be run as root"
    exit 1
fi

# Check Ubuntu version
if ! grep -q "Ubuntu 24.04" /etc/os-release; then
    echo "⚠️  Warning: This script is optimized for Ubuntu 24.04"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to create system user
create_system_user() {
    echo "👤 Creating system user and group..."
    if ! id "$USER" &>/dev/null; then
        useradd --system --create-home --home-dir "$DATA_DIR" \
                --shell /usr/sbin/nologin --user-group "$USER"
        usermod -aG libvirt "$USER"
        echo "✅ Created user: $USER"
    else
        echo "✅ User $USER already exists"
    fi
}

# Function to create directories
create_directories() {
    echo "📁 Creating directories..."
    
    # Create main directories
    mkdir -p "$INSTALL_PREFIX"/{bin,lib,share}
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"/{vms,storage,cache,logs}
    mkdir -p "$LOG_DIR"
    
    # Set ownership
    chown -R "$USER:$GROUP" "$DATA_DIR"
    chown -R "$USER:$GROUP" "$LOG_DIR"
    
    # Set permissions
    chmod 755 "$INSTALL_PREFIX"/{bin,lib,share}
    chmod 750 "$CONFIG_DIR"
    chmod 750 "$DATA_DIR"
    chmod 755 "$LOG_DIR"
    
    echo "✅ Directories created with proper permissions"
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing system dependencies..."
    
    # Update package list
    apt-get update
    
    # Install dependencies
    apt-get install -y \
        postgresql postgresql-contrib \
        libvirt-daemon-system libvirt-clients \
        qemu-kvm qemu-utils \
        bridge-utils vlan \
        curl wget jq \
        build-essential \
        apparmor-utils \
        ufw \
        redis-server \
        python3 python3-pip \
        git
    
    echo "✅ Dependencies installed"
}

# Function to build application
build_application() {
    echo "🏗️  Building NovaCron application..."
    
    cd "$PROJECT_ROOT/backend/cmd/api-server"
    CGO_ENABLED=1 go build -o api-server-real -ldflags "-s -w" .
    
    # Install binary
    install -m 755 api-server-real "$INSTALL_PREFIX/bin/api-server"
    
    echo "✅ Application built and installed"
}

# Function to setup database
setup_database() {
    echo "🗄️  Setting up PostgreSQL database..."
    
    # Start PostgreSQL
    systemctl start postgresql
    systemctl enable postgresql
    
    # Create database and user
    sudo -u postgres psql <<EOF
CREATE DATABASE novacron;
CREATE USER novacron WITH ENCRYPTED PASSWORD 'novacron123';
GRANT ALL PRIVILEGES ON DATABASE novacron TO novacron;
ALTER USER novacron CREATEDB;
\q
EOF

    # Test connection
    export PGPASSWORD='novacron123'
    if psql -h localhost -U novacron -d novacron -c "SELECT 1;" > /dev/null 2>&1; then
        echo "✅ Database setup completed"
    else
        echo "❌ Database setup failed"
        exit 1
    fi
}

# Function to install configuration
install_configuration() {
    echo "⚙️  Installing configuration files..."
    
    # Create environment configuration
    cat > "$CONFIG_DIR/environment" <<EOF
# NovaCron Environment Configuration
NODE_ENV=production
LOG_LEVEL=info

# Database Configuration
DB_URL=postgresql://novacron:novacron123@localhost:5432/novacron

# API Configuration
API_HOST=0.0.0.0
API_PORT=8090
WS_PORT=8091

# Authentication
AUTH_SECRET=changeme_in_production_$(openssl rand -base64 32)

# Storage Configuration
STORAGE_PATH=$DATA_DIR/vms
CACHE_PATH=$DATA_DIR/cache

# Hypervisor Configuration
HYPERVISOR_DRIVER=kvm
LIBVIRT_URI=qemu:///system

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Logging
LOG_DIR=$LOG_DIR
EOF
    
    # Set permissions
    chmod 600 "$CONFIG_DIR/environment"
    chown root:$GROUP "$CONFIG_DIR/environment"
    
    echo "✅ Configuration installed"
}

# Function to install systemd services
install_systemd_services() {
    echo "🎯 Installing systemd services..."
    
    # Install service files
    cp "$PROJECT_ROOT/systemd/novacron-api.service" /etc/systemd/system/
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    systemctl enable novacron-api
    
    echo "✅ Systemd services installed"
}

# Function to install AppArmor profiles
install_apparmor_profiles() {
    echo "🛡️  Installing AppArmor profiles..."
    
    if command -v apparmor_parser &> /dev/null; then
        # Install profiles
        cp "$PROJECT_ROOT/apparmor/novacron-api" /etc/apparmor.d/
        
        # Parse and load profiles
        apparmor_parser -r /etc/apparmor.d/novacron-api
        
        echo "✅ AppArmor profiles installed and loaded"
    else
        echo "⚠️  AppArmor not available, skipping security profiles"
    fi
}

# Function to configure UFW firewall
configure_ufw_firewall() {
    echo "🔥 Configuring UFW firewall..."
    
    # Install UFW profiles
    if [[ -x "$PROJECT_ROOT/scripts/install-ufw-profiles.sh" ]]; then
        "$PROJECT_ROOT/scripts/install-ufw-profiles.sh"
    else
        echo "⚠️  UFW profile installer not found, installing manually..."
        
        # Ensure UFW applications directory exists
        mkdir -p /etc/ufw/applications.d
        
        # Copy UFW profile
        if [[ -f "$PROJECT_ROOT/configs/ufw/applications.d/novacron" ]]; then
            cp "$PROJECT_ROOT/configs/ufw/applications.d/novacron" /etc/ufw/applications.d/
            ufw app update all >/dev/null 2>&1 || true
        fi
    fi
    
    # Check if UFW is already configured
    if ufw status | grep -q "Status: active"; then
        echo "⚠️  UFW is already active, skipping automatic rule configuration"
        echo "   Run manually: sudo $PROJECT_ROOT/scripts/setup-ufw-rules.sh production"
    else
        echo "📋 UFW profiles installed but firewall not configured"
        echo "   To configure: sudo $PROJECT_ROOT/scripts/setup-ufw-rules.sh production"
        echo "   To enable: sudo ufw enable"
    fi
    
    echo "✅ UFW configuration completed"
}

# Function to start services
start_services() {
    echo "🚀 Starting services..."
    
    # Start database
    systemctl start postgresql
    
    # Start Redis
    systemctl start redis-server
    systemctl enable redis-server
    
    # Start NovaCron services
    systemctl start novacron-api
    
    # Check service status
    sleep 5
    
    if systemctl is-active --quiet novacron-api; then
        echo "✅ NovaCron API service is running"
    else
        echo "❌ NovaCron API service failed to start"
        journalctl -u novacron-api -n 20 --no-pager
        exit 1
    fi
    
    echo "✅ All services started"
}

# Function to run health checks
run_health_checks() {
    echo "🏥 Running health checks..."
    
    # Check database connectivity
    if sudo -u postgres psql -d novacron -c "SELECT 1;" > /dev/null 2>&1; then
        echo "✅ Database connectivity OK"
    else
        echo "❌ Database connectivity failed"
        exit 1
    fi
    
    # Check Redis connectivity
    if redis-cli ping | grep -q PONG; then
        echo "✅ Redis connectivity OK"
    else
        echo "❌ Redis connectivity failed"
        exit 1
    fi
    
    # Check API endpoint
    sleep 10
    if curl -f http://localhost:8090/health > /dev/null 2>&1; then
        echo "✅ API health check OK"
        echo "📄 API Info:"
        curl -s http://localhost:8090/api/info | jq . 2>/dev/null || echo "Could not format JSON"
    else
        echo "❌ API health check failed"
        echo "Service logs:"
        journalctl -u novacron-api -n 20 --no-pager
        exit 1
    fi
    
    echo "✅ All health checks passed"
}

# Function to print deployment summary
print_summary() {
    echo ""
    echo "🎉 NovaCron Real Backend Deployment Complete!"
    echo "============================================="
    echo ""
    echo "📍 Installation Details:"
    echo "  - Install prefix: $INSTALL_PREFIX"
    echo "  - Configuration: $CONFIG_DIR"
    echo "  - Data directory: $DATA_DIR"
    echo "  - Log directory: $LOG_DIR"
    echo "  - System user: $USER"
    echo ""
    echo "🌐 Service URLs:"
    echo "  - API Server: http://localhost:8090"
    echo "  - WebSocket: ws://localhost:8091"
    echo "  - Health Check: http://localhost:8090/health"
    echo "  - API Info: http://localhost:8090/api/info"
    echo ""
    echo "🔧 Management Commands:"
    echo "  - Start service: systemctl start novacron-api"
    echo "  - Stop service: systemctl stop novacron-api"
    echo "  - View logs: journalctl -u novacron-api -f"
    echo "  - Check status: systemctl status novacron-api"
    echo ""
    echo "🔥 Firewall Commands:"
    echo "  - Configure UFW rules: sudo $PROJECT_ROOT/scripts/setup-ufw-rules.sh production"
    echo "  - Enable firewall: sudo ufw enable"
    echo "  - Check firewall status: sudo ufw status verbose"
    echo "  - List NovaCron profiles: ufw app list | grep NovaCron"
    echo ""
    echo "📚 Configuration files:"
    echo "  - Environment: $CONFIG_DIR/environment"
    echo "  - UFW profiles: /etc/ufw/applications.d/novacron"
    echo ""
    echo "🔐 Default credentials:"
    echo "  - Database: novacron:novacron123"
    echo "  - Default admin user: admin:admin123 (change after first login)"
    echo ""
    echo "⚠️  Important: Update AUTH_SECRET in production!"
    echo "⚠️  Important: Configure firewall before exposing to network!"
    echo ""
}

# Main deployment function
main() {
    echo "Starting deployment process..."
    
    create_system_user
    create_directories
    install_dependencies
    build_application
    setup_database
    install_configuration
    install_systemd_services
    install_apparmor_profiles
    configure_ufw_firewall
    start_services
    run_health_checks
    print_summary
    
    echo "🎊 Deployment completed successfully!"
}

# Run main function
main "$@"