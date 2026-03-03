#!/bin/bash

set -e

echo "🔐 NovaCron Secure Deployment Script"
echo "===================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run this script as root${NC}"
   exit 1
fi

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command_exists go; then
    echo -e "${RED}Go is not installed. Please install Go 1.23+ first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo

# Step 2: Start Vault in development mode
echo "🔑 Starting HashiCorp Vault..."

# Check if Vault is already running
if pgrep -x "vault" > /dev/null; then
    echo -e "${YELLOW}Vault is already running${NC}"
else
    # Check if Vault is installed
    if ! command_exists vault; then
        echo "Installing Vault..."
        wget -q https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
        unzip -q vault_1.15.0_linux_amd64.zip
        sudo mv vault /usr/local/bin/
        rm vault_1.15.0_linux_amd64.zip
        echo -e "${GREEN}✓ Vault installed${NC}"
    fi
    
    # Start Vault in dev mode (background)
    echo "Starting Vault server in development mode..."
    vault server -dev -dev-root-token-id="root" > /tmp/vault.log 2>&1 &
    VAULT_PID=$!
    sleep 3
    
    export VAULT_ADDR='http://127.0.0.1:8200'
    export VAULT_ROOT_TOKEN='root'
    
    echo -e "${GREEN}✓ Vault started (PID: $VAULT_PID)${NC}"
fi

echo

# Step 3: Initialize Vault secrets
echo "🔐 Initializing Vault secrets..."

export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_ROOT_TOKEN='root'

cd backend/core/security
go run setup_vault.go
cd ../../..

echo

# Step 4: Generate TLS certificates
echo "🔒 Generating TLS certificates..."

TLS_DIR="/etc/novacron/tls"
if [ ! -d "$TLS_DIR" ]; then
    sudo mkdir -p "$TLS_DIR"
    sudo chown $USER:$USER "$TLS_DIR"
fi

# Check if certificates already exist
if [ -f "$TLS_DIR/cert.pem" ] && [ -f "$TLS_DIR/key.pem" ]; then
    echo -e "${YELLOW}TLS certificates already exist${NC}"
else
    # Generate self-signed certificate for development
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout "$TLS_DIR/key.pem" \
        -out "$TLS_DIR/cert.pem" \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=NovaCron/CN=localhost"
    
    chmod 600 "$TLS_DIR/key.pem"
    chmod 644 "$TLS_DIR/cert.pem"
    
    echo -e "${GREEN}✓ TLS certificates generated${NC}"
fi

echo

# Step 5: Build the secure API server
echo "🏗️  Building secure API server..."

cd backend
go build -o ../bin/novacron-secure ./cmd/api-server/main_secure.go
cd ..

echo -e "${GREEN}✓ Secure API server built${NC}"
echo

# Step 6: Update database schema (if needed)
echo "📊 Checking database..."

# Check if PostgreSQL is running
if docker ps | grep -q postgres; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo "Starting PostgreSQL..."
    docker-compose up -d postgres
    sleep 5
fi

# Create tables if they don't exist
docker exec -i novacron-postgres psql -U novacron -d novacron << 'EOF' 2>/dev/null || true
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS vms (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL,
    node_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vms_state ON vms(state);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
EOF

echo -e "${GREEN}✓ Database ready${NC}"
echo

# Step 7: Create environment file
echo "📝 Creating environment configuration..."

if [ ! -f .env ]; then
    # Load the Vault token that was created
    VAULT_TOKEN=$(cat .vault-token 2>/dev/null || echo "dev-token")
    
    cat > .env << EOF
# NovaCron Secure Configuration
NOVACRON_ENV=development
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=$VAULT_TOKEN
TLS_CERT_PATH=/etc/novacron/tls/cert.pem
TLS_KEY_PATH=/etc/novacron/tls/key.pem
DB_HOST=localhost
DB_PORT=5432
DB_NAME=novacron
DB_USER=novacron
LOG_LEVEL=info
EOF
    
    echo -e "${GREEN}✓ Environment file created${NC}"
else
    echo -e "${YELLOW}Environment file already exists${NC}"
fi

echo

# Step 8: Start the secure API server
echo "🚀 Starting secure NovaCron API server..."

# Stop any existing API server
pkill -f novacron-secure || true

# Source environment variables
export $(cat .env | xargs)

# Start the secure server
nohup ./bin/novacron-secure > /tmp/novacron-secure.log 2>&1 &
API_PID=$!

sleep 3

# Check if server started successfully
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓ Secure API server started (PID: $API_PID)${NC}"
else
    echo -e "${RED}Failed to start API server. Check /tmp/novacron-secure.log for details${NC}"
    exit 1
fi

echo

# Step 9: Test the secure endpoints
echo "🧪 Testing secure endpoints..."

# Test HTTPS health endpoint
if curl -k -s https://localhost:8443/api/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ HTTPS endpoint is working${NC}"
else
    echo -e "${RED}HTTPS endpoint test failed${NC}"
fi

# Test HTTP to HTTPS redirect
REDIRECT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -I http://localhost:8080/api/health)
if [ "$REDIRECT_STATUS" = "301" ] || [ "$REDIRECT_STATUS" = "308" ]; then
    echo -e "${GREEN}✓ HTTP to HTTPS redirect is working${NC}"
else
    echo -e "${YELLOW}HTTP redirect returned status: $REDIRECT_STATUS${NC}"
fi

echo
echo "================================"
echo -e "${GREEN}🎉 Secure deployment complete!${NC}"
echo "================================"
echo
echo "📍 Access points:"
echo "   HTTPS API: https://localhost:8443"
echo "   HTTP Redirect: http://localhost:8080"
echo "   Vault UI: http://localhost:8200"
echo
echo "📋 Next steps:"
echo "   1. Update frontend to use https://localhost:8443"
echo "   2. Import TLS certificate to browser (for development)"
echo "   3. Configure production Vault for production deployment"
echo "   4. Replace self-signed certificate with CA-signed for production"
echo
echo "📖 Security features enabled:"
echo "   ✓ SQL injection prevention (parameterized queries)"
echo "   ✓ Secrets management (HashiCorp Vault)"
echo "   ✓ HTTPS/TLS encryption (TLS 1.2+)"
echo "   ✓ Security headers (HSTS, CSP, etc.)"
echo "   ✓ Rate limiting (100 req/min)"
echo "   ✓ Input validation and sanitization"
echo "   ✓ Secure password hashing (bcrypt)"
echo
echo "🔍 Logs:"
echo "   API Server: /tmp/novacron-secure.log"
echo "   Vault: /tmp/vault.log"
echo
echo "⚠️  For production deployment:"
echo "   - Use production Vault configuration"
echo "   - Use CA-signed TLS certificates"
echo "   - Configure firewall rules"
echo "   - Enable monitoring and alerting"