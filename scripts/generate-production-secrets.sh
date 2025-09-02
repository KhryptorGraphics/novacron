#!/bin/bash

# NovaCron Production Secrets Generator
# Generates secure secrets for production deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NovaCron Production Secrets Generator ===${NC}"
echo

# Function to generate secure random string
generate_secret() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

# Function to generate strong password
generate_password() {
    local length=${1:-24}
    # Ensure mix of uppercase, lowercase, numbers, and special characters
    local password=""
    password+=$(openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-8)
    password+=$(echo $RANDOM | md5sum | head -c 4)
    password+="@#$"
    password+=$(openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-8)
    echo "$password" | fold -w1 | shuf | tr -d '\n' | cut -c1-"$length"
}

# Create .env.production if it doesn't exist
ENV_FILE=".env.production"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Warning: $ENV_FILE already exists. Creating backup...${NC}"
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
fi

echo -e "${GREEN}Generating secure production secrets...${NC}"
echo

# Generate secrets
AUTH_SECRET=$(generate_secret 64)
JWT_SECRET=$(generate_secret 32)
SESSION_SECRET=$(generate_secret 32)
ENCRYPTION_KEY=$(generate_secret 32)
API_KEY=$(generate_secret 48)
WEBHOOK_SECRET=$(generate_secret 32)

# Database passwords
DB_PASSWORD=$(generate_password 24)
REDIS_PASSWORD=$(generate_password 20)
RABBITMQ_PASSWORD=$(generate_password 20)

# Cloud provider keys (placeholders - should be replaced with actual keys)
AWS_SECRET_KEY=$(generate_secret 40)
AZURE_CLIENT_SECRET=$(generate_secret 32)
GCP_PRIVATE_KEY=$(generate_secret 64)

# Admin credentials
ADMIN_PASSWORD=$(generate_password 16)
ADMIN_API_KEY=$(generate_secret 32)

# Write to .env.production
cat > "$ENV_FILE" << EOF
# ============================================
# NovaCron Production Environment Variables
# Generated: $(date)
# ============================================
# CRITICAL: Never commit this file to version control!

# Application Settings
NODE_ENV=production
APP_NAME=NovaCron
APP_URL=https://api.novacron.io
APP_PORT=8090
WS_PORT=8091

# Security Keys (256-bit)
AUTH_SECRET=$AUTH_SECRET
JWT_SECRET=$JWT_SECRET
SESSION_SECRET=$SESSION_SECRET
ENCRYPTION_KEY=$ENCRYPTION_KEY

# API Keys
API_KEY=$API_KEY
WEBHOOK_SECRET=$WEBHOOK_SECRET
ADMIN_API_KEY=$ADMIN_API_KEY

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=novacron_prod
DB_USER=novacron
DB_PASSWORD=$DB_PASSWORD
DB_SSL_MODE=require
DB_MAX_CONNECTIONS=100
DB_CONNECTION_TIMEOUT=30

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=$REDIS_PASSWORD
REDIS_DB=0
REDIS_MAX_RETRIES=3
REDIS_TLS_ENABLED=true

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=novacron
RABBITMQ_PASSWORD=$RABBITMQ_PASSWORD
RABBITMQ_VHOST=/novacron

# TLS Configuration
TLS_ENABLED=true
TLS_CERT_FILE=/etc/novacron/certs/server.crt
TLS_KEY_FILE=/etc/novacron/certs/server.key
TLS_CA_FILE=/etc/novacron/certs/ca.crt
TLS_MIN_VERSION=1.3

# Cloud Provider Credentials (Replace with actual values)
AWS_ACCESS_KEY_ID=REPLACE_WITH_ACTUAL_KEY
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_KEY
AWS_REGION=us-east-1

AZURE_CLIENT_ID=REPLACE_WITH_ACTUAL_ID
AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET
AZURE_TENANT_ID=REPLACE_WITH_ACTUAL_TENANT

GCP_PROJECT_ID=REPLACE_WITH_ACTUAL_PROJECT
GCP_PRIVATE_KEY=$GCP_PRIVATE_KEY
GCP_CLIENT_EMAIL=REPLACE_WITH_ACTUAL_EMAIL

# Monitoring & Logging
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
LOG_LEVEL=info
LOG_FORMAT=json
LOG_OUTPUT=/var/log/novacron/app.log

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_MS=60000

# CORS Settings
CORS_ENABLED=true
CORS_ORIGINS=https://app.novacron.io,https://novacron.io
CORS_CREDENTIALS=true

# Admin Credentials (Change immediately after first login!)
ADMIN_EMAIL=admin@novacron.io
ADMIN_PASSWORD=$ADMIN_PASSWORD

# Feature Flags
FEATURE_AI_OPTIMIZATION=true
FEATURE_MULTI_CLOUD=true
FEATURE_FEDERATION=true
FEATURE_BACKUP=true
FEATURE_MONITORING=true

# Performance Settings
MAX_WORKERS=10
CONNECTION_POOL_SIZE=50
CACHE_TTL=3600
REQUEST_TIMEOUT=30000

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
BACKUP_S3_BUCKET=novacron-backups

# Email Configuration (Optional)
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=REPLACE_WITH_ACTUAL_SENDGRID_KEY
SMTP_FROM=noreply@novacron.io

# Sentry Error Tracking (Optional)
SENTRY_DSN=REPLACE_WITH_ACTUAL_SENTRY_DSN
SENTRY_ENVIRONMENT=production

EOF

echo -e "${GREEN}✓ Production secrets generated successfully!${NC}"
echo

# Create TLS certificate generation script
TLS_SCRIPT="scripts/generate-tls-certs.sh"
cat > "$TLS_SCRIPT" << 'EOFTLS'
#!/bin/bash

# Generate self-signed TLS certificates for development/testing
# For production, use Let's Encrypt or proper CA-signed certificates

CERT_DIR="/etc/novacron/certs"
mkdir -p "$CERT_DIR"

# Generate private key
openssl genrsa -out "$CERT_DIR/server.key" 4096

# Generate certificate signing request
openssl req -new -key "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" \
    -subj "/C=US/ST=State/L=City/O=NovaCron/CN=api.novacron.io"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in "$CERT_DIR/server.csr" \
    -signkey "$CERT_DIR/server.key" -out "$CERT_DIR/server.crt"

# Generate CA certificate (for client verification)
openssl req -x509 -new -nodes -key "$CERT_DIR/server.key" \
    -sha256 -days 365 -out "$CERT_DIR/ca.crt" \
    -subj "/C=US/ST=State/L=City/O=NovaCron/CN=NovaCron CA"

# Set proper permissions
chmod 600 "$CERT_DIR/server.key"
chmod 644 "$CERT_DIR/server.crt"
chmod 644 "$CERT_DIR/ca.crt"

echo "TLS certificates generated in $CERT_DIR"
EOFTLS

chmod +x "$TLS_SCRIPT"
echo -e "${GREEN}✓ TLS certificate generation script created${NC}"
echo

# Create password strength validation script
cat > "scripts/validate-passwords.sh" << 'EOFVAL'
#!/bin/bash

# Validate password strength in .env.production

ENV_FILE=".env.production"

check_password_strength() {
    local password=$1
    local name=$2
    local min_length=12
    
    # Check length
    if [ ${#password} -lt $min_length ]; then
        echo "❌ $name is too short (minimum $min_length characters)"
        return 1
    fi
    
    # Check for uppercase
    if ! [[ "$password" =~ [A-Z] ]]; then
        echo "❌ $name missing uppercase letters"
        return 1
    fi
    
    # Check for lowercase
    if ! [[ "$password" =~ [a-z] ]]; then
        echo "❌ $name missing lowercase letters"
        return 1
    fi
    
    # Check for numbers
    if ! [[ "$password" =~ [0-9] ]]; then
        echo "❌ $name missing numbers"
        return 1
    fi
    
    # Check for special characters
    if ! [[ "$password" =~ [^a-zA-Z0-9] ]]; then
        echo "❌ $name missing special characters"
        return 1
    fi
    
    echo "✓ $name meets security requirements"
    return 0
}

# Extract and validate passwords from .env.production
if [ -f "$ENV_FILE" ]; then
    echo "Validating password strength..."
    
    DB_PASSWORD=$(grep "^DB_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    REDIS_PASSWORD=$(grep "^REDIS_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    ADMIN_PASSWORD=$(grep "^ADMIN_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    
    check_password_strength "$DB_PASSWORD" "Database password"
    check_password_strength "$REDIS_PASSWORD" "Redis password"
    check_password_strength "$ADMIN_PASSWORD" "Admin password"
else
    echo "Error: $ENV_FILE not found"
    exit 1
fi
EOFVAL

chmod +x "scripts/validate-passwords.sh"

echo -e "${YELLOW}⚠️  Important Security Reminders:${NC}"
echo "1. Never commit .env.production to version control"
echo "2. Replace placeholder cloud provider credentials with actual values"
echo "3. Use Let's Encrypt or CA-signed certificates for production TLS"
echo "4. Change the admin password immediately after first login"
echo "5. Enable 2FA for all admin accounts"
echo "6. Rotate all secrets regularly (at least every 90 days)"
echo "7. Use a secrets management service (Vault, AWS Secrets Manager) in production"
echo
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review and update .env.production with actual cloud credentials"
echo "2. Run: ./scripts/generate-tls-certs.sh (or use Let's Encrypt)"
echo "3. Run: ./scripts/validate-passwords.sh"
echo "4. Secure the .env.production file: chmod 600 .env.production"
echo "5. Test the configuration with: npm run validate:env"
echo
echo -e "${GREEN}✅ Security configuration complete!${NC}"