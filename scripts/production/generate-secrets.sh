#!/bin/bash

# NovaCron Secret Generation Script
# Usage: ./generate-secrets.sh [staging|production]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENVIRONMENT="${1:-production}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; exit 1; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"; }

log "Generating secrets for $ENVIRONMENT environment"

# Create secrets directory
SECRETS_DIR="$PROJECT_ROOT/deployment/secrets/$ENVIRONMENT"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# Generate random password
generate_password() {
    local length="${1:-32}"
    local complexity="${2:-high}"
    
    case "$complexity" in
        "high")
            # High complexity: alphanumeric + special chars
            openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length" | tr -d '\n'
            ;;
        "medium")
            # Medium complexity: alphanumeric only
            cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w "$length" | head -n 1 | tr -d '\n'
            ;;
        "low")
            # Low complexity: letters only
            cat /dev/urandom | tr -dc 'a-zA-Z' | fold -w "$length" | head -n 1 | tr -d '\n'
            ;;
    esac
}

# Generate JWT secret
generate_jwt_secret() {
    local length="${1:-64}"
    # JWT secret should be very secure
    openssl rand -hex "$length" | tr -d '\n'
}

# Generate database credentials
generate_database_secrets() {
    log "Generating database secrets..."
    
    local db_user="novacron_${ENVIRONMENT}"
    local db_password
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        db_password=$(generate_password 48 high)
    else
        db_password=$(generate_password 32 high)
    fi
    
    echo "$db_user" > "$SECRETS_DIR/postgres_user"
    echo "$db_password" > "$SECRETS_DIR/postgres_password"
    
    chmod 600 "$SECRETS_DIR/postgres_user" "$SECRETS_DIR/postgres_password"
    success "Database secrets generated"
}

# Generate Redis credentials
generate_redis_secrets() {
    log "Generating Redis secrets..."
    
    local redis_password
    if [[ "$ENVIRONMENT" == "production" ]]; then
        redis_password=$(generate_password 48 high)
    else
        redis_password=$(generate_password 32 high)
    fi
    
    echo "$redis_password" > "$SECRETS_DIR/redis_password"
    chmod 600 "$SECRETS_DIR/redis_password"
    
    success "Redis secrets generated"
}

# Generate JWT secrets
generate_jwt_secrets() {
    log "Generating JWT secrets..."
    
    local jwt_secret
    if [[ "$ENVIRONMENT" == "production" ]]; then
        jwt_secret=$(generate_jwt_secret 64)
    else
        jwt_secret=$(generate_jwt_secret 32)
    fi
    
    echo "$jwt_secret" > "$SECRETS_DIR/jwt_secret"
    chmod 600 "$SECRETS_DIR/jwt_secret"
    
    success "JWT secrets generated"
}

# Generate Grafana credentials
generate_grafana_secrets() {
    log "Generating Grafana secrets..."
    
    local grafana_admin_user="admin"
    local grafana_admin_password
    local grafana_secret_key
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        grafana_admin_password=$(generate_password 32 high)
        grafana_secret_key=$(generate_password 32 high)
    else
        grafana_admin_password=$(generate_password 16 medium)
        grafana_secret_key=$(generate_password 16 medium)
    fi
    
    echo "$grafana_admin_user" > "$SECRETS_DIR/grafana_admin_user"
    echo "$grafana_admin_password" > "$SECRETS_DIR/grafana_admin_password"
    echo "$grafana_secret_key" > "$SECRETS_DIR/grafana_secret_key"
    
    chmod 600 "$SECRETS_DIR"/grafana_*
    success "Grafana secrets generated"
}

# Generate encryption keys
generate_encryption_keys() {
    log "Generating encryption keys..."
    
    # Application encryption key
    local encryption_key=$(generate_password 32 high)
    echo "$encryption_key" > "$SECRETS_DIR/encryption_key"
    
    # Session signing key
    local session_key=$(generate_password 32 high)
    echo "$session_key" > "$SECRETS_DIR/session_key"
    
    # CSRF token key
    local csrf_key=$(generate_password 32 high)
    echo "$csrf_key" > "$SECRETS_DIR/csrf_key"
    
    chmod 600 "$SECRETS_DIR"/*_key
    success "Encryption keys generated"
}

# Generate API keys
generate_api_keys() {
    log "Generating API keys..."
    
    # Internal service API key
    local internal_api_key=$(generate_password 48 high)
    echo "$internal_api_key" > "$SECRETS_DIR/internal_api_key"
    
    # Webhook signing key
    local webhook_secret=$(generate_password 32 high)
    echo "$webhook_secret" > "$SECRETS_DIR/webhook_secret"
    
    chmod 600 "$SECRETS_DIR"/*_key "$SECRETS_DIR"/*_secret
    success "API keys generated"
}

# Generate AWS credentials template
generate_aws_template() {
    log "Generating AWS credentials template..."
    
    cat > "$SECRETS_DIR/aws_access_key" << EOF
# Replace with actual AWS Access Key ID
AKIA...
EOF
    
    cat > "$SECRETS_DIR/aws_secret_key" << EOF
# Replace with actual AWS Secret Access Key
...
EOF
    
    chmod 600 "$SECRETS_DIR"/aws_*
    warn "AWS credentials template created - update with real values"
}

# Create Docker secrets (for Swarm)
create_docker_secrets() {
    if docker info | grep -q "Swarm: active"; then
        log "Creating Docker Swarm secrets..."
        
        local timestamp=$(date +%s)
        local secrets=(
            "postgres_user"
            "postgres_password" 
            "jwt_secret"
            "redis_password"
            "grafana_admin_user"
            "grafana_admin_password"
            "grafana_secret_key"
            "encryption_key"
            "session_key"
            "csrf_key"
            "internal_api_key"
            "webhook_secret"
        )
        
        for secret in "${secrets[@]}"; do
            if [[ -f "$SECRETS_DIR/$secret" ]]; then
                local secret_name="${secret}_v${timestamp}"
                if docker secret create "$secret_name" "$SECRETS_DIR/$secret" 2>/dev/null; then
                    log "Created Docker secret: $secret_name"
                    
                    # Create a symlink with the standard name
                    docker secret create "$secret" "$SECRETS_DIR/$secret" 2>/dev/null || true
                else
                    warn "Failed to create Docker secret: $secret_name"
                fi
            fi
        done
        
        success "Docker secrets created"
    else
        log "Docker Swarm not active, skipping Docker secret creation"
    fi
}

# Create Kubernetes secrets
create_kubernetes_secrets() {
    if kubectl cluster-info &>/dev/null; then
        log "Creating Kubernetes secrets..."
        
        # Create generic secrets
        kubectl create secret generic database-credentials \
            --from-file=username="$SECRETS_DIR/postgres_user" \
            --from-file=password="$SECRETS_DIR/postgres_password" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        kubectl create secret generic redis-credentials \
            --from-file=password="$SECRETS_DIR/redis_password" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        kubectl create secret generic jwt-secret \
            --from-file=secret="$SECRETS_DIR/jwt_secret" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        kubectl create secret generic grafana-credentials \
            --from-file=admin-user="$SECRETS_DIR/grafana_admin_user" \
            --from-file=admin-password="$SECRETS_DIR/grafana_admin_password" \
            --from-file=secret-key="$SECRETS_DIR/grafana_secret_key" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        kubectl create secret generic encryption-keys \
            --from-file=encryption-key="$SECRETS_DIR/encryption_key" \
            --from-file=session-key="$SECRETS_DIR/session_key" \
            --from-file=csrf-key="$SECRETS_DIR/csrf_key" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        kubectl create secret generic api-keys \
            --from-file=internal-api-key="$SECRETS_DIR/internal_api_key" \
            --from-file=webhook-secret="$SECRETS_DIR/webhook_secret" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f - || true
        
        success "Kubernetes secrets created"
    else
        log "Kubernetes not accessible, skipping Kubernetes secret creation"
    fi
}

# Generate environment file
generate_env_file() {
    log "Generating environment file..."
    
    local env_file="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT.generated"
    
    cat > "$env_file" << EOF
# NovaCron $ENVIRONMENT Environment - Generated $(date)
# WARNING: This file contains sensitive information

# Database Configuration
POSTGRES_USER=$(cat "$SECRETS_DIR/postgres_user")
POSTGRES_PASSWORD=$(cat "$SECRETS_DIR/postgres_password")
POSTGRES_DB=novacron_${ENVIRONMENT}

# Redis Configuration  
REDIS_PASSWORD=$(cat "$SECRETS_DIR/redis_password")

# Authentication
JWT_SECRET=$(cat "$SECRETS_DIR/jwt_secret")

# Grafana
GRAFANA_ADMIN_USER=$(cat "$SECRETS_DIR/grafana_admin_user")
GRAFANA_ADMIN_PASSWORD=$(cat "$SECRETS_DIR/grafana_admin_password")

# Encryption
ENCRYPTION_KEY=$(cat "$SECRETS_DIR/encryption_key")
SESSION_KEY=$(cat "$SECRETS_DIR/session_key")
CSRF_KEY=$(cat "$SECRETS_DIR/csrf_key")

# API Keys
INTERNAL_API_KEY=$(cat "$SECRETS_DIR/internal_api_key")
WEBHOOK_SECRET=$(cat "$SECRETS_DIR/webhook_secret")

# AWS (update with real values)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-1
EOF
    
    chmod 600 "$env_file"
    success "Environment file generated: $env_file"
}

# Create secrets summary
create_summary() {
    log "Creating secrets summary..."
    
    local summary_file="$SECRETS_DIR/secrets_summary.txt"
    
    cat > "$summary_file" << EOF
NovaCron Secrets Summary - $ENVIRONMENT Environment
Generated: $(date)

Database:
- User: $(cat "$SECRETS_DIR/postgres_user")
- Password: [REDACTED] ($(wc -c < "$SECRETS_DIR/postgres_password") chars)

Redis:
- Password: [REDACTED] ($(wc -c < "$SECRETS_DIR/redis_password") chars)

JWT:
- Secret: [REDACTED] ($(wc -c < "$SECRETS_DIR/jwt_secret") chars)

Grafana:
- Admin User: $(cat "$SECRETS_DIR/grafana_admin_user")
- Admin Password: [REDACTED] ($(wc -c < "$SECRETS_DIR/grafana_admin_password") chars)

Encryption Keys:
- Encryption Key: [REDACTED] ($(wc -c < "$SECRETS_DIR/encryption_key") chars)
- Session Key: [REDACTED] ($(wc -c < "$SECRETS_DIR/session_key") chars)
- CSRF Key: [REDACTED] ($(wc -c < "$SECRETS_DIR/csrf_key") chars)

API Keys:
- Internal API Key: [REDACTED] ($(wc -c < "$SECRETS_DIR/internal_api_key") chars)
- Webhook Secret: [REDACTED] ($(wc -c < "$SECRETS_DIR/webhook_secret") chars)

Files Location: $SECRETS_DIR
Permissions: 700 (directory), 600 (files)

IMPORTANT:
1. Store these secrets securely
2. Never commit to version control
3. Use environment-specific secrets
4. Rotate secrets regularly
5. Update AWS credentials with real values

Backup Command:
tar czf novacron-secrets-$ENVIRONMENT-$(date +%Y%m%d).tar.gz -C $(dirname "$SECRETS_DIR") $(basename "$SECRETS_DIR")
EOF
    
    chmod 600 "$summary_file"
    success "Secrets summary created: $summary_file"
}

# Backup existing secrets
backup_existing_secrets() {
    if [[ -d "$SECRETS_DIR" ]]; then
        log "Backing up existing secrets..."
        
        local backup_dir="$SECRETS_DIR.backup.$(date +%Y%m%d-%H%M%S)"
        cp -r "$SECRETS_DIR" "$backup_dir"
        
        success "Existing secrets backed up to: $backup_dir"
    fi
}

# Main execution
main() {
    log "=== NovaCron Secret Generation Started ==="
    
    # Backup existing secrets
    backup_existing_secrets
    
    # Generate all secrets
    generate_database_secrets
    generate_redis_secrets
    generate_jwt_secrets
    generate_grafana_secrets
    generate_encryption_keys
    generate_api_keys
    generate_aws_template
    
    # Create deployment-specific secrets
    create_docker_secrets
    create_kubernetes_secrets
    
    # Generate configuration files
    generate_env_file
    create_summary
    
    success "=== Secret Generation Completed ==="
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Review generated secrets in: $SECRETS_DIR"
    echo "2. Update AWS credentials with real values"
    echo "3. Securely store the secrets backup"
    echo "4. Test the deployment with new secrets"
    echo "5. Set up secret rotation schedule"
    echo ""
    
    # Security reminders
    warn "SECURITY REMINDERS:"
    warn "- Never commit secrets to version control"
    warn "- Rotate secrets regularly (quarterly recommended)"
    warn "- Use different secrets for each environment"
    warn "- Store production secrets in a secure vault"
    warn "- Limit access to secrets on a need-to-know basis"
}

# Verify environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment. Use 'staging' or 'production'"
fi

# Check dependencies
for cmd in openssl kubectl docker; do
    if ! command -v "$cmd" &> /dev/null; then
        warn "Command not found: $cmd (some features may not work)"
    fi
done

# Error handling
trap 'error "Secret generation failed at line $LINENO"' ERR

# Run main function
main