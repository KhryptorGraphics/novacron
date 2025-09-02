#!/bin/bash

# NovaCron Production Secrets Generator
# Version: 1.0.0
# Description: Generate secure secrets for production deployment

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../deployment/secrets"
K8S_DIR="${SCRIPT_DIR}/../deployment/kubernetes"
BACKUP_DIR="${OUTPUT_DIR}/backup"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            ;;
    esac
}

# Check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    local deps=("openssl" "base64" "kubectl")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log "ERROR" "$dep is required but not installed"
            exit 1
        fi
    done
    
    log "INFO" "All dependencies satisfied"
}

# Create directory structure
create_directories() {
    log "INFO" "Creating directory structure..."
    
    mkdir -p "$OUTPUT_DIR" "$BACKUP_DIR"
    
    log "INFO" "Directory structure created"
}

# Generate random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d '\n'
}

# Generate random hex string
generate_hex() {
    local length=${1:-32}
    openssl rand -hex "$length" | tr -d '\n'
}

# Base64 encode string
b64encode() {
    echo -n "$1" | base64 -w 0
}

# Generate JWT secret (256-bit key)
generate_jwt_secret() {
    log "INFO" "Generating JWT secret..."
    openssl rand -base64 64 | tr -d '\n'
}

# Generate API encryption key (256-bit key)
generate_api_key() {
    log "INFO" "Generating API encryption key..."
    openssl rand -hex 64 | tr -d '\n'
}

# Generate database credentials
generate_db_credentials() {
    log "INFO" "Generating database credentials..."
    
    local db_user="novacron_$(openssl rand -hex 4)"
    local db_password=$(generate_password 32)
    local db_name="novacron_prod"
    local db_host="${DB_HOST:-novacron-postgres.novacron.svc.cluster.local}"
    local db_port="${DB_PORT:-5432}"
    
    local db_url="postgres://${db_user}:${db_password}@${db_host}:${db_port}/${db_name}?sslmode=require"
    local readonly_db_url="postgres://readonly_${db_user}:${db_password}@${db_host}:${db_port}/${db_name}?sslmode=require&default_transaction_isolation=repeatable%20read"
    
    echo "DB_USER=$db_user"
    echo "DB_PASSWORD=$db_password"
    echo "DB_NAME=$db_name"
    echo "DATABASE_URL=$db_url"
    echo "READONLY_DATABASE_URL=$readonly_db_url"
}

# Generate Redis credentials
generate_redis_credentials() {
    log "INFO" "Generating Redis credentials..."
    
    local redis_password=$(generate_password 24)
    local redis_host="${REDIS_HOST:-novacron-redis.novacron.svc.cluster.local}"
    local redis_port="${REDIS_PORT:-6379}"
    local redis_url="redis://:${redis_password}@${redis_host}:${redis_port}/0"
    
    echo "REDIS_PASSWORD=$redis_password"
    echo "REDIS_URL=$redis_url"
}

# Generate monitoring credentials
generate_monitoring_credentials() {
    log "INFO" "Generating monitoring credentials..."
    
    local grafana_admin_password=$(generate_password 16)
    local grafana_secret_key=$(generate_hex 32)
    local grafana_db_user="grafana_$(openssl rand -hex 4)"
    local grafana_db_password=$(generate_password 24)
    
    echo "GRAFANA_ADMIN_PASSWORD=$grafana_admin_password"
    echo "GRAFANA_SECRET_KEY=$grafana_secret_key"
    echo "GRAFANA_DB_USER=$grafana_db_user"
    echo "GRAFANA_DB_PASSWORD=$grafana_db_password"
}

# Generate SMTP credentials template
generate_smtp_credentials() {
    log "INFO" "Generating SMTP credentials template..."
    
    cat << EOF
# SMTP Configuration (fill in your values)
SMTP_HOST=smtp.your-provider.com
SMTP_PORT=587
SMTP_USER=noreply@novacron.local
SMTP_PASSWORD=your-smtp-password
SMTP_FROM=noreply@novacron.local
EOF
}

# Generate AWS credentials template
generate_aws_credentials() {
    log "INFO" "Generating AWS credentials template..."
    
    cat << EOF
# AWS Configuration (fill in your values)
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=us-west-2
BACKUP_S3_BUCKET=novacron-backups-$(openssl rand -hex 8)
EOF
}

# Generate TLS certificates
generate_tls_certificates() {
    log "INFO" "Generating TLS certificates..."
    
    local cert_dir="$OUTPUT_DIR/certificates"
    mkdir -p "$cert_dir"
    
    # Generate CA private key
    openssl genrsa -out "$cert_dir/ca-key.pem" 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 365 -key "$cert_dir/ca-key.pem" -sha256 -out "$cert_dir/ca.pem" -subj "/C=US/ST=CA/L=San Francisco/O=NovaCron/OU=IT/CN=NovaCron CA"
    
    # Generate server private key
    openssl genrsa -out "$cert_dir/server-key.pem" 4096
    
    # Generate server certificate signing request
    openssl req -subj "/C=US/ST=CA/L=San Francisco/O=NovaCron/OU=IT/CN=*.novacron.local" -sha256 -new -key "$cert_dir/server-key.pem" -out "$cert_dir/server.csr"
    
    # Create extensions file
    cat > "$cert_dir/server-extfile.cnf" << EOF
subjectAltName = DNS:novacron.local,DNS:*.novacron.local,DNS:api.novacron.local,DNS:ws.novacron.local,IP:127.0.0.1
extendedKeyUsage = serverAuth
EOF
    
    # Generate server certificate
    openssl x509 -req -days 365 -in "$cert_dir/server.csr" -CA "$cert_dir/ca.pem" -CAkey "$cert_dir/ca-key.pem" -out "$cert_dir/server.pem" -extfile "$cert_dir/server-extfile.cnf"
    
    # Generate database client certificate
    openssl genrsa -out "$cert_dir/db-client-key.pem" 4096
    openssl req -subj "/C=US/ST=CA/L=San Francisco/O=NovaCron/OU=IT/CN=db-client" -new -key "$cert_dir/db-client-key.pem" -out "$cert_dir/db-client.csr"
    openssl x509 -req -days 365 -in "$cert_dir/db-client.csr" -CA "$cert_dir/ca.pem" -CAkey "$cert_dir/ca-key.pem" -out "$cert_dir/db-client.pem"
    
    # Set appropriate permissions
    chmod 400 "$cert_dir"/*-key.pem
    chmod 444 "$cert_dir"/*.pem
    
    log "INFO" "TLS certificates generated in $cert_dir"
    
    # Return certificate values for Kubernetes secrets
    echo "CA_CERT=$(cat "$cert_dir/ca.pem" | base64 -w 0)"
    echo "SERVER_CERT=$(cat "$cert_dir/server.pem" | base64 -w 0)"
    echo "SERVER_KEY=$(cat "$cert_dir/server-key.pem" | base64 -w 0)"
    echo "DB_CLIENT_CERT=$(cat "$cert_dir/db-client.pem" | base64 -w 0)"
    echo "DB_CLIENT_KEY=$(cat "$cert_dir/db-client-key.pem" | base64 -w 0)"
}

# Generate webhook URLs template
generate_webhook_urls() {
    log "INFO" "Generating webhook URLs template..."
    
    cat << EOF
# Webhook URLs (fill in your values)
SECURITY_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SECURITY/WEBHOOK
HEALTH_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/HEALTH/WEBHOOK
ALERTS_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/ALERTS/WEBHOOK
EOF
}

# Create environment file
create_env_file() {
    log "INFO" "Creating environment file..."
    
    local env_file="$OUTPUT_DIR/.env.production"
    
    cat > "$env_file" << EOF
# NovaCron Production Environment Configuration
# Generated on $(date)
# Version: 1.0.0

# Core Application
JWT_SECRET=$(generate_jwt_secret)
API_ENCRYPTION_KEY=$(generate_api_key)
ADMIN_API_KEY=$(generate_hex 32)

# Database Configuration
$(generate_db_credentials)

# Redis Configuration
$(generate_redis_credentials)

# Monitoring Configuration
$(generate_monitoring_credentials)

# TLS Certificates
$(generate_tls_certificates)

# External Services Templates
$(generate_smtp_credentials)

$(generate_aws_credentials)

$(generate_webhook_urls)

# Domain Configuration
DOMAIN_NAME=novacron.local
API_PORT=8091
WS_PORT=8093
METRICS_PORT=9090

# Security Configuration
ENVIRONMENT=production
DEBUG=false
CORS_ALLOWED_ORIGINS=https://*.novacron.local

# Storage Paths
DATA_DIR=/var/lib/novacron
BACKUP_DIR=/var/lib/novacron/backups
LOG_DIR=/var/log/novacron
CERT_DIR=/etc/novacron/certs

EOF
    
    chmod 600 "$env_file"
    log "INFO" "Environment file created: $env_file"
}

# Create Kubernetes secrets YAML
create_k8s_secrets() {
    log "INFO" "Creating Kubernetes secrets..."
    
    local secrets_file="$OUTPUT_DIR/secrets-generated.yaml"
    
    # Source the environment file to get the variables
    source "$OUTPUT_DIR/.env.production"
    
    cat > "$secrets_file" << EOF
# Generated NovaCron Kubernetes Secrets
# Created on $(date)

apiVersion: v1
kind: Secret
metadata:
  name: novacron-app-secrets
  namespace: novacron
type: Opaque
data:
  jwt-secret: $(b64encode "$JWT_SECRET")
  api-encryption-key: $(b64encode "$API_ENCRYPTION_KEY")
  admin-api-key: $(b64encode "$ADMIN_API_KEY")

---
apiVersion: v1
kind: Secret
metadata:
  name: novacron-db-secrets
  namespace: novacron
type: Opaque
data:
  postgres-user: $(b64encode "$DB_USER")
  postgres-password: $(b64encode "$DB_PASSWORD")
  postgres-database: $(b64encode "$DB_NAME")
  database-url: $(b64encode "$DATABASE_URL")
  readonly-database-url: $(b64encode "$READONLY_DATABASE_URL")

---
apiVersion: v1
kind: Secret
metadata:
  name: novacron-redis-secrets
  namespace: novacron
type: Opaque
data:
  redis-password: $(b64encode "$REDIS_PASSWORD")
  redis-url: $(b64encode "$REDIS_URL")

---
apiVersion: v1
kind: Secret
metadata:
  name: novacron-tls-certs
  namespace: novacron
type: kubernetes.io/tls
data:
  tls.crt: $SERVER_CERT
  tls.key: $SERVER_KEY
  ca.crt: $CA_CERT

---
apiVersion: v1
kind: Secret
metadata:
  name: novacron-db-tls-certs
  namespace: novacron
type: Opaque
data:
  client.crt: $DB_CLIENT_CERT
  client.key: $DB_CLIENT_KEY
  ca.crt: $CA_CERT

---
apiVersion: v1
kind: Secret
metadata:
  name: novacron-monitoring-secrets
  namespace: novacron
type: Opaque
data:
  grafana-admin-password: $(b64encode "$GRAFANA_ADMIN_PASSWORD")
  grafana-secret-key: $(b64encode "$GRAFANA_SECRET_KEY")

EOF
    
    chmod 600 "$secrets_file"
    log "INFO" "Kubernetes secrets file created: $secrets_file"
}

# Create setup instructions
create_setup_instructions() {
    log "INFO" "Creating setup instructions..."
    
    local instructions_file="$OUTPUT_DIR/SETUP_INSTRUCTIONS.md"
    
    cat > "$instructions_file" << EOF
# NovaCron Production Deployment Setup Instructions

## Overview
This directory contains generated secrets and configuration for NovaCron production deployment.

## Generated Files
- \`.env.production\` - Environment variables for production
- \`secrets-generated.yaml\` - Kubernetes secrets (ready to apply)
- \`certificates/\` - TLS certificates directory
- \`SETUP_INSTRUCTIONS.md\` - This file

## Prerequisites
1. Kubernetes cluster with ingress controller
2. Storage classes configured
3. DNS configured for *.novacron.local
4. External services configured (SMTP, AWS, etc.)

## Deployment Steps

### 1. Review and Update External Service Configuration
Edit the following sections in \`.env.production\`:
- SMTP configuration
- AWS credentials (if using S3 for backups)
- Webhook URLs for notifications

### 2. Create Kubernetes Namespace
\`\`\`bash
kubectl apply -f ../kubernetes/namespace.yaml
\`\`\`

### 3. Apply Generated Secrets
\`\`\`bash
kubectl apply -f secrets-generated.yaml
\`\`\`

### 4. Apply ConfigMaps and RBAC
\`\`\`bash
kubectl apply -f ../kubernetes/configmap.yaml
kubectl apply -f ../kubernetes/rbac.yaml
\`\`\`

### 5. Deploy Application
\`\`\`bash
kubectl apply -f ../kubernetes/deployments.yaml
kubectl apply -f ../kubernetes/services.yaml
kubectl apply -f ../kubernetes/ingress.yaml
\`\`\`

### 6. Verify Deployment
\`\`\`bash
kubectl get pods -n novacron
kubectl get services -n novacron
curl -f https://novacron.local/health
\`\`\`

## Post-Deployment Configuration

### Database Setup
Run migrations after the database is ready:
\`\`\`bash
kubectl exec -it deployment/novacron-api -n novacron -- /app/migrate up
\`\`\`

### Create Admin User
\`\`\`bash
kubectl exec -it deployment/novacron-api -n novacron -- /app/create-admin-user
\`\`\`

### Configure Monitoring
1. Access Grafana at https://grafana.novacron.local
2. Login with admin/\$GRAFANA_ADMIN_PASSWORD
3. Configure data sources and dashboards

## Security Recommendations

1. **Rotate Secrets Regularly**: Set up automated secret rotation
2. **Enable Network Policies**: Apply strict network policies
3. **Monitor Access**: Enable audit logging and monitoring
4. **Backup Strategy**: Configure automated backups
5. **Certificate Management**: Set up cert-manager for automatic renewal

## Backup and Recovery

### Manual Backup
\`\`\`bash
kubectl create job --from=cronjob/novacron-backup manual-backup-\$(date +%s) -n novacron
\`\`\`

### Restore from Backup
\`\`\`bash
# Restore instructions are in the backup service documentation
\`\`\`

## Troubleshooting

### Check Logs
\`\`\`bash
kubectl logs deployment/novacron-api -n novacron
kubectl logs deployment/novacron-frontend -n novacron
\`\`\`

### Debug Database Connection
\`\`\`bash
kubectl exec -it deployment/novacron-api -n novacron -- psql \$DATABASE_URL
\`\`\`

### Check Ingress
\`\`\`bash
kubectl describe ingress novacron-ingress -n novacron
\`\`\`

## Support
For support, check the documentation at /docs/ or contact the operations team.

Generated on: $(date)
Version: 1.0.0
EOF
    
    log "INFO" "Setup instructions created: $instructions_file"
}

# Apply secrets to Kubernetes (optional)
apply_to_kubernetes() {
    if [ "${APPLY_TO_K8S:-false}" = "true" ]; then
        log "INFO" "Applying secrets to Kubernetes..."
        
        if kubectl cluster-info &> /dev/null; then
            kubectl apply -f "$OUTPUT_DIR/secrets-generated.yaml"
            log "INFO" "Secrets applied to Kubernetes cluster"
        else
            log "WARN" "No Kubernetes cluster available, skipping apply"
        fi
    fi
}

# Main execution
main() {
    log "INFO" "NovaCron Production Secrets Generator v1.0.0"
    
    check_dependencies
    create_directories
    create_env_file
    create_k8s_secrets
    create_setup_instructions
    apply_to_kubernetes
    
    log "INFO" "Secrets generation completed successfully!"
    log "INFO" "Generated files location: $OUTPUT_DIR"
    log "WARN" "IMPORTANT: Review and update external service configurations before deployment"
    log "WARN" "IMPORTANT: Store secrets securely and never commit them to version control"
}

# Display usage information
usage() {
    cat << EOF
NovaCron Production Secrets Generator

Usage: $0 [OPTIONS]

Options:
    -a, --apply-k8s     Apply secrets to Kubernetes cluster
    -h, --help          Show this help message

Environment Variables:
    DB_HOST             Database host (default: novacron-postgres.novacron.svc.cluster.local)
    DB_PORT             Database port (default: 5432)
    REDIS_HOST          Redis host (default: novacron-redis.novacron.svc.cluster.local)
    REDIS_PORT          Redis port (default: 6379)
    APPLY_TO_K8S        Apply secrets to Kubernetes (default: false)

Examples:
    $0                  Generate secrets only
    $0 -a               Generate secrets and apply to Kubernetes
    APPLY_TO_K8S=true $0  Generate and apply using environment variable

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--apply-k8s)
            export APPLY_TO_K8S=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"