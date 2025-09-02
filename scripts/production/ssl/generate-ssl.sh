#!/bin/bash

# SSL Certificate Generation and Management Script
# Usage: ./generate-ssl.sh [self-signed|letsencrypt|import] [domain]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SSL_DIR="$PROJECT_ROOT/deployment/ssl"

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

# Default values
CERT_TYPE="${1:-self-signed}"
DOMAIN="${2:-novacron.local}"
DAYS_VALID=365
KEY_SIZE=4096

# Ensure SSL directory exists
mkdir -p "$SSL_DIR"

log "SSL Certificate Management for domain: $DOMAIN"

# Generate self-signed certificate
generate_self_signed() {
    log "Generating self-signed SSL certificate..."
    
    local key_file="$SSL_DIR/key.pem"
    local cert_file="$SSL_DIR/cert.pem"
    local config_file="$SSL_DIR/openssl.conf"
    
    # Create OpenSSL config
    cat > "$config_file" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = San Francisco
O = NovaCron
OU = IT Department
CN = $DOMAIN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = *.$DOMAIN
DNS.3 = localhost
DNS.4 = api.$DOMAIN
DNS.5 = admin.$DOMAIN
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    
    # Generate private key
    openssl genrsa -out "$key_file" $KEY_SIZE
    
    # Generate certificate
    openssl req -new -x509 -key "$key_file" -out "$cert_file" \
        -days $DAYS_VALID -config "$config_file" -extensions v3_req
    
    # Set proper permissions
    chmod 600 "$key_file"
    chmod 644 "$cert_file"
    
    success "Self-signed certificate generated"
    display_certificate_info "$cert_file"
}

# Generate Let's Encrypt certificate
generate_letsencrypt() {
    log "Generating Let's Encrypt SSL certificate..."
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        log "Installing certbot..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y certbot python3-certbot-nginx
        elif command -v yum &> /dev/null; then
            sudo yum install -y certbot python3-certbot-nginx
        else
            error "Package manager not supported. Please install certbot manually."
        fi
    fi
    
    # Generate certificate
    local email="${SSL_EMAIL:-admin@$DOMAIN}"
    
    log "Generating certificate for $DOMAIN with email: $email"
    
    # Use webroot method for validation
    sudo certbot certonly --webroot \
        --webroot-path=/var/www/html \
        --email "$email" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        -d "www.$DOMAIN" \
        -d "api.$DOMAIN" \
        -d "admin.$DOMAIN"
    
    # Copy certificates to our SSL directory
    sudo cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/cert.pem"
    sudo cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/key.pem"
    
    # Set proper permissions
    sudo chown $(id -u):$(id -g) "$SSL_DIR"/*.pem
    chmod 600 "$SSL_DIR/key.pem"
    chmod 644 "$SSL_DIR/cert.pem"
    
    # Setup auto-renewal
    setup_auto_renewal
    
    success "Let's Encrypt certificate generated"
    display_certificate_info "$SSL_DIR/cert.pem"
}

# Setup auto-renewal for Let's Encrypt
setup_auto_renewal() {
    log "Setting up auto-renewal for Let's Encrypt..."
    
    # Create renewal script
    local renewal_script="/usr/local/bin/novacron-ssl-renewal.sh"
    
    sudo tee "$renewal_script" > /dev/null << EOF
#!/bin/bash
# NovaCron SSL Certificate Renewal Script

set -euo pipefail

LOG_FILE="/var/log/novacron/ssl-renewal.log"
mkdir -p "\$(dirname "\$LOG_FILE")"

log() {
    echo "[\$(date +'%Y-%m-%d %H:%M:%S')] \$1" | tee -a "\$LOG_FILE"
}

log "Starting SSL certificate renewal check..."

# Renew certificates
if certbot renew --quiet --no-self-upgrade; then
    log "Certificates renewed successfully"
    
    # Copy renewed certificates
    cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/cert.pem"
    cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/key.pem"
    
    # Restart services to use new certificates
    if systemctl is-active --quiet docker; then
        if docker stack ls | grep -q novacron; then
            log "Restarting Docker Swarm services..."
            docker service update --force novacron_frontend
        elif docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps &>/dev/null; then
            log "Restarting Docker Compose services..."
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" restart frontend
        fi
    fi
    
    log "SSL renewal completed successfully"
else
    log "Certificate renewal failed"
    exit 1
fi
EOF
    
    sudo chmod +x "$renewal_script"
    
    # Add to crontab for automatic renewal
    local cron_entry="0 2 * * 1 $renewal_script"
    
    if ! sudo crontab -l 2>/dev/null | grep -q "$renewal_script"; then
        (sudo crontab -l 2>/dev/null || true; echo "$cron_entry") | sudo crontab -
        log "Auto-renewal cron job added"
    fi
    
    success "Auto-renewal configured"
}

# Import existing certificates
import_certificates() {
    log "Importing existing SSL certificates..."
    
    local cert_source="${SSL_CERT_SOURCE:-}"
    local key_source="${SSL_KEY_SOURCE:-}"
    
    if [[ -z "$cert_source" || -z "$key_source" ]]; then
        error "SSL_CERT_SOURCE and SSL_KEY_SOURCE environment variables must be set"
    fi
    
    if [[ ! -f "$cert_source" ]]; then
        error "Certificate file not found: $cert_source"
    fi
    
    if [[ ! -f "$key_source" ]]; then
        error "Private key file not found: $key_source"
    fi
    
    # Validate certificate and key
    if ! openssl x509 -in "$cert_source" -noout -text &>/dev/null; then
        error "Invalid certificate file"
    fi
    
    if ! openssl rsa -in "$key_source" -check -noout &>/dev/null; then
        error "Invalid private key file"
    fi
    
    # Copy certificates
    cp "$cert_source" "$SSL_DIR/cert.pem"
    cp "$key_source" "$SSL_DIR/key.pem"
    
    # Set proper permissions
    chmod 600 "$SSL_DIR/key.pem"
    chmod 644 "$SSL_DIR/cert.pem"
    
    success "Certificates imported successfully"
    display_certificate_info "$SSL_DIR/cert.pem"
}

# Display certificate information
display_certificate_info() {
    local cert_file="$1"
    
    echo ""
    echo "=== Certificate Information ==="
    
    # Subject
    echo "Subject: $(openssl x509 -in "$cert_file" -noout -subject | sed 's/subject=//')"
    
    # Issuer
    echo "Issuer: $(openssl x509 -in "$cert_file" -noout -issuer | sed 's/issuer=//')"
    
    # Valid dates
    echo "Valid From: $(openssl x509 -in "$cert_file" -noout -startdate | sed 's/notBefore=//')"
    echo "Valid Until: $(openssl x509 -in "$cert_file" -noout -enddate | sed 's/notAfter=//')"
    
    # Subject Alternative Names
    local san=$(openssl x509 -in "$cert_file" -noout -text | grep -A1 "Subject Alternative Name" | tail -1 | sed 's/^[[:space:]]*//' || echo "None")
    echo "SAN: $san"
    
    # Fingerprint
    echo "SHA256 Fingerprint: $(openssl x509 -in "$cert_file" -noout -fingerprint -sha256 | sed 's/SHA256 Fingerprint=//')"
    
    echo ""
}

# Validate certificate chain
validate_certificate() {
    log "Validating SSL certificate..."
    
    local cert_file="$SSL_DIR/cert.pem"
    local key_file="$SSL_DIR/key.pem"
    
    if [[ ! -f "$cert_file" || ! -f "$key_file" ]]; then
        error "Certificate or key file not found"
    fi
    
    # Check if certificate and key match
    local cert_hash=$(openssl x509 -noout -modulus -in "$cert_file" | openssl md5)
    local key_hash=$(openssl rsa -noout -modulus -in "$key_file" | openssl md5)
    
    if [[ "$cert_hash" != "$key_hash" ]]; then
        error "Certificate and private key do not match"
    fi
    
    # Check certificate expiration
    local expiry_date=$(openssl x509 -in "$cert_file" -noout -enddate | sed 's/notAfter=//')
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [[ $days_until_expiry -lt 0 ]]; then
        error "Certificate has expired"
    elif [[ $days_until_expiry -lt 30 ]]; then
        warn "Certificate expires in $days_until_expiry days"
    else
        log "Certificate expires in $days_until_expiry days"
    fi
    
    success "Certificate validation passed"
}

# Create Diffie-Hellman parameters
generate_dhparam() {
    log "Generating Diffie-Hellman parameters..."
    
    local dhparam_file="$SSL_DIR/dhparam.pem"
    
    if [[ ! -f "$dhparam_file" ]]; then
        openssl dhparam -out "$dhparam_file" 2048
        chmod 644 "$dhparam_file"
        success "Diffie-Hellman parameters generated"
    else
        log "Diffie-Hellman parameters already exist"
    fi
}

# Update Docker secrets
update_docker_secrets() {
    log "Updating Docker secrets..."
    
    local cert_file="$SSL_DIR/cert.pem"
    local key_file="$SSL_DIR/key.pem"
    
    # Create new secrets with timestamp
    local timestamp=$(date +%s)
    
    if docker secret create "ssl_cert_v$timestamp" "$cert_file" 2>/dev/null; then
        log "SSL certificate secret created: ssl_cert_v$timestamp"
    fi
    
    if docker secret create "ssl_key_v$timestamp" "$key_file" 2>/dev/null; then
        log "SSL private key secret created: ssl_key_v$timestamp"
    fi
    
    success "Docker secrets updated"
}

# Main execution
main() {
    log "=== SSL Certificate Management ==="
    
    case "$CERT_TYPE" in
        "self-signed")
            generate_self_signed
            ;;
        "letsencrypt")
            generate_letsencrypt
            ;;
        "import")
            import_certificates
            ;;
        "validate")
            validate_certificate
            exit 0
            ;;
        *)
            error "Invalid certificate type. Use: self-signed, letsencrypt, import, or validate"
            ;;
    esac
    
    generate_dhparam
    validate_certificate
    update_docker_secrets
    
    success "=== SSL Certificate Management Completed ==="
}

# Error handling
trap 'error "SSL certificate generation failed at line $LINENO"' ERR

# Run main function
main