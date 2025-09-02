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
