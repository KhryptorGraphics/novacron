#!/bin/sh
# NovaCron API Entrypoint Script
set -e

# Create config file if it doesn't exist
if [ ! -f /etc/novacron/config.yaml ]; then
    echo "Creating default configuration file..."
    mkdir -p /etc/novacron
    cat > /etc/novacron/config.yaml << EOF
# NovaCron API Configuration
logLevel: ${LOG_LEVEL:-info}
api:
  host: 0.0.0.0
  port: ${API_PORT:-8090}
  tlsEnabled: false
database:
  url: ${DB_URL:-postgresql://postgres:postgres@postgres:5432/novacron}
auth:
  secret: ${AUTH_SECRET:-changeme}
  tokenExpiration: 24h
hypervisors:
  addresses: ${HYPERVISOR_ADDRS:-novacron-hypervisor:9000}
websocket:
  enabled: true
  host: 0.0.0.0
  port: 8091
metrics:
  enabled: true
  host: 0.0.0.0
  port: 9090
EOF
    echo "Configuration file created at /etc/novacron/config.yaml"
fi

# Create log directory if it doesn't exist
if [ ! -d /var/log/novacron ]; then
    echo "Creating log directory..."
    mkdir -p /var/log/novacron
    chown -R novacron:novacron /var/log/novacron
fi

# Wait for the database to be ready
if [ ! -z "$DB_URL" ]; then
    echo "Waiting for database to be ready..."
    DB_HOST=$(echo $DB_URL | sed -n 's/.*@\(.*\):.*/\1/p')
    DB_PORT=$(echo $DB_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if [ -z "$DB_PORT" ]; then
        DB_PORT=5432
    fi
    
    RETRIES=30
    until nc -z $DB_HOST $DB_PORT || [ $RETRIES -eq 0 ]; do
        echo "Waiting for database at $DB_HOST:$DB_PORT, $RETRIES retries left..."
        RETRIES=$((RETRIES-1))
        sleep 1
    done
    
    if [ $RETRIES -eq 0 ]; then
        echo "WARNING: Could not connect to database, continuing startup anyway"
    else
        echo "Database is ready"
    fi
fi

# Check hypervisor connectivity
echo "Checking hypervisor connectivity..."
for HYPERVISOR in $(echo ${HYPERVISOR_ADDRS:-novacron-hypervisor:9000} | tr ',' ' '); do
    HOST=$(echo $HYPERVISOR | cut -d: -f1)
    PORT=$(echo $HYPERVISOR | cut -d: -f2)
    
    if nc -z $HOST $PORT 2>/dev/null; then
        echo "Hypervisor at $HOST:$PORT is reachable"
    else
        echo "WARNING: Hypervisor at $HOST:$PORT is not reachable"
    fi
done

# Print startup message
echo "Starting NovaCron API Service..."
echo "Log Level: ${LOG_LEVEL:-info}"
echo "API Port: ${API_PORT:-8090}"
echo "Database URL: ${DB_URL:-postgresql://postgres:postgres@postgres:5432/novacron}"
echo "Hypervisor Addresses: ${HYPERVISOR_ADDRS:-novacron-hypervisor:9000}"

# Start the application
if [ "$1" = "novacron-api" ]; then
    echo "Starting Go API service..."
    exec "$@"
elif [ "$1" = "python" ]; then
    echo "Starting Python service..."
    exec "$@"
else
    echo "Starting default service..."
    exec "$@"
fi
