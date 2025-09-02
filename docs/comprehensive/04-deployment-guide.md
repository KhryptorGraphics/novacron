# NovaCron Production Deployment Guide

## Deployment Overview

This guide provides comprehensive instructions for deploying NovaCron in production environments, including prerequisites, infrastructure setup, configuration management, and operational procedures.

## Prerequisites

### System Requirements

#### Minimum Hardware Requirements
- **CPU**: 8 cores (16 threads recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 500GB SSD (1TB+ recommended)
- **Network**: 10Gbps network interface

#### Recommended Production Hardware
- **CPU**: 16-32 cores with virtualization support (Intel VT-x or AMD-V)
- **RAM**: 64-128GB ECC memory
- **Storage**: 
  - OS: 100GB NVMe SSD
  - VM Storage: 2TB+ NVMe SSD with RAID 10
  - Backup Storage: 5TB+ HDD with RAID 5
- **Network**: Dual 10Gbps NICs with bonding/teaming

#### Operating System Support
- **Primary**: Ubuntu 22.04 LTS, RHEL 9, CentOS Stream 9
- **Container**: Docker 24.0+, Kubernetes 1.28+
- **Hypervisor**: KVM/QEMU 8.0+, libvirt 9.0+

### Software Dependencies

#### Required Services
```bash
# Database
PostgreSQL 15+
Redis 7.0+

# Monitoring
Prometheus 2.45+
Grafana 10.0+
AlertManager 0.25+

# Security
HashiCorp Vault 1.14+ (optional)
```

#### Development Tools
```bash
# Build Tools
Go 1.23.0+
Node.js 18.0+
Docker 24.0+
kubectl 1.28+

# Utilities
git 2.40+
curl 8.0+
jq 1.6+
```

## Infrastructure Setup

### 1. Container Deployment (Recommended)

#### Docker Compose Deployment

**Create deployment directory:**
```bash
mkdir -p /opt/novacron
cd /opt/novacron
```

**Docker Compose Configuration (`docker-compose.yml`):**
```yaml
version: '3.8'

services:
  # Database Services
  postgresql:
    image: postgres:15-alpine
    container_name: novacron-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: novacron
      POSTGRES_USER: novacron
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U novacron -d novacron"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: novacron-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Application Services
  novacron-api:
    image: novacron/api:latest
    container_name: novacron-api
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgres://novacron:${POSTGRES_PASSWORD}@postgresql:5432/novacron?sslmode=disable
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - JWT_SECRET=${JWT_SECRET}
      - LOG_LEVEL=info
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - /var/lib/libvirt:/var/lib/libvirt
      - /var/run/libvirt:/var/run/libvirt
    ports:
      - "8080:8080"
      - "8081:8081"
    depends_on:
      postgresql:
        condition: service_healthy
      redis:
        condition: service_healthy
    privileged: true
    devices:
      - /dev/kvm:/dev/kvm
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  novacron-frontend:
    image: novacron/frontend:latest
    container_name: novacron-frontend
    restart: unless-stopped
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1
      - NEXT_PUBLIC_WS_URL=ws://localhost:8081
    ports:
      - "3000:3000"
    depends_on:
      - novacron-api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Monitoring Services
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: novacron-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.0.0
    container_name: novacron-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3001:3000"

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

**Environment Configuration (`.env`):**
```bash
# Database Configuration
POSTGRES_PASSWORD=your-secure-postgres-password
REDIS_PASSWORD=your-secure-redis-password

# Application Security
JWT_SECRET=your-jwt-secret-key-minimum-32-characters
API_SECRET=your-api-secret-key

# Monitoring
GRAFANA_PASSWORD=your-grafana-admin-password

# SSL/TLS Configuration
TLS_CERT_PATH=/etc/ssl/certs/novacron.crt
TLS_KEY_PATH=/etc/ssl/private/novacron.key

# Cloud Provider Credentials (if using)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AZURE_CLIENT_ID=your-azure-client-id
AZURE_CLIENT_SECRET=your-azure-secret
GCP_SERVICE_ACCOUNT_KEY=/path/to/gcp-key.json
```

**Deploy with Docker Compose:**
```bash
# Generate strong passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 64)
export GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Start services
docker-compose up -d

# Verify deployment
docker-compose ps
docker-compose logs -f
```

### 2. Kubernetes Deployment

#### Kubernetes Manifests

**Namespace (`namespace.yaml`):**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: novacron
  labels:
    name: novacron
```

**ConfigMap (`configmap.yaml`):**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: novacron-config
  namespace: novacron
data:
  app.yaml: |
    server:
      api_port: "8080"
      ws_port: "8081"
      read_timeout: 30s
      write_timeout: 30s
      idle_timeout: 120s
      shutdown_timeout: 30s
    
    database:
      url: "postgres://novacron:password@postgresql:5432/novacron?sslmode=disable"
      max_connections: 50
      max_idle_connections: 25
      conn_max_lifetime: 300s
    
    redis:
      url: "redis://:password@redis:6379"
      db: 0
      max_retries: 3
    
    logging:
      level: "info"
      format: "json"
      output: "stdout"
```

**Secrets (`secrets.yaml`):**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: novacron-secrets
  namespace: novacron
type: Opaque
data:
  postgres-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  jwt-secret: <base64-encoded-jwt-secret>
```

**PostgreSQL Deployment (`postgresql.yaml`):**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: novacron
spec:
  serviceName: postgresql
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgresql
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: novacron
        - name: POSTGRES_USER
          value: novacron
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        ports:
        - containerPort: 5432
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd

---
apiVersion: v1
kind: Service
metadata:
  name: postgresql
  namespace: novacron
spec:
  selector:
    app: postgresql
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

**NovaCron API Deployment (`api.yaml`):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-api
  namespace: novacron
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novacron-api
  template:
    metadata:
      labels:
        app: novacron-api
    spec:
      containers:
      - name: novacron-api
        image: novacron/api:latest
        env:
        - name: DATABASE_URL
          value: "postgres://novacron:$(POSTGRES_PASSWORD)@postgresql:5432/novacron?sslmode=disable"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: postgres-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: jwt-secret
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: kvm-device
          mountPath: /dev/kvm
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: websocket
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          privileged: true
      volumes:
      - name: config
        configMap:
          name: novacron-config
      - name: kvm-device
        hostPath:
          path: /dev/kvm
      nodeSelector:
        node-type: hypervisor

---
apiVersion: v1
kind: Service
metadata:
  name: novacron-api
  namespace: novacron
spec:
  selector:
    app: novacron-api
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  - port: 8081
    targetPort: 8081
    name: websocket
  type: ClusterIP
```

**Ingress Configuration (`ingress.yaml`):**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: novacron-ingress
  namespace: novacron
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  tls:
  - hosts:
    - api.novacron.com
    - app.novacron.com
    secretName: novacron-tls
  rules:
  - host: api.novacron.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: novacron-api
            port:
              number: 8080
  - host: app.novacron.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: novacron-frontend
            port:
              number: 3000
```

**Deploy to Kubernetes:**
```bash
# Create namespace and secrets
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml

# Deploy database
kubectl apply -f postgresql.yaml

# Deploy application
kubectl apply -f api.yaml
kubectl apply -f frontend.yaml

# Configure ingress
kubectl apply -f ingress.yaml

# Verify deployment
kubectl get pods -n novacron
kubectl get services -n novacron
kubectl get ingress -n novacron
```

### 3. Bare Metal Deployment

#### System Preparation

**Install Dependencies (Ubuntu 22.04):**
```bash
#!/bin/bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y build-essential git curl wget

# Install Go
wget https://golang.org/dl/go1.23.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib
sudo -u postgres createuser -s novacron
sudo -u postgres createdb novacron

# Install Redis
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Install KVM/QEMU
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils
sudo usermod -aG libvirt $USER
sudo usermod -aG kvm $USER

# Install monitoring tools
sudo apt install -y prometheus grafana
```

**Build Application:**
```bash
# Clone repository
git clone https://github.com/novacron/novacron.git
cd novacron

# Build backend
cd backend
go mod download
go build -o bin/novacron-api cmd/api-server/main.go

# Build frontend
cd ../frontend
npm install
npm run build
```

**System Service Configuration:**

**Backend Service (`/etc/systemd/system/novacron-api.service`):**
```ini
[Unit]
Description=NovaCron API Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=novacron
Group=novacron
WorkingDirectory=/opt/novacron
ExecStart=/opt/novacron/bin/novacron-api
Restart=always
RestartSec=10
Environment=DATABASE_URL=postgres://novacron:password@localhost/novacron?sslmode=disable
Environment=REDIS_URL=redis://localhost:6379
Environment=JWT_SECRET=your-jwt-secret
Environment=LOG_LEVEL=info

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/novacron/logs /var/lib/libvirt
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**Frontend Service (`/etc/systemd/system/novacron-frontend.service`):**
```ini
[Unit]
Description=NovaCron Frontend
After=network.target novacron-api.service

[Service]
Type=simple
User=novacron
Group=novacron
WorkingDirectory=/opt/novacron/frontend
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10
Environment=NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
```

**Enable and Start Services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable novacron-api
sudo systemctl enable novacron-frontend
sudo systemctl start novacron-api
sudo systemctl start novacron-frontend

# Verify services
sudo systemctl status novacron-api
sudo systemctl status novacron-frontend
```

## Configuration Management

### 1. Application Configuration

**Main Configuration (`config/app.yaml`):**
```yaml
# Server Configuration
server:
  api_port: "8080"
  ws_port: "8081"
  read_timeout: 30s
  write_timeout: 30s
  idle_timeout: 120s
  shutdown_timeout: 30s
  tls:
    enabled: true
    cert_file: "/etc/ssl/certs/novacron.crt"
    key_file: "/etc/ssl/private/novacron.key"

# Database Configuration
database:
  url: "postgres://novacron:password@localhost:5432/novacron?sslmode=require"
  max_connections: 50
  max_idle_connections: 25
  conn_max_lifetime: 300s
  conn_max_idle_time: 60s

# Redis Configuration
redis:
  url: "redis://localhost:6379"
  password: "your-redis-password"
  db: 0
  max_retries: 3
  dial_timeout: 5s
  read_timeout: 3s
  write_timeout: 3s
  pool_size: 50

# Authentication Configuration
auth:
  jwt_secret: "your-jwt-secret-key"
  token_expiry: 24h
  refresh_token_expiry: 168h
  bcrypt_cost: 12

# VM Configuration
vm:
  storage_path: "/var/lib/novacron/storage"
  image_path: "/var/lib/novacron/images"
  network_bridge: "virbr0"
  default_driver: "kvm"

# Monitoring Configuration
monitoring:
  metrics_enabled: true
  metrics_path: "/metrics"
  tracing_enabled: true
  jaeger_endpoint: "http://jaeger:14268/api/traces"

# Logging Configuration
logging:
  level: "info"
  format: "json"
  output: "stdout"
  file: "/var/log/novacron/app.log"
  max_size: 100
  max_backups: 3
  max_age: 28
  compress: true
```

**Security Configuration (`config/security.yaml`):**
```yaml
# TLS Configuration
tls:
  min_version: "1.2"
  cipher_suites:
    - "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
    - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
    - "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305"
    - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"

# CORS Configuration
cors:
  allowed_origins:
    - "https://app.novacron.com"
    - "https://api.novacron.com"
  allowed_methods:
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
    - "OPTIONS"
  allowed_headers:
    - "Content-Type"
    - "Authorization"
    - "X-API-Key"
  credentials: true
  max_age: 86400

# Rate Limiting
rate_limiting:
  enabled: true
  requests_per_hour: 1000
  burst_size: 100
  cleanup_interval: 60s

# Security Headers
security_headers:
  x_frame_options: "DENY"
  x_content_type_options: "nosniff"
  x_xss_protection: "1; mode=block"
  strict_transport_security: "max-age=31536000; includeSubDomains"
  content_security_policy: "default-src 'self'; script-src 'self' 'unsafe-inline'"
```

### 2. Database Configuration

**Database Initialization Script:**
```bash
#!/bin/bash
# Initialize NovaCron Database

set -e

DB_NAME="novacron"
DB_USER="novacron"
DB_PASSWORD="${POSTGRES_PASSWORD:-your-secure-password}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

# Create database and user
sudo -u postgres psql << EOF
CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
EOF

# Run migrations
cd /opt/novacron/backend/migrations
for migration in *.sql; do
    echo "Running migration: $migration"
    PGPASSWORD=${DB_PASSWORD} psql -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME} -f "$migration"
done

echo "Database initialization completed successfully"
```

### 3. Monitoring Configuration

**Prometheus Configuration (`prometheus.yml`):**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'novacron-api'
    static_configs:
      - targets: ['novacron-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgresql:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

**Grafana Datasources (`datasources.yaml`):**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: 30s
```

## SSL/TLS Configuration

### 1. Generate SSL Certificates

**Self-Signed Certificate (Development):**
```bash
# Generate private key
openssl genrsa -out novacron.key 2048

# Generate certificate
openssl req -new -x509 -key novacron.key -out novacron.crt -days 365 \
  -subj "/C=US/ST=CA/L=San Francisco/O=NovaCron/OU=IT/CN=novacron.com"

# Install certificates
sudo cp novacron.crt /etc/ssl/certs/
sudo cp novacron.key /etc/ssl/private/
sudo chmod 644 /etc/ssl/certs/novacron.crt
sudo chmod 600 /etc/ssl/private/novacron.key
```

**Let's Encrypt Certificate (Production):**
```bash
# Install certbot
sudo apt install -y certbot

# Generate certificate
sudo certbot certonly --standalone -d api.novacron.com -d app.novacron.com

# Setup auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Nginx Reverse Proxy

**Nginx Configuration (`/etc/nginx/sites-available/novacron`):**
```nginx
upstream novacron_api {
    server 127.0.0.1:8080;
    server 127.0.0.1:8080;
    server 127.0.0.1:8080;
}

upstream novacron_frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name api.novacron.com app.novacron.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.novacron.com;

    ssl_certificate /etc/ssl/certs/novacron.crt;
    ssl_certificate_key /etc/ssl/private/novacron.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS;
    ssl_prefer_server_ciphers off;

    client_max_body_size 100M;

    location / {
        proxy_pass http://novacron_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /ws {
        proxy_pass http://novacron_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 443 ssl http2;
    server_name app.novacron.com;

    ssl_certificate /etc/ssl/certs/novacron.crt;
    ssl_certificate_key /etc/ssl/private/novacron.key;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://novacron_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Recovery

### 1. Database Backup

**Automated Backup Script (`backup-db.sh`):**
```bash
#!/bin/bash

BACKUP_DIR="/opt/novacron/backups"
DB_NAME="novacron"
DB_USER="novacron"
DB_PASSWORD="${POSTGRES_PASSWORD}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/novacron_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Create database backup
PGPASSWORD=${DB_PASSWORD} pg_dump -h localhost -U ${DB_USER} ${DB_NAME} > ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Remove backups older than 30 days
find ${BACKUP_DIR} -name "novacron_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

**Backup Cron Job:**
```bash
# Add to crontab
0 2 * * * /opt/novacron/scripts/backup-db.sh
```

### 2. Application Data Backup

**VM Storage Backup:**
```bash
#!/bin/bash

VM_STORAGE="/var/lib/novacron/storage"
BACKUP_DIR="/opt/novacron/backups/vm-storage"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup using rsync
rsync -av --delete ${VM_STORAGE}/ ${BACKUP_DIR}/latest/

# Create timestamped snapshot
cp -al ${BACKUP_DIR}/latest ${BACKUP_DIR}/${TIMESTAMP}

echo "VM storage backup completed: ${BACKUP_DIR}/${TIMESTAMP}"
```

### 3. Disaster Recovery

**Recovery Procedure:**
```bash
#!/bin/bash
# Disaster Recovery Script

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
sudo systemctl stop novacron-api
sudo systemctl stop novacron-frontend

# Restore database
PGPASSWORD=${POSTGRES_PASSWORD} psql -h localhost -U novacron -d novacron < ${BACKUP_FILE}

# Restore VM storage
if [ -d "/opt/novacron/backups/vm-storage/latest" ]; then
    sudo rsync -av /opt/novacron/backups/vm-storage/latest/ /var/lib/novacron/storage/
fi

# Start services
sudo systemctl start novacron-api
sudo systemctl start novacron-frontend

# Verify recovery
curl -f http://localhost:8080/health

echo "Disaster recovery completed"
```

## Security Hardening

### 1. System Security

**Firewall Configuration:**
```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow NovaCron API (if direct access needed)
sudo ufw allow 8080/tcp
sudo ufw allow 8081/tcp

# Allow monitoring (restrict to monitoring subnet)
sudo ufw allow from 10.0.1.0/24 to any port 9090
```

**Fail2Ban Configuration (`/etc/fail2ban/jail.local`):**
```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[novacron-api]
enabled = true
port = 8080
logpath = /var/log/novacron/app.log
filter = novacron-api
maxretry = 10
bantime = 1800
```

### 2. Application Security

**Security Middleware Configuration:**
```go
// Security middleware configuration
func SecurityMiddleware() gin.HandlerFunc {
    config := secure.Config{
        SSLRedirect:           true,
        SSLHost:              "api.novacron.com",
        STSSeconds:           315360000,
        STSIncludeSubdomains: true,
        FrameDeny:            true,
        ContentTypeNosniff:   true,
        BrowserXssFilter:     true,
        ContentSecurityPolicy: "default-src 'self'",
    }
    return secure.New(config)
}
```

## Monitoring and Alerting

### 1. Health Checks

**Application Health Check:**
```bash
#!/bin/bash
# Health check script for monitoring

API_URL="http://localhost:8080/health"
FRONTEND_URL="http://localhost:3000"

# Check API health
if ! curl -f -s ${API_URL} > /dev/null; then
    echo "API health check failed"
    exit 1
fi

# Check frontend
if ! curl -f -s ${FRONTEND_URL} > /dev/null; then
    echo "Frontend health check failed"
    exit 1
fi

echo "All services healthy"
```

### 2. Log Management

**Logrotate Configuration (`/etc/logrotate.d/novacron`):**
```
/var/log/novacron/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
```

## Troubleshooting

### 1. Common Issues

**Database Connection Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test database connection
PGPASSWORD=password psql -h localhost -U novacron -d novacron -c "SELECT version();"

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

**KVM/Libvirt Issues:**
```bash
# Check libvirt status
sudo systemctl status libvirtd

# Test KVM functionality
lsmod | grep kvm
ls -la /dev/kvm

# Check virtualization support
sudo virt-host-validate
```

**Permission Issues:**
```bash
# Fix file permissions
sudo chown -R novacron:novacron /opt/novacron
sudo chmod -R 755 /opt/novacron
sudo chmod 600 /opt/novacron/config/*.yaml

# Add user to required groups
sudo usermod -aG libvirt novacron
sudo usermod -aG kvm novacron
```

### 2. Performance Tuning

**PostgreSQL Optimization:**
```sql
-- Optimize PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

**Go Application Optimization:**
```bash
# Set Go runtime environment variables
export GOGC=100
export GOMAXPROCS=4
export GOMEMLIMIT=1GiB
```

## Maintenance Procedures

### 1. Regular Maintenance

**Weekly Maintenance Script:**
```bash
#!/bin/bash
# Weekly maintenance tasks

# Update system packages (with approval)
sudo apt update

# Clean up old logs
sudo find /var/log -type f -name "*.log" -mtime +30 -delete

# Clean up old backups
find /opt/novacron/backups -type f -mtime +90 -delete

# Restart services (if needed)
sudo systemctl restart novacron-api
sudo systemctl restart novacron-frontend

# Verify services
curl -f http://localhost:8080/health
```

### 2. Update Procedures

**Application Update:**
```bash
#!/bin/bash
# Application update script

# Backup current version
sudo cp -r /opt/novacron /opt/novacron.backup.$(date +%Y%m%d)

# Stop services
sudo systemctl stop novacron-api
sudo systemctl stop novacron-frontend

# Update application
git pull origin main
go build -o bin/novacron-api cmd/api-server/main.go
npm install && npm run build

# Run database migrations
./scripts/migrate.sh

# Start services
sudo systemctl start novacron-api
sudo systemctl start novacron-frontend

# Verify update
curl -f http://localhost:8080/health
```

---

**Document Classification**: Operational - Internal Use  
**Last Updated**: September 2, 2025  
**Version**: 1.0  
**Next Review**: October 15, 2025