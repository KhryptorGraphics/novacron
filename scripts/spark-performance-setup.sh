#!/bin/bash

# Spark Dating App Performance Optimization Setup Script
# Implements comprehensive performance optimizations for sub-second response times

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config/performance"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.spark-performance.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check available memory (require at least 8GB for Redis cluster)
    available_memory=$(free -g | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 8 ]; then
        log_warning "Less than 8GB available memory. Performance may be impacted."
    fi
    
    log_success "Prerequisites check completed"
}

# Create performance-optimized Docker Compose file
create_docker_compose() {
    log_info "Creating Docker Compose configuration for performance optimization..."
    
    cat > "$DOCKER_COMPOSE_FILE" << 'EOF'
version: '3.8'

services:
  # Redis Cluster for L2 Cache
  redis-1:
    image: redis:7.2-alpine
    container_name: spark-redis-1
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7001 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7001:7001"
      - "17001:17001"
    volumes:
      - redis1-data:/data
    networks:
      - spark-performance
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  redis-2:
    image: redis:7.2-alpine
    container_name: spark-redis-2
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7002 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7002:7002"
      - "17002:17002"
    volumes:
      - redis2-data:/data
    networks:
      - spark-performance

  redis-3:
    image: redis:7.2-alpine
    container_name: spark-redis-3
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7003 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7003:7003"
      - "17003:17003"
    volumes:
      - redis3-data:/data
    networks:
      - spark-performance

  redis-4:
    image: redis:7.2-alpine
    container_name: spark-redis-4
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7004 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7004:7004"
      - "17004:17004"
    volumes:
      - redis4-data:/data
    networks:
      - spark-performance

  redis-5:
    image: redis:7.2-alpine
    container_name: spark-redis-5
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7005 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7005:7005"
      - "17005:17005"
    volumes:
      - redis5-data:/data
    networks:
      - spark-performance

  redis-6:
    image: redis:7.2-alpine
    container_name: spark-redis-6
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 7006 --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "7006:7006"
      - "17006:17006"
    volumes:
      - redis6-data:/data
    networks:
      - spark-performance

  # Redis Cluster Setup
  redis-cluster-setup:
    image: redis:7.2-alpine
    container_name: spark-redis-cluster-setup
    command: >
      sh -c "sleep 10 && 
             redis-cli --cluster create 
             redis-1:7001 redis-2:7002 redis-3:7003 
             redis-4:7004 redis-5:7005 redis-6:7006 
             --cluster-replicas 1 --cluster-yes"
    depends_on:
      - redis-1
      - redis-2
      - redis-3
      - redis-4
      - redis-5
      - redis-6
    networks:
      - spark-performance

  # PostgreSQL Primary Database
  postgres-primary:
    image: postgres:15-alpine
    container_name: spark-postgres-primary
    environment:
      POSTGRES_DB: spark_dating
      POSTGRES_USER: spark_user
      POSTGRES_PASSWORD: spark_password
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-primary-data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./config/postgres/init-replication.sql:/docker-entrypoint-initdb.d/init-replication.sql
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c shared_preload_libraries=pg_stat_statements
      -c log_statement=ddl
      -c log_min_duration_statement=100ms
    networks:
      - spark-performance
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  # PostgreSQL Read Replica 1
  postgres-read-1:
    image: postgres:15-alpine
    container_name: spark-postgres-read-1
    environment:
      POSTGRES_DB: spark_dating
      POSTGRES_USER: spark_user
      POSTGRES_PASSWORD: spark_password
      PGUSER: postgres
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    ports:
      - "5433:5432"
    volumes:
      - postgres-read1-data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary
    networks:
      - spark-performance

  # PostgreSQL Read Replica 2
  postgres-read-2:
    image: postgres:15-alpine
    container_name: spark-postgres-read-2
    environment:
      POSTGRES_DB: spark_dating
      POSTGRES_USER: spark_user
      POSTGRES_PASSWORD: spark_password
      PGUSER: postgres
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    ports:
      - "5434:5432"
    volumes:
      - postgres-read2-data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary
    networks:
      - spark-performance

  # Nginx Load Balancer
  nginx-lb:
    image: nginx:alpine
    container_name: spark-nginx-lb
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/nginx/spark-performance.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - spark-app-1
      - spark-app-2
    networks:
      - spark-performance

  # Spark Application Instances
  spark-app-1:
    build:
      context: .
      dockerfile: Dockerfile.spark
    container_name: spark-app-1
    environment:
      - NODE_ENV=production
      - REDIS_CLUSTER_NODES=redis-1:7001,redis-2:7002,redis-3:7003,redis-4:7004,redis-5:7005,redis-6:7006
      - DATABASE_URL=postgresql://spark_user:spark_password@postgres-primary:5432/spark_dating
      - DATABASE_READ_URL=postgresql://spark_user:spark_password@postgres-read-1:5432/spark_dating
      - PORT=3000
    ports:
      - "3001:3000"
    volumes:
      - ./config/performance:/app/config/performance
    depends_on:
      - postgres-primary
      - redis-cluster-setup
    networks:
      - spark-performance
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  spark-app-2:
    build:
      context: .
      dockerfile: Dockerfile.spark
    container_name: spark-app-2
    environment:
      - NODE_ENV=production
      - REDIS_CLUSTER_NODES=redis-1:7001,redis-2:7002,redis-3:7003,redis-4:7004,redis-5:7005,redis-6:7006
      - DATABASE_URL=postgresql://spark_user:spark_password@postgres-primary:5432/spark_dating
      - DATABASE_READ_URL=postgresql://spark_user:spark_password@postgres-read-2:5432/spark_dating
      - PORT=3000
    ports:
      - "3002:3000"
    volumes:
      - ./config/performance:/app/config/performance
    depends_on:
      - postgres-primary
      - redis-cluster-setup
    networks:
      - spark-performance

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: spark-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./config/prometheus/spark-rules.yml:/etc/prometheus/spark-rules.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - spark-performance

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: spark-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=spark_admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - spark-performance

networks:
  spark-performance:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  redis1-data:
  redis2-data:
  redis3-data:
  redis4-data:
  redis5-data:
  redis6-data:
  postgres-primary-data:
  postgres-read1-data:
  postgres-read2-data:
  prometheus-data:
  grafana-data:
EOF

    log_success "Docker Compose configuration created"
}

# Create optimized PostgreSQL configuration
create_postgres_config() {
    log_info "Creating optimized PostgreSQL configuration..."
    
    mkdir -p "$PROJECT_ROOT/config/postgres"
    
    cat > "$PROJECT_ROOT/config/postgres/postgresql.conf" << 'EOF'
# PostgreSQL Configuration optimized for Spark Dating App
# Performance-focused settings for high-concurrency workloads

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = 512MB
work_mem = 8MB
maintenance_work_mem = 128MB
effective_cache_size = 2GB
effective_io_concurrency = 200

# Checkpoint Settings  
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_timeout = 15min
max_wal_size = 2GB
min_wal_size = 512MB

# Query Planner Settings
random_page_cost = 1.1
cpu_tuple_cost = 0.01
cpu_index_tuple_cost = 0.005
cpu_operator_cost = 0.0025

# Logging Settings
log_destination = 'stderr'
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_min_duration_statement = 100ms
log_checkpoints = on
log_connections = off
log_disconnections = off
log_lock_waits = on
log_temp_files = 0

# Autovacuum Settings (important for dating app with frequent updates)
autovacuum = on
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_delay = 10ms
autovacuum_vacuum_cost_limit = 1000

# Stats Collection
track_activities = on
track_counts = on
track_io_timing = on
track_functions = pl
shared_preload_libraries = 'pg_stat_statements'

# Spark Dating App Specific Optimizations
# Enable extensions for location-based queries
shared_preload_libraries = 'pg_stat_statements,postgis'

# Timezone
timezone = 'UTC'
log_timezone = 'UTC'

# Locale
lc_messages = 'en_US.utf8'
lc_monetary = 'en_US.utf8'
lc_numeric = 'en_US.utf8'
lc_time = 'en_US.utf8'
default_text_search_config = 'pg_catalog.english'
EOF

    # Create replication initialization script
    cat > "$PROJECT_ROOT/config/postgres/init-replication.sql" << 'EOF'
-- Initialize replication for Spark Dating App
-- Creates replication user and optimized indexes

-- Create replication user
CREATE USER replicator REPLICATION LOGIN ENCRYPTED PASSWORD 'repl_password';

-- Create performance monitoring extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Spark Dating App Schema
CREATE SCHEMA IF NOT EXISTS spark;

-- Users table with location support
CREATE TABLE IF NOT EXISTS spark.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    age INTEGER CHECK (age >= 18 AND age <= 100),
    bio TEXT,
    location GEOMETRY(POINT, 4326),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimized indexes for user matching
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_location_age_active 
ON spark.users USING GIST(location, age, is_active, updated_at) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_age_active 
ON spark.users(age, is_active, last_active_at) 
WHERE is_active = true;

-- Swipes table for match tracking
CREATE TABLE IF NOT EXISTS spark.swipes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES spark.users(id),
    target_id UUID NOT NULL REFERENCES spark.users(id),
    direction VARCHAR(10) CHECK (direction IN ('left', 'right', 'super')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, target_id)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_swipes_user_target 
ON spark.swipes(user_id, target_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_swipes_target_direction 
ON spark.swipes(target_id, direction, created_at DESC) 
WHERE direction = 'right';

-- Matches table
CREATE TABLE IF NOT EXISTS spark.matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user1_id UUID NOT NULL REFERENCES spark.users(id),
    user2_id UUID NOT NULL REFERENCES spark.users(id),
    matched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    CHECK (user1_id < user2_id),
    UNIQUE(user1_id, user2_id)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_matches_users_timestamp 
ON spark.matches(user1_id, user2_id, matched_at DESC) 
WHERE is_active = true;

-- Messages table
CREATE TABLE IF NOT EXISTS spark.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    sender_id UUID NOT NULL REFERENCES spark.users(id),
    recipient_id UUID NOT NULL REFERENCES spark.users(id),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    read_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_timestamp
ON spark.messages(conversation_id, created_at DESC)
WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_recipient_unread
ON spark.messages(recipient_id, conversation_id, created_at DESC)
WHERE read_at IS NULL AND deleted_at IS NULL;

-- Performance monitoring views
CREATE VIEW spark.performance_stats AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'spark';

-- User activity statistics
CREATE MATERIALIZED VIEW spark.user_activity_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as new_users,
    COUNT(*) FILTER (WHERE last_active_at > NOW() - INTERVAL '24 hours') as active_users,
    COUNT(*) FILTER (WHERE last_active_at > NOW() - INTERVAL '7 days') as weekly_active_users
FROM spark.users
GROUP BY DATE(created_at);

CREATE UNIQUE INDEX ON spark.user_activity_stats(date);

-- Function to refresh activity stats
CREATE OR REPLACE FUNCTION spark.refresh_activity_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY spark.user_activity_stats;
END;
$$ LANGUAGE plpgsql;
EOF

    log_success "PostgreSQL configuration created"
}

# Create Nginx configuration for load balancing
create_nginx_config() {
    log_info "Creating Nginx load balancer configuration..."
    
    mkdir -p "$PROJECT_ROOT/config/nginx"
    
    cat > "$PROJECT_ROOT/config/nginx/nginx.conf" << 'EOF'
# Nginx Configuration optimized for Spark Dating App
# High-performance reverse proxy and load balancer

user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

# Optimize for high concurrency
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;
    
    # Include server configurations
    include /etc/nginx/conf.d/*.conf;
}
EOF

    cat > "$PROJECT_ROOT/config/nginx/spark-performance.conf" << 'EOF'
# Spark Dating App Performance-Optimized Server Configuration

# Upstream application servers
upstream spark_app {
    least_conn;
    server spark-app-1:3000 max_fails=3 fail_timeout=30s;
    server spark-app-2:3000 max_fails=3 fail_timeout=30s;
    
    # Health checks (if nginx-plus)
    # health_check interval=30s fails=3 passes=2;
}

# WebSocket upstream for real-time features
upstream spark_websocket {
    ip_hash; # Sticky sessions for WebSocket
    server spark-app-1:3000;
    server spark-app-2:3000;
}

server {
    listen 80;
    server_name spark-dating.local;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Client body size for image uploads
    client_max_body_size 10M;
    
    # Timeouts
    proxy_connect_timeout 30s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;

    # API endpoints with rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://spark_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Custom headers for performance monitoring
        proxy_set_header X-Request-Start $msec;
        proxy_set_header X-Response-Time $request_time;
    }
    
    # Authentication endpoints with stricter rate limiting
    location /api/auth/ {
        limit_req zone=auth burst=5 nodelay;
        
        proxy_pass http://spark_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket connections for real-time chat
    location /socket.io/ {
        proxy_pass http://spark_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific timeouts
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # Static file caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
        
        proxy_pass http://spark_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://spark_app;
        proxy_set_header Host $host;
    }

    # Default location
    location / {
        proxy_pass http://spark_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

    log_success "Nginx configuration created"
}

# Create Prometheus monitoring configuration
create_monitoring_config() {
    log_info "Creating monitoring configuration..."
    
    mkdir -p "$PROJECT_ROOT/config/prometheus"
    mkdir -p "$PROJECT_ROOT/config/grafana/dashboards"
    mkdir -p "$PROJECT_ROOT/config/grafana/datasources"
    
    # Prometheus configuration
    cat > "$PROJECT_ROOT/config/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "spark-rules.yml"

scrape_configs:
  - job_name: 'spark-app'
    static_configs:
      - targets: ['spark-app-1:3000', 'spark-app-2:3000']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-primary:5432']
    scrape_interval: 30s

  - job_name: 'redis-cluster'
    static_configs:
      - targets: ['redis-1:7001', 'redis-2:7002', 'redis-3:7003', 
                  'redis-4:7004', 'redis-5:7005', 'redis-6:7006']
    scrape_interval: 15s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-lb:80']
    scrape_interval: 15s
EOF

    # Prometheus alerting rules
    cat > "$PROJECT_ROOT/config/prometheus/spark-rules.yml" << 'EOF'
groups:
  - name: spark_performance
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="spark-app"}[5m])) > 0.2
        for: 2m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High API response time"
          description: "P95 API response time is {{ $value }}s"

      - alert: LowCacheHitRate
        expr: rate(cache_requests_total{result="hit"}[5m]) / rate(cache_requests_total[5m]) < 0.85
        for: 5m
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      - alert: DatabaseSlowQueries
        expr: histogram_quantile(0.95, rate(postgresql_query_duration_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "Slow database queries"
          description: "P95 query time is {{ $value }}s"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.005
        for: 1m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: RedisClusterDown
        expr: up{job="redis-cluster"} == 0
        for: 30s
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "Redis cluster node down"
          description: "Redis cluster node {{ $labels.instance }} is down"
EOF

    # Grafana datasource configuration
    cat > "$PROJECT_ROOT/config/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log_success "Monitoring configuration created"
}

# Initialize performance optimization
init_performance() {
    log_info "Initializing performance optimization setup..."
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/docs/performance"
    mkdir -p "$PROJECT_ROOT/config/performance"
    mkdir -p "$PROJECT_ROOT/scripts"
    mkdir -p "$PROJECT_ROOT/config/postgres"
    mkdir -p "$PROJECT_ROOT/config/nginx"
    mkdir -p "$PROJECT_ROOT/config/prometheus"
    
    # Set appropriate permissions
    chmod +x "$0"
    
    log_success "Performance optimization initialized"
}

# Deploy performance infrastructure
deploy() {
    log_info "Deploying performance-optimized infrastructure..."
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Start infrastructure
    log_info "Starting performance infrastructure..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Verify Redis cluster
    log_info "Verifying Redis cluster setup..."
    docker exec spark-redis-1 redis-cli --cluster info || log_warning "Redis cluster may need manual setup"
    
    # Check database connectivity
    log_info "Verifying database connectivity..."
    docker exec spark-postgres-primary pg_isready -U spark_user || log_warning "Database may not be ready yet"
    
    log_success "Performance infrastructure deployed successfully"
    log_info "Access points:"
    log_info "  - Application: http://localhost"
    log_info "  - Grafana: http://localhost:3000 (admin/spark_admin)"
    log_info "  - Prometheus: http://localhost:9090"
}

# Test performance setup
test_performance() {
    log_info "Testing performance setup..."
    
    # Test Redis cluster
    log_info "Testing Redis cluster..."
    docker exec spark-redis-1 redis-cli --cluster check redis-1:7001 || log_error "Redis cluster test failed"
    
    # Test database
    log_info "Testing database connection..."
    docker exec spark-postgres-primary psql -U spark_user -d spark_dating -c "SELECT version();" || log_error "Database test failed"
    
    # Test application endpoints
    log_info "Testing application endpoints..."
    curl -f http://localhost/health || log_error "Application health check failed"
    
    log_success "Performance setup tests completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up performance infrastructure..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    log_success "Cleanup completed"
}

# Show performance metrics
show_metrics() {
    log_info "Performance Metrics:"
    echo "=================================="
    echo "Redis Cluster Status:"
    docker exec spark-redis-1 redis-cli --cluster info 2>/dev/null || echo "Redis cluster not ready"
    echo ""
    echo "Database Status:"
    docker exec spark-postgres-primary pg_isready -U spark_user 2>/dev/null || echo "Database not ready"
    echo ""
    echo "Application Status:"
    curl -s http://localhost/health 2>/dev/null || echo "Application not responding"
    echo ""
    echo "Monitoring URLs:"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
}

# Main function
main() {
    case "${1:-}" in
        "init")
            check_prerequisites
            init_performance
            create_docker_compose
            create_postgres_config
            create_nginx_config
            create_monitoring_config
            log_success "Performance optimization setup completed"
            log_info "Next steps:"
            log_info "  1. Run './spark-performance-setup.sh deploy' to start infrastructure"
            log_info "  2. Run './spark-performance-setup.sh test' to verify setup"
            ;;
        "deploy")
            deploy
            ;;
        "test")
            test_performance
            ;;
        "metrics")
            show_metrics
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Spark Dating App Performance Setup"
            echo ""
            echo "Usage: $0 {init|deploy|test|metrics|cleanup}"
            echo ""
            echo "Commands:"
            echo "  init     - Initialize performance optimization configuration"
            echo "  deploy   - Deploy performance-optimized infrastructure"  
            echo "  test     - Test the performance setup"
            echo "  metrics  - Show current performance metrics"
            echo "  cleanup  - Clean up all performance infrastructure"
            echo ""
            echo "Example workflow:"
            echo "  $0 init    # Set up configuration files"
            echo "  $0 deploy  # Deploy infrastructure" 
            echo "  $0 test    # Verify everything works"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"