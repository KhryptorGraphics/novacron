# NovaCron Performance Optimization Guide

## Performance Overview

This guide provides comprehensive strategies for optimizing NovaCron performance across all system components, from application-level optimizations to infrastructure tuning and monitoring.

### Current Performance Baseline

**System Performance Metrics (as of September 2025):**
- **API Response Time**: P95 < 300ms (Target: <1000ms) ✅
- **System Uptime**: 99.95% (Target: >99.9%) ✅
- **Throughput**: 850 RPS (Target: >1000 RPS) ⚠️
- **Database Query Performance**: P95 < 50ms
- **VM Operation Latency**: P95 < 2s for lifecycle operations
- **Memory Usage**: 65% average utilization
- **CPU Usage**: 45% average utilization

## Application-Level Optimization

### 1. Go Backend Optimization

#### Memory Management
```go
// Optimize memory allocations
type VMPool struct {
    vms     sync.Pool
    metrics sync.Pool
}

func NewVMPool() *VMPool {
    return &VMPool{
        vms: sync.Pool{
            New: func() interface{} {
                return &VM{
                    Metrics: make([]Metric, 0, 100), // Pre-allocate capacity
                }
            },
        },
        metrics: sync.Pool{
            New: func() interface{} {
                return &Metrics{
                    Data: make(map[string]float64, 20),
                }
            },
        },
    }
}

// Use buffer pools for JSON marshaling
var jsonBufferPool = sync.Pool{
    New: func() interface{} {
        return bytes.NewBuffer(make([]byte, 0, 1024))
    },
}

func (h *Handler) WriteJSONResponse(w http.ResponseWriter, data interface{}) error {
    buf := jsonBufferPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer jsonBufferPool.Put(buf)
    
    if err := json.NewEncoder(buf).Encode(data); err != nil {
        return err
    }
    
    w.Header().Set("Content-Type", "application/json")
    _, err := w.Write(buf.Bytes())
    return err
}
```

#### Goroutine Pool Management
```go
type WorkerPool struct {
    workerCount int
    jobQueue    chan Job
    workers     []Worker
    quit        chan bool
}

func NewWorkerPool(workerCount, queueSize int) *WorkerPool {
    return &WorkerPool{
        workerCount: workerCount,
        jobQueue:    make(chan Job, queueSize),
        workers:     make([]Worker, workerCount),
        quit:        make(chan bool),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workerCount; i++ {
        wp.workers[i] = Worker{
            id:       i,
            jobQueue: wp.jobQueue,
            quit:     make(chan bool),
        }
        go wp.workers[i].start()
    }
}

// Optimized VM operations with worker pool
func (vm *VMManager) CreateVMAsync(config VMConfig) (*Job, error) {
    job := &Job{
        ID:     generateJobID(),
        Type:   JobTypeCreateVM,
        Config: config,
        Status: JobStatusPending,
    }
    
    select {
    case vm.workerPool.jobQueue <- job:
        return job, nil
    default:
        return nil, ErrWorkerPoolFull
    }
}
```

#### HTTP Server Optimization
```go
func NewOptimizedServer() *http.Server {
    return &http.Server{
        Addr:              ":8080",
        Handler:           setupRoutes(),
        ReadTimeout:       10 * time.Second,
        WriteTimeout:      10 * time.Second,
        IdleTimeout:       120 * time.Second,
        ReadHeaderTimeout: 5 * time.Second,
        MaxHeaderBytes:    1 << 20, // 1 MB
    }
}

// Connection pooling for HTTP client
var httpClient = &http.Client{
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        DisableCompression:  false,
        ForceAttemptHTTP2:   true,
    },
    Timeout: 30 * time.Second,
}

// Request/Response compression middleware
func CompressionMiddleware() gin.HandlerFunc {
    return gin.DefaultWriter.Header().Set("Content-Encoding", "gzip")
    return gzip.New(gzip.Config{
        Level: gzip.BestCompression,
    })
}
```

### 2. Database Performance Optimization

#### Query Optimization
```sql
-- Optimized VM listing with proper indexing
CREATE INDEX CONCURRENTLY idx_vms_tenant_state_created 
ON vms(tenant_id, state, created_at DESC) 
WHERE state != 'deleted';

CREATE INDEX CONCURRENTLY idx_vm_metrics_vm_time 
ON vm_metrics(vm_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_vm_metrics_timestamp_btree 
ON vm_metrics USING BTREE(timestamp) 
WHERE timestamp > (NOW() - INTERVAL '7 days');

-- Partitioning for metrics table
CREATE TABLE vm_metrics (
    id BIGSERIAL,
    vm_id VARCHAR(255) NOT NULL,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    network_sent BIGINT,
    network_recv BIGINT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE vm_metrics_2025_09 PARTITION OF vm_metrics
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE vm_metrics_2025_10 PARTITION OF vm_metrics
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
```

#### Connection Pool Tuning
```yaml
database:
  # Connection pool configuration
  max_connections: 50
  max_idle_connections: 25
  conn_max_lifetime: 300s
  conn_max_idle_time: 60s
  
  # Query performance
  statement_timeout: "30s"
  idle_in_transaction_session_timeout: "60s"
  
  # Memory settings
  shared_buffers: "256MB"
  effective_cache_size: "1GB"
  work_mem: "4MB"
  maintenance_work_mem: "64MB"
  
  # WAL and checkpoints
  wal_buffers: "16MB"
  checkpoint_completion_target: 0.9
  max_wal_size: "1GB"
  min_wal_size: "80MB"
```

#### Query Performance Monitoring
```go
type QueryMonitor struct {
    slowQueries map[string]*QueryStats
    mutex       sync.RWMutex
    threshold   time.Duration
}

func (qm *QueryMonitor) MonitorQuery(query string, duration time.Duration) {
    if duration < qm.threshold {
        return // Only track slow queries
    }
    
    qm.mutex.Lock()
    defer qm.mutex.Unlock()
    
    stats, exists := qm.slowQueries[query]
    if !exists {
        stats = &QueryStats{
            Query:      query,
            Count:      0,
            TotalTime:  0,
            MaxTime:    0,
            MinTime:    duration,
        }
        qm.slowQueries[query] = stats
    }
    
    stats.Count++
    stats.TotalTime += duration
    if duration > stats.MaxTime {
        stats.MaxTime = duration
    }
    if duration < stats.MinTime {
        stats.MinTime = duration
    }
}
```

### 3. Caching Strategy Implementation

#### Multi-Level Caching
```go
type CacheManager struct {
    l1Cache *sync.Map              // In-memory cache
    l2Cache *redis.Client          // Redis cache
    l3Cache *sql.DB               // Database cache
    ttl     map[string]time.Duration
}

func (cm *CacheManager) Get(key string) (interface{}, error) {
    // L1: In-memory cache (fastest)
    if value, exists := cm.l1Cache.Load(key); exists {
        if entry, ok := value.(*CacheEntry); ok && !entry.IsExpired() {
            cacheHits.WithLabelValues("l1").Inc()
            return entry.Value, nil
        }
        cm.l1Cache.Delete(key) // Remove expired entry
    }
    
    // L2: Redis cache
    if cm.l2Cache != nil {
        value, err := cm.l2Cache.Get(context.Background(), key).Result()
        if err == nil {
            // Store in L1 for faster access
            cm.l1Cache.Store(key, &CacheEntry{
                Value:     value,
                ExpiresAt: time.Now().Add(cm.ttl[key]),
            })
            cacheHits.WithLabelValues("l2").Inc()
            return value, nil
        }
    }
    
    // L3: Database/Source of truth
    cacheMisses.Inc()
    return nil, ErrCacheMiss
}

func (cm *CacheManager) Set(key string, value interface{}) error {
    ttl := cm.ttl[key]
    
    // Store in L1
    cm.l1Cache.Store(key, &CacheEntry{
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
    })
    
    // Store in L2
    if cm.l2Cache != nil {
        return cm.l2Cache.Set(context.Background(), key, value, ttl).Err()
    }
    
    return nil
}
```

#### Smart Cache Invalidation
```go
type CacheInvalidator struct {
    patterns map[string][]string // Resource type -> cache key patterns
    manager  *CacheManager
}

func (ci *CacheInvalidator) InvalidateResource(resourceType, resourceID string) error {
    patterns, exists := ci.patterns[resourceType]
    if !exists {
        return nil
    }
    
    for _, pattern := range patterns {
        key := fmt.Sprintf(pattern, resourceID)
        ci.manager.Delete(key)
        
        // Also invalidate related keys
        if resourceType == "vm" {
            ci.manager.Delete(fmt.Sprintf("vm:metrics:%s", resourceID))
            ci.manager.Delete(fmt.Sprintf("vm:status:%s", resourceID))
        }
    }
    
    return nil
}

// Cache warming for frequently accessed data
func (cm *CacheManager) WarmCache() error {
    // Pre-load frequently accessed VM data
    vms, err := cm.db.GetActiveVMs()
    if err != nil {
        return err
    }
    
    for _, vm := range vms {
        cm.Set(fmt.Sprintf("vm:%s", vm.ID), vm)
        
        // Pre-load metrics
        metrics, err := cm.db.GetVMMetrics(vm.ID, time.Hour)
        if err == nil {
            cm.Set(fmt.Sprintf("vm:metrics:%s", vm.ID), metrics)
        }
    }
    
    return nil
}
```

### 4. Frontend Performance Optimization

#### React Component Optimization
```typescript
// Memoization for expensive computations
const VMMetricsChart = React.memo(({ vmId }: { vmId: string }) => {
  const { data: metrics, isLoading } = useQuery(
    ['vm-metrics', vmId],
    () => fetchVMMetrics(vmId),
    {
      staleTime: 30 * 1000, // 30 seconds
      cacheTime: 5 * 60 * 1000, // 5 minutes
      refetchInterval: 30 * 1000, // Auto-refresh every 30s
    }
  );

  const chartData = useMemo(() => {
    if (!metrics) return null;
    
    return {
      labels: metrics.timestamps,
      datasets: [
        {
          label: 'CPU Usage',
          data: metrics.cpu_usage,
          borderColor: 'rgb(75, 192, 192)',
        },
        {
          label: 'Memory Usage',
          data: metrics.memory_usage,
          borderColor: 'rgb(255, 99, 132)',
        },
      ],
    };
  }, [metrics]);

  if (isLoading) return <ChartSkeleton />;
  
  return <Chart data={chartData} />;
});

// Virtual scrolling for large lists
const VMList = ({ vms }: { vms: VM[] }) => {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 50 });
  
  const visibleVMs = useMemo(() => 
    vms.slice(visibleRange.start, visibleRange.end),
    [vms, visibleRange]
  );

  const handleScroll = useCallback(
    throttle((scrollTop: number) => {
      const itemHeight = 100;
      const containerHeight = 800;
      const start = Math.floor(scrollTop / itemHeight);
      const end = start + Math.ceil(containerHeight / itemHeight) + 5;
      
      setVisibleRange({ start, end });
    }, 100),
    []
  );

  return (
    <VirtualizedList
      itemCount={vms.length}
      itemSize={100}
      onScroll={handleScroll}
    >
      {visibleVMs.map(vm => (
        <VMListItem key={vm.id} vm={vm} />
      ))}
    </VirtualizedList>
  );
};
```

#### Bundle Optimization
```typescript
// Code splitting by route
const Dashboard = lazy(() => import('./pages/Dashboard'));
const VMManagement = lazy(() => import('./pages/VMManagement'));
const Monitoring = lazy(() => import('./pages/Monitoring'));

// Preload critical components
const preloadComponents = () => {
  import('./pages/Dashboard');
  import('./components/VMList');
  import('./components/MetricsChart');
};

// Resource preloading
export const preloadCriticalResources = () => {
  // Preload critical API calls
  queryClient.prefetchQuery(['user-profile'], fetchUserProfile);
  queryClient.prefetchQuery(['vm-list'], fetchVMList);
  
  // Preload critical images
  const criticalImages = [
    '/images/vm-status-running.svg',
    '/images/vm-status-stopped.svg',
    '/images/loading-spinner.svg',
  ];
  
  criticalImages.forEach(src => {
    const img = new Image();
    img.src = src;
  });
};
```

#### Service Worker for Caching
```javascript
// sw.js - Service Worker for caching
const CACHE_NAME = 'novacron-v1.0.0';
const urlsToCache = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/images/logo.svg',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
  // Cache-first strategy for static assets
  if (event.request.url.includes('/static/')) {
    event.respondWith(
      caches.match(event.request)
        .then(response => {
          return response || fetch(event.request);
        })
    );
  }
  
  // Network-first strategy for API calls
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Cache successful API responses
          if (response.status === 200) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseClone);
              });
          }
          return response;
        })
        .catch(() => {
          // Fallback to cache on network failure
          return caches.match(event.request);
        })
    );
  }
});
```

## Infrastructure Optimization

### 1. Load Balancing Configuration

#### Nginx Load Balancer
```nginx
upstream novacron_api {
    # Weighted round-robin with health checks
    server 10.0.1.10:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 weight=2 max_fails=3 fail_timeout=30s;
    
    # Connection keepalive
    keepalive 32;
}

server {
    listen 80;
    server_name api.novacron.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req_zone $http_authorization zone=auth_limit:10m rate=1000r/m;
    
    location /api/v1/ {
        limit_req zone=api_limit burst=50 nodelay;
        limit_req zone=auth_limit burst=100 nodelay;
        
        proxy_pass http://novacron_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Connection pooling and timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # Compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1000;
        gzip_types
            application/json
            application/javascript
            text/css
            text/javascript
            text/plain
            text/xml;
    }
    
    # WebSocket proxy with optimizations
    location /ws/ {
        proxy_pass http://novacron_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

#### HAProxy Configuration
```
global
    maxconn 4096
    log stdout local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    retries 3
    option redispatch

frontend novacron_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/novacron.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 100 }
    
    default_backend novacron_api

backend novacron_api
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server api1 10.0.1.10:8080 check inter 5s rise 2 fall 3
    server api2 10.0.1.11:8080 check inter 5s rise 2 fall 3
    server api3 10.0.1.12:8080 check inter 5s rise 2 fall 3

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
```

### 2. Database Scaling

#### Read Replicas Configuration
```yaml
postgresql_cluster:
  primary:
    host: "pg-primary.internal"
    port: 5432
    max_connections: 100
    
  read_replicas:
    - host: "pg-read1.internal"
      port: 5432
      weight: 1
      max_connections: 100
      
    - host: "pg-read2.internal"
      port: 5432
      weight: 1
      max_connections: 100
      
  connection_routing:
    write_operations: "primary"
    read_operations: "read_replicas"
    fallback: "primary"
```

#### Database Connection Pooling
```go
type DatabaseManager struct {
    primary   *sql.DB
    replicas  []*sql.DB
    semaphore chan struct{}
}

func NewDatabaseManager(config DatabaseConfig) *DatabaseManager {
    dm := &DatabaseManager{
        semaphore: make(chan struct{}, config.MaxConnections),
    }
    
    // Primary database connection
    dm.primary = sql.Open("postgres", config.PrimaryDSN)
    dm.primary.SetMaxOpenConns(config.MaxConnections)
    dm.primary.SetMaxIdleConns(config.MaxIdleConnections)
    dm.primary.SetConnMaxLifetime(config.ConnMaxLifetime)
    
    // Read replica connections
    for _, replicaDSN := range config.ReplicaDSNs {
        replica := sql.Open("postgres", replicaDSN)
        replica.SetMaxOpenConns(config.MaxConnections / 2)
        replica.SetMaxIdleConns(config.MaxIdleConnections / 2)
        dm.replicas = append(dm.replicas, replica)
    }
    
    return dm
}

func (dm *DatabaseManager) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    // Use read replica for SELECT queries
    if strings.HasPrefix(strings.ToUpper(strings.TrimSpace(query)), "SELECT") {
        replica := dm.selectReplica()
        return replica.QueryContext(ctx, query, args...)
    }
    
    // Use primary for write operations
    return dm.primary.QueryContext(ctx, query, args...)
}

func (dm *DatabaseManager) selectReplica() *sql.DB {
    if len(dm.replicas) == 0 {
        return dm.primary
    }
    
    // Round-robin selection with health checking
    for i := 0; i < len(dm.replicas); i++ {
        replica := dm.replicas[rand.Intn(len(dm.replicas))]
        if dm.isHealthy(replica) {
            return replica
        }
    }
    
    return dm.primary // Fallback to primary
}
```

### 3. Kubernetes Optimization

#### Resource Management
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: novacron-api
        image: novacron/api:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: GOMAXPROCS
          valueFrom:
            resourceFieldRef:
              resource: limits.cpu
        - name: GOMEMLIMIT
          valueFrom:
            resourceFieldRef:
              resource: limits.memory
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      nodeSelector:
        node-type: compute-optimized
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - novacron-api
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: novacron-api-service
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  selector:
    app: novacron-api
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: novacron-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: novacron-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Monitoring & Profiling

### 1. Application Performance Monitoring

#### Go Profiling Integration
```go
import (
    _ "net/http/pprof"
    "github.com/pkg/profile"
)

func main() {
    // Enable profiling in development
    if os.Getenv("ENABLE_PROFILING") == "true" {
        defer profile.Start(
            profile.CPUProfile,
            profile.ProfilePath("/tmp/profiles"),
        ).Stop()
    }
    
    // Start pprof endpoint
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // Application startup
    startApplication()
}

// Custom metrics for performance monitoring
var (
    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "endpoint", "status_code"},
    )
    
    activeConnections = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "http_active_connections",
            Help: "Number of active HTTP connections",
        },
    )
    
    databaseQueryDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "database_query_duration_seconds",
            Help:    "Database query duration in seconds",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
        },
        []string{"query_type", "table"},
    )
)
```

#### Performance Middleware
```go
func PerformanceMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        
        // Track active connections
        activeConnections.Inc()
        defer activeConnections.Dec()
        
        // Process request
        c.Next()
        
        // Record metrics
        duration := time.Since(start)
        status := strconv.Itoa(c.Writer.Status())
        
        requestDuration.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
            status,
        ).Observe(duration.Seconds())
        
        // Log slow requests
        if duration > 1*time.Second {
            logger.Warn("Slow request detected",
                "method", c.Request.Method,
                "path", c.Request.URL.Path,
                "duration", duration,
                "status", status,
            )
        }
    }
}
```

### 2. Database Performance Monitoring

#### Query Performance Tracking
```sql
-- Enable query statistics
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
SELECT pg_reload_conf();

-- Create extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Monitor slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
WHERE mean_time > 100 -- queries slower than 100ms
ORDER BY total_time DESC
LIMIT 20;

-- Index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_tup_read::float / NULLIF(idx_tup_fetch, 0) AS ratio
FROM pg_stat_user_indexes
WHERE idx_tup_read > 0
ORDER BY ratio DESC;
```

#### Connection Monitoring
```go
type DatabaseMonitor struct {
    db     *sql.DB
    ticker *time.Ticker
    quit   chan struct{}
}

func (dm *DatabaseMonitor) Start() {
    dm.ticker = time.NewTicker(30 * time.Second)
    go dm.monitorLoop()
}

func (dm *DatabaseMonitor) monitorLoop() {
    for {
        select {
        case <-dm.ticker.C:
            dm.collectMetrics()
        case <-dm.quit:
            dm.ticker.Stop()
            return
        }
    }
}

func (dm *DatabaseMonitor) collectMetrics() {
    stats := dm.db.Stats()
    
    // Connection pool metrics
    connectionPoolSize.Set(float64(stats.MaxOpenConnections))
    connectionPoolInUse.Set(float64(stats.InUse))
    connectionPoolIdle.Set(float64(stats.Idle))
    connectionPoolWaitCount.Set(float64(stats.WaitCount))
    connectionPoolWaitDuration.Set(stats.WaitDuration.Seconds())
    
    // Query active connections
    var activeQueries int
    err := dm.db.QueryRow("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'").Scan(&activeQueries)
    if err == nil {
        databaseActiveQueries.Set(float64(activeQueries))
    }
    
    // Check for long-running queries
    rows, err := dm.db.Query(`
        SELECT pid, query, state, query_start 
        FROM pg_stat_activity 
        WHERE state = 'active' 
        AND query_start < NOW() - INTERVAL '30 seconds'
        AND query NOT LIKE '%pg_stat_activity%'
    `)
    if err == nil {
        defer rows.Close()
        
        for rows.Next() {
            var pid int
            var query, state string
            var queryStart time.Time
            
            if err := rows.Scan(&pid, &query, &state, &queryStart); err == nil {
                logger.Warn("Long-running query detected",
                    "pid", pid,
                    "duration", time.Since(queryStart),
                    "query", truncateString(query, 100),
                )
            }
        }
    }
}
```

### 3. Real-time Performance Dashboards

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "NovaCron Performance Dashboard",
    "panels": [
      {
        "title": "API Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "Response Time (seconds)",
            "max": 5,
            "min": 0
          }
        ]
      },
      {
        "title": "Database Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(database_query_duration_seconds_sum[5m]) / rate(database_query_duration_seconds_count[5m])",
            "legendFormat": "Average Query Time",
            "refId": "A"
          },
          {
            "expr": "database_active_connections",
            "legendFormat": "Active Connections",
            "refId": "B"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %",
            "refId": "A"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %",
            "refId": "B"
          }
        ]
      }
    ]
  }
}
```

## Performance Testing & Benchmarking

### 1. Load Testing Strategy

#### K6 Load Testing Script
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Maintain 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200 users
    { duration: '5m', target: 200 },   // Maintain 200 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests under 1s
    http_req_failed: ['rate<0.01'],    // Error rate under 1%
  },
};

const BASE_URL = 'https://api.novacron.com';
const AUTH_TOKEN = __ENV.AUTH_TOKEN;

export default function() {
  // Authentication
  const authHeaders = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };
  
  // Test VM listing
  let response = http.get(`${BASE_URL}/api/v1/vms`, {
    headers: authHeaders,
  });
  
  check(response, {
    'VM list status is 200': (r) => r.status === 200,
    'VM list response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  errorRate.add(response.status !== 200);
  
  // Test VM creation (reduced frequency)
  if (Math.random() < 0.1) { // 10% of requests
    const vmConfig = {
      name: `test-vm-${Date.now()}`,
      config: {
        cpu: 2,
        memory: 4096,
        disk: 50,
        image: 'ubuntu-22.04',
      },
    };
    
    response = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify(vmConfig), {
      headers: authHeaders,
    });
    
    check(response, {
      'VM creation status is 201': (r) => r.status === 201,
      'VM creation response time < 2s': (r) => r.timings.duration < 2000,
    });
  }
  
  sleep(Math.random() * 3 + 1); // 1-4 seconds think time
}
```

#### Go Benchmark Tests
```go
func BenchmarkVMList(b *testing.B) {
    server := setupTestServer()
    defer server.Close()
    
    client := &http.Client{Timeout: 10 * time.Second}
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            resp, err := client.Get(server.URL + "/api/v1/vms")
            if err != nil {
                b.Error(err)
            }
            resp.Body.Close()
            
            if resp.StatusCode != http.StatusOK {
                b.Errorf("Expected status 200, got %d", resp.StatusCode)
            }
        }
    })
}

func BenchmarkDatabaseQuery(b *testing.B) {
    db := setupTestDatabase()
    defer db.Close()
    
    query := "SELECT id, name, state FROM vms WHERE tenant_id = $1 LIMIT 50"
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        rows, err := db.Query(query, "test-tenant")
        if err != nil {
            b.Error(err)
        }
        
        for rows.Next() {
            var id, name, state string
            rows.Scan(&id, &name, &state)
        }
        rows.Close()
    }
}

func BenchmarkJSONMarshal(b *testing.B) {
    vm := &VM{
        ID:       "vm-123",
        Name:     "test-vm",
        State:    "running",
        TenantID: "tenant-123",
        Config: VMConfig{
            CPU:    2,
            Memory: 4096,
            Disk:   50,
        },
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := json.Marshal(vm)
        if err != nil {
            b.Error(err)
        }
    }
}
```

### 2. Performance Regression Testing

#### Automated Performance CI/CD Pipeline
```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *' # Daily at 2 AM

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: novacron_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.23'
    
    - name: Run benchmark tests
      run: |
        go test -bench=. -benchmem -run=^$ ./... | tee benchmark.txt
    
    - name: Load test with k6
      uses: grafana/k6-action@v0.3.1
      with:
        filename: tests/performance/load-test.js
        flags: --out json=results.json
    
    - name: Performance regression check
      run: |
        go run scripts/performance-check.go \
          --baseline=performance-baseline.json \
          --current=results.json \
          --threshold=10
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: |
          benchmark.txt
          results.json
```

## Performance Optimization Roadmap

### 1. Short-term Optimizations (Next 30 days)
- [ ] Implement connection pooling improvements
- [ ] Add Redis caching for frequently accessed data
- [ ] Optimize database queries with proper indexing
- [ ] Enable HTTP/2 and compression
- [ ] Implement query result caching

### 2. Medium-term Improvements (3-6 months)
- [ ] Database read replica implementation
- [ ] CDN integration for static assets
- [ ] Advanced caching strategies (cache warming, invalidation)
- [ ] API response pagination and filtering
- [ ] Kubernetes horizontal pod autoscaling

### 3. Long-term Enhancements (6-12 months)
- [ ] Database sharding for multi-tenant scalability
- [ ] Edge computing deployment for global performance
- [ ] Advanced load balancing with geographic routing
- [ ] Real-time stream processing optimization
- [ ] AI-powered performance prediction and auto-scaling

## Performance Monitoring Checklist

### Daily Monitoring
- [ ] Review application performance metrics
- [ ] Check database slow query logs
- [ ] Monitor system resource utilization
- [ ] Validate cache hit rates
- [ ] Review error rates and response times

### Weekly Analysis
- [ ] Performance trend analysis
- [ ] Capacity planning review
- [ ] Database maintenance tasks
- [ ] Cache optimization review
- [ ] Load testing results analysis

### Monthly Optimization
- [ ] Performance baseline updates
- [ ] Infrastructure scaling decisions
- [ ] Code optimization prioritization
- [ ] Performance testing strategy review
- [ ] Architecture improvement planning

---

**Document Classification**: Technical - Performance Team  
**Last Updated**: September 2, 2025  
**Version**: 1.0  
**Performance Review**: Required Monthly  
**Benchmark Update**: Required Quarterly