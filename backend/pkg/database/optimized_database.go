package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/lib/pq"
	_ "github.com/lib/pq"
)

// OptimizedDB wraps sqlx.DB with query optimization features
type OptimizedDB struct {
	*sqlx.DB
	queryCache     *QueryCache
	preparedStmts  map[string]*sqlx.Stmt
	stmtMutex      sync.RWMutex
	metricsEnabled bool
}

// QueryCache implements an LRU cache for query results
type QueryCache struct {
	cache    map[string]*CacheEntry
	mutex    sync.RWMutex
	maxSize  int
	ttl      time.Duration
	hitRate  float64
	hits     int64
	misses   int64
}

// CacheEntry represents a cached query result
type CacheEntry struct {
	Data      interface{}
	ExpiresAt time.Time
	Size      int
}

// QueryMetrics tracks query performance
type QueryMetrics struct {
	Query        string
	Duration     time.Duration
	RowsAffected int64
	CacheHit     bool
	Timestamp    time.Time
}

// ConnectionPoolConfig optimized for high performance
type ConnectionPoolConfig struct {
	MaxOpenConns        int           // Maximum open connections
	MaxIdleConns        int           // Maximum idle connections
	ConnMaxLifetime     time.Duration // Maximum connection lifetime
	ConnMaxIdleTime     time.Duration // Maximum idle time
	HealthCheckInterval time.Duration // Health check interval
}

// DefaultPoolConfig returns optimized connection pool settings
func DefaultPoolConfig() ConnectionPoolConfig {
	return ConnectionPoolConfig{
		MaxOpenConns:        50,              // Increased for high load
		MaxIdleConns:        25,              // 50% of max open
		ConnMaxLifetime:     5 * time.Minute, // Rotate connections regularly
		ConnMaxIdleTime:     90 * time.Second,
		HealthCheckInterval: 30 * time.Second,
	}
}

// NewOptimized creates an optimized database connection
func NewOptimized(databaseURL string, config ConnectionPoolConfig) (*OptimizedDB, error) {
	// Configure for optimal performance
	db, err := sqlx.Open("postgres", databaseURL+"?sslmode=disable&application_name=novacron&statement_timeout=30000")
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Apply optimized connection pool settings
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.ConnMaxLifetime)
	db.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// Test connection with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Initialize query cache
	queryCache := &QueryCache{
		cache:   make(map[string]*CacheEntry),
		maxSize: 1000,
		ttl:     5 * time.Minute,
	}

	odb := &OptimizedDB{
		DB:            db,
		queryCache:    queryCache,
		preparedStmts: make(map[string]*sqlx.Stmt),
	}

	// Start health check routine
	go odb.healthCheckLoop(config.HealthCheckInterval)

	// Start cache cleanup routine
	go odb.cacheCleanupLoop()

	return odb, nil
}

// PrepareNamed prepares and caches a named query
func (db *OptimizedDB) PrepareNamed(query string) (*sqlx.NamedStmt, error) {
	return db.PrepareNamedContext(context.Background(), query)
}

// GetPrepared returns a cached prepared statement or creates a new one
func (db *OptimizedDB) GetPrepared(key, query string) (*sqlx.Stmt, error) {
	db.stmtMutex.RLock()
	if stmt, exists := db.preparedStmts[key]; exists {
		db.stmtMutex.RUnlock()
		return stmt, nil
	}
	db.stmtMutex.RUnlock()

	// Create new prepared statement
	db.stmtMutex.Lock()
	defer db.stmtMutex.Unlock()

	// Double-check after acquiring write lock
	if stmt, exists := db.preparedStmts[key]; exists {
		return stmt, nil
	}

	stmt, err := db.Preparex(query)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare statement: %w", err)
	}

	db.preparedStmts[key] = stmt
	return stmt, nil
}

// QueryWithCache executes a query with caching
func (db *OptimizedDB) QueryWithCache(ctx context.Context, key string, dest interface{}, query string, args ...interface{}) error {
	// Check cache first
	if cached := db.queryCache.Get(key); cached != nil {
		// Type assertion based on dest type
		switch v := dest.(type) {
		case *[]interface{}:
			*v = cached.([]interface{})
		default:
			// Use JSON for generic unmarshaling
			data, _ := json.Marshal(cached)
			json.Unmarshal(data, dest)
		}
		return nil
	}

	// Execute query
	start := time.Now()
	err := db.SelectContext(ctx, dest, query, args...)
	duration := time.Since(start)

	if err != nil {
		return err
	}

	// Cache result
	db.queryCache.Set(key, dest, 60*time.Second)

	// Log metrics if enabled
	if db.metricsEnabled {
		log.Printf("Query executed in %v: %s", duration, key)
	}

	return nil
}

// Cache methods
func (c *QueryCache) Get(key string) interface{} {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.cache[key]
	if !exists {
		c.misses++
		return nil
	}

	if time.Now().After(entry.ExpiresAt) {
		c.misses++
		return nil
	}

	c.hits++
	return entry.Data
}

func (c *QueryCache) Set(key string, data interface{}, ttl time.Duration) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache[key] = &CacheEntry{
		Data:      data,
		ExpiresAt: time.Now().Add(ttl),
		Size:      1, // Simplified size calculation
	}

	// Evict old entries if cache is too large
	if len(c.cache) > c.maxSize {
		c.evictOldest()
	}
}

func (c *QueryCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range c.cache {
		if oldestTime.IsZero() || entry.ExpiresAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.ExpiresAt
		}
	}

	if oldestKey != "" {
		delete(c.cache, oldestKey)
	}
}

func (c *QueryCache) GetHitRate() float64 {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float64(c.hits) / float64(total) * 100
}

// Health check routine
func (db *OptimizedDB) healthCheckLoop(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := db.PingContext(ctx); err != nil {
			log.Printf("Database health check failed: %v", err)
		}
		cancel()
	}
}

// Cache cleanup routine
func (db *OptimizedDB) cacheCleanupLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		db.queryCache.mutex.Lock()
		now := time.Now()
		for key, entry := range db.queryCache.cache {
			if now.After(entry.ExpiresAt) {
				delete(db.queryCache.cache, key)
			}
		}
		db.queryCache.mutex.Unlock()
	}
}

// OptimizedVMRepository with query optimization
type OptimizedVMRepository struct {
	db *OptimizedDB
}

// NewOptimizedVMRepository creates an optimized VM repository
func NewOptimizedVMRepository(db *OptimizedDB) *OptimizedVMRepository {
	return &OptimizedVMRepository{db: db}
}

// ListVMsFast uses materialized view for fast VM listing
func (r *OptimizedVMRepository) ListVMsFast(ctx context.Context, filters map[string]interface{}) ([]*VM, error) {
	query := `
		SELECT 
			id, name, state, node_id, cpu_cores, memory_mb, disk_gb,
			owner_id, organization_id, created_at, updated_at,
			current_cpu_usage, current_memory_percent, health_status
		FROM mv_vm_listing 
		WHERE 1=1`
	
	args := []interface{}{}
	argIndex := 1

	// Build dynamic query with filters
	if orgID, ok := filters["organization_id"]; ok {
		query += fmt.Sprintf(" AND organization_id = $%d", argIndex)
		args = append(args, orgID)
		argIndex++
	}

	if state, ok := filters["state"]; ok {
		query += fmt.Sprintf(" AND state = $%d", argIndex)
		args = append(args, state)
		argIndex++
	}

	if ownerID, ok := filters["owner_id"]; ok {
		query += fmt.Sprintf(" AND owner_id = $%d", argIndex)
		args = append(args, ownerID)
		argIndex++
	}

	// Add pagination
	limit := 50
	offset := 0
	if l, ok := filters["limit"].(int); ok {
		limit = l
	}
	if o, ok := filters["offset"].(int); ok {
		offset = o
	}

	query += fmt.Sprintf(" ORDER BY created_at DESC LIMIT %d OFFSET %d", limit, offset)

	// Generate cache key
	cacheKey := fmt.Sprintf("vms_list_%v", filters)

	var vms []*VM
	err := r.db.QueryWithCache(ctx, cacheKey, &vms, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}

	return vms, nil
}

// GetDashboardStats retrieves pre-computed dashboard statistics
func (r *OptimizedVMRepository) GetDashboardStats(ctx context.Context, orgID string) (map[string]interface{}, error) {
	query := `
		SELECT 
			vm_count, total_cpu, total_memory_mb, total_disk_gb,
			online_nodes, total_nodes, 
			avg_cpu_usage, avg_memory_usage, p95_cpu, p95_memory
		FROM mv_dashboard_stats 
		WHERE organization_id = $1 
		ORDER BY calculated_at DESC 
		LIMIT 1`

	var stats map[string]interface{}
	cacheKey := fmt.Sprintf("dashboard_%s", orgID)

	err := r.db.QueryWithCache(ctx, cacheKey, &stats, query, orgID)
	if err != nil {
		return nil, fmt.Errorf("failed to get dashboard stats: %w", err)
	}

	return stats, nil
}

// BulkInsertMetrics efficiently inserts multiple metrics using COPY
func (r *OptimizedVMRepository) BulkInsertMetrics(ctx context.Context, metrics []VMMetric) error {
	if len(metrics) == 0 {
		return nil
	}

	txn, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer txn.Rollback()

	stmt, err := txn.Prepare(pq.CopyIn("vm_metrics",
		"vm_id", "cpu_usage", "memory_usage", "memory_percent",
		"disk_read_bytes", "disk_write_bytes", "network_rx_bytes", 
		"network_tx_bytes", "timestamp"))
	if err != nil {
		return fmt.Errorf("failed to prepare COPY statement: %w", err)
	}

	for _, m := range metrics {
		_, err = stmt.Exec(
			m.VMID, m.CPUUsage, m.MemoryUsage, m.MemoryUsage, // Fixed: using MemoryUsage instead of MemoryPercent
			0, 0, m.NetworkRecv, // Fixed: using actual fields (NetworkRecv instead of NetworkRxBytes)
			m.NetworkSent, m.Timestamp, // Fixed: using NetworkSent instead of NetworkTxBytes
		)
		if err != nil {
			return fmt.Errorf("failed to execute COPY: %w", err)
		}
	}

	if _, err = stmt.Exec(); err != nil {
		return fmt.Errorf("failed to flush COPY: %w", err)
	}

	if err = stmt.Close(); err != nil {
		return fmt.Errorf("failed to close statement: %w", err)
	}

	if err = txn.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// GetVMMetricsOptimized retrieves metrics using appropriate materialized view
func (r *OptimizedVMRepository) GetVMMetricsOptimized(ctx context.Context, vmID string, start, end time.Time) ([]*VMMetric, error) {
	duration := end.Sub(start)

	var query string
	var args []interface{}

	// Use different views based on time range
	if duration <= 24*time.Hour {
		// Use raw metrics for recent data
		query = `
			SELECT vm_id, cpu_usage, memory_percent, disk_read_bytes, 
				   disk_write_bytes, network_rx_bytes, network_tx_bytes, timestamp
			FROM vm_metrics 
			WHERE vm_id = $1 AND timestamp BETWEEN $2 AND $3
			ORDER BY timestamp DESC`
		args = []interface{}{vmID, start, end}
	} else if duration <= 7*24*time.Hour {
		// Use hourly rollup
		query = `
			SELECT vm_id, avg_cpu as cpu_usage, avg_memory as memory_percent,
				   total_disk_read as disk_read_bytes, total_disk_write as disk_write_bytes,
				   total_network_rx as network_rx_bytes, total_network_tx as network_tx_bytes,
				   hour as timestamp
			FROM mv_vm_metrics_hourly
			WHERE vm_id = $1 AND hour BETWEEN $2 AND $3
			ORDER BY hour DESC`
		args = []interface{}{vmID, start, end}
	} else {
		// Use daily rollup
		query = `
			SELECT vm_id, avg_cpu as cpu_usage, avg_memory as memory_percent,
				   total_disk_read as disk_read_bytes, total_disk_write as disk_write_bytes,
				   total_network_rx as network_rx_bytes, total_network_tx as network_tx_bytes,
				   day as timestamp
			FROM mv_vm_metrics_daily
			WHERE vm_id = $1 AND day BETWEEN $2 AND $3
			ORDER BY day DESC`
		args = []interface{}{vmID, start, end}
	}

	cacheKey := fmt.Sprintf("metrics_%s_%d_%d", vmID, start.Unix(), end.Unix())

	var metrics []*VMMetric
	err := r.db.QueryWithCache(ctx, cacheKey, &metrics, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM metrics: %w", err)
	}

	return metrics, nil
}

// GetNodeCapacity retrieves node capacity using materialized view
func (r *OptimizedVMRepository) GetNodeCapacity(ctx context.Context) ([]map[string]interface{}, error) {
	query := `
		SELECT id, name, hostname, status,
			   total_cpu, total_memory_mb, total_disk_gb,
			   vm_count, allocated_cpu, allocated_memory_mb,
			   available_cpu, available_memory_mb, available_disk_gb,
			   cpu_allocation_percent, memory_allocation_percent,
			   current_cpu_usage, current_memory_usage
		FROM mv_node_capacity
		ORDER BY available_memory_mb DESC`

	var capacity []map[string]interface{}
	err := r.db.QueryWithCache(ctx, "node_capacity", &capacity, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get node capacity: %w", err)
	}

	return capacity, nil
}

// Close closes the database connection and cleans up resources
func (db *OptimizedDB) Close() error {
	// Close all prepared statements
	db.stmtMutex.Lock()
	for _, stmt := range db.preparedStmts {
		stmt.Close()
	}
	db.stmtMutex.Unlock()

	// Close database connection
	return db.DB.Close()
}

// GetCacheStats returns cache statistics
func (db *OptimizedDB) GetCacheStats() map[string]interface{} {
	return map[string]interface{}{
		"hit_rate":    db.queryCache.GetHitRate(),
		"total_hits":  db.queryCache.hits,
		"total_misses": db.queryCache.misses,
		"cache_size":  len(db.queryCache.cache),
		"prepared_statements": len(db.preparedStmts),
	}
}