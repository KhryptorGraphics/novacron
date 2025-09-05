package performance

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// DatabaseConfig represents optimized database configuration
type DatabaseConfig struct {
	// Connection Pool Settings
	MaxOpenConns    int           `yaml:"maxOpenConns" json:"maxOpenConns"`       // Maximum open connections
	MaxIdleConns    int           `yaml:"maxIdleConns" json:"maxIdleConns"`       // Maximum idle connections  
	ConnMaxLifetime time.Duration `yaml:"connMaxLifetime" json:"connMaxLifetime"` // Connection max lifetime
	ConnMaxIdleTime time.Duration `yaml:"connMaxIdleTime" json:"connMaxIdleTime"` // Connection max idle time
	
	// Query Performance Settings
	QueryTimeout    time.Duration `yaml:"queryTimeout" json:"queryTimeout"`       // Query timeout
	PrepareEnabled  bool          `yaml:"prepareEnabled" json:"prepareEnabled"`   // Use prepared statements
	
	// Monitoring Settings
	SlowQueryThreshold time.Duration `yaml:"slowQueryThreshold" json:"slowQueryThreshold"` // Slow query threshold
	MetricsEnabled     bool          `yaml:"metricsEnabled" json:"metricsEnabled"`         // Enable metrics collection
}

// DefaultDatabaseConfig returns optimized default configuration
func DefaultDatabaseConfig() DatabaseConfig {
	return DatabaseConfig{
		MaxOpenConns:       100,                // Increased from typical 25
		MaxIdleConns:       25,                 // Increased from typical 5
		ConnMaxLifetime:    30 * time.Minute,  // Increased from typical 5m
		ConnMaxIdleTime:    5 * time.Minute,   // Increased from typical 1m
		QueryTimeout:       30 * time.Second,  // Query timeout
		PrepareEnabled:     true,              // Enable prepared statements
		SlowQueryThreshold: 100 * time.Millisecond, // Log slow queries
		MetricsEnabled:     true,              // Enable metrics
	}
}

// OptimizedDB represents an optimized database connection
type OptimizedDB struct {
	*sql.DB
	config  DatabaseConfig
	metrics *DatabaseMetrics
	logger  logger.Logger
}

// DatabaseMetrics tracks database performance metrics
type DatabaseMetrics struct {
	TotalQueries     int64         `json:"totalQueries"`
	SlowQueries      int64         `json:"slowQueries"`
	FailedQueries    int64         `json:"failedQueries"`
	AverageLatency   time.Duration `json:"averageLatency"`
	ConnectionsOpen  int           `json:"connectionsOpen"`
	ConnectionsIdle  int           `json:"connectionsIdle"`
	ConnectionsInUse int           `json:"connectionsInUse"`
	LastUpdate       time.Time     `json:"lastUpdate"`
}

// NewOptimizedDB creates a new optimized database connection
func NewOptimizedDB(driverName, dataSourceName string, config DatabaseConfig, logger logger.Logger) (*OptimizedDB, error) {
	db, err := sql.Open(driverName, dataSourceName)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Apply optimized configuration
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.ConnMaxLifetime)
	db.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	optimizedDB := &OptimizedDB{
		DB:      db,
		config:  config,
		metrics: &DatabaseMetrics{},
		logger:  logger,
	}

	logger.Info("Database connection optimized",
		"maxOpenConns", config.MaxOpenConns,
		"maxIdleConns", config.MaxIdleConns,
		"connMaxLifetime", config.ConnMaxLifetime,
		"connMaxIdleTime", config.ConnMaxIdleTime,
	)

	return optimizedDB, nil
}

// QueryWithMetrics executes a query with performance tracking
func (db *OptimizedDB) QueryWithMetrics(query string, args ...interface{}) (*sql.Rows, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		db.trackQuery(duration, nil)
		
		if duration > db.config.SlowQueryThreshold {
			db.logger.Warn("Slow query detected",
				"query", query,
				"duration", duration,
				"args", args,
			)
		}
	}()

	rows, err := db.DB.Query(query, args...)
	if err != nil {
		db.trackQuery(time.Since(start), err)
		return nil, err
	}

	return rows, nil
}

// ExecWithMetrics executes a statement with performance tracking
func (db *OptimizedDB) ExecWithMetrics(query string, args ...interface{}) (sql.Result, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		db.trackQuery(duration, nil)
		
		if duration > db.config.SlowQueryThreshold {
			db.logger.Warn("Slow exec detected",
				"query", query,
				"duration", duration,
				"args", args,
			)
		}
	}()

	result, err := db.DB.Exec(query, args...)
	if err != nil {
		db.trackQuery(time.Since(start), err)
		return nil, err
	}

	return result, nil
}

// PrepareWithMetrics creates a prepared statement with metrics
func (db *OptimizedDB) PrepareWithMetrics(query string) (*sql.Stmt, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		if duration > db.config.SlowQueryThreshold {
			db.logger.Warn("Slow prepare detected",
				"query", query,
				"duration", duration,
			)
		}
	}()

	return db.DB.Prepare(query)
}

// trackQuery tracks query performance metrics
func (db *OptimizedDB) trackQuery(duration time.Duration, err error) {
	db.metrics.TotalQueries++
	
	if err != nil {
		db.metrics.FailedQueries++
	}
	
	if duration > db.config.SlowQueryThreshold {
		db.metrics.SlowQueries++
	}
	
	// Update average latency (simple moving average)
	if db.metrics.TotalQueries == 1 {
		db.metrics.AverageLatency = duration
	} else {
		db.metrics.AverageLatency = time.Duration(
			(int64(db.metrics.AverageLatency) + int64(duration)) / 2,
		)
	}
	
	db.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current database performance metrics
func (db *OptimizedDB) GetMetrics() DatabaseMetrics {
	stats := db.DB.Stats()
	
	return DatabaseMetrics{
		TotalQueries:     db.metrics.TotalQueries,
		SlowQueries:      db.metrics.SlowQueries,
		FailedQueries:    db.metrics.FailedQueries,
		AverageLatency:   db.metrics.AverageLatency,
		ConnectionsOpen:  stats.OpenConnections,
		ConnectionsIdle:  stats.Idle,
		ConnectionsInUse: stats.InUse,
		LastUpdate:       time.Now(),
	}
}

// Close closes the database connection gracefully
func (db *OptimizedDB) Close() error {
	db.logger.Info("Closing optimized database connection")
	return db.DB.Close()
}

// HealthCheck performs a database health check
func (db *OptimizedDB) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	return db.DB.PingContext(ctx)
}

// GetConnectionPoolStatus returns detailed connection pool status
func (db *OptimizedDB) GetConnectionPoolStatus() map[string]interface{} {
	stats := db.DB.Stats()
	
	return map[string]interface{}{
		"maxOpenConnections": db.config.MaxOpenConns,
		"openConnections":    stats.OpenConnections,
		"inUse":             stats.InUse,
		"idle":              stats.Idle,
		"waitCount":         stats.WaitCount,
		"waitDuration":      stats.WaitDuration,
		"maxIdleClosed":     stats.MaxIdleClosed,
		"maxLifetimeClosed": stats.MaxLifetimeClosed,
	}
}