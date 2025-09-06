package database

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq" // PostgreSQL driver
)

// DB wraps sqlx.DB with additional functionality
type DB struct {
	*sqlx.DB
}

// New creates a new database connection
func New(databaseURL string) (*DB, error) {
	db, err := sqlx.Connect("postgres", databaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure optimized connection pool for high performance
	db.SetMaxOpenConns(100)              // Increased from 25 to handle more concurrent requests
	db.SetMaxIdleConns(25)               // Increased from 12 to reduce connection overhead
	db.SetConnMaxLifetime(30 * time.Minute) // Increased from 5 minutes for better reuse
	db.SetConnMaxIdleTime(5 * time.Minute)  // Increased from 1 minute to reduce reconnection overhead

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DB{DB: db}, nil
}

// UserRepository provides database operations for users
type UserRepository struct {
	db *DB
}

// NewUserRepository creates a new user repository
func NewUserRepository(db *DB) *UserRepository {
	return &UserRepository{db: db}
}

// Create creates a new user
func (r *UserRepository) Create(ctx context.Context, user *User) error {
	query := `
		INSERT INTO users (username, email, password_hash, role, tenant_id, created_at, updated_at)
		VALUES (:username, :email, :password_hash, :role, :tenant_id, :created_at, :updated_at)
		RETURNING id`
	
	user.CreatedAt = time.Now()
	user.UpdatedAt = user.CreatedAt

	rows, err := r.db.NamedQueryContext(ctx, query, user)
	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	defer rows.Close()

	if rows.Next() {
		if err := rows.Scan(&user.ID); err != nil {
			return fmt.Errorf("failed to scan user ID: %w", err)
		}
	}

	return nil
}

// GetByUsername retrieves a user by username
func (r *UserRepository) GetByUsername(ctx context.Context, username string) (*User, error) {
	var user User
	query := `SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at 
			  FROM users WHERE username = $1`
	
	if err := r.db.GetContext(ctx, &user, query, username); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return &user, nil
}

// GetByID retrieves a user by ID
func (r *UserRepository) GetByID(ctx context.Context, id int) (*User, error) {
	var user User
	query := `SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at 
			  FROM users WHERE id = $1`
	
	if err := r.db.GetContext(ctx, &user, query, id); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return &user, nil
}

// VMRepository provides database operations for VMs
type VMRepository struct {
	db *DB
}

// NewVMRepository creates a new VM repository
func NewVMRepository(db *DB) *VMRepository {
	return &VMRepository{db: db}
}

// Create creates a new VM record
func (r *VMRepository) Create(ctx context.Context, vm *VM) error {
	query := `
		INSERT INTO vms (id, name, state, node_id, owner_id, tenant_id, config, created_at, updated_at)
		VALUES (:id, :name, :state, :node_id, :owner_id, :tenant_id, :config, :created_at, :updated_at)`
	
	vm.CreatedAt = time.Now()
	vm.UpdatedAt = vm.CreatedAt

	if _, err := r.db.NamedExecContext(ctx, query, vm); err != nil {
		return fmt.Errorf("failed to create VM: %w", err)
	}

	return nil
}

// GetByID retrieves a VM by ID
func (r *VMRepository) GetByID(ctx context.Context, id string) (*VM, error) {
	var vm VM
	query := `SELECT id, name, state, node_id, owner_id, tenant_id, config, created_at, updated_at 
			  FROM vms WHERE id = $1`
	
	if err := r.db.GetContext(ctx, &vm, query, id); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	return &vm, nil
}

// List retrieves all VMs with optional filtering and eager loading to prevent N+1 queries
func (r *VMRepository) List(ctx context.Context, filters map[string]interface{}) ([]*VM, error) {
	// Use eager loading with JOINs to avoid N+1 queries
	query := `
		SELECT 
			v.id, v.name, v.state, v.node_id, v.owner_id, v.tenant_id, v.config, v.created_at, v.updated_at,
			u.username as owner_username,
			t.name as tenant_name,
			COALESCE(latest_metrics.cpu_usage, 0) as latest_cpu,
			COALESCE(latest_metrics.memory_usage, 0) as latest_memory
		FROM vms v
		LEFT JOIN users u ON v.owner_id = u.id
		LEFT JOIN tenants t ON v.tenant_id = t.id
		LEFT JOIN LATERAL (
			SELECT cpu_usage, memory_usage 
			FROM vm_metrics 
			WHERE vm_id = v.id 
			ORDER BY timestamp DESC 
			LIMIT 1
		) latest_metrics ON true
		WHERE 1=1`
	args := []interface{}{}
	argIndex := 1

	// Add filters
	if tenantID, ok := filters["tenant_id"]; ok {
		query += fmt.Sprintf(" AND v.tenant_id = $%d", argIndex)
		args = append(args, tenantID)
		argIndex++
	}

	if ownerID, ok := filters["owner_id"]; ok {
		query += fmt.Sprintf(" AND v.owner_id = $%d", argIndex)
		args = append(args, ownerID)
		argIndex++
	}

	if state, ok := filters["state"]; ok {
		query += fmt.Sprintf(" AND v.state = $%d", argIndex)
		args = append(args, state)
		argIndex++
	}

	query += " ORDER BY v.created_at DESC"

	// Use prepared statement for better performance
	stmt, err := r.db.PrepareContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare VM list query: %w", err)
	}
	defer stmt.Close()

	rows, err := stmt.QueryContext(ctx, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute VM list query: %w", err)
	}
	defer rows.Close()

	var vms []*VM
	for rows.Next() {
		vm := &VM{}
		var ownerUsername, tenantName sql.NullString
		var latestCPU, latestMemory sql.NullFloat64
		
		err := rows.Scan(
			&vm.ID, &vm.Name, &vm.State, &vm.NodeID, &vm.OwnerID, &vm.TenantID, 
			&vm.Config, &vm.CreatedAt, &vm.UpdatedAt,
			&ownerUsername, &tenantName, &latestCPU, &latestMemory,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan VM row: %w", err)
		}
		
		// Add preloaded data to avoid additional queries
		if ownerUsername.Valid {
			vm.OwnerUsername = ownerUsername.String
		}
		if tenantName.Valid {
			vm.TenantName = tenantName.String
		}
		if latestCPU.Valid {
			vm.LatestCPU = latestCPU.Float64
		}
		if latestMemory.Valid {
			vm.LatestMemory = latestMemory.Float64
		}
		
		vms = append(vms, vm)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating VM rows: %w", err)
	}

	return vms, nil
}

// Update updates a VM record
func (r *VMRepository) Update(ctx context.Context, vm *VM) error {
	vm.UpdatedAt = time.Now()
	
	query := `
		UPDATE vms 
		SET name = :name, state = :state, node_id = :node_id, config = :config, updated_at = :updated_at
		WHERE id = :id`

	if _, err := r.db.NamedExecContext(ctx, query, vm); err != nil {
		return fmt.Errorf("failed to update VM: %w", err)
	}

	return nil
}

// Delete deletes a VM record
func (r *VMRepository) Delete(ctx context.Context, id string) error {
	query := `DELETE FROM vms WHERE id = $1`
	if _, err := r.db.ExecContext(ctx, query, id); err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}
	return nil
}

// MetricsRepository provides database operations for metrics
type MetricsRepository struct {
	db *DB
}

// NewMetricsRepository creates a new metrics repository
func NewMetricsRepository(db *DB) *MetricsRepository {
	return &MetricsRepository{db: db}
}

// CreateVMMetric creates a new VM metric record
func (r *MetricsRepository) CreateVMMetric(ctx context.Context, metric *VMMetric) error {
	query := `
		INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, disk_usage, network_sent, network_recv, iops, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`
	
	if metric.Timestamp.IsZero() {
		metric.Timestamp = time.Now()
	}

	if _, err := r.db.ExecContext(ctx, query, 
		metric.VMID, metric.CPUUsage, metric.MemoryUsage, metric.DiskUsage,
		metric.NetworkSent, metric.NetworkRecv, metric.IOPS, metric.Timestamp); err != nil {
		return fmt.Errorf("failed to create VM metric: %w", err)
	}

	return nil
}

// GetVMMetrics retrieves VM metrics within a time range
func (r *MetricsRepository) GetVMMetrics(ctx context.Context, vmID string, start, end time.Time) ([]*VMMetric, error) {
	query := `
		SELECT id, vm_id, cpu_usage, memory_usage, disk_usage, network_sent, network_recv, iops, timestamp
		FROM vm_metrics 
		WHERE vm_id = $1 AND timestamp >= $2 AND timestamp <= $3
		ORDER BY timestamp DESC`

	var metrics []*VMMetric
	if err := r.db.SelectContext(ctx, &metrics, query, vmID, start, end); err != nil {
		return nil, fmt.Errorf("failed to get VM metrics: %w", err)
	}

	return metrics, nil
}

// GetLatestVMMetrics retrieves the latest metrics for all VMs with optimized query
func (r *MetricsRepository) GetLatestVMMetrics(ctx context.Context) ([]*VMMetric, error) {
	// Optimized query using DISTINCT ON for better performance
	query := `
		SELECT DISTINCT ON (vm_id) 
			id, vm_id, cpu_usage, memory_usage, disk_usage, 
			network_sent, network_recv, iops, timestamp
		FROM vm_metrics
		WHERE timestamp > NOW() - INTERVAL '1 hour'
		ORDER BY vm_id, timestamp DESC`

	// Use prepared statement with connection pooling
	stmt, err := r.db.PrepareContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare latest VM metrics query: %w", err)
	}
	defer stmt.Close()

	rows, err := stmt.QueryContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to execute latest VM metrics query: %w", err)
	}
	defer rows.Close()

	var metrics []*VMMetric
	for rows.Next() {
		metric := &VMMetric{}
		err := rows.Scan(
			&metric.ID, &metric.VMID, &metric.CPUUsage, &metric.MemoryUsage,
			&metric.DiskUsage, &metric.NetworkSent, &metric.NetworkRecv,
			&metric.IOPS, &metric.Timestamp,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan VM metric row: %w", err)
		}
		metrics = append(metrics, metric)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating VM metric rows: %w", err)
	}

	return metrics, nil
}

// CreateSystemMetric creates a system metric record
func (r *MetricsRepository) CreateSystemMetric(ctx context.Context, metric *SystemMetric) error {
	query := `
		INSERT INTO system_metrics (node_id, cpu_usage, memory_usage, memory_total, memory_available, 
									disk_usage, disk_total, disk_available, network_sent, network_recv,
									load_average_1, load_average_5, load_average_15, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`
	
	if metric.Timestamp.IsZero() {
		metric.Timestamp = time.Now()
	}

	if _, err := r.db.ExecContext(ctx, query, 
		metric.NodeID, metric.CPUUsage, metric.MemoryUsage, metric.MemoryTotal, metric.MemoryAvailable,
		metric.DiskUsage, metric.DiskTotal, metric.DiskAvailable, metric.NetworkSent, metric.NetworkRecv,
		metric.LoadAverage1, metric.LoadAverage5, metric.LoadAverage15, metric.Timestamp); err != nil {
		return fmt.Errorf("failed to create system metric: %w", err)
	}

	return nil
}

// GetSystemMetrics retrieves system metrics within a time range
func (r *MetricsRepository) GetSystemMetrics(ctx context.Context, nodeID string, start, end time.Time) ([]*SystemMetric, error) {
	query := `
		SELECT id, node_id, cpu_usage, memory_usage, memory_total, memory_available,
			   disk_usage, disk_total, disk_available, network_sent, network_recv,
			   load_average_1, load_average_5, load_average_15, timestamp
		FROM system_metrics 
		WHERE node_id = $1 AND timestamp >= $2 AND timestamp <= $3
		ORDER BY timestamp DESC`

	var metrics []*SystemMetric
	if err := r.db.SelectContext(ctx, &metrics, query, nodeID, start, end); err != nil {
		return nil, fmt.Errorf("failed to get system metrics: %w", err)
	}

	return metrics, nil
}

// AlertRepository provides database operations for alerts
type AlertRepository struct {
	db *DB
}

// NewAlertRepository creates a new alert repository
func NewAlertRepository(db *DB) *AlertRepository {
	return &AlertRepository{db: db}
}

// Create creates a new alert
func (r *AlertRepository) Create(ctx context.Context, alert *Alert) error {
	query := `
		INSERT INTO alerts (id, name, description, severity, status, resource, resource_id, 
						   metric_name, threshold, current_value, labels, start_time, created_at, updated_at)
		VALUES (:id, :name, :description, :severity, :status, :resource, :resource_id,
				:metric_name, :threshold, :current_value, :labels, :start_time, :created_at, :updated_at)`
	
	alert.CreatedAt = time.Now()
	alert.UpdatedAt = alert.CreatedAt

	if _, err := r.db.NamedExecContext(ctx, query, alert); err != nil {
		return fmt.Errorf("failed to create alert: %w", err)
	}

	return nil
}

// GetActive retrieves all active alerts
func (r *AlertRepository) GetActive(ctx context.Context) ([]*Alert, error) {
	query := `
		SELECT id, name, description, severity, status, resource, resource_id,
			   metric_name, threshold, current_value, labels, start_time, end_time,
			   acknowledged_by, acknowledged_at, created_at, updated_at
		FROM alerts 
		WHERE status IN ('firing', 'acknowledged') 
		ORDER BY created_at DESC`

	var alerts []*Alert
	if err := r.db.SelectContext(ctx, &alerts, query); err != nil {
		return nil, fmt.Errorf("failed to get active alerts: %w", err)
	}

	return alerts, nil
}

// Acknowledge acknowledges an alert
func (r *AlertRepository) Acknowledge(ctx context.Context, alertID, acknowledgedBy string) error {
	query := `
		UPDATE alerts 
		SET status = 'acknowledged', acknowledged_by = $2, acknowledged_at = $3, updated_at = $4
		WHERE id = $1`

	if _, err := r.db.ExecContext(ctx, query, alertID, acknowledgedBy, time.Now(), time.Now()); err != nil {
		return fmt.Errorf("failed to acknowledge alert: %w", err)
	}

	return nil
}

// AuditRepository provides database operations for audit logs
type AuditRepository struct {
	db *DB
}

// NewAuditRepository creates a new audit repository
func NewAuditRepository(db *DB) *AuditRepository {
	return &AuditRepository{db: db}
}

// Log creates an audit log entry
func (r *AuditRepository) Log(ctx context.Context, entry *AuditLog) error {
	query := `
		INSERT INTO audit_logs (user_id, action, resource, resource_id, details, ip_address, user_agent, success, error_message, timestamp)
		VALUES (:user_id, :action, :resource, :resource_id, :details, :ip_address, :user_agent, :success, :error_message, :timestamp)`
	
	entry.Timestamp = time.Now()

	if _, err := r.db.NamedExecContext(ctx, query, entry); err != nil {
		log.Printf("Failed to create audit log: %v", err)
		// Don't return error - audit logging shouldn't break operations
	}

	return nil
}

// SessionRepository provides database operations for sessions
type SessionRepository struct {
	db *DB
}

// NewSessionRepository creates a new session repository
func NewSessionRepository(db *DB) *SessionRepository {
	return &SessionRepository{db: db}
}

// Create creates a new session
func (r *SessionRepository) Create(ctx context.Context, session *Session) error {
	query := `
		INSERT INTO sessions (id, user_id, token, expires_at, ip_address, user_agent, is_active, created_at, updated_at)
		VALUES (:id, :user_id, :token, :expires_at, :ip_address, :user_agent, :is_active, :created_at, :updated_at)`
	
	session.CreatedAt = time.Now()
	session.UpdatedAt = session.CreatedAt
	session.IsActive = true

	if _, err := r.db.NamedExecContext(ctx, query, session); err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	return nil
}

// GetByToken retrieves a session by token
func (r *SessionRepository) GetByToken(ctx context.Context, token string) (*Session, error) {
	var session Session
	query := `SELECT id, user_id, token, expires_at, ip_address, user_agent, is_active, created_at, updated_at 
			  FROM sessions WHERE token = $1 AND is_active = true AND expires_at > NOW()`
	
	if err := r.db.GetContext(ctx, &session, query, token); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	return &session, nil
}

// Invalidate invalidates a session
func (r *SessionRepository) Invalidate(ctx context.Context, token string) error {
	query := `UPDATE sessions SET is_active = false, updated_at = NOW() WHERE token = $1`
	if _, err := r.db.ExecContext(ctx, query, token); err != nil {
		return fmt.Errorf("failed to invalidate session: %w", err)
	}
	return nil
}

// Repositories bundles all repositories together
type Repositories struct {
	Users    *UserRepository
	VMs      *VMRepository
	Metrics  *MetricsRepository
	Alerts   *AlertRepository
	Audit    *AuditRepository
	Sessions *SessionRepository
}

// NewRepositories creates all repositories
func NewRepositories(db *DB) *Repositories {
	return &Repositories{
		Users:    NewUserRepository(db),
		VMs:      NewVMRepository(db),
		Metrics:  NewMetricsRepository(db),
		Alerts:   NewAlertRepository(db),
		Audit:    NewAuditRepository(db),
		Sessions: NewSessionRepository(db),
	}
}