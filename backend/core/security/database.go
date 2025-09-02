package security

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

// SecureDB wraps database operations with parameterized queries
type SecureDB struct {
	db *sql.DB
}

// NewSecureDB creates a new secure database wrapper
func NewSecureDB(db *sql.DB) *SecureDB {
	return &SecureDB{db: db}
}

// QueryBuilder helps construct parameterized queries safely
type QueryBuilder struct {
	query      strings.Builder
	args       []interface{}
	paramCount int
}

// NewQueryBuilder creates a new query builder
func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{
		args:       make([]interface{}, 0),
		paramCount: 0,
	}
}

// Select adds a SELECT clause
func (qb *QueryBuilder) Select(fields ...string) *QueryBuilder {
	qb.query.WriteString("SELECT ")
	for i, field := range fields {
		if i > 0 {
			qb.query.WriteString(", ")
		}
		qb.query.WriteString(field)
	}
	return qb
}

// From adds a FROM clause
func (qb *QueryBuilder) From(table string) *QueryBuilder {
	qb.query.WriteString(" FROM ")
	qb.query.WriteString(table)
	return qb
}

// Where adds a WHERE clause with parameterized values
func (qb *QueryBuilder) Where(condition string, args ...interface{}) *QueryBuilder {
	qb.query.WriteString(" WHERE ")
	// Replace ? with $1, $2, etc for PostgreSQL
	for range args {
		qb.paramCount++
		condition = strings.Replace(condition, "?", fmt.Sprintf("$%d", qb.paramCount), 1)
	}
	qb.query.WriteString(condition)
	qb.args = append(qb.args, args...)
	return qb
}

// And adds an AND condition
func (qb *QueryBuilder) And(condition string, args ...interface{}) *QueryBuilder {
	qb.query.WriteString(" AND ")
	for range args {
		qb.paramCount++
		condition = strings.Replace(condition, "?", fmt.Sprintf("$%d", qb.paramCount), 1)
	}
	qb.query.WriteString(condition)
	qb.args = append(qb.args, args...)
	return qb
}

// OrderBy adds an ORDER BY clause
func (qb *QueryBuilder) OrderBy(field string, direction string) *QueryBuilder {
	qb.query.WriteString(" ORDER BY ")
	qb.query.WriteString(field)
	qb.query.WriteString(" ")
	// Validate direction to prevent injection
	if direction == "ASC" || direction == "DESC" {
		qb.query.WriteString(direction)
	} else {
		qb.query.WriteString("ASC")
	}
	return qb
}

// Limit adds a LIMIT clause
func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
	qb.query.WriteString(fmt.Sprintf(" LIMIT %d", limit))
	return qb
}

// Build returns the final query and arguments
func (qb *QueryBuilder) Build() (string, []interface{}) {
	return qb.query.String(), qb.args
}

// VMRepository provides secure VM database operations
type VMRepository struct {
	db *SecureDB
}

// NewVMRepository creates a new VM repository
func NewVMRepository(db *SecureDB) *VMRepository {
	return &VMRepository{db: db}
}

// GetVMs retrieves all VMs with optional filtering
func (r *VMRepository) GetVMs(ctx context.Context, state string) ([]*VM, error) {
	qb := NewQueryBuilder()
	qb.Select("id", "name", "state", "node_id", "created_at", "updated_at").
		From("vms")
	
	if state != "" {
		qb.Where("state = ?", state)
	}
	
	qb.OrderBy("created_at", "DESC")
	
	query, args := qb.Build()
	rows, err := r.db.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query VMs: %w", err)
	}
	defer rows.Close()
	
	var vms []*VM
	for rows.Next() {
		vm := &VM{}
		err := rows.Scan(&vm.ID, &vm.Name, &vm.State, &vm.NodeID, &vm.CreatedAt, &vm.UpdatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan VM: %w", err)
		}
		vms = append(vms, vm)
	}
	
	return vms, rows.Err()
}

// GetVMByID retrieves a specific VM by ID
func (r *VMRepository) GetVMByID(ctx context.Context, id string) (*VM, error) {
	qb := NewQueryBuilder()
	query, args := qb.Select("id", "name", "state", "node_id", "created_at", "updated_at").
		From("vms").
		Where("id = ?", id).
		Build()
	
	vm := &VM{}
	err := r.db.db.QueryRowContext(ctx, query, args...).Scan(
		&vm.ID, &vm.Name, &vm.State, &vm.NodeID, &vm.CreatedAt, &vm.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("VM not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}
	
	return vm, nil
}

// CreateVM creates a new VM record
func (r *VMRepository) CreateVM(ctx context.Context, vm *VM) error {
	query := `
		INSERT INTO vms (id, name, state, node_id, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	
	now := time.Now()
	_, err := r.db.db.ExecContext(ctx, query,
		vm.ID, vm.Name, vm.State, vm.NodeID, now, now,
	)
	if err != nil {
		return fmt.Errorf("failed to create VM: %w", err)
	}
	
	vm.CreatedAt = now
	vm.UpdatedAt = now
	return nil
}

// UpdateVM updates an existing VM
func (r *VMRepository) UpdateVM(ctx context.Context, vm *VM) error {
	query := `
		UPDATE vms 
		SET name = $2, state = $3, node_id = $4, updated_at = $5
		WHERE id = $1
	`
	
	now := time.Now()
	result, err := r.db.db.ExecContext(ctx, query,
		vm.ID, vm.Name, vm.State, vm.NodeID, now,
	)
	if err != nil {
		return fmt.Errorf("failed to update VM: %w", err)
	}
	
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("VM not found")
	}
	
	vm.UpdatedAt = now
	return nil
}

// DeleteVM deletes a VM by ID
func (r *VMRepository) DeleteVM(ctx context.Context, id string) error {
	query := "DELETE FROM vms WHERE id = $1"
	
	result, err := r.db.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}
	
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("VM not found")
	}
	
	return nil
}

// VM represents a virtual machine
type VM struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	State     string    `json:"state"`
	NodeID    string    `json:"node_id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UserRepository provides secure user database operations
type UserRepository struct {
	db *SecureDB
}

// NewUserRepository creates a new user repository
func NewUserRepository(db *SecureDB) *UserRepository {
	return &UserRepository{db: db}
}

// GetUserByEmail retrieves a user by email (safe from SQL injection)
func (r *UserRepository) GetUserByEmail(ctx context.Context, email string) (*User, error) {
	query := `
		SELECT id, email, password_hash, role, created_at, updated_at
		FROM users
		WHERE email = $1
	`
	
	user := &User{}
	err := r.db.db.QueryRowContext(ctx, query, email).Scan(
		&user.ID, &user.Email, &user.PasswordHash, &user.Role,
		&user.CreatedAt, &user.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	
	return user, nil
}

// CreateUser creates a new user (safe from SQL injection)
func (r *UserRepository) CreateUser(ctx context.Context, user *User) error {
	query := `
		INSERT INTO users (id, email, password_hash, role, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	
	now := time.Now()
	_, err := r.db.db.ExecContext(ctx, query,
		user.ID, user.Email, user.PasswordHash, user.Role, now, now,
	)
	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	
	user.CreatedAt = now
	user.UpdatedAt = now
	return nil
}

// User represents a system user
type User struct {
	ID           string    `json:"id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	Role         string    `json:"role"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}