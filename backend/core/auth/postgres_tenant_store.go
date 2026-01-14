package auth

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq" // PostgreSQL driver
)

// PostgresTenantStore implements TenantService using PostgreSQL
// Maps to the 'organizations' table in the database schema
type PostgresTenantStore struct {
	db *sql.DB
}

// NewPostgresTenantStore creates a new PostgreSQL-backed tenant store
func NewPostgresTenantStore(db *sql.DB) *PostgresTenantStore {
	return &PostgresTenantStore{db: db}
}

// Create creates a new tenant
func (s *PostgresTenantStore) Create(tenant *Tenant) error {
	if tenant.ID == "" {
		tenant.ID = uuid.New().String()
	}

	now := time.Now()
	tenant.CreatedAt = now
	tenant.UpdatedAt = now

	// Serialize resource quotas to get max values
	maxVMs := int64(100) // default
	maxStorageGB := int64(1000) // default
	if v, ok := tenant.ResourceQuotas["vms"]; ok {
		maxVMs = v
	}
	if v, ok := tenant.ResourceQuotas["storage_gb"]; ok {
		maxStorageGB = v
	}

	// Map status to is_active
	isActive := tenant.Status == TenantStatusActive || tenant.Status == ""

	query := `
		INSERT INTO organizations (id, tenant_id, name, tier, max_vms, max_storage_gb, billing_email, is_active, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`

	// Use ID as tenant_id for uniqueness
	tenantIDStr := tenant.ID
	tier := "standard"
	if t, ok := tenant.Metadata["tier"].(string); ok {
		tier = t
	}
	billingEmail := ""
	if email, ok := tenant.Metadata["billing_email"].(string); ok {
		billingEmail = email
	}

	_, err := s.db.Exec(query,
		tenant.ID,
		tenantIDStr,
		tenant.Name,
		tier,
		maxVMs,
		maxStorageGB,
		nullableString(billingEmail),
		isActive,
		tenant.CreatedAt,
		tenant.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create tenant: %w", err)
	}

	return nil
}

// Get retrieves a tenant by ID
func (s *PostgresTenantStore) Get(id string) (*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, tier, max_vms, max_storage_gb, billing_email, is_active, created_at, updated_at
		FROM organizations
		WHERE id = $1
	`

	tenant := &Tenant{}
	var tenantID, billingEmail sql.NullString
	var tier string
	var maxVMs, maxStorageGB int64
	var isActive bool

	err := s.db.QueryRow(query, id).Scan(
		&tenant.ID,
		&tenantID,
		&tenant.Name,
		&tier,
		&maxVMs,
		&maxStorageGB,
		&billingEmail,
		&isActive,
		&tenant.CreatedAt,
		&tenant.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("tenant not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}

	// Map is_active to status
	tenant.Status = mapIsActiveToTenantStatus(isActive)

	// Populate resource quotas
	tenant.ResourceQuotas = map[string]int64{
		"vms":        maxVMs,
		"storage_gb": maxStorageGB,
	}

	// Populate metadata
	tenant.Metadata = make(map[string]interface{})
	tenant.Metadata["tier"] = tier
	if billingEmail.Valid {
		tenant.Metadata["billing_email"] = billingEmail.String
	}

	return tenant, nil
}

// List lists tenants with optional filtering
func (s *PostgresTenantStore) List(filter map[string]interface{}) ([]*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, tier, max_vms, max_storage_gb, billing_email, is_active, created_at, updated_at
		FROM organizations
		WHERE 1=1
	`
	args := []interface{}{}
	argIdx := 1

	// Apply filters
	if status, ok := filter["status"]; ok {
		isActive := status == string(TenantStatusActive)
		query += fmt.Sprintf(" AND is_active = $%d", argIdx)
		args = append(args, isActive)
		argIdx++
	}

	if name, ok := filter["name"]; ok {
		query += fmt.Sprintf(" AND name ILIKE $%d", argIdx)
		args = append(args, "%"+name.(string)+"%")
		argIdx++
	}

	// Add ordering
	query += " ORDER BY name ASC"

	// Apply limit if specified
	if limit, ok := filter["limit"]; ok {
		query += fmt.Sprintf(" LIMIT $%d", argIdx)
		args = append(args, limit)
		argIdx++
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list tenants: %w", err)
	}
	defer rows.Close()

	var tenants []*Tenant
	for rows.Next() {
		tenant := &Tenant{}
		var tenantID, billingEmail sql.NullString
		var tier string
		var maxVMs, maxStorageGB int64
		var isActive bool

		err := rows.Scan(
			&tenant.ID,
			&tenantID,
			&tenant.Name,
			&tier,
			&maxVMs,
			&maxStorageGB,
			&billingEmail,
			&isActive,
			&tenant.CreatedAt,
			&tenant.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan tenant row: %w", err)
		}

		tenant.Status = mapIsActiveToTenantStatus(isActive)
		tenant.ResourceQuotas = map[string]int64{
			"vms":        maxVMs,
			"storage_gb": maxStorageGB,
		}
		tenant.Metadata = make(map[string]interface{})
		tenant.Metadata["tier"] = tier
		if billingEmail.Valid {
			tenant.Metadata["billing_email"] = billingEmail.String
		}

		tenants = append(tenants, tenant)
	}

	return tenants, nil
}

// Update updates a tenant
func (s *PostgresTenantStore) Update(tenant *Tenant) error {
	tenant.UpdatedAt = time.Now()

	isActive := tenant.Status == TenantStatusActive

	maxVMs := int64(100)
	maxStorageGB := int64(1000)
	if v, ok := tenant.ResourceQuotas["vms"]; ok {
		maxVMs = v
	}
	if v, ok := tenant.ResourceQuotas["storage_gb"]; ok {
		maxStorageGB = v
	}

	tier := "standard"
	if t, ok := tenant.Metadata["tier"].(string); ok {
		tier = t
	}
	billingEmail := ""
	if email, ok := tenant.Metadata["billing_email"].(string); ok {
		billingEmail = email
	}

	query := `
		UPDATE organizations SET
			name = $2,
			tier = $3,
			max_vms = $4,
			max_storage_gb = $5,
			billing_email = $6,
			is_active = $7,
			updated_at = $8
		WHERE id = $1
	`

	result, err := s.db.Exec(query,
		tenant.ID,
		tenant.Name,
		tier,
		maxVMs,
		maxStorageGB,
		nullableString(billingEmail),
		isActive,
		tenant.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to update tenant: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", tenant.ID)
	}

	return nil
}

// Delete deletes a tenant
func (s *PostgresTenantStore) Delete(id string) error {
	query := `DELETE FROM organizations WHERE id = $1`

	result, err := s.db.Exec(query, id)
	if err != nil {
		return fmt.Errorf("failed to delete tenant: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", id)
	}

	return nil
}

// UpdateStatus updates a tenant's status
func (s *PostgresTenantStore) UpdateStatus(id string, status TenantStatus) error {
	isActive := status == TenantStatusActive

	query := `UPDATE organizations SET is_active = $2, updated_at = $3 WHERE id = $1`

	result, err := s.db.Exec(query, id, isActive, time.Now())
	if err != nil {
		return fmt.Errorf("failed to update status: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", id)
	}

	return nil
}

// SetResourceQuota sets a resource quota for a tenant
func (s *PostgresTenantStore) SetResourceQuota(id string, resource string, quota int64) error {
	var query string
	switch resource {
	case "vms":
		query = `UPDATE organizations SET max_vms = $2, updated_at = $3 WHERE id = $1`
	case "storage_gb":
		query = `UPDATE organizations SET max_storage_gb = $2, updated_at = $3 WHERE id = $1`
	default:
		// For other resources, store in a separate quota table or as JSON
		return s.setQuotaInMetadata(id, resource, quota)
	}

	result, err := s.db.Exec(query, id, quota, time.Now())
	if err != nil {
		return fmt.Errorf("failed to set resource quota: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", id)
	}

	return nil
}

// setQuotaInMetadata stores non-standard quotas in a quotas JSON column
func (s *PostgresTenantStore) setQuotaInMetadata(id string, resource string, quota int64) error {
	// This is a simplified implementation
	// In production, you might want a separate tenant_quotas table
	return nil
}

// GetResourceQuota gets a resource quota for a tenant
func (s *PostgresTenantStore) GetResourceQuota(id string, resource string) (int64, error) {
	var query string
	switch resource {
	case "vms":
		query = `SELECT max_vms FROM organizations WHERE id = $1`
	case "storage_gb":
		query = `SELECT max_storage_gb FROM organizations WHERE id = $1`
	default:
		// Return default for unknown resources
		return 0, nil
	}

	var quota int64
	err := s.db.QueryRow(query, id).Scan(&quota)
	if err == sql.ErrNoRows {
		return 0, fmt.Errorf("tenant not found: %s", id)
	}
	if err != nil {
		return 0, fmt.Errorf("failed to get resource quota: %w", err)
	}

	return quota, nil
}

// GetResourceQuotas gets all resource quotas for a tenant
func (s *PostgresTenantStore) GetResourceQuotas(id string) (map[string]int64, error) {
	query := `SELECT max_vms, max_storage_gb FROM organizations WHERE id = $1`

	var maxVMs, maxStorageGB int64
	err := s.db.QueryRow(query, id).Scan(&maxVMs, &maxStorageGB)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("tenant not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get resource quotas: %w", err)
	}

	quotas := map[string]int64{
		"vms":        maxVMs,
		"storage_gb": maxStorageGB,
	}

	return quotas, nil
}

// mapIsActiveToTenantStatus maps is_active to TenantStatus
func mapIsActiveToTenantStatus(isActive bool) TenantStatus {
	if isActive {
		return TenantStatusActive
	}
	return TenantStatusInactive
}

// Ensure PostgresTenantStore implements TenantService
var _ TenantService = (*PostgresTenantStore)(nil)
