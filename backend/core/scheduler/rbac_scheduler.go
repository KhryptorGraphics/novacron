package scheduler

import (
	"context"
	"fmt"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// RBACScheduler is a wrapper that adds RBAC authorization to a scheduler
type RBACScheduler struct {
	// The wrapped scheduler
	scheduler SchedulerInterface

	// The auth manager for authorization checks
	authManager *auth.AuthManager
}

// NewRBACScheduler creates a new RBAC-enabled scheduler
func NewRBACScheduler(scheduler SchedulerInterface, authManager *auth.AuthManager) *RBACScheduler {
	return &RBACScheduler{
		scheduler:   scheduler,
		authManager: authManager,
	}
}

// Start starts the scheduler
func (s *RBACScheduler) Start() error {
	// No authorization needed for system functions
	return s.scheduler.Start()
}

// Stop stops the scheduler
func (s *RBACScheduler) Stop() error {
	// No authorization needed for system functions
	return s.scheduler.Stop()
}

// UpdateNodeResources updates resources for a node
func (s *RBACScheduler) UpdateNodeResources(ctx context.Context, nodeID string, resources map[ResourceType]*Resource) error {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeNode,
		ResourceID:   nodeID,
		Action:       auth.AuthorizationTypeUpdate,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return s.scheduler.UpdateNodeResources(nodeID, resources)
}

// RequestPlacement requests placement of a VM
func (s *RBACScheduler) RequestPlacement(ctx context.Context, vmID string, policy PlacementPolicy, constraints []PlacementConstraint, resources map[string]float64, priority int) (string, error) {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return "", fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeVM,
		ResourceID:   vmID,
		Action:       auth.AuthorizationTypeCreate,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return "", fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return "", fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return s.scheduler.RequestPlacement(vmID, policy, constraints, resources, priority)
}

// GetPlacementResult gets the result of a placement request
func (s *RBACScheduler) GetPlacementResult(ctx context.Context, requestID string) (*PlacementResult, error) {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return nil, fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeSystem,
		ResourceID:   "placement",
		Action:       auth.AuthorizationTypeRead,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return nil, fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return nil, fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return s.scheduler.GetPlacementResult(requestID)
}

// CancelPlacementRequest cancels a pending placement request
func (s *RBACScheduler) CancelPlacementRequest(ctx context.Context, requestID string) error {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeSystem,
		ResourceID:   "placement",
		Action:       auth.AuthorizationTypeDelete,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return nil // Not implemented yet in base scheduler
}

// RequestMigration requests migration of a VM
func (s *RBACScheduler) RequestMigration(ctx context.Context, vmID string, destNodeID string, options *MigrationOptions) (string, error) {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return "", fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeVM,
		ResourceID:   vmID,
		Action:       auth.AuthorizationTypeUpdate,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return "", fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return "", fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return "", fmt.Errorf("not implemented") // Not implemented yet in base scheduler
}

// GetMigrationResult gets the result of a migration request
func (s *RBACScheduler) GetMigrationResult(ctx context.Context, requestID string) (*MigrationResult, error) {
	// Extract user and tenant from context
	userID, ok := auth.GetContextUserID(ctx)
	if !ok {
		return nil, fmt.Errorf("user ID not found in context")
	}

	tenantID, _ := auth.GetContextTenantID(ctx)

	// Create authorization request
	authReq := auth.AuthorizationRequest{
		UserID:       userID,
		TenantID:     tenantID,
		ResourceType: auth.ResourceTypeSystem,
		ResourceID:   "migration",
		Action:       auth.AuthorizationTypeRead,
	}

	// Check authorization
	result, err := s.authManager.Authorize(authReq)
	if err != nil {
		return nil, fmt.Errorf("authorization check failed: %w", err)
	}

	if !result.Authorized {
		return nil, fmt.Errorf("not authorized: %s", result.Reason)
	}

	// If authorized, perform the operation
	return nil, fmt.Errorf("not implemented") // Not implemented yet in base scheduler
}

// MigrationOptions contains options for migration
type MigrationOptions struct {
	// Priority is the priority of the migration
	Priority int

	// MaxDowntime is the maximum acceptable downtime in milliseconds
	MaxDowntime int64

	// LiveMigration indicates if live migration should be used
	LiveMigration bool
}

// MigrationResult contains the result of a migration
type MigrationResult struct {
	// Status is the status of the migration
	Status string

	// SourceNode is the source node
	SourceNode string

	// DestinationNode is the destination node
	DestinationNode string

	// StartTime is when the migration started
	StartTime string

	// EndTime is when the migration ended
	EndTime string

	// Downtime is the downtime in milliseconds
	Downtime int64

	// Error is any error that occurred
	Error string
}
