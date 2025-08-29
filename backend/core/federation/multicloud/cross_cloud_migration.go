package multicloud

import (
	"context"
	"fmt"
	"time"
)

// CrossCloudMigrationEngine handles VM migration between different cloud providers
type CrossCloudMigrationEngine struct {
	registry *ProviderRegistry
}

// NewCrossCloudMigrationEngine creates a new cross-cloud migration engine
func NewCrossCloudMigrationEngine(registry *ProviderRegistry) *CrossCloudMigrationEngine {
	return &CrossCloudMigrationEngine{
		registry: registry,
	}
}

// MigrateVM migrates a VM between cloud providers
func (e *CrossCloudMigrationEngine) MigrateVM(
	ctx context.Context,
	request *CrossCloudMigrationRequest,
	sourceProvider, destProvider CloudProvider) (*CrossCloudMigrationStatus, error) {

	// Create migration status tracker
	status := &CrossCloudMigrationStatus{
		MigrationID:         request.MigrationID,
		VMID:               request.VMID,
		SourceProviderID:   request.SourceProviderID,
		DestinationProviderID: request.DestinationProviderID,
		Status:             MigrationStatusPending,
		StartTime:          time.Now(),
		Progress:           0,
		Steps:              []MigrationStep{},
	}

	// Step 1: Pre-migration validation
	if err := e.preMigrationValidation(ctx, request, sourceProvider, destProvider, status); err != nil {
		return e.updateStatusWithError(status, fmt.Errorf("pre-migration validation failed: %v", err))
	}

	// Step 2: Export VM from source provider
	exportData, err := e.exportVM(ctx, request, sourceProvider, status)
	if err != nil {
		return e.updateStatusWithError(status, fmt.Errorf("VM export failed: %v", err))
	}

	// Step 3: Transform VM data for destination provider
	transformedData, err := e.transformVMData(ctx, exportData, destProvider, status)
	if err != nil {
		return e.updateStatusWithError(status, fmt.Errorf("VM data transformation failed: %v", err))
	}

	// Step 4: Import VM to destination provider
	destVM, err := e.importVM(ctx, transformedData, destProvider, status)
	if err != nil {
		return e.updateStatusWithError(status, fmt.Errorf("VM import failed: %v", err))
	}

	// Step 5: Post-migration validation
	if err := e.postMigrationValidation(ctx, request, destVM, status); err != nil {
		return e.updateStatusWithError(status, fmt.Errorf("post-migration validation failed: %v", err))
	}

	// Step 6: Cleanup source VM (if requested)
	if request.DeleteSource {
		if err := e.cleanupSourceVM(ctx, request, sourceProvider, status); err != nil {
			// Log error but don't fail the migration
			fmt.Printf("Warning: cleanup failed: %v\n", err)
		}
	}

	// Complete migration
	status.Status = MigrationStatusCompleted
	status.Progress = 100
	status.EndTime = time.Now()
	status.DestinationVMID = destVM.ID
	status.CompletionMessage = "Migration completed successfully"

	return status, nil
}

// preMigrationValidation validates the migration request
func (e *CrossCloudMigrationEngine) preMigrationValidation(
	ctx context.Context,
	request *CrossCloudMigrationRequest,
	sourceProvider, destProvider CloudProvider,
	status *CrossCloudMigrationStatus) error {

	step := MigrationStep{
		Name:      "Pre-migration Validation",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// Check VM exists and is in valid state
	vm, err := sourceProvider.GetVM(ctx, request.VMID)
	if err != nil {
		return fmt.Errorf("source VM not found: %v", err)
	}

	if vm.State != VMStateRunning && vm.State != VMStateStopped {
		return fmt.Errorf("VM is in invalid state for migration: %s", vm.State)
	}

	// Check destination provider capabilities
	destCapabilities := destProvider.GetCapabilities()
	hasImportCapability := false
	for _, cap := range destCapabilities {
		if cap == CapabilityVMLiveMigration {
			hasImportCapability = true
			break
		}
	}

	if !hasImportCapability {
		return fmt.Errorf("destination provider does not support VM import")
	}

	// Check resource quota at destination
	quota, err := destProvider.GetResourceQuota(ctx)
	if err != nil {
		return fmt.Errorf("failed to get destination quota: %v", err)
	}

	usage, err := destProvider.GetResourceUsage(ctx)
	if err != nil {
		return fmt.Errorf("failed to get destination usage: %v", err)
	}

	if usage.UsedVMs >= quota.MaxVMs {
		return fmt.Errorf("destination provider has no VM quota available")
	}

	if usage.UsedCPU+vm.CPU > quota.MaxCPU {
		return fmt.Errorf("destination provider has insufficient CPU quota")
	}

	if usage.UsedMemory+vm.Memory > quota.MaxMemory {
		return fmt.Errorf("destination provider has insufficient memory quota")
	}

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()
	status.Progress = 15

	return nil
}

// exportVM exports the VM from the source provider
func (e *CrossCloudMigrationEngine) exportVM(
	ctx context.Context,
	request *CrossCloudMigrationRequest,
	sourceProvider CloudProvider,
	status *CrossCloudMigrationStatus) (*VMExportData, error) {

	step := MigrationStep{
		Name:      "VM Export",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// Determine export format based on destination provider
	format := e.selectExportFormat(request.DestinationProviderID)

	// Export VM
	exportData, err := sourceProvider.ExportVM(ctx, request.VMID, format)
	if err != nil {
		return nil, err
	}

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()
	status.Steps[stepIdx].Details = map[string]interface{}{
		"format": format,
		"size":   exportData.Size,
	}
	status.Progress = 40

	return exportData, nil
}

// transformVMData transforms VM data for the destination provider
func (e *CrossCloudMigrationEngine) transformVMData(
	ctx context.Context,
	exportData *VMExportData,
	destProvider CloudProvider,
	status *CrossCloudMigrationStatus) (*VMExportData, error) {

	step := MigrationStep{
		Name:      "Data Transformation",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// For now, assume data is compatible
	// In a real implementation, this would handle format conversions,
	// metadata transformations, etc.

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()
	status.Progress = 60

	return exportData, nil
}

// importVM imports the VM to the destination provider
func (e *CrossCloudMigrationEngine) importVM(
	ctx context.Context,
	exportData *VMExportData,
	destProvider CloudProvider,
	status *CrossCloudMigrationStatus) (*VMInstance, error) {

	step := MigrationStep{
		Name:      "VM Import",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// Import VM
	vm, err := destProvider.ImportVM(ctx, exportData)
	if err != nil {
		return nil, err
	}

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()
	status.Steps[stepIdx].Details = map[string]interface{}{
		"vm_id": vm.ID,
	}
	status.Progress = 85

	return vm, nil
}

// postMigrationValidation validates the migrated VM
func (e *CrossCloudMigrationEngine) postMigrationValidation(
	ctx context.Context,
	request *CrossCloudMigrationRequest,
	destVM *VMInstance,
	status *CrossCloudMigrationStatus) error {

	step := MigrationStep{
		Name:      "Post-migration Validation",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// Wait for VM to be ready
	maxWait := 5 * time.Minute
	startTime := time.Now()

	destProvider, err := e.registry.GetProvider(request.DestinationProviderID)
	if err != nil {
		return err
	}

	for {
		vm, err := destProvider.GetVM(ctx, destVM.ID)
		if err != nil {
			return fmt.Errorf("failed to get destination VM status: %v", err)
		}

		if vm.State == VMStateRunning {
			break
		}

		if time.Since(startTime) > maxWait {
			return fmt.Errorf("VM did not start within timeout period")
		}

		time.Sleep(10 * time.Second)
	}

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()
	status.Progress = 95

	return nil
}

// cleanupSourceVM removes the source VM
func (e *CrossCloudMigrationEngine) cleanupSourceVM(
	ctx context.Context,
	request *CrossCloudMigrationRequest,
	sourceProvider CloudProvider,
	status *CrossCloudMigrationStatus) error {

	step := MigrationStep{
		Name:      "Source Cleanup",
		Status:    StepStatusInProgress,
		StartTime: time.Now(),
	}
	status.Steps = append(status.Steps, step)

	// Delete source VM
	err := sourceProvider.DeleteVM(ctx, request.VMID)
	if err != nil {
		stepIdx := len(status.Steps) - 1
		status.Steps[stepIdx].Status = StepStatusFailed
		status.Steps[stepIdx].EndTime = time.Now()
		status.Steps[stepIdx].Error = err.Error()
		return err
	}

	// Update step status
	stepIdx := len(status.Steps) - 1
	status.Steps[stepIdx].Status = StepStatusCompleted
	status.Steps[stepIdx].EndTime = time.Now()

	return nil
}

// selectExportFormat selects the best export format for the destination provider
func (e *CrossCloudMigrationEngine) selectExportFormat(destProviderID string) VMExportFormat {
	// Get destination provider
	provider, err := e.registry.GetProvider(destProviderID)
	if err != nil {
		return ExportFormatOVF // Default fallback
	}

	// Select format based on provider type
	switch provider.GetProviderType() {
	case ProviderAWS:
		return ExportFormatVMDK
	case ProviderAzure:
		return ExportFormatVHD
	case ProviderGCP:
		return ExportFormatRAW
	case ProviderVMware:
		return ExportFormatOVF
	case ProviderKVM:
		return ExportFormatQCOW2
	default:
		return ExportFormatOVF
	}
}

// updateStatusWithError updates migration status with error information
func (e *CrossCloudMigrationEngine) updateStatusWithError(status *CrossCloudMigrationStatus, err error) (*CrossCloudMigrationStatus, error) {
	status.Status = MigrationStatusFailed
	status.EndTime = time.Now()
	status.ErrorMessage = err.Error()

	// Mark current step as failed
	if len(status.Steps) > 0 {
		stepIdx := len(status.Steps) - 1
		if status.Steps[stepIdx].Status == StepStatusInProgress {
			status.Steps[stepIdx].Status = StepStatusFailed
			status.Steps[stepIdx].EndTime = time.Now()
			status.Steps[stepIdx].Error = err.Error()
		}
	}

	return status, err
}

// GetMigrationStatus retrieves the status of an ongoing migration
func (e *CrossCloudMigrationEngine) GetMigrationStatus(migrationID string) (*CrossCloudMigrationStatus, error) {
	// In a real implementation, this would retrieve status from a database
	// For now, return a placeholder
	return &CrossCloudMigrationStatus{
		MigrationID: migrationID,
		Status:      MigrationStatusInProgress,
	}, nil
}

// ListMigrations lists all migrations
func (e *CrossCloudMigrationEngine) ListMigrations(filters *MigrationFilters) ([]*CrossCloudMigrationStatus, error) {
	// In a real implementation, this would query a database
	// For now, return empty list
	return []*CrossCloudMigrationStatus{}, nil
}

// Migration request and status types

// CrossCloudMigrationRequest represents a cross-cloud migration request
type CrossCloudMigrationRequest struct {
	MigrationID           string                 `json:"migration_id"`
	VMID                  string                 `json:"vm_id"`
	SourceProviderID      string                 `json:"source_provider_id"`
	DestinationProviderID string                 `json:"destination_provider_id"`
	DestinationRegion     string                 `json:"destination_region"`
	DestinationConfig     *VMCreateRequest       `json:"destination_config,omitempty"`
	DeleteSource          bool                   `json:"delete_source"`
	Options               map[string]interface{} `json:"options,omitempty"`
	ScheduledTime         *time.Time             `json:"scheduled_time,omitempty"`
	MaxDowntime           time.Duration          `json:"max_downtime,omitempty"`
	Rollback              bool                   `json:"rollback"`
}

// CrossCloudMigrationStatus represents the status of a cross-cloud migration
type CrossCloudMigrationStatus struct {
	MigrationID           string          `json:"migration_id"`
	VMID                  string          `json:"vm_id"`
	SourceProviderID      string          `json:"source_provider_id"`
	DestinationProviderID string          `json:"destination_provider_id"`
	Status                MigrationStatus `json:"status"`
	Progress              int             `json:"progress"` // 0-100
	Steps                 []MigrationStep `json:"steps"`
	StartTime             time.Time       `json:"start_time"`
	EndTime               time.Time       `json:"end_time,omitempty"`
	EstimatedCompletion   time.Time       `json:"estimated_completion,omitempty"`
	DestinationVMID       string          `json:"destination_vm_id,omitempty"`
	ErrorMessage          string          `json:"error_message,omitempty"`
	CompletionMessage     string          `json:"completion_message,omitempty"`
	Metrics               *MigrationMetrics `json:"metrics,omitempty"`
}

// MigrationStatus represents the status of a migration
type MigrationStatus string

const (
	MigrationStatusPending    MigrationStatus = "pending"
	MigrationStatusInProgress MigrationStatus = "in_progress"
	MigrationStatusCompleted  MigrationStatus = "completed"
	MigrationStatusFailed     MigrationStatus = "failed"
	MigrationStatusCancelled  MigrationStatus = "cancelled"
	MigrationStatusRolledBack MigrationStatus = "rolled_back"
)

// MigrationStep represents a step in the migration process
type MigrationStep struct {
	Name      string                 `json:"name"`
	Status    StepStatus             `json:"status"`
	StartTime time.Time              `json:"start_time"`
	EndTime   time.Time              `json:"end_time,omitempty"`
	Duration  time.Duration          `json:"duration,omitempty"`
	Progress  int                    `json:"progress"` // 0-100
	Details   map[string]interface{} `json:"details,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// StepStatus represents the status of a migration step
type StepStatus string

const (
	StepStatusPending    StepStatus = "pending"
	StepStatusInProgress StepStatus = "in_progress"
	StepStatusCompleted  StepStatus = "completed"
	StepStatusFailed     StepStatus = "failed"
	StepStatusSkipped    StepStatus = "skipped"
)

// MigrationMetrics represents metrics for a migration
type MigrationMetrics struct {
	DataTransferred   int64         `json:"data_transferred"` // bytes
	TransferRate      float64       `json:"transfer_rate"`    // bytes/second
	TotalDowntime     time.Duration `json:"total_downtime"`
	NetworkLatency    time.Duration `json:"network_latency"`
	CompressionRatio  float64       `json:"compression_ratio"`
	VerificationTime  time.Duration `json:"verification_time"`
}

// MigrationFilters represents filters for listing migrations
type MigrationFilters struct {
	Status            MigrationStatus `json:"status,omitempty"`
	SourceProvider    string          `json:"source_provider,omitempty"`
	DestinationProvider string        `json:"destination_provider,omitempty"`
	StartTimeFrom     *time.Time      `json:"start_time_from,omitempty"`
	StartTimeTo       *time.Time      `json:"start_time_to,omitempty"`
	VMID              string          `json:"vm_id,omitempty"`
}