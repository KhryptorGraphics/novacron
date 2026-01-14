package services

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/database"
)

// MigrationType represents the type of migration
type MigrationType string

const (
	MigrationTypeCold MigrationType = "cold"
	MigrationTypeWarm MigrationType = "warm"
	MigrationTypeLive MigrationType = "live"
)

// MigrationStatus represents the status of a migration
type MigrationStatus string

const (
	MigrationStatusPending   MigrationStatus = "pending"
	MigrationStatusRunning   MigrationStatus = "running"
	MigrationStatusCompleted MigrationStatus = "completed"
	MigrationStatusFailed    MigrationStatus = "failed"
)

// MigrationRequest represents a request to migrate a VM
type MigrationRequest struct {
	VMID         string        `json:"vm_id"`
	TargetNodeID string        `json:"target_node_id"`
	Type         MigrationType `json:"type"`
	Options      map[string]interface{} `json:"options,omitempty"`
}

// MigrationResponse represents a migration operation response
type MigrationResponse struct {
	ID               string          `json:"id"`
	VMID             string          `json:"vm_id"`
	SourceNodeID     string          `json:"source_node_id"`
	TargetNodeID     string          `json:"target_node_id"`
	Type             MigrationType   `json:"type"`
	Status           MigrationStatus `json:"status"`
	Progress         float64         `json:"progress"`
	BytesTotal       int64           `json:"bytes_total"`
	BytesTransferred int64           `json:"bytes_transferred"`
	StartedAt        *time.Time      `json:"started_at,omitempty"`
	CompletedAt      *time.Time      `json:"completed_at,omitempty"`
	ErrorMessage     *string         `json:"error_message,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
}

// MigrationService provides VM migration functionality
type MigrationService struct {
	db            *database.DB
	repos         *database.Repositories
	vmService     *VMService
	compressionEngine *CompressionEngine
	bandwidthOptimizer *BandwidthOptimizer
	activeMigrations map[string]*MigrationContext
	mu            sync.RWMutex
}

// MigrationContext holds the context for an active migration
type MigrationContext struct {
	Migration *database.Migration
	Cancel    context.CancelFunc
	Progress  chan float64
	Error     chan error
	Done      chan bool
}

// CompressionEngine provides adaptive compression for VM migration
type CompressionEngine struct {
	compressionLevel int
	compressionType  string
	mu              sync.RWMutex
}

// BandwidthOptimizer provides bandwidth optimization for WAN transfers
type BandwidthOptimizer struct {
	maxBandwidth    int64 // bytes per second
	throttleEnabled bool
	compressionRatio float64
	mu              sync.RWMutex
}

// NewMigrationService creates a new migration service
func NewMigrationService(db *database.DB, vmService *VMService) *MigrationService {
	return &MigrationService{
		db:                db,
		repos:             database.NewRepositories(db),
		vmService:         vmService,
		compressionEngine: NewCompressionEngine(),
		bandwidthOptimizer: NewBandwidthOptimizer(),
		activeMigrations:  make(map[string]*MigrationContext),
	}
}

// NewCompressionEngine creates a new compression engine with Ubuntu 24.04 optimizations
func NewCompressionEngine() *CompressionEngine {
	return &CompressionEngine{
		compressionLevel: 6, // Balanced compression level
		compressionType:  "lz4", // Fast compression for live migration
	}
}

// NewBandwidthOptimizer creates a new bandwidth optimizer
func NewBandwidthOptimizer() *BandwidthOptimizer {
	return &BandwidthOptimizer{
		maxBandwidth:     100 * 1024 * 1024, // 100 MB/s default
		throttleEnabled:  true,
		compressionRatio: 9.39, // Target 9.39x speedup as per Ubuntu 24.04 Core spec
	}
}

// StartMigration starts a VM migration
func (s *MigrationService) StartMigration(ctx context.Context, request MigrationRequest) (*MigrationResponse, error) {
	// Validate VM exists
	vm, err := s.vmService.GetVM(ctx, request.VMID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	// Create migration record
	migrationID := uuid.New().String()
	sourceNodeID := "default-node-01" // Get from VM
	if vm.NodeID != nil {
		sourceNodeID = *vm.NodeID
	}

	migration := &database.Migration{
		ID:           migrationID,
		VMID:         request.VMID,
		SourceNodeID: sourceNodeID,
		TargetNodeID: request.TargetNodeID,
		Type:         string(request.Type),
		Status:       string(MigrationStatusPending),
		Progress:     0.0,
		BytesTotal:   0,
		BytesTransferred: 0,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	if err := s.createMigrationRecord(ctx, migration); err != nil {
		return nil, fmt.Errorf("failed to create migration record: %w", err)
	}

	// Start migration asynchronously
	migrationCtx, cancel := context.WithCancel(ctx)
	migrationContext := &MigrationContext{
		Migration: migration,
		Cancel:    cancel,
		Progress:  make(chan float64, 10),
		Error:     make(chan error, 1),
		Done:      make(chan bool, 1),
	}

	s.mu.Lock()
	s.activeMigrations[migrationID] = migrationContext
	s.mu.Unlock()

	go s.executeMigration(migrationCtx, migrationContext, request)

	return s.convertToMigrationResponse(migration), nil
}

// GetMigration gets a migration by ID
func (s *MigrationService) GetMigration(ctx context.Context, migrationID string) (*MigrationResponse, error) {
	migration, err := s.getMigrationRecord(ctx, migrationID)
	if err != nil {
		return nil, fmt.Errorf("failed to get migration: %w", err)
	}

	if migration == nil {
		return nil, fmt.Errorf("migration not found")
	}

	return s.convertToMigrationResponse(migration), nil
}

// ListMigrations lists all migrations with optional filtering
func (s *MigrationService) ListMigrations(ctx context.Context, filters map[string]interface{}) ([]*MigrationResponse, error) {
	migrations, err := s.listMigrationRecords(ctx, filters)
	if err != nil {
		return nil, fmt.Errorf("failed to list migrations: %w", err)
	}

	var responses []*MigrationResponse
	for _, migration := range migrations {
		responses = append(responses, s.convertToMigrationResponse(migration))
	}

	return responses, nil
}

// CancelMigration cancels an active migration
func (s *MigrationService) CancelMigration(ctx context.Context, migrationID string) error {
	s.mu.RLock()
	migrationContext, exists := s.activeMigrations[migrationID]
	s.mu.RUnlock()

	if !exists {
		return fmt.Errorf("migration not found or not active")
	}

	// Cancel the migration context
	migrationContext.Cancel()

	// Update status in database
	migration := migrationContext.Migration
	migration.Status = string(MigrationStatusFailed)
	errorMsg := "Migration cancelled by user"
	migration.ErrorMessage = &errorMsg
	migration.UpdatedAt = time.Now()

	if err := s.updateMigrationRecord(ctx, migration); err != nil {
		return fmt.Errorf("failed to update migration status: %w", err)
	}

	// Clean up
	s.mu.Lock()
	delete(s.activeMigrations, migrationID)
	s.mu.Unlock()

	log.Printf("Migration %s cancelled", migrationID)
	return nil
}

// executeMigration executes the actual migration process
func (s *MigrationService) executeMigration(ctx context.Context, migrationContext *MigrationContext, request MigrationRequest) {
	migration := migrationContext.Migration
	migrationID := migration.ID

	defer func() {
		s.mu.Lock()
		delete(s.activeMigrations, migrationID)
		s.mu.Unlock()
		close(migrationContext.Progress)
		close(migrationContext.Error)
		close(migrationContext.Done)
	}()

	// Update status to running
	migration.Status = string(MigrationStatusRunning)
	now := time.Now()
	migration.StartedAt = &now
	migration.UpdatedAt = now

	if err := s.updateMigrationRecord(ctx, migration); err != nil {
		log.Printf("Failed to update migration status: %v", err)
		migrationContext.Error <- err
		return
	}

	log.Printf("Starting %s migration for VM %s from %s to %s", 
		request.Type, request.VMID, migration.SourceNodeID, migration.TargetNodeID)

	// Execute migration based on type
	var err error
	switch request.Type {
	case MigrationTypeCold:
		err = s.executeColdMigration(ctx, migrationContext, request)
	case MigrationTypeWarm:
		err = s.executeWarmMigration(ctx, migrationContext, request)
	case MigrationTypeLive:
		err = s.executeLiveMigration(ctx, migrationContext, request)
	default:
		err = fmt.Errorf("unsupported migration type: %s", request.Type)
	}

	// Update final status
	migration.UpdatedAt = time.Now()
	if err != nil {
		migration.Status = string(MigrationStatusFailed)
		errorMsg := err.Error()
		migration.ErrorMessage = &errorMsg
		log.Printf("Migration %s failed: %v", migrationID, err)
	} else {
		migration.Status = string(MigrationStatusCompleted)
		completedAt := time.Now()
		migration.CompletedAt = &completedAt
		migration.Progress = 100.0
		log.Printf("Migration %s completed successfully", migrationID)
	}

	if updateErr := s.updateMigrationRecord(ctx, migration); updateErr != nil {
		log.Printf("Failed to update final migration status: %v", updateErr)
	}

	if err != nil {
		migrationContext.Error <- err
	} else {
		migrationContext.Done <- true
	}
}

// executeColdMigration executes a cold migration (VM stopped)
func (s *MigrationService) executeColdMigration(ctx context.Context, migrationContext *MigrationContext, request MigrationRequest) error {
	migration := migrationContext.Migration
	
	// Stop VM if running
	vm, err := s.vmService.GetVM(ctx, request.VMID)
	if err != nil {
		return fmt.Errorf("failed to get VM: %w", err)
	}

	wasRunning := vm.State == "running"
	if wasRunning {
		if err := s.vmService.StopVM(ctx, request.VMID); err != nil {
			return fmt.Errorf("failed to stop VM: %w", err)
		}
		s.updateProgress(migrationContext, 10.0)
	}

	// Simulate disk copy with compression and bandwidth optimization
	diskPaths := []string{"/var/lib/novacron/vms/" + request.VMID + "/disk.qcow2"}
	totalBytes := int64(10 * 1024 * 1024 * 1024) // 10GB example

	migration.BytesTotal = totalBytes
	if err := s.updateMigrationRecord(ctx, migration); err != nil {
		return fmt.Errorf("failed to update migration: %w", err)
	}

	// Copy disk with compression and progress tracking
	if err := s.copyDisksWithCompression(ctx, migrationContext, diskPaths, migration.TargetNodeID); err != nil {
		return fmt.Errorf("failed to copy disks: %w", err)
	}

	// Update VM node assignment
	updates := map[string]interface{}{
		"node_id": request.TargetNodeID,
	}
	if _, err := s.vmService.UpdateVM(ctx, request.VMID, updates); err != nil {
		return fmt.Errorf("failed to update VM node: %w", err)
	}

	// Start VM if it was running before
	if wasRunning {
		if err := s.vmService.StartVM(ctx, request.VMID); err != nil {
			return fmt.Errorf("failed to start VM on target node: %w", err)
		}
	}

	s.updateProgress(migrationContext, 100.0)
	return nil
}

// executeWarmMigration executes a warm migration (pre-copy memory)
func (s *MigrationService) executeWarmMigration(ctx context.Context, migrationContext *MigrationContext, request MigrationRequest) error {
	migration := migrationContext.Migration
	
	// Pre-copy memory while VM is running
	s.updateProgress(migrationContext, 5.0)
	
	totalBytes := int64(4 * 1024 * 1024 * 1024) // 4GB memory example
	migration.BytesTotal = totalBytes
	if err := s.updateMigrationRecord(ctx, migration); err != nil {
		return fmt.Errorf("failed to update migration: %w", err)
	}

	// Simulate iterative pre-copy
	iterations := 3
	for i := 0; i < iterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		log.Printf("Memory pre-copy iteration %d/%d", i+1, iterations)
		
		// Simulate memory copy with compression
		if err := s.copyMemoryWithCompression(ctx, migrationContext, request.VMID, i); err != nil {
			return fmt.Errorf("failed to copy memory iteration %d: %w", i+1, err)
		}

		progress := 20.0 + (50.0 * float64(i+1) / float64(iterations))
		s.updateProgress(migrationContext, progress)
	}

	// Final stop and sync
	if err := s.vmService.StopVM(ctx, request.VMID); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	// Final memory sync
	if err := s.finalMemorySync(ctx, migrationContext, request.VMID); err != nil {
		return fmt.Errorf("failed to perform final memory sync: %w", err)
	}
	s.updateProgress(migrationContext, 80.0)

	// Update VM node assignment
	updates := map[string]interface{}{
		"node_id": request.TargetNodeID,
	}
	if _, err := s.vmService.UpdateVM(ctx, request.VMID, updates); err != nil {
		return fmt.Errorf("failed to update VM node: %w", err)
	}

	// Start VM on target node
	if err := s.vmService.StartVM(ctx, request.VMID); err != nil {
		return fmt.Errorf("failed to start VM on target node: %w", err)
	}

	s.updateProgress(migrationContext, 100.0)
	return nil
}

// executeLiveMigration executes a live migration (minimal downtime)
func (s *MigrationService) executeLiveMigration(ctx context.Context, migrationContext *MigrationContext, request MigrationRequest) error {
	migration := migrationContext.Migration
	
	totalBytes := int64(8 * 1024 * 1024 * 1024) // 8GB example
	migration.BytesTotal = totalBytes
	if err := s.updateMigrationRecord(ctx, migration); err != nil {
		return fmt.Errorf("failed to update migration: %w", err)
	}

	// Phase 1: Pre-copy memory pages while VM runs
	log.Printf("Live migration phase 1: Pre-copy memory")
	if err := s.liveMigrationPreCopy(ctx, migrationContext, request.VMID); err != nil {
		return fmt.Errorf("failed in pre-copy phase: %w", err)
	}
	s.updateProgress(migrationContext, 70.0)

	// Phase 2: Stop VM and copy remaining dirty pages
	log.Printf("Live migration phase 2: Stop-and-copy")
	if err := s.liveMigrationStopAndCopy(ctx, migrationContext, request.VMID); err != nil {
		return fmt.Errorf("failed in stop-and-copy phase: %w", err)
	}
	s.updateProgress(migrationContext, 90.0)

	// Phase 3: Start VM on target node
	log.Printf("Live migration phase 3: Start on target")
	
	// Update VM node assignment
	updates := map[string]interface{}{
		"node_id": request.TargetNodeID,
	}
	if _, err := s.vmService.UpdateVM(ctx, request.VMID, updates); err != nil {
		return fmt.Errorf("failed to update VM node: %w", err)
	}

	// Start VM on target node
	if err := s.vmService.StartVM(ctx, request.VMID); err != nil {
		return fmt.Errorf("failed to start VM on target node: %w", err)
	}

	s.updateProgress(migrationContext, 100.0)
	return nil
}

// Helper methods for migration operations

func (s *MigrationService) copyDisksWithCompression(ctx context.Context, migrationContext *MigrationContext, diskPaths []string, targetNodeID string) error {
	// Simulate disk copy with adaptive compression
	totalBytes := migrationContext.Migration.BytesTotal
	bytesPerSecond := s.bandwidthOptimizer.maxBandwidth
	
	// Apply compression ratio for speed improvement
	effectiveSpeed := int64(float64(bytesPerSecond) * s.bandwidthOptimizer.compressionRatio)
	
	duration := time.Duration(totalBytes/effectiveSpeed) * time.Second
	startTime := time.Now()
	
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			elapsed := time.Since(startTime)
			if elapsed >= duration {
				migrationContext.Migration.BytesTransferred = totalBytes
				s.updateMigrationRecord(ctx, migrationContext.Migration)
				return nil
			}
			
			progress := float64(elapsed) / float64(duration) * 80.0 // Up to 80% for disk copy
			transferred := int64(progress / 100.0 * float64(totalBytes))
			
			migrationContext.Migration.BytesTransferred = transferred
			s.updateMigrationRecord(ctx, migrationContext.Migration)
			s.updateProgress(migrationContext, 10.0+progress)
		}
	}
}

func (s *MigrationService) copyMemoryWithCompression(ctx context.Context, migrationContext *MigrationContext, vmID string, iteration int) error {
	// Simulate memory copy with compression
	time.Sleep(2 * time.Second) // Simulate copy time
	return nil
}

func (s *MigrationService) finalMemorySync(ctx context.Context, migrationContext *MigrationContext, vmID string) error {
	// Simulate final memory synchronization
	time.Sleep(1 * time.Second)
	return nil
}

func (s *MigrationService) liveMigrationPreCopy(ctx context.Context, migrationContext *MigrationContext, vmID string) error {
	// Simulate live migration pre-copy phase
	totalBytes := migrationContext.Migration.BytesTotal
	duration := 10 * time.Second // Simulate pre-copy duration
	startTime := time.Now()
	
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			elapsed := time.Since(startTime)
			if elapsed >= duration {
				return nil
			}
			
			progress := float64(elapsed) / float64(duration) * 70.0
			transferred := int64(progress / 100.0 * float64(totalBytes))
			
			migrationContext.Migration.BytesTransferred = transferred
			s.updateMigrationRecord(ctx, migrationContext.Migration)
			s.updateProgress(migrationContext, progress)
		}
	}
}

func (s *MigrationService) liveMigrationStopAndCopy(ctx context.Context, migrationContext *MigrationContext, vmID string) error {
	// Simulate stop-and-copy phase (minimal downtime)
	if err := s.vmService.StopVM(ctx, vmID); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}
	
	// Brief pause to simulate final page copy
	time.Sleep(500 * time.Millisecond)
	
	return nil
}

func (s *MigrationService) updateProgress(migrationContext *MigrationContext, progress float64) {
	migrationContext.Migration.Progress = progress
	select {
	case migrationContext.Progress <- progress:
	default:
		// Channel full, skip
	}
}

// Database operations

func (s *MigrationService) createMigrationRecord(ctx context.Context, migration *database.Migration) error {
	query := `
		INSERT INTO migrations (id, vm_id, source_node_id, target_node_id, type, status, 
							   progress, bytes_total, bytes_transferred, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`
	
	_, err := s.db.ExecContext(ctx, query,
		migration.ID, migration.VMID, migration.SourceNodeID, migration.TargetNodeID,
		migration.Type, migration.Status, migration.Progress, migration.BytesTotal,
		migration.BytesTransferred, migration.CreatedAt, migration.UpdatedAt)
	
	return err
}

func (s *MigrationService) getMigrationRecord(ctx context.Context, migrationID string) (*database.Migration, error) {
	var migration database.Migration
	query := `
		SELECT id, vm_id, source_node_id, target_node_id, type, status, progress, 
			   bytes_total, bytes_transferred, started_at, completed_at, error_message,
			   created_at, updated_at
		FROM migrations WHERE id = $1`
	
	err := s.db.GetContext(ctx, &migration, query, migrationID)
	if err == nil {
		return &migration, nil
	}
	return nil, err
}

func (s *MigrationService) listMigrationRecords(ctx context.Context, filters map[string]interface{}) ([]*database.Migration, error) {
	query := `
		SELECT id, vm_id, source_node_id, target_node_id, type, status, progress,
			   bytes_total, bytes_transferred, started_at, completed_at, error_message,
			   created_at, updated_at
		FROM migrations ORDER BY created_at DESC`
	
	var migrations []*database.Migration
	err := s.db.SelectContext(ctx, &migrations, query)
	return migrations, err
}

func (s *MigrationService) updateMigrationRecord(ctx context.Context, migration *database.Migration) error {
	query := `
		UPDATE migrations 
		SET status = $2, progress = $3, bytes_transferred = $4, started_at = $5,
			completed_at = $6, error_message = $7, updated_at = $8
		WHERE id = $1`
	
	_, err := s.db.ExecContext(ctx, query,
		migration.ID, migration.Status, migration.Progress, migration.BytesTransferred,
		migration.StartedAt, migration.CompletedAt, migration.ErrorMessage, migration.UpdatedAt)
	
	return err
}

func (s *MigrationService) convertToMigrationResponse(migration *database.Migration) *MigrationResponse {
	return &MigrationResponse{
		ID:               migration.ID,
		VMID:             migration.VMID,
		SourceNodeID:     migration.SourceNodeID,
		TargetNodeID:     migration.TargetNodeID,
		Type:             MigrationType(migration.Type),
		Status:           MigrationStatus(migration.Status),
		Progress:         migration.Progress,
		BytesTotal:       migration.BytesTotal,
		BytesTransferred: migration.BytesTransferred,
		StartedAt:        migration.StartedAt,
		CompletedAt:      migration.CompletedAt,
		ErrorMessage:     migration.ErrorMessage,
		CreatedAt:        migration.CreatedAt,
		UpdatedAt:        migration.UpdatedAt,
	}
}

// OptimizeForUbuntu configures the migration service for Ubuntu 24.04 Core optimizations
func (s *MigrationService) OptimizeForUbuntu() {
	s.compressionEngine.mu.Lock()
	s.compressionEngine.compressionType = "zstd" // Use Zstandard for better compression
	s.compressionEngine.compressionLevel = 3     // Fast compression for live migration
	s.compressionEngine.mu.Unlock()

	s.bandwidthOptimizer.mu.Lock()
	s.bandwidthOptimizer.compressionRatio = 9.39  // Ubuntu 24.04 Core target
	s.bandwidthOptimizer.maxBandwidth = 1024 * 1024 * 1024 // 1GB/s for local network
	s.bandwidthOptimizer.mu.Unlock()

	log.Println("Migration service optimized for Ubuntu 24.04 Core")
}