package vm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

var (
	// ErrMigrationNotFound is returned when a migration record is not found
	ErrMigrationNotFound = errors.New("migration not found")
	
	// ErrMigrationInProgress is returned when a migration is already in progress
	ErrMigrationInProgress = errors.New("migration already in progress")
	
	// ErrMigrationCancelled is returned when a migration is cancelled
	ErrMigrationCancelled = errors.New("migration cancelled")
	
	// ErrVMNotRunning is returned when a warm or live migration is attempted on a non-running VM
	ErrVMNotRunning = errors.New("VM is not running")
	
	// ErrInvalidMigrationType is returned when an invalid migration type is specified
	ErrInvalidMigrationType = errors.New("invalid migration type")
	
	// ErrInvalidTarget is returned when the target node is invalid
	ErrInvalidTarget = errors.New("invalid target node")
	
	// ErrTargetNeedsResources is returned when the target node doesn't have enough resources
	ErrTargetNeedsResources = errors.New("target node has insufficient resources")
)

// Migration represents a VM migration operation
type Migration struct {
	record    *MigrationRecord
	vm        *VM
	sourceNode Node
	targetNode Node
	manager    *MigrationManagerImpl
	executor   MigrationExecutor
	ctx        context.Context
	cancel     context.CancelFunc
	eventCh    chan MigrationEvent
	logger     *logrus.Entry
	mu         sync.Mutex
}

// NewMigration creates a new Migration instance
func NewMigration(record *MigrationRecord, vm *VM, sourceNode, targetNode Node, manager *MigrationManagerImpl, executor MigrationExecutor) *Migration {
	ctx, cancel := context.WithCancel(context.Background())
	logger := logrus.WithFields(logrus.Fields{
		"migration_id":   record.ID,
		"vm_id":          record.VMID,
		"vm_name":        record.VMName,
		"source_node":    record.SourceNodeID,
		"target_node":    record.TargetNodeID,
		"migration_type": record.MigrationType,
	})
	
	return &Migration{
		record:    record,
		vm:        vm,
		sourceNode: sourceNode,
		targetNode: targetNode,
		manager:    manager,
		executor:   executor,
		ctx:        ctx,
		cancel:     cancel,
		eventCh:    make(chan MigrationEvent, 100),
		logger:     logger,
	}
}

// Start begins the migration process
func (m *Migration) Start() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.logger.Info("Starting migration")
	
	// Update record state
	m.record.State = MigrationStateInitiating
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
		return err
	}
	
	// Send initiated event
	m.emitEvent(MigrationEventInitiated, "Migration initiated", 0.0)
	
	// Start the migration in a separate goroutine
	go m.run()
	
	return nil
}

// Cancel stops the migration process
func (m *Migration) Cancel() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Check if the migration is already completed
	if m.record.State == MigrationStateCompleted ||
		m.record.State == MigrationStateFailed ||
		m.record.State == MigrationStateRolledBack {
		return fmt.Errorf("cannot cancel migration in state: %s", m.record.State)
	}
	
	m.logger.Info("Cancelling migration")
	
	// Cancel the context
	m.cancel()
	
	return nil
}

// GetStatus returns the current migration status
func (m *Migration) GetStatus() *MigrationStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	return &MigrationStatus{
		ID:               m.record.ID,
		VMID:             m.record.VMID,
		VMName:           m.record.VMName,
		SourceNodeID:     m.record.SourceNodeID,
		TargetNodeID:     m.record.TargetNodeID,
		MigrationType:    m.record.MigrationType,
		State:            m.record.State,
		Progress:         m.record.Progress,
		StartTime:        m.record.StartTime,
		CompletionTime:   m.record.CompletionTime,
		ErrorMessage:     m.record.ErrorMessage,
		BytesTransferred: m.record.BytesTransferred,
		TotalBytes:       m.record.TotalBytes,
		TransferRate:     m.record.TransferRate,
	}
}

// Events returns the event channel
func (m *Migration) Events() <-chan MigrationEvent {
	return m.eventCh
}

// run executes the migration
func (m *Migration) run() {
	var err error
	
	defer func() {
		close(m.eventCh)
	}()
	
	// Set start time
	m.record.StartTime = time.Now()
	
	// Save record
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
		m.handleError(err)
		return
	}
	
	// Emit started event
	m.emitEvent(MigrationEventStarted, "Migration started", 0.05)
	
	// Execute the appropriate migration type
	switch m.record.MigrationType {
	case MigrationTypeCold:
		err = m.executor.ExecuteColdMigration(m.record.ID, m.vm, m.targetNode)
	case MigrationTypeWarm:
		err = m.executor.ExecuteWarmMigration(m.record.ID, m.vm, m.targetNode)
	case MigrationTypeLive:
		err = m.executor.ExecuteLiveMigration(m.record.ID, m.vm, m.targetNode)
	default:
		err = ErrInvalidMigrationType
	}
	
	// Check if the migration was cancelled
	select {
	case <-m.ctx.Done():
		m.handleCancellation()
		return
	default:
		// Migration not cancelled, check for errors
		if err != nil {
			m.handleError(err)
			return
		}
	}
	
	// Migration successful
	m.handleSuccess()
}

// handleSuccess updates the record when migration is successful
func (m *Migration) handleSuccess() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.logger.Info("Migration completed successfully")
	
	// Update record
	m.record.State = MigrationStateCompleted
	m.record.Progress = 1.0
	m.record.CompletionTime = time.Now()
	
	// Save record
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
	}
	
	// Emit completion event
	m.emitEvent(MigrationEventCompleted, "Migration completed successfully", 1.0)
}

// handleError updates the record when migration fails
func (m *Migration) handleError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.logger.WithError(err).Error("Migration failed")
	
	// Update record
	m.record.State = MigrationStateFailed
	m.record.ErrorMessage = err.Error()
	m.record.CompletionTime = time.Now()
	
	// Save record
	if saveErr := m.manager.SaveMigrationRecord(m.record); saveErr != nil {
		m.logger.WithError(saveErr).Error("Failed to save migration record")
	}
	
	// Emit failure event
	m.emitEvent(MigrationEventFailed, fmt.Sprintf("Migration failed: %s", err.Error()), m.record.Progress)
	
	// Attempt rollback if not already in rollback state
	if m.record.State != MigrationStateRollingBack && m.record.State != MigrationStateRolledBack {
		m.attemptRollback()
	}
}

// handleCancellation updates the record when migration is cancelled
func (m *Migration) handleCancellation() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.logger.Info("Migration cancelled")
	
	// Update record
	m.record.State = MigrationStateFailed
	m.record.ErrorMessage = "Migration cancelled by user"
	m.record.CompletionTime = time.Now()
	
	// Save record
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
	}
	
	// Emit cancellation event
	m.emitEvent(MigrationEventFailed, "Migration cancelled by user", m.record.Progress)
	
	// Attempt rollback
	m.attemptRollback()
}

// attemptRollback tries to roll back the migration
func (m *Migration) attemptRollback() {
	m.logger.Info("Attempting to roll back migration")
	
	// Update record
	m.record.State = MigrationStateRollingBack
	
	// Save record
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
	}
	
	// Emit rollback started event
	m.emitEvent(MigrationEventRollbackStarted, "Starting rollback", m.record.Progress)
	
	// Perform rollback
	err := m.executor.RollbackMigration(m.record.ID, m.vm, m.sourceNode)
	if err != nil {
		m.logger.WithError(err).Error("Failed to roll back migration")
		
		// Update record
		m.record.State = MigrationStateFailed
		m.record.ErrorMessage = fmt.Sprintf("%s (rollback failed: %s)", m.record.ErrorMessage, err.Error())
		
		// Save record
		if saveErr := m.manager.SaveMigrationRecord(m.record); saveErr != nil {
			m.logger.WithError(saveErr).Error("Failed to save migration record")
		}
		
		// Emit rollback failed event
		m.emitEvent(MigrationEventFailed, fmt.Sprintf("Rollback failed: %s", err.Error()), m.record.Progress)
		return
	}
	
	// Rollback successful
	m.logger.Info("Rollback completed successfully")
	
	// Update record
	m.record.State = MigrationStateRolledBack
	
	// Save record
	if err := m.manager.SaveMigrationRecord(m.record); err != nil {
		m.logger.WithError(err).Error("Failed to save migration record")
	}
	
	// Emit rollback done event
	m.emitEvent(MigrationEventRollbackDone, "Rollback completed successfully", m.record.Progress)
}

// emitEvent sends a migration event
func (m *Migration) emitEvent(eventType, message string, progress float64) {
	event := MigrationEvent{
		ID:          fmt.Sprintf("%s-%d", m.record.ID, time.Now().UnixNano()),
		MigrationID: m.record.ID,
		Type:        eventType,
		Timestamp:   time.Now(),
		Message:     message,
		Progress:    progress,
	}
	
	// Add transfer metrics if available
	if m.record.BytesTransferred > 0 {
		event.BytesTransferred = m.record.BytesTransferred
	}
	if m.record.TransferRate > 0 {
		event.TransferRate = m.record.TransferRate
	}
	
	// Update progress in record if changed
	if m.record.Progress != progress {
		m.record.Progress = progress
		if err := m.manager.SaveMigrationRecord(m.record); err != nil {
			m.logger.WithError(err).Warn("Failed to save migration record")
		}
	}
	
	// Send event
	select {
	case m.eventCh <- event:
		// Event sent successfully
	default:
		// Channel full, log warning
		m.logger.Warn("Event channel full, dropping event")
	}
}
