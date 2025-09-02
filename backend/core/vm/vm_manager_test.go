package vm

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/go-redis/redis/v8"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Mock dependencies
type MockHypervisorManager struct {
	mock.Mock
}

func (m *MockHypervisorManager) CreateVM(ctx context.Context, config VMConfig) (*VM, error) {
	args := m.Called(ctx, config)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*VM), args.Error(1)
}

func (m *MockHypervisorManager) StartVM(ctx context.Context, vmID string) error {
	args := m.Called(ctx, vmID)
	return args.Error(0)
}

func (m *MockHypervisorManager) StopVM(ctx context.Context, vmID string, force bool) error {
	args := m.Called(ctx, vmID, force)
	return args.Error(0)
}

func (m *MockHypervisorManager) DeleteVM(ctx context.Context, vmID string) error {
	args := m.Called(ctx, vmID)
	return args.Error(0)
}

func (m *MockHypervisorManager) GetVMStatus(ctx context.Context, vmID string) (VMStatus, error) {
	args := m.Called(ctx, vmID)
	return args.Get(0).(VMStatus), args.Error(1)
}

type MockEventBus struct {
	mock.Mock
}

func (m *MockEventBus) Publish(ctx context.Context, event Event) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockEventBus) Subscribe(ctx context.Context, eventType string, handler func(Event)) error {
	args := m.Called(ctx, eventType, handler)
	return args.Error(0)
}

// Test VMManager creation
func TestNewVMManager(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	redisClient := redis.NewClient(&redis.Options{})
	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}

	manager := NewVMManager(db, redisClient, hypervisor, eventBus)
	
	assert.NotNil(t, manager)
	assert.Equal(t, db, manager.db)
	assert.Equal(t, redisClient, manager.cache)
	assert.Equal(t, hypervisor, manager.hypervisor)
	assert.Equal(t, eventBus, manager.eventBus)
}

// Test CreateVM success
func TestVMManager_CreateVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	redisClient := redis.NewClient(&redis.Options{})
	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}

	manager := NewVMManager(db, redisClient, hypervisor, eventBus)

	ctx := context.Background()
	config := VMConfig{
		Name: "test-vm",
		Resources: Resources{
			CPU:    4,
			Memory: 8192,
			Disk:   100,
		},
	}

	expectedVM := &VM{
		ID:     "vm-123",
		Name:   config.Name,
		Status: VMStatusRunning,
		Resources: config.Resources,
		CreatedAt: time.Now(),
	}

	// Mock hypervisor CreateVM
	hypervisor.On("CreateVM", ctx, config).Return(expectedVM, nil)

	// Mock database insert
	sqlMock.ExpectBegin()
	sqlMock.ExpectExec("INSERT INTO virtual_machines").
		WithArgs(expectedVM.ID, expectedVM.Name, expectedVM.Status, 
			sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))
	sqlMock.ExpectCommit()

	// Mock event publish
	eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil)

	// Execute
	vm, err := manager.CreateVM(ctx, config)

	// Assert
	require.NoError(t, err)
	assert.Equal(t, expectedVM.ID, vm.ID)
	assert.Equal(t, expectedVM.Name, vm.Name)
	assert.Equal(t, expectedVM.Status, vm.Status)
	
	hypervisor.AssertExpectations(t)
	eventBus.AssertExpectations(t)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test CreateVM validation error
func TestVMManager_CreateVM_ValidationError(t *testing.T) {
	db, _, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	
	// Test empty name
	config := VMConfig{
		Name: "",
		Resources: Resources{
			CPU:    4,
			Memory: 8192,
			Disk:   100,
		},
	}

	_, err = manager.CreateVM(ctx, config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "name is required")

	// Test invalid resources
	config.Name = "test-vm"
	config.Resources.CPU = 0

	_, err = manager.CreateVM(ctx, config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "CPU must be greater than 0")
}

// Test StartVM success
func TestVMManager_StartVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}
	manager := NewVMManager(db, nil, hypervisor, eventBus)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query
	rows := sqlmock.NewRows([]string{"status"}).AddRow(VMStatusStopped)
	sqlMock.ExpectQuery("SELECT status FROM virtual_machines").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Mock hypervisor StartVM
	hypervisor.On("StartVM", ctx, vmID).Return(nil)

	// Mock database update
	sqlMock.ExpectExec("UPDATE virtual_machines SET status").
		WithArgs(VMStatusRunning, vmID).
		WillReturnResult(sqlmock.NewResult(1, 1))

	// Mock event publish
	eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil)

	// Execute
	err = manager.StartVM(ctx, vmID)

	// Assert
	require.NoError(t, err)
	hypervisor.AssertExpectations(t)
	eventBus.AssertExpectations(t)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test StartVM already running
func TestVMManager_StartVM_AlreadyRunning(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query - VM already running
	rows := sqlmock.NewRows([]string{"status"}).AddRow(VMStatusRunning)
	sqlMock.ExpectQuery("SELECT status FROM virtual_machines").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Execute
	err = manager.StartVM(ctx, vmID)

	// Assert
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already running")
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test StopVM success
func TestVMManager_StopVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}
	manager := NewVMManager(db, nil, hypervisor, eventBus)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query
	rows := sqlmock.NewRows([]string{"status"}).AddRow(VMStatusRunning)
	sqlMock.ExpectQuery("SELECT status FROM virtual_machines").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Mock hypervisor StopVM
	hypervisor.On("StopVM", ctx, vmID, false).Return(nil)

	// Mock database update
	sqlMock.ExpectExec("UPDATE virtual_machines SET status").
		WithArgs(VMStatusStopped, vmID).
		WillReturnResult(sqlmock.NewResult(1, 1))

	// Mock event publish
	eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil)

	// Execute
	err = manager.StopVM(ctx, vmID, false)

	// Assert
	require.NoError(t, err)
	hypervisor.AssertExpectations(t)
	eventBus.AssertExpectations(t)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test DeleteVM success
func TestVMManager_DeleteVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}
	manager := NewVMManager(db, nil, hypervisor, eventBus)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query - VM is stopped
	rows := sqlmock.NewRows([]string{"status"}).AddRow(VMStatusStopped)
	sqlMock.ExpectQuery("SELECT status FROM virtual_machines").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Mock hypervisor DeleteVM
	hypervisor.On("DeleteVM", ctx, vmID).Return(nil)

	// Mock database delete
	sqlMock.ExpectBegin()
	sqlMock.ExpectExec("DELETE FROM virtual_machines").
		WithArgs(vmID).
		WillReturnResult(sqlmock.NewResult(1, 1))
	sqlMock.ExpectCommit()

	// Mock event publish
	eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil)

	// Execute
	err = manager.DeleteVM(ctx, vmID)

	// Assert
	require.NoError(t, err)
	hypervisor.AssertExpectations(t)
	eventBus.AssertExpectations(t)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test DeleteVM while running
func TestVMManager_DeleteVM_WhileRunning(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query - VM is running
	rows := sqlmock.NewRows([]string{"status"}).AddRow(VMStatusRunning)
	sqlMock.ExpectQuery("SELECT status FROM virtual_machines").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Execute
	err = manager.DeleteVM(ctx, vmID)

	// Assert
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot delete running VM")
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test ListVMs with pagination
func TestVMManager_ListVMs_Pagination(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	page := 2
	limit := 10

	// Mock count query
	countRows := sqlmock.NewRows([]string{"count"}).AddRow(25)
	sqlMock.ExpectQuery("SELECT COUNT").
		WillReturnRows(countRows)

	// Mock list query
	listRows := sqlmock.NewRows([]string{"id", "name", "status", "resources", "created_at"}).
		AddRow("vm-1", "test-vm-1", VMStatusRunning, `{"cpu":4,"memory":8192}`, time.Now()).
		AddRow("vm-2", "test-vm-2", VMStatusStopped, `{"cpu":2,"memory":4096}`, time.Now())
	
	sqlMock.ExpectQuery("SELECT .* FROM virtual_machines").
		WithArgs(limit, (page-1)*limit).
		WillReturnRows(listRows)

	// Execute
	result, err := manager.ListVMs(ctx, page, limit, nil)

	// Assert
	require.NoError(t, err)
	assert.Len(t, result.VMs, 2)
	assert.Equal(t, 25, result.Total)
	assert.Equal(t, page, result.Page)
	assert.Equal(t, limit, result.Limit)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test GetVM success
func TestVMManager_GetVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	vmID := "vm-123"

	// Mock database query
	rows := sqlMock.NewRows([]string{"id", "name", "status", "resources", "created_at", "updated_at"}).
		AddRow(vmID, "test-vm", VMStatusRunning, `{"cpu":4,"memory":8192}`, time.Now(), time.Now())

	sqlMock.ExpectQuery("SELECT .* FROM virtual_machines WHERE id").
		WithArgs(vmID).
		WillReturnRows(rows)

	// Execute
	vm, err := manager.GetVM(ctx, vmID)

	// Assert
	require.NoError(t, err)
	assert.Equal(t, vmID, vm.ID)
	assert.Equal(t, "test-vm", vm.Name)
	assert.Equal(t, VMStatusRunning, vm.Status)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test GetVM not found
func TestVMManager_GetVM_NotFound(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	manager := NewVMManager(db, nil, nil, nil)

	ctx := context.Background()
	vmID := "vm-nonexistent"

	// Mock database query - no rows
	sqlMock.ExpectQuery("SELECT .* FROM virtual_machines WHERE id").
		WithArgs(vmID).
		WillReturnError(sql.ErrNoRows)

	// Execute
	_, err = manager.GetVM(ctx, vmID)

	// Assert
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test UpdateVM success
func TestVMManager_UpdateVM_Success(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	eventBus := &MockEventBus{}
	manager := NewVMManager(db, nil, nil, eventBus)

	ctx := context.Background()
	vmID := "vm-123"
	
	update := VMUpdate{
		Name: strPtr("updated-vm"),
		Resources: &Resources{
			CPU:    8,
			Memory: 16384,
			Disk:   200,
		},
		Metadata: map[string]interface{}{
			"environment": "production",
		},
	}

	// Mock check existence
	checkRows := sqlMock.NewRows([]string{"id"}).AddRow(vmID)
	sqlMock.ExpectQuery("SELECT id FROM virtual_machines WHERE id").
		WithArgs(vmID).
		WillReturnRows(checkRows)

	// Mock update
	sqlMock.ExpectBegin()
	sqlMock.ExpectExec("UPDATE virtual_machines SET").
		WithArgs(update.Name, sqlmock.AnyArg(), sqlmock.AnyArg(), vmID).
		WillReturnResult(sqlmock.NewResult(1, 1))
	sqlMock.ExpectCommit()

	// Mock event publish
	eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil)

	// Execute
	err = manager.UpdateVM(ctx, vmID, update)

	// Assert
	require.NoError(t, err)
	eventBus.AssertExpectations(t)
	assert.NoError(t, sqlMock.ExpectationsWereMet())
}

// Test concurrent operations
func TestVMManager_ConcurrentOperations(t *testing.T) {
	db, sqlMock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()

	hypervisor := &MockHypervisorManager{}
	eventBus := &MockEventBus{}
	manager := NewVMManager(db, nil, hypervisor, eventBus)

	ctx := context.Background()
	vmCount := 10

	// Set up expectations for concurrent VM creations
	for i := 0; i < vmCount; i++ {
		config := VMConfig{
			Name: fmt.Sprintf("vm-%d", i),
			Resources: Resources{
				CPU:    2,
				Memory: 4096,
				Disk:   50,
			},
		}

		expectedVM := &VM{
			ID:     fmt.Sprintf("vm-id-%d", i),
			Name:   config.Name,
			Status: VMStatusRunning,
		}

		hypervisor.On("CreateVM", ctx, config).Return(expectedVM, nil).Maybe()
		eventBus.On("Publish", ctx, mock.AnythingOfType("Event")).Return(nil).Maybe()
		
		sqlMock.ExpectBegin()
		sqlMock.ExpectExec("INSERT INTO virtual_machines").
			WillReturnResult(sqlmock.NewResult(1, 1)).
			WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), 
				sqlmock.AnyArg(), sqlmock.AnyArg())
		sqlMock.ExpectCommit()
	}

	// Execute concurrent operations
	errChan := make(chan error, vmCount)
	for i := 0; i < vmCount; i++ {
		go func(index int) {
			config := VMConfig{
				Name: fmt.Sprintf("vm-%d", index),
				Resources: Resources{
					CPU:    2,
					Memory: 4096,
					Disk:   50,
				},
			}
			_, err := manager.CreateVM(ctx, config)
			errChan <- err
		}(i)
	}

	// Collect results
	for i := 0; i < vmCount; i++ {
		err := <-errChan
		assert.NoError(t, err)
	}
}

// Helper function
func strPtr(s string) *string {
	return &s
}