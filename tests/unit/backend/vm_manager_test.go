package backend_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// MockVMDriver is a mock implementation of VMDriver
type MockVMDriver struct {
	mock.Mock
}

func (m *MockVMDriver) CreateVM(ctx context.Context, config *vm.VMConfig) (*vm.VM, error) {
	args := m.Called(ctx, config)
	return args.Get(0).(*vm.VM), args.Error(1)
}

func (m *MockVMDriver) StartVM(ctx context.Context, vmID string) error {
	args := m.Called(ctx, vmID)
	return args.Error(0)
}

func (m *MockVMDriver) StopVM(ctx context.Context, vmID string) error {
	args := m.Called(ctx, vmID)
	return args.Error(0)
}

func (m *MockVMDriver) DestroyVM(ctx context.Context, vmID string) error {
	args := m.Called(ctx, vmID)
	return args.Error(0)
}

func (m *MockVMDriver) GetVM(ctx context.Context, vmID string) (*vm.VM, error) {
	args := m.Called(ctx, vmID)
	return args.Get(0).(*vm.VM), args.Error(1)
}

func (m *MockVMDriver) ListVMs(ctx context.Context) ([]*vm.VM, error) {
	args := m.Called(ctx)
	return args.Get(0).([]*vm.VM), args.Error(1)
}

func (m *MockVMDriver) GetVMStatus(ctx context.Context, vmID string) (vm.VMStatus, error) {
	args := m.Called(ctx, vmID)
	return args.Get(0).(vm.VMStatus), args.Error(1)
}

// MockVMScheduler is a mock implementation of VMScheduler
type MockVMScheduler struct {
	mock.Mock
}

func (m *MockVMScheduler) ScheduleVM(ctx context.Context, config *vm.VMConfig) (*vm.VMPlacement, error) {
	args := m.Called(ctx, config)
	return args.Get(0).(*vm.VMPlacement), args.Error(1)
}

func (m *MockVMScheduler) GetOptimalNodes(ctx context.Context, requirements *vm.ResourceRequirements) ([]*vm.Node, error) {
	args := m.Called(ctx, requirements)
	return args.Get(0).([]*vm.Node), args.Error(1)
}

// VMManagerTestSuite defines the test suite for VM manager
type VMManagerTestSuite struct {
	suite.Suite
	vmManager   *vm.VMManager
	mockDriver  *MockVMDriver
	mockScheduler *MockVMScheduler
	ctx         context.Context
	cancel      context.CancelFunc
}

func (suite *VMManagerTestSuite) SetupTest() {
	suite.mockDriver = new(MockVMDriver)
	suite.mockScheduler = new(MockVMScheduler)
	suite.ctx, suite.cancel = context.WithCancel(context.Background())
	
	config := &vm.VMManagerConfig{
		DefaultDriver:   vm.VMTypeQEMU,
		UpdateInterval:  time.Second * 30,
		CleanupInterval: time.Minute * 5,
		DefaultVMType:   vm.VMTypeQEMU,
		RetentionPeriod: time.Hour * 24,
	}
	
	suite.vmManager = vm.NewVMManager(config)
	// Register mock driver
	suite.vmManager.RegisterDriver(vm.VMTypeQEMU, suite.mockDriver)
	suite.vmManager.SetScheduler(suite.mockScheduler)
}

func (suite *VMManagerTestSuite) TearDownTest() {
	suite.cancel()
	if suite.vmManager != nil {
		suite.vmManager.Shutdown(suite.ctx)
	}
}

func (suite *VMManagerTestSuite) TestCreateVM_Success() {
	// Arrange
	config := &vm.VMConfig{
		Name:     "test-vm",
		Type:     vm.VMTypeQEMU,
		CPU:      2,
		Memory:   4096,
		Disk:     20,
		Networks: []string{"default"},
	}
	
	expectedVM := &vm.VM{
		ID:     "test-vm-id",
		Name:   "test-vm",
		Status: vm.VMStatusCreated,
		Config: config,
	}
	
	placement := &vm.VMPlacement{
		NodeID:   "node-1",
		NodeType: "compute",
	}

	suite.mockScheduler.On("ScheduleVM", mock.AnythingOfType("*context.cancelCtx"), config).Return(placement, nil)
	suite.mockDriver.On("CreateVM", mock.AnythingOfType("*context.cancelCtx"), config).Return(expectedVM, nil)

	// Act
	vm, err := suite.vmManager.CreateVM(suite.ctx, config)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), vm)
	assert.Equal(suite.T(), expectedVM.ID, vm.ID)
	assert.Equal(suite.T(), expectedVM.Name, vm.Name)
	suite.mockDriver.AssertExpectations(suite.T())
	suite.mockScheduler.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestCreateVM_SchedulingFailure() {
	// Arrange
	config := &vm.VMConfig{
		Name: "test-vm",
		Type: vm.VMTypeQEMU,
		CPU:  8, // High resource requirement
		Memory: 32768,
	}

	suite.mockScheduler.On("ScheduleVM", mock.AnythingOfType("*context.cancelCtx"), config).Return((*vm.VMPlacement)(nil), vm.ErrNoSuitableNodes)

	// Act
	vm, err := suite.vmManager.CreateVM(suite.ctx, config)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
	assert.Equal(suite.T(), vm.ErrNoSuitableNodes, err)
	suite.mockScheduler.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestCreateVM_DriverFailure() {
	// Arrange
	config := &vm.VMConfig{
		Name: "test-vm",
		Type: vm.VMTypeQEMU,
	}
	
	placement := &vm.VMPlacement{
		NodeID: "node-1",
	}

	suite.mockScheduler.On("ScheduleVM", mock.AnythingOfType("*context.cancelCtx"), config).Return(placement, nil)
	suite.mockDriver.On("CreateVM", mock.AnythingOfType("*context.cancelCtx"), config).Return((*vm.VM)(nil), errors.New("driver error"))

	// Act
	vm, err := suite.vmManager.CreateVM(suite.ctx, config)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
	suite.mockDriver.AssertExpectations(suite.T())
	suite.mockScheduler.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestStartVM_Success() {
	// Arrange
	vmID := "test-vm-id"
	suite.mockDriver.On("StartVM", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(nil)

	// Act
	err := suite.vmManager.StartVM(suite.ctx, vmID)

	// Assert
	assert.NoError(suite.T(), err)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestStartVM_VMNotFound() {
	// Arrange
	vmID := "nonexistent-vm-id"
	suite.mockDriver.On("StartVM", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(vm.ErrVMNotFound)

	// Act
	err := suite.vmManager.StartVM(suite.ctx, vmID)

	// Assert
	assert.Error(suite.T(), err)
	assert.Equal(suite.T(), vm.ErrVMNotFound, err)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestStopVM_Success() {
	// Arrange
	vmID := "test-vm-id"
	suite.mockDriver.On("StopVM", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(nil)

	// Act
	err := suite.vmManager.StopVM(suite.ctx, vmID)

	// Assert
	assert.NoError(suite.T(), err)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestDestroyVM_Success() {
	// Arrange
	vmID := "test-vm-id"
	suite.mockDriver.On("DestroyVM", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(nil)

	// Act
	err := suite.vmManager.DestroyVM(suite.ctx, vmID)

	// Assert
	assert.NoError(suite.T(), err)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestGetVM_Success() {
	// Arrange
	vmID := "test-vm-id"
	expectedVM := &vm.VM{
		ID:     vmID,
		Name:   "test-vm",
		Status: vm.VMStatusRunning,
	}

	suite.mockDriver.On("GetVM", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(expectedVM, nil)

	// Act
	vm, err := suite.vmManager.GetVM(suite.ctx, vmID)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), vm)
	assert.Equal(suite.T(), expectedVM.ID, vm.ID)
	assert.Equal(suite.T(), expectedVM.Name, vm.Name)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestListVMs_Success() {
	// Arrange
	expectedVMs := []*vm.VM{
		{ID: "vm-1", Name: "test-vm-1", Status: vm.VMStatusRunning},
		{ID: "vm-2", Name: "test-vm-2", Status: vm.VMStatusStopped},
		{ID: "vm-3", Name: "test-vm-3", Status: vm.VMStatusCreated},
	}

	suite.mockDriver.On("ListVMs", mock.AnythingOfType("*context.cancelCtx")).Return(expectedVMs, nil)

	// Act
	vms, err := suite.vmManager.ListVMs(suite.ctx)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), vms)
	assert.Len(suite.T(), vms, 3)
	assert.Equal(suite.T(), expectedVMs, vms)
	suite.mockDriver.AssertExpectations(suite.T())
}

func (suite *VMManagerTestSuite) TestGetVMStatus_Success() {
	// Arrange
	vmID := "test-vm-id"
	expectedStatus := vm.VMStatusRunning

	suite.mockDriver.On("GetVMStatus", mock.AnythingOfType("*context.cancelCtx"), vmID).Return(expectedStatus, nil)

	// Act
	status, err := suite.vmManager.GetVMStatus(suite.ctx, vmID)

	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), expectedStatus, status)
	suite.mockDriver.AssertExpectations(suite.T())
}

// Test concurrent operations
func (suite *VMManagerTestSuite) TestConcurrentOperations() {
	// Arrange
	vmConfigs := []*vm.VMConfig{
		{Name: "vm-1", Type: vm.VMTypeQEMU},
		{Name: "vm-2", Type: vm.VMTypeQEMU},
		{Name: "vm-3", Type: vm.VMTypeQEMU},
	}
	
	placement := &vm.VMPlacement{NodeID: "node-1"}
	
	for i, config := range vmConfigs {
		expectedVM := &vm.VM{
			ID:   fmt.Sprintf("vm-%d-id", i+1),
			Name: config.Name,
			Status: vm.VMStatusCreated,
		}
		suite.mockScheduler.On("ScheduleVM", mock.AnythingOfType("*context.cancelCtx"), config).Return(placement, nil)
		suite.mockDriver.On("CreateVM", mock.AnythingOfType("*context.cancelCtx"), config).Return(expectedVM, nil)
	}

	// Act - Create VMs concurrently
	results := make(chan error, len(vmConfigs))
	for _, config := range vmConfigs {
		go func(cfg *vm.VMConfig) {
			_, err := suite.vmManager.CreateVM(suite.ctx, cfg)
			results <- err
		}(config)
	}

	// Assert
	for i := 0; i < len(vmConfigs); i++ {
		err := <-results
		assert.NoError(suite.T(), err)
	}
	
	suite.mockDriver.AssertExpectations(suite.T())
	suite.mockScheduler.AssertExpectations(suite.T())
}

// Test context cancellation
func (suite *VMManagerTestSuite) TestContextCancellation() {
	// Arrange
	ctx, cancel := context.WithCancel(context.Background())
	config := &vm.VMConfig{Name: "test-vm", Type: vm.VMTypeQEMU}
	
	// Cancel context immediately
	cancel()

	// Act
	vm, err := suite.vmManager.CreateVM(ctx, config)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
	assert.Equal(suite.T(), context.Canceled, err)
}

// Test timeout scenarios
func (suite *VMManagerTestSuite) TestTimeout() {
	// Arrange
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*100)
	defer cancel()
	
	config := &vm.VMConfig{Name: "test-vm", Type: vm.VMTypeQEMU}
	placement := &vm.VMPlacement{NodeID: "node-1"}
	
	suite.mockScheduler.On("ScheduleVM", mock.AnythingOfType("*context.timerCtx"), config).Return(placement, nil)
	
	// Simulate slow driver operation
	suite.mockDriver.On("CreateVM", mock.AnythingOfType("*context.timerCtx"), config).Return(func(ctx context.Context, config *vm.VMConfig) (*vm.VM, error) {
		time.Sleep(time.Millisecond * 200) // Longer than timeout
		return nil, context.DeadlineExceeded
	})

	// Act
	vm, err := suite.vmManager.CreateVM(ctx, config)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
}

// Edge case tests
func (suite *VMManagerTestSuite) TestEdgeCases() {
	// Test with nil config
	vm, err := suite.vmManager.CreateVM(suite.ctx, nil)
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
	
	// Test with empty VM name
	config := &vm.VMConfig{Name: "", Type: vm.VMTypeQEMU}
	vm, err = suite.vmManager.CreateVM(suite.ctx, config)
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
	
	// Test with invalid VM type
	config = &vm.VMConfig{Name: "test", Type: "invalid"}
	vm, err = suite.vmManager.CreateVM(suite.ctx, config)
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), vm)
}

// Run the test suite
func TestVMManagerTestSuite(t *testing.T) {
	suite.Run(t, new(VMManagerTestSuite))
}

// Individual unit tests for VM struct
func TestVM_IsRunning(t *testing.T) {
	runningVM := &vm.VM{Status: vm.VMStatusRunning}
	assert.True(t, runningVM.IsRunning())
	
	stoppedVM := &vm.VM{Status: vm.VMStatusStopped}
	assert.False(t, stoppedVM.IsRunning())
}

func TestVM_IsStopped(t *testing.T) {
	stoppedVM := &vm.VM{Status: vm.VMStatusStopped}
	assert.True(t, stoppedVM.IsStopped())
	
	runningVM := &vm.VM{Status: vm.VMStatusRunning}
	assert.False(t, runningVM.IsStopped())
}

func TestVM_ValidateConfig(t *testing.T) {
	// Valid config
	validConfig := &vm.VMConfig{
		Name:   "test-vm",
		Type:   vm.VMTypeQEMU,
		CPU:    2,
		Memory: 4096,
		Disk:   20,
	}
	err := validConfig.Validate()
	assert.NoError(t, err)
	
	// Invalid config - no name
	invalidConfig := &vm.VMConfig{
		Type:   vm.VMTypeQEMU,
		CPU:    2,
		Memory: 4096,
	}
	err = invalidConfig.Validate()
	assert.Error(t, err)
	
	// Invalid config - zero memory
	invalidConfig = &vm.VMConfig{
		Name:   "test",
		Type:   vm.VMTypeQEMU,
		CPU:    2,
		Memory: 0,
	}
	err = invalidConfig.Validate()
	assert.Error(t, err)
}

// Benchmark tests
func BenchmarkCreateVM(b *testing.B) {
	mockDriver := new(MockVMDriver)
	mockScheduler := new(MockVMScheduler)
	
	config := &vm.VMManagerConfig{
		DefaultDriver: vm.VMTypeQEMU,
	}
	
	vmManager := vm.NewVMManager(config)
	vmManager.RegisterDriver(vm.VMTypeQEMU, mockDriver)
	vmManager.SetScheduler(mockScheduler)
	
	vmConfig := &vm.VMConfig{Name: "bench-vm", Type: vm.VMTypeQEMU}
	placement := &vm.VMPlacement{NodeID: "node-1"}
	expectedVM := &vm.VM{ID: "vm-id", Name: "bench-vm"}
	
	mockScheduler.On("ScheduleVM", mock.Anything, vmConfig).Return(placement, nil)
	mockDriver.On("CreateVM", mock.Anything, vmConfig).Return(expectedVM, nil)
	
	ctx := context.Background()
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		vmManager.CreateVM(ctx, vmConfig)
	}
}

func BenchmarkListVMs(b *testing.B) {
	mockDriver := new(MockVMDriver)
	
	vms := []*vm.VM{
		{ID: "vm-1", Name: "vm-1"},
		{ID: "vm-2", Name: "vm-2"},
	}
	
	config := &vm.VMManagerConfig{DefaultDriver: vm.VMTypeQEMU}
	vmManager := vm.NewVMManager(config)
	vmManager.RegisterDriver(vm.VMTypeQEMU, mockDriver)
	
	mockDriver.On("ListVMs", mock.Anything).Return(vms, nil)
	
	ctx := context.Background()
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		vmManager.ListVMs(ctx)
	}
}