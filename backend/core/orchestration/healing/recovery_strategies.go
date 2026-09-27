package healing

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// RestartRecoveryStrategy implements recovery by restarting the target
type RestartRecoveryStrategy struct {
	logger   *logrus.Logger
	priority int

	mu           sync.RWMutex
	vmController VMController
}

// MigrateRecoveryStrategy implements recovery by migrating the target
type MigrateRecoveryStrategy struct {
	logger   *logrus.Logger
	priority int

	mu             sync.RWMutex
	vmController   VMController
	targetSelector MigrationTargetSelector
}

// ScaleRecoveryStrategy implements recovery by scaling the target
type ScaleRecoveryStrategy struct {
	logger   *logrus.Logger
	priority int
}

// FailoverRecoveryStrategy implements recovery by failing over to backup
type FailoverRecoveryStrategy struct {
	logger   *logrus.Logger
	priority int
}

// NewRestartRecoveryStrategy creates a new restart recovery strategy
func NewRestartRecoveryStrategy(logger *logrus.Logger) *RestartRecoveryStrategy {
	return &RestartRecoveryStrategy{
		logger:   logger,
		priority: 5, // Medium priority
	}
}

// NewMigrateRecoveryStrategy creates a new migrate recovery strategy
func NewMigrateRecoveryStrategy(logger *logrus.Logger) *MigrateRecoveryStrategy {
	return &MigrateRecoveryStrategy{
		logger:   logger,
		priority: 3, // Lower priority due to complexity
	}
}

// NewScaleRecoveryStrategy creates a new scale recovery strategy
func NewScaleRecoveryStrategy(logger *logrus.Logger) *ScaleRecoveryStrategy {
	return &ScaleRecoveryStrategy{
		logger:   logger,
		priority: 7, // High priority for scalable services
	}
}

// NewFailoverRecoveryStrategy creates a new failover recovery strategy
func NewFailoverRecoveryStrategy(logger *logrus.Logger) *FailoverRecoveryStrategy {
	return &FailoverRecoveryStrategy{
		logger:   logger,
		priority: 8, // High priority for critical services
	}
}

// SetVMController injects the real VM control backend. Safe to call at any
// time; Recover reads it under mu. Nil means "not yet wired" (Recover then
// returns an explicit error rather than a fake success).
func (r *RestartRecoveryStrategy) SetVMController(vc VMController) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.vmController = vc
}

// SetVMController injects the real VM control backend for migration.
func (m *MigrateRecoveryStrategy) SetVMController(vc VMController) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.vmController = vc
}

// SetMigrationTargetSelector injects the destination-picking backend.
func (m *MigrateRecoveryStrategy) SetMigrationTargetSelector(sel MigrationTargetSelector) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.targetSelector = sel
}

// Restart Recovery Strategy Implementation

// GetName returns the strategy name
func (r *RestartRecoveryStrategy) GetName() string {
	return "restart"
}

// CanRecover determines if restart strategy can handle the failure
func (r *RestartRecoveryStrategy) CanRecover(failure *FailureInfo) bool {
	// Restart can handle most failure types except hardware failures
	switch failure.FailureType {
	case FailureTypeUnresponsive, FailureTypeHighError, FailureTypeServiceDown:
		return true
	case FailureTypeResourceExhaustion:
		// Only if it's not a persistent resource issue
		return failure.Severity != SeverityCritical
	case FailureTypeNetworkIssue:
		// Restart won't help with network issues
		return false
	case FailureTypeCustom:
		// Manual operator trigger (TriggerHealing): restart is the least invasive real action.
		// VM restarts still go through RestartSupervisor.RequestRestart, which refuses
		// user-stopped, restart-policy=no and exhausted VMs.
		return true
	default:
		return false
	}
}

// Recover executes the restart recovery action against the injected VM
// controller. Only VM targets have a real backend; every other target type
// (service/node/cluster) returns an explicit unsupported-target error instead
// of a simulated success.
func (r *RestartRecoveryStrategy) Recover(failure *FailureInfo, target *HealingTarget) (*RecoveryResult, error) {
	r.logger.WithFields(logrus.Fields{
		"target_id":    target.ID,
		"target_type":  target.Type,
		"failure_type": failure.FailureType,
	}).Info("Executing restart recovery strategy")

	startTime := time.Now()
	result := &RecoveryResult{
		ActionsExecuted: []string{},
		Errors:          []string{},
		Metadata:        make(map[string]interface{}),
	}

	if target.Type != TargetTypeVM {
		err := fmt.Errorf("restart not supported for target type %q: only vm targets have a real backend", target.Type)
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	r.mu.RLock()
	vc := r.vmController
	r.mu.RUnlock()
	if vc == nil {
		err := fmt.Errorf("restart strategy has no VM controller configured")
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	if err := vc.RestartVM(context.Background(), target.ID); err != nil {
		result.Success = false
		result.Message = "Failed to restart VM"
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	result.ActionsExecuted = append(result.ActionsExecuted, "restart_vm")
	result.Success = true
	result.Message = "Restart completed successfully"
	result.Duration = time.Since(startTime)

	r.logger.WithFields(logrus.Fields{
		"target_id": target.ID,
		"duration":  result.Duration,
		"success":   result.Success,
	}).Info("Restart recovery strategy completed")

	return result, nil
}

// GetPriority returns the strategy priority
func (r *RestartRecoveryStrategy) GetPriority() int {
	return r.priority
}

// EstimateTime estimates recovery time for restart
func (r *RestartRecoveryStrategy) EstimateTime(failure *FailureInfo) time.Duration {
	// Estimate based on failure type and target complexity
	baseTime := 30 * time.Second

	switch failure.FailureType {
	case FailureTypeUnresponsive:
		return baseTime * 2 // May need forced restart
	case FailureTypeServiceDown:
		return baseTime
	case FailureTypeHighError:
		return baseTime * 3 // May need cleanup
	default:
		return baseTime
	}
}

// Migrate Recovery Strategy Implementation

// GetName returns the strategy name
func (m *MigrateRecoveryStrategy) GetName() string {
	return "migrate"
}

// CanRecover determines if migrate strategy can handle the failure
func (m *MigrateRecoveryStrategy) CanRecover(failure *FailureInfo) bool {
	// Migration can handle node-level failures and resource exhaustion
	switch failure.FailureType {
	case FailureTypeResourceExhaustion, FailureTypeNetworkIssue:
		return true
	case FailureTypeUnresponsive:
		// Only if it's a node-level issue
		return failure.Severity == SeverityHigh || failure.Severity == SeverityCritical
	default:
		return false
	}
}

// Recover executes the migrate recovery action: the injected target selector
// picks a destination and the injected VM controller performs the move. Only
// VM targets have a real backend.
func (m *MigrateRecoveryStrategy) Recover(failure *FailureInfo, target *HealingTarget) (*RecoveryResult, error) {
	m.logger.WithFields(logrus.Fields{
		"target_id":    target.ID,
		"target_type":  target.Type,
		"failure_type": failure.FailureType,
	}).Info("Executing migrate recovery strategy")

	startTime := time.Now()
	result := &RecoveryResult{
		ActionsExecuted: []string{},
		Errors:          []string{},
		Metadata:        make(map[string]interface{}),
	}

	if target.Type != TargetTypeVM {
		err := fmt.Errorf("migration not supported for target type %q: only vm targets have a real backend", target.Type)
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	m.mu.RLock()
	vc, sel := m.vmController, m.targetSelector
	m.mu.RUnlock()
	if vc == nil || sel == nil {
		err := fmt.Errorf("migrate strategy has no VM controller/target selector configured")
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	ctx := context.Background()
	targetNode, err := sel.SelectTarget(ctx, target.ID)
	if err != nil {
		result.Success = false
		result.Message = "Failed to select migration target"
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	if err := vc.MigrateVM(ctx, target.ID, targetNode, nil); err != nil {
		result.Success = false
		result.Message = "Failed to migrate VM"
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	result.ActionsExecuted = append(result.ActionsExecuted, "migrate_vm")
	result.Metadata["target_node"] = targetNode
	result.Success = true
	result.Message = "Migration completed successfully"
	result.Duration = time.Since(startTime)

	return result, nil
}

// GetPriority returns the strategy priority
func (m *MigrateRecoveryStrategy) GetPriority() int {
	return m.priority
}

// EstimateTime estimates recovery time for migration
func (m *MigrateRecoveryStrategy) EstimateTime(failure *FailureInfo) time.Duration {
	// Migration typically takes longer
	baseTime := 5 * time.Minute

	switch failure.FailureType {
	case FailureTypeResourceExhaustion:
		return baseTime
	case FailureTypeNetworkIssue:
		return baseTime * 2 // May need to find network-isolated location
	default:
		return baseTime
	}
}

// Scale Recovery Strategy Implementation

// GetName returns the strategy name
func (s *ScaleRecoveryStrategy) GetName() string {
	return "scale"
}

// CanRecover determines if scale strategy can handle the failure
func (s *ScaleRecoveryStrategy) CanRecover(failure *FailureInfo) bool {
	// Scaling can handle resource exhaustion and high load scenarios
	switch failure.FailureType {
	case FailureTypeResourceExhaustion, FailureTypeHighLatency:
		return true
	case FailureTypeHighError:
		// Only if it's due to overload
		return failure.Severity == SeverityMedium || failure.Severity == SeverityHigh
	default:
		return false
	}
}

// Recover executes the scale recovery action
func (s *ScaleRecoveryStrategy) Recover(failure *FailureInfo, target *HealingTarget) (*RecoveryResult, error) {
	s.logger.WithFields(logrus.Fields{
		"target_id":    target.ID,
		"target_type":  target.Type,
		"failure_type": failure.FailureType,
	}).Info("Executing scale recovery strategy")

	startTime := time.Now()
	result := &RecoveryResult{
		ActionsExecuted: []string{},
		Errors:          []string{},
		Metadata:        make(map[string]interface{}),
	}

	// Simulate scaling process
	switch target.Type {
	case TargetTypeService:
		if err := s.scaleService(target, failure, result); err != nil {
			result.Success = false
			result.Message = "Failed to scale service"
			result.Errors = append(result.Errors, err.Error())
			return result, err
		}

	case TargetTypeCluster:
		if err := s.scaleCluster(target, failure, result); err != nil {
			result.Success = false
			result.Message = "Failed to scale cluster"
			result.Errors = append(result.Errors, err.Error())
			return result, err
		}

	default:
		err := fmt.Errorf("scaling not supported for target type %s", target.Type)
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	result.Success = true
	result.Message = "Scaling completed successfully"
	result.Duration = time.Since(startTime)

	return result, nil
}

// GetPriority returns the strategy priority
func (s *ScaleRecoveryStrategy) GetPriority() int {
	return s.priority
}

// EstimateTime estimates recovery time for scaling
func (s *ScaleRecoveryStrategy) EstimateTime(failure *FailureInfo) time.Duration {
	// Scaling time depends on the type of scaling needed
	baseTime := 2 * time.Minute

	switch failure.FailureType {
	case FailureTypeResourceExhaustion:
		return baseTime
	case FailureTypeHighLatency:
		return baseTime * 2 // May need multiple scaling steps
	default:
		return baseTime
	}
}

// Failover Recovery Strategy Implementation

// GetName returns the strategy name
func (f *FailoverRecoveryStrategy) GetName() string {
	return "failover"
}

// CanRecover determines if failover strategy can handle the failure
func (f *FailoverRecoveryStrategy) CanRecover(failure *FailureInfo) bool {
	// Failover can handle most critical failures if backup exists
	return failure.Severity == SeverityHigh || failure.Severity == SeverityCritical
}

// Recover executes the failover recovery action
func (f *FailoverRecoveryStrategy) Recover(failure *FailureInfo, target *HealingTarget) (*RecoveryResult, error) {
	f.logger.WithFields(logrus.Fields{
		"target_id":    target.ID,
		"target_type":  target.Type,
		"failure_type": failure.FailureType,
	}).Info("Executing failover recovery strategy")

	startTime := time.Now()
	result := &RecoveryResult{
		ActionsExecuted: []string{},
		Errors:          []string{},
		Metadata:        make(map[string]interface{}),
	}

	// Check if failover target exists
	if !f.hasFailoverTarget(target) {
		err := fmt.Errorf("no failover target available for %s", target.ID)
		result.Success = false
		result.Message = err.Error()
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	// Simulate failover process
	if err := f.executeFailover(target, result); err != nil {
		result.Success = false
		result.Message = "Failover execution failed"
		result.Errors = append(result.Errors, err.Error())
		return result, err
	}

	result.Success = true
	result.Message = "Failover completed successfully"
	result.Duration = time.Since(startTime)

	return result, nil
}

// GetPriority returns the strategy priority
func (f *FailoverRecoveryStrategy) GetPriority() int {
	return f.priority
}

// EstimateTime estimates recovery time for failover
func (f *FailoverRecoveryStrategy) EstimateTime(failure *FailureInfo) time.Duration {
	// Failover is typically quick if backup is ready
	return 1 * time.Minute
}

// Private helper methods for scale and failover strategies (restart and
// migrate delegate to the injected VMController/MigrationTargetSelector
// directly in Recover, above).

func (s *ScaleRecoveryStrategy) scaleService(target *HealingTarget, failure *FailureInfo, result *RecoveryResult) error {
	// Service scaling not implemented: requires orchestrator integration
	result.ActionsExecuted = append(result.ActionsExecuted, "scale_service_not_implemented")
	return fmt.Errorf("scaleService not implemented: requires orchestrator integration")
}

func (s *ScaleRecoveryStrategy) scaleCluster(target *HealingTarget, failure *FailureInfo, result *RecoveryResult) error {
	// Cluster scaling not implemented: requires cluster manager integration
	result.ActionsExecuted = append(result.ActionsExecuted, "scale_cluster_not_implemented")
	return fmt.Errorf("scaleCluster not implemented: requires cluster manager integration")
}

func (f *FailoverRecoveryStrategy) hasFailoverTarget(target *HealingTarget) bool {
	// Check if failover target exists (simplified check)
	if metadata, exists := target.Metadata["failover_target"]; exists {
		return metadata != nil && metadata != ""
	}
	return false
}

func (f *FailoverRecoveryStrategy) executeFailover(target *HealingTarget, result *RecoveryResult) error {
	// Failover not implemented: requires cluster and load balancer integration
	result.ActionsExecuted = append(result.ActionsExecuted, "failover_not_implemented")
	return fmt.Errorf("executeFailover not implemented: requires cluster and load balancer integration")
}
