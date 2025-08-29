package controllers

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/novacron"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VirtualMachineReconciler reconciles a VirtualMachine object
type VirtualMachineReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	NovaCronClient *novacron.Client
	Recorder       record.EventRecorder
}

const (
	VirtualMachineFinalizerName = "novacron.io/virtualMachine"
	RequeueInterval             = 30 * time.Second
	VirtualMachineAnnotationVMID = "novacron.io/vm-id"
)

//+kubebuilder:rbac:groups=novacron.io,resources=virtualmachines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=virtualmachines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=virtualmachines/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile implements the reconciliation logic for VirtualMachine resources
func (r *VirtualMachineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("virtualmachine", req.NamespacedName)

	// Fetch the VirtualMachine resource
	vm := &novacronv1.VirtualMachine{}
	if err := r.Get(ctx, req.NamespacedName, vm); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("VirtualMachine resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get VirtualMachine")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if vm.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, vm)
	}

	// Add finalizer if not present
	if !containsFinalizer(vm.Finalizers, VirtualMachineFinalizerName) {
		vm.Finalizers = append(vm.Finalizers, VirtualMachineFinalizerName)
		if err := r.Update(ctx, vm); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update observed generation
	if vm.Status.ObservedGeneration != vm.Generation {
		vm.Status.ObservedGeneration = vm.Generation
		if err := r.Status().Update(ctx, vm); err != nil {
			logger.Error(err, "Failed to update observed generation")
			return ctrl.Result{}, err
		}
	}

	// Handle VM lifecycle based on current phase
	return r.handleVMLifecycle(ctx, vm)
}

func (r *VirtualMachineReconciler) handleVMLifecycle(ctx context.Context, vm *novacronv1.VirtualMachine) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get VM ID from annotations if exists
	vmID := vm.Annotations[VirtualMachineAnnotationVMID]

	switch vm.Status.Phase {
	case novacronv1.VMPhasePending, "":
		return r.handlePendingPhase(ctx, vm)
	case novacronv1.VMPhaseRunning:
		return r.handleRunningPhase(ctx, vm, vmID)
	case novacronv1.VMPhaseFailed:
		return r.handleFailedPhase(ctx, vm, vmID)
	default:
		logger.Info("Unknown VM phase", "phase", vm.Status.Phase)
		return ctrl.Result{RequeueAfter: RequeueInterval}, nil
	}
}

func (r *VirtualMachineReconciler) handlePendingPhase(ctx context.Context, vm *novacronv1.VirtualMachine) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Convert Kubernetes VM spec to NovaCron format
	vmConfig, err := r.convertVMSpec(ctx, vm)
	if err != nil {
		logger.Error(err, "Failed to convert VM spec")
		return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseFailed, "", "", err.Error())
	}

	// Create VM in NovaCron
	createReq := &novacron.CreateVMRequest{
		Name:       vm.Spec.Name,
		Command:    "",
		Args:       nil,
		CPUShares:  vmConfig.CPUShares,
		MemoryMB:   vmConfig.MemoryMB,
		DiskSizeGB: vmConfig.DiskSizeGB,
		Tags:       map[string]string{
			"kubernetes.namespace": vm.Namespace,
			"kubernetes.name":      vm.Name,
		},
		TenantID: "kubernetes-" + vm.Namespace,
	}

	// Extract command and args from config
	if vmConfig.Command != "" {
		createReq.Command = vmConfig.Command
	}
	if len(vmConfig.Args) > 0 {
		createReq.Args = vmConfig.Args
	}

	vmResponse, err := r.NovaCronClient.CreateVM(ctx, createReq)
	if err != nil {
		logger.Error(err, "Failed to create VM in NovaCron")
		r.Recorder.Event(vm, "Warning", "VMCreationFailed", fmt.Sprintf("Failed to create VM: %v", err))
		return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseFailed, "", "", err.Error())
	}

	// Store VM ID in annotations
	if vm.Annotations == nil {
		vm.Annotations = make(map[string]string)
	}
	vm.Annotations[VirtualMachineAnnotationVMID] = vmResponse.ID

	if err := r.Update(ctx, vm); err != nil {
		logger.Error(err, "Failed to update VM annotations")
		return ctrl.Result{}, err
	}

	// Start the VM
	if err := r.NovaCronClient.StartVM(ctx, vmResponse.ID); err != nil {
		logger.Error(err, "Failed to start VM")
		r.Recorder.Event(vm, "Warning", "VMStartFailed", fmt.Sprintf("Failed to start VM: %v", err))
		return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseFailed, vmResponse.ID, "", err.Error())
	}

	logger.Info("VM created and started successfully", "vmID", vmResponse.ID)
	r.Recorder.Event(vm, "Normal", "VMCreated", fmt.Sprintf("VM %s created successfully", vmResponse.ID))

	return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseRunning, vmResponse.ID, vmResponse.State, "")
}

func (r *VirtualMachineReconciler) handleRunningPhase(ctx context.Context, vm *novacronv1.VirtualMachine, vmID string) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if vmID == "" {
		logger.Error(nil, "VM ID not found in annotations")
		return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseFailed, "", "", "VM ID not found")
	}

	// Get VM status from NovaCron
	vmResponse, err := r.NovaCronClient.GetVM(ctx, vmID)
	if err != nil {
		logger.Error(err, "Failed to get VM status")
		return ctrl.Result{RequeueAfter: RequeueInterval}, err
	}

	// Get VM metrics
	metrics, err := r.NovaCronClient.GetVMMetrics(ctx, vmID)
	if err != nil {
		logger.V(1).Info("Failed to get VM metrics", "error", err)
		// Don't fail reconciliation if metrics unavailable
		metrics = nil
	}

	// Update status based on VM state
	var phase novacronv1.VMPhase
	switch vmResponse.State {
	case "running":
		phase = novacronv1.VMPhaseRunning
	case "failed":
		phase = novacronv1.VMPhaseFailed
	case "stopped", "succeeded":
		phase = novacronv1.VMPhaseSucceeded
	default:
		phase = novacronv1.VMPhaseUnknown
	}

	// Update resource usage if metrics available
	var resourceUsage *novacronv1.ResourceUsage
	if metrics != nil {
		resourceUsage = &novacronv1.ResourceUsage{
			CPU:    fmt.Sprintf("%.2f%%", metrics.CPUUsage),
			Memory: fmt.Sprintf("%.2f%%", metrics.MemoryUsage),
		}
	}

	// Update VM status
	vm.Status.Phase = phase
	vm.Status.State = vmResponse.State
	vm.Status.VMID = vmResponse.ID
	vm.Status.NodeID = ""
	if vmResponse.NodeID != nil {
		vm.Status.NodeID = *vmResponse.NodeID
	}
	vm.Status.ResourceUsage = resourceUsage

	// Update conditions
	now := metav1.Now()
	vm.Status.Conditions = r.updateConditions(vm.Status.Conditions, novacronv1.VirtualMachineReady, 
		metav1.ConditionTrue, "VMRunning", "VM is running successfully", now)

	if err := r.Status().Update(ctx, vm); err != nil {
		logger.Error(err, "Failed to update VM status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: RequeueInterval}, nil
}

func (r *VirtualMachineReconciler) handleFailedPhase(ctx context.Context, vm *novacronv1.VirtualMachine, vmID string) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Check restart policy
	if vm.Spec.RestartPolicy == novacronv1.VMRestartPolicyNever {
		logger.Info("VM failed, restart policy is Never")
		return ctrl.Result{}, nil
	}

	if vm.Spec.RestartPolicy == novacronv1.VMRestartPolicyOnFailure || vm.Spec.RestartPolicy == novacronv1.VMRestartPolicyAlways {
		logger.Info("Attempting to restart failed VM", "vmID", vmID)
		
		if vmID != "" {
			if err := r.NovaCronClient.StartVM(ctx, vmID); err != nil {
				logger.Error(err, "Failed to restart VM")
				return ctrl.Result{RequeueAfter: RequeueInterval}, nil
			}
			
			r.Recorder.Event(vm, "Normal", "VMRestarted", "VM restarted after failure")
			return r.updateVMStatus(ctx, vm, novacronv1.VMPhaseRunning, vmID, "running", "")
		}
	}

	return ctrl.Result{RequeueAfter: RequeueInterval}, nil
}

func (r *VirtualMachineReconciler) handleDeletion(ctx context.Context, vm *novacronv1.VirtualMachine) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(vm.Finalizers, VirtualMachineFinalizerName) {
		// Clean up VM in NovaCron
		vmID := vm.Annotations[VirtualMachineAnnotationVMID]
		if vmID != "" {
			logger.Info("Deleting VM from NovaCron", "vmID", vmID)
			if err := r.NovaCronClient.DeleteVM(ctx, vmID); err != nil {
				logger.Error(err, "Failed to delete VM from NovaCron")
				// Continue with finalizer removal even if deletion fails
				// The VM might already be deleted
			}
		}

		// Remove finalizer
		vm.Finalizers = removeFinalizer(vm.Finalizers, VirtualMachineFinalizerName)
		if err := r.Update(ctx, vm); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *VirtualMachineReconciler) updateVMStatus(ctx context.Context, vm *novacronv1.VirtualMachine, 
	phase novacronv1.VMPhase, vmID, state, errorMsg string) (ctrl.Result, error) {
	
	vm.Status.Phase = phase
	vm.Status.VMID = vmID
	vm.Status.State = state

	// Update conditions
	now := metav1.Now()
	var conditionType novacronv1.VirtualMachineConditionType
	var conditionStatus metav1.ConditionStatus
	var reason, message string

	switch phase {
	case novacronv1.VMPhaseRunning:
		conditionType = novacronv1.VirtualMachineReady
		conditionStatus = metav1.ConditionTrue
		reason = "VMRunning"
		message = "VM is running successfully"
	case novacronv1.VMPhaseFailed:
		conditionType = novacronv1.VirtualMachineReady
		conditionStatus = metav1.ConditionFalse
		reason = "VMFailed"
		message = errorMsg
		if message == "" {
			message = "VM has failed"
		}
	default:
		conditionType = novacronv1.VirtualMachineReady
		conditionStatus = metav1.ConditionUnknown
		reason = "VMPending"
		message = "VM is in pending state"
	}

	vm.Status.Conditions = r.updateConditions(vm.Status.Conditions, conditionType, conditionStatus, reason, message, now)

	if err := r.Status().Update(ctx, vm); err != nil {
		return ctrl.Result{}, err
	}

	var requeueAfter time.Duration
	if phase == novacronv1.VMPhaseRunning {
		requeueAfter = RequeueInterval
	} else if phase == novacronv1.VMPhaseFailed {
		requeueAfter = RequeueInterval
	}

	return ctrl.Result{RequeueAfter: requeueAfter}, nil
}

func (r *VirtualMachineReconciler) convertVMSpec(ctx context.Context, vm *novacronv1.VirtualMachine) (*VMConfigInternal, error) {
	config := &VMConfigInternal{}

	if vm.Spec.Config != nil {
		// Parse CPU request
		if vm.Spec.Config.Resources.CPU.Request != "" {
			cpuShares, err := parseCPURequest(vm.Spec.Config.Resources.CPU.Request)
			if err != nil {
				return nil, fmt.Errorf("invalid CPU request: %w", err)
			}
			config.CPUShares = cpuShares
		} else {
			config.CPUShares = 1024 // Default
		}

		// Parse memory request
		if vm.Spec.Config.Resources.Memory.Request != "" {
			memoryMB, err := parseMemoryRequest(vm.Spec.Config.Resources.Memory.Request)
			if err != nil {
				return nil, fmt.Errorf("invalid memory request: %w", err)
			}
			config.MemoryMB = memoryMB
		} else {
			config.MemoryMB = 512 // Default
		}

		// Parse disk size
		if vm.Spec.Config.Resources.Disk.Request != "" {
			diskGB, err := parseDiskRequest(vm.Spec.Config.Resources.Disk.Request)
			if err != nil {
				return nil, fmt.Errorf("invalid disk request: %w", err)
			}
			config.DiskSizeGB = diskGB
		} else {
			config.DiskSizeGB = 10 // Default
		}

		// Set command and args
		if len(vm.Spec.Config.Command) > 0 {
			config.Command = vm.Spec.Config.Command[0]
			if len(vm.Spec.Config.Command) > 1 {
				config.Args = vm.Spec.Config.Command[1:]
			}
		}
		if len(vm.Spec.Config.Args) > 0 {
			config.Args = append(config.Args, vm.Spec.Config.Args...)
		}
	} else {
		// Use defaults
		config.CPUShares = 1024
		config.MemoryMB = 512
		config.DiskSizeGB = 10
	}

	return config, nil
}

func (r *VirtualMachineReconciler) updateConditions(conditions []novacronv1.VirtualMachineCondition, 
	condType novacronv1.VirtualMachineConditionType, status metav1.ConditionStatus, 
	reason, message string, now metav1.Time) []novacronv1.VirtualMachineCondition {

	// Find existing condition
	for i, condition := range conditions {
		if condition.Type == condType {
			if condition.Status != status {
				conditions[i].Status = status
				conditions[i].LastTransitionTime = now
			}
			conditions[i].Reason = reason
			conditions[i].Message = message
			return conditions
		}
	}

	// Add new condition
	newCondition := novacronv1.VirtualMachineCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	return append(conditions, newCondition)
}

// SetupWithManager sets up the controller with the Manager
func (r *VirtualMachineReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.VirtualMachine{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		WithEventFilter(predicate.Or(predicate.GenerationChangedPredicate{}, predicate.LabelChangedPredicate{})).
		Complete(r)
}

// Helper types and functions

type VMConfigInternal struct {
	CPUShares  int
	MemoryMB   int
	DiskSizeGB int
	Command    string
	Args       []string
}

func containsFinalizer(finalizers []string, finalizer string) bool {
	for _, f := range finalizers {
		if f == finalizer {
			return true
		}
	}
	return false
}

func removeFinalizer(finalizers []string, finalizer string) []string {
	var result []string
	for _, f := range finalizers {
		if f != finalizer {
			result = append(result, f)
		}
	}
	return result
}

func parseCPURequest(cpu string) (int, error) {
	// Simple CPU parsing - in production, use resource.Quantity
	// For now, assume CPU is in millicores (e.g., "500m" = 500 millicores)
	if cpu == "" {
		return 1024, nil
	}
	
	// Simplified parsing - convert to CPU shares
	// 1 CPU = 1024 shares, 500m = 512 shares
	if cpu == "1" || cpu == "1000m" {
		return 1024, nil
	}
	if cpu == "500m" {
		return 512, nil
	}
	
	return 1024, nil // Default fallback
}

func parseMemoryRequest(memory string) (int, error) {
	// Simple memory parsing - in production, use resource.Quantity
	// For now, assume memory is in MB or GB
	if memory == "" {
		return 512, nil
	}
	
	// Simplified parsing
	if memory == "1Gi" || memory == "1G" {
		return 1024, nil
	}
	if memory == "512Mi" || memory == "512M" {
		return 512, nil
	}
	
	return 512, nil // Default fallback
}

func parseDiskRequest(disk string) (int, error) {
	// Simple disk parsing - in production, use resource.Quantity
	if disk == "" {
		return 10, nil
	}
	
	// Simplified parsing
	if disk == "20Gi" || disk == "20G" {
		return 20, nil
	}
	if disk == "10Gi" || disk == "10G" {
		return 10, nil
	}
	
	return 10, nil // Default fallback
}