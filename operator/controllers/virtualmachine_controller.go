/*
Copyright 2024 NovaCron.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
*/

package controllers

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	
	novacronv1alpha1 "github.com/novacron/operator/api/v1alpha1"
	"github.com/novacron/operator/pkg/provider"
)

const (
	vmFinalizer = "virtualmachine.novacron.io/finalizer"
)

// VirtualMachineReconciler reconciles a VirtualMachine object
type VirtualMachineReconciler struct {
	client.Client
	Scheme    *runtime.Scheme
	Log       logr.Logger
	VMManager provider.VMManager
}

// +kubebuilder:rbac:groups=novacron.io,resources=virtualmachines,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=novacron.io,resources=virtualmachines/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=novacron.io,resources=virtualmachines/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile handles VirtualMachine resource changes
func (r *VirtualMachineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("virtualmachine", req.NamespacedName)

	// Fetch the VirtualMachine instance
	vm := &novacronv1alpha1.VirtualMachine{}
	if err := r.Get(ctx, req.NamespacedName, vm); err != nil {
		if errors.IsNotFound(err) {
			// Object not found, could have been deleted
			log.Info("VirtualMachine resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get VirtualMachine")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if !vm.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, vm, log)
	}

	// Add finalizer if not present
	if !controllerutil.ContainsFinalizer(vm, vmFinalizer) {
		log.Info("Adding finalizer to VirtualMachine")
		controllerutil.AddFinalizer(vm, vmFinalizer)
		if err := r.Update(ctx, vm); err != nil {
			log.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Reconcile VM state based on current phase
	switch vm.Status.Phase {
	case "":
		// New VM, need to create
		return r.handleCreate(ctx, vm, log)
	case novacronv1alpha1.VMPending:
		// VM is being created, check status
		return r.handlePending(ctx, vm, log)
	case novacronv1alpha1.VMRunning:
		// VM is running, ensure desired state
		return r.handleRunning(ctx, vm, log)
	case novacronv1alpha1.VMMigrating:
		// VM is migrating, check migration status
		return r.handleMigrating(ctx, vm, log)
	case novacronv1alpha1.VMStopped:
		// VM is stopped, check if should be started
		return r.handleStopped(ctx, vm, log)
	case novacronv1alpha1.VMFailed:
		// VM failed, attempt recovery
		return r.handleFailed(ctx, vm, log)
	default:
		log.Error(fmt.Errorf("unknown phase: %s", vm.Status.Phase), "Unknown VM phase")
		return ctrl.Result{}, fmt.Errorf("unknown phase: %s", vm.Status.Phase)
	}
}

// handleCreate handles VM creation
func (r *VirtualMachineReconciler) handleCreate(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Creating new VirtualMachine")

	// Select node for VM placement
	node, err := r.selectNode(ctx, vm)
	if err != nil {
		log.Error(err, "Failed to select node for VM")
		// Update status to Failed
		vm.Status.Phase = novacronv1alpha1.VMFailed
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "NodeSelectionFailed", err.Error())
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
		}
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	log.Info("Selected node for VM", "node", node)

	// Create VM on selected node
	vmID, err := r.VMManager.CreateVM(ctx, node, vm.Spec)
	if err != nil {
		log.Error(err, "Failed to create VM")
		vm.Status.Phase = novacronv1alpha1.VMFailed
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "CreateFailed", err.Error())
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
		}
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	// Update status
	vm.Status.Phase = novacronv1alpha1.VMPending
	vm.Status.NodeName = node
	vm.Status.VMID = vmID
	r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Creating", "VM is being created")

	if err := r.Status().Update(ctx, vm); err != nil {
		log.Error(err, "Failed to update VM status")
		return ctrl.Result{}, err
	}

	log.Info("VM creation initiated", "vmID", vmID, "node", node)

	// Requeue to check status
	return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}

// handlePending handles pending VM
func (r *VirtualMachineReconciler) handlePending(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Checking pending VM status")

	// Check VM status from provider
	status, err := r.VMManager.GetVMStatus(ctx, vm.Status.VMID)
	if err != nil {
		log.Error(err, "Failed to get VM status")
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	switch status.State {
	case "running":
		// VM is now running
		vm.Status.Phase = novacronv1alpha1.VMRunning
		vm.Status.IPAddresses = status.IPAddresses
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionTrue, "Running", "VM is running")
		
		// Set LiveMigratable condition based on policy
		if vm.Spec.MigrationPolicy != nil && vm.Spec.MigrationPolicy.AllowLiveMigration {
			r.setCondition(vm, novacronv1alpha1.VMLiveMigratable, corev1.ConditionTrue, "MigrationEnabled", "VM can be live migrated")
		} else {
			r.setCondition(vm, novacronv1alpha1.VMLiveMigratable, corev1.ConditionFalse, "MigrationDisabled", "Live migration is disabled")
		}

		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
			return ctrl.Result{}, err
		}
		log.Info("VM is now running", "ipAddresses", status.IPAddresses)
		return ctrl.Result{RequeueAfter: 60 * time.Second}, nil

	case "failed":
		// VM creation failed
		vm.Status.Phase = novacronv1alpha1.VMFailed
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Failed", status.Message)
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
			return ctrl.Result{}, err
		}
		log.Error(fmt.Errorf(status.Message), "VM creation failed")
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil

	default:
		// Still pending
		log.Info("VM still pending", "state", status.State)
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}
}

// handleRunning handles running VM
func (r *VirtualMachineReconciler) handleRunning(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Handling running VM")

	// Check if VM should be stopped
	if !vm.Spec.Running {
		log.Info("Stopping VM as per spec")
		if err := r.VMManager.StopVM(ctx, vm.Status.VMID); err != nil {
			log.Error(err, "Failed to stop VM")
			return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
		}
		vm.Status.Phase = novacronv1alpha1.VMStopped
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Stopped", "VM is stopped")
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: 60 * time.Second}, nil
	}

	// Update resource usage
	usage, err := r.VMManager.GetVMResourceUsage(ctx, vm.Status.VMID)
	if err != nil {
		log.Error(err, "Failed to get VM resource usage")
		// Non-critical error, continue
	} else {
		vm.Status.ResourceUsage = novacronv1alpha1.ResourceUsage{
			CPU: novacronv1alpha1.ResourceMetric{
				Used:       usage.CPUUsed,
				Available:  usage.CPUAvailable,
				Percentage: usage.CPUPercentage,
			},
			Memory: novacronv1alpha1.ResourceMetric{
				Used:       usage.MemoryUsed,
				Available:  usage.MemoryAvailable,
				Percentage: usage.MemoryPercentage,
			},
			Disk: novacronv1alpha1.ResourceMetric{
				Used:       usage.DiskUsed,
				Available:  usage.DiskAvailable,
				Percentage: usage.DiskPercentage,
			},
		}
	}

	// Check health
	healthy, err := r.VMManager.CheckVMHealth(ctx, vm.Status.VMID)
	if err != nil || !healthy {
		log.Warn("VM health check failed", "error", err, "healthy", healthy)
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Unhealthy", "VM health check failed")
	} else {
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionTrue, "Healthy", "VM is healthy")
	}

	// Update last updated timestamp
	vm.Status.LastUpdated = metav1.Now()

	if err := r.Status().Update(ctx, vm); err != nil {
		log.Error(err, "Failed to update VM status")
		return ctrl.Result{}, err
	}

	// Regular reconciliation interval
	return ctrl.Result{RequeueAfter: 60 * time.Second}, nil
}

// handleMigrating handles migrating VM
func (r *VirtualMachineReconciler) handleMigrating(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Checking migration status")

	if vm.Status.Migration == nil {
		log.Error(fmt.Errorf("migration status is nil"), "Invalid migration state")
		vm.Status.Phase = novacronv1alpha1.VMFailed
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
		}
		return ctrl.Result{}, nil
	}

	// Check migration progress
	progress, err := r.VMManager.GetMigrationProgress(ctx, vm.Status.VMID)
	if err != nil {
		log.Error(err, "Failed to get migration progress")
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}

	vm.Status.Migration.Progress = progress.Percentage
	vm.Status.Migration.State = progress.State

	switch progress.State {
	case "completed":
		// Migration completed
		log.Info("Migration completed successfully")
		vm.Status.Phase = novacronv1alpha1.VMRunning
		vm.Status.NodeName = vm.Status.Migration.TargetNode
		vm.Status.Migration = nil
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionTrue, "Running", "VM is running after migration")

	case "failed":
		// Migration failed
		log.Error(fmt.Errorf(progress.Error), "Migration failed")
		vm.Status.Phase = novacronv1alpha1.VMFailed
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "MigrationFailed", progress.Error)

	default:
		// Still migrating
		log.Info("Migration in progress", "progress", progress.Percentage)
	}

	if err := r.Status().Update(ctx, vm); err != nil {
		log.Error(err, "Failed to update VM status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

// handleStopped handles stopped VM
func (r *VirtualMachineReconciler) handleStopped(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Handling stopped VM")

	// Check if VM should be started
	if vm.Spec.Running {
		log.Info("Starting VM as per spec")
		if err := r.VMManager.StartVM(ctx, vm.Status.VMID); err != nil {
			log.Error(err, "Failed to start VM")
			return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
		}
		vm.Status.Phase = novacronv1alpha1.VMPending
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Starting", "VM is starting")
		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	// VM should remain stopped
	return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}

// handleFailed handles failed VM
func (r *VirtualMachineReconciler) handleFailed(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Handling failed VM")

	// Check if auto-recovery is enabled
	if vm.Spec.Running {
		log.Info("Attempting to recover failed VM")
		
		// Delete failed VM
		if err := r.VMManager.DeleteVM(ctx, vm.Status.VMID); err != nil {
			log.Error(err, "Failed to delete failed VM")
			// Continue anyway
		}

		// Reset status to trigger recreation
		vm.Status.Phase = ""
		vm.Status.VMID = ""
		vm.Status.NodeName = ""
		vm.Status.IPAddresses = nil
		r.setCondition(vm, novacronv1alpha1.VMReady, corev1.ConditionFalse, "Recovering", "Attempting to recover VM")

		if err := r.Status().Update(ctx, vm); err != nil {
			log.Error(err, "Failed to update VM status")
			return ctrl.Result{}, err
		}

		return ctrl.Result{Requeue: true}, nil
	}

	// VM should remain failed
	return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}

// handleDeletion handles VM deletion
func (r *VirtualMachineReconciler) handleDeletion(ctx context.Context, vm *novacronv1alpha1.VirtualMachine, log logr.Logger) (ctrl.Result, error) {
	log.Info("Handling VirtualMachine deletion")

	// Check if finalizer is present
	if controllerutil.ContainsFinalizer(vm, vmFinalizer) {
		// Delete the VM if it exists
		if vm.Status.VMID != "" {
			log.Info("Deleting VM", "vmID", vm.Status.VMID)
			if err := r.VMManager.DeleteVM(ctx, vm.Status.VMID); err != nil {
				log.Error(err, "Failed to delete VM")
				// Don't block deletion
			}
		}

		// Remove finalizer
		log.Info("Removing finalizer")
		controllerutil.RemoveFinalizer(vm, vmFinalizer)
		if err := r.Update(ctx, vm); err != nil {
			log.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

// selectNode selects a node for VM placement
func (r *VirtualMachineReconciler) selectNode(ctx context.Context, vm *novacronv1alpha1.VirtualMachine) (string, error) {
	// List all nodes
	nodes := &corev1.NodeList{}
	if err := r.List(ctx, nodes); err != nil {
		return "", fmt.Errorf("failed to list nodes: %w", err)
	}

	// Filter nodes based on selector
	var candidateNodes []corev1.Node
	for _, node := range nodes.Items {
		// Skip nodes that are not ready
		if !r.isNodeReady(&node) {
			continue
		}

		// Check node selector
		if !r.matchesNodeSelector(&node, vm.Spec.Template.Spec.NodeSelector) {
			continue
		}

		// Check if node has capacity
		if !r.hasCapacity(&node, &vm.Spec.Template.Spec.Resources) {
			continue
		}

		candidateNodes = append(candidateNodes, node)
	}

	if len(candidateNodes) == 0 {
		return "", fmt.Errorf("no suitable nodes found")
	}

	// Simple selection: pick the first suitable node
	// TODO: Implement more sophisticated scheduling algorithms
	return candidateNodes[0].Name, nil
}

// isNodeReady checks if a node is ready
func (r *VirtualMachineReconciler) isNodeReady(node *corev1.Node) bool {
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// matchesNodeSelector checks if node matches selector
func (r *VirtualMachineReconciler) matchesNodeSelector(node *corev1.Node, selector map[string]string) bool {
	if selector == nil {
		return true
	}

	for key, value := range selector {
		if nodeValue, ok := node.Labels[key]; !ok || nodeValue != value {
			return false
		}
	}

	return true
}

// hasCapacity checks if node has capacity for VM
func (r *VirtualMachineReconciler) hasCapacity(node *corev1.Node, resources *novacronv1alpha1.Resources) bool {
	// TODO: Implement proper capacity checking
	// For now, always return true
	return true
}

// setCondition sets a condition on the VM
func (r *VirtualMachineReconciler) setCondition(vm *novacronv1alpha1.VirtualMachine, condType novacronv1alpha1.VMConditionType, status corev1.ConditionStatus, reason, message string) {
	now := metav1.Now()
	
	// Find existing condition
	var condition *novacronv1alpha1.VMCondition
	for i := range vm.Status.Conditions {
		if vm.Status.Conditions[i].Type == condType {
			condition = &vm.Status.Conditions[i]
			break
		}
	}

	if condition == nil {
		// Add new condition
		vm.Status.Conditions = append(vm.Status.Conditions, novacronv1alpha1.VMCondition{
			Type:               condType,
			Status:             status,
			LastProbeTime:      now,
			LastTransitionTime: now,
			Reason:             reason,
			Message:            message,
		})
	} else {
		// Update existing condition
		if condition.Status != status {
			condition.LastTransitionTime = now
		}
		condition.Status = status
		condition.LastProbeTime = now
		condition.Reason = reason
		condition.Message = message
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *VirtualMachineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1alpha1.VirtualMachine{}).
		WithEventFilter(predicate.GenerationChangedPredicate{}).
		Complete(r)
}