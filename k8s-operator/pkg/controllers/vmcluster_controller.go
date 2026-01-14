package controllers

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/log"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/novacron"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VMClusterReconciler reconciles a VMCluster object
type VMClusterReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	NovaCronClient *novacron.Client
	Recorder       record.EventRecorder
}

const (
	VMClusterFinalizerName = "novacron.io/vmcluster"
	VMClusterRequeueInterval = 30 * time.Second
)

//+kubebuilder:rbac:groups=novacron.io,resources=vmclusters,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=vmclusters/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=vmclusters/finalizers,verbs=update

// Reconcile implements the reconciliation logic for VMCluster resources
func (r *VMClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("vmcluster", req.NamespacedName)

	// Fetch the VMCluster resource
	cluster := &novacronv1.VMCluster{}
	if err := r.Get(ctx, req.NamespacedName, cluster); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("VMCluster resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get VMCluster")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if cluster.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, cluster)
	}

	// Add finalizer if not present
	if !containsFinalizer(cluster.Finalizers, VMClusterFinalizerName) {
		cluster.Finalizers = append(cluster.Finalizers, VMClusterFinalizerName)
		if err := r.Update(ctx, cluster); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update observed generation
	if cluster.Status.ObservedGeneration != cluster.Generation {
		cluster.Status.ObservedGeneration = cluster.Generation
		if err := r.Status().Update(ctx, cluster); err != nil {
			logger.Error(err, "Failed to update observed generation")
			return ctrl.Result{}, err
		}
	}

	// Handle cluster lifecycle
	return r.handleClusterLifecycle(ctx, cluster)
}

func (r *VMClusterReconciler) handleClusterLifecycle(ctx context.Context, cluster *novacronv1.VMCluster) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get desired replica count
	desiredReplicas := int32(1)
	if cluster.Spec.Replicas != nil {
		desiredReplicas = *cluster.Spec.Replicas
	}

	// List existing VMs for this cluster
	vmList := &novacronv1.VirtualMachineList{}
	listOpts := []client.ListOption{
		client.InNamespace(cluster.Namespace),
		client.MatchingLabels{
			"novacron.io/cluster":    cluster.Name,
			"novacron.io/managed-by": "vmcluster-controller",
		},
	}

	if err := r.List(ctx, vmList, listOpts...); err != nil {
		logger.Error(err, "Failed to list VMs for cluster")
		return ctrl.Result{}, err
	}

	currentReplicas := int32(len(vmList.Items))
	
	// Count ready VMs
	var readyReplicas int32
	var availableReplicas int32
	for _, vm := range vmList.Items {
		if vm.Status.Phase == novacronv1.VMPhaseRunning {
			readyReplicas++
			if r.isVMAvailable(&vm) {
				availableReplicas++
			}
		}
	}

	logger.Info("Cluster status", 
		"desired", desiredReplicas, 
		"current", currentReplicas, 
		"ready", readyReplicas, 
		"available", availableReplicas)

	// Handle scaling
	var err error
	if currentReplicas < desiredReplicas {
		err = r.scaleUp(ctx, cluster, vmList.Items, desiredReplicas-currentReplicas)
	} else if currentReplicas > desiredReplicas {
		err = r.scaleDown(ctx, cluster, vmList.Items, currentReplicas-desiredReplicas)
	}

	if err != nil {
		logger.Error(err, "Failed to scale cluster")
		return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, err
	}

	// Update status
	cluster.Status.Replicas = currentReplicas
	cluster.Status.ReadyReplicas = readyReplicas
	cluster.Status.AvailableReplicas = availableReplicas

	// Update conditions
	now := metav1.Now()
	if currentReplicas == desiredReplicas && readyReplicas == desiredReplicas {
		cluster.Status.Conditions = r.updateClusterConditions(cluster.Status.Conditions,
			novacronv1.VMClusterReady, metav1.ConditionTrue, "AllReplicasReady",
			"All replicas are ready", now)
		cluster.Status.Conditions = r.updateClusterConditions(cluster.Status.Conditions,
			novacronv1.VMClusterProgressing, metav1.ConditionFalse, "ScalingComplete",
			"Scaling operations complete", now)
	} else {
		cluster.Status.Conditions = r.updateClusterConditions(cluster.Status.Conditions,
			novacronv1.VMClusterReady, metav1.ConditionFalse, "ReplicasNotReady",
			fmt.Sprintf("Ready: %d/%d", readyReplicas, desiredReplicas), now)
		cluster.Status.Conditions = r.updateClusterConditions(cluster.Status.Conditions,
			novacronv1.VMClusterProgressing, metav1.ConditionTrue, "Scaling",
			fmt.Sprintf("Scaling to %d replicas", desiredReplicas), now)
	}

	if err := r.Status().Update(ctx, cluster); err != nil {
		logger.Error(err, "Failed to update VMCluster status")
		return ctrl.Result{}, err
	}

	// Handle autoscaling if enabled
	if cluster.Spec.AutoScaling != nil && cluster.Spec.AutoScaling.Enabled {
		return r.handleAutoScaling(ctx, cluster)
	}

	return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, nil
}

func (r *VMClusterReconciler) scaleUp(ctx context.Context, cluster *novacronv1.VMCluster, 
	existingVMs []novacronv1.VirtualMachine, count int32) error {
	
	logger := log.FromContext(ctx)

	// Get template
	template := &novacronv1.VMTemplate{}
	templateKey := client.ObjectKey{
		Name:      cluster.Spec.Template.Name,
		Namespace: cluster.Spec.Template.Namespace,
	}
	if templateKey.Namespace == "" {
		templateKey.Namespace = cluster.Namespace
	}

	if err := r.Get(ctx, templateKey, template); err != nil {
		logger.Error(err, "Failed to get VM template")
		r.Recorder.Event(cluster, "Warning", "TemplateNotFound", 
			fmt.Sprintf("Template %s not found", cluster.Spec.Template.Name))
		return err
	}

	if !template.Status.Valid {
		logger.Error(nil, "Template validation failed")
		r.Recorder.Event(cluster, "Warning", "InvalidTemplate", 
			fmt.Sprintf("Template %s is invalid", template.Name))
		return fmt.Errorf("template %s is invalid", template.Name)
	}

	// Create new VMs
	for i := int32(0); i < count; i++ {
		vmName := fmt.Sprintf("%s-%s", cluster.Name, r.generateVMSuffix(existingVMs, i))
		
		vm := &novacronv1.VirtualMachine{
			ObjectMeta: metav1.ObjectMeta{
				Name:      vmName,
				Namespace: cluster.Namespace,
				Labels: map[string]string{
					"novacron.io/cluster":    cluster.Name,
					"novacron.io/template":   template.Name,
					"novacron.io/managed-by": "vmcluster-controller",
				},
				OwnerReferences: []metav1.OwnerReference{
					*metav1.NewControllerRef(cluster, novacronv1.SchemeGroupVersion.WithKind("VMCluster")),
				},
			},
			Spec: novacronv1.VirtualMachineSpec{
				Name:   vmName,
				Config: &template.Spec.Config,
				NodeSelector: cluster.Spec.Template.Namespace,
				RestartPolicy: novacronv1.VMRestartPolicyAlways,
			},
		}

		// Apply template defaults
		if template.Spec.DefaultNodeSelector != nil {
			if vm.Spec.NodeSelector == nil {
				vm.Spec.NodeSelector = make(map[string]string)
			}
			for k, v := range template.Spec.DefaultNodeSelector {
				if _, exists := vm.Spec.NodeSelector[k]; !exists {
					vm.Spec.NodeSelector[k] = v
				}
			}
		}

		if template.Spec.DefaultAffinity != nil {
			vm.Spec.Affinity = template.Spec.DefaultAffinity
		}

		if err := r.Create(ctx, vm); err != nil {
			logger.Error(err, "Failed to create VM", "vmName", vmName)
			r.Recorder.Event(cluster, "Warning", "VMCreationFailed", 
				fmt.Sprintf("Failed to create VM %s: %v", vmName, err))
			return err
		}

		logger.Info("Created VM for cluster", "vmName", vmName, "cluster", cluster.Name)
		r.Recorder.Event(cluster, "Normal", "VMCreated", 
			fmt.Sprintf("Created VM %s", vmName))
	}

	return nil
}

func (r *VMClusterReconciler) scaleDown(ctx context.Context, cluster *novacronv1.VMCluster, 
	existingVMs []novacronv1.VirtualMachine, count int32) error {
	
	logger := log.FromContext(ctx)

	// Select VMs to delete (prefer non-ready VMs first)
	var vmsToDelete []novacronv1.VirtualMachine
	
	// First, add non-ready VMs
	for _, vm := range existingVMs {
		if len(vmsToDelete) >= int(count) {
			break
		}
		if vm.Status.Phase != novacronv1.VMPhaseRunning {
			vmsToDelete = append(vmsToDelete, vm)
		}
	}

	// If we need more, add ready VMs
	for _, vm := range existingVMs {
		if len(vmsToDelete) >= int(count) {
			break
		}
		if vm.Status.Phase == novacronv1.VMPhaseRunning {
			// Check if already in deletion list
			found := false
			for _, deleteVM := range vmsToDelete {
				if deleteVM.Name == vm.Name {
					found = true
					break
				}
			}
			if !found {
				vmsToDelete = append(vmsToDelete, vm)
			}
		}
	}

	// Delete selected VMs
	for _, vm := range vmsToDelete {
		if err := r.Delete(ctx, &vm); err != nil {
			logger.Error(err, "Failed to delete VM", "vmName", vm.Name)
			r.Recorder.Event(cluster, "Warning", "VMDeletionFailed", 
				fmt.Sprintf("Failed to delete VM %s: %v", vm.Name, err))
			return err
		}

		logger.Info("Deleted VM from cluster", "vmName", vm.Name, "cluster", cluster.Name)
		r.Recorder.Event(cluster, "Normal", "VMDeleted", 
			fmt.Sprintf("Deleted VM %s", vm.Name))
	}

	return nil
}

func (r *VMClusterReconciler) handleAutoScaling(ctx context.Context, cluster *novacronv1.VMCluster) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get current metrics for all VMs in the cluster
	vmList := &novacronv1.VirtualMachineList{}
	listOpts := []client.ListOption{
		client.InNamespace(cluster.Namespace),
		client.MatchingLabels{
			"novacron.io/cluster":    cluster.Name,
			"novacron.io/managed-by": "vmcluster-controller",
		},
	}

	if err := r.List(ctx, vmList, listOpts...); err != nil {
		logger.Error(err, "Failed to list VMs for autoscaling")
		return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, err
	}

	// Calculate average resource utilization
	var totalCPU, totalMemory float64
	var runningVMs int32

	for _, vm := range vmList.Items {
		if vm.Status.Phase == novacronv1.VMPhaseRunning && vm.Status.ResourceUsage != nil {
			if cpuUsage := parseUsagePercentage(vm.Status.ResourceUsage.CPU); cpuUsage > 0 {
				totalCPU += cpuUsage
				runningVMs++
			}
			if memUsage := parseUsagePercentage(vm.Status.ResourceUsage.Memory); memUsage > 0 {
				totalMemory += memUsage
			}
		}
	}

	if runningVMs == 0 {
		return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, nil
	}

	avgCPU := totalCPU / float64(runningVMs)
	avgMemory := totalMemory / float64(runningVMs)

	// Determine if scaling is needed
	autoScale := cluster.Spec.AutoScaling
	scaleUp := false
	scaleDown := false

	if autoScale.TargetCPUUtilization != nil {
		target := float64(*autoScale.TargetCPUUtilization)
		if avgCPU > target+10 { // Scale up if 10% above target
			scaleUp = true
		} else if avgCPU < target-10 && runningVMs > autoScale.MinReplicas { // Scale down if 10% below target
			scaleDown = true
		}
	}

	if autoScale.TargetMemoryUtilization != nil {
		target := float64(*autoScale.TargetMemoryUtilization)
		if avgMemory > target+10 {
			scaleUp = true
		} else if avgMemory < target-10 && runningVMs > autoScale.MinReplicas {
			scaleDown = true
		}
	}

	// Apply scaling
	currentReplicas := cluster.Status.Replicas
	var newReplicas int32

	if scaleUp && currentReplicas < autoScale.MaxReplicas {
		newReplicas = currentReplicas + 1
		if newReplicas > autoScale.MaxReplicas {
			newReplicas = autoScale.MaxReplicas
		}
	} else if scaleDown && currentReplicas > autoScale.MinReplicas {
		newReplicas = currentReplicas - 1
		if newReplicas < autoScale.MinReplicas {
			newReplicas = autoScale.MinReplicas
		}
	} else {
		newReplicas = currentReplicas
	}

	if newReplicas != currentReplicas {
		logger.Info("Autoscaling cluster", 
			"from", currentReplicas, 
			"to", newReplicas, 
			"avgCPU", avgCPU, 
			"avgMemory", avgMemory)

		cluster.Spec.Replicas = &newReplicas
		if err := r.Update(ctx, cluster); err != nil {
			logger.Error(err, "Failed to update cluster replicas for autoscaling")
			return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, err
		}

		r.Recorder.Event(cluster, "Normal", "AutoScaling", 
			fmt.Sprintf("Autoscaled from %d to %d replicas (CPU: %.1f%%, Memory: %.1f%%)", 
				currentReplicas, newReplicas, avgCPU, avgMemory))
	}

	return ctrl.Result{RequeueAfter: VMClusterRequeueInterval}, nil
}

func (r *VMClusterReconciler) handleDeletion(ctx context.Context, cluster *novacronv1.VMCluster) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(cluster.Finalizers, VMClusterFinalizerName) {
		// Delete all VMs in the cluster
		vmList := &novacronv1.VirtualMachineList{}
		listOpts := []client.ListOption{
			client.InNamespace(cluster.Namespace),
			client.MatchingLabels{
				"novacron.io/cluster":    cluster.Name,
				"novacron.io/managed-by": "vmcluster-controller",
			},
		}

		if err := r.List(ctx, vmList, listOpts...); err != nil {
			logger.Error(err, "Failed to list VMs for cluster deletion")
			return ctrl.Result{}, err
		}

		// Delete all VMs
		for _, vm := range vmList.Items {
			if err := r.Delete(ctx, &vm); err != nil {
				if !client.IgnoreNotFound(err) == nil {
					logger.Error(err, "Failed to delete VM during cluster cleanup", "vmName", vm.Name)
					return ctrl.Result{}, err
				}
			} else {
				logger.Info("Deleted VM during cluster cleanup", "vmName", vm.Name)
			}
		}

		// Wait for all VMs to be deleted
		if len(vmList.Items) > 0 {
			logger.Info("Waiting for VMs to be deleted", "count", len(vmList.Items))
			return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
		}

		// Remove finalizer
		cluster.Finalizers = removeFinalizer(cluster.Finalizers, VMClusterFinalizerName)
		if err := r.Update(ctx, cluster); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}

		logger.Info("VMCluster deleted successfully")
		r.Recorder.Event(cluster, "Normal", "ClusterDeleted", "Cluster deleted successfully")
	}

	return ctrl.Result{}, nil
}

func (r *VMClusterReconciler) updateClusterConditions(conditions []novacronv1.VMClusterCondition,
	condType novacronv1.VMClusterConditionType, status metav1.ConditionStatus,
	reason, message string, now metav1.Time) []novacronv1.VMClusterCondition {

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
	newCondition := novacronv1.VMClusterCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	return append(conditions, newCondition)
}

func (r *VMClusterReconciler) isVMAvailable(vm *novacronv1.VirtualMachine) bool {
	// VM is available if it's running and ready for a reasonable amount of time
	for _, condition := range vm.Status.Conditions {
		if condition.Type == novacronv1.VirtualMachineReady &&
			condition.Status == metav1.ConditionTrue &&
			time.Since(condition.LastTransitionTime.Time) > 30*time.Second {
			return true
		}
	}
	return false
}

func (r *VMClusterReconciler) generateVMSuffix(existingVMs []novacronv1.VirtualMachine, offset int32) string {
	// Generate a unique suffix for the VM name
	suffix := offset
	for {
		found := false
		for _, vm := range existingVMs {
			expectedName := fmt.Sprintf("-%d", suffix)
			if vm.Name[len(vm.Name)-len(expectedName):] == expectedName {
				found = true
				break
			}
		}
		if !found {
			break
		}
		suffix++
	}
	return strconv.Itoa(int(suffix))
}

// SetupWithManager sets up the controller with the Manager
func (r *VMClusterReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.VMCluster{}).
		Owns(&novacronv1.VirtualMachine{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		Complete(r)
}

// Helper functions

func parseUsagePercentage(usage string) float64 {
	// Parse usage string like "75.5%" to float64
	if usage == "" {
		return 0
	}
	
	// Remove '%' if present
	if usage[len(usage)-1] == '%' {
		usage = usage[:len(usage)-1]
	}
	
	if val, err := strconv.ParseFloat(usage, 64); err == nil {
		return val
	}
	
	return 0
}