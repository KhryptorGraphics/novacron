package controllers

import (
	"context"
	"fmt"

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

// VMTemplateReconciler reconciles a VMTemplate object
type VMTemplateReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	NovaCronClient *novacron.Client
	Recorder       record.EventRecorder
}

const (
	VMTemplateFinalizerName = "novacron.io/vmtemplate"
)

//+kubebuilder:rbac:groups=novacron.io,resources=vmtemplates,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=vmtemplates/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=vmtemplates/finalizers,verbs=update

// Reconcile implements the reconciliation logic for VMTemplate resources
func (r *VMTemplateReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("vmtemplate", req.NamespacedName)

	// Fetch the VMTemplate resource
	template := &novacronv1.VMTemplate{}
	if err := r.Get(ctx, req.NamespacedName, template); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("VMTemplate resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get VMTemplate")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if template.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, template)
	}

	// Add finalizer if not present
	if !containsFinalizer(template.Finalizers, VMTemplateFinalizerName) {
		template.Finalizers = append(template.Finalizers, VMTemplateFinalizerName)
		if err := r.Update(ctx, template); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Validate template
	return r.validateTemplate(ctx, template)
}

func (r *VMTemplateReconciler) validateTemplate(ctx context.Context, template *novacronv1.VMTemplate) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var validationErrors []string
	
	// Validate required fields
	if template.Spec.Config.Resources.CPU.Request == "" && template.Spec.Config.Resources.CPU.Limit == "" {
		validationErrors = append(validationErrors, "CPU resource must be specified")
	}
	
	if template.Spec.Config.Resources.Memory.Request == "" && template.Spec.Config.Resources.Memory.Limit == "" {
		validationErrors = append(validationErrors, "Memory resource must be specified")
	}

	if template.Spec.Config.Image == "" && len(template.Spec.Config.Command) == 0 {
		validationErrors = append(validationErrors, "Either image or command must be specified")
	}

	// Validate parameters
	for _, param := range template.Spec.Parameters {
		if param.Name == "" {
			validationErrors = append(validationErrors, "Parameter name cannot be empty")
		}
		if param.Type == "" {
			validationErrors = append(validationErrors, fmt.Sprintf("Parameter %s must have a type", param.Name))
		}
		if param.Required && param.DefaultValue == nil {
			validationErrors = append(validationErrors, fmt.Sprintf("Required parameter %s must have a default value", param.Name))
		}
	}

	// Update status
	now := metav1.Now()
	template.Status.Valid = len(validationErrors) == 0
	template.Status.ValidationErrors = validationErrors
	template.Status.LastUpdated = &now

	// Count VMs created from this template
	vmList := &novacronv1.VirtualMachineList{}
	listOpts := []client.ListOption{
		client.InNamespace(template.Namespace),
		client.MatchingLabels{"novacron.io/template": template.Name},
	}
	
	if err := r.List(ctx, vmList, listOpts...); err != nil {
		logger.Error(err, "Failed to list VMs for template")
	} else {
		template.Status.VMCount = int32(len(vmList.Items))
	}

	if err := r.Status().Update(ctx, template); err != nil {
		logger.Error(err, "Failed to update VMTemplate status")
		return ctrl.Result{}, err
	}

	if !template.Status.Valid {
		r.Recorder.Event(template, "Warning", "ValidationFailed", 
			fmt.Sprintf("Template validation failed: %v", validationErrors))
		logger.Info("Template validation failed", "errors", validationErrors)
	} else {
		r.Recorder.Event(template, "Normal", "ValidationSucceeded", "Template validation passed")
		logger.Info("Template validation succeeded")
	}

	return ctrl.Result{}, nil
}

func (r *VMTemplateReconciler) handleDeletion(ctx context.Context, template *novacronv1.VMTemplate) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(template.Finalizers, VMTemplateFinalizerName) {
		// Check if there are any VMs using this template
		vmList := &novacronv1.VirtualMachineList{}
		listOpts := []client.ListOption{
			client.InNamespace(template.Namespace),
			client.MatchingLabels{"novacron.io/template": template.Name},
		}
		
		if err := r.List(ctx, vmList, listOpts...); err != nil {
			logger.Error(err, "Failed to list VMs for template deletion check")
			return ctrl.Result{}, err
		}

		if len(vmList.Items) > 0 {
			logger.Info("Cannot delete template, VMs still reference it", "vmCount", len(vmList.Items))
			r.Recorder.Event(template, "Warning", "DeleteBlocked", 
				fmt.Sprintf("Cannot delete template: %d VMs still reference it", len(vmList.Items)))
			// Don't remove finalizer yet
			return ctrl.Result{}, nil
		}

		// Remove finalizer
		template.Finalizers = removeFinalizer(template.Finalizers, VMTemplateFinalizerName)
		if err := r.Update(ctx, template); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}

		logger.Info("VMTemplate deleted successfully")
		r.Recorder.Event(template, "Normal", "TemplateDeleted", "Template deleted successfully")
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *VMTemplateReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.VMTemplate{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		Complete(r)
}