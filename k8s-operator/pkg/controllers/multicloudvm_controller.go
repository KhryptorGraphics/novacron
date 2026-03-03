package controllers

import (
	"context"
	"fmt"
	"strings"
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
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/providers"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MultiCloudVMReconciler reconciles a MultiCloudVM object
type MultiCloudVMReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	NovaCronClient *novacron.Client
	CloudProviders providers.CloudProviderManager
	Recorder       record.EventRecorder
}

const (
	MultiCloudVMFinalizerName = "novacron.io/multiCloudVM"
	MultiCloudRequeueInterval = 45 * time.Second
	ProviderAnnotationPrefix  = "novacron.io/provider-"
)

//+kubebuilder:rbac:groups=novacron.io,resources=multicloudvms,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=multicloudvms/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=multicloudvms/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch
//+kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch

// Reconcile implements the reconciliation logic for MultiCloudVM resources
func (r *MultiCloudVMReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("multicloudvm", req.NamespacedName)

	// Fetch the MultiCloudVM resource
	mcvm := &novacronv1.MultiCloudVM{}
	if err := r.Get(ctx, req.NamespacedName, mcvm); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("MultiCloudVM resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get MultiCloudVM")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if mcvm.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, mcvm)
	}

	// Add finalizer if not present
	if !containsFinalizer(mcvm.Finalizers, MultiCloudVMFinalizerName) {
		mcvm.Finalizers = append(mcvm.Finalizers, MultiCloudVMFinalizerName)
		if err := r.Update(ctx, mcvm); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update observed generation
	if mcvm.Status.ObservedGeneration != mcvm.Generation {
		mcvm.Status.ObservedGeneration = mcvm.Generation
		if err := r.Status().Update(ctx, mcvm); err != nil {
			logger.Error(err, "Failed to update observed generation")
			return ctrl.Result{}, err
		}
	}

	// Handle multi-cloud VM lifecycle
	return r.handleMultiCloudVMLifecycle(ctx, mcvm)
}

func (r *MultiCloudVMReconciler) handleMultiCloudVMLifecycle(ctx context.Context, mcvm *novacronv1.MultiCloudVM) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Validate deployment strategy
	if err := r.validateDeploymentStrategy(mcvm); err != nil {
		return r.updateStatus(ctx, mcvm, "Failed", err.Error())
	}

	// Get VM template
	template, err := r.getVMTemplate(ctx, mcvm)
	if err != nil {
		logger.Error(err, "Failed to get VM template")
		return r.updateStatus(ctx, mcvm, "Failed", fmt.Sprintf("Template error: %v", err))
	}

	// Determine optimal deployment strategy
	deploymentPlan, err := r.planDeployment(ctx, mcvm, template)
	if err != nil {
		logger.Error(err, "Failed to plan deployment")
		return r.updateStatus(ctx, mcvm, "Failed", fmt.Sprintf("Planning error: %v", err))
	}

	// Execute deployment across selected providers
	deploymentResults, err := r.executeDeployment(ctx, mcvm, deploymentPlan)
	if err != nil {
		logger.Error(err, "Failed to execute deployment")
		return r.updateStatus(ctx, mcvm, "Failed", fmt.Sprintf("Deployment error: %v", err))
	}

	// Update status with deployment results
	if err := r.updateDeploymentStatus(ctx, mcvm, deploymentResults); err != nil {
		logger.Error(err, "Failed to update deployment status")
		return ctrl.Result{}, err
	}

	// Handle cost optimization
	if mcvm.Spec.CostOptimization != nil {
		if err := r.optimizeCosts(ctx, mcvm, deploymentResults); err != nil {
			logger.V(1).Info("Cost optimization failed", "error", err)
			// Don't fail reconciliation for cost optimization errors
		}
	}

	// Handle migration if needed
	if needsMigration, reason := r.evaluateMigrationNeeds(ctx, mcvm); needsMigration {
		logger.Info("Migration needed", "reason", reason)
		if err := r.handleMigration(ctx, mcvm, reason); err != nil {
			logger.Error(err, "Migration failed")
			// Don't fail reconciliation, will retry
		}
	}

	logger.Info("MultiCloudVM reconciliation completed successfully")
	return ctrl.Result{RequeueAfter: MultiCloudRequeueInterval}, nil
}

func (r *MultiCloudVMReconciler) validateDeploymentStrategy(mcvm *novacronv1.MultiCloudVM) error {
	strategy := mcvm.Spec.DeploymentStrategy
	
	// Validate strategy type
	validTypes := []string{"active-passive", "active-active", "burst", "cost-optimized"}
	if !contains(validTypes, strategy.Type) {
		return fmt.Errorf("invalid deployment strategy type: %s", strategy.Type)
	}

	// Validate primary provider exists in providers list
	primaryFound := false
	for _, provider := range mcvm.Spec.Providers {
		if provider.Name == strategy.Primary {
			primaryFound = true
			break
		}
	}
	if !primaryFound {
		return fmt.Errorf("primary provider '%s' not found in providers list", strategy.Primary)
	}

	// Validate secondary providers
	for _, secondary := range strategy.Secondary {
		secondaryFound := false
		for _, provider := range mcvm.Spec.Providers {
			if provider.Name == secondary {
				secondaryFound = true
				break
			}
		}
		if !secondaryFound {
			return fmt.Errorf("secondary provider '%s' not found in providers list", secondary)
		}
	}

	return nil
}

func (r *MultiCloudVMReconciler) getVMTemplate(ctx context.Context, mcvm *novacronv1.MultiCloudVM) (*novacronv1.VMTemplate, error) {
	template := &novacronv1.VMTemplate{}
	templateKey := types.NamespacedName{
		Name:      mcvm.Spec.VMTemplate.Name,
		Namespace: mcvm.Spec.VMTemplate.Namespace,
	}
	if templateKey.Namespace == "" {
		templateKey.Namespace = mcvm.Namespace
	}

	if err := r.Get(ctx, templateKey, template); err != nil {
		return nil, fmt.Errorf("failed to get VM template %s: %w", templateKey, err)
	}

	return template, nil
}

func (r *MultiCloudVMReconciler) planDeployment(ctx context.Context, mcvm *novacronv1.MultiCloudVM, template *novacronv1.VMTemplate) (*DeploymentPlan, error) {
	plan := &DeploymentPlan{
		Strategy:    mcvm.Spec.DeploymentStrategy.Type,
		Primary:     mcvm.Spec.DeploymentStrategy.Primary,
		Secondary:   mcvm.Spec.DeploymentStrategy.Secondary,
		Template:    template,
		Deployments: make(map[string]*ProviderDeployment),
	}

	// Plan deployment for primary provider
	primaryProvider, err := r.getProvider(mcvm, mcvm.Spec.DeploymentStrategy.Primary)
	if err != nil {
		return nil, fmt.Errorf("failed to get primary provider: %w", err)
	}

	primaryDeployment, err := r.planProviderDeployment(ctx, primaryProvider, template, true)
	if err != nil {
		return nil, fmt.Errorf("failed to plan primary deployment: %w", err)
	}
	plan.Deployments[mcvm.Spec.DeploymentStrategy.Primary] = primaryDeployment

	// Plan deployment for secondary providers based on strategy
	switch mcvm.Spec.DeploymentStrategy.Type {
	case "active-active":
		// Deploy to all providers
		for _, secondary := range mcvm.Spec.DeploymentStrategy.Secondary {
			provider, err := r.getProvider(mcvm, secondary)
			if err != nil {
				return nil, fmt.Errorf("failed to get secondary provider %s: %w", secondary, err)
			}
			
			deployment, err := r.planProviderDeployment(ctx, provider, template, false)
			if err != nil {
				return nil, fmt.Errorf("failed to plan secondary deployment for %s: %w", secondary, err)
			}
			plan.Deployments[secondary] = deployment
		}
	case "active-passive":
		// Only prepare secondary providers, don't deploy yet
		for _, secondary := range mcvm.Spec.DeploymentStrategy.Secondary {
			provider, err := r.getProvider(mcvm, secondary)
			if err != nil {
				return nil, fmt.Errorf("failed to get secondary provider %s: %w", secondary, err)
			}
			
			deployment, err := r.planProviderDeployment(ctx, provider, template, false)
			if err != nil {
				return nil, fmt.Errorf("failed to plan secondary deployment for %s: %w", secondary, err)
			}
			deployment.Status = "standby"
			plan.Deployments[secondary] = deployment
		}
	case "burst", "cost-optimized":
		// Plan deployment based on current needs and cost analysis
		if err := r.planBurstDeployment(ctx, plan, mcvm, template); err != nil {
			return nil, fmt.Errorf("failed to plan burst deployment: %w", err)
		}
	}

	return plan, nil
}

func (r *MultiCloudVMReconciler) planProviderDeployment(ctx context.Context, provider *novacronv1.CloudProvider, template *novacronv1.VMTemplate, isPrimary bool) (*ProviderDeployment, error) {
	deployment := &ProviderDeployment{
		Provider:  provider.Name,
		Region:    provider.Region,
		IsPrimary: isPrimary,
		Status:    "planned",
		Resources: ResourceRequirements{
			CPU:     template.Spec.Config.Resources.CPU.Request,
			Memory:  template.Spec.Config.Resources.Memory.Request,
			Storage: template.Spec.Config.Resources.Disk.Request,
		},
	}

	// Calculate estimated cost
	cost, err := r.CloudProviders.EstimateCost(provider.Name, provider.Region, deployment.Resources)
	if err != nil {
		return nil, fmt.Errorf("failed to estimate cost for %s: %w", provider.Name, err)
	}
	deployment.EstimatedCost = cost

	return deployment, nil
}

func (r *MultiCloudVMReconciler) planBurstDeployment(ctx context.Context, plan *DeploymentPlan, mcvm *novacronv1.MultiCloudVM, template *novacronv1.VMTemplate) error {
	// Analyze current load and predict burst needs
	// This would integrate with AI scheduling policy if available
	
	// For now, implement simple cost-based selection
	if mcvm.Spec.CostOptimization != nil {
		return r.planCostOptimizedDeployment(ctx, plan, mcvm, template)
	}

	return nil
}

func (r *MultiCloudVMReconciler) planCostOptimizedDeployment(ctx context.Context, plan *DeploymentPlan, mcvm *novacronv1.MultiCloudVM, template *novacronv1.VMTemplate) error {
	// Find the most cost-effective secondary providers
	costAnalysis := make(map[string]float64)
	
	for _, secondary := range mcvm.Spec.DeploymentStrategy.Secondary {
		provider, err := r.getProvider(mcvm, secondary)
		if err != nil {
			continue
		}
		
		resources := ResourceRequirements{
			CPU:     template.Spec.Config.Resources.CPU.Request,
			Memory:  template.Spec.Config.Resources.Memory.Request,
			Storage: template.Spec.Config.Resources.Disk.Request,
		}
		
		cost, err := r.CloudProviders.EstimateCost(provider.Name, provider.Region, resources)
		if err != nil {
			continue
		}
		
		costAnalysis[secondary] = cost.HourlyCost
	}

	// Select the most cost-effective provider if cost optimization is enabled
	if len(costAnalysis) > 0 {
		bestProvider := ""
		bestCost := float64(^uint(0) >> 1) // Max float64
		
		for provider, cost := range costAnalysis {
			if cost < bestCost {
				bestCost = cost
				bestProvider = provider
			}
		}
		
		if bestProvider != "" {
			provider, _ := r.getProvider(mcvm, bestProvider)
			deployment, err := r.planProviderDeployment(ctx, provider, template, false)
			if err != nil {
				return err
			}
			plan.Deployments[bestProvider] = deployment
		}
	}

	return nil
}

func (r *MultiCloudVMReconciler) executeDeployment(ctx context.Context, mcvm *novacronv1.MultiCloudVM, plan *DeploymentPlan) (map[string]*DeploymentResult, error) {
	results := make(map[string]*DeploymentResult)

	for providerName, deployment := range plan.Deployments {
		if deployment.Status == "standby" {
			// Skip standby deployments for active-passive
			continue
		}

		result, err := r.deployToProvider(ctx, mcvm, providerName, deployment)
		if err != nil {
			return nil, fmt.Errorf("deployment to %s failed: %w", providerName, err)
		}
		results[providerName] = result
	}

	return results, nil
}

func (r *MultiCloudVMReconciler) deployToProvider(ctx context.Context, mcvm *novacronv1.MultiCloudVM, providerName string, deployment *ProviderDeployment) (*DeploymentResult, error) {
	logger := log.FromContext(ctx)
	
	// Get cloud provider client
	cloudClient, err := r.CloudProviders.GetClient(providerName)
	if err != nil {
		return nil, fmt.Errorf("failed to get cloud client for %s: %w", providerName, err)
	}

	// Deploy VM through the cloud provider
	vmRequest := &providers.VMRequest{
		Name:      fmt.Sprintf("%s-%s", mcvm.Name, providerName),
		Region:    deployment.Region,
		Resources: deployment.Resources,
		Template:  deployment.Template,
		Tags: map[string]string{
			"kubernetes.namespace":     mcvm.Namespace,
			"kubernetes.name":          mcvm.Name,
			"novacron.provider":        providerName,
			"novacron.deployment-type": deployment.Status,
		},
	}

	vmResult, err := cloudClient.CreateVM(ctx, vmRequest)
	if err != nil {
		return nil, fmt.Errorf("cloud provider VM creation failed: %w", err)
	}

	logger.Info("VM deployed to cloud provider", "provider", providerName, "vmId", vmResult.ID)
	r.Recorder.Event(mcvm, "Normal", "VMDeployed", 
		fmt.Sprintf("VM deployed to %s with ID %s", providerName, vmResult.ID))

	result := &DeploymentResult{
		Provider:     providerName,
		Status:       "deployed",
		VMID:         vmResult.ID,
		IPAddress:    vmResult.IPAddress,
		Cost:         deployment.EstimatedCost,
		DeployedAt:   metav1.Now(),
		InstanceType: vmResult.InstanceType,
	}

	// Store provider-specific VM ID in annotations
	if mcvm.Annotations == nil {
		mcvm.Annotations = make(map[string]string)
	}
	mcvm.Annotations[ProviderAnnotationPrefix+providerName] = vmResult.ID

	return result, nil
}

func (r *MultiCloudVMReconciler) updateDeploymentStatus(ctx context.Context, mcvm *novacronv1.MultiCloudVM, results map[string]*DeploymentResult) error {
	// Update cloud deployments status
	var cloudDeployments []novacronv1.CloudDeploymentStatus
	var totalCost float64
	var primaryProvider string

	for providerName, result := range results {
		deployment := novacronv1.CloudDeploymentStatus{
			Provider: providerName,
			Status:   result.Status,
			Instances: []novacronv1.VMInstance{
				{
					ID:     result.VMID,
					Status: "running", // Assume running for now
					IP:     result.IPAddress,
				},
			},
			Cost: &novacronv1.ResourceCost{
				Currency:   result.Cost.Currency,
				HourlyCost: result.Cost.HourlyCost,
				TotalCost:  result.Cost.TotalCost,
			},
			LastUpdated: &result.DeployedAt,
		}
		
		cloudDeployments = append(cloudDeployments, deployment)
		totalCost += result.Cost.HourlyCost
		
		if result.Provider == mcvm.Spec.DeploymentStrategy.Primary {
			primaryProvider = providerName
		}
	}

	// Update status
	mcvm.Status.CloudDeployments = cloudDeployments
	mcvm.Status.PrimaryProvider = primaryProvider
	mcvm.Status.TotalCost = &novacronv1.ResourceCost{
		Currency:   "USD", // Default currency
		HourlyCost: totalCost,
		TotalCost:  totalCost, // Will be calculated over time
	}
	mcvm.Status.Health = novacronv1.MultiCloudHealthStatus{
		Status:       "Healthy",
		Availability: 100.0, // Will be calculated based on actual health checks
	}

	// Update conditions
	now := metav1.Now()
	mcvm.Status.Conditions = r.updateMultiCloudConditions(mcvm.Status.Conditions,
		novacronv1.MultiCloudVMReady, metav1.ConditionTrue,
		"DeploymentSuccessful", "MultiCloud VM deployed successfully", now)

	return r.Status().Update(ctx, mcvm)
}

func (r *MultiCloudVMReconciler) optimizeCosts(ctx context.Context, mcvm *novacronv1.MultiCloudVM, results map[string]*DeploymentResult) error {
	logger := log.FromContext(ctx)
	
	if mcvm.Spec.CostOptimization == nil {
		return nil
	}

	// Check if total cost exceeds maximum allowed cost
	if mcvm.Spec.CostOptimization.MaxCostPerHour != "" {
		// Parse max cost and compare
		// Implementation would parse the cost string and compare
		logger.V(1).Info("Cost optimization check", "maxCost", mcvm.Spec.CostOptimization.MaxCostPerHour)
	}

	// Handle spot instance optimization
	if mcvm.Spec.CostOptimization.UseSpotInstances {
		return r.optimizeWithSpotInstances(ctx, mcvm, results)
	}

	// Cost-based scaling
	if mcvm.Spec.CostOptimization.CostBasedScaling {
		return r.handleCostBasedScaling(ctx, mcvm, results)
	}

	return nil
}

func (r *MultiCloudVMReconciler) optimizeWithSpotInstances(ctx context.Context, mcvm *novacronv1.MultiCloudVM, results map[string]*DeploymentResult) error {
	// Check if spot instances can be used for secondary deployments
	for providerName, result := range results {
		if result.Provider != mcvm.Spec.DeploymentStrategy.Primary {
			// Try to migrate to spot instances for cost savings
			// Implementation would check spot availability and migrate
			log.FromContext(ctx).V(1).Info("Checking spot instance availability", "provider", providerName)
		}
	}
	return nil
}

func (r *MultiCloudVMReconciler) handleCostBasedScaling(ctx context.Context, mcvm *novacronv1.MultiCloudVM, results map[string]*DeploymentResult) error {
	// Implement cost-based scaling logic
	// This would integrate with the AI scheduling policy for intelligent decisions
	return nil
}

func (r *MultiCloudVMReconciler) evaluateMigrationNeeds(ctx context.Context, mcvm *novacronv1.MultiCloudVM) (bool, string) {
	if mcvm.Spec.MigrationPolicy == nil {
		return false, ""
	}

	// Check migration triggers
	for _, trigger := range mcvm.Spec.MigrationPolicy.Triggers {
		switch trigger.Type {
		case "cost":
			if needsCostMigration, reason := r.evaluateCostMigration(mcvm, trigger); needsCostMigration {
				return true, reason
			}
		case "performance":
			if needsPerfMigration, reason := r.evaluatePerformanceMigration(mcvm, trigger); needsPerfMigration {
				return true, reason
			}
		case "availability":
			if needsAvailMigration, reason := r.evaluateAvailabilityMigration(mcvm, trigger); needsAvailMigration {
				return true, reason
			}
		}
	}

	return false, ""
}

func (r *MultiCloudVMReconciler) evaluateCostMigration(mcvm *novacronv1.MultiCloudVM, trigger novacronv1.MigrationTrigger) (bool, string) {
	// Check if current costs exceed thresholds
	if mcvm.Status.TotalCost != nil && trigger.Threshold != "" {
		// Parse threshold and compare
		// Implementation would parse cost threshold and compare with current costs
		return false, ""
	}
	return false, ""
}

func (r *MultiCloudVMReconciler) evaluatePerformanceMigration(mcvm *novacronv1.MultiCloudVM, trigger novacronv1.MigrationTrigger) (bool, string) {
	// Check performance metrics against thresholds
	// Implementation would check CPU, memory, network performance
	return false, ""
}

func (r *MultiCloudVMReconciler) evaluateAvailabilityMigration(mcvm *novacronv1.MultiCloudVM, trigger novacronv1.MigrationTrigger) (bool, string) {
	// Check availability metrics
	if mcvm.Status.Health.Availability < 99.0 { // Example threshold
		return true, "Availability below threshold"
	}
	return false, ""
}

func (r *MultiCloudVMReconciler) handleMigration(ctx context.Context, mcvm *novacronv1.MultiCloudVM, reason string) error {
	logger := log.FromContext(ctx)
	
	logger.Info("Starting cross-cloud migration", "reason", reason)
	r.Recorder.Event(mcvm, "Normal", "MigrationStarted", 
		fmt.Sprintf("Cross-cloud migration started: %s", reason))

	// Update migration status
	mcvm.Status.Migration = &novacronv1.CrossCloudMigrationStatus{
		Type:        mcvm.Spec.MigrationPolicy.Type,
		Progress:    0.0,
		StartTime:   &metav1.Time{Time: time.Now()},
		Phases:      []novacronv1.MigrationPhase{},
	}

	// Implementation would handle the actual migration process
	// For now, just log the migration intent
	logger.V(1).Info("Migration handling not fully implemented", "type", mcvm.Spec.MigrationPolicy.Type)
	
	return nil
}

func (r *MultiCloudVMReconciler) handleDeletion(ctx context.Context, mcvm *novacronv1.MultiCloudVM) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(mcvm.Finalizers, MultiCloudVMFinalizerName) {
		// Clean up VMs in all cloud providers
		for _, deployment := range mcvm.Status.CloudDeployments {
			vmID := ""
			if len(deployment.Instances) > 0 {
				vmID = deployment.Instances[0].ID
			}
			
			if vmID != "" {
				logger.Info("Deleting VM from cloud provider", "provider", deployment.Provider, "vmID", vmID)
				
				cloudClient, err := r.CloudProviders.GetClient(deployment.Provider)
				if err != nil {
					logger.Error(err, "Failed to get cloud client", "provider", deployment.Provider)
					continue
				}
				
				if err := cloudClient.DeleteVM(ctx, vmID); err != nil {
					logger.Error(err, "Failed to delete VM from cloud provider", "provider", deployment.Provider, "vmID", vmID)
					// Continue with cleanup even if deletion fails
				}
			}
		}

		// Remove finalizer
		mcvm.Finalizers = removeFinalizer(mcvm.Finalizers, MultiCloudVMFinalizerName)
		if err := r.Update(ctx, mcvm); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *MultiCloudVMReconciler) updateStatus(ctx context.Context, mcvm *novacronv1.MultiCloudVM, status, message string) (ctrl.Result, error) {
	now := metav1.Now()
	
	var conditionType novacronv1.MultiCloudVMConditionType
	var conditionStatus metav1.ConditionStatus
	
	switch status {
	case "Failed":
		conditionType = novacronv1.MultiCloudVMReady
		conditionStatus = metav1.ConditionFalse
	default:
		conditionType = novacronv1.MultiCloudVMReady
		conditionStatus = metav1.ConditionUnknown
	}
	
	mcvm.Status.Conditions = r.updateMultiCloudConditions(mcvm.Status.Conditions,
		conditionType, conditionStatus, status, message, now)
	
	if err := r.Status().Update(ctx, mcvm); err != nil {
		return ctrl.Result{}, err
	}
	
	return ctrl.Result{RequeueAfter: MultiCloudRequeueInterval}, nil
}

func (r *MultiCloudVMReconciler) updateMultiCloudConditions(conditions []novacronv1.MultiCloudVMCondition,
	condType novacronv1.MultiCloudVMConditionType, status metav1.ConditionStatus,
	reason, message string, now metav1.Time) []novacronv1.MultiCloudVMCondition {

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
	newCondition := novacronv1.MultiCloudVMCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	return append(conditions, newCondition)
}

func (r *MultiCloudVMReconciler) getProvider(mcvm *novacronv1.MultiCloudVM, providerName string) (*novacronv1.CloudProvider, error) {
	for _, provider := range mcvm.Spec.Providers {
		if provider.Name == providerName {
			return &provider, nil
		}
	}
	return nil, fmt.Errorf("provider %s not found", providerName)
}

// SetupWithManager sets up the controller with the Manager
func (r *MultiCloudVMReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.MultiCloudVM{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		WithEventFilter(predicate.Or(predicate.GenerationChangedPredicate{}, predicate.LabelChangedPredicate{})).
		Complete(r)
}

// Helper types and functions

type DeploymentPlan struct {
	Strategy    string
	Primary     string
	Secondary   []string
	Template    *novacronv1.VMTemplate
	Deployments map[string]*ProviderDeployment
}

type ProviderDeployment struct {
	Provider      string
	Region        string
	IsPrimary     bool
	Status        string
	Resources     ResourceRequirements
	EstimatedCost *novacronv1.ResourceCost
	Template      interface{}
}

type ResourceRequirements struct {
	CPU     string
	Memory  string
	Storage string
}

type DeploymentResult struct {
	Provider     string
	Status       string
	VMID         string
	IPAddress    string
	Cost         *novacronv1.ResourceCost
	DeployedAt   metav1.Time
	InstanceType string
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}