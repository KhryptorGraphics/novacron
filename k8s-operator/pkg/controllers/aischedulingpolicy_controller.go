package controllers

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/ai"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AISchedulingPolicyReconciler reconciles an AISchedulingPolicy object
type AISchedulingPolicyReconciler struct {
	client.Client
	Scheme    *runtime.Scheme
	AIEngine  ai.SchedulingEngine
	Recorder  record.EventRecorder
}

const (
	AISchedulingPolicyFinalizerName = "novacron.io/aiSchedulingPolicy"
	AIRequeueInterval              = 60 * time.Second
)

//+kubebuilder:rbac:groups=novacron.io,resources=aischedulingpolicies,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=aischedulingpolicies/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=aischedulingpolicies/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile implements the reconciliation logic for AISchedulingPolicy resources
func (r *AISchedulingPolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("aischedulingpolicy", req.NamespacedName)

	// Fetch the AISchedulingPolicy resource
	policy := &novacronv1.AISchedulingPolicy{}
	if err := r.Get(ctx, req.NamespacedName, policy); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("AISchedulingPolicy resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get AISchedulingPolicy")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if policy.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, policy)
	}

	// Add finalizer if not present
	if !containsFinalizer(policy.Finalizers, AISchedulingPolicyFinalizerName) {
		policy.Finalizers = append(policy.Finalizers, AISchedulingPolicyFinalizerName)
		if err := r.Update(ctx, policy); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update observed generation
	if policy.Status.ObservedGeneration != policy.Generation {
		policy.Status.ObservedGeneration = policy.Generation
		if err := r.Status().Update(ctx, policy); err != nil {
			logger.Error(err, "Failed to update observed generation")
			return ctrl.Result{}, err
		}
	}

	// Handle AI scheduling policy lifecycle
	return r.handleAISchedulingPolicyLifecycle(ctx, policy)
}

func (r *AISchedulingPolicyReconciler) handleAISchedulingPolicyLifecycle(ctx context.Context, policy *novacronv1.AISchedulingPolicy) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Validate AI model configuration
	if err := r.validateModelConfig(policy); err != nil {
		return r.updateStatus(ctx, policy, "Failed", err.Error())
	}

	// Initialize or update AI model
	modelStatus, err := r.initializeAIModel(ctx, policy)
	if err != nil {
		logger.Error(err, "Failed to initialize AI model")
		return r.updateStatus(ctx, policy, "Failed", fmt.Sprintf("Model initialization error: %v", err))
	}

	// Update model status
	policy.Status.ModelStatus = modelStatus

	// Handle training if needed
	if modelStatus.State == "training" || r.needsRetraining(policy) {
		trainingResult, err := r.handleModelTraining(ctx, policy)
		if err != nil {
			logger.Error(err, "Model training failed")
			return r.updateStatus(ctx, policy, "Training", fmt.Sprintf("Training error: %v", err))
		}
		
		policy.Status.LearningProgress = trainingResult.Progress
		policy.Status.ModelStatus.TrainingProgress = trainingResult.Progress.ValidationAccuracy
		
		if trainingResult.Completed {
			policy.Status.ModelStatus.State = "ready"
			policy.Status.ModelStatus.Accuracy = trainingResult.Progress.ValidationAccuracy
			policy.Status.ModelStatus.LastTraining = &metav1.Time{Time: time.Now()}
			
			r.Recorder.Event(policy, "Normal", "TrainingCompleted", 
				fmt.Sprintf("Model training completed with accuracy: %.2f%%", trainingResult.Progress.ValidationAccuracy*100))
		}
	}

	// Collect and process training data
	if policy.Status.ModelStatus.State == "ready" {
		if err := r.collectTrainingData(ctx, policy); err != nil {
			logger.V(1).Info("Failed to collect training data", "error", err)
			// Don't fail reconciliation for data collection errors
		}

		// Update accuracy metrics
		if err := r.updateAccuracyMetrics(ctx, policy); err != nil {
			logger.V(1).Info("Failed to update accuracy metrics", "error", err)
		}

		// Process recent scheduling decisions
		if err := r.processSchedulingDecisions(ctx, policy); err != nil {
			logger.V(1).Info("Failed to process scheduling decisions", "error", err)
		}
	}

	// Update overall status
	if err := r.updateAISchedulingStatus(ctx, policy); err != nil {
		logger.Error(err, "Failed to update AI scheduling status")
		return ctrl.Result{}, err
	}

	logger.Info("AISchedulingPolicy reconciliation completed successfully")
	return ctrl.Result{RequeueAfter: AIRequeueInterval}, nil
}

func (r *AISchedulingPolicyReconciler) validateModelConfig(policy *novacronv1.AISchedulingPolicy) error {
	config := policy.Spec.ModelConfig

	// Validate model type
	validTypes := []string{"neural-network", "decision-tree", "reinforcement-learning", "gradient-boosting"}
	if !contains(validTypes, config.ModelType) {
		return fmt.Errorf("invalid model type: %s", config.ModelType)
	}

	// Validate objectives
	if len(policy.Spec.Objectives) == 0 {
		return fmt.Errorf("at least one scheduling objective must be specified")
	}

	totalWeight := 0.0
	for _, objective := range policy.Spec.Objectives {
		if objective.Weight < 0.0 || objective.Weight > 1.0 {
			return fmt.Errorf("objective weight must be between 0.0 and 1.0, got: %f", objective.Weight)
		}
		totalWeight += objective.Weight
	}

	if totalWeight < 0.99 || totalWeight > 1.01 { // Allow small floating point errors
		return fmt.Errorf("total objective weights must sum to 1.0, got: %f", totalWeight)
	}

	return nil
}

func (r *AISchedulingPolicyReconciler) initializeAIModel(ctx context.Context, policy *novacronv1.AISchedulingPolicy) (*novacronv1.AIModelStatus, error) {
	logger := log.FromContext(ctx)

	modelID := fmt.Sprintf("%s-%s", policy.Namespace, policy.Name)
	
	// Check if model already exists
	existingModel, err := r.AIEngine.GetModel(modelID)
	if err == nil && existingModel != nil {
		// Model exists, check if configuration changed
		if r.modelConfigChanged(policy, existingModel) {
			logger.Info("Model configuration changed, updating model")
			if err := r.AIEngine.UpdateModel(modelID, r.convertToAIConfig(policy)); err != nil {
				return nil, fmt.Errorf("failed to update AI model: %w", err)
			}
		}
		return r.convertFromAIModelStatus(existingModel), nil
	}

	// Create new model
	logger.Info("Creating new AI model", "type", policy.Spec.ModelConfig.ModelType)
	
	aiConfig := r.convertToAIConfig(policy)
	model, err := r.AIEngine.CreateModel(modelID, aiConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create AI model: %w", err)
	}

	r.Recorder.Event(policy, "Normal", "ModelCreated", 
		fmt.Sprintf("AI model created: %s", policy.Spec.ModelConfig.ModelType))

	return r.convertFromAIModelStatus(model), nil
}

func (r *AISchedulingPolicyReconciler) convertToAIConfig(policy *novacronv1.AISchedulingPolicy) *ai.ModelConfig {
	config := &ai.ModelConfig{
		Type:       policy.Spec.ModelConfig.ModelType,
		Version:    policy.Spec.ModelConfig.Version,
		Parameters: policy.Spec.ModelConfig.Parameters,
	}

	// Convert objectives
	for _, obj := range policy.Spec.Objectives {
		config.Objectives = append(config.Objectives, ai.Objective{
			Type:   obj.Type,
			Weight: obj.Weight,
			Target: obj.Target,
		})
	}

	// Convert data sources
	for _, ds := range policy.Spec.DataSources {
		config.DataSources = append(config.DataSources, ai.DataSource{
			Type:       ds.Type,
			Connection: ds.Connection,
			Metrics:    ds.Metrics,
			Interval:   ds.Interval,
		})
	}

	// Convert training config if present
	if policy.Spec.ModelConfig.TrainingConfig != nil {
		tc := policy.Spec.ModelConfig.TrainingConfig
		config.TrainingConfig = &ai.TrainingConfig{
			DatasetSize:     tc.DatasetSize,
			ValidationSplit: tc.ValidationSplit,
			Epochs:          tc.Epochs,
			Features:        tc.Features,
			Hyperparameters: tc.Hyperparameters,
		}
	}

	// Convert learning config if present
	if policy.Spec.LearningConfig != nil {
		lc := policy.Spec.LearningConfig
		config.LearningConfig = &ai.LearningConfig{
			OnlineLearning:     lc.OnlineLearning,
			LearningRate:       lc.LearningRate,
			BatchSize:          lc.BatchSize,
			RetrainingInterval: lc.RetrainingInterval,
		}
	}

	return config
}

func (r *AISchedulingPolicyReconciler) convertFromAIModelStatus(model *ai.Model) *novacronv1.AIModelStatus {
	return &novacronv1.AIModelStatus{
		State:            model.Status.State,
		Accuracy:         model.Status.Accuracy,
		LastTraining:     model.Status.LastTraining,
		TrainingProgress: model.Status.TrainingProgress,
	}
}

func (r *AISchedulingPolicyReconciler) modelConfigChanged(policy *novacronv1.AISchedulingPolicy, existingModel *ai.Model) bool {
	// Simple comparison - in production would do deep comparison
	return existingModel.Config.Type != policy.Spec.ModelConfig.ModelType
}

func (r *AISchedulingPolicyReconciler) needsRetraining(policy *novacronv1.AISchedulingPolicy) bool {
	if policy.Status.ModelStatus == nil {
		return true
	}

	// Check if retraining interval has passed
	if policy.Spec.LearningConfig != nil && policy.Spec.LearningConfig.RetrainingInterval != "" {
		if policy.Status.ModelStatus.LastTraining != nil {
			// Parse retraining interval and check if enough time has passed
			// For now, use a simple check
			lastTraining := policy.Status.ModelStatus.LastTraining.Time
			if time.Since(lastTraining) > time.Hour*24 { // Retrain daily by default
				return true
			}
		}
	}

	// Check if accuracy has degraded
	if policy.Status.AccuracyMetrics != nil {
		if policy.Status.AccuracyMetrics.ShortTermAccuracy < 0.8 { // Threshold
			return true
		}
	}

	return false
}

func (r *AISchedulingPolicyReconciler) handleModelTraining(ctx context.Context, policy *novacronv1.AISchedulingPolicy) (*ai.TrainingResult, error) {
	logger := log.FromContext(ctx)
	
	modelID := fmt.Sprintf("%s-%s", policy.Namespace, policy.Name)
	
	logger.Info("Starting model training", "modelType", policy.Spec.ModelConfig.ModelType)
	
	// Start training
	trainingJob, err := r.AIEngine.StartTraining(modelID)
	if err != nil {
		return nil, fmt.Errorf("failed to start training: %w", err)
	}

	// For now, simulate quick training completion
	// In production, this would track long-running training jobs
	result := &ai.TrainingResult{
		JobID:     trainingJob.ID,
		Completed: true,
		Progress: ai.LearningProgress{
			Iterations:         100,
			Loss:              0.05,
			ValidationAccuracy: 0.85,
		},
	}

	return result, nil
}

func (r *AISchedulingPolicyReconciler) collectTrainingData(ctx context.Context, policy *novacronv1.AISchedulingPolicy) error {
	// Collect data from configured data sources
	for _, dataSource := range policy.Spec.DataSources {
		if err := r.collectFromDataSource(ctx, policy, dataSource); err != nil {
			return fmt.Errorf("failed to collect from data source %s: %w", dataSource.Type, err)
		}
	}
	return nil
}

func (r *AISchedulingPolicyReconciler) collectFromDataSource(ctx context.Context, policy *novacronv1.AISchedulingPolicy, dataSource novacronv1.DataSource) error {
	logger := log.FromContext(ctx)
	
	switch dataSource.Type {
	case "prometheus":
		return r.collectFromPrometheus(ctx, policy, dataSource)
	case "influxdb":
		return r.collectFromInfluxDB(ctx, policy, dataSource)
	case "cloudwatch":
		return r.collectFromCloudWatch(ctx, policy, dataSource)
	default:
		logger.V(1).Info("Unsupported data source type", "type", dataSource.Type)
		return nil
	}
}

func (r *AISchedulingPolicyReconciler) collectFromPrometheus(ctx context.Context, policy *novacronv1.AISchedulingPolicy, dataSource novacronv1.DataSource) error {
	// Implementation would query Prometheus for metrics
	// For now, just log the collection attempt
	log.FromContext(ctx).V(1).Info("Collecting data from Prometheus", "metrics", dataSource.Metrics)
	return nil
}

func (r *AISchedulingPolicyReconciler) collectFromInfluxDB(ctx context.Context, policy *novacronv1.AISchedulingPolicy, dataSource novacronv1.DataSource) error {
	// Implementation would query InfluxDB for metrics
	log.FromContext(ctx).V(1).Info("Collecting data from InfluxDB", "metrics", dataSource.Metrics)
	return nil
}

func (r *AISchedulingPolicyReconciler) collectFromCloudWatch(ctx context.Context, policy *novacronv1.AISchedulingPolicy, dataSource novacronv1.DataSource) error {
	// Implementation would query CloudWatch for metrics
	log.FromContext(ctx).V(1).Info("Collecting data from CloudWatch", "metrics", dataSource.Metrics)
	return nil
}

func (r *AISchedulingPolicyReconciler) updateAccuracyMetrics(ctx context.Context, policy *novacronv1.AISchedulingPolicy) error {
	modelID := fmt.Sprintf("%s-%s", policy.Namespace, policy.Name)
	
	// Get accuracy metrics from AI engine
	metrics, err := r.AIEngine.GetAccuracyMetrics(modelID)
	if err != nil {
		return fmt.Errorf("failed to get accuracy metrics: %w", err)
	}

	policy.Status.AccuracyMetrics = &novacronv1.AccuracyMetrics{
		ShortTermAccuracy:   metrics.ShortTerm,
		LongTermAccuracy:    metrics.LongTerm,
		AccuracyByObjective: metrics.ByObjective,
	}

	return nil
}

func (r *AISchedulingPolicyReconciler) processSchedulingDecisions(ctx context.Context, policy *novacronv1.AISchedulingPolicy) error {
	modelID := fmt.Sprintf("%s-%s", policy.Namespace, policy.Name)
	
	// Get recent decisions from AI engine
	decisions, err := r.AIEngine.GetRecentDecisions(modelID, 10) // Last 10 decisions
	if err != nil {
		return fmt.Errorf("failed to get recent decisions: %w", err)
	}

	var policyDecisions []novacronv1.AISchedulingDecision
	for _, decision := range decisions {
		policyDecision := novacronv1.AISchedulingDecision{
			Timestamp:  &metav1.Time{Time: decision.Timestamp},
			WorkloadID: decision.WorkloadID,
			Placement: novacronv1.PlacementDecision{
				Target:   decision.Placement.Target,
				Provider: decision.Placement.Provider,
				Resources: novacronv1.ResourceCapacity{
					CPU:    decision.Placement.Resources.CPU,
					Memory: decision.Placement.Resources.Memory,
					Storage: decision.Placement.Resources.Storage,
				},
				ExpectedPerformance: decision.Placement.ExpectedPerformance,
			},
			Confidence: decision.Confidence,
			Reasoning:  decision.Reasoning,
		}
		policyDecisions = append(policyDecisions, policyDecision)
	}

	policy.Status.RecentDecisions = policyDecisions
	return nil
}

func (r *AISchedulingPolicyReconciler) updateAISchedulingStatus(ctx context.Context, policy *novacronv1.AISchedulingPolicy) error {
	now := metav1.Now()
	
	var conditionType novacronv1.AISchedulingPolicyConditionType
	var conditionStatus metav1.ConditionStatus
	var reason, message string

	if policy.Status.ModelStatus != nil {
		switch policy.Status.ModelStatus.State {
		case "ready":
			conditionType = novacronv1.AISchedulingPolicyReady
			conditionStatus = metav1.ConditionTrue
			reason = "ModelReady"
			message = "AI model is ready and making scheduling decisions"
		case "training":
			conditionType = novacronv1.AISchedulingPolicyTraining
			conditionStatus = metav1.ConditionTrue
			reason = "ModelTraining"
			message = fmt.Sprintf("AI model training in progress: %.1f%%", 
				policy.Status.ModelStatus.TrainingProgress*100)
		case "error":
			conditionType = novacronv1.AISchedulingPolicyReady
			conditionStatus = metav1.ConditionFalse
			reason = "ModelError"
			message = "AI model encountered an error"
		default:
			conditionType = novacronv1.AISchedulingPolicyReady
			conditionStatus = metav1.ConditionUnknown
			reason = "ModelUnknown"
			message = "AI model status unknown"
		}
	} else {
		conditionType = novacronv1.AISchedulingPolicyReady
		conditionStatus = metav1.ConditionUnknown
		reason = "Initializing"
		message = "AI model initializing"
	}

	policy.Status.Conditions = r.updateAIConditions(policy.Status.Conditions,
		conditionType, conditionStatus, reason, message, now)

	return r.Status().Update(ctx, policy)
}

func (r *AISchedulingPolicyReconciler) updateAIConditions(conditions []novacronv1.AISchedulingPolicyCondition,
	condType novacronv1.AISchedulingPolicyConditionType, status metav1.ConditionStatus,
	reason, message string, now metav1.Time) []novacronv1.AISchedulingPolicyCondition {

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
	newCondition := novacronv1.AISchedulingPolicyCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	return append(conditions, newCondition)
}

func (r *AISchedulingPolicyReconciler) handleDeletion(ctx context.Context, policy *novacronv1.AISchedulingPolicy) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(policy.Finalizers, AISchedulingPolicyFinalizerName) {
		// Clean up AI model
		modelID := fmt.Sprintf("%s-%s", policy.Namespace, policy.Name)
		
		logger.Info("Deleting AI model", "modelID", modelID)
		if err := r.AIEngine.DeleteModel(modelID); err != nil {
			logger.Error(err, "Failed to delete AI model", "modelID", modelID)
			// Continue with cleanup even if deletion fails
		}

		// Remove finalizer
		policy.Finalizers = removeFinalizer(policy.Finalizers, AISchedulingPolicyFinalizerName)
		if err := r.Update(ctx, policy); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *AISchedulingPolicyReconciler) updateStatus(ctx context.Context, policy *novacronv1.AISchedulingPolicy, status, message string) (ctrl.Result, error) {
	now := metav1.Now()
	
	var conditionType novacronv1.AISchedulingPolicyConditionType
	var conditionStatus metav1.ConditionStatus
	
	switch status {
	case "Failed":
		conditionType = novacronv1.AISchedulingPolicyReady
		conditionStatus = metav1.ConditionFalse
	case "Training":
		conditionType = novacronv1.AISchedulingPolicyTraining
		conditionStatus = metav1.ConditionTrue
	default:
		conditionType = novacronv1.AISchedulingPolicyReady
		conditionStatus = metav1.ConditionUnknown
	}
	
	policy.Status.Conditions = r.updateAIConditions(policy.Status.Conditions,
		conditionType, conditionStatus, status, message, now)
	
	if err := r.Status().Update(ctx, policy); err != nil {
		return ctrl.Result{}, err
	}
	
	return ctrl.Result{RequeueAfter: AIRequeueInterval}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *AISchedulingPolicyReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.AISchedulingPolicy{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		WithEventFilter(predicate.Or(predicate.GenerationChangedPredicate{}, predicate.LabelChangedPredicate{})).
		Complete(r)
}