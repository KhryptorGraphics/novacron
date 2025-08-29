package controllers

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/cache"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// CacheIntegrationReconciler reconciles a CacheIntegration object
type CacheIntegrationReconciler struct {
	client.Client
	Scheme       *runtime.Scheme
	CacheManager cache.Manager
	Recorder     record.EventRecorder
}

const (
	CacheIntegrationFinalizerName = "novacron.io/cacheIntegration"
	CacheRequeueInterval         = 30 * time.Second
)

//+kubebuilder:rbac:groups=novacron.io,resources=cacheintegrations,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=novacron.io,resources=cacheintegrations/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=novacron.io,resources=cacheintegrations/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch
//+kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch

// Reconcile implements the reconciliation logic for CacheIntegration resources
func (r *CacheIntegrationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("cacheintegration", req.NamespacedName)

	// Fetch the CacheIntegration resource
	cacheInt := &novacronv1.CacheIntegration{}
	if err := r.Get(ctx, req.NamespacedName, cacheInt); err != nil {
		if client.IgnoreNotFound(err) == nil {
			logger.Info("CacheIntegration resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get CacheIntegration")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if cacheInt.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, cacheInt)
	}

	// Add finalizer if not present
	if !containsFinalizer(cacheInt.Finalizers, CacheIntegrationFinalizerName) {
		cacheInt.Finalizers = append(cacheInt.Finalizers, CacheIntegrationFinalizerName)
		if err := r.Update(ctx, cacheInt); err != nil {
			logger.Error(err, "Failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update observed generation
	if cacheInt.Status.ObservedGeneration != cacheInt.Generation {
		cacheInt.Status.ObservedGeneration = cacheInt.Generation
		if err := r.Status().Update(ctx, cacheInt); err != nil {
			logger.Error(err, "Failed to update observed generation")
			return ctrl.Result{}, err
		}
	}

	// Handle cache integration lifecycle
	return r.handleCacheIntegrationLifecycle(ctx, cacheInt)
}

func (r *CacheIntegrationReconciler) handleCacheIntegrationLifecycle(ctx context.Context, cacheInt *novacronv1.CacheIntegration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Validate cache configuration
	if err := r.validateCacheConfig(cacheInt); err != nil {
		return r.updateStatus(ctx, cacheInt, "Failed", err.Error())
	}

	// Get Redis credentials if needed
	credentials, err := r.getRedisCredentials(ctx, cacheInt)
	if err != nil {
		logger.Error(err, "Failed to get Redis credentials")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("Credentials error: %v", err))
	}

	// Initialize or update cache cluster
	clusterConfig, err := r.buildClusterConfig(cacheInt, credentials)
	if err != nil {
		logger.Error(err, "Failed to build cluster configuration")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("Configuration error: %v", err))
	}

	// Connect to Redis cluster
	cacheID := fmt.Sprintf("%s-%s", cacheInt.Namespace, cacheInt.Name)
	cluster, err := r.CacheManager.GetOrCreateCluster(cacheID, clusterConfig)
	if err != nil {
		logger.Error(err, "Failed to connect to Redis cluster")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("Connection error: %v", err))
	}

	// Configure cache strategy
	if err := r.configureCacheStrategy(ctx, cluster, cacheInt); err != nil {
		logger.Error(err, "Failed to configure cache strategy")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("Strategy error: %v", err))
	}

	// Configure TTL policies
	if err := r.configureTTLPolicies(ctx, cluster, cacheInt); err != nil {
		logger.Error(err, "Failed to configure TTL policies")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("TTL policy error: %v", err))
	}

	// Configure eviction policy
	if err := r.configureEvictionPolicy(ctx, cluster, cacheInt); err != nil {
		logger.Error(err, "Failed to configure eviction policy")
		return r.updateStatus(ctx, cacheInt, "Failed", fmt.Sprintf("Eviction policy error: %v", err))
	}

	// Handle cache warming if enabled
	if cacheInt.Spec.WarmingConfig != nil && cacheInt.Spec.WarmingConfig.Enabled {
		if err := r.handleCacheWarming(ctx, cluster, cacheInt); err != nil {
			logger.V(1).Info("Cache warming failed", "error", err)
			// Don't fail reconciliation for cache warming errors
		}
	}

	// Update cluster health status
	health, err := r.getClusterHealth(ctx, cluster)
	if err != nil {
		logger.V(1).Info("Failed to get cluster health", "error", err)
		health = &novacronv1.RedisClusterHealth{Status: "unknown"}
	}
	cacheInt.Status.ClusterHealth = health

	// Update performance metrics
	metrics, err := r.getPerformanceMetrics(ctx, cluster)
	if err != nil {
		logger.V(1).Info("Failed to get performance metrics", "error", err)
	} else {
		cacheInt.Status.PerformanceMetrics = metrics
	}

	// Update memory usage
	memUsage, err := r.getMemoryUsage(ctx, cluster)
	if err != nil {
		logger.V(1).Info("Failed to get memory usage", "error", err)
	} else {
		cacheInt.Status.MemoryUsage = memUsage
	}

	// Update overall status
	if err := r.updateCacheIntegrationStatus(ctx, cacheInt); err != nil {
		logger.Error(err, "Failed to update cache integration status")
		return ctrl.Result{}, err
	}

	logger.Info("CacheIntegration reconciliation completed successfully")
	return ctrl.Result{RequeueAfter: CacheRequeueInterval}, nil
}

func (r *CacheIntegrationReconciler) validateCacheConfig(cacheInt *novacronv1.CacheIntegration) error {
	// Validate Redis endpoints
	if len(cacheInt.Spec.RedisConfig.Endpoints) == 0 {
		return fmt.Errorf("at least one Redis endpoint must be specified")
	}

	for _, endpoint := range cacheInt.Spec.RedisConfig.Endpoints {
		if endpoint == "" {
			return fmt.Errorf("Redis endpoint cannot be empty")
		}
		// Basic endpoint validation
		if !strings.Contains(endpoint, ":") {
			return fmt.Errorf("Redis endpoint must include port: %s", endpoint)
		}
	}

	// Validate cache strategy
	validStrategies := []string{"write-through", "write-behind", "read-through", "cache-aside"}
	if !contains(validStrategies, cacheInt.Spec.Strategy.Type) {
		return fmt.Errorf("invalid cache strategy: %s", cacheInt.Spec.Strategy.Type)
	}

	// Validate eviction algorithm
	if cacheInt.Spec.EvictionPolicy != nil {
		validAlgorithms := []string{"LRU", "LFU", "FIFO", "Random"}
		if !contains(validAlgorithms, cacheInt.Spec.EvictionPolicy.Algorithm) {
			return fmt.Errorf("invalid eviction algorithm: %s", cacheInt.Spec.EvictionPolicy.Algorithm)
		}
	}

	return nil
}

func (r *CacheIntegrationReconciler) getRedisCredentials(ctx context.Context, cacheInt *novacronv1.CacheIntegration) (map[string]string, error) {
	credentials := make(map[string]string)

	if cacheInt.Spec.RedisConfig.CredentialsSecret == "" {
		// No credentials specified, use empty credentials
		return credentials, nil
	}

	// Get credentials from secret
	secret := &corev1.Secret{}
	secretKey := types.NamespacedName{
		Name:      cacheInt.Spec.RedisConfig.CredentialsSecret,
		Namespace: cacheInt.Namespace,
	}

	if err := r.Get(ctx, secretKey, secret); err != nil {
		return nil, fmt.Errorf("failed to get credentials secret %s: %w", secretKey, err)
	}

	// Extract credentials
	for key, value := range secret.Data {
		credentials[key] = string(value)
	}

	return credentials, nil
}

func (r *CacheIntegrationReconciler) buildClusterConfig(cacheInt *novacronv1.CacheIntegration, credentials map[string]string) (*cache.ClusterConfig, error) {
	config := &cache.ClusterConfig{
		Endpoints:   cacheInt.Spec.RedisConfig.Endpoints,
		Credentials: credentials,
	}

	// Configure high availability
	if cacheInt.Spec.RedisConfig.HA != nil {
		config.HA = &cache.HAConfig{
			SentinelEnabled: cacheInt.Spec.RedisConfig.HA.SentinelEnabled,
			Replicas:        cacheInt.Spec.RedisConfig.HA.Replicas,
			SentinelHosts:   cacheInt.Spec.RedisConfig.HA.SentinelHosts,
		}
	}

	// Configure security
	if cacheInt.Spec.RedisConfig.Security != nil {
		config.Security = &cache.SecurityConfig{
			TLSEnabled:  cacheInt.Spec.RedisConfig.Security.TLSEnabled,
			AuthEnabled: cacheInt.Spec.RedisConfig.Security.AuthEnabled,
			CertSecret:  cacheInt.Spec.RedisConfig.Security.CertSecret,
		}
	}

	return config, nil
}

func (r *CacheIntegrationReconciler) configureCacheStrategy(ctx context.Context, cluster cache.Cluster, cacheInt *novacronv1.CacheIntegration) error {
	strategy := &cache.Strategy{
		Type:        cacheInt.Spec.Strategy.Type,
		Consistency: cacheInt.Spec.Strategy.Consistency,
	}

	// Configure cache levels
	for _, level := range cacheInt.Spec.Strategy.Levels {
		strategy.Levels = append(strategy.Levels, cache.Level{
			Name:     level.Name,
			Size:     level.Size,
			TTL:      level.TTL,
			Strategy: level.Strategy,
		})
	}

	return cluster.ConfigureStrategy(strategy)
}

func (r *CacheIntegrationReconciler) configureTTLPolicies(ctx context.Context, cluster cache.Cluster, cacheInt *novacronv1.CacheIntegration) error {
	for _, ttlPolicy := range cacheInt.Spec.TTLPolicies {
		policy := &cache.TTLPolicy{
			Pattern:         ttlPolicy.Pattern,
			TTL:            ttlPolicy.TTL,
			RefreshStrategy: ttlPolicy.RefreshStrategy,
		}
		if err := cluster.SetTTLPolicy(policy); err != nil {
			return fmt.Errorf("failed to set TTL policy for pattern %s: %w", ttlPolicy.Pattern, err)
		}
	}
	return nil
}

func (r *CacheIntegrationReconciler) configureEvictionPolicy(ctx context.Context, cluster cache.Cluster, cacheInt *novacronv1.CacheIntegration) error {
	if cacheInt.Spec.EvictionPolicy == nil {
		return nil
	}

	policy := &cache.EvictionPolicy{
		Algorithm: cacheInt.Spec.EvictionPolicy.Algorithm,
	}

	if cacheInt.Spec.EvictionPolicy.MemoryThresholds != nil {
		policy.MemoryThresholds = &cache.MemoryThresholds{
			Warning:  cacheInt.Spec.EvictionPolicy.MemoryThresholds.Warning,
			Critical: cacheInt.Spec.EvictionPolicy.MemoryThresholds.Critical,
			Eviction: cacheInt.Spec.EvictionPolicy.MemoryThresholds.Eviction,
		}
	}

	return cluster.ConfigureEviction(policy)
}

func (r *CacheIntegrationReconciler) handleCacheWarming(ctx context.Context, cluster cache.Cluster, cacheInt *novacronv1.CacheIntegration) error {
	logger := log.FromContext(ctx)
	
	warmingConfig := cacheInt.Spec.WarmingConfig
	if !warmingConfig.Enabled {
		return nil
	}

	logger.Info("Starting cache warming")

	for _, strategy := range warmingConfig.Strategies {
		warmingStrategy := &cache.WarmingStrategy{
			Type:     strategy.Type,
			Patterns: strategy.Patterns,
			Priority: strategy.Priority,
		}

		if err := cluster.WarmCache(warmingStrategy); err != nil {
			return fmt.Errorf("cache warming failed for strategy %s: %w", strategy.Type, err)
		}
	}

	r.Recorder.Event(cacheInt, "Normal", "CacheWarmed", "Cache warming completed successfully")
	return nil
}

func (r *CacheIntegrationReconciler) getClusterHealth(ctx context.Context, cluster cache.Cluster) (*novacronv1.RedisClusterHealth, error) {
	health, err := cluster.GetHealth()
	if err != nil {
		return nil, err
	}

	var nodes []novacronv1.RedisNodeStatus
	for _, node := range health.Nodes {
		nodes = append(nodes, novacronv1.RedisNodeStatus{
			NodeID:   node.NodeID,
			Status:   node.Status,
			Role:     node.Role,
			Memory:   node.Memory,
			LastSeen: node.LastSeen,
		})
	}

	return &novacronv1.RedisClusterHealth{
		Status:      health.Status,
		Nodes:       nodes,
		Replication: health.Replication,
	}, nil
}

func (r *CacheIntegrationReconciler) getPerformanceMetrics(ctx context.Context, cluster cache.Cluster) (*novacronv1.CachePerformanceMetrics, error) {
	metrics, err := cluster.GetPerformanceMetrics()
	if err != nil {
		return nil, err
	}

	return &novacronv1.CachePerformanceMetrics{
		HitRate:      metrics.HitRate,
		MissRate:     metrics.MissRate,
		ResponseTime: metrics.ResponseTime,
		Throughput:   metrics.Throughput,
	}, nil
}

func (r *CacheIntegrationReconciler) getMemoryUsage(ctx context.Context, cluster cache.Cluster) (*novacronv1.CacheMemoryUsage, error) {
	usage, err := cluster.GetMemoryUsage()
	if err != nil {
		return nil, err
	}

	// Calculate usage percentage
	var usagePercentage float64
	if totalMB := r.parseMemory(usage.TotalMemory); totalMB > 0 {
		usedMB := r.parseMemory(usage.UsedMemory)
		usagePercentage = (usedMB / totalMB) * 100
	}

	return &novacronv1.CacheMemoryUsage{
		TotalMemory:     usage.TotalMemory,
		UsedMemory:      usage.UsedMemory,
		UsagePercentage: usagePercentage,
		UsageByLevel:    usage.UsageByLevel,
	}, nil
}

func (r *CacheIntegrationReconciler) parseMemory(memStr string) float64 {
	// Simple memory parsing - extract number from strings like "1.5GB", "512MB"
	memStr = strings.ToUpper(strings.TrimSpace(memStr))
	
	var multiplier float64 = 1
	if strings.HasSuffix(memStr, "GB") {
		multiplier = 1024
		memStr = strings.TrimSuffix(memStr, "GB")
	} else if strings.HasSuffix(memStr, "MB") {
		multiplier = 1
		memStr = strings.TrimSuffix(memStr, "MB")
	}
	
	value, err := strconv.ParseFloat(memStr, 64)
	if err != nil {
		return 0
	}
	
	return value * multiplier
}

func (r *CacheIntegrationReconciler) updateCacheIntegrationStatus(ctx context.Context, cacheInt *novacronv1.CacheIntegration) error {
	now := metav1.Now()
	
	var conditionType novacronv1.CacheIntegrationConditionType
	var conditionStatus metav1.ConditionStatus
	var reason, message string

	// Determine overall status based on cluster health
	if cacheInt.Status.ClusterHealth != nil {
		switch cacheInt.Status.ClusterHealth.Status {
		case "healthy":
			conditionType = novacronv1.CacheIntegrationReady
			conditionStatus = metav1.ConditionTrue
			reason = "CacheHealthy"
			message = "Redis cluster is healthy and performing well"
			
			// Check performance
			if cacheInt.Status.PerformanceMetrics != nil {
				if cacheInt.Status.PerformanceMetrics.HitRate > 0.8 {
					conditionType = novacronv1.CacheIntegrationPerforming
					conditionStatus = metav1.ConditionTrue
					reason = "HighPerformance"
					message = fmt.Sprintf("Cache performing well with %.1f%% hit rate", 
						cacheInt.Status.PerformanceMetrics.HitRate*100)
				}
			}
		case "degraded":
			conditionType = novacronv1.CacheIntegrationHealthy
			conditionStatus = metav1.ConditionFalse
			reason = "CacheDegraded"
			message = "Redis cluster performance is degraded"
		case "failed":
			conditionType = novacronv1.CacheIntegrationReady
			conditionStatus = metav1.ConditionFalse
			reason = "CacheFailed"
			message = "Redis cluster has failed"
		default:
			conditionType = novacronv1.CacheIntegrationReady
			conditionStatus = metav1.ConditionUnknown
			reason = "CacheUnknown"
			message = "Redis cluster status unknown"
		}
	} else {
		conditionType = novacronv1.CacheIntegrationReady
		conditionStatus = metav1.ConditionUnknown
		reason = "Initializing"
		message = "Cache integration initializing"
	}

	cacheInt.Status.Conditions = r.updateCacheConditions(cacheInt.Status.Conditions,
		conditionType, conditionStatus, reason, message, now)

	return r.Status().Update(ctx, cacheInt)
}

func (r *CacheIntegrationReconciler) updateCacheConditions(conditions []novacronv1.CacheIntegrationCondition,
	condType novacronv1.CacheIntegrationConditionType, status metav1.ConditionStatus,
	reason, message string, now metav1.Time) []novacronv1.CacheIntegrationCondition {

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
	newCondition := novacronv1.CacheIntegrationCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	return append(conditions, newCondition)
}

func (r *CacheIntegrationReconciler) handleDeletion(ctx context.Context, cacheInt *novacronv1.CacheIntegration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if containsFinalizer(cacheInt.Finalizers, CacheIntegrationFinalizerName) {
		// Clean up cache cluster connection
		cacheID := fmt.Sprintf("%s-%s", cacheInt.Namespace, cacheInt.Name)
		
		logger.Info("Disconnecting from Redis cluster", "cacheID", cacheID)
		if err := r.CacheManager.DisconnectCluster(cacheID); err != nil {
			logger.Error(err, "Failed to disconnect from Redis cluster", "cacheID", cacheID)
			// Continue with cleanup even if disconnection fails
		}

		// Remove finalizer
		cacheInt.Finalizers = removeFinalizer(cacheInt.Finalizers, CacheIntegrationFinalizerName)
		if err := r.Update(ctx, cacheInt); err != nil {
			logger.Error(err, "Failed to remove finalizer")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *CacheIntegrationReconciler) updateStatus(ctx context.Context, cacheInt *novacronv1.CacheIntegration, status, message string) (ctrl.Result, error) {
	now := metav1.Now()
	
	var conditionType novacronv1.CacheIntegrationConditionType
	var conditionStatus metav1.ConditionStatus
	
	switch status {
	case "Failed":
		conditionType = novacronv1.CacheIntegrationReady
		conditionStatus = metav1.ConditionFalse
	default:
		conditionType = novacronv1.CacheIntegrationReady
		conditionStatus = metav1.ConditionUnknown
	}
	
	cacheInt.Status.Conditions = r.updateCacheConditions(cacheInt.Status.Conditions,
		conditionType, conditionStatus, status, message, now)
	
	if err := r.Status().Update(ctx, cacheInt); err != nil {
		return ctrl.Result{}, err
	}
	
	return ctrl.Result{RequeueAfter: CacheRequeueInterval}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *CacheIntegrationReconciler) SetupWithManager(mgr ctrl.Manager, concurrency int) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&novacronv1.CacheIntegration{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: concurrency,
		}).
		WithEventFilter(predicate.Or(predicate.GenerationChangedPredicate{}, predicate.LabelChangedPredicate{})).
		Complete(r)
}