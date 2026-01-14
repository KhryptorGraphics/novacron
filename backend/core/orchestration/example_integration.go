package orchestration

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/policy"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
)

// AdvancedOrchestrationDemo demonstrates the complete orchestration system
func AdvancedOrchestrationDemo(logger *logrus.Logger) error {
	logger.Info("Starting Advanced Orchestration Demo")

	// 1. Initialize Event Bus
	eventBus := events.NewNATSEventBus(logger)
	eventConfig := events.EventBusConfig{
		URL:           "nats://localhost:4222",
		ClusterID:     "novacron-demo",
		ClientID:      "orchestration-demo",
		MaxReconnects: 5,
		ReconnectWait: 2 * time.Second,
	}

	ctx := context.Background()
	if err := eventBus.Connect(ctx, eventConfig); err != nil {
		logger.WithError(err).Warn("Failed to connect to NATS, continuing with mock event bus")
		// Continue without event bus for demo
	}

	// 2. Initialize Placement Engine
	placementEngine := placement.NewDefaultPlacementEngine(logger)
	logger.Info("Placement engine initialized")

	// 3. Initialize Auto-Scaling
	autoScaler := autoscaling.NewDefaultAutoScaler(logger, eventBus)

	// Add auto-scaling target
	autoScalingTarget := &autoscaling.AutoScalerTarget{
		ID:      "web-service",
		Type:    "service",
		Enabled: true,
		Thresholds: &autoscaling.ScalingThresholds{
			CPUScaleUpThreshold:      0.7,
			CPUScaleDownThreshold:    0.3,
			MemoryScaleUpThreshold:   0.8,
			MemoryScaleDownThreshold: 0.4,
			MinReplicas:              2,
			MaxReplicas:              20,
			CooldownPeriod:           5 * time.Minute,
			PredictionWeight:         0.3,
		},
	}

	if err := autoScaler.AddTarget(autoScalingTarget); err != nil {
		return fmt.Errorf("failed to add auto-scaling target: %w", err)
	}

	if err := autoScaler.StartMonitoring(); err != nil {
		return fmt.Errorf("failed to start auto-scaling monitoring: %w", err)
	}
	logger.Info("Auto-scaling system started")

	// 4. Initialize Self-Healing
	healingController := healing.NewDefaultHealingController(logger, eventBus)

	// Add healing target
	healingTarget := &healing.HealingTarget{
		ID:      "database-cluster",
		Type:    healing.TargetTypeService,
		Name:    "PostgreSQL Cluster",
		Enabled: true,
		HealthCheckConfig: &healing.HealthCheckConfig{
			Interval:           30 * time.Second,
			Timeout:            10 * time.Second,
			HealthyThreshold:   3,
			UnhealthyThreshold: 2,
			FailureThreshold:   5,
			CheckType:          healing.HealthCheckTypeHTTP,
		},
		RecoveryConfig: &healing.RecoveryConfig{
			EnableAutoRecovery:  true,
			MaxRecoveryAttempts: 3,
			RecoveryTimeout:     10 * time.Minute,
			BackoffStrategy:     healing.BackoffExponential,
		},
	}

	if err := healingController.RegisterTarget(healingTarget); err != nil {
		return fmt.Errorf("failed to register healing target: %w", err)
	}

	if err := healingController.StartMonitoring(); err != nil {
		return fmt.Errorf("failed to start healing monitoring: %w", err)
	}
	logger.Info("Self-healing system started")

	// 5. Initialize Policy Engine
	policyEngine := policy.NewDefaultPolicyEngine(logger, eventBus)

	// Create sample policies
	placementPolicy := &policy.OrchestrationPolicy{
		Name:        "High Availability Placement",
		Description: "Ensures VMs are distributed across different availability zones",
		Enabled:     true,
		Priority:    10,
		Selector: &policy.PolicySelector{
			ResourceTypes: []policy.ResourceType{policy.ResourceTypeVM},
			MatchLabels:   map[string]string{"tier": "production"},
		},
		Rules: []*policy.PolicyRule{
			{
				Name:     "Anti-Affinity Rule",
				Type:     policy.RuleTypePlacement,
				Enabled:  true,
				Priority: 10,
				Conditions: []*policy.RuleCondition{
					{
						Type:     policy.ConditionTypeLabel,
						Field:    "labels.environment",
						Operator: policy.OperatorEquals,
						Value:    "production",
					},
				},
				Actions: []*policy.RuleAction{
					{
						Type: policy.ActionTypeSchedule,
						Parameters: map[string]interface{}{
							"strategy":      "anti-affinity",
							"spread_domain": "availability_zone",
						},
					},
				},
			},
		},
	}

	scalingPolicy := &policy.OrchestrationPolicy{
		Name:        "Reactive Scaling Policy",
		Description: "Scale services based on resource utilization",
		Enabled:     true,
		Priority:    8,
		Selector: &policy.PolicySelector{
			ResourceTypes: []policy.ResourceType{policy.ResourceTypeService},
		},
		Rules: []*policy.PolicyRule{
			{
				Name:     "CPU-based Scaling",
				Type:     policy.RuleTypeAutoScaling,
				Enabled:  true,
				Priority: 5,
				Conditions: []*policy.RuleCondition{
					{
						Type:     policy.ConditionTypeMetric,
						Field:    "metrics.cpu_usage",
						Operator: policy.OperatorGreaterThan,
						Value:    0.8,
					},
				},
				Actions: []*policy.RuleAction{
					{
						Type: policy.ActionTypeScale,
						Parameters: map[string]interface{}{
							"direction": "up",
							"factor":    1.5,
						},
					},
				},
			},
		},
	}

	if err := policyEngine.CreatePolicy(placementPolicy); err != nil {
		return fmt.Errorf("failed to create placement policy: %w", err)
	}

	if err := policyEngine.CreatePolicy(scalingPolicy); err != nil {
		return fmt.Errorf("failed to create scaling policy: %w", err)
	}
	logger.Info("Policy engine initialized with sample policies")

	// 6. Initialize Main Orchestration Engine
	orchestrationEngine := NewDefaultOrchestrationEngine(logger)
	// Inject a default evacuation handler with minimal adapters
	listFn := func(nodeID string) ([]string, error) {
		// Example only: integrate with your VM manager in real usage
		return []string{"vm-1", "vm-2"}, nil
	}
	selectFn := func(vmID string, sourceNodeID string) (string, error) {
		// Example only: choose a dummy target; in production, use placementEngine
		if sourceNodeID == "nodeA" { return "nodeB", nil }
		return "nodeA", nil
	}
	migrateFn := func(ctx context.Context, vmID, targetNodeID string) error {
		// Example only: call into migration manager/driver in real usage
		logger.WithFields(logrus.Fields{"vm_id": vmID, "target": targetNodeID}).Info("Mock migrate invoked")
		return nil
	}
	orchestrationEngine.SetEvacuationHandler(NewDefaultEvacuationHandler(listFn, selectFn, migrateFn, logger))


	// Register policies with orchestration engine
	for _, pol := range []*OrchestrationPolicy{
		{
			ID:       placementPolicy.ID,
			Name:     placementPolicy.Name,
			Enabled:  placementPolicy.Enabled,
			Priority: placementPolicy.Priority,
		},
		{
			ID:       scalingPolicy.ID,
			Name:     scalingPolicy.Name,
			Enabled:  scalingPolicy.Enabled,
			Priority: scalingPolicy.Priority,
		},
	} {
		if err := orchestrationEngine.RegisterPolicy(pol); err != nil {
			logger.WithError(err).Warn("Failed to register policy with orchestration engine")
		}
	}

	if err := orchestrationEngine.Start(ctx); err != nil {
		return fmt.Errorf("failed to start orchestration engine: %w", err)
	}
	logger.Info("Main orchestration engine started")

	// 7. Demonstrate System Operations
	logger.Info("=== Demonstrating System Operations ===")

	// Demonstrate placement decision
	vmSpec := placement.VMSpec{
		CPU:     4,
		Memory:  8192,
		Storage: 100,
		Labels: map[string]string{
			"environment": "production",
			"tier":        "web",
		},
	}

	placementRequest := &placement.PlacementRequest{
		VMID:     "web-vm-001",
		VMSpec:   vmSpec,
		Strategy: placement.StrategyBalanced,
	}

	_, err := placementEngine.PlaceVM(ctx, placementRequest)
	if err != nil {
		logger.WithError(err).Error("Placement decision failed")
	} else {
		logger.Info("✅ VM placement decision completed")
	}

	// Demonstrate auto-scaling decision
	decision, err := autoScaler.GetScalingDecision("web-service")
	if err != nil {
		logger.WithError(err).Error("Auto-scaling decision failed")
	} else {
		logger.WithFields(logrus.Fields{
			"action":        decision.Action,
			"current_scale": decision.CurrentScale,
			"target_scale":  decision.TargetScale,
			"confidence":    decision.Confidence,
		}).Info("✅ Auto-scaling decision completed")
	}

	// Demonstrate healing trigger
	healingDecision, err := healingController.TriggerHealing("database-cluster", "Manual health check")
	if err != nil {
		logger.WithError(err).Error("Healing trigger failed")
	} else {
		logger.WithFields(logrus.Fields{
			"strategy":   healingDecision.Strategy,
			"confidence": healingDecision.Confidence,
			"reason":     healingDecision.Reason,
		}).Info("✅ Self-healing triggered")
	}

	// Demonstrate policy evaluation
	policyContext := &policy.PolicyEvaluationContext{
		RequestID:    "demo-request-001",
		Timestamp:    time.Now(),
		ResourceType: policy.ResourceTypeVM,
		ResourceID:   "web-vm-001",
		Labels: map[string]string{
			"environment": "production",
			"tier":        "web",
		},
		Metrics: map[string]float64{
			"cpu_usage":    0.85,
			"memory_usage": 0.70,
		},
	}

	policyResults, err := policyEngine.EvaluateAllPolicies(policyContext)
	if err != nil {
		logger.WithError(err).Error("Policy evaluation failed")
	} else {
		logger.WithField("results_count", len(policyResults)).Info("✅ Policy evaluation completed")
		for _, result := range policyResults {
			if result.Matched {
				logger.WithFields(logrus.Fields{
					"policy_name": result.PolicyName,
					"score":       result.Score,
					"actions":     len(result.Actions),
				}).Info("Policy matched")
			}
		}
	}

	// 8. Show system status
	logger.Info("=== System Status ===")

	engineStatus := orchestrationEngine.GetStatus()
	logger.WithFields(logrus.Fields{
		"state":            engineStatus.State,
		"active_policies":  engineStatus.ActivePolicies,
		"events_processed": engineStatus.EventsProcessed,
	}).Info("Orchestration Engine Status")

	autoScalerStatus := autoScaler.GetStatus()
	logger.WithFields(logrus.Fields{
		"running":           autoScalerStatus.Running,
		"targets_count":     autoScalerStatus.TargetsCount,
		"decisions_count":   autoScalerStatus.DecisionsCount,
		"predictions_count": autoScalerStatus.PredictionsCount,
	}).Info("Auto-Scaler Status")

	healingStatus := healingController.GetStatus()
	logger.WithFields(logrus.Fields{
		"running":           healingStatus.Running,
		"targets_count":     healingStatus.TargetsCount,
		"active_healings":   healingStatus.ActiveHealings,
		"success_rate":      healingStatus.SuccessRate,
	}).Info("Healing Controller Status")

	policyEngineStatus := policyEngine.GetStatus()
	logger.WithFields(logrus.Fields{
		"total_policies":    policyEngineStatus.TotalPolicies,
		"enabled_policies":  policyEngineStatus.EnabledPolicies,
		"total_rules":       policyEngineStatus.TotalRules,
		"evaluations_count": policyEngineStatus.EvaluationsCount,
	}).Info("Policy Engine Status")

	// 9. Cleanup
	logger.Info("=== Cleaning Up ===")

	if err := autoScaler.StopMonitoring(); err != nil {
		logger.WithError(err).Error("Failed to stop auto-scaling")
	}

	if err := healingController.StopMonitoring(); err != nil {
		logger.WithError(err).Error("Failed to stop healing controller")
	}

	if err := orchestrationEngine.Stop(ctx); err != nil {
		logger.WithError(err).Error("Failed to stop orchestration engine")
	}

	if err := eventBus.Disconnect(); err != nil {
		logger.WithError(err).Error("Failed to disconnect event bus")
	}

	logger.Info("🎉 Advanced Orchestration Demo completed successfully!")
	return nil
}