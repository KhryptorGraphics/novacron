#!/usr/bin/env python3
"""
Specific example demonstrating migration action clarity with forced migration scenario.
"""

import sys
import os
from datetime import datetime, timedelta

# Add ai_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))

from predictive_scaling import PredictiveScalingEngine, ResourceType, ScalingAction, ResourceForecast


def demonstrate_migration_action():
    """Demonstrate migration action with clear execution details"""

    print("🔄 Migration Action Clarity Example")
    print("=" * 50)

    engine = PredictiveScalingEngine(db_path="/tmp/migration_specific.db")

    # Test the migration logic directly
    print("\n1️⃣ Testing Migration Decision Logic:")

    # This should trigger a migration (high CPU + high confidence)
    action, target, reasoning, target_node, migration_plan = engine._determine_scaling_action(
        ResourceType.CPU,
        current=0.75,   # 75% current utilization
        peak=0.85,      # 85% peak
        avg=0.78,       # 78% average - above migration threshold
        forecast=None   # Will use mock forecast with high confidence
    )

    # Create a mock forecast with high confidence to trigger migration
    mock_forecast = ResourceForecast(
        resource_type=ResourceType.CPU,
        vm_id="vm-test",
        forecast_horizon=60,
        predicted_values=[0.78] * 60,
        confidence_intervals=[(0.75, 0.82)] * 60,
        peak_prediction=0.85,
        peak_time=datetime.now() + timedelta(minutes=30),
        valley_prediction=0.72,
        valley_time=datetime.now() + timedelta(minutes=45),
        forecast_accuracy=0.9,  # High confidence
        model_used="test"
    )

    # Test with high confidence forecast - values that will trigger migration
    action, target, reasoning, target_node, migration_plan = engine._determine_scaling_action(
        ResourceType.CPU,
        current=0.80,   # Above 0.75 threshold
        peak=0.75,      # Below scale_up threshold (0.8)
        avg=0.78,       # Above 0.7 threshold
        forecast=mock_forecast  # High confidence forecast
    )

    print(f"   Action: {action.value}")
    print(f"   Target Utilization: {target:.3f}")
    print(f"   Reasoning: {reasoning}")

    if action == ScalingAction.MIGRATE:
        print(f"   Target Node: {target_node}")
        print(f"   Migration Plan Available: {'Yes' if migration_plan else 'No'}")

        if migration_plan:
            print("\n2️⃣ Migration Plan Details:")

            # Show expected improvement
            improvement = migration_plan['expected_utilization_improvement']
            print(f"   📈 Utilization Improvement:")
            print(f"     • Current: {improvement['current_utilization']:.1f}%")
            print(f"     • Expected: {improvement['expected_utilization']:.1f}%")
            print(f"     • Improvement: {improvement['improvement_percentage']:.1f}%")

            print(f"   🏗️ Migration Type: {migration_plan['migration_type']}")
            print(f"   ⏱️ Completion Time: {migration_plan['estimated_completion_minutes']} minutes")
            print(f"   ⏸️ Downtime: {migration_plan['estimated_downtime_seconds']} seconds")

            print("\n3️⃣ Execution Steps:")
            for step in migration_plan['execution_steps']:
                print(f"     Step {step['step']}: {step['action']}")
                print(f"       • {step['description']}")
                print(f"       • Timeout: {step['timeout_seconds']}s")

            print("\n4️⃣ Success Criteria:")
            for i, criterion in enumerate(migration_plan['success_criteria'], 1):
                print(f"     {i}. {criterion}")

            print("\n5️⃣ Rollback Conditions:")
            for i, condition in enumerate(migration_plan['rollback_conditions'], 1):
                print(f"     {i}. {condition}")

            print("\n6️⃣ Resource Requirements:")
            for req, value in migration_plan['resource_requirements'].items():
                print(f"     • {req}: {value}")

    print("\n" + "=" * 50)

    # Test the legacy API with migration details
    print("\n7️⃣ Legacy API Response:")
    from predictive_scaling import ScalingDecision

    # Create a sample migration decision
    if action == ScalingAction.MIGRATE and migration_plan:
        decision = ScalingDecision(
            decision_id="test-migration-001",
            vm_id="vm-production-001",
            resource_type=ResourceType.CPU,
            scaling_action=action,
            current_value=0.75,
            target_value=target,
            confidence=0.9,
            reasoning=reasoning,
            cost_impact=0.25,
            performance_impact=2.5,
            urgency_score=0.8,
            execution_time=datetime.now() + timedelta(minutes=5),
            rollback_plan={'original_value': 0.75},
            created_at=datetime.now(),
            target_node_id=target_node,
            migration_plan=migration_plan
        )

        # Show how this would appear to legacy systems
        legacy_format = {
            'action': decision.scaling_action.value,
            'resource': decision.resource_type.value,
            'current': decision.current_value,
            'target': decision.target_value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }

        # Add migration-specific info
        legacy_format.update({
            'target_node_id': decision.target_node_id,
            'migration_details': {
                'migration_type': decision.migration_plan.get('migration_type'),
                'expected_improvement': decision.migration_plan.get('expected_utilization_improvement'),
                'execution_steps': decision.migration_plan.get('execution_steps'),
                'success_criteria': decision.migration_plan.get('success_criteria'),
                'estimated_completion_minutes': decision.migration_plan.get('estimated_completion_minutes')
            }
        })

        print("   Legacy API would receive:")
        print(f"     • Action: {legacy_format['action']}")
        print(f"     • Current: {legacy_format['current']*100:.0f}%")
        print(f"     • Target: {legacy_format['target']*100:.0f}%")
        print(f"     • Target Node: {legacy_format['target_node_id']}")
        print(f"     • Migration Type: {legacy_format['migration_details']['migration_type']}")

        expected_improvement = legacy_format['migration_details']['expected_improvement']
        print(f"     • Expected Improvement: {expected_improvement['improvement_percentage']:.1f}%")
        print(f"     • Completion Time: {legacy_format['migration_details']['estimated_completion_minutes']} minutes")

        print("\n✅ Key Improvements Demonstrated:")
        print("   • target_value now shows expected utilization AFTER migration")
        print("   • target_node_id clearly identifies destination")
        print("   • migration_plan provides complete execution roadmap")
        print("   • Success criteria define measurable outcomes")
        print("   • Rollback conditions provide safety guarantees")
        print("   • Time estimates help with planning")

    else:
        print("   ⚠️ Migration not triggered in this scenario")
        print("   This would be a regular scaling action instead")


if __name__ == "__main__":
    demonstrate_migration_action()