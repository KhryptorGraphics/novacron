#!/usr/bin/env python3
"""
Example demonstrating improved migration action clarity in PredictiveScalingEngine.

This example shows how MIGRATE actions now provide clear, actionable information
for VM migration executors, including target nodes and detailed execution plans.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import json

# Add ai_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))

from predictive_scaling import PredictiveScalingEngine, ResourceType, ScalingAction


def demonstrate_migration_clarity():
    """Demonstrate how migration actions provide clear execution information"""

    print("🚀 Migration Action Clarity Demonstration")
    print("=" * 50)

    # Initialize the predictive scaling engine
    engine = PredictiveScalingEngine(db_path="/tmp/demo_migration.db")

    # Create sample historical data that will trigger migration
    print("\n📊 Creating sample data with high CPU utilization...")
    historical_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(120, 0, -1)],
        'cpu_usage': [82.0 + (i % 10) for i in range(120)]  # High CPU with variation
    })

    print(f"   • Historical data: {len(historical_data)} data points")
    print(f"   • Average CPU usage: {historical_data['cpu_usage'].mean():.1f}%")
    print(f"   • Peak CPU usage: {historical_data['cpu_usage'].max():.1f}%")

    # Generate forecast
    print("\n🔮 Generating resource demand forecast...")
    forecast = engine.predict_resource_demand("vm-production-001", ResourceType.CPU, historical_data)

    print(f"   • Forecast horizon: {forecast.forecast_horizon} minutes")
    print(f"   • Predicted peak: {forecast.peak_prediction:.1f}%")
    print(f"   • Model used: {forecast.model_used}")
    print(f"   • Forecast accuracy: {forecast.forecast_accuracy:.3f}")

    # Make scaling decisions
    print("\n⚖️ Making scaling decisions...")
    current_resources = {ResourceType.CPU: 85.0}  # Current high utilization
    forecasts = {ResourceType.CPU: forecast}

    decisions = engine.make_scaling_decision("vm-production-001", forecasts, current_resources)

    print(f"   • Number of decisions generated: {len(decisions)}")

    # Analyze decisions
    for i, decision in enumerate(decisions):
        print(f"\n📋 Decision {i+1}: {decision.scaling_action.value.upper()}")
        print(f"   • Resource: {decision.resource_type.value}")
        print(f"   • Current utilization: {decision.current_value:.1f}%")
        print(f"   • Target utilization: {decision.target_value:.1f}%")
        print(f"   • Confidence: {decision.confidence:.3f}")
        print(f"   • Reasoning: {decision.reasoning}")
        print(f"   • Cost impact: ${decision.cost_impact:.2f}")
        print(f"   • Performance impact: {decision.performance_impact:.2f}")
        print(f"   • Urgency score: {decision.urgency_score:.3f}")

        # Show migration-specific details if this is a migration
        if decision.scaling_action == ScalingAction.MIGRATE:
            print("\n🔄 Migration Details:")
            print(f"   • Target node: {decision.target_node_id}")

            if decision.migration_plan:
                plan = decision.migration_plan
                print(f"   • Migration type: {plan['migration_type']}")

                # Show expected improvement
                improvement = plan['expected_utilization_improvement']
                print(f"   • Expected improvement: {improvement['improvement_percentage']:.1f}%")
                print(f"     - Current: {improvement['current_utilization']:.1f}%")
                print(f"     - Expected after migration: {improvement['expected_utilization']:.1f}%")

                # Show execution timeline
                print(f"   • Estimated completion: {plan['estimated_completion_minutes']} minutes")
                print(f"   • Estimated downtime: {plan['estimated_downtime_seconds']} seconds")

                # Show execution steps
                print("\n   📝 Execution Steps:")
                for step in plan['execution_steps']:
                    print(f"     {step['step']}. {step['action']}")
                    print(f"        Description: {step['description']}")
                    print(f"        Timeout: {step['timeout_seconds']}s")

                # Show success criteria
                print("\n   ✅ Success Criteria:")
                for criterion in plan['success_criteria']:
                    print(f"     • {criterion}")

                # Show rollback conditions
                print("\n   🔙 Rollback Conditions:")
                for condition in plan['rollback_conditions']:
                    print(f"     • {condition}")

                # Show resource requirements
                print("\n   💾 Resource Requirements:")
                requirements = plan['resource_requirements']
                for req, value in requirements.items():
                    print(f"     • {req}: {value}")

    # Demonstrate legacy API compatibility
    print("\n🔗 Legacy API Compatibility:")
    print("-" * 30)

    # Test with legacy wrapper
    from predictive_scaling import AutoScaler
    legacy_scaler = AutoScaler(db_path="/tmp/demo_legacy.db")

    result = legacy_scaler.scale_decision("vm-production-001", {'cpu_usage': 85.0})

    print(f"   • Action: {result['action']}")
    print(f"   • Reasoning: {result['reasoning']}")

    if 'migration_details' in result:
        print("\n   🔄 Migration Details (Legacy Format):")
        details = result['migration_details']
        print(f"     • Migration type: {details.get('migration_type', 'N/A')}")
        print(f"     • Completion time: {details.get('estimated_completion_minutes', 'N/A')} minutes")

        if 'expected_improvement' in details:
            improvement = details['expected_improvement']
            print(f"     • Expected improvement: {improvement.get('improvement_percentage', 0):.1f}%")


def show_migration_vs_scaling_comparison():
    """Show the difference between migration and scaling actions"""

    print("\n" + "=" * 60)
    print("📊 Migration vs Scaling Action Comparison")
    print("=" * 60)

    engine = PredictiveScalingEngine(db_path="/tmp/demo_comparison.db")

    # Test different scenarios
    scenarios = [
        {
            "name": "High CPU - Migration Candidate",
            "current": 0.85,
            "peak": 0.90,
            "avg": 0.82,
            "description": "High sustained CPU usage with good forecast confidence"
        },
        {
            "name": "Peak Load - Scale Up",
            "current": 0.60,
            "peak": 0.95,
            "avg": 0.65,
            "description": "Occasional peaks requiring more capacity"
        },
        {
            "name": "Low Utilization - Scale Down",
            "current": 0.40,
            "peak": 0.25,
            "avg": 0.20,
            "description": "Consistently low usage allowing resource reduction"
        }
    ]

    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Current: {scenario['current']*100:.0f}%, Peak: {scenario['peak']*100:.0f}%, Average: {scenario['avg']*100:.0f}%")

        # Simulate the decision process
        action, target, reasoning, target_node, migration_plan = engine._determine_scaling_action(
            ResourceType.CPU,
            scenario['current'],
            scenario['peak'],
            scenario['avg'],
            None  # forecast not used in this simplified test
        )

        print(f"\n   🎬 Action: {action.value}")
        print(f"   🎯 Target Value: {target:.2f}")
        print(f"   💭 Reasoning: {reasoning}")

        if action == ScalingAction.MIGRATE:
            print(f"   🏠 Target Node: {target_node}")
            print(f"   📋 Migration Plan: Available ({len(migration_plan)} fields)")

            # Show key migration plan details
            if migration_plan:
                improvement = migration_plan['expected_utilization_improvement']
                print(f"   📈 Expected Improvement: {improvement['improvement_percentage']:.1f}%")
                print(f"   ⏱️ Estimated Time: {migration_plan['estimated_completion_minutes']} minutes")

        print()


if __name__ == "__main__":
    print("Migration Action Clarity Example")
    print("This demonstrates how MIGRATE actions now provide clear, actionable information")
    print()

    try:
        demonstrate_migration_clarity()
        show_migration_vs_scaling_comparison()

        print("\n✅ Migration action clarity demonstration completed successfully!")
        print("\nKey improvements:")
        print("  • Migration decisions include target_node_id")
        print("  • target_value represents expected utilization after migration")
        print("  • Detailed migration_plan with execution steps")
        print("  • Success criteria and rollback conditions")
        print("  • Resource requirements and time estimates")
        print("  • Legacy API compatibility maintained")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()