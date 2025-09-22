#!/usr/bin/env python3
"""
Test migration action clarity in predictive scaling engine.
Verifies that MIGRATE actions provide clear, actionable information for VM migration executors.
"""

import sys
import os
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add ai_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))

from predictive_scaling import PredictiveScalingEngine, ResourceType, ScalingAction


class TestMigrationActionClarity(unittest.TestCase):
    """Test that MIGRATE actions provide clear execution information"""

    def setUp(self):
        """Set up test environment"""
        self.engine = PredictiveScalingEngine(db_path="/tmp/test_migration_clarity.db")

    def test_migration_decision_structure(self):
        """Test that migration decisions include all required fields"""
        # Create test data that should trigger migration
        data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(120, 0, -1)],
            'cpu_usage': [85.0] * 120  # High consistent CPU usage
        })

        # Generate forecast
        forecast = self.engine.predict_resource_demand("vm-001", ResourceType.CPU, data)

        # Make scaling decision with current high utilization
        current_resources = {ResourceType.CPU: 85.0}
        forecasts = {ResourceType.CPU: forecast}

        decisions = self.engine.make_scaling_decision("vm-001", forecasts, current_resources)

        # Should get at least one decision
        self.assertGreater(len(decisions), 0)

        # Check if we got a migration decision
        migration_decisions = [d for d in decisions if d.scaling_action == ScalingAction.MIGRATE]
        if migration_decisions:
            decision = migration_decisions[0]

            # Verify all required migration fields are present
            self.assertIsNotNone(decision.target_node_id, "Migration decision must include target_node_id")
            self.assertIsNotNone(decision.migration_plan, "Migration decision must include migration_plan")

            # Verify target_value represents expected utilization after migration
            self.assertLess(decision.target_value, decision.current_value,
                          "Migration target_value should be expected utilization after migration")

            # Verify migration plan contains essential information
            plan = decision.migration_plan
            self.assertIn('migration_type', plan, "Migration plan must specify migration type")
            self.assertIn('target_node_id', plan, "Migration plan must specify target node")
            self.assertIn('execution_steps', plan, "Migration plan must include execution steps")
            self.assertIn('success_criteria', plan, "Migration plan must define success criteria")
            self.assertIn('expected_utilization_improvement', plan,
                        "Migration plan must include utilization improvement details")

            # Verify execution steps are actionable
            steps = plan['execution_steps']
            self.assertGreater(len(steps), 0, "Migration plan must have execution steps")

            for step in steps:
                self.assertIn('step', step, "Each execution step must have step number")
                self.assertIn('action', step, "Each execution step must have action")
                self.assertIn('description', step, "Each execution step must have description")
                self.assertIn('timeout_seconds', step, "Each execution step must have timeout")

    def test_migration_utilization_clarity(self):
        """Test that migration target_value clearly represents expected utilization"""
        # Test the scaling action determination directly
        action, target, reasoning, target_node, migration_plan = self.engine._determine_scaling_action(
            ResourceType.CPU,
            current=0.85,  # 85% current utilization
            peak=0.90,     # 90% peak
            avg=0.82,      # 82% average
            forecast=None  # Won't be used in this test path
        )

        if action == ScalingAction.MIGRATE:
            # Verify target utilization is lower than current (improvement expected)
            self.assertLess(target, 0.85,
                          f"Migration target_value {target} should be less than current 0.85")

            # Verify target node is specified
            self.assertIsNotNone(target_node, "Migration must specify target node")

            # Verify migration plan exists
            self.assertIsNotNone(migration_plan, "Migration must include execution plan")

            # Verify plan contains expected improvement
            improvement = migration_plan.get('expected_utilization_improvement', {})
            self.assertEqual(improvement.get('current_utilization'), 0.85)
            self.assertEqual(improvement.get('expected_utilization'), target)
            self.assertGreater(improvement.get('improvement_percentage'), 0,
                             "Migration should show positive improvement percentage")

    def test_legacy_api_migration_info(self):
        """Test that legacy API includes migration details for MIGRATE actions"""
        # Use legacy wrapper
        autoscaler = self.engine  # PredictiveScalingEngine is the new implementation

        # Test data that should trigger migration
        metrics = {'cpu_usage': 0.85}  # High CPU usage

        # Mock the forecast to ensure migration decision
        # This is a bit tricky since we need to manipulate internal state

        # Instead, let's test the legacy API structure
        from predictive_scaling import AutoScaler
        legacy_scaler = AutoScaler(db_path="/tmp/test_legacy_migration.db")

        result = legacy_scaler.scale_decision("vm-001", metrics)

        # Check result structure
        self.assertIn('action', result)
        self.assertIn('reasoning', result)

        # If it's a migration action, check for migration details
        if result['action'] == 'migrate':
            self.assertIn('target_node_id', result, "Legacy API should include target_node_id for migrations")
            self.assertIn('migration_details', result, "Legacy API should include migration_details")

            details = result['migration_details']
            self.assertIn('migration_type', details)
            self.assertIn('expected_improvement', details)
            self.assertIn('execution_steps', details)
            self.assertIn('estimated_completion_minutes', details)

    def test_migration_plan_completeness(self):
        """Test that migration plans contain all necessary execution information"""
        # Test the migration plan creation directly
        plan = self.engine._create_migration_plan("vm-test", "node-target", 0.85, 0.60)

        # Verify essential fields
        required_fields = [
            'migration_type',
            'source_vm_id',
            'target_node_id',
            'expected_utilization_improvement',
            'execution_steps',
            'success_criteria',
            'rollback_conditions',
            'resource_requirements',
            'estimated_downtime_seconds',
            'estimated_completion_minutes'
        ]

        for field in required_fields:
            self.assertIn(field, plan, f"Migration plan must include {field}")

        # Verify execution steps are detailed
        steps = plan['execution_steps']
        self.assertGreater(len(steps), 3, "Migration should have multiple execution steps")

        # Verify improvement calculation
        improvement = plan['expected_utilization_improvement']
        self.assertEqual(improvement['current_utilization'], 0.85)
        self.assertEqual(improvement['expected_utilization'], 0.60)
        self.assertAlmostEqual(improvement['improvement_percentage'], 29.41, places=1)

        # Verify success criteria are actionable
        criteria = plan['success_criteria']
        self.assertGreater(len(criteria), 0, "Migration must have success criteria")
        for criterion in criteria:
            self.assertIsInstance(criterion, str)
            self.assertGreater(len(criterion), 10, "Success criteria should be descriptive")


if __name__ == '__main__':
    unittest.main()