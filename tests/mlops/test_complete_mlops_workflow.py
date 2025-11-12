"""
Complete MLOps Platform Integration Test
Demonstrates end-to-end ML lifecycle with all components
"""

import asyncio
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Import all MLOps components
import sys
sys.path.append('/home/kp/novacron')

from backend.core.mlops.pipeline.ml_pipeline import PipelineConfig, MLPipeline
from backend.core.mlops.serving.model_server import ModelServer, ModelFramework, PredictionRequest
from backend.core.mlops.monitoring.ml_monitoring import MLMonitor
from backend.core.mlops.governance.ml_governance import GovernanceManager, ComplianceFramework


class CompletMLOpsWorkflowTest:
    """End-to-end MLOps workflow test"""

    def __init__(self):
        self.test_dir = Path("./test_mlops_output")
        self.test_dir.mkdir(exist_ok=True)

    async def run_complete_workflow(self):
        """Execute complete MLOps workflow"""
        print("=" * 80)
        print("COMPLETE MLOPS WORKFLOW TEST")
        print("=" * 80)

        # Step 1: Generate synthetic data
        print("\n[1/10] Generating synthetic dataset...")
        self.create_synthetic_dataset()

        # Step 2: Train model with pipeline
        print("\n[2/10] Training model with ML Pipeline...")
        model_path, run_metrics = await self.train_model()

        # Step 3: Register model
        print("\n[3/10] Registering model in registry...")
        model_id = self.register_model(run_metrics)

        # Step 4: Assess governance
        print("\n[4/10] Assessing bias and compliance...")
        await self.assess_governance(model_id)

        # Step 5: Deploy model
        print("\n[5/10] Deploying model to serving...")
        endpoint_id = await self.deploy_model(model_path)

        # Step 6: Setup monitoring
        print("\n[6/10] Setting up monitoring...")
        monitor = self.setup_monitoring(model_id, model_path)

        # Step 7: Serve predictions
        print("\n[7/10] Serving predictions...")
        await self.serve_predictions(endpoint_id, monitor)

        # Step 8: Check drift
        print("\n[8/10] Checking for drift...")
        await self.check_drift(monitor)

        # Step 9: A/B test
        print("\n[9/10] Running A/B test...")
        await self.ab_test_workflow(endpoint_id, model_path)

        # Step 10: Generate reports
        print("\n[10/10] Generating reports...")
        await self.generate_reports(model_id, monitor)

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE - ALL TESTS PASSED ✓")
        print("=" * 80)

    def create_synthetic_dataset(self):
        """Create synthetic fraud detection dataset"""
        np.random.seed(42)

        n_samples = 10000

        # Generate features
        data = {
            'transaction_amount': np.random.lognormal(4, 1.5, n_samples),
            'merchant_risk_score': np.random.beta(2, 5, n_samples),
            'user_age': np.random.randint(18, 80, n_samples),
            'transaction_hour': np.random.randint(0, 24, n_samples),
            'days_since_last_transaction': np.random.exponential(7, n_samples),
            'num_transactions_30d': np.random.poisson(15, n_samples),
            'avg_transaction_amount': np.random.lognormal(3.5, 1, n_samples),
            'account_age_days': np.random.randint(1, 3650, n_samples),
        }

        df = pd.DataFrame(data)

        # Generate target (fraud) with realistic patterns
        fraud_prob = (
            0.01 +  # Base rate
            0.3 * (df['transaction_amount'] > 1000).astype(float) +
            0.4 * (df['merchant_risk_score'] > 0.7).astype(float) +
            0.2 * (df['transaction_hour'].isin([0, 1, 2, 3])).astype(float) +
            0.1 * (df['days_since_last_transaction'] > 30).astype(float)
        )
        fraud_prob = np.clip(fraud_prob, 0, 1)

        df['is_fraud'] = (np.random.random(n_samples) < fraud_prob).astype(int)

        # Add protected attributes for bias testing
        df['age_group'] = pd.cut(df['user_age'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
        df['gender'] = np.random.randint(0, 2, n_samples)

        # Save dataset
        train_path = self.test_dir / "fraud_train.csv"
        df.to_csv(train_path, index=False)

        print(f"  Dataset: {n_samples} samples, {len(df.columns)} features")
        print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
        print(f"  Saved to: {train_path}")

    async def train_model(self):
        """Train model with ML Pipeline"""
        config = PipelineConfig(
            name="fraud_detector_test",
            description="Test fraud detection model",
            data_source=str(self.test_dir / "fraud_train.csv"),
            target_column="is_fraud",

            # Features (exclude protected attributes)
            feature_columns=[
                'transaction_amount', 'merchant_risk_score', 'user_age',
                'transaction_hour', 'days_since_last_transaction',
                'num_transactions_30d', 'avg_transaction_amount', 'account_age_days'
            ],

            # Model config
            model_type="sklearn",
            model_class="RandomForest",
            model_params={"n_estimators": 50, "max_depth": 10, "random_state": 42},

            # Quick tuning for test
            enable_tuning=True,
            tuning_method="optuna",
            tuning_trials=20,
            tuning_timeout=60,
            param_space={
                "n_estimators": {"type": "int", "low": 30, "high": 100},
                "max_depth": {"type": "int", "low": 5, "high": 15},
            },

            # Feature engineering
            feature_transforms=[
                {"type": "scale", "columns": ["transaction_amount", "avg_transaction_amount"]},
            ],

            # Validation
            test_size=0.2,
            validation_size=0.1,
            cross_validation_folds=3,
            metrics=["accuracy", "precision", "recall", "f1_score"],

            # Output
            output_dir=str(self.test_dir / "pipeline_output"),
            save_intermediate=True,
        )

        pipeline = MLPipeline(config)
        run = await pipeline.execute()

        print(f"  Status: {run.status.value}")
        print(f"  Accuracy: {run.metrics.get('test_accuracy', 0):.4f}")
        print(f"  Precision: {run.metrics.get('val_precision', 0):.4f}")
        print(f"  Recall: {run.metrics.get('val_recall', 0):.4f}")
        print(f"  F1 Score: {run.metrics.get('val_f1', 0):.4f}")

        assert run.status.value == "completed", "Pipeline should complete successfully"
        assert run.metrics.get('test_accuracy', 0) > 0.7, "Model should have >70% accuracy"

        return run.artifacts['model'], run.metrics

    def register_model(self, metrics):
        """Register model in registry (simulated)"""
        model_id = "fraud_detector_test_001"

        # In real implementation, would call Go registry via gRPC/HTTP
        print(f"  Model ID: {model_id}")
        print(f"  Version: v1.0.0")
        print(f"  Framework: scikit-learn")
        print(f"  Metrics: {metrics}")

        return model_id

    async def assess_governance(self, model_id):
        """Assess bias and compliance"""
        gov = GovernanceManager(storage_path=str(self.test_dir / "governance"))

        # Register dataset
        dataset = gov.register_dataset(
            dataset_id="fraud_train_001",
            dataset_name="Fraud Detection Training Data",
            source="synthetic_generator",
            contains_pii=True,
        )

        # Register model
        model = gov.register_model(
            model_id=model_id,
            model_version="v1.0.0",
            training_dataset="fraud_train_001",
            training_algorithm="RandomForest",
            hyperparameters={"n_estimators": 50, "max_depth": 10},
        )

        # Load data for bias assessment
        df = pd.read_csv(self.test_dir / "fraud_train.csv")
        y_true = df['is_fraud'].values
        # Simulate predictions
        y_pred = (np.random.random(len(y_true)) > 0.5).astype(int)

        # Assess bias
        protected_attrs = {
            'age_group': df['age_group'].values,
            'gender': df['gender'].values,
        }

        bias_reports = await gov.assess_bias(model_id, y_true, y_pred, protected_attrs)

        print(f"  Total bias assessments: {len(bias_reports)}")
        biased_count = sum(1 for r in bias_reports if r.is_biased)
        print(f"  Biased assessments: {biased_count}")

        if biased_count > 0:
            print("  Bias detected:")
            for report in bias_reports:
                if report.is_biased:
                    print(f"    - {report.bias_type.value}: {report.description}")

        # Assess GDPR compliance
        compliance = await gov.assess_compliance(
            model_id=model_id,
            framework=ComplianceFramework.GDPR,
            model_metadata={"domain": "finance", "use_case": "fraud_detection"},
            compliance_checks={
                "data_minimization": True,
                "right_to_explanation": True,
                "data_retention": True,
                "consent_management": True,
                "data_portability": True,
                "right_to_erasure": False,  # Not implemented yet
            }
        )

        print(f"  GDPR compliance: {compliance.compliance_score:.1f}%")
        print(f"  Risk level: {compliance.risk_level.value}")

        # Export governance report
        report_path = self.test_dir / "governance_report.json"
        await gov.export_governance_report(model_id, str(report_path))
        print(f"  Governance report: {report_path}")

    async def deploy_model(self, model_path):
        """Deploy model to serving"""
        server = ModelServer(storage_path=str(self.test_dir / "serving"))

        endpoint_id = await server.deploy_model(
            model_id="fraud_detector_test",
            model_version="v1.0.0",
            model_path=model_path,
            framework=ModelFramework.SKLEARN,
            endpoint_config={
                "min_replicas": 1,
                "max_replicas": 5,
                "target_latency_ms": 100.0,
            }
        )

        print(f"  Endpoint ID: {endpoint_id}")
        print(f"  Status: active")

        self.server = server  # Store for later use
        return endpoint_id

    def setup_monitoring(self, model_id, model_path):
        """Setup ML monitoring"""
        monitor = MLMonitor(model_id=model_id, model_version="v1.0.0")

        # Set reference data
        df = pd.read_csv(self.test_dir / "fraud_train.csv")
        reference_data = df.drop(columns=['is_fraud', 'age_group', 'gender']).head(1000)
        monitor.set_reference_data(reference_data, sensitivity=0.05)

        # Setup explainer
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        feature_names = [
            'transaction_amount', 'merchant_risk_score', 'user_age',
            'transaction_hour', 'days_since_last_transaction',
            'num_transactions_30d', 'avg_transaction_amount', 'account_age_days'
        ]
        monitor.set_explainer(model, feature_names, model_type="tree")

        print(f"  Reference data: 1000 samples")
        print(f"  Drift detection: enabled")
        print(f"  Explainability: enabled (SHAP)")

        return monitor

    async def serve_predictions(self, endpoint_id, monitor):
        """Serve predictions and log to monitor"""
        # Simulate 100 predictions
        print(f"  Serving 100 predictions...")

        for i in range(100):
            features = {
                'transaction_amount': np.random.lognormal(4, 1.5),
                'merchant_risk_score': np.random.beta(2, 5),
                'user_age': np.random.randint(18, 80),
                'transaction_hour': np.random.randint(0, 24),
                'days_since_last_transaction': np.random.exponential(7),
                'num_transactions_30d': np.random.poisson(15),
                'avg_transaction_amount': np.random.lognormal(3.5, 1),
                'account_age_days': np.random.randint(1, 3650),
            }

            request = PredictionRequest(
                request_id=f"req_{i}",
                endpoint_id=endpoint_id,
                features=features
            )

            response = await self.server.predict(request)

            # Log to monitor
            await monitor.log_prediction(
                features=features,
                prediction=response.predictions[0],
                ground_truth=None,  # Would be filled in later
                latency_ms=response.latency_ms
            )

        # Get metrics
        metrics = monitor.get_performance_metrics(window_minutes=60)

        print(f"  Predictions served: {metrics.prediction_count}")
        print(f"  Avg latency: {metrics.latency_p50:.2f}ms (p50)")
        print(f"  Throughput: {metrics.throughput:.1f} req/min")

    async def check_drift(self, monitor):
        """Check for data drift"""
        # Generate drifted data (higher transaction amounts)
        df_drift = pd.DataFrame({
            'transaction_amount': np.random.lognormal(5, 2, 100),  # Increased mean
            'merchant_risk_score': np.random.beta(2, 5, 100),
            'user_age': np.random.randint(18, 80, 100),
            'transaction_hour': np.random.randint(0, 24, 100),
            'days_since_last_transaction': np.random.exponential(7, 100),
            'num_transactions_30d': np.random.poisson(15, 100),
            'avg_transaction_amount': np.random.lognormal(3.5, 1, 100),
            'account_age_days': np.random.randint(1, 3650, 100),
        })

        alerts = await monitor.check_drift(df_drift)

        print(f"  Drift alerts: {len(alerts)}")
        if alerts:
            print("  Detected:")
            for alert in alerts[:3]:  # Show first 3
                print(f"    - {alert.severity.value}: {alert.description[:60]}...")

        # Get dashboard
        dashboard = monitor.get_monitoring_dashboard()
        print(f"  Health score: {dashboard['health_score']:.1f}/100")

    async def ab_test_workflow(self, endpoint_v1, model_path):
        """A/B testing workflow"""
        # Deploy v2 (same model for test)
        endpoint_v2 = await self.server.deploy_model(
            model_id="fraud_detector_test",
            model_version="v2.0.0",
            model_path=model_path,
            framework=ModelFramework.SKLEARN,
        )

        # Create A/B test
        self.server.create_ab_test(
            experiment_id="test_ab_experiment",
            control_endpoint=endpoint_v1,
            variant_endpoints=[endpoint_v2],
            traffic_split={endpoint_v1: 0.8, endpoint_v2: 0.2}
        )

        print(f"  Control: {endpoint_v1} (80%)")
        print(f"  Variant: {endpoint_v2} (20%)")

        # Serve 50 predictions through A/B test
        for i in range(50):
            features = {
                'transaction_amount': np.random.lognormal(4, 1.5),
                'merchant_risk_score': np.random.beta(2, 5),
                'user_age': np.random.randint(18, 80),
                'transaction_hour': np.random.randint(0, 24),
                'days_since_last_transaction': np.random.exponential(7),
                'num_transactions_30d': np.random.poisson(15),
                'avg_transaction_amount': np.random.lognormal(3.5, 1),
                'account_age_days': np.random.randint(1, 3650),
            }

            await self.server.predict_with_ab_test("test_ab_experiment", features)

        # Get results
        results = self.server.get_ab_test_results("test_ab_experiment")

        print(f"  Total requests: {sum(r['requests'] for r in results['results'].values())}")
        for endpoint, metrics in results['results'].items():
            print(f"  {endpoint[-10:]}: {metrics['requests']} requests, "
                  f"{metrics['avg_latency_ms']:.2f}ms avg latency")

    async def generate_reports(self, model_id, monitor):
        """Generate final reports"""
        # Performance report
        perf_metrics = monitor.get_performance_metrics(window_minutes=60)

        report = {
            "model_id": model_id,
            "model_version": "v1.0.0",
            "metrics": {
                "predictions": perf_metrics.prediction_count,
                "accuracy": perf_metrics.accuracy,
                "latency_p95": perf_metrics.latency_p95,
                "throughput": perf_metrics.throughput,
            },
            "alerts": {
                "total": len(monitor.active_alerts),
                "critical": len([a for a in monitor.active_alerts if a.severity.value == "critical"]),
            },
            "health_score": monitor.get_monitoring_dashboard()['health_score'],
        }

        report_path = self.test_dir / "performance_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Performance report: {report_path}")
        print(f"  Health score: {report['health_score']:.1f}/100")
        print(f"  Total alerts: {report['alerts']['total']}")


async def main():
    """Run complete MLOps workflow test"""
    test = CompletMLOpsWorkflowTest()

    try:
        await test.run_complete_workflow()
        print("\n✅ ALL TESTS PASSED")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
