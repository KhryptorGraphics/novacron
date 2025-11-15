#!/usr/bin/env python3
"""
DWCP Master Training Orchestrator
Trains all 4 neural models with 98% accuracy targets

Usage:
    python train_dwcp_models.py --data-path data/dwcp_metrics.csv --target-accuracy 0.98
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DWCPModelOrchestrator:
    """Master orchestrator for training all DWCP neural models"""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None

    def validate_data_schema(self, data_path):
        """Validate input data against schema"""
        logger.info("Validating data schema...")

        try:
            df = pd.read_csv(data_path)

            required_cols = [
                'timestamp', 'rtt_ms', 'jitter_ms', 'throughput_mbps',
                'packet_loss', 'link_type', 'network_tier', 'congestion_window',
                'hde_compression_ratio', 'hde_delta_hit_rate', 'amst_transfer_rate',
                'uptime_pct', 'failure_rate', 'retransmits', 'error_budget_burn_rate',
                'consensus_latency_ms'
            ]

            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return False

            logger.info(f"Data schema validation passed - {len(df)} records")
            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

    def train_model(self, model_name, script_path, args):
        """Train a single model"""
        import subprocess

        logger.info(f"Starting training for {model_name}...")
        start = time.time()

        try:
            # Build command
            cmd = ['python', script_path] + args

            logger.info(f"Command: {' '.join(cmd)}")

            # Run training script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            duration = time.time() - start

            logger.info(f"{model_name} training completed in {duration:.2f}s")

            return {
                'model': model_name,
                'success': True,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.CalledProcessError as e:
            duration = time.time() - start
            logger.error(f"{model_name} training failed: {e}")

            return {
                'model': model_name,
                'success': False,
                'duration': duration,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }

    def train_all_models_sequential(self):
        """Train all models sequentially"""
        logger.info("Training models sequentially...")

        models = self.get_model_configs()

        for model_name, config in models.items():
            if model_name not in self.config['models_to_train']:
                logger.info(f"Skipping {model_name} (not selected)")
                continue

            result = self.train_model(model_name, config['script'], config['args'])
            self.results[model_name] = result

    def train_all_models_parallel(self):
        """Train all models in parallel"""
        logger.info("Training models in parallel...")

        models = self.get_model_configs()

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}

            for model_name, config in models.items():
                if model_name not in self.config['models_to_train']:
                    logger.info(f"Skipping {model_name} (not selected)")
                    continue

                future = executor.submit(
                    self.train_model,
                    model_name,
                    config['script'],
                    config['args']
                )
                futures[future] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    self.results[model_name] = result
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    self.results[model_name] = {
                        'model': model_name,
                        'success': False,
                        'error': str(e)
                    }

    def get_model_configs(self):
        """Get training configurations for all models"""
        base_path = Path(__file__).parent.parent
        output_dir = Path(self.config['output_dir'])

        models = {
            'bandwidth_predictor': {
                'script': str(base_path / 'core/network/dwcp/prediction/training/train_lstm.py'),
                'args': [
                    '--data-path', self.config['data_path'],
                    '--output', str(output_dir / 'bandwidth_predictor.keras'),
                    '--target-correlation', str(self.config['target_accuracy']),
                    '--target-mape', '5.0',
                    '--epochs', str(self.config['epochs']),
                    '--batch-size', str(self.config['batch_size']),
                    '--seed', str(self.config['seed'])
                ]
            },
            'compression_selector': {
                'script': str(base_path / 'core/network/dwcp/compression/training/train_compression_selector.py'),
                'args': [
                    '--data-path', self.config['data_path'],
                    '--output', str(output_dir / 'compression_selector.keras'),
                    '--target-accuracy', str(self.config['target_accuracy']),
                    '--epochs', str(self.config['epochs'] // 2),  # Half epochs for policy net
                    '--batch-size', str(self.config['batch_size']),
                    '--seed', str(self.config['seed'])
                ]
            },
            'reliability_detector': {
                'script': str(base_path / 'core/network/dwcp/monitoring/training/train_isolation_forest.py'),
                'args': [
                    '--data-path', self.config['data_path'],
                    '--incidents-path', self.config['incidents_path'],
                    '--output', str(output_dir / 'reliability_model.pkl'),
                    '--target-recall', str(self.config['target_accuracy']),
                    '--target-pr-auc', '0.90',
                    '--seed', str(self.config['seed'])
                ]
            },
            'consensus_latency': {
                'script': str(base_path / 'core/network/dwcp/monitoring/training/train_lstm_autoencoder.py'),
                'args': [
                    '--data-path', self.config['data_path'],
                    '--output', str(output_dir / 'consensus_latency.keras'),
                    '--target-accuracy', str(self.config['target_accuracy']),
                    '--epochs', str(self.config['epochs']),
                    '--batch-size', str(self.config['batch_size']),
                    '--window-size', '20',
                    '--seed', str(self.config['seed'])
                ]
            }
        }

        return models

    def load_evaluation_reports(self):
        """Load evaluation reports from trained models"""
        logger.info("Loading evaluation reports...")

        output_dir = Path(self.config['output_dir'])
        reports = {}

        model_files = {
            'bandwidth_predictor': 'bandwidth_predictor_report.json',
            'compression_selector': 'compression_selector_policy_net_report.json',
            'reliability_detector': 'reliability_model_report.json',
            'consensus_latency': 'consensus_latency_lstm_autoencoder_report.json'
        }

        for model_name, report_file in model_files.items():
            report_path = output_dir / report_file

            if report_path.exists():
                with open(report_path, 'r') as f:
                    reports[model_name] = json.load(f)
                logger.info(f"Loaded report for {model_name}")
            else:
                logger.warning(f"Report not found for {model_name}: {report_path}")
                reports[model_name] = {'success': False, 'error': 'Report not found'}

        return reports

    def generate_master_report(self):
        """Generate aggregated master report"""
        logger.info("Generating master report...")

        # Load individual evaluation reports
        evaluation_reports = self.load_evaluation_reports()

        # Count successes
        models_passed = sum(
            1 for report in evaluation_reports.values()
            if report.get('success', False)
        )

        all_targets_met = models_passed == len(self.config['models_to_train'])

        # Build master report
        master_report = {
            'training_session': f"dwcp_neural_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'overall_success': all_targets_met,
            'models_trained': len(self.config['models_to_train']),
            'models_passed': models_passed,
            'total_training_time_seconds': self.end_time - self.start_time if self.end_time else 0,
            'config': self.config,
            'training_results': self.results,
            'evaluation_reports': evaluation_reports,
            'production_readiness': {
                'all_targets_met': all_targets_met,
                'integration_tests_passed': None,  # To be filled by integration tests
                'go_api_compatibility': None,  # To be filled by Go tests
                'deployment_recommendation': 'APPROVED' if all_targets_met else 'FAILED'
            }
        }

        # Save master report
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'master_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(master_report, f, indent=2)

        logger.info(f"Master report saved to {report_path}")

        # Generate markdown summary
        md_path = output_dir / 'master_training_report.md'
        self.generate_markdown_summary(master_report, md_path)

        return master_report

    def generate_markdown_summary(self, report, output_path):
        """Generate human-readable markdown summary"""
        md = f"""# DWCP Neural Training Report

**Session:** {report['training_session']}
**Date:** {datetime.now().isoformat()}
**Status:** {'✅ SUCCESS' if report['overall_success'] else '❌ FAILED'}

---

## Summary

- **Models Trained:** {report['models_trained']}
- **Models Passed:** {report['models_passed']}
- **Total Training Time:** {report['total_training_time_seconds']:.2f}s

---

## Model Results

"""

        for model_name, eval_report in report['evaluation_reports'].items():
            status = '✅' if eval_report.get('success', False) else '❌'
            md += f"### {status} {model_name}\n\n"

            if 'achieved_metrics' in eval_report:
                md += "**Metrics:**\n"
                for metric, value in eval_report['achieved_metrics'].items():
                    md += f"- {metric}: {value}\n"

            if 'training_time_seconds' in eval_report:
                md += f"- Training Time: {eval_report['training_time_seconds']:.2f}s\n"

            if 'model_size_mb' in eval_report:
                md += f"- Model Size: {eval_report['model_size_mb']:.2f} MB\n"

            md += "\n---\n\n"

        md += f"""## Deployment Recommendation

**Status:** {report['production_readiness']['deployment_recommendation']}

"""

        with open(output_path, 'w') as f:
            f.write(md)

        logger.info(f"Markdown summary saved to {output_path}")

    def run(self):
        """Run the complete training orchestration"""
        logger.info("=" * 80)
        logger.info("DWCP Neural Model Training Orchestrator")
        logger.info("=" * 80)

        self.start_time = time.time()

        # Validate data
        if not self.validate_data_schema(self.config['data_path']):
            logger.error("Data schema validation failed. Aborting.")
            return 1

        # Train models
        if self.config['parallel']:
            self.train_all_models_parallel()
        else:
            self.train_all_models_sequential()

        self.end_time = time.time()

        # Generate master report
        master_report = self.generate_master_report()

        # Print summary
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Overall Success: {master_report['overall_success']}")
        logger.info(f"Models Passed: {master_report['models_passed']}/{master_report['models_trained']}")
        logger.info(f"Total Time: {master_report['total_training_time_seconds']:.2f}s")
        logger.info("=" * 80)

        return 0 if master_report['overall_success'] else 1


def main():
    parser = argparse.ArgumentParser(description='DWCP Master Training Orchestrator')
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--output-dir', default='checkpoints/dwcp_v1', help='Output directory for models')
    parser.add_argument('--incidents-path', default='data/labeled_incidents.json',
                       help='Path to labeled incidents JSON')
    parser.add_argument('--models', default='bandwidth,compression,reliability,latency',
                       help='Comma-separated list of models to train')
    parser.add_argument('--target-accuracy', type=float, default=0.98, help='Target accuracy')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--parallel', action='store_true', help='Train models in parallel')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Parse model list
    models_to_train = [m.strip() for m in args.models.split(',')]

    model_map = {
        'bandwidth': 'bandwidth_predictor',
        'compression': 'compression_selector',
        'reliability': 'reliability_detector',
        'latency': 'consensus_latency'
    }

    models_to_train = [model_map.get(m, m) for m in models_to_train]

    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'incidents_path': args.incidents_path,
        'models_to_train': models_to_train,
        'target_accuracy': args.target_accuracy,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'parallel': args.parallel,
        'seed': args.seed
    }

    orchestrator = DWCPModelOrchestrator(config)
    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
