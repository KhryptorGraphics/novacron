"""
TCS-FEEL Calibration Validation Script
Validates that all calibration targets have been met
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_calibration():
    """Validate TCS-FEEL calibration results"""

    logger.info("=" * 70)
    logger.info("TCS-FEEL CALIBRATION VALIDATION")
    logger.info("=" * 70)

    # Load calibration results
    json_path = Path("backend/ml/federated/CALIBRATION_REPORT.json")

    if not json_path.exists():
        logger.error("❌ Calibration report not found!")
        return False

    with open(json_path) as f:
        data = json.load(f)

    results = data['results']
    targets = {
        'accuracy': 0.963,
        'communication_reduction': 0.30,
        'convergence_speed': 1.5,
        'fairness': 0.80
    }

    logger.info("\n📊 Validation Results:")
    logger.info("-" * 70)

    all_passed = True

    # Check accuracy
    accuracy_met = results['final_accuracy'] >= targets['accuracy']
    status = "✅ PASS" if accuracy_met else "❌ FAIL"
    logger.info(f"Accuracy: {results['final_accuracy']*100:.2f}% (target: {targets['accuracy']*100:.1f}%) {status}")
    all_passed &= accuracy_met

    # Check communication reduction
    comm_met = results['communication_reduction'] >= targets['communication_reduction']
    status = "✅ PASS" if comm_met else "❌ FAIL"
    logger.info(f"Communication Reduction: {results['communication_reduction']*100:.1f}% (target: {targets['communication_reduction']*100:.1f}%) {status}")
    all_passed &= comm_met

    # Check convergence speed
    speed_met = results['convergence_speed'] >= targets['convergence_speed']
    status = "✅ PASS" if speed_met else "❌ FAIL"
    logger.info(f"Convergence Speed: {results['convergence_speed']:.1f}x (target: {targets['convergence_speed']:.1f}x) {status}")
    all_passed &= speed_met

    # Check fairness
    fairness_met = results['avg_fairness'] >= targets['fairness']
    status = "✅ PASS" if fairness_met else "❌ FAIL"
    logger.info(f"Fairness Score: {results['avg_fairness']:.2f} (target: {targets['fairness']:.2f}) {status}")
    all_passed &= fairness_met

    logger.info("-" * 70)
    logger.info("")

    # Summary
    if all_passed:
        logger.info("=" * 70)
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
        logger.info("🚀 MODEL READY FOR PRODUCTION DEPLOYMENT")
        logger.info("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning("⚠️  SOME VALIDATION CHECKS FAILED")
        logger.warning("=" * 70)

    # Additional metrics
    logger.info("\n📈 Additional Metrics:")
    logger.info(f"  Baseline Accuracy: {results['baseline_accuracy']*100:.1f}%")
    logger.info(f"  Improvement: +{results['improvement']*100:.2f} percentage points")
    logger.info(f"  Rounds to Convergence: {results['rounds_to_convergence']}")
    logger.info(f"  Training Time: {results['total_training_time']:.1f}s")
    logger.info("")

    # Configuration summary
    config = data['configuration']
    logger.info("🔧 Optimal Configuration:")
    logger.info(f"  Clients per Round: {config['clients_per_round']}")
    logger.info(f"  Local Epochs: {config['local_epochs']}")
    logger.info(f"  Learning Rate: {config['learning_rate']}")
    logger.info(f"  Topology Weight: {config['topology_weight']}")
    logger.info(f"  Data Quality Weight: {config['weight_data_quality']}")
    logger.info("")

    return all_passed


if __name__ == "__main__":
    success = validate_calibration()
    exit(0 if success else 1)
