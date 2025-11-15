#!/usr/bin/env python3
"""
Fast training script for Node Reliability Isolation Forest (reduced hyperparameter search).
For demonstration and quick testing.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_isolation_forest import *

class FastIsolationForestTuner(IsolationForestTuner):
    """Fast tuner with reduced hyperparameter search."""

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters with reduced search space.
        """
        logger.info("Starting FAST hyperparameter tuning...")

        # Calculate contamination from labeled data
        incident_rate = y_train.sum() / len(y_train)
        contamination_values = [
            incident_rate,
            min(0.5, incident_rate * 1.5)
        ]

        logger.info(f"Incident rate in training data: {incident_rate:.4f}")
        logger.info(f"Testing contamination values: {contamination_values}")

        # Reduced parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_samples': [256, 'auto'],
            'max_features': [0.75, 1.0],
            'contamination': contamination_values
        }

        best_recall = 0
        best_config = None
        results = []

        total_configs = (len(param_grid['n_estimators']) *
                        len(param_grid['max_samples']) *
                        len(param_grid['max_features']) *
                        len(param_grid['contamination']))

        logger.info(f"Evaluating {total_configs} configurations (FAST mode)...")

        config_idx = 0
        for n_est in param_grid['n_estimators']:
            for max_samp in param_grid['max_samples']:
                for max_feat in param_grid['max_features']:
                    for contam in param_grid['contamination']:
                        config_idx += 1
                        logger.info(f"Progress: {config_idx}/{total_configs} configurations tested")

                        # Train model
                        scaler = RobustScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)

                        model = IsolationForest(
                            n_estimators=n_est,
                            max_samples=max_samp,
                            max_features=max_feat,
                            contamination=contam,
                            random_state=42,
                            n_jobs=-1
                        )

                        model.fit(X_train_scaled)

                        # Get anomaly scores
                        scores = model.score_samples(X_val_scaled)

                        # Tune threshold for target recall
                        threshold, metrics = self._tune_threshold(scores, y_val)

                        recall = metrics['recall']
                        fp_rate = metrics['fp_rate']

                        results.append({
                            'n_estimators': n_est,
                            'max_samples': max_samp,
                            'max_features': max_feat,
                            'contamination': contam,
                            'threshold': threshold,
                            'recall': recall,
                            'precision': metrics['precision'],
                            'f1': metrics['f1'],
                            'fp_rate': fp_rate
                        })

                        logger.info(f"  Config {config_idx}: Recall={recall:.4f}, FP={fp_rate:.4f}")

                        # Check if this is the best configuration
                        if recall >= self.target_recall and fp_rate <= self.max_fp_rate:
                            if recall > best_recall:
                                best_recall = recall
                                best_config = {
                                    'model': model,
                                    'scaler': scaler,
                                    'threshold': threshold,
                                    'params': {
                                        'n_estimators': n_est,
                                        'max_samples': max_samp,
                                        'max_features': max_feat,
                                        'contamination': contam
                                    },
                                    'metrics': metrics
                                }

        if best_config:
            self.best_model = best_config['model']
            self.best_scaler = best_config['scaler']
            self.best_threshold = best_config['threshold']
            self.best_params = best_config['params']
            logger.info(f"Best configuration found: Recall={best_recall:.4f}")
            logger.info(f"Best params: {self.best_params}")
        else:
            # Fallback: select config with highest recall
            results_df = pd.DataFrame(results)
            best_idx = results_df['recall'].idxmax()
            best_result = results_df.iloc[best_idx]

            logger.warning(f"Target recall/FP rate not achieved. Using best recall configuration.")
            logger.warning(f"Best recall: {best_result['recall']:.4f}, FP rate: {best_result['fp_rate']:.4f}")

            # Retrain with best params
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = IsolationForest(
                n_estimators=int(best_result['n_estimators']),
                max_samples=best_result['max_samples'],
                max_features=best_result['max_features'],
                contamination=best_result['contamination'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled)

            self.best_model = model
            self.best_scaler = scaler
            self.best_threshold = best_result['threshold']
            self.best_params = {
                'n_estimators': int(best_result['n_estimators']),
                'max_samples': best_result['max_samples'],
                'max_features': best_result['max_features'],
                'contamination': best_result['contamination']
            }

        return {
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'all_results': results
        }


def main_fast():
    parser = argparse.ArgumentParser(
        description='Fast Train Isolation Forest for Node Reliability (DEMO)'
    )
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--incident-rate', type=float, default=0.02)
    parser.add_argument('--output', type=str, default='../models')
    parser.add_argument('--report', type=str, default='../../../../../../docs/models/node_reliability_eval.md')
    parser.add_argument('--target-recall', type=float, default=0.98)
    parser.add_argument('--max-fp-rate', type=float, default=0.05)
    parser.add_argument('--test-size', type=float, default=0.2)

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Node Reliability Isolation Forest Training (FAST MODE)")
    logger.info("=" * 80)

    # Generate synthetic data
    df = generate_synthetic_data(n_samples=args.n_samples, incident_rate=args.incident_rate)
    y = df['label'].values

    # Feature engineering
    feature_engineer = FeatureEngineer(rolling_windows=[5, 15])  # Reduced windows
    df_engineered = feature_engineer.engineer_features(df)

    # Extract features
    X = df_engineered[feature_engineer.feature_names].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(feature_engineer.feature_names)}")

    # Temporal train/test split
    split_idx = int(len(X) * (1 - args.test_size))
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]

    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Tune model with FAST tuner
    tuner = FastIsolationForestTuner(
        target_recall=args.target_recall,
        max_fp_rate=args.max_fp_rate
    )

    tuning_results = tuner.tune_hyperparameters(X_train, y_train, X_val, y_val)

    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter Tuning Complete")
    logger.info("=" * 80)
    logger.info(f"Best parameters: {tuning_results['best_params']}")
    logger.info(f"Best threshold: {tuning_results['best_threshold']:.6f}")

    # Evaluate on test set (skip feature importance for speed)
    logger.info("Evaluating on test set...")
    X_test_scaled = tuner.best_scaler.transform(X_test)
    scores = tuner.best_model.score_samples(X_test_scaled)
    predictions = (scores <= tuner.best_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='binary', zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    try:
        roc_auc = roc_auc_score(y_test, -scores)
    except:
        roc_auc = 0.0

    evaluation_results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'fp_rate': float(fp_rate),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'threshold': float(tuner.best_threshold),
        'n_test_samples': len(y_test),
        'n_incidents': int(y_test.sum())
    }

    # Save model
    save_model(
        tuner.best_model,
        tuner.best_scaler,
        tuner.best_threshold,
        tuner.best_params,
        feature_engineer.feature_names,
        args.output
    )

    # Generate report
    generate_evaluation_report(
        evaluation_results,
        tuner.best_params,
        args.report
    )

    # Print summary
    recall_status = "✓ PASS" if evaluation_results['recall'] >= args.target_recall else "✗ FAIL"
    fp_status = "✓ PASS" if evaluation_results['fp_rate'] < args.max_fp_rate else "✗ FAIL"

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Recall:          {evaluation_results['recall']*100:.2f}% (Target: {args.target_recall*100:.0f}%) {recall_status}")
    print(f"Precision:       {evaluation_results['precision']*100:.2f}%")
    print(f"F1 Score:        {evaluation_results['f1_score']:.4f}")
    print(f"FP Rate:         {evaluation_results['fp_rate']*100:.2f}% (Target: <{args.max_fp_rate*100:.0f}%) {fp_status}")
    print(f"ROC-AUC:         {evaluation_results['roc_auc']:.4f}")
    print("=" * 80)
    print(f"\nModel artifacts saved to: {Path(args.output).absolute()}")
    print(f"Evaluation report: {Path(args.report).absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main_fast()
