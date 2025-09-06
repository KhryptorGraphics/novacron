# AI/ML Operations Runbook
## NovaCron v10 Extended - AI/ML Model Management & Operations

### Document Information
- **Version**: 1.0.0
- **Last Updated**: 2025-01-05
- **Classification**: OPERATIONAL
- **Review Frequency**: Weekly

---

## 1. Model Monitoring Dashboard

### Real-Time Model Performance Metrics

```yaml
ml_monitoring_dashboard:
  model_performance:
    panels:
      - prediction_accuracy:
          metrics: [accuracy, precision, recall, f1_score]
          threshold_alerts:
            accuracy: "< 0.95"
            precision: "< 0.90"
            recall: "< 0.85"
          refresh_rate: 30s
      
      - inference_latency:
          p50: "< 10ms"
          p95: "< 50ms"
          p99: "< 100ms"
          visualization: time_series
      
      - model_drift:
          feature_drift: kolmogorov_smirnov_test
          prediction_drift: population_stability_index
          alert_threshold: 0.1
          
  resource_utilization:
    gpu_metrics:
      - utilization: percentage
      - memory: GB_used
      - temperature: celsius
      - power: watts
    
    cpu_metrics:
      - cores_used: count
      - utilization: percentage
      - memory: GB
      
  data_pipeline:
    ingestion_rate: events_per_second
    processing_lag: milliseconds
    data_quality_score: percentage
    missing_features: count
```

### Model Performance Tracking

```python
#!/usr/bin/env python3
# model_monitoring.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.metrics_history = []
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()
        
    def monitor_inference(self, inputs: np.ndarray, outputs: np.ndarray, 
                         ground_truth: Optional[np.ndarray] = None) -> Dict:
        """Monitor model inference in real-time"""
        
        metrics = {
            'timestamp': datetime.now(),
            'model': self.model_name,
            'version': self.model_version,
            'batch_size': len(inputs)
        }
        
        # Measure inference latency
        start_time = datetime.now()
        predictions = self.model.predict(inputs)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        metrics['inference_latency_ms'] = inference_time
        metrics['throughput'] = len(inputs) / (inference_time / 1000)
        
        # Calculate accuracy if ground truth available
        if ground_truth is not None:
            metrics['accuracy'] = accuracy_score(ground_truth, predictions)
            metrics['precision'] = precision_score(ground_truth, predictions, average='weighted')
            metrics['recall'] = recall_score(ground_truth, predictions, average='weighted')
            metrics['f1'] = f1_score(ground_truth, predictions, average='weighted')
            
            # Check for performance degradation
            self.check_performance_degradation(metrics)
        
        # Check for data drift
        drift_score = self.drift_detector.detect_drift(inputs)
        metrics['drift_score'] = drift_score
        
        if drift_score > 0.1:
            self.alert_manager.send_alert(
                'data_drift',
                f'Data drift detected: score={drift_score:.3f}',
                'high'
            )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def check_performance_degradation(self, current_metrics: Dict):
        """Check for model performance degradation"""
        
        # Get baseline performance
        baseline = self.get_baseline_performance()
        
        # Check accuracy degradation
        if current_metrics['accuracy'] < baseline['accuracy'] * 0.95:
            self.alert_manager.send_alert(
                'performance_degradation',
                f"Accuracy dropped to {current_metrics['accuracy']:.3f} from baseline {baseline['accuracy']:.3f}",
                'critical'
            )
            
            # Trigger automatic retraining if severe
            if current_metrics['accuracy'] < baseline['accuracy'] * 0.90:
                self.trigger_retraining()
    
    def get_baseline_performance(self) -> Dict:
        """Get baseline model performance"""
        # In production, this would fetch from a metrics database
        return {
            'accuracy': 0.98,
            'precision': 0.97,
            'recall': 0.96,
            'f1': 0.965
        }
    
    def trigger_retraining(self):
        """Trigger model retraining pipeline"""
        logger.warning(f"Triggering retraining for {self.model_name}")
        
        # Start retraining job
        retraining_job = {
            'model': self.model_name,
            'trigger': 'performance_degradation',
            'timestamp': datetime.now(),
            'config': self.get_retraining_config()
        }
        
        # Submit to training queue
        self.submit_training_job(retraining_job)

class DriftDetector:
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        self.reference_data = reference_data
        self.drift_history = []
        
    def detect_drift(self, current_data: np.ndarray) -> float:
        """Detect data drift using statistical tests"""
        
        if self.reference_data is None:
            return 0.0
            
        drift_scores = []
        
        # Kolmogorov-Smirnov test for each feature
        for i in range(current_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )
            drift_scores.append(1 - p_value)
        
        # Overall drift score
        overall_drift = np.mean(drift_scores)
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_score': overall_drift,
            'feature_scores': drift_scores
        })
        
        return overall_drift
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                     buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        def psi_bucket(e, a):
            if a == 0:
                a = 0.0001
            if e == 0:
                e = 0.0001
            return (e - a) * np.log(e / a)
        
        # Create bins
        breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
        
        # Calculate frequencies
        expected_freq = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_freq = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Calculate PSI
        psi_values = [psi_bucket(e, a) for e, a in zip(expected_freq, actual_freq)]
        
        return sum(psi_values)

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.deployments = {}
        
    def register_model(self, model_name: str, model_path: str, 
                      metadata: Dict) -> str:
        """Register a new model version"""
        
        version = self.generate_version(model_name)
        
        model_entry = {
            'name': model_name,
            'version': version,
            'path': model_path,
            'metadata': metadata,
            'registered_at': datetime.now(),
            'status': 'registered',
            'performance_metrics': {},
            'deployment_history': []
        }
        
        self.models[f"{model_name}:{version}"] = model_entry
        
        logger.info(f"Registered model {model_name}:{version}")
        
        return version
    
    def promote_model(self, model_name: str, version: str, 
                     environment: str = 'staging'):
        """Promote model to environment"""
        
        model_key = f"{model_name}:{version}"
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        # Run validation tests
        if not self.validate_model(model_name, version):
            raise ValueError("Model validation failed")
        
        # Update deployment
        self.deployments[environment] = {
            'model': model_key,
            'deployed_at': datetime.now(),
            'previous_model': self.deployments.get(environment, {}).get('model')
        }
        
        # Update model status
        self.models[model_key]['status'] = f'deployed_{environment}'
        self.models[model_key]['deployment_history'].append({
            'environment': environment,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Promoted {model_key} to {environment}")
    
    def rollback_model(self, environment: str):
        """Rollback to previous model version"""
        
        current_deployment = self.deployments.get(environment)
        
        if not current_deployment or not current_deployment.get('previous_model'):
            raise ValueError(f"No previous model to rollback in {environment}")
        
        previous_model = current_deployment['previous_model']
        
        # Rollback deployment
        self.deployments[environment] = {
            'model': previous_model,
            'deployed_at': datetime.now(),
            'rollback_from': current_deployment['model']
        }
        
        logger.warning(f"Rolled back {environment} from {current_deployment['model']} to {previous_model}")
    
    def validate_model(self, model_name: str, version: str) -> bool:
        """Validate model before deployment"""
        
        model_key = f"{model_name}:{version}"
        model = self.models[model_key]
        
        # Load model
        if 'tensorflow' in model['metadata'].get('framework', ''):
            loaded_model = tf.keras.models.load_model(model['path'])
        elif 'pytorch' in model['metadata'].get('framework', ''):
            loaded_model = torch.load(model['path'])
        else:
            raise ValueError(f"Unsupported framework for {model_key}")
        
        # Run validation tests
        validation_results = {
            'load_test': True,
            'inference_test': self.run_inference_test(loaded_model),
            'performance_test': self.run_performance_test(loaded_model),
            'compatibility_test': self.run_compatibility_test(loaded_model)
        }
        
        # All tests must pass
        return all(validation_results.values())
    
    def run_inference_test(self, model) -> bool:
        """Test model inference"""
        try:
            # Create sample input
            sample_input = np.random.randn(1, *model.input_shape[1:])
            
            # Run inference
            output = model.predict(sample_input)
            
            # Validate output shape
            return output.shape[0] == 1
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return False
    
    def run_performance_test(self, model) -> bool:
        """Test model performance"""
        try:
            # Create batch of inputs
            batch_input = np.random.randn(100, *model.input_shape[1:])
            
            # Measure inference time
            start = datetime.now()
            _ = model.predict(batch_input)
            inference_time = (datetime.now() - start).total_seconds()
            
            # Check latency requirement (< 100ms for batch of 100)
            return inference_time < 0.1
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
```

---

## 2. Model Retraining Procedures

### Automated Retraining Pipeline

```python
#!/usr/bin/env python3
# model_retraining.py

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
import mlflow
import optuna

class RetrainingPipeline:
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.experiment_tracker = ExperimentTracker()
        self.data_processor = DataProcessor()
        self.hyperparameter_tuner = HyperparameterTuner()
        
    def run_retraining(self, trigger_reason: str) -> Dict:
        """Run complete retraining pipeline"""
        
        logger.info(f"Starting retraining pipeline - Trigger: {trigger_reason}")
        
        pipeline_run = {
            'run_id': self.generate_run_id(),
            'start_time': datetime.now(),
            'trigger': trigger_reason,
            'status': 'running'
        }
        
        try:
            # Step 1: Data preparation
            training_data = self.prepare_training_data()
            
            # Step 2: Feature engineering
            features = self.engineer_features(training_data)
            
            # Step 3: Hyperparameter tuning
            best_params = self.tune_hyperparameters(features)
            
            # Step 4: Model training
            model = self.train_model(features, best_params)
            
            # Step 5: Model evaluation
            evaluation = self.evaluate_model(model, features['test'])
            
            # Step 6: Model validation
            if self.validate_model(model, evaluation):
                # Step 7: Register model
                model_version = self.register_model(model, evaluation)
                pipeline_run['model_version'] = model_version
                pipeline_run['status'] = 'completed'
            else:
                pipeline_run['status'] = 'validation_failed'
                
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            pipeline_run['status'] = 'failed'
            pipeline_run['error'] = str(e)
            
        pipeline_run['end_time'] = datetime.now()
        pipeline_run['duration'] = (pipeline_run['end_time'] - pipeline_run['start_time']).total_seconds()
        
        # Log pipeline run
        self.experiment_tracker.log_pipeline_run(pipeline_run)
        
        return pipeline_run
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data"""
        
        logger.info("Preparing training data...")
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.model_config['training_window_days'])
        
        # Query data warehouse
        query = f"""
        SELECT *
        FROM ml_training_data
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        AND quality_score > 0.8
        """
        
        data = pd.read_sql(query, self.get_db_connection())
        
        # Data quality checks
        data = self.data_processor.clean_data(data)
        data = self.data_processor.handle_missing_values(data)
        data = self.data_processor.remove_outliers(data)
        
        # Balance dataset if needed
        if self.model_config.get('balance_classes'):
            data = self.data_processor.balance_dataset(data)
        
        logger.info(f"Prepared {len(data)} training samples")
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> Dict:
        """Feature engineering"""
        
        logger.info("Engineering features...")
        
        # Apply feature transformations
        features = self.data_processor.extract_features(data)
        
        # Feature selection
        selected_features = self.data_processor.select_features(
            features,
            method=self.model_config.get('feature_selection', 'mutual_info')
        )
        
        # Split data
        X = selected_features.drop(columns=['target'])
        y = selected_features['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if self.model_config.get('stratify') else None
        )
        
        # Scale features
        X_train_scaled = self.data_processor.scale_features(X_train, fit=True)
        X_test_scaled = self.data_processor.scale_features(X_test, fit=False)
        
        return {
            'train': (X_train_scaled, y_train),
            'test': (X_test_scaled, y_test),
            'feature_names': list(X.columns),
            'scaler': self.data_processor.scaler
        }
    
    def tune_hyperparameters(self, features: Dict) -> Dict:
        """Hyperparameter tuning using Optuna"""
        
        logger.info("Tuning hyperparameters...")
        
        X_train, y_train = features['train']
        X_val, y_val = features['test']
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'num_layers': trial.suggest_int('num_layers', 2, 8),
                'hidden_units': trial.suggest_int('hidden_units', 32, 512),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
            }
            
            # Train model with suggested parameters
            model = self.build_model(params, features['feature_names'])
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,  # Quick training for tuning
                batch_size=params['batch_size'],
                verbose=0
            )
            
            # Return validation loss
            return min(history.history['val_loss'])
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.model_config.get('tuning_trials', 50))
        
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def train_model(self, features: Dict, hyperparameters: Dict):
        """Train model with best hyperparameters"""
        
        logger.info("Training model...")
        
        X_train, y_train = features['train']
        X_val, y_val = features['test']
        
        # Build model
        model = self.build_model(hyperparameters, features['feature_names'])
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model_checkpoint.h5',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config.get('max_epochs', 100),
            batch_size=hyperparameters['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training metrics
        self.experiment_tracker.log_training_metrics(history.history)
        
        return model
    
    def evaluate_model(self, model, test_data: Tuple) -> Dict:
        """Evaluate trained model"""
        
        logger.info("Evaluating model...")
        
        X_test, y_test = test_data
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred.round()),
            'precision': precision_score(y_test, y_pred.round(), average='weighted'),
            'recall': recall_score(y_test, y_pred.round(), average='weighted'),
            'f1': f1_score(y_test, y_pred.round(), average='weighted'),
            'auc_roc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) == 2 else None
        }
        
        # Additional metrics
        evaluation['inference_latency'] = self.measure_inference_latency(model, X_test[:100])
        evaluation['model_size_mb'] = self.get_model_size(model) / (1024 * 1024)
        
        logger.info(f"Model evaluation: {evaluation}")
        
        return evaluation
    
    def validate_model(self, model, evaluation: Dict) -> bool:
        """Validate model meets requirements"""
        
        # Performance thresholds
        thresholds = self.model_config.get('validation_thresholds', {
            'accuracy': 0.95,
            'precision': 0.90,
            'recall': 0.90,
            'inference_latency': 100  # ms
        })
        
        # Check all thresholds
        for metric, threshold in thresholds.items():
            if metric in evaluation:
                if metric == 'inference_latency':
                    if evaluation[metric] > threshold:
                        logger.warning(f"Validation failed: {metric}={evaluation[metric]} > {threshold}")
                        return False
                else:
                    if evaluation[metric] < threshold:
                        logger.warning(f"Validation failed: {metric}={evaluation[metric]} < {threshold}")
                        return False
        
        logger.info("Model validation passed")
        return True

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data"""
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle invalid values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with too many missing values
        threshold = len(data.columns) * 0.5
        data = data.dropna(thresh=threshold)
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        
        # Numerical columns: fill with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        
        # Categorical columns: fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
        
        return data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from data"""
        
        if method == 'iqr':
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter data
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                data = data[(data[col] >= lower_bound[col]) & (data[col] <= upper_bound[col])]
                
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            data = data[(z_scores < 3).all(axis=1)]
        
        return data
    
    def balance_dataset(self, data: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Balance dataset using SMOTE or undersampling"""
        
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Check class distribution
        class_counts = y.value_counts()
        
        if class_counts.min() / class_counts.max() < 0.5:
            # Use SMOTE for severe imbalance
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        else:
            # Use undersampling for mild imbalance
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        
        # Combine back
        balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
        
        return balanced_data

class ExperimentTracker:
    def __init__(self):
        mlflow.set_tracking_uri("http://mlflow.novacron.io")
        mlflow.set_experiment("model_retraining")
        
    def log_pipeline_run(self, pipeline_run: Dict):
        """Log pipeline run to MLflow"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("trigger", pipeline_run['trigger'])
            mlflow.log_param("run_id", pipeline_run['run_id'])
            
            # Log metrics
            if 'duration' in pipeline_run:
                mlflow.log_metric("duration_seconds", pipeline_run['duration'])
            
            # Log status
            mlflow.set_tag("status", pipeline_run['status'])
            
            if pipeline_run['status'] == 'failed':
                mlflow.set_tag("error", pipeline_run.get('error', 'Unknown'))
    
    def log_training_metrics(self, history: Dict):
        """Log training metrics"""
        
        for epoch, values in enumerate(history['loss']):
            mlflow.log_metric("train_loss", values, step=epoch)
            
        for epoch, values in enumerate(history['val_loss']):
            mlflow.log_metric("val_loss", values, step=epoch)
            
        if 'accuracy' in history:
            for epoch, values in enumerate(history['accuracy']):
                mlflow.log_metric("train_accuracy", values, step=epoch)
                
        if 'val_accuracy' in history:
            for epoch, values in enumerate(history['val_accuracy']):
                mlflow.log_metric("val_accuracy", values, step=epoch)
```

---

## 3. Anomaly Investigation

### Anomaly Detection and Response

```bash
#!/bin/bash
# anomaly_investigation.sh

investigate_anomaly() {
    ANOMALY_TYPE=$1
    ANOMALY_ID=$2
    SEVERITY=$3
    
    echo "=== Anomaly Investigation ==="
    echo "Type: $ANOMALY_TYPE"
    echo "ID: $ANOMALY_ID"
    echo "Severity: $SEVERITY"
    echo "Time: $(date)"
    
    case $ANOMALY_TYPE in
        "prediction_drift")
            investigate_prediction_drift $ANOMALY_ID
            ;;
        "feature_anomaly")
            investigate_feature_anomaly $ANOMALY_ID
            ;;
        "performance_degradation")
            investigate_performance_degradation $ANOMALY_ID
            ;;
        "data_quality")
            investigate_data_quality $ANOMALY_ID
            ;;
        "system_anomaly")
            investigate_system_anomaly $ANOMALY_ID
            ;;
        *)
            echo "Unknown anomaly type: $ANOMALY_TYPE"
            ;;
    esac
}

investigate_prediction_drift() {
    local anomaly_id=$1
    echo ""
    echo "Investigating prediction drift..."
    
    # 1. Get drift metrics
    drift_metrics=$(curl -s http://ml-monitor.novacron.io/api/drift/$anomaly_id)
    echo "Drift metrics: $drift_metrics"
    
    # 2. Analyze feature distributions
    python3 << EOF
import json
import numpy as np
from scipy import stats

# Load drift data
drift_data = json.loads('$drift_metrics')

# Analyze each feature
for feature, scores in drift_data['feature_scores'].items():
    ks_statistic = scores['ks_statistic']
    p_value = scores['p_value']
    
    if p_value < 0.05:
        print(f"Feature '{feature}' shows significant drift:")
        print(f"  KS statistic: {ks_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        # Recommend action
        if ks_statistic > 0.3:
            print(f"  Action: Consider retraining with recent data")
        elif ks_statistic > 0.2:
            print(f"  Action: Monitor closely, prepare for retraining")
        else:
            print(f"  Action: Continue monitoring")

# Overall recommendation
overall_drift = drift_data['overall_drift_score']
if overall_drift > 0.15:
    print("\nRECOMMENDATION: Initiate model retraining")
elif overall_drift > 0.10:
    print("\nRECOMMENDATION: Increase monitoring frequency")
else:
    print("\nRECOMMENDATION: Continue normal operations")
EOF
    
    # 3. Check for data pipeline issues
    echo ""
    echo "Checking data pipeline..."
    pipeline_status=$(check_data_pipeline_health)
    
    if [ "$pipeline_status" != "healthy" ]; then
        echo "  ⚠️  Data pipeline issues detected"
        fix_data_pipeline
    fi
    
    # 4. Generate drift report
    generate_drift_report $anomaly_id
}

investigate_feature_anomaly() {
    local anomaly_id=$1
    echo ""
    echo "Investigating feature anomaly..."
    
    # Get anomaly details
    anomaly_details=$(get_anomaly_details $anomaly_id)
    
    # Check for data corruption
    echo "Checking for data corruption..."
    corrupted_records=$(find_corrupted_records "$anomaly_details")
    
    if [ -n "$corrupted_records" ]; then
        echo "  Found corrupted records: $corrupted_records"
        quarantine_corrupted_data "$corrupted_records"
    fi
    
    # Check for upstream changes
    echo "Checking upstream data sources..."
    upstream_changes=$(check_upstream_changes)
    
    if [ -n "$upstream_changes" ]; then
        echo "  Upstream changes detected: $upstream_changes"
        update_feature_pipeline "$upstream_changes"
    fi
}

investigate_performance_degradation() {
    local anomaly_id=$1
    echo ""
    echo "Investigating performance degradation..."
    
    # 1. Get performance metrics
    current_metrics=$(get_current_model_metrics)
    baseline_metrics=$(get_baseline_model_metrics)
    
    # 2. Compare metrics
    python3 << EOF
import json

current = json.loads('$current_metrics')
baseline = json.loads('$baseline_metrics')

print("Performance Comparison:")
print(f"  Accuracy: {current['accuracy']:.3f} (baseline: {baseline['accuracy']:.3f})")
print(f"  Precision: {current['precision']:.3f} (baseline: {baseline['precision']:.3f})")
print(f"  Recall: {current['recall']:.3f} (baseline: {baseline['recall']:.3f})")
print(f"  F1 Score: {current['f1']:.3f} (baseline: {baseline['f1']:.3f})")

# Calculate degradation
accuracy_drop = (baseline['accuracy'] - current['accuracy']) / baseline['accuracy'] * 100
print(f"\nAccuracy degradation: {accuracy_drop:.1f}%")

if accuracy_drop > 10:
    print("ACTION: Immediate retraining required")
elif accuracy_drop > 5:
    print("ACTION: Schedule retraining within 24 hours")
else:
    print("ACTION: Continue monitoring")
EOF
    
    # 3. Check for concept drift
    echo ""
    echo "Checking for concept drift..."
    concept_drift=$(detect_concept_drift)
    
    if [ "$concept_drift" == "detected" ]; then
        echo "  Concept drift detected - updating training strategy"
        update_training_strategy "concept_drift"
    fi
    
    # 4. Analyze error patterns
    echo ""
    echo "Analyzing error patterns..."
    analyze_model_errors
}

fix_data_pipeline() {
    echo "Attempting to fix data pipeline..."
    
    # Restart data ingestion
    kubectl rollout restart deployment/data-ingestion
    
    # Clear corrupted cache
    redis-cli FLUSHDB
    
    # Reprocess last hour of data
    reprocess_recent_data 1h
    
    # Verify fix
    sleep 30
    if [ "$(check_data_pipeline_health)" == "healthy" ]; then
        echo "✅ Data pipeline fixed"
    else
        echo "❌ Manual intervention required"
        alert_data_engineering_team
    fi
}

generate_drift_report() {
    local anomaly_id=$1
    
    cat > /reports/drift-report-$anomaly_id.md << EOF
# Data Drift Report
**Anomaly ID**: $anomaly_id
**Generated**: $(date)

## Executive Summary
Significant data drift detected requiring immediate attention.

## Drift Analysis
$(get_drift_analysis $anomaly_id)

## Affected Models
$(list_affected_models $anomaly_id)

## Recommended Actions
1. Review feature engineering pipeline
2. Validate data quality checks
3. Consider model retraining
4. Update monitoring thresholds

## Next Steps
- [ ] Review this report with ML team
- [ ] Approve retraining if necessary
- [ ] Update data validation rules
- [ ] Document root cause
EOF
    
    echo "Report generated: /reports/drift-report-$anomaly_id.md"
}

# Execute investigation
investigate_anomaly "$@"
```

---

## 4. Model Deployment

### Blue-Green Model Deployment

```python
#!/usr/bin/env python3
# model_deployment.py

import os
import time
import requests
from typing import Dict, List, Optional
import kubernetes
from kubernetes import client, config
import boto3

class ModelDeployer:
    def __init__(self):
        config.load_incluster_config()
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()
        self.load_balancer = LoadBalancerManager()
        
    def deploy_model(self, model_name: str, model_version: str, 
                    strategy: str = 'blue_green') -> bool:
        """Deploy model using specified strategy"""
        
        logger.info(f"Deploying {model_name}:{model_version} using {strategy}")
        
        if strategy == 'blue_green':
            return self.blue_green_deployment(model_name, model_version)
        elif strategy == 'canary':
            return self.canary_deployment(model_name, model_version)
        elif strategy == 'rolling':
            return self.rolling_deployment(model_name, model_version)
        else:
            raise ValueError(f"Unknown deployment strategy: {strategy}")
    
    def blue_green_deployment(self, model_name: str, model_version: str) -> bool:
        """Blue-green deployment strategy"""
        
        # Step 1: Deploy to green environment
        green_deployment = self.create_deployment(
            name=f"{model_name}-green",
            model_version=model_version,
            replicas=3,
            labels={'app': model_name, 'version': 'green'}
        )
        
        # Step 2: Wait for green deployment to be ready
        if not self.wait_for_deployment(f"{model_name}-green"):
            logger.error("Green deployment failed")
            return False
        
        # Step 3: Run smoke tests on green
        if not self.run_smoke_tests(f"{model_name}-green"):
            logger.error("Green deployment failed smoke tests")
            self.cleanup_deployment(f"{model_name}-green")
            return False
        
        # Step 4: Switch traffic to green
        self.load_balancer.switch_traffic(model_name, 'green')
        
        # Step 5: Monitor for issues
        if not self.monitor_deployment(model_name, duration=300):
            logger.error("Issues detected, rolling back")
            self.load_balancer.switch_traffic(model_name, 'blue')
            self.cleanup_deployment(f"{model_name}-green")
            return False
        
        # Step 6: Remove blue deployment
        self.cleanup_deployment(f"{model_name}-blue")
        
        # Step 7: Rename green to blue for next deployment
        self.rename_deployment(f"{model_name}-green", f"{model_name}-blue")
        
        logger.info(f"Successfully deployed {model_name}:{model_version}")
        return True
    
    def canary_deployment(self, model_name: str, model_version: str) -> bool:
        """Canary deployment strategy"""
        
        # Step 1: Deploy canary with small replica count
        canary_deployment = self.create_deployment(
            name=f"{model_name}-canary",
            model_version=model_version,
            replicas=1,
            labels={'app': model_name, 'version': 'canary'}
        )
        
        # Step 2: Route small percentage of traffic
        self.load_balancer.split_traffic(
            model_name,
            weights={'stable': 95, 'canary': 5}
        )
        
        # Step 3: Monitor canary metrics
        canary_healthy = True
        for i in range(10):  # Monitor for 10 minutes
            metrics = self.get_deployment_metrics(f"{model_name}-canary")
            
            if metrics['error_rate'] > 0.01:  # 1% error threshold
                logger.warning(f"High error rate in canary: {metrics['error_rate']}")
                canary_healthy = False
                break
                
            time.sleep(60)
        
        if not canary_healthy:
            # Rollback canary
            self.load_balancer.split_traffic(
                model_name,
                weights={'stable': 100, 'canary': 0}
            )
            self.cleanup_deployment(f"{model_name}-canary")
            return False
        
        # Step 4: Gradually increase traffic
        for weight in [10, 25, 50, 75, 100]:
            self.load_balancer.split_traffic(
                model_name,
                weights={'stable': 100 - weight, 'canary': weight}
            )
            
            time.sleep(120)  # Wait 2 minutes between increases
            
            if not self.monitor_deployment(f"{model_name}-canary", duration=60):
                # Rollback on issues
                self.load_balancer.split_traffic(
                    model_name,
                    weights={'stable': 100, 'canary': 0}
                )
                self.cleanup_deployment(f"{model_name}-canary")
                return False
        
        # Step 5: Promote canary to stable
        self.cleanup_deployment(f"{model_name}-stable")
        self.rename_deployment(f"{model_name}-canary", f"{model_name}-stable")
        
        return True
    
    def create_deployment(self, name: str, model_version: str, 
                         replicas: int, labels: Dict) -> bool:
        """Create Kubernetes deployment"""
        
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(match_labels=labels),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=labels),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="model-server",
                                image=f"novacron/model-server:{model_version}",
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={"memory": "2Gi", "cpu": "1"},
                                    limits={"memory": "4Gi", "cpu": "2"}
                                ),
                                env=[
                                    client.V1EnvVar(name="MODEL_VERSION", value=model_version),
                                    client.V1EnvVar(name="LOG_LEVEL", value="INFO")
                                ],
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/health",
                                        port=8080
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/ready",
                                        port=8080
                                    ),
                                    initial_delay_seconds=10,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        try:
            self.k8s_apps.create_namespaced_deployment(
                namespace="ml-models",
                body=deployment
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def run_smoke_tests(self, deployment_name: str) -> bool:
        """Run smoke tests on deployment"""
        
        # Get service endpoint
        service = self.k8s_core.read_namespaced_service(
            name=deployment_name,
            namespace="ml-models"
        )
        
        endpoint = f"http://{service.status.load_balancer.ingress[0].ip}:8080"
        
        # Test cases
        test_cases = [
            {
                'name': 'Health check',
                'endpoint': f"{endpoint}/health",
                'expected_status': 200
            },
            {
                'name': 'Model info',
                'endpoint': f"{endpoint}/model/info",
                'expected_status': 200
            },
            {
                'name': 'Prediction test',
                'endpoint': f"{endpoint}/predict",
                'method': 'POST',
                'data': {'features': [1.0, 2.0, 3.0]},
                'expected_status': 200
            }
        ]
        
        for test in test_cases:
            try:
                if test.get('method') == 'POST':
                    response = requests.post(
                        test['endpoint'],
                        json=test.get('data'),
                        timeout=5
                    )
                else:
                    response = requests.get(test['endpoint'], timeout=5)
                
                if response.status_code != test['expected_status']:
                    logger.error(f"Test '{test['name']}' failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"Test '{test['name']}' failed: {e}")
                return False
        
        logger.info("All smoke tests passed")
        return True

class LoadBalancerManager:
    def __init__(self):
        self.elb = boto3.client('elbv2')
        
    def switch_traffic(self, model_name: str, target_version: str):
        """Switch all traffic to target version"""
        
        # Update target group
        self.elb.modify_target_group_attributes(
            TargetGroupArn=self.get_target_group_arn(model_name),
            Attributes=[
                {
                    'Key': 'stickiness.enabled',
                    'Value': 'true'
                },
                {
                    'Key': 'stickiness.type',
                    'Value': 'app_cookie'
                }
            ]
        )
        
        # Update routing rules
        self.elb.modify_rule(
            RuleArn=self.get_rule_arn(model_name),
            Actions=[
                {
                    'Type': 'forward',
                    'TargetGroupArn': self.get_target_group_arn(f"{model_name}-{target_version}")
                }
            ]
        )
        
        logger.info(f"Switched traffic to {target_version}")
    
    def split_traffic(self, model_name: str, weights: Dict[str, int]):
        """Split traffic between versions"""
        
        target_groups = []
        for version, weight in weights.items():
            if weight > 0:
                target_groups.append({
                    'TargetGroupArn': self.get_target_group_arn(f"{model_name}-{version}"),
                    'Weight': weight
                })
        
        self.elb.modify_rule(
            RuleArn=self.get_rule_arn(model_name),
            Actions=[
                {
                    'Type': 'forward',
                    'ForwardConfig': {
                        'TargetGroups': target_groups
                    }
                }
            ]
        )
        
        logger.info(f"Split traffic: {weights}")
```

---

## 5. Neural Pattern Management

### Neural Network Pattern Analysis

```go
// neural_pattern_manager.go
package ml

import (
    "context"
    "encoding/json"
    "fmt"
    "math"
    "time"
)

type NeuralPatternManager struct {
    PatternDB      PatternDatabase
    Analyzer       PatternAnalyzer
    Optimizer      PatternOptimizer
    QuantumBridge  QuantumProcessor
}

type Pattern struct {
    ID          string
    Type        string
    Weights     [][]float64
    Activations []float64
    Metadata    PatternMetadata
    Performance PatternPerformance
}

type PatternMetadata struct {
    CreatedAt    time.Time
    UpdatedAt    time.Time
    Version      int
    Framework    string
    Architecture string
}

type PatternPerformance struct {
    Accuracy     float64
    Latency      time.Duration
    Throughput   float64
    MemoryUsage  int64
    PowerUsage   float64
}

func (npm *NeuralPatternManager) AnalyzePattern(ctx context.Context, modelID string) (*Pattern, error) {
    // Load model
    model, err := npm.loadModel(modelID)
    if err != nil {
        return nil, fmt.Errorf("failed to load model: %w", err)
    }
    
    // Extract patterns
    pattern := &Pattern{
        ID:   generatePatternID(),
        Type: model.Type,
    }
    
    // Analyze weights
    pattern.Weights = npm.extractWeights(model)
    
    // Analyze activations
    pattern.Activations = npm.analyzeActivations(model)
    
    // Measure performance
    pattern.Performance = npm.measurePerformance(model)
    
    // Optimize pattern
    optimized := npm.Optimizer.Optimize(pattern)
    
    // Store pattern
    if err := npm.PatternDB.Store(optimized); err != nil {
        return nil, fmt.Errorf("failed to store pattern: %w", err)
    }
    
    return optimized, nil
}

func (npm *NeuralPatternManager) OptimizePattern(pattern *Pattern) *Pattern {
    // Prune small weights
    pattern = npm.pruneWeights(pattern, 0.01)
    
    // Quantize weights
    pattern = npm.quantizeWeights(pattern, 8)
    
    // Optimize activation functions
    pattern = npm.optimizeActivations(pattern)
    
    // Knowledge distillation
    pattern = npm.distillKnowledge(pattern)
    
    return pattern
}

func (npm *NeuralPatternManager) pruneWeights(pattern *Pattern, threshold float64) *Pattern {
    pruned := &Pattern{
        ID:          pattern.ID,
        Type:        pattern.Type,
        Weights:     make([][]float64, len(pattern.Weights)),
        Activations: pattern.Activations,
    }
    
    totalWeights := 0
    prunedCount := 0
    
    for i, layer := range pattern.Weights {
        pruned.Weights[i] = make([]float64, len(layer))
        for j, weight := range layer {
            totalWeights++
            if math.Abs(weight) < threshold {
                pruned.Weights[i][j] = 0
                prunedCount++
            } else {
                pruned.Weights[i][j] = weight
            }
        }
    }
    
    // Log pruning statistics
    pruningRatio := float64(prunedCount) / float64(totalWeights)
    fmt.Printf("Pruned %d/%d weights (%.2f%%)\n", prunedCount, totalWeights, pruningRatio*100)
    
    return pruned
}

func (npm *NeuralPatternManager) quantizeWeights(pattern *Pattern, bits int) *Pattern {
    quantized := &Pattern{
        ID:          pattern.ID,
        Type:        pattern.Type,
        Weights:     make([][]float64, len(pattern.Weights)),
        Activations: pattern.Activations,
    }
    
    // Calculate quantization scale
    scale := math.Pow(2, float64(bits)) - 1
    
    for i, layer := range pattern.Weights {
        quantized.Weights[i] = make([]float64, len(layer))
        
        // Find min and max for layer
        min, max := layer[0], layer[0]
        for _, w := range layer {
            if w < min {
                min = w
            }
            if w > max {
                max = w
            }
        }
        
        // Quantize weights
        for j, weight := range layer {
            normalized := (weight - min) / (max - min)
            quantizedValue := math.Round(normalized * scale) / scale
            quantized.Weights[i][j] = quantizedValue*(max-min) + min
        }
    }
    
    return quantized
}

// Quantum-enhanced pattern processing
func (npm *NeuralPatternManager) QuantumEnhancePattern(pattern *Pattern) (*Pattern, error) {
    // Prepare quantum state
    qstate := npm.QuantumBridge.PrepareState(pattern.Weights)
    
    // Apply quantum optimization
    optimized := npm.QuantumBridge.OptimizeWeights(qstate)
    
    // Measure and collapse state
    result := npm.QuantumBridge.Measure(optimized)
    
    // Update pattern
    enhanced := &Pattern{
        ID:          pattern.ID + "_quantum",
        Type:        pattern.Type + "_quantum_enhanced",
        Weights:     result,
        Activations: pattern.Activations,
    }
    
    return enhanced, nil
}

// Pattern similarity analysis
func (npm *NeuralPatternManager) FindSimilarPatterns(pattern *Pattern, threshold float64) ([]*Pattern, error) {
    allPatterns, err := npm.PatternDB.GetAll()
    if err != nil {
        return nil, err
    }
    
    similar := []*Pattern{}
    
    for _, p := range allPatterns {
        similarity := npm.calculateSimilarity(pattern, p)
        if similarity > threshold {
            similar = append(similar, p)
        }
    }
    
    return similar, nil
}

func (npm *NeuralPatternManager) calculateSimilarity(p1, p2 *Pattern) float64 {
    if len(p1.Weights) != len(p2.Weights) {
        return 0
    }
    
    totalSimilarity := 0.0
    layerCount := 0
    
    for i := range p1.Weights {
        if len(p1.Weights[i]) != len(p2.Weights[i]) {
            continue
        }
        
        // Cosine similarity for layer weights
        dotProduct := 0.0
        norm1 := 0.0
        norm2 := 0.0
        
        for j := range p1.Weights[i] {
            dotProduct += p1.Weights[i][j] * p2.Weights[i][j]
            norm1 += p1.Weights[i][j] * p1.Weights[i][j]
            norm2 += p2.Weights[i][j] * p2.Weights[i][j]
        }
        
        if norm1 > 0 && norm2 > 0 {
            similarity := dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
            totalSimilarity += similarity
            layerCount++
        }
    }
    
    if layerCount == 0 {
        return 0
    }
    
    return totalSimilarity / float64(layerCount)
}

// Auto-ML pattern generation
func (npm *NeuralPatternManager) GenerateAutoMLPattern(taskType string, constraints map[string]interface{}) (*Pattern, error) {
    // Neural Architecture Search (NAS)
    architecture := npm.searchArchitecture(taskType, constraints)
    
    // Initialize pattern
    pattern := &Pattern{
        ID:   generatePatternID(),
        Type: taskType,
        Metadata: PatternMetadata{
            CreatedAt:    time.Now(),
            Architecture: architecture,
        },
    }
    
    // Generate initial weights
    pattern.Weights = npm.initializeWeights(architecture)
    
    // Optimize using evolutionary algorithm
    pattern = npm.evolvePattern(pattern, constraints)
    
    // Fine-tune with gradient descent
    pattern = npm.fineTunePattern(pattern)
    
    return pattern, nil
}
```

---

## 6. Performance Optimization

### ML Model Performance Tuning

```bash
#!/bin/bash
# ml_performance_optimization.sh

optimize_ml_performance() {
    echo "=== ML Performance Optimization ==="
    
    MODEL_NAME=$1
    OPTIMIZATION_TYPE=$2  # inference|training|memory|all
    
    case $OPTIMIZATION_TYPE in
        "inference")
            optimize_inference_performance $MODEL_NAME
            ;;
        "training")
            optimize_training_performance $MODEL_NAME
            ;;
        "memory")
            optimize_memory_usage $MODEL_NAME
            ;;
        "all")
            optimize_inference_performance $MODEL_NAME
            optimize_training_performance $MODEL_NAME
            optimize_memory_usage $MODEL_NAME
            ;;
    esac
}

optimize_inference_performance() {
    local model=$1
    echo ""
    echo "Optimizing inference performance for $model..."
    
    # 1. Model quantization
    echo "Applying model quantization..."
    python3 << EOF
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('/models/$model')

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_dataset_gen()
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert and save
tflite_model = converter.convert()
with open('/models/${model}_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size reduced from {len(model.to_json())/1024:.2f}KB to {len(tflite_model)/1024:.2f}KB")
EOF
    
    # 2. ONNX conversion for optimization
    echo "Converting to ONNX format..."
    python3 -m tf2onnx.convert \
        --saved-model /models/$model \
        --output /models/${model}.onnx \
        --opset 13
    
    # 3. TensorRT optimization
    echo "Applying TensorRT optimization..."
    trtexec --onnx=/models/${model}.onnx \
            --saveEngine=/models/${model}.trt \
            --fp16 \
            --workspace=1024 \
            --batch=1 \
            --minShapes=input:1x224x224x3 \
            --optShapes=input:8x224x224x3 \
            --maxShapes=input:32x224x224x3
    
    # 4. Benchmark inference speed
    echo ""
    echo "Benchmarking inference performance..."
    benchmark_inference $model
}

optimize_training_performance() {
    local model=$1
    echo ""
    echo "Optimizing training performance for $model..."
    
    # 1. Mixed precision training
    echo "Enabling mixed precision training..."
    cat > /configs/${model}_training_optimized.yaml << EOF
training_config:
  mixed_precision:
    enabled: true
    policy: "mixed_float16"
    loss_scale: 128
  
  distributed_training:
    strategy: "mirrored"
    num_gpus: 4
    
  gradient_accumulation:
    steps: 4
    
  optimizer:
    type: "adam"
    learning_rate: 0.001
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-7
    
  data_pipeline:
    prefetch_size: 2
    num_parallel_calls: 8
    cache_dataset: true
    
  callbacks:
    - learning_rate_scheduler:
        schedule: "cosine"
        warmup_epochs: 5
    - gradient_clipping:
        max_norm: 1.0
    - early_stopping:
        patience: 10
        monitor: "val_loss"
EOF
    
    # 2. Optimize data pipeline
    echo "Optimizing data pipeline..."
    python3 << EOF
import tensorflow as tf

# Configure data pipeline optimization
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

# Apply to dataset
dataset = load_dataset('$model')
dataset = dataset.with_options(options)
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("Data pipeline optimized")
EOF
    
    # 3. Setup distributed training
    echo "Configuring distributed training..."
    horovodrun -np 4 -H localhost:4 \
        python train_distributed.py --model $model
}

optimize_memory_usage() {
    local model=$1
    echo ""
    echo "Optimizing memory usage for $model..."
    
    # 1. Model pruning
    echo "Applying model pruning..."
    python3 << EOF
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load model
model = tf.keras.models.load_model('/models/$model')

# Apply pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)

# Compile and save
model_for_pruning.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_for_pruning.save('/models/${model}_pruned')
print("Model pruned successfully")
EOF
    
    # 2. Optimize memory allocation
    echo "Optimizing GPU memory allocation..."
    export TF_GPU_ALLOCATOR=cuda_malloc_async
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_GPU_THREAD_MODE=gpu_private
    export TF_GPU_THREAD_COUNT=2
    
    # 3. Clear cache and unused memory
    echo "Clearing memory cache..."
    python3 << EOF
import gc
import torch
import tensorflow as tf

# Clear TensorFlow session
tf.keras.backend.clear_session()

# Clear PyTorch cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

print("Memory cache cleared")
EOF
}

benchmark_inference() {
    local model=$1
    
    echo "Running inference benchmark..."
    
    # Warm up
    for i in {1..10}; do
        curl -s -X POST http://localhost:8080/predict \
            -H "Content-Type: application/json" \
            -d '{"model": "'$model'", "data": [1,2,3,4,5]}' > /dev/null
    done
    
    # Benchmark
    total_time=0
    iterations=100
    
    for i in $(seq 1 $iterations); do
        start=$(date +%s%N)
        curl -s -X POST http://localhost:8080/predict \
            -H "Content-Type: application/json" \
            -d '{"model": "'$model'", "data": [1,2,3,4,5]}' > /dev/null
        end=$(date +%s%N)
        
        latency=$((($end - $start) / 1000000))
        total_time=$((total_time + latency))
    done
    
    avg_latency=$((total_time / iterations))
    throughput=$((1000 / avg_latency))
    
    echo ""
    echo "Benchmark Results:"
    echo "  Average Latency: ${avg_latency}ms"
    echo "  Throughput: ${throughput} req/s"
    echo "  P50 Latency: $(calculate_percentile 50)ms"
    echo "  P95 Latency: $(calculate_percentile 95)ms"
    echo "  P99 Latency: $(calculate_percentile 99)ms"
}

# Execute optimization
optimize_ml_performance "$@"
```

---

## 7. Compliance & Governance

### AI/ML Compliance Monitoring

```python
#!/usr/bin/env python3
# ml_compliance.py

import json
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

class MLComplianceMonitor:
    def __init__(self):
        self.compliance_standards = {
            'GDPR': self.check_gdpr_compliance,
            'CCPA': self.check_ccpa_compliance,
            'EU_AI_Act': self.check_eu_ai_act_compliance,
            'FDA': self.check_fda_compliance
        }
        
    def run_compliance_audit(self, model_name: str) -> Dict:
        """Run comprehensive compliance audit"""
        
        audit_results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'compliance_status': {},
            'findings': [],
            'recommendations': []
        }
        
        for standard, check_func in self.compliance_standards.items():
            result = check_func(model_name)
            audit_results['compliance_status'][standard] = result['compliant']
            
            if not result['compliant']:
                audit_results['findings'].extend(result['findings'])
                audit_results['recommendations'].extend(result['recommendations'])
        
        # Generate audit report
        self.generate_audit_report(audit_results)
        
        return audit_results
    
    def check_gdpr_compliance(self, model_name: str) -> Dict:
        """Check GDPR compliance for AI/ML models"""
        
        findings = []
        recommendations = []
        compliant = True
        
        # Check data minimization
        if not self.verify_data_minimization(model_name):
            findings.append("Model uses excessive personal data")
            recommendations.append("Implement data minimization techniques")
            compliant = False
        
        # Check right to explanation
        if not self.verify_explainability(model_name):
            findings.append("Model lacks explainability features")
            recommendations.append("Implement LIME or SHAP for explanations")
            compliant = False
        
        # Check data retention
        if not self.verify_data_retention(model_name):
            findings.append("Training data retained beyond necessary period")
            recommendations.append("Implement automatic data deletion policies")
            compliant = False
        
        # Check consent management
        if not self.verify_consent_tracking(model_name):
            findings.append("Inadequate consent tracking for training data")
            recommendations.append("Implement consent management system")
            compliant = False
        
        return {
            'compliant': compliant,
            'findings': findings,
            'recommendations': recommendations
        }
    
    def check_eu_ai_act_compliance(self, model_name: str) -> Dict:
        """Check EU AI Act compliance"""
        
        findings = []
        recommendations = []
        compliant = True
        
        # Determine risk category
        risk_level = self.assess_ai_risk_level(model_name)
        
        if risk_level == 'high_risk':
            # Additional requirements for high-risk AI
            
            # Check human oversight
            if not self.verify_human_oversight(model_name):
                findings.append("Insufficient human oversight mechanisms")
                recommendations.append("Implement human-in-the-loop controls")
                compliant = False
            
            # Check robustness and accuracy
            if not self.verify_robustness(model_name):
                findings.append("Model lacks robustness testing")
                recommendations.append("Implement adversarial testing")
                compliant = False
            
            # Check transparency
            if not self.verify_transparency(model_name):
                findings.append("Insufficient transparency documentation")
                recommendations.append("Create comprehensive model cards")
                compliant = False
        
        return {
            'compliant': compliant,
            'risk_level': risk_level,
            'findings': findings,
            'recommendations': recommendations
        }
    
    def verify_data_minimization(self, model_name: str) -> bool:
        """Verify data minimization practices"""
        
        # Check feature usage
        model_config = self.get_model_config(model_name)
        features = model_config.get('features', [])
        
        # Check for unnecessary PII
        pii_features = [f for f in features if self.is_pii(f)]
        
        # Verify each PII feature is necessary
        for feature in pii_features:
            if not self.is_feature_necessary(model_name, feature):
                return False
        
        return True
    
    def verify_explainability(self, model_name: str) -> bool:
        """Verify model explainability"""
        
        # Check if explainability module exists
        explainer_exists = self.check_explainer_exists(model_name)
        
        # Verify explanation quality
        if explainer_exists:
            explanation_quality = self.assess_explanation_quality(model_name)
            return explanation_quality > 0.8
        
        return False
    
    def generate_audit_report(self, audit_results: Dict):
        """Generate compliance audit report"""
        
        report = f"""
# AI/ML Compliance Audit Report

**Model**: {audit_results['model']}
**Date**: {audit_results['timestamp']}

## Compliance Summary

"""
        
        for standard, compliant in audit_results['compliance_status'].items():
            status = "✅ Compliant" if compliant else "❌ Non-Compliant"
            report += f"- **{standard}**: {status}\n"
        
        if audit_results['findings']:
            report += "\n## Findings\n\n"
            for finding in audit_results['findings']:
                report += f"- {finding}\n"
        
        if audit_results['recommendations']:
            report += "\n## Recommendations\n\n"
            for rec in audit_results['recommendations']:
                report += f"- {rec}\n"
        
        # Save report
        filename = f"/reports/ml-compliance-{audit_results['model']}-{datetime.now().strftime('%Y%m%d')}.md"
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Compliance report saved: {filename}")
```

---

## 8. Appendix

### ML Operations Commands

```bash
# Model management
mlflow models serve -m models:/ModelName/Production -p 5000
bentoml serve model_service:latest
torchserve --start --model-store model_store --models model.mar

# Training commands
python train.py --distributed --mixed-precision --gpu 4
horovodrun -np 4 python train_horovod.py
deepspeed train.py --deepspeed_config ds_config.json

# Inference optimization
python -m torch.utils.bottleneck inference.py
nsys profile python inference.py
tensorboard --logdir logs/profiling

# Model conversion
python -m tf2onnx.convert --saved-model model_dir --output model.onnx
trtexec --onnx=model.onnx --saveEngine=model.trt
torch.jit.script(model).save("model_scripted.pt")

# Monitoring
prometheus --config.file=prometheus.yml
grafana-server --config=/etc/grafana/grafana.ini
mlflow ui --host 0.0.0.0 --port 5000

# Dataset management
dvc add data/training_data.csv
dvc push
label-studio start --port 8080
```

### ML Metrics Reference

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Model Accuracy | >95% | 90-95% | <90% |
| Inference Latency P50 | <10ms | 10-50ms | >50ms |
| Inference Latency P99 | <100ms | 100-500ms | >500ms |
| Training Loss | <0.01 | 0.01-0.1 | >0.1 |
| Data Drift Score | <0.05 | 0.05-0.15 | >0.15 |
| GPU Utilization | 70-90% | 50-70% | <50% |
| Memory Usage | <80% | 80-90% | >90% |
| Model Size | <100MB | 100-500MB | >500MB |

### Troubleshooting Guide

| Issue | Symptoms | Solution |
|-------|----------|----------|
| High inference latency | P99 >500ms | Quantize model, use TensorRT, batch requests |
| Training not converging | Loss not decreasing | Adjust learning rate, check data quality |
| GPU OOM | CUDA out of memory | Reduce batch size, use gradient accumulation |
| Data drift | Accuracy dropping | Retrain with recent data, adjust features |
| Model too large | >1GB size | Prune weights, use knowledge distillation |

---

**Document Review Schedule**: Weekly
**Last Review**: 2025-01-05
**Next Review**: 2025-01-12
**Owner**: ML Operations Team