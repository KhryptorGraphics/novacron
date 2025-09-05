#!/usr/bin/env python3
"""
NovaCron ML Model Metrics Exporter

Custom Prometheus exporter for ML pipeline and MLE-Star workflow metrics.
Provides comprehensive monitoring for machine learning operations.
"""

import os
import time
import logging
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from enum import Enum

import prometheus_client
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info, Summary
from prometheus_client.core import CollectorRegistry, REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    SERVING = "serving"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

@dataclass
class MLMetrics:
    """Container for ML metrics data"""
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    training_time: float
    inference_latency: float
    prediction_count: int
    error_count: int
    status: ModelStatus
    last_updated: datetime
    drift_score: float = 0.0
    resource_usage: Dict[str, float] = None

class MLModelExporter:
    """Prometheus exporter for ML model metrics"""
    
    def __init__(self, ml_service_url: str, export_port: int = 8000, 
                 scrape_interval: int = 30):
        self.ml_service_url = ml_service_url.rstrip('/')
        self.export_port = export_port
        self.scrape_interval = scrape_interval
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Cache for model data
        self.model_cache: Dict[str, MLMetrics] = {}
        self.last_scrape_time = datetime.now()
        
        # Threading
        self.scrape_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info(f"ML Model Exporter initialized for {ml_service_url}")

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'ml_model_accuracy', 
            'Model accuracy score',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'ml_model_precision',
            'Model precision score', 
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'ml_model_recall',
            'Model recall score',
            ['model_name', 'version', 'environment'], 
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'ml_model_f1_score',
            'Model F1 score',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_loss = Gauge(
            'ml_model_loss',
            'Model training/validation loss',
            ['model_name', 'version', 'environment', 'type'],
            registry=self.registry
        )
        
        # Training metrics
        self.training_duration = Histogram(
            'ml_training_duration_seconds',
            'Model training duration in seconds',
            ['model_name', 'version', 'environment'],
            registry=self.registry,
            buckets=(60, 300, 900, 1800, 3600, 7200, 14400, 28800)
        )
        
        self.training_failures = Counter(
            'ml_training_failures_total',
            'Total number of training failures',
            ['model_name', 'version', 'environment', 'error_type'],
            registry=self.registry
        )
        
        # Inference metrics
        self.inference_latency = Histogram(
            'ml_inference_duration_seconds',
            'Model inference latency in seconds',
            ['model_name', 'version', 'environment'],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.predictions_total = Counter(
            'ml_predictions_total',
            'Total number of predictions made',
            ['model_name', 'version', 'environment', 'status'],
            registry=self.registry
        )
        
        self.prediction_errors = Counter(
            'ml_prediction_errors_total', 
            'Total number of prediction errors',
            ['model_name', 'version', 'environment', 'error_type'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            'Data drift detection score',
            ['model_name', 'dataset', 'feature'],
            registry=self.registry
        )
        
        self.feature_importance = Gauge(
            'ml_feature_importance',
            'Feature importance scores',
            ['model_name', 'version', 'feature_name'],
            registry=self.registry
        )
        
        # Resource usage metrics
        self.model_memory_usage = Gauge(
            'ml_model_memory_bytes',
            'Memory usage by model in bytes',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_cpu_usage = Gauge(
            'ml_model_cpu_usage_percent',
            'CPU usage by model',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_gpu_usage = Gauge(
            'ml_model_gpu_usage_percent',
            'GPU usage by model',
            ['model_name', 'version', 'environment', 'gpu_id'],
            registry=self.registry
        )
        
        # Model status
        self.model_status = Gauge(
            'ml_model_status',
            'Model status (0=stopped, 1=training, 2=serving, 3=failed)',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        self.model_uptime = Gauge(
            'ml_model_uptime_seconds',
            'Model uptime in seconds',
            ['model_name', 'version', 'environment'],
            registry=self.registry
        )
        
        # MLE-Star workflow metrics
        self.workflow_stage_duration = Histogram(
            'mle_star_stage_duration_seconds',
            'MLE-Star workflow stage duration',
            ['workflow_name', 'stage', 'model_name'],
            registry=self.registry,
            buckets=(10, 30, 60, 180, 300, 600, 1800, 3600)
        )
        
        self.workflow_failures = Counter(
            'mle_star_workflow_failures_total',
            'MLE-Star workflow failures',
            ['workflow_name', 'stage', 'model_name', 'error_type'],
            registry=self.registry
        )
        
        self.workflow_success_rate = Gauge(
            'mle_star_workflow_success_rate',
            'MLE-Star workflow success rate',
            ['workflow_name', 'stage'],
            registry=self.registry
        )
        
        # Model registry metrics
        self.model_versions = Gauge(
            'ml_model_versions_count',
            'Number of model versions in registry',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_deployments = Gauge(
            'ml_model_deployments_active',
            'Number of active model deployments', 
            ['model_name', 'environment'],
            registry=self.registry
        )
        
        # Pipeline health
        self.pipeline_health = Gauge(
            'ml_pipeline_health_status',
            'ML pipeline health status (1=healthy, 0=unhealthy)',
            ['pipeline_name', 'component'],
            registry=self.registry
        )
        
        self.scrape_duration = Summary(
            'ml_exporter_scrape_duration_seconds',
            'Time spent scraping ML metrics',
            registry=self.registry
        )
        
        self.scrape_errors = Counter(
            'ml_exporter_scrape_errors_total',
            'Errors encountered during scraping',
            ['error_type'],
            registry=self.registry
        )

    def _fetch_model_metrics(self) -> Dict[str, Any]:
        """Fetch metrics from ML service API"""
        try:
            # Get model list
            models_response = requests.get(
                f"{self.ml_service_url}/api/v1/models",
                timeout=10
            )
            models_response.raise_for_status()
            models = models_response.json()
            
            metrics_data = {}
            
            for model in models.get('models', []):
                model_name = model.get('name')
                if not model_name:
                    continue
                    
                try:
                    # Get detailed metrics for each model
                    model_metrics = self._fetch_single_model_metrics(model_name)
                    if model_metrics:
                        metrics_data[model_name] = model_metrics
                except Exception as e:
                    logger.error(f"Error fetching metrics for model {model_name}: {e}")
                    self.scrape_errors.labels(error_type="model_fetch_error").inc()
            
            # Get workflow metrics
            try:
                workflow_data = self._fetch_workflow_metrics()
                metrics_data['workflows'] = workflow_data
            except Exception as e:
                logger.error(f"Error fetching workflow metrics: {e}")
                self.scrape_errors.labels(error_type="workflow_fetch_error").inc()
            
            return metrics_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to ML service: {e}")
            self.scrape_errors.labels(error_type="connection_error").inc()
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching metrics: {e}")
            self.scrape_errors.labels(error_type="unexpected_error").inc()
            return {}

    def _fetch_single_model_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Fetch metrics for a single model"""
        endpoints = [
            f"/api/v1/models/{model_name}/metrics",
            f"/api/v1/models/{model_name}/status", 
            f"/api/v1/models/{model_name}/performance",
            f"/api/v1/models/{model_name}/drift"
        ]
        
        model_data = {}
        
        for endpoint in endpoints:
            try:
                response = requests.get(
                    f"{self.ml_service_url}{endpoint}",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    model_data.update(data)
            except Exception as e:
                logger.warning(f"Error fetching {endpoint} for {model_name}: {e}")
        
        return model_data if model_data else None

    def _fetch_workflow_metrics(self) -> Dict[str, Any]:
        """Fetch MLE-Star workflow metrics"""
        try:
            response = requests.get(
                f"{self.ml_service_url}/api/v1/workflows/metrics",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching workflow metrics: {e}")
            return {}

    def _update_metrics(self, metrics_data: Dict[str, Any]):
        """Update Prometheus metrics with fetched data"""
        for model_name, model_data in metrics_data.items():
            if model_name == 'workflows':
                self._update_workflow_metrics(model_data)
                continue
                
            self._update_single_model_metrics(model_name, model_data)

    def _update_single_model_metrics(self, model_name: str, model_data: Dict[str, Any]):
        """Update metrics for a single model"""
        version = model_data.get('version', 'unknown')
        environment = model_data.get('environment', 'production')
        
        # Performance metrics
        if 'accuracy' in model_data:
            self.model_accuracy.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).set(model_data['accuracy'])
            
        if 'precision' in model_data:
            self.model_precision.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).set(model_data['precision'])
            
        if 'recall' in model_data:
            self.model_recall.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).set(model_data['recall'])
            
        if 'f1_score' in model_data:
            self.model_f1_score.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).set(model_data['f1_score'])
            
        if 'loss' in model_data:
            self.model_loss.labels(
                model_name=model_name, 
                version=version, 
                environment=environment,
                type='validation'
            ).set(model_data['loss'])
            
        # Training metrics
        if 'training_time' in model_data:
            self.training_duration.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).observe(model_data['training_time'])
            
        # Inference metrics
        if 'inference_latency' in model_data:
            self.inference_latency.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).observe(model_data['inference_latency'])
            
        if 'predictions_count' in model_data:
            self.predictions_total.labels(
                model_name=model_name, 
                version=version, 
                environment=environment,
                status='success'
            )._value._value = model_data['predictions_count']
            
        # Data drift
        if 'drift_score' in model_data:
            self.data_drift_score.labels(
                model_name=model_name,
                dataset='production',
                feature='overall'
            ).set(model_data['drift_score'])
            
        # Resource usage
        if 'resource_usage' in model_data:
            resources = model_data['resource_usage']
            if 'memory_bytes' in resources:
                self.model_memory_usage.labels(
                    model_name=model_name, 
                    version=version, 
                    environment=environment
                ).set(resources['memory_bytes'])
                
            if 'cpu_percent' in resources:
                self.model_cpu_usage.labels(
                    model_name=model_name, 
                    version=version, 
                    environment=environment
                ).set(resources['cpu_percent'])
                
            if 'gpu_percent' in resources:
                for gpu_id, gpu_usage in resources['gpu_percent'].items():
                    self.model_gpu_usage.labels(
                        model_name=model_name, 
                        version=version, 
                        environment=environment,
                        gpu_id=str(gpu_id)
                    ).set(gpu_usage)
                    
        # Status
        status_map = {
            'training': 1,
            'serving': 2, 
            'failed': 3,
            'stopped': 0
        }
        status = model_data.get('status', 'unknown')
        self.model_status.labels(
            model_name=model_name, 
            version=version, 
            environment=environment
        ).set(status_map.get(status, 0))
        
        # Uptime
        if 'uptime_seconds' in model_data:
            self.model_uptime.labels(
                model_name=model_name, 
                version=version, 
                environment=environment
            ).set(model_data['uptime_seconds'])

    def _update_workflow_metrics(self, workflow_data: Dict[str, Any]):
        """Update MLE-Star workflow metrics"""
        for workflow_name, workflow_info in workflow_data.items():
            if not isinstance(workflow_info, dict):
                continue
                
            # Stage durations
            for stage, stage_data in workflow_info.get('stages', {}).items():
                if 'duration' in stage_data:
                    self.workflow_stage_duration.labels(
                        workflow_name=workflow_name,
                        stage=stage,
                        model_name=workflow_info.get('model_name', 'unknown')
                    ).observe(stage_data['duration'])
                    
                # Success rate
                if 'success_rate' in stage_data:
                    self.workflow_success_rate.labels(
                        workflow_name=workflow_name,
                        stage=stage
                    ).set(stage_data['success_rate'])

    def _scrape_metrics(self):
        """Main scraping loop"""
        while not self.shutdown_event.is_set():
            start_time = time.time()
            
            try:
                with self.scrape_duration.time():
                    logger.info("Starting metrics scrape")
                    metrics_data = self._fetch_model_metrics()
                    
                    if metrics_data:
                        self._update_metrics(metrics_data)
                        logger.info(f"Updated metrics for {len(metrics_data)} items")
                    else:
                        logger.warning("No metrics data received")
                        
            except Exception as e:
                logger.error(f"Error during metrics scrape: {e}")
                self.scrape_errors.labels(error_type="scrape_error").inc()
                
            # Calculate sleep time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.scrape_interval - elapsed)
            
            if self.shutdown_event.wait(timeout=sleep_time):
                break

    def start(self):
        """Start the exporter"""
        # Start Prometheus HTTP server
        start_http_server(self.export_port, registry=self.registry)
        logger.info(f"Started Prometheus exporter on port {self.export_port}")
        
        # Start scraping thread
        self.scrape_thread = threading.Thread(target=self._scrape_metrics)
        self.scrape_thread.daemon = True
        self.scrape_thread.start()
        logger.info("Started metrics scraping thread")

    def stop(self):
        """Stop the exporter"""
        logger.info("Stopping ML Model Exporter")
        self.shutdown_event.set()
        
        if self.scrape_thread and self.scrape_thread.is_alive():
            self.scrape_thread.join(timeout=5)
            
        logger.info("ML Model Exporter stopped")

def main():
    """Main function"""
    # Configuration from environment variables
    ml_service_url = os.getenv('ML_MODEL_ENDPOINT', 'http://ml-service:8000')
    export_port = int(os.getenv('EXPORT_PORT', '8000'))
    scrape_interval = int(os.getenv('SCRAPE_INTERVAL', '30'))
    
    # Create and start exporter
    exporter = MLModelExporter(
        ml_service_url=ml_service_url,
        export_port=export_port,
        scrape_interval=scrape_interval
    )
    
    try:
        exporter.start()
        logger.info(f"ML Model Exporter running on port {export_port}")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        exporter.stop()

if __name__ == '__main__':
    main()