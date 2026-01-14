"""
FastAPI application for NovaCron AI Operations Engine.

Provides REST API endpoints for all AI services including failure prediction,
workload optimization, anomaly detection, and resource optimization.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST

from ..config import get_settings, Settings
from ..core.failure_predictor import FailurePredictionService
from ..core.workload_optimizer import WorkloadPlacementService
from ..core.anomaly_detector import AnomalyDetectionService
from ..core.resource_optimizer import ResourceOptimizationService
from ..models.base import PredictionRequest, PredictionResponse
from ..utils.logging import setup_logging

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(
    level=settings.monitoring.log_level,
    format=settings.monitoring.log_format,
    log_file=settings.monitoring.log_file
)

logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()
request_count = Counter('ai_engine_requests_total', 'Total requests', ['method', 'endpoint'], registry=registry)
request_duration = Histogram('ai_engine_request_duration_seconds', 'Request duration', ['method', 'endpoint'], registry=registry)
active_models = Gauge('ai_engine_active_models', 'Number of active models', ['model_type'], registry=registry)

# Global service instances
failure_service: Optional[FailurePredictionService] = None
placement_service: Optional[WorkloadPlacementService] = None  
anomaly_service: Optional[AnomalyDetectionService] = None
resource_service: Optional[ResourceOptimizationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global failure_service, placement_service, anomaly_service, resource_service
    
    logger.info("Initializing NovaCron AI Operations Engine...")
    
    try:
        # Initialize services
        failure_service = FailurePredictionService(settings)
        await failure_service.initialize()
        
        placement_service = WorkloadPlacementService(settings)
        await placement_service.initialize()
        
        anomaly_service = AnomalyDetectionService(settings)
        await anomaly_service.initialize()
        
        resource_service = ResourceOptimizationService(settings)
        await resource_service.initialize()
        
        logger.info("AI Operations Engine initialized successfully")
        
        yield
        
        # Cleanup
        logger.info("Shutting down AI Operations Engine...")
        
        if failure_service:
            await failure_service.stop_monitoring()
        
        if anomaly_service:
            await anomaly_service.stop_monitoring()
            
        if resource_service:
            await resource_service.stop_optimization_monitoring()
        
        logger.info("AI Operations Engine shutdown complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Operations Engine: {str(e)}")
        raise


# Create FastAPI app
app = FastAPI(
    title="NovaCron AI Operations Engine",
    description="AI-powered operations engine for predictive VM management",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware
if settings.security.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency to get services
async def get_failure_service() -> FailurePredictionService:
    """Get failure prediction service."""
    if failure_service is None:
        raise HTTPException(status_code=503, detail="Failure prediction service not initialized")
    return failure_service


async def get_placement_service() -> WorkloadPlacementService:
    """Get workload placement service."""
    if placement_service is None:
        raise HTTPException(status_code=503, detail="Workload placement service not initialized")
    return placement_service


async def get_anomaly_service() -> AnomalyDetectionService:
    """Get anomaly detection service."""
    if anomaly_service is None:
        raise HTTPException(status_code=503, detail="Anomaly detection service not initialized")
    return anomaly_service


async def get_resource_service() -> ResourceOptimizationService:
    """Get resource optimization service."""
    if resource_service is None:
        raise HTTPException(status_code=503, detail="Resource optimization service not initialized")  
    return resource_service


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NovaCron AI Operations Engine",
        "version": "0.1.0",
        "status": "operational",
        "services": {
            "failure_prediction": failure_service is not None,
            "workload_placement": placement_service is not None,
            "anomaly_detection": anomaly_service is not None,
            "resource_optimization": resource_service is not None
        },
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs",
            "failure_prediction": "/api/v1/failure",
            "workload_placement": "/api/v1/placement", 
            "anomaly_detection": "/api/v1/anomaly",
            "resource_optimization": "/api/v1/resource"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "services": {
                "failure_prediction": {
                    "status": "healthy" if failure_service and failure_service.active_model else "no_model",
                    "active_model": failure_service.active_model.metadata.model_id if failure_service and failure_service.active_model else None
                },
                "workload_placement": {
                    "status": "healthy" if placement_service and placement_service.active_model else "no_model",
                    "active_model": placement_service.active_model.metadata.model_id if placement_service and placement_service.active_model else None
                },
                "anomaly_detection": {
                    "status": "healthy" if anomaly_service and anomaly_service.active_model else "no_model",
                    "active_model": anomaly_service.active_model.metadata.model_id if anomaly_service and anomaly_service.active_model else None
                },
                "resource_optimization": {
                    "status": "healthy" if resource_service and resource_service.active_model else "no_model",
                    "active_model": resource_service.active_model.metadata.model_id if resource_service and resource_service.active_model else None
                }
            }
        }
        
        # Determine overall status
        service_statuses = [service["status"] for service in health_status["services"].values()]
        if all(status == "healthy" for status in service_statuses):
            health_status["status"] = "healthy"
        elif any(status == "healthy" for status in service_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update active models gauge
        if failure_service and failure_service.active_model:
            active_models.labels(model_type="failure_prediction").set(1)
        else:
            active_models.labels(model_type="failure_prediction").set(0)
            
        if placement_service and placement_service.active_model:
            active_models.labels(model_type="workload_placement").set(1) 
        else:
            active_models.labels(model_type="workload_placement").set(0)
            
        if anomaly_service and anomaly_service.active_model:
            active_models.labels(model_type="anomaly_detection").set(1)
        else:
            active_models.labels(model_type="anomaly_detection").set(0)
            
        if resource_service and resource_service.active_model:
            active_models.labels(model_type="resource_optimization").set(1)
        else:
            active_models.labels(model_type="resource_optimization").set(0)
        
        metrics_output = generate_latest(registry)
        return JSONResponse(
            content=metrics_output.decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Metrics generation failed")


# Failure Prediction API
@app.post("/api/v1/failure/predict", response_model=PredictionResponse)
async def predict_failure(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    service: FailurePredictionService = Depends(get_failure_service)
):
    """Predict potential hardware failures."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/failure/predict").inc()
        
        with request_duration.labels(method="POST", endpoint="/api/v1/failure/predict").time():
            response = await service.predict_failures(request)
        
        # Log high-risk predictions in background
        if response.prediction == 1:
            background_tasks.add_task(
                _log_high_risk_prediction,
                "failure",
                request.request_id,
                response.confidence
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Failure prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/failure/models")
async def list_failure_models(
    service: FailurePredictionService = Depends(get_failure_service)
):
    """List available failure prediction models."""
    try:
        models = []
        for model_id, model in service.models.items():
            model_info = {
                "model_id": model_id,
                "is_active": service.active_model and service.active_model.metadata.model_id == model_id,
                "metadata": model.metadata.dict()
            }
            models.append(model_info)
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to list failure models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Workload Placement API  
@app.post("/api/v1/placement/optimize")
async def optimize_workload_placement(
    workload_request: Dict[str, Any],
    available_nodes: List[Dict[str, Any]],
    service: WorkloadPlacementService = Depends(get_placement_service)
):
    """Get optimal workload placement recommendations."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/placement/optimize").inc()
        
        with request_duration.labels(method="POST", endpoint="/api/v1/placement/optimize").time():
            candidates = await service.optimize_placement(workload_request, available_nodes)
        
        # Convert candidates to serializable format
        candidates_data = [
            {
                "node_id": candidate.node_id,
                "score": candidate.score,
                "resource_utilization": candidate.resource_utilization,
                "estimated_performance": candidate.estimated_performance,
                "constraints_satisfied": candidate.constraints_satisfied,
                "reasoning": candidate.reasoning
            }
            for candidate in candidates
        ]
        
        return {
            "workload_id": workload_request.get("workload_id", "unknown"),
            "candidates": candidates_data,
            "recommendation": candidates_data[0] if candidates_data else None
        }
        
    except Exception as e:
        logger.error(f"Workload placement optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/placement/batch")
async def batch_optimize_placement(
    workload_requests: List[Dict[str, Any]],
    available_nodes: List[Dict[str, Any]],
    service: WorkloadPlacementService = Depends(get_placement_service)
):
    """Optimize placement for multiple workloads."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/placement/batch").inc()
        
        with request_duration.labels(method="POST", endpoint="/api/v1/placement/batch").time():
            results = await service.batch_optimize(workload_requests, available_nodes)
        
        # Convert results to serializable format
        batch_results = {}
        for workload_id, candidates in results.items():
            batch_results[workload_id] = [
                {
                    "node_id": candidate.node_id,
                    "score": candidate.score,
                    "resource_utilization": candidate.resource_utilization,
                    "estimated_performance": candidate.estimated_performance,
                    "constraints_satisfied": candidate.constraints_satisfied,
                    "reasoning": candidate.reasoning
                }
                for candidate in candidates
            ]
        
        return {"results": batch_results}
        
    except Exception as e:
        logger.error(f"Batch placement optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/placement/factors")
async def get_placement_factors(
    service: WorkloadPlacementService = Depends(get_placement_service)
):
    """Get the complete list of placement factors."""
    try:
        factors = service.get_placement_factors()
        return {"placement_factors": factors, "total_factors": len(factors)}
        
    except Exception as e:
        logger.error(f"Failed to get placement factors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Anomaly Detection API
@app.post("/api/v1/anomaly/detect", response_model=PredictionResponse)
async def detect_anomaly(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """Detect anomalies in system metrics."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/anomaly/detect").inc()
        
        with request_duration.labels(method="POST", endpoint="/api/v1/anomaly/detect").time():
            response = await service.detect_anomalies(request)
        
        # Log anomalies in background
        if response.prediction == 1:
            background_tasks.add_task(
                _log_anomaly_detection,
                request.request_id,
                response.metadata.get("anomaly_types", []),
                response.metadata.get("severity", "unknown")
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/anomaly/batch")
async def batch_detect_anomalies(
    data: List[Dict[str, Any]],
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """Detect anomalies in batch data."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/anomaly/batch").inc()
        
        import pandas as pd
        
        with request_duration.labels(method="POST", endpoint="/api/v1/anomaly/batch").time():
            data_df = pd.DataFrame(data)
            results = await service.batch_detect(data_df)
        
        return {"results": results, "total_samples": len(data)}
        
    except Exception as e:
        logger.error(f"Batch anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/anomaly/trends")
async def get_anomaly_trends(
    hours: int = 24,
    service: AnomalyDetectionService = Depends(get_anomaly_service)
):
    """Get anomaly trends and statistics."""
    try:
        from datetime import timedelta
        
        time_window = timedelta(hours=hours)
        trends = await service.get_anomaly_trends(time_window)
        
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get anomaly trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Resource Optimization API
@app.post("/api/v1/resource/optimize")
async def optimize_resources(
    resource_data: List[Dict[str, Any]],
    objective: str = "balanced",
    service: ResourceOptimizationService = Depends(get_resource_service)
):
    """Get resource optimization recommendations."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/resource/optimize").inc()
        
        import pandas as pd
        from ..core.resource_optimizer import OptimizationObjective
        
        # Validate objective
        valid_objectives = [obj.value for obj in OptimizationObjective]
        if objective not in valid_objectives:
            raise HTTPException(status_code=400, detail=f"Invalid objective. Must be one of: {valid_objectives}")
        
        with request_duration.labels(method="POST", endpoint="/api/v1/resource/optimize").time():
            resource_df = pd.DataFrame(resource_data)
            recommendations = await service.get_optimization_recommendations(resource_df, objective)
        
        # Convert recommendations to serializable format
        recommendations_data = [rec.to_dict() for rec in recommendations]
        
        return {
            "recommendations": recommendations_data,
            "total_recommendations": len(recommendations_data),
            "objective": objective
        }
        
    except Exception as e:
        logger.error(f"Resource optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/resource/optimize-single")
async def optimize_single_workload_resources(
    workload_data: Dict[str, Any],
    objective: str = "balanced", 
    service: ResourceOptimizationService = Depends(get_resource_service)
):
    """Optimize resources for a single workload."""
    try:
        request_count.labels(method="POST", endpoint="/api/v1/resource/optimize-single").inc()
        
        from ..core.resource_optimizer import OptimizationObjective
        
        # Validate objective
        valid_objectives = [obj.value for obj in OptimizationObjective] 
        if objective not in valid_objectives:
            raise HTTPException(status_code=400, detail=f"Invalid objective. Must be one of: {valid_objectives}")
        
        with request_duration.labels(method="POST", endpoint="/api/v1/resource/optimize-single").time():
            recommendation = await service.optimize_single_workload(workload_data, objective)
        
        if recommendation:
            return {
                "recommendation": recommendation.to_dict(),
                "workload_id": workload_data.get("workload_id", "unknown"),
                "objective": objective
            }
        else:
            return {
                "recommendation": None,
                "message": "No optimization needed",
                "workload_id": workload_data.get("workload_id", "unknown"),
                "objective": objective
            }
        
    except Exception as e:
        logger.error(f"Single workload resource optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/resource/summary")
async def get_optimization_summary(
    hours: int = 24,
    service: ResourceOptimizationService = Depends(get_resource_service)
):
    """Get resource optimization summary and statistics."""
    try:
        from datetime import timedelta
        
        time_window = timedelta(hours=hours)
        summary = await service.get_optimization_summary(time_window)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get optimization summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Management API
@app.post("/api/v1/models/{service_type}/activate")
async def activate_model(
    service_type: str,
    model_id: str,
    failure_svc: FailurePredictionService = Depends(get_failure_service),
    placement_svc: WorkloadPlacementService = Depends(get_placement_service),
    anomaly_svc: AnomalyDetectionService = Depends(get_anomaly_service),
    resource_svc: ResourceOptimizationService = Depends(get_resource_service)
):
    """Activate a specific model for a service."""
    try:
        if service_type == "failure":
            failure_svc.set_active_model(model_id)
        elif service_type == "placement":
            placement_svc.set_active_model(model_id)
        elif service_type == "anomaly":
            anomaly_svc.set_active_model(model_id)
        elif service_type == "resource":
            resource_svc.set_active_model(model_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid service type")
        
        return {
            "status": "success",
            "message": f"Model {model_id} activated for {service_type} service"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Model activation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{service_type}/performance")
async def get_model_performance(
    service_type: str,
    model_id: str,
    failure_svc: FailurePredictionService = Depends(get_failure_service),
    placement_svc: WorkloadPlacementService = Depends(get_placement_service),
    anomaly_svc: AnomalyDetectionService = Depends(get_anomaly_service),
    resource_svc: ResourceOptimizationService = Depends(get_resource_service)
):
    """Get performance metrics for a specific model."""
    try:
        if service_type == "failure":
            metrics = failure_svc.get_model_metrics(model_id)
        elif service_type == "placement":
            metrics = placement_svc.get_placement_factors()  # Placeholder
        elif service_type == "anomaly":
            metrics = anomaly_svc.get_model_performance(model_id)
        elif service_type == "resource":
            metrics = resource_svc.get_model_performance(model_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid service type")
        
        return {"model_id": model_id, "service_type": service_type, "metrics": metrics}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _log_high_risk_prediction(prediction_type: str, request_id: str, confidence: float):
    """Log high-risk predictions for monitoring."""
    logger.warning(
        f"High-risk {prediction_type} prediction detected",
        extra={
            "request_id": request_id,
            "confidence": confidence,
            "prediction_type": prediction_type
        }
    )


async def _log_anomaly_detection(request_id: str, anomaly_types: List[str], severity: str):
    """Log anomaly detections for monitoring."""
    logger.warning(
        f"Anomaly detected",
        extra={
            "request_id": request_id,
            "anomaly_types": anomaly_types,
            "severity": severity
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "ai_engine.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.monitoring.log_level.lower(),
        access_log=True,
        reload=settings.debug
    )