"""
AI-driven resource optimization for NovaCron.

Provides intelligent resource allocation and scaling recommendations
using reinforcement learning and predictive analytics.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

from ..models.base import BaseMLModel, ModelMetadata, ModelType, PredictionRequest, PredictionResponse
from ..utils.feature_engineering import ResourceFeatureExtractor
from ..utils.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class ResourceAction:
    """Enumeration of possible resource optimization actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE = "migrate"
    CONSOLIDATE = "consolidate"
    SPLIT = "split"
    MAINTAIN = "maintain"


class OptimizationObjective:
    """Enumeration of optimization objectives."""
    COST = "cost"
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    SUSTAINABILITY = "sustainability"
    BALANCED = "balanced"


class ResourceRecommendation:
    """Resource optimization recommendation."""
    
    def __init__(self, action: str, target_resources: Dict[str, float], 
                 expected_impact: Dict[str, float], confidence: float,
                 reasoning: List[str], priority: int = 1):
        self.action = action
        self.target_resources = target_resources
        self.expected_impact = expected_impact
        self.confidence = confidence
        self.reasoning = reasoning
        self.priority = priority
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            'action': self.action,
            'target_resources': self.target_resources,
            'expected_impact': self.expected_impact,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat()
        }


class ResourceOptimizationModel(BaseMLModel):
    """Multi-objective resource optimization model."""
    
    def __init__(self, model_metadata: ModelMetadata):
        """Initialize resource optimization model."""
        super().__init__(model_metadata)
        
        # Prediction models for different objectives
        self._cost_model: Optional[lgb.LGBMRegressor] = None
        self._performance_model: Optional[RandomForestRegressor] = None
        self._efficiency_model: Optional[lgb.LGBMRegressor] = None
        self._sustainability_model: Optional[RandomForestRegressor] = None
        
        # Resource utilization predictor
        self._utilization_model: Optional[lgb.LGBMRegressor] = None
        
        # Feature processing
        self._scaler: Optional[StandardScaler] = None
        self._feature_extractor = ResourceFeatureExtractor()
        
        # Optimization parameters
        self._optimization_objectives = {
            OptimizationObjective.COST: 0.3,
            OptimizationObjective.PERFORMANCE: 0.4,
            OptimizationObjective.EFFICIENCY: 0.2,
            OptimizationObjective.SUSTAINABILITY: 0.1
        }
        
        # Resource constraints
        self._resource_constraints = {
            'cpu_min': 0.1,  # Minimum CPU cores
            'cpu_max': 64,   # Maximum CPU cores
            'memory_min': 0.5,  # Minimum memory GB
            'memory_max': 512,  # Maximum memory GB
            'storage_min': 10,   # Minimum storage GB
            'storage_max': 10000  # Maximum storage GB
        }
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame,
              validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None) -> Dict[str, float]:
        """
        Train the resource optimization model.
        
        Args:
            X: Training features (resource states, workload characteristics, usage patterns)
            y: Multi-target outcomes (cost, performance, efficiency, sustainability metrics)
            validation_data: Optional validation dataset
            
        Returns:
            Training metrics
        """
        logger.info(f"Training resource optimization model with {len(X)} samples")
        start_time = datetime.utcnow()
        
        try:
            # Extract resource optimization features
            logger.info("Extracting resource optimization features...")
            X_features = self._feature_extractor.extract_features(X)
            
            # Split data if validation not provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_features, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X_features, y
                X_val_features = self._feature_extractor.extract_features(validation_data[0])
                X_val, y_val = X_val_features, validation_data[1]
            
            # Scale features
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)
            
            # Store feature names
            self._feature_names = X_train.columns.tolist()
            
            # Train multi-objective models
            metrics = self._train_multi_objective_models(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            # Update metadata
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            self.update_metadata(
                training_status="completed",
                trained_at=datetime.utcnow(),
                training_duration=training_duration,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=len(self._feature_names),
                **metrics
            )
            
            self._is_trained = True
            logger.info(f"Model training completed in {training_duration:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.update_metadata(training_status="failed")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal resource configurations.
        
        Args:
            X: Input features (current resource state and workload)
            
        Returns:
            Predicted optimal resource allocations
        """
        recommendations = self.get_optimization_recommendations(X)
        
        # Extract target CPU allocation as the primary prediction
        cpu_allocations = []
        for rec in recommendations:
            if rec and 'cpu_cores' in rec.target_resources:
                cpu_allocations.append(rec.target_resources['cpu_cores'])
            else:
                cpu_allocations.append(0.0)
        
        return np.array(cpu_allocations)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence probabilities for resource recommendations.
        
        Args:
            X: Input features
            
        Returns:
            Confidence probabilities
        """
        recommendations = self.get_optimization_recommendations(X)
        
        # Extract confidence scores
        confidences = []
        for rec in recommendations:
            if rec:
                confidences.append(rec.confidence)
            else:
                confidences.append(0.0)
        
        # Convert to probability format
        confidences = np.array(confidences)
        return np.column_stack([1 - confidences, confidences])
    
    def get_optimization_recommendations(self, X: pd.DataFrame, 
                                       objective: str = OptimizationObjective.BALANCED) -> List[Optional[ResourceRecommendation]]:
        """
        Get detailed resource optimization recommendations.
        
        Args:
            X: Input features (resource states and workload characteristics)
            objective: Optimization objective
            
        Returns:
            List of resource optimization recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Process features
        X_features = self._feature_extractor.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        
        # Get predictions for all objectives
        cost_pred = self._cost_model.predict(X_scaled) if self._cost_model else None
        performance_pred = self._performance_model.predict(X_scaled) if self._performance_model else None
        efficiency_pred = self._efficiency_model.predict(X_scaled) if self._efficiency_model else None
        sustainability_pred = self._sustainability_model.predict(X_scaled) if self._sustainability_model else None
        utilization_pred = self._utilization_model.predict(X_scaled) if self._utilization_model else None
        
        recommendations = []
        for i in range(len(X)):
            try:
                rec = self._generate_recommendation(
                    X.iloc[i], X_features.iloc[i],
                    {
                        'cost': cost_pred[i] if cost_pred is not None else 0,
                        'performance': performance_pred[i] if performance_pred is not None else 0,
                        'efficiency': efficiency_pred[i] if efficiency_pred is not None else 0,
                        'sustainability': sustainability_pred[i] if sustainability_pred is not None else 0,
                        'utilization': utilization_pred[i] if utilization_pred is not None else 0
                    },
                    objective
                )
                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Failed to generate recommendation for sample {i}: {str(e)}")
                recommendations.append(None)
        
        return recommendations
    
    def _train_multi_objective_models(self, X_train: np.ndarray, y_train: pd.DataFrame,
                                    X_val: np.ndarray, y_val: pd.DataFrame) -> Dict[str, float]:
        """Train models for different optimization objectives."""
        
        metrics = {}
        
        # Cost optimization model
        if 'cost' in y_train.columns:
            logger.info("Training cost optimization model...")
            self._cost_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self._cost_model.fit(X_train, y_train['cost'])
            
            cost_pred = self._cost_model.predict(X_val)
            metrics['cost_r2'] = r2_score(y_val['cost'], cost_pred)
        
        # Performance optimization model
        if 'performance' in y_train.columns:
            logger.info("Training performance optimization model...")
            self._performance_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self._performance_model.fit(X_train, y_train['performance'])
            
            perf_pred = self._performance_model.predict(X_val)
            metrics['performance_r2'] = r2_score(y_val['performance'], perf_pred)
        
        # Efficiency optimization model
        if 'efficiency' in y_train.columns:
            logger.info("Training efficiency optimization model...")
            self._efficiency_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self._efficiency_model.fit(X_train, y_train['efficiency'])
            
            eff_pred = self._efficiency_model.predict(X_val)
            metrics['efficiency_r2'] = r2_score(y_val['efficiency'], eff_pred)
        
        # Sustainability model
        if 'sustainability' in y_train.columns:
            logger.info("Training sustainability model...")
            self._sustainability_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self._sustainability_model.fit(X_train, y_train['sustainability'])
            
            sust_pred = self._sustainability_model.predict(X_val)
            metrics['sustainability_r2'] = r2_score(y_val['sustainability'], sust_pred)
        
        # Resource utilization predictor
        if 'resource_utilization' in y_train.columns:
            logger.info("Training utilization model...")
            self._utilization_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self._utilization_model.fit(X_train, y_train['resource_utilization'])
            
            util_pred = self._utilization_model.predict(X_val)
            metrics['utilization_r2'] = r2_score(y_val['resource_utilization'], util_pred)
        
        # Overall metrics
        r2_scores = [v for k, v in metrics.items() if k.endswith('_r2')]
        if r2_scores:
            metrics['overall_r2'] = np.mean(r2_scores)
        
        return metrics
    
    def _generate_recommendation(self, original_features: pd.Series, processed_features: pd.Series,
                               predictions: Dict[str, float], objective: str) -> Optional[ResourceRecommendation]:
        """Generate a specific resource optimization recommendation."""
        
        current_cpu = original_features.get('cpu_cores', 1.0)
        current_memory = original_features.get('memory_gb', 4.0)
        current_storage = original_features.get('storage_gb', 100.0)
        
        cpu_utilization = original_features.get('cpu_utilization', 0.5)
        memory_utilization = original_features.get('memory_utilization', 0.5)
        storage_utilization = original_features.get('storage_utilization', 0.5)
        
        # Analyze current resource usage patterns
        analysis = self._analyze_resource_usage(
            cpu_utilization, memory_utilization, storage_utilization
        )
        
        # Determine optimization action
        action = self._determine_action(analysis, predictions, objective)
        
        if action == ResourceAction.MAINTAIN:
            return None  # No changes needed
        
        # Calculate target resource allocation
        target_resources = self._calculate_target_resources(
            current_cpu, current_memory, current_storage,
            analysis, action, predictions
        )
        
        # Calculate expected impact
        expected_impact = self._calculate_expected_impact(
            original_features, target_resources, predictions
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(analysis, action, target_resources, expected_impact)
        
        # Calculate confidence
        confidence = self._calculate_recommendation_confidence(analysis, predictions)
        
        # Determine priority
        priority = self._calculate_priority(action, expected_impact, confidence)
        
        return ResourceRecommendation(
            action=action,
            target_resources=target_resources,
            expected_impact=expected_impact,
            confidence=confidence,
            reasoning=reasoning,
            priority=priority
        )
    
    def _analyze_resource_usage(self, cpu_util: float, memory_util: float, 
                              storage_util: float) -> Dict[str, Any]:
        """Analyze current resource usage patterns."""
        
        # Define utilization thresholds
        high_threshold = 0.8
        low_threshold = 0.3
        
        analysis = {
            'cpu_status': 'normal',
            'memory_status': 'normal',
            'storage_status': 'normal',
            'overall_utilization': (cpu_util + memory_util + storage_util) / 3.0,
            'bottlenecks': [],
            'underutilized': []
        }
        
        # Analyze CPU
        if cpu_util > high_threshold:
            analysis['cpu_status'] = 'high'
            analysis['bottlenecks'].append('cpu')
        elif cpu_util < low_threshold:
            analysis['cpu_status'] = 'low'
            analysis['underutilized'].append('cpu')
        
        # Analyze Memory
        if memory_util > high_threshold:
            analysis['memory_status'] = 'high'
            analysis['bottlenecks'].append('memory')
        elif memory_util < low_threshold:
            analysis['memory_status'] = 'low'
            analysis['underutilized'].append('memory')
        
        # Analyze Storage
        if storage_util > high_threshold:
            analysis['storage_status'] = 'high'
            analysis['bottlenecks'].append('storage')
        elif storage_util < low_threshold:
            analysis['storage_status'] = 'low'
            analysis['underutilized'].append('storage')
        
        return analysis
    
    def _determine_action(self, analysis: Dict[str, Any], predictions: Dict[str, float],
                         objective: str) -> str:
        """Determine the optimization action to take."""
        
        # Check for bottlenecks (scale up)
        if analysis['bottlenecks']:
            if len(analysis['bottlenecks']) >= 2:
                return ResourceAction.SCALE_UP
            elif analysis['overall_utilization'] > 0.7:
                return ResourceAction.SCALE_UP
        
        # Check for underutilization (scale down or consolidate)
        if len(analysis['underutilized']) >= 2 and analysis['overall_utilization'] < 0.4:
            if objective == OptimizationObjective.COST:
                return ResourceAction.SCALE_DOWN
            else:
                return ResourceAction.CONSOLIDATE
        
        # Check for migration opportunities based on predictions
        if predictions.get('performance', 0) < 0.5:
            return ResourceAction.MIGRATE
        
        # Check for splitting opportunities (high load variance)
        if analysis['overall_utilization'] > 0.8 and len(analysis['bottlenecks']) == 1:
            return ResourceAction.SPLIT
        
        return ResourceAction.MAINTAIN
    
    def _calculate_target_resources(self, current_cpu: float, current_memory: float,
                                  current_storage: float, analysis: Dict[str, Any],
                                  action: str, predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate target resource allocation based on action."""
        
        target_resources = {
            'cpu_cores': current_cpu,
            'memory_gb': current_memory,
            'storage_gb': current_storage
        }
        
        if action == ResourceAction.SCALE_UP:
            # Increase resources for bottlenecked components
            if 'cpu' in analysis['bottlenecks']:
                target_resources['cpu_cores'] = min(
                    current_cpu * 1.5,
                    self._resource_constraints['cpu_max']
                )
            if 'memory' in analysis['bottlenecks']:
                target_resources['memory_gb'] = min(
                    current_memory * 1.4,
                    self._resource_constraints['memory_max']
                )
            if 'storage' in analysis['bottlenecks']:
                target_resources['storage_gb'] = min(
                    current_storage * 1.3,
                    self._resource_constraints['storage_max']
                )
        
        elif action == ResourceAction.SCALE_DOWN:
            # Decrease underutilized resources
            if 'cpu' in analysis['underutilized']:
                target_resources['cpu_cores'] = max(
                    current_cpu * 0.7,
                    self._resource_constraints['cpu_min']
                )
            if 'memory' in analysis['underutilized']:
                target_resources['memory_gb'] = max(
                    current_memory * 0.8,
                    self._resource_constraints['memory_min']
                )
            if 'storage' in analysis['underutilized']:
                target_resources['storage_gb'] = max(
                    current_storage * 0.9,
                    self._resource_constraints['storage_min']
                )
        
        elif action == ResourceAction.CONSOLIDATE:
            # Optimize for efficiency
            efficiency_factor = predictions.get('efficiency', 0.5)
            target_resources['cpu_cores'] = max(
                current_cpu * (0.8 + 0.2 * efficiency_factor),
                self._resource_constraints['cpu_min']
            )
            target_resources['memory_gb'] = max(
                current_memory * (0.8 + 0.2 * efficiency_factor),
                self._resource_constraints['memory_min']
            )
        
        elif action == ResourceAction.SPLIT:
            # Increase resources slightly for splitting
            target_resources['cpu_cores'] = min(
                current_cpu * 1.2,
                self._resource_constraints['cpu_max']
            )
            target_resources['memory_gb'] = min(
                current_memory * 1.2,
                self._resource_constraints['memory_max']
            )
        
        return target_resources
    
    def _calculate_expected_impact(self, original_features: pd.Series,
                                 target_resources: Dict[str, float],
                                 predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate expected impact of resource changes."""
        
        current_cpu = original_features.get('cpu_cores', 1.0)
        current_memory = original_features.get('memory_gb', 4.0)
        
        cpu_change = target_resources['cpu_cores'] / current_cpu - 1.0
        memory_change = target_resources['memory_gb'] / current_memory - 1.0
        
        # Estimate impact based on resource changes and predictions
        performance_impact = (cpu_change * 0.6 + memory_change * 0.4) * predictions.get('performance', 0.5)
        cost_impact = (cpu_change * 0.7 + memory_change * 0.3) * predictions.get('cost', 0.5)
        efficiency_impact = predictions.get('efficiency', 0.5) - (abs(cpu_change) + abs(memory_change)) * 0.1
        
        return {
            'performance_change': performance_impact,
            'cost_change': cost_impact,
            'efficiency_change': efficiency_impact,
            'resource_utilization_change': predictions.get('utilization', 0.5) - 0.5
        }
    
    def _generate_reasoning(self, analysis: Dict[str, Any], action: str,
                          target_resources: Dict[str, float], expected_impact: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = []
        
        # Analysis-based reasoning
        if analysis['bottlenecks']:
            reasoning.append(f"Detected bottlenecks in: {', '.join(analysis['bottlenecks'])}")
        
        if analysis['underutilized']:
            reasoning.append(f"Underutilized resources: {', '.join(analysis['underutilized'])}")
        
        # Action-specific reasoning
        if action == ResourceAction.SCALE_UP:
            reasoning.append("Scaling up to address performance bottlenecks")
        elif action == ResourceAction.SCALE_DOWN:
            reasoning.append("Scaling down to reduce costs and improve efficiency")
        elif action == ResourceAction.CONSOLIDATE:
            reasoning.append("Consolidating resources for better efficiency")
        elif action == ResourceAction.MIGRATE:
            reasoning.append("Migration recommended for better performance")
        elif action == ResourceAction.SPLIT:
            reasoning.append("Splitting workload to handle high resource demands")
        
        # Impact-based reasoning
        if expected_impact['performance_change'] > 0.1:
            reasoning.append(f"Expected performance improvement: {expected_impact['performance_change']:.1%}")
        elif expected_impact['performance_change'] < -0.1:
            reasoning.append(f"Expected performance trade-off: {abs(expected_impact['performance_change']):.1%}")
        
        if expected_impact['cost_change'] > 0.1:
            reasoning.append(f"Expected cost increase: {expected_impact['cost_change']:.1%}")
        elif expected_impact['cost_change'] < -0.1:
            reasoning.append(f"Expected cost savings: {abs(expected_impact['cost_change']):.1%}")
        
        return reasoning
    
    def _calculate_recommendation_confidence(self, analysis: Dict[str, Any],
                                           predictions: Dict[str, float]) -> float:
        """Calculate confidence in the recommendation."""
        
        base_confidence = 0.7
        
        # Increase confidence for clear resource issues
        if len(analysis['bottlenecks']) >= 2:
            base_confidence += 0.2
        elif len(analysis['underutilized']) >= 2:
            base_confidence += 0.15
        
        # Adjust based on prediction quality
        prediction_confidence = np.mean(list(predictions.values()))
        confidence = base_confidence * (0.5 + 0.5 * prediction_confidence)
        
        return min(confidence, 1.0)
    
    def _calculate_priority(self, action: str, expected_impact: Dict[str, float],
                          confidence: float) -> int:
        """Calculate recommendation priority (1 = highest, 5 = lowest)."""
        
        # Base priority by action type
        action_priorities = {
            ResourceAction.SCALE_UP: 2,      # High priority for performance issues
            ResourceAction.SCALE_DOWN: 4,    # Lower priority for cost optimization
            ResourceAction.MIGRATE: 3,       # Medium priority
            ResourceAction.CONSOLIDATE: 4,   # Lower priority
            ResourceAction.SPLIT: 2,         # High priority for capacity issues
            ResourceAction.MAINTAIN: 5       # Lowest priority
        }
        
        base_priority = action_priorities.get(action, 3)
        
        # Adjust based on expected impact magnitude
        impact_magnitude = abs(expected_impact.get('performance_change', 0)) + \
                          abs(expected_impact.get('cost_change', 0))
        
        if impact_magnitude > 0.3:
            base_priority = max(1, base_priority - 1)
        elif impact_magnitude < 0.1:
            base_priority = min(5, base_priority + 1)
        
        # Adjust based on confidence
        if confidence > 0.9:
            base_priority = max(1, base_priority - 1)
        elif confidence < 0.6:
            base_priority = min(5, base_priority + 1)
        
        return base_priority
    
    def save_model(self, filepath: str) -> None:
        """Save the resource optimization model."""
        import joblib
        
        model_data = {
            'cost_model': self._cost_model,
            'performance_model': self._performance_model,
            'efficiency_model': self._efficiency_model,
            'sustainability_model': self._sustainability_model,
            'utilization_model': self._utilization_model,
            'scaler': self._scaler,
            'feature_names': self._feature_names,
            'optimization_objectives': self._optimization_objectives,
            'resource_constraints': self._resource_constraints,
            'metadata': self.metadata.dict()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the resource optimization model."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self._cost_model = model_data['cost_model']
        self._performance_model = model_data['performance_model']
        self._efficiency_model = model_data['efficiency_model']
        self._sustainability_model = model_data['sustainability_model']
        self._utilization_model = model_data['utilization_model']
        self._scaler = model_data['scaler']
        self._feature_names = model_data['feature_names']
        self._optimization_objectives = model_data['optimization_objectives']
        self._resource_constraints = model_data['resource_constraints']
        
        self._is_trained = True
        logger.info(f"Model loaded from {filepath}")


class ResourceOptimizationService:
    """Service for AI-driven resource optimization and recommendations."""
    
    def __init__(self, settings):
        """Initialize resource optimization service."""
        self.settings = settings
        self.models: Dict[str, ResourceOptimizationModel] = {}
        self.active_model: Optional[ResourceOptimizationModel] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Recommendation history
        self._recommendation_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
    
    async def initialize(self) -> None:
        """Initialize the service and load models."""
        logger.info("Initializing resource optimization service...")
        
        # Load existing models
        await self._load_models()
        
        # Start optimization monitoring if we have an active model
        if self.active_model:
            await self.start_optimization_monitoring()
    
    async def get_optimization_recommendations(self, resource_data: pd.DataFrame,
                                             objective: str = OptimizationObjective.BALANCED) -> List[ResourceRecommendation]:
        """
        Get resource optimization recommendations.
        
        Args:
            resource_data: Current resource states and workload data
            objective: Optimization objective
            
        Returns:
            List of resource optimization recommendations
        """
        if not self.active_model or not self.active_model.is_trained:
            raise ValueError("No active trained model available for optimization")
        
        # Get recommendations from model
        recommendations = self.active_model.get_optimization_recommendations(
            resource_data, objective
        )
        
        # Filter out None recommendations and sort by priority
        valid_recommendations = [rec for rec in recommendations if rec is not None]
        valid_recommendations.sort(key=lambda x: x.priority)
        
        # Store recommendations in history
        for rec in valid_recommendations:
            await self._store_recommendation(rec)
        
        logger.info(f"Generated {len(valid_recommendations)} optimization recommendations")
        
        return valid_recommendations
    
    async def optimize_single_workload(self, workload_data: Dict[str, Any],
                                     objective: str = OptimizationObjective.BALANCED) -> Optional[ResourceRecommendation]:
        """
        Optimize resources for a single workload.
        
        Args:
            workload_data: Single workload resource data
            objective: Optimization objective
            
        Returns:
            Resource optimization recommendation
        """
        workload_df = pd.DataFrame([workload_data])
        recommendations = await self.get_optimization_recommendations(workload_df, objective)
        
        return recommendations[0] if recommendations else None
    
    async def batch_optimize(self, workloads_data: List[Dict[str, Any]],
                           objective: str = OptimizationObjective.BALANCED) -> List[ResourceRecommendation]:
        """
        Optimize resources for multiple workloads.
        
        Args:
            workloads_data: List of workload data
            objective: Optimization objective
            
        Returns:
            List of resource optimization recommendations
        """
        workloads_df = pd.DataFrame(workloads_data)
        return await self.get_optimization_recommendations(workloads_df, objective)
    
    async def start_optimization_monitoring(self) -> None:
        """Start continuous resource optimization monitoring."""
        if self._optimization_task and not self._optimization_task.done():
            logger.warning("Optimization monitoring task is already running")
            return
        
        logger.info("Starting resource optimization monitoring...")
        self._optimization_task = asyncio.create_task(self._optimization_monitoring_loop())
    
    async def stop_optimization_monitoring(self) -> None:
        """Stop continuous resource optimization monitoring."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
            logger.info("Resource optimization monitoring stopped")
    
    async def get_optimization_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Get optimization summary and statistics.
        
        Args:
            time_window: Time window for summary
            
        Returns:
            Optimization summary statistics
        """
        cutoff_time = datetime.utcnow() - time_window
        
        # Filter recent recommendations
        recent_recommendations = [
            rec for rec in self._recommendation_history
            if datetime.fromisoformat(rec['timestamp']) > cutoff_time
        ]
        
        if not recent_recommendations:
            return {
                'total_recommendations': 0,
                'actions_distribution': {},
                'priority_distribution': {},
                'average_confidence': 0.0,
                'potential_savings': 0.0
            }
        
        # Calculate statistics
        total_recommendations = len(recent_recommendations)
        
        # Action distribution
        actions_dist = {}
        for rec in recent_recommendations:
            action = rec['action']
            actions_dist[action] = actions_dist.get(action, 0) + 1
        
        # Priority distribution
        priority_dist = {}
        for rec in recent_recommendations:
            priority = rec['priority']
            priority_dist[f"priority_{priority}"] = priority_dist.get(f"priority_{priority}", 0) + 1
        
        # Average confidence
        avg_confidence = np.mean([rec['confidence'] for rec in recent_recommendations])
        
        # Potential savings (simplified calculation)
        potential_savings = sum([
            rec.get('expected_impact', {}).get('cost_change', 0)
            for rec in recent_recommendations
            if rec.get('expected_impact', {}).get('cost_change', 0) < 0
        ])
        
        return {
            'total_recommendations': total_recommendations,
            'actions_distribution': actions_dist,
            'priority_distribution': priority_dist,
            'average_confidence': float(avg_confidence),
            'potential_savings': abs(potential_savings),
            'time_window_hours': time_window.total_seconds() / 3600
        }
    
    async def _optimization_monitoring_loop(self) -> None:
        """Main monitoring loop for continuous resource optimization."""
        while True:
            try:
                # Get current resource states from NovaCron API
                resource_data = await self._fetch_current_resource_states()
                
                if resource_data is None or resource_data.empty:
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
                    continue
                
                # Get optimization recommendations
                recommendations = await self.get_optimization_recommendations(
                    resource_data, OptimizationObjective.BALANCED
                )
                
                # Process high-priority recommendations
                high_priority_recs = [rec for rec in recommendations if rec.priority <= 2]
                
                for rec in high_priority_recs:
                    await self._process_optimization_recommendation(rec, resource_data)
                
                # Wait before next optimization cycle (30 minutes)
                await asyncio.sleep(1800)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in optimization monitoring loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _fetch_current_resource_states(self) -> Optional[pd.DataFrame]:
        """Fetch current resource states from NovaCron API."""
        # This would integrate with the NovaCron monitoring API
        # For now, return mock data structure
        return None
    
    async def _process_optimization_recommendation(self, recommendation: ResourceRecommendation,
                                                 resource_data: pd.DataFrame) -> None:
        """Process and potentially execute a high-priority recommendation."""
        
        logger.info(
            f"Processing high-priority recommendation: {recommendation.action} "
            f"(confidence: {recommendation.confidence:.2f})"
        )
        
        # Log the recommendation
        for reason in recommendation.reasoning:
            logger.info(f"  Reason: {reason}")
        
        # In a real implementation, this would:
        # 1. Validate the recommendation against current state
        # 2. Check if similar recommendations are already being executed
        # 3. Send the recommendation to the NovaCron orchestration system
        # 4. Track implementation status
        
        # For now, just log the recommendation
        logger.info(f"Recommendation logged for manual review: {recommendation.to_dict()}")
    
    async def _store_recommendation(self, recommendation: ResourceRecommendation) -> None:
        """Store recommendation in history buffer."""
        rec_dict = recommendation.to_dict()
        
        # Add to history buffer
        self._recommendation_history.append(rec_dict)
        
        # Maintain buffer size
        if len(self._recommendation_history) > self._max_history_size:
            self._recommendation_history = self._recommendation_history[-self._max_history_size:]
        
        # Store in database
        # Implementation would store in PostgreSQL via SQLAlchemy
    
    async def _load_models(self) -> None:
        """Load existing models from storage."""
        # Implementation would scan model storage directory and load models
        pass
    
    def set_active_model(self, model_id: str) -> None:
        """Set the active model for optimization."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.active_model = self.models[model_id]
        logger.info(f"Active resource optimization model set to {model_id}")
    
    def update_optimization_objectives(self, objectives: Dict[str, float]) -> None:
        """Update optimization objective weights."""
        if not self.active_model:
            raise ValueError("No active model to update")
        
        # Validate weights sum to 1.0
        if abs(sum(objectives.values()) - 1.0) > 0.01:
            raise ValueError("Optimization objective weights must sum to 1.0")
        
        self.active_model._optimization_objectives.update(objectives)
        logger.info(f"Updated optimization objectives: {objectives}")
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        return {
            'model_id': model_id,
            'overall_r2': getattr(model.metadata, 'overall_r2', None),
            'cost_r2': getattr(model.metadata, 'cost_r2', None),
            'performance_r2': getattr(model.metadata, 'performance_r2', None),
            'efficiency_r2': getattr(model.metadata, 'efficiency_r2', None),
            'training_samples': model.metadata.training_samples,
            'feature_count': model.metadata.feature_count,
            'trained_at': model.metadata.trained_at
        }