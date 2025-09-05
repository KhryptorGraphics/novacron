"""
A/B Testing Framework
====================

A/B testing framework for model comparison in production:
- Statistical experiment design and power analysis
- Traffic splitting and randomization
- Performance monitoring and statistical significance testing
- Multi-armed bandit optimization
- Automated decision making and rollout strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

try:
    from scipy import stats
    from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ABTestConfig:
    """Configuration for A/B testing framework"""
    # Statistical configuration
    significance_level: float = 0.05
    power: float = 0.8
    minimum_effect_size: float = 0.02  # 2% relative improvement
    
    # Traffic allocation
    default_control_split: float = 0.5
    min_sample_size_per_variant: int = 1000
    
    # Experiment duration
    min_experiment_duration_days: int = 7
    max_experiment_duration_days: int = 30
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_check_interval_hours: int = 24
    futility_threshold: float = 0.1  # Stop if probability of success < 10%
    
    # Multi-armed bandit
    enable_bandit_optimization: bool = False
    bandit_algorithm: str = "thompson_sampling"  # epsilon_greedy, ucb, thompson_sampling
    exploration_rate: float = 0.1
    
    # Metrics and monitoring
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = field(default_factory=lambda: ["revenue", "engagement"])
    
    # Data collection
    data_retention_days: int = 90
    enable_real_time_monitoring: bool = True
    
    # Output
    results_storage_path: str = "./ab_test_results"


@dataclass
class ExperimentMetrics:
    """Experiment metrics structure"""
    metric_name: str
    control_value: float
    treatment_value: float
    lift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    is_significant: bool


@dataclass 
class ABTestExperiment:
    """A/B test experiment definition"""
    experiment_id: str
    name: str
    description: str
    
    # Models being compared
    control_model_id: str
    treatment_model_ids: List[str]
    
    # Traffic allocation
    traffic_allocation: Dict[str, float]  # variant_id -> proportion
    
    # Statistical configuration
    primary_metric: str
    secondary_metrics: List[str]
    significance_level: float
    minimum_effect_size: float
    
    # Timing
    start_time: datetime
    planned_end_time: datetime
    actual_end_time: Optional[datetime] = None
    
    # Status and results
    status: ExperimentStatus = ExperimentStatus.DRAFT
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    results: Dict[str, ExperimentMetrics] = field(default_factory=dict)
    
    # Configuration
    enable_early_stopping: bool = True
    enable_bandit: bool = False


class StatisticalEngine:
    """Statistical analysis engine for A/B tests"""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
    
    def calculate_sample_size(self, baseline_rate: float, effect_size: float,
                            alpha: float = None, power: float = None) -> int:
        """Calculate required sample size for experiment"""
        if not SCIPY_AVAILABLE:
            # Simple approximation without scipy
            return max(1000, int(10000 * effect_size))
        
        alpha = alpha or self.config.significance_level
        power = power or self.config.power
        
        try:
            # Two-proportion z-test sample size calculation
            p1 = baseline_rate
            p2 = baseline_rate * (1 + effect_size)
            
            # Pooled proportion
            p_pool = (p1 + p2) / 2
            
            # Z-scores
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            # Sample size per group
            n = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + 
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
            
            return max(self.config.min_sample_size_per_variant, int(np.ceil(n)))
            
        except Exception as e:
            logger.warning(f"Sample size calculation failed: {e}")
            return self.config.min_sample_size_per_variant
    
    def analyze_experiment(self, control_data: np.ndarray, treatment_data: np.ndarray,
                          metric_name: str, metric_type: str = "continuous") -> ExperimentMetrics:
        """Analyze experiment results"""
        try:
            if metric_type == "binary":
                return self._analyze_binary_metric(control_data, treatment_data, metric_name)
            else:
                return self._analyze_continuous_metric(control_data, treatment_data, metric_name)
                
        except Exception as e:
            logger.error(f"Experiment analysis failed for {metric_name}: {e}")
            # Return default result
            return ExperimentMetrics(
                metric_name=metric_name,
                control_value=0.0,
                treatment_value=0.0,
                lift=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                statistical_power=0.0,
                is_significant=False
            )
    
    def _analyze_binary_metric(self, control_data: np.ndarray, treatment_data: np.ndarray,
                              metric_name: str) -> ExperimentMetrics:
        """Analyze binary metric (conversion rate, click-through rate, etc.)"""
        control_rate = np.mean(control_data)
        treatment_rate = np.mean(treatment_data)
        lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        
        if SCIPY_AVAILABLE:
            # Chi-square test
            control_successes = np.sum(control_data)
            control_failures = len(control_data) - control_successes
            treatment_successes = np.sum(treatment_data)
            treatment_failures = len(treatment_data) - treatment_successes
            
            contingency_table = [[control_successes, control_failures],
                                [treatment_successes, treatment_failures]]
            
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Confidence interval for difference in proportions
            n1, n2 = len(control_data), len(treatment_data)
            p1, p2 = control_rate, treatment_rate
            
            se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
            z_score = stats.norm.ppf(1 - self.config.significance_level/2)
            
            diff = p2 - p1
            ci_lower = diff - z_score * se_diff
            ci_upper = diff + z_score * se_diff
            
            confidence_interval = (ci_lower, ci_upper)
            
            # Statistical power calculation
            effect_size = abs(diff) / np.sqrt(p1*(1-p1))
            power = self._calculate_power(effect_size, n1, n2)
            
        else:
            # Simple approximation
            p_value = 0.5  # Placeholder
            confidence_interval = (lift - 0.1, lift + 0.1)
            power = 0.8
        
        return ExperimentMetrics(
            metric_name=metric_name,
            control_value=control_rate,
            treatment_value=treatment_rate,
            lift=lift,
            p_value=p_value,
            confidence_interval=confidence_interval,
            statistical_power=power,
            is_significant=p_value < self.config.significance_level
        )
    
    def _analyze_continuous_metric(self, control_data: np.ndarray, treatment_data: np.ndarray,
                                  metric_name: str) -> ExperimentMetrics:
        """Analyze continuous metric (revenue, latency, etc.)"""
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        
        if SCIPY_AVAILABLE:
            # Two-sample t-test
            t_stat, p_value = ttest_ind(treatment_data, control_data)
            
            # Confidence interval for difference in means
            n1, n2 = len(control_data), len(treatment_data)
            s1, s2 = np.std(control_data, ddof=1), np.std(treatment_data, ddof=1)
            
            # Pooled standard error
            se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
            
            # Degrees of freedom (Welch's t-test)
            df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            
            t_critical = stats.t.ppf(1 - self.config.significance_level/2, df)
            
            diff = treatment_mean - control_mean
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            confidence_interval = (ci_lower, ci_upper)
            
            # Statistical power
            effect_size = abs(diff) / np.sqrt((s1**2 + s2**2) / 2)
            power = self._calculate_power(effect_size, n1, n2)
            
        else:
            # Simple approximation
            p_value = 0.5
            confidence_interval = (lift - 0.1, lift + 0.1)
            power = 0.8
        
        return ExperimentMetrics(
            metric_name=metric_name,
            control_value=control_mean,
            treatment_value=treatment_mean,
            lift=lift,
            p_value=p_value,
            confidence_interval=confidence_interval,
            statistical_power=power,
            is_significant=p_value < self.config.significance_level
        )
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power"""
        if not SCIPY_AVAILABLE:
            return 0.8  # Default assumption
        
        try:
            # Approximate power calculation
            alpha = self.config.significance_level
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(n1 * n2 / (n1 + n2)) - z_alpha
            power = stats.norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))
            
        except:
            return 0.8
    
    def should_stop_early(self, experiment: ABTestExperiment) -> Tuple[bool, str]:
        """Determine if experiment should be stopped early"""
        if not experiment.enable_early_stopping:
            return False, "Early stopping disabled"
        
        # Check minimum runtime
        runtime = datetime.now() - experiment.start_time
        if runtime.days < 3:  # Minimum 3 days
            return False, "Minimum runtime not met"
        
        # Check if we have results for primary metric
        primary_result = experiment.results.get(experiment.primary_metric)
        if not primary_result:
            return False, "No results for primary metric"
        
        # Early stopping for significance
        if primary_result.is_significant and primary_result.statistical_power > 0.8:
            return True, "Statistical significance achieved with adequate power"
        
        # Futility stopping
        if primary_result.p_value > 0.8:  # Very unlikely to achieve significance
            return True, "Futility threshold met - unlikely to achieve significance"
        
        return False, "Continue experiment"


class MultiArmedBandit:
    """Multi-armed bandit for dynamic traffic allocation"""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.algorithm = config.bandit_algorithm
        self.exploration_rate = config.exploration_rate
    
    def select_variant(self, variants: List[str], performance_history: Dict[str, List[float]]) -> str:
        """Select variant using bandit algorithm"""
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy(variants, performance_history)
        elif self.algorithm == "ucb":
            return self._upper_confidence_bound(variants, performance_history)
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling(variants, performance_history)
        else:
            # Random selection as fallback
            return np.random.choice(variants)
    
    def _epsilon_greedy(self, variants: List[str], performance_history: Dict[str, List[float]]) -> str:
        """Epsilon-greedy selection"""
        if np.random.random() < self.exploration_rate:
            # Explore: random selection
            return np.random.choice(variants)
        else:
            # Exploit: select best performing variant
            avg_performance = {}
            for variant in variants:
                history = performance_history.get(variant, [])
                avg_performance[variant] = np.mean(history) if history else 0.0
            
            return max(avg_performance, key=avg_performance.get)
    
    def _upper_confidence_bound(self, variants: List[str], performance_history: Dict[str, List[float]]) -> str:
        """Upper Confidence Bound selection"""
        total_plays = sum(len(performance_history.get(v, [])) for v in variants)
        
        if total_plays == 0:
            return np.random.choice(variants)
        
        ucb_values = {}
        for variant in variants:
            history = performance_history.get(variant, [])
            n_plays = len(history)
            
            if n_plays == 0:
                ucb_values[variant] = float('inf')  # Prioritize unplayed variants
            else:
                mean_reward = np.mean(history)
                confidence_bonus = np.sqrt(2 * np.log(total_plays) / n_plays)
                ucb_values[variant] = mean_reward + confidence_bonus
        
        return max(ucb_values, key=ucb_values.get)
    
    def _thompson_sampling(self, variants: List[str], performance_history: Dict[str, List[float]]) -> str:
        """Thompson Sampling selection"""
        samples = {}
        
        for variant in variants:
            history = performance_history.get(variant, [])
            
            if not history:
                # Prior: assume Beta(1,1) for binary rewards
                samples[variant] = np.random.beta(1, 1)
            else:
                # Update Beta distribution based on history
                successes = sum(history)
                failures = len(history) - successes
                samples[variant] = np.random.beta(1 + successes, 1 + failures)
        
        return max(samples, key=samples.get)


class ABTestingFramework(BaseEnhancement):
    """A/B Testing Framework enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="ab_testing_framework",
            version="1.0.0",
            enabled=True,
            priority=10,
            parameters={
                "significance_level": 0.05,
                "power": 0.8,
                "minimum_effect_size": 0.02,
                "default_control_split": 0.5,
                "min_sample_size_per_variant": 1000,
                "min_experiment_duration_days": 7,
                "max_experiment_duration_days": 30,
                "enable_early_stopping": True,
                "early_stopping_check_interval_hours": 24,
                "futility_threshold": 0.1,
                "enable_bandit_optimization": False,
                "bandit_algorithm": "thompson_sampling",
                "exploration_rate": 0.1,
                "primary_metric": "conversion_rate",
                "secondary_metrics": ["revenue", "engagement"],
                "data_retention_days": 90,
                "enable_real_time_monitoring": True,
                "results_storage_path": "./ab_test_results"
            }
        )
    
    def initialize(self) -> bool:
        """Initialize A/B Testing Framework"""
        try:
            # Create configuration
            self.ab_config = ABTestConfig(**self.config.parameters)
            
            # Initialize engines
            self.statistical_engine = StatisticalEngine(self.ab_config)
            self.bandit_engine = MultiArmedBandit(self.ab_config)
            
            # Storage for active experiments
            self.active_experiments: Dict[str, ABTestExperiment] = {}
            
            # Create results directory
            Path(self.ab_config.results_storage_path).mkdir(parents=True, exist_ok=True)
            
            self._logger.info("A/B Testing Framework initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize A/B Testing Framework: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with A/B testing capabilities"""
        enhanced = workflow.copy()
        
        # Add A/B testing configuration
        if 'ab_testing' not in enhanced:
            enhanced['ab_testing'] = {}
        
        enhanced['ab_testing'] = {
            'enabled': True,
            'statistical_config': {
                'significance_level': self.ab_config.significance_level,
                'power': self.ab_config.power,
                'minimum_effect_size': self.ab_config.minimum_effect_size
            },
            'experiment_config': {
                'min_duration_days': self.ab_config.min_experiment_duration_days,
                'max_duration_days': self.ab_config.max_experiment_duration_days,
                'min_sample_size': self.ab_config.min_sample_size_per_variant
            },
            'early_stopping': {
                'enabled': self.ab_config.enable_early_stopping,
                'check_interval_hours': self.ab_config.early_stopping_check_interval_hours,
                'futility_threshold': self.ab_config.futility_threshold
            },
            'bandit_optimization': {
                'enabled': self.ab_config.enable_bandit_optimization,
                'algorithm': self.ab_config.bandit_algorithm,
                'exploration_rate': self.ab_config.exploration_rate
            },
            'metrics': {
                'primary': self.ab_config.primary_metric,
                'secondary': self.ab_config.secondary_metrics
            }
        }
        
        # Enhance MLE-Star stages with A/B testing
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 5: Results Evaluation - Add A/B test design
            if '5_results_evaluation' in stages:
                if 'ab_test_design' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['ab_test_design'] = [
                        'experiment_hypothesis_formulation',
                        'statistical_power_analysis',
                        'sample_size_calculation',
                        'success_metrics_definition'
                    ]
            
            # Stage 7: Deployment Prep - Add A/B test setup
            if '7_deployment_prep' in stages:
                if 'ab_test_setup' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['ab_test_setup'] = [
                        'experiment_configuration',
                        'traffic_allocation_strategy',
                        'monitoring_and_alerting_setup',
                        'statistical_analysis_pipeline'
                    ]
        
        self._logger.debug("Enhanced workflow with A/B testing capabilities")
        return enhanced
    
    def design_experiment(self, name: str, description: str,
                         control_model_id: str, treatment_model_ids: List[str],
                         baseline_conversion_rate: float,
                         minimum_detectable_effect: float = None,
                         duration_days: int = None,
                         traffic_split: Dict[str, float] = None) -> ABTestExperiment:
        """Design an A/B test experiment"""
        try:
            self._logger.info(f"Designing A/B test experiment: {name}")
            
            # Use defaults if not specified
            minimum_detectable_effect = minimum_detectable_effect or self.ab_config.minimum_effect_size
            duration_days = duration_days or self.ab_config.min_experiment_duration_days
            
            # Calculate sample size
            required_sample_size = self.statistical_engine.calculate_sample_size(
                baseline_conversion_rate, minimum_detectable_effect
            )
            
            # Default traffic allocation
            all_variants = [control_model_id] + treatment_model_ids
            if traffic_split is None:
                equal_split = 1.0 / len(all_variants)
                traffic_split = {variant: equal_split for variant in all_variants}
            
            # Generate experiment ID
            experiment_id = self._generate_experiment_id(name)
            
            # Create experiment
            experiment = ABTestExperiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                control_model_id=control_model_id,
                treatment_model_ids=treatment_model_ids,
                traffic_allocation=traffic_split,
                primary_metric=self.ab_config.primary_metric,
                secondary_metrics=self.ab_config.secondary_metrics,
                significance_level=self.ab_config.significance_level,
                minimum_effect_size=minimum_detectable_effect,
                start_time=datetime.now(),
                planned_end_time=datetime.now() + timedelta(days=duration_days),
                enable_early_stopping=self.ab_config.enable_early_stopping,
                enable_bandit=self.ab_config.enable_bandit_optimization
            )
            
            self._logger.info(f"Experiment designed: {experiment_id}")
            self._logger.info(f"Required sample size per variant: {required_sample_size}")
            
            return experiment
            
        except Exception as e:
            self._logger.error(f"Failed to design experiment: {e}")
            raise
    
    def start_experiment(self, experiment: ABTestExperiment) -> bool:
        """Start running an A/B test experiment"""
        try:
            # Validate experiment
            if not self._validate_experiment(experiment):
                return False
            
            # Update status and timing
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            
            # Store active experiment
            self.active_experiments[experiment.experiment_id] = experiment
            
            # Save experiment configuration
            self._save_experiment_config(experiment)
            
            self._logger.info(f"Started experiment: {experiment.experiment_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start experiment: {e}")
            return False
    
    def record_observation(self, experiment_id: str, variant_id: str,
                          metric_values: Dict[str, float]) -> bool:
        """Record an observation for the experiment"""
        try:
            if experiment_id not in self.active_experiments:
                self._logger.error(f"Experiment not found: {experiment_id}")
                return False
            
            experiment = self.active_experiments[experiment_id]
            
            # Update sample size
            if variant_id not in experiment.sample_sizes:
                experiment.sample_sizes[variant_id] = 0
            experiment.sample_sizes[variant_id] += 1
            
            # Store observation (in a real implementation, this would go to a database)
            self._store_observation(experiment_id, variant_id, metric_values)
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to record observation: {e}")
            return False
    
    def analyze_experiment(self, experiment_id: str,
                          metric_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, ExperimentMetrics]:
        """Analyze experiment results"""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            experiment = self.active_experiments[experiment_id]
            results = {}
            
            # Analyze each metric
            all_metrics = [experiment.primary_metric] + experiment.secondary_metrics
            
            for metric_name in all_metrics:
                if metric_name not in metric_data:
                    continue
                
                variant_data = metric_data[metric_name]
                control_data = variant_data.get(experiment.control_model_id)
                
                if control_data is None:
                    continue
                
                # Analyze each treatment variant
                for treatment_id in experiment.treatment_model_ids:
                    treatment_data = variant_data.get(treatment_id)
                    
                    if treatment_data is None:
                        continue
                    
                    # Determine metric type
                    metric_type = "binary" if metric_name == "conversion_rate" else "continuous"
                    
                    # Perform statistical analysis
                    result = self.statistical_engine.analyze_experiment(
                        control_data, treatment_data, 
                        f"{metric_name}_{treatment_id}", metric_type
                    )
                    
                    results[f"{metric_name}_{treatment_id}"] = result
            
            # Update experiment with results
            experiment.results.update(results)
            
            # Check for early stopping
            should_stop, reason = self.statistical_engine.should_stop_early(experiment)
            if should_stop:
                self._logger.info(f"Early stopping recommended for {experiment_id}: {reason}")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Failed to analyze experiment: {e}")
            return {}
    
    def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop running experiment"""
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            experiment = self.active_experiments[experiment_id]
            experiment.status = ExperimentStatus.COMPLETED
            experiment.actual_end_time = datetime.now()
            
            # Save final results
            self._save_experiment_results(experiment)
            
            # Remove from active experiments
            del self.active_experiments[experiment_id]
            
            self._logger.info(f"Stopped experiment {experiment_id}: {reason}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to stop experiment: {e}")
            return False
    
    def get_variant_allocation(self, experiment_id: str) -> Optional[str]:
        """Get variant allocation for bandit optimization"""
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            
            if not experiment.enable_bandit:
                # Static allocation
                variants = [experiment.control_model_id] + experiment.treatment_model_ids
                weights = [experiment.traffic_allocation.get(v, 0) for v in variants]
                return np.random.choice(variants, p=weights)
            
            # Dynamic bandit allocation
            variants = [experiment.control_model_id] + experiment.treatment_model_ids
            
            # Get performance history (placeholder - would come from stored data)
            performance_history = {}
            for variant in variants:
                # In real implementation, this would query historical performance
                performance_history[variant] = []
            
            return self.bandit_engine.select_variant(variants, performance_history)
            
        except Exception as e:
            self._logger.error(f"Failed to get variant allocation: {e}")
            return None
    
    def _validate_experiment(self, experiment: ABTestExperiment) -> bool:
        """Validate experiment configuration"""
        # Check traffic allocation sums to 1
        total_traffic = sum(experiment.traffic_allocation.values())
        if not (0.99 <= total_traffic <= 1.01):  # Allow small floating point errors
            self._logger.error(f"Traffic allocation must sum to 1.0, got {total_traffic}")
            return False
        
        # Check all variants have allocation
        all_variants = [experiment.control_model_id] + experiment.treatment_model_ids
        for variant in all_variants:
            if variant not in experiment.traffic_allocation:
                self._logger.error(f"Missing traffic allocation for variant: {variant}")
                return False
        
        return True
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID"""
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_name}_{timestamp}"
    
    def _save_experiment_config(self, experiment: ABTestExperiment):
        """Save experiment configuration to file"""
        try:
            config_path = Path(self.ab_config.results_storage_path) / f"{experiment.experiment_id}_config.json"
            
            # Convert to serializable format
            config_data = {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'description': experiment.description,
                'control_model_id': experiment.control_model_id,
                'treatment_model_ids': experiment.treatment_model_ids,
                'traffic_allocation': experiment.traffic_allocation,
                'primary_metric': experiment.primary_metric,
                'secondary_metrics': experiment.secondary_metrics,
                'significance_level': experiment.significance_level,
                'minimum_effect_size': experiment.minimum_effect_size,
                'start_time': experiment.start_time.isoformat(),
                'planned_end_time': experiment.planned_end_time.isoformat(),
                'status': experiment.status.value,
                'enable_early_stopping': experiment.enable_early_stopping,
                'enable_bandit': experiment.enable_bandit
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self._logger.error(f"Failed to save experiment config: {e}")
    
    def _save_experiment_results(self, experiment: ABTestExperiment):
        """Save experiment results to file"""
        try:
            results_path = Path(self.ab_config.results_storage_path) / f"{experiment.experiment_id}_results.json"
            
            # Convert results to serializable format
            results_data = {}
            for metric_name, result in experiment.results.items():
                results_data[metric_name] = {
                    'metric_name': result.metric_name,
                    'control_value': result.control_value,
                    'treatment_value': result.treatment_value,
                    'lift': result.lift,
                    'p_value': result.p_value,
                    'confidence_interval': result.confidence_interval,
                    'statistical_power': result.statistical_power,
                    'is_significant': result.is_significant
                }
            
            final_data = {
                'experiment_id': experiment.experiment_id,
                'status': experiment.status.value,
                'actual_end_time': experiment.actual_end_time.isoformat() if experiment.actual_end_time else None,
                'sample_sizes': experiment.sample_sizes,
                'results': results_data
            }
            
            with open(results_path, 'w') as f:
                json.dump(final_data, f, indent=2)
                
        except Exception as e:
            self._logger.error(f"Failed to save experiment results: {e}")
    
    def _store_observation(self, experiment_id: str, variant_id: str, metric_values: Dict[str, float]):
        """Store observation data (placeholder implementation)"""
        # In a real implementation, this would store data in a database
        # For now, we'll just log it
        self._logger.debug(f"Observation recorded for {experiment_id}/{variant_id}: {metric_values}")
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment summary"""
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            
            summary = {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'status': experiment.status.value,
                'start_time': experiment.start_time.isoformat(),
                'runtime_days': (datetime.now() - experiment.start_time).days,
                'sample_sizes': experiment.sample_sizes,
                'total_samples': sum(experiment.sample_sizes.values()),
                'significant_results': [],
                'primary_metric_result': None
            }
            
            # Add results summary
            for metric_name, result in experiment.results.items():
                if result.is_significant:
                    summary['significant_results'].append({
                        'metric': metric_name,
                        'lift': result.lift,
                        'p_value': result.p_value
                    })
                
                if experiment.primary_metric in metric_name:
                    summary['primary_metric_result'] = {
                        'lift': result.lift,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'confidence_interval': result.confidence_interval
                    }
            
            return summary
            
        except Exception as e:
            self._logger.error(f"Failed to get experiment summary: {e}")
            return None