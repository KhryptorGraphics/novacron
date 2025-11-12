#!/usr/bin/env python3
"""
Prophet-based Capacity Planning Model for DWCP v3
Forecasts capacity needs and provides planning recommendations
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Prophet-based capacity planning and forecasting"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_dir: str = "/tmp/ml_models"
    ):
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models for different metrics
        self.models = {}
        self.forecasts = {}
        self.capacity_history = []

        # Try to import Prophet
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.prophet_available = True
        except ImportError:
            logger.warning("Prophet not available, using fallback forecasting")
            self.Prophet = None
            self.prophet_available = False

        logger.info("Initialized CapacityPlanner")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'forecast_horizon_days': 30,
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'growth': 'linear',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'confidence_interval': 0.95,
            'capacity_buffer': 0.20,  # 20% buffer
            'alert_threshold': 0.85,  # Alert at 85% capacity
            'metrics': [
                'cpu_usage', 'memory_usage', 'disk_usage',
                'network_bandwidth', 'request_rate',
                'latency', 'throughput'
            ]
        }

    def prepare_data(self, data: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Prepare data for Prophet"""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        # Prophet requires 'ds' (datetime) and 'y' (value) columns
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(data['timestamp']),
            'y': data[metric].values
        })

        # Remove outliers
        prophet_df = self._remove_outliers(prophet_df)

        # Handle missing values
        prophet_df = prophet_df.dropna()

        return prophet_df.sort_values('ds').reset_index(drop=True)

    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
        """Remove outliers using standard deviation method"""
        mean = df['y'].mean()
        std = df['y'].std()

        df_cleaned = df[
            (df['y'] >= mean - n_std * std) &
            (df['y'] <= mean + n_std * std)
        ].copy()

        removed = len(df) - len(df_cleaned)
        if removed > 0:
            logger.info(f"Removed {removed} outliers from {len(df)} samples")

        return df_cleaned

    def train_model(self, data: pd.DataFrame, metric: str) -> Dict:
        """Train Prophet model for specific metric"""
        logger.info(f"Training capacity model for {metric}")

        # Prepare data
        prophet_df = self.prepare_data(data, metric)

        if len(prophet_df) < 100:
            logger.warning(f"Insufficient data for {metric}: {len(prophet_df)} samples")
            return {'status': 'insufficient_data', 'metric': metric}

        if self.prophet_available:
            # Use Prophet
            model = self.Prophet(
                growth=self.config['growth'],
                seasonality_mode=self.config['seasonality_mode'],
                yearly_seasonality=self.config['yearly_seasonality'],
                weekly_seasonality=self.config['weekly_seasonality'],
                daily_seasonality=self.config['daily_seasonality'],
                changepoint_prior_scale=self.config['changepoint_prior_scale'],
                seasonality_prior_scale=self.config['seasonality_prior_scale'],
                interval_width=self.config['confidence_interval']
            )

            # Fit model
            model.fit(prophet_df)
        else:
            # Use fallback simple forecasting
            model = self._create_fallback_model(prophet_df)

        self.models[metric] = model

        training_result = {
            'status': 'success',
            'metric': metric,
            'training_samples': len(prophet_df),
            'date_range': {
                'start': str(prophet_df['ds'].min()),
                'end': str(prophet_df['ds'].max())
            }
        }

        logger.info(f"Training complete for {metric}")
        return training_result

    def _create_fallback_model(self, df: pd.DataFrame) -> Dict:
        """Create simple fallback forecasting model"""
        return {
            'type': 'fallback',
            'trend': np.polyfit(range(len(df)), df['y'].values, 1),
            'seasonal': self._extract_seasonal_pattern(df),
            'last_value': float(df['y'].iloc[-1]),
            'mean': float(df['y'].mean()),
            'std': float(df['y'].std())
        }

    def _extract_seasonal_pattern(self, df: pd.DataFrame, period: int = 24) -> np.ndarray:
        """Extract simple seasonal pattern"""
        if len(df) < period * 2:
            return np.zeros(period)

        # Group by hour and calculate average
        df['hour'] = pd.to_datetime(df['ds']).dt.hour
        seasonal = df.groupby('hour')['y'].mean().values

        # Pad if necessary
        if len(seasonal) < period:
            seasonal = np.pad(seasonal, (0, period - len(seasonal)), mode='edge')

        return seasonal[:period]

    def forecast(
        self,
        metric: str,
        periods: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate forecast for metric"""
        periods = periods or self.config['forecast_horizon_days'] * 24  # Hourly forecast

        if metric not in self.models:
            raise ValueError(f"Model for {metric} not trained")

        logger.info(f"Forecasting {metric} for {periods} periods")

        model = self.models[metric]

        if self.prophet_available and not isinstance(model, dict):
            # Prophet forecast
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)

            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_df.columns = ['timestamp', 'forecast', 'lower_bound', 'upper_bound']

        else:
            # Fallback forecast
            forecast_df = self._fallback_forecast(model, periods)

        self.forecasts[metric] = forecast_df

        logger.info(f"Forecast complete for {metric}")
        return forecast_df

    def _fallback_forecast(self, model: Dict, periods: int) -> pd.DataFrame:
        """Generate fallback forecast"""
        trend_coef = model['trend']
        seasonal = model['seasonal']
        last_value = model['last_value']
        mean = model['mean']
        std = model['std']

        forecasts = []
        timestamps = []

        base_time = datetime.now()

        for i in range(periods):
            # Trend component
            trend = trend_coef[0] * i + trend_coef[1]

            # Seasonal component
            seasonal_idx = i % len(seasonal)
            seasonal_comp = seasonal[seasonal_idx]

            # Combine
            forecast = last_value + trend + (seasonal_comp - mean) * 0.5

            # Add some uncertainty
            lower = forecast - 1.96 * std
            upper = forecast + 1.96 * std

            forecasts.append({
                'timestamp': base_time + timedelta(hours=i),
                'forecast': forecast,
                'lower_bound': lower,
                'upper_bound': upper
            })

        return pd.DataFrame(forecasts)

    def predict_capacity_needs(self) -> Dict:
        """Predict when additional capacity will be needed"""
        logger.info("Predicting capacity needs")

        capacity_predictions = {}

        for metric in self.config['metrics']:
            if metric not in self.forecasts:
                continue

            forecast = self.forecasts[metric]

            # Determine capacity thresholds based on metric type
            if 'usage' in metric:
                threshold = self.config['alert_threshold']  # 85%
                critical = 0.95  # 95%
            elif metric == 'latency':
                threshold = 200  # 200ms
                critical = 500  # 500ms
            elif metric == 'request_rate':
                # Predict based on current capacity
                threshold = float('inf')
                critical = float('inf')
            else:
                threshold = float('inf')
                critical = float('inf')

            # Find when threshold will be exceeded
            if threshold != float('inf'):
                exceeds_threshold = forecast[forecast['forecast'] > threshold]

                if not exceeds_threshold.empty:
                    first_exceed = exceeds_threshold.iloc[0]
                    time_to_threshold = (
                        pd.to_datetime(first_exceed['timestamp']) - datetime.now()
                    )

                    capacity_predictions[metric] = {
                        'current_forecast': float(forecast.iloc[0]['forecast']),
                        'threshold': threshold,
                        'critical_threshold': critical,
                        'exceeds_at': str(first_exceed['timestamp']),
                        'time_to_threshold_hours': time_to_threshold.total_seconds() / 3600,
                        'severity': 'high' if time_to_threshold.days < 7 else 'medium',
                        'recommended_action': self._get_capacity_recommendation(
                            metric,
                            time_to_threshold.days
                        )
                    }
                else:
                    capacity_predictions[metric] = {
                        'current_forecast': float(forecast.iloc[0]['forecast']),
                        'threshold': threshold,
                        'status': 'healthy',
                        'severity': 'low'
                    }

        return capacity_predictions

    def _get_capacity_recommendation(self, metric: str, days_until_threshold: int) -> str:
        """Get capacity recommendation"""
        if days_until_threshold < 7:
            urgency = "URGENT"
        elif days_until_threshold < 14:
            urgency = "HIGH"
        elif days_until_threshold < 30:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        recommendations = {
            'cpu_usage': f"[{urgency}] Scale up compute resources or optimize CPU-intensive operations",
            'memory_usage': f"[{urgency}] Increase memory allocation or investigate memory leaks",
            'disk_usage': f"[{urgency}] Add storage capacity or implement data archival",
            'network_bandwidth': f"[{urgency}] Upgrade network capacity or optimize data transfer",
            'request_rate': f"[{urgency}] Scale out application instances",
            'latency': f"[{urgency}] Optimize performance or scale resources",
            'throughput': f"[{urgency}] Scale infrastructure to handle increased load"
        }

        return recommendations.get(metric, f"[{urgency}] Review and scale {metric} capacity")

    def generate_capacity_plan(
        self,
        forecast_horizon_days: Optional[int] = None
    ) -> Dict:
        """Generate comprehensive capacity plan"""
        horizon = forecast_horizon_days or self.config['forecast_horizon_days']

        logger.info(f"Generating {horizon}-day capacity plan")

        # Get capacity predictions
        capacity_needs = self.predict_capacity_needs()

        # Calculate cost estimates
        cost_estimate = self._estimate_costs(capacity_needs)

        # Generate timeline
        timeline = self._generate_capacity_timeline(capacity_needs, horizon)

        # Risk assessment
        risks = self._assess_capacity_risks(capacity_needs)

        capacity_plan = {
            'generated_at': datetime.now().isoformat(),
            'forecast_horizon_days': horizon,
            'capacity_needs': capacity_needs,
            'cost_estimate': cost_estimate,
            'timeline': timeline,
            'risks': risks,
            'recommendations': self._generate_overall_recommendations(capacity_needs),
            'confidence': self._calculate_forecast_confidence()
        }

        self.capacity_history.append(capacity_plan)

        logger.info("Capacity plan generated")
        return capacity_plan

    def _estimate_costs(self, capacity_needs: Dict) -> Dict:
        """Estimate costs for capacity additions"""
        # Simplified cost estimation
        costs = {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'total': 0.0
        }

        # Cost per unit (example values)
        unit_costs = {
            'cpu_usage': 50,  # $50 per additional CPU
            'memory_usage': 20,  # $20 per GB
            'disk_usage': 0.10,  # $0.10 per GB
            'network_bandwidth': 100  # $100 per Gbps
        }

        for metric, prediction in capacity_needs.items():
            if prediction.get('status') == 'healthy':
                continue

            if metric in unit_costs:
                # Estimate required capacity increase (simplified)
                required_increase = 1.0  # 1 unit

                if 'cpu' in metric:
                    costs['compute'] += unit_costs[metric] * required_increase
                elif 'memory' in metric:
                    costs['compute'] += unit_costs[metric] * required_increase
                elif 'disk' in metric:
                    costs['storage'] += unit_costs[metric] * required_increase * 1000  # GB
                elif 'network' in metric:
                    costs['network'] += unit_costs[metric] * required_increase

        costs['total'] = costs['compute'] + costs['storage'] + costs['network']

        # Add buffer for uncertainty
        costs['total_with_buffer'] = costs['total'] * (1 + self.config['capacity_buffer'])

        return costs

    def _generate_capacity_timeline(
        self,
        capacity_needs: Dict,
        horizon_days: int
    ) -> List[Dict]:
        """Generate capacity addition timeline"""
        timeline = []

        for metric, prediction in capacity_needs.items():
            if prediction.get('status') == 'healthy':
                continue

            if 'exceeds_at' in prediction:
                timeline.append({
                    'date': prediction['exceeds_at'],
                    'metric': metric,
                    'action': prediction['recommended_action'],
                    'severity': prediction['severity'],
                    'days_until': prediction.get('time_to_threshold_hours', 0) / 24
                })

        # Sort by date
        timeline = sorted(timeline, key=lambda x: x['date'])

        return timeline

    def _assess_capacity_risks(self, capacity_needs: Dict) -> List[Dict]:
        """Assess capacity-related risks"""
        risks = []

        high_severity_count = sum(
            1 for p in capacity_needs.values()
            if p.get('severity') == 'high'
        )

        if high_severity_count > 0:
            risks.append({
                'risk': 'Imminent capacity shortage',
                'severity': 'high',
                'affected_metrics': [
                    m for m, p in capacity_needs.items()
                    if p.get('severity') == 'high'
                ],
                'mitigation': 'Immediate capacity scaling required'
            })

        # Check for multiple metrics approaching threshold
        threshold_approaching = sum(
            1 for p in capacity_needs.values()
            if p.get('time_to_threshold_hours', float('inf')) < 7 * 24
        )

        if threshold_approaching >= 2:
            risks.append({
                'risk': 'Multiple capacity constraints',
                'severity': 'medium',
                'impact': 'May require comprehensive infrastructure upgrade',
                'mitigation': 'Plan coordinated capacity additions'
            })

        return risks

    def _generate_overall_recommendations(self, capacity_needs: Dict) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []

        # Priority actions
        high_priority = [
            p['recommended_action']
            for p in capacity_needs.values()
            if p.get('severity') == 'high'
        ]

        if high_priority:
            recommendations.extend(high_priority[:3])

        # General recommendations
        recommendations.extend([
            "Monitor capacity metrics daily",
            "Implement automated alerting for threshold violations",
            "Review and update capacity models monthly",
            "Consider implementing auto-scaling policies"
        ])

        return recommendations[:5]

    def _calculate_forecast_confidence(self) -> float:
        """Calculate overall forecast confidence"""
        # Simplified confidence calculation
        # In production, this would analyze forecast errors

        if not self.models:
            return 0.0

        # Base confidence
        confidence = 0.85

        # Adjust based on data availability
        if len(self.models) < len(self.config['metrics']):
            confidence *= 0.9

        return confidence

    def get_forecast_accuracy(self, actual_data: pd.DataFrame, metric: str) -> Dict:
        """Calculate forecast accuracy against actual data"""
        if metric not in self.forecasts:
            return {'status': 'no_forecast', 'metric': metric}

        forecast = self.forecasts[metric]
        actual = actual_data[['timestamp', metric]].copy()
        actual.columns = ['timestamp', 'actual']

        # Merge forecast and actual
        comparison = pd.merge(
            forecast,
            actual,
            on='timestamp',
            how='inner'
        )

        if len(comparison) == 0:
            return {'status': 'no_overlap', 'metric': metric}

        # Calculate metrics
        mae = np.mean(np.abs(comparison['forecast'] - comparison['actual']))
        rmse = np.sqrt(np.mean((comparison['forecast'] - comparison['actual']) ** 2))
        mape = np.mean(np.abs((comparison['forecast'] - comparison['actual']) / (comparison['actual'] + 1e-10))) * 100

        # Calculate accuracy (within 10% threshold)
        accuracy = np.mean(
            np.abs((comparison['forecast'] - comparison['actual']) / (comparison['actual'] + 1e-10)) < 0.10
        ) * 100

        accuracy_metrics = {
            'metric': metric,
            'samples': len(comparison),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'accuracy_10pct': float(accuracy),
            'mean_actual': float(comparison['actual'].mean()),
            'mean_forecast': float(comparison['forecast'].mean())
        }

        logger.info(f"Forecast accuracy for {metric}: {accuracy:.1f}%")

        return accuracy_metrics

    def save_models(self, filename: str) -> None:
        """Save capacity planning models"""
        model_path = self.model_dir / filename

        # For Prophet models, we need to serialize differently
        models_to_save = {}
        for metric, model in self.models.items():
            if self.prophet_available and hasattr(model, 'to_json'):
                models_to_save[metric] = model.to_json()
            else:
                models_to_save[metric] = model

        with open(model_path, 'w') as f:
            json.dump({
                'models': models_to_save,
                'forecasts': {k: v.to_dict('records') for k, v in self.forecasts.items()},
                'config': self.config,
                'capacity_history': self.capacity_history
            }, f, indent=2)

        logger.info(f"Models saved to {model_path}")

    def load_models(self, filename: str) -> None:
        """Load capacity planning models"""
        model_path = self.model_dir / filename

        with open(model_path, 'r') as f:
            data = json.load(f)

        self.config = data['config']
        self.capacity_history = data['capacity_history']

        # Load forecasts
        self.forecasts = {
            k: pd.DataFrame(v)
            for k, v in data['forecasts'].items()
        }

        # Note: Prophet models need to be retrained, not deserialized
        # This is a known limitation
        logger.warning("Prophet models need to be retrained after loading")

        logger.info(f"Models loaded from {model_path}")


def main():
    """Main capacity planning pipeline"""
    logger.info("Starting Capacity Planner")

    # Initialize planner
    planner = CapacityPlanner()

    logger.info("Capacity planner ready")
    logger.info("Use planner.train_model() to train forecasting models")
    logger.info("Use planner.forecast() to generate forecasts")
    logger.info("Use planner.generate_capacity_plan() for comprehensive planning")

    return planner


if __name__ == "__main__":
    planner = main()
