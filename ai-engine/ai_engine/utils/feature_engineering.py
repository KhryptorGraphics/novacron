"""
Feature engineering utilities for the AI Operations Engine.

Provides specialized feature extractors for different ML models including
time-series features, workload characteristics, and resource patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


logger = logging.getLogger(__name__)


class TimeSeriesFeatureExtractor:
    """Extract time-series features for failure prediction and anomaly detection."""
    
    def __init__(self):
        """Initialize time-series feature extractor."""
        self.fc_parameters = ComprehensiveFCParameters()
        self._scaler = StandardScaler()
    
    def extract_features(self, df: pd.DataFrame, column_id: str = 'node_id', 
                        column_sort: str = 'timestamp') -> pd.DataFrame:
        """
        Extract comprehensive time-series features.
        
        Args:
            df: Input DataFrame with time-series data
            column_id: Column identifying different time series
            column_sort: Column for sorting (usually timestamp)
            
        Returns:
            DataFrame with extracted features
        """
        try:
            # Ensure we have the required columns
            if column_id not in df.columns:
                df[column_id] = 'default_id'
            
            if column_sort not in df.columns:
                df[column_sort] = pd.date_range(start='2024-01-01', periods=len(df), freq='1T')
            
            # Extract tsfresh features
            extracted_features = extract_features(
                df, column_id=column_id, column_sort=column_sort,
                default_fc_parameters=self.fc_parameters,
                impute_function=impute
            )
            
            # Add custom time-series features
            custom_features = self._extract_custom_features(df, column_id, column_sort)
            
            # Combine features
            if not custom_features.empty:
                extracted_features = pd.concat([extracted_features, custom_features], axis=1)
            
            # Remove features with NaN or infinite values
            extracted_features = extracted_features.replace([np.inf, -np.inf], np.nan)
            extracted_features = extracted_features.fillna(0)
            
            logger.info(f"Extracted {extracted_features.shape[1]} time-series features")
            
            return extracted_features
            
        except Exception as e:
            logger.warning(f"Time-series feature extraction failed: {str(e)}")
            # Return basic statistical features as fallback
            return self._extract_basic_features(df)
    
    def _extract_custom_features(self, df: pd.DataFrame, column_id: str, 
                               column_sort: str) -> pd.DataFrame:
        """Extract custom time-series features."""
        
        custom_features = {}
        
        # Get numeric columns (excluding ID and timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = {column_id, column_sort}
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in feature_cols:
            if col not in df.columns or df[col].empty:
                continue
                
            values = df[col].values
            
            # Trend features
            if len(values) > 1:
                # Linear trend slope
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                custom_features[f'{col}_trend_slope'] = slope
                custom_features[f'{col}_trend_r2'] = r_value ** 2
                
                # Rate of change features
                rate_of_change = np.diff(values)
                if len(rate_of_change) > 0:
                    custom_features[f'{col}_roc_mean'] = np.mean(rate_of_change)
                    custom_features[f'{col}_roc_std'] = np.std(rate_of_change)
                    custom_features[f'{col}_roc_max'] = np.max(rate_of_change)
                    custom_features[f'{col}_roc_min'] = np.min(rate_of_change)
            
            # Statistical moments
            custom_features[f'{col}_skewness'] = stats.skew(values)
            custom_features[f'{col}_kurtosis'] = stats.kurtosis(values)
            
            # Percentile features
            percentiles = [10, 25, 75, 90, 95, 99]
            for p in percentiles:
                custom_features[f'{col}_percentile_{p}'] = np.percentile(values, p)
            
            # Volatility features
            if len(values) > 1:
                volatility = np.std(rate_of_change) if len(rate_of_change) > 0 else 0
                custom_features[f'{col}_volatility'] = volatility
                
                # Coefficient of variation
                mean_val = np.mean(values)
                if mean_val != 0:
                    custom_features[f'{col}_cv'] = np.std(values) / abs(mean_val)
                else:
                    custom_features[f'{col}_cv'] = 0
        
        return pd.DataFrame([custom_features])
    
    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic statistical features as fallback."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = {}
        
        for col in numeric_cols:
            values = df[col].values
            
            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_min'] = np.min(values)
            features[f'{col}_max'] = np.max(values)
            features[f'{col}_median'] = np.median(values)
            features[f'{col}_range'] = np.max(values) - np.min(values)
        
        return pd.DataFrame([features])


class WorkloadFeatureExtractor:
    """Extract features for workload placement optimization."""
    
    def __init__(self):
        """Initialize workload feature extractor."""
        self.label_encoders = {}
        self._scaler = StandardScaler()
    
    def extract_placement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive placement features.
        
        Args:
            df: DataFrame with workload and node characteristics
            
        Returns:
            DataFrame with placement features
        """
        features_list = []
        
        for _, row in df.iterrows():
            features = {}
            
            # Resource features
            features.update(self._extract_resource_features(row))
            
            # Performance features
            features.update(self._extract_performance_features(row))
            
            # Infrastructure features
            features.update(self._extract_infrastructure_features(row))
            
            # Network features
            features.update(self._extract_network_features(row))
            
            # Operational features
            features.update(self._extract_operational_features(row))
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle categorical features
        features_df = self._encode_categorical_features(features_df)
        
        logger.info(f"Extracted {features_df.shape[1]} placement features")
        
        return features_df
    
    def _extract_resource_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract resource-related features."""
        features = {}
        
        # CPU features
        features['cpu_cores_available'] = row.get('cpu_cores_available', 0)
        features['cpu_frequency'] = row.get('cpu_frequency', 2400)  # MHz
        features['cpu_architecture'] = row.get('cpu_architecture', 'x86_64')
        features['cpu_utilization'] = row.get('cpu_utilization', 0.5)
        features['cpu_load_avg'] = row.get('cpu_load_avg', 1.0)
        
        # Memory features
        features['memory_available'] = row.get('memory_available', 8)  # GB
        features['memory_bandwidth'] = row.get('memory_bandwidth', 25600)  # MB/s
        features['memory_type'] = row.get('memory_type', 'DDR4')
        features['memory_utilization'] = row.get('memory_utilization', 0.5)
        
        # Storage features
        features['storage_available'] = row.get('storage_available', 100)  # GB
        features['storage_type'] = row.get('storage_type', 'SSD')
        features['storage_iops'] = row.get('storage_iops', 10000)
        features['storage_bandwidth'] = row.get('storage_bandwidth', 500)  # MB/s
        features['storage_utilization'] = row.get('storage_utilization', 0.3)
        
        # GPU features (if available)
        features['gpu_available'] = row.get('gpu_available', 0)
        features['gpu_memory'] = row.get('gpu_memory', 0)  # GB
        features['gpu_compute_capability'] = row.get('gpu_compute_capability', 0.0)
        
        # Resource ratios and derived features
        if features['cpu_cores_available'] > 0:
            features['memory_per_core'] = features['memory_available'] / features['cpu_cores_available']
        else:
            features['memory_per_core'] = 0
        
        features['resource_balance_score'] = self._calculate_resource_balance(
            features['cpu_utilization'], features['memory_utilization'], features['storage_utilization']
        )
        
        return features
    
    def _extract_performance_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract performance-related features."""
        features = {}
        
        # Historical performance metrics
        features['historical_cpu_performance'] = row.get('cpu_performance_score', 0.8)
        features['historical_memory_performance'] = row.get('memory_performance_score', 0.8)
        features['historical_disk_performance'] = row.get('disk_performance_score', 0.8)
        features['historical_network_performance'] = row.get('network_performance_score', 0.8)
        
        # Benchmark scores
        features['cpu_benchmark'] = row.get('cpu_benchmark_score', 1000)
        features['memory_benchmark'] = row.get('memory_benchmark_score', 1000)
        features['disk_benchmark'] = row.get('disk_benchmark_score', 1000)
        
        # Workload-specific features
        features['workload_type'] = row.get('workload_type', 'general')
        features['workload_priority'] = row.get('workload_priority', 3)  # 1-5 scale
        features['sla_requirements'] = row.get('sla_requirements', 0.95)  # Uptime SLA
        features['latency_sensitivity'] = row.get('latency_sensitivity', 0.5)  # 0-1 scale
        features['throughput_requirements'] = row.get('throughput_requirements', 100)  # requests/sec
        
        # Performance isolation features
        features['hypervisor_type'] = row.get('hypervisor_type', 'kvm')
        features['virtualization_overhead'] = row.get('virtualization_overhead', 0.05)  # 5%
        features['performance_isolation_score'] = row.get('performance_isolation_score', 0.8)
        
        return features
    
    def _extract_infrastructure_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract infrastructure-related features."""
        features = {}
        
        # Location features
        features['datacenter_location'] = row.get('datacenter_location', 'us-east-1')
        features['rack_position'] = row.get('rack_position', 'rack_01')
        features['availability_zone'] = row.get('availability_zone', 'az-a')
        
        # Environmental features
        features['temperature'] = row.get('temperature', 22.0)  # Celsius
        features['humidity'] = row.get('humidity', 45.0)  # Percentage
        features['power_efficiency'] = row.get('power_efficiency', 0.85)  # PUE
        features['cooling_efficiency'] = row.get('cooling_efficiency', 0.9)
        
        # Reliability features
        features['hardware_age'] = row.get('hardware_age', 2.0)  # Years
        features['failure_history'] = row.get('failure_history', 0.02)  # Failure rate
        features['mtbf_rating'] = row.get('mtbf_rating', 50000)  # Hours
        features['redundancy_level'] = row.get('redundancy_level', 2)  # N+1, N+2, etc.
        
        # Compliance and security features
        features['security_level'] = row.get('security_level', 3)  # 1-5 scale
        features['compliance_certifications'] = row.get('compliance_certifications', 'iso27001')
        features['physical_security_level'] = row.get('physical_security_level', 4)  # 1-5 scale
        
        return features
    
    def _extract_network_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract network-related features."""
        features = {}
        
        # Network capacity and performance
        features['network_bandwidth'] = row.get('network_bandwidth', 1000)  # Mbps
        features['network_latency'] = row.get('network_latency', 5.0)  # ms
        features['network_utilization'] = row.get('network_utilization', 0.3)
        features['packet_loss_rate'] = row.get('packet_loss_rate', 0.001)  # 0.1%
        features['jitter_variance'] = row.get('jitter_variance', 1.0)  # ms
        
        # Network topology features
        features['hop_count'] = row.get('hop_count', 3)
        features['network_distance'] = row.get('network_distance', 100)  # km
        features['bgp_path_length'] = row.get('bgp_path_length', 5)
        features['routing_efficiency'] = row.get('routing_efficiency', 0.9)
        
        # Network services
        features['cdn_proximity'] = row.get('cdn_proximity', 0.8)  # 0-1 scale
        features['load_balancer_affinity'] = row.get('load_balancer_affinity', 0.5)
        features['network_provider'] = row.get('network_provider', 'tier1')
        
        return features
    
    def _extract_operational_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract operational features."""
        features = {}
        
        # Current operational state
        features['current_load'] = row.get('current_load', 0.5)  # 0-1 scale
        features['capacity_utilization'] = row.get('capacity_utilization', 0.6)
        features['operational_cost'] = row.get('operational_cost', 100)  # $/hour
        features['licensing_cost'] = row.get('licensing_cost', 50)  # $/hour
        
        # Operational efficiency
        features['automation_level'] = row.get('automation_level', 3)  # 1-5 scale
        features['monitoring_coverage'] = row.get('monitoring_coverage', 0.9)
        features['incident_response_time'] = row.get('incident_response_time', 15)  # minutes
        
        # Migration and deployment features
        features['migration_cost'] = row.get('migration_cost', 50)  # Cost units
        features['deployment_time'] = row.get('deployment_time', 300)  # seconds
        features['startup_time'] = row.get('startup_time', 60)  # seconds
        
        # Service dependencies
        features['service_dependencies'] = row.get('service_dependencies', 2)  # Count
        features['integration_complexity'] = row.get('integration_complexity', 0.3)  # 0-1 scale
        
        return features
    
    def _calculate_resource_balance(self, cpu_util: float, memory_util: float, 
                                  storage_util: float) -> float:
        """Calculate resource balance score."""
        utilizations = [cpu_util, memory_util, storage_util]
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        
        # Higher balance = lower standard deviation
        balance_score = max(0, 1.0 - (std_util / (mean_util + 0.01)))
        
        return balance_score
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = self.label_encoders[col].classes_
                df_encoded[col] = df[col].astype(str).apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in unique_values else -1
                )
        
        return df_encoded


class AnomalyFeatureExtractor:
    """Extract features for anomaly detection."""
    
    def __init__(self):
        """Initialize anomaly feature extractor."""
        self._scaler = StandardScaler()
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract anomaly detection features.
        
        Args:
            df: Input DataFrame with system metrics
            
        Returns:
            DataFrame with anomaly detection features
        """
        features_list = []
        
        for _, row in df.iterrows():
            features = {}
            
            # System resource features
            features.update(self._extract_system_features(row))
            
            # Application performance features
            features.update(self._extract_application_features(row))
            
            # Network behavior features
            features.update(self._extract_network_behavior_features(row))
            
            # Security features
            features.update(self._extract_security_features(row))
            
            # Temporal features
            features.update(self._extract_temporal_features(row))
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Add derived features
        features_df = self._add_derived_features(features_df)
        
        logger.info(f"Extracted {features_df.shape[1]} anomaly detection features")
        
        return features_df
    
    def _extract_system_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract system resource features."""
        features = {}
        
        # CPU features
        features['cpu_utilization'] = row.get('cpu_utilization', 0.5)
        features['cpu_load_1min'] = row.get('cpu_load_1min', 1.0)
        features['cpu_load_5min'] = row.get('cpu_load_5min', 1.0)
        features['cpu_load_15min'] = row.get('cpu_load_15min', 1.0)
        features['cpu_steal_time'] = row.get('cpu_steal_time', 0.01)
        features['cpu_wait_time'] = row.get('cpu_wait_time', 0.05)
        
        # Memory features
        features['memory_utilization'] = row.get('memory_utilization', 0.5)
        features['memory_available'] = row.get('memory_available', 4.0)  # GB
        features['memory_cached'] = row.get('memory_cached', 1.0)  # GB
        features['memory_buffers'] = row.get('memory_buffers', 0.5)  # GB
        features['swap_utilization'] = row.get('swap_utilization', 0.0)
        features['page_faults'] = row.get('page_faults', 100)  # per second
        
        # Disk features
        features['disk_utilization'] = row.get('disk_utilization', 0.3)
        features['disk_read_iops'] = row.get('disk_read_iops', 50)
        features['disk_write_iops'] = row.get('disk_write_iops', 50)
        features['disk_read_bandwidth'] = row.get('disk_read_bandwidth', 10)  # MB/s
        features['disk_write_bandwidth'] = row.get('disk_write_bandwidth', 10)  # MB/s
        features['disk_queue_depth'] = row.get('disk_queue_depth', 2)
        features['disk_await'] = row.get('disk_await', 10)  # ms
        
        return features
    
    def _extract_application_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract application performance features."""
        features = {}
        
        # Process features
        features['process_count'] = row.get('process_count', 100)
        features['thread_count'] = row.get('thread_count', 500)
        features['zombie_processes'] = row.get('zombie_processes', 0)
        features['context_switches'] = row.get('context_switches', 1000)  # per second
        
        # Application metrics
        features['response_time'] = row.get('response_time', 100)  # ms
        features['throughput'] = row.get('throughput', 100)  # requests/sec
        features['error_rate'] = row.get('error_rate', 0.01)  # 1%
        features['active_connections'] = row.get('active_connections', 50)
        features['connection_pool_utilization'] = row.get('connection_pool_utilization', 0.5)
        
        # Database features (if applicable)
        features['db_query_time'] = row.get('db_query_time', 50)  # ms
        features['db_connections'] = row.get('db_connections', 20)
        features['db_locks'] = row.get('db_locks', 5)
        features['db_deadlocks'] = row.get('db_deadlocks', 0)
        
        return features
    
    def _extract_network_behavior_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract network behavior features."""
        features = {}
        
        # Network traffic features
        features['network_rx_bytes'] = row.get('network_rx_bytes', 1000000)  # bytes/sec
        features['network_tx_bytes'] = row.get('network_tx_bytes', 1000000)  # bytes/sec
        features['network_rx_packets'] = row.get('network_rx_packets', 1000)  # packets/sec
        features['network_tx_packets'] = row.get('network_tx_packets', 1000)  # packets/sec
        features['network_rx_errors'] = row.get('network_rx_errors', 0)
        features['network_tx_errors'] = row.get('network_tx_errors', 0)
        features['network_rx_dropped'] = row.get('network_rx_dropped', 0)
        features['network_tx_dropped'] = row.get('network_tx_dropped', 0)
        
        # Connection features
        features['tcp_established_connections'] = row.get('tcp_established_connections', 50)
        features['tcp_time_wait_connections'] = row.get('tcp_time_wait_connections', 10)
        features['tcp_listen_connections'] = row.get('tcp_listen_connections', 5)
        features['udp_connections'] = row.get('udp_connections', 10)
        
        # Network quality features
        features['network_latency'] = row.get('network_latency', 5.0)  # ms
        features['packet_loss'] = row.get('packet_loss', 0.001)  # 0.1%
        features['jitter'] = row.get('jitter', 1.0)  # ms
        features['bandwidth_utilization'] = row.get('bandwidth_utilization', 0.3)
        
        return features
    
    def _extract_security_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract security-related features."""
        features = {}
        
        # Authentication features
        features['failed_login_attempts'] = row.get('failed_login_attempts', 0)
        features['successful_logins'] = row.get('successful_logins', 5)
        features['privileged_access_attempts'] = row.get('privileged_access_attempts', 1)
        features['unusual_user_activity'] = row.get('unusual_user_activity', 0)
        
        # System security features
        features['firewall_blocks'] = row.get('firewall_blocks', 10)
        features['suspicious_file_access'] = row.get('suspicious_file_access', 0)
        features['privilege_escalations'] = row.get('privilege_escalations', 0)
        features['system_file_modifications'] = row.get('system_file_modifications', 0)
        
        # Network security features
        features['suspicious_network_connections'] = row.get('suspicious_network_connections', 0)
        features['port_scan_attempts'] = row.get('port_scan_attempts', 0)
        features['ddos_indicators'] = row.get('ddos_indicators', 0)
        features['malware_detections'] = row.get('malware_detections', 0)
        
        return features
    
    def _extract_temporal_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract temporal features."""
        features = {}
        
        # Time-based features
        timestamp = row.get('timestamp', pd.Timestamp.now())
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        features['hour_of_day'] = timestamp.hour if hasattr(timestamp, 'hour') else 12
        features['day_of_week'] = timestamp.dayofweek if hasattr(timestamp, 'dayofweek') else 1
        features['is_weekend'] = features['day_of_week'] >= 5
        features['is_business_hours'] = 9 <= features['hour_of_day'] <= 17
        
        # System uptime
        features['system_uptime'] = row.get('system_uptime', 86400)  # seconds
        features['process_runtime'] = row.get('process_runtime', 3600)  # seconds
        
        return features
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features based on existing features."""
        df_derived = df.copy()
        
        # Resource pressure indicators
        if 'cpu_utilization' in df.columns and 'memory_utilization' in df.columns:
            df_derived['resource_pressure'] = df['cpu_utilization'] + df['memory_utilization']
        
        # I/O pressure
        if 'disk_read_iops' in df.columns and 'disk_write_iops' in df.columns:
            df_derived['io_pressure'] = df['disk_read_iops'] + df['disk_write_iops']
        
        # Network activity ratio
        if 'network_rx_bytes' in df.columns and 'network_tx_bytes' in df.columns:
            total_bytes = df['network_rx_bytes'] + df['network_tx_bytes']
            df_derived['network_rx_ratio'] = df['network_rx_bytes'] / (total_bytes + 1)
            df_derived['network_tx_ratio'] = df['network_tx_bytes'] / (total_bytes + 1)
        
        # Error rates
        if 'network_rx_errors' in df.columns and 'network_rx_packets' in df.columns:
            df_derived['network_error_rate'] = df['network_rx_errors'] / (df['network_rx_packets'] + 1)
        
        # Load balance
        if all(col in df.columns for col in ['cpu_load_1min', 'cpu_load_5min', 'cpu_load_15min']):
            df_derived['load_trend'] = df['cpu_load_1min'] - df['cpu_load_15min']
        
        return df_derived


class ResourceFeatureExtractor:
    """Extract features for resource optimization."""
    
    def __init__(self):
        """Initialize resource feature extractor."""
        self._scaler = StandardScaler()
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract resource optimization features.
        
        Args:
            df: Input DataFrame with resource states and workload data
            
        Returns:
            DataFrame with resource optimization features
        """
        features_list = []
        
        for _, row in df.iterrows():
            features = {}
            
            # Current resource state
            features.update(self._extract_current_state_features(row))
            
            # Historical patterns
            features.update(self._extract_historical_features(row))
            
            # Workload characteristics
            features.update(self._extract_workload_features(row))
            
            # Cost features
            features.update(self._extract_cost_features(row))
            
            # Performance requirements
            features.update(self._extract_performance_requirements(row))
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Add optimization-specific derived features
        features_df = self._add_optimization_features(features_df)
        
        logger.info(f"Extracted {features_df.shape[1]} resource optimization features")
        
        return features_df
    
    def _extract_current_state_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract current resource state features."""
        features = {}
        
        # Resource allocations
        features['cpu_cores'] = row.get('cpu_cores', 2.0)
        features['memory_gb'] = row.get('memory_gb', 4.0)
        features['storage_gb'] = row.get('storage_gb', 100.0)
        features['network_bandwidth'] = row.get('network_bandwidth', 1000)  # Mbps
        
        # Current utilization
        features['cpu_utilization'] = row.get('cpu_utilization', 0.5)
        features['memory_utilization'] = row.get('memory_utilization', 0.5)
        features['storage_utilization'] = row.get('storage_utilization', 0.3)
        features['network_utilization'] = row.get('network_utilization', 0.2)
        
        # Resource efficiency
        features['cpu_efficiency'] = row.get('cpu_efficiency', 0.8)
        features['memory_efficiency'] = row.get('memory_efficiency', 0.8)
        features['storage_efficiency'] = row.get('storage_efficiency', 0.9)
        
        return features
    
    def _extract_historical_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract historical usage patterns."""
        features = {}
        
        # Historical averages
        features['cpu_utilization_avg_7d'] = row.get('cpu_utilization_avg_7d', 0.5)
        features['memory_utilization_avg_7d'] = row.get('memory_utilization_avg_7d', 0.5)
        features['storage_utilization_avg_7d'] = row.get('storage_utilization_avg_7d', 0.3)
        
        # Peak usage
        features['cpu_utilization_max_7d'] = row.get('cpu_utilization_max_7d', 0.8)
        features['memory_utilization_max_7d'] = row.get('memory_utilization_max_7d', 0.8)
        features['storage_utilization_max_7d'] = row.get('storage_utilization_max_7d', 0.5)
        
        # Usage variability
        features['cpu_utilization_std_7d'] = row.get('cpu_utilization_std_7d', 0.1)
        features['memory_utilization_std_7d'] = row.get('memory_utilization_std_7d', 0.1)
        features['storage_utilization_std_7d'] = row.get('storage_utilization_std_7d', 0.05)
        
        # Growth trends
        features['cpu_usage_trend'] = row.get('cpu_usage_trend', 0.0)  # % change per week
        features['memory_usage_trend'] = row.get('memory_usage_trend', 0.0)
        features['storage_usage_trend'] = row.get('storage_usage_trend', 0.05)
        
        return features
    
    def _extract_workload_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract workload characteristics."""
        features = {}
        
        # Workload type and characteristics
        features['workload_type'] = row.get('workload_type', 'web_server')
        features['workload_priority'] = row.get('workload_priority', 3)  # 1-5 scale
        features['workload_criticality'] = row.get('workload_criticality', 0.5)  # 0-1 scale
        
        # Performance characteristics
        features['cpu_intensive'] = row.get('cpu_intensive', 0.5)  # 0-1 scale
        features['memory_intensive'] = row.get('memory_intensive', 0.5)
        features['io_intensive'] = row.get('io_intensive', 0.3)
        features['network_intensive'] = row.get('network_intensive', 0.3)
        
        # Scaling characteristics
        features['horizontal_scalable'] = row.get('horizontal_scalable', True)
        features['vertical_scalable'] = row.get('vertical_scalable', True)
        features['auto_scaling_enabled'] = row.get('auto_scaling_enabled', False)
        features['scaling_responsiveness'] = row.get('scaling_responsiveness', 0.5)  # 0-1 scale
        
        return features
    
    def _extract_cost_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract cost-related features."""
        features = {}
        
        # Direct costs
        features['cpu_cost_per_core'] = row.get('cpu_cost_per_core', 50)  # $/month
        features['memory_cost_per_gb'] = row.get('memory_cost_per_gb', 10)  # $/month
        features['storage_cost_per_gb'] = row.get('storage_cost_per_gb', 0.5)  # $/month
        features['network_cost_per_gbps'] = row.get('network_cost_per_gbps', 100)  # $/month
        
        # Operational costs
        features['licensing_cost'] = row.get('licensing_cost', 100)  # $/month
        features['management_cost'] = row.get('management_cost', 50)  # $/month
        features['support_cost'] = row.get('support_cost', 25)  # $/month
        
        # Cost efficiency
        features['cost_per_transaction'] = row.get('cost_per_transaction', 0.01)  # $
        features['cost_efficiency_score'] = row.get('cost_efficiency_score', 0.7)  # 0-1 scale
        
        return features
    
    def _extract_performance_requirements(self, row: pd.Series) -> Dict[str, Any]:
        """Extract performance requirements."""
        features = {}
        
        # SLA requirements
        features['sla_uptime'] = row.get('sla_uptime', 0.99)  # 99%
        features['sla_response_time'] = row.get('sla_response_time', 200)  # ms
        features['sla_throughput'] = row.get('sla_throughput', 1000)  # requests/sec
        
        # Performance targets
        features['target_cpu_utilization'] = row.get('target_cpu_utilization', 0.7)
        features['target_memory_utilization'] = row.get('target_memory_utilization', 0.8)
        features['target_response_time'] = row.get('target_response_time', 100)  # ms
        
        # Business impact
        features['revenue_impact'] = row.get('revenue_impact', 1000)  # $/hour downtime
        features['user_impact'] = row.get('user_impact', 100)  # affected users
        features['business_criticality'] = row.get('business_criticality', 0.5)  # 0-1 scale
        
        return features
    
    def _add_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optimization-specific derived features."""
        df_derived = df.copy()
        
        # Resource over/under provisioning
        if 'cpu_utilization' in df.columns:
            df_derived['cpu_overprovisioned'] = (df['cpu_utilization'] < 0.3).astype(int)
            df_derived['cpu_underprovisioned'] = (df['cpu_utilization'] > 0.8).astype(int)
        
        if 'memory_utilization' in df.columns:
            df_derived['memory_overprovisioned'] = (df['memory_utilization'] < 0.4).astype(int)
            df_derived['memory_underprovisioned'] = (df['memory_utilization'] > 0.9).astype(int)
        
        # Resource balance
        if all(col in df.columns for col in ['cpu_utilization', 'memory_utilization']):
            df_derived['resource_imbalance'] = abs(df['cpu_utilization'] - df['memory_utilization'])
        
        # Efficiency scores
        if all(col in df.columns for col in ['cpu_utilization', 'cpu_cores']):
            df_derived['cpu_efficiency_ratio'] = df['cpu_utilization'] * df['cpu_cores']
        
        if all(col in df.columns for col in ['memory_utilization', 'memory_gb']):
            df_derived['memory_efficiency_ratio'] = df['memory_utilization'] * df['memory_gb']
        
        # Cost efficiency
        if all(col in df.columns for col in ['cpu_cost_per_core', 'cpu_cores', 'cpu_utilization']):
            total_cpu_cost = df['cpu_cost_per_core'] * df['cpu_cores']
            df_derived['cpu_cost_efficiency'] = df['cpu_utilization'] / (total_cpu_cost + 1)
        
        return df_derived