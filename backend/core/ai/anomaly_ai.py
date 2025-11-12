"""
AI-Driven Anomaly Detection for DWCP v3
Deep autoencoder with attention mechanisms and explainable AI
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    input_dim: int = 64
    encoding_dim: int = 16
    hidden_layers: List[int] = None
    attention_heads: int = 8
    dropout_rate: float = 0.2
    contamination_rate: float = 0.01
    threshold_percentile: float = 99.0
    window_size: int = 100
    stride: int = 10
    use_attention: bool = True
    use_variational: bool = True
    explainability_samples: int = 100

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]

class AttentionMechanism(nn.Module):
    """Multi-head attention for anomaly detection"""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Linear transformations and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )

        output = self.out(context)

        return output, attention_weights

class VariationalEncoder(nn.Module):
    """Variational encoder for robust feature extraction"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

class Decoder(nn.Module):
    """Decoder for reconstruction"""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class DeepAutoencoder(nn.Module):
    """Deep autoencoder with attention for anomaly detection"""

    def __init__(self, config: AnomalyConfig):
        super().__init__()
        self.config = config

        # Encoder
        if config.use_variational:
            self.encoder = VariationalEncoder(
                config.input_dim,
                config.hidden_layers,
                config.encoding_dim
            )
        else:
            encoder_layers = []
            prev_dim = config.input_dim
            for hidden_dim in config.hidden_layers:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate)
                ])
                prev_dim = hidden_dim
            encoder_layers.append(nn.Linear(prev_dim, config.encoding_dim))
            self.encoder = nn.Sequential(*encoder_layers)

        # Attention mechanism
        if config.use_attention:
            self.attention = AttentionMechanism(
                config.encoding_dim,
                config.attention_heads,
                config.dropout_rate
            )

        # Decoder
        self.decoder = Decoder(
            config.encoding_dim,
            config.hidden_layers,
            config.input_dim
        )

        self.use_variational = config.use_variational
        self.use_attention = config.use_attention

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Encode
        if self.use_variational:
            z, mu, logvar = self.encoder(x)
            encoding = z
            aux_outputs = {'mu': mu, 'logvar': logvar}
        else:
            encoding = self.encoder(x)
            aux_outputs = {}

        # Apply attention if enabled
        if self.use_attention and len(encoding.shape) == 3:
            attended, attention_weights = self.attention(encoding)
            aux_outputs['attention_weights'] = attention_weights
            encoding = attended.mean(dim=1) if len(attended.shape) == 3 else attended

        # Decode
        reconstruction = self.decoder(encoding)

        return reconstruction, aux_outputs

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly score for input"""
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction='none')
        return mse.mean(dim=-1)

class RootCauseAnalyzer:
    """Analyze root causes of anomalies using attention and SHAP"""

    def __init__(self, model: DeepAutoencoder, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

    def analyze_anomaly(self, anomaly_sample: torch.Tensor,
                       normal_samples: torch.Tensor) -> Dict[str, Any]:
        """Analyze root cause of anomaly"""
        self.model.eval()

        with torch.no_grad():
            # Get reconstruction and attention weights
            anomaly_recon, anomaly_aux = self.model(anomaly_sample)
            normal_recon, _ = self.model(normal_samples.mean(dim=0, keepdim=True))

            # Calculate feature-wise reconstruction errors
            anomaly_errors = (anomaly_sample - anomaly_recon).abs()
            normal_errors = (normal_samples.mean(dim=0, keepdim=True) - normal_recon).abs()

            # Identify most anomalous features
            feature_importance = (anomaly_errors - normal_errors).squeeze().cpu().numpy()

        # Get attention weights if available
        attention_analysis = {}
        if 'attention_weights' in anomaly_aux:
            attention_weights = anomaly_aux['attention_weights'].squeeze().cpu().numpy()
            attention_analysis = self._analyze_attention(attention_weights)

        # Rank features by importance
        feature_ranking = sorted(
            [(self.feature_names[i], float(feature_importance[i]))
             for i in range(len(self.feature_names))],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            'top_anomalous_features': feature_ranking[:5],
            'attention_analysis': attention_analysis,
            'anomaly_score': float(self.model.get_anomaly_score(anomaly_sample).item()),
            'reconstruction_error': float(F.mse_loss(anomaly_recon, anomaly_sample).item())
        }

    def _analyze_attention(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Analyze attention patterns"""
        # Find which parts of the sequence get most attention
        avg_attention = attention_weights.mean(axis=0) if len(attention_weights.shape) > 2 else attention_weights

        max_attention_idx = np.unravel_index(avg_attention.argmax(), avg_attention.shape)

        return {
            'max_attention_position': max_attention_idx,
            'attention_entropy': -np.sum(avg_attention * np.log(avg_attention + 1e-10)),
            'attention_concentration': float(avg_attention.max())
        }

    def explain_with_shap(self, samples: np.ndarray, n_samples: int = 100) -> Dict[str, Any]:
        """Use SHAP to explain anomaly predictions"""

        def model_predict(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                scores = self.model.get_anomaly_score(x_tensor)
            return scores.cpu().numpy()

        # Create SHAP explainer
        background = samples[np.random.choice(len(samples), min(100, len(samples)), replace=False)]
        explainer = shap.KernelExplainer(model_predict, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(samples[:n_samples])

        # Get feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_ranking = sorted(
            [(self.feature_names[i], float(feature_importance[i]))
             for i in range(len(self.feature_names))],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'feature_importance': importance_ranking,
            'shap_values': shap_values.tolist(),
            'expected_value': float(explainer.expected_value)
        }

class AnomalyDataset(Dataset):
    """Dataset for anomaly detection training"""

    def __init__(self, data: np.ndarray, window_size: int, stride: int):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.windows = self._create_windows()

    def _create_windows(self) -> List[np.ndarray]:
        """Create sliding windows from data"""
        windows = []
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            window = self.data[i:i + self.window_size]
            windows.append(window.flatten())
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])

class RealTimeAnomalyScorer:
    """Real-time anomaly scoring system"""

    def __init__(self, model: DeepAutoencoder, threshold: float):
        self.model = model
        self.threshold = threshold
        self.score_history = []
        self.anomaly_buffer = []
        self.device = next(model.parameters()).device

    def score_sample(self, sample: np.ndarray) -> Dict[str, Any]:
        """Score a single sample in real-time"""
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            score = self.model.get_anomaly_score(sample_tensor).item()

        is_anomaly = score > self.threshold
        severity = self._calculate_severity(score)

        # Update history
        self.score_history.append({
            'timestamp': datetime.now(),
            'score': score,
            'is_anomaly': is_anomaly,
            'severity': severity
        })

        # Buffer anomalies for pattern analysis
        if is_anomaly:
            self.anomaly_buffer.append({
                'sample': sample,
                'score': score,
                'timestamp': datetime.now()
            })

        return {
            'score': score,
            'is_anomaly': is_anomaly,
            'severity': severity,
            'threshold': self.threshold,
            'percentile': self._calculate_percentile(score)
        }

    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity"""
        ratio = score / self.threshold

        if ratio < 1:
            return "normal"
        elif ratio < 1.5:
            return "low"
        elif ratio < 2:
            return "medium"
        elif ratio < 3:
            return "high"
        else:
            return "critical"

    def _calculate_percentile(self, score: float) -> float:
        """Calculate score percentile from history"""
        if len(self.score_history) < 2:
            return 50.0

        scores = [h['score'] for h in self.score_history[-1000:]]
        percentile = (np.array(scores) < score).mean() * 100
        return float(percentile)

    def get_anomaly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        if len(self.anomaly_buffer) < 2:
            return {'pattern': 'insufficient_data'}

        # Time-based clustering
        timestamps = [a['timestamp'] for a in self.anomaly_buffer[-100:]]
        time_diffs = [(timestamps[i+1] - timestamps[i]).seconds
                     for i in range(len(timestamps)-1)]

        avg_interval = np.mean(time_diffs) if time_diffs else 0

        # Determine pattern type
        if avg_interval < 60:  # Less than 1 minute
            pattern_type = "burst"
        elif avg_interval < 3600:  # Less than 1 hour
            pattern_type = "frequent"
        elif avg_interval < 86400:  # Less than 1 day
            pattern_type = "periodic"
        else:
            pattern_type = "sporadic"

        # Score analysis
        recent_scores = [a['score'] for a in self.anomaly_buffer[-20:]]
        score_trend = "increasing" if np.diff(recent_scores).mean() > 0 else "decreasing"

        return {
            'pattern_type': pattern_type,
            'average_interval_seconds': avg_interval,
            'score_trend': score_trend,
            'recent_count': len(self.anomaly_buffer),
            'max_score': max(recent_scores) if recent_scores else 0
        }

class AnomalyDetectionSystem:
    """Complete AI-driven anomaly detection system"""

    def __init__(self, config: AnomalyConfig = None):
        self.config = config or AnomalyConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DeepAutoencoder(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = StandardScaler()
        self.threshold = None
        self.root_cause_analyzer = None
        self.real_time_scorer = None

        self.model_path = Path("/home/kp/novacron/models/anomaly_detection")
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.training_history = []

    def train(self, normal_data: np.ndarray, epochs: int = 100,
             batch_size: int = 32, validation_split: float = 0.2):
        """Train the anomaly detection model"""
        logger.info("Training anomaly detection model...")

        # Normalize data
        normalized_data = self.scaler.fit_transform(normal_data)

        # Create dataset
        dataset = AnomalyDataset(
            normalized_data,
            self.config.window_size,
            self.config.stride
        )

        # Split data
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)

                reconstruction, aux = self.model(batch)

                # Calculate loss
                recon_loss = F.mse_loss(reconstruction, batch)
                loss = recon_loss

                # Add KL divergence for variational autoencoder
                if self.config.use_variational and 'mu' in aux:
                    kl_loss = -0.5 * torch.sum(
                        1 + aux['logvar'] - aux['mu'].pow(2) - aux['logvar'].exp()
                    ) / batch.size(0)
                    loss = recon_loss + 0.1 * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstruction, aux = self.model(batch)
                    loss = F.mse_loss(reconstruction, batch)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")

        # Set threshold
        self._set_threshold(val_loader)

        # Initialize analyzers
        feature_names = [f"feature_{i}" for i in range(self.config.input_dim)]
        self.root_cause_analyzer = RootCauseAnalyzer(self.model, feature_names)
        self.real_time_scorer = RealTimeAnomalyScorer(self.model, self.threshold)

        logger.info("Training complete")

    def _set_threshold(self, data_loader: DataLoader):
        """Set anomaly threshold based on reconstruction errors"""
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                scores = self.model.get_anomaly_score(batch)
                errors.extend(scores.cpu().numpy())

        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, self.config.threshold_percentile)
        logger.info(f"Anomaly threshold set to: {self.threshold:.4f}")

    def detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in data"""
        # Normalize
        normalized = self.scaler.transform(data)

        # Create windows
        dataset = AnomalyDataset(normalized, self.config.window_size, self.config.stride)
        loader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        all_scores = []
        anomalies = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                scores = self.model.get_anomaly_score(batch).cpu().numpy()
                all_scores.extend(scores)

                # Identify anomalies
                anomaly_indices = np.where(scores > self.threshold)[0]
                for idx in anomaly_indices:
                    anomalies.append({
                        'index': i * loader.batch_size + idx,
                        'score': float(scores[idx]),
                        'severity': self._calculate_severity(scores[idx])
                    })

        # Analyze patterns
        pattern_analysis = self._analyze_anomaly_patterns(anomalies)

        return {
            'n_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(all_scores),
            'anomalies': anomalies[:20],  # Top 20 anomalies
            'score_statistics': {
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores)),
                'max': float(np.max(all_scores)),
                'min': float(np.min(all_scores))
            },
            'pattern_analysis': pattern_analysis,
            'threshold': self.threshold
        }

    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity level"""
        if self.threshold is None:
            return "unknown"

        ratio = score / self.threshold
        if ratio < 1:
            return "normal"
        elif ratio < 1.5:
            return "low"
        elif ratio < 2:
            return "medium"
        elif ratio < 3:
            return "high"
        else:
            return "critical"

    def _analyze_anomaly_patterns(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        if not anomalies:
            return {'pattern': 'no_anomalies'}

        indices = [a['index'] for a in anomalies]
        if len(indices) < 2:
            return {'pattern': 'isolated'}

        # Check for clustering
        diffs = np.diff(indices)
        avg_diff = np.mean(diffs)

        if avg_diff < 5:
            pattern = "clustered"
        elif avg_diff < 20:
            pattern = "periodic"
        else:
            pattern = "scattered"

        # Severity distribution
        severities = [a['severity'] for a in anomalies]
        severity_counts = {s: severities.count(s) for s in set(severities)}

        return {
            'pattern': pattern,
            'average_spacing': float(avg_diff),
            'severity_distribution': severity_counts,
            'trend': self._calculate_trend(anomalies)
        }

    def _calculate_trend(self, anomalies: List[Dict]) -> str:
        """Calculate anomaly trend"""
        if len(anomalies) < 3:
            return "insufficient_data"

        scores = [a['score'] for a in anomalies]
        # Simple linear regression
        x = np.arange(len(scores))
        z = np.polyfit(x, scores, 1)
        slope = z[0]

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def explain_anomaly(self, anomaly_data: np.ndarray,
                       normal_data: np.ndarray) -> Dict[str, Any]:
        """Explain why data is anomalous"""
        # Normalize
        anomaly_normalized = self.scaler.transform(anomaly_data)
        normal_normalized = self.scaler.transform(normal_data)

        # Convert to tensors
        anomaly_tensor = torch.FloatTensor(anomaly_normalized).to(self.device)
        normal_tensor = torch.FloatTensor(normal_normalized).to(self.device)

        # Get root cause analysis
        root_cause = self.root_cause_analyzer.analyze_anomaly(
            anomaly_tensor[0:1],
            normal_tensor
        )

        # Get SHAP explanation
        shap_explanation = self.root_cause_analyzer.explain_with_shap(
            anomaly_normalized[:10],
            n_samples=min(10, len(anomaly_normalized))
        )

        return {
            'root_cause_analysis': root_cause,
            'shap_explanation': shap_explanation,
            'recommendation': self._generate_recommendation(root_cause)
        }

    def _generate_recommendation(self, root_cause: Dict[str, Any]) -> str:
        """Generate actionable recommendation based on root cause"""
        top_features = root_cause.get('top_anomalous_features', [])
        if not top_features:
            return "Monitor system for additional anomalies"

        feature_name = top_features[0][0]
        severity = self._calculate_severity(root_cause.get('anomaly_score', 0))

        if severity == "critical":
            return f"CRITICAL: Immediately investigate {feature_name}. Consider emergency response."
        elif severity == "high":
            return f"HIGH PRIORITY: Review {feature_name} configuration and recent changes."
        else:
            return f"Monitor {feature_name} closely for further anomalies."

    def save_model(self):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'threshold': self.threshold,
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            'training_history': self.training_history
        }, self.model_path / "anomaly_model.pt")

        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path / "anomaly_model.pt")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']
        self.training_history = checkpoint['training_history']

        if checkpoint['scaler_mean'] is not None:
            self.scaler.mean_ = checkpoint['scaler_mean']
            self.scaler.scale_ = checkpoint['scaler_scale']

        # Reinitialize analyzers
        feature_names = [f"feature_{i}" for i in range(self.config.input_dim)]
        self.root_cause_analyzer = RootCauseAnalyzer(self.model, feature_names)
        self.real_time_scorer = RealTimeAnomalyScorer(self.model, self.threshold)

        logger.info("Model loaded successfully")

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get anomaly detection metrics"""
        if not self.training_history:
            return {}

        return {
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_val_loss': self.training_history[-1]['val_loss'],
            'threshold': self.threshold,
            'false_positive_estimate': 100 - self.config.threshold_percentile,
            'model_complexity': sum(p.numel() for p in self.model.parameters()),
            'detection_ready': self.threshold is not None
        }

if __name__ == "__main__":
    # Example usage
    config = AnomalyConfig(
        input_dim=64,
        encoding_dim=16,
        use_attention=True,
        use_variational=True
    )

    system = AnomalyDetectionSystem(config)

    # Generate sample normal data
    normal_data = np.random.randn(10000, 64) * 0.5 + 1.0

    # Add some anomalies
    anomaly_data = np.random.randn(100, 64) * 2.0 + 3.0

    # Train model
    system.train(normal_data, epochs=50)

    # Detect anomalies
    test_data = np.vstack([normal_data[:500], anomaly_data[:10]])
    results = system.detect_anomalies(test_data)

    print(f"Detected {results['n_anomalies']} anomalies")
    print(f"Anomaly rate: {results['anomaly_rate']:.2%}")
    print(f"Pattern analysis: {results['pattern_analysis']}")

    # Explain an anomaly
    explanation = system.explain_anomaly(anomaly_data[:1], normal_data[:100])
    print(f"Root cause: {explanation['root_cause_analysis']['top_anomalous_features'][:3]}")
    print(f"Recommendation: {explanation['recommendation']}")