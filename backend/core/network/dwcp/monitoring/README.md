# DWCP Monitoring - ML-Based Anomaly Detection

Intelligent anomaly detection system for proactive DWCP monitoring using machine learning.

## Quick Start

### 1. Train Models

```bash
cd training
pip install -r requirements.txt

# Generate synthetic training data or use real data
python train_isolation_forest.py --synthetic --output ../models
python train_lstm_autoencoder.py --synthetic --output ../models
```

### 2. Configure

Edit `/home/kp/novacron/configs/monitoring/anomaly-detection.yaml`:

```yaml
enabled: true
check_interval: 10s

alert:
  slack_enabled: true
  slack_webhook_url: "YOUR_WEBHOOK_URL"
  slack_channel: "#dwcp-alerts"
```

### 3. Run

```go
package main

import (
    "context"
    "github.com/yourusername/novacron/backend/core/network/dwcp/monitoring"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()

    // Load configuration
    config, _ := monitoring.LoadConfigFromFile("configs/monitoring/anomaly-detection.yaml")

    // Create detector
    detector, _ := monitoring.NewAnomalyDetector(config.Detector, logger)

    // Create alert manager
    alertManager := monitoring.NewAlertManager(config.Alert, logger)

    // Create pipeline
    pipeline := monitoring.NewMonitoringPipeline(
        detector,
        alertManager,
        config.CheckInterval,
        logger,
    )

    // Train with historical normal data
    normalData := loadNormalData()
    pipeline.TrainDetector(context.Background(), normalData)

    // Start monitoring
    pipeline.Start()
    defer pipeline.Stop()

    // Feed metrics
    go feedMetrics(pipeline)

    select {}
}

func feedMetrics(pipeline *monitoring.MonitoringPipeline) {
    ticker := time.NewTicker(1 * time.Second)
    for range ticker.C {
        metrics := &monitoring.MetricVector{
            Timestamp:   time.Now(),
            Bandwidth:   getCurrentBandwidth(),
            Latency:     getCurrentLatency(),
            PacketLoss:  getCurrentPacketLoss(),
            Jitter:      getCurrentJitter(),
            CPUUsage:    getCurrentCPUUsage(),
            MemoryUsage: getCurrentMemoryUsage(),
            ErrorRate:   getCurrentErrorRate(),
        }

        pipeline.ProcessMetrics(metrics)
    }
}
```

## Components

### Detectors

- **anomaly_detector.go** - Main coordinator
- **isolation_forest.go** - Isolation Forest implementation
- **lstm_autoencoder.go** - LSTM Autoencoder for time series
- **zscore_detector.go** - Statistical Z-score detector
- **seasonal_esd.go** - Seasonal ESD detector
- **ensemble_detector.go** - Ensemble voting aggregator

### Pipeline

- **monitoring_pipeline.go** - Real-time monitoring pipeline
- **alert_manager.go** - Multi-channel alert dispatch

### Configuration

- **config.go** - Configuration management

## Architecture

```
MetricVector → MetricsBuffer → AnomalyDetector → AlertManager
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              IsolationForest   LSTMAutoencoder   ZScore
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                              EnsembleDetector
                                     │
                                  Anomaly
```

## Detection Models

### Isolation Forest
- **Type**: Unsupervised outlier detection
- **Best for**: Multivariate anomalies
- **Training**: Requires normal data only
- **Inference**: Fast (<1ms)

### LSTM Autoencoder
- **Type**: Deep learning reconstruction
- **Best for**: Time-series pattern deviations
- **Training**: Requires sequential normal data
- **Inference**: Moderate (1-10ms)

### Z-Score
- **Type**: Statistical outlier detection
- **Best for**: Simple statistical anomalies
- **Training**: Online learning (rolling window)
- **Inference**: Very fast (<0.1ms)

### Seasonal ESD
- **Type**: Time-series decomposition
- **Best for**: Seasonal patterns
- **Training**: Requires seasonal data
- **Inference**: Fast (1ms)

## Metrics

7 metrics monitored per vector:

1. **Bandwidth** (Mbps)
2. **Latency** (ms)
3. **Packet Loss** (%)
4. **Jitter** (ms)
5. **CPU Usage** (%)
6. **Memory Usage** (%)
7. **Error Rate** (%)

## Alerts

### Severity Levels

- **Info**: Confidence <0.7, log only
- **Warning**: Confidence 0.7-0.9, Slack alert
- **Critical**: Confidence >0.9, PagerDuty + Slack

### Channels

- Slack (webhook)
- PagerDuty (events API)
- Generic webhook (JSON POST)
- Email (SMTP)

### Throttling

- Per metric+severity combination
- Configurable duration (default: 5 minutes)
- Prevents alert fatigue

## Testing

```bash
# Run tests
go test ./...

# Run benchmarks
go test -bench=. ./...

# Run specific test
go test -run TestIsolationForest_Detection

# With coverage
go test -cover ./...
```

## Grafana Dashboard

Import dashboard from:
`/home/kp/novacron/configs/grafana/dwcp-anomaly-detection-dashboard.json`

Includes:
- Real-time anomaly timeline
- Anomaly count by metric
- Severity distribution
- Confidence heatmap
- Model comparison
- Detection latency

## Performance

### Targets
- Detection latency: <30s
- False positive rate: <5%
- True positive rate: >90%
- Throughput: 1000 metrics/s

### Optimization
- Use ONNX for model inference
- Parallel detector execution
- Efficient metric buffering
- Smart alert throttling

## Troubleshooting

### High False Positives
- Increase ensemble threshold
- Retrain with more diverse data
- Adjust individual detector thresholds

### Missing Anomalies
- Decrease ensemble threshold
- Check training data coverage
- Verify metrics are accurate

### High Latency
- Reduce check interval
- Optimize model inference
- Scale horizontally

## Documentation

Full documentation: `/home/kp/novacron/docs/monitoring/ANOMALY_DETECTION.md`

## File Structure

```
monitoring/
├── anomaly_detector.go         # Main detector
├── isolation_forest.go         # Isolation Forest
├── lstm_autoencoder.go         # LSTM Autoencoder
├── zscore_detector.go          # Z-Score detector
├── seasonal_esd.go             # Seasonal ESD
├── ensemble_detector.go        # Ensemble voting
├── monitoring_pipeline.go      # Real-time pipeline
├── alert_manager.go            # Alert dispatch
├── config.go                   # Configuration
├── anomaly_test.go            # Tests
├── models/                     # Trained models
│   ├── isolation_forest.onnx
│   ├── lstm_autoencoder.onnx
│   └── *.json                  # Metadata
├── training/                   # Training scripts
│   ├── requirements.txt
│   ├── train_isolation_forest.py
│   ├── train_lstm_autoencoder.py
│   └── README.md
├── testdata/                   # Test data
└── README.md
```

## License

Part of the NovaCron project.
