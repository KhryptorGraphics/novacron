# PBA v3 Quick Reference

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow numpy scikit-learn matplotlib tf2onnx
```

### 2. Train Models
```bash
cd /home/kp/novacron/ai_engine
python train_bandwidth_predictor_v3.py --mode both --samples 2000 --epochs 50 --plot
```

### 3. Deploy Models
```bash
sudo mkdir -p /var/lib/dwcp/models
sudo cp models/datacenter/datacenter_bandwidth_predictor.onnx /var/lib/dwcp/models/
sudo cp models/internet/internet_bandwidth_predictor.onnx /var/lib/dwcp/models/
```

### 4. Enable v3
```go
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Prediction:  true,
    V3RolloutPercentage: 10,
})
```

## 📊 Performance Targets

| Mode | Accuracy | Latency | Sequence | LSTM Units |
|------|----------|---------|----------|------------|
| Datacenter | 85%+ | <100ms | 30 steps | 128/64 |
| Internet | 70%+ | <150ms | 60 steps | 256/128 |

## 📁 File Locations

```
ai_engine/
├── bandwidth_predictor_v3.py          # Python ML model
├── train_bandwidth_predictor_v3.py    # Training script
└── test_bandwidth_predictor_v3.py     # Python tests

backend/core/network/dwcp/v3/prediction/
├── pba_v3.go                          # Go integration
├── lstm_predictor_v3.go               # v3 predictor
└── pba_v3_test.go                     # Go tests

docs/
├── PBA-V3-UPGRADE-GUIDE.md            # Complete guide
├── PBA-V3-IMPLEMENTATION-SUMMARY.md   # Implementation details
└── PBA-V3-QUICK-REFERENCE.md          # This file
```

## 🔧 Common Commands

### Train Datacenter Model Only
```bash
python train_bandwidth_predictor_v3.py --mode datacenter --samples 1000 --epochs 30
```

### Train Internet Model Only
```bash
python train_bandwidth_predictor_v3.py --mode internet --samples 2000 --epochs 50
```

### Run Python Tests
```bash
python test_bandwidth_predictor_v3.py
```

### Run Go Tests
```bash
go test -v ./backend/core/network/dwcp/v3/prediction/
```

### Export to ONNX
```python
from bandwidth_predictor_v3 import BandwidthPredictorV3

predictor = BandwidthPredictorV3(mode='datacenter')
predictor.load_model('models/datacenter')
predictor.export_to_onnx('models/datacenter')
```

## 🎯 Usage Examples

### Python Prediction
```python
from bandwidth_predictor_v3 import BandwidthPredictorV3, NetworkMetrics

predictor = BandwidthPredictorV3(mode='internet')
predictor.load_model('models/internet')

prediction, confidence = predictor.predict(historical_data)
print(f"Bandwidth: {prediction:.2f} Mbps (confidence: {confidence:.2%})")
```

### Go Prediction
```go
config := prediction.DefaultPBAv3Config()
pba, _ := prediction.NewPBAv3(config)
defer pba.Close()

pred, _ := pba.PredictBandwidth(ctx)
fmt.Printf("Bandwidth: %.2f Mbps\n", pred.PredictedBandwidthMbps)
```

## 🐛 Troubleshooting

### TensorFlow Not Installed
```bash
pip install tensorflow
```

### ONNX Export Fails
```bash
pip install tf2onnx
```

### Go Tests Fail
```bash
# Models not found - train and deploy first
python train_bandwidth_predictor_v3.py --mode both
sudo cp models/*/*.onnx /var/lib/dwcp/models/
```

## 📈 Monitoring

### Check Metrics
```go
metrics := pba.GetMetrics()
fmt.Printf("Total predictions: %d\n", metrics.TotalPredictions)
fmt.Printf("Datacenter: %d (avg latency: %v)\n",
    metrics.DatacenterPredictions, metrics.DatacenterAvgLatency)
fmt.Printf("Internet: %d (avg latency: %v)\n",
    metrics.InternetPredictions, metrics.InternetAvgLatency)
```

### Feature Flags Status
```go
stats := upgrade.GetRolloutStats()
fmt.Printf("Rollout: %d%%\n", stats["rolloutPercentage"])
fmt.Printf("Prediction enabled: %v\n",
    stats["enabledComponents"].(map[string]bool)["prediction"])
```

## 🔄 Migration Checklist

- [x] Implement v3 models (datacenter + internet)
- [x] Create training infrastructure
- [x] Add comprehensive tests
- [x] Write documentation
- [ ] Install TensorFlow
- [ ] Train models with real data
- [ ] Deploy ONNX models
- [ ] Run integration tests
- [ ] Enable feature flags (10%)
- [ ] Monitor in production
- [ ] Increase rollout (50%, 100%)
- [ ] Deprecate v1

## 📚 References

- **Full Guide**: [PBA-V3-UPGRADE-GUIDE.md](PBA-V3-UPGRADE-GUIDE.md)
- **Implementation**: [PBA-V3-IMPLEMENTATION-SUMMARY.md](PBA-V3-IMPLEMENTATION-SUMMARY.md)
- **DWCP v3**: [DWCP-V3-QUICK-START.md](DWCP-V3-QUICK-START.md)
