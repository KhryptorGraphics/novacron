# ML/AI Integration Recommendations for NovaCron

**Document Type:** Technical Implementation Guide
**Target Audience:** ML Engineers, DevOps, Backend Developers
**Priority:** High (Phase 2 completion blocker)

---

## 1. Critical Integration Gaps

### 1.1 Feature Engineering Consistency

**Problem:**
Python and Go implement different feature extraction logic, causing training-serving skew.

**Impact:**
- Model accuracy degrades by 10-30% in production
- Predictions are unreliable
- Debugging is extremely difficult

**Solution: Implement Feature Store**

```yaml
Option 1: Feast (Recommended)
  Pros:
    - Lightweight (Python + Redis/PostgreSQL)
    - Online + Offline stores
    - Point-in-time correct features
    - Go SDK available

  Cons:
    - Additional infrastructure
    - Learning curve

  Implementation:
    1. Install Feast:
       pip install feast

    2. Define feature repository:
       feast init novacron_features

    3. Define features:
       # features.py
       from feast import Entity, Feature, FeatureView, ValueType
       from feast.data_source import FileSource

       vm_entity = Entity(name="vm_id", value_type=ValueType.STRING)

       vm_metrics = FeatureView(
           name="vm_metrics",
           entities=["vm_id"],
           features=[
               Feature("cpu_usage", ValueType.DOUBLE),
               Feature("memory_usage", ValueType.DOUBLE),
               Feature("disk_io", ValueType.DOUBLE),
               Feature("network_io", ValueType.DOUBLE),
               Feature("latency", ValueType.DOUBLE),
               Feature("error_rate", ValueType.DOUBLE)
           ],
           ttl=timedelta(hours=24),
           input=FileSource(path="data/vm_metrics.parquet")
       )

    4. Apply to online store:
       feast apply

    5. Use in Python training:
       from feast import FeatureStore
       store = FeatureStore(repo_path=".")
       features = store.get_online_features(
           features=["vm_metrics:cpu_usage", "vm_metrics:memory_usage"],
           entity_rows=[{"vm_id": "vm-123"}]
       ).to_dict()

    6. Use in Go serving:
       import "github.com/feast-dev/feast/sdk/go"
       client := feast.NewClient("localhost:6566")
       features := client.GetOnlineFeatures(...)

Option 2: Custom Feature Store (Lightweight)
  Implementation:
    1. Create shared feature library:
       # features/extractor.py
       def extract_features(metrics: dict) -> dict:
           return {
               "cpu_normalized": metrics["cpu"] / 100.0,
               "memory_normalized": metrics["memory"] / 100.0,
               # ... all features
           }

    2. Expose as gRPC service:
       # features/service.py
       import grpc
       from concurrent import futures
       import features_pb2
       import features_pb2_grpc

       class FeatureExtractor(features_pb2_grpc.FeatureExtractorServicer):
           def ExtractFeatures(self, request, context):
               features = extract_features(request.metrics)
               return features_pb2.FeaturesResponse(features=features)

    3. Call from Go:
       conn, _ := grpc.Dial("localhost:50051")
       client := pb.NewFeatureExtractorClient(conn)
       response, _ := client.ExtractFeatures(context.Background(), request)
```

**Recommendation:** Start with Option 2 (custom gRPC service) for Phase 2, migrate to Feast for production.

---

## 2. Model Registry Implementation

**Problem:**
No centralized model versioning, rollback, or A/B testing capability.

**Solution: MLflow Model Registry**

```bash
# 1. Install MLflow
pip install mlflow

# 2. Start MLflow server
mlflow server \
  --backend-store-uri postgresql://mlflow:password@localhost/mlflow \
  --default-artifact-root s3://mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000

# 3. Register model in training
import mlflow
import mlflow.keras

with mlflow.start_run():
    # Train model
    model = train_lstm(X_train, y_train)

    # Log parameters
    mlflow.log_param("hidden_size", 128)
    mlflow.log_param("sequence_length", 10)

    # Log metrics
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("mae", 0.03)

    # Register model
    mlflow.keras.log_model(
        model,
        "bandwidth_predictor",
        registered_model_name="BandwidthPredictorLSTM"
    )

# 4. Promote to production
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="BandwidthPredictorLSTM",
    version=3,
    stage="Production"
)

# 5. Load in Go (via REST API)
# GET http://mlflow-server:5000/model-versions/get-latest?name=BandwidthPredictorLSTM&stage=Production
# Download artifact and load

# 6. Rollback if needed
client.transition_model_version_stage(
    name="BandwidthPredictorLSTM",
    version=2,  # Previous version
    stage="Production"
)
```

**Benefits:**
- Version control for models
- Staged rollout (Staging → Production)
- Quick rollback (revert to previous version)
- Experiment tracking (hyperparameters, metrics)
- Model lineage (which data, code version)

---

## 3. Replace HTTP with gRPC

**Problem:**
HTTP/JSON has high serialization overhead and latency.

**Performance Comparison:**
```
HTTP/JSON:  10-50ms latency, 100-500 bytes payload
gRPC/Proto: 2-10ms latency, 30-150 bytes payload
Improvement: 5-10x throughput, 60-80% latency reduction
```

**Implementation:**

```protobuf
// proto/ai_service.proto
syntax = "proto3";

package novacron.ai;

service AIService {
  rpc PredictBandwidth(BandwidthRequest) returns (BandwidthResponse);
  rpc OptimizePlacement(PlacementRequest) returns (PlacementResponse);
  rpc DetectAnomaly(AnomalyRequest) returns (AnomalyResponse);
}

message BandwidthRequest {
  string vm_id = 1;
  repeated double historical_metrics = 2;
  int32 horizon_minutes = 3;
}

message BandwidthResponse {
  repeated double predictions = 1;
  double confidence = 2;
  string model_version = 3;
}

message PlacementRequest {
  string workload_id = 1;
  map<string, double> requirements = 2;
  repeated NodeInfo available_nodes = 3;
}

message PlacementResponse {
  repeated PlacementCandidate candidates = 1;
  double overall_confidence = 2;
}

message NodeInfo {
  string node_id = 1;
  double cpu_available = 2;
  double memory_available = 3;
  // ... 100+ fields
}

message PlacementCandidate {
  string node_id = 1;
  double score = 2;
  string reasoning = 3;
}

message AnomalyRequest {
  string vm_id = 1;
  VMMetrics current_metrics = 2;
}

message AnomalyResponse {
  bool is_anomaly = 1;
  double score = 2;
  string anomaly_type = 3;
  string severity = 4;
}

message VMMetrics {
  double cpu = 1;
  double memory = 2;
  double disk_io = 3;
  double network_io = 4;
  double latency = 5;
  double error_rate = 6;
}
```

```python
# Python server
import grpc
from concurrent import futures
import ai_service_pb2
import ai_service_pb2_grpc
from bandwidth_predictor import BandwidthPredictor

class AIServiceServicer(ai_service_pb2_grpc.AIServiceServicer):
    def __init__(self):
        self.bandwidth_predictor = BandwidthPredictor()

    def PredictBandwidth(self, request, context):
        predictions = self.bandwidth_predictor.predict(
            vm_id=request.vm_id,
            historical_metrics=request.historical_metrics,
            horizon_minutes=request.horizon_minutes
        )

        return ai_service_pb2.BandwidthResponse(
            predictions=predictions["values"],
            confidence=predictions["confidence"],
            model_version="v1.3.0"
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(
        AIServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

```go
// Go client
package ai

import (
	"context"
	"time"

	pb "novacron/proto/ai_service"
	"google.golang.org/grpc"
)

type GRPCAIClient struct {
	conn   *grpc.ClientConn
	client pb.AIServiceClient
}

func NewGRPCAIClient(endpoint string) (*GRPCAIClient, error) {
	conn, err := grpc.Dial(endpoint,
		grpc.WithInsecure(),
		grpc.WithTimeout(30*time.Second),
		grpc.WithMaxMsgSize(10*1024*1024), // 10MB max
	)
	if err != nil {
		return nil, err
	}

	return &GRPCAIClient{
		conn:   conn,
		client: pb.NewAIServiceClient(conn),
	}, nil
}

func (c *GRPCAIClient) PredictBandwidth(ctx context.Context, vmID string, metrics []float64, horizon int32) (*pb.BandwidthResponse, error) {
	req := &pb.BandwidthRequest{
		VmId:              vmID,
		HistoricalMetrics: metrics,
		HorizonMinutes:    horizon,
	}

	resp, err := c.client.PredictBandwidth(ctx, req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}
```

**Migration Plan:**
1. Week 1: Define protobuf schemas
2. Week 2: Implement Python gRPC server (parallel with HTTP)
3. Week 3: Implement Go gRPC client
4. Week 4: A/B test (HTTP vs gRPC)
5. Week 5: Migrate 100% to gRPC, deprecate HTTP

---

## 4. Batch Inference Optimization

**Problem:**
Individual predictions have high overhead (network, serialization).

**Solution: Batch Inference API**

```python
# Python batch endpoint
@app.post("/api/v1/predict/batch")
async def predict_batch(batch: BatchPredictionRequest):
    """Process multiple predictions in a single call"""

    # Extract features for all VMs in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        features = list(executor.map(extract_features, batch.vms))

    # Batch predict (efficient)
    features_array = np.array(features)
    predictions = model.predict(features_array)  # Single model call

    # Format responses
    results = []
    for i, vm in enumerate(batch.vms):
        results.append(PredictionResponse(
            vm_id=vm.vm_id,
            predictions=predictions[i].tolist(),
            confidence=calculate_confidence(predictions[i])
        ))

    return BatchPredictionResponse(results=results)
```

```go
// Go batch client
func (pa *PredictiveAllocator) BatchPredict(vmIDs []string) (map[string]*Prediction, error) {
	// Collect metrics for all VMs
	batch := make([]*VMMetrics, len(vmIDs))
	for i, vmID := range vmIDs {
		batch[i] = pa.getMetrics(vmID)
	}

	// Single API call for all VMs
	req := BatchPredictionRequest{VMs: batch}
	resp, err := pa.aiClient.PredictBatch(ctx, req)
	if err != nil {
		return nil, err
	}

	// Map results
	results := make(map[string]*Prediction)
	for _, result := range resp.Results {
		results[result.VmId] = result
	}

	return results, nil
}
```

**Expected Improvement:**
- 10 VMs: 10x faster (1 call vs 10 calls)
- 100 VMs: 50x faster (batched processing)
- Latency: 250ms (batch) vs 2500ms (serial)

---

## 5. Drift Detection Implementation

**Problem:**
No monitoring for data drift or model drift in production.

**Solution: Evidently AI Integration**

```python
# Production monitoring
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, NumTargetDriftTab

# Reference data (training set)
reference_data = load_training_data()

# Production data (last 7 days)
production_data = load_production_data(days=7)

# Create drift report
drift_dashboard = Dashboard(tabs=[
    DataDriftTab(),
    NumTargetDriftTab()
])

drift_dashboard.calculate(
    reference_data=reference_data,
    current_data=production_data,
    column_mapping={'target': 'bandwidth_usage'}
)

drift_dashboard.save('reports/drift_report.html')

# Check for drift
drift_detected = drift_dashboard.get_metrics()['data_drift']['n_drifted_features'] > 0

if drift_detected:
    # Alert and trigger retraining
    alert_ops_team()
    trigger_retraining_pipeline()
```

**Monitoring Dashboard:**
```python
# Grafana dashboard metrics
from prometheus_client import Gauge

# Data drift metrics
data_drift_score = Gauge('model_data_drift_score', 'Data drift score (0-1)')
features_drifted = Gauge('model_features_drifted', 'Number of drifted features')

# Model performance metrics
prediction_accuracy = Gauge('model_prediction_accuracy', 'Rolling 7-day accuracy')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')

# Update periodically
def monitor_drift():
    score = calculate_drift_score()
    data_drift_score.set(score)

    if score > 0.3:  # Drift threshold
        send_alert("Data drift detected: %.2f" % score)
```

---

## 6. Automated Retraining Pipeline

**Problem:**
Manual retraining is error-prone and slow.

**Solution: Kubeflow Pipelines**

```python
# training_pipeline.py
import kfp
from kfp import dsl

@dsl.component
def collect_data():
    """Collect last 30 days of production metrics"""
    # Query time series DB
    data = query_metrics(days=30)
    return data

@dsl.component
def preprocess_data(data):
    """Feature engineering and validation"""
    features = extract_features(data)
    validate_data_quality(features)
    return features

@dsl.component
def train_model(features):
    """Train LSTM model"""
    model = LSTMModel()
    model.fit(features)
    return model

@dsl.component
def evaluate_model(model, test_data):
    """Evaluate on holdout set"""
    accuracy = model.evaluate(test_data)
    if accuracy < 0.85:
        raise ValueError("Accuracy below threshold")
    return accuracy

@dsl.component
def deploy_model(model):
    """Deploy to staging, then production"""
    deploy_to_staging(model)
    run_smoke_tests()
    deploy_to_production(model)

@dsl.pipeline(name='Bandwidth Predictor Retraining')
def retraining_pipeline():
    data = collect_data()
    features = preprocess_data(data)
    model = train_model(features)
    accuracy = evaluate_model(model, features)
    deploy_model(model)

# Trigger on schedule or drift detection
if __name__ == '__main__':
    kfp.Client().create_run_from_pipeline_func(
        retraining_pipeline,
        arguments={}
    )
```

**Triggers:**
1. **Scheduled:** Weekly retraining (cron)
2. **Drift-based:** When data drift > 0.3
3. **Performance-based:** When accuracy < 0.80
4. **Manual:** Operator-initiated

---

## 7. Production Deployment Architecture

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-service
spec:
  replicas: 3  # Horizontal scaling
  selector:
    matchLabels:
      app: ai-service
  template:
    metadata:
      labels:
        app: ai-service
        version: v1.3.0
    spec:
      containers:
      - name: ai-service
        image: novacron/ai-service:1.3.0
        ports:
        - containerPort: 50051  # gRPC
        env:
        - name: MODEL_VERSION
          value: "v1.3.0"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ai-service
spec:
  selector:
    app: ai-service
  ports:
  - port: 50051
    targetPort: 50051
    protocol: TCP
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Monitoring Stack:**
```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'ai-service'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: ai-service

# Grafana dashboard
{
  "dashboard": {
    "title": "AI Service Metrics",
    "panels": [
      {
        "title": "Prediction Latency",
        "targets": [
          "histogram_quantile(0.95, rate(model_prediction_latency_seconds[5m]))"
        ]
      },
      {
        "title": "Prediction Accuracy",
        "targets": ["model_prediction_accuracy"]
      },
      {
        "title": "Data Drift Score",
        "targets": ["model_data_drift_score"]
      },
      {
        "title": "Request Rate",
        "targets": ["rate(grpc_server_handled_total[5m])"]
      }
    ]
  }
}
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Week 1:**
- [ ] Deploy feature extraction gRPC service
- [ ] Set up MLflow server (tracking + registry)
- [ ] Implement batch inference endpoint
- [ ] Add comprehensive logging

**Week 2:**
- [ ] Implement gRPC API (parallel with HTTP)
- [ ] Set up Prometheus metrics
- [ ] Create Grafana dashboards
- [ ] Deploy to staging environment

### Phase 2: Integration (Weeks 3-4)

**Week 3:**
- [ ] Migrate Go client to gRPC
- [ ] A/B test HTTP vs gRPC
- [ ] Implement drift detection (Evidently)
- [ ] Set up automated alerts

**Week 4:**
- [ ] Full gRPC migration
- [ ] Integrate with DWCP v3
- [ ] End-to-end testing
- [ ] Performance benchmarking

### Phase 3: Automation (Weeks 5-6)

**Week 5:**
- [ ] Implement automated retraining pipeline
- [ ] Set up CI/CD for models
- [ ] Deploy model registry
- [ ] Create runbooks

**Week 6:**
- [ ] Production deployment (canary)
- [ ] Monitor for 1 week
- [ ] Tune thresholds
- [ ] Full rollout

---

## 9. Success Metrics

**Integration Quality:**
- ✅ Feature consistency: 100% (training = serving)
- ✅ Prediction latency: <50ms (gRPC)
- ✅ Batch throughput: >1000 predictions/sec
- ✅ Model versioning: 100% of models registered
- ✅ Drift detection: <1 hour detection time

**Operational:**
- ✅ Uptime: 99.9% (3 nines)
- ✅ Automated retraining: <4 hours end-to-end
- ✅ Rollback time: <5 minutes
- ✅ Alert response: <15 minutes

**Business:**
- ✅ PBA accuracy: ≥85%
- ✅ ITP speed improvement: 2x
- ✅ Resource utilization: +20%
- ✅ Cost savings: 15%

---

## 10. Conclusion

These integration improvements are **critical for Phase 2 success**. Priority order:

1. **Feature consistency** (Week 1) - Blocks accuracy validation
2. **Model registry** (Week 1) - Enables versioning and rollback
3. **gRPC migration** (Weeks 2-3) - Performance requirement
4. **Drift detection** (Week 3) - Production reliability
5. **Automated retraining** (Week 5) - Long-term sustainability

**Estimated effort:** 6 weeks (2 engineers)
**Estimated cost:** $50K (infrastructure + tooling)
**Risk level:** Medium (proven technologies, clear path)

---

**Next Steps:**
1. Review recommendations with team
2. Allocate resources (2 ML engineers)
3. Set up infrastructure (MLflow, Feast/gRPC)
4. Begin implementation (Week 1 priorities)
