# NovaCron Monitoring and Observability Stack

Comprehensive monitoring, alerting, and observability platform for the NovaCron virtualization and ML pipeline system.

## 📊 Overview

This monitoring stack provides:
- **System Monitoring**: Infrastructure metrics (CPU, memory, storage, network)
- **Application Performance Monitoring**: Service health, response times, error rates
- **ML Pipeline Monitoring**: Model performance, training metrics, data drift detection
- **Log Aggregation**: Centralized logging with parsing and analysis
- **Distributed Tracing**: End-to-end request tracing across microservices
- **Intelligent Alerting**: Multi-channel notifications with correlation and suppression

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │    │   Infrastructure │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • NovaCron API  │    │ • Node Exporter │    │ • ML Models     │
│ • VM Manager    │    │ • cAdvisor      │    │ • MLE-Star      │
│ • Migration Svc │    │ • Blackbox Exp  │    │ • Training Jobs │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
        ┌─────────────────────────▼─────────────────────────┐
        │              Prometheus                           │
        │           (Metrics Storage)                       │
        └─────────────────────┬───────────────────────────┘
                              │
        ┌─────────────────────▼─────────────────────────────┐
        │             AlertManager                          │
        │         (Alert Routing & Notifications)           │
        └─────────────────────┬───────────────────────────┘
                              │
   ┌──────────┬───────────────┼───────────────┬──────────┐
   │          │               │               │          │
   ▼          ▼               ▼               ▼          ▼
┌─────┐   ┌─────┐         ┌─────┐         ┌─────┐   ┌─────┐
│Slack│   │Email│         │Teams│         │PagerD│   │Webhook
└─────┘   └─────┘         └─────┘         └─────┘   └─────┘

┌─────────────────────────────────────────────────────────────┐
│                      Grafana                                │
│               (Visualization & Dashboards)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Elasticsearch │    │     Jaeger      │    │      Loki       │
│  (Log Storage)  │    │   (Tracing)     │    │  (Log Storage)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Development Deployment (Docker Compose)

1. **Start the monitoring stack:**
   ```bash
   cd config/monitoring
   docker-compose up -d
   ```

2. **Access services:**
   - Grafana: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090
   - AlertManager: http://localhost:9093
   - Jaeger: http://localhost:16686
   - Kibana: http://localhost:5601

3. **Configure data sources:**
   ```bash
   # Data sources are auto-provisioned via config files
   # Check grafana/provisioning/datasources/
   ```

### Production Deployment (Kubernetes)

1. **Create monitoring namespace:**
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   ```

2. **Deploy Prometheus:**
   ```bash
   kubectl apply -f kubernetes/prometheus-deployment.yaml
   ```

3. **Deploy Grafana:**
   ```bash
   kubectl apply -f kubernetes/grafana-deployment.yaml
   ```

4. **Deploy AlertManager:**
   ```bash
   kubectl apply -f kubernetes/alertmanager-deployment.yaml
   ```

## 📈 Monitoring Components

### System Monitoring

#### Node Exporter
- **Purpose**: System-level metrics collection
- **Metrics**: CPU, memory, disk, network, filesystem
- **Port**: 9100
- **Configuration**: Auto-discovered by Prometheus

#### cAdvisor  
- **Purpose**: Container metrics collection
- **Metrics**: Container CPU, memory, network, filesystem usage
- **Port**: 8080
- **Integration**: Kubernetes native, Docker daemon

#### Blackbox Exporter
- **Purpose**: External endpoint monitoring
- **Capabilities**: HTTP, TCP, DNS, ICMP probes
- **Port**: 9115
- **Configuration**: `exporters/blackbox.yml`

### Application Monitoring

#### Prometheus Metrics
Services expose metrics on `/metrics` endpoints:
- **NovaCron API**: Port 8080/metrics
- **VM Manager**: Port 8081/metrics  
- **Migration Service**: Port 8082/metrics
- **Scheduler**: Port 8083/metrics
- **ML Pipeline**: Port 8000/metrics

#### Custom Exporters
- **ML Model Exporter**: Custom exporter for ML pipeline metrics
- **Location**: `exporters/ml-model/`
- **Metrics**: Model accuracy, training time, inference latency, drift scores
- **Port**: 9200

### ML Pipeline Monitoring

#### MLE-Star Workflow Monitoring
- **Workflow Stage Tracking**: Duration and success rate per stage
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Training Metrics**: Duration, failures, resource usage
- **Data Quality**: Drift detection, feature importance

#### Key Metrics
```prometheus
# Model performance
ml_model_accuracy{model_name="classifier", version="v1.2"}
ml_model_precision{model_name="classifier", version="v1.2"}
ml_model_recall{model_name="classifier", version="v1.2"}
ml_model_f1_score{model_name="classifier", version="v1.2"}

# Training metrics  
ml_training_duration_seconds{model_name="classifier"}
ml_training_failures_total{model_name="classifier", error_type="data_error"}

# Inference metrics
ml_inference_duration_seconds{model_name="classifier"}
ml_predictions_total{model_name="classifier", status="success"}

# Data quality
ml_data_drift_score{model_name="classifier", dataset="production"}
```

## 🔔 Alerting Framework

### Alert Rules

#### Infrastructure Alerts
- **HighCPUUsage**: CPU > 80% for 2 minutes
- **CriticalCPUUsage**: CPU > 95% for 1 minute  
- **HighMemoryUsage**: Memory > 85% for 2 minutes
- **HighDiskUsage**: Disk > 80% for 2 minutes
- **NodeDown**: Node unreachable for 30 seconds

#### Application Alerts
- **ApplicationDown**: Service down for 30 seconds
- **HighResponseTime**: 95th percentile > 1s for 2 minutes
- **HighErrorRate**: Error rate > 5% for 2 minutes
- **VMCreationFailure**: > 3 failures in 5 minutes
- **MigrationFailure**: Any failure in 5 minutes

#### ML Pipeline Alerts
- **ModelPerformanceDegradation**: Accuracy < 85% for 5 minutes
- **DataDriftDetected**: Drift score > 0.3 for 5 minutes
- **ModelTrainingFailure**: > 2 failures in 10 minutes
- **HighInferenceLatency**: 95th percentile > 1s for 2 minutes

### Notification Channels

#### Slack Integration
```yaml
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#alerts'
    title: 'NovaCron Alert: {{ .GroupLabels.alertname }}'
```

#### Email Notifications
```yaml
email_configs:
  - to: 'ops@novacron.com'
    subject: '[NovaCron] {{ .GroupLabels.alertname }}'
```

#### PagerDuty Integration
```yaml
pagerduty_configs:
  - routing_key: 'your-pagerduty-integration-key'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

## 📊 Dashboards

### Infrastructure Dashboard
- **System Overview**: Node status, resource utilization
- **CPU & Memory**: Time series charts with thresholds
- **Disk Usage**: Bar charts by filesystem
- **Network I/O**: Traffic patterns and utilization
- **Container Resources**: Docker/K8s container metrics

### Application Dashboard  
- **Service Health**: Up/down status for all services
- **Response Times**: 95th percentile latency trends
- **Error Rates**: Error percentage by service
- **Request Volume**: Requests per second
- **Database Performance**: Query times and connections

### ML Pipeline Dashboard
- **Model Performance**: Accuracy, precision, recall over time
- **Training Metrics**: Job duration and success rates
- **Inference Latency**: Response time distributions  
- **Data Quality**: Drift scores and feature importance
- **Resource Usage**: GPU utilization and memory

### Custom Dashboards
Create custom dashboards by:
1. Importing JSON from `dashboards/` directory
2. Using Grafana UI to build new dashboards
3. Saving dashboard JSON for version control

## 🔍 Log Aggregation

### ELK Stack Configuration

#### Elasticsearch
- **Purpose**: Log storage and indexing
- **Retention**: Hot-warm-cold architecture
- **Indices**: 
  - `novacron-application-*`: Application logs
  - `novacron-ml-*`: ML pipeline logs  
  - `novacron-access-*`: Access logs
  - `novacron-errors-*`: Error logs

#### Logstash
- **Purpose**: Log processing and enrichment
- **Features**:
  - Structured log parsing
  - GeoIP enrichment
  - Metric extraction from logs
  - Error correlation

#### Kibana  
- **Purpose**: Log visualization and search
- **Features**:
  - Full-text search across logs
  - Log pattern analysis
  - Custom dashboards
  - Alerting on log patterns

### Log Parsing Rules

#### Application Logs
```ruby
# API Request logs
%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:log_level} \[%{DATA:thread}\] %{DATA:logger} - %{GREEDYDATA:log_message}

# ML Pipeline logs  
%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:log_level} \[%{DATA:stage}\] Model-%{DATA:model_name}: %{GREEDYDATA:log_message}

# MLE-Star workflow logs
%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:log_level} Workflow-%{DATA:workflow_id} Stage-%{DATA:stage}: %{GREEDYDATA:log_message}
```

## 🕵️ Distributed Tracing

### Jaeger Configuration
- **Collector**: Port 14268 (HTTP), 14250 (gRPC)
- **Query**: Port 16686 (UI)
- **Agent**: Port 6831 (UDP), 6832 (UDP)

### Trace Instrumentation
Services should instrument HTTP requests, database calls, and external service calls:

```python
# Example Python instrumentation
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Trace Analysis
- **Service Map**: Visualize service dependencies
- **Performance Analysis**: Identify bottlenecks
- **Error Correlation**: Link errors across services
- **SLA Monitoring**: Track request latencies

## ☁️ Cloud Provider Integrations

### AWS CloudWatch
- **Configuration**: `cloud/aws-cloudwatch-config.yaml`
- **Features**: Custom metrics, log groups, alarms, dashboards
- **X-Ray Integration**: Distributed tracing

### Google Cloud Monitoring
- **Service Account**: Required for API access
- **Custom Metrics**: Pushed via Cloud Monitoring API
- **Log Export**: To Cloud Logging

### Azure Monitor  
- **Application Insights**: APM and custom metrics
- **Log Analytics**: Centralized logging
- **Azure Alerts**: Integration with AlertManager

## 🛠️ Configuration Files

### Key Configuration Files
```
config/monitoring/
├── docker-compose.yml              # Development deployment
├── prometheus/
│   ├── prometheus.yml             # Main Prometheus config
│   └── rules/                     # Alert rules
│       ├── infrastructure.yml     # Infrastructure alerts  
│       ├── application.yml        # Application alerts
│       └── ml-pipeline.yml       # ML pipeline alerts
├── grafana/
│   ├── provisioning/             # Auto-provisioning config
│   └── dashboards/               # Dashboard definitions
├── alertmanager/
│   └── alertmanager.yml          # Alert routing config
├── logstash/
│   └── pipeline/
│       └── logstash.conf         # Log processing rules
├── kubernetes/                   # K8s manifests
├── exporters/                    # Custom exporters
├── cloud/                        # Cloud provider configs
└── README.md                     # This file
```

### Environment Variables

#### Common Settings
```bash
# Prometheus
PROMETHEUS_RETENTION_TIME=30d
PROMETHEUS_RETENTION_SIZE=50GB

# Grafana  
GF_SECURITY_ADMIN_PASSWORD=your-secure-password
GF_USERS_ALLOW_SIGN_UP=false

# ML Exporter
ML_MODEL_ENDPOINT=http://ml-service:8000
EXPORT_PORT=8000
SCRAPE_INTERVAL=30

# AlertManager
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SMTP_HOST=smtp.gmail.com:587
SMTP_USER=alerts@novacron.com
SMTP_PASSWORD=your-app-password
```

## 🔧 Maintenance

### Regular Tasks

#### Daily
- Monitor alert volume and noise
- Check dashboard for anomalies  
- Verify backup completion
- Review error logs

#### Weekly  
- Update alert thresholds based on trends
- Clean old log indices
- Review ML model performance trends
- Capacity planning review

#### Monthly
- Update monitoring components
- Review and tune alert rules
- Analyze long-term performance trends
- Security updates

### Backup Strategy

#### Prometheus Data
- **Retention**: 30 days local, 1 year remote storage
- **Backup**: Daily snapshots to object storage
- **Recovery**: Point-in-time restore capability

#### Grafana Dashboards
- **Version Control**: All dashboards in Git
- **Export**: Automated JSON export to repository
- **Import**: Provisioning from configuration files

#### Log Data
- **Hot**: Last 7 days on SSD storage
- **Warm**: 8-30 days on standard storage  
- **Cold**: 31-365 days on cold storage
- **Archive**: >1 year compressed archive

### Troubleshooting

#### Common Issues

**Prometheus High Memory Usage**
```bash
# Check memory usage
kubectl top pod prometheus-0 -n monitoring

# Reduce retention or increase memory limits
# Edit prometheus-deployment.yaml
```

**Missing Metrics**
```bash
# Check target health
curl http://prometheus:9090/api/v1/targets

# Verify service discovery
kubectl get endpoints -n monitoring
```

**Alert Fatigue**
```bash
# Review firing alerts
curl http://alertmanager:9093/api/v1/alerts

# Tune alert rules in prometheus/rules/
```

**Dashboard Not Loading**
```bash
# Check Grafana logs
kubectl logs grafana-0 -n monitoring

# Verify data source connectivity
# Grafana > Configuration > Data Sources
```

### Performance Tuning

#### Prometheus Optimization
- Adjust scrape intervals based on requirements
- Use recording rules for complex queries
- Configure appropriate retention policies
- Enable remote storage for long-term data

#### Grafana Optimization
- Use query caching for expensive dashboards
- Implement dashboard folders and permissions
- Enable alerting with proper notification policies
- Optimize panel queries and time ranges

#### Elasticsearch Optimization
- Configure proper shard and replica settings
- Use index lifecycle management (ILM)
- Monitor cluster health and performance
- Implement proper mapping templates

## 🤝 Contributing

### Adding New Metrics
1. Define metrics in the appropriate service
2. Update Prometheus scrape configuration
3. Create dashboard panels in Grafana  
4. Add relevant alert rules
5. Update documentation

### Creating Dashboards
1. Use Grafana UI to build dashboard
2. Export JSON configuration
3. Save to `dashboards/` directory
4. Add to provisioning configuration
5. Test with sample data

### Custom Exporters
1. Follow Prometheus client library conventions
2. Implement health checks and error handling
3. Add Dockerfile and deployment manifests
4. Include comprehensive metrics documentation
5. Add integration tests

## 📚 References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

---

For support and questions, please contact the NovaCron DevOps team or create an issue in the project repository.