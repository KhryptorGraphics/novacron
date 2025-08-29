# NovaCron Enhanced SDK Framework

Production-ready multi-language SDKs providing seamless access to the enhanced NovaCron platform with multi-cloud federation, AI integration, and advanced reliability features.

## Features

### 🚀 **Multi-Cloud Federation**
- Unified API across AWS, Azure, GCP, OpenStack, and VMware
- Cross-cloud VM migration and workload placement
- Cost optimization across cloud providers
- Regional failover and disaster recovery

### 🤖 **AI-Powered Operations**
- Intelligent VM placement recommendations
- Predictive scaling based on workload patterns
- Anomaly detection for proactive monitoring
- Cost optimization suggestions

### ⚡ **Advanced Performance & Reliability**
- Redis-based caching for improved response times
- Circuit breaker pattern for fault tolerance
- Automatic JWT token refresh
- Exponential backoff retry logic
- Real-time metrics and monitoring

### 🔄 **Real-Time Event Streaming**
- WebSocket support for live updates
- Federated event streaming across clouds
- Batch operations with controlled concurrency
- Asynchronous operations support

## Language Support

| Language | Version | Features |
|----------|---------|----------|
| **Python** | 3.8+ | Full async/await, type hints, Redis caching |
| **JavaScript/TypeScript** | Node 16+ | ES2020+, TypeScript definitions, Promise-based |
| **Go** | 1.21+ | Context support, goroutines, structured logging |

## Quick Start

### Python

```python
import asyncio
from novacron.enhanced_client import EnhancedNovaCronClient, CloudProvider

async def main():
    async with EnhancedNovaCronClient(
        base_url="https://api.novacron.io",
        api_token="your_jwt_token",
        cloud_provider=CloudProvider.AWS,
        region="us-west-2",
        enable_ai_features=True,
        redis_url="redis://localhost:6379"
    ) as client:
        
        # Get AI-powered placement recommendation
        vm_specs = {
            "name": "web-server",
            "cpu_shares": 2048,
            "memory_mb": 4096,
            "disk_size_gb": 50
        }
        
        recommendation = await client.get_intelligent_placement_recommendation(
            vm_specs=vm_specs,
            constraints={"availability_zone": "us-west-2a"}
        )
        
        print(f"Recommended node: {recommendation['recommended_node']}")
        print(f"Confidence: {recommendation['confidence_score']}")
        
        # Create VM with AI placement
        vm = await client.create_vm_with_ai_placement(
            vm_specs,
            use_ai_placement=True
        )
        
        print(f"VM created: {vm.id}")

asyncio.run(main())
```

### TypeScript

```typescript
import { EnhancedNovaCronClient, CloudProvider } from '@novacron/enhanced-sdk';

async function main() {
  const client = new EnhancedNovaCronClient({
    baseURL: 'https://api.novacron.io',
    apiToken: 'your_jwt_token',
    cloudProvider: CloudProvider.AWS,
    region: 'us-west-2',
    enableAIFeatures: true,
    redisUrl: 'redis://localhost:6379'
  });

  try {
    // Batch create VMs with AI placement
    const vmSpecs = [
      { name: 'web-1', cpu_shares: 1024, memory_mb: 2048, disk_size_gb: 20 },
      { name: 'web-2', cpu_shares: 1024, memory_mb: 2048, disk_size_gb: 20 },
      { name: 'web-3', cpu_shares: 1024, memory_mb: 2048, disk_size_gb: 20 }
    ];

    const results = await client.batchCreateVMs(
      vmSpecs,
      3, // concurrency
      true // use AI placement
    );

    console.log(`Created ${results.filter(r => !(r instanceof Error)).length} VMs`);

    // Stream federated events
    const eventStream = client.streamFederatedEvents(
      ['vm.created', 'vm.migrated'],
      [CloudProvider.AWS, CloudProvider.GCP]
    );

    eventStream.on('event', (event) => {
      console.log('Federated event:', event);
    });

  } finally {
    await client.close();
  }
}

main().catch(console.error);
```

### Go

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/khryptorgraphics/novacron-enhanced-sdk"
)

func main() {
    config := novacron.EnhancedClientConfig{
        BaseURL:           "https://api.novacron.io",
        APIToken:          "your_jwt_token",
        CloudProvider:     novacron.CloudProviderAWS,
        Region:            "us-west-2",
        EnableAIFeatures:  true,
        RedisURL:         "redis://localhost:6379",
        EnableMetrics:    true,
    }

    client, err := novacron.NewEnhancedClient(config)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Get cost comparison for cross-cloud migration
    vmSpecs := map[string]interface{}{
        "cpu_shares":   2048,
        "memory_mb":    4096,
        "disk_size_gb": 100,
    }

    costs, err := client.GetCrossCloudCosts(
        ctx,
        novacron.CloudProviderAWS,
        novacron.CloudProviderGCP,
        vmSpecs,
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("AWS cost: $%.2f/month\n", costs["aws_cost"])
    fmt.Printf("GCP cost: $%.2f/month\n", costs["gcp_cost"])
    fmt.Printf("Savings: $%.2f/month\n", costs["potential_savings"])

    // Batch migrate VMs
    migrations := []novacron.MigrationSpec{
        {VMID: "vm-1", TargetNodeID: "node-gcp-1", Type: "live"},
        {VMID: "vm-2", TargetNodeID: "node-gcp-2", Type: "live"},
    }

    results, err := client.BatchMigrateVMs(ctx, migrations, 2)
    if err != nil {
        log.Fatal(err)
    }

    for i, result := range results {
        if err, ok := result.(error); ok {
            fmt.Printf("Migration %d failed: %v\n", i, err)
        } else {
            fmt.Printf("Migration %d succeeded\n", i)
        }
    }

    // Get performance metrics
    metrics := client.GetRequestMetrics()
    for endpoint, metric := range metrics {
        fmt.Printf("%s: avg=%.2fms, p95=%.2fms\n",
            endpoint, metric.AvgDuration*1000, metric.P95Duration*1000)
    }
}
```

## Advanced Configuration

### Caching Strategy

```python
# Python - Redis caching with custom TTL
client = EnhancedNovaCronClient(
    base_url="https://api.novacron.io",
    redis_url="redis://localhost:6379",
    cache_ttl=600,  # 10 minutes
)

# Cache specific operations
vms = await client.list_vms(use_cache=True, cache_ttl=300)
```

```typescript
// TypeScript - Multi-level caching
const client = new EnhancedNovaCronClient({
  redisUrl: 'redis://localhost:6379',
  cacheTTL: 600,
  // Automatic caching for GET operations
});
```

### Circuit Breaker Configuration

```go
// Go - Custom circuit breaker settings
config := novacron.EnhancedClientConfig{
    CircuitBreakerThreshold: 10,    // 10 failures
    CircuitBreakerTimeout:   30 * time.Second,
    EnableMetrics:          true,
}

client, _ := novacron.NewEnhancedClient(config)

// Monitor circuit breaker status
status := client.GetCircuitBreakerStatus()
for endpoint, state := range status {
    fmt.Printf("%s: %s (failures: %d)\n", 
        endpoint, state.State, state.Failures)
}
```

### AI Feature Configuration

```python
# Enable specific AI features
client = EnhancedNovaCronClient(
    enable_ai_features=True,
    ai_features=[
        AIFeature.INTELLIGENT_PLACEMENT,
        AIFeature.COST_OPTIMIZATION
    ]
)

# Get predictive scaling recommendations
forecast = await client.get_predictive_scaling_forecast(
    vm_id="vm-123",
    forecast_hours=48
)

# Detect anomalies
anomalies = await client.detect_anomalies(
    vm_id="vm-123",
    time_window=3600  # 1 hour
)
```

## Multi-Cloud Federation Examples

### Cross-Cloud Migration

```python
# Migrate VM from AWS to GCP
migration = await client.create_cross_cloud_migration(
    vm_id="vm-aws-123",
    target_cluster="gcp-cluster-west",
    target_provider=CloudProvider.GCP,
    target_region="us-west1",
    migration_options={
        "compression": True,
        "bandwidth_limit": 1000,  # Mbps
        "live_migration": True
    }
)
```

### Federated Resource Management

```typescript
// List resources across all federated clusters
const clusters = await client.listFederatedClusters();

for (const cluster of clusters) {
    console.log(`${cluster.name}: ${cluster.provider} (${cluster.region})`);
    console.log(`  VMs: ${cluster.vm_count}`);
    console.log(`  Capacity: ${cluster.available_capacity}%`);
}
```

### Cost Optimization

```go
// Get cost optimization recommendations
recommendations, err := client.GetCostOptimizationRecommendations(ctx, "")
if err != nil {
    log.Fatal(err)
}

for _, rec := range recommendations {
    fmt.Printf("Recommendation: %s\n", rec["title"])
    fmt.Printf("Potential savings: $%.2f/month\n", rec["monthly_savings"])
    fmt.Printf("Action: %s\n", rec["action"])
}
```

## Performance Monitoring

### Request Metrics

```python
# Get detailed performance metrics
metrics = client.get_request_metrics()

for endpoint, metric in metrics.items():
    print(f"{endpoint}:")
    print(f"  Requests: {metric['count']}")
    print(f"  Average: {metric['avg_duration']:.3f}s")
    print(f"  95th percentile: {metric['p95_duration']:.3f}s")
```

### Circuit Breaker Status

```typescript
// Monitor circuit breaker health
const status = client.getCircuitBreakerStatus();

for (const [endpoint, state] of Object.entries(status)) {
    if (state.isOpen) {
        console.warn(`⚠️  ${endpoint} circuit breaker is OPEN`);
        console.warn(`   Failures: ${state.failures}`);
        console.warn(`   Last failure: ${state.lastFailure}`);
    }
}
```

## Error Handling

### Comprehensive Error Types

```python
from novacron.exceptions import (
    AuthenticationError,
    ValidationError,
    ResourceNotFoundError,
    CircuitBreakerError,
    APIError
)

try:
    vm = await client.create_vm(vm_request)
except AuthenticationError:
    # Token expired, refresh automatically handled
    pass
except ValidationError as e:
    print(f"Invalid VM configuration: {e}")
except CircuitBreakerError:
    print("Service temporarily unavailable")
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
```

### Retry Logic with Exponential Backoff

```typescript
// Automatic retry with exponential backoff
const client = new EnhancedNovaCronClient({
  maxRetries: 5,
  retryDelay: 1000, // Start with 1 second
  // Exponential backoff: 1s, 2s, 4s, 8s, 16s
});

// Custom retry for specific operations
import { backoff } from 'exponential-backoff';

const result = await backoff(
  () => client.createVM(vmSpec),
  {
    numOfAttempts: 3,
    startingDelay: 1000,
    maxDelay: 10000,
  }
);
```

## Real-Time Event Streaming

### WebSocket Event Streaming

```python
# Stream VM events with filtering
async for event in client.stream_vm_events(vm_id="vm-123"):
    if event["type"] == "vm.state_changed":
        print(f"VM {event['vm_id']} is now {event['state']}")
    elif event["type"] == "vm.metrics_updated":
        print(f"CPU: {event['cpu_usage']:.1f}%")
```

### Federated Event Streaming

```go
// Stream events from multiple cloud providers
import "context"

ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
defer cancel()

events := client.StreamFederatedEvents(ctx, []string{"vm.migrated", "cluster.scaled"})

for event := range events {
    fmt.Printf("Event from %s: %s\n", event.CloudProvider, event.Type)
    fmt.Printf("Details: %+v\n", event.Data)
}
```

## Batch Operations

### Concurrent VM Operations

```python
# Create 50 VMs concurrently with controlled concurrency
vm_requests = [
    CreateVMRequest(name=f"worker-{i}", cpu_shares=1024, memory_mb=2048)
    for i in range(50)
]

results = await client.batch_create_vms(
    requests=vm_requests,
    concurrency=10,  # Max 10 concurrent operations
    use_ai_placement=True
)

successful = [r for r in results if isinstance(r, VM)]
failed = [r for r in results if isinstance(r, Exception)]

print(f"Created: {len(successful)}, Failed: {len(failed)}")
```

### Batch Migration with Progress Tracking

```typescript
// Migrate multiple VMs with progress monitoring
const migrations = [
  { vm_id: 'vm-1', target_node_id: 'node-2', type: 'live' },
  { vm_id: 'vm-2', target_node_id: 'node-3', type: 'live' },
  // ... more migrations
];

const results = await client.batchMigrateVMs(migrations, 3);

// Track progress
for (let i = 0; i < results.length; i++) {
  const result = results[i];
  if (result instanceof Error) {
    console.error(`Migration ${i} failed:`, result.message);
  } else {
    console.log(`Migration ${i} started:`, result.id);
    
    // Monitor migration progress
    const migration = await client.getMigration(result.id);
    console.log(`Progress: ${migration.progress}%`);
  }
}
```

## Testing

### Unit Tests

```python
# Python - pytest with async support
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_vm_creation_with_ai():
    client = EnhancedNovaCronClient("https://test.api")
    client._request = AsyncMock(return_value={"id": "vm-123"})
    
    vm = await client.create_vm_with_ai_placement(
        vm_specs={"name": "test-vm"},
        use_ai_placement=True
    )
    
    assert vm.id == "vm-123"
```

### Integration Tests

```typescript
// TypeScript - Jest integration tests
describe('Enhanced NovaCron Client', () => {
  let client: EnhancedNovaCronClient;

  beforeEach(() => {
    client = new EnhancedNovaCronClient({
      baseURL: process.env.TEST_API_URL,
      apiToken: process.env.TEST_API_TOKEN,
    });
  });

  afterEach(async () => {
    await client.close();
  });

  test('should create VM with AI placement', async () => {
    const vm = await client.createVMWithAIPlacement({
      name: 'test-vm',
      cpu_shares: 1024,
      memory_mb: 2048,
      disk_size_gb: 20,
    });

    expect(vm.id).toBeDefined();
    expect(vm.placement_reasoning).toBeDefined();
  });
});
```

## Production Deployment

### Environment Configuration

```bash
# Environment variables
export NOVACRON_API_URL="https://api.novacron.io"
export NOVACRON_API_TOKEN="your_jwt_token"
export NOVACRON_REDIS_URL="redis://redis-cluster:6379"
export NOVACRON_CLOUD_PROVIDER="aws"
export NOVACRON_REGION="us-west-2"
export NOVACRON_ENABLE_AI="true"
export NOVACRON_LOG_LEVEL="info"
```

### Docker Configuration

```dockerfile
# Dockerfile for Python SDK
FROM python:3.11-slim

WORKDIR /app

COPY requirements-enhanced.txt .
RUN pip install -r requirements-enhanced.txt

COPY . .

ENV PYTHONPATH=/app
ENV NOVACRON_ENABLE_METRICS=true

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: novacron-app:latest
        env:
        - name: NOVACRON_API_URL
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: api-url
        - name: NOVACRON_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: api-token
        - name: NOVACRON_REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Security Best Practices

### Token Management

```python
# Automatic token refresh with secure storage
import keyring

class SecureTokenManager:
    def __init__(self):
        self.service_name = "novacron-sdk"
    
    def get_token(self):
        return keyring.get_password(self.service_name, "api_token")
    
    def set_token(self, token):
        keyring.set_password(self.service_name, "api_token", token)

# Use with SDK
token_manager = SecureTokenManager()
client = EnhancedNovaCronClient(
    api_token=token_manager.get_token(),
    # Token refresh callback
    token_refresh_callback=token_manager.set_token
)
```

### TLS Configuration

```go
// Go - Custom TLS configuration
import (
    "crypto/tls"
    "net/http"
)

config := novacron.EnhancedClientConfig{
    BaseURL: "https://api.novacron.io",
    TLSConfig: &tls.Config{
        MinVersion: tls.VersionTLS12,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
        },
    },
}
```

## Monitoring and Observability

### Prometheus Metrics

```python
# Expose SDK metrics to Prometheus
from prometheus_client import Counter, Histogram, start_http_server

# Custom metrics
sdk_requests = Counter('novacron_sdk_requests_total', 
                      'Total SDK requests', ['method', 'endpoint'])
sdk_duration = Histogram('novacron_sdk_request_duration_seconds',
                        'SDK request duration')

# Integrate with client
client = EnhancedNovaCronClient(
    metrics_callback=lambda method, endpoint, duration: (
        sdk_requests.labels(method=method, endpoint=endpoint).inc(),
        sdk_duration.observe(duration)
    )
)

# Start metrics server
start_http_server(8000)
```

### Structured Logging

```typescript
// Structured logging with correlation IDs
import { createLogger } from 'winston';

const logger = createLogger({
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [new winston.transports.Console()]
});

const client = new EnhancedNovaCronClient({
  logger: {
    info: (msg, meta) => logger.info(msg, meta),
    warn: (msg, meta) => logger.warn(msg, meta),
    error: (msg, meta) => logger.error(msg, meta),
  }
});
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `npm test` / `pytest` / `go test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@novacron.io
- 💬 Discord: https://discord.gg/novacron
- 📖 Documentation: https://docs.novacron.io
- 🐛 Issues: https://github.com/khryptorgraphics/novacron/issues