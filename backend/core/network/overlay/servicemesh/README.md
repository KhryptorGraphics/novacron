# Service Mesh System

The Service Mesh System provides advanced service discovery, traffic management, and secure service-to-service communication capabilities. It enables NovaCron to manage microservices connectivity with fine-grained control over traffic flow, security, and observability.

## Architecture

The service mesh architecture consists of the following components:

1. **Service Mesh Manager**: Core management component that orchestrates the service mesh
2. **Service Discovery**: Registry for locating services across the mesh
3. **Proxy Sidecar**: Data plane component that intercepts and controls service traffic
4. **Traffic Policies**: Rules that define how traffic should be routed between services

## Key Features

### Service Discovery and Load Balancing

- Dynamic service registration and discovery
- Multiple load balancing algorithms (round-robin, least connections, etc.)
- Health checking with automatic failover
- Weighted traffic distribution

### Traffic Management

- Request routing based on path, headers, or other criteria
- Circuit breaking to prevent cascade failures
- Timeouts and retries with backoff
- Rate limiting to protect services from overload
- Fault injection for reliability testing

### Security

- Mutual TLS (mTLS) for service-to-service authentication
- Certificate management and rotation
- Authorization policies for fine-grained access control
- Traffic encryption for sensitive data

### Observability

- Distributed tracing of requests across services
- Detailed metrics for service traffic
- Access logging for audit and debugging
- Automatic anomaly detection

## Using the Service Mesh

### Registering a Service

```go
// Create a service mesh manager
discoveryClient := consul.NewConsulServiceDiscovery("localhost:8500")
proxyManager := envoy.NewEnvoyProxyManager()
meshManager := servicemesh.NewServiceMeshManager(discoveryClient, proxyManager)

// Initialize the manager
ctx := context.Background()
meshManager.Initialize(ctx)

// Create and register a service
service := &servicemesh.Service{
    Name:      "payment-service",
    Type:      servicemesh.ServiceHTTP,
    VirtualIP: "10.0.0.1",
    Port:      8090,
    Endpoints: []*servicemesh.ServiceEndpoint{
        {
            ID:      "payment-1",
            Address: "192.168.1.10",
            Port:    8090,
            Weight:  100,
            Healthy: true,
            Labels: map[string]string{
                "version": "v1",
                "env":     "prod",
            },
        },
    },
    Settings: map[string]string{
        "protocol": "http/1.1",
    },
}

// Register the service
meshManager.RegisterService(ctx, service)
```

### Applying Traffic Policies

```go
// Create a traffic policy
policy := &servicemesh.TrafficPolicy{
    Name:          "payment-policy",
    ServiceName:   "payment-service",
    LoadBalancing: "round-robin",
    CircuitBreaking: map[string]string{
        "max_connections":     "100",
        "max_pending_requests": "1000",
        "max_requests":        "1000",
        "max_retries":         "3",
    },
    Timeouts: map[string]time.Duration{
        "connect_timeout": 5 * time.Second,
        "request_timeout": 30 * time.Second,
    },
    Retries: map[string]interface{}{
        "attempts":      3,
        "per_try_timeout": 2 * time.Second,
        "retry_on":        "connect-failure,refused-stream,unavailable",
    },
}

// Apply the policy
meshManager.ApplyTrafficPolicy(ctx, policy)
```

### Enabling Mutual TLS

```go
// Enable mTLS for a service
meshManager.EnableMutualTLS(
    ctx,
    "payment-service",
    "/etc/certs/service.crt",
    "/etc/certs/service.key",
    "/etc/certs/ca.crt",
)
```

## Integration with Overlay Networks

The Service Mesh integrates with NovaCron's Network Overlay System to provide:

1. **Virtual Network Connectivity**: Overlay networks provide L2/L3 connectivity for service mesh traffic
2. **Network Isolation**: Services in different overlay networks can communicate securely through the mesh
3. **Traffic Encryption**: Adds TLS encryption on top of overlay network security
4. **Network Policies**: Overlay network policies enforce connection-level security, while service mesh enforces service-level security

## Implementation Details

- Thread-safe implementation with RWMutex protection
- Support for multiple service discovery backends (Consul, etcd, etc.)
- Support for multiple proxy implementations (Envoy, NGINX, etc.)
- Graceful handling of proxy reconfigurations
- Context-based operations for cancellation support
