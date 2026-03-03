# Network Overlay System

The Network Overlay System provides virtualized network infrastructure on top of physical networks, enabling advanced network topologies, isolation, and policy enforcement. This component enables NovaCron to create and manage software-defined networking capabilities across diverse environments.

## Architecture

The network overlay system is built on the following components:

1. **Overlay Manager**: Core coordination component that manages drivers and networks
2. **Overlay Drivers**: Implementations for specific overlay technologies (VXLAN, GENEVE, etc.)
3. **Network Policies**: Rules that define connectivity and security between endpoints
4. **Endpoints**: Virtual network interfaces that connect to overlay networks

## Supported Overlay Technologies

The system supports multiple overlay network technologies:

1. **VXLAN (Virtual Extensible LAN)**
   - Standard Layer 2 overlay over Layer 3 network
   - 24-bit VNI supports up to 16 million networks
   - Uses UDP encapsulation with efficient multicast support

2. **GENEVE (Generic Network Virtualization Encapsulation)**
   - Extensible protocol with option capabilities
   - Designed for flexibility and future extensions
   - Compatible with various control planes

3. **GRE (Generic Routing Encapsulation)**
   - Simple IP encapsulation protocol 
   - Long-established standard with wide support
   - Lightweight encapsulation with minimal overhead

4. **NVGRE (Network Virtualization using GRE)**
   - Microsoft's network virtualization technology
   - Uses GRE with 24-bit Tenant Network Identifier
   - Compatible with Windows environments

5. **MPLS over UDP**
   - MPLS label-switching with UDP transport
   - Retains QoS capabilities of MPLS
   - Enables MPLS across non-MPLS infrastructure

6. **VLAN (Virtual LAN)**
   - Traditional network segmentation (not a true overlay)
   - Limited to 4094 networks (12-bit VLAN ID)
   - Included for completeness and legacy support

## Core Capabilities

### Network Isolation

Creates fully isolated virtual networks with their own address spaces, enabling:
- Multi-tenancy on shared infrastructure
- Workload separation with strong isolation guarantees
- Project-specific network topologies

### Network Policies

Applies fine-grained security policies to control traffic:
- Allow/deny rules based on protocol, port, and source/destination
- Traffic redirection for service insertion
- Quality of Service (QoS) rules
- Rate limiting and traffic shaping

### Layer 2 Extensions

Extends Layer 2 domains across physical networks:
- VM migration without IP address changes
- Topology-agnostic connectivity
- Legacy application support requiring Layer 2 adjacency

### Service Mesh Integration

For overlay drivers that support it:
- Service discovery integration
- Traffic management and load balancing
- Advanced observability for network traffic
- Mutual TLS for service-to-service communication

## Using the Overlay Network System

### Creating an Overlay Network

```go
// Get the overlay manager
overlayManager := overlay.NewOverlayManager()

// Register a VXLAN driver
vxlanDriver, _ := overlay.GetDriverFactory("vxlan")()
overlayManager.RegisterDriver(vxlanDriver)

// Initialize the manager
ctx := context.Background()
overlayManager.Initialize(ctx)

// Create a VXLAN-based overlay network
network := overlay.OverlayNetwork{
    ID:         "net-001",
    Name:       "application-network",
    Type:       overlay.VXLAN,
    CIDR:       "10.0.0.0/24",
    VNI:        1001,
    MTU:        1450,
    Active:     true,
    Interfaces: []string{"eth0"},
    Options: map[string]string{
        "multicast_group": "239.1.1.1",
    },
}

overlayManager.CreateNetwork(ctx, network, "vxlan")
```

### Adding Endpoints

```go
// Add an endpoint to the network
endpoint := overlay.EndpointConfig{
    NetworkID:  "net-001",
    Name:       "app-server-1",
    MACAddress: "02:42:ac:11:00:02",
    IPAddress:  "10.0.0.2/24",
    Options: map[string]string{
        "hostname": "app-server-1",
    },
}

overlayManager.CreateEndpoint(ctx, endpoint)
```

### Creating Network Policies

```go
// Create a network policy
policy := overlay.NetworkPolicy{
    ID:        "policy-001",
    Name:      "web-tier-policy",
    NetworkID: "net-001",
    Priority:  100,
    Rules: []overlay.PolicyRule{
        {
            Type:                "allow",
            SourceSelector:      "role=web",
            DestinationSelector: "role=app",
            Protocol:            "tcp",
            DestinationPortRange: "8090-8100",
        },
        {
            Type:                "deny",
            SourceSelector:      "*",
            DestinationSelector: "role=db",
            Protocol:            "*",
        },
    },
}

overlayManager.ApplyNetworkPolicy(ctx, policy)
```

## Implementation Details

- Thread-safe design with RWMutex protection for concurrent access
- Context-based operations for cancellation support
- Pluggable driver architecture for modular implementation
- Factory pattern for driver instantiation
- Clear error handling and logging
