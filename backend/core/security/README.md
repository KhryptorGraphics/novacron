# Enhanced Security Features

This package provides comprehensive security capabilities for NovaCron, implementing authentication, authorization, network isolation, and encryption services. The components work together to provide a secure environment for multi-tenant operations.

## Features

- **Authentication Provider**: JWT-based authentication with token management
- **Network Isolation**: VLAN-like network policies for tenant isolation
- **Encryption Services**: Data protection with AES and RSA encryption
- **RBAC**: Role-based access control (integrated with auth system)
- **Audit Logging**: Security event tracking and auditing

## Components

### Authentication Provider

The authentication provider manages user authentication and token-based access:

- JWT-based authentication with RSA signing
- Token management with refresh and revocation
- Claims-based authorization information
- Session management

### Network Isolation

The network isolation system provides VLAN-like isolation between tenants:

- Policy-based network traffic control
- Fine-grained rules for ingress and egress traffic
- CIDR and port-based filtering
- Priority-based rule evaluation
- Default policies with tenant isolation

### Encryption Services

The encryption services provide data protection capabilities:

- AES-GCM encryption for data at rest
- RSA-OAEP encryption for asymmetric operations
- Key management with rotation capabilities
- Field-level encryption for sensitive data

## Usage Examples

### Authentication

```go
// Create auth provider
authProvider, err := security.NewBaseAuthProvider(
    "novacron-auth",
    1*time.Hour,      // Token expiration
    24*time.Hour,     // Refresh token expiration
)
if err != nil {
    log.Fatalf("Failed to create auth provider: %v", err)
}

// Authenticate user
token, err := authProvider.Authenticate("admin", "password")
if err != nil {
    log.Fatalf("Authentication failed: %v", err)
}

// Validate token
claims, err := authProvider.ValidateToken(token)
if err != nil {
    log.Fatalf("Token validation failed: %v", err)
}

// Check permissions
if !contains(claims.Permissions, "admin:read") {
    log.Fatalf("User doesn't have required permission")
}
```

### Network Isolation

```go
// Create network isolation manager
netManager := security.NewNetworkIsolationManager()

// Create default policy for tenant
policy, err := netManager.CreateDefaultPolicy("tenant-123")
if err != nil {
    log.Fatalf("Failed to create policy: %v", err)
}

// Add a specific rule
rule := &security.NetworkRule{
    ID:                 "db-access",
    Name:               "Database Access",
    Description:        "Allow access to database servers",
    Type:               security.AllowPolicy,
    Direction:          security.Ingress,
    Protocol:           "tcp",
    SourceCIDR:         []string{"10.0.0.0/16"},
    DestinationCIDR:    []string{"10.1.0.0/24"},
    DestinationPortRange: []string{"5432"},
    Priority:           100,
    Enabled:            true,
}
err = netManager.AddRule(policy.ID, rule)
if err != nil {
    log.Fatalf("Failed to add rule: %v", err)
}

// Check connectivity
allowed, err := netManager.CheckConnectivity(
    net.ParseIP("10.0.1.5"),    // Source IP
    net.ParseIP("10.1.0.10"),   // Destination IP
    12345,                      // Source port
    5432,                       // Destination port
    "tcp",                      // Protocol
    "tenant-123",               // Tenant ID
)
if err != nil {
    log.Fatalf("Failed to check connectivity: %v", err)
}
if !allowed {
    log.Println("Connection not allowed by network policy")
}
```

### Encryption

```go
// Create encryption manager
encManager := security.NewEncryptionManager()

// Create AES key
aesKey, err := encManager.CreateAESKey(
    "data-encryption-key",
    "Data Encryption Key",
    32, // 256-bit
)
if err != nil {
    log.Fatalf("Failed to create AES key: %v", err)
}

// Encrypt data
sensitiveData := []byte("sensitive information")
encryptedData, err := encManager.EncryptWithAES(sensitiveData, aesKey.ID)
if err != nil {
    log.Fatalf("Failed to encrypt data: %v", err)
}

// Decrypt data
decryptedData, err := encManager.DecryptWithAES(encryptedData, aesKey.ID)
if err != nil {
    log.Fatalf("Failed to decrypt data: %v", err)
}
```

## Integration Points

The security system integrates with other NovaCron components:

- **Authentication**: Used by API and Web services for user access
- **Network Isolation**: Used by the VM manager for network configuration
- **Encryption**: Used by the storage manager for data protection
- **Multi-tenant Architecture**: Security is tenant-aware across all components

## Security Best Practices

The security implementation follows these best practices:

1. **Defense in Depth**: Multiple security layers work together
2. **Least Privilege**: RBAC controls access to only what's needed
3. **Secure by Default**: Secure defaults with explicit opt-out
4. **Encryption**: Sensitive data protected at rest and in transit
5. **Isolation**: Tenants isolated at network and resource levels

## Future Enhancements

- **Hardware Security Module (HSM)**: Integration for key protection
- **Zero Trust**: Expand to full zero trust architecture
- **Compliance Frameworks**: Support for regulatory requirements
- **Identity Federation**: Support for external identity providers
- **Advanced Threat Protection**: Integration with threat detection systems
