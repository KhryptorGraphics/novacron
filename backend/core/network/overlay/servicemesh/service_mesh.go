package servicemesh

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ServiceType defines the type of service in the mesh
type ServiceType string

const (
	// ServiceHTTP represents an HTTP service
	ServiceHTTP ServiceType = "http"
	// ServiceGRPC represents a gRPC service
	ServiceGRPC ServiceType = "grpc"
	// ServiceTCP represents a generic TCP service
	ServiceTCP ServiceType = "tcp"
	// ServiceUDP represents a UDP service
	ServiceUDP ServiceType = "udp"
)

// ServiceEndpoint represents a single endpoint for a service
type ServiceEndpoint struct {
	// ID is a unique identifier for this endpoint
	ID string
	// Address is the IP address or hostname of the endpoint
	Address string
	// Port is the port number
	Port int
	// Weight for load balancing (higher values get more traffic)
	Weight int
	// Labels for metadata filtering
	Labels map[string]string
	// Health status of the endpoint
	Healthy bool
	// Last health check time
	LastHealthCheck time.Time
}

// Service represents a service in the mesh
type Service struct {
	// Name of the service
	Name string
	// Type of service
	Type ServiceType
	// List of endpoints for this service
	Endpoints []*ServiceEndpoint
	// Virtual IP for this service
	VirtualIP string
	// Target port for the service
	Port int
	// Protocol-specific settings
	Settings map[string]string
}

// TrafficPolicy defines how traffic should be routed to a service
type TrafficPolicy struct {
	// Name of the policy
	Name string
	// Service this policy applies to
	ServiceName string
	// Load balancing method (round-robin, least-conn, etc.)
	LoadBalancing string
	// Circuit breaking settings
	CircuitBreaking map[string]string
	// Timeout settings
	Timeouts map[string]time.Duration
	// Retry settings
	Retries map[string]interface{}
	// Fault injection for testing
	FaultInjection map[string]interface{}
	// TLS settings
	TLS map[string]string
}

// ServiceMeshManager manages the service mesh
type ServiceMeshManager struct {
	// Mutex for protecting concurrent access
	mu sync.RWMutex
	// Map of service name to service
	services map[string]*Service
	// Map of service name to traffic policy
	policies map[string]*TrafficPolicy
	// Service discovery client
	discoveryClient ServiceDiscoveryClient
	// Proxy sidecar manager
	proxyManager ProxyManager
	// Is the manager initialized
	initialized bool
}

// ServiceDiscoveryClient defines the interface for service discovery
type ServiceDiscoveryClient interface {
	// RegisterService registers a service with the discovery system
	RegisterService(ctx context.Context, service *Service) error
	// DeregisterService deregisters a service from the discovery system
	DeregisterService(ctx context.Context, serviceName string) error
	// DiscoverService finds a service by name
	DiscoverService(ctx context.Context, serviceName string) (*Service, error)
	// ListServices lists all services
	ListServices(ctx context.Context) ([]*Service, error)
	// WatchService watches for changes to a service
	WatchService(ctx context.Context, serviceName string, callback func(*Service))
}

// ProxyManager defines the interface for proxy management
type ProxyManager interface {
	// ConfigureProxy configures a proxy for a service
	ConfigureProxy(ctx context.Context, service *Service, policy *TrafficPolicy) error
	// RemoveProxy removes a proxy configuration for a service
	RemoveProxy(ctx context.Context, serviceName string) error
	// GetProxyStatus gets the status of a proxy
	GetProxyStatus(ctx context.Context, serviceName string) (map[string]interface{}, error)
	// ReloadProxy reloads a proxy configuration
	ReloadProxy(ctx context.Context, serviceName string) error
}

// NewServiceMeshManager creates a new service mesh manager
func NewServiceMeshManager(discoveryClient ServiceDiscoveryClient, proxyManager ProxyManager) *ServiceMeshManager {
	return &ServiceMeshManager{
		services:        make(map[string]*Service),
		policies:        make(map[string]*TrafficPolicy),
		discoveryClient: discoveryClient,
		proxyManager:    proxyManager,
		initialized:     false,
	}
}

// Initialize initializes the service mesh manager
func (m *ServiceMeshManager) Initialize(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		return fmt.Errorf("service mesh manager already initialized")
	}

	// Load existing services from discovery
	services, err := m.discoveryClient.ListServices(ctx)
	if err != nil {
		return fmt.Errorf("failed to list services: %v", err)
	}

	for _, service := range services {
		m.services[service.Name] = service
	}

	m.initialized = true
	return nil
}

// RegisterService registers a new service with the mesh
func (m *ServiceMeshManager) RegisterService(ctx context.Context, service *Service) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	// Validate service
	if service.Name == "" {
		return fmt.Errorf("service name cannot be empty")
	}

	if _, exists := m.services[service.Name]; exists {
		return fmt.Errorf("service %s already exists", service.Name)
	}

	// Register with discovery
	if err := m.discoveryClient.RegisterService(ctx, service); err != nil {
		return fmt.Errorf("failed to register service with discovery: %v", err)
	}

	// Store the service
	m.services[service.Name] = service

	// Configure proxy if a policy exists
	if policy, exists := m.policies[service.Name]; exists {
		if err := m.proxyManager.ConfigureProxy(ctx, service, policy); err != nil {
			return fmt.Errorf("failed to configure proxy: %v", err)
		}
	}

	return nil
}

// DeregisterService removes a service from the mesh
func (m *ServiceMeshManager) DeregisterService(ctx context.Context, serviceName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	if _, exists := m.services[serviceName]; !exists {
		return fmt.Errorf("service %s not found", serviceName)
	}

	// Deregister from discovery
	if err := m.discoveryClient.DeregisterService(ctx, serviceName); err != nil {
		return fmt.Errorf("failed to deregister service from discovery: %v", err)
	}

	// Remove proxy configuration
	if err := m.proxyManager.RemoveProxy(ctx, serviceName); err != nil {
		return fmt.Errorf("failed to remove proxy configuration: %v", err)
	}

	// Remove from local cache
	delete(m.services, serviceName)
	delete(m.policies, serviceName)

	return nil
}

// GetService returns information about a service
func (m *ServiceMeshManager) GetService(ctx context.Context, serviceName string) (*Service, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("service mesh manager not initialized")
	}

	// Try local cache first
	if service, exists := m.services[serviceName]; exists {
		return service, nil
	}

	// Try discovery
	service, err := m.discoveryClient.DiscoverService(ctx, serviceName)
	if err != nil {
		return nil, fmt.Errorf("failed to discover service: %v", err)
	}

	// Update local cache
	m.services[serviceName] = service

	return service, nil
}

// ListServices returns a list of all services in the mesh
func (m *ServiceMeshManager) ListServices(ctx context.Context) ([]*Service, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("service mesh manager not initialized")
	}

	services := make([]*Service, 0, len(m.services))
	for _, service := range m.services {
		services = append(services, service)
	}

	return services, nil
}

// ApplyTrafficPolicy applies a traffic policy to a service
func (m *ServiceMeshManager) ApplyTrafficPolicy(ctx context.Context, policy *TrafficPolicy) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	// Validate policy
	if policy.ServiceName == "" {
		return fmt.Errorf("service name cannot be empty")
	}

	service, exists := m.services[policy.ServiceName]
	if !exists {
		return fmt.Errorf("service %s not found", policy.ServiceName)
	}

	// Configure proxy
	if err := m.proxyManager.ConfigureProxy(ctx, service, policy); err != nil {
		return fmt.Errorf("failed to configure proxy: %v", err)
	}

	// Store the policy
	m.policies[policy.ServiceName] = policy

	return nil
}

// RemoveTrafficPolicy removes a traffic policy from a service
func (m *ServiceMeshManager) RemoveTrafficPolicy(ctx context.Context, serviceName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	if _, exists := m.policies[serviceName]; !exists {
		return fmt.Errorf("no policy found for service %s", serviceName)
	}

	// Remove proxy configuration
	if err := m.proxyManager.RemoveProxy(ctx, serviceName); err != nil {
		return fmt.Errorf("failed to remove proxy configuration: %v", err)
	}

	// Remove policy
	delete(m.policies, serviceName)

	return nil
}

// GetTrafficPolicy returns the traffic policy for a service
func (m *ServiceMeshManager) GetTrafficPolicy(ctx context.Context, serviceName string) (*TrafficPolicy, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("service mesh manager not initialized")
	}

	policy, exists := m.policies[serviceName]
	if !exists {
		return nil, fmt.Errorf("no policy found for service %s", serviceName)
	}

	return policy, nil
}

// AddServiceEndpoint adds an endpoint to a service
func (m *ServiceMeshManager) AddServiceEndpoint(ctx context.Context, serviceName string, endpoint *ServiceEndpoint) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	service, exists := m.services[serviceName]
	if !exists {
		return fmt.Errorf("service %s not found", serviceName)
	}

	// Check for duplicate endpoint ID
	for _, ep := range service.Endpoints {
		if ep.ID == endpoint.ID {
			return fmt.Errorf("endpoint with ID %s already exists", endpoint.ID)
		}
	}

	// Add endpoint
	service.Endpoints = append(service.Endpoints, endpoint)

	// Update service in discovery
	if err := m.discoveryClient.RegisterService(ctx, service); err != nil {
		// Rollback the change
		service.Endpoints = service.Endpoints[:len(service.Endpoints)-1]
		return fmt.Errorf("failed to update service in discovery: %v", err)
	}

	// Reconfigure proxy if necessary
	if policy, exists := m.policies[serviceName]; exists {
		if err := m.proxyManager.ConfigureProxy(ctx, service, policy); err != nil {
			return fmt.Errorf("failed to reconfigure proxy: %v", err)
		}
	}

	return nil
}

// RemoveServiceEndpoint removes an endpoint from a service
func (m *ServiceMeshManager) RemoveServiceEndpoint(ctx context.Context, serviceName, endpointID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	service, exists := m.services[serviceName]
	if !exists {
		return fmt.Errorf("service %s not found", serviceName)
	}

	// Find and remove the endpoint
	found := false
	var newEndpoints []*ServiceEndpoint
	for _, ep := range service.Endpoints {
		if ep.ID != endpointID {
			newEndpoints = append(newEndpoints, ep)
		} else {
			found = true
		}
	}

	if !found {
		return fmt.Errorf("endpoint %s not found in service %s", endpointID, serviceName)
	}

	// Update endpoints
	service.Endpoints = newEndpoints

	// Update service in discovery
	if err := m.discoveryClient.RegisterService(ctx, service); err != nil {
		return fmt.Errorf("failed to update service in discovery: %v", err)
	}

	// Reconfigure proxy if necessary
	if policy, exists := m.policies[serviceName]; exists {
		if err := m.proxyManager.ConfigureProxy(ctx, service, policy); err != nil {
			return fmt.Errorf("failed to reconfigure proxy: %v", err)
		}
	}

	return nil
}

// Shutdown shuts down the service mesh manager
func (m *ServiceMeshManager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return nil
	}

	// Nothing specific to clean up in this implementation
	m.initialized = false
	return nil
}

// EnableMutualTLS enables mutual TLS for a service
func (m *ServiceMeshManager) EnableMutualTLS(ctx context.Context, serviceName, certPath, keyPath, caPath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.initialized {
		return fmt.Errorf("service mesh manager not initialized")
	}

	service, exists := m.services[serviceName]
	if !exists {
		return fmt.Errorf("service %s not found", serviceName)
	}

	policy, exists := m.policies[serviceName]
	if !exists {
		policy = &TrafficPolicy{
			Name:        serviceName + "-policy",
			ServiceName: serviceName,
			TLS:         make(map[string]string),
		}
	}

	// Update TLS settings
	policy.TLS["enabled"] = "true"
	policy.TLS["cert_path"] = certPath
	policy.TLS["key_path"] = keyPath
	policy.TLS["ca_path"] = caPath

	// Apply the policy
	if err := m.proxyManager.ConfigureProxy(ctx, service, policy); err != nil {
		return fmt.Errorf("failed to configure proxy with mTLS: %v", err)
	}

	// Store the policy
	m.policies[serviceName] = policy

	return nil
}
