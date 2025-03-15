package multi_tenant

import (
	"context"
	"fmt"
	"log"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
)

// This example demonstrates how to use the RBAC scheduler with multi-tenancy
// showing how the system integrates authorization with the network-aware scheduler

func RunMultiTenantExample() {
	// Create the required auth components
	roleService := auth.NewRoleMemoryStore()
	userService := auth.NewUserMemoryStore()
	tenantService := createTenantService()
	auditService := auth.NewInMemoryAuditService()

	// Create the auth manager
	authManager := auth.NewAuthManager(
		auth.DefaultAuthManagerConfig(),
		userService,
		roleService,
		tenantService,
		auditService,
	)
	if err := authManager.Start(); err != nil {
		log.Fatalf("Failed to start auth manager: %v", err)
	}
	defer authManager.Stop()

	// Create the scheduler components
	factoryConfig := scheduler.DefaultSchedulerFactoryConfig()
	factoryConfig.SchedulerType = scheduler.SchedulerTypeNetworkAware
	factory := scheduler.NewSchedulerFactory(factoryConfig)

	// Create the network-aware scheduler
	schedulerInstance, err := factory.CreateScheduler()
	if err != nil {
		log.Fatalf("Failed to create scheduler: %v", err)
	}

	networkScheduler, ok := schedulerInstance.(*scheduler.NetworkAwareScheduler)
	if !ok {
		log.Fatalf("Failed to cast to NetworkAwareScheduler")
	}

	// Start the scheduler
	if err := networkScheduler.Start(); err != nil {
		log.Fatalf("Failed to start scheduler: %v", err)
	}
	defer networkScheduler.Stop()

	// Create the RBAC wrapper for the scheduler
	rbacScheduler := scheduler.NewRBACScheduler(networkScheduler, authManager)

	// Set up the network topology and nodes (same as in network_aware_example.go)
	setupMTNetworkTopology(factory.GetNetworkTopology())
	setupMTNodes(networkScheduler)

	// Set up multi-tenancy example with users and tenants
	fmt.Println("=== Setting up multi-tenancy demo ===")
	setUpMultiTenancy(userService, roleService, tenantService)

	// Demonstrate tenant isolation with VM placement
	fmt.Println("\n=== Demonstrating multi-tenant VM placement ===")
	demonstrateMultiTenantPlacement(rbacScheduler)

	// Demonstrate authorization for different operations
	fmt.Println("\n=== Demonstrating role-based access control ===")
	demonstrateRBACOperations(rbacScheduler)
}

// createTenantService creates a tenant service
func createTenantService() auth.TenantService {
	// In a real implementation, this would connect to a database
	// For this example, we'll use a placeholder implementation
	return &dummyTenantService{
		tenants: make(map[string]*auth.Tenant),
	}
}

// dummyTenantService is a minimal implementation of TenantService
type dummyTenantService struct {
	tenants map[string]*auth.Tenant
}

func (s *dummyTenantService) Get(id string) (*auth.Tenant, error) {
	tenant, exists := s.tenants[id]
	if !exists {
		return nil, fmt.Errorf("tenant not found: %s", id)
	}
	return tenant, nil
}

func (s *dummyTenantService) List(filter map[string]interface{}) ([]*auth.Tenant, error) {
	result := make([]*auth.Tenant, 0, len(s.tenants))
	for _, tenant := range s.tenants {
		result = append(result, tenant)
	}
	return result, nil
}

func (s *dummyTenantService) Create(tenant *auth.Tenant) error {
	s.tenants[tenant.ID] = tenant
	return nil
}

func (s *dummyTenantService) Update(tenant *auth.Tenant) error {
	s.tenants[tenant.ID] = tenant
	return nil
}

func (s *dummyTenantService) Delete(id string) error {
	delete(s.tenants, id)
	return nil
}

func (s *dummyTenantService) AddUser(tenantID, userID string) error {
	// In a real implementation, this would update the tenant's user list
	return nil
}

func (s *dummyTenantService) RemoveUser(tenantID, userID string) error {
	// In a real implementation, this would update the tenant's user list
	return nil
}

func (s *dummyTenantService) GetUsers(tenantID string) ([]*auth.User, error) {
	// In a real implementation, this would retrieve the tenant's users
	return []*auth.User{}, nil
}

// setupMTNetworkTopology defines nodes and links in the network topology (same as network_aware_example.go)
func setupMTNetworkTopology(topology *network.NetworkTopology) {
	// Define datacenter locations
	dc1 := network.NetworkLocation{
		Datacenter: "dc-east",
		Zone:       "zone-1",
		Rack:       "rack-1",
	}

	dc2 := network.NetworkLocation{
		Datacenter: "dc-west",
		Zone:       "zone-1",
		Rack:       "rack-1",
	}

	// Add nodes to topology
	nodes := []struct {
		id       string
		nodeType string
		location network.NetworkLocation
	}{
		{"node-1", "hypervisor", dc1},
		{"node-2", "hypervisor", dc1},
		{"node-3", "hypervisor", dc1},
		{"node-4", "hypervisor", dc2},
		{"node-5", "hypervisor", dc2},
	}

	for _, n := range nodes {
		topology.AddNode(&network.NetworkNode{
			ID:       n.id,
			Type:     n.nodeType,
			Location: n.location,
			Attributes: map[string]interface{}{
				"cores":   24,
				"memory":  128,
				"storage": 2048,
			},
		})
	}

	// Add network links (simplified for this example)
	links := []struct {
		source    string
		dest      string
		bandwidth float64
		latency   float64
		linkType  network.LinkType
	}{
		{"node-1", "node-2", 10000, 0.5, network.LinkTypeSameDatacenter},
		{"node-2", "node-3", 10000, 0.5, network.LinkTypeSameDatacenter},
		{"node-4", "node-5", 10000, 0.5, network.LinkTypeSameDatacenter},
		{"node-1", "node-4", 1000, 50, network.LinkTypeInterDatacenter},
	}

	for _, l := range links {
		topology.AddLink(&network.NetworkLink{
			SourceID:      l.source,
			DestinationID: l.dest,
			Type:          l.linkType,
			Bandwidth:     l.bandwidth,
			Latency:       l.latency,
			Utilization:   0.2, // 20% baseline utilization
		})

		// Add reverse link
		topology.AddLink(&network.NetworkLink{
			SourceID:      l.dest,
			DestinationID: l.source,
			Type:          l.linkType,
			Bandwidth:     l.bandwidth,
			Latency:       l.latency,
			Utilization:   0.2,
		})
	}
}

// setupMTNodes registers nodes with the scheduler (same as network_aware_example.go)
func setupMTNodes(s *scheduler.NetworkAwareScheduler) {
	nodes := []struct {
		id        string
		resources map[scheduler.ResourceType]*scheduler.Resource
	}{
		{
			"node-1",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 10.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 32 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 512 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 2000},
			},
		},
		{
			"node-2",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 24.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 64 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 1024 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 5000},
			},
		},
		{
			"node-3",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 48.0, Used: 12.0},
				scheduler.ResourceMemory:  {Capacity: 128 * 1024, Used: 40 * 1024},
				scheduler.ResourceDisk:    {Capacity: 2048 * 1024, Used: 768 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 3000},
			},
		},
		{
			"node-4",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 64.0, Used: 16.0},
				scheduler.ResourceMemory:  {Capacity: 256 * 1024, Used: 64 * 1024},
				scheduler.ResourceDisk:    {Capacity: 4096 * 1024, Used: 1024 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 2500},
			},
		},
		{
			"node-5",
			map[scheduler.ResourceType]*scheduler.Resource{
				scheduler.ResourceCPU:     {Capacity: 64.0, Used: 32.0},
				scheduler.ResourceMemory:  {Capacity: 256 * 1024, Used: 128 * 1024},
				scheduler.ResourceDisk:    {Capacity: 4096 * 1024, Used: 2048 * 1024},
				scheduler.ResourceNetwork: {Capacity: 10000, Used: 5000},
			},
		},
	}

	for _, node := range nodes {
		if err := s.UpdateNodeResources(node.id, node.resources); err != nil {
			log.Fatalf("Failed to update node resources for %s: %v", node.id, err)
		}
	}
}

// setUpMultiTenancy creates users, roles, and tenants for the multi-tenancy example
func setUpMultiTenancy(userService auth.UserService, roleService auth.RoleService, tenantService auth.TenantService) {
	// Create tenants
	tenants := []struct {
		id   string
		name string
	}{
		{"tenant1", "Finance Department"},
		{"tenant2", "Engineering Department"},
		{"tenant3", "Marketing Department"},
	}

	for _, t := range tenants {
		tenant := &auth.Tenant{
			ID:   t.id,
			Name: t.name,
		}
		if err := tenantService.Create(tenant); err != nil {
			log.Fatalf("Failed to create tenant %s: %v", t.id, err)
		}
		fmt.Printf("Created tenant: %s (%s)\n", t.name, t.id)
	}

	// Create roles
	roles := []struct {
		id          string
		name        string
		description string
		tenantID    string
		permissions []auth.Permission
	}{
		{
			"admin-role",
			"Administrator",
			"Full access to all resources",
			"",
			[]auth.Permission{
				{Resource: "*", Action: "*"},
			},
		},
		{
			"vm-operator-role",
			"VM Operator",
			"Can manage VMs",
			"",
			[]auth.Permission{
				{Resource: string(auth.ResourceTypeVM), Action: string(auth.AuthorizationTypeCreate)},
				{Resource: string(auth.ResourceTypeVM), Action: string(auth.AuthorizationTypeRead)},
				{Resource: string(auth.ResourceTypeVM), Action: string(auth.AuthorizationTypeUpdate)},
				{Resource: string(auth.ResourceTypeVM), Action: string(auth.AuthorizationTypeDelete)},
				{Resource: string(auth.ResourceTypeSystem), Action: string(auth.AuthorizationTypeRead)},
			},
		},
		{
			"vm-viewer-role",
			"VM Viewer",
			"Can view VMs but not modify them",
			"",
			[]auth.Permission{
				{Resource: string(auth.ResourceTypeVM), Action: string(auth.AuthorizationTypeRead)},
				{Resource: string(auth.ResourceTypeSystem), Action: string(auth.AuthorizationTypeRead)},
			},
		},
	}

	for _, r := range roles {
		role := &auth.Role{
			ID:          r.id,
			Name:        r.name,
			Description: r.description,
			TenantID:    r.tenantID,
			Permissions: r.permissions,
		}
		if err := roleService.Create(role); err != nil {
			log.Fatalf("Failed to create role %s: %v", r.id, err)
		}
		fmt.Printf("Created role: %s (%s)\n", r.name, r.id)
	}

	// Create users
	users := []struct {
		id       string
		username string
		email    string
		tenantID string
		roles    []string
		password string
	}{
		{
			"admin-user",
			"admin",
			"admin@example.com",
			"",
			[]string{"admin-role"},
			"admin123",
		},
		{
			"operator1",
			"operator1",
			"operator1@example.com",
			"tenant1",
			[]string{"vm-operator-role"},
			"password123",
		},
		{
			"operator2",
			"operator2",
			"operator2@example.com",
			"tenant2",
			[]string{"vm-operator-role"},
			"password123",
		},
		{
			"viewer1",
			"viewer1",
			"viewer1@example.com",
			"tenant1",
			[]string{"vm-viewer-role"},
			"password123",
		},
	}

	for _, u := range users {
		user := auth.NewUser(u.username, u.email, u.tenantID)
		user.ID = u.id
		if err := userService.Create(user, u.password); err != nil {
			log.Fatalf("Failed to create user %s: %v", u.id, err)
		}

		// Add roles to the user
		for _, roleID := range u.roles {
			if err := userService.AddRole(u.id, roleID); err != nil {
				log.Fatalf("Failed to add role %s to user %s: %v", roleID, u.id, err)
			}
		}

		fmt.Printf("Created user: %s (%s) in tenant %s with roles %v\n", u.username, u.id, u.tenantID, u.roles)
	}
}

// demonstrateMultiTenantPlacement shows how VM placement works with tenants
func demonstrateMultiTenantPlacement(rbacScheduler *scheduler.RBACScheduler) {
	// Create a context for the admin user
	adminCtx := context.Background()
	adminCtx = context.WithValue(adminCtx, "auth_user_id", "admin-user")

	// Create a context for tenant1 operator
	tenant1Ctx := context.Background()
	tenant1Ctx = context.WithValue(tenant1Ctx, "auth_user_id", "operator1")
	tenant1Ctx = context.WithValue(tenant1Ctx, "auth_tenant_id", "tenant1")

	// Create a context for tenant2 operator
	tenant2Ctx := context.Background()
	tenant2Ctx = context.WithValue(tenant2Ctx, "auth_user_id", "operator2")
	tenant2Ctx = context.WithValue(tenant2Ctx, "auth_tenant_id", "tenant2")

	// Create a context for tenant1 viewer
	viewerCtx := context.Background()
	viewerCtx = context.WithValue(viewerCtx, "auth_user_id", "viewer1")
	viewerCtx = context.WithValue(viewerCtx, "auth_tenant_id", "tenant1")

	// Tenant1 operator creates a VM
	vm1ID := "tenant1-vm-1"
	requestID, err := rbacScheduler.RequestPlacement(
		tenant1Ctx,
		vm1ID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     2.0,
			"memory":  4096.0,
			"disk":    10240.0,
			"network": 1000.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Tenant1 operator VM creation failed: %v\n", err)
	} else {
		fmt.Printf("Tenant1 operator created VM %s with request ID %s\n", vm1ID, requestID)
	}

	// Tenant2 operator creates a VM
	vm2ID := "tenant2-vm-1"
	requestID, err = rbacScheduler.RequestPlacement(
		tenant2Ctx,
		vm2ID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     2.0,
			"memory":  4096.0,
			"disk":    10240.0,
			"network": 1000.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Tenant2 operator VM creation failed: %v\n", err)
	} else {
		fmt.Printf("Tenant2 operator created VM %s with request ID %s\n", vm2ID, requestID)
	}

	// Tenant1 viewer tries to create a VM (should fail)
	vm3ID := "tenant1-unauthorized-vm"
	_, err = rbacScheduler.RequestPlacement(
		viewerCtx,
		vm3ID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     1.0,
			"memory":  2048.0,
			"disk":    5120.0,
			"network": 500.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Tenant1 viewer VM creation correctly failed: %v\n", err)
	} else {
		fmt.Printf("ERROR: Tenant1 viewer should not be able to create VMs\n")
	}

	// Tenant2 operator tries to create a VM in tenant1 (should fail)
	vm4ID := "tenant1-vm-from-tenant2"
	crossTenantCtx := context.WithValue(tenant2Ctx, "auth_tenant_id", "tenant1")
	_, err = rbacScheduler.RequestPlacement(
		crossTenantCtx,
		vm4ID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     1.0,
			"memory":  2048.0,
			"disk":    5120.0,
			"network": 500.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Cross-tenant VM creation correctly failed: %v\n", err)
	} else {
		fmt.Printf("ERROR: Cross-tenant VM creation should have failed\n")
	}

	// Admin can create VMs in any tenant
	vm5ID := "admin-vm-in-tenant1"
	adminWithTenantCtx := context.WithValue(adminCtx, "auth_tenant_id", "tenant1")
	requestID, err = rbacScheduler.RequestPlacement(
		adminWithTenantCtx,
		vm5ID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     4.0,
			"memory":  8192.0,
			"disk":    20480.0,
			"network": 2000.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Admin VM creation in tenant1 failed: %v\n", err)
	} else {
		fmt.Printf("Admin created VM %s in tenant1 with request ID %s\n", vm5ID, requestID)
	}
}

// demonstrateRBACOperations shows different operations with RBAC
func demonstrateRBACOperations(rbacScheduler *scheduler.RBACScheduler) {
	// Create contexts for different users
	adminCtx := context.Background()
	adminCtx = context.WithValue(adminCtx, "auth_user_id", "admin-user")

	operatorCtx := context.Background()
	operatorCtx = context.WithValue(operatorCtx, "auth_user_id", "operator1")
	operatorCtx = context.WithValue(operatorCtx, "auth_tenant_id", "tenant1")

	viewerCtx := context.Background()
	viewerCtx = context.WithValue(viewerCtx, "auth_user_id", "viewer1")
	viewerCtx = context.WithValue(viewerCtx, "auth_tenant_id", "tenant1")

	// 1. Node resource update (admin only)
	nodeID := "node-1"
	resources := map[scheduler.ResourceType]*scheduler.Resource{
		scheduler.ResourceCPU:     {Capacity: 64.0, Used: 16.0},
		scheduler.ResourceMemory:  {Capacity: 256 * 1024, Used: 64 * 1024},
		scheduler.ResourceDisk:    {Capacity: 4096 * 1024, Used: 1024 * 1024},
		scheduler.ResourceNetwork: {Capacity: 20000, Used: 5000},
	}

	// Admin updating node resources (should succeed)
	err := rbacScheduler.UpdateNodeResources(adminCtx, nodeID, resources)
	if err != nil {
		fmt.Printf("Admin node resource update failed: %v\n", err)
	} else {
		fmt.Printf("Admin successfully updated node %s resources\n", nodeID)
	}

	// Operator updating node resources (should fail)
	err = rbacScheduler.UpdateNodeResources(operatorCtx, nodeID, resources)
	if err != nil {
		fmt.Printf("Operator node resource update correctly failed: %v\n", err)
	} else {
		fmt.Printf("ERROR: Operator should not be able to update node resources\n")
	}

	// 2. Placement result retrieval (operators and viewers)
	// Create a VM first (using the admin)
	vmID := "test-vm-for-viewing"
	requestID, err := rbacScheduler.RequestPlacement(
		adminCtx,
		vmID,
		scheduler.PolicyNetworkAware,
		[]scheduler.PlacementConstraint{},
		map[string]float64{
			"cpu":     2.0,
			"memory":  4096.0,
			"disk":    10240.0,
			"network": 1000.0,
		},
		1,
	)

	if err != nil {
		fmt.Printf("Failed to create test VM: %v\n", err)
	} else {
		fmt.Printf("Created test VM with request ID %s\n", requestID)

		// Admin viewing the placement result
		_, err = rbacScheduler.GetPlacementResult(adminCtx, requestID)
		if err != nil {
			fmt.Printf("Admin placement result retrieval failed: %v\n", err)
		} else {
			fmt.Printf("Admin successfully retrieved placement result\n")
		}

		// Operator viewing the placement result
		_, err = rbacScheduler.GetPlacementResult(operatorCtx, requestID)
		if err != nil {
			fmt.Printf("Operator placement result retrieval failed: %v\n", err)
		} else {
			fmt.Printf("Operator successfully retrieved placement result\n")
		}

		// Viewer viewing the placement result
		_, err = rbacScheduler.GetPlacementResult(viewerCtx, requestID)
		if err != nil {
			fmt.Printf("Viewer placement result retrieval failed: %v\n", err)
		} else {
			fmt.Printf("Viewer successfully retrieved placement result\n")
		}
	}

	// Show migration actions (currently not implemented)
	fmt.Printf("Migration operations are not fully implemented in this demo\n")
}
