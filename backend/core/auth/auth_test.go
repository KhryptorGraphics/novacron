package auth

import (
	"testing"
	"time"
)

func TestAuthServiceIntegration(t *testing.T) {
	// Create services
	users := NewUserMemoryStore()
	roles := NewRoleMemoryStore()
	tenants := NewTenantMemoryStore()
	auditLog := NewInMemoryAuditLogService()

	// Create auth service with default config
	auth := NewAuthService(DefaultAuthConfiguration(), users, roles, tenants, auditLog)

	// Test tenant creation
	tenant := NewTenant("test-tenant", "Test Tenant", "Test tenant for integration test")
	tenant.Status = TenantStatusActive
	err := auth.CreateTenant(tenant)
	if err != nil {
		t.Fatalf("Failed to create tenant: %v", err)
	}

	// Test user creation
	user := NewUser("testuser", "test@example.com", "test-tenant")
	user.Status = UserStatusActive
	err = auth.CreateUser(user, "Password@123")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}

	// Test login with invalid credentials
	_, err = auth.Login("testuser", "wrongpassword")
	if err == nil {
		t.Fatal("Login succeeded with wrong password")
	}

	// Test login with valid credentials
	session, err := auth.Login("testuser", "Password@123")
	if err != nil {
		t.Fatalf("Login failed: %v", err)
	}
	if session == nil {
		t.Fatal("Session is nil")
	}
	if session.UserID != "testuser" {
		t.Fatalf("Session user ID mismatch: expected 'testuser', got '%s'", session.UserID)
	}

	// Test session validation
	validatedSession, err := auth.ValidateSession(session.ID, session.Token)
	if err != nil {
		t.Fatalf("Session validation failed: %v", err)
	}
	if validatedSession == nil {
		t.Fatal("Validated session is nil")
	}

	// Test session validation with invalid token
	_, err = auth.ValidateSession(session.ID, "invalid-token")
	if err == nil {
		t.Fatal("Session validation succeeded with invalid token")
	}

	// Test session refresh
	refreshedSession, err := auth.RefreshSession(session.ID, session.Token)
	if err != nil {
		t.Fatalf("Session refresh failed: %v", err)
	}
	if refreshedSession == nil {
		t.Fatal("Refreshed session is nil")
	}

	// Test role creation
	role := NewRole("test-role", "Test Role", "Test role for integration test", "test-tenant")
	role.Permissions = []Permission{
		{
			Resource: "vm",
			Action:   "read",
			Effect:   "allow",
		},
	}
	err = auth.CreateRole(role)
	if err != nil {
		t.Fatalf("Failed to create role: %v", err)
	}

	// Add role to user
	err = users.AddRole("testuser", "test-role")
	if err != nil {
		t.Fatalf("Failed to add role to user: %v", err)
	}

	// Test permission check with a permission the user should have
	hasPermission, err := auth.HasPermission("testuser", "vm", "read")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}
	if !hasPermission {
		t.Fatal("User should have 'vm:read' permission")
	}

	// Test permission check with a permission the user should not have
	hasPermission, err = auth.HasPermission("testuser", "vm", "delete")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}
	if hasPermission {
		t.Fatal("User should not have 'vm:delete' permission")
	}

	// Test logout
	err = auth.Logout(session.ID)
	if err != nil {
		t.Fatalf("Logout failed: %v", err)
	}

	// Test session validation after logout (should fail)
	_, err = auth.ValidateSession(session.ID, session.Token)
	if err == nil {
		t.Fatal("Session validation succeeded after logout")
	}
}

func TestAuthServicePasswordValidation(t *testing.T) {
	// Create services
	users := NewUserMemoryStore()
	roles := NewRoleMemoryStore()
	tenants := NewTenantMemoryStore()
	auditLog := NewInMemoryAuditLogService()

	// Create auth service with default config
	config := DefaultAuthConfiguration()
	auth := NewAuthService(config, users, roles, tenants, auditLog)

	// Create a tenant for testing
	tenant := NewTenant("test-tenant", "Test Tenant", "Test tenant for password validation")
	tenant.Status = TenantStatusActive
	err := auth.CreateTenant(tenant)
	if err != nil {
		t.Fatalf("Failed to create tenant: %v", err)
	}

	// Test creating a user with a too short password
	user := NewUser("shortpw", "short@example.com", "test-tenant")
	err = auth.CreateUser(user, "Short1!")
	if err == nil {
		t.Fatal("User creation succeeded with too short password")
	}

	// Test creating a user with a password missing uppercase
	user = NewUser("nouppercase", "nouppercase@example.com", "test-tenant")
	err = auth.CreateUser(user, "password123!")
	if err == nil {
		t.Fatal("User creation succeeded with password missing uppercase")
	}

	// Test creating a user with a password missing lowercase
	user = NewUser("nolowercase", "nolowercase@example.com", "test-tenant")
	err = auth.CreateUser(user, "PASSWORD123!")
	if err == nil {
		t.Fatal("User creation succeeded with password missing lowercase")
	}

	// Test creating a user with a password missing numbers
	user = NewUser("nonumbers", "nonumbers@example.com", "test-tenant")
	err = auth.CreateUser(user, "Password!")
	if err == nil {
		t.Fatal("User creation succeeded with password missing numbers")
	}

	// Test creating a user with a password missing special characters
	user = NewUser("nospecial", "nospecial@example.com", "test-tenant")
	err = auth.CreateUser(user, "Password123")
	if err == nil {
		t.Fatal("User creation succeeded with password missing special characters")
	}

	// Test creating a user with a valid password
	user = NewUser("validpw", "validpw@example.com", "test-tenant")
	err = auth.CreateUser(user, "ValidPassword123!")
	if err != nil {
		t.Fatalf("User creation failed with valid password: %v", err)
	}
}

func TestTenantResourceQuotas(t *testing.T) {
	// Create tenant service
	tenants := NewTenantMemoryStore()

	// Create a tenant
	tenant := NewTenant("quota-tenant", "Quota Tenant", "Tenant for quota testing")
	tenant.Status = TenantStatusActive

	// Set custom quotas
	tenant.ResourceQuotas["vm.count"] = 5
	tenant.ResourceQuotas["vm.cpu"] = 10

	// Create the tenant
	err := tenants.Create(tenant)
	if err != nil {
		t.Fatalf("Failed to create tenant: %v", err)
	}

	// Get quota
	quota, err := tenants.GetResourceQuota("quota-tenant", "vm.count")
	if err != nil {
		t.Fatalf("Failed to get resource quota: %v", err)
	}
	if quota != 5 {
		t.Fatalf("Resource quota mismatch: expected 5, got %d", quota)
	}

	// Set quota
	err = tenants.SetResourceQuota("quota-tenant", "vm.count", 20)
	if err != nil {
		t.Fatalf("Failed to set resource quota: %v", err)
	}

	// Verify updated quota
	quota, err = tenants.GetResourceQuota("quota-tenant", "vm.count")
	if err != nil {
		t.Fatalf("Failed to get updated resource quota: %v", err)
	}
	if quota != 20 {
		t.Fatalf("Updated resource quota mismatch: expected 20, got %d", quota)
	}

	// Get all quotas
	quotas, err := tenants.GetResourceQuotas("quota-tenant")
	if err != nil {
		t.Fatalf("Failed to get all resource quotas: %v", err)
	}
	if quotas["vm.count"] != 20 {
		t.Fatalf("Resource quota in map mismatch: expected 20, got %d", quotas["vm.count"])
	}
	if quotas["vm.cpu"] != 10 {
		t.Fatalf("Resource quota in map mismatch: expected 10, got %d", quotas["vm.cpu"])
	}
}

func TestRolePermissions(t *testing.T) {
	// Create role service
	roles := NewRoleMemoryStore()

	// Create a custom role
	role := NewRole("custom-role", "Custom Role", "Role for permission testing", "test-tenant")

	// Add permissions
	role.Permissions = []Permission{
		{
			Resource: "vm",
			Action:   "read",
			Effect:   "allow",
		},
		{
			Resource: "storage",
			Action:   "read",
			Effect:   "allow",
		},
	}

	// Create the role
	err := roles.Create(role)
	if err != nil {
		t.Fatalf("Failed to create role: %v", err)
	}

	// Check permissions
	hasPermission, err := roles.HasPermission("custom-role", "vm", "read")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}
	if !hasPermission {
		t.Fatal("Role should have 'vm:read' permission")
	}

	hasPermission, err = roles.HasPermission("custom-role", "vm", "write")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}
	if hasPermission {
		t.Fatal("Role should not have 'vm:write' permission")
	}

	// Add a permission
	err = roles.AddPermission("custom-role", Permission{
		Resource: "vm",
		Action:   "write",
		Effect:   "allow",
	})
	if err != nil {
		t.Fatalf("Failed to add permission: %v", err)
	}

	// Check the new permission
	hasPermission, err = roles.HasPermission("custom-role", "vm", "write")
	if err != nil {
		t.Fatalf("Permission check failed after addition: %v", err)
	}
	if !hasPermission {
		t.Fatal("Role should now have 'vm:write' permission")
	}

	// Remove a permission
	err = roles.RemovePermission("custom-role", "vm", "write")
	if err != nil {
		t.Fatalf("Failed to remove permission: %v", err)
	}

	// Check the removed permission
	hasPermission, err = roles.HasPermission("custom-role", "vm", "write")
	if err != nil {
		t.Fatalf("Permission check failed after removal: %v", err)
	}
	if hasPermission {
		t.Fatal("Role should no longer have 'vm:write' permission")
	}
}

func TestSystemRoles(t *testing.T) {
	// Create role service
	roles := NewRoleMemoryStore()

	// Check system roles exist
	role, err := roles.Get("admin")
	if err != nil {
		t.Fatalf("Failed to get admin role: %v", err)
	}
	if !role.IsSystem {
		t.Fatal("Admin role should be a system role")
	}

	// Check admin has wildcard permission
	hasPermission, err := roles.HasPermission("admin", "anything", "anything")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}
	if !hasPermission {
		t.Fatal("Admin role should have wildcard permission")
	}

	// Try to modify a system role (should fail)
	err = roles.AddPermission("admin", Permission{
		Resource: "new-resource",
		Action:   "new-action",
		Effect:   "allow",
	})
	if err == nil {
		t.Fatal("Modifying system role should fail")
	}

	// Try to delete a system role (should fail)
	err = roles.Delete("admin")
	if err == nil {
		t.Fatal("Deleting system role should fail")
	}
}

func TestAuditLogging(t *testing.T) {
	// Create audit log service
	auditLog := NewInMemoryAuditLogService()

	// Log an entry
	entry := AuditEntry{
		Action:      string(AuditActionLogin),
		Resource:    "user",
		ResourceID:  "testuser",
		Description: "Test login",
		UserID:      "testuser",
		TenantID:    "test-tenant",
		Timestamp:   time.Now(),
	}

	err := auditLog.Log(entry)
	if err != nil {
		t.Fatalf("Failed to log audit entry: %v", err)
	}

	// Search for the entry
	results, err := auditLog.Search(map[string]interface{}{
		"userId": "testuser",
	}, 10, 0)
	if err != nil {
		t.Fatalf("Failed to search audit log: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Expected 1 audit entry, got %d", len(results))
	}
	if results[0].Action != string(AuditActionLogin) {
		t.Fatalf("Audit entry action mismatch: expected '%s', got '%s'", AuditActionLogin, results[0].Action)
	}

	// Count entries
	count, err := auditLog.Count(map[string]interface{}{
		"userId": "testuser",
	})
	if err != nil {
		t.Fatalf("Failed to count audit entries: %v", err)
	}
	if count != 1 {
		t.Fatalf("Expected count of 1, got %d", count)
	}
}
