package auth

import (
	"fmt"
	"sync"
	"testing"
)

// TestAuthServiceConcurrentLoginLogoutValidate is the concrete reproduction
// of "Login sets LastLogin then calls Update, racing another Login on the
// same user" at the full AuthServiceImpl level. Multiple goroutines log into
// the *same* underlying user concurrently; individual Login/ValidateSession/
// HasPermission/Logout errors are tolerated (benign contention), but the
// whole run must be -race clean.
func TestAuthServiceConcurrentLoginLogoutValidate(t *testing.T) {
	users := NewUserMemoryStore()
	roles := NewRoleMemoryStore()
	tenants := NewTenantMemoryStore()
	auditLog := NewInMemoryAuditService()

	authSvc := NewAuthService(DefaultAuthConfiguration(), users, roles, tenants, auditLog)

	tenant := NewTenant("race-tenant", "Race Tenant", "concurrency test tenant")
	tenant.Status = TenantStatusActive
	if err := authSvc.CreateTenant(tenant); err != nil {
		t.Fatalf("CreateTenant: %v", err)
	}

	const userCount = 8
	for i := 0; i < userCount; i++ {
		username := fmt.Sprintf("race-user-%d", i)
		user := NewUser(username, fmt.Sprintf("%s@example.com", username), "race-tenant")
		user.Status = UserStatusActive
		if err := authSvc.CreateUser(user, "Password@123"); err != nil {
			t.Fatalf("CreateUser(%s): %v", username, err)
		}
	}

	const goroutines = 24
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			username := fmt.Sprintf("race-user-%d", g%userCount)

			session, err := authSvc.Login(username, "Password@123")
			if err != nil {
				// Benign contention (e.g. concurrent session-limit eviction
				// racing a validate on an evicted session); not a data race.
				return
			}
			_, _ = authSvc.ValidateSession(session.ID, session.Token)
			_, _ = authSvc.HasPermission(session.UserID, "vm", "read")
			_ = authSvc.Logout(session.ID)
		}(g)
	}
	wg.Wait()
}
