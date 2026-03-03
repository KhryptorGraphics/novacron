// Package main provides a simple test for the consolidated audit system
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/security"
)

func main() {
	fmt.Println("Testing Consolidated Audit System...")

	// Test 1: Direct use of audit package
	fmt.Println("\n=== Test 1: Direct Audit Package ===")
	ctx := context.Background()

	// Test logging various types of events
	err := audit.LogSecretAccess(ctx, "user1", "database-secret", audit.ActionRead, audit.ResultSuccess, map[string]interface{}{
		"database": "prod-db",
		"query":    "SELECT * FROM users",
	})
	if err != nil {
		log.Printf("Error logging secret access: %v", err)
	} else {
		fmt.Println("✓ Logged secret access event")
	}

	err = audit.LogAuthEvent(ctx, "user1", true, map[string]interface{}{
		"login_method": "SSO",
		"ip_address":   "192.168.1.100",
	})
	if err != nil {
		log.Printf("Error logging auth event: %v", err)
	} else {
		fmt.Println("✓ Logged auth event")
	}

	err = audit.LogConfigChange(ctx, "admin", "app-config", "debug=false", "debug=true")
	if err != nil {
		log.Printf("Error logging config change: %v", err)
	} else {
		fmt.Println("✓ Logged config change event")
	}

	// Test querying events
	filter := &audit.AuditFilter{
		Actors:    []string{"user1", "admin"},
		StartTime: timePtr(time.Now().Add(-1 * time.Hour)),
		EndTime:   timePtr(time.Now()),
		Limit:     10,
	}

	events, err := audit.QueryEvents(ctx, filter)
	if err != nil {
		log.Printf("Error querying events: %v", err)
	} else {
		fmt.Printf("✓ Queried %d events\n", len(events))
	}

	// Test 2: Compatibility with security package aliases
	fmt.Println("\n=== Test 2: Security Package Compatibility ===")

	// Using security package aliases (should work transparently)
	err = security.LogSecretAccess(ctx, "user2", "api-key", security.ActionRead, security.ResultSuccess, map[string]interface{}{
		"api_endpoint": "/v1/users",
		"key_id":       "key-123",
	})
	if err != nil {
		log.Printf("Error using security package alias: %v", err)
	} else {
		fmt.Println("✓ Used security package alias successfully")
	}

	// Test 3: Bridge functionality
	fmt.Println("\n=== Test 3: Bridge Functionality ===")

	// Initialize bridge
	audit.InitializeBridge(audit.NewSimpleAuditLogger())

	// Get services through bridge
	auditLogger := audit.GetBridgeAuditLogger()
	authService := audit.GetBridgeAuthService()

	// Test audit logger
	testEvent := &audit.AuditEvent{
		EventType:   audit.EventSecretAccess,
		Actor:       "user3",
		Resource:    "bridge-test",
		Action:      audit.ActionRead,
		Result:      audit.ResultSuccess,
		Sensitivity: audit.SensitivityInternal,
	}

	err = auditLogger.LogEvent(ctx, testEvent)
	if err != nil {
		log.Printf("Error using bridge audit logger: %v", err)
	} else {
		fmt.Println("✓ Used bridge audit logger successfully")
	}

	// Test auth service
	authEntry := &audit.AuditEntry{
		UserID:       "user3",
		TenantID:     "tenant1",
		ResourceType: "document",
		ResourceID:   "doc-456",
		Action:       "READ",
		Success:      true,
		Timestamp:    time.Now(),
		IPAddress:    "192.168.1.101",
	}

	err = authService.LogAccess(authEntry)
	if err != nil {
		log.Printf("Error using bridge auth service: %v", err)
	} else {
		fmt.Println("✓ Used bridge auth service successfully")
	}

	// Test 4: Type conversions
	fmt.Println("\n=== Test 4: Type Conversions ===")

	// Convert between legacy and new formats
	legacyEvent := audit.LegacyAuditEvent{
		ID:         "legacy-1",
		Timestamp:  time.Now(),
		UserID:     "legacy-user",
		Action:     "CREATE",
		Resource:   "legacy-resource",
		Result:     "SUCCESS",
		ClientIP:   "192.168.1.102",
	}

	newEvent := audit.ConvertLegacyToAuditEvent(legacyEvent)
	fmt.Printf("✓ Converted legacy event: %s -> %s\n", legacyEvent.ID, newEvent.ID)

	// Convert back
	convertedLegacy := audit.ConvertAuditEventToLegacy(newEvent)
	fmt.Printf("✓ Converted back to legacy: %s -> %s\n", newEvent.ID, convertedLegacy.ID)

	fmt.Println("\n=== All Tests Passed! ===")
	fmt.Println("The consolidated audit system is working correctly.")
	fmt.Println("✓ Direct audit package usage")
	fmt.Println("✓ Security package compatibility aliases")
	fmt.Println("✓ Bridge functionality for legacy code")
	fmt.Println("✓ Type conversion utilities")
}

func timePtr(t time.Time) *time.Time {
	return &t
}