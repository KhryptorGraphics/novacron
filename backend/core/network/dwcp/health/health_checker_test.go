package health

import (
	"context"
	"testing"
	"time"
)

func TestNewChecker(t *testing.T) {
	checker := NewChecker(30*time.Second, 5*time.Second, "1.0.0")

	if checker == nil {
		t.Fatal("NewChecker returned nil")
	}

	if checker.interval != 30*time.Second {
		t.Errorf("Expected interval 30s, got %v", checker.interval)
	}

	if checker.timeout != 5*time.Second {
		t.Errorf("Expected timeout 5s, got %v", checker.timeout)
	}

	if checker.version != "1.0.0" {
		t.Errorf("Expected version 1.0.0, got %s", checker.version)
	}
}

func TestRegisterCheck(t *testing.T) {
	checker := NewChecker(30*time.Second, 5*time.Second, "1.0.0")

	checkFunc := func(ctx context.Context) (*ComponentHealth, error) {
		return &ComponentHealth{
			Name:   "test",
			Status: StatusHealthy,
		}, nil
	}

	checker.RegisterCheck("test", checkFunc)

	if len(checker.components) != 1 {
		t.Errorf("Expected 1 component, got %d", len(checker.components))
	}
}

func TestAMSTHealthCheck(t *testing.T) {
	tests := []struct {
		name          string
		activeStreams int
		minStreams    int
		maxStreams    int
		expectedStatus Status
	}{
		{"Healthy", 32, 16, 256, StatusHealthy},
		{"Below minimum", 8, 16, 256, StatusDegraded},
		{"Above maximum", 300, 16, 256, StatusUnhealthy},
		{"No streams", 0, 16, 256, StatusUnhealthy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check := AMSTHealthCheck(tt.activeStreams, tt.minStreams, tt.maxStreams)
			health, err := check(context.Background())

			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}

			if health.Status != tt.expectedStatus {
				t.Errorf("Expected status %s, got %s", tt.expectedStatus, health.Status)
			}
		})
	}
}

func TestHDEHealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		enabled        bool
		avgRatio       float64
		minRatio       float64
		baselineCount  int
		expectedStatus Status
	}{
		{"Healthy", true, 5.0, 2.0, 10, StatusHealthy},
		{"Disabled", false, 0, 2.0, 0, StatusDegraded},
		{"Low ratio", true, 1.5, 2.0, 10, StatusDegraded},
		{"No baselines", true, 5.0, 2.0, 0, StatusDegraded},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check := HDEHealthCheck(tt.enabled, tt.avgRatio, tt.minRatio, tt.baselineCount)
			health, err := check(context.Background())

			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}

			if health.Status != tt.expectedStatus {
				t.Errorf("Expected status %s, got %s", tt.expectedStatus, health.Status)
			}
		})
	}
}

func TestErrorRateHealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		errorCount     int
		totalRequests  int
		maxErrorRate   float64
		expectedStatus Status
	}{
		{"Healthy", 1, 1000, 5.0, StatusHealthy},
		{"Degraded", 30, 1000, 5.0, StatusDegraded},
		{"Unhealthy", 100, 1000, 5.0, StatusUnhealthy},
		{"No requests", 0, 0, 5.0, StatusHealthy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check := ErrorRateHealthCheck(tt.errorCount, tt.totalRequests, tt.maxErrorRate)
			health, err := check(context.Background())

			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}

			if health.Status != tt.expectedStatus {
				t.Errorf("Expected status %s, got %s", tt.expectedStatus, health.Status)
			}
		})
	}
}

func TestBaselineSyncHealthCheck(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name           string
		syncedNodes    int
		totalNodes     int
		lastSyncTime   time.Time
		maxSyncAge     time.Duration
		expectedStatus Status
	}{
		{"Fully synced", 10, 10, now.Add(-1 * time.Minute), 5 * time.Minute, StatusHealthy},
		{"Partially synced", 5, 10, now.Add(-1 * time.Minute), 5 * time.Minute, StatusDegraded},
		{"Old sync", 10, 10, now.Add(-10 * time.Minute), 5 * time.Minute, StatusDegraded},
		{"Few nodes synced", 2, 10, now.Add(-1 * time.Minute), 5 * time.Minute, StatusUnhealthy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check := BaselineSyncHealthCheck(tt.syncedNodes, tt.totalNodes, tt.lastSyncTime, tt.maxSyncAge)
			health, err := check(context.Background())

			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}

			if health.Status != tt.expectedStatus {
				t.Errorf("Expected status %s, got %s", tt.expectedStatus, health.Status)
			}
		})
	}
}

func TestGetHealth(t *testing.T) {
	checker := NewChecker(30*time.Second, 5*time.Second, "1.0.0")

	// Register some checks
	checker.RegisterCheck("amst", AMSTHealthCheck(32, 16, 256))
	checker.RegisterCheck("hde", HDEHealthCheck(true, 5.0, 2.0, 10))

	// Run checks
	checker.runChecks(context.Background())

	// Get health
	health := checker.GetHealth()

	if health == nil {
		t.Fatal("GetHealth returned nil")
	}

	if health.Status != StatusHealthy {
		t.Errorf("Expected healthy status, got %s", health.Status)
	}

	if len(health.Components) != 2 {
		t.Errorf("Expected 2 components, got %d", len(health.Components))
	}

	if health.Version != "1.0.0" {
		t.Errorf("Expected version 1.0.0, got %s", health.Version)
	}
}

func TestIsHealthy(t *testing.T) {
	checker := NewChecker(30*time.Second, 5*time.Second, "1.0.0")

	// Register healthy checks
	checker.RegisterCheck("amst", AMSTHealthCheck(32, 16, 256))
	checker.RegisterCheck("hde", HDEHealthCheck(true, 5.0, 2.0, 10))

	checker.runChecks(context.Background())

	if !checker.IsHealthy() {
		t.Error("Expected IsHealthy to return true")
	}

	// Register unhealthy check
	checker.RegisterCheck("unhealthy", AMSTHealthCheck(0, 16, 256))
	checker.runChecks(context.Background())

	if checker.IsHealthy() {
		t.Error("Expected IsHealthy to return false")
	}
}
