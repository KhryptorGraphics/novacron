package main

import (
	"context"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestRuntimeMobilityPolicyEndpointGatesLiveMigration(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.MigrationMode = vm.MigrationModeLive
	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	response := getJSONResponse[runtimeMobilityPolicyResponse](t, apiServer, "/internal/runtime/v1/mobility/policy")
	if got, want := response.Mode, vm.MigrationModeLive; got != want {
		t.Fatalf("mobility mode = %q, want %q", got, want)
	}
	if response.Policy.LiveMigrationGate.Enabled {
		t.Fatal("live migration gate should remain disabled by default")
	}
	if got, want := response.Policy.DefaultMigrationMode, vm.MigrationModeCheckpoint; got != want {
		t.Fatalf("policy default mode = %q, want gated fallback %q", got, want)
	}
	if containsRuntimeMobilityString(response.Policy.AllowedMigrationModes, vm.MigrationModeLive) {
		t.Fatalf("live migration should not be in allowed modes without gate: %#v", response.Policy.AllowedMigrationModes)
	}
	if !containsRuntimeMobilityString(response.Policy.AllowedMigrationModes, vm.MigrationModeCheckpoint) {
		t.Fatalf("checkpoint should be allowed before live migration: %#v", response.Policy.AllowedMigrationModes)
	}
}

func containsRuntimeMobilityString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
