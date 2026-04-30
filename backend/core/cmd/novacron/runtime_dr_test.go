package main

import (
	"context"
	"net/http"
	"testing"
)

func TestRuntimeDRStatusEndpointRequiresInitializedRuntime(t *testing.T) {
	t.Parallel()

	router := newRuntimeRouter(defaultRuntimeConfig("node-a", t.TempDir()), nil, nil, nil, nil, nil, nil, nil, nil, nil, nil)

	response := getRuntimeMobilityJSON[map[string]string](t, router, "/internal/runtime/v1/dr/status", http.StatusServiceUnavailable)
	if response["error"] == "" {
		t.Fatalf("expected DR status endpoint to report unavailable runtime, got %#v", response)
	}
}

func TestInitializeAPIStartsRuntimeDRWhenBackupServiceEnabled(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.EnabledServices = []string{"api", "backup"}
	config.Auth.Enabled = false
	config.Services.AuthMode = "disabled"

	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	report := getJSONResponse[runtimeServiceReport](t, apiServer, "/internal/runtime/v1/services")
	serviceStates := make(map[string]runtimeServiceStatus, len(report.Services))
	for _, service := range report.Services {
		serviceStates[service.Name] = service
	}
	if got := serviceStates["backup"].State; got != runtimeServiceStateRunning {
		t.Fatalf("backup state = %q, want %q", got, runtimeServiceStateRunning)
	}

	status := getJSONResponse[runtimeDRStatusResponse](t, apiServer, "/internal/runtime/v1/dr/status")
	if !status.Enabled {
		t.Fatal("expected DR runtime to be enabled")
	}
	if got, want := status.State, "Normal"; got != want {
		t.Fatalf("DR state = %q, want %q", got, want)
	}
	if status.HealthScore <= 0 {
		t.Fatalf("health score = %f, want positive", status.HealthScore)
	}
}
