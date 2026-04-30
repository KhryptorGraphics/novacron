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

	backupsResponse := getRuntimeMobilityJSON[map[string]string](t, router, "/internal/runtime/v1/dr/backups", http.StatusServiceUnavailable)
	if backupsResponse["error"] == "" {
		t.Fatalf("expected DR backups endpoint to report unavailable runtime, got %#v", backupsResponse)
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

func TestRuntimeDRBackupsPersistAcrossRestart(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dataDir := t.TempDir()
	config := defaultRuntimeConfig("node-a", dataDir)
	config.Services.EnabledServices = []string{"api", "backup"}
	config.Auth.Enabled = false
	config.Services.AuthMode = "disabled"

	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}

	registered := postJSONResponse[runtimeDRBackupRecord](t, apiServer, "/internal/runtime/v1/dr/backups", runtimeDRBackupRequest{
		BackupID: "backup-persist-1",
		VMID:     "vm-1",
		Type:     "full",
		Location: "runtime-local",
		Metadata: map[string]string{"source": "test"},
	}, http.StatusAccepted)
	if got, want := registered.BackupID, "backup-persist-1"; got != want {
		t.Fatalf("registered backup_id = %q, want %q", got, want)
	}
	if err := apiServer.Shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown returned error: %v", err)
	}

	restarted, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("restart initializeAPI returned error: %v", err)
	}
	defer restarted.Shutdown(context.Background())

	backups := getJSONResponse[[]runtimeDRBackupRecord](t, restarted, "/internal/runtime/v1/dr/backups")
	if len(backups) != 1 {
		t.Fatalf("backup count = %d, want 1: %#v", len(backups), backups)
	}
	if got, want := backups[0].BackupID, "backup-persist-1"; got != want {
		t.Fatalf("persisted backup_id = %q, want %q", got, want)
	}

	status := getJSONResponse[runtimeDRStatusResponse](t, restarted, "/internal/runtime/v1/dr/status")
	if got, want := status.BackupCount, int64(1); got != want {
		t.Fatalf("status backup_count = %d, want %d", got, want)
	}
}
