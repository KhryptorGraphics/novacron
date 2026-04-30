package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func TestRuntimeInventoryRouterListsVMsFromSQLPersistence(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	nowRows := sqlmock.NewRows([]string{"id", "name", "state", "node_id", "tenant_id", "created_at", "updated_at"}).
		AddRow("vm-1", "alpha", "running", "node-a", "default", time.Unix(0, 0).UTC(), time.Unix(0, 0).UTC())
	mock.ExpectQuery(`SELECT id, name, state, node_id, tenant_id, created_at, updated_at FROM vms ORDER BY created_at DESC`).
		WillReturnRows(nowRows)

	router := newRuntimeRouter(defaultRuntimeConfig("node-a", t.TempDir()), nil, nil, nil, nil, nil, nil, &runtimeInventoryStore{db: db}, nil, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/internal/runtime/v1/vms", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload []map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(payload) != 1 {
		t.Fatalf("expected 1 VM, got %d", len(payload))
	}
	if got := payload[0]["id"]; got != "vm-1" {
		t.Fatalf("vm id = %#v, want vm-1", got)
	}
	if got := payload[0]["status"]; got != "running" {
		t.Fatalf("vm status = %#v, want running", got)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestRuntimeInventoryRouterGetsNetworkFromSQLPersistence(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "type", "subnet", "gateway", "status", "created_at", "updated_at"}).
		AddRow("net-1", "blue", "bridged", "10.0.0.0/24", "10.0.0.1", "active", time.Unix(0, 0).UTC(), time.Unix(0, 0).UTC())
	mock.ExpectQuery(`SELECT id, name, type, subnet, gateway, status, created_at, updated_at\s+FROM networks WHERE id = \$1`).
		WithArgs("net-1").
		WillReturnRows(rows)

	router := newRuntimeRouter(defaultRuntimeConfig("node-a", t.TempDir()), nil, nil, nil, nil, nil, nil, &runtimeInventoryStore{db: db}, nil, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/internal/runtime/v1/networks/net-1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if got := payload["id"]; got != "net-1" {
		t.Fatalf("network id = %#v, want net-1", got)
	}
	if got := payload["gateway"]; got != "10.0.0.1" {
		t.Fatalf("network gateway = %#v, want 10.0.0.1", got)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}
