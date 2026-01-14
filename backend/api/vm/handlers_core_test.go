package vm

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"fmt"
	"github.com/gorilla/mux"
	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

// testRouter builds a mux router with auth+envelope middleware and registers VM routes
func testRouter(t *testing.T, vmManager *corevm.VMManager) *mux.Router {
	t.Helper()
	r := mux.NewRouter()
	auth := middleware.NewAuthMiddleware(nil)
	r.Use(auth.RequireAuth)
	r.Use(middleware.ResponseEnvelopeMiddleware)
	require := func(role string, h http.HandlerFunc) http.Handler { return auth.RequireRole(role)(h) }
	RegisterRoutes(r, vmManager, require)
	return r
}

// seedVMs creates VMs directly through the manager for test setup
func seedVMs(t *testing.T, m *corevm.VMManager, n int) []string {
	t.Helper()
	ids := make([]string, 0, n)
	ctx := context.Background()
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("vm-%02d", i)
		req := corevm.CreateVMRequest{
			Name: name,
			Spec: corevm.VMConfig{Name: name, Command: "/bin/true", MemoryMB: 128, CPUShares: 1},
		}
		v, err := m.CreateVM(ctx, req)
		if err != nil { t.Fatalf("create vm: %v", err) }
		ids = append(ids, v.ID())
	}
	return ids
}

func makeRequest(r *mux.Router, method, path string, role string, body any) *httptest.ResponseRecorder {
	var buf *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		buf = bytes.NewReader(b)
	} else { buf = bytes.NewReader(nil) }
	req := httptest.NewRequest(method, path, buf)
	if body != nil { req.Header.Set("Content-Type", "application/json; charset=utf-8") }
	if role != "" { req.Header.Set("X-Role", role) }
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec
}

// ---------- Tests ----------

func TestListVMs_DefaultsAndEnvelope(t *testing.T) {
	cfg := corevm.VMManagerConfig{ DefaultDriver: corevm.VMTypeKVM, Drivers: map[corevm.VMType]corevm.VMDriverConfigManager{ corevm.VMTypeKVM: {Enabled: true, Config: map[string]any{}}, }, }
	m, _ := corevm.NewVMManager(cfg)
	seedVMs(t, m, 12)
	r := testRouter(t, m)

	rec := makeRequest(r, "GET", "/api/v1/vms", "viewer", nil)
	if rec.Code != http.StatusOK { t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String()) }
	if ct := rec.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" { t.Fatalf("unexpected content-type: %s", ct) }
	if rec.Header().Get("X-Pagination") == "" { t.Fatalf("missing X-Pagination header") }

	var env struct{ Data any; Error any; Pagination any }
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil { t.Fatalf("bad json: %v", err) }
	if env.Error != nil { t.Fatalf("unexpected error: %v", env.Error) }
}

func TestListVMs_CustomParamsAndValidation(t *testing.T) {
	cfg := corevm.VMManagerConfig{ DefaultDriver: corevm.VMTypeKVM, Drivers: map[corevm.VMType]corevm.VMDriverConfigManager{ corevm.VMTypeKVM: {Enabled: true, Config: map[string]any{}}, }, }
	m, _ := corevm.NewVMManager(cfg)
	ids := seedVMs(t, m, 15)
	// Mark some running and rename to include foo
	for i, id := range ids {
		v, _ := m.GetVM(id)
		if i%2 == 0 { v.SetState(corevm.StateRunning) }
		if i%3 == 0 { v.SetName("foo-"+v.Name()) }
	}
	r := testRouter(t, m)

	u := url.URL{ Path: "/api/v1/vms" }
	q := u.Query()
	q.Set("page", "2")
	q.Set("pageSize", "5")
	q.Set("sortBy", "name")
	q.Set("sortDir", "desc")
	q.Set("state", "running")
	q.Set("q", "foo")
	u.RawQuery = q.Encode()
	rec := makeRequest(r, "GET", u.String(), "viewer", nil)
	if rec.Code != http.StatusOK { t.Fatalf("expected 200, got %d", rec.Code) }
	if rec.Header().Get("X-Pagination") == "" { t.Fatalf("missing pagination header") }
}

func TestRBAC_ViewerForbiddenMutations(t *testing.T) {
	cfg := corevm.VMManagerConfig{ DefaultDriver: corevm.VMTypeKVM, Drivers: map[corevm.VMType]corevm.VMDriverConfigManager{ corevm.VMTypeKVM: {Enabled: true, Config: map[string]any{}}, }, }
	m, _ := corevm.NewVMManager(cfg)
	r := testRouter(t, m)

	rec := makeRequest(r, "POST", "/api/v1/vms", "viewer", map[string]any{"name":"x"})
	if rec.Code != http.StatusForbidden { t.Fatalf("expected 403, got %d", rec.Code) }
	var env map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &env)
	if env["error"] == nil { t.Fatalf("expected envelope error") }
}

func TestCreateStartStopAndDeleteAsOperator(t *testing.T) {
	cfg := corevm.VMManagerConfig{ DefaultDriver: corevm.VMTypeKVM, Drivers: map[corevm.VMType]corevm.VMDriverConfigManager{ corevm.VMTypeKVM: {Enabled: true, Config: map[string]any{}}, }, }
	m, _ := corevm.NewVMManager(cfg)
	r := testRouter(t, m)

	// Create
	rec := makeRequest(r, "POST", "/api/v1/vms", "operator", map[string]any{"name":"vm-a","command":"/bin/true"})
	if rec.Code != http.StatusCreated { t.Fatalf("expected 201, got %d: %s", rec.Code, rec.Body.String()) }
	var env map[string]any; _ = json.Unmarshal(rec.Body.Bytes(), &env)
	data := env["data"].(map[string]any)
	id := data["id"].(string)

	// Start
	rec = makeRequest(r, "POST", "/api/v1/vms/"+id+"/start", "operator", nil)
	if rec.Code != http.StatusOK { t.Fatalf("start expected 200, got %d", rec.Code) }
	if rec.Code != http.StatusOK && rec.Code != http.StatusBadRequest && rec.Code != http.StatusConflict {
		t.Fatalf("start expected 200/400/409, got %d", rec.Code)
	}

	// Stop
	rec = makeRequest(r, "POST", "/api/v1/vms/"+id+"/stop", "operator", nil)
	if rec.Code != http.StatusOK { t.Fatalf("stop expected 200, got %d", rec.Code) }
	// Delete
	rec = makeRequest(r, "DELETE", "/api/v1/vms/"+id, "operator", nil)
	if rec.Code != http.StatusOK { t.Fatalf("delete expected 200, got %d", rec.Code) }
}

func TestUpdateVM_PatchValidation(t *testing.T) {
	cfg := corevm.VMManagerConfig{ DefaultDriver: corevm.VMTypeKVM, Drivers: map[corevm.VMType]corevm.VMDriverConfigManager{ corevm.VMTypeKVM: {Enabled: true, Config: map[string]any{}}, }, }
	m, _ := corevm.NewVMManager(cfg)
	ctx := context.Background()
	v, _ := m.CreateVM(ctx, corevm.CreateVMRequest{Name: "p1", Spec: corevm.VMConfig{Name: "p1", Command: "/bin/true"}})
	r := testRouter(t, m)

	// Unsupported field
	rec := makeRequest(r, "PATCH", "/api/v1/vms/"+v.ID(), "operator", map[string]any{"cpu_shares":2})
	if rec.Code != http.StatusBadRequest { t.Fatalf("expected 400, got %d", rec.Code) }
	// No allowed fields
	rec = makeRequest(r, "PATCH", "/api/v1/vms/"+v.ID(), "operator", map[string]any{"memory_mb":128})
	if rec.Code != http.StatusBadRequest { t.Fatalf("expected 400, got %d", rec.Code) }
	// Valid name+tags
	rec = makeRequest(r, "PATCH", "/api/v1/vms/"+v.ID(), "operator", map[string]any{"name":" new-name ", "tags": map[string]string{"a":"b"}})
	if rec.Code != http.StatusOK { t.Fatalf("expected 200, got %d", rec.Code) }
}

