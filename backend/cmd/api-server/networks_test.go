//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/network/provision"
	"github.com/lib/pq"
)

const (
	testNetworkID = "0f3c2a9e-51d4-4b7a-9c2e-7d1f00aa1234"
	testBridge    = "ncbr-0f3c2a9e51"
	testAdminID   = "3d0f4b2a-1111-4c2e-9a3b-000000000001"
)

var networkCols = []string{"id", "name", "bridge", "cidr", "gateway", "vlan_id", "mtu", "created_by", "created_at", "updated_at", "vm_count"}

// newNetworksRouter mounts the network routes behind a stand-in for
// requireAuth that takes role/org/user from request headers.
func newNetworksRouter(t *testing.T) (*mux.Router, sqlmock.Sqlmock, *provision.Fake) {
	t.Helper()
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	fake := &provision.Fake{}
	router := mux.NewRouter()
	api := router.PathPrefix("/api").Subrouter()
	api.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := context.WithValue(r.Context(), "role", r.Header.Get("X-Test-Role"))
			ctx = context.WithValue(ctx, "organization_id", r.Header.Get("X-Test-Org"))
			ctx = context.WithValue(ctx, "user_id", r.Header.Get("X-Test-User"))
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	})
	registerNetworkRoutes(api, db, fake)
	return router, mock, fake
}

func doNetworks(t *testing.T, router *mux.Router, role, method, path, body string) (*httptest.ResponseRecorder, map[string]interface{}) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Test-Role", role)
	req.Header.Set("X-Test-User", testAdminID)
	if role == "user" {
		req.Header.Set("X-Test-Org", "7a1f9c40-2222-4d3e-8b4c-000000000002")
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	var out map[string]interface{}
	if strings.HasPrefix(strings.TrimSpace(rec.Body.String()), "{") {
		_ = json.Unmarshal(rec.Body.Bytes(), &out)
	}
	return rec, out
}

func TestNetworksCreateProvisionsInsideTransaction(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	now := time.Now().UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(regexp.QuoteMeta("INSERT INTO networks")).
		WithArgs(sqlmock.AnyArg(), "prod", sqlmock.AnyArg(), "10.20.0.0/24", "10.20.0.1", 100, 9000, testAdminID).
		WillReturnRows(sqlmock.NewRows([]string{"created_by", "created_at", "updated_at"}).AddRow(testAdminID, now, now))
	mock.ExpectCommit()

	rec, out := doNetworks(t, router, "admin", http.MethodPost, "/api/networks",
		`{"name":"prod","cidr":"10.20.0.0/24","gateway":"10.20.0.1","vlan_id":100,"mtu":9000}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	id, _ := out["id"].(string)
	bridge, _ := out["bridge"].(string)
	if !strings.HasPrefix(bridge, "ncbr-") || out["name"] != "prod" || out["cidr"] != "10.20.0.0/24" || out["gateway"] != "10.20.0.1" || out["vlan_id"] != float64(100) || out["mtu"] != float64(9000) || out["created_by"] != testAdminID || out["vm_count"] != float64(0) {
		t.Fatalf("response %v", out)
	}
	spec, ok := fake.Provisioned(bridge)
	if !ok || spec.NetworkID != id || spec.VLANID != 100 || spec.MTU != 9000 || spec.Gateway.String() != "10.20.0.1" {
		t.Fatalf("provisioned %+v (ok=%v) for id %s", spec, ok, id)
	}
	if calls := fake.Calls(); len(calls) != 1 || calls[0] != "ensure "+bridge {
		t.Fatalf("provisioner calls %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksCreateRollsBackWhenProvisionFails(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	fake.FailEnsure(errors.New("ioctl: operation not permitted"))
	now := time.Now().UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(regexp.QuoteMeta("INSERT INTO networks")).
		WillReturnRows(sqlmock.NewRows([]string{"created_by", "created_at", "updated_at"}).AddRow(nil, now, now))
	mock.ExpectRollback()

	rec, out := doNetworks(t, router, "admin", http.MethodPost, "/api/networks", `{"name":"prod","cidr":"10.20.0.0/24"}`)
	if rec.Code != http.StatusInternalServerError || !strings.Contains(out["error"].(string), "operation not permitted") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	// The failed Ensure is compensated by a Remove so no half-built bridge
	// survives, and the row never committed.
	if calls := fake.Calls(); len(calls) != 2 || !strings.HasPrefix(calls[0], "ensure ") || !strings.HasPrefix(calls[1], "remove ") {
		t.Fatalf("provisioner calls %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksCreateValidation(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	cases := []struct {
		name, body, want string
	}{
		{"bad name", `{"name":"-prod","cidr":"10.20.0.0/24"}`, "name must be"},
		{"long name", `{"name":"` + strings.Repeat("a", 64) + `","cidr":"10.20.0.0/24"}`, "name must be"},
		{"bad cidr", `{"name":"prod","cidr":"10.20.0.0"}`, "not a valid network prefix"},
		{"host bits", `{"name":"prod","cidr":"10.20.0.7/24"}`, "host bits"},
		{"gateway outside", `{"name":"prod","cidr":"10.20.0.0/24","gateway":"10.21.0.1"}`, "not inside"},
		{"vlan 0", `{"name":"prod","cidr":"10.20.0.0/24","vlan_id":0}`, "vlan_id must be between 1 and 4094"},
		{"vlan 4095", `{"name":"prod","cidr":"10.20.0.0/24","vlan_id":4095}`, "vlan_id must be between 1 and 4094"},
		{"mtu 575", `{"name":"prod","cidr":"10.20.0.0/24","mtu":575}`, "mtu 575 must be between 576 and 9000"},
		{"mtu 9001", `{"name":"prod","cidr":"10.20.0.0/24","mtu":9001}`, "mtu 9001 must be between 576 and 9000"},
		{"unknown field", `{"name":"prod","cidr":"10.20.0.0/24","subnet":"x"}`, "invalid request body"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec, out := doNetworks(t, router, "admin", http.MethodPost, "/api/networks", tc.body)
			if rec.Code != http.StatusBadRequest || !strings.Contains(out["error"].(string), tc.want) {
				t.Fatalf("status %d body %s, want 400 containing %q", rec.Code, rec.Body, tc.want)
			}
		})
	}
	if calls := fake.Calls(); len(calls) != 0 {
		t.Fatalf("invalid requests reached the provisioner: %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksCreateConflicts(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	for _, tc := range []struct{ code, want string }{
		{"23505", `a network named "prod" already exists`},
		{"23P01", "cidr 10.20.0.0/24 overlaps an existing network"},
	} {
		mock.ExpectBegin()
		mock.ExpectQuery(regexp.QuoteMeta("INSERT INTO networks")).WillReturnError(&pq.Error{Code: pq.ErrorCode(tc.code)})
		mock.ExpectRollback()
		rec, out := doNetworks(t, router, "admin", http.MethodPost, "/api/networks", `{"name":"prod","cidr":"10.20.0.0/24"}`)
		if rec.Code != http.StatusConflict || out["error"] != tc.want {
			t.Fatalf("pg %s: status %d body %s", tc.code, rec.Code, rec.Body)
		}
	}
	if calls := fake.Calls(); len(calls) != 0 {
		t.Fatalf("conflicting rows reached the provisioner: %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksWritesAreAdminOnly(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	for _, role := range []string{"user", "operator", "viewer", ""} {
		if rec, _ := doNetworks(t, router, role, http.MethodPost, "/api/networks", `{"name":"prod","cidr":"10.20.0.0/24"}`); rec.Code != http.StatusForbidden {
			t.Fatalf("POST as %q: %d", role, rec.Code)
		}
		if rec, _ := doNetworks(t, router, role, http.MethodDelete, "/api/networks/"+testNetworkID, ""); rec.Code != http.StatusForbidden {
			t.Fatalf("DELETE as %q: %d", role, rec.Code)
		}
	}
	if calls := fake.Calls(); len(calls) != 0 {
		t.Fatalf("forbidden requests reached the provisioner: %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksListAndGetScopeVMCount(t *testing.T) {
	router, mock, _ := newNetworksRouter(t)
	now := time.Now().UTC()
	row := func() *sqlmock.Rows {
		return sqlmock.NewRows(networkCols).AddRow(testNetworkID, "prod", testBridge, "10.20.0.0/24", "10.20.0.1", nil, 1500, nil, now, now, 2)
	}
	// Admin: every VM on the network is counted, no org parameter.
	mock.ExpectQuery(`SELECT .* FROM networks n ORDER BY`).WithoutArgs().WillReturnRows(row())
	rec, _ := doNetworks(t, router, "admin", http.MethodGet, "/api/networks", "")
	var list []map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil || rec.Code != http.StatusOK || len(list) != 1 {
		t.Fatalf("admin list: %d %s (%v)", rec.Code, rec.Body, err)
	}
	if list[0]["vlan_id"] != nil || list[0]["gateway"] != "10.20.0.1" || list[0]["vm_count"] != float64(2) || list[0]["bridge"] != testBridge {
		t.Fatalf("admin list entry %v", list[0])
	}
	// Scoped user: the count is restricted to their org (2nd query arg).
	mock.ExpectQuery(`SELECT .* v\.organization_id = \$2\) FROM networks n WHERE n\.id = \$1`).
		WithArgs(testNetworkID, "7a1f9c40-2222-4d3e-8b4c-000000000002").WillReturnRows(row())
	rec, out := doNetworks(t, router, "user", http.MethodGet, "/api/networks/"+testNetworkID, "")
	if rec.Code != http.StatusOK || out["id"] != testNetworkID {
		t.Fatalf("user get: %d %s", rec.Code, rec.Body)
	}
	// Unknown / malformed ids are 404 without a query for the malformed one.
	mock.ExpectQuery(`SELECT .* FROM networks n WHERE n\.id = \$1`).WithArgs(testNetworkID).WillReturnRows(sqlmock.NewRows(networkCols))
	if rec, _ := doNetworks(t, router, "admin", http.MethodGet, "/api/networks/"+testNetworkID, ""); rec.Code != http.StatusNotFound {
		t.Fatalf("missing network: %d", rec.Code)
	}
	if rec, _ := doNetworks(t, router, "admin", http.MethodGet, "/api/networks/not-a-uuid", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("malformed id: %d", rec.Code)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksDeleteRefusesWhileVMsAttached(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	now := time.Now().UTC()
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT .* FROM networks n WHERE n\.id = \$1 FOR UPDATE`).WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows(networkCols).AddRow(testNetworkID, "prod", testBridge, "10.20.0.0/24", nil, nil, 1500, nil, now, now, 0))
	mock.ExpectQuery(`SELECT id::text FROM vms WHERE network_id = \$1`).WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow("vm-1").AddRow("vm-2"))
	mock.ExpectRollback()

	rec, out := doNetworks(t, router, "admin", http.MethodDelete, "/api/networks/"+testNetworkID, "")
	if rec.Code != http.StatusConflict || !strings.Contains(out["error"].(string), "in use by 2 VM(s)") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ids, _ := out["vm_ids"].([]interface{}); len(ids) != 2 || ids[0] != "vm-1" {
		t.Fatalf("vm_ids %v", out["vm_ids"])
	}
	if calls := fake.Calls(); len(calls) != 0 {
		t.Fatalf("in-use network reached the provisioner: %v", calls)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksDeleteRemovesBridgeThenCommits(t *testing.T) {
	router, mock, fake := newNetworksRouter(t)
	now := time.Now().UTC()
	spec, _ := provision.NewSpec(testNetworkID, "10.20.0.0/24", "10.20.0.1", 0, 1500)
	if err := fake.Ensure(context.Background(), spec); err != nil {
		t.Fatal(err)
	}
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT .* FROM networks n WHERE n\.id = \$1 FOR UPDATE`).WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows(networkCols).AddRow(testNetworkID, "prod", testBridge, "10.20.0.0/24", "10.20.0.1", nil, 1500, nil, now, now, 0))
	mock.ExpectQuery(`SELECT id::text FROM vms WHERE network_id = \$1`).WithArgs(testNetworkID).WillReturnRows(sqlmock.NewRows([]string{"id"}))
	mock.ExpectExec(`DELETE FROM networks WHERE id = \$1`).WithArgs(testNetworkID).WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()

	rec, out := doNetworks(t, router, "admin", http.MethodDelete, "/api/networks/"+testNetworkID, "")
	if rec.Code != http.StatusOK || out["status"] != "deleted" || out["id"] != testNetworkID {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if _, still := fake.Provisioned(testBridge); still {
		t.Fatal("bridge still provisioned after delete")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}

	// A bridge that still has a guest tap keeps the row: the DELETE is
	// rolled back and the caller sees why.
	fake.FailRemove(provision.ErrBridgeInUse)
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT .* FROM networks n WHERE n\.id = \$1 FOR UPDATE`).WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows(networkCols).AddRow(testNetworkID, "prod", testBridge, "10.20.0.0/24", "10.20.0.1", nil, 1500, nil, now, now, 0))
	mock.ExpectQuery(`SELECT id::text FROM vms WHERE network_id = \$1`).WithArgs(testNetworkID).WillReturnRows(sqlmock.NewRows([]string{"id"}))
	mock.ExpectExec(`DELETE FROM networks WHERE id = \$1`).WithArgs(testNetworkID).WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectRollback()
	rec, out = doNetworks(t, router, "admin", http.MethodDelete, "/api/networks/"+testNetworkID, "")
	if rec.Code != http.StatusConflict || !strings.Contains(out["error"].(string), "attached ports") {
		t.Fatalf("in-use bridge: status %d: %s", rec.Code, rec.Body)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworksDeleteMissingIs404(t *testing.T) {
	router, mock, _ := newNetworksRouter(t)
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT .* FROM networks n WHERE n\.id = \$1 FOR UPDATE`).WithArgs(testNetworkID).WillReturnRows(sqlmock.NewRows(networkCols))
	mock.ExpectRollback()
	if rec, _ := doNetworks(t, router, "admin", http.MethodDelete, "/api/networks/"+testNetworkID, ""); rec.Code != http.StatusNotFound {
		t.Fatalf("status %d", rec.Code)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestReconcileNetworksReprovisionsEveryRow(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	now := time.Now().UTC()
	const other = "9b8a7c6d-5e4f-4a3b-8c2d-1e0f00bb5678"
	mock.ExpectQuery(`SELECT .* FROM networks n ORDER BY n\.created_at`).WillReturnRows(sqlmock.NewRows(networkCols).
		AddRow(testNetworkID, "prod", testBridge, "10.20.0.0/24", "10.20.0.1", 100, 9000, nil, now, now, 0).
		AddRow(other, "lab", "ncbr-9b8a7c6d5e", "fd00:10::/64", nil, nil, 1500, nil, now, now, 0))
	fake := &provision.Fake{}
	reconcileNetworks(db, fake)
	if calls := fake.Calls(); len(calls) != 2 || calls[0] != "ensure "+testBridge || calls[1] != "ensure ncbr-9b8a7c6d5e" {
		t.Fatalf("calls %v", calls)
	}
	if spec, ok := fake.Provisioned(testBridge); !ok || spec.VLANID != 100 || spec.MTU != 9000 || spec.Gateway.String() != "10.20.0.1" {
		t.Fatalf("reconciled spec %+v ok=%v", spec, ok)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestVMCreateWithNetworkIDIsLocalAndPersisted(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	router := mux.NewRouter()
	api := router.PathPrefix("/api").Subrouter()
	registerSecureAPIRoutes(api, db, nil, t.TempDir())

	// Unknown network: rejected before any VM work.
	mock.ExpectQuery("SELECT EXISTS \\(SELECT 1 FROM networks WHERE id = \\$1\\)").WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/vms", map[string]interface{}{"name": "net-vm", "network_id": testNetworkID}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "no network "+testNetworkID) {
		t.Fatalf("unknown network: %d %s", rec.Code, rec.Body)
	}
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/vms", map[string]interface{}{"name": "net-vm", "network_id": "bridge0"}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "not a network id") {
		t.Fatalf("legacy label: %d %s", rec.Code, rec.Body)
	}

	// Known network but a peer requested: the bridge is local-only.
	mock.ExpectQuery("SELECT EXISTS \\(SELECT 1 FROM networks WHERE id = \\$1\\)").WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/vms", map[string]interface{}{"name": "net-vm", "network_id": testNetworkID, "node_id": "peer-b"}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "cannot host it") {
		t.Fatalf("peer placement: %d %s", rec.Code, rec.Body)
	}

	// Known network, auto placement: created locally with network_id in the
	// row (13th INSERT arg) even though no manager is wired.
	mock.ExpectQuery("SELECT EXISTS \\(SELECT 1 FROM networks WHERE id = \\$1\\)").WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectQuery("SELECT EXISTS \\(SELECT 1 FROM networks WHERE id = \\$1\\)").WithArgs(testNetworkID).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectExec("INSERT INTO vms").
		WithArgs(sqlmock.AnyArg(), "net-vm", "stopped", 1, 512, 1, sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), testNetworkID).
		WillReturnResult(sqlmock.NewResult(1, 1))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/vms", map[string]interface{}{"name": "net-vm", "memory_mb": 512, "disk_size_gb": 1, "network_id": testNetworkID}))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create: %d %s", rec.Code, rec.Body)
	}
	var out map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out["placed_on"] != selfNodeID() || out["placed_by"] != "explicit-local" {
		t.Fatalf("placement %v", out)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
