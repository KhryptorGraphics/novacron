package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestOrgScopePowerRoutesHideCrossOrgAndPreserveSameOrg(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()
	vmID := "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	orgA := "11111111-1111-1111-1111-111111111111"
	orgB := "22222222-2222-2222-2222-222222222222"
	if _, err := manager.CreateVM(context.Background(), vm.CreateVMRequest{
		Name: vmID, AllowMissingOwnership: true,
		Spec: vm.VMConfig{ID: vmID, Name: vmID, Type: vm.VMTypeKVM, TenantID: "default"},
	}); err != nil {
		t.Fatalf("seed manager VM: %v", err)
	}

	router, authMgr := newOrgScopeRouterWithManager(t, db, manager)
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgB).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
	crossOrg := httptest.NewRequest(http.MethodPost, "/api/vms/"+vmID+"/start", nil)
	crossOrg.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	crossRec := httptest.NewRecorder()
	router.ServeHTTP(crossRec, crossOrg)
	if crossRec.Code != http.StatusNotFound {
		t.Fatalf("cross-org start must return 404, got %d body=%s", crossRec.Code, crossRec.Body.String())
	}
	managedVM, err := manager.GetVM(vmID)
	if err != nil || managedVM.State() != vm.StateStopped {
		t.Fatalf("cross-org start changed manager state: vm=%v err=%v", managedVM, err)
	}

	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgA).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectExec(`UPDATE vms SET state = \$2, updated_at = NOW\(\) WHERE id = \$1 AND organization_id = \$3`).
		WithArgs(vmID, "running", orgA).WillReturnResult(sqlmock.NewResult(0, 1))
	sameOrg := httptest.NewRequest(http.MethodPost, "/api/vms/"+vmID+"/start", nil)
	sameOrg.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	sameRec := httptest.NewRecorder()
	router.ServeHTTP(sameRec, sameOrg)
	if sameRec.Code != http.StatusOK || !strings.Contains(sameRec.Body.String(), `"state":"running"`) {
		t.Fatalf("same-org start should succeed with running state, got %d body=%s", sameRec.Code, sameRec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

func TestOrgScopeMetricsAndInterfacesHideCrossOrgVM(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgB := "22222222-2222-2222-2222-222222222222"
	vmID := "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
	router, authMgr := newOrgScopeRouter(t, db)
	paths := []struct {
		method string
		path   string
		body   []byte
	}{
		{http.MethodGet, "/api/vms/" + vmID + "/metrics", nil},
		{http.MethodGet, "/api/vms/" + vmID + "/interfaces", nil},
		{http.MethodPost, "/api/vms/" + vmID + "/interfaces", marshalBody(t, map[string]string{"name": "eth0", "mac_address": "02:00:00:00:00:01"})},
		{http.MethodGet, "/api/vms/" + vmID + "/interfaces/cccccccc-cccc-cccc-cccc-cccccccccccc", nil},
		{http.MethodPut, "/api/vms/" + vmID + "/interfaces/cccccccc-cccc-cccc-cccc-cccccccccccc", marshalBody(t, map[string]string{"name": "changed"})},
		{http.MethodDelete, "/api/vms/" + vmID + "/interfaces/cccccccc-cccc-cccc-cccc-cccccccccccc", nil},
	}
	for _, route := range paths {
		t.Run(route.method+" "+route.path, func(t *testing.T) {
			mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
				WithArgs(vmID, orgB).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
			req := httptest.NewRequest(route.method, route.path, strings.NewReader(string(route.body)))
			req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), `"error":"vm not found"`) {
				t.Fatalf("out-of-scope VM should be hidden with 404, got %d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

func TestOrgScopeMonitoringVMListExcludesOtherOrganizations(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgB := "22222222-2222-2222-2222-222222222222"
	mock.ExpectQuery(`SELECT id, name, state, organization_id FROM vms WHERE organization_id = \$1 ORDER BY created_at DESC`).
		WithArgs(orgB).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "state", "organization_id"}).
			AddRow("vm-b", "tenant-b-vm", "running", orgB))
	mock.ExpectQuery(`SELECT cpu_usage, memory_usage FROM vm_metrics\s+WHERE vm_id = \$1 ORDER BY timestamp DESC LIMIT 1`).
		WithArgs("vm-b").WillReturnRows(sqlmock.NewRows([]string{"cpu_usage", "memory_usage"}))
	router, authMgr := newOrgScopeRouter(t, db)
	req := httptest.NewRequest(http.MethodGet, "/api/monitoring/vms", nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "tenant-b-vm") || strings.Contains(rec.Body.String(), "tenant-a-vm") {
		t.Fatalf("monitoring list should contain only caller-org VMs, got %d body=%s", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

func TestOrgScopeMetricsReturnsNoContentWhenNoSampleExists(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgID := "22222222-2222-2222-2222-222222222222"
	vmID := "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
	router, authMgr := newOrgScopeRouter(t, db)
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgID).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectQuery(`SELECT COALESCE\(cpu_usage, 0\), COALESCE\(memory_usage, 0\)\s+FROM vm_metrics WHERE vm_id = \$1\s+ORDER BY timestamp DESC\s+LIMIT 1`).
		WithArgs(vmID).WillReturnRows(sqlmock.NewRows([]string{"cpu_usage", "memory_usage"}))
	req := httptest.NewRequest(http.MethodGet, "/api/vms/"+vmID+"/metrics", nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgID, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent || rec.Body.Len() != 0 {
		t.Fatalf("missing metric sample must not return fake zero payload, got %d body=%s", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}
