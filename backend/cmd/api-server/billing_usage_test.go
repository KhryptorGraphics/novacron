package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// TestComputeUsageTotalsZeroRates: with the default all-zero rate card every
// quantity is still measurable (the whole point of metering-before-pricing)
// but no revenue is invented.
func TestComputeUsageTotalsZeroRates(t *testing.T) {
	totals := computeUsageTotals(64<<30, 3, 7200, 14400, usageRates{})
	if totals.EgressGB != 64 {
		t.Fatalf("egress GB conversion wrong: got %v want 64", totals.EgressGB)
	}
	if totals.VCPUHours != 4 {
		t.Fatalf("vcpu hours conversion wrong: got %v want 4", totals.VCPUHours)
	}
	if totals.EstimatedCost != 0 {
		t.Fatalf("zero rate card must estimate $0, got %v", totals.EstimatedCost)
	}
}

// TestComputeUsageTotalsPricing: each term contributes its own unit price.
func TestComputeUsageTotalsPricing(t *testing.T) {
	rates := usageRates{PerGBEgress: 0.04, PerVCPUHour: 0.027, PerJobSecond: 0.001, PerMigration: 0.50}
	totals := computeUsageTotals(100<<30 /*100 GiB*/, 2, 200, 7200 /*2 vCPU-hours*/, rates)
	want := 100*0.04 + 2*0.027 + 200*0.001 + 2*0.50
	if absDiff(totals.EstimatedCost, want) > 1e-9 {
		t.Fatalf("estimated cost: got %v want %v", totals.EstimatedCost, want)
	}
}

// TestLoadUsageRatesRejectsGarbage: negative or unparseable rates are forced
// to zero so a misconfigured operator env never invents revenue.
func TestLoadUsageRatesRejectsGarbage(t *testing.T) {
	t.Setenv("NOVACRON_RATE_PER_GB_EGRESS", "-1")
	t.Setenv("NOVACRON_RATE_PER_VCPU_HOUR", "not-a-number")
	t.Setenv("NOVACRON_RATE_PER_JOB_SECOND", "0.005")
	os.Unsetenv("NOVACRON_RATE_PER_MIGRATION")

	rates := loadUsageRates()
	if rates.PerGBEgress != 0 || rates.PerVCPUHour != 0 {
		t.Fatalf("invalid rates must zero out: %+v", rates)
	}
	if rates.PerJobSecond != 0.005 {
		t.Fatalf("valid rate lost: %+v", rates)
	}
	if rates.PerMigration != 0 {
		t.Fatalf("unset rate must default to 0: %+v", rates)
	}
}

// TestRecordUsageEventGuards: nil DB and non-positive quantities are silently
// dropped — billing telemetry must never fail the operation it observes.
func TestRecordUsageEventGuards(t *testing.T) {
	recordUsageEvent(t.Context(), nil, usageEvent{EventType: "egress_bytes", Quantity: 100})
	recordUsageEvent(t.Context(), nil, usageEvent{EventType: "egress_bytes", Quantity: 0})
	recordUsageEvent(t.Context(), nil, usageEvent{})
}

// TestUsageOrgForVMUsesVMOrganization: a VM row that carries an org is
// attributed to it -- the whole point of stamping organization_id at create.
func TestUsageOrgForVMUsesVMOrganization(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	const org = "11111111-2222-3333-4444-555555555555"
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT organization_id FROM vms WHERE id = $1`)).
		WithArgs("vm-with-org").
		WillReturnRows(sqlmock.NewRows([]string{"organization_id"}).AddRow(org))

	if got := usageOrgForVM(t.Context(), db, "vm-with-org"); got != org {
		t.Fatalf("expected the VM's org %s, got %s", org, got)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestUsageOrgForVMFallsBackToDefault: rows that predate org stamping (NULL
// org), VMs with no row at all, and the no-DB/no-id paths all attribute to the
// seeded default org -- a usage event is never unattributed and never invented.
func TestUsageOrgForVMFallsBackToDefault(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT organization_id FROM vms WHERE id = $1`)).
		WithArgs("vm-null-org").
		WillReturnRows(sqlmock.NewRows([]string{"organization_id"}).AddRow(nil))
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT organization_id FROM vms WHERE id = $1`)).
		WithArgs("vm-no-row").
		WillReturnError(sql.ErrNoRows)

	for _, vmID := range []string{"vm-null-org", "vm-no-row"} {
		if got := usageOrgForVM(t.Context(), db, vmID); got != defaultOrganizationID {
			t.Fatalf("vm %s: expected default org %s, got %s", vmID, defaultOrganizationID, got)
		}
	}
	if got := usageOrgForVM(t.Context(), nil, "vm-any"); got != defaultOrganizationID {
		t.Fatalf("nil db: expected default org %s, got %s", defaultOrganizationID, got)
	}
	if got := usageOrgForVM(t.Context(), db, ""); got != defaultOrganizationID {
		t.Fatalf("empty vm id: expected default org %s, got %s", defaultOrganizationID, got)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCreateVMLocalStampsOrganization pins what a create writes into
// vms.organization_id: the positional WithArgs assertion fails if the org is
// smuggled into another column, and a non-uuid tenant label (legacy tokens
// carry e.g. "default") must be dropped rather than handed to the ::uuid cast.
// Whether THIS node's organizations directory can resolve the id is decided
// inside the INSERT expression itself (a scalar subquery), which sqlmock never
// sees -- so this pins the parameter, not the directory lookup.
func TestCreateVMLocalStampsOrganization(t *testing.T) {
	const org = "11111111-2222-3333-4444-555555555555"
	cases := []struct {
		name    string
		specOrg string
		wantArg string // value the INSERT must receive for organization_id
	}{
		{"uuid org is stamped", org, org},
		{"legacy tenant label is dropped", "default", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			db, mock, err := sqlmock.New()
			if err != nil {
				t.Fatalf("sqlmock: %v", err)
			}
			defer db.Close()

			mock.ExpectExec("INSERT INTO vms").
				WithArgs(sqlmock.AnyArg(), "org-vm", "stopped", 1, 512, 1,
					sqlmock.AnyArg(), // os_type
					sqlmock.AnyArg(), // node_id: selfNodeID(), depends on NOVACRON_NODE_ID
					"",               // owner_id (no owner on this request)
					"",               // requested_owner_id
					tc.wantArg,       // organization_id
					sqlmock.AnyArg(), // metadata JSON
				).
				WillReturnResult(sqlmock.NewResult(1, 1))

			vmID, state, err := createVMLocal(t.Context(), db, nil, clusterCreateSpec{
				Name: "org-vm", MemoryMB: 512, DiskSizeGB: 1, OrganizationID: tc.specOrg,
			})
			if err != nil {
				t.Fatalf("createVMLocal: %v", err)
			}
			if vmID == "" || state != "stopped" {
				t.Fatalf("unexpected create result: id=%q state=%q", vmID, state)
			}
			if err := mock.ExpectationsWereMet(); err != nil {
				t.Fatalf("unmet sql expectations: %v", err)
			}
		})
	}
}

// TestClusterCreateStampsCallerOrganization drives the canonical POST /vms
// route end to end: the org the row carries (and therefore every usage event
// metered off it) is the AUTHENTICATED caller's org -- requireAuth's tenant
// claim -- not something the request body can set. A legacy token whose tenant
// claim is not a uuid ("default") creates the VM unattributed rather than
// failing the INSERT on the ::uuid cast.
func TestClusterCreateStampsCallerOrganization(t *testing.T) {
	const org = "11111111-2222-3333-4444-555555555555"
	cases := []struct {
		name     string
		tenant   string
		wantArg  string
		wantEcho interface{}
	}{
		{"uuid tenant claim is stamped", org, org, org},
		{"non-uuid tenant label is dropped", "default", "", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			db, mock, err := sqlmock.New()
			if err != nil {
				t.Fatalf("sqlmock: %v", err)
			}
			defer db.Close()

			authManager := auth.NewSimpleAuthManager("test-secret", nil)
			router := mux.NewRouter()
			protected := router.PathPrefix("/api").Subrouter()
			protected.Use(requireAuth(authManager))
			// nil manager: metadata-only create, no qemu/driver in the way.
			registerSecureAPIRoutes(protected, db, nil, t.TempDir())

			// The org argument must land in the organization_id slot (11th) --
			// the owner columns right before it stay empty for this token.
			mock.ExpectExec("INSERT INTO vms").
				WithArgs(sqlmock.AnyArg(), "org-create", "stopped", 1, 512, 1,
					sqlmock.AnyArg(), // os_type
					sqlmock.AnyArg(), // node_id: selfNodeID(), depends on NOVACRON_NODE_ID
					"",               // owner_id
					"",               // requested_owner_id
					tc.wantArg,       // organization_id
					sqlmock.AnyArg(), // metadata JSON
				).
				WillReturnResult(sqlmock.NewResult(1, 1))

			req := httptest.NewRequest(http.MethodPost, "/api/vms",
				strings.NewReader(`{"name":"org-create","memory_mb":512,"disk_size_gb":1}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", signedBearerToken(t, authManager, "7", tc.tenant, "admin"))

			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusCreated {
				t.Fatalf("expected 201, got %d (%s)", rec.Code, rec.Body.String())
			}
			var created map[string]interface{}
			if err := json.NewDecoder(rec.Body).Decode(&created); err != nil {
				t.Fatalf("decode create response: %v", err)
			}
			if created["organization_id"] != tc.wantEcho {
				t.Fatalf("create response organization_id: expected %#v, got %#v", tc.wantEcho, created["organization_id"])
			}
			if err := mock.ExpectationsWereMet(); err != nil {
				t.Fatalf("unmet sql expectations: %v", err)
			}
		})
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}
