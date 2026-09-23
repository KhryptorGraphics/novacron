package main

import (
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

// nodeProfile fixtures: a fast local link and a slow remote one.
func fastLink() *linkProfile { return &linkProfile{RTTMS: 0.4} }
func slowLink() *linkProfile { return &linkProfile{RTTMS: 40} }

// TestEstimatedMoveCostPrefersFastLink: same bytes cost less over the lower-RTT
// link, and zero bytes means only the run-time proxy remains.
func TestEstimatedMoveCostPrefersFastLink(t *testing.T) {
	fast := nodeProfile{NodeCapacity: NodeCapacity{NodeID: "fast"}, Link: fastLink()}
	slow := nodeProfile{NodeCapacity: NodeCapacity{NodeID: "slow"}, Link: slowLink()}
	spec := fabricJobSpec{BytesToMove: 1 << 30, MemoryMB: 512}

	cf, cs := estimatedMoveCostS(fast, spec), estimatedMoveCostS(slow, spec)
	if !(cf < cs) {
		t.Fatalf("fast-link cost %v should be below slow-link cost %v", cf, cs)
	}

	// With no bytes to move, cost is the run-time proxy only: equal across links.
	spec0 := fabricJobSpec{MemoryMB: 512}
	if estimatedMoveCostS(fast, spec0) != estimatedMoveCostS(slow, spec0) {
		t.Fatal("zero-byte jobs should have link-independent cost")
	}
}

// TestRTTToBpsFloorsAndDefaults: an unmeasured link is treated as fast (not
// zero, which would divide to +Inf), and a huge RTT never goes below the floor.
func TestRTTToBpsFloorsAndDefaults(t *testing.T) {
	if got := rttToBps(nodeProfile{}); got <= 0 {
		t.Fatalf("unmeasured link must yield a positive rate, got %v", got)
	}
	if got := rttToBps(nodeProfile{Link: &linkProfile{RTTMS: 100000}}); got < 1e7 {
		t.Fatalf("rate floor violated: %v", got)
	}
}

// TestRemoteStatusEffectiveStatus maps peer-reported driver truth onto job
// outcomes, including the killed-vs-finished distinction.
func TestRemoteStatusEffectiveStatus(t *testing.T) {
	zero := 0
	three := 3
	cases := []struct {
		name   string
		remote remoteVMStatus
		stored string
		want   string
	}{
		{"running", remoteVMStatus{State: "running"}, jobStatusRunning, jobStatusRunning},
		{"exited zero", remoteVMStatus{State: "stopped", ExitCode: &zero}, jobStatusRunning, jobStatusCompleted},
		{"exited nonzero", remoteVMStatus{State: "stopped", ExitCode: &three}, jobStatusRunning, jobStatusFailed},
		{"failed state", remoteVMStatus{State: "failed"}, jobStatusRunning, jobStatusFailed},
		{"unknown keeps stored", remoteVMStatus{State: "unknown"}, jobStatusRunning, jobStatusRunning},
		{"stopped without code keeps stored running", remoteVMStatus{State: "stopped"}, jobStatusRunning, jobStatusCompleted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.remote.effectiveStatus(tc.stored); got != tc.want {
				t.Fatalf("effectiveStatus = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestClampJobMemory bounds the placement inputs.
func TestClampJobMemory(t *testing.T) {
	if clampJobMemory(0) != 128 {
		t.Fatal("zero memory must default to 128 MB")
	}
	if clampJobMemory(1<<30) != 32768 {
		t.Fatal("oversized memory must clamp to 32768 MB")
	}
	if clampJobMemory(512) != 512 {
		t.Fatal("valid memory must pass through")
	}
}

// TestFabricJobNameStable: an unnamed job gets a deterministic prefix from the
// command and the job id, so it is identifiable in the VM list.
func TestFabricJobNameStable(t *testing.T) {
	spec := fabricJobSpec{Command: "/usr/bin/env"}
	got := fabricJobName(spec, "01234567-89ab-cdef-0123-456789abcdef")
	if got == "" || got[:11] != "fabric-job-" {
		t.Fatalf("unexpected job name %q", got)
	}
	if named := fabricJobName(fabricJobSpec{Command: "x", Name: "explicit"}, "01234567-89ab-cdef-0123-456789abcdef"); named != "explicit" {
		t.Fatalf("explicit name not honored: %q", named)
	}
}

// Cross-node jobs have no executor VM row in the submitter's database. Their
// persisted organization must still be returned for tenant authorization.
func TestGetFabricJobUsesPersistedOrganization(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	created := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT j.id, j.vm_id, j.node_id, j.command, COALESCE(j.name,''), j.status, COALESCE(j.error,''), COALESCE(j.placed_by,''), j.created_at,
		       j.organization_id
		FROM fabric_jobs j
		WHERE j.id = $1`)).
		WithArgs("job-1").
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "vm_id", "node_id", "command", "name", "status", "error", "placed_by", "created_at", "organization_id",
		}).AddRow("job-1", "vm-on-peer", "node-b", "echo hi", "job", jobStatusRunning, "", "pin", created, "org-a"))

	job, org, err := getFabricJob(t.Context(), db, "job-1")
	if err != nil {
		t.Fatalf("getFabricJob: %v", err)
	}
	if job.VMID != "vm-on-peer" || org.String != "org-a" || !org.Valid {
		t.Fatalf("cross-node job ownership = vm %q org %#v, want vm-on-peer/org-a", job.VMID, org)
	}
	if !orgVisible("org-a", org) || orgVisible("org-b", org) {
		t.Fatalf("job org visibility should allow org-a and hide org-b")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// Scoped listings retain jobs whose VM lives only in a remote node's database.
func TestListFabricJobsFiltersByPersistedOrganization(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	query := regexp.QuoteMeta(`
		SELECT id, vm_id, node_id, command, COALESCE(name,''), status, COALESCE(error,''), COALESCE(placed_by,''), created_at
		FROM fabric_jobs WHERE organization_id = $1
		ORDER BY created_at DESC LIMIT 200`)
	created := time.Now().UTC()
	mock.ExpectQuery(query).WithArgs("org-a").
		WillReturnRows(sqlmock.NewRows([]string{"id", "vm_id", "node_id", "command", "name", "status", "error", "placed_by", "created_at"}).
			AddRow("job-1", "vm-on-peer", "node-b", "echo hi", "job", jobStatusRunning, "", "pin", created))
	mock.ExpectQuery(query).WithArgs("org-b").
		WillReturnRows(sqlmock.NewRows([]string{"id", "vm_id", "node_id", "command", "name", "status", "error", "placed_by", "created_at"}))

	jobs, err := listFabricJobs(t.Context(), db, nil, "org-a", false)
	if err != nil || len(jobs) != 1 || jobs[0].VMID != "vm-on-peer" {
		t.Fatalf("org-a cross-node jobs = %#v, err=%v", jobs, err)
	}
	jobs, err = listFabricJobs(t.Context(), db, nil, "org-b", false)
	if err != nil || len(jobs) != 0 {
		t.Fatalf("org-b should not see org-a job: %#v, err=%v", jobs, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}
