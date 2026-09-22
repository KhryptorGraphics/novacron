package main

// Usage-metered billing foundation (PR-1 of the profitability roadmap,
// research/profitability/business-models.md): persist MEASURED resource
// consumption from the fabric's own telemetry and expose it through
// org-scoped read endpoints.
//
// What this is:
//   - egress bytes/migrations: written at transfer completion from the
//     admission store's finished record (measured bytes, not estimates).
//   - job seconds: written when a fabric job reaches a terminal state.
//   - vCPU-seconds: written on VM stop/delete/terminal transitions from
//     row timestamps (created_at -> transition time).
//
// What this is NOT:
//   - Not payment collection, invoices, dunning, or Stripe integration — the
//     1,160-line in-memory-only backend/enterprise/billing stub stays
//     unreferenced; this is the metering plane it always lacked.
//   - Not monetized by default: the rate card defaults to zero USD. An
//     operator sets rates explicitly (env) when they choose to charge.

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// defaultOrganizationID is seeded by migration 000010. Every usage event must
// carry an organization; creators that predate tenant-aware request plumbing
// attribute to this org rather than writing unattributed rows.
const defaultOrganizationID = "00000000-0000-0000-0000-000000000001"

// usageEvent is one persisted unit of measured consumption.
type usageEvent struct {
	OrganizationID string                 `json:"organization_id"`
	UserID         string                 `json:"user_id,omitempty"`  // may be empty (system-initiated)
	EventType      string                 `json:"event_type"`         // egress_bytes | migration | job_seconds | vcpu_seconds
	VMID           string                 `json:"vm_id,omitempty"`
	JobID          string                 `json:"job_id,omitempty"`
	SourceNodeID   string                 `json:"source_node_id,omitempty"`
	TargetNodeID   string                 `json:"target_node_id,omitempty"`
	Quantity       float64                `json:"quantity"`
	Unit           string                 `json:"unit"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	OccurredAt     time.Time              `json:"occurred_at"`
}

// recordUsageEvent persists one measured event. Billing must never fail the
// operation that produced the usage, so errors are logged, never returned
// into the hot path; callers treat recordUsageEvent as best-effort telemetry
// with an auditable gap on failure (logged warn).
func recordUsageEvent(ctx context.Context, db *sql.DB, ev usageEvent) {
	if db == nil || ev.Quantity <= 0 {
		return
	}
	if ev.OrganizationID == "" {
		ev.OrganizationID = defaultOrganizationID
	}
	if ev.Unit == "" {
		ev.Unit = "count"
	}
	meta, _ := json.Marshal(ev.Metadata)
	if _, err := db.ExecContext(ctx, `
		INSERT INTO usage_events
			(organization_id, user_id, event_type, vm_id, job_id, source_node_id, target_node_id, quantity, unit, metadata, occurred_at)
		VALUES ($1, NULLIF($2, '')::uuid, $3, NULLIF($4, '')::uuid, NULLIF($5, '')::uuid, NULLIF($6, ''), NULLIF($7, ''), $8, $9, $10, $11)
	`, ev.OrganizationID, ev.UserID, ev.EventType, ev.VMID, ev.JobID, ev.SourceNodeID, ev.TargetNodeID,
		ev.Quantity, ev.Unit, meta, ev.OccurredAt); err != nil {
		logger.Warn("usage event persist failed (billing gap)",
			"event_type", ev.EventType, "org", ev.OrganizationID, "quantity", ev.Quantity, "error", err)
	}
}

// usageOrgForVM resolves a VM's organization for attribution; falls back to
// the default org for rows that predate org stamping.
func usageOrgForVM(ctx context.Context, db *sql.DB, vmID string) string {
	if db == nil || vmID == "" {
		return defaultOrganizationID
	}
	var org sql.NullString
	if err := db.QueryRowContext(ctx,
		`SELECT organization_id FROM vms WHERE id = $1`, vmID).Scan(&org); err != nil {
		return defaultOrganizationID
	}
	if org.Valid && org.String != "" {
		return org.String
	}
	return defaultOrganizationID
}

// usageRates is the operator-configurable rate card. All values are USD and
// default to 0 — metering is real even before anyone charges for it, and a
// zero rate keeps every summary "measured but unpriced" rather than
// pretending revenue exists.
type usageRates struct {
	PerGBEgress        float64 `json:"usd_per_gb_egress"`         // NOVACRON_RATE_PER_GB_EGRESS
	PerVCPUHour        float64 `json:"usd_per_vcpu_hour"`         // NOVACRON_RATE_PER_VCPU_HOUR
	PerJobSecond       float64 `json:"usd_per_job_second"`        // NOVACRON_RATE_PER_JOB_SECOND
	PerMigration       float64 `json:"usd_per_migration"`         // NOVACRON_RATE_PER_MIGRATION
}

// loadUsageRates reads the rate card from env. Invalid/negative values are
// rejected to zero rather than billed — a misconfigured rate must never
// invent revenue.
func loadUsageRates() usageRates {
	parse := func(key string) float64 {
		v, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv(key)), 64)
		if err != nil || v < 0 {
			return 0
		}
		return v
	}
	return usageRates{
		PerGBEgress:  parse("NOVACRON_RATE_PER_GB_EGRESS"),
		PerVCPUHour:  parse("NOVACRON_RATE_PER_VCPU_HOUR"),
		PerJobSecond: parse("NOVACRON_RATE_PER_JOB_SECOND"),
		PerMigration: parse("NOVACRON_RATE_PER_MIGRATION"),
	}
}

// usageTotals is the aggregated consumption for one org over a window.
type usageTotals struct {
	EgressBytes   float64 `json:"egress_bytes"`
	EgressGB      float64 `json:"egress_gb"`
	Migrations    float64 `json:"migrations"`
	JobSeconds    float64 `json:"job_seconds"`
	VCPUSeconds   float64 `json:"vcpu_seconds"`
	VCPUHours     float64 `json:"vcpu_hours"`
	EstimatedCost float64 `json:"estimated_cost_usd"`
}

// computeUsageTotals aggregates raw per-type quantities into totals and a
// cost estimate. Pure function — the unit-testable core of the summary
// endpoint; no DB, no clock.
func computeUsageTotals(egressBytes, migrations, jobSeconds, vcpuSeconds float64, rates usageRates) usageTotals {
	t := usageTotals{
		EgressBytes: egressBytes,
		EgressGB:    egressBytes / (1 << 30),
		Migrations:  migrations,
		JobSeconds:  jobSeconds,
		VCPUSeconds: vcpuSeconds,
		VCPUHours:   vcpuSeconds / 3600,
	}
	t.EstimatedCost = t.EgressGB*rates.PerGBEgress +
		t.VCPUHours*rates.PerVCPUHour +
		t.JobSeconds*rates.PerJobSecond +
		t.Migrations*rates.PerMigration
	return t
}

// billingOrgForRequest resolves which org a caller may see: non-admins are
// forced to their own organization; admins may pass ?org_id= to inspect any.
func billingOrgForRequest(ctx context.Context, db *sql.DB, r *http.Request) (orgID string, allOrgs bool, err error) {
	userID, _ := r.Context().Value("user_id").(string)
	role, _ := r.Context().Value("role").(string)
	isAdmin := role == "admin" || role == "super-admin"

	if isAdmin {
		if q := strings.TrimSpace(r.URL.Query().Get("org_id")); q != "" {
			return q, false, nil
		}
		return "", true, nil // no filter: all orgs
	}
	if userID == "" {
		return "", false, errUnauthorizedBilling
	}
	var org sql.NullString
	if err := db.QueryRowContext(ctx,
		`SELECT organization_id FROM users WHERE id = $1`, userID).Scan(&org); err != nil {
		return "", false, err
	}
	if org.Valid && org.String != "" {
		return org.String, false, nil
	}
	return defaultOrganizationID, false, nil
}

var errUnauthorizedBilling = &billingError{code: http.StatusUnauthorized, msg: "authentication required"}

type billingError struct {
	code int
	msg  string
}

func (e *billingError) Error() string { return e.msg }

// registerBillingUsageRoutes mounts the org-scoped metering read API on the
// authenticated API router. Endpoints:
//
//	GET /billing/usage?from=&to=&limit=         raw events for the caller's org
//	GET /billing/usage/summary?from=&to=        aggregates + estimated cost
func registerBillingUsageRoutes(apiRouter *mux.Router, db *sql.DB) {
	apiRouter.HandleFunc("/billing/usage", func(w http.ResponseWriter, r *http.Request) {
		orgID, allOrgs, err := billingOrgForRequest(r.Context(), db, r)
		if err != nil {
			writeJSONError(w, errStatusOf(err), "usage access denied")
			return
		}
		from, to := parseUsageWindow(r)
		limit := 1000
		if v, perr := strconv.Atoi(r.URL.Query().Get("limit")); perr == nil && v > 0 && v <= 10000 {
			limit = v
		}

		query := `
			SELECT organization_id, COALESCE(user_id::text,''), event_type,
			       COALESCE(vm_id::text,''), COALESCE(job_id::text,''),
			       COALESCE(source_node_id,''), COALESCE(target_node_id,''),
			       quantity, unit, metadata, occurred_at
			FROM usage_events
			WHERE occurred_at >= $1 AND occurred_at < $2`
		args := []interface{}{from, to}
		if !allOrgs {
			query += " AND organization_id = $3"
			args = append(args, orgID)
		}
		query += " ORDER BY occurred_at DESC LIMIT " + strconv.Itoa(limit)

		rows, err := db.QueryContext(r.Context(), query, args...)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "usage query failed")
			return
		}
		defer rows.Close()

		events := make([]usageEvent, 0)
		for rows.Next() {
			var ev usageEvent
			var meta []byte
			var org, uid, vmid, jid, sn, tn string
			if err := rows.Scan(&org, &uid, &ev.EventType, &vmid, &jid, &sn, &tn,
				&ev.Quantity, &ev.Unit, &meta, &ev.OccurredAt); err != nil {
				continue
			}
			ev.OrganizationID, ev.UserID, ev.VMID, ev.JobID = org, uid, vmid, jid
			ev.SourceNodeID, ev.TargetNodeID = sn, tn
			if len(meta) > 0 {
				_ = json.Unmarshal(meta, &ev.Metadata)
			}
			events = append(events, ev)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"org_id": nullableOrg(orgID, allOrgs),
			"from":   from.Format(time.RFC3339),
			"to":     to.Format(time.RFC3339),
			"events": events,
		})
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/billing/usage/summary", func(w http.ResponseWriter, r *http.Request) {
		orgID, allOrgs, err := billingOrgForRequest(r.Context(), db, r)
		if err != nil {
			writeJSONError(w, errStatusOf(err), "usage access denied")
			return
		}
		from, to := parseUsageWindow(r)

		query := `
			SELECT event_type, SUM(quantity)
			FROM usage_events
			WHERE occurred_at >= $1 AND occurred_at < $2`
		args := []interface{}{from, to}
		if !allOrgs {
			query += " AND organization_id = $3"
			args = append(args, orgID)
		}
		query += " GROUP BY event_type"

		rows, err := db.QueryContext(r.Context(), query, args...)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "usage summary query failed")
			return
		}
		defer rows.Close()

		var egressBytes, migrations, jobSeconds, vcpuSeconds float64
		for rows.Next() {
			var et string
			var q float64
			if err := rows.Scan(&et, &q); err != nil {
				continue
			}
			switch et {
			case "egress_bytes":
				egressBytes = q
			case "migration":
				migrations = q
			case "job_seconds":
				jobSeconds = q
			case "vcpu_seconds":
				vcpuSeconds = q
			}
		}

		rates := loadUsageRates()
		totals := computeUsageTotals(egressBytes, migrations, jobSeconds, vcpuSeconds, rates)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"org_id":     nullableOrg(orgID, allOrgs),
			"from":       from.Format(time.RFC3339),
			"to":         to.Format(time.RFC3339),
			"totals":     totals,
			"rate_card":  rates,
			"note":       "measured consumption; rates are operator-configured via NOVACRON_RATE_* env (0 defaults mean unpriced, not free)",
		})
	}).Methods(http.MethodGet)
}

// parseUsageWindow accepts RFC3339 from/to; defaults: trailing 30 days.
func parseUsageWindow(r *http.Request) (time.Time, time.Time) {
	now := time.Now().UTC()
	from, to := now.AddDate(0, 0, -30), now
	if v := strings.TrimSpace(r.URL.Query().Get("from")); v != "" {
		if parsed, err := time.Parse(time.RFC3339, v); err == nil {
			from = parsed
		}
	}
	if v := strings.TrimSpace(r.URL.Query().Get("to")); v != "" {
		if parsed, err := time.Parse(time.RFC3339, v); err == nil {
			to = parsed
		}
	}
	return from, to
}

func nullableOrg(orgID string, allOrgs bool) interface{} {
	if allOrgs {
		return nil
	}
	return orgID
}

func errStatusOf(err error) int {
	if be, ok := err.(*billingError); ok {
		return be.code
	}
	return http.StatusInternalServerError
}

// Secondary guard: the unique partial index idx_usage_events_job_once on
// usage_events(job_id, event_type) WHERE event_type = 'job_seconds' prevents
// a concurrent double-fire (two watchers seeing the same terminal poll)
// from ever writing two rows for the same job.
func maybeMeterJobOnce(db *sql.DB, job *fabricJob, terminalStatus string, vmManager *core_vm.VMManager) {
	if db == nil || job.VMID == "" || terminalStatus == "" {
		return
	}
	// Only meter once per job; if we fail, log and move on.
	var orgID string
	err := db.QueryRowContext(context.Background(),
		`SELECT organization_id FROM vms WHERE id = $1`, job.VMID).Scan(&orgID)
	if err != nil || orgID == "" {
		orgID = defaultOrganizationID
	}

		// Compute elapsed: job was created and started at created_at. Terminal now.
	// We can't know the exact exit time from the ProcessDriver since it's
	// already exited; created_at to now is the longest possible billable
	// duration (upper bound). Accuracy: honest label "observed terminal
	// transition time, not measured exit".
	elapsed := time.Since(job.CreatedAt).Seconds()
	if elapsed < 0 || !isFiniteFloat(elapsed) {
		elapsed = 0
	}

	recordUsageEvent(context.Background(), db, usageEvent{
		OrganizationID: orgID,
		EventType:      "job_seconds",
		JobID:          job.ID,
		VMID:           job.VMID,
		TargetNodeID:   job.NodeID,
		Quantity:       elapsed,
		Unit:           "seconds",
		OccurredAt:     time.Now().UTC(),
		Metadata: map[string]interface{}{
			"terminal_status": terminalStatus,
			"command":         job.Command,
			"note":            "terminated at observation time, wall clock since create",
		},
	})
}

// isFiniteFloat guards against time arithmetic NaN/Inf.
func isFiniteFloat(f float64) bool {
	return f == f && f-f == 0 // NaN check + Inf check (Inf-Inf=NaN, Inf==Inf is true in Go)
}
