//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

type phase4ContractStore struct {
	mu             sync.Mutex
	firewallRules  []map[string]interface{}
	loadBalancers  []map[string]interface{}
	migrationPlans []map[string]interface{}
	migrationJobs  []map[string]interface{}
	backupPolicies []map[string]interface{}
	backupRuns     []map[string]interface{}
	restoreJobs    []map[string]interface{}
}

var phase4Contracts = &phase4ContractStore{}

func registerNetworkPolicyContractRoutes(router *mux.Router) {
	router.HandleFunc("/network/firewall-rules", listFirewallRulesHandler).Methods(http.MethodGet)
	router.HandleFunc("/network/firewall-rules", createFirewallRuleHandler).Methods(http.MethodPost)
	router.HandleFunc("/network/firewall-rules/{id}", updateFirewallRuleHandler).Methods(http.MethodPut)
	router.HandleFunc("/network/firewall-rules/{id}", deleteFirewallRuleHandler).Methods(http.MethodDelete)

	router.HandleFunc("/network/load-balancers", listLoadBalancersHandler).Methods(http.MethodGet)
	router.HandleFunc("/network/load-balancers", createLoadBalancerHandler).Methods(http.MethodPost)
	router.HandleFunc("/network/load-balancers/{id}", updateLoadBalancerHandler).Methods(http.MethodPut)
	router.HandleFunc("/network/load-balancers/{id}", deleteLoadBalancerHandler).Methods(http.MethodDelete)
	router.HandleFunc("/network/load-balancers/{id}/backends", addLoadBalancerBackendHandler).Methods(http.MethodPost)
	router.HandleFunc("/network/load-balancers/{id}/backends/{backendId}", deleteLoadBalancerBackendHandler).Methods(http.MethodDelete)
}

func registerMigrationPlanningContractRoutes(router *mux.Router) {
	router.HandleFunc("/migration/plans", listMigrationPlansHandler).Methods(http.MethodGet)
	router.HandleFunc("/migration/plans", createMigrationPlanHandler).Methods(http.MethodPost)
	router.HandleFunc("/migration/preflight", runMigrationPreflightHandler).Methods(http.MethodPost)
	router.HandleFunc("/migration/jobs/{id}", getMigrationJobHandler).Methods(http.MethodGet)
	router.HandleFunc("/migration/jobs/{id}/rollback", rollbackMigrationJobHandler).Methods(http.MethodPost)
}

func registerBackupPolicyContractRoutes(router *mux.Router) {
	router.HandleFunc("/backup/policies", listBackupPoliciesHandler).Methods(http.MethodGet)
	router.HandleFunc("/backup/policies", createBackupPolicyHandler).Methods(http.MethodPost)
	router.HandleFunc("/backup/policies/{id}", updateBackupPolicyHandler).Methods(http.MethodPut)
	router.HandleFunc("/backup/policies/{id}", deleteBackupPolicyHandler).Methods(http.MethodDelete)
	router.HandleFunc("/backup/policies/{id}/run", runBackupPolicyHandler).Methods(http.MethodPost)
	router.HandleFunc("/backup/backups", listBackupsHandler).Methods(http.MethodGet)
	router.HandleFunc("/backup/backups/{id}", getBackupHandler).Methods(http.MethodGet)
	router.HandleFunc("/backup/backups/{id}/verify", verifyBackupHandler).Methods(http.MethodPost)
	router.HandleFunc("/backup/restore", startRestoreHandler).Methods(http.MethodPost)
	router.HandleFunc("/backup/restore/{id}", getRestoreHandler).Methods(http.MethodGet)
}

func listFirewallRulesHandler(w http.ResponseWriter, r *http.Request) {
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusOK, cloneRecords(phase4Contracts.firewallRules))
}

func createFirewallRuleHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name        string `json:"name"`
		NetworkID   string `json:"networkId"`
		Direction   string `json:"direction"`
		Action      string `json:"action"`
		Protocol    string `json:"protocol"`
		Source      string `json:"source"`
		Destination string `json:"destination"`
		Port        string `json:"port"`
		Priority    int    `json:"priority"`
		Enabled     *bool  `json:"enabled"`
	}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Name) == "" {
		writeJSONError(w, http.StatusBadRequest, "name is required")
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	enabled := true
	if req.Enabled != nil {
		enabled = *req.Enabled
	}
	rule := map[string]interface{}{
		"id":          contractID("fw"),
		"name":        req.Name,
		"networkId":   req.NetworkID,
		"direction":   defaultString(req.Direction, "inbound"),
		"action":      defaultString(req.Action, "allow"),
		"protocol":    defaultString(req.Protocol, "tcp"),
		"source":      defaultString(req.Source, "0.0.0.0/0"),
		"destination": defaultString(req.Destination, "any"),
		"port":        defaultString(req.Port, "any"),
		"priority":    defaultInt(req.Priority, 100),
		"enabled":     enabled,
		"hits":        0,
		"status":      "not_configured",
		"createdAt":   now,
		"updatedAt":   now,
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.firewallRules = append([]map[string]interface{}{rule}, phase4Contracts.firewallRules...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusCreated, rule)
}

func updateFirewallRuleHandler(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, rule := range phase4Contracts.firewallRules {
		if rule["id"] == id {
			mergeContractRecord(rule, req)
			writeJSON(w, http.StatusOK, rule)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "firewall rule not found")
}

func deleteFirewallRuleHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for i, rule := range phase4Contracts.firewallRules {
		if rule["id"] == id {
			phase4Contracts.firewallRules = append(phase4Contracts.firewallRules[:i], phase4Contracts.firewallRules[i+1:]...)
			writeJSON(w, http.StatusOK, map[string]interface{}{"id": id, "status": "deleted"})
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "firewall rule not found")
}

func listLoadBalancersHandler(w http.ResponseWriter, r *http.Request) {
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusOK, cloneRecords(phase4Contracts.loadBalancers))
}

func createLoadBalancerHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name      string                   `json:"name"`
		NetworkID string                   `json:"networkId"`
		VIP       string                   `json:"vip"`
		Port      int                      `json:"port"`
		Algorithm string                   `json:"algorithm"`
		Type      string                   `json:"type"`
		Backends  []map[string]interface{} `json:"backends"`
	}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Name) == "" {
		writeJSONError(w, http.StatusBadRequest, "name is required")
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	lb := map[string]interface{}{
		"id":          contractID("lb"),
		"name":        req.Name,
		"networkId":   req.NetworkID,
		"vip":         req.VIP,
		"port":        defaultInt(req.Port, 80),
		"algorithm":   defaultString(req.Algorithm, "round_robin"),
		"type":        defaultString(req.Type, "layer4"),
		"status":      "not_configured",
		"backends":    req.Backends,
		"healthCheck": map[string]interface{}{"enabled": false, "path": "", "intervalSeconds": 30},
		"createdAt":   now,
		"updatedAt":   now,
	}
	if req.Backends == nil {
		lb["backends"] = []map[string]interface{}{}
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.loadBalancers = append([]map[string]interface{}{lb}, phase4Contracts.loadBalancers...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusCreated, lb)
}

func updateLoadBalancerHandler(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, lb := range phase4Contracts.loadBalancers {
		if lb["id"] == id {
			mergeContractRecord(lb, req)
			writeJSON(w, http.StatusOK, lb)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "load balancer not found")
}

func deleteLoadBalancerHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for i, lb := range phase4Contracts.loadBalancers {
		if lb["id"] == id {
			phase4Contracts.loadBalancers = append(phase4Contracts.loadBalancers[:i], phase4Contracts.loadBalancers[i+1:]...)
			writeJSON(w, http.StatusOK, map[string]interface{}{"id": id, "status": "deleted"})
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "load balancer not found")
}

func addLoadBalancerBackendHandler(w http.ResponseWriter, r *http.Request) {
	var backend map[string]interface{}
	if !decodeJSONRequest(w, r, &backend) {
		return
	}
	id := mux.Vars(r)["id"]
	if _, ok := backend["id"]; !ok {
		backend["id"] = contractID("backend")
	}
	if _, ok := backend["status"]; !ok {
		backend["status"] = "not_configured"
	}
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, lb := range phase4Contracts.loadBalancers {
		if lb["id"] == id {
			backends, _ := lb["backends"].([]map[string]interface{})
			backends = append(backends, backend)
			lb["backends"] = backends
			lb["updatedAt"] = time.Now().UTC().Format(time.RFC3339)
			writeJSON(w, http.StatusCreated, backend)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "load balancer not found")
}

func deleteLoadBalancerBackendHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, lb := range phase4Contracts.loadBalancers {
		if lb["id"] != vars["id"] {
			continue
		}
		backends, _ := lb["backends"].([]map[string]interface{})
		for i, backend := range backends {
			if backend["id"] == vars["backendId"] {
				lb["backends"] = append(backends[:i], backends[i+1:]...)
				lb["updatedAt"] = time.Now().UTC().Format(time.RFC3339)
				writeJSON(w, http.StatusOK, map[string]interface{}{"id": vars["backendId"], "status": "deleted"})
				return
			}
		}
		writeJSONError(w, http.StatusNotFound, "load balancer backend not found")
		return
	}
	writeJSONError(w, http.StatusNotFound, "load balancer not found")
}

func listMigrationPlansHandler(w http.ResponseWriter, r *http.Request) {
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusOK, cloneRecords(phase4Contracts.migrationPlans))
}

func createMigrationPlanHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SourceCluster      string   `json:"sourceCluster"`
		TargetCluster      string   `json:"targetCluster"`
		VMIDs              []string `json:"vmIds"`
		MigrationStrategy  string   `json:"migrationStrategy"`
		BandwidthMbps      int      `json:"bandwidthMbps"`
		MaxDowntimeSeconds int      `json:"maxDowntimeSeconds"`
	}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.TargetCluster) == "" || len(req.VMIDs) == 0 {
		writeJSONError(w, http.StatusBadRequest, "targetCluster and vmIds are required")
		return
	}
	strategy := defaultString(req.MigrationStrategy, "cold")
	checks := migrationPreflightChecks(strategy, req.BandwidthMbps, req.MaxDowntimeSeconds)
	now := time.Now().UTC().Format(time.RFC3339)
	plan := map[string]interface{}{
		"planId":                   contractID("migration-plan"),
		"status":                   preflightStatus(checks),
		"sourceCluster":            defaultString(req.SourceCluster, "local"),
		"targetCluster":            req.TargetCluster,
		"vmIds":                    req.VMIDs,
		"vmCount":                  len(req.VMIDs),
		"migrationStrategy":        strategy,
		"estimatedDurationSeconds": len(req.VMIDs) * 300,
		"estimatedDowntimeSeconds": estimatedDowntime(strategy, req.MaxDowntimeSeconds),
		"checks":                   checks,
		"createdAt":                now,
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.migrationPlans = append([]map[string]interface{}{plan}, phase4Contracts.migrationPlans...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusCreated, plan)
}

func runMigrationPreflightHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		PlanID             string   `json:"planId"`
		VMIDs              []string `json:"vmIds"`
		MigrationStrategy  string   `json:"migrationStrategy"`
		BandwidthMbps      int      `json:"bandwidthMbps"`
		MaxDowntimeSeconds int      `json:"maxDowntimeSeconds"`
	}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	checks := migrationPreflightChecks(defaultString(req.MigrationStrategy, "cold"), req.BandwidthMbps, req.MaxDowntimeSeconds)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"planId":    req.PlanID,
		"status":    preflightStatus(checks),
		"vmCount":   len(req.VMIDs),
		"checks":    checks,
		"createdAt": time.Now().UTC().Format(time.RFC3339),
	})
}

func getMigrationJobHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, job := range phase4Contracts.migrationJobs {
		if job["jobId"] == id || job["id"] == id {
			writeJSON(w, http.StatusOK, job)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "migration job not found")
}

func rollbackMigrationJobHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	rollback := map[string]interface{}{
		"jobId":      id,
		"rollbackId": contractID("migration-rollback"),
		"status":     "queued",
		"createdAt":  time.Now().UTC().Format(time.RFC3339),
		"condition":  "operator_requested",
	}
	writeJSON(w, http.StatusAccepted, rollback)
}

func listBackupPoliciesHandler(w http.ResponseWriter, r *http.Request) {
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusOK, cloneRecords(phase4Contracts.backupPolicies))
}

func createBackupPolicyHandler(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	name, _ := req["name"].(string)
	if strings.TrimSpace(name) == "" {
		writeJSONError(w, http.StatusBadRequest, "name is required")
		return
	}
	now := time.Now().UTC().Format(time.RFC3339)
	policy := map[string]interface{}{
		"id":            contractID("backup-policy"),
		"name":          name,
		"enabled":       boolValue(req["enabled"], true),
		"schedule":      stringValue(req["schedule"], "manual"),
		"retentionDays": numberValue(req["retentionDays"], 30),
		"target":        stringValue(req["target"], "local"),
		"status":        "not_configured",
		"createdAt":     now,
		"updatedAt":     now,
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.backupPolicies = append([]map[string]interface{}{policy}, phase4Contracts.backupPolicies...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusCreated, policy)
}

func updateBackupPolicyHandler(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, policy := range phase4Contracts.backupPolicies {
		if policy["id"] == id {
			mergeContractRecord(policy, req)
			writeJSON(w, http.StatusOK, policy)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "backup policy not found")
}

func deleteBackupPolicyHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for i, policy := range phase4Contracts.backupPolicies {
		if policy["id"] == id {
			phase4Contracts.backupPolicies = append(phase4Contracts.backupPolicies[:i], phase4Contracts.backupPolicies[i+1:]...)
			writeJSON(w, http.StatusOK, map[string]interface{}{"id": id, "status": "deleted"})
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "backup policy not found")
}

func runBackupPolicyHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	now := time.Now().UTC().Format(time.RFC3339)
	run := map[string]interface{}{
		"id":        contractID("backup-run"),
		"policyId":  id,
		"status":    "queued",
		"createdAt": now,
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.backupRuns = append([]map[string]interface{}{run}, phase4Contracts.backupRuns...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusAccepted, run)
}

func listBackupsHandler(w http.ResponseWriter, r *http.Request) {
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusOK, cloneRecords(phase4Contracts.backupRuns))
}

func getBackupHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, backup := range phase4Contracts.backupRuns {
		if backup["id"] == id {
			writeJSON(w, http.StatusOK, backup)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "backup not found")
}

func verifyBackupHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"id":        mux.Vars(r)["id"],
		"status":    "verification_queued",
		"createdAt": time.Now().UTC().Format(time.RFC3339),
	})
}

func startRestoreHandler(w http.ResponseWriter, r *http.Request) {
	var req map[string]interface{}
	if !decodeJSONRequest(w, r, &req) {
		return
	}
	backupID := stringValue(req["backupId"], "")
	if backupID == "" {
		writeJSONError(w, http.StatusBadRequest, "backupId is required")
		return
	}
	restore := map[string]interface{}{
		"id":        contractID("restore"),
		"backupId":  backupID,
		"target":    stringValue(req["target"], "original"),
		"status":    "queued",
		"createdAt": time.Now().UTC().Format(time.RFC3339),
	}
	phase4Contracts.mu.Lock()
	phase4Contracts.restoreJobs = append([]map[string]interface{}{restore}, phase4Contracts.restoreJobs...)
	phase4Contracts.mu.Unlock()
	writeJSON(w, http.StatusAccepted, restore)
}

func getRestoreHandler(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	phase4Contracts.mu.Lock()
	defer phase4Contracts.mu.Unlock()
	for _, restore := range phase4Contracts.restoreJobs {
		if restore["id"] == id {
			writeJSON(w, http.StatusOK, restore)
			return
		}
	}
	writeJSONError(w, http.StatusNotFound, "restore job not found")
}

func decodeJSONRequest(w http.ResponseWriter, r *http.Request, target interface{}) bool {
	if err := json.NewDecoder(r.Body).Decode(target); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid request body")
		return false
	}
	return true
}

func contractID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func defaultString(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}

func defaultInt(value int, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}

func mergeContractRecord(record map[string]interface{}, updates map[string]interface{}) {
	for key, value := range updates {
		if key == "id" || key == "createdAt" {
			continue
		}
		record[key] = value
	}
	record["updatedAt"] = time.Now().UTC().Format(time.RFC3339)
}

func cloneRecords(records []map[string]interface{}) []map[string]interface{} {
	cloned := make([]map[string]interface{}, 0, len(records))
	for _, record := range records {
		next := make(map[string]interface{}, len(record))
		for key, value := range record {
			next[key] = value
		}
		cloned = append(cloned, next)
	}
	return cloned
}

func migrationPreflightChecks(strategy string, bandwidthMbps int, maxDowntimeSeconds int) []map[string]interface{} {
	checks := []map[string]interface{}{
		{"name": "trusted_cluster", "status": "passed", "message": "target cluster must be configured as a trusted seed"},
		{"name": "storage_reachability", "status": "passed", "message": "storage copy path will be validated by the migration executor"},
	}
	if strategy == "live" {
		checks = append(checks, map[string]interface{}{"name": "live_migration_gate", "status": "warning", "message": "live migration is feature-gated in the convergence RC"})
	}
	if bandwidthMbps > 0 && bandwidthMbps < 50 {
		checks = append(checks, map[string]interface{}{"name": "bandwidth_floor", "status": "warning", "message": "bandwidth below 50 Mbps may require checkpoint or cold migration"})
	}
	if maxDowntimeSeconds > 0 && strategy == "cold" && maxDowntimeSeconds < 60 {
		checks = append(checks, map[string]interface{}{"name": "downtime_window", "status": "warning", "message": "cold migration cannot guarantee sub-minute downtime"})
	}
	return checks
}

func preflightStatus(checks []map[string]interface{}) string {
	for _, check := range checks {
		if check["status"] == "failed" {
			return "failed"
		}
	}
	for _, check := range checks {
		if check["status"] == "warning" {
			return "warning"
		}
	}
	return "passed"
}

func estimatedDowntime(strategy string, requested int) int {
	switch strategy {
	case "live":
		return defaultInt(requested, 30)
	case "checkpoint":
		return defaultInt(requested, 120)
	default:
		return defaultInt(requested, 300)
	}
}

func stringValue(value interface{}, fallback string) string {
	if typed, ok := value.(string); ok && strings.TrimSpace(typed) != "" {
		return strings.TrimSpace(typed)
	}
	return fallback
}

func numberValue(value interface{}, fallback int) int {
	switch typed := value.(type) {
	case float64:
		if typed > 0 {
			return int(typed)
		}
	case int:
		if typed > 0 {
			return typed
		}
	}
	return fallback
}

func boolValue(value interface{}, fallback bool) bool {
	if typed, ok := value.(bool); ok {
		return typed
	}
	return fallback
}
