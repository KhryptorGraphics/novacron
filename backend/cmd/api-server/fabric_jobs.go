package main

// Fabric compute jobs (P2/G2) — the thinnest surface that reuses the existing
// cluster-dispatch path end to end. A fabric job IS a Process VM: submit →
// placement (bandwidth/locality cost) → create + start on the chosen node
// (local createVMLocal or the /internal/vms/create dispatch RPC) → status and
// logs read back from the VM the Process driver runs.
//
// No second executor exists: dispatch, placement, VM lifecycle, process
// supervision, stdout/stderr capture and restart re-adoption are all the
// already-verified VM paths. This file adds only the job record + HTTP surface
// and the placement decision.

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// jobStatus* mirror the VM lifecycle as a job outcome. "pending" covers the
// window between create and start; the VM state then drives everything.
const (
	jobStatusPending   = "pending"
	jobStatusRunning   = "running"
	jobStatusCompleted = "completed"
	jobStatusFailed    = "failed"
	jobStatusCancelled = "cancelled"
	jobStatusQueued    = "queued"
)

// fabricJob is the persisted record mapping a job to its executor VM + node.
type fabricJob struct {
	ID        string    `json:"job_id"`
	Name      string    `json:"name,omitempty"`
	VMID      string    `json:"vm_id"`
	NodeID    string    `json:"node_id"`
	Command   string    `json:"command"`
	Status    string    `json:"status"`
	Error     string    `json:"error,omitempty"`
	PlacedBy  string    `json:"placed_by,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// fabricJobSpec is the submit payload.
type fabricJobSpec struct {
	Name         string            `json:"name,omitempty"`
	Command      string            `json:"command"`
	Args         []string          `json:"args,omitempty"`
	Env          map[string]string `json:"env,omitempty"`
	NodeID       string            `json:"node_id,omitempty"` // pin to a node
	BytesToMove  int               `json:"bytes_to_move,omitempty"`
	MemoryMB     int               `json:"memory_mb,omitempty"`
	VCPUs        int               `json:"vcpus,omitempty"`
	InputsNodeID string            `json:"inputs_node_id,omitempty"` // where the data already is
}

// placementDecision is the observable result of the placement cost — exposed on
// submit so the user can see WHY a node was chosen (G2: "the node the placement
// cost picks").
type placementDecision struct {
	Decision      string  `json:"decision"` // pinned | locality | cost | local-only | default
	CostEstimateS float64 `json:"cost_estimate_s,omitempty"`
	Reason        string  `json:"reason"`
}

// registerFabricJobRoutes wires the fabric job API on the authed /api router.
func registerFabricJobRoutes(apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	apiRouter.HandleFunc("/compute/jobs", func(w http.ResponseWriter, r *http.Request) {
		var spec fabricJobSpec
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&spec); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		spec.Command = strings.TrimSpace(spec.Command)
		if spec.Command == "" {
			writeJSONError(w, http.StatusBadRequest, "command is required")
			return
		}
		job, decision, err := submitFabricJob(r.Context(), db, vmManager, storagePath, spec)
		if err != nil {
			code := http.StatusInternalServerError
			if errors.Is(err, errNoNodeFits) || errors.Is(err, errPinnedNodeUnreachable) {
				code = http.StatusServiceUnavailable
			}
			writeJSONError(w, code, err.Error())
			return
		}
		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"job_id": job.ID, "vm_id": job.VMID, "node_id": job.NodeID,
			"status": job.Status, "placement": decision,
		})
	}).Methods(http.MethodPost)

	apiRouter.HandleFunc("/compute/jobs", func(w http.ResponseWriter, r *http.Request) {
		jobs, err := listFabricJobs(r.Context(), db, vmManager)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, err.Error())
			return
		}
		out := make([]map[string]interface{}, 0, len(jobs))
		for i := range jobs {
			j := &jobs[i]
			out = append(out, map[string]interface{}{
				"job_id": j.ID, "name": j.Name, "command": j.Command,
				"status":  liveJobStatus(vmManager, j),
				"node_id": j.NodeID, "vm_id": j.VMID,
				"created_at": j.CreatedAt.UTC().Format(time.RFC3339),
				"error":      j.Error,
			})
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"jobs": out})
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/compute/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		job, err := getFabricJob(r.Context(), db, mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "job not found")
			return
		}
		status := liveJobStatus(vmManager, job)
		payload := map[string]interface{}{
			"job_id": job.ID, "name": job.Name, "command": job.Command,
			"status": status, "node_id": job.NodeID, "vm_id": job.VMID,
			"created_at": job.CreatedAt.UTC().Format(time.RFC3339),
			"error":      job.Error,
		}
		// Logs exist only where the process runs: locally when the job's node
		// is this node; a peer's logs are fetched through its own API surface.
		if job.NodeID == selfNodeID() || job.NodeID == "" {
			payload["logs"] = processLogTails(storagePath, job.VMID)
		} else if addr := vmManager.MigrationPeers()[job.NodeID]; addr != "" {
			if remote, err := fetchRemoteJobLogs(addr, job.VMID); err == nil {
				payload["logs"] = remote
			} else {
				payload["logs"] = map[string]string{"stdout": "", "stderr": ""}
				payload["logs_error"] = err.Error()
			}
		}
		writeJSON(w, http.StatusOK, payload)
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/compute/jobs/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {
		job, err := getFabricJob(r.Context(), db, mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "job not found")
			return
		}
		var cancelErr error
		if job.NodeID == selfNodeID() || job.NodeID == "" {
			cancelErr = vmManager.StopVM(r.Context(), job.VMID)
		} else if addr := vmManager.MigrationPeers()[job.NodeID]; addr != "" {
			cancelErr = cancelRemoteJob(addr, job.VMID)
		} else {
			cancelErr = fmt.Errorf("job's node %q is not registered", job.NodeID)
		}
		if cancelErr != nil {
			writeJSON(w, http.StatusBadGateway, map[string]interface{}{
				"cancelled": false, "status": "failed-to-cancel", "error": cancelErr.Error(),
			})
			return
		}
		if db != nil {
			_, _ = db.ExecContext(r.Context(),
				`UPDATE fabric_jobs SET status = $1, updated_at = NOW() WHERE id = $2`,
				jobStatusCancelled, job.ID)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"cancelled": true, "status": jobStatusCancelled})
	}).Methods(http.MethodPost)
}

var (
	errNoNodeFits            = errors.New("no node in the fabric can fit the job")
	errPinnedNodeUnreachable = errors.New("pinned node is not reachable")
)

// submitFabricJob places and starts one job, returning the persisted record and
// the placement decision that chose its node.
func submitFabricJob(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, storagePath string, spec fabricJobSpec) (*fabricJob, *placementDecision, error) {
	jobID := uuid.NewString()
	spec.MemoryMB = clampJobMemory(spec.MemoryMB)
	jobVCPUs := spec.VCPUs
	if jobVCPUs <= 0 {
		jobVCPUs = 1
	}

	node, decision, err := placeFabricJob(vmManager, storagePath, spec)
	if err != nil {
		return nil, nil, err
	}

	// The job's VM is created with the Process driver (Command non-empty), on
	// the selected node: local create, or dispatch RPC to the peer.
	createSpec := clusterCreateSpec{
		Name:     fabricJobName(spec, jobID),
		MemoryMB: spec.MemoryMB,
		VCPUs:    jobVCPUs,
		Command:  spec.Command,
		Args:     spec.Args,
		Env:      spec.Env,
		Tags:     map[string]interface{}{"fabric_job_id": jobID},
	}

	var vmID string
	if node.Addr == "" || node.NodeID == selfNodeID() {
		var state string
		vmID, state, err = createVMLocal(ctx, db, vmManager, createSpec)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create job VM locally: %w", err)
		}
		_ = state
	} else {
		out, derr := dispatchCreateToPeer(node.Addr, createSpec)
		if derr != nil {
			return nil, nil, fmt.Errorf("failed to dispatch job to %s: %w", node.NodeID, derr)
		}
		vmID, _ = out["id"].(string)
		if vmID == "" {
			return nil, nil, fmt.Errorf("peer %s returned no VM id for the job", node.NodeID)
		}
	}

	// Start it: local start, or the peer's own start RPC.
	if node.Addr == "" || node.NodeID == selfNodeID() {
		if err := vmManager.StartVM(ctx, vmID); err != nil {
			// Roll back the orphaned VM so a failed start leaves no debris.
			_ = vmManager.DeleteVM(context.Background(), vmID)
			if db != nil {
				_, _ = db.ExecContext(ctx, `DELETE FROM vms WHERE id = $1`, vmID)
			}
			return nil, nil, fmt.Errorf("failed to start job VM: %w", err)
		}
	} else if err := startRemoteJob(node.Addr, vmID); err != nil {
		_ = deleteRemoteJob(node.Addr, vmID)
		return nil, nil, fmt.Errorf("failed to start job on %s: %w", node.NodeID, err)
	}

	job := &fabricJob{
		ID: jobID, Name: spec.Name, VMID: vmID, NodeID: node.NodeID,
		Command: spec.Command, Status: jobStatusRunning,
		PlacedBy: decision.Decision, CreatedAt: time.Now().UTC(),
	}
	if db != nil {
		if _, err := db.ExecContext(ctx, `
			INSERT INTO fabric_jobs (id, vm_id, node_id, command, name, status, placed_by, created_at, updated_at)
			VALUES ($1, $2, $3, $4, NULLIF($5, ''), $6, $7, NOW(), NOW())
		`, job.ID, job.VMID, job.NodeID, job.Command, job.Name, job.Status, job.PlacedBy); err != nil {
			// Created and started, but not recorded: stop it so no untracked
			// process keeps running, and surface the failure.
			if node.NodeID == selfNodeID() {
				_ = vmManager.StopVM(context.Background(), vmID)
			} else {
				_ = cancelRemoteJob(node.Addr, vmID)
			}
			return nil, nil, fmt.Errorf("failed to persist job record: %w", err)
		}
	}
	return job, decision, nil
}

// placeFabricJob picks the execution node. Order of precedence:
//  1. explicit pin (node_id) — data-locality floor: the caller knows where the
//     work must run; an unreachable pin is an error, never a silent move.
//  2. inputs_node_id — the node that already holds the inputs, unless its
//     link/locality cost is worse than running elsewhere AND bytes_to_move
//     makes the transfer dominate (see estimatedMoveCostS).
//  3. cost-optimal: min estimated transfer time over reachable nodes that fit.
//  4. single reachable node → it (default).
func placeFabricJob(vmManager *core_vm.VMManager, storagePath string, spec fabricJobSpec) (NodeCapacity, *placementDecision, error) {
	nodes := nodeProfiles(vmManager, storagePath, nil)
	byID := make(map[string]nodeProfile, len(nodes))
	for _, n := range nodes {
		byID[n.NodeID] = n
	}

	if pin := strings.TrimSpace(spec.NodeID); pin != "" {
		n, ok := byID[pin]
		if !ok || !n.Reachable {
			return NodeCapacity{}, nil, fmt.Errorf("%w: %q", errPinnedNodeUnreachable, pin)
		}
		return n.NodeCapacity, &placementDecision{
			Decision: "pinned",
			Reason:   fmt.Sprintf("node %s pinned by caller", pin),
		}, nil
	}

	if src := strings.TrimSpace(spec.InputsNodeID); src != "" {
		if n, ok := byID[src]; ok && n.Reachable && n.memAvailMB() >= int64(spec.MemoryMB) {
			return n.NodeCapacity, &placementDecision{
				Decision: "locality",
				Reason:   fmt.Sprintf("inputs live on %s; running there moves zero bytes", src),
			}, nil
		}
	}

	memMB := int64(spec.MemoryMB)
	candidates := make([]nodeProfile, 0, len(nodes))
	for _, n := range nodes {
		if n.Reachable && n.MemTotalMB > 0 && n.memAvailMB() >= memMB {
			candidates = append(candidates, n)
		}
	}
	if len(candidates) == 0 {
		return NodeCapacity{}, nil, errNoNodeFits
	}
	if len(candidates) == 1 {
		c := candidates[0]
		return c.NodeCapacity, &placementDecision{
			Decision: "default",
			Reason:   fmt.Sprintf("only reachable node with capacity: %s", c.NodeID),
		}, nil
	}

	// Cost-optimal: bytes_to_move / measured link bps + estimated run time
	// proxy (memory size as a stand-in when no measured runtime exists).
	type scored struct {
		n    nodeProfile
		cost float64 // seconds
	}
	scoredNodes := make([]scored, 0, len(candidates))
	for _, c := range candidates {
		scoredNodes = append(scoredNodes, scored{n: c, cost: estimatedMoveCostS(c, spec)})
	}
	sort.SliceStable(scoredNodes, func(i, j int) bool { return scoredNodes[i].cost < scoredNodes[j].cost })
	best := scoredNodes[0]
	return best.n.NodeCapacity, &placementDecision{
		Decision:      "cost",
		CostEstimateS: best.cost,
		Reason:        fmt.Sprintf("lowest estimated cost (%.3fs): transfer %d bytes over %s link", best.cost, spec.BytesToMove, linkBandwidthLabel(best.n)),
	}, nil
}

// estimatedMoveCostS is bytes/link_bps plus a fixed run-time proxy. The link
// term uses the measured RTT as the only live signal available on this path
// today; when a throughput probe lands (P3) it replaces rttToBps verbatim.
func estimatedMoveCostS(n nodeProfile, spec fabricJobSpec) float64 {
	bytes := float64(spec.BytesToMove)
	linkBps := rttToBps(n)
	transfer := 0.0
	if bytes > 0 {
		transfer = bytes / linkBps
	}
	// Run-time proxy: guest memory footprint / 1 GiB/s baseline.
	run := float64(spec.MemoryMB) / 1024.0
	return transfer + run
}

// rttToBps converts a link profile to a capacity estimate for placement:
// a FRESH measured throughput wins; otherwise the estimate degrades to the
// RTT heuristic (documented placeholder: it ranks LAN ahead of WAN, which is
// what ordering needs), and an unmeasured link ranks as fast-local.
func rttToBps(n nodeProfile) float64 {
	if n.Link != nil && n.Link.ThroughputBps != nil && !n.Link.Stale && *n.Link.ThroughputBps > 0 {
		return *n.Link.ThroughputBps
	}
	if n.Link == nil || n.Link.RTTMS <= 0 {
		return 1e9 // unknown link: treat as fast local, not as zero
	}
	// 1 Gbps at ≤1ms, degrading inversely with RTT; floor at 10 Mbps so a huge
	// RTT never produces division-by-tiny.
	est := 1e9 * (1.0 / n.Link.RTTMS)
	if est < 1e7 {
		est = 1e7
	}
	return est
}

func linkBandwidthLabel(n nodeProfile) string {
	if n.Link == nil {
		return "unmeasured"
	}
	if n.Link.ThroughputBps != nil {
		state := "measured"
		if n.Link.Stale {
			state = "stale"
		}
		return fmt.Sprintf("%s %.1f Mbps", state, *n.Link.ThroughputBps/1e6)
	}
	if n.Link.RTTMS > 0 {
		return fmt.Sprintf("rtt %.2fms (throughput unmeasured)", n.Link.RTTMS)
	}
	return "unmeasured"
}

func clampJobMemory(mb int) int {
	if mb <= 0 {
		return 128
	}
	if mb > 32768 {
		return 32768
	}
	return mb
}

func fabricJobName(spec fabricJobSpec, jobID string) string {
	if strings.TrimSpace(spec.Name) != "" {
		return spec.Name
	}
	base := filepath.Base(strings.Fields(spec.Command)[0])
	return fmt.Sprintf("fabric-job-%s-%s", base, jobID[:8])
}

// liveJobStatus derives a job's outcome from DRIVER truth (pid liveness +
// recorded exit code), not from the manager's cached VM state: the manager's
// maintenance loop is a placeholder, so vm.State() stays frozen at whatever
// create/start set and a finished process would read "running" forever.
//
// A job placed on a peer is resolved through that peer's status RPC: the local
// manager does not know the VM at all, and falling back to the stored row
// would report "running" forever alongside an already-written log tail.
func liveJobStatus(vmManager *core_vm.VMManager, job *fabricJob) string {
	if job.Status == jobStatusCancelled {
		return jobStatusCancelled
	}
	if vmManager == nil {
		return job.Status
	}
	if job.NodeID != "" && job.NodeID != selfNodeID() {
		addr := vmManager.MigrationPeers()[job.NodeID]
		if addr == "" {
			return job.Status
		}
		remote, err := fetchRemoteJobStatus(addr, job.VMID)
		if err != nil {
			return job.Status
		}
		return remote.effectiveStatus(job.Status)
	}
	return localJobStatus(vmManager, job)
}

// localJobStatus is liveJobStatus against THIS node's manager.
func localJobStatus(vmManager *core_vm.VMManager, job *fabricJob) string {
	vm, err := vmManager.GetVM(job.VMID)
	if err != nil {
		return job.Status
	}
	drv, derr := vmManager.GetDriverForConfig(vm.Config())
	if derr != nil {
		return job.Status
	}
	st, serr := drv.GetStatus(context.Background(), job.VMID)
	if serr != nil {
		return job.Status
	}
	switch st {
	case core_vm.StateRunning, core_vm.StateStarting:
		return jobStatusRunning
	case core_vm.StateStopped:
		// The process is gone. The exit code tells success from failure; when
		// the driver doesn't record it (non-process VM) a started job counts
		// as completed.
		if pd, ok := drv.(*core_vm.ProcessDriver); ok {
			if code, has := pd.ExitCode(job.VMID); has {
				if code == 0 {
					return jobStatusCompleted
				}
				return jobStatusFailed
			}
		}
		if job.Status == jobStatusRunning {
			return jobStatusCompleted
		}
		return job.Status
	case core_vm.StateUnknown, core_vm.StateDeleting:
		return jobStatusFailed
	default:
		return job.Status
	}
}

// remoteVMStatus is the peer's answer for one VM.
type remoteVMStatus struct {
	State    string `json:"state"` // running | stopped | failed | unknown
	ExitCode *int   `json:"exit_code,omitempty"`
	Error    string `json:"error,omitempty"`
}

// effectiveStatus maps a peer-reported VM state onto a job status, with the
// stored row as the fallback for states the peer couldn't determine.
func (r remoteVMStatus) effectiveStatus(stored string) string {
	switch r.State {
	case "running":
		return jobStatusRunning
	case "stopped":
		if r.ExitCode != nil && *r.ExitCode != 0 {
			return jobStatusFailed
		}
		if stored == jobStatusRunning {
			return jobStatusCompleted
		}
		return stored
	case "failed":
		return jobStatusFailed
	default:
		return stored
	}
}

// fetchRemoteJobStatus asks a peer for driver-truth status of a VM it owns.
func fetchRemoteJobStatus(addr, vmID string) (remoteVMStatus, error) {
	var out remoteVMStatus
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+"/internal/fabric/vm-status/"+vmID, nil)
	if err != nil {
		return out, err
	}
	if secret := os.Getenv("NOVACRON_MIGRATION_SECRET"); secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("status RPC %s: %s", addr, resp.Status)
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&out); err != nil {
		return out, err
	}
	return out, nil
}

// localVMStatus answers the peer-facing status RPC for a VM this node owns.
func localVMStatus(vmManager *core_vm.VMManager, vmID string) remoteVMStatus {
	var out remoteVMStatus
	if vmManager == nil {
		out.State = "unknown"
		out.Error = "no vm manager"
		return out
	}
	vm, err := vmManager.GetVM(vmID)
	if err != nil {
		out.State = "unknown"
		out.Error = "vm not found on this node"
		return out
	}
	drv, derr := vmManager.GetDriverForConfig(vm.Config())
	if derr != nil {
		out.State = "unknown"
		out.Error = derr.Error()
		return out
	}
	st, serr := drv.GetStatus(context.Background(), vmID)
	if serr != nil {
		out.State = "unknown"
		out.Error = serr.Error()
		return out
	}
	switch st {
	case core_vm.StateRunning, core_vm.StateStarting:
		out.State = "running"
	case core_vm.StateStopped:
		out.State = "stopped"
		if pd, ok := drv.(*core_vm.ProcessDriver); ok {
			if code, has := pd.ExitCode(vmID); has {
				c := code
				out.ExitCode = &c
			}
		}
	case core_vm.StateUnknown, core_vm.StateDeleting:
		out.State = "failed"
	default:
		out.State = "unknown"
	}
	return out
}

// getFabricJob loads one job row.
func getFabricJob(ctx context.Context, db *sql.DB, id string) (*fabricJob, error) {
	if db == nil {
		return nil, errors.New("no database")
	}
	var j fabricJob
	var created time.Time
	err := db.QueryRowContext(ctx, `
		SELECT id, vm_id, node_id, command, COALESCE(name,''), status, COALESCE(error,''), COALESCE(placed_by,''), created_at
		FROM fabric_jobs WHERE id = $1`, id).
		Scan(&j.ID, &j.VMID, &j.NodeID, &j.Command, &j.Name, &j.Status, &j.Error, &j.PlacedBy, &created)
	if err != nil {
		return nil, err
	}
	j.CreatedAt = created
	return &j, nil
}

// listFabricJobs returns the most recent jobs, newest first.
func listFabricJobs(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager) ([]fabricJob, error) {
	if db == nil {
		return nil, errors.New("no database")
	}
	rows, err := db.QueryContext(ctx, `
		SELECT id, vm_id, node_id, command, COALESCE(name,''), status, COALESCE(error,''), COALESCE(placed_by,''), created_at
		FROM fabric_jobs ORDER BY created_at DESC LIMIT 200`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]fabricJob, 0, 16)
	for rows.Next() {
		var j fabricJob
		if err := rows.Scan(&j.ID, &j.VMID, &j.NodeID, &j.Command, &j.Name, &j.Status, &j.Error, &j.PlacedBy, &j.CreatedAt); err != nil {
			return nil, err
		}
		// Meter once when a job reaches a terminal state. The stored job row
		// never flips from 'running' — the status is computed fresh from the
		// driver on every read — so we gate on a usage_events marker keyed on
		// the job's ID to avoid double-billing on retries.
		if j.ID != "" {
			status := liveJobStatus(vmManager, &j)
			if j.Status != status && (status == jobStatusCompleted || status == jobStatusFailed) {
				maybeMeterJobOnce(db, &j, status, vmManager)
				// Advance the row to match the observed terminal state.
				_, _ = db.ExecContext(context.Background(), `UPDATE fabric_jobs SET status = $1, updated_at = NOW() WHERE id = $2`, status, j.ID)
			}
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// processLogTails reads the tail of the Process driver's captured output for a
// VM. Missing files are empty, not an error (the process may not have written
// anything yet).
func processLogTails(storagePath, vmID string) map[string]string {
	const tailBytes = 64 << 10
	read := func(name string) string {
		path := filepath.Join(storagePath, "processes", vmID, name)
		f, err := os.Open(path)
		if err != nil {
			return ""
		}
		defer f.Close()
		st, err := f.Stat()
		if err != nil {
			return ""
		}
		off := int64(0)
		if st.Size() > tailBytes {
			off = st.Size() - tailBytes
		}
		if _, err := f.Seek(off, io.SeekStart); err != nil {
			return ""
		}
		b, err := io.ReadAll(io.LimitReader(f, tailBytes))
		if err != nil {
			return ""
		}
		return string(b)
	}
	return map[string]string{"stdout": read("stdout.log"), "stderr": read("stderr.log")}
}

// fetchRemoteJobLogs pulls a peer's log tail through its own API. The peer
// exposes log tails on the same authed route, so the caller's JWT is forwarded.
func fetchRemoteJobLogs(addr, vmID string) (map[string]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+"/internal/fabric/vm-logs/"+vmID, nil)
	if err != nil {
		return nil, err
	}
	if secret := os.Getenv("NOVACRON_MIGRATION_SECRET"); secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("log fetch from %s: %s", addr, resp.Status)
	}
	var out map[string]string
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// startRemoteJob tells a peer to start a VM it owns.
func startRemoteJob(addr, vmID string) error {
	return postInternalVM(addr, "/internal/fabric/vm-start", map[string]string{"vm_id": vmID})
}

// cancelRemoteJob tells a peer to stop a VM it owns.
func cancelRemoteJob(addr, vmID string) error {
	return postInternalVM(addr, "/internal/fabric/vm-stop", map[string]string{"vm_id": vmID})
}

// deleteRemoteJob tells a peer to delete a VM it owns (rollback path).
func deleteRemoteJob(addr, vmID string) error {
	return postInternalVM(addr, "/internal/fabric/vm-delete", map[string]string{"vm_id": vmID})
}

func postInternalVM(addr, path string, payload map[string]string) error {
	body, _ := json.Marshal(payload)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+path, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if secret := os.Getenv("NOVACRON_MIGRATION_SECRET"); secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return fmt.Errorf("%s: %s: %s", path, resp.Status, strings.TrimSpace(string(b)))
	}
	return nil
}

// registerFabricNodeRPCs wires the node-to-node helpers the job API needs on a
// peer: start/stop/delete of a VM this node owns, plus its process log tails.
// All are shared-secret gated, like every other /internal route.
func registerFabricNodeRPCs(root *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	vmOp := func(action string, fn func(context.Context, string) error) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if !internalSecretOK(r) {
				writeJSONError(w, http.StatusForbidden, "forbidden")
				return
			}
			var req struct {
				VMID string `json:"vm_id"`
			}
			if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&req); err != nil || strings.TrimSpace(req.VMID) == "" {
				writeJSONError(w, http.StatusBadRequest, "vm_id is required")
				return
			}
			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()
			if err := fn(ctx, req.VMID); err != nil {
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("%s %s: %v", action, req.VMID, err))
				return
			}
			writeJSON(w, http.StatusOK, map[string]interface{}{action: true, "vm_id": req.VMID})
		}
	}
	root.HandleFunc("/internal/fabric/vm-start", vmOp("started", vmManager.StartVM)).Methods(http.MethodPost)
	root.HandleFunc("/internal/fabric/vm-stop", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		var req struct {
			VMID string `json:"vm_id"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&req); err != nil || strings.TrimSpace(req.VMID) == "" {
			writeJSONError(w, http.StatusBadRequest, "vm_id is required")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		if err := vmManager.StopVM(ctx, req.VMID); err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("stop %s: %v", req.VMID, err))
			return
		}
		// A remote stop is a job cancel: record it on the local job row too,
		// otherwise this node would keep showing the job as running until its
		// driver-truth reconcile reports the killed process as failed.
		if db != nil {
			if _, err := db.ExecContext(ctx,
				`UPDATE fabric_jobs SET status = $1, updated_at = NOW() WHERE vm_id = $2 AND status <> $3`,
				jobStatusCancelled, req.VMID, jobStatusCancelled); err != nil {
				log.Printf("cancel marking failed for vm %s: %v", req.VMID, err)
			}
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"stopped": true, "vm_id": req.VMID})
	}).Methods(http.MethodPost)
	root.HandleFunc("/internal/fabric/vm-delete", vmOp("deleted", vmManager.DeleteVM)).Methods(http.MethodPost)

	root.HandleFunc("/internal/fabric/vm-logs/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		writeJSON(w, http.StatusOK, processLogTails(storagePath, mux.Vars(r)["id"]))
	}).Methods(http.MethodGet)

	// Driver-truth status for one VM this node owns: pid liveness plus the
	// recorded exit code. The job API on another node reads this to report a
	// remote job's outcome instead of a stale stored row.
	root.HandleFunc("/internal/fabric/vm-status/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		writeJSON(w, http.StatusOK, localVMStatus(vmManager, mux.Vars(r)["id"]))
	}).Methods(http.MethodGet)
}
