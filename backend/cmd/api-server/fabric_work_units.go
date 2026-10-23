package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// Fabric work units: a fabric job plus content-addressed inputs and a
// checkpoint that survives a failed attempt.
//
// The unit IS a Process-driver VM (the existing job executor) started with five
// reserved environment variables. Inputs are addressed by SHA-256, verified on
// arrival at the executing node, and never re-sent once that node's cache holds
// a byte-identical copy. A retry restarts the SAME VM with the SAME checkpoint
// path and a bumped attempt counter, so the workload resumes where it stopped.
//
// Honest scope: the workload is a native process running as the api-server's
// account, so submission and retry are admin-only. This is trusted-operator
// execution, not a sandbox. Resumability is the workload's responsibility: the
// platform guarantees the checkpoint path is unchanged and the attempt number
// advances; it cannot know what an application wrote there.

const (
	maxWorkUnitArtifactBytes int64 = 1 << 30
	maxWorkUnitInputBytes    int64 = 4 << 30
	maxWorkUnitArtifacts           = 64
	maxWorkUnitAttempts            = 10

	// defaultWorkUnitCacheBytes bounds the per-node artifact cache. Without a
	// bound the cache grows by one copy of every input ever submitted.
	defaultWorkUnitCacheBytes int64 = 10 << 30
	// workUnitCacheEvictGrace keeps an artifact that was uploaded but not yet
	// submitted: eviction only considers files older than this.
	workUnitCacheEvictGrace = 10 * time.Minute
	// workUnitCacheScanLimit bounds the reference lookup; a truncated scan
	// cannot prove an artifact unreferenced, so it prunes nothing.
	workUnitCacheScanLimit = 5000
)

var (
	errWorkUnitArtifactTooLarge     = errors.New("work-unit artifact exceeds the 1 GiB limit")
	errWorkUnitArtifactHashMismatch = errors.New("work-unit artifact SHA-256 does not match its address")
	errWorkUnitInputTooLarge        = errors.New("work-unit inputs exceed the 4 GiB limit")
	errWorkUnitAttemptsExhausted    = errors.New("work-unit attempt limit is exhausted")
	errWorkUnitRunning              = errors.New("work unit is still running")
	errWorkUnitInternal             = errors.New("work-unit storage failure")
)

// Work-unit environment contract. These keys are reserved: a caller cannot set
// them, so the runner always agrees with the platform about where inputs and
// resume state live.
const (
	envWorkUnitArtifactDir = "NOVACRON_ARTIFACT_DIR"
	envWorkUnitArtifacts   = "NOVACRON_ARTIFACTS"
	envWorkUnitCheckpoint  = "NOVACRON_CHECKPOINT_PATH"
	envWorkUnitAttemptFile = "NOVACRON_ATTEMPT_FILE"
	envWorkUnitID          = "NOVACRON_WORK_UNIT_ID"
)

var workUnitReservedEnv = []string{
	envWorkUnitArtifactDir, envWorkUnitArtifacts, envWorkUnitCheckpoint, envWorkUnitAttemptFile, envWorkUnitID,
}

// workUnitSpec is the submit payload.
type workUnitSpec struct {
	Name           string            `json:"name,omitempty"`
	Command        string            `json:"command"`
	Args           []string          `json:"args,omitempty"`
	Env            map[string]string `json:"env,omitempty"`
	ArtifactHashes []string          `json:"artifact_hashes"`
	NodeID         string            `json:"node_id,omitempty"`
	InputsNodeID   string            `json:"inputs_node_id,omitempty"`
	MemoryMB       int               `json:"memory_mb,omitempty"`
	VCPUs          int               `json:"vcpus,omitempty"`
	AttemptLimit   int               `json:"attempt_limit,omitempty"`
}

// workUnitRecord is one persisted unit, read back with its resume state.
type workUnitRecord struct {
	ID              string
	VMID            string
	NodeID          string
	Command         string
	Name            string
	Status          string
	Error           string
	Attempts        int
	AttemptLimit    int
	ArtifactHashes  []string
	InputBytes      int64
	CheckpointPath  string
	CheckpointState string
	CreatedAt       time.Time
	Organization    sql.NullString
}

type workUnitCheckpoint struct {
	Available bool  `json:"available"`
	SizeBytes int64 `json:"size_bytes"`
}

type workUnitPrepareRequest struct {
	WorkUnitID     string            `json:"work_unit_id"`
	ArtifactHashes []string          `json:"artifact_hashes"`
	Attempt        int               `json:"attempt,omitempty"`
	Env            map[string]string `json:"env,omitempty"`
}

// registerFabricWorkUnitRoutes wires the admin-only work-unit surface and the
// node-to-node artifact/prepare/checkpoint RPCs.
func registerFabricWorkUnitRoutes(root, apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	apiRouter.HandleFunc("/compute/work-units", func(w http.ResponseWriter, r *http.Request) {
		if !workUnitAdmin(r.Context()) {
			writeJSONError(w, http.StatusForbidden, "work-unit execution is restricted to admins")
			return
		}
		var spec workUnitSpec
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&spec); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		spec.Command = strings.TrimSpace(spec.Command)
		if spec.Command == "" {
			writeJSONError(w, http.StatusBadRequest, "command is required")
			return
		}
		rec, decision, err := submitWorkUnit(r.Context(), db, vmManager, storagePath, spec)
		if err != nil {
			writeJSONError(w, workUnitErrorStatus(err), err.Error())
			return
		}
		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"work_unit_id": rec.ID, "job_id": rec.ID, "vm_id": rec.VMID, "node_id": rec.NodeID,
			"status": rec.Status, "attempts": rec.Attempts, "attempt_limit": rec.AttemptLimit,
			"artifact_hashes": rec.ArtifactHashes, "input_bytes": rec.InputBytes,
			"placement": decision,
		})
	}).Methods(http.MethodPost)

	apiRouter.HandleFunc("/compute/work-units/{id}", func(w http.ResponseWriter, r *http.Request) {
		rec, err := getWorkUnit(r.Context(), db, mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "work unit not found")
			return
		}
		scopeOrg, isAdmin, _ := requireOrgScope(r.Context(), nil, "")
		if !isAdmin && !orgVisible(scopeOrg, rec.Organization) {
			writeJSONError(w, http.StatusNotFound, "work unit not found")
			return
		}
		status := workUnitLiveStatus(vmManager, rec)
		checkpoint := workUnitCheckpointState(vmManager, storagePath, rec)
		payload := map[string]interface{}{
			"work_unit_id": rec.ID, "job_id": rec.ID, "vm_id": rec.VMID, "node_id": rec.NodeID,
			"name": rec.Name, "command": rec.Command, "status": status, "error": rec.Error,
			"attempts": rec.Attempts, "attempt_limit": rec.AttemptLimit,
			"artifact_hashes": rec.ArtifactHashes, "input_bytes": rec.InputBytes,
			"checkpoint": checkpoint, "checkpoint_state": rec.CheckpointState,
			"created_at": rec.CreatedAt.UTC().Format(time.RFC3339),
		}
		if rec.NodeID == selfNodeID() || rec.NodeID == "" {
			payload["logs"] = processLogTails(storagePath, rec.VMID)
		} else if addr := vmManager.MigrationPeers()[rec.NodeID]; addr != "" {
			if remote, err := fetchRemoteJobLogs(addr, rec.VMID); err == nil {
				payload["logs"] = remote
			} else {
				payload["logs_error"] = err.Error()
			}
		}
		writeJSON(w, http.StatusOK, payload)
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/compute/work-units/{id}/retry", func(w http.ResponseWriter, r *http.Request) {
		if !workUnitAdmin(r.Context()) {
			writeJSONError(w, http.StatusForbidden, "work-unit retry is restricted to admins")
			return
		}
		rec, err := getWorkUnit(r.Context(), db, mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "work unit not found")
			return
		}
		if err := retryWorkUnit(r.Context(), db, vmManager, storagePath, rec); err != nil {
			writeJSONError(w, workUnitErrorStatus(err), err.Error())
			return
		}
		updated, err := getWorkUnit(r.Context(), db, rec.ID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "work unit restarted but could not be re-read")
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"work_unit_id": updated.ID, "status": workUnitLiveStatus(vmManager, updated),
			"attempts": updated.Attempts, "attempt_limit": updated.AttemptLimit,
			"checkpoint": workUnitCheckpointState(vmManager, storagePath, updated),
		})
	}).Methods(http.MethodPost)

	apiRouter.HandleFunc("/compute/work-units/artifacts", func(w http.ResponseWriter, r *http.Request) {
		if !workUnitAdmin(r.Context()) {
			writeJSONError(w, http.StatusForbidden, "work-unit artifacts are restricted to admins")
			return
		}
		if r.ContentLength > maxWorkUnitArtifactBytes {
			writeJSONError(w, http.StatusRequestEntityTooLarge, errWorkUnitArtifactTooLarge.Error())
			return
		}
		sha256Hex, size, cached, err := storeWorkUnitArtifact(storagePath, r.Body)
		if err != nil {
			writeJSONError(w, workUnitErrorStatus(err), err.Error())
			return
		}
		pruneWorkUnitArtifacts(r.Context(), db, storagePath)
		status := http.StatusCreated
		if cached {
			status = http.StatusOK
		}
		writeJSON(w, status, map[string]interface{}{
			"sha256": sha256Hex, "size_bytes": size, "cached": cached,
		})
	}).Methods(http.MethodPost)

	// HEAD reports whether the node already holds a byte-identical copy; PUT
	// stages one under its SHA-256 address and rejects a body that hashes
	// differently, so a corrupted transfer never enters the cache.
	root.HandleFunc("/internal/fabric/work-units/artifacts/{hash}", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		hash := mux.Vars(r)["hash"]
		if !validWorkUnitArtifactHash(hash) {
			writeJSONError(w, http.StatusBadRequest, "invalid artifact hash")
			return
		}
		if r.Method == http.MethodHead {
			size, err := verifyWorkUnitArtifact(storagePath, hash)
			if err != nil {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.ContentLength > maxWorkUnitArtifactBytes {
			writeJSONError(w, http.StatusRequestEntityTooLarge, errWorkUnitArtifactTooLarge.Error())
			return
		}
		_, size, cached, err := storeWorkUnitArtifactAtHash(storagePath, hash, r.Body)
		if err != nil {
			writeJSONError(w, workUnitErrorStatus(err), err.Error())
			return
		}
		status := http.StatusCreated
		if cached {
			status = http.StatusOK
		}
		writeJSON(w, status, map[string]interface{}{"size_bytes": size, "cached": cached})
	}).Methods(http.MethodHead, http.MethodPut)

	root.HandleFunc("/internal/fabric/work-units/prepare", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		var req workUnitPrepareRequest
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid prepare request")
			return
		}
		env, err := prepareWorkUnitEnvironment(storagePath, req.WorkUnitID, req.ArtifactHashes, req.Attempt, req.Env)
		if err != nil {
			writeJSONError(w, workUnitErrorStatus(err), err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"env": env})
	}).Methods(http.MethodPost)

	root.HandleFunc("/internal/fabric/work-units/{id}/checkpoint", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		id := mux.Vars(r)["id"]
		if _, err := uuid.Parse(id); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid work-unit id")
			return
		}
		state, err := inspectWorkUnitCheckpoint(storagePath, id)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to inspect checkpoint")
			return
		}
		writeJSON(w, http.StatusOK, state)
	}).Methods(http.MethodGet)
}

func workUnitAdmin(ctx context.Context) bool {
	_, isAdmin, _ := requireOrgScope(ctx, nil, "")
	return isAdmin
}

func workUnitErrorStatus(err error) int {
	switch {
	case errors.Is(err, errWorkUnitArtifactTooLarge), errors.Is(err, errWorkUnitInputTooLarge):
		return http.StatusRequestEntityTooLarge
	case errors.Is(err, errWorkUnitAttemptsExhausted), errors.Is(err, errWorkUnitRunning):
		return http.StatusConflict
	case errors.Is(err, errWorkUnitInternal):
		return http.StatusInternalServerError
	case errors.Is(err, errNoNodeFits), errors.Is(err, errPinnedNodeUnreachable):
		return http.StatusServiceUnavailable
	default:
		return http.StatusBadRequest
	}
}

// placementBytes converts an input total for the placement cost without
// overflowing a 32-bit int; the cost only ranks nodes, so the clamp is safe.
func placementBytes(total int64) int {
	if int64(int(total)) != total {
		return int(^uint(0) >> 1)
	}
	return int(total)
}

// submitWorkUnit places, stages, and starts one unit. Order matters: artifacts
// are validated in THIS node's cache first (the submitter uploads them here),
// then staged onto the node that will run them, then the VM is created with the
// environment that node computed for itself.
func submitWorkUnit(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, storagePath string, spec workUnitSpec) (*workUnitRecord, *placementDecision, error) {
	if db == nil {
		return nil, nil, errors.New("work units need a database")
	}
	hashes, err := normalizedWorkUnitHashes(spec.ArtifactHashes)
	if err != nil {
		return nil, nil, err
	}
	inputBytes, err := validateWorkUnitArtifacts(storagePath, hashes)
	if err != nil {
		return nil, nil, err
	}
	attemptLimit := spec.AttemptLimit
	if attemptLimit <= 0 {
		attemptLimit = 1
	}
	if attemptLimit > maxWorkUnitAttempts {
		attemptLimit = maxWorkUnitAttempts
	}

	workUnitID := uuid.NewString()
	jobSpec := fabricJobSpec{
		Name: spec.Name, Command: spec.Command, Args: spec.Args, Env: spec.Env,
		NodeID: spec.NodeID, InputsNodeID: spec.InputsNodeID,
		BytesToMove: placementBytes(inputBytes), MemoryMB: spec.MemoryMB, VCPUs: spec.VCPUs,
	}
	node, decision, err := placeFabricJob(db, vmManager, storagePath, jobSpec)
	if err != nil {
		return nil, nil, err
	}
	if err := stageWorkUnitArtifacts(node, storagePath, hashes); err != nil {
		return nil, nil, err
	}
	env, err := prepareWorkUnitOnNode(node, storagePath, workUnitID, hashes, 1, spec.Env)
	if err != nil {
		return nil, nil, err
	}

	submitterOrg, _ := ctx.Value("organization_id").(string)
	createSpec := clusterCreateSpec{
		Name:     fabricJobName(jobSpec, workUnitID),
		MemoryMB: clampJobMemory(spec.MemoryMB),
		VCPUs:    spec.VCPUs,
		Command:  spec.Command,
		Args:     spec.Args,
		Env:      env,
		Tags: map[string]interface{}{
			"fabric_job_id":    workUnitID,
			"fabric_work_unit": true,
		},
		OrganizationID: orgLabelForVM(submitterOrg),
	}
	vmID, err := createAndStartJobVM(ctx, db, vmManager, node, createSpec)
	if err != nil {
		return nil, nil, err
	}

	hashesJSON, err := json.Marshal(hashes)
	if err != nil {
		rollbackJobVM(node, vmManager, vmID)
		return nil, nil, err
	}
	if _, err := db.ExecContext(ctx, `
		INSERT INTO fabric_jobs (id, vm_id, node_id, command, name, status, placed_by, organization_id,
			work_unit, artifact_hashes, input_bytes, attempts, attempt_limit, checkpoint_path, checkpoint_state, created_at, updated_at)
		VALUES ($1, $2, $3, $4, NULLIF($5, ''), $6, $7, NULLIF($8, '')::uuid,
			TRUE, $9, $10, $11, $12, NULLIF($13, ''), $14, NOW(), NOW())
	`, workUnitID, vmID, node.NodeID, spec.Command, spec.Name, jobStatusRunning, decision.Decision,
		orgLabelForVM(submitterOrg), hashesJSON, inputBytes, 1, attemptLimit,
		env[envWorkUnitCheckpoint], "none",
	); err != nil {
		rollbackJobVM(node, vmManager, vmID)
		return nil, nil, fmt.Errorf("%w: persist work unit: %v", errWorkUnitInternal, err)
	}
	return &workUnitRecord{
		ID: workUnitID, VMID: vmID, NodeID: node.NodeID, Command: spec.Command, Name: spec.Name,
		Status: jobStatusRunning, Attempts: 1, AttemptLimit: attemptLimit,
		ArtifactHashes: hashes, InputBytes: inputBytes,
		CheckpointPath: env[envWorkUnitCheckpoint], CheckpointState: "none",
		CreatedAt: time.Now().UTC(), Organization: sql.NullString{String: orgLabelForVM(submitterOrg), Valid: submitterOrg != ""},
	}, decision, nil
}

// retryWorkUnit reserves one attempt atomically, then restarts the SAME VM so
// it re-reads the same checkpoint path with an advanced attempt number. The
// reservation is a single UPDATE ... WHERE attempts < attempt_limit, so two
// concurrent retries cannot both start a run.
func retryWorkUnit(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, storagePath string, rec *workUnitRecord) error {
	if db == nil || vmManager == nil {
		return errors.New("work-unit retry needs a database and VM manager")
	}
	switch workUnitLiveStatus(vmManager, rec) {
	case jobStatusRunning, jobStatusPending:
		return errWorkUnitRunning
	}
	var attempts, limit int
	var vmID, nodeID, checkpointPath string
	err := db.QueryRowContext(ctx, `
		UPDATE fabric_jobs
		SET attempts = attempts + 1, status = $2, error = NULL, updated_at = NOW()
		WHERE id = $1 AND work_unit AND attempts < attempt_limit
		RETURNING attempts, attempt_limit, vm_id, node_id, COALESCE(checkpoint_path, '')
	`, rec.ID, jobStatusRunning).Scan(&attempts, &limit, &vmID, &nodeID, &checkpointPath)
	if errors.Is(err, sql.ErrNoRows) {
		return errWorkUnitAttemptsExhausted
	}
	if err != nil {
		return fmt.Errorf("%w: reserve attempt: %v", errWorkUnitInternal, err)
	}

	node := NodeCapacity{NodeID: nodeID, Addr: vmManager.MigrationPeers()[nodeID]}
	local := nodeID == "" || nodeID == selfNodeID()
	// Re-preparing is the pre-flight: it re-verifies every input is still in
	// that node's cache and rewrites the attempt file the workload reads.
	if _, err := prepareWorkUnitOnNode(node, storagePath, rec.ID, rec.ArtifactHashes, attempts, nil); err != nil {
		_, _ = db.ExecContext(ctx, `UPDATE fabric_jobs SET status = $2, error = $3, updated_at = NOW() WHERE id = $1`,
			rec.ID, jobStatusFailed, err.Error())
		return fmt.Errorf("work unit cannot resume: %w", err)
	}
	if local {
		err = restartLocalWorkUnitVM(ctx, vmManager, vmID)
	} else if node.Addr == "" {
		err = fmt.Errorf("work unit's node %q is not registered", nodeID)
	} else {
		err = startRemoteJob(node.Addr, vmID)
	}
	if err != nil {
		_, _ = db.ExecContext(ctx, `UPDATE fabric_jobs SET status = $2, error = $3, updated_at = NOW() WHERE id = $1`,
			rec.ID, jobStatusFailed, err.Error())
		return fmt.Errorf("failed to restart work unit: %w", err)
	}
	if _, err := os.Stat(checkpointPath); err == nil {
		_, _ = db.ExecContext(ctx, `UPDATE fabric_jobs SET checkpoint_state = $2, updated_at = NOW() WHERE id = $1`,
			rec.ID, "consumed")
	}
	return nil
}

// restartLocalWorkUnitVM re-runs a finished executor process. It drives the
// driver directly instead of VMManager.StartVM because the manager's cached VM
// state is frozen at "running" (its maintenance loop is inert), so StartVM
// would report success without spawning the retry.
func restartLocalWorkUnitVM(ctx context.Context, vmManager *core_vm.VMManager, vmID string) error {
	vm, err := vmManager.GetVM(vmID)
	if err != nil {
		return err
	}
	drv, err := vmManager.GetDriverForConfig(vm.Config())
	if err != nil {
		return err
	}
	if err := drv.Start(ctx, vmID); err != nil {
		return err
	}
	vm.SetState(core_vm.StateRunning)
	return nil
}

// getWorkUnit loads one unit including its resume fields. Only work-unit rows
// are visible here; plain jobs keep the original job loader and its contract.
func getWorkUnit(ctx context.Context, db *sql.DB, id string) (*workUnitRecord, error) {
	if db == nil {
		return nil, errors.New("no database")
	}
	var rec workUnitRecord
	var hashes []byte
	err := db.QueryRowContext(ctx, `
		SELECT id, vm_id, node_id, command, COALESCE(name, ''), status, COALESCE(error, ''),
		       attempts, attempt_limit, artifact_hashes, input_bytes,
		       COALESCE(checkpoint_path, ''), COALESCE(checkpoint_state, ''), created_at, organization_id
		FROM fabric_jobs WHERE id = $1 AND work_unit`, id).
		Scan(&rec.ID, &rec.VMID, &rec.NodeID, &rec.Command, &rec.Name, &rec.Status, &rec.Error,
			&rec.Attempts, &rec.AttemptLimit, &hashes, &rec.InputBytes,
			&rec.CheckpointPath, &rec.CheckpointState, &rec.CreatedAt, &rec.Organization)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(hashes, &rec.ArtifactHashes); err != nil {
		rec.ArtifactHashes = nil
	}
	return &rec, nil
}

func workUnitLiveStatus(vmManager *core_vm.VMManager, rec *workUnitRecord) string {
	return liveJobStatus(vmManager, &fabricJob{
		ID: rec.ID, VMID: rec.VMID, NodeID: rec.NodeID, Status: rec.Status,
	})
}

// workUnitCheckpointState reports the resume file the workload wrote. The
// executing node owns the file, so a remote unit is asked over its own RPC.
func workUnitCheckpointState(vmManager *core_vm.VMManager, storagePath string, rec *workUnitRecord) workUnitCheckpoint {
	if rec.NodeID == "" || rec.NodeID == selfNodeID() {
		state, err := inspectWorkUnitCheckpoint(storagePath, rec.ID)
		if err != nil {
			return workUnitCheckpoint{}
		}
		return state
	}
	addr := vmManager.MigrationPeers()[rec.NodeID]
	if addr == "" {
		return workUnitCheckpoint{}
	}
	secret, ok := nodeCredential(rec.NodeID)
	if !ok {
		return workUnitCheckpoint{}
	}
	return fetchRemoteWorkUnitCheckpoint(addr, secret, rec.ID)
}

func prepareWorkUnitOnNode(node NodeCapacity, storagePath, workUnitID string, hashes []string, attempt int, baseEnv map[string]string) (map[string]string, error) {
	if node.Addr == "" || node.NodeID == selfNodeID() {
		return prepareWorkUnitEnvironment(storagePath, workUnitID, hashes, attempt, baseEnv)
	}
	secret, ok := nodeCredential(node.NodeID)
	if !ok {
		return nil, fmt.Errorf("no credential configured for %s", node.NodeID)
	}
	body, err := json.Marshal(workUnitPrepareRequest{
		WorkUnitID: workUnitID, ArtifactHashes: hashes, Attempt: attempt, Env: baseEnv,
	})
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+node.Addr+"/internal/fabric/work-units/prepare", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Migration-Secret", secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("prepare work unit on %s: %w", node.NodeID, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, fmt.Errorf("prepare work unit on %s: %s: %s", node.NodeID, resp.Status, strings.TrimSpace(string(b)))
	}
	var out struct {
		Env map[string]string `json:"env"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, err
	}
	if out.Env[envWorkUnitCheckpoint] == "" || out.Env[envWorkUnitArtifacts] == "" {
		return nil, fmt.Errorf("node %s returned an incomplete work-unit environment", node.NodeID)
	}
	return out.Env, nil
}

// stageWorkUnitArtifacts copies inputs to the executing node's cache, skipping
// every hash it already holds intact. A presence probe is cheap and the upload
// only happens on a miss, which is what keeps a retry from re-sending bytes.
func stageWorkUnitArtifacts(node NodeCapacity, storagePath string, hashes []string) error {
	if node.Addr == "" || node.NodeID == selfNodeID() {
		return nil // already validated against this node's cache
	}
	secret, ok := nodeCredential(node.NodeID)
	if !ok {
		return fmt.Errorf("no credential configured for %s", node.NodeID)
	}
	for _, hash := range hashes {
		present, err := peerHasWorkUnitArtifact(node.Addr, secret, hash)
		if err != nil {
			return err
		}
		if present {
			continue
		}
		if err := uploadWorkUnitArtifact(node.Addr, secret, storagePath, hash); err != nil {
			return err
		}
	}
	return nil
}

func peerHasWorkUnitArtifact(addr, secret, hash string) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodHead,
		"http://"+addr+"/internal/fabric/work-units/artifacts/"+hash, nil)
	if err != nil {
		return false, err
	}
	req.Header.Set("X-Migration-Secret", secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("artifact probe %s: %w", hash, err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("artifact probe %s: %s", hash, resp.Status)
	}
}

func uploadWorkUnitArtifact(addr, secret, storagePath, hash string) error {
	file, err := os.Open(workUnitArtifactPath(storagePath, hash))
	if err != nil {
		return fmt.Errorf("open local artifact %s: %w", hash, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPut,
		"http://"+addr+"/internal/fabric/work-units/artifacts/"+hash, file)
	if err != nil {
		return err
	}
	req.ContentLength = info.Size()
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Migration-Secret", secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("transfer artifact %s: %w", hash, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return fmt.Errorf("transfer artifact %s: %s: %s", hash, resp.Status, strings.TrimSpace(string(b)))
	}
	return nil
}

func fetchRemoteWorkUnitCheckpoint(addr, secret, workUnitID string) workUnitCheckpoint {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"http://"+addr+"/internal/fabric/work-units/"+workUnitID+"/checkpoint", nil)
	if err != nil {
		return workUnitCheckpoint{}
	}
	req.Header.Set("X-Migration-Secret", secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return workUnitCheckpoint{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return workUnitCheckpoint{}
	}
	var state workUnitCheckpoint
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&state); err != nil {
		return workUnitCheckpoint{}
	}
	return state
}

func inspectWorkUnitCheckpoint(storagePath, workUnitID string) (workUnitCheckpoint, error) {
	info, err := os.Lstat(workUnitCheckpointPath(storagePath, workUnitID))
	if os.IsNotExist(err) {
		return workUnitCheckpoint{Available: false}, nil
	}
	if err != nil {
		return workUnitCheckpoint{}, err
	}
	if !info.Mode().IsRegular() {
		return workUnitCheckpoint{}, fmt.Errorf("checkpoint is not a regular file")
	}
	return workUnitCheckpoint{Available: true, SizeBytes: info.Size()}, nil
}

func normalizedWorkUnitHashes(hashes []string) ([]string, error) {
	if len(hashes) == 0 {
		return nil, fmt.Errorf("artifact_hashes is required")
	}
	if len(hashes) > maxWorkUnitArtifacts {
		return nil, fmt.Errorf("work unit accepts at most %d artifacts", maxWorkUnitArtifacts)
	}
	out := make([]string, 0, len(hashes))
	seen := make(map[string]struct{}, len(hashes))
	for _, raw := range hashes {
		hash := strings.ToLower(strings.TrimSpace(raw))
		if !validWorkUnitArtifactHash(hash) {
			return nil, fmt.Errorf("invalid artifact hash %q", raw)
		}
		if _, dup := seen[hash]; dup {
			return nil, fmt.Errorf("duplicate artifact hash %q", hash)
		}
		seen[hash] = struct{}{}
		out = append(out, hash)
	}
	return out, nil
}

func workUnitArtifactDir(storagePath string) string {
	return filepath.Join(storagePath, "fabric", "artifacts")
}

func workUnitArtifactPath(storagePath, sha256Hex string) string {
	return filepath.Join(workUnitArtifactDir(storagePath), sha256Hex)
}

func workUnitDir(storagePath, workUnitID string) string {
	return filepath.Join(storagePath, "fabric", "work-units", workUnitID)
}

func workUnitCheckpointPath(storagePath, workUnitID string) string {
	return filepath.Join(workUnitDir(storagePath, workUnitID), "checkpoint")
}

func workUnitAttemptPath(storagePath, workUnitID string) string {
	return filepath.Join(workUnitDir(storagePath, workUnitID), "attempt")
}

func validWorkUnitArtifactHash(sha256Hex string) bool {
	if len(sha256Hex) != sha256.Size*2 || strings.ToLower(sha256Hex) != sha256Hex {
		return false
	}
	decoded, err := hex.DecodeString(sha256Hex)
	return err == nil && len(decoded) == sha256.Size
}

func validateWorkUnitArtifacts(storagePath string, hashes []string) (int64, error) {
	if len(hashes) == 0 || len(hashes) > maxWorkUnitArtifacts {
		return 0, fmt.Errorf("work unit needs 1 to %d artifacts", maxWorkUnitArtifacts)
	}
	var total int64
	for _, hash := range hashes {
		size, err := verifyWorkUnitArtifact(storagePath, hash)
		if err != nil {
			return 0, fmt.Errorf("artifact %s is not available in the local cache: %w", hash, err)
		}
		if total > maxWorkUnitInputBytes-size {
			return 0, errWorkUnitInputTooLarge
		}
		total += size
	}
	return total, nil
}

// prepareWorkUnitEnvironment creates the unit's state directory, records the
// attempt number the workload can read, and returns the environment the VM must
// be created with. attempt <= 0 means the first run.
func prepareWorkUnitEnvironment(storagePath, workUnitID string, hashes []string, attempt int, base map[string]string) (map[string]string, error) {
	if _, err := uuid.Parse(workUnitID); err != nil {
		return nil, fmt.Errorf("invalid work-unit id")
	}
	if attempt <= 0 {
		attempt = 1
	}
	env := make(map[string]string, len(base)+5)
	for key, value := range base {
		env[key] = value
	}
	for _, key := range workUnitReservedEnv {
		if _, exists := env[key]; exists {
			return nil, fmt.Errorf("environment key %s is reserved for work units", key)
		}
	}
	if _, err := validateWorkUnitArtifacts(storagePath, hashes); err != nil {
		return nil, err
	}
	dir := workUnitDir(storagePath, workUnitID)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("create work-unit directory: %w", err)
	}
	if err := os.WriteFile(workUnitAttemptPath(storagePath, workUnitID),
		[]byte(strconv.Itoa(attempt)), 0600); err != nil {
		return nil, fmt.Errorf("record work-unit attempt: %w", err)
	}
	env[envWorkUnitArtifactDir] = workUnitArtifactDir(storagePath)
	env[envWorkUnitArtifacts] = strings.Join(hashes, ",")
	env[envWorkUnitCheckpoint] = workUnitCheckpointPath(storagePath, workUnitID)
	env[envWorkUnitAttemptFile] = workUnitAttemptPath(storagePath, workUnitID)
	env[envWorkUnitID] = workUnitID
	return env, nil
}

func storeWorkUnitArtifact(storagePath string, source io.Reader) (string, int64, bool, error) {
	return storeWorkUnitArtifactAtHash(storagePath, "", source)
}

func storeWorkUnitArtifactAtHash(storagePath, expectedHash string, source io.Reader) (string, int64, bool, error) {
	if strings.TrimSpace(storagePath) == "" {
		return "", 0, false, fmt.Errorf("storage path is required")
	}
	dir := workUnitArtifactDir(storagePath)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", 0, false, fmt.Errorf("create artifact cache: %w", err)
	}
	file, err := os.CreateTemp(dir, ".artifact-*")
	if err != nil {
		return "", 0, false, fmt.Errorf("create temporary artifact: %w", err)
	}
	tempPath := file.Name()
	keepTemp := false
	defer func() {
		_ = file.Close()
		if !keepTemp {
			_ = os.Remove(tempPath)
		}
	}()

	digest := sha256.New()
	size, err := io.Copy(io.MultiWriter(file, digest), io.LimitReader(source, maxWorkUnitArtifactBytes+1))
	if err != nil {
		return "", 0, false, fmt.Errorf("read artifact body: %w", err)
	}
	if size > maxWorkUnitArtifactBytes {
		return "", 0, false, errWorkUnitArtifactTooLarge
	}
	if err := file.Sync(); err != nil {
		return "", 0, false, fmt.Errorf("sync artifact cache file: %w", err)
	}
	if err := file.Close(); err != nil {
		return "", 0, false, fmt.Errorf("close artifact cache file: %w", err)
	}

	sha256Hex := hex.EncodeToString(digest.Sum(nil))
	if expectedHash != "" && sha256Hex != expectedHash {
		return "", 0, false, errWorkUnitArtifactHashMismatch
	}
	destination := workUnitArtifactPath(storagePath, sha256Hex)
	if _, err := os.Lstat(destination); err == nil {
		if existingSize, verifyErr := verifyWorkUnitArtifact(storagePath, sha256Hex); verifyErr == nil && existingSize == size {
			return sha256Hex, size, true, nil
		}
		if err := os.Remove(destination); err != nil && !os.IsNotExist(err) {
			return "", 0, false, fmt.Errorf("replace corrupt cached artifact: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return "", 0, false, fmt.Errorf("inspect artifact cache: %w", err)
	}
	if err := os.Rename(tempPath, destination); err != nil {
		return "", 0, false, fmt.Errorf("commit artifact cache file: %w", err)
	}
	keepTemp = true
	return sha256Hex, size, false, nil
}

func verifyWorkUnitArtifact(storagePath, expectedHash string) (int64, error) {
	if !validWorkUnitArtifactHash(expectedHash) {
		return 0, fmt.Errorf("invalid artifact hash %q", expectedHash)
	}
	path := workUnitArtifactPath(storagePath, expectedHash)
	info, err := os.Lstat(path)
	if err != nil {
		return 0, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxWorkUnitArtifactBytes {
		return 0, fmt.Errorf("cached artifact %s is not a regular file within the size limit", expectedHash)
	}
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	digest := sha256.New()
	size, err := io.Copy(digest, io.LimitReader(file, maxWorkUnitArtifactBytes+1))
	if err != nil {
		return 0, err
	}
	if size != info.Size() || hex.EncodeToString(digest.Sum(nil)) != expectedHash {
		return 0, fmt.Errorf("cached artifact %s failed SHA-256 verification", expectedHash)
	}
	return size, nil
}

// workUnitCacheBudget is the cache ceiling from NOVACRON_WORK_UNIT_CACHE_BYTES.
// Zero, negative or unparseable values fall back to the 10 GiB default; there is
// no "unlimited" value, because that is the failure this exists to prevent.
func workUnitCacheBudget() int64 {
	raw := strings.TrimSpace(os.Getenv("NOVACRON_WORK_UNIT_CACHE_BYTES"))
	if raw == "" {
		return defaultWorkUnitCacheBytes
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n <= 0 {
		return defaultWorkUnitCacheBytes
	}
	return n
}

// pruneWorkUnitArtifacts keeps the artifact cache under its byte budget by
// deleting the oldest artifacts that no work-unit row references.
//
// Fail-safe by construction: if the reference lookup errors or is truncated, if
// the cache cannot be listed, or if a file is still inside the grace window, the
// artifact stays. Deleting a referenced input would break a later retry, and
// deleting a freshly uploaded one would race a submit that already validated it.
func pruneWorkUnitArtifacts(ctx context.Context, db *sql.DB, storagePath string) {
	if db == nil || strings.TrimSpace(storagePath) == "" {
		return
	}
	budget := workUnitCacheBudget()

	type cachedArtifact struct {
		hash    string
		size    int64
		modTime time.Time
	}
	entries, err := os.ReadDir(workUnitArtifactDir(storagePath))
	if err != nil {
		return
	}
	artifacts := make([]cachedArtifact, 0, len(entries))
	var total int64
	for _, entry := range entries {
		if entry.IsDir() || !validWorkUnitArtifactHash(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		artifacts = append(artifacts, cachedArtifact{hash: entry.Name(), size: info.Size(), modTime: info.ModTime()})
		total += info.Size()
	}
	if total <= budget {
		return
	}

	rows, err := db.QueryContext(ctx,
		`SELECT artifact_hashes FROM fabric_jobs WHERE work_unit LIMIT $1`, workUnitCacheScanLimit+1)
	if err != nil {
		return
	}
	defer rows.Close()
	referenced := make(map[string]struct{})
	seen := 0
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			return
		}
		seen++
		var hashes []string
		if err := json.Unmarshal(raw, &hashes); err != nil {
			continue
		}
		for _, hash := range hashes {
			referenced[hash] = struct{}{}
		}
	}
	if rows.Err() != nil || seen > workUnitCacheScanLimit {
		return // incomplete view: prune nothing rather than guess
	}

	sort.Slice(artifacts, func(i, j int) bool { return artifacts[i].modTime.Before(artifacts[j].modTime) })
	now := time.Now()
	for _, artifact := range artifacts {
		if total <= budget {
			return
		}
		if _, inUse := referenced[artifact.hash]; inUse {
			continue
		}
		if now.Sub(artifact.modTime) < workUnitCacheEvictGrace {
			continue
		}
		if err := os.Remove(workUnitArtifactPath(storagePath, artifact.hash)); err == nil {
			total -= artifact.size
		}
	}
}
