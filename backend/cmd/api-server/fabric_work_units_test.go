package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// workUnitAdminRequest builds an admin request that also carries the
// trusted-operator opt-in marker the execution routes require. Admin role
// alone is what authorizes the call; the header records the intent to run
// unsandboxed native code.
func workUnitAdminRequest(req *http.Request) *http.Request {
	req = req.WithContext(context.WithValue(req.Context(), "role", "admin"))
	req.Header.Set(trustedOperatorHeader, "1")
	return req
}

func TestWorkUnitArtifactUploadIsContentAddressedAndAdminOnly(t *testing.T) {
	storagePath := t.TempDir()
	rootRouter, apiRouter := mux.NewRouter(), mux.NewRouter()
	registerFabricWorkUnitRoutes(rootRouter, apiRouter, nil, nil, storagePath)

	artifact := []byte("checkpointed workload input\n")
	upload := func(role string) (*httptest.ResponseRecorder, struct {
		SHA256    string `json:"sha256"`
		SizeBytes int64  `json:"size_bytes"`
		Cached    bool   `json:"cached"`
	}) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/compute/work-units/artifacts", bytes.NewReader(artifact))
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set(trustedOperatorHeader, "1")
		if role != "" {
			req = req.WithContext(context.WithValue(req.Context(), "role", role))
		}
		response := httptest.NewRecorder()
		apiRouter.ServeHTTP(response, req)
		var body struct {
			SHA256    string `json:"sha256"`
			SizeBytes int64  `json:"size_bytes"`
			Cached    bool   `json:"cached"`
		}
		if response.Code == http.StatusCreated || response.Code == http.StatusOK {
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatalf("decode artifact upload response: %v", err)
			}
		}
		return response, body
	}

	if response, _ := upload(""); response.Code != http.StatusForbidden {
		t.Fatalf("unauthorized artifact upload status = %d, want 403", response.Code)
	}
	first, firstBody := upload("admin")
	if first.Code != http.StatusCreated {
		t.Fatalf("first artifact upload status = %d, body %s", first.Code, first.Body)
	}
	if firstBody.SHA256 != "5267c7d8ee36f743375cbbbd21f3eb73b0bc87c2b8b2d1e8a944514d93237a32" ||
		firstBody.SizeBytes != int64(len(artifact)) || firstBody.Cached {
		t.Fatalf("first upload = %+v, want the known digest, exact size, and cache miss", firstBody)
	}

	second, secondBody := upload("admin")
	if second.Code != http.StatusOK {
		t.Fatalf("repeat artifact upload status = %d, body %s", second.Code, second.Body)
	}
	if secondBody.SHA256 != firstBody.SHA256 || secondBody.SizeBytes != firstBody.SizeBytes || !secondBody.Cached {
		t.Fatalf("repeat upload = %+v, want the same digest and size with a cache hit", secondBody)
	}
}
func TestWorkUnitResumesFromCheckpointAfterFailure(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	nodeCredentialsTestEnv(t, "cluster-secret", "")
	storagePath := t.TempDir()

	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeProcess] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  map[string]interface{}{"base_path": t.TempDir()},
	}
	vmManager, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer vmManager.Stop()

	artifact := []byte("checkpointed workload input\n")
	digest, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader(artifact))
	if err != nil {
		t.Fatalf("seed artifact cache: %v", err)
	}

	// Attempt 1 verifies the staged input, writes a checkpoint, and fails.
	// Attempt 2 must find that checkpoint and finish the second phase; it exits
	// non-zero if the input digest is wrong or the checkpoint is missing.
	script := `set -eu
hash=$(printf '%s' "$NOVACRON_ARTIFACTS" | cut -d, -f1)
staged="$NOVACRON_ARTIFACT_DIR/$hash"
digest=$(sha256sum "$staged" | cut -d' ' -f1)
if [ "$digest" != "$hash" ]; then
  echo "artifact digest mismatch: $digest != $hash" >&2
  exit 9
fi
dir=$(dirname "$NOVACRON_CHECKPOINT_PATH")
attempt=$(cat "$NOVACRON_ATTEMPT_FILE")
echo "attempt=$attempt digest=$digest"
if [ "$attempt" = "1" ]; then
  echo half > "$NOVACRON_CHECKPOINT_PATH"
  echo phase-one >> "$dir/progress.log"
  echo "interrupted after first phase" >&2
  exit 3
fi
grep -q half "$NOVACRON_CHECKPOINT_PATH"
echo done > "$dir/result.txt"
echo phase-two >> "$dir/progress.log"
exit 0`

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	// Placement: no peers, so the one reachable node is this node ("default").
	mock.ExpectQuery("SELECT node_id, last_rtt_ms").
		WillReturnRows(sqlmock.NewRows([]string{"node_id", "last_rtt_ms", "last_heartbeat", "link"}))
	// createVMLocal
	mock.ExpectExec("INSERT INTO vms").
		WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(),
			sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), "node-a", sqlmock.AnyArg(),
			sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))
	// work-unit record
	mock.ExpectExec("INSERT INTO fabric_jobs").
		WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg(), "node-a", "/bin/sh", sqlmock.AnyArg(),
			jobStatusRunning, "default", sqlmock.AnyArg(), sqlmock.AnyArg(), int64(len(artifact)),
			1, 2, sqlmock.AnyArg(), "none").
		WillReturnResult(sqlmock.NewResult(1, 1))

	apiRouter := mux.NewRouter()
	registerFabricWorkUnitRoutes(mux.NewRouter(), apiRouter, db, vmManager, storagePath)

	submitBody, err := json.Marshal(workUnitSpec{
		Command: "/bin/sh", Args: []string{"-c", script},
		ArtifactHashes: []string{digest}, AttemptLimit: 2, MemoryMB: 128,
	})
	if err != nil {
		t.Fatalf("marshal submit: %v", err)
	}
	response := httptest.NewRecorder()
	apiRouter.ServeHTTP(response, workUnitAdminRequest(
		httptest.NewRequest(http.MethodPost, "/compute/work-units", bytes.NewReader(submitBody))))
	if response.Code != http.StatusCreated {
		t.Fatalf("submit status = %d, body %s", response.Code, response.Body)
	}
	var submitted struct {
		WorkUnitID string   `json:"work_unit_id"`
		VMID       string   `json:"vm_id"`
		NodeID     string   `json:"node_id"`
		Status     string   `json:"status"`
		Attempts   int      `json:"attempts"`
		Limit      int      `json:"attempt_limit"`
		Hashes     []string `json:"artifact_hashes"`
		InputBytes int64    `json:"input_bytes"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &submitted); err != nil {
		t.Fatalf("decode submit: %v", err)
	}
	if submitted.NodeID != "node-a" || submitted.Attempts != 1 || submitted.Limit != 2 ||
		submitted.InputBytes != int64(len(artifact)) || len(submitted.Hashes) != 1 || submitted.Hashes[0] != digest {
		t.Fatalf("submission = %+v, want one attempt reserved and the artifact recorded", submitted)
	}

	rec := &workUnitRecord{ID: submitted.WorkUnitID, VMID: submitted.VMID, NodeID: submitted.NodeID, Status: jobStatusRunning}
	waitForWorkUnitStatus(t, vmManager, storagePath, rec, jobStatusFailed)

	workUnitDir := filepath.Join(storagePath, "fabric", "work-units", submitted.WorkUnitID)
	checkpointPath := filepath.Join(workUnitDir, "checkpoint")
	if data, err := os.ReadFile(checkpointPath); err != nil || string(data) != "half\n" {
		t.Fatalf("checkpoint after attempt 1 = %q, %v; want a resumable marker", data, err)
	}

	hashesJSON, _ := json.Marshal([]string{digest})
	failedRow := func(attempts int) *sqlmock.Rows {
		return sqlmock.NewRows([]string{
			"id", "vm_id", "node_id", "command", "name", "status", "error",
			"attempts", "attempt_limit", "artifact_hashes", "input_bytes",
			"checkpoint_path", "checkpoint_state", "created_at", "organization_id",
		}).AddRow(submitted.WorkUnitID, submitted.VMID, "node-a", "/bin/sh", "", jobStatusFailed, "",
			attempts, 2, hashesJSON, len(artifact), checkpointPath, "available", time.Now().UTC(), nil)
	}
	mock.ExpectQuery("SELECT id, vm_id, node_id, command").
		WithArgs(submitted.WorkUnitID).WillReturnRows(failedRow(1))
	// The atomic attempt reservation is the only thing that authorizes a run.
	mock.ExpectQuery("UPDATE fabric_jobs").
		WithArgs(submitted.WorkUnitID, jobStatusRunning).
		WillReturnRows(sqlmock.NewRows([]string{"attempts", "attempt_limit", "vm_id", "node_id"}).
			AddRow(2, 2, submitted.VMID, "node-a"))
	mock.ExpectExec("UPDATE fabric_jobs").
		WithArgs(submitted.WorkUnitID, "consumed").
		WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectQuery("SELECT id, vm_id, node_id, command").
		WithArgs(submitted.WorkUnitID).WillReturnRows(failedRow(2))

	retry := httptest.NewRecorder()
	apiRouter.ServeHTTP(retry, workUnitAdminRequest(
		httptest.NewRequest(http.MethodPost, "/compute/work-units/"+submitted.WorkUnitID+"/retry", nil)))
	if retry.Code != http.StatusOK {
		t.Fatalf("retry status = %d, body %s", retry.Code, retry.Body)
	}
	var retried struct {
		Status   string `json:"status"`
		Attempts int    `json:"attempts"`
		Limit    int    `json:"attempt_limit"`
	}
	if err := json.Unmarshal(retry.Body.Bytes(), &retried); err != nil {
		t.Fatalf("decode retry: %v", err)
	}
	if retried.Attempts != 2 || retried.Limit != 2 {
		t.Fatalf("retry = %+v, want attempt 2 of 2", retried)
	}

	rec.Status = jobStatusRunning
	waitForWorkUnitStatus(t, vmManager, storagePath, rec, jobStatusCompleted)

	// Resume proof: phase two ran, with the phase-one checkpoint still in place
	// and the attempt counter advanced. A fresh run would have failed the
	// grep or written phase-one twice.
	if data, err := os.ReadFile(filepath.Join(workUnitDir, "progress.log")); err != nil || string(data) != "phase-one\nphase-two\n" {
		t.Fatalf("progress log = %q, %v; want both phases in order", data, err)
	}
	if data, err := os.ReadFile(filepath.Join(workUnitDir, "result.txt")); err != nil || string(data) != "done\n" {
		t.Fatalf("result = %q, %v; want the completed output", data, err)
	}
	if data, err := os.ReadFile(filepath.Join(workUnitDir, "attempt")); err != nil || string(data) != "2" {
		t.Fatalf("attempt file = %q, %v; want the advanced attempt number", data, err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}

func waitForWorkUnitStatus(t *testing.T, vmManager *core_vm.VMManager, storagePath string, rec *workUnitRecord, want string) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		if got := workUnitLiveStatus(vmManager, rec); got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("work unit %s status = %q, want %q; logs: %v", rec.ID,
				workUnitLiveStatus(vmManager, rec), want, processLogTails(storagePath, rec.VMID))
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestStageWorkUnitArtifactsTransfersOnlyMissingInputs(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	nodeCredentialsTestEnv(t, "cluster-secret", "peer-b=peer-secret")
	storagePath := t.TempDir()

	held := []byte("already on the peer\n")
	missing := []byte("must be transferred\n")
	heldHash, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader(held))
	if err != nil {
		t.Fatalf("seed held artifact: %v", err)
	}
	missingHash, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader(missing))
	if err != nil {
		t.Fatalf("seed missing artifact: %v", err)
	}

	var transfers []string
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Migration-Secret") != "peer-secret" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		hash := strings.TrimPrefix(r.URL.Path, "/internal/fabric/work-units/artifacts/")
		switch r.Method {
		case http.MethodHead:
			if hash == heldHash {
				w.WriteHeader(http.StatusOK)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPut:
			body, _ := io.ReadAll(r.Body)
			if hash != missingHash || string(body) != string(missing) {
				t.Errorf("transfer %s carried %q; want %s's bytes", hash, body, missingHash)
			}
			transfers = append(transfers, hash)
			w.WriteHeader(http.StatusCreated)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer peer.Close()

	node := NodeCapacity{NodeID: "peer-b", Addr: joinTestRequestAddr(t, peer.URL)}
	if err := stageWorkUnitArtifacts(node, storagePath, []string{heldHash, missingHash}); err != nil {
		t.Fatalf("stage artifacts: %v", err)
	}
	if len(transfers) != 1 || transfers[0] != missingHash {
		t.Fatalf("transfers = %v, want exactly the missing artifact", transfers)
	}
}

func TestWorkUnitRetryRefusesExhaustedBudget(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	nodeCredentialsTestEnv(t, "cluster-secret", "")
	storagePath := t.TempDir()

	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeProcess] = core_vm.VMDriverConfigManager{
		Enabled: true, Config: map[string]interface{}{"base_path": t.TempDir()},
	}
	vmManager, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer vmManager.Stop()

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	const id = "0b3b7f5e-8a41-4d1a-9c2e-1f0a2b3c4d5e"
	hashesJSON, _ := json.Marshal([]string{"5267c7d8ee36f743375cbbbd21f3eb73b0bc87c2b8b2d1e8a944514d93237a32"})
	row := func(status string) *sqlmock.Rows {
		return sqlmock.NewRows([]string{
			"id", "vm_id", "node_id", "command", "name", "status", "error",
			"attempts", "attempt_limit", "artifact_hashes", "input_bytes",
			"checkpoint_path", "checkpoint_state", "created_at", "organization_id",
		}).AddRow(id, "vm-gone", "node-a", "/bin/sh", "", status, "attempt 2 failed",
			2, 2, hashesJSON, int64(28), filepath.Join(storagePath, "fabric", "work-units", id, "checkpoint"),
			"consumed", time.Now().UTC(), nil)
	}
	mock.ExpectQuery("SELECT id, vm_id, node_id, command").WithArgs(id).WillReturnRows(row(jobStatusFailed))
	// The atomic guard refuses the third attempt: no row returned.
	mock.ExpectQuery("UPDATE fabric_jobs").WithArgs(id, jobStatusRunning).
		WillReturnRows(sqlmock.NewRows([]string{"attempts", "attempt_limit", "vm_id", "node_id"}))

	apiRouter := mux.NewRouter()
	registerFabricWorkUnitRoutes(mux.NewRouter(), apiRouter, db, vmManager, storagePath)
	response := httptest.NewRecorder()
	apiRouter.ServeHTTP(response, workUnitAdminRequest(
		httptest.NewRequest(http.MethodPost, "/compute/work-units/"+id+"/retry", nil)))
	if response.Code != http.StatusConflict {
		t.Fatalf("exhausted retry status = %d, body %s; want 409", response.Code, response.Body)
	}
	if !strings.Contains(response.Body.String(), "attempt limit") {
		t.Fatalf("exhausted retry body = %s, want the attempt limit stated", response.Body)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}

func TestWorkUnitArtifactTransferRejectsCorruptBody(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	nodeCredentialsTestEnv(t, "cluster-secret", "node-a=node-a-secret")
	storagePath := t.TempDir()
	rootRouter, apiRouter := mux.NewRouter(), mux.NewRouter()
	registerFabricWorkUnitRoutes(rootRouter, apiRouter, nil, nil, storagePath)

	const digest = "5267c7d8ee36f743375cbbbd21f3eb73b0bc87c2b8b2d1e8a944514d93237a32"
	req := httptest.NewRequest(http.MethodPut, "/internal/fabric/work-units/artifacts/"+digest,
		strings.NewReader("this is not the addressed content\n"))
	req.Header.Set("X-Migration-Secret", "node-a-secret")
	response := httptest.NewRecorder()
	rootRouter.ServeHTTP(response, req)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("corrupt transfer status = %d, want 400", response.Code)
	}
	if _, err := os.Lstat(workUnitArtifactPath(storagePath, digest)); !os.IsNotExist(err) {
		t.Fatalf("a body that failed its digest check must not enter the cache (stat err = %v)", err)
	}
}

func TestWorkUnitPeerCacheVerifiesAndPreparesArtifacts(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	nodeCredentialsTestEnv(t, "cluster-secret", "node-a=node-a-secret")
	storagePath := t.TempDir()
	rootRouter, apiRouter := mux.NewRouter(), mux.NewRouter()
	registerFabricWorkUnitRoutes(rootRouter, apiRouter, nil, nil, storagePath)

	const digest = "5267c7d8ee36f743375cbbbd21f3eb73b0bc87c2b8b2d1e8a944514d93237a32"
	const workUnitID = "7d54a1f7-a2df-4c4b-badd-cadee8a0a011"
	artifact := []byte("checkpointed workload input\n")
	request := func(method, path, secret string, body []byte) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, path, bytes.NewReader(body))
		if secret != "" {
			req.Header.Set("X-Migration-Secret", secret)
		}
		response := httptest.NewRecorder()
		rootRouter.ServeHTTP(response, req)
		return response
	}

	if response := request(http.MethodHead, "/internal/fabric/work-units/artifacts/"+digest, "", nil); response.Code != http.StatusForbidden {
		t.Fatalf("unauthenticated cache probe status = %d, want 403", response.Code)
	}
	if response := request(http.MethodHead, "/internal/fabric/work-units/artifacts/"+digest, "node-a-secret", nil); response.Code != http.StatusNotFound {
		t.Fatalf("cache miss status = %d, want 404", response.Code)
	}
	if response := request(http.MethodPut, "/internal/fabric/work-units/artifacts/"+digest, "node-a-secret", artifact); response.Code != http.StatusCreated {
		t.Fatalf("artifact transfer status = %d, body %s", response.Code, response.Body)
	}
	if response := request(http.MethodHead, "/internal/fabric/work-units/artifacts/"+digest, "node-a-secret", nil); response.Code != http.StatusOK {
		t.Fatalf("cache hit status = %d, want 200", response.Code)
	}

	prepareBody, err := json.Marshal(map[string]interface{}{
		"work_unit_id":    workUnitID,
		"artifact_hashes": []string{digest},
		"env":             map[string]string{"WORK_MODE": "test"},
	})
	if err != nil {
		t.Fatalf("marshal prepare request: %v", err)
	}
	response := request(http.MethodPost, "/internal/fabric/work-units/prepare", "node-a-secret", prepareBody)
	if response.Code != http.StatusOK {
		t.Fatalf("work-unit prepare status = %d, body %s", response.Code, response.Body)
	}
	var prepared struct {
		Env map[string]string `json:"env"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &prepared); err != nil {
		t.Fatalf("decode prepare response: %v", err)
	}
	if prepared.Env["WORK_MODE"] != "test" ||
		prepared.Env["NOVACRON_ARTIFACT_DIR"] != workUnitArtifactDir(storagePath) ||
		prepared.Env["NOVACRON_ARTIFACTS"] != digest ||
		prepared.Env["NOVACRON_CHECKPOINT_PATH"] != filepath.Join(storagePath, "fabric", "work-units", workUnitID, "checkpoint") {
		t.Fatalf("prepared environment = %#v", prepared.Env)
	}
}

func TestPruneWorkUnitArtifactsKeepsReferencedAndRecentInputs(t *testing.T) {
	t.Setenv("NOVACRON_WORK_UNIT_CACHE_BYTES", "10") // bytes: any artifact exceeds it
	storagePath := t.TempDir()

	referenced, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader([]byte("referenced input\n")))
	if err != nil {
		t.Fatalf("seed referenced artifact: %v", err)
	}
	oldest, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader([]byte("old unreferenced\n")))
	if err != nil {
		t.Fatalf("seed oldest artifact: %v", err)
	}
	fresh, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader([]byte("fresh unreferenced\n")))
	if err != nil {
		t.Fatalf("seed fresh artifact: %v", err)
	}
	// Both unreferenced inputs must be evictable by age, but only the fresh one
	// stays inside the grace window — that is the race guard under test.
	stale := time.Now().Add(-2 * workUnitCacheEvictGrace)
	for _, hash := range []string{referenced, oldest} {
		if err := os.Chtimes(workUnitArtifactPath(storagePath, hash), stale, stale); err != nil {
			t.Fatalf("backdate %s: %v", hash, err)
		}
	}

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	hashesJSON, _ := json.Marshal([]string{referenced})
	mock.ExpectQuery("SELECT artifact_hashes FROM fabric_jobs").
		WithArgs(workUnitCacheScanLimit + 1).
		WillReturnRows(sqlmock.NewRows([]string{"artifact_hashes"}).AddRow(hashesJSON))

	pruneWorkUnitArtifacts(t.Context(), db, storagePath)

	if _, err := os.Lstat(workUnitArtifactPath(storagePath, referenced)); err != nil {
		t.Fatalf("a referenced artifact must never be evicted: %v", err)
	}
	if _, err := os.Lstat(workUnitArtifactPath(storagePath, fresh)); err != nil {
		t.Fatalf("an artifact inside the grace window must survive: %v", err)
	}
	if _, err := os.Lstat(workUnitArtifactPath(storagePath, oldest)); !os.IsNotExist(err) {
		t.Fatalf("the oldest unreferenced artifact should be evicted (stat err = %v)", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}

func TestPruneWorkUnitArtifactsDoesNothingWithoutProvenReferences(t *testing.T) {
	t.Setenv("NOVACRON_WORK_UNIT_CACHE_BYTES", "10")
	storagePath := t.TempDir()
	hash, _, _, err := storeWorkUnitArtifact(storagePath, bytes.NewReader([]byte("input\n")))
	if err != nil {
		t.Fatalf("seed artifact: %v", err)
	}
	stale := time.Now().Add(-2 * workUnitCacheEvictGrace)
	if err := os.Chtimes(workUnitArtifactPath(storagePath, hash), stale, stale); err != nil {
		t.Fatalf("backdate: %v", err)
	}

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	// A truncated scan cannot prove anything is unreferenced.
	rows := sqlmock.NewRows([]string{"artifact_hashes"})
	for i := 0; i <= workUnitCacheScanLimit; i++ {
		rows.AddRow([]byte("[]"))
	}
	mock.ExpectQuery("SELECT artifact_hashes FROM fabric_jobs").
		WithArgs(workUnitCacheScanLimit + 1).WillReturnRows(rows)

	pruneWorkUnitArtifacts(t.Context(), db, storagePath)

	if _, err := os.Lstat(workUnitArtifactPath(storagePath, hash)); err != nil {
		t.Fatalf("a truncated reference scan must prune nothing: %v", err)
	}
}

// A work unit IS a fabric_jobs row, so the pre-existing job API must keep
// working for it: that is how an operator watches and cancels an expensive unit.
// The status assertion also pins the in-flight rule — a Process VM with no
// exit.code has not finished, and must not read as completed.
func TestWorkUnitIsVisibleAndCancellableThroughJobAPI(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node-a")
	storagePath := t.TempDir()

	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeProcess] = core_vm.VMDriverConfigManager{
		Enabled: true, Config: map[string]interface{}{"base_path": t.TempDir()},
	}
	vmManager, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer vmManager.Stop()
	vm, err := vmManager.CreateVM(t.Context(), core_vm.CreateVMRequest{
		Name:                  "work-unit-executor",
		AllowMissingOwnership: true,
		Spec: core_vm.VMConfig{
			Type: core_vm.VMTypeProcess, Command: "/bin/true", TenantID: "default",
		},
	})
	if err != nil {
		t.Fatalf("create executor VM: %v", err)
	}

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	const id = "5f2a7c1e-3d4b-4a9f-8e6d-2b1c0d9e8f7a"
	created := time.Now().UTC()
	jobRow := func() *sqlmock.Rows {
		return sqlmock.NewRows([]string{
			"id", "vm_id", "node_id", "command", "name", "status", "error", "placed_by", "created_at", "organization_id",
		}).AddRow(id, vm.ID(), "node-a", "/bin/true", "unit", jobStatusRunning, "", "default", created, nil)
	}
	mock.ExpectQuery("SELECT j.id, j.vm_id").WithArgs(id).WillReturnRows(jobRow())
	mock.ExpectQuery("SELECT j.id, j.vm_id").WithArgs(id).WillReturnRows(jobRow())
	mock.ExpectExec("UPDATE fabric_jobs").WithArgs(jobStatusCancelled, id).WillReturnResult(sqlmock.NewResult(1, 1))

	apiRouter := mux.NewRouter()
	registerFabricJobRoutes(apiRouter, db, vmManager, storagePath)

	detail := httptest.NewRecorder()
	apiRouter.ServeHTTP(detail, workUnitAdminRequest(
		httptest.NewRequest(http.MethodGet, "/compute/jobs/"+id, nil)))
	if detail.Code != http.StatusOK {
		t.Fatalf("GET /compute/jobs/{id} status = %d, body %s", detail.Code, detail.Body)
	}
	var shown struct {
		VMID   string `json:"vm_id"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(detail.Body.Bytes(), &shown); err != nil {
		t.Fatalf("decode job detail: %v", err)
	}
	if shown.VMID != vm.ID() || shown.Status != jobStatusRunning {
		t.Fatalf("job detail = %+v, want the unit's VM and an in-flight status", shown)
	}

	cancel := httptest.NewRecorder()
	apiRouter.ServeHTTP(cancel, workUnitAdminRequest(
		httptest.NewRequest(http.MethodPost, "/compute/jobs/"+id+"/cancel", nil)))
	if cancel.Code != http.StatusOK {
		t.Fatalf("cancel status = %d, body %s", cancel.Code, cancel.Body)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}

// The harness runs with NOVACRON_PROBE_BYTES=2MiB, above the 1 MiB global JSON
// cap. That cap wrapped this binary stream and truncated it mid-body, which read
// as a 400 at the receiver and silently killed the outbound measurement. This
// pins the exemption at the middleware level, not just the handler.
func TestProbeAcceptsBodyAboveTheJSONCap(t *testing.T) {
	nodeCredentialsTestEnv(t, "cluster-secret", "")
	const n = 2 << 20
	if n <= int(defaultMaxBodyBytes) {
		t.Fatalf("test body %d must exceed the JSON cap %d", n, defaultMaxBodyBytes)
	}

	router := mux.NewRouter()
	router.Use(maxBodyBytesMiddleware(maxBodyBytes()))
	router.HandleFunc("/internal/cluster/probe", clusterProbeHandler).Methods(http.MethodGet, http.MethodPost)

	req := httptest.NewRequest(http.MethodPost, "/internal/cluster/probe", bytes.NewReader(make([]byte, n)))
	req.Header.Set("X-Migration-Secret", "cluster-secret")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, req)
	if response.Code != http.StatusOK {
		t.Fatalf("probe status = %d, body %s; want 200 for a body above the JSON cap", response.Code, response.Body)
	}
	var receipt probeReceipt
	if err := json.Unmarshal(response.Body.Bytes(), &receipt); err != nil {
		t.Fatalf("decode probe receipt: %v", err)
	}
	if receipt.BytesReceived != n || receipt.ElapsedNS <= 0 {
		t.Fatalf("receipt = %+v, want all %d bytes received", receipt, n)
	}
}

// A work-unit input larger than the JSON cap must hash as the bytes actually
// sent: a truncated body would otherwise be stored under the wrong address.
func TestWorkUnitArtifactUploadAboveTheJSONCapIsNotTruncated(t *testing.T) {
	storagePath := t.TempDir()
	artifact := bytes.Repeat([]byte("payload-"), (2<<20)/8)
	want := sha256.Sum256(artifact)

	rootRouter, apiRouter := mux.NewRouter(), mux.NewRouter()
	rootRouter.Use(maxBodyBytesMiddleware(maxBodyBytes()))
	apiRouter.Use(maxBodyBytesMiddleware(maxBodyBytes()))
	registerFabricWorkUnitRoutes(rootRouter, apiRouter, nil, nil, storagePath)

	response := httptest.NewRecorder()
	apiRouter.ServeHTTP(response, workUnitAdminRequest(
		httptest.NewRequest(http.MethodPost, "/compute/work-units/artifacts", bytes.NewReader(artifact))))
	if response.Code != http.StatusCreated {
		t.Fatalf("artifact upload status = %d, body %s", response.Code, response.Body)
	}
	var body struct {
		SHA256    string `json:"sha256"`
		SizeBytes int64  `json:"size_bytes"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if body.SHA256 != hex.EncodeToString(want[:]) || body.SizeBytes != int64(len(artifact)) {
		t.Fatalf("upload = %s/%d bytes, want %s/%d; the body was truncated",
			body.SHA256, body.SizeBytes, hex.EncodeToString(want[:]), len(artifact))
	}
}
