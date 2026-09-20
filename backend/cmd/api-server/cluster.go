package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// This file adds real cross-node pooling to the canonical api-server: a create
// submitted to any node is placed on whichever node has capacity (best-fit) and
// dispatched there, and /api/cluster reports the aggregate inventory. Placement is
// STATELESS -- each create fetches live capacity from every node, so there is no
// scheduler-state divergence across the independent api-servers.
//
// Honest scope: a VM runs on exactly one node (no single-system-image); the pool
// is the set of nodes' aggregate capacity, usable in parallel via placement.

// selfNodeID is this node's cluster id (NOVACRON_NODE_ID), defaulting to "local".
func selfNodeID() string {
	if id := strings.TrimSpace(os.Getenv("NOVACRON_NODE_ID")); id != "" {
		return id
	}
	return "local"
}

func internalSecretOK(r *http.Request) bool {
	// Delegate to the fail-closed migration auth helper (migration_auth.go): a node
	// with no NOVACRON_MIGRATION_SECRET configured rejects all internal RPCs.
	return migrationAuthOK(r)
}

// maxProbeBytes caps a single throughput probe payload (the heartbeat asks for
// NOVACRON_PROBE_BYTES, default 1 MiB; the cap bounds a hostile request).
const maxProbeBytes = 8 << 20

// probePayload is a fixed pool of incompressible bytes served by
// /internal/cluster/probe. Random content keeps transparent compression in the
// path from inflating a measured rate; generated once (lazily) and shared.
var probePayload = func() []byte {
	b := make([]byte, maxProbeBytes)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is fatal-grade; a deterministic fill still gives
		// the probe a payload, and the measurement stays honest about bytes.
		for i := range b {
			b[i] = byte(i * 7)
		}
	}
	return b
}()

// NodeCapacity is one node's real capacity and live VM reservation. Reported per
// node -- the cluster aggregate is a SUM of these, not a shared pool a single VM
// can draw on.
type NodeCapacity struct {
	NodeID         string `json:"node_id"`
	Addr           string `json:"addr,omitempty"`
	Arch           string `json:"arch"`
	Cores          int    `json:"cores"`
	MemTotalMB     int64  `json:"mem_total_mb"`
	MemAllocatedMB int64  `json:"mem_allocated_mb"`
	StorageTotalGB int64  `json:"storage_total_gb"`
	StorageFreeGB  int64  `json:"storage_free_gb"`
	VMCount        int    `json:"vm_count"`
	Reachable      bool   `json:"reachable"`
}

// memAvailMB is the memory a node can still reserve for VMs (total minus what its
// running VMs already reserve).
func (c NodeCapacity) memAvailMB() int64 { return c.MemTotalMB - c.MemAllocatedMB }

func localNodeCapacity(vmManager *core_vm.VMManager, storagePath string) NodeCapacity {
	var memAlloc int64
	var count int
	if vmManager != nil {
		_, memAlloc, count = vmManager.ClusterUsage()
	}
	nc := NodeCapacity{
		NodeID:         selfNodeID(),
		Arch:           runtime.GOARCH,
		Cores:          runtime.NumCPU(),
		MemTotalMB:     memTotalMB(),
		MemAllocatedMB: memAlloc,
		VMCount:        count,
		Reachable:      true,
	}
	nc.StorageTotalGB, nc.StorageFreeGB = storageGB(storagePath)
	return nc
}

func memTotalMB() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			f := strings.Fields(line) // "MemTotal:  N kB"
			if len(f) >= 2 {
				kb, _ := strconv.ParseInt(f[1], 10, 64)
				return kb / 1024
			}
		}
	}
	return 0
}

func storageGB(path string) (totalGB, freeGB int64) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, 0
	}
	bs := int64(st.Bsize)
	totalGB = int64(st.Blocks) * bs / (1 << 30)
	freeGB = int64(st.Bavail) * bs / (1 << 30)
	return
}

// allNodeCapacities returns this node's capacity plus every peer's (fetched live).
// Unreachable peers are included with Reachable=false so the inventory is honest.
func allNodeCapacities(vmManager *core_vm.VMManager, storagePath string) []NodeCapacity {
	caps := []NodeCapacity{localNodeCapacity(vmManager, storagePath)}
	if vmManager == nil {
		return caps
	}
	for id, addr := range vmManager.MigrationPeers() {
		if c, err := fetchPeerCapacity(addr); err == nil {
			c.NodeID, c.Addr = id, addr
			caps = append(caps, c)
		} else {
			caps = append(caps, NodeCapacity{NodeID: id, Addr: addr, Reachable: false})
		}
	}
	return caps
}

func fetchPeerCapacity(addr string) (NodeCapacity, error) {
	var c NodeCapacity
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+"/internal/cluster/capacity", nil)
	if secret := os.Getenv("NOVACRON_MIGRATION_SECRET"); secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return c, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c, fmt.Errorf("capacity RPC %s: %s", addr, resp.Status)
	}
	err = json.NewDecoder(resp.Body).Decode(&c)
	return c, err
}

// fetchPeerCapacityWithSecret is fetchPeerCapacity with the internal secret
// supplied by the caller (the join/heartbeat paths must use the configured
// secret explicitly rather than re-reading env at call time).
func fetchPeerCapacityWithSecret(addr, secret string) (NodeCapacity, error) {
	var c NodeCapacity
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+"/internal/cluster/capacity", nil)
	if secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return c, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c, fmt.Errorf("capacity RPC %s: %s", addr, resp.Status)
	}
	err = json.NewDecoder(resp.Body).Decode(&c)
	return c, err
}

// placeVM picks the reachable node that can fit the request and is least loaded by
// memory-utilization fraction (proportional balancing across heterogeneous nodes).
// Returns false if no node has room.
// ponytail: stateless fetch has a TOCTOU under CONCURRENT auto-creates -- two at
// once can pick the same node before either reservation lands. Sequential use is
// exact; add a reserve step only if concurrent cluster placement is needed.
func placeVM(caps []NodeCapacity, memMB int64, diskGB int) (NodeCapacity, bool) {
	var best NodeCapacity
	found := false
	bestFrac := 2.0
	for _, c := range caps {
		if !c.Reachable || c.MemTotalMB <= 0 {
			continue
		}
		if c.memAvailMB() < memMB || c.StorageFreeGB < int64(diskGB) {
			continue
		}
		frac := float64(c.MemAllocatedMB+memMB) / float64(c.MemTotalMB)
		if !found || frac < bestFrac {
			best, bestFrac, found = c, frac, true
		}
	}
	return best, found
}

// clusterCreateSpec is the VM create payload passed to createVMLocal and over the
// /internal/vms/create dispatch RPC.
type clusterCreateSpec struct {
	Name       string                 `json:"name"`
	VCPUs      int                    `json:"vcpus,omitempty"`
	CPUShares  int                    `json:"cpu_shares,omitempty"`
	MemoryMB   int                    `json:"memory_mb,omitempty"`
	DiskSizeGB int                    `json:"disk_size_gb,omitempty"`
	Image      string                 `json:"image,omitempty"`
	Tags       map[string]interface{} `json:"tags,omitempty"`
	OwnerID    string                 `json:"owner_id,omitempty"`
	// TenantID is accepted on the wire for backward compatibility; the
	// canonical vms table has no tenancy column, so it is not persisted.
	TenantID string `json:"tenant_id,omitempty"`
	// Fabric job (P2/G2): when Command is non-empty the Process driver runs
	// it instead of booting a KVM guest — the thinnest executor surface that
	// reuses VM create/dispatch/start/stop/logs end to end.
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// createVMLocal provisions a VM on THIS node (manager create + DB row) and returns
// its id and state. Shared by the /vms route (local placement) and the
// /internal/vms/create dispatch RPC. node_id is stored as this node's id so the
// row records where the guest actually runs.

// runtimeTenant returns the in-memory quota-accounting bucket for a create.
// The canonical vms table has no tenancy column, so this label is used only
// for runtime accounting, never persisted.
func runtimeTenant(tenantID string) string {
	if strings.TrimSpace(tenantID) == "" {
		return "default"
	}
	return strings.TrimSpace(tenantID)
}

// vcpusOrDefault fills the canonical vms.cpu_cores NOT NULL column with the
// REAL guest vCPU count (the column used to hold the CPUShares scheduling
// weight). A legacy request without vcpus persists 1 core, which matches what
// the KVM driver boots when both VCPUs and CPUShares are unset (see
// driver_kvm_enhanced.go -smp derivation).
func vcpusOrDefault(vcpus int) int {
	if vcpus <= 0 {
		return 1
	}
	return vcpus
}

// createVMLocal provisions a VM on THIS node (manager create + DB row) and returns
// its id and state. Shared by the /vms route (local placement) and the
// /internal/vms/create dispatch RPC. node_id is stored as selfNodeID() (the
// cluster node id, not a nodes(id) UUID -- see novacron-ok7) so the row
// records where the guest actually runs and is queryable via plain SQL.
func createVMLocal(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, spec clusterCreateSpec) (vmID, state string, err error) {
	vmID = uuid.NewString()
	state = "stopped" // canonical vm_state enum; reconciled to live state below
	ownerID := spec.OwnerID
	if _, err := uuid.Parse(ownerID); err != nil {
		ownerID = "" // non-uuid ownership is dropped; vms.owner_id is a users FK
	}
	// requestedOwnerID is set only when a validly-formatted owner UUID could
	// not be honoured locally (see the existence check below); owner_id
	// itself already answers "who owns this" when the owner does resolve, so
	// duplicating it into requested_owner_id there would be redundant.
	requestedOwnerID := ownerID
	// Cross-node creates land in the PEER's database, which has its own users
	// table: an owner that exists only on the submitting node would violate
	// vms_owner_id_fkey. Ownership is a local-directory concept, so when the
	// local DB has no such user the column is NULL and the requested owner is
	// preserved in vms.requested_owner_id instead (never silently invented).
	if ownerID != "" && db != nil {
		var ownerExists bool
		if err := db.QueryRow(`SELECT EXISTS (SELECT 1 FROM users WHERE id = $1)`, ownerID).Scan(&ownerExists); err != nil || !ownerExists {
			ownerID = ""
		}
	}
	if ownerID != "" {
		requestedOwnerID = "" // resolved locally: no divergence to record
	}
	if vmManager != nil {
		if _, cerr := vmManager.CreateVM(ctx, core_vm.CreateVMRequest{
			Name:                  spec.Name,
			AllowMissingOwnership: true,
			Spec: core_vm.VMConfig{
				ID: vmID, Name: spec.Name,
				// A fabric job (spec.Command set) runs as a Process VM; a
				// plain VM create boots a KVM guest as before.
				Type:  vmTypeForSpec(spec),
				VCPUs: spec.VCPUs, CPUShares: spec.CPUShares, MemoryMB: spec.MemoryMB, DiskSizeGB: spec.DiskSizeGB,
				Image: spec.Image, OwnerID: ownerID,
				Command: spec.Command, Args: spec.Args, Env: spec.Env,
				// Runtime quota accounting needs a bucket even though the
				// canonical vms table has no tenancy column to persist.
				TenantID: runtimeTenant(spec.TenantID),
			},
		}); cerr != nil {
			return "", "", cerr
		}
		state = liveVMState(vmManager, vmID, state)
	}
	// cpu_cores is NOT NULL in the canonical schema and carries the REAL guest
	// vCPU count (novacron schema residue: it used to store the CPUShares
	// scheduling weight). vcpusOrDefault keeps the column NOT NULL for legacy
	// requests that do not set vcpus; cpu_shares stays in metadata.
	cores := vcpusOrDefault(spec.VCPUs)
	if spec.CPUShares <= 0 {
		spec.CPUShares = 1024 // manager-path NewVM default, for the metadata record
	}
	metadataPayload, _ := json.Marshal(map[string]interface{}{
		"cpu_shares": spec.CPUShares, "vcpus": cores, "memory_mb": spec.MemoryMB,
		"disk_size_gb": spec.DiskSizeGB, "image": spec.Image, "tags": spec.Tags,
	})
	if _, dberr := db.Exec(`
		INSERT INTO vms (id, name, state, cpu_cores, memory_mb, disk_gb, os_type, node_id, owner_id, requested_owner_id, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULLIF($9, '')::uuid, NULLIF($10, '')::uuid, $11, NOW(), NOW())
	`, vmID, spec.Name, state, cores, spec.MemoryMB, spec.DiskSizeGB, nullableStringValue(spec.Image), selfNodeID(), ownerID, requestedOwnerID, metadataPayload); dberr != nil {
		if vmManager != nil {
			_ = vmManager.DeleteVM(context.Background(), vmID)
		}
		return "", "", dberr
	}
	return vmID, state, nil
}

// vmTypeForSpec picks the VM type for a create: a fabric job (Command set)
// runs as a Process VM; anything else boots a KVM guest as before.
func vmTypeForSpec(spec clusterCreateSpec) core_vm.VMType {
	if strings.TrimSpace(spec.Command) != "" {
		return core_vm.VMTypeProcess
	}
	return core_vm.VMTypeKVM
}

// dispatchCreateToPeer sends a create to a chosen peer's /internal/vms/create and
// returns its {id, node_id, state}.
func dispatchCreateToPeer(addr string, spec clusterCreateSpec) (map[string]interface{}, error) {
	body, _ := json.Marshal(spec)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+addr+"/internal/vms/create", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if secret := os.Getenv("NOVACRON_MIGRATION_SECRET"); secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("dispatch create to %s: %w", addr, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("dispatch create %s: %s: %s", addr, resp.Status, strings.TrimSpace(string(b)))
	}
	var out map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&out)
	return out, err
}

// clusteredCreateHandler replaces the node-local /vms create with cluster
// placement: node_id ""/"auto"/"cluster" best-fits across all nodes; an explicit
// peer id dispatches there; this node's own id (or an unknown label) creates local.
func clusteredCreateHandler(db *sql.DB, vmManager *core_vm.VMManager, storagePath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Name       string                 `json:"name"`
			NodeID     string                 `json:"node_id"`
			Tags       map[string]interface{} `json:"tags,omitempty"`
			VCPUs      int                    `json:"vcpus,omitempty"`
			CPUShares  int                    `json:"cpu_shares,omitempty"`
			MemoryMB   int                    `json:"memory_mb,omitempty"`
			DiskSizeGB int                    `json:"disk_size_gb,omitempty"`
			Image      string                 `json:"image,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(req.Name) == "" {
			writeJSONError(w, http.StatusBadRequest, "name is required")
			return
		}
		// Upper bound only: runtime.NumCPU clamping stays in the KVM driver.
		if req.VCPUs > 256 {
			writeJSONError(w, http.StatusBadRequest, "vcpus must be between 1 and 256")
			return
		}
		// requireAuth puts the (string) user_id claim on the context; only a
		// well-formed uuid can own a canonical vms row (owner_id -> users.id).
		userID, _ := r.Context().Value("user_id").(string)
		if _, err := uuid.Parse(userID); err != nil {
			userID = ""
		}
		spec := clusterCreateSpec{
			Name: req.Name, VCPUs: req.VCPUs, CPUShares: req.CPUShares, MemoryMB: req.MemoryMB,
			DiskSizeGB: req.DiskSizeGB, Image: req.Image, Tags: req.Tags,
			OwnerID: userID,
		}

		target := strings.TrimSpace(req.NodeID)
		auto := target == "" || target == "auto" || target == "cluster"
		peers := map[string]string{}
		if vmManager != nil {
			peers = vmManager.MigrationPeers()
		}

		// Decide placement.
		placedNode := selfNodeID()
		placedBy := "explicit-local"
		var dispatchAddr string
		if auto {
			caps := allNodeCapacities(vmManager, storagePath)
			chosen, ok := placeVM(caps, int64(req.MemoryMB), req.DiskSizeGB)
			if !ok {
				writeJSONError(w, http.StatusServiceUnavailable, "no cluster node has capacity for this VM")
				return
			}
			placedNode, placedBy = chosen.NodeID, "cluster-scheduler(best-fit)"
			if chosen.NodeID != selfNodeID() {
				dispatchAddr = chosen.Addr
			}
		} else if addr, isPeer := peers[target]; isPeer && target != selfNodeID() {
			placedNode, placedBy, dispatchAddr = target, "explicit-peer", addr
		}

		// Remote placement: dispatch to the chosen peer, pass its result through.
		if dispatchAddr != "" {
			out, err := dispatchCreateToPeer(dispatchAddr, spec)
			if err != nil {
				writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("cluster dispatch failed: %v", err))
				return
			}
			out["placed_by"] = placedBy
			out["placed_on"] = placedNode
			writeJSON(w, http.StatusCreated, out)
			return
		}
		// Local placement.
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		vmID, state, err := createVMLocal(ctx, db, vmManager, spec)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to create VM: %v", err))
			return
		}
		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id": vmID, "name": req.Name, "state": state, "status": state,
			"vcpus": vcpusOrDefault(spec.VCPUs), "memory_mb": req.MemoryMB, "disk_gb": req.DiskSizeGB,
			"node_id": selfNodeID(), "placed_on": placedNode, "placed_by": placedBy,
			"organization_id": nil,
		})
	}
}

// registerClusterRoutes wires the cluster inventory + placement RPCs. /api/cluster
// is on the authed apiRouter; the /internal/* endpoints are node-to-node (shared
// secret), off the JWT router.
func registerClusterRoutes(root, apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	// GET /api/cluster -- aggregate inventory across all nodes (per-node + sum).
	apiRouter.HandleFunc("/cluster", func(w http.ResponseWriter, r *http.Request) {
		nodes := allNodeCapacities(vmManager, storagePath)
		agg := struct {
			Cores          int   `json:"cores"`
			MemTotalMB     int64 `json:"mem_total_mb"`
			MemAllocatedMB int64 `json:"mem_allocated_mb"`
			StorageTotalGB int64 `json:"storage_total_gb"`
			StorageFreeGB  int64 `json:"storage_free_gb"`
			VMCount        int   `json:"vm_count"`
			NodesReachable int   `json:"nodes_reachable"`
			NodesTotal     int   `json:"nodes_total"`
		}{NodesTotal: len(nodes)}
		for _, n := range nodes {
			if !n.Reachable {
				continue
			}
			agg.NodesReachable++
			agg.Cores += n.Cores
			agg.MemTotalMB += n.MemTotalMB
			agg.MemAllocatedMB += n.MemAllocatedMB
			agg.StorageTotalGB += n.StorageTotalGB
			agg.StorageFreeGB += n.StorageFreeGB
			agg.VMCount += n.VMCount
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"nodes":     nodes,
			"aggregate": agg,
			"note":      "aggregate is the SUM of independent nodes' capacity, usable in parallel via placement -- not a single shared pool a VM can span",
		})
	}).Methods(http.MethodGet)

	// GET /internal/cluster/capacity -- this node's own capacity (node-to-node).
	root.HandleFunc("/internal/cluster/capacity", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		writeJSON(w, http.StatusOK, localNodeCapacity(vmManager, storagePath))
	}).Methods(http.MethodGet)

	// GET /internal/cluster/probe?bytes=N -- N bytes of incompressible payload
	// for the heartbeat's throughput measurement. Capped; 0 disables.
	root.HandleFunc("/internal/cluster/probe", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		n, _ := strconv.Atoi(r.URL.Query().Get("bytes"))
		if n < 0 {
			n = 0
		}
		if n > maxProbeBytes {
			n = maxProbeBytes
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", strconv.Itoa(n))
		w.WriteHeader(http.StatusOK)
		if n > 0 {
			_, _ = w.Write(probePayload[:n])
		}
	}).Methods(http.MethodGet)

	// POST /internal/vms/create -- a coordinator dispatches a create here; we create
	// it locally and return {id, node_id, state}.
	root.HandleFunc("/internal/vms/create", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		var spec clusterCreateSpec
		if err := json.NewDecoder(r.Body).Decode(&spec); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(spec.Name) == "" {
			writeJSONError(w, http.StatusBadRequest, "name is required")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		vmID, state, err := createVMLocal(ctx, db, vmManager, spec)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to create VM: %v", err))
			return
		}
		// A dispatched fabric job is recorded HERE too, so the node actually
		// running the work shows it in its own job list (and a remote cancel
		// can mark it cancelled locally). Best-effort: the submitter keeps the
		// authoritative row, so a failure here only costs local visibility.
		if jobID, _ := spec.Tags["fabric_job_id"].(string); jobID != "" && db != nil {
			if _, jerr := db.ExecContext(ctx, `
				INSERT INTO fabric_jobs (id, vm_id, node_id, command, status, placed_by, created_at, updated_at)
				VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
				ON CONFLICT (id) DO NOTHING
			`, jobID, vmID, selfNodeID(), spec.Command, jobStatusRunning, "dispatched"); jerr != nil {
				log.Printf("fabric job row for dispatched %s not recorded: %v", jobID, jerr)
			}
		}
		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id": vmID, "name": spec.Name, "state": state, "status": state, "node_id": selfNodeID(),
		})
	}).Methods(http.MethodPost)
}
