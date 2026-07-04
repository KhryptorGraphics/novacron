package vm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

// kvmMigDiskNode is the stable -blockdev node-name given to a block-migration
// VM's primary disk so it can be drive-mirror'd (source) and NBD-exported (dest).
const kvmMigDiskNode = "migdisk"

// This file implements real live migration for KVMDriverEnhanced by driving
// QEMU's own `migrate` command over QMP. The design is memory-only migration
// over shared storage: the destination reuses the source's exact QEMU args
// (via buildQEMUArgs) plus -incoming, and opens the SAME boot disk and UEFI
// vars with file.locking=off so both processes can hold them open. We do NOT
// reimplement dirty-page copy; QEMU transfers RAM+device state itself.
//
// ponytail: memory-only over shared storage with locking=off on both ends.
// Block migration (dest-own disk, `migrate -b`) is the fallback if a real
// deployment cannot share storage; not implemented here to avoid scope creep.

// runtimeDir returns the per-process directory for a VM's sockets, console and
// pidfile. It defaults to the disk's directory (normal VMs) but a migration
// dest overrides it so it can share the source disk while keeping its own
// runtime files.
func (d *KVMDriverEnhanced) runtimeDir(vmInfo *KVMVMInfo) string {
	if vmInfo.RuntimeDir != "" {
		return vmInfo.RuntimeDir
	}
	return filepath.Dir(vmInfo.DiskPath)
}

// StartMigrationSource starts an already-created VM in shared-storage mode
// (file.locking=off) so a migration destination can open the same disk/vars.
// Caller = the source side of a migration. It is otherwise identical to Start.
func (d *KVMDriverEnhanced) StartMigrationSource(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, ok := d.vms[vmID]
	if !ok {
		return fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State == StateRunning {
		return fmt.Errorf("VM %s is already running", vmID)
	}
	vmInfo.ShareStorage = true
	vmInfo.VNCPort = freeVNCPort()
	log.Printf("Starting KVM migration source %s (shared storage)", vmID)
	return d.launchVM(vmID, vmInfo)
}

// StartIncoming launches a migration destination that mirrors srcVMID (looked up
// in this driver) over shared storage. It is a thin wrapper over
// StartIncomingWithDisk for same-host migration where source and dest live in
// the same driver; cross-node callers (whose driver has no source entry) call
// StartIncomingWithDisk directly with the source's disk path + config.
func (d *KVMDriverEnhanced) StartIncoming(ctx context.Context, srcVMID, destID, destDir, incomingURI string) (string, error) {
	d.vmLock.RLock()
	src, ok := d.vms[srcVMID]
	var diskPath string
	var config VMConfig
	if ok {
		diskPath, config = src.DiskPath, src.Config
	}
	d.vmLock.RUnlock()
	if !ok {
		return "", fmt.Errorf("source VM %s not found", srcVMID)
	}
	return d.StartIncomingWithDisk(ctx, destID, destDir, incomingURI, diskPath, config)
}

// StartIncomingWithDisk launches a migration destination that opens the given
// (shared) disk with the given config and waits for an incoming migration on
// incomingURI, using destDir for its own sockets/console/pidfile. diskPath must
// be reachable on this host (shared storage / same filesystem as the source)
// and the source must run with file.locking=off. Returns the dest VM ID.
func (d *KVMDriverEnhanced) StartIncomingWithDisk(ctx context.Context, destID, destDir, incomingURI, diskPath string, config VMConfig) (string, error) {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	if diskPath == "" {
		return "", fmt.Errorf("incoming migration requires a shared disk path")
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create dest dir: %w", err)
	}

	dest := &KVMVMInfo{
		ID:           destID,
		Config:       config,
		State:        StateCreated,
		DiskPath:     diskPath, // shared storage (memory-only migration)
		ConfigPath:   filepath.Join(destDir, "config.json"),
		MonitorPath:  filepath.Join(destDir, "monitor.sock"),
		VNCPort:      freeVNCPort(),
		RuntimeDir:   destDir,
		ShareStorage: true,
		IncomingURI:  incomingURI,
	}
	d.vms[destID] = dest
	// Persist config.json so a dest-node restart can re-adopt this migrated VM
	// into the manager (adoptManagerVM reads it); else the qemu would be orphaned.
	if err := d.saveVMConfig(dest); err != nil {
		log.Printf("migration dest %s: could not persist config (won't re-adopt on restart): %v", destID, err)
	}

	log.Printf("Starting KVM migration dest %s <- incoming %s (disk %s)", destID, incomingURI, dest.DiskPath)
	if err := d.launchVM(destID, dest); err != nil {
		delete(d.vms, destID)
		return "", err
	}

	// Block until the dest is actually waiting for the incoming stream (QMP
	// status "inmigrate"), so the caller can issue migrate without racing the
	// listener. Probed over QMP, never by dialing the migration port.
	if err := waitIncomingReady(filepath.Join(destDir, "qmp.sock"), 60*time.Second); err != nil {
		_ = d.stopVMInternal(dest)
		delete(d.vms, destID)
		return "", fmt.Errorf("dest %s not ready for incoming: %w", destID, err)
	}
	return destID, nil
}

// waitIncomingReady polls the destination QMP socket until the VM reports it is
// waiting for an incoming migration ("inmigrate").
func waitIncomingReady(qmpSock string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if q, err := qmpDial(qmpSock, 2*time.Second); err == nil {
			raw, e := q.execute("query-status", nil)
			q.Close()
			if e == nil {
				var st struct {
					Status string `json:"status"`
				}
				if json.Unmarshal(raw, &st) == nil && st.Status == "inmigrate" {
					return nil
				}
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for QMP inmigrate on %s", qmpSock)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// Migrate drives QEMU's migration on the source: it waits for the destination
// to be listening, issues `migrate` over the source QMP socket, polls
// query-migrate to completion, then quits the (now paused) source so it exits.
// target is the QEMU migration URI (e.g. "tcp:127.0.0.1:4444"); params["uri"]
// overrides it. Returns after the source has been asked to quit.
func (d *KVMDriverEnhanced) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	_, _, err := d.migrateWithStats(ctx, vmID, target, params)
	return err
}

// migrateWithStats is Migrate's implementation, additionally returning the QEMU
// downtime and total_time (ms) read from query-migrate on completion. Kept
// separate so callers/tests can observe the recorded migration stats without
// widening the VMDriver.Migrate interface signature.
func (d *KVMDriverEnhanced) migrateWithStats(ctx context.Context, vmID, target string, params map[string]string) (downtimeMs, totalMs int64, err error) {
	d.vmLock.RLock()
	vmInfo, ok := d.vms[vmID]
	d.vmLock.RUnlock()
	if !ok {
		return 0, 0, fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return 0, 0, fmt.Errorf("VM %s is not running", vmID)
	}

	uri := target
	if p := params["uri"]; p != "" {
		uri = p
	}
	if uri == "" {
		return 0, 0, fmt.Errorf("migration target URI required (target_node is a node id, not a QEMU URI)")
	}

	// Dest readiness is ensured by StartIncoming (it waits for QMP inmigrate).
	// We must NOT dial the migration URI to probe it: qemu consumes the first
	// connection on -incoming as the migration stream, which would poison it
	// ("Not a migration stream").
	sock := filepath.Join(d.runtimeDir(vmInfo), "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		return 0, 0, fmt.Errorf("connect source QMP %s: %w", sock, err)
	}
	defer q.Close()

	if _, err := q.execute("migrate", map[string]interface{}{"uri": uri}); err != nil {
		return 0, 0, fmt.Errorf("migrate command: %w", err)
	}

	downtimeMs, totalMs, err = q.pollMigration(ctx)
	if err != nil {
		return 0, 0, err
	}

	// Migration completed: the source is paused (postmigrate). Quit it so the
	// process exits and only the destination remains running.
	_, _ = q.execute("quit", nil)
	log.Printf("VM %s migrated to %s (downtime %dms, total %dms)", vmID, uri, downtimeMs, totalMs)
	return downtimeMs, totalMs, nil
}

// --- minimal QMP (QEMU Machine Protocol) client ---------------------------

type qmpConn struct {
	c net.Conn
	r *bufio.Reader
}

// qmpDial connects to a QMP unix socket, reads the greeting and negotiates
// capabilities so the connection is ready for commands.
func qmpDial(sock string, timeout time.Duration) (*qmpConn, error) {
	c, err := net.DialTimeout("unix", sock, timeout)
	if err != nil {
		return nil, err
	}
	q := &qmpConn{c: c, r: bufio.NewReader(c)}
	if _, err := q.readObject(); err != nil { // QMP greeting
		q.Close()
		return nil, fmt.Errorf("read QMP greeting: %w", err)
	}
	if _, err := q.execute("qmp_capabilities", nil); err != nil {
		q.Close()
		return nil, fmt.Errorf("qmp_capabilities: %w", err)
	}
	return q, nil
}

func (q *qmpConn) Close() error { return q.c.Close() }

// readObject reads one newline-delimited JSON object from the QMP stream.
func (q *qmpConn) readObject() (map[string]json.RawMessage, error) {
	line, err := q.r.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(line, &m); err != nil {
		return nil, fmt.Errorf("bad QMP json %q: %w", string(line), err)
	}
	return m, nil
}

// execute sends a QMP command and returns its "return" payload, skipping any
// asynchronous events that arrive before the reply.
func (q *qmpConn) execute(cmd string, args map[string]interface{}) (json.RawMessage, error) {
	req := map[string]interface{}{"execute": cmd}
	if args != nil {
		req["arguments"] = args
	}
	b, _ := json.Marshal(req)
	if _, err := q.c.Write(append(b, '\n')); err != nil {
		return nil, err
	}
	for {
		m, err := q.readObject()
		if err != nil {
			return nil, err
		}
		if e, ok := m["error"]; ok {
			return nil, fmt.Errorf("qmp %s: %s", cmd, string(e))
		}
		if r, ok := m["return"]; ok {
			return r, nil
		}
		// otherwise it is an async event; keep reading for the reply
	}
}

// pollMigration polls query-migrate until the migration reaches a terminal
// state, returning downtime and total_time in milliseconds on success.
func (q *qmpConn) pollMigration(ctx context.Context) (downtimeMs, totalMs int64, err error) {
	for {
		if ctx.Err() != nil {
			return 0, 0, ctx.Err()
		}
		raw, err := q.execute("query-migrate", nil)
		if err != nil {
			return 0, 0, fmt.Errorf("query-migrate: %w", err)
		}
		var st struct {
			Status    string `json:"status"`
			Downtime  int64  `json:"downtime"`
			TotalTime int64  `json:"total_time"`
		}
		if err := json.Unmarshal(raw, &st); err != nil {
			return 0, 0, fmt.Errorf("parse query-migrate: %w", err)
		}
		switch st.Status {
		case "completed":
			return st.Downtime, st.TotalTime, nil
		case "failed", "cancelled":
			return 0, 0, fmt.Errorf("migration %s", st.Status)
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// --- block (non-shared storage) migration: NBD drive-mirror + RAM ------------
//
// Empirically verified on QEMU 8.2.2 (x86 KVM + arm64 TCG): the destination has
// its OWN empty disk exposed over NBD writable; the source drive-mirrors its live
// disk into that export (copy-mode=write-blocking so no last-moment guest write is
// lost at the RAM cutover) until "ready", migrates RAM, then CANCELS the mirror
// (the dest owns its disk; pivoting would redirect the departing source onto the
// NBD target). This is the modern `virsh migrate --copy-storage-all` equivalent;
// the legacy `migrate -b` capability (deprecated 8.2, removed 9.1) is not used.

// The primary disk is a named -blockdev ("migdisk") for EVERY VM, so any running
// VM can be drive-mirror'd without relaunch (block migration = host evacuation of
// arbitrary VMs); no special source-launch mode is needed.

// StartIncomingBlock launches a block-migration destination: it creates its OWN
// empty qcow2 of virtualSizeBytes (must equal the source disk's virtual size),
// launches with -incoming + the named -blockdev, waits for QMP inmigrate, then
// exposes its disk over NBD writable. Returns the dest VM ID and the nbd:// URI
// the source must drive-mirror into (advertiseHost is the address the source
// reaches this host on: 127.0.0.1 for same-host, the node IP for cross-node).
func (d *KVMDriverEnhanced) StartIncomingBlock(ctx context.Context, destID, destDir, incomingURI, advertiseHost string, virtualSizeBytes int64, config VMConfig) (string, string, error) {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	if virtualSizeBytes <= 0 {
		return "", "", fmt.Errorf("block-migration dest requires the source disk virtual size")
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", "", fmt.Errorf("failed to create dest dir: %w", err)
	}
	destDisk := filepath.Join(destDir, "disk.qcow2")
	if err := createEmptyQcow2(ctx, destDisk, virtualSizeBytes); err != nil {
		return "", "", err
	}

	dest := &KVMVMInfo{
		ID:           destID,
		Config:       config,
		State:        StateCreated,
		DiskPath:     destDisk, // its OWN disk (not shared)
		ConfigPath:   filepath.Join(destDir, "config.json"),
		MonitorPath:  filepath.Join(destDir, "monitor.sock"),
		VNCPort:      freeVNCPort(),
		RuntimeDir:   destDir,
		IncomingURI:  incomingURI,
	}
	d.vms[destID] = dest
	// Persist config.json so a dest-node restart can re-adopt this migrated VM into
	// the manager (adoptManagerVM reads it); without it the qemu would be orphaned.
	if err := d.saveVMConfig(dest); err != nil {
		log.Printf("block-migration dest %s: could not persist config (won't re-adopt on restart): %v", destID, err)
	}

	log.Printf("Starting KVM block-migration dest %s <- incoming %s (own disk %s)", destID, incomingURI, destDisk)
	if err := d.launchVM(destID, dest); err != nil {
		delete(d.vms, destID)
		return "", "", err
	}

	qmpSock := filepath.Join(destDir, "qmp.sock")
	if err := waitIncomingReady(qmpSock, 60*time.Second); err != nil {
		_ = d.stopVMInternal(dest)
		delete(d.vms, destID)
		return "", "", fmt.Errorf("dest %s not ready for incoming: %w", destID, err)
	}

	nbdPort, err := freeMigrationPort()
	if err != nil {
		_ = d.stopVMInternal(dest)
		delete(d.vms, destID)
		return "", "", err
	}
	if err := nbdExportDisk(qmpSock, nbdPort); err != nil {
		_ = d.stopVMInternal(dest)
		delete(d.vms, destID)
		return "", "", fmt.Errorf("dest %s NBD export failed: %w", destID, err)
	}
	nbdURI := fmt.Sprintf("nbd://%s:%d/%s", advertiseHost, nbdPort, kvmMigDiskNode)
	log.Printf("Block-migration dest %s exposes disk over %s", destID, nbdURI)
	return destID, nbdURI, nil
}

// migrateBlockWithStats runs block migration on the source: drive-mirror the disk
// into the dest's NBD export (write-blocking) -> wait ready -> migrate RAM ->
// cancel the mirror -> quit the source. Returns QEMU downtime/total (ms). The
// caller must invoke FinishIncomingBlock on the dest afterwards to tear down the
// export. nbdURI comes from StartIncomingBlock.
func (d *KVMDriverEnhanced) migrateBlockWithStats(ctx context.Context, vmID, ramURI, nbdURI string) (downtimeMs, totalMs int64, err error) {
	d.vmLock.RLock()
	vmInfo, ok := d.vms[vmID]
	d.vmLock.RUnlock()
	if !ok {
		return 0, 0, fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return 0, 0, fmt.Errorf("VM %s is not running", vmID)
	}
	if ramURI == "" || nbdURI == "" {
		return 0, 0, fmt.Errorf("block migration requires both a RAM URI and an NBD target URI")
	}

	sock := filepath.Join(d.runtimeDir(vmInfo), "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		return 0, 0, fmt.Errorf("connect source QMP %s: %w", sock, err)
	}
	defer q.Close()

	const jobID = "mir0"
	// sync=full copies the whole device; mode=existing opens the pre-created NBD
	// target; format=raw = how to read the NBD target (raw guest sectors),
	// independent of the dest's qcow2 storage; copy-mode=write-blocking makes
	// guest writes block until mirrored so the target is current at the RAM
	// cutover (the default background mode would silently drop last writes).
	if _, err := q.execute("drive-mirror", map[string]interface{}{
		"job-id":    jobID,
		"device":    kvmMigDiskNode,
		"target":    nbdURI,
		"sync":      "full",
		"mode":      "existing",
		"format":    "raw",
		"copy-mode": "write-blocking",
	}); err != nil {
		return 0, 0, fmt.Errorf("drive-mirror: %w", err)
	}

	if err := pollBlockJobReady(ctx, q, jobID, 10*time.Minute); err != nil {
		return 0, 0, err
	}

	// Generous downtime ceiling so a slow (TCG) guest still converges in one
	// stop-and-copy; harmless under KVM (actual downtime stays small).
	_, _ = q.execute("migrate-set-parameters", map[string]interface{}{"downtime-limit": 5000})

	if _, err := q.execute("migrate", map[string]interface{}{"uri": ramURI}); err != nil {
		return 0, 0, fmt.Errorf("migrate command: %w", err)
	}
	downtimeMs, totalMs, err = q.pollMigration(ctx)
	if err != nil {
		return 0, 0, err
	}

	// Dest now owns its fully-mirrored disk. Cancel the source mirror (NOT
	// block-job-complete, which would pivot the departing source onto the target).
	if _, err := q.execute("block-job-cancel", map[string]interface{}{"device": jobID}); err != nil {
		log.Printf("block-migration %s: block-job-cancel warning: %v", vmID, err)
	}
	// block-job-cancel is ASYNC: the source's NBD client disconnects only when the
	// job concludes. Wait for that here so the caller's FinishIncomingBlock (dest
	// block-export-del) can't race into "export 'exp0' still in use".
	if err := pollBlockJobGone(ctx, q, jobID, 30*time.Second); err != nil {
		log.Printf("block-migration %s: mirror job did not conclude cleanly: %v", vmID, err)
	}
	_, _ = q.execute("quit", nil)
	log.Printf("VM %s block-migrated to %s (nbd %s, downtime %dms, total %dms)", vmID, ramURI, nbdURI, downtimeMs, totalMs)
	return downtimeMs, totalMs, nil
}

// FinishIncomingBlock tears down the dest's NBD export after the source has cut
// over and closed its mirror client. Call on the DEST after migrateBlockWithStats
// returns (ordering matters: the source's block-job-cancel/quit must happen first,
// else block-export-del fails with "export still in use"). Best-effort.
func (d *KVMDriverEnhanced) FinishIncomingBlock(ctx context.Context, destID string) error {
	d.vmLock.RLock()
	dest, ok := d.vms[destID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("dest VM %s not found", destID)
	}
	q, err := qmpDial(filepath.Join(d.runtimeDir(dest), "qmp.sock"), 10*time.Second)
	if err != nil {
		return fmt.Errorf("connect dest QMP: %w", err)
	}
	defer q.Close()
	if _, err := q.execute("block-export-del", map[string]interface{}{"id": "exp0"}); err != nil {
		log.Printf("block-migration dest %s: block-export-del warning: %v", destID, err)
	}
	if _, err := q.execute("nbd-server-stop", nil); err != nil {
		log.Printf("block-migration dest %s: nbd-server-stop warning: %v", destID, err)
	}
	return nil
}

// WaitResumed blocks until the dest VM's incoming migration completes and the
// guest resumes (query-status transitions inmigrate -> running), or the timeout
// elapses. It gates both the block-migration NBD teardown and registering the
// migrated VM in the destination node's manager/DB (the guest is only known-good
// on the dest once it has resumed).
func (d *KVMDriverEnhanced) WaitResumed(destID string, timeout time.Duration) error {
	d.vmLock.RLock()
	dest, ok := d.vms[destID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("dest %s not found", destID)
	}
	qmpSock := filepath.Join(d.runtimeDir(dest), "qmp.sock")
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if q, err := qmpDial(qmpSock, 5*time.Second); err == nil {
			raw, e := q.execute("query-status", nil)
			q.Close()
			if e == nil {
				var st struct {
					Status string `json:"status"`
				}
				if json.Unmarshal(raw, &st) == nil && st.Status == "running" {
					return nil
				}
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("dest %s did not resume within %s", destID, timeout)
}

// AwaitFinishIncomingBlock waits for the incoming migration to resume, then tears
// down the dest's NBD export. Cross-node, the dest cannot observe the source's
// async block-job-cancel, so it retries block-export-del (which fails "in use"
// until the source's mirror client disconnects) until it succeeds or the timeout
// elapses, then stops the NBD server. Returns nil once the guest has resumed
// (teardown is best-effort after that); non-nil if it never resumed -- the caller
// uses that to decide whether to register the migrated VM. Meant to run in a
// goroutine on the dest after StartIncomingBlock returns.
func (d *KVMDriverEnhanced) AwaitFinishIncomingBlock(destID string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	// 1. WAIT for the incoming migration to complete on this dest. Deleting the
	// export before this would kill the source's in-flight drive-mirror -- the
	// export only reports "in use" while a client is attached, so a premature
	// delete silently succeeds and leaves the source mirroring into a dead target.
	if err := d.WaitResumed(destID, timeout); err != nil {
		log.Printf("block-migration dest %s: never resumed; leaving NBD export up", destID)
		return err
	}

	d.vmLock.RLock()
	dest, ok := d.vms[destID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("dest %s not found", destID)
	}
	qmpSock := filepath.Join(d.runtimeDir(dest), "qmp.sock")

	// 2. Now retry block-export-del until the source's async block-job-cancel
	// disconnects its NBD client (fails "in use" until then), then stop the server.
	for time.Now().Before(deadline) {
		if q, err := qmpDial(qmpSock, 5*time.Second); err == nil {
			_, delErr := q.execute("block-export-del", map[string]interface{}{"id": "exp0"})
			if delErr == nil {
				_, _ = q.execute("nbd-server-stop", nil)
				q.Close()
				log.Printf("block-migration dest %s: NBD export torn down", destID)
				return nil
			}
			q.Close()
		}
		time.Sleep(1 * time.Second)
	}
	log.Printf("block-migration dest %s: NBD export teardown timed out", destID)
	return nil // resumed; teardown is best-effort
}

// nbdExportDisk starts the QEMU NBD server on the dest QMP socket and exports the
// migration disk node writable, so a source can drive-mirror into this dest.
func nbdExportDisk(qmpSock string, nbdPort int) error {
	q, err := qmpDial(qmpSock, 10*time.Second)
	if err != nil {
		return fmt.Errorf("connect dest QMP: %w", err)
	}
	defer q.Close()
	// port must be a JSON STRING in the inet SocketAddress.
	if _, err := q.execute("nbd-server-start", map[string]interface{}{
		"addr": map[string]interface{}{
			"type": "inet",
			"data": map[string]interface{}{"host": "0.0.0.0", "port": strconv.Itoa(nbdPort)},
		},
	}); err != nil {
		return fmt.Errorf("nbd-server-start: %w", err)
	}
	// block-export-add is the modern export command (not deprecated nbd-server-add).
	if _, err := q.execute("block-export-add", map[string]interface{}{
		"type":      "nbd",
		"id":        "exp0",
		"node-name": kvmMigDiskNode,
		"writable":  true,
		"name":      kvmMigDiskNode,
	}); err != nil {
		return fmt.Errorf("block-export-add: %w", err)
	}
	return nil
}

// pollBlockJobReady polls query-block-jobs until the named job reports ready (its
// mirror has caught up to the source), or ctx/timeout fires.
func pollBlockJobReady(ctx context.Context, q *qmpConn, jobID string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		raw, err := q.execute("query-block-jobs", nil)
		if err != nil {
			return fmt.Errorf("query-block-jobs: %w", err)
		}
		var jobs []struct {
			Device string `json:"device"`
			Ready  bool   `json:"ready"`
			Status string `json:"status"`
		}
		if err := json.Unmarshal(raw, &jobs); err != nil {
			return fmt.Errorf("parse query-block-jobs: %w", err)
		}
		for _, j := range jobs {
			if j.Device == jobID {
				if j.Ready {
					return nil
				}
				if j.Status == "aborting" || j.Status == "concluded" {
					return fmt.Errorf("mirror job %s ended without ready (status %s)", jobID, j.Status)
				}
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("mirror job %s not ready within %s", jobID, timeout)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// pollBlockJobGone polls query-block-jobs until the named job is no longer present
// (a cancelled/concluded mirror auto-dismisses), i.e. the source's NBD client has
// disconnected -- so the dest export can then be deleted without "still in use".
func pollBlockJobGone(ctx context.Context, q *qmpConn, jobID string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		raw, err := q.execute("query-block-jobs", nil)
		if err != nil {
			return fmt.Errorf("query-block-jobs: %w", err)
		}
		var jobs []struct {
			Device string `json:"device"`
		}
		if err := json.Unmarshal(raw, &jobs); err != nil {
			return fmt.Errorf("parse query-block-jobs: %w", err)
		}
		present := false
		for _, j := range jobs {
			if j.Device == jobID {
				present = true
			}
		}
		if !present {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("mirror job %s did not conclude within %s", jobID, timeout)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// createEmptyQcow2 creates an empty qcow2 of exactly virtualSizeBytes -- the
// source disk's virtual size, which a sync=full drive-mirror into it requires.
func createEmptyQcow2(ctx context.Context, path string, virtualSizeBytes int64) error {
	cmd := exec.CommandContext(ctx, "qemu-img", "create", "-f", "qcow2", path, strconv.FormatInt(virtualSizeBytes, 10))
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("create dest disk: %w: %s", err, string(out))
	}
	return nil
}

// sourceDiskVirtualSize returns the virtual size (bytes) of a qcow2 disk via
// qemu-img info, so a block-migration dest can create a matching empty target.
func sourceDiskVirtualSize(ctx context.Context, diskPath string) (int64, error) {
	// -U (force-share): the source qemu holds the disk's write lock while running;
	// virtual-size is static header metadata, safe to read without the lock.
	out, err := exec.CommandContext(ctx, "qemu-img", "info", "--output=json", "-U", diskPath).Output()
	if err != nil {
		return 0, fmt.Errorf("qemu-img info %s: %w", diskPath, err)
	}
	var info struct {
		VirtualSize int64 `json:"virtual-size"`
	}
	if err := json.Unmarshal(out, &info); err != nil {
		return 0, fmt.Errorf("parse qemu-img info: %w", err)
	}
	if info.VirtualSize <= 0 {
		return 0, fmt.Errorf("qemu-img info returned non-positive virtual-size for %s", diskPath)
	}
	return info.VirtualSize, nil
}

// freeMigrationPort asks the kernel for an unused TCP port for the NBD server.
func freeMigrationPort() (int, error) {
	ln, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}
