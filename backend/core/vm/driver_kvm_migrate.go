package vm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"time"
)

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
