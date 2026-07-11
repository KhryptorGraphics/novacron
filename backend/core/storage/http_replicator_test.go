package storage

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestReplicaHelperProcess is not a real test: it is the entry point for the
// child storage-agent processes spawned by the multi-process tests below. When
// run normally (env unset) it returns immediately. When the parent re-execs
// this test binary with NOVACRON_REPLICA_HELPER=1, it binds a fresh localhost
// port, prints "READY <addr>" so the parent can dial it, and serves the
// replica handler until the parent kills the process. This gives the tests
// genuinely separate OS processes — real network replication, not in-process
// loopback.
func TestReplicaHelperProcess(t *testing.T) {
	if os.Getenv("NOVACRON_REPLICA_HELPER") != "1" {
		return
	}
	root := os.Getenv("NOVACRON_REPLICA_ROOT")
	secret := os.Getenv("NOVACRON_REPLICA_SECRET")

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Fprintln(os.Stderr, "helper listen:", err)
		os.Exit(1)
	}
	// Announce the actual bound address; the parent scans stdout for this.
	fmt.Printf("READY %s\n", ln.Addr().String())

	srv := &http.Server{Handler: NewReplicaHandler(root, secret)}
	_ = srv.Serve(ln) // blocks until the parent kills this process
}

// replicaProc is a running child storage-agent process.
type replicaProc struct {
	cmd  *exec.Cmd
	addr string
	root string
}

func (rp *replicaProc) baseURL() string { return "http://" + rp.addr }

func (rp *replicaProc) kill() {
	if rp.cmd != nil && rp.cmd.Process != nil {
		_ = rp.cmd.Process.Kill()
		_ = rp.cmd.Wait()
	}
}

// startReplicaProc spawns a real separate OS process running the replica
// storage-agent on its own temp root, and waits for it to become ready.
func startReplicaProc(t *testing.T, secret string) *replicaProc {
	t.Helper()
	root := t.TempDir()

	// -test.timeout=0 so the child never self-aborts while serving; the parent
	// kills it via t.Cleanup.
	cmd := exec.Command(os.Args[0], "-test.run=^TestReplicaHelperProcess$", "-test.timeout=0")
	cmd.Env = append(os.Environ(),
		"NOVACRON_REPLICA_HELPER=1",
		"NOVACRON_REPLICA_ROOT="+root,
		"NOVACRON_REPLICA_SECRET="+secret,
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	cmd.Stderr = os.Stderr // surface child panics/errors in the test log
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper: %v", err)
	}

	rp := &replicaProc{cmd: cmd, root: root}
	t.Cleanup(rp.kill)

	addrCh := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			line := sc.Text()
			if strings.HasPrefix(line, "READY ") {
				addrCh <- strings.TrimPrefix(line, "READY ")
				// Keep draining so the child's stdout never blocks on a full pipe.
				for sc.Scan() {
				}
				return
			}
		}
		close(addrCh) // EOF before READY -> helper failed to start
	}()

	select {
	case addr, ok := <-addrCh:
		if !ok || addr == "" {
			t.Fatalf("helper process exited before signalling READY")
		}
		rp.addr = addr
	case <-time.After(20 * time.Second):
		t.Fatalf("timed out waiting for helper process READY")
	}
	return rp
}

func waitUntil(pred func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if pred() {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return pred()
}

func mustReadFile(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("expected replica on disk at %s: %v", path, err)
	}
	if string(got) != want {
		t.Fatalf("on-disk replica at %s = %q, want %q", path, got, want)
	}
}

// TestHTTPReplicator_MultiProcess is the core proof that the transport is real:
// two genuinely separate OS processes act as two storage nodes, and replicas
// written to each travel over HTTP to that process, land in its own distinct
// on-disk root, and survive independently when the other process dies. A
// same-process loopback could never demonstrate this.
func TestHTTPReplicator_MultiProcess(t *testing.T) {
	const secret = "multiproc-replica-secret"
	a := startReplicaProc(t, secret)
	b := startReplicaProc(t, secret)

	// Genuinely distinct processes and backends.
	if a.cmd.Process.Pid == b.cmd.Process.Pid {
		t.Fatalf("expected two distinct OS processes, both are pid %d", a.cmd.Process.Pid)
	}
	if a.addr == b.addr {
		t.Fatalf("expected two distinct listen addresses, both are %s", a.addr)
	}
	if a.root == b.root {
		t.Fatalf("expected two distinct on-disk roots")
	}

	r := NewHTTPReplicator(secret)
	r.SetNode("node-a", a.baseURL())
	r.SetNode("node-b", b.baseURL())

	ctx := context.Background()
	const volumeID = "vol-multiproc"
	const shardIndex = 0

	// Both nodes must be reachable over the network before we write.
	if !r.NodeAvailable("node-a") || !r.NodeAvailable("node-b") {
		t.Fatalf("both helper nodes should be available; a=%v b=%v",
			r.NodeAvailable("node-a"), r.NodeAvailable("node-b"))
	}
	if r.NodeAvailable("node-unregistered") {
		t.Fatalf("unregistered node must not report available")
	}

	// Write distinct data to each node over HTTP.
	if err := r.WriteReplica(ctx, "node-a", volumeID, shardIndex, []byte("from-a")); err != nil {
		t.Fatalf("write to node-a: %v", err)
	}
	if err := r.WriteReplica(ctx, "node-b", volumeID, shardIndex, []byte("from-b")); err != nil {
		t.Fatalf("write to node-b: %v", err)
	}

	// Read back over the network: each node returns only its own data (no leak).
	dataA, err := r.ReadReplica(ctx, "node-a", volumeID, shardIndex)
	if err != nil || string(dataA) != "from-a" {
		t.Fatalf("read node-a = %q, %v; want %q", dataA, err, "from-a")
	}
	dataB, err := r.ReadReplica(ctx, "node-b", volumeID, shardIndex)
	if err != nil || string(dataB) != "from-b" {
		t.Fatalf("read node-b = %q, %v; want %q", dataB, err, "from-b")
	}

	// Physical proof: the bytes were persisted by each SEPARATE process into
	// its own root, at distinct paths, by a request that crossed the network.
	pathA := filepath.Join(a.root, volumeID, fmt.Sprintf("shard_%d", shardIndex))
	pathB := filepath.Join(b.root, volumeID, fmt.Sprintf("shard_%d", shardIndex))
	if pathA == pathB {
		t.Fatalf("distinct processes must have distinct backend paths")
	}
	mustReadFile(t, pathA, "from-a")
	mustReadFile(t, pathB, "from-b")

	// Independent failability: kill node-b's PROCESS. node-b must become
	// unavailable and reads from it must error (never fabricate); node-a must
	// be entirely unaffected.
	b.kill()
	if !waitUntil(func() bool { return !r.NodeAvailable("node-b") }, 5*time.Second) {
		t.Fatalf("node-b should become unavailable after its process is killed")
	}
	if _, err := r.ReadReplica(ctx, "node-b", volumeID, shardIndex); err == nil {
		t.Fatalf("read from killed node-b must fail, not fabricate data")
	}
	if !r.NodeAvailable("node-a") {
		t.Fatalf("node-a must remain available after node-b dies")
	}
	dataA, err = r.ReadReplica(ctx, "node-a", volumeID, shardIndex)
	if err != nil || string(dataA) != "from-a" {
		t.Fatalf("node-a must survive node-b's loss: got %q, %v", dataA, err)
	}

	// A wrong secret must be rejected (fail-closed auth over the wire).
	bad := NewHTTPReplicator("wrong-secret")
	bad.SetNode("node-a", a.baseURL())
	if err := bad.WriteReplica(ctx, "node-a", volumeID, 1, []byte("nope")); err == nil {
		t.Fatalf("write with wrong secret must be rejected")
	}
}

// TestDistributedStorageService_HTTPReplicator_MultiProcess proves the network
// transport is a drop-in for the Replicator interface: injected via the
// SetReplicator hook, a real DistributedStorageService shards a volume across
// two separate storage-agent PROCESSES and round-trips the data through its
// full write/read pipeline. This does NOT wire the service into the api-server
// (per the single-node-GA boundary, D1) — it is an in-test injection only.
func TestDistributedStorageService_HTTPReplicator_MultiProcess(t *testing.T) {
	const secret = "dss-replica-secret"
	a := startReplicaProc(t, secret)
	b := startReplicaProc(t, secret)

	baseManager, err := NewStorageManager(StorageManagerConfig{
		BasePath:    t.TempDir(),
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	})
	if err != nil {
		t.Fatalf("base manager: %v", err)
	}

	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = t.TempDir()
	distConfig.SynchronousReplication = true // honor the full replication factor before returning

	dss, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("distributed service: %v", err)
	}
	if err := dss.Start(); err != nil {
		t.Fatalf("start distributed service: %v", err)
	}
	defer dss.Stop()

	// Register the two node processes with BOTH the service (for placement) and
	// the network replicator (for transport), then inject the replicator.
	repl := NewHTTPReplicator(secret)
	nodeA := NodeInfo{ID: "node-a", Name: "node-a", Available: true, JoinedAt: time.Now(), LastSeen: time.Now()}
	nodeB := NodeInfo{ID: "node-b", Name: "node-b", Available: true, JoinedAt: time.Now(), LastSeen: time.Now()}
	dss.AddNode(nodeA)
	dss.AddNode(nodeB)
	repl.SetNode("node-a", a.baseURL())
	repl.SetNode("node-b", b.baseURL())
	dss.SetReplicator(repl)

	ctx := context.Background()
	const replicationFactor = 2 // both live nodes must hold a replica
	volume, err := dss.CreateDistributedVolume(ctx, VolumeCreateOptions{
		Name:   "dss-multiproc-volume",
		Type:   VolumeTypeDistributed,
		Format: VolumeFormatRAW,
		Size:   1,
	}, replicationFactor)
	if err != nil {
		t.Fatalf("create distributed volume: %v", err)
	}

	const shardIndex = 0
	payload := []byte("distributed-storage-over-real-network")
	if err := dss.WriteShard(ctx, volume.ID, shardIndex, payload); err != nil {
		t.Fatalf("WriteShard over network: %v", err)
	}

	// The full pipeline round-trips through the two remote processes.
	got, err := dss.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("ReadShard over network: %v", err)
	}
	if string(got) != string(payload) {
		t.Fatalf("ReadShard = %q, want %q", got, payload)
	}

	// Both remote processes physically hold a shard replica on their own root.
	shardName := fmt.Sprintf("shard_%d", shardIndex)
	for _, root := range []string{a.root, b.root} {
		p := filepath.Join(root, volume.ID, shardName)
		info, statErr := os.Stat(p)
		if statErr != nil {
			t.Fatalf("expected replica on remote root %s at %s: %v", root, p, statErr)
		}
		if info.Size() == 0 {
			t.Fatalf("replica at %s is empty", p)
		}
	}

	// Lose one whole process; the shard must still read via failover to the
	// surviving remote replica.
	b.kill()
	afterLoss, err := dss.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("ReadShard after losing a remote process (failover expected): %v", err)
	}
	if string(afterLoss) != string(payload) {
		t.Fatalf("ReadShard after loss = %q, want %q", afterLoss, payload)
	}
}
