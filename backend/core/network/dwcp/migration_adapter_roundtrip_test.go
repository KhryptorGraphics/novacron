package dwcp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// receivedMemory/receivedDisk capture what a test's MigrationAdapter
// received via OnMemoryReceived/OnDiskReceived, synchronized for the
// receiving goroutine (handleIncomingMigration runs independently of the
// sending MigrateVMMemory/MigrateVMDisk call) to hand off to the test.
type receivedMemory struct {
	vmID string
	data []byte
}

type receivedDiskBlock struct {
	vmID    string
	blockID int
	data    []byte
}

// newLoopbackMigrationAdapter creates one MigrationAdapter listening on a
// free loopback port, wired to report every received memory/disk payload
// on the returned channels. The same adapter is used as both sender and
// receiver (self-loopback: MigrateVMMemory/MigrateVMDisk target the
// adapter's own ListenPort on 127.0.0.1) — a valid, minimal topology for
// proving the wire protocol round-trips correctly end to end.
func newLoopbackMigrationAdapter(t *testing.T, enableDWCP bool) (adapter *dwcp.MigrationAdapter, memCh chan receivedMemory, diskCh chan receivedDiskBlock) {
	t.Helper()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to find free port: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()

	memCh = make(chan receivedMemory, 8)
	diskCh = make(chan receivedDiskBlock, 64)

	adapter, err = dwcp.NewMigrationAdapter(dwcp.MigrationAdapterConfig{
		EnableDWCP:        enableDWCP,
		EnableFallback:    false,
		ListenPort:        port,
		ConnectionTimeout: 5 * time.Second,
		OnMemoryReceived: func(vmID string, data []byte) {
			cp := make([]byte, len(data))
			copy(cp, data)
			memCh <- receivedMemory{vmID: vmID, data: cp}
		},
		OnDiskReceived: func(vmID string, blockID int, data []byte) {
			cp := make([]byte, len(data))
			copy(cp, data)
			diskCh <- receivedDiskBlock{vmID: vmID, blockID: blockID, data: cp}
		},
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter failed: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	listenErrCh := make(chan error, 1)
	go func() { listenErrCh <- adapter.ListenForMigrations(ctx) }()

	// Give the listener a moment to bind before any test dials it.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 50*time.Millisecond)
		if err == nil {
			c.Close()
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	return adapter, memCh, diskCh
}

func waitForMemory(t *testing.T, ch chan receivedMemory, timeout time.Duration) receivedMemory {
	t.Helper()
	select {
	case m := <-ch:
		return m
	case <-time.After(timeout):
		t.Fatal("timed out waiting for OnMemoryReceived callback")
		return receivedMemory{}
	}
}

func waitForDiskBlocks(t *testing.T, ch chan receivedDiskBlock, count int, timeout time.Duration) map[int]receivedDiskBlock {
	t.Helper()
	blocks := make(map[int]receivedDiskBlock, count)
	deadline := time.After(timeout)
	for len(blocks) < count {
		select {
		case b := <-ch:
			blocks[b.blockID] = b
		case <-deadline:
			t.Fatalf("timed out waiting for disk blocks: got %d of %d", len(blocks), count)
		}
	}
	return blocks
}

// TestMigrationRoundTrip_StandardMemory proves the standard (non-DWCP)
// memory path — long broken (handleIncomingMigration's single type-byte
// read never matched what migrateMemoryStandard's 8-byte size header
// actually sends, so it silently misframed every connection; see
// novacron-lce) — now round-trips byte-exact end to end.
func TestMigrationRoundTrip_StandardMemory(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapter(t, false)

	original := make([]byte, 256*1024)
	rand.Read(original)

	if err := adapter.MigrateVMMemory(context.Background(), "vm-std-mem", original, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMMemory failed: %v", err)
	}

	got := waitForMemory(t, memCh, 5*time.Second)
	if got.vmID != "vm-std-mem" {
		t.Errorf("vmID mismatch: got %q, want %q", got.vmID, "vm-std-mem")
	}
	if !bytes.Equal(got.data, original) {
		t.Fatalf("standard memory round trip corrupted data: got %d bytes, want %d bytes matching original", len(got.data), len(original))
	}
}

// TestMigrationRoundTrip_StandardDisk proves the standard disk path
// round-trips byte-exact end to end, including correct per-block
// reassembly.
func TestMigrationRoundTrip_StandardDisk(t *testing.T) {
	adapter, _, diskCh := newLoopbackMigrationAdapter(t, false)

	blocks := make(map[int][]byte)
	for i := 0; i < 5; i++ {
		b := make([]byte, 16*1024)
		rand.Read(b)
		blocks[i] = b
	}

	if err := adapter.MigrateVMDisk(context.Background(), "vm-std-disk", blocks, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMDisk failed: %v", err)
	}

	got := waitForDiskBlocks(t, diskCh, len(blocks), 5*time.Second)
	for id, want := range blocks {
		block, ok := got[id]
		if !ok {
			t.Fatalf("block %d never received", id)
		}
		if block.vmID != "vm-std-disk" {
			t.Errorf("block %d vmID mismatch: got %q, want %q", id, block.vmID, "vm-std-disk")
		}
		if !bytes.Equal(block.data, want) {
			t.Fatalf("block %d round trip corrupted data: got %d bytes, want %d bytes matching original", id, len(block.data), len(want))
		}
	}
}

// TestMigrationRoundTrip_DWCPMemory proves the DWCP (AMST+HDE) memory
// path — receiveMemory was an empty stub before this fix (novacron-lce)
// — round-trips byte-exact end to end: real AMST.Transfer over a real
// loopback TCP connection, real HDE.CompressMemory on the sender, real
// AMST.Receive + HDE.Decompress on the receiver.
func TestMigrationRoundTrip_DWCPMemory(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapter(t, true)

	// Realistic VM-memory-shaped data (not pure random): HDE compresses
	// this meaningfully, unlike pure noise, exercising the compress+
	// transfer+receive+decompress path the way it would actually be used.
	original := make([]byte, 1024*1024)
	for i := 0; i < len(original); i += 4096 {
		end := i + 4096
		if end > len(original) {
			end = len(original)
		}
		switch (i / 4096) % 3 {
		case 0:
		case 1:
			for j := i; j < end; j++ {
				original[j] = byte(j % 256)
			}
		case 2:
			rand.Read(original[i:end])
		}
	}

	if err := adapter.MigrateVMMemory(context.Background(), "vm-dwcp-mem", original, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMMemory failed: %v", err)
	}

	got := waitForMemory(t, memCh, 10*time.Second)
	if got.vmID != "vm-dwcp-mem" {
		t.Errorf("vmID mismatch: got %q, want %q", got.vmID, "vm-dwcp-mem")
	}
	if !bytes.Equal(got.data, original) {
		t.Fatalf("DWCP memory round trip corrupted data: got %d bytes, want %d bytes matching original", len(got.data), len(original))
	}
}

// TestMigrationRoundTrip_DWCPDisk proves the DWCP disk path round-trips
// byte-exact end to end, including correct per-block
// compress/transfer/receive/decompress/reassembly.
func TestMigrationRoundTrip_DWCPDisk(t *testing.T) {
	adapter, _, diskCh := newLoopbackMigrationAdapter(t, true)

	blocks := make(map[int][]byte)
	for i := 0; i < 5; i++ {
		b := make([]byte, 32*1024)
		switch i % 3 {
		case 0:
			// zero block
		case 1:
			for j := range b {
				b[j] = byte(j % 256)
			}
		case 2:
			rand.Read(b)
		}
		blocks[i] = b
	}

	if err := adapter.MigrateVMDisk(context.Background(), "vm-dwcp-disk", blocks, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMDisk failed: %v", err)
	}

	got := waitForDiskBlocks(t, diskCh, len(blocks), 10*time.Second)
	for id, want := range blocks {
		block, ok := got[id]
		if !ok {
			t.Fatalf("block %d never received", id)
		}
		if !bytes.Equal(block.data, want) {
			t.Fatalf("DWCP disk block %d round trip corrupted data: got %d bytes, want %d bytes matching original", id, len(block.data), len(want))
		}
	}
}

// TestMigrationRoundTrip_SecondMigrationSameVM proves createConnection's
// always-fresh-connection design (no reuse across migrations — see its
// doc comment) actually holds under a second migration of the same VM,
// and that HDE.CompressMemory's per-vmID baseline tracking (which would
// try to delta-encode the second call if EnableDelta were on) does not
// corrupt the result now that EnableDelta is forced off for this
// adapter's HDE (NewMigrationAdapter) — both migrations must round-trip
// byte-exact independently.
func TestMigrationRoundTrip_SecondMigrationSameVM(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapter(t, true)

	first := make([]byte, 512*1024)
	rand.Read(first)
	if err := adapter.MigrateVMMemory(context.Background(), "vm-repeat", first, "127.0.0.1", nil); err != nil {
		t.Fatalf("first MigrateVMMemory failed: %v", err)
	}
	got1 := waitForMemory(t, memCh, 10*time.Second)
	if !bytes.Equal(got1.data, first) {
		t.Fatalf("first migration corrupted data: got %d bytes, want %d bytes", len(got1.data), len(first))
	}

	// Second migration of the SAME vmID with DIFFERENT data — if
	// createConnection reused a cached, already-Transfer'd connection, or
	// if delta encoding were silently active, this would fail to
	// round-trip cleanly.
	second := make([]byte, 512*1024)
	rand.Read(second)
	if err := adapter.MigrateVMMemory(context.Background(), "vm-repeat", second, "127.0.0.1", nil); err != nil {
		t.Fatalf("second MigrateVMMemory failed: %v", err)
	}
	got2 := waitForMemory(t, memCh, 10*time.Second)
	if !bytes.Equal(got2.data, second) {
		t.Fatalf("second migration corrupted data: got %d bytes, want %d bytes", len(got2.data), len(second))
	}
	if bytes.Equal(got2.data, first) {
		t.Fatal("second migration returned the FIRST migration's data — connection reuse or delta corruption regression")
	}
}

// TestMigrationRoundTrip_DictionaryPresentButDisabled proves the
// EnableDictionary=false kill-switch (NewMigrationAdapter) holds even
// when a trained dictionary already exists in the adapter's HDE instance
// for the exact key CompressDisk would look up — guards against a future
// regression where CompressDisk's guard changes from "EnableDictionary"
// to "dictionary exists" (see novacron-o0e).
func TestMigrationRoundTrip_DictionaryPresentButDisabled(t *testing.T) {
	adapter, _, diskCh := newLoopbackMigrationAdapter(t, true)

	vmID := "vm-dict-present"
	samples := make([][]byte, 20)
	for i := range samples {
		s := make([]byte, 4096)
		switch i % 3 {
		case 0:
		case 1:
			for j := range s {
				s[j] = byte(j % 256)
			}
		case 2:
			rand.Read(s)
		}
		samples[i] = s
	}
	// CompressDisk looks up hde.getDictionary(vmID + "_disk").
	if err := adapter.TrainDictionary(vmID+"_disk", samples); err != nil {
		t.Fatalf("TrainDictionary failed: %v", err)
	}

	block := make([]byte, 32*1024)
	rand.Read(block)
	blocks := map[int][]byte{0: block}

	if err := adapter.MigrateVMDisk(context.Background(), vmID, blocks, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMDisk failed: %v", err)
	}

	got := waitForDiskBlocks(t, diskCh, 1, 10*time.Second)
	if !bytes.Equal(got[0].data, block) {
		t.Fatalf("disk round trip corrupted with a trained dictionary present: got %d bytes, want %d bytes matching original", len(got[0].data), len(block))
	}
}

// TestMigrationRoundTrip_Concurrent proves multiple sequential-per-VM but
// concurrent-across-VM migrations on one adapter do not cross-contaminate
// — each vmID's data must arrive intact and attributed to the right vmID.
func TestMigrationRoundTrip_Concurrent(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapter(t, true)

	const n = 4
	want := make(map[string][]byte, n)
	var wg sync.WaitGroup
	errCh := make(chan error, n)
	for i := 0; i < n; i++ {
		vmID := fmt.Sprintf("vm-concurrent-%d", i)
		data := make([]byte, 64*1024)
		rand.Read(data)
		want[vmID] = data

		wg.Add(1)
		go func(vmID string, data []byte) {
			defer wg.Done()
			if err := adapter.MigrateVMMemory(context.Background(), vmID, data, "127.0.0.1", nil); err != nil {
				errCh <- fmt.Errorf("%s: %w", vmID, err)
			}
		}(vmID, data)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent migration failed: %v", err)
	}

	got := make(map[string][]byte, n)
	for len(got) < n {
		m := waitForMemory(t, memCh, 10*time.Second)
		got[m.vmID] = m.data
	}

	for vmID, wantData := range want {
		gotData, ok := got[vmID]
		if !ok {
			t.Fatalf("%s: never received", vmID)
		}
		if !bytes.Equal(gotData, wantData) {
			t.Fatalf("%s: round trip corrupted data (cross-contamination between concurrent migrations?)", vmID)
		}
	}
}

// TestMigrationRoundTrip_StandardMemoryRejectsOversizedHeader proves a
// wire-supplied size that exceeds MaxMemoryUsage is rejected before
// attempting to allocate it — a garbage or hostile 8-byte size header
// must not be able to drive an unbounded make([]byte, size) and crash
// the process. Connects raw (bypassing MigrateVMMemory) to control the
// wire bytes directly.
func TestMigrationRoundTrip_StandardMemoryRejectsOversizedHeader(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to find free port: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()

	recvAdapter, err := dwcp.NewMigrationAdapter(dwcp.MigrationAdapterConfig{
		EnableDWCP:        false,
		ListenPort:        port,
		ConnectionTimeout: 2 * time.Second,
		MaxMemoryUsage:    1024, // deliberately tiny cap for this test
		OnMemoryReceived: func(vmID string, data []byte) {
			t.Errorf("OnMemoryReceived unexpectedly fired for an oversized payload (vmID=%s, %d bytes) — the size cap did not reject it", vmID, len(data))
		},
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter failed: %v", err)
	}
	t.Cleanup(func() { recvAdapter.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go recvAdapter.ListenForMigrations(ctx)
	time.Sleep(100 * time.Millisecond)

	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 2*time.Second)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	defer conn.Close()

	// Envelope: protocol=0 (standard memory), vmID="x".
	vmID := "x"
	envelope := []byte{0}
	envelope = binary.BigEndian.AppendUint16(envelope, uint16(len(vmID)))
	envelope = append(envelope, vmID...)
	if _, err := conn.Write(envelope); err != nil {
		t.Fatalf("failed to write envelope: %v", err)
	}

	// A deliberately huge size header (1TB) — must be rejected by the
	// receiver's maxSize check, not attempted as an allocation.
	sizeHeader := make([]byte, 8)
	binary.BigEndian.PutUint64(sizeHeader, 1<<40)
	if _, err := conn.Write(sizeHeader); err != nil {
		t.Fatalf("failed to write size header: %v", err)
	}

	// The receiver should close the connection after rejecting the
	// oversized header (handleIncomingMigration's defer conn.Close()) —
	// observe that as a clean read failure (EOF/reset) rather than the
	// connection hanging or the process attempting the allocation.
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	buf := make([]byte, 1)
	_, readErr := conn.Read(buf)
	if readErr == nil {
		t.Fatal("expected connection to be closed by the receiver after rejecting an oversized size header, but a read succeeded")
	}

	// No further assertion needed beyond the read failure above: the
	// OnMemoryReceived callback (which would t.Errorf if invoked) proves
	// on its own that no oversized allocation/delivery happened.
}
