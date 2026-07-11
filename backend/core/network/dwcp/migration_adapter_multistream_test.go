package dwcp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// newLoopbackMigrationAdapterWithAMST is a variant of
// newLoopbackMigrationAdapter (migration_adapter_roundtrip_test.go) that
// lets a test configure AMSTConfig directly, to drive MigrationAdapter's
// real multi-stream AMST session correlation (novacron-hpa) instead of the
// single-stream default every other round-trip test exercises. Kept as a
// separate helper (rather than adding a parameter to the existing one) so
// this file doesn't touch already-passing tests' shared setup.
func newLoopbackMigrationAdapterWithAMST(t testing.TB, amstConfig dwcp.AMSTConfig) (adapter *dwcp.MigrationAdapter, memCh chan receivedMemory, diskCh chan receivedDiskBlock) {
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
		EnableDWCP:        true,
		EnableFallback:    false,
		ListenPort:        port,
		ConnectionTimeout: 5 * time.Second,
		AMSTConfig:        amstConfig,
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
	go func() { adapter.ListenForMigrations(ctx) }()

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

// TestMigrationRoundTrip_DWCPMemoryMultiStream proves real N>1 AMST stream
// session correlation (novacron-hpa): the sender dials 4 concurrent AMST
// streams (MinStreams=MaxStreams=InitialStreams=4), the listener accepts 4
// independent connections that must be regrouped via the sessionID written
// on each (writeDWCPStreamEnvelope/handleIncomingDWCPStream), and the
// receiver reassembles the payload correctly regardless of which stream
// carried which chunk. Payload is large enough (2MB, default 64KB chunks =
// 32 chunks) that chunks are genuinely spread across multiple streams, not
// just one — TestMigrationRoundTrip_DWCPMemory (migration_adapter_roundtrip_test.go)
// already covers the N=1 case this generalizes.
func TestMigrationRoundTrip_DWCPMemoryMultiStream(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapterWithAMST(t, dwcp.AMSTConfig{
		MinStreams:     4,
		MaxStreams:     4,
		InitialStreams: 4,
	})

	original := generateMixedVMData(2 * 1024 * 1024)

	if err := adapter.MigrateVMMemory(context.Background(), "vm-multistream-mem", original, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMMemory failed: %v", err)
	}

	got := waitForMemory(t, memCh, 10*time.Second)
	if got.vmID != "vm-multistream-mem" {
		t.Errorf("vmID mismatch: got %q, want %q", got.vmID, "vm-multistream-mem")
	}
	if !bytes.Equal(got.data, original) {
		t.Fatalf("multi-stream DWCP memory round trip corrupted data: got %d bytes, want %d bytes matching original", len(got.data), len(original))
	}
}

// TestMigrationRoundTrip_DWCPMoreStreamsThanChunks is the edge case the
// originating ticket (novacron-hpa, discovered from novacron-lce) flagged
// as never tested: N (8 streams) > actual chunk count (1, forced by a
// payload smaller than the 64KB default chunk size) — 7 of the 8 streams
// receive zero chunks. Must degrade gracefully: every one of the 8
// connections still gets correlated into the session (dialing and writing
// the per-stream envelope happens at Connect time, before Transfer even
// computes how many chunks there are), and AMST.Receive's zero-chunk
// streams see a clean io.EOF once the sender closes every stream after a
// successful Transfer, rather than hanging or erroring. waitForMemory's
// timeout converts a regression (hang) into a clean test failure instead
// of blocking the test run forever.
func TestMigrationRoundTrip_DWCPMoreStreamsThanChunks(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapterWithAMST(t, dwcp.AMSTConfig{
		MinStreams:     8,
		MaxStreams:     8,
		InitialStreams: 8,
	})

	// Well under the default 64KB chunk size, so AMST.Transfer computes
	// exactly 1 chunk for the (already tiny, and CompressionLevelNone on
	// loopback per selectCompressionTier) compressed payload.
	original := make([]byte, 4*1024)
	if _, err := rand.Read(original); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}

	if err := adapter.MigrateVMMemory(context.Background(), "vm-more-streams-than-chunks", original, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMMemory failed: %v", err)
	}

	got := waitForMemory(t, memCh, 10*time.Second)
	if got.vmID != "vm-more-streams-than-chunks" {
		t.Errorf("vmID mismatch: got %q, want %q", got.vmID, "vm-more-streams-than-chunks")
	}
	if !bytes.Equal(got.data, original) {
		t.Fatalf("N>chunk-count round trip corrupted data: got %d bytes, want %d bytes matching original", len(got.data), len(original))
	}
}

// TestMigrationRoundTrip_DWCPDiskMultiStream proves the disk path also
// round-trips correctly through the same multi-stream session correlation
// machinery — completeDWCPSession only dispatches memory vs disk after all
// streams have already been correlated, wrapped in one AMST, and received.
func TestMigrationRoundTrip_DWCPDiskMultiStream(t *testing.T) {
	adapter, _, diskCh := newLoopbackMigrationAdapterWithAMST(t, dwcp.AMSTConfig{
		MinStreams:     4,
		MaxStreams:     4,
		InitialStreams: 4,
	})

	blocks := make(map[int][]byte)
	for i := 0; i < 8; i++ {
		blocks[i] = generateMixedVMData(64 * 1024)
	}

	if err := adapter.MigrateVMDisk(context.Background(), "vm-multistream-disk", blocks, "127.0.0.1", nil); err != nil {
		t.Fatalf("MigrateVMDisk failed: %v", err)
	}

	got := waitForDiskBlocks(t, diskCh, len(blocks), 10*time.Second)
	for id, want := range blocks {
		block, ok := got[id]
		if !ok {
			t.Fatalf("block %d never received", id)
		}
		if !bytes.Equal(block.data, want) {
			t.Fatalf("multi-stream DWCP disk block %d round trip corrupted data: got %d bytes, want %d bytes matching original", id, len(block.data), len(want))
		}
	}
}

// TestMigrationRoundTrip_DWCPMultiStreamConcurrent proves multiple
// concurrent multi-stream migrations (4 streams each) on one adapter do
// not cross-contaminate — each vmID's data must arrive intact, exercising
// registerDWCPStream's sessionID-keyed map under concurrent sessions
// forming at once (mirrors TestMigrationRoundTrip_Concurrent's
// single-stream coverage in migration_adapter_roundtrip_test.go).
func TestMigrationRoundTrip_DWCPMultiStreamConcurrent(t *testing.T) {
	adapter, memCh, _ := newLoopbackMigrationAdapterWithAMST(t, dwcp.AMSTConfig{
		MinStreams:     4,
		MaxStreams:     4,
		InitialStreams: 4,
	})

	const n = 4
	want := make(map[string][]byte, n)
	errCh := make(chan error, n)
	for i := 0; i < n; i++ {
		vmID := fmt.Sprintf("vm-multistream-concurrent-%d", i)
		data := generateMixedVMData(256 * 1024)
		want[vmID] = data

		go func(vmID string, data []byte) {
			if err := adapter.MigrateVMMemory(context.Background(), vmID, data, "127.0.0.1", nil); err != nil {
				errCh <- fmt.Errorf("%s: %w", vmID, err)
				return
			}
			errCh <- nil
		}(vmID, data)
	}
	for i := 0; i < n; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("concurrent multi-stream migration failed: %v", err)
		}
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
			t.Fatalf("%s: round trip corrupted data (cross-contamination between concurrent multi-stream sessions?)", vmID)
		}
	}
}
