package dwcp

import (
	"bytes"
	"crypto/rand"
	"testing"
)

// TestCompressionNone_UncompressedAndDistinct proves CompressionLevelNone is a
// genuine "skip compression entirely" tier whose output is distinct from
// CompressionLocal's compressed output (novacron-94l).
//
// Before the fix, CompressionLevelNone (config.go, value 0) aliased exactly to
// CompressionLocal (hde.go iota, also value 0), so passing "None" to
// CompressMemory looked up the zstd LocalLevel encoder and compressed anyway —
// there was no functional skip path anywhere in the pipeline. The fix re-bases
// the HDE tier iota (Local/Regional/Global => 1/2/3) so 0 is a real,
// distinct skip value that CompressMemory/CompressDisk honor by emitting the
// payload verbatim.
func TestCompressionNone_UncompressedAndDistinct(t *testing.T) {
	hde, err := NewHDE(HDEConfig{LocalLevel: 0, EnableDelta: false})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	// All-zero payload: CompressionLocal collapses it to near-nothing, so a
	// genuine skip (full size) is unmistakably distinct.
	data := make([]byte, 64*1024)

	noneOut, err := hde.CompressMemory("vm-none", data, CompressionLevelNone)
	if err != nil {
		t.Fatalf("CompressMemory(None) failed: %v", err)
	}
	localOut, err := hde.CompressMemory("vm-local", data, CompressionLocal)
	if err != nil {
		t.Fatalf("CompressMemory(Local) failed: %v", err)
	}

	// The None packet payload (after the 16-byte HDE header) must be the
	// verbatim original bytes — genuinely uncompressed.
	if len(noneOut) < 16 {
		t.Fatalf("None packet too short: %d bytes", len(noneOut))
	}
	if !bytes.Equal(noneOut[16:], data) {
		t.Errorf("CompressionLevelNone payload is not the verbatim original (payload=%d bytes, data=%d bytes): compression was applied when it should have been skipped", len(noneOut)-16, len(data))
	}

	// Local must actually compress (the whole point of a real tier): for
	// all-zero data it collapses far below the None (uncompressed) size.
	if len(localOut) >= len(noneOut) {
		t.Errorf("CompressionLocal output (%d bytes) is not smaller than CompressionLevelNone output (%d bytes) — None is not distinct from Local", len(localOut), len(noneOut))
	}

	// A None packet must round-trip back to the original.
	back, err := hde.Decompress(noneOut)
	if err != nil {
		t.Fatalf("Decompress(None packet) failed: %v", err)
	}
	if !bytes.Equal(back, data) {
		t.Errorf("Decompress(None) round trip mismatch: got %d bytes, want %d", len(back), len(data))
	}

	// A Local packet must still round-trip too (no regression).
	backLocal, err := hde.Decompress(localOut)
	if err != nil {
		t.Fatalf("Decompress(Local packet) failed: %v", err)
	}
	if !bytes.Equal(backLocal, data) {
		t.Errorf("Decompress(Local) round trip mismatch: got %d bytes, want %d", len(backLocal), len(data))
	}
}

// TestSelectCompressionTier_SelectsNoneOnFastLink proves the MigrationAdapter
// can actually SELECT the genuine skip path for a fast/LAN link (novacron-94l):
// on a sub-threshold-latency link, compression is a measured net loss, so
// selectCompressionTier must pick a tier that yields uncompressed output.
//
// End-to-end discrimination: whatever tier is selected, feeding it to
// CompressMemory must produce verbatim (uncompressed) output. Pre-fix, a fast
// link selects CompressionLocal (== value 0 == "None"), which zstd-wraps even
// incompressible data — so the payload never equals the raw input until None
// is a real, selectable skip tier.
func TestSelectCompressionTier_SelectsNoneOnFastLink(t *testing.T) {
	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{EnableDWCP: true})
	if err != nil {
		t.Fatalf("NewMigrationAdapter failed: %v", err)
	}
	defer adapter.Close()

	amst, err := NewAMST(AMSTConfig{MinStreams: 1, MaxStreams: 1, InitialStreams: 1})
	if err != nil {
		t.Fatalf("NewAMST failed: %v", err)
	}
	defer amst.Close()
	amst.UpdateMetrics(0, 0, 0) // 0ms latency = fastest possible link

	conn := &MigrationConnection{AMST: amst}
	tier := adapter.selectCompressionTier(conn)

	// Incompressible (random) data: a zstd frame of it is never byte-equal to
	// the raw bytes, so bytes.Equal below flips false the moment any real
	// compression tier is chosen.
	data := make([]byte, 64*1024)
	if _, err := rand.Read(data); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	out, err := adapter.hde.CompressMemory("vm-fastlink", data, tier)
	if err != nil {
		t.Fatalf("CompressMemory(selected tier) failed: %v", err)
	}
	if len(out) < 16 || !bytes.Equal(out[16:], data) {
		t.Errorf("fast-link tier %v did not skip compression: payload=%d bytes, want verbatim %d bytes (selectCompressionTier cannot select a genuine no-compression path)", tier, len(out)-16, len(data))
	}
}
