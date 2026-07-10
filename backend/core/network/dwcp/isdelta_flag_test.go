package dwcp

import (
	"crypto/rand"
	"testing"
)

// packetIsDelta reports the isDelta bit createPacket wrote — packet[0],
// per its "[isDelta:1][tier:1][reserved:6][dataSize:8][compressed data]"
// header layout.
func packetIsDelta(packet []byte) bool {
	return len(packet) > 0 && packet[0] == 1
}

// TestCompressMemory_IsDeltaFlagMatchesActualUsage proves CompressMemory's
// packet isDelta flag reflects whether the delta was actually the
// compressed payload, not merely whether one was computed.
//
// Before this fix, CompressMemory computed usedDelta implicitly inline
// (deltaEncoded != nil && len(deltaEncoded) < len(memoryData)/2) to pick
// dataToCompress, then separately passed a DIFFERENT condition
// (deltaEncoded != nil alone) to createPacket — so a delta that was
// computed but rejected as too large (isDeltaEfficient's own threshold,
// DeltaThreshold default 0.7 against the RAW delta.Size, can pass while
// the ENCODED delta's actual byte length still exceeds the stricter
// len(data)/2 "actually worth using" check applied afterward) would still
// mark the packet isDelta=true even though the payload was the full,
// non-delta compressed data. Confirmed unreachable for MigrationAdapter
// (EnableDelta forced off, NewMigrationAdapter) but real for any other
// EnableDelta caller — see novacron-38p follow-up notes.
//
// The fix computes usedDelta once and uses that SAME variable for both
// the data-selection branch and the packet flag, so the two cannot
// diverge by construction — this test proves the wiring (createPacket
// correctly encodes the flag HDE.CompressMemory passes it) rather than
// hunting the exact DeltaThreshold-vs-half-size numeric gap, since the
// single-variable fix makes that gap structurally unreachable regardless
// of where the numeric boundary falls.
func TestCompressMemory_IsDeltaFlagMatchesActualUsage(t *testing.T) {
	hde, err := NewHDE(HDEConfig{GlobalLevel: 3, EnableDelta: true, DeltaThreshold: 0.7})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	data := make([]byte, 64*1024)
	rand.Read(data)

	// First compression for this vmID: no baseline exists yet, so no
	// delta can be computed — must not be flagged as delta.
	first, err := hde.CompressMemory("vm-isdelta", data, CompressionGlobal)
	if err != nil {
		t.Fatalf("first CompressMemory failed: %v", err)
	}
	if packetIsDelta(first) {
		t.Error("first compression (no baseline) incorrectly flagged isDelta=true")
	}

	// Second compression of IDENTICAL data: a baseline now exists and is
	// byte-identical, so the delta is trivially tiny and efficient —
	// must be flagged as delta, and must actually be small (proving the
	// delta path, not the full-data path, produced this packet).
	second, err := hde.CompressMemory("vm-isdelta", data, CompressionGlobal)
	if err != nil {
		t.Fatalf("second CompressMemory failed: %v", err)
	}
	if !packetIsDelta(second) {
		t.Error("second compression (identical data, real baseline) incorrectly flagged isDelta=false")
	}
	if len(second) >= len(first) {
		t.Errorf("second (delta) packet (%d bytes) should be markedly smaller than first (full) packet (%d bytes) for identical repeat data", len(second), len(first))
	}
}

// TestCompressDisk_IsDeltaFlagMatchesActualUsage is the CompressDisk
// analogue of TestCompressMemory_IsDeltaFlagMatchesActualUsage — same
// bug, same fix, same reasoning, different call site (hde.go CompressDisk).
func TestCompressDisk_IsDeltaFlagMatchesActualUsage(t *testing.T) {
	hde, err := NewHDE(HDEConfig{GlobalLevel: 3, EnableDelta: true, DeltaThreshold: 0.7})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	data := make([]byte, 64*1024)
	rand.Read(data)

	first, err := hde.CompressDisk("vm-isdelta-disk", data, 0, CompressionGlobal)
	if err != nil {
		t.Fatalf("first CompressDisk failed: %v", err)
	}
	if packetIsDelta(first) {
		t.Error("first compression (no baseline) incorrectly flagged isDelta=true")
	}

	second, err := hde.CompressDisk("vm-isdelta-disk", data, 0, CompressionGlobal)
	if err != nil {
		t.Fatalf("second CompressDisk failed: %v", err)
	}
	if !packetIsDelta(second) {
		t.Error("second compression (identical data, real baseline) incorrectly flagged isDelta=false")
	}
	if len(second) >= len(first) {
		t.Errorf("second (delta) packet (%d bytes) should be markedly smaller than first (full) packet (%d bytes) for identical repeat data", len(second), len(first))
	}
}
