package dwcp

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

// dictSamples builds a mixed zero/pattern/random sample set that is known not
// to trigger the zstd.BuildDict panic (see dictionary_fix_test.go) so these
// tests exercise the dictionary success path cleanly.
func dictSamples() [][]byte {
	samples := make([][]byte, 100)
	for i := range samples {
		s := make([]byte, 4096)
		switch i % 3 {
		case 0:
			// zero page: leave as-is
		case 1:
			for j := range s {
				s[j] = byte(j % 256)
			}
		case 2:
			rand.Read(s)
		}
		samples[i] = s
	}
	return samples
}

// packetDictID reads the dictionary ID createPacket writes at bytes [4:8] of
// the HDE packet header. Zero means "no dictionary was used".
func packetDictID(packet []byte) uint32 {
	if len(packet) < 16 {
		return 0
	}
	return binary.BigEndian.Uint32(packet[4:8])
}

// TestDictionaryRoundTrip_DiskCompressDecompress proves CompressDisk with
// EnableDictionary now produces a packet HDE.Decompress can correctly reverse
// back to the ORIGINAL bytes (novacron-976, parts 2 & 3).
//
// Before the fix this was impossible: CompressDisk ran compressWithDict and
// THEN plain hde.compress on top of the result (double compression), while
// Decompress did exactly one plain decompress pass with no dictionary — so it
// could only peel the outer plain layer and returned the inner, still
// dictionary-compressed frame as "decompressed" garbage. The fix stops the
// double compression (single dictionary pass, dict ID recorded in the packet)
// and builds a dictionary-aware decoder in TrainDictionary that Decompress
// selects by that ID.
func TestDictionaryRoundTrip_DiskCompressDecompress(t *testing.T) {
	hde, err := NewHDE(HDEConfig{
		GlobalLevel:      9,
		EnableDictionary: true,
		EnableDelta:      false, // isolate the dictionary path from delta encoding
		DictSize:         1024,
		TrainingSamples:  100,
	})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	const vmID = "vm-dict-disk"
	// Key MUST match the vmID+"_disk" lookup CompressDisk performs.
	if err := hde.TrainDictionary(vmID+"_disk", dictSamples()); err != nil {
		t.Fatalf("TrainDictionary failed: %v", err)
	}

	// Disk block overlapping the dictionary's training pattern so the
	// dictionary is genuinely engaged.
	original := make([]byte, 8192)
	if _, err := rand.Read(original); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	copy(original, byte0toN(4096))

	packet, err := hde.CompressDisk(vmID, original, 0, CompressionRegional)
	if err != nil {
		t.Fatalf("CompressDisk failed: %v", err)
	}
	if packetDictID(packet) == 0 {
		t.Error("CompressDisk did not record a dictionary ID in the packet (dictionary not engaged / not signalled to Decompress)")
	}

	back, err := hde.Decompress(packet)
	if err != nil {
		t.Fatalf("Decompress failed: %v", err)
	}
	if !bytes.Equal(back, original) {
		t.Fatalf("dictionary disk round trip corrupted data: got %d bytes, want %d (double-compression without a matching decode, or decoder lacks the dictionary)", len(back), len(original))
	}
}

// TestDictionaryRoundTrip_MemoryCompressDecompress proves CompressMemory is now
// dictionary-aware and round-trips through Decompress (novacron-976, part 1).
//
// Before the fix, CompressMemory never called compressWithDict regardless of
// EnableDictionary — it silently plain-compressed, so a naive round trip would
// pass while the dictionary was never used. The discriminating assertion is
// that the packet now carries a non-zero dictionary ID (the dictionary was
// actually engaged); the round trip then proves that engagement is reversible.
func TestDictionaryRoundTrip_MemoryCompressDecompress(t *testing.T) {
	hde, err := NewHDE(HDEConfig{
		GlobalLevel:      9,
		EnableDictionary: true,
		EnableDelta:      false,
		DictSize:         1024,
		TrainingSamples:  100,
	})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	const vmID = "vm-dict-mem"
	// Key MUST match the vmID+"_memory" lookup CompressMemory performs.
	if err := hde.TrainDictionary(vmID+"_memory", dictSamples()); err != nil {
		t.Fatalf("TrainDictionary failed: %v", err)
	}

	original := make([]byte, 8192)
	if _, err := rand.Read(original); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	copy(original, byte0toN(4096))

	packet, err := hde.CompressMemory(vmID, original, CompressionRegional)
	if err != nil {
		t.Fatalf("CompressMemory failed: %v", err)
	}
	if packetDictID(packet) == 0 {
		t.Error("CompressMemory did not engage the dictionary (packet dict ID = 0): not dictionary-aware")
	}

	back, err := hde.Decompress(packet)
	if err != nil {
		t.Fatalf("Decompress failed: %v", err)
	}
	if !bytes.Equal(back, original) {
		t.Fatalf("dictionary memory round trip corrupted data: got %d bytes, want %d (CompressMemory not dictionary-aware, or decoder lacks the dictionary)", len(back), len(original))
	}
}

// byte0toN returns a deterministic byte ramp matching dictSamples' case-1
// pattern, so overlaid payload bytes reference the trained dictionary.
func byte0toN(n int) []byte {
	b := make([]byte, n)
	for j := range b {
		b[j] = byte(j % 256)
	}
	return b
}
