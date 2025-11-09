package cache

import (
	"testing"
)

func TestContentAddressedStorage_StoreChunk(t *testing.T) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	data := []byte("test data")

	hash, err := cas.StoreChunk(data)
	if err != nil {
		t.Fatalf("StoreChunk failed: %v", err)
	}

	if len(hash) == 0 {
		t.Errorf("Expected non-empty hash")
	}

	// Store same data again
	hash2, err := cas.StoreChunk(data)
	if err != nil {
		t.Fatalf("StoreChunk failed: %v", err)
	}

	// Should get same hash
	if string(hash) != string(hash2) {
		t.Errorf("Same data should produce same hash")
	}

	// Should have deduplication
	if cas.TotalChunks() != 2 {
		t.Errorf("Expected 2 total chunks, got %d", cas.TotalChunks())
	}

	if cas.UniqueChunks() != 1 {
		t.Errorf("Expected 1 unique chunk, got %d", cas.UniqueChunks())
	}
}

func TestContentAddressedStorage_GetChunk(t *testing.T) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	data := []byte("test data")

	hash, err := cas.StoreChunk(data)
	if err != nil {
		t.Fatalf("StoreChunk failed: %v", err)
	}

	retrieved, err := cas.GetChunk(hash)
	if err != nil {
		t.Fatalf("GetChunk failed: %v", err)
	}

	if string(retrieved) != string(data) {
		t.Errorf("Retrieved data doesn't match original")
	}
}

func TestContentAddressedStorage_RefCounting(t *testing.T) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	data := []byte("test data")

	hash, _ := cas.StoreChunk(data)

	// Add references
	cas.AddRef(hash)
	cas.AddRef(hash)

	// Release one reference
	cas.ReleaseRef(hash)

	// GC should not remove chunk (still has references)
	freed, _ := cas.GC()
	if freed > 0 {
		t.Errorf("GC should not free referenced chunks")
	}

	// Release remaining references
	cas.ReleaseRef(hash)

	// GC should remove chunk
	freed, _ = cas.GC()
	if freed == 0 {
		t.Errorf("GC should free unreferenced chunks")
	}
}

func TestContentAddressedStorage_DeduplicationRatio(t *testing.T) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	// Store same data multiple times
	data := []byte("duplicate data")

	for i := 0; i < 10; i++ {
		cas.StoreChunk(data)
	}

	ratio := cas.DeduplicationRatio()
	if ratio < 2.0 {
		t.Errorf("Expected deduplication ratio >= 2.0, got %f", ratio)
	}
}

func TestContentAddressedStorage_ChunkData(t *testing.T) {
	config := DefaultConfig()
	config.ChunkSize = 10 // 10 bytes per chunk
	cas := NewContentAddressedStorage(config)

	data := []byte("0123456789ABCDEFGHIJ") // 20 bytes

	chunks, err := cas.ChunkData(data)
	if err != nil {
		t.Fatalf("ChunkData failed: %v", err)
	}

	if len(chunks) != 2 {
		t.Errorf("Expected 2 chunks, got %d", len(chunks))
	}

	if len(chunks[0]) != 10 {
		t.Errorf("First chunk should be 10 bytes, got %d", len(chunks[0]))
	}

	if len(chunks[1]) != 10 {
		t.Errorf("Second chunk should be 10 bytes, got %d", len(chunks[1]))
	}
}

func TestContentAddressedStorage_ReconstructData(t *testing.T) {
	config := DefaultConfig()
	config.ChunkSize = 10
	cas := NewContentAddressedStorage(config)

	originalData := []byte("0123456789ABCDEFGHIJ")

	// Chunk and store
	chunks, _ := cas.ChunkData(originalData)
	hashes := make([][]byte, len(chunks))

	for i, chunk := range chunks {
		hash, _ := cas.StoreChunk(chunk)
		hashes[i] = hash
	}

	// Reconstruct
	reconstructed, err := cas.ReconstructData(hashes)
	if err != nil {
		t.Fatalf("ReconstructData failed: %v", err)
	}

	if string(reconstructed) != string(originalData) {
		t.Errorf("Reconstructed data doesn't match original")
	}
}

func BenchmarkContentAddressedStorage_Store(b *testing.B) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	data := make([]byte, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cas.StoreChunk(data)
	}
}

func BenchmarkContentAddressedStorage_Get(b *testing.B) {
	config := DefaultConfig()
	cas := NewContentAddressedStorage(config)

	data := make([]byte, 1024)
	hash, _ := cas.StoreChunk(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cas.GetChunk(hash)
	}
}
