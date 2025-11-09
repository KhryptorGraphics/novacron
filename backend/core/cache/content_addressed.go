package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

// ContentAddressedStorageImpl implements content-addressed storage with deduplication
type ContentAddressedStorageImpl struct {
	config *CacheConfig

	// Chunk storage
	chunks    map[string]*Chunk
	chunksMu  sync.RWMutex

	// Statistics
	totalChunks  int64
	uniqueChunks int64
	savedBytes   int64
}

// Chunk represents a deduplicated chunk
type Chunk struct {
	Hash     []byte
	Data     []byte
	Size     int64
	RefCount int
}

// NewContentAddressedStorage creates a new content-addressed storage
func NewContentAddressedStorage(config *CacheConfig) *ContentAddressedStorageImpl {
	return &ContentAddressedStorageImpl{
		config: config,
		chunks: make(map[string]*Chunk),
	}
}

// StoreChunk stores a chunk and returns its hash
func (cas *ContentAddressedStorageImpl) StoreChunk(data []byte) ([]byte, error) {
	// Compute hash
	hash := sha256.Sum256(data)
	hashStr := hex.EncodeToString(hash[:])

	cas.chunksMu.Lock()
	defer cas.chunksMu.Unlock()

	// Check if chunk already exists
	if chunk, ok := cas.chunks[hashStr]; ok {
		// Already exists, increment ref count
		cas.totalChunks++
		cas.savedBytes += int64(len(data))
		return chunk.Hash, nil
	}

	// Store new chunk
	chunk := &Chunk{
		Hash:     hash[:],
		Data:     append([]byte{}, data...),
		Size:     int64(len(data)),
		RefCount: 0,
	}

	cas.chunks[hashStr] = chunk
	cas.totalChunks++
	cas.uniqueChunks++

	return chunk.Hash, nil
}

// GetChunk retrieves a chunk by hash
func (cas *ContentAddressedStorageImpl) GetChunk(hash []byte) ([]byte, error) {
	hashStr := hex.EncodeToString(hash)

	cas.chunksMu.RLock()
	defer cas.chunksMu.RUnlock()

	chunk, ok := cas.chunks[hashStr]
	if !ok {
		return nil, ErrNotFound
	}

	return append([]byte{}, chunk.Data...), nil
}

// AddRef increments the reference count
func (cas *ContentAddressedStorageImpl) AddRef(hash []byte) error {
	hashStr := hex.EncodeToString(hash)

	cas.chunksMu.Lock()
	defer cas.chunksMu.Unlock()

	chunk, ok := cas.chunks[hashStr]
	if !ok {
		return ErrNotFound
	}

	chunk.RefCount++
	return nil
}

// ReleaseRef decrements the reference count
func (cas *ContentAddressedStorageImpl) ReleaseRef(hash []byte) error {
	hashStr := hex.EncodeToString(hash)

	cas.chunksMu.Lock()
	defer cas.chunksMu.Unlock()

	chunk, ok := cas.chunks[hashStr]
	if !ok {
		return ErrNotFound
	}

	if chunk.RefCount > 0 {
		chunk.RefCount--
	}

	return nil
}

// GC performs garbage collection of unreferenced chunks
func (cas *ContentAddressedStorageImpl) GC() (int64, error) {
	cas.chunksMu.Lock()
	defer cas.chunksMu.Unlock()

	var freedBytes int64

	for hashStr, chunk := range cas.chunks {
		if chunk.RefCount == 0 {
			freedBytes += chunk.Size
			delete(cas.chunks, hashStr)
			cas.uniqueChunks--
		}
	}

	return freedBytes, nil
}

// DeduplicationRatio returns the deduplication ratio
func (cas *ContentAddressedStorageImpl) DeduplicationRatio() float64 {
	cas.chunksMu.RLock()
	defer cas.chunksMu.RUnlock()

	if cas.uniqueChunks == 0 {
		return 1.0
	}

	return float64(cas.totalChunks) / float64(cas.uniqueChunks)
}

// TotalChunks returns the total number of chunks stored
func (cas *ContentAddressedStorageImpl) TotalChunks() int64 {
	cas.chunksMu.RLock()
	defer cas.chunksMu.RUnlock()

	return cas.totalChunks
}

// UniqueChunks returns the number of unique chunks
func (cas *ContentAddressedStorageImpl) UniqueChunks() int64 {
	cas.chunksMu.RLock()
	defer cas.chunksMu.RUnlock()

	return cas.uniqueChunks
}

// ChunkData splits data into chunks
func (cas *ContentAddressedStorageImpl) ChunkData(data []byte) ([][]byte, error) {
	chunkSize := cas.config.ChunkSize
	if chunkSize <= 0 {
		chunkSize = 64 * 1024 // 64KB default
	}

	chunks := make([][]byte, 0)

	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}

		chunk := data[i:end]
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// ReconstructData reconstructs data from chunks
func (cas *ContentAddressedStorageImpl) ReconstructData(hashes [][]byte) ([]byte, error) {
	result := make([]byte, 0)

	for _, hash := range hashes {
		chunk, err := cas.GetChunk(hash)
		if err != nil {
			return nil, err
		}
		result = append(result, chunk...)
	}

	return result, nil
}
