package backup

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/chmduquesne/rollinghash/rabinkarp64"
)

const (
	// Default chunk sizes for Rabin fingerprinting
	DefaultMinChunkSize = 1024      // 1KB
	DefaultAvgChunkSize = 8192      // 8KB  
	DefaultMaxChunkSize = 65536     // 64KB
	
	// Deduplication index file
	DedupIndexFile = "dedup_index.json"
	
	// Storage directories
	ChunksDir = "chunks"
	IndexDir  = "index"
)

// DeduplicationEngine implements content-aware chunking and deduplication
type DeduplicationEngine struct {
	basePath     string
	chunkStore   *ChunkStore
	index        *DedupIndex
	rabinPoly    uint64
	minChunkSize int
	avgChunkSize int
	maxChunkSize int
	mutex        sync.RWMutex
}

// ChunkStore manages storage of deduplicated chunks
type ChunkStore struct {
	basePath string
	chunks   map[string]*ChunkInfo
	mutex    sync.RWMutex
}

// ChunkInfo contains metadata about a stored chunk
type ChunkInfo struct {
	Hash      string    `json:"hash"`
	Size      int64     `json:"size"`
	RefCount  int64     `json:"ref_count"`
	StorePath string    `json:"store_path"`
	CreatedAt time.Time `json:"created_at"`
	LastUsed  time.Time `json:"last_used"`
}

// DedupIndex maintains the deduplication index
type DedupIndex struct {
	chunks   map[string]*ChunkInfo
	stats    *DedupStats
	mutex    sync.RWMutex
	filePath string
}

// DedupStats tracks deduplication statistics
type DedupStats struct {
	TotalBytes        int64   `json:"total_bytes"`
	UniqueBytes       int64   `json:"unique_bytes"`
	DeduplicatedBytes int64   `json:"deduplicated_bytes"`
	CompressionRatio  float64 `json:"compression_ratio"`
	ChunkCount        int64   `json:"chunk_count"`
	UniqueChunks      int64   `json:"unique_chunks"`
	LastUpdated       time.Time `json:"last_updated"`
}

// ChunkReference represents a reference to a deduplicated chunk
type ChunkReference struct {
	Hash   string `json:"hash"`
	Offset int64  `json:"offset"`
	Size   int64  `json:"size"`
}

// DedupResult contains the result of deduplication operation
type DedupResult struct {
	OriginalSize int64            `json:"original_size"`
	StoredSize   int64            `json:"stored_size"`
	ChunkRefs    []ChunkReference `json:"chunk_refs"`
	Ratio        float64          `json:"ratio"`
}

// NewDeduplicationEngine creates a new deduplication engine
func NewDeduplicationEngine(basePath string) (*DeduplicationEngine, error) {
	// Ensure directories exist
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create deduplication base path: %w", err)
	}
	
	chunksPath := filepath.Join(basePath, ChunksDir)
	if err := os.MkdirAll(chunksPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create chunks directory: %w", err)
	}
	
	indexPath := filepath.Join(basePath, IndexDir)
	if err := os.MkdirAll(indexPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create index directory: %w", err)
	}
	
	// Initialize chunk store
	chunkStore := &ChunkStore{
		basePath: chunksPath,
		chunks:   make(map[string]*ChunkInfo),
	}
	
	// Initialize deduplication index
	indexFilePath := filepath.Join(indexPath, DedupIndexFile)
	dedupIndex := &DedupIndex{
		chunks:   make(map[string]*ChunkInfo),
		stats:    &DedupStats{LastUpdated: time.Now()},
		filePath: indexFilePath,
	}
	
	// Load existing index
	if err := dedupIndex.load(); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to load deduplication index: %w", err)
	}
	
	engine := &DeduplicationEngine{
		basePath:     basePath,
		chunkStore:   chunkStore,
		index:        dedupIndex,
		rabinPoly:    0x3DA3358B4DC173, // Rabin polynomial for chunking
		minChunkSize: DefaultMinChunkSize,
		avgChunkSize: DefaultAvgChunkSize,
		maxChunkSize: DefaultMaxChunkSize,
	}
	
	// Load existing chunks into memory
	if err := engine.loadExistingChunks(); err != nil {
		return nil, fmt.Errorf("failed to load existing chunks: %w", err)
	}
	
	return engine, nil
}

// ProcessBlock processes a block of data for deduplication
func (e *DeduplicationEngine) ProcessBlock(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}
	
	// Content-aware chunking using Rabin fingerprinting
	chunks := e.chunkData(data)
	
	// Process each chunk for deduplication
	var result DedupResult
	result.OriginalSize = int64(len(data))
	result.ChunkRefs = make([]ChunkReference, 0, len(chunks))
	
	var offset int64
	for _, chunk := range chunks {
		hash := calculateChunkHash(chunk)
		
		// Check if chunk already exists
		e.index.mutex.RLock()
		chunkInfo, exists := e.index.chunks[hash]
		e.index.mutex.RUnlock()
		
		if exists {
			// Chunk exists, just increment reference count
			e.index.mutex.Lock()
			chunkInfo.RefCount++
			chunkInfo.LastUsed = time.Now()
			e.index.mutex.Unlock()
		} else {
			// Store new chunk
			chunkInfo = &ChunkInfo{
				Hash:      hash,
				Size:      int64(len(chunk)),
				RefCount:  1,
				CreatedAt: time.Now(),
				LastUsed:  time.Now(),
			}
			
			if err := e.storeChunk(hash, chunk, chunkInfo); err != nil {
				return nil, fmt.Errorf("failed to store chunk %s: %w", hash, err)
			}
			
			// Add to index
			e.index.mutex.Lock()
			e.index.chunks[hash] = chunkInfo
			e.index.stats.UniqueChunks++
			e.index.stats.UniqueBytes += int64(len(chunk))
			e.index.mutex.Unlock()
		}
		
		// Add chunk reference
		result.ChunkRefs = append(result.ChunkRefs, ChunkReference{
			Hash:   hash,
			Offset: offset,
			Size:   int64(len(chunk)),
		})
		
		offset += int64(len(chunk))
	}
	
	// Update statistics
	e.index.mutex.Lock()
	e.index.stats.TotalBytes += result.OriginalSize
	e.index.stats.ChunkCount += int64(len(chunks))
	e.index.stats.DeduplicatedBytes = e.index.stats.TotalBytes - e.index.stats.UniqueBytes
	if e.index.stats.TotalBytes > 0 {
		e.index.stats.CompressionRatio = float64(e.index.stats.UniqueBytes) / float64(e.index.stats.TotalBytes)
	}
	e.index.stats.LastUpdated = time.Now()
	e.index.mutex.Unlock()
	
	// Save index periodically (every 1000 operations)
	if e.index.stats.ChunkCount%1000 == 0 {
		e.index.save()
	}
	
	// Return deduplicated representation (chunk references as bytes)
	return e.serializeChunkRefs(result.ChunkRefs), nil
}

// chunkData performs content-aware chunking using Rabin fingerprinting
func (e *DeduplicationEngine) chunkData(data []byte) [][]byte {
	if len(data) <= e.minChunkSize {
		return [][]byte{data}
	}
	
	chunks := make([][]byte, 0)
	start := 0
	
	// Initialize Rabin hash
	hash := rabinkarp64.NewFromPoly(e.rabinPoly)
	
	for i := e.minChunkSize; i < len(data); i++ {
		hash.Write([]byte{data[i]})
		
		// Check if we should break here (based on hash value and constraints)
		shouldBreak := (hash.Sum64()%uint64(e.avgChunkSize) == 0) || (i-start >= e.maxChunkSize)
		
		if shouldBreak {
			chunks = append(chunks, data[start:i])
			start = i
			hash = rabinkarp64.NewFromPoly(e.rabinPoly)
		}
	}
	
	// Add remaining data as last chunk
	if start < len(data) {
		chunks = append(chunks, data[start:])
	}
	
	return chunks
}

// storeChunk stores a chunk to disk
func (e *DeduplicationEngine) storeChunk(hash string, data []byte, chunkInfo *ChunkInfo) error {
	// Create subdirectories based on hash prefix for better distribution
	subDir := hash[:2]
	dirPath := filepath.Join(e.chunkStore.basePath, subDir)
	
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create chunk directory: %w", err)
	}
	
	// Store chunk
	chunkPath := filepath.Join(dirPath, hash)
	if err := os.WriteFile(chunkPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write chunk file: %w", err)
	}
	
	chunkInfo.StorePath = chunkPath
	
	// Add to chunk store
	e.chunkStore.mutex.Lock()
	e.chunkStore.chunks[hash] = chunkInfo
	e.chunkStore.mutex.Unlock()
	
	return nil
}

// RetrieveChunk retrieves a chunk by hash
func (e *DeduplicationEngine) RetrieveChunk(hash string) ([]byte, error) {
	e.chunkStore.mutex.RLock()
	chunkInfo, exists := e.chunkStore.chunks[hash]
	e.chunkStore.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("chunk not found: %s", hash)
	}
	
	data, err := os.ReadFile(chunkInfo.StorePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read chunk file: %w", err)
	}
	
	// Update last used time
	e.index.mutex.Lock()
	chunkInfo.LastUsed = time.Now()
	e.index.mutex.Unlock()
	
	return data, nil
}

// ReconstructData reconstructs data from chunk references
func (e *DeduplicationEngine) ReconstructData(chunkRefs []ChunkReference) ([]byte, error) {
	if len(chunkRefs) == 0 {
		return nil, nil
	}
	
	// Calculate total size
	var totalSize int64
	for _, ref := range chunkRefs {
		totalSize += ref.Size
	}
	
	// Reconstruct data
	result := make([]byte, totalSize)
	var offset int64
	
	for _, ref := range chunkRefs {
		chunkData, err := e.RetrieveChunk(ref.Hash)
		if err != nil {
			return nil, fmt.Errorf("failed to retrieve chunk %s: %w", ref.Hash, err)
		}
		
		copy(result[offset:offset+ref.Size], chunkData)
		offset += ref.Size
	}
	
	return result, nil
}

// serializeChunkRefs serializes chunk references to bytes
func (e *DeduplicationEngine) serializeChunkRefs(refs []ChunkReference) []byte {
	// Simple serialization - in practice you might use protobuf or similar
	var result []byte
	for _, ref := range refs {
		result = append(result, []byte(fmt.Sprintf("%s:%d:%d;", ref.Hash, ref.Offset, ref.Size))...)
	}
	return result
}

// deserializeChunkRefs deserializes chunk references from bytes
func (e *DeduplicationEngine) deserializeChunkRefs(data []byte) ([]ChunkReference, error) {
	if len(data) == 0 {
		return nil, nil
	}
	
	// Simple deserialization - in practice you might use protobuf or similar
	parts := string(data)
	refs := make([]ChunkReference, 0)
	
	for _, part := range splitString(parts, ";") {
		if part == "" {
			continue
		}
		
		segments := splitString(part, ":")
		if len(segments) != 3 {
			return nil, fmt.Errorf("invalid chunk reference format: %s", part)
		}
		
		var offset, size int64
		if _, err := fmt.Sscanf(segments[1], "%d", &offset); err != nil {
			return nil, fmt.Errorf("invalid offset: %s", segments[1])
		}
		if _, err := fmt.Sscanf(segments[2], "%d", &size); err != nil {
			return nil, fmt.Errorf("invalid size: %s", segments[2])
		}
		
		refs = append(refs, ChunkReference{
			Hash:   segments[0],
			Offset: offset,
			Size:   size,
		})
	}
	
	return refs, nil
}

// GetStats returns current deduplication statistics
func (e *DeduplicationEngine) GetStats() *DedupStats {
	e.index.mutex.RLock()
	defer e.index.mutex.RUnlock()
	
	// Return a copy
	stats := *e.index.stats
	return &stats
}

// GarbageCollect removes unreferenced chunks
func (e *DeduplicationEngine) GarbageCollect() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	
	collected := 0
	var freedBytes int64
	
	e.index.mutex.Lock()
	for hash, chunkInfo := range e.index.chunks {
		if chunkInfo.RefCount <= 0 {
			// Remove chunk file
			if err := os.Remove(chunkInfo.StorePath); err != nil && !os.IsNotExist(err) {
				// Log error but continue
				continue
			}
			
			freedBytes += chunkInfo.Size
			delete(e.index.chunks, hash)
			
			// Remove from chunk store
			e.chunkStore.mutex.Lock()
			delete(e.chunkStore.chunks, hash)
			e.chunkStore.mutex.Unlock()
			
			collected++
		}
	}
	
	// Update stats
	e.index.stats.UniqueBytes -= freedBytes
	e.index.stats.UniqueChunks -= int64(collected)
	e.index.stats.DeduplicatedBytes = e.index.stats.TotalBytes - e.index.stats.UniqueBytes
	if e.index.stats.TotalBytes > 0 {
		e.index.stats.CompressionRatio = float64(e.index.stats.UniqueBytes) / float64(e.index.stats.TotalBytes)
	}
	e.index.stats.LastUpdated = time.Now()
	e.index.mutex.Unlock()
	
	// Save updated index
	return e.index.save()
}

// loadExistingChunks loads existing chunks from disk
func (e *DeduplicationEngine) loadExistingChunks() error {
	return filepath.Walk(e.chunkStore.basePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if !info.IsDir() {
			// Extract hash from filename
			hash := filepath.Base(path)
			if len(hash) == 64 { // SHA-256 hash length
				chunkInfo := &ChunkInfo{
					Hash:      hash,
					Size:      info.Size(),
					StorePath: path,
					CreatedAt: info.ModTime(),
					LastUsed:  info.ModTime(),
				}
				
				// Check if exists in index
				e.index.mutex.RLock()
				indexInfo, exists := e.index.chunks[hash]
				e.index.mutex.RUnlock()
				
				if exists {
					chunkInfo.RefCount = indexInfo.RefCount
				} else {
					// Orphaned chunk - mark for garbage collection
					chunkInfo.RefCount = 0
				}
				
				e.chunkStore.mutex.Lock()
				e.chunkStore.chunks[hash] = chunkInfo
				e.chunkStore.mutex.Unlock()
			}
		}
		
		return nil
	})
}

// load loads the deduplication index from disk
func (idx *DedupIndex) load() error {
	return loadJSON(idx.filePath, idx)
}

// save saves the deduplication index to disk
func (idx *DedupIndex) save() error {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()
	
	return saveJSON(idx.filePath, idx)
}

// calculateChunkHash calculates SHA-256 hash for chunk data
func calculateChunkHash(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// splitString splits a string by delimiter
func splitString(s, delim string) []string {
	if s == "" {
		return nil
	}
	
	var result []string
	var current string
	
	for _, char := range s {
		if string(char) == delim {
			result = append(result, current)
			current = ""
		} else {
			current += string(char)
		}
	}
	
	if current != "" {
		result = append(result, current)
	}
	
	return result
}

// Close closes the deduplication engine and saves state
func (e *DeduplicationEngine) Close() error {
	return e.index.save()
}