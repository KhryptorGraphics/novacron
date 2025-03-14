# Data Deduplication for NovaCron

This package implements data deduplication capabilities for the NovaCron distributed storage system, enabling efficient storage utilization by eliminating redundant data blocks.

## Overview

The deduplication system works at the block level to identify and eliminate duplicate data across files and volumes. It supports multiple chunking strategies and can be integrated with compression and encryption for a complete data efficiency solution.

## Features

### Multiple Deduplication Methods

- **Fixed-size Chunking**: Divides data into chunks of a predetermined size
- **Variable-size Chunking**: Uses content-defined boundaries for more effective deduplication
- **Content-defined Chunking**: Identifies natural boundaries in the data for optimal deduplication

### Deduplication Store

- **Block Storage**: Unique data blocks are stored once and referenced by multiple files
- **Reference Counting**: Tracks how many files use each block for garbage collection
- **Metadata Management**: Stores information about block relationships and file reconstruction

### Integration

- **Pipeline Integration**: Seamlessly works with compression and encryption
- **Transparent Operation**: Higher-level components read/write normally without knowledge of deduplication
- **Volume-Level Configuration**: Deduplication can be enabled/disabled per volume

## Usage

### Creating a Deduplicator

```go
// Create a deduplication configuration
config := deduplication.DefaultDedupConfig()
config.Algorithm = deduplication.DedupContent
config.MinBlockSize = 4 * 1024    // 4 KB minimum
config.TargetBlockSize = 64 * 1024 // 64 KB target
config.MaxBlockSize = 1024 * 1024  // 1 MB maximum
config.Enabled = true

// Create a deduplicator
deduplicator, err := deduplication.NewDeduplicator(config)
if err != nil {
    log.Fatalf("Failed to create deduplicator: %v", err)
}
```

### Deduplicating Data

```go
// Deduplicate data and get deduplication information
dedupInfo, err := deduplicator.Deduplicate(data)
if err != nil {
    log.Fatalf("Failed to deduplicate data: %v", err)
}

// Check deduplication results
fmt.Printf("Algorithm: %s\n", dedupInfo.Algorithm)
fmt.Printf("Deduplication ratio: %.2f\n", dedupInfo.DedupRatio)
fmt.Printf("Unique blocks: %d\n", len(dedupInfo.Blocks))
```

### Reconstructing Data

```go
// Reconstruct original data from deduplication information
originalData, err := deduplicator.Reconstruct(dedupInfo)
if err != nil {
    log.Fatalf("Failed to reconstruct data: %v", err)
}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `Algorithm` | Deduplication algorithm (Fixed, Variable, Content) | `DedupContent` |
| `MinBlockSize` | Minimum block size for variable chunking | `4 KB` |
| `TargetBlockSize` | Target block size for chunking | `64 KB` |
| `MaxBlockSize` | Maximum block size for variable chunking | `1 MB` |
| `Enabled` | Whether deduplication is enabled | `true` |
| `HashAlgorithm` | Hash algorithm for block identification | `SHA-256` |
| `InlineSmallBlocks` | Store small blocks inline rather than deduplicated | `true` |
| `SmallBlockThreshold` | Threshold for inline small blocks | `1 KB` |

## Architecture

The deduplication system consists of the following key components:

### 1. Deduplicator

The main component that orchestrates the deduplication process. It:
- Receives raw data from the storage system
- Chunks the data according to the chosen algorithm
- Identifies duplicate blocks
- Stores unique blocks and metadata
- Returns deduplication information

### 2. Chunker

Responsible for breaking data into chunks according to different strategies:
- `FixedChunker`: Creates chunks of a fixed size
- `VariableChunker`: Creates chunks with variable sizes within constraints
- `ContentChunker`: Uses content-defined chunking for optimal deduplication

### 3. BlockStore

Manages storage of unique data blocks:
- Stores blocks keyed by their hash
- Handles reference counting for garbage collection
- Provides efficient block retrieval

### 4. DedupFileInfo

Contains all information needed to reconstruct a file:
- Metadata about the original file
- List of block references
- Deduplication statistics

## Performance Considerations

### Deduplication Ratio

The effectiveness of deduplication depends on the nature of the data:
- Virtual machine images: 50-80% reduction
- Backups: 60-95% reduction
- Database dumps: 30-70% reduction
- Media files: 0-10% reduction (already compressed)

### CPU Usage

Deduplication is CPU-intensive, especially with content-defined chunking. Considerations:
- `FixedChunker` is fastest but least effective
- `ContentChunker` is most effective but most CPU-intensive
- Use lower-CPU hash algorithms for less critical data

### Memory Usage

Block processing requires memory for:
- Hash calculation
- Block metadata caching
- Reference mapping

For large datasets, configure memory limits or use streaming approaches.

## Examples

### Configuring Different Chunking Strategies

#### Fixed-size Chunking

```go
config := deduplication.DefaultDedupConfig()
config.Algorithm = deduplication.DedupFixed
config.TargetBlockSize = 32 * 1024 // 32 KB chunks
```

#### Content-defined Chunking

```go
config := deduplication.DefaultDedupConfig()
config.Algorithm = deduplication.DedupContent
config.MinBlockSize = 8 * 1024     // 8 KB minimum
config.TargetBlockSize = 32 * 1024 // 32 KB target
config.MaxBlockSize = 128 * 1024   // 128 KB maximum
```

### Deduplication with Compression and Encryption

For maximum space efficiency, deduplication should be applied before compression and encryption:

```go
// 1. Deduplicate
dedupInfo, _ := deduplicator.Deduplicate(data)

// 2. Compress (if storing the blocks directly)
compressedData, _ := compressor.Compress(dedupInfo.ToBytes())

// 3. Encrypt (if storing the blocks directly)
encryptedData, _ := encryptor.Encrypt(compressedData)
```

## Implementation Notes

### Block Identification

Blocks are identified by their content hash (default SHA-256):
- Collision resistance is important
- Performance vs. security tradeoff
- Faster algorithms available for non-security-critical data

### Block Storage

Blocks can be stored in various backends:
- Local filesystem with flat or hierarchical structure
- Object storage (S3, Azure Blob, etc.)
- Custom storage engines

### Garbage Collection

To maintain storage efficiency:
- Reference counting tracks block usage
- Unused blocks are removed when reference count hits zero
- Periodic validation ensures block integrity

## Future Enhancements

1. **Global Deduplication**: Cross-volume deduplication for maximum efficiency
2. **Predictive Loading**: Pre-fetch blocks based on access patterns
3. **Tiered Deduplication**: Different strategies for different data types
4. **ML-based Chunking**: Use machine learning to optimize chunk boundaries
5. **Distributed Block Store**: High-performance distributed storage for blocks
