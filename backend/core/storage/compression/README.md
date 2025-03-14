# Data Compression for NovaCron

This package implements data compression capabilities for the NovaCron distributed storage system, enabling efficient storage utilization by reducing the size of stored data.

## Overview

The compression system provides multiple compression algorithms and levels, allowing for a balance between compression ratio, CPU usage, and speed. It supports adaptive compression and can be integrated with deduplication and encryption for a complete data efficiency solution.

## Features

### Multiple Compression Algorithms

- **Gzip**: Standard compression with good balance of speed and compression ratio
- **Zlib**: Similar to Gzip but with a different header format
- **LZW**: Dictionary-based compression good for text and structured data
- **Zstd**: Modern algorithm with high compression ratios and fast decompression
- **LZMA**: High compression ratio at the cost of slower compression speed

### Compression Levels

- **Fastest**: Minimal compression but very fast processing
- **Fast**: Good balance for speed-sensitive workloads
- **Default**: Balanced compression/speed tradeoff
- **Better**: Improved compression at moderate speed cost
- **Best**: Maximum compression with slowest processing

### Smart Features

- **Auto-detection**: Automatically detects if data is compressible
- **Adaptive Compression**: Selects algorithms based on data type
- **Size Control**: Skip compression for very small or very large data
- **Metadata Tracking**: Stores information about compressed data

## Usage

### Creating a Compressor

```go
// Create a compression configuration
config := compression.DefaultCompressionConfig()
config.Algorithm = compression.CompressionGzip
config.Level = compression.CompressionDefault
config.AutoDetect = true
config.MinSizeBytes = 4 * 1024        // 4KB minimum
config.MaxSizeBytes = 32 * 1024 * 1024 // 32MB maximum

// Create a compressor
compressor := compression.NewCompressor(config)
```

### Compressing Data

```go
// Compress data
compressedData, err := compressor.Compress(data)
if err != nil {
    log.Fatalf("Failed to compress data: %v", err)
}

// Compress with metadata
compressedData, err := compressor.CompressWithMetadata(data)
if err != nil {
    log.Fatalf("Failed to compress data: %v", err)
}
fmt.Printf("Algorithm: %s\n", compressedData.Algorithm)
fmt.Printf("Original size: %d\n", compressedData.OriginalSize)
fmt.Printf("Compressed size: %d\n", len(compressedData.Data))
fmt.Printf("Compression ratio: %.2f\n", compressedData.CompressionRatio)
```

### Decompressing Data

```go
// Decompress data
originalData, err := compressor.Decompress(compressedData, compression.CompressionGzip)
if err != nil {
    log.Fatalf("Failed to decompress data: %v", err)
}

// Decompress from compressed data with metadata
originalData, err := compressor.Decompress(compressedData.Data, compressedData.Algorithm)
if err != nil {
    log.Fatalf("Failed to decompress data: %v", err)
}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `Algorithm` | Compression algorithm | `CompressionGzip` |
| `Level` | Compression level | `CompressionDefault` |
| `AutoDetect` | Whether to detect if data is compressible | `true` |
| `MinSizeBytes` | Minimum size for compression | `4 KB` |
| `MaxSizeBytes` | Maximum size for compression | `32 MB` |
| `CompressionThreshold` | Ratio threshold to keep compression | `1.1` |
| `SampleSize` | Sample size for compression detection | `4 KB` |

## Architecture

The compression system consists of the following key components:

### 1. Compressor

The main component that orchestrates the compression process. It:
- Receives raw data from the storage system
- Determines if compression should be applied
- Selects and applies the appropriate compression algorithm
- Returns compressed data with metadata

### 2. CompressionAlgorithm

Enum defining the available compression algorithms:
- `CompressionNone`: No compression
- `CompressionGzip`: Standard GZIP compression
- `CompressionZlib`: ZLIB compression
- `CompressionLZW`: LZW dictionary-based compression
- `CompressionZstd`: Zstandard compression
- `CompressionLZMA`: LZMA high-ratio compression

### 3. CompressionLevel

Enum defining compression levels:
- `CompressionFastest`: Fastest compression with lowest ratio
- `CompressionFast`: Fast compression with moderate ratio
- `CompressionDefault`: Balanced compression/speed
- `CompressionBetter`: Higher ratio, slower speed
- `CompressionBest`: Highest ratio, slowest speed

### 4. CompressedData

Structure containing compressed data and metadata:
- `Data`: The compressed data bytes
- `Algorithm`: The algorithm used
- `OriginalSize`: Size before compression
- `CompressionRatio`: Achieved compression ratio

## Performance Considerations

### Compression Ratio

The effectiveness of compression depends on the nature of the data:
- Text files: 60-80% reduction
- Database dumps: 70-90% reduction
- Structured data: 40-80% reduction
- Images/videos: 0-5% reduction (already compressed)
- Executables: 20-60% reduction

### CPU Usage

Compression is CPU-intensive. Considerations:
- `CompressionFastest` uses minimal CPU but has lower ratios
- `CompressionBest` uses significant CPU but achieves higher ratios
- Consider workload characteristics when selecting algorithms

### Memory Usage

Different algorithms have different memory requirements:
- Gzip/Zlib: Moderate memory usage
- LZMA: Higher memory usage for better compression
- Zstd: Good balance of memory usage and compression

## Examples

### Configuring Different Compression Algorithms

#### Gzip Compression

```go
config := compression.DefaultCompressionConfig()
config.Algorithm = compression.CompressionGzip
config.Level = compression.CompressionDefault
```

#### High-ratio LZMA Compression

```go
config := compression.DefaultCompressionConfig()
config.Algorithm = compression.CompressionLZMA
config.Level = compression.CompressionBest
```

#### Fast Zstd Compression

```go
config := compression.DefaultCompressionConfig()
config.Algorithm = compression.CompressionZstd
config.Level = compression.CompressionFast
```

### Integration with Deduplication and Encryption

For maximum efficiency, compression should be applied after deduplication but before encryption:

```go
// 1. Deduplicate
dedupInfo, _ := deduplicator.Deduplicate(data)

// 2. Convert to bytes for compression
dedupBytes := dedupInfo.ToBytes()

// 3. Compress 
compressedData, _ := compressor.Compress(dedupBytes)

// 4. Encrypt
encryptedData, _ := encryptor.Encrypt(compressedData)
```

## Implementation Notes

### Compression Detection

The auto-detection mechanism:
- Takes a small sample of the data
- Attempts compression with the configured algorithm
- Keeps the compressed version only if the ratio exceeds the threshold
- Falls back to uncompressed data if the ratio is insufficient

### Algorithm Selection

Consider these guidelines when selecting compression algorithms:
- Gzip: Good general-purpose algorithm for most data
- Zstd: Modern alternative with better performance characteristics
- LZMA: Best for cold storage or archive data where speed is less important
- LZW: Good for specific text formats but generally outperformed by others

### Level Selection

Compression level selection depends on the use case:
- Live storage: Use `CompressionFast` or `CompressionDefault`
- Archival: Use `CompressionBetter` or `CompressionBest`
- Time-sensitive operations: Use `CompressionFastest`

## Future Enhancements

1. **Machine Learning Compression**: Use ML to select optimal algorithm and level
2. **Domain-specific Compressors**: Specialized for VM images, database dumps, etc.
3. **Parallel Compression**: Utilize multiple cores for faster compression
4. **Streaming API**: Support for streamed compression/decompression
5. **Hardware Acceleration**: Leverage hardware compression when available
