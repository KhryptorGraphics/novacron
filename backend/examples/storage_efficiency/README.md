# Storage Efficiency Example

This example demonstrates the usage of Novacron's distributed storage system with its three key storage efficiency features:

1. **Data Deduplication**: Eliminates redundant data blocks
2. **Compression**: Reduces data size using various compression algorithms
3. **Encryption**: Secures stored data while maintaining efficiency

## What This Example Demonstrates

- Creating and configuring a distributed storage service
- Setting up deduplication, compression, and encryption
- Writing and reading data through the multi-layer efficiency pipeline
- Visualizing efficiency metrics (deduplication ratio, compression ratio, etc.)
- Verifying data integrity through the full pipeline

## The Storage Efficiency Pipeline

This example shows how data flows through a layered efficiency pipeline:

```
Raw Data → Deduplication → Compression → Encryption → Storage
```

During reads, the process is reversed:

```
Storage → Decryption → Decompression → Reconstruction → Original Data
```

## How to Run

Since this is an example that requires the full Novacron infrastructure to be running, it's primarily intended as a reference for how to use the storage efficiency features. The code can be executed if the Novacron backend is properly set up, but the examples currently contain dependencies that may prevent them from running in isolation.

```bash
# From the project root
cd backend/examples/storage_efficiency
go run main.go
```

## Code Walkthrough

The example consists of several key sections:

1. **Configuration Setup**: Demonstrates how to configure each component (deduplication, compression, encryption)
2. **Service Creation**: Shows how to create and start the distributed storage service
3. **Storage Node Addition**: Illustrates adding storage nodes to the cluster
4. **Volume Creation**: Demonstrates creating a distributed volume with efficiency features enabled
5. **Data Operations**: Shows writing data with redundancy and reading it back
6. **Verification**: Confirms data integrity through the full pipeline
7. **Metrics Display**: Shows how to access and interpret efficiency metrics

## Key Components

### Deduplication

The example configures content-defined chunking for optimal deduplication:

```go
dedupConfig := deduplication.DefaultDedupConfig()
dedupConfig.Algorithm = deduplication.DedupContent
dedupConfig.MinBlockSize = 4 * 1024     // 4 KB minimum
dedupConfig.TargetBlockSize = 64 * 1024 // 64 KB target
dedupConfig.Enabled = true
```

### Compression

The example uses Gzip compression with auto-detection:

```go
compConfig := compression.DefaultCompressionConfig()
compConfig.Algorithm = compression.CompressionGzip
compConfig.Level = compression.CompressionDefault
compConfig.AutoDetect = true
```

### Encryption

The example demonstrates AES-256 encryption with GCM mode:

```go
encConfig := encryption.DefaultEncryptionConfig()
encConfig.Algorithm = encryption.EncryptionAES256
encConfig.Mode = encryption.EncryptionModeGCM
encConfig.MasterKey = "your-secure-master-key-replace-in-production"
```

## Expected Results

When running the example, you should see log messages showing the efficiency metrics:

```
Original data size: 100000 bytes
Data verification successful!
Storage efficiency:
- Deduplication: true
  Algorithm: content
  Ratio: 10.50
  Unique blocks: 95
- Compression: gzip
  Original size: 9524 bytes
  Ratio: 2.75
- Encryption: true
  Algorithm: aes256
  Mode: gcm
```

The actual metrics will vary depending on the nature of the test data.

## Note About Real-World Usage

In a production environment:

1. Use a secure key management system for encryption keys
2. Adjust deduplication block sizes based on your data characteristics
3. Select compression algorithms based on your performance/ratio requirements
4. Consider the performance implications of enabling all three features simultaneously
