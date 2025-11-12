# DWCP Phase 1 Test Data

This directory contains test data samples for DWCP Phase 1 validation tests.

## Directory Structure

```
testdata/
├── vm_memory_samples/    # Realistic VM memory patterns
├── vm_disk_samples/      # VM disk block samples
├── cluster_state_samples/# Federation state samples
└── dictionaries/         # Pre-trained test dictionaries
```

## VM Memory Samples

Simulated VM memory pages with typical patterns:
- Zero pages (60-70% of VM memory)
- Kernel data structures (repetitive patterns)
- User space heap data (mixed patterns)
- Stack frames (highly compressible)

## VM Disk Samples

Simulated VM disk blocks:
- Filesystem metadata (highly compressible)
- Application binaries (moderately compressible)
- User data (variable compression)

## Cluster State Samples

Federation state data:
- Node metadata (JSON structures)
- Resource allocations (repetitive patterns)
- Configuration data (highly compressible)

## Dictionaries

Pre-trained Zstandard dictionaries for:
- VM memory compression
- VM disk compression
- Cluster state compression

## Generating Test Data

Test data is generated programmatically by the test suite. This directory
provides samples for:

1. **Baseline testing** - Consistent results across test runs
2. **Performance benchmarks** - Realistic data patterns
3. **Compression validation** - Known compression ratios

## Usage in Tests

```go
// Load VM memory sample
vmMemorySample := loadTestData("vm_memory_samples/8gb_sample.bin")

// Load pre-trained dictionary
dict := loadDictionary("dictionaries/vm_memory.dict")

// Use in compression tests
encoder.TrainDictionary("vm-test", dict)
encoded := encoder.Encode("vm-test", vmMemorySample)
```

## Sample Characteristics

### VM Memory (8GB sample)
- **Size**: 8 GB
- **Pattern**: 65% zeros, 25% repetitive, 10% random
- **Expected compression**: 15-20x with dictionary

### VM Disk (4GB sample)
- **Size**: 4 GB
- **Pattern**: Filesystem with typical usage
- **Expected compression**: 8-12x with delta encoding

### Cluster State (4MB sample)
- **Size**: 4 MB
- **Pattern**: JSON structures with node metadata
- **Expected compression**: 10-15x with dictionary

## Generating Samples

To generate test samples:

```bash
cd backend/core/network/dwcp
go test -v -run TestGenerateSamples
```

This will create all required test data samples in this directory.
