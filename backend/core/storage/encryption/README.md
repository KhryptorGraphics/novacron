# Data Encryption for NovaCron

This package implements data encryption capabilities for the NovaCron distributed storage system, providing security and privacy for stored data.

## Overview

The encryption system provides multiple encryption algorithms and modes, allowing for a balance between security, performance, and compatibility. It integrates with the deduplication and compression systems to provide a complete data security and efficiency solution.

## Features

### Multiple Encryption Algorithms

- **AES-256**: Advanced Encryption Standard with 256-bit keys (recommended)
- **AES-192**: AES with 192-bit keys
- **AES-128**: AES with 128-bit keys
- **ChaCha20**: Stream cipher optimized for software implementations

### Encryption Modes

- **GCM (Galois/Counter Mode)**: Authenticated encryption with associated data (AEAD)
- **CBC (Cipher Block Chaining)**: Traditional block cipher mode
- **CTR (Counter)**: Converts a block cipher into a stream cipher

### Security Features

- **Per-Volume Keys**: Each volume gets a unique encryption key
- **Key Derivation**: Master key used to derive volume-specific keys
- **Authenticated Encryption**: Integrity protection with GCM mode
- **Nonce Management**: Secure handling of initialization vectors
- **Key Verification**: Verification data to detect incorrect keys

## Usage

### Creating an Encryptor

```go
// Create an encryption configuration
config := encryption.DefaultEncryptionConfig()
config.Algorithm = encryption.EncryptionAES256
config.Mode = encryption.EncryptionModeGCM
config.MasterKey = "your-secure-master-key"
config.MinSizeBytes = 1024  // Only encrypt data >= 1KB

// Create an encryptor
encryptor, err := encryption.NewEncryptor(config)
if err != nil {
    log.Fatalf("Failed to create encryptor: %v", err)
}
```

### Encrypting Data

```go
// Encrypt data for a specific volume
encryptedData, err := encryptor.Encrypt(data, volumeID)
if err != nil {
    log.Fatalf("Failed to encrypt data: %v", err)
}

// The returned encryptedData contains:
// - Algorithm used
// - Mode used
// - Encrypted data
// - IV/Nonce (securely generated per encryption operation)
// - Additional authentication data (for GCM mode)
```

### Decrypting Data

```go
// Decrypt data for a specific volume
originalData, err := encryptor.Decrypt(encryptedData, volumeID)
if err != nil {
    log.Fatalf("Failed to decrypt data: %v", err)
}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `Algorithm` | Encryption algorithm | `EncryptionAES256` |
| `Mode` | Encryption mode | `EncryptionModeGCM` |
| `MasterKey` | Master key for deriving volume keys | Required |
| `MinSizeBytes` | Minimum size to apply encryption | `0` (encrypt everything) |
| `IncludeVerification` | Include verification data | `true` |
| `RotationInterval` | Key rotation interval | `720h` (30 days) |
| `KeyDerivationIterations` | PBKDF2 iterations for key derivation | `10000` |

## Architecture

The encryption system consists of the following key components:

### 1. Encryptor

The main component that orchestrates the encryption process. It:
- Receives data from the storage system
- Determines if encryption should be applied
- Derives the correct key for the volume
- Performs encryption/decryption operations
- Manages nonces and IVs securely

### 2. EncryptionAlgorithm

Enum defining the available encryption algorithms:
- `EncryptionNone`: No encryption
- `EncryptionAES128`: AES with 128-bit keys
- `EncryptionAES192`: AES with 192-bit keys
- `EncryptionAES256`: AES with 256-bit keys
- `EncryptionChaCha20`: ChaCha20 stream cipher

### 3. EncryptionMode

Enum defining encryption modes:
- `EncryptionModeGCM`: Authenticated encryption with associated data
- `EncryptionModeCBC`: Cipher Block Chaining mode
- `EncryptionModeCTR`: Counter mode

### 4. EncryptedData

Structure containing encrypted data and metadata:
- `Data`: The encrypted data bytes
- `Algorithm`: The algorithm used
- `Mode`: The encryption mode used
- `IV`: Initialization vector or nonce
- `AuthTag`: Authentication tag (for GCM mode)

## Security Considerations

### Key Management

- The master key should be stored securely, ideally in a key management service
- Volume keys are derived from the master key and volume ID using PBKDF2
- Keys are never stored in plaintext on disk
- Consider key rotation for long-term security

### Algorithm Selection

- AES-256 is recommended for most use cases
- AES-192 and AES-128 provide good security with slightly better performance
- ChaCha20 is a good alternative on platforms without AES hardware acceleration

### Mode Selection

- GCM is recommended for most use cases as it provides authenticated encryption
- CBC requires proper padding and careful IV management
- CTR is fast but requires careful nonce management

### Nonce/IV Management

- Each encryption operation uses a unique nonce/IV
- For GCM and CTR, nonces must never be reused with the same key
- Nonces are generated using a cryptographically secure random number generator

## Examples

### Configuring Different Encryption Algorithms

#### AES-256 with GCM (recommended)

```go
config := encryption.DefaultEncryptionConfig()
config.Algorithm = encryption.EncryptionAES256
config.Mode = encryption.EncryptionModeGCM
```

#### AES-128 with CBC

```go
config := encryption.DefaultEncryptionConfig()
config.Algorithm = encryption.EncryptionAES128
config.Mode = encryption.EncryptionModeCBC
```

#### ChaCha20

```go
config := encryption.DefaultEncryptionConfig()
config.Algorithm = encryption.EncryptionChaCha20
```

### Integration with Deduplication and Compression

For maximum security and efficiency, encryption should be applied after deduplication and compression:

```go
// 1. Deduplicate
dedupInfo, _ := deduplicator.Deduplicate(data)

// 2. Convert to bytes for compression
dedupBytes := dedupInfo.ToBytes()

// 3. Compress
compressedData, _ := compressor.Compress(dedupBytes)

// 4. Encrypt
encryptedData, _ := encryptor.Encrypt(compressedData, volumeID)
```

## Implementation Notes

### Key Derivation

Volume keys are derived from the master key using:
- PBKDF2 key derivation function
- Volume ID as the salt
- Configurable number of iterations
- SHA-256 as the hash function

This ensures each volume has a unique key even with the same master key.

### Authenticated Encryption

When using GCM mode:
- Data authenticity is verified during decryption
- Tampering will cause decryption to fail
- Associated data can be included for additional verification

### Padding

For block cipher modes that require padding (CBC):
- PKCS#7 padding is used
- Padding is automatically handled during encryption/decryption

### Performance Optimization

- Small data (configurable threshold) can skip encryption
- AES-NI hardware acceleration is used when available
- Parallelization for large data blocks

## Future Enhancements

1. **Hardware Security Module (HSM) Integration**: Support for external key storage
2. **Additional Algorithms**: Support for post-quantum cryptography
3. **Vault Integration**: Integration with Hashicorp Vault for key management
4. **Envelope Encryption**: Support for envelope encryption models
5. **Multi-tenant Key Management**: Isolated keys for multi-tenant deployments
