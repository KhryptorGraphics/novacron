// +build amd64

package simd

import (
	"hash/crc32"
	"github.com/klauspost/cpuid/v2"
)

// ChecksumCalculator provides SIMD-accelerated checksum calculation
type ChecksumCalculator struct {
	hasCLMUL bool
	table    *crc32.Table
}

// NewChecksumCalculator creates a new SIMD-accelerated checksum calculator
func NewChecksumCalculator() *ChecksumCalculator {
	return &ChecksumCalculator{
		hasCLMUL: cpuid.CPU.Supports(cpuid.CLMUL),
		table:    crc32.MakeTable(crc32.IEEE),
	}
}

// CalculateCRC32 computes CRC32 checksum with SIMD acceleration
func (c *ChecksumCalculator) CalculateCRC32(data []byte) uint32 {
	if c.hasCLMUL && len(data) >= 256 {
		return crc32CLMUL(data)
	}
	return crc32.Checksum(data, c.table)
}

// CalculateCRC32C computes CRC32C (Castagnoli) checksum
func (c *ChecksumCalculator) CalculateCRC32C(data []byte) uint32 {
	if c.hasCLMUL && len(data) >= 256 {
		return crc32cCLMUL(data)
	}
	return crc32.Checksum(data, crc32.MakeTable(crc32.Castagnoli))
}

// VerifyChecksum verifies data integrity using CRC32
func (c *ChecksumCalculator) VerifyChecksum(data []byte, expected uint32) bool {
	return c.CalculateCRC32(data) == expected
}

// Rolling hash for incremental checksum updates
type RollingHash struct {
	window []byte
	size   int
	pos    int
	hash   uint32
}

// NewRollingHash creates a new rolling hash with given window size
func NewRollingHash(windowSize int) *RollingHash {
	return &RollingHash{
		window: make([]byte, windowSize),
		size:   windowSize,
		pos:    0,
		hash:   0,
	}
}

// Update updates rolling hash with new byte
func (rh *RollingHash) Update(b byte) uint32 {
	// Remove oldest byte
	old := rh.window[rh.pos]
	rh.hash = rh.hash - uint32(old)

	// Add new byte
	rh.window[rh.pos] = b
	rh.hash = rh.hash + uint32(b)

	rh.pos = (rh.pos + 1) % rh.size
	return rh.hash
}

// Hash returns current hash value
func (rh *RollingHash) Hash() uint32 {
	return rh.hash
}

// Adler32 computes Adler-32 checksum (faster than CRC32)
func Adler32(data []byte) uint32 {
	const mod = 65521
	a := uint32(1)
	b := uint32(0)

	// Process 16 bytes at a time for better performance
	i := 0
	for i+16 <= len(data) {
		for j := 0; j < 16; j++ {
			a += uint32(data[i+j])
			b += a
		}
		a %= mod
		b %= mod
		i += 16
	}

	// Process remaining bytes
	for i < len(data) {
		a += uint32(data[i])
		b += a
		i++
	}

	a %= mod
	b %= mod

	return (b << 16) | a
}

// Fletcher32 computes Fletcher-32 checksum
func Fletcher32(data []byte) uint32 {
	// Pad to even length
	if len(data)%2 != 0 {
		data = append(data, 0)
	}

	sum1 := uint32(0)
	sum2 := uint32(0)

	for i := 0; i < len(data); i += 2 {
		word := uint32(data[i])<<8 | uint32(data[i+1])
		sum1 = (sum1 + word) % 65535
		sum2 = (sum2 + sum1) % 65535
	}

	return (sum2 << 16) | sum1
}

// xxHash32 computes xxHash (very fast, good distribution)
func xxHash32(data []byte, seed uint32) uint32 {
	const (
		prime1 = 2654435761
		prime2 = 2246822519
		prime3 = 3266489917
		prime4 = 668265263
		prime5 = 374761393
	)

	h32 := seed + prime5
	length := uint32(len(data))

	// Process 16-byte chunks
	i := 0
	for i+16 <= len(data) {
		v1 := seed + prime1 + prime2
		v2 := seed + prime2
		v3 := seed
		v4 := seed - prime1

		for j := 0; j < 4; j++ {
			k := uint32(data[i])<<24 | uint32(data[i+1])<<16 | uint32(data[i+2])<<8 | uint32(data[i+3])
			i += 4

			switch j {
			case 0:
				v1 += k * prime2
				v1 = (v1 << 13) | (v1 >> 19)
				v1 *= prime1
			case 1:
				v2 += k * prime2
				v2 = (v2 << 13) | (v2 >> 19)
				v2 *= prime1
			case 2:
				v3 += k * prime2
				v3 = (v3 << 13) | (v3 >> 19)
				v3 *= prime1
			case 3:
				v4 += k * prime2
				v4 = (v4 << 13) | (v4 >> 19)
				v4 *= prime1
			}
		}

		h32 = ((v1 << 1) | (v1 >> 31)) + ((v2 << 7) | (v2 >> 25)) +
			((v3 << 12) | (v3 >> 20)) + ((v4 << 18) | (v4 >> 14))
	}

	h32 += length

	// Process remaining bytes
	for i+4 <= len(data) {
		k := uint32(data[i])<<24 | uint32(data[i+1])<<16 | uint32(data[i+2])<<8 | uint32(data[i+3])
		h32 += k * prime3
		h32 = ((h32 << 17) | (h32 >> 15)) * prime4
		i += 4
	}

	for i < len(data) {
		h32 += uint32(data[i]) * prime5
		h32 = ((h32 << 11) | (h32 >> 21)) * prime1
		i++
	}

	// Final mix
	h32 ^= h32 >> 15
	h32 *= prime2
	h32 ^= h32 >> 13
	h32 *= prime3
	h32 ^= h32 >> 16

	return h32
}
