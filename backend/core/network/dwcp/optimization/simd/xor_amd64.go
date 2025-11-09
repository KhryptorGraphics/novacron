// +build amd64

package simd

import (
	"github.com/klauspost/cpuid/v2"
)

// XORDeltaEncoder provides SIMD-accelerated XOR operations for delta encoding
type XORDeltaEncoder struct {
	hasAVX2 bool
	hasSSSE3 bool
}

// NewXORDeltaEncoder creates a new SIMD-accelerated XOR delta encoder
func NewXORDeltaEncoder() *XORDeltaEncoder {
	return &XORDeltaEncoder{
		hasAVX2:  cpuid.CPU.Supports(cpuid.AVX2),
		hasSSSE3: cpuid.CPU.Supports(cpuid.SSSE3),
	}
}

// XORBytes performs XOR operation on two byte slices
func (e *XORDeltaEncoder) XORBytes(dst, src1, src2 []byte) {
	if len(dst) != len(src1) || len(dst) != len(src2) {
		panic("XORBytes: slice length mismatch")
	}

	// Use AVX2 for large buffers
	if e.hasAVX2 && len(dst) >= 256 {
		xorBytesAVX2(dst, src1, src2)
		return
	}

	// Use SSSE3 for medium buffers
	if e.hasSSSE3 && len(dst) >= 128 {
		xorBytesSSSE3(dst, src1, src2)
		return
	}

	// Fallback to scalar
	xorBytesScalar(dst, src1, src2)
}

// xorBytesScalar is the fallback scalar implementation
func xorBytesScalar(dst, src1, src2 []byte) {
	// Process 8 bytes at a time using uint64
	n := len(dst)
	i := 0

	// Unroll loop for better performance
	for i+8 <= n {
		d1 := (*uint64)(unsafe.Pointer(&dst[i]))
		s1 := (*uint64)(unsafe.Pointer(&src1[i]))
		s2 := (*uint64)(unsafe.Pointer(&src2[i]))
		*d1 = *s1 ^ *s2
		i += 8
	}

	// Handle remaining bytes
	for i < n {
		dst[i] = src1[i] ^ src2[i]
		i++
	}
}

// EncodeDelta creates delta between two frames using SIMD
func (e *XORDeltaEncoder) EncodeDelta(current, previous []byte) []byte {
	delta := make([]byte, len(current))
	e.XORBytes(delta, current, previous)
	return delta
}

// DecodeDelta applies delta to previous frame using SIMD
func (e *XORDeltaEncoder) DecodeDelta(delta, previous []byte) []byte {
	current := make([]byte, len(delta))
	e.XORBytes(current, delta, previous)
	return current
}

// FindChangedRegions identifies non-zero regions in delta
func (e *XORDeltaEncoder) FindChangedRegions(delta []byte, minSize int) []Region {
	regions := make([]Region, 0, 16)
	start := -1

	for i := 0; i < len(delta); i++ {
		if delta[i] != 0 {
			if start == -1 {
				start = i
			}
		} else if start != -1 {
			if i-start >= minSize {
				regions = append(regions, Region{
					Offset: start,
					Length: i - start,
				})
			}
			start = -1
		}
	}

	// Handle final region
	if start != -1 && len(delta)-start >= minSize {
		regions = append(regions, Region{
			Offset: start,
			Length: len(delta) - start,
		})
	}

	return regions
}

// Region represents a changed region in delta
type Region struct {
	Offset int
	Length int
}

// CompressDelta compresses delta by storing only changed regions
func (e *XORDeltaEncoder) CompressDelta(delta []byte, minRegionSize int) []byte {
	regions := e.FindChangedRegions(delta, minRegionSize)

	if len(regions) == 0 {
		return []byte{0} // No changes
	}

	// Calculate compressed size
	size := 1 + len(regions)*8 // Header + region descriptors
	for _, r := range regions {
		size += r.Length
	}

	compressed := make([]byte, 0, size)
	compressed = append(compressed, byte(len(regions)))

	// Write region descriptors
	for _, r := range regions {
		compressed = append(compressed,
			byte(r.Offset>>24), byte(r.Offset>>16),
			byte(r.Offset>>8), byte(r.Offset),
			byte(r.Length>>24), byte(r.Length>>16),
			byte(r.Length>>8), byte(r.Length),
		)
	}

	// Write region data
	for _, r := range regions {
		compressed = append(compressed, delta[r.Offset:r.Offset+r.Length]...)
	}

	return compressed
}

// DecompressDelta restores full delta from compressed form
func (e *XORDeltaEncoder) DecompressDelta(compressed []byte, fullSize int) []byte {
	if len(compressed) == 0 || compressed[0] == 0 {
		return make([]byte, fullSize)
	}

	delta := make([]byte, fullSize)
	numRegions := int(compressed[0])
	offset := 1

	// Read region descriptors
	regions := make([]Region, numRegions)
	for i := 0; i < numRegions; i++ {
		regions[i].Offset = int(compressed[offset])<<24 | int(compressed[offset+1])<<16 |
			int(compressed[offset+2])<<8 | int(compressed[offset+3])
		regions[i].Length = int(compressed[offset+4])<<24 | int(compressed[offset+5])<<16 |
			int(compressed[offset+6])<<8 | int(compressed[offset+7])
		offset += 8
	}

	// Read region data
	for _, r := range regions {
		copy(delta[r.Offset:], compressed[offset:offset+r.Length])
		offset += r.Length
	}

	return delta
}

import "unsafe"
