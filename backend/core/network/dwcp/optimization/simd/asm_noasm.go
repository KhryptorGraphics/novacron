//go:build !amd64

package simd

// Pure-Go fallbacks for non-amd64 arches. These are never invoked at runtime —
// the CPU-feature flags (CLMUL/AVX2/SSSE3) are false off amd64, so the callers
// always take their scalar paths — but they must exist to satisfy the linker.

func crc32CLMUL(data []byte) uint32  { return 0 }
func crc32cCLMUL(data []byte) uint32 { return 0 }

func xorBytesAVX2(dst, src1, src2 []byte)  { xorBytesScalar(dst, src1, src2) }
func xorBytesSSSE3(dst, src1, src2 []byte) { xorBytesScalar(dst, src1, src2) }
