//go:build amd64

package simd

// Assembly-accelerated primitives, implemented in the *_amd64.s files.

//go:noescape
func crc32CLMUL(data []byte) uint32

//go:noescape
func crc32cCLMUL(data []byte) uint32

//go:noescape
func xorBytesAVX2(dst, src1, src2 []byte)

//go:noescape
func xorBytesSSSE3(dst, src1, src2 []byte)
