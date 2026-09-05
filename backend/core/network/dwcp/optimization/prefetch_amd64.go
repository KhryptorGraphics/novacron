//go:build amd64

package optimization

import "unsafe"

// prefetch and prefetchw are the hardware cache hints (PREFETCHT0/PREFETCHW),
// implemented in prefetch_amd64.s. runtime.prefetch/prefetchw do not exist as
// linkable symbols in the Go 1.25 runtime — the pre-fix //go:linkname to
// runtime.prefetch produced "relocation target runtime.prefetch not defined"
// at link time (reproduced locally with GOARCH=amd64 on Go 1.25.x).
func prefetch(addr unsafe.Pointer)
func prefetchw(addr unsafe.Pointer)
