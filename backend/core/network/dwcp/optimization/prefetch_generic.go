//go:build !amd64

package optimization

import "unsafe"

// runtime.prefetch/prefetchw are amd64-only. On other arches a prefetch is just
// a performance hint, so no-ops are semantically correct.
func prefetch(addr unsafe.Pointer)  {}
func prefetchw(addr unsafe.Pointer) {}
