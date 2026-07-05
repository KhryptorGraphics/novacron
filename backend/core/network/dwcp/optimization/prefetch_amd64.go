//go:build amd64

package optimization

import "unsafe"

//go:linkname prefetch runtime.prefetch
func prefetch(addr unsafe.Pointer)

//go:linkname prefetchw runtime.prefetchw
func prefetchw(addr unsafe.Pointer)
