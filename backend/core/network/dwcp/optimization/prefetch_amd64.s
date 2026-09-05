// prefetch and prefetchw are the hardware cache hints (PREFETCHT0/PREFETCHW).
// runtime.prefetch/prefetchw do not exist as linkable symbols in the Go 1.25
// runtime (the runtime's own hint is the internal/runtime/sys.Prefetch compiler
// intrinsic, not a linkable symbol), so implement the hints directly.

#include "textflag.h"

// func prefetch(addr unsafe.Pointer)
TEXT ·prefetch(SB), NOSPLIT, $0-8
	MOVQ addr+0(FP), AX
	PREFETCHT0 (AX)
	RET

// func prefetchw(addr unsafe.Pointer)
TEXT ·prefetchw(SB), NOSPLIT, $0-8
	MOVQ addr+0(FP), AX
	BYTE $0x0F; BYTE $0x18; BYTE $0x30
	RET
