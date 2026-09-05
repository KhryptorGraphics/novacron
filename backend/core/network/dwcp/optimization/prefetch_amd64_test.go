//go:build amd64

package optimization

import (
	"testing"
	"unsafe"
)

// TestPrefetchHintExecutes: the assembly prefetch hints must at least execute
// correctly on real amd64 hardware (smoke — no observable cache effect, but a
// bad opcode would fault and a bad frame would panic).
func TestPrefetchHintExecutes(t *testing.T) {
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i)
	}
	for i := 0; i < len(data); i += 64 {
		prefetch(unsafe.Pointer(&data[i]))
		prefetchw(unsafe.Pointer(&data[i]))
	}
	if data[123] != 123 {
		t.Fatal("data corrupted by prefetch hints")
	}
}
