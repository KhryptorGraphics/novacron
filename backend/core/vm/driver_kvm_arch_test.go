package vm

import (
	"runtime"
	"testing"
)

// TestDefaultQEMUBinarySelection verifies the arch->binary mapping. The x86
// selection is asserted here because qemu-system-x86_64 cannot be live-booted
// on this arm64 host (deferred to a CI x86 runner).
func TestDefaultQEMUBinarySelection(t *testing.T) {
	if got := defaultQEMUBinary("amd64"); got != "qemu-system-x86_64" {
		t.Fatalf("amd64: got %q, want qemu-system-x86_64", got)
	}
	if got := defaultQEMUBinary("x86_64"); got != "qemu-system-x86_64" {
		t.Fatalf("x86_64: got %q, want qemu-system-x86_64", got)
	}
	if got := defaultQEMUBinary("arm64"); got != "qemu-system-aarch64" {
		t.Fatalf("arm64: got %q, want qemu-system-aarch64", got)
	}
	if got := defaultQEMUBinary("aarch64"); got != "qemu-system-aarch64" {
		t.Fatalf("aarch64: got %q, want qemu-system-aarch64", got)
	}

	want := "qemu-system-x86_64"
	if runtime.GOARCH == "arm64" {
		want = "qemu-system-aarch64"
	}
	if got := defaultQEMUBinary(""); got != want {
		t.Fatalf("host default: got %q, want %q", got, want)
	}
}
