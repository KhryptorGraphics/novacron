package vm

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"golang.org/x/sys/unix"
)

// EBPFCapability represents the eBPF support level
type EBPFCapability struct {
	Supported         bool
	KernelVersion     string
	HasBPFSyscall     bool
	HasBPFHelpers     bool
	HasBTF            bool
	HasCORESupport    bool
	CanLoadPrograms   bool
	CanAttachKprobes  bool
	CanAttachTracing  bool
	ErrorMessage      string
	RequiredKernelMin string
}

const (
	// MinKernelVersion is the minimum kernel version required for eBPF support
	MinKernelVersion = "4.18"

	// RecommendedKernelVersion is the recommended kernel version for full eBPF support
	RecommendedKernelVersion = "5.8"
)

// IsEBPFSupported checks if eBPF is supported on the current system
func IsEBPFSupported() bool {
	cap := CheckEBPFCapability()
	return cap.Supported
}

// CheckEBPFCapability performs a comprehensive check of eBPF capabilities
func CheckEBPFCapability() *EBPFCapability {
	cap := &EBPFCapability{
		Supported:         false,
		RequiredKernelMin: MinKernelVersion,
	}

	// Check if we're on Linux
	if runtime.GOOS != "linux" {
		cap.ErrorMessage = "eBPF is only supported on Linux"
		return cap
	}

	// Check kernel version
	kernelVersion, err := getKernelVersion()
	if err != nil {
		cap.ErrorMessage = fmt.Sprintf("failed to get kernel version: %v", err)
		return cap
	}
	cap.KernelVersion = kernelVersion

	// Check if kernel version meets minimum requirements
	if !isKernelVersionSupported(kernelVersion) {
		cap.ErrorMessage = fmt.Sprintf("kernel version %s is below minimum required %s", kernelVersion, MinKernelVersion)
		return cap
	}

	// Check if BPF syscall is available
	cap.HasBPFSyscall = checkBPFSyscall()
	if !cap.HasBPFSyscall {
		cap.ErrorMessage = "BPF syscall not available"
		return cap
	}

	// Check if we have necessary capabilities (CAP_BPF or CAP_SYS_ADMIN)
	if !checkBPFCapabilities() {
		cap.ErrorMessage = "insufficient capabilities (need CAP_BPF or CAP_SYS_ADMIN)"
		return cap
	}

	// Try to load a simple eBPF program to verify we can actually use eBPF
	canLoad, err := tryLoadSimpleProgram()
	cap.CanLoadPrograms = canLoad
	if !canLoad {
		cap.ErrorMessage = fmt.Sprintf("cannot load eBPF programs: %v", err)
		return cap
	}

	// Check for BTF support
	cap.HasBTF = checkBTFSupport()

	// Check for CO-RE support
	cap.HasCORESupport = checkCORESupport()

	// Check if we can attach kprobes
	cap.CanAttachKprobes = checkKprobeSupport()

	// Check if we can attach tracing programs
	cap.CanAttachTracing = checkTracingSupport()

	// If we got here, eBPF is supported
	cap.Supported = true

	return cap
}

// getKernelVersion retrieves the kernel version string
func getKernelVersion() (string, error) {
	var utsname unix.Utsname
	if err := unix.Uname(&utsname); err != nil {
		return "", err
	}

	// Convert [65]int8 to string
	release := make([]byte, 0, 65)
	for _, b := range utsname.Release {
		if b == 0 {
			break
		}
		release = append(release, byte(b))
	}

	return string(release), nil
}

// isKernelVersionSupported checks if the kernel version meets minimum requirements
func isKernelVersionSupported(version string) bool {
	// Extract major.minor version
	parts := strings.Split(version, ".")
	if len(parts) < 2 {
		return false
	}

	var major, minor int
	fmt.Sscanf(parts[0], "%d", &major)
	fmt.Sscanf(parts[1], "%d", &minor)

	// Require at least kernel 4.18
	if major > 4 {
		return true
	}
	if major == 4 && minor >= 18 {
		return true
	}

	return false
}

// checkBPFSyscall checks if the BPF syscall is available
func checkBPFSyscall() bool {
	// Try to make a simple BPF syscall
	// BPF_MAP_CREATE with invalid parameters should fail with EINVAL, not ENOSYS
	attr := &unix.BPFMapCreateAttr{
		MapType:    unix.BPF_MAP_TYPE_ARRAY,
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
	}

	_, err := unix.BPFMapCreate(attr)
	if err == nil {
		return true
	}

	// If we get ENOSYS, the syscall doesn't exist
	if errors.Is(err, unix.ENOSYS) {
		return false
	}

	// Any other error (like EINVAL, EPERM) means the syscall exists
	return true
}

// checkBPFCapabilities checks if we have the necessary capabilities
func checkBPFCapabilities() bool {
	// Try to check for CAP_BPF (kernel 5.8+) or CAP_SYS_ADMIN
	var hdr unix.CapUserHeader
	var data [2]unix.CapUserData

	hdr.Version = unix.LINUX_CAPABILITY_VERSION_3
	hdr.Pid = 0 // Current process

	if err := unix.Capget(&hdr, &data[0]); err != nil {
		// Can't check capabilities, assume we don't have them
		return false
	}

	// Check for CAP_SYS_ADMIN (bit 21)
	if (data[0].Effective & (1 << unix.CAP_SYS_ADMIN)) != 0 {
		return true
	}

	// Check for CAP_BPF (bit 39, in data[1] on 64-bit systems)
	// CAP_BPF = 39, so it's bit 7 in data[1] (39 - 32 = 7)
	const capBPF = 39
	if capBPF < 32 {
		if (data[0].Effective & (1 << capBPF)) != 0 {
			return true
		}
	} else {
		if (data[1].Effective & (1 << (capBPF - 32))) != 0 {
			return true
		}
	}

	return false
}

// tryLoadSimpleProgram tries to load a minimal eBPF program
func tryLoadSimpleProgram() (bool, error) {
	// Create a minimal valid eBPF program
	prog := &ebpf.ProgramSpec{
		Type: ebpf.SocketFilter,
		Instructions: asm.Instructions{
			// mov r0, 0
			asm.Mov.Imm(asm.R0, 0),
			// exit
			asm.Return(),
		},
		License: "GPL",
	}

	// Try to load the program
	p, err := ebpf.NewProgram(prog)
	if err != nil {
		return false, err
	}
	defer p.Close()

	return true, nil
}

// checkBTFSupport checks if BTF (BPF Type Format) is supported
func checkBTFSupport() bool {
	// Check if /sys/kernel/btf/vmlinux exists
	_, err := os.Stat("/sys/kernel/btf/vmlinux")
	return err == nil
}

// checkCORESupport checks if CO-RE (Compile Once Run Everywhere) is supported
func checkCORESupport() bool {
	// CO-RE requires BTF support
	return checkBTFSupport()
}

// checkKprobeSupport checks if we can attach kprobes
func checkKprobeSupport() bool {
	// Check if /sys/kernel/debug/tracing/kprobe_events exists
	_, err := os.Stat("/sys/kernel/debug/tracing/kprobe_events")
	if err != nil {
		// Try alternative path
		_, err = os.Stat("/sys/kernel/tracing/kprobe_events")
	}
	return err == nil
}

// checkTracingSupport checks if we can attach tracing programs
func checkTracingSupport() bool {
	// Check if /sys/kernel/debug/tracing/events exists
	_, err := os.Stat("/sys/kernel/debug/tracing/events")
	if err != nil {
		// Try alternative path
		_, err = os.Stat("/sys/kernel/tracing/events")
	}
	return err == nil
}

// GetEBPFDiagnostics returns detailed diagnostics about eBPF support
func GetEBPFDiagnostics() string {
	cap := CheckEBPFCapability()

	var sb strings.Builder
	sb.WriteString("eBPF Capability Diagnostics:\n")
	sb.WriteString(fmt.Sprintf("  Supported: %v\n", cap.Supported))
	sb.WriteString(fmt.Sprintf("  Kernel Version: %s\n", cap.KernelVersion))
	sb.WriteString(fmt.Sprintf("  Required Minimum: %s\n", cap.RequiredKernelMin))
	sb.WriteString(fmt.Sprintf("  Has BPF Syscall: %v\n", cap.HasBPFSyscall))
	sb.WriteString(fmt.Sprintf("  Has BPF Helpers: %v\n", cap.HasBPFHelpers))
	sb.WriteString(fmt.Sprintf("  Has BTF: %v\n", cap.HasBTF))
	sb.WriteString(fmt.Sprintf("  Has CO-RE Support: %v\n", cap.HasCORESupport))
	sb.WriteString(fmt.Sprintf("  Can Load Programs: %v\n", cap.CanLoadPrograms))
	sb.WriteString(fmt.Sprintf("  Can Attach Kprobes: %v\n", cap.CanAttachKprobes))
	sb.WriteString(fmt.Sprintf("  Can Attach Tracing: %v\n", cap.CanAttachTracing))

	if !cap.Supported && cap.ErrorMessage != "" {
		sb.WriteString(fmt.Sprintf("  Error: %s\n", cap.ErrorMessage))
	}

	return sb.String()
}
