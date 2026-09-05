package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestUbuntu2404ProfilePrepareConfig(t *testing.T) {
	t.Setenv("PATH", installUbuntu2404TestTools(t))

	baseDir := t.TempDir()
	baseImagePath := filepath.Join(baseDir, "ubuntu-24.04-base.qcow2")
	if err := os.WriteFile(baseImagePath, []byte("base-image"), 0o644); err != nil {
		t.Fatalf("write base image: %v", err)
	}

	profile, err := vm.NewUbuntu2404Profile(baseImagePath, filepath.Join(baseDir, "cloud-init"))
	if err != nil {
		t.Fatalf("NewUbuntu2404Profile returned error: %v", err)
	}

	config, err := profile.PrepareConfig("vm-ubuntu-2404", vm.VMConfig{Name: "ubuntu-test"}, vm.Ubuntu2404CloudInitOptions{
		Hostname:          "ubuntu-test",
		SSHAuthorizedKeys: []string{"ssh-ed25519 AAAATEST user@test"},
	})
	if err != nil {
		t.Fatalf("PrepareConfig returned error: %v", err)
	}

	if got, want := config.Type, vm.VMTypeKVM; got != want {
		t.Fatalf("vm type = %q, want %q", got, want)
	}
	if got, want := config.Image, baseImagePath; got != want {
		t.Fatalf("image = %q, want %q", got, want)
	}
	if config.CloudInitISO == "" {
		t.Fatal("cloud-init iso path is empty")
	}
	if _, err := os.Stat(config.CloudInitISO); err != nil {
		t.Fatalf("cloud-init iso not created: %v", err)
	}

	userDataPath := filepath.Join(baseDir, "cloud-init", "vm-ubuntu-2404", "user-data")
	userData, err := os.ReadFile(userDataPath)
	if err != nil {
		t.Fatalf("read user-data: %v", err)
	}
	userDataText := string(userData)
	if !strings.Contains(userDataText, "hostname: ubuntu-test") {
		t.Fatalf("user-data missing hostname: %s", userDataText)
	}
	if !strings.Contains(userDataText, "ssh-ed25519 AAAATEST user@test") {
		t.Fatalf("user-data missing ssh key: %s", userDataText)
	}
}

func TestVMManagerCreateUbuntu2404VMUsesBaseImageAndCloudInitISO(t *testing.T) {
	t.Setenv("PATH", installUbuntu2404TestTools(t))

	baseDir := t.TempDir()
	baseImagePath := filepath.Join(baseDir, "ubuntu-24.04-base.qcow2")
	if err := os.WriteFile(baseImagePath, []byte("base-image"), 0o644); err != nil {
		t.Fatalf("write base image: %v", err)
	}

	// The QMP socket lives under the VM runtime dir; unix socket paths cap at
	// 107 bytes and t.TempDir() nests under the long test name, so give the
	// driver a short, test-scoped vm_path instead.
	vmDir, err := os.MkdirTemp("/tmp", "nvu24")
	if err != nil {
		t.Fatalf("create short vm_path: %v", err)
	}
	defer os.RemoveAll(vmDir)

	managerConfig := vm.DefaultVMManagerConfig()
	managerConfig.Drivers[vm.VMTypeKVM] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"node_id":   "ubuntu-test-node",
			"qemu_path": "qemu-system-x86_64",
			"vm_path":   vmDir,
		},
	}

	manager, err := vm.NewVMManager(managerConfig)
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	profile, err := vm.NewUbuntu2404Profile(baseImagePath, filepath.Join(baseDir, "cloud-init"))
	if err != nil {
		t.Fatalf("NewUbuntu2404Profile returned error: %v", err)
	}

	vmInstance, err := manager.CreateUbuntu2404VM(context.Background(), vm.CreateVMRequest{
		Name: "ubuntu-24-04-test",
		Spec: vm.VMConfig{
			OwnerID:    "owner-a",
			TenantID:   "tenant-a",
			CPUShares:  2,
			MemoryMB:   1024,
			DiskSizeGB: 20,
		},
		Tags: map[string]string{
			"purpose": "ubuntu-profile-test",
		},
	}, profile, vm.Ubuntu2404CloudInitOptions{
		Hostname: "ubuntu-24-04-test",
	})
	if err != nil {
		t.Fatalf("CreateUbuntu2404VM returned error: %v", err)
	}

	config := vmInstance.Config()
	if got, want := config.Type, vm.VMTypeKVM; got != want {
		t.Fatalf("vm type = %q, want %q", got, want)
	}
	if got, want := config.Image, baseImagePath; got != want {
		t.Fatalf("image = %q, want %q", got, want)
	}
	if config.CloudInitISO == "" {
		t.Fatal("cloud-init iso path is empty")
	}
	if _, err := os.Stat(config.CloudInitISO); err != nil {
		t.Fatalf("cloud-init iso not created: %v", err)
	}
	if got, want := config.Tags["purpose"], "ubuntu-profile-test"; got != want {
		t.Fatalf("purpose tag = %q, want %q", got, want)
	}
	if got, want := config.Tags["version"], "24.04"; got != want {
		t.Fatalf("version tag = %q, want %q", got, want)
	}

	if err := manager.StartVM(context.Background(), config.ID); err != nil {
		t.Fatalf("StartVM returned error: %v", err)
	}

	time.Sleep(50 * time.Millisecond)

	qemuImgArgs, err := os.ReadFile(os.Getenv("NOVACRON_TEST_QEMU_IMG_LOG"))
	if err != nil {
		t.Fatalf("read qemu-img args: %v", err)
	}
	qemuImgArgsText := string(qemuImgArgs)
	if !strings.Contains(qemuImgArgsText, "-b") || !strings.Contains(qemuImgArgsText, baseImagePath) {
		t.Fatalf("qemu-img args missing base image: %s", qemuImgArgsText)
	}

	qemuSystemArgs, err := os.ReadFile(os.Getenv("NOVACRON_TEST_QEMU_SYSTEM_LOG"))
	if err != nil {
		t.Fatalf("read qemu-system args: %v", err)
	}
	qemuSystemArgsText := string(qemuSystemArgs)
	if !strings.Contains(qemuSystemArgsText, config.CloudInitISO) {
		t.Fatalf("qemu-system args missing cloud-init iso %q: %s", config.CloudInitISO, qemuSystemArgsText)
	}
}

func installUbuntu2404TestTools(t *testing.T) string {
	t.Helper()

	toolsDir := filepath.Join(t.TempDir(), "bin")
	if err := os.MkdirAll(toolsDir, 0o755); err != nil {
		t.Fatalf("create tools dir: %v", err)
	}

	qemuImgArgsPath := filepath.Join(t.TempDir(), "qemu-img-args.txt")
	cloudLocalDSArgsPath := filepath.Join(t.TempDir(), "cloud-localds-args.txt")
	qemuSystemArgsPath := filepath.Join(t.TempDir(), "qemu-system-args.txt")

	t.Setenv("NOVACRON_TEST_QEMU_IMG_LOG", qemuImgArgsPath)
	t.Setenv("NOVACRON_TEST_CLOUD_LOCALDS_LOG", cloudLocalDSArgsPath)
	t.Setenv("NOVACRON_TEST_QEMU_SYSTEM_LOG", qemuSystemArgsPath)

	// The fake qemu must honor the driver's launch contract: a real qemu
	// opens its QMP unix socket ("-qmp unix:<path>,server,nowait") and stays
	// alive, which is exactly what the driver's waitQMPUp liveness probe
	// checks. The old log-and-exit stub failed that probe after the liveness
	// check landed. Parse the -qmp argument, hold the socket open with nc
	// until a fifo writer closes (3 s), then exit — the driver sees a live
	// qemu that came up and shut down cleanly after the probe window.
	writeExecutable(t, filepath.Join(toolsDir, "qemu-system-x86_64"), `#!/bin/sh
printf '%s\n' "$@" > "$NOVACRON_TEST_QEMU_SYSTEM_LOG"
qmp_path=""
prev=""
for arg in "$@"; do
  case "$prev" in
    -qmp) qmp_path="$arg" ;;
  esac
  prev="$arg"
done
if [ -n "$qmp_path" ]; then
  sock="${qmp_path#unix:}"
  sock="${sock%%,*}"
  fifo="$sock.stub-ctl"
  mkfifo "$fifo" 2>/dev/null
  (sleep 3 > "$fifo" 2>/dev/null &)
  nc -U -l "$sock" < "$fifo" >/dev/null 2>&1
  rm -f "$fifo"
fi
exit 0
`)
	writeExecutable(t, filepath.Join(toolsDir, "cloud-localds"), "#!/bin/sh\noutput=\"$1\"\nshift\nprintf '%s\\n' \"$@\" > \"$NOVACRON_TEST_CLOUD_LOCALDS_LOG\"\n: > \"$output\"\n")
	// The disk-create path issues multiple qemu-img invocations (convert from
	// base, then resize); append every invocation's args so the assertion can
	// see the base-image convert, and touch the size-detected output file.
	writeExecutable(t, filepath.Join(toolsDir, "qemu-img"), "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$NOVACRON_TEST_QEMU_IMG_LOG\"\nprev=''\nlast=''\nfor arg in \"$@\"; do\n  prev=\"$last\"\n  last=\"$arg\"\ndone\ncase \"$last\" in\n  *K|*M|*G|*T) output=\"$prev\" ;;\n  *) output=\"$last\" ;;\nesac\n: > \"$output\"\n")

	return toolsDir + string(os.PathListSeparator) + os.Getenv("PATH")
}

func writeExecutable(t *testing.T, path, contents string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(contents), 0o755); err != nil {
		t.Fatalf("write executable %s: %v", path, err)
	}
}
