package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ContainerdDriver implements VMDriver against a real containerd daemon by
// shelling out to the `ctr` CLI (mirroring how ContainerDriver shells out to
// `docker`). Every operation talks to containerd — there is no simulation
// fallback; if the daemon is unreachable the constructor fails.
//
// Scope (deliberate, documented ceilings):
//   - CPU shares: `ctr` exposes no --cpu-shares flag (only quota/period), so
//     shares are recorded as a container label for scheduler round-trip and
//     NOT enforced in the OCI spec. Memory IS enforced via --memory-limit.
//   - Networking: `ctr` runs tasks in the host network namespace unless CNI
//     plugins are installed and --cni is passed; this driver does not wire CNI.
//   - Snapshots/migration: not supported by containerd tasks; reported false.
type ContainerdDriver struct {
	nodeID    string
	address   string
	namespace string
	// pullMu serializes image pulls so N concurrent Creates of the same image
	// don't spawn N redundant `ctr image pull` processes.
	pullMu sync.Mutex
}

// Container labels used to round-trip VMConfig fields that have no native
// containerd representation. Kept under a novacron. prefix so user Tags
// (also stored as labels) can be separated back out in GetInfo.
const (
	labelName      = "novacron.name"
	labelCPUShares = "novacron.cpu_shares"
	labelMemoryMB  = "novacron.memory_mb"
)

// NewContainerdDriver creates a new containerd driver. The daemon must be
// reachable at the configured socket; there is no simulation fallback.
func NewContainerdDriver(config map[string]interface{}) (VMDriver, error) {
	nodeID := "default-node"
	if id, ok := config["node_id"].(string); ok {
		nodeID = id
	}

	address := "/run/containerd/containerd.sock"
	if addr, ok := config["address"].(string); ok {
		address = addr
	}

	namespace := "novacron"
	if ns, ok := config["namespace"].(string); ok {
		namespace = ns
	}

	if !isContainerdReachable(address) {
		return nil, fmt.Errorf("containerd socket not reachable at %s", address)
	}

	log.Printf("Initialized containerd driver: address=%s namespace=%s", address, namespace)
	return &ContainerdDriver{
		nodeID:    nodeID,
		address:   address,
		namespace: namespace,
	}, nil
}

// isContainerdReachable checks if the containerd socket accepts connections.
func isContainerdReachable(address string) bool {
	conn, err := net.DialTimeout("unix", address, 2*time.Second)
	if err != nil {
		return false
	}
	defer conn.Close()
	return true
}

// ctr runs a ctr subcommand scoped to this driver's address and namespace,
// returning combined output. Namespaces are auto-created by containerd on
// first use, so no explicit namespace setup is needed.
func (d *ContainerdDriver) ctr(ctx context.Context, args ...string) ([]byte, error) {
	full := append([]string{"-a", d.address, "-n", d.namespace}, args...)
	cmd := exec.CommandContext(ctx, "ctr", full...)
	return cmd.CombinedOutput()
}

// normalizeImageRef turns docker-style short names ("alpine:latest") into the
// fully-qualified refs containerd requires ("docker.io/library/alpine:latest").
func normalizeImageRef(ref string) string {
	if ref == "" {
		return "docker.io/library/alpine:latest"
	}
	slash := strings.Index(ref, "/")
	if slash < 0 {
		// No registry/repo component at all: docker official image shorthand.
		return "docker.io/library/" + ref
	}
	// Has a slash; if the first component looks like a host (contains "." or
	// ":" or is "localhost") it is already qualified, otherwise assume docker.io.
	host := ref[:slash]
	if strings.ContainsAny(host, ".:") || host == "localhost" {
		return ref
	}
	return "docker.io/" + ref
}

// ensureImage pulls the image if it is not already in the namespace's store.
func (d *ContainerdDriver) ensureImage(ctx context.Context, ref string) error {
	d.pullMu.Lock()
	defer d.pullMu.Unlock()

	out, err := d.ctr(ctx, "image", "ls", "-q", "name=="+ref)
	if err == nil && strings.Contains(string(out), ref) {
		return nil // already present
	}

	log.Printf("Pulling image %s into namespace %s", ref, d.namespace)
	out, err = d.ctr(ctx, "image", "pull", ref)
	if err != nil {
		return fmt.Errorf("failed to pull image %s: %w (output: %s)", ref, err, lastLine(out))
	}
	return nil
}

// lastLine returns the last non-empty line of output for compact error messages.
func lastLine(out []byte) string {
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return ""
	}
	return lines[len(lines)-1]
}

// Create pulls the image (if needed) and creates a containerd container.
// No task exists yet, so the container is in "created" state until Start.
func (d *ContainerdDriver) Create(ctx context.Context, config VMConfig) (string, error) {
	containerID := config.ID
	if containerID == "" {
		return "", fmt.Errorf("container ID is required")
	}

	image := config.Image
	if image == "" {
		image = config.RootFS // docker driver reads the image from RootFS
	}
	ref := normalizeImageRef(image)

	if err := d.ensureImage(ctx, ref); err != nil {
		return "", err
	}

	args := []string{"container", "create"}

	// Memory limit is enforced in the OCI spec; CPU shares have no ctr flag
	// (see type comment) and ride along as a label instead.
	if config.MemoryMB > 0 {
		args = append(args, "--memory-limit", strconv.FormatInt(int64(config.MemoryMB)*1024*1024, 10))
	}
	args = append(args,
		"--label", labelName+"="+config.Name,
		"--label", labelCPUShares+"="+strconv.Itoa(config.CPUShares),
		"--label", labelMemoryMB+"="+strconv.Itoa(config.MemoryMB),
	)
	for k, v := range config.Tags {
		args = append(args, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	for k, v := range config.Env {
		args = append(args, "--env", fmt.Sprintf("%s=%s", k, v))
	}
	for _, mount := range config.Mounts {
		args = append(args, "--mount",
			fmt.Sprintf("type=bind,src=%s,dst=%s,options=rbind:rw", mount.Source, mount.Target))
	}

	args = append(args, ref, containerID)
	if config.Command != "" {
		args = append(args, config.Command)
		args = append(args, config.Args...)
	}

	if out, err := d.ctr(ctx, args...); err != nil {
		return "", fmt.Errorf("failed to create container %s: %w (output: %s)", containerID, err, lastLine(out))
	}

	log.Printf("Created containerd container %s (image %s)", containerID, ref)
	return containerID, nil
}

// containerExists reports whether the container is present in the namespace.
func (d *ContainerdDriver) containerExists(ctx context.Context, vmID string) bool {
	out, err := d.ctr(ctx, "container", "ls", "-q", "id=="+vmID)
	return err == nil && strings.Contains(string(out), vmID)
}

// taskState returns the raw task status ("RUNNING", "PAUSED", "STOPPED", ...)
// for vmID, or "" if the container has no task (i.e. created / never started).
func (d *ContainerdDriver) taskState(ctx context.Context, vmID string) (string, error) {
	out, err := d.ctr(ctx, "task", "ls")
	if err != nil {
		return "", fmt.Errorf("failed to list tasks: %w (output: %s)", err, lastLine(out))
	}
	// Output: "TASK    PID    STATUS" header, then one row per task.
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 3 && fields[0] == vmID {
			return strings.ToUpper(fields[2]), nil
		}
	}
	return "", nil
}

// Start creates and starts the container's task. A leftover STOPPED task from
// a previous run is deleted first so the container can be restarted.
func (d *ContainerdDriver) Start(ctx context.Context, vmID string) error {
	if !d.containerExists(ctx, vmID) {
		return fmt.Errorf("container %s not found", vmID)
	}

	state, err := d.taskState(ctx, vmID)
	if err != nil {
		return err
	}
	switch state {
	case "RUNNING":
		return nil // already running
	case "STOPPED":
		// Clear the exited task so `task start` can create a fresh one.
		if out, err := d.ctr(ctx, "task", "delete", vmID); err != nil {
			return fmt.Errorf("failed to remove stopped task for %s: %w (output: %s)", vmID, err, lastLine(out))
		}
	}

	if out, err := d.ctr(ctx, "task", "start", "-d", vmID); err != nil {
		return fmt.Errorf("failed to start container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	log.Printf("Started containerd container %s", vmID)
	return nil
}

// Stop sends SIGTERM to the task and escalates to SIGKILL if it does not exit.
// The exited task is left in place (STOPPED) so GetStatus reports StateStopped;
// it is cleaned up by Delete (or by Start on restart).
func (d *ContainerdDriver) Stop(ctx context.Context, vmID string) error {
	if !d.containerExists(ctx, vmID) {
		return fmt.Errorf("container %s not found", vmID)
	}

	state, err := d.taskState(ctx, vmID)
	if err != nil {
		return err
	}
	if state == "" || state == "STOPPED" {
		return nil // never started, or already exited
	}

	if out, err := d.ctr(ctx, "task", "kill", "-s", "SIGTERM", vmID); err != nil {
		return fmt.Errorf("failed to signal container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	if d.waitTaskStopped(ctx, vmID, 10*time.Second) {
		return nil
	}

	log.Printf("Container %s did not exit on SIGTERM, escalating to SIGKILL", vmID)
	if out, err := d.ctr(ctx, "task", "kill", "-s", "SIGKILL", vmID); err != nil {
		return fmt.Errorf("failed to SIGKILL container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	if !d.waitTaskStopped(ctx, vmID, 10*time.Second) {
		return fmt.Errorf("container %s still not stopped after SIGKILL", vmID)
	}
	return nil
}

// waitTaskStopped polls until the task reports STOPPED (or vanishes).
func (d *ContainerdDriver) waitTaskStopped(ctx context.Context, vmID string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		state, err := d.taskState(ctx, vmID)
		if err == nil && (state == "" || state == "STOPPED") {
			return true
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false
}

// Delete force-removes the task (if any) and then the container + snapshot.
func (d *ContainerdDriver) Delete(ctx context.Context, vmID string) error {
	if !d.containerExists(ctx, vmID) {
		return fmt.Errorf("container %s not found", vmID)
	}

	// Force-delete any task first (running or stopped); ignore "no task" errors.
	_, _ = d.ctr(ctx, "task", "delete", "-f", vmID)

	if out, err := d.ctr(ctx, "container", "delete", vmID); err != nil {
		return fmt.Errorf("failed to delete container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	log.Printf("Deleted containerd container %s", vmID)
	return nil
}

// GetStatus maps container/task state onto the VM State enum.
func (d *ContainerdDriver) GetStatus(ctx context.Context, vmID string) (State, error) {
	if !d.containerExists(ctx, vmID) {
		return StateUnknown, fmt.Errorf("container %s not found", vmID)
	}

	state, err := d.taskState(ctx, vmID)
	if err != nil {
		return StateUnknown, err
	}
	switch state {
	case "":
		return StateCreated, nil // container exists, no task yet
	case "CREATED":
		return StateCreated, nil
	case "RUNNING":
		return StateRunning, nil
	case "PAUSED", "PAUSING":
		return StatePaused, nil
	case "STOPPED":
		return StateStopped, nil
	default:
		return StateUnknown, nil
	}
}

// containerInfoJSON is the subset of `ctr container info` output we consume.
type containerInfoJSON struct {
	ID        string            `json:"ID"`
	Labels    map[string]string `json:"Labels"`
	Image     string            `json:"Image"`
	CreatedAt time.Time         `json:"CreatedAt"`
}

// GetInfo returns container metadata plus (when running) memory usage read
// from `ctr task metrics`.
func (d *ContainerdDriver) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	status, err := d.GetStatus(ctx, vmID)
	if err != nil {
		return nil, err
	}

	out, err := d.ctr(ctx, "container", "info", vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get container info for %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	var info containerInfoJSON
	if err := json.Unmarshal(out, &info); err != nil {
		return nil, fmt.Errorf("failed to parse container info for %s: %w", vmID, err)
	}

	vmInfo := &VMInfo{
		ID:        vmID,
		Name:      vmID,
		State:     status,
		CreatedAt: info.CreatedAt,
		Image:     info.Image,
	}

	// Unpack novacron.* labels back into typed fields; everything else is Tags.
	tags := make(map[string]string)
	for k, v := range info.Labels {
		switch k {
		case labelName:
			if v != "" {
				vmInfo.Name = v
			}
		case labelCPUShares:
			if shares, err := strconv.Atoi(v); err == nil {
				vmInfo.CPUShares = shares
			}
		case labelMemoryMB:
			if mb, err := strconv.Atoi(v); err == nil {
				vmInfo.MemoryMB = mb
			}
		default:
			tags[k] = v
		}
	}
	if len(tags) > 0 {
		vmInfo.Tags = tags
	}

	if status == StateRunning {
		d.fillMemoryUsage(ctx, vmID, vmInfo)
	}
	return vmInfo, nil
}

// fillMemoryUsage best-effort parses `ctr task metrics` (cgroup v1 or v2 key
// names) into VMInfo.MemoryUsage. CPU percent needs two samples to compute, so
// CPUUsage is deliberately left 0 rather than reporting a made-up number.
func (d *ContainerdDriver) fillMemoryUsage(ctx context.Context, vmID string, vmInfo *VMInfo) {
	out, err := d.ctr(ctx, "task", "metrics", vmID)
	if err != nil {
		return // metrics are best-effort
	}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		if fields[0] == "memory.usage_in_bytes" || fields[0] == "memory.current" {
			if bytes, err := strconv.ParseInt(fields[1], 10, 64); err == nil {
				vmInfo.MemoryUsage = bytes
			}
			return
		}
	}
}

// GetMetrics returns the same data as GetInfo (which includes live memory
// usage for running containers) — mirroring the docker driver's behavior.
func (d *ContainerdDriver) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	return d.GetInfo(ctx, vmID)
}

// ListVMs lists containers in the namespace with their task-derived state.
func (d *ContainerdDriver) ListVMs(ctx context.Context) ([]VMInfo, error) {
	out, err := d.ctr(ctx, "container", "ls", "-q")
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w (output: %s)", err, lastLine(out))
	}

	// One `task ls` for the whole namespace instead of a query per container.
	states := make(map[string]string)
	if taskOut, err := d.ctr(ctx, "task", "ls"); err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(taskOut)), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 3 && fields[0] != "TASK" {
				states[fields[0]] = strings.ToUpper(fields[2])
			}
		}
	}

	var vms []VMInfo
	for _, id := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if id == "" {
			continue
		}
		state := StateCreated
		switch states[id] {
		case "RUNNING":
			state = StateRunning
		case "PAUSED", "PAUSING":
			state = StatePaused
		case "STOPPED":
			state = StateStopped
		}
		vms = append(vms, VMInfo{ID: id, Name: id, State: state})
	}
	return vms, nil
}

// SupportsPause returns whether the driver supports pausing VMs
func (d *ContainerdDriver) SupportsPause() bool { return true }

// SupportsResume returns whether the driver supports resuming VMs
func (d *ContainerdDriver) SupportsResume() bool { return true }

// SupportsSnapshot returns whether the driver supports snapshots
func (d *ContainerdDriver) SupportsSnapshot() bool { return false }

// SupportsMigrate returns whether the driver supports migration
func (d *ContainerdDriver) SupportsMigrate() bool { return false }

// Pause freezes the container's task via the cgroup freezer.
func (d *ContainerdDriver) Pause(ctx context.Context, vmID string) error {
	if !d.containerExists(ctx, vmID) {
		return fmt.Errorf("container %s not found", vmID)
	}
	if out, err := d.ctr(ctx, "task", "pause", vmID); err != nil {
		return fmt.Errorf("failed to pause container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	log.Printf("Paused containerd container %s", vmID)
	return nil
}

// Resume thaws a paused task.
func (d *ContainerdDriver) Resume(ctx context.Context, vmID string) error {
	if !d.containerExists(ctx, vmID) {
		return fmt.Errorf("container %s not found", vmID)
	}
	if out, err := d.ctr(ctx, "task", "resume", vmID); err != nil {
		return fmt.Errorf("failed to resume container %s: %w (output: %s)", vmID, err, lastLine(out))
	}
	log.Printf("Resumed containerd container %s", vmID)
	return nil
}

// Snapshot creates a snapshot of a containerd container VM (not supported).
func (d *ContainerdDriver) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	return "", fmt.Errorf("snapshots not supported by containerd driver")
}

// Migrate migrates a containerd container VM (not supported).
func (d *ContainerdDriver) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	return fmt.Errorf("migration not supported by containerd driver")
}

// SupportsLiveMigration returns whether the driver supports live migration
func (d *ContainerdDriver) SupportsLiveMigration() bool { return false }

// SupportsHotPlug returns whether the driver supports hot-plugging devices
func (d *ContainerdDriver) SupportsHotPlug() bool { return false }

// SupportsGPUPassthrough returns whether the driver supports GPU passthrough
func (d *ContainerdDriver) SupportsGPUPassthrough() bool { return false }

// SupportsSRIOV returns whether the driver supports SR-IOV
func (d *ContainerdDriver) SupportsSRIOV() bool { return false }

// SupportsNUMA returns whether the driver supports NUMA configuration
func (d *ContainerdDriver) SupportsNUMA() bool { return false }

// serverVersion returns the containerd daemon version ("v2.2.5") or "unknown".
func (d *ContainerdDriver) serverVersion(ctx context.Context) string {
	out, err := d.ctr(ctx, "version")
	if err != nil {
		return "unknown"
	}
	// Output has Client: and Server: sections, each with a "  Version:  vX.Y.Z" line.
	lines := strings.Split(string(out), "\n")
	inServer := false
	for _, line := range lines {
		if strings.HasPrefix(line, "Server:") {
			inServer = true
			continue
		}
		if inServer && strings.Contains(line, "Version:") {
			fields := strings.Fields(line)
			return fields[len(fields)-1]
		}
	}
	return "unknown"
}

// GetCapabilities returns the capabilities of the containerd driver.
func (d *ContainerdDriver) GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error) {
	return &HypervisorCapabilities{
		Type:                   VMTypeContainerd,
		Version:                d.serverVersion(ctx),
		SupportsPause:          d.SupportsPause(),
		SupportsResume:         d.SupportsResume(),
		SupportsSnapshot:       d.SupportsSnapshot(),
		SupportsMigrate:        d.SupportsMigrate(),
		SupportsLiveMigration:  d.SupportsLiveMigration(),
		SupportsHotPlug:        d.SupportsHotPlug(),
		SupportsGPUPassthrough: d.SupportsGPUPassthrough(),
		SupportsSRIOV:          d.SupportsSRIOV(),
		SupportsNUMA:           d.SupportsNUMA(),
		MaxVCPUs:               1024,
		MaxMemoryMB:            1024 * 1024, // 1TB
		SupportedFeatures:      []string{"pause", "resume", "memory_limits", "labels", "bind_mounts"},
		HardwareExtensions:     []string{},
	}, nil
}

// GetHypervisorInfo returns information about the containerd runtime.
func (d *ContainerdDriver) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) {
	capabilities, err := d.GetCapabilities(ctx)
	if err != nil {
		return nil, err
	}

	activeVMs := 0
	if vms, err := d.ListVMs(ctx); err == nil {
		activeVMs = len(vms)
	}

	return &HypervisorInfo{
		Type:           VMTypeContainerd,
		Version:        capabilities.Version,
		ConnectionURI:  d.address,
		Hostname:       d.nodeID,
		CPUModel:       "Container",
		Virtualization: "Container",
		IOMMUEnabled:   false,
		NUMANodes:      1,
		GPUDevices:     []GPUDevice{},
		NetworkDevices: []NetworkDevice{},
		StorageDevices: []StorageDevice{},
		ActiveVMs:      activeVMs,
		Capabilities:   capabilities,
		Metadata: map[string]interface{}{
			"runtime":   "containerd",
			"namespace": d.namespace,
			"address":   d.address,
		},
	}, nil
}

// HotPlugDevice hot-plugs a device (not supported for containers).
func (d *ContainerdDriver) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error {
	return fmt.Errorf("hot-plug not supported by containerd driver")
}

// HotUnplugDevice hot-unplugs a device (not supported for containers).
func (d *ContainerdDriver) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error {
	return fmt.Errorf("hot-unplug not supported by containerd driver")
}

// ConfigureCPUPinning configures CPU pinning (not supported for containers).
func (d *ContainerdDriver) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error {
	return fmt.Errorf("CPU pinning not supported by containerd driver")
}

// ConfigureNUMA configures NUMA topology (not supported for containers).
func (d *ContainerdDriver) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error {
	return fmt.Errorf("NUMA configuration not supported by containerd driver")
}
