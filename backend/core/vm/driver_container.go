package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ContainerDriver implements the VMDriver interface for container-based VMs
type ContainerDriver struct {
	nodeID string
}

// NewContainerDriver creates a new container driver
func NewContainerDriver(config map[string]interface{}) (VMDriver, error) {
	nodeID := ""
	if id, ok := config["node_id"].(string); ok {
		nodeID = id
	}
	
	return &ContainerDriver{
		nodeID: nodeID,
	}, nil
}

// Create creates a new container VM
func (d *ContainerDriver) Create(ctx context.Context, config VMConfig) (string, error) {
	log.Printf("Creating container VM %s", config.Name)
	
	// Generate a container name based on VM config
	containerName := fmt.Sprintf("novacron-%s-%s", config.Name, strconv.FormatInt(time.Now().UnixNano(), 16))
	
	// Build the docker command to create a container
	args := []string{
		"create",
		"--name", containerName,
	}
	
	// Set resource limits
	if config.CPUShares > 0 {
		args = append(args, "--cpu-shares", fmt.Sprintf("%d", config.CPUShares))
	}
	
	if config.MemoryMB > 0 {
		args = append(args, "--memory", fmt.Sprintf("%dm", config.MemoryMB))
	}
	
	// Set environment variables
	for k, v := range config.Env {
		args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Set labels from tags
	for k, v := range config.Tags {
		args = append(args, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Mount volumes
	for _, mount := range config.Mounts {
		args = append(args, "-v", fmt.Sprintf("%s:%s", mount.Source, mount.Target))
	}
	
	// Configure network
	if config.NetworkID != "" {
		args = append(args, "--network", config.NetworkID)
	}
	
	// Add the command as the last argument (assuming RootFS contains the image name)
	image := config.RootFS
	if image == "" {
		image = "alpine:latest" // Default image
	}
	args = append(args, image)
	
	// Add the command if specified
	if config.Command != "" {
		args = append(args, config.Command)
		args = append(args, config.Args...)
	}
	
	// Run the command
	cmd := exec.CommandContext(ctx, "docker", args...)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to create container: %v, output: %s", err, string(output))
		return "", fmt.Errorf("failed to create container: %w", err)
	}
	
	// Return the container ID as the VM ID
	return containerName, nil
}

// Start starts a container VM
func (d *ContainerDriver) Start(ctx context.Context, vmID string) error {
	log.Printf("Starting container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "start", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to start container %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to start container: %w", err)
	}
	
	return nil
}

// Stop stops a container VM
func (d *ContainerDriver) Stop(ctx context.Context, vmID string) error {
	log.Printf("Stopping container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "stop", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to stop container %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to stop container: %w", err)
	}
	
	return nil
}

// Delete deletes a container VM
func (d *ContainerDriver) Delete(ctx context.Context, vmID string) error {
	log.Printf("Deleting container VM %s", vmID)
	
	// Stop the container first to ensure it's not running
	_ = d.Stop(ctx, vmID) // Ignore error, it might already be stopped
	
	cmd := exec.CommandContext(ctx, "docker", "rm", "-f", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to delete container %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to delete container: %w", err)
	}
	
	return nil
}

// GetStatus gets the status of a container VM
func (d *ContainerDriver) GetStatus(ctx context.Context, vmID string) (State, error) {
	log.Printf("Getting status of container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Status}}", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		if strings.Contains(string(output), "No such container") {
			return StateUnknown, errors.New("container not found")
		}
		log.Printf("Failed to get status of container %s: %v, output: %s", vmID, err, string(output))
		return StateUnknown, fmt.Errorf("failed to get container status: %w", err)
	}
	
	status := strings.TrimSpace(string(output))
	
	switch status {
	case "running":
		return StateRunning, nil
	case "exited":
		return StateStopped, nil
	case "created":
		return StateStopped, nil
	case "paused":
		return StatePaused, nil
	case "restarting":
		return StateRestarting, nil
	default:
		return StateUnknown, nil
	}
}

// GetInfo gets information about a container VM
func (d *ContainerDriver) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	log.Printf("Getting info of container VM %s", vmID)
	
	// First check if the container exists
	status, err := d.GetStatus(ctx, vmID)
	if err != nil {
		return nil, err
	}
	
	// Get detailed container info
	inspectFormat := `{
		"id": "{{.Id}}",
		"name": "{{.Name}}",
		"image": "{{.Config.Image}}",
		"created": "{{.Created}}",
		"status": "{{.State.Status}}",
		"pid": {{.State.Pid}},
		"memory": {{if .State.MemoryStats}}{{.State.MemoryStats.Usage}}{{else}}0{{end}},
		"cpu": {{if .State.CPUStats}}{{.State.CPUStats.CPUUsage.TotalUsage}}{{else}}0{{end}},
		"startTime": "{{.State.StartedAt}}",
		"networks": {{json .NetworkSettings.Networks}}
	}`
	
	cmd := exec.CommandContext(ctx, "docker", "inspect", "-f", inspectFormat, vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to get info of container %s: %v, output: %s", vmID, err, string(output))
		return nil, fmt.Errorf("failed to get container info: %w", err)
	}
	
	// Parse output (in a real implementation, we would parse the JSON properly)
	// For now, we'll create a minimal VMInfo object with the information
	
	vmInfo := &VMInfo{
		ID:       vmID,
		Name:     vmID, // Use container ID as name for now
		State:    status,
		CreatedAt: time.Now(),
	}
	
	// Get container stats for more accurate resource usage
	if status == StateRunning {
		statsCmd := exec.CommandContext(ctx, "docker", "stats", "--no-stream", "--format", 
			"{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}", vmID)
		statsOutput, statsErr := statsCmd.CombinedOutput()
		
		if statsErr == nil {
			statsParts := strings.Split(strings.TrimSpace(string(statsOutput)), "|")
			if len(statsParts) >= 4 {
				// Parse CPU percentage (e.g., "5.10%")
				cpuPct := statsParts[0]
				cpuPct = strings.TrimSuffix(cpuPct, "%")
				cpuFloat, _ := strconv.ParseFloat(cpuPct, 64)
				
				// Parse memory usage (e.g., "50MiB / 4GiB")
				memParts := strings.Split(statsParts[1], "/")
				memUsage := strings.TrimSpace(memParts[0])
				memValue, _ := strconv.ParseFloat(strings.TrimSuffix(memUsage, "MiB"), 64)
				
				// Add resource usage info
				vmInfo.CPUUsage = cpuFloat
				vmInfo.MemoryUsage = int64(memValue * 1024 * 1024) // Convert MiB to bytes
				
				// Parse network I/O (e.g., "648B / 648B")
				// For a full implementation, we would parse these values properly
				// and set them in NetworkInfo
			}
		}
	}
	
	return vmInfo, nil
}

// GetMetrics gets metrics for a container VM
func (d *ContainerDriver) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	// For containers, metrics are the same as info for now
	return d.GetInfo(ctx, vmID)
}

// ListVMs lists all container VMs
func (d *ContainerDriver) ListVMs(ctx context.Context) ([]VMInfo, error) {
	cmd := exec.CommandContext(ctx, "docker", "ps", "-a", "--format", "{{.ID}}|{{.Names}}|{{.Status}}")
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to list containers: %v, output: %s", err, string(output))
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}
	
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	var vms []VMInfo
	
	for _, line := range lines {
		if line == "" {
			continue
		}
		
		parts := strings.Split(line, "|")
		if len(parts) != 3 {
			continue
		}
		
		vmID := parts[0]
		name := parts[1]
		statusStr := parts[2]
		
		// Convert Docker status to VM state
		var state State
		if strings.Contains(statusStr, "Up") {
			state = StateRunning
		} else if strings.Contains(statusStr, "Exited") {
			state = StateStopped
		} else if strings.Contains(statusStr, "Created") {
			state = StateCreated
		} else {
			state = StateUnknown
		}
		
		vms = append(vms, VMInfo{
			ID:    vmID,
			Name:  name,
			State: state,
		})
	}
	
	return vms, nil
}

// SupportsPause returns whether the driver supports pausing VMs
func (d *ContainerDriver) SupportsPause() bool {
	return true
}

// SupportsResume returns whether the driver supports resuming VMs
func (d *ContainerDriver) SupportsResume() bool {
	return true
}

// SupportsSnapshot returns whether the driver supports snapshots
func (d *ContainerDriver) SupportsSnapshot() bool {
	return false // Docker containers don't support native snapshots
}

// SupportsMigrate returns whether the driver supports migration
func (d *ContainerDriver) SupportsMigrate() bool {
	return false // Docker containers don't support live migration
}

// Pause pauses a container VM
func (d *ContainerDriver) Pause(ctx context.Context, vmID string) error {
	log.Printf("Pausing container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "pause", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to pause container %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to pause container: %w", err)
	}
	
	return nil
}

// Resume resumes a container VM
func (d *ContainerDriver) Resume(ctx context.Context, vmID string) error {
	log.Printf("Resuming container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "unpause", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		log.Printf("Failed to resume container %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to resume container: %w", err)
	}
	
	return nil
}

// Snapshot creates a snapshot of a container VM (not supported)
func (d *ContainerDriver) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	return "", errors.New("snapshots not supported for container driver")
}

// Migrate migrates a container VM (not supported)
func (d *ContainerDriver) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	return errors.New("migration not supported for container driver")
}
