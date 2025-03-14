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
func NewContainerDriver(nodeID string) *ContainerDriver {
	return &ContainerDriver{
		nodeID: nodeID,
	}
}

// Create creates a new container VM
func (d *ContainerDriver) Create(ctx context.Context, spec VMSpec) (string, error) {
	log.Printf("Creating container VM with image %s", spec.Image)
	
	// Generate a container name based on VM specs
	containerName := fmt.Sprintf("novacron-%s-%s", spec.Type, strconv.FormatInt(time.Now().UnixNano(), 16))
	
	// Build the docker command to create a container
	args := []string{
		"create",
		"--name", containerName,
	}
	
	// Set resource limits
	if spec.VCPU > 0 {
		args = append(args, "--cpus", fmt.Sprintf("%d", spec.VCPU))
	}
	
	if spec.MemoryMB > 0 {
		args = append(args, "--memory", fmt.Sprintf("%dm", spec.MemoryMB))
	}
	
	// Set environment variables
	for k, v := range spec.Env {
		args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Set labels
	for k, v := range spec.Labels {
		args = append(args, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Mount volumes
	for _, vol := range spec.Volumes {
		mountOptions := "rw"
		if vol.ReadOnly {
			mountOptions = "ro"
		}
		args = append(args, "-v", fmt.Sprintf("%s:%s:%s", vol.VolumeID, vol.Path, mountOptions))
	}
	
	// Configure network
	if len(spec.Networks) > 0 {
		for _, net := range spec.Networks {
			args = append(args, "--network", net.NetworkID)
			
			if net.IPAddress != "" {
				args = append(args, "--ip", net.IPAddress)
			}
			
			if net.MACAddress != "" {
				args = append(args, "--mac-address", net.MACAddress)
			}
		}
	}
	
	// Add the image name as the last argument
	args = append(args, spec.Image)
	
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
func (d *ContainerDriver) GetStatus(ctx context.Context, vmID string) (VMState, error) {
	log.Printf("Getting status of container VM %s", vmID)
	
	cmd := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Status}}", vmID)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		if strings.Contains(string(output), "No such container") {
			return VMStateUnknown, errors.New("container not found")
		}
		log.Printf("Failed to get status of container %s: %v, output: %s", vmID, err, string(output))
		return VMStateUnknown, fmt.Errorf("failed to get container status: %w", err)
	}
	
	status := strings.TrimSpace(string(output))
	
	switch status {
	case "running":
		return VMStateRunning, nil
	case "exited":
		return VMStateStopped, nil
	case "created":
		return VMStateStopped, nil
	case "paused":
		return VMStatePaused, nil
	case "restarting":
		return VMStateRestarting, nil
	default:
		return VMStateUnknown, nil
	}
}

// GetInfo gets information about a container VM
func (d *ContainerDriver) GetInfo(ctx context.Context, vmID string) (*VM, error) {
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
	// For now, we'll create a minimal VM object with the information
	
	vm := &VM{
		ID:      vmID,
		State:   status,
		NodeID:  d.nodeID,
		UpdatedAt: time.Now(),
	}
	
	// Get container stats for more accurate resource usage
	if status == VMStateRunning {
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
				
				// Add process info
				vm.ProcessInfo = VMProcessInfo{
					CPUUsagePercent: cpuFloat,
					MemoryUsageMB:   int(memValue),
					LastUpdatedAt:   time.Now(),
				}
				
				// Parse network I/O (e.g., "648B / 648B")
				// For a full implementation, we would parse these values properly
				// and set them in NetworkInfo
			}
		}
	}
	
	return vm, nil
}
