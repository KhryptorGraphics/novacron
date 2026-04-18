package vm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

// Ubuntu2404Profile describes the canonical Ubuntu 24.04 guest pipeline.
type Ubuntu2404Profile struct {
	BaseImagePath     string
	CloudInitBasePath string
}

// Ubuntu2404CloudInitOptions configures the guest's NoCloud seed data.
type Ubuntu2404CloudInitOptions struct {
	Hostname          string
	SSHAuthorizedKeys []string
	Packages          []string
	UserData          string
}

// NewUbuntu2404Profile validates and returns the canonical Ubuntu 24.04 guest profile.
func NewUbuntu2404Profile(baseImagePath, cloudInitBasePath string) (*Ubuntu2404Profile, error) {
	if baseImagePath == "" {
		return nil, fmt.Errorf("ubuntu 24.04 base image path is required")
	}
	if _, err := os.Stat(baseImagePath); err != nil {
		return nil, fmt.Errorf("ubuntu 24.04 base image is unavailable: %w", err)
	}
	if cloudInitBasePath == "" {
		return nil, fmt.Errorf("cloud-init base path is required")
	}
	if err := os.MkdirAll(cloudInitBasePath, 0o755); err != nil {
		return nil, fmt.Errorf("create cloud-init base path: %w", err)
	}

	return &Ubuntu2404Profile{
		BaseImagePath:     baseImagePath,
		CloudInitBasePath: cloudInitBasePath,
	}, nil
}

// PrepareConfig converts a generic VM config into the canonical Ubuntu 24.04 KVM config.
func (p *Ubuntu2404Profile) PrepareConfig(vmID string, base VMConfig, options Ubuntu2404CloudInitOptions) (VMConfig, error) {
	if p == nil {
		return VMConfig{}, fmt.Errorf("ubuntu 24.04 profile is required")
	}
	if vmID == "" {
		return VMConfig{}, fmt.Errorf("vm id is required")
	}

	hostname := options.Hostname
	if hostname == "" {
		if base.Name != "" {
			hostname = base.Name
		} else {
			hostname = vmID
		}
	}

	cloudInitDir := filepath.Join(p.CloudInitBasePath, vmID)
	if err := os.MkdirAll(cloudInitDir, 0o755); err != nil {
		return VMConfig{}, fmt.Errorf("create cloud-init directory: %w", err)
	}

	userDataPath := filepath.Join(cloudInitDir, "user-data")
	metaDataPath := filepath.Join(cloudInitDir, "meta-data")
	seedISOPath := filepath.Join(cloudInitDir, "seed.iso")

	userData := options.UserData
	if userData == "" {
		userData = renderUbuntu2404UserData(hostname, options.SSHAuthorizedKeys, options.Packages)
	}
	if err := os.WriteFile(userDataPath, []byte(userData), 0o644); err != nil {
		return VMConfig{}, fmt.Errorf("write user-data: %w", err)
	}

	metaData := fmt.Sprintf("instance-id: %s\nlocal-hostname: %s\n", vmID, hostname)
	if err := os.WriteFile(metaDataPath, []byte(metaData), 0o644); err != nil {
		return VMConfig{}, fmt.Errorf("write meta-data: %w", err)
	}

	if err := createUbuntu2404SeedISO(seedISOPath, userDataPath, metaDataPath); err != nil {
		return VMConfig{}, fmt.Errorf("create cloud-init seed iso: %w", err)
	}

	if base.ID == "" {
		base.ID = vmID
	}
	if base.Name == "" {
		base.Name = hostname
	}
	if base.Type == "" {
		base.Type = VMTypeKVM
	}
	if base.CPUShares == 0 {
		base.CPUShares = 2
	}
	if base.MemoryMB == 0 {
		base.MemoryMB = 2048
	}
	if base.DiskSizeGB == 0 {
		base.DiskSizeGB = 20
	}
	if base.Tags == nil {
		base.Tags = make(map[string]string)
	}
	base.Tags["os"] = "ubuntu"
	base.Tags["version"] = "24.04"
	base.Tags["lts"] = "true"
	base.Image = p.BaseImagePath
	base.CloudInitISO = seedISOPath

	return base, nil
}

// CreateUbuntu2404VM creates a VM using the canonical Ubuntu 24.04 profile.
func (m *VMManager) CreateUbuntu2404VM(
	ctx context.Context,
	req CreateVMRequest,
	profile *Ubuntu2404Profile,
	options Ubuntu2404CloudInitOptions,
) (*VM, error) {
	if profile == nil {
		return nil, fmt.Errorf("ubuntu 24.04 profile is required")
	}

	if req.Spec.ID == "" {
		req.Spec.ID = uuid.New().String()
	}
	if req.Spec.Name == "" {
		if req.Name != "" {
			req.Spec.Name = req.Name
		} else {
			req.Spec.Name = req.Spec.ID
		}
	}

	preparedConfig, err := profile.PrepareConfig(req.Spec.ID, req.Spec, options)
	if err != nil {
		return nil, err
	}

	if len(req.Tags) > 0 {
		if preparedConfig.Tags == nil {
			preparedConfig.Tags = make(map[string]string)
		}
		for key, value := range req.Tags {
			preparedConfig.Tags[key] = value
		}
	}

	req.Spec = preparedConfig
	return m.CreateVM(ctx, req)
}

func renderUbuntu2404UserData(hostname string, sshKeys, packages []string) string {
	packageList := uniqueStrings(append(
		[]string{"qemu-guest-agent", "cloud-init", "cloud-initramfs-growroot"},
		packages...,
	))

	var builder strings.Builder
	builder.WriteString("#cloud-config\n")
	builder.WriteString(fmt.Sprintf("hostname: %s\n", hostname))
	builder.WriteString("manage_etc_hosts: true\n")
	builder.WriteString("users:\n")
	builder.WriteString("  - name: ubuntu\n")
	builder.WriteString("    sudo: ALL=(ALL) NOPASSWD:ALL\n")
	builder.WriteString("    shell: /bin/bash\n")
	if len(sshKeys) > 0 {
		builder.WriteString("    ssh_authorized_keys:\n")
		for _, key := range sshKeys {
			builder.WriteString(fmt.Sprintf("      - %s\n", key))
		}
	}
	builder.WriteString("package_update: true\n")
	builder.WriteString("package_upgrade: true\n")
	builder.WriteString("packages:\n")
	for _, pkg := range packageList {
		builder.WriteString(fmt.Sprintf("  - %s\n", pkg))
	}
	builder.WriteString("runcmd:\n")
	builder.WriteString("  - systemctl enable qemu-guest-agent.service\n")
	builder.WriteString("  - systemctl start qemu-guest-agent.service\n")

	return builder.String()
}

func createUbuntu2404SeedISO(isoPath, userDataPath, metaDataPath string) error {
	if toolPath, err := exec.LookPath("cloud-localds"); err == nil {
		cmd := exec.Command(toolPath, isoPath, userDataPath, metaDataPath)
		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("cloud-localds failed: %w, output: %s", err, strings.TrimSpace(string(output)))
		}
		return nil
	}

	for _, toolName := range []string{"genisoimage", "mkisofs"} {
		toolPath, err := exec.LookPath(toolName)
		if err != nil {
			continue
		}

		cmd := exec.Command(
			toolPath,
			"-output", isoPath,
			"-volid", "cidata",
			"-joliet",
			"-rock",
			userDataPath,
			metaDataPath,
		)
		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("%s failed: %w, output: %s", toolName, err, strings.TrimSpace(string(output)))
		}
		return nil
	}

	return fmt.Errorf("no cloud-init ISO builder found in PATH (need cloud-localds, genisoimage, or mkisofs)")
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}
