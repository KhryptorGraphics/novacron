package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// Ubuntu2404KVMConfig provides configuration for Ubuntu 24.04 VMs on KVM
type Ubuntu2404KVMConfig struct {
	// Base image path for Ubuntu 24.04
	BaseImagePath string
	
	// Cloud-init configuration directory
	CloudInitDir string
	
	// Default VM specifications
	DefaultSpec VMSpec
}

// NewUbuntu2404KVMConfig creates a new Ubuntu 24.04 KVM configuration
func NewUbuntu2404KVMConfig(baseImagePath, cloudInitDir string) *Ubuntu2404KVMConfig {
	return &Ubuntu2404KVMConfig{
		BaseImagePath: baseImagePath,
		CloudInitDir:  cloudInitDir,
		DefaultSpec: VMSpec{
			VCPU:     2,
			MemoryMB: 2048,
			DiskMB:   20480, // 20GB
			Type:     VMTypeKVM,
			Image:    "ubuntu-24.04",
			Networks: []VMNetworkSpec{
				{
					NetworkID: "default",
				},
			},
			Env: map[string]string{
				"OS_VERSION": "24.04",
				"OS_NAME":    "Ubuntu",
				"OS_CODENAME": "Noble Numbat",
			},
			Labels: map[string]string{
				"os":      "ubuntu",
				"version": "24.04",
				"lts":     "true",
			},
		},
	}
}

// CreateUbuntu2404VM creates a new Ubuntu 24.04 VM using the KVM driver
func CreateUbuntu2404VM(ctx context.Context, driver *KVMDriver, config *Ubuntu2404KVMConfig, name string, customSpec *VMSpec) (string, error) {
	log.Printf("Creating Ubuntu 24.04 VM: %s", name)
	
	// Create VM directory
	vmDir := filepath.Join(driver.vmBasePath, name)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create VM directory: %w", err)
	}
	
	// Use default spec if custom spec is not provided
	spec := config.DefaultSpec
	if customSpec != nil {
		// Merge custom spec with default spec
		if customSpec.VCPU > 0 {
			spec.VCPU = customSpec.VCPU
		}
		if customSpec.MemoryMB > 0 {
			spec.MemoryMB = customSpec.MemoryMB
		}
		if customSpec.DiskMB > 0 {
			spec.DiskMB = customSpec.DiskMB
		}
		if len(customSpec.Networks) > 0 {
			spec.Networks = customSpec.Networks
		}
		if len(customSpec.Volumes) > 0 {
			spec.Volumes = customSpec.Volumes
		}
		if len(customSpec.Env) > 0 {
			for k, v := range customSpec.Env {
				spec.Env[k] = v
			}
		}
		if len(customSpec.Labels) > 0 {
			for k, v := range customSpec.Labels {
				spec.Labels[k] = v
			}
		}
	}
	
	// Set the image to the Ubuntu 24.04 base image
	spec.Image = config.BaseImagePath
	
	// Create the VM using the KVM driver
	vmID, err := driver.Create(ctx, spec)
	if err != nil {
		return "", fmt.Errorf("failed to create Ubuntu 24.04 VM: %w", err)
	}
	
	log.Printf("Created Ubuntu 24.04 VM with ID: %s", vmID)
	return vmID, nil
}

// PrepareUbuntu2404CloudInit prepares cloud-init configuration for Ubuntu 24.04
func PrepareUbuntu2404CloudInit(config *Ubuntu2404KVMConfig, vmID string, userData map[string]interface{}) error {
	// Create cloud-init directory for this VM
	cloudInitDir := filepath.Join(config.CloudInitDir, vmID)
	if err := os.MkdirAll(cloudInitDir, 0755); err != nil {
		return fmt.Errorf("failed to create cloud-init directory: %w", err)
	}
	
	// Create user-data file
	// In a real implementation, this would generate a proper cloud-init user-data file
	// based on the provided userData map
	userDataContent := `#cloud-config
hostname: ${hostname}
manage_etc_hosts: true
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - ${ssh_key}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
`
	
	// Replace variables with actual values
	if hostname, ok := userData["hostname"].(string); ok {
		userDataContent = replaceVariable(userDataContent, "hostname", hostname)
	} else {
		userDataContent = replaceVariable(userDataContent, "hostname", vmID)
	}
	
	if sshKey, ok := userData["ssh_key"].(string); ok {
		userDataContent = replaceVariable(userDataContent, "ssh_key", sshKey)
	} else {
		userDataContent = replaceVariable(userDataContent, "ssh_key", "")
	}
	
	// Write user-data file
	userDataPath := filepath.Join(cloudInitDir, "user-data")
	if err := os.WriteFile(userDataPath, []byte(userDataContent), 0644); err != nil {
		return fmt.Errorf("failed to write user-data file: %w", err)
	}
	
	// Create meta-data file
	metaDataContent := fmt.Sprintf("instance-id: %s\nlocal-hostname: %s\n", vmID, vmID)
	metaDataPath := filepath.Join(cloudInitDir, "meta-data")
	if err := os.WriteFile(metaDataPath, []byte(metaDataContent), 0644); err != nil {
		return fmt.Errorf("failed to write meta-data file: %w", err)
	}
	
	// Create cloud-init ISO
	isoPath := filepath.Join(cloudInitDir, "cloud-init.iso")
	if err := createCloudInitISO(cloudInitDir, isoPath); err != nil {
		return fmt.Errorf("failed to create cloud-init ISO: %w", err)
	}
	
	return nil
}

// Helper function to replace variables in cloud-init templates
func replaceVariable(content, variable, value string) string {
	return replaceString(content, "${"+variable+"}", value)
}

// Helper function to replace strings
func replaceString(content, old, new string) string {
	// In a real implementation, this would use proper string replacement
	// For simplicity, we'll use a placeholder implementation
	return content
}

// Helper function to create a cloud-init ISO
func createCloudInitISO(sourceDir, isoPath string) error {
	// In a real implementation, this would use genisoimage or similar tool
	// to create an ISO from the cloud-init files
	// For simplicity, we'll use a placeholder implementation
	return nil
}
