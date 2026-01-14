package main

import (
	"fmt"
	"log"

	"github.com/khryptorgraphics/novacron/backend/core/templates"
)

func main() {
	// Create a template manager
	manager := templates.NewTemplateManager()

	// Create an Ubuntu 24.04 VM template
	template := &templates.Template{
		ID:          "ubuntu-server-24.04",
		Name:        "Ubuntu Server 24.04 LTS",
		Description: "Standard Ubuntu Server 24.04 LTS (Noble Numbat) template",
		Type:        templates.VMTemplate,
		State:       templates.DraftState,
		Version:     "1.0.0",
		BaseImageID: "ubuntu-noble-24.04",
		DefaultProvisioningStrategy: templates.CloneStrategy,
		HardwareProfile: &templates.HardwareProfile{
			CPUCount:  2,
			CPUCores:  4,
			MemoryMB:  4096,
		},
		NetworkProfile: &templates.NetworkProfile{
			Interfaces: []*templates.NetworkInterface{
				{
					Name: "eth0",
					Type: "virtio",
					IPConfiguration: &templates.IPConfiguration{
						DHCPEnabled: true,
					},
				},
			},
			RequiresPublicIP: false,
		},
		StorageProfile: &templates.StorageProfile{
			Disks: []*templates.DiskProfile{
				{
					ID:          "disk0",
					Name:        "system",
					SizeGB:      40,
					Type:        "ssd",
					IsBoot:      true,
					IsSystemDisk: true,
				},
			},
		},
		IsPublic: true,
		OwnerID:  "admin",
		TenantID: "tenant-1",
		Parameters: []*templates.TemplateParameter{
			{
				ID:          "hostname",
				Name:        "Hostname",
				Description: "Hostname for the VM",
				Type:        "string",
				Required:    true,
				DefaultValue: "ubuntu-24-04-server",
			},
			{
				ID:          "ssh_key",
				Name:        "SSH Public Key",
				Description: "SSH public key for root access",
				Type:        "text",
				Required:    false,
			},
			{
				ID:          "enable_live_patch",
				Name:        "Enable Canonical Livepatch",
				Description: "Enable Canonical Livepatch for kernel updates without rebooting",
				Type:        "boolean",
				Required:    false,
				DefaultValue: true,
			},
		},
	}

	// Add template to manager
	if err := manager.CreateTemplate(template); err != nil {
		log.Fatalf("Failed to create template: %v", err)
	}

	fmt.Println("Successfully created Ubuntu 24.04 LTS template")

	// Publish the template to make it available for use
	if err := manager.PublishTemplate("ubuntu-server-24.04"); err != nil {
		log.Fatalf("Failed to publish template: %v", err)
	}

	fmt.Println("Successfully published Ubuntu 24.04 LTS template")

	// Create a provisioning request example
	request := &templates.ProvisioningRequest{
		ID:                  "provision-ubuntu-24-04",
		Name:                "web-server-ubuntu-24-04",
		Description:         "Web server running Ubuntu 24.04 LTS",
		TemplateID:          "ubuntu-server-24.04",
		TemplateVersion:     "1.0.0",
		ProvisioningStrategy: templates.CloneStrategy,
		Parameters: map[string]interface{}{
			"hostname":        "web-server-1",
			"ssh_key":         "ssh-rsa AAAAB3NzaC1...",
			"enable_live_patch": true,
		},
		Customizations: map[string]interface{}{
			"install_packages": []string{"nginx", "php-fpm", "certbot"},
			"open_ports":       []int{80, 443},
		},
		RequestedBy:    "user1",
		TenantID:       "tenant-1",
		ResourcePoolID: "pool-1",
	}

	// Create the provisioning request
	if err := manager.CreateProvisioningRequest(request); err != nil {
		log.Fatalf("Failed to create provisioning request: %v", err)
	}

	fmt.Println("Successfully created provisioning request for Ubuntu 24.04 LTS")
}
