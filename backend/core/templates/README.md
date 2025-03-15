# VM Templates and Provisioning

This package provides comprehensive VM template management and rapid provisioning capabilities for NovaCron, including template versioning, sharing, and customization options.

## Features

- **Multiple Template Types**: VM, container, and application templates
- **Template Versioning**: Track and manage template versions
- **Rapid Provisioning**: Fast deployment from templates
- **Customization**: Parameter-based customization during provisioning
- **Access Control**: Fine-grained template access control
- **Multi-tenant**: Full tenant isolation for templates
- **Template Sharing**: Public and private templates with sharing options
- **Hardware Profiles**: Define hardware requirements for templates
- **Network Profiles**: Configure networking for templates
- **Storage Profiles**: Manage storage configurations

## Components

### Template Manager

The `TemplateManager` is the central component that manages templates and provisioning:

- Creating and managing templates
- Versioning templates
- Provisioning resources from templates
- Managing template access control
- Tracking provisioning requests

### Template Types

The system supports multiple template types:

- **VM Templates**: Templates for virtual machines
- **Container Templates**: Templates for containerized applications
- **Application Templates**: Multi-component application templates

### Provisioning Strategies

Different provisioning strategies are supported:

- **Clone**: Full clone of the template
- **Linked Clone**: Space-efficient clone that shares base with template
- **Instantiate**: Create a new instance from the template

## Usage Examples

### Creating a Template

```go
// Create a template manager
manager := templates.NewTemplateManager()

// Create a VM template
template := &templates.Template{
    ID:          "ubuntu-server-20.04",
    Name:        "Ubuntu Server 20.04 LTS",
    Description: "Standard Ubuntu Server 20.04 LTS template",
    Type:        templates.VMTemplate,
    State:       templates.DraftState,
    Version:     "1.0.0",
    BaseImageID: "ubuntu-focal-20.04",
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
    IsPublic: false,
    OwnerID:  "admin",
    TenantID: "tenant-1",
    Parameters: []*templates.TemplateParameter{
        {
            ID:          "hostname",
            Name:        "Hostname",
            Description: "Hostname for the VM",
            Type:        "string",
            Required:    true,
            DefaultValue: "ubuntu-server",
        },
        {
            ID:          "ssh_key",
            Name:        "SSH Public Key",
            Description: "SSH public key for root access",
            Type:        "text",
            Required:    false,
        },
    },
}

// Add template to manager
if err := manager.CreateTemplate(template); err != nil {
    log.Fatalf("Failed to create template: %v", err)
}
```

### Publishing a Template

```go
// Publish a template to make it available for use
if err := manager.PublishTemplate("ubuntu-server-20.04"); err != nil {
    log.Fatalf("Failed to publish template: %v", err)
}
```

### Provisioning from a Template

```go
// Create a provisioning request
request := &templates.ProvisioningRequest{
    ID:                  "provision-1",
    Name:                "web-server-1",
    Description:         "Web server for production",
    TemplateID:          "ubuntu-server-20.04",
    TemplateVersion:     "1.0.0",
    ProvisioningStrategy: templates.CloneStrategy,
    Parameters: map[string]interface{}{
        "hostname": "web-server-1",
        "ssh_key":  "ssh-rsa AAAAB3NzaC1...",
    },
    Customizations: map[string]interface{}{
        "install_packages": []string{"nginx", "php-fpm"},
        "open_ports":       []int{80, 443},
    },
    RequestedBy:    "user1",
    TenantID:       "tenant-1",
    ResourcePoolID: "pool-1",
}

// Create provisioning request
if err := manager.CreateProvisioningRequest(request); err != nil {
    log.Fatalf("Failed to create provisioning request: %v", err)
}

// Start provisioning
if err := manager.StartProvisioning(request.ID); err != nil {
    log.Fatalf("Failed to start provisioning: %v", err)
}

// Check provisioning status
status, err := manager.GetProvisioningRequest(request.ID)
if err != nil {
    log.Fatalf("Failed to get provisioning status: %v", err)
}

log.Printf("Provisioning status: %s", status.Status)
```

### Creating a Template Version

```go
// Create a new version of an existing template
version, err := manager.CreateTemplateVersion(
    "ubuntu-server-20.04",  // Template ID
    "1.1.0",                // New version
    "Updated with security patches",  // Description
    "templates/ubuntu-server-20.04/1.1.0",  // Storage location
    "admin",                // Created by
)
if err != nil {
    log.Fatalf("Failed to create template version: %v", err)
}

log.Printf("Created template version: %s", version.Version)
```

## Integration Points

The templates system integrates with other NovaCron components:

- **VM Manager**: For VM template provisioning
- **Storage Manager**: For template storage and disk handling
- **Network Manager**: For network configuration during provisioning
- **Security Manager**: For access control and secure provisioning
- **Backup Manager**: For template backup and recovery

## Performance Considerations

The template system is designed for performance and efficiency:

1. **Template Library**: Efficient storage and indexing of templates
2. **Linked Clones**: Space-efficient provisioning with linked clones
3. **Parallel Provisioning**: Multiple simultaneous provisioning operations
4. **Caching**: Caching frequently used templates for faster provisioning
5. **Incremental Updates**: Efficient template versioning with incremental updates

## Future Enhancements

- **Template Marketplace**: Public marketplace for template sharing
- **AI-Assisted Template Creation**: ML-based template recommendations
- **Automated Testing**: Automated validation of templates
- **Template Dependencies**: Support for template dependencies and composition
- **Infrastructure as Code Integration**: Integration with IaC tools like Terraform
