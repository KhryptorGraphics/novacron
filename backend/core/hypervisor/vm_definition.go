package hypervisor

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"text/template"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// VMDefinition represents a virtual machine definition with extended configuration options
type VMDefinition struct {
	// Basic VM information
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	CPUCores int               `json:"cpu_cores"`
	MemoryMB int               `json:"memory_mb"`
	Tags     map[string]string `json:"tags"`

	// OS configuration
	OSType    string `json:"os_type"`    // e.g., "linux", "windows"
	OSVariant string `json:"os_variant"` // e.g., "ubuntu20.04", "win10"

	// Storage configuration
	Volumes []Volume `json:"volumes"`

	// Network configuration
	NetworkInterfaces []NetworkInterface `json:"network_interfaces"`

	// Advanced configuration
	Features     []string          `json:"features"`      // e.g., "acpi", "apic"
	BootDevices  []string          `json:"boot_devices"`  // e.g., "hd", "cdrom", "network"
	ClockOffset  string            `json:"clock_offset"`  // e.g., "utc", "localtime"
	EmulatorPath string            `json:"emulator_path"` // Path to the emulator binary
	Metadata     map[string]string `json:"metadata"`      // Custom metadata
}

// Volume represents a storage volume for a VM
type Volume struct {
	Type      string `json:"type"`       // e.g., "file", "block"
	Device    string `json:"device"`     // e.g., "disk", "cdrom"
	Format    string `json:"format"`     // e.g., "qcow2", "raw"
	Path      string `json:"path"`       // Path to the disk image
	TargetDev string `json:"target_dev"` // e.g., "vda", "hda"
	TargetBus string `json:"target_bus"` // e.g., "virtio", "ide"
	SizeGB    int    `json:"size_gb"`    // Size in GB
	ReadOnly  bool   `json:"read_only"`  // Whether the disk is read-only
	Shareable bool   `json:"shareable"`  // Whether the disk is shareable
	CacheMode string `json:"cache_mode"` // e.g., "none", "writeback"
	IOMode    string `json:"io_mode"`    // e.g., "native", "threads"
}

// NetworkInterface represents a network interface for a VM
type NetworkInterface struct {
	Type      string `json:"type"`       // e.g., "network", "bridge"
	Source    string `json:"source"`     // e.g., "default", "br0"
	Model     string `json:"model"`      // e.g., "virtio", "e1000"
	MAC       string `json:"mac"`        // MAC address
	PortGroup string `json:"port_group"` // Port group for virtual networks
	Managed   bool   `json:"managed"`    // Whether the interface is managed
	MTU       int    `json:"mtu"`        // MTU size
}

// XMLTemplateGenerator generates libvirt XML from VM definitions
type XMLTemplateGenerator interface {
	// GenerateXML generates a complete XML definition for libvirt
	GenerateXML(def *VMDefinition) (string, error)

	// ValidateDefinition validates a VM definition for completeness and correctness
	ValidateDefinition(def *VMDefinition) error

	// LoadTemplate loads a template from a file or embedded resource
	LoadTemplate(name string) error

	// ListAvailableTemplates lists available templates
	ListAvailableTemplates() []string
}

// templateData holds data for template rendering
type templateData struct {
	VM       *VMDefinition
	Defaults map[string]interface{}
}

// xmlTemplateGenerator implements XMLTemplateGenerator
type xmlTemplateGenerator struct {
	templates       map[string]*template.Template
	defaultTemplate string
	osTemplates     map[string]string // Maps OS type/variant to template name
	templateDir     string
	defaults        map[string]interface{}
}

// NewXMLTemplateGenerator creates a new XML template generator
func NewXMLTemplateGenerator(templateDir string) (XMLTemplateGenerator, error) {
	generator := &xmlTemplateGenerator{
		templates:       make(map[string]*template.Template),
		osTemplates:     make(map[string]string),
		templateDir:     templateDir,
		defaultTemplate: "default",
		defaults: map[string]interface{}{
			"emulator": "/usr/bin/qemu-system-x86_64",
			"features": []string{"acpi", "apic"},
			"clock":    "utc",
		},
	}

	// Load default template
	if err := generator.LoadTemplate(generator.defaultTemplate); err != nil {
		return nil, fmt.Errorf("failed to load default template: %w", err)
	}

	// Set up OS template mappings
	generator.osTemplates["linux/ubuntu20.04"] = "ubuntu"
	generator.osTemplates["linux/debian10"] = "debian"
	generator.osTemplates["windows/win10"] = "windows"
	generator.osTemplates["windows/win2019"] = "windows-server"

	// Try to load OS-specific templates
	for _, tmpl := range generator.osTemplates {
		_ = generator.LoadTemplate(tmpl) // Ignore errors, will fall back to default
	}

	return generator, nil
}

// LoadTemplate loads a template from a file or embedded resource
func (g *xmlTemplateGenerator) LoadTemplate(name string) error {
	// Try to load from template directory
	templatePath := filepath.Join(g.templateDir, name+".xml.tmpl")

	// Check if file exists
	if _, err := os.Stat(templatePath); os.IsNotExist(err) {
		// If not in template directory, try embedded templates
		return fmt.Errorf("template %s not found at %s", name, templatePath)
	}

	// Read template file
	content, err := ioutil.ReadFile(templatePath)
	if err != nil {
		return fmt.Errorf("failed to read template file %s: %w", templatePath, err)
	}

	// Parse template
	tmpl, err := template.New(name).Parse(string(content))
	if err != nil {
		return fmt.Errorf("failed to parse template %s: %w", name, err)
	}

	// Store template
	g.templates[name] = tmpl
	return nil
}

// ListAvailableTemplates lists available templates
func (g *xmlTemplateGenerator) ListAvailableTemplates() []string {
	templates := make([]string, 0, len(g.templates))
	for name := range g.templates {
		templates = append(templates, name)
	}
	return templates
}

// GenerateXML generates a complete XML definition for libvirt
func (g *xmlTemplateGenerator) GenerateXML(def *VMDefinition) (string, error) {
	// Validate definition
	if err := g.ValidateDefinition(def); err != nil {
		return "", err
	}

	// Determine which template to use
	templateName := g.defaultTemplate
	if def.OSType != "" && def.OSVariant != "" {
		osKey := fmt.Sprintf("%s/%s", def.OSType, def.OSVariant)
		if tmpl, ok := g.osTemplates[osKey]; ok && g.templates[tmpl] != nil {
			templateName = tmpl
		}
	}

	// Get template
	tmpl, ok := g.templates[templateName]
	if !ok {
		return "", fmt.Errorf("template %s not found", templateName)
	}

	// Prepare template data
	data := templateData{
		VM:       def,
		Defaults: g.defaults,
	}

	// Render template
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to render template: %w", err)
	}

	// Format XML
	xmlBytes := buf.Bytes()
	var formattedXML bytes.Buffer
	if err := xml.Unmarshal(xmlBytes, &formattedXML); err != nil {
		// If XML is invalid, return the unformatted version
		return buf.String(), nil
	}

	return formattedXML.String(), nil
}

// ValidateDefinition validates a VM definition for completeness and correctness
func (g *xmlTemplateGenerator) ValidateDefinition(def *VMDefinition) error {
	if def == nil {
		return fmt.Errorf("VM definition is nil")
	}

	if def.Name == "" {
		return fmt.Errorf("VM name is required")
	}

	if def.CPUCores <= 0 {
		return fmt.Errorf("VM must have at least 1 CPU core")
	}

	if def.MemoryMB <= 0 {
		return fmt.Errorf("VM must have positive memory allocation")
	}

	// Validate volumes
	for i, vol := range def.Volumes {
		if vol.Type == "" {
			return fmt.Errorf("volume %d: type is required", i)
		}

		if vol.Device == "" {
			return fmt.Errorf("volume %d: device is required", i)
		}

		if vol.Path == "" && vol.SizeGB <= 0 {
			return fmt.Errorf("volume %d: either path or size must be specified", i)
		}

		if vol.TargetDev == "" {
			return fmt.Errorf("volume %d: target device is required", i)
		}
	}

	// Validate network interfaces
	for i, iface := range def.NetworkInterfaces {
		if iface.Type == "" {
			return fmt.Errorf("network interface %d: type is required", i)
		}

		if iface.Source == "" {
			return fmt.Errorf("network interface %d: source is required", i)
		}
	}

	return nil
}

// ConvertVMConfigToDefinition converts a vm.VMConfig to a VMDefinition
func ConvertVMConfigToDefinition(config vm.VMConfig) *VMDefinition {
	def := &VMDefinition{
		ID:       config.ID,
		Name:     config.Name,
		CPUCores: config.CPUShares,
		MemoryMB: config.MemoryMB,
		Tags:     config.Tags,
		OSType:   "linux", // Default to Linux
	}

	// Add a default volume for the root filesystem
	if config.RootFS != "" {
		def.Volumes = append(def.Volumes, Volume{
			Type:      "file",
			Device:    "disk",
			Format:    "qcow2",
			Path:      config.RootFS,
			TargetDev: "vda",
			TargetBus: "virtio",
		})
	}

	// Add a default network interface
	if config.NetworkID != "" {
		def.NetworkInterfaces = append(def.NetworkInterfaces, NetworkInterface{
			Type:   "network",
			Source: config.NetworkID,
			Model:  "virtio",
		})
	}

	return def
}

// ConvertDefinitionToVMConfig converts a VMDefinition to a vm.VMConfig
func ConvertDefinitionToVMConfig(def *VMDefinition) vm.VMConfig {
	config := vm.VMConfig{
		ID:        def.ID,
		Name:      def.Name,
		CPUShares: def.CPUCores,
		MemoryMB:  def.MemoryMB,
		Tags:      def.Tags,
	}

	// Set root filesystem from the first volume
	if len(def.Volumes) > 0 {
		config.RootFS = def.Volumes[0].Path
	}

	// Set network ID from the first network interface
	if len(def.NetworkInterfaces) > 0 {
		config.NetworkID = def.NetworkInterfaces[0].Source
	}

	return config
}
