package hypervisor

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"log"
	"net/url" // Import net/url for URI parsing
	"time"
	"unsafe" // Required for CGo pointer conversions with libvirt

	"github.com/digitalocean/go-libvirt"
	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// KVMManager manages KVM virtual machines using libvirt
type KVMManager struct {
	libvirtURI string
	conn       *libvirt.Libvirt // Pointer type is correct
}

// NewKVMManager creates a new KVM manager instance
func NewKVMManager(uri string) (*KVMManager, error) {
	if uri == "" {
		uri = string(libvirt.QEMUSystem) // Use libvirt constant
	}
	// Use ConnectToURI with parsed URI
	parsedURI, err := url.Parse(uri)
	if err != nil {
		return nil, fmt.Errorf("failed to parse libvirt URI %s: %w", uri, err)
	}
	l, err := libvirt.ConnectToURI(parsedURI) // Correct connection function
	if err != nil {
		return nil, fmt.Errorf("failed to connect to libvirt URI %s: %w", uri, err)
	}
	log.Printf("Connected to libvirt at %s", uri)
	return &KVMManager{
		libvirtURI: uri,
		conn:       l,
	}, nil
}

// Close closes the connection to libvirt
func (m *KVMManager) Close() error {
	if m.conn != nil {
		if err := m.conn.Disconnect(); err != nil {
			return fmt.Errorf("failed to disconnect from libvirt: %w", err)
		}
		log.Println("Disconnected from libvirt")
		m.conn = nil
	}
	return nil
}

// --- VM Lifecycle Management ---

// CreateVM creates a KVM virtual machine based on the provided config
func (m *KVMManager) CreateVM(ctx context.Context, vmConfig vm.VMConfig) (*vm.VMInfo, error) {
	log.Printf("Creating KVM VM: %s", vmConfig.Name)
	xmlDef, err := generateDomainXML(vmConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate domain XML: %w", err)
	}
	// Define the domain
	domain, err := m.conn.DomainDefineXML(xmlDef)
	if err != nil {
		return nil, fmt.Errorf("failed to define domain: %w", err)
	}
	// Start the domain
	if err := m.conn.DomainCreate(domain); err != nil {
		_ = m.conn.DomainUndefine(domain)
		return nil, fmt.Errorf("failed to start domain: %w", err)
	}
	// Get domain info (returns 6 values)
	state, maxMem, memory, nrVirtCpu, cpuTime, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		_ = m.conn.DomainDestroy(domain)
		_ = m.conn.DomainUndefine(domain)
		return nil, fmt.Errorf("failed to get domain info: %w", err)
	}
	// Get UUID
	uuidStr := uuid.UUID(domain.UUID).String()
	// Construct VMInfo
	vmInfo := vm.VMInfo{
		ID:        uuidStr,
		Name:      vmConfig.Name,
		State:     mapLibvirtState(libvirt.DomainState(state)),
		CPUShares: vmConfig.CPUShares,
		MemoryMB:  vmConfig.MemoryMB,
		CreatedAt: time.Now(),
		RootFS:    vmConfig.RootFS,
		Tags:      vmConfig.Tags,
	}
	_ = maxMem
	_ = memory
	_ = nrVirtCpu
	_ = cpuTime
	return &vmInfo, nil
}

// DeleteVM deletes a KVM virtual machine
func (m *KVMManager) DeleteVM(ctx context.Context, vmID string) error {
	domain, err := m.findDomain(vmID)
	if err != nil {
		return err
	}
	// Get domain info (returns 6 values)
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err == nil && libvirt.DomainState(state) == libvirt.DomainRunning {
		if err := m.conn.DomainDestroy(domain); err != nil {
			log.Printf("Warning: failed to destroy domain %s before undefining: %v", vmID, err)
		}
	} else if err != nil {
		log.Printf("Warning: failed to get domain info for %s before undefining: %v", vmID, err)
	}
	// Undefine the domain
	if err := m.conn.DomainUndefine(domain); err != nil {
		return fmt.Errorf("failed to undefine domain %s: %w", vmID, err)
	}
	log.Printf("Deleted KVM VM: %s", vmID)
	return nil
}

// StartVM starts a KVM virtual machine
func (m *KVMManager) StartVM(ctx context.Context, vmID string) error {
	domain, err := m.findDomain(vmID)
	if err != nil {
		return err
	}
	// Get domain info (returns 6 values)
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return fmt.Errorf("failed to get domain info for %s: %w", vmID, err)
	}
	if libvirt.DomainState(state) == libvirt.DomainRunning {
		log.Printf("VM %s is already running", vmID)
		return nil // Already running
	}
	if err := m.conn.DomainCreate(domain); err != nil {
		return fmt.Errorf("failed to start domain %s: %w", vmID, err)
	}
	log.Printf("Started KVM VM: %s", vmID)
	return nil
}

// StopVM stops a KVM virtual machine
func (m *KVMManager) StopVM(ctx context.Context, vmID string, force bool) error {
	domain, err := m.findDomain(vmID)
	if err != nil {
		return err
	}
	// Get domain info (returns 6 values)
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return fmt.Errorf("failed to get domain info for %s: %w", vmID, err)
	}
	if libvirt.DomainState(state) == libvirt.DomainShutoff {
		log.Printf("VM %s is already stopped", vmID)
		return nil // Already stopped
	}
	if force {
		// DomainDestroy takes *Domain
		if err := m.conn.DomainDestroy(domain); err != nil {
			return fmt.Errorf("failed to force stop (destroy) domain %s: %w", vmID, err)
		}
		log.Printf("Force stopped KVM VM: %s", vmID)
	} else {
		if err := m.conn.DomainShutdown(domain); err != nil {
			return fmt.Errorf("failed to gracefully stop (shutdown) domain %s: %w", vmID, err)
		}
		log.Printf("Requested graceful shutdown for KVM VM: %s", vmID)
	}
	return nil
}

// GetVMStatus retrieves the status of a KVM virtual machine
func (m *KVMManager) GetVMStatus(ctx context.Context, vmID string) (vm.State, error) {
	domain, err := m.findDomain(vmID) // findDomain returns *Domain
	if err != nil {
		log.Printf("Could not find domain %s: %v", vmID, err)
		return vm.StateStopped, nil // Assume stopped if not found
	}
	// Get domain info (returns 6 values)
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return vm.StateFailed, fmt.Errorf("failed to get domain info for %s: %w", vmID, err)
	}
	return mapLibvirtState(libvirt.DomainState(state)), nil
}

// ListVMs lists all KVM virtual machines managed by this hypervisor
func (m *KVMManager) ListVMs(ctx context.Context) ([]vm.VMInfo, error) {
	// Use ConnectListAllDomains
	domains, _, err := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive|libvirt.ConnectListDomainsInactive)
	if err != nil {
		return nil, fmt.Errorf("failed to list all domains: %w", err)
	}
	// No need to free domains in go-libvirt

	var vms []vm.VMInfo
	for _, dom := range domains {
		// DomainGetInfo takes Domain value
		state, maxMem, _, nrVirtCpu, _, errInfo := m.conn.DomainGetInfo(dom)
		if errInfo != nil {
			log.Printf("Warning: failed to get info for domain ID %d: %v", dom.ID, errInfo)
			continue
		}
		// DomainGetName takes Domain value
		name := dom.Name
		if name == "" {
			name = fmt.Sprintf("domain-%d", dom.ID)
		}
		// Use dom.UUID for UUID
		domainUUID := uuid.UUID(dom.UUID).String()

		vms = append(vms, vm.VMInfo{
			ID:        domainUUID,
			Name:      name,
			State:     mapLibvirtState(libvirt.DomainState(state)),
			CPUShares: int(nrVirtCpu),     // Map from libvirt info
			MemoryMB:  int(maxMem / 1024), // Map from libvirt info
			// CreatedAt, RootFS, Tags would require fetching/parsing domain XML or metadata
		})
	}
	return vms, nil
}

// --- Metric Collection ---

// GetVMMetrics retrieves performance metrics for a specific KVM VM
func (m *KVMManager) GetVMMetrics(ctx context.Context, vmID string) (*vm.VMInfo, error) {
	domain, err := m.findDomain(vmID) // findDomain returns *Domain
	if err != nil {
		return nil, err
	}
	// Get domain info (returns 6 values)
	state, maxMem, _, nrVirtCpu, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return nil, fmt.Errorf("failed to get domain info for %s: %w", vmID, err)
	}
	// Use domain.UUID for UUID
	domainUUID := uuid.UUID(domain.UUID).String()
	// Get domain name
	name := domain.Name
	if name == "" {
		name = vmID
	}
	vmInfo := &vm.VMInfo{
		ID:        domainUUID,
		Name:      name,
		State:     mapLibvirtState(libvirt.DomainState(state)),
		CPUShares: int(nrVirtCpu),
		MemoryMB:  int(maxMem / 1024),
		// MemoryUsage: not implemented (would require additional API calls)
	}
	return vmInfo, nil
}

// GetHypervisorMetrics retrieves performance metrics for the KVM host
func (m *KVMManager) GetHypervisorMetrics(ctx context.Context) (*ResourceInfo, error) {
	// Use ConnectGetNodeInfo
	// Node info retrieval not implemented in go-libvirt; set to zero values or implement if needed
	var nodeInfo ResourceInfo
	return &nodeInfo, nil
	// allDomains, numDomains, errAll := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive|libvirt.ConnectListDomainsInactive)
	// Not implemented in go-libvirt; return zero/default values
	return &ResourceInfo{
		CPUCores:    0,
		MemoryTotal: 0,
		VMs:         0,
		VMsRunning:  0,
	}, nil
}

// --- Helper Functions ---

// findDomain finds a libvirt domain by UUID or Name
func (m *KVMManager) findDomain(identifier string) (libvirt.Domain, error) {
	// Try lookup by UUID first
	parsedUUID, err := uuid.Parse(identifier)
	if err == nil {
		// Convert uuid.UUID to libvirt.UUID
		var libvirtUUID libvirt.UUID
		copy(libvirtUUID[:], parsedUUID[:])
		domain, errLookup := m.conn.DomainLookupByUUID(libvirtUUID)
		if errLookup == nil {
			return domain, nil // Found by UUID
		}
		log.Printf("Domain lookup by UUID %s failed (may not exist or error): %v", identifier, errLookup)
	} else {
		log.Printf("Identifier '%s' is not a valid UUID: %v. Trying lookup by name.", identifier, err)
	}
	// If not found by UUID or identifier is not a UUID, try by name
	domain, err := m.conn.DomainLookupByName(identifier)
	if err != nil {
		return libvirt.Domain{}, fmt.Errorf("failed to find domain by UUID or Name '%s': %w", identifier, err)
	}
	return domain, nil
}

// generateDomainXML generates a libvirt domain XML definition from a VMConfig
func generateDomainXML(cfg vm.VMConfig) (string, error) {
	type Disk struct {
		Type   string `xml:"type,attr"`
		Device string `xml:"device,attr"`
		Driver struct {
			Name string `xml:"name,attr"`
			Type string `xml:"type,attr"`
		} `xml:"driver"`
		Source struct {
			File string `xml:"file,attr"`
		} `xml:"source"`
		Target struct {
			Dev string `xml:"dev,attr"`
			Bus string `xml:"bus,attr"`
		} `xml:"target"`
	}
	type Interface struct {
		Type   string `xml:"type,attr"`
		Source struct {
			Network string `xml:"network,attr"`
		} `xml:"source"`
		Model struct {
			Type string `xml:"type,attr"`
		} `xml:"model"`
	}
	type Domain struct {
		XMLName xml.Name `xml:"domain"`
		Type    string   `xml:"type,attr"`
		Name    string   `xml:"name"`
		UUID    string   `xml:"uuid,omitempty"` // Add UUID
		Memory  struct {
			Unit  string `xml:"unit,attr"`
			Value int    `xml:",chardata"`
		} `xml:"memory"`
		VCPU struct {
			Placement string `xml:"placement,attr,omitempty"`
			Value     int    `xml:",chardata"`
		} `xml:"vcpu"`
		OS struct {
			Type struct {
				Arch    string `xml:"arch,attr"`
				Machine string `xml:"machine,attr,omitempty"`
				Value   string `xml:",chardata"`
			} `xml:"type"`
			Boot struct {
				Dev string `xml:"dev,attr"`
			} `xml:"boot,omitempty"`
		} `xml:"os"`
		Features struct {
			ACPI   *struct{} `xml:"acpi"`
			APIC   *struct{} `xml:"apic"`
			VMport *struct{} `xml:"vmport,omitempty"`
		} `xml:"features"`
		Clock struct {
			Offset string `xml:"offset,attr"`
			Timer  []struct {
				Name       string `xml:"name,attr"`
				TickPolicy string `xml:"tickpolicy,attr,omitempty"`
				Present    string `xml:"present,attr,omitempty"`
			} `xml:"timer"`
		} `xml:"clock"`
		OnPoweroff string `xml:"on_poweroff,omitempty"`
		OnReboot   string `xml:"on_reboot,omitempty"`
		OnCrash    string `xml:"on_crash,omitempty"`
		Devices    struct {
			Emulator   string      `xml:"emulator,omitempty"`
			Disks      []Disk      `xml:"disk"`
			Interfaces []Interface `xml:"interface"`
			Serials    []struct {
				Type   string `xml:"type,attr"`
				Target *struct {
					Type string `xml:"type,attr,omitempty"`
					Port *int   `xml:"port,attr"`
				} `xml:"target,omitempty"`
			} `xml:"serial"`
			Consoles []struct {
				Type   string `xml:"type,attr"`
				Target *struct {
					Type string `xml:"type,attr,omitempty"`
					Port *int   `xml:"port,attr"`
				} `xml:"target,omitempty"`
			} `xml:"console"`
			Channels []struct {
				Type   string `xml:"type,attr"`
				Target *struct {
					Type string `xml:"type,attr"`
					Name string `xml:"name,attr"`
				} `xml:"target"`
			} `xml:"channel"`
			MemBalloon *struct {
				Model string `xml:"model,attr"`
			} `xml:"memballoon,omitempty"`
		} `xml:"devices"`
	}

	d := Domain{
		Type: "kvm",
		Name: cfg.Name,
		UUID: cfg.ID, // Use config ID as UUID
	}
	d.Memory.Unit = "MiB"
	d.Memory.Value = cfg.MemoryMB
	d.VCPU.Placement = "static"
	d.VCPU.Value = cfg.CPUShares // Use CPUShares from config
	d.OS.Type.Arch = "x86_64"    // Assuming x86_64
	d.OS.Type.Machine = "pc-q35-latest"
	d.OS.Type.Value = "hvm"
	d.OS.Boot.Dev = "hd"

	d.Features.ACPI = &struct{}{}
	d.Features.APIC = &struct{}{}
	d.Features.VMport = &struct{}{}

	d.Clock.Offset = "utc"
	d.Clock.Timer = []struct {
		Name       string `xml:"name,attr"`
		TickPolicy string `xml:"tickpolicy,attr,omitempty"`
		Present    string `xml:"present,attr,omitempty"`
	}{
		{Name: "rtc", TickPolicy: "catchup"},
		{Name: "pit", TickPolicy: "delay"},
		{Name: "hpet", Present: "no"},
	}

	d.OnPoweroff = "destroy"
	d.OnReboot = "restart"
	d.OnCrash = "destroy"

	d.Devices.Emulator = "/usr/bin/qemu-system-x86_64" // Make configurable

	// Disk
	disk := Disk{
		Type:   "file",
		Device: "disk",
	}
	disk.Driver.Name = "qemu"
	disk.Driver.Type = "qcow2"    // Assuming qcow2
	disk.Source.File = cfg.RootFS // Use RootFS from config
	disk.Target.Dev = "vda"
	disk.Target.Bus = "virtio"
	d.Devices.Disks = []Disk{disk}

	// Add additional mounts
	for i, mount := range cfg.Mounts {
		addDisk := Disk{
			Type:   "file",
			Device: "disk",
		}
		addDisk.Driver.Name = "qemu"
		addDisk.Driver.Type = "qcow2" // Assuming qcow2, adjust if needed
		addDisk.Source.File = mount.Source
		addDisk.Target.Dev = fmt.Sprintf("vd%c", 'b'+i)
		addDisk.Target.Bus = "virtio"
		d.Devices.Disks = append(d.Devices.Disks, addDisk)
	}

	// Network Interface
	iface := Interface{
		Type: "network",
	}
	if cfg.NetworkID != "" {
		iface.Source.Network = cfg.NetworkID
	} else {
		iface.Source.Network = "default"
	}
	iface.Model.Type = "virtio"
	d.Devices.Interfaces = []Interface{iface}

	// Serial Console
	serialPort := 0
	d.Devices.Serials = append(d.Devices.Serials, struct {
		Type   string `xml:"type,attr"`
		Target *struct {
			Type string `xml:"type,attr,omitempty"`
			Port *int   `xml:"port,attr"`
		} `xml:"target,omitempty"`
	}{
		Type: "pty",
		Target: &struct {
			Type string `xml:"type,attr,omitempty"`
			Port *int   `xml:"port,attr"`
		}{Port: &serialPort},
	})

	// Console Device
	consolePort := 0
	d.Devices.Consoles = append(d.Devices.Consoles, struct {
		Type   string `xml:"type,attr"`
		Target *struct {
			Type string `xml:"type,attr,omitempty"`
			Port *int   `xml:"port,attr"`
		} `xml:"target,omitempty"`
	}{
		Type: "pty",
		Target: &struct {
			Type string `xml:"type,attr,omitempty"`
			Port *int   `xml:"port,attr"`
		}{Type: "serial", Port: &consolePort},
	})

	// Virtio RNG Channel
	d.Devices.Channels = append(d.Devices.Channels, struct {
		Type   string `xml:"type,attr"`
		Target *struct {
			Type string `xml:"type,attr"`
			Name string `xml:"name,attr"`
		} `xml:"target"`
	}{
		Type: "unix",
		Target: &struct {
			Type string `xml:"type,attr"`
			Name string `xml:"name,attr"`
		}{Type: "virtio", Name: "org.qemu.guest_agent.0"},
	})

	// Memory Balloon
	d.Devices.MemBalloon = &struct {
		Model string `xml:"model,attr"`
	}{Model: "virtio"}

	var buf bytes.Buffer
	enc := xml.NewEncoder(&buf)
	enc.Indent("", "  ")
	if err := enc.Encode(d); err != nil {
		return "", fmt.Errorf("failed to encode domain XML: %w", err)
	}
	return buf.String(), nil
}

// mapLibvirtState maps libvirt.DomainState to vm.State
func mapLibvirtState(state libvirt.DomainState) vm.State {
	switch state {
	case libvirt.DomainRunning:
		return vm.StateRunning
	case libvirt.DomainShutoff, libvirt.DomainShutdown, libvirt.DomainPaused, libvirt.DomainPmsuspended:
		// Group stopped/paused states
		return vm.StateStopped
	case libvirt.DomainCrashed, libvirt.DomainNostate: // Treat NoState as Failed
		return vm.StateFailed
	case libvirt.DomainBlocked:
		// Decide how to map Blocked - could be Running or a specific Blocked state if added to vm.State
		return vm.StateRunning // Mapping Blocked to Running for now
	default:
		log.Printf("Warning: Unhandled libvirt domain state: %v", state)
		return vm.StateFailed // Default unknown states to Failed
	}
}

// Helper function to convert C array pointer to Go byte slice (use with caution)
// This might not be needed if the library handles conversions appropriately.
func goBytes(cArray *byte, size int) []byte {
	return (*[1 << 30]byte)(unsafe.Pointer(cArray))[:size:size]
}
