package hypervisor

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"log"
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
		uri = "qemu:///system" // Default system connection
	}
	// Use NewConnect for standard URI connection
	l, err := libvirt.NewConnect(uri)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to libvirt URI %s: %w", uri, err)
	}
	log.Printf("Connected to libvirt at %s", uri)
	return &KVMManager{
		libvirtURI: uri,
		conn:       l, // Assign the connection pointer
	}, nil
}

// Close closes the connection to libvirt
func (m *KVMManager) Close() error {
	if m.conn != nil {
		if err := m.conn.Disconnect(); err != nil {
			// Check if it's already disconnected
			if err != libvirt.ErrDisconnected {
				return fmt.Errorf("failed to disconnect from libvirt: %w", err)
			}
		}
		log.Println("Disconnected from libvirt")
		m.conn = nil // Nil out the connection
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
	// DomainDefineXML returns *Domain
	domain, err := m.conn.DomainDefineXML(xmlDef)
	if err != nil {
		return nil, fmt.Errorf("failed to define domain: %w", err)
	}
	// DomainCreate takes *Domain
	if err := m.conn.DomainCreate(*domain); err != nil { // Pass value
		// Attempt cleanup if start fails
		_ = m.conn.DomainUndefine(*domain) // Undefine takes Domain value
		return nil, fmt.Errorf("failed to start domain: %w", err)
	}
	// DomainGetInfo takes Domain value, returns multiple values
	state, maxMem, _, nrVirtCpu, _, err := m.conn.DomainGetInfo(*domain) // Pass value, ignore memory and cpuTime for now
	if err != nil {
		// Attempt cleanup if get info fails
		_ = m.conn.DomainDestroy(*domain)  // Destroy takes Domain value
		_ = m.conn.DomainUndefine(*domain) // Undefine takes Domain value
		return nil, fmt.Errorf("failed to get domain info: %w", err)
	}

	// Get UUID
	domainUUIDBytes, err := m.conn.DomainGetUUID(*domain) // Pass value
	domainUUID := uuid.UUID(domainUUIDBytes).String()
	if err != nil {
		log.Printf("Warning: failed to get UUID for domain %s: %v", vmConfig.Name, err)
		domainUUID = "unknown-uuid-" + vmConfig.Name // Use name as part of fallback
	}

	// Construct VMInfo based on available data
	vmInfo := vm.VMInfo{
		ID:        domainUUID, // Use UUID as ID
		Name:      vmConfig.Name,
		State:     mapLibvirtState(libvirt.DomainState(state)), // Cast state
		CPUShares: int(nrVirtCpu),                              // Use nrVirtCpu
		MemoryMB:  int(maxMem / 1024),                          // Use maxMem (assuming KiB)
		CreatedAt: time.Now(),                                  // Placeholder
		RootFS:    vmConfig.RootFS,
		Tags:      vmConfig.Tags,
	}

	return &vmInfo, nil
}

// DeleteVM deletes a KVM virtual machine
func (m *KVMManager) DeleteVM(ctx context.Context, vmID string) error {
	domain, err := m.findDomain(vmID) // findDomain returns *Domain
	if err != nil {
		return err // Domain not found
	}
	// DomainGetInfo takes Domain value
	stateVal, _, _, _, _, errInfo := m.conn.DomainGetInfo(*domain) // Pass value
	if errInfo == nil && libvirt.DomainState(stateVal) == libvirt.DomainRunning {
		// DomainDestroy takes Domain value
		if err := m.conn.DomainDestroy(*domain); err != nil {
			log.Printf("Warning: failed to destroy domain %s before undefining: %v", vmID, err)
		}
	} else if errInfo != nil {
		log.Printf("Warning: failed to get domain info for %s before undefining: %v", vmID, errInfo)
	}

	// DomainUndefineFlags takes Domain value
	if err := m.conn.DomainUndefineFlags(*domain, libvirt.DomainUndefineManagedSave); err != nil {
		if errSnap := m.conn.DomainUndefineFlags(*domain, libvirt.DomainUndefineSnapshotsMetadata); errSnap != nil {
			log.Printf("Warning: failed to undefine snapshots metadata for domain %s: %v", vmID, errSnap)
		}
		return fmt.Errorf("failed to undefine domain %s: %w", vmID, err)
	}

	log.Printf("Deleted KVM VM: %s", vmID)
	return nil
}

// StartVM starts a KVM virtual machine
func (m *KVMManager) StartVM(ctx context.Context, vmID string) error {
	domain, err := m.findDomain(vmID) // findDomain returns *Domain
	if err != nil {
		return err
	}
	// DomainGetInfo takes Domain value
	stateVal, _, _, _, _, errInfo := m.conn.DomainGetInfo(*domain) // Pass value
	if errInfo != nil {
		return fmt.Errorf("failed to get domain info for %s: %w", vmID, errInfo)
	}
	if libvirt.DomainState(stateVal) == libvirt.DomainRunning {
		log.Printf("VM %s is already running", vmID)
		return nil // Already running
	}
	// DomainCreate takes Domain value
	if err := m.conn.DomainCreate(*domain); err != nil { // Pass value
		return fmt.Errorf("failed to start domain %s: %w", vmID, err)
	}
	log.Printf("Started KVM VM: %s", vmID)
	return nil
}

// StopVM stops a KVM virtual machine
func (m *KVMManager) StopVM(ctx context.Context, vmID string, force bool) error {
	domain, err := m.findDomain(vmID) // findDomain returns *Domain
	if err != nil {
		return err
	}
	// DomainGetInfo takes Domain value
	stateVal, _, _, _, _, errInfo := m.conn.DomainGetInfo(*domain) // Pass value
	if errInfo != nil {
		return fmt.Errorf("failed to get domain info for %s: %w", vmID, errInfo)
	}
	if libvirt.DomainState(stateVal) == libvirt.DomainShutoff {
		log.Printf("VM %s is already stopped", vmID)
		return nil // Already stopped
	}

	if force {
		// DomainDestroy takes Domain value
		if err := m.conn.DomainDestroy(*domain); err != nil { // Pass value
			return fmt.Errorf("failed to force stop (destroy) domain %s: %w", vmID, err)
		}
		log.Printf("Force stopped KVM VM: %s", vmID)
	} else {
		// DomainShutdown takes Domain value
		if err := m.conn.DomainShutdown(*domain); err != nil { // Pass value
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
	// DomainGetInfo takes Domain value
	stateVal, _, _, _, _, errInfo := m.conn.DomainGetInfo(*domain) // Pass value
	if errInfo != nil {
		return vm.StateFailed, fmt.Errorf("failed to get domain info for %s: %w", vmID, errInfo)
	}
	return mapLibvirtState(libvirt.DomainState(stateVal)), nil
}

// ListVMs lists all KVM virtual machines managed by this hypervisor
func (m *KVMManager) ListVMs(ctx context.Context) ([]vm.VMInfo, error) {
	// Use ConnectListAllDomains to get both active and inactive
	domains, _, err := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive|libvirt.ConnectListDomainsInactive) // Correct flags
	if err != nil {
		return nil, fmt.Errorf("failed to list all domains: %w", err)
	}
	defer func() { // Ensure domains are freed
		for i := range domains {
			_ = m.conn.DomainFree(domains[i])
		}
	}()

	var vms []vm.VMInfo
	for _, dom := range domains {
		// DomainGetInfo takes Domain value
		stateVal, maxMem, _, nrVirtCpu, _, errInfo := m.conn.DomainGetInfo(dom) // Pass value
		if errInfo != nil {
			log.Printf("Warning: failed to get info for domain ID %d: %v", dom.ID, errInfo)
			continue
		}
		name, err := m.conn.DomainGetName(dom) // DomainGetName takes Domain value
		if err != nil {
			log.Printf("Warning: failed to get name for domain ID %d: %v", dom.ID, err)
			name = fmt.Sprintf("domain-%d", dom.ID)
		}
		domainUUIDBytes, err := m.conn.DomainGetUUID(dom) // DomainGetUUID takes Domain value
		domainUUID := uuid.UUID(domainUUIDBytes).String()
		if err != nil {
			log.Printf("Warning: failed to get UUID for domain %s: %v", name, err)
			domainUUID = fmt.Sprintf("unknown-uuid-%d", dom.ID)
		}

		vms = append(vms, vm.VMInfo{
			ID:        domainUUID,
			Name:      name,
			State:     mapLibvirtState(libvirt.DomainState(stateVal)),
			CPUShares: int(nrVirtCpu),
			MemoryMB:  int(maxMem / 1024),
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
	// DomainGetInfo takes Domain value
	stateVal, maxMem, _, nrVirtCpu, _, errInfo := m.conn.DomainGetInfo(*domain) // Pass value
	if errInfo != nil {
		return nil, fmt.Errorf("failed to get domain info for %s: %w", vmID, errInfo)
	}

	// Get Memory Stats
	// DomainMemoryStats takes Domain value
	memStats, err := m.conn.DomainMemoryStats(*domain, uint32(libvirt.DomainMemoryStatNr), 0) // Pass value
	memUsage := int64(0)
	if err == nil {
		for _, stat := range memStats {
			if stat.Tag == int32(libvirt.DomainMemoryStatActualBalloon) {
				memUsage = int64(stat.Val * 1024) // Assuming KiB -> Bytes
				break
			}
			if stat.Tag == int32(libvirt.DomainMemoryStatRss) && memUsage == 0 {
				memUsage = int64(stat.Val * 1024)
			}
		}
	} else {
		log.Printf("Warning: failed to get memory stats for %s: %v", vmID, err)
	}

	// Get Name and UUID
	name, _ := m.conn.DomainGetName(*domain)            // Pass value
	domainUUIDBytes, _ := m.conn.DomainGetUUID(*domain) // Pass value
	domainUUID := uuid.UUID(domainUUIDBytes).String()

	vmInfo := &vm.VMInfo{
		ID:          domainUUID, // Use UUID
		Name:        name,
		State:       mapLibvirtState(libvirt.DomainState(stateVal)),
		CPUShares:   int(nrVirtCpu),
		MemoryMB:    int(maxMem / 1024),
		MemoryUsage: memUsage,
		// Add other stats (CPU, Disk, Network) if possible and needed
	}

	return vmInfo, nil
}

// GetHypervisorMetrics retrieves performance metrics for the KVM host
func (m *KVMManager) GetHypervisorMetrics(ctx context.Context) (*ResourceInfo, error) {
	// Use ConnectGetNodeInfo
	nodeInfo, err := m.conn.ConnectGetNodeInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get node info: %w", err)
	}

	// Get VM counts using ConnectListAllDomains
	allDomains, numDomains, err := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive|libvirt.ConnectListDomainsInactive)
	if err != nil {
		log.Printf("Warning: failed to get all domains count: %v", err)
		numDomains = 0 // Default to 0 on error
	}
	activeDomains, numActive, err := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive)
	if err != nil {
		log.Printf("Warning: failed to get active domains count: %v", err)
		numActive = 0 // Default to 0 on error
	}
	// Free the domain lists
	for i := range allDomains {
		_ = m.conn.DomainFree(allDomains[i])
	}
	for i := range activeDomains {
		_ = m.conn.DomainFree(activeDomains[i])
	}

	return &ResourceInfo{
		CPUCores:    int(nodeInfo.Cpus),
		MemoryTotal: int64(nodeInfo.Memory * 1024), // Assuming Memory is in KiB
		VMs:         int(numDomains),
		VMsRunning:  int(numActive),
		// Other fields like MemoryUsed, CPUUsage, Disk, Network require host-level collection
	}, nil
}

// --- Helper Functions ---

// findDomain finds a libvirt domain by UUID or Name
func (m *KVMManager) findDomain(identifier string) (*libvirt.Domain, error) {
	// Try lookup by UUID first - Use ConnectLookupByUUIDString
	parsedUUID := uuid.Parse(identifier)
	if parsedUUID != uuid.Nil {
		// ConnectLookupByUUID takes []byte
		domain, err := m.conn.ConnectLookupByUUID(parsedUUID[:])
		if err == nil {
			return domain, nil // Found by UUID
		}
		// Log if UUID lookup failed for a valid UUID, but continue to try by name
		log.Printf("Domain lookup by UUID %s failed (may not exist or error): %v", identifier, err)
	}

	// If not found by UUID or identifier is not a UUID, try by name
	// Use ConnectLookupByName
	domain, err := m.conn.ConnectLookupByName(identifier)
	if err != nil {
		return nil, fmt.Errorf("failed to find domain by UUID or Name '%s': %w", identifier, err)
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
	d.VCPU.Value = cfg.CPUShares // Using CPUShares as vCPU count
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
	disk.Driver.Type = "qcow2"
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
	// Map libvirt states to defined vm.State constants
	case libvirt.DomainRunning:
		return vm.StateRunning
	case libvirt.DomainShutoff:
		return vm.StateStopped
	case libvirt.DomainCrashed:
		return vm.StateFailed
	// Add mappings for other relevant states if needed, defaulting others
	case libvirt.DomainNostate, libvirt.DomainBlocked, libvirt.DomainPaused, libvirt.DomainShutdown, libvirt.DomainPmsuspended:
		// Decide how to map these intermediate/uncommon states
		// Mapping Paused/Suspended/Shutdown to Stopped might be reasonable
		// Mapping Blocked to Running might be reasonable
		// Mapping NoState to Failed or Stopped
		return vm.StateStopped // Example: Mapping paused/suspended/shutdown to stopped
	default:
		return vm.StateFailed // Default to Failed for unknown states
	}
}

// Helper function to convert C array pointer to Go byte slice (use with caution)
func goBytes(cArray *byte, size int) []byte {
	return (*[1 << 30]byte)(unsafe.Pointer(cArray))[:size:size]
}
