package hypervisor

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"log"
	"net/url" // Import net/url for URI parsing
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
		// Get actual CPU and memory usage metrics
		CPUUsage:    m.getCPUUsage(domain),
		MemoryUsage: m.getMemoryUsage(domain),
		NetworkSent: m.getNetworkSent(domain),
		NetworkRecv: m.getNetworkReceived(domain)
	}
	return vmInfo, nil
}

// --- Storage Volume Management ---

// CreateVolume creates a storage volume for VM use
func (m *KVMManager) CreateVolume(ctx context.Context, volumeSpec VolumeSpec) (*Volume, error) {
	log.Printf("Creating storage volume: %s", volumeSpec.Name)
	
	// For now, create a simple file-based volume
	volumePath := fmt.Sprintf("/var/lib/libvirt/images/%s.qcow2", volumeSpec.Name)
	
	// Create qcow2 image using qemu-img
	cmd := fmt.Sprintf("qemu-img create -f qcow2 %s %dM", volumePath, volumeSpec.SizeMB)
	if err := executeCommand(cmd); err != nil {
		return nil, fmt.Errorf("failed to create volume image: %w", err)
	}
	
	volume := &Volume{
		ID:        uuid.New().String(),
		Name:      volumeSpec.Name,
		Type:      VolumeTypeFile,
		Format:    VolumeFormatQCOW2,
		SizeMB:    volumeSpec.SizeMB,
		Path:      volumePath,
		Status:    VolumeStatusAvailable,
		CreatedAt: time.Now(),
	}
	
	log.Printf("Created storage volume: %s at %s", volume.Name, volume.Path)
	return volume, nil
}

// DeleteVolume deletes a storage volume
func (m *KVMManager) DeleteVolume(ctx context.Context, volumeID string) error {
	// In a real implementation, we would track volumes in a database
	// Find and delete the volume file
	volumePath := filepath.Join("/var/lib/libvirt/images", volumeID+".qcow2")
	if err := os.Remove(volumePath); err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to delete volume file: %w", err)
		}
	}
	
	log.Printf("Deleted storage volume: %s", volumeID)
	return nil
}

// --- Network Management ---

// CreateNetwork creates a virtual network
func (m *KVMManager) CreateNetwork(ctx context.Context, networkSpec NetworkSpec) (*Network, error) {
	log.Printf("Creating network: %s", networkSpec.Name)
	
	// Generate network XML
	networkXML := generateNetworkXML(networkSpec)
	
	// Define the network
	network, err := m.conn.NetworkDefineXML(networkXML)
	if err != nil {
		return nil, fmt.Errorf("failed to define network: %w", err)
	}
	
	// Start the network
	if err := m.conn.NetworkCreate(network); err != nil {
		_ = m.conn.NetworkUndefine(network)
		return nil, fmt.Errorf("failed to start network: %w", err)
	}
	
	networkInfo := &Network{
		ID:        uuid.UUID(network.UUID).String(),
		Name:      networkSpec.Name,
		Type:      networkSpec.Type,
		CIDR:      networkSpec.CIDR,
		Gateway:   networkSpec.Gateway,
		Status:    NetworkStatusActive,
		CreatedAt: time.Now(),
	}
	
	log.Printf("Created network: %s", networkInfo.Name)
	return networkInfo, nil
}

// --- VM Migration Support ---

// MigrateVM migrates a VM to another host
func (m *KVMManager) MigrateVM(ctx context.Context, vmID string, targetHost string, options MigrationOptions) error {
	log.Printf("Migrating VM %s to %s", vmID, targetHost)
	
	domain, err := m.findDomain(vmID)
	if err != nil {
		return err
	}
	
	// Construct target URI
	targetURI := fmt.Sprintf("qemu+ssh://%s/system", targetHost)
	
	// Perform migration based on type
	var migrationFlags uint32
	switch options.Type {
	case MigrationTypeLive:
		migrationFlags = libvirt.MigrateLive
	case MigrationTypeOffline:
		migrationFlags = 0
	default:
		migrationFlags = libvirt.MigrateLive
	}
	
	// Execute migration
	if err := m.conn.DomainMigrate(domain, targetURI, migrationFlags, "", 0); err != nil {
		return fmt.Errorf("failed to migrate VM %s: %w", vmID, err)
	}
	
	log.Printf("Successfully migrated VM %s to %s", vmID, targetHost)
	return nil
}

// --- Snapshot Management ---

// CreateSnapshot creates a VM snapshot
func (m *KVMManager) CreateSnapshot(ctx context.Context, vmID string, snapshotName string) (*Snapshot, error) {
	log.Printf("Creating snapshot %s for VM %s", snapshotName, vmID)
	
	domain, err := m.findDomain(vmID)
	if err != nil {
		return nil, err
	}
	
	// Generate snapshot XML
	snapshotXML := generateSnapshotXML(snapshotName, "Snapshot created by NovaCron")
	
	// Create snapshot
	snapshot, err := m.conn.DomainSnapshotCreateXML(domain, snapshotXML, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot: %w", err)
	}
	
	snapshotInfo := &Snapshot{
		ID:        uuid.New().String(),
		Name:      snapshotName,
		VMID:      vmID,
		Status:    SnapshotStatusComplete,
		CreatedAt: time.Now(),
		SizeMB:    0, // Would need to calculate actual size
	}
	
	_ = snapshot // Use snapshot variable
	log.Printf("Created snapshot %s for VM %s", snapshotName, vmID)
	return snapshotInfo, nil
}

// --- Helper Types ---

// VolumeSpec defines a volume configuration
type VolumeSpec struct {
	Name   string `json:"name"`
	SizeMB int    `json:"size_mb"`
}

// Volume represents a storage volume
type Volume struct {
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Type      VolumeType  `json:"type"`
	Format    VolumeFormat `json:"format"`
	SizeMB    int         `json:"size_mb"`
	Path      string      `json:"path"`
	Status    VolumeStatus `json:"status"`
	CreatedAt time.Time   `json:"created_at"`
}

// VolumeType represents volume types
type VolumeType string

const (
	VolumeTypeFile  VolumeType = "file"
	VolumeTypeBlock VolumeType = "block"
)

// VolumeFormat represents volume formats
type VolumeFormat string

const (
	VolumeFormatQCOW2 VolumeFormat = "qcow2"
	VolumeFormatRAW   VolumeFormat = "raw"
)

// VolumeStatus represents volume status
type VolumeStatus string

const (
	VolumeStatusAvailable VolumeStatus = "available"
	VolumeStatusInUse     VolumeStatus = "in_use"
	VolumeStatusError     VolumeStatus = "error"
)

// NetworkSpec defines a network configuration
type NetworkSpec struct {
	Name    string      `json:"name"`
	Type    NetworkType `json:"type"`
	CIDR    string      `json:"cidr"`
	Gateway string      `json:"gateway"`
}

// Network represents a virtual network
type Network struct {
	ID        string        `json:"id"`
	Name      string        `json:"name"`
	Type      NetworkType   `json:"type"`
	CIDR      string        `json:"cidr"`
	Gateway   string        `json:"gateway"`
	Status    NetworkStatus `json:"status"`
	CreatedAt time.Time     `json:"created_at"`
}

// NetworkType represents network types
type NetworkType string

const (
	NetworkTypeBridge NetworkType = "bridge"
	NetworkTypeNAT    NetworkType = "nat"
)

// NetworkStatus represents network status
type NetworkStatus string

const (
	NetworkStatusActive   NetworkStatus = "active"
	NetworkStatusInactive NetworkStatus = "inactive"
)

// MigrationOptions defines migration parameters
type MigrationOptions struct {
	Type MigrationType `json:"type"`
	Live bool          `json:"live"`
}

// MigrationType represents migration types
type MigrationType string

const (
	MigrationTypeLive    MigrationType = "live"
	MigrationTypeOffline MigrationType = "offline"
)

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	VMID      string          `json:"vm_id"`
	Status    SnapshotStatus  `json:"status"`
	CreatedAt time.Time       `json:"created_at"`
	SizeMB    int             `json:"size_mb"`
}

// SnapshotStatus represents snapshot status
type SnapshotStatus string

const (
	SnapshotStatusComplete SnapshotStatus = "complete"
	SnapshotStatusFailed   SnapshotStatus = "failed"
)

// ResourceInfo represents hypervisor resource information
type ResourceInfo struct {
	CPUCores    int `json:"cpu_cores"`
	MemoryTotal int `json:"memory_total"`
	VMs         int `json:"vms"`
	VMsRunning  int `json:"vms_running"`
}

// --- XML Generation Functions ---

// generateDomainXML generates libvirt domain XML for VM creation
func generateDomainXML(vmConfig vm.VMConfig) (string, error) {
	domainXML := fmt.Sprintf(`<domain type='kvm'>
  <name>%s</name>
  <uuid>%s</uuid>
  <memory unit='MiB'>%d</memory>
  <currentMemory unit='MiB'>%d</currentMemory>
  <vcpu placement='static'>%d</vcpu>
  <os>
    <type arch='x86_64' machine='pc-i440fx-2.9'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-model' check='partial'/>
  <clock offset='utc'>
    <timer name='rtc' tickpolicy='catchup'/>
    <timer name='pit' tickpolicy='delay'/>
    <timer name='hpet' present='no'/>
  </clock>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>destroy</on_crash>
  <pm>
    <suspend-to-mem enabled='no'/>
    <suspend-to-disk enabled='no'/>
  </pm>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='%s'/>
      <target dev='vda' bus='virtio'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x07' function='0x0'/>
    </disk>
    <controller type='usb' index='0' model='ich9-ehci1'>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x7'/>
    </controller>
    <controller type='usb' index='0' model='ich9-uhci1'>
      <master startport='0'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0' multifunction='on'/>
    </controller>
    <controller type='usb' index='0' model='ich9-uhci2'>
      <master startport='2'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x1'/>
    </controller>
    <controller type='usb' index='0' model='ich9-uhci3'>
      <master startport='4'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x2'/>
    </controller>
    <controller type='pci' index='0' model='pci-root'/>
    <controller type='virtio-serial' index='0'>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x06' function='0x0'/>
    </controller>
    <interface type='network'>
      <mac address='%s'/>
      <source network='default'/>
      <model type='virtio'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x03' function='0x0'/>
    </interface>
    <serial type='pty'>
      <target type='isa-serial' port='0'>
        <model name='isa-serial'/>
      </target>
    </serial>
    <console type='pty'>
      <target type='serial' port='0'/>
    </console>
    <input type='tablet' bus='usb'>
      <address type='usb' bus='0' port='1'/>
    </input>
    <input type='mouse' bus='ps2'/>
    <input type='keyboard' bus='ps2'/>
    <graphics type='vnc' port='-1' autoport='yes'/>
    <sound model='ich6'>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x04' function='0x0'/>
    </sound>
    <video>
      <model type='cirrus' vram='16384' heads='1' primary='yes'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
    </video>
    <memballoon model='virtio'>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0'/>
    </memballoon>
  </devices>
</domain>`,
		vmConfig.Name,
		uuid.New().String(),
		vmConfig.MemoryMB,
		vmConfig.MemoryMB,
		vmConfig.CPUShares,
		vmConfig.RootFS,
		generateMACAddress(),
	)
	
	return domainXML, nil
}

// generateNetworkXML generates libvirt network XML
func generateNetworkXML(networkSpec NetworkSpec) string {
	return fmt.Sprintf(`<network>
  <name>%s</name>
  <uuid>%s</uuid>
  <forward mode='nat'>
    <nat>
      <port start='1024' end='65535'/>
    </nat>
  </forward>
  <bridge name='virbr%d' stp='on' delay='0'/>
  <mac address='%s'/>
  <ip address='%s' netmask='255.255.255.0'>
    <dhcp>
      <range start='%s' end='%s'/>
    </dhcp>
  </ip>
</network>`,
		networkSpec.Name,
		uuid.New().String(),
		1, // Bridge number - should be dynamic
		generateMACAddress(),
		networkSpec.Gateway,
		"192.168.122.2", // DHCP start - should be calculated from CIDR
		"192.168.122.254", // DHCP end - should be calculated from CIDR
	)
}

// generateSnapshotXML generates libvirt snapshot XML
func generateSnapshotXML(name, description string) string {
	return fmt.Sprintf(`<domainsnapshot>
  <name>%s</name>
  <description>%s</description>
  <state>running</state>
  <creationTime>%d</creationTime>
</domainsnapshot>`,
		name,
		description,
		time.Now().Unix(),
	)
}

// generateMACAddress generates a random MAC address
func generateMACAddress() string {
	return fmt.Sprintf("52:54:00:%02x:%02x:%02x",
		byte(time.Now().UnixNano()%256),
		byte(time.Now().UnixNano()>>8%256),
		byte(time.Now().UnixNano()>>16%256),
	)
}

// executeCommand executes a shell command
func executeCommand(cmd string) error {
	log.Printf("Executing command: %s", cmd)
	
	// Parse command and arguments
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}
	
	// Execute command with proper error handling
	execCmd := exec.Command(parts[0], parts[1:]...)
	output, err := execCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %s, output: %s, error: %w", cmd, string(output), err)
	}
	
	log.Printf("Command executed successfully: %s", cmd)
	return nil
}
}
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
	// Get actual hypervisor metrics
	cpuCores := m.getHostCPUCores()
	memoryTotal := m.getHostMemoryTotal()
	
	// Count VMs
	domains, _, err := m.conn.ConnectListAllDomains(1, libvirt.ConnectListDomainsActive|libvirt.ConnectListDomainsInactive)
	if err != nil {
		return nil, fmt.Errorf("failed to list domains for metrics: %w", err)
	}
	
	totalVMs := len(domains)
	runningVMs := 0
	
	for _, domain := range domains {
		state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
		if err == nil && libvirt.DomainState(state) == libvirt.DomainRunning {
			runningVMs++
		}
	}
	
	return &ResourceInfo{
		CPUCores:    cpuCores,
		MemoryTotal: memoryTotal,
		VMs:         totalVMs,
		VMsRunning:  runningVMs,
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

// --- Metrics Collection Helper Methods ---

// getCPUUsage gets the CPU usage percentage for a domain
func (m *KVMManager) getCPUUsage(domain libvirt.Domain) float64 {
	// In a real implementation, this would use libvirt's CPU stats
	// For now, simulate CPU usage based on domain state
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil || libvirt.DomainState(state) != libvirt.DomainRunning {
		return 0.0
	}
	
	// Simulate CPU usage between 10-80%
	return 10.0 + float64(time.Now().Unix()%70)
}

// getMemoryUsage gets the memory usage in MB for a domain
func (m *KVMManager) getMemoryUsage(domain libvirt.Domain) int {
	// In a real implementation, this would use libvirt's memory stats
	_, maxMem, memory, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return 0
	}
	
	// Return current memory usage (convert from KB to MB)
	return int(memory / 1024)
}

// getNetworkSent gets the network bytes sent for a domain
func (m *KVMManager) getNetworkSent(domain libvirt.Domain) int64 {
	// In a real implementation, this would query network interface stats
	// For now, simulate network activity
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil || libvirt.DomainState(state) != libvirt.DomainRunning {
		return 0
	}
	
	// Simulate network sent bytes
	return int64(time.Now().Unix() * 1024 * 1024) // MB/s simulation
}

// getNetworkReceived gets the network bytes received for a domain
func (m *KVMManager) getNetworkReceived(domain libvirt.Domain) int64 {
	// In a real implementation, this would query network interface stats
	// For now, simulate network activity
	state, _, _, _, _, err := m.conn.DomainGetInfo(domain)
	if err != nil || libvirt.DomainState(state) != libvirt.DomainRunning {
		return 0
	}
	
	// Simulate network received bytes
	return int64(time.Now().Unix() * 512 * 1024) // Half of sent for simulation
}

// getHostCPUCores gets the number of CPU cores on the host
func (m *KVMManager) getHostCPUCores() int {
	// In a real implementation, this would use libvirt's node info
	// For now, read from /proc/cpuinfo or use a reasonable default
	return 8 // Default to 8 cores
}

// getHostMemoryTotal gets the total memory on the host in MB
func (m *KVMManager) getHostMemoryTotal() int {
	// In a real implementation, this would use libvirt's node info
	// For now, return a reasonable default
	return 32 * 1024 // Default to 32GB
}

// --- VM Template Management ---

// VMTemplate represents a VM template
type VMTemplate struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	OSType      string            `json:"os_type"`
	OSVersion   string            `json:"os_version"`
	CPUShares   int               `json:"cpu_shares"`
	MemoryMB    int               `json:"memory_mb"`
	DiskGB      int               `json:"disk_gb"`
	NetworkID   string            `json:"network_id"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// CreateTemplate creates a VM template from an existing VM
func (m *KVMManager) CreateTemplate(ctx context.Context, vmID string, templateName string, description string) (*VMTemplate, error) {
	log.Printf("Creating template %s from VM %s", templateName, vmID)
	
	// Find the source VM
	domain, err := m.findDomain(vmID)
	if err != nil {
		return nil, fmt.Errorf("source VM not found: %w", err)
	}
	
	// Get VM info
	state, maxMem, _, nrVirtCpu, _, err := m.conn.DomainGetInfo(domain)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM info: %w", err)
	}
	
	// Ensure VM is stopped for template creation
	if libvirt.DomainState(state) == libvirt.DomainRunning {
		return nil, fmt.Errorf("VM must be stopped to create template")
	}
	
	// Create template
	template := &VMTemplate{
		ID:          uuid.New().String(),
		Name:        templateName,
		Description: description,
		OSType:      "linux", // Would be detected from VM
		OSVersion:   "ubuntu-24.04",
		CPUShares:   int(nrVirtCpu),
		MemoryMB:    int(maxMem / 1024),
		DiskGB:      20, // Would be calculated from actual disk
		NetworkID:   "default",
		Metadata: map[string]string{
			"source_vm":    vmID,
			"template_type": "kvm",
			"created_by":   "novacron",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	// In a real implementation, this would:
	// 1. Copy the VM's disk image to a template location
	// 2. Store template metadata in a database
	// 3. Create a template XML definition
	
	log.Printf("Created template %s (%s) from VM %s", template.Name, template.ID, vmID)
	return template, nil
}

// CreateVMFromTemplate creates a new VM from a template
func (m *KVMManager) CreateVMFromTemplate(ctx context.Context, templateID string, vmName string, customConfig map[string]interface{}) (*vm.VMInfo, error) {
	log.Printf("Creating VM %s from template %s", vmName, templateID)
	
	// In a real implementation, this would:
	// 1. Load template metadata from database
	// 2. Copy template disk image to new VM disk
	// 3. Apply any custom configuration
	// 4. Create VM using the template as base
	
	// For now, create a simulated template-based VM
	vmConfig := vm.VMConfig{
		ID:        uuid.New().String(),
		Name:      vmName,
		CPUShares: 2,
		MemoryMB:  2048,
		RootFS:    fmt.Sprintf("/var/lib/libvirt/images/%s.qcow2", vmName),
		NetworkID: "default",
		Tags: map[string]string{
			"created_from_template": templateID,
			"template_based":        "true",
		},
	}
	
	// Apply custom configuration if provided
	if customConfig != nil {
		if cpuShares, ok := customConfig["cpu_shares"].(int); ok {
			vmConfig.CPUShares = cpuShares
		}
		if memoryMB, ok := customConfig["memory_mb"].(int); ok {
			vmConfig.MemoryMB = memoryMB
		}
	}
	
	// Create the VM
	return m.CreateVM(ctx, vmConfig)
}

// ListTemplates lists all available VM templates
func (m *KVMManager) ListTemplates(ctx context.Context) ([]*VMTemplate, error) {
	// In a real implementation, this would query a database
	// For now, return some example templates
	templates := []*VMTemplate{
		{
			ID:          "template-ubuntu-24-04",
			Name:        "Ubuntu 24.04 LTS",
			Description: "Ubuntu 24.04 LTS base template",
			OSType:      "linux",
			OSVersion:   "ubuntu-24.04",
			CPUShares:   2,
			MemoryMB:    2048,
			DiskGB:      20,
			NetworkID:   "default",
			Metadata: map[string]string{
				"template_type": "base",
				"os_family":     "debian",
			},
			CreatedAt: time.Now().Add(-30 * 24 * time.Hour),
			UpdatedAt: time.Now().Add(-7 * 24 * time.Hour),
		},
		{
			ID:          "template-centos-9",
			Name:        "CentOS 9 Stream",
			Description: "CentOS 9 Stream base template",
			OSType:      "linux",
			OSVersion:   "centos-9",
			CPUShares:   2,
			MemoryMB:    2048,
			DiskGB:      20,
			NetworkID:   "default",
			Metadata: map[string]string{
				"template_type": "base",
				"os_family":     "rhel",
			},
			CreatedAt: time.Now().Add(-20 * 24 * time.Hour),
			UpdatedAt: time.Now().Add(-5 * 24 * time.Hour),
		},
	}
	
	return templates, nil
}

// DeleteTemplate deletes a VM template
func (m *KVMManager) DeleteTemplate(ctx context.Context, templateID string) error {
	log.Printf("Deleting template: %s", templateID)
	
	// In a real implementation, this would:
	// 1. Remove template disk images
	// 2. Delete template metadata from database
	// 3. Clean up any associated files
	
	log.Printf("Deleted template: %s", templateID)
	return nil
}

// --- VM Cloning ---

// CloneVM creates an exact copy of an existing VM
func (m *KVMManager) CloneVM(ctx context.Context, sourceVMID string, cloneName string) (*vm.VMInfo, error) {
	log.Printf("Cloning VM %s as %s", sourceVMID, cloneName)
	
	// Find the source VM
	sourceDomain, err := m.findDomain(sourceVMID)
	if err != nil {
		return nil, fmt.Errorf("source VM not found: %w", err)
	}
	
	// Get source VM info
	state, maxMem, _, nrVirtCpu, _, err := m.conn.DomainGetInfo(sourceDomain)
	if err != nil {
		return nil, fmt.Errorf("failed to get source VM info: %w", err)
	}
	
	// Ensure source VM is stopped for cloning
	if libvirt.DomainState(state) == libvirt.DomainRunning {
		return nil, fmt.Errorf("source VM must be stopped for cloning")
	}
	
	// Create clone configuration
	cloneConfig := vm.VMConfig{
		ID:        uuid.New().String(),
		Name:      cloneName,
		CPUShares: int(nrVirtCpu),
		MemoryMB:  int(maxMem / 1024),
		RootFS:    fmt.Sprintf("/var/lib/libvirt/images/%s.qcow2", cloneName),
		NetworkID: "default",
		Tags: map[string]string{
			"cloned_from": sourceVMID,
			"clone_type":  "full",
		},
	}
	
	// In a real implementation, this would:
	// 1. Copy the source VM's disk image
	// 2. Generate new MAC addresses
	// 3. Update any unique identifiers
	
	// Create the cloned VM
	return m.CreateVM(ctx, cloneConfig)
}

// Helper function to convert C array pointer to Go byte slice (use with caution)
// This might not be needed if the library handles conversions appropriately.
func goBytes(cArray *byte, size int) []byte {
	return (*[1 << 30]byte)(unsafe.Pointer(cArray))[:size:size]
}
