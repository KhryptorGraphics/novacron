package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// HardwareVirtualization provides hardware-specific virtualization features
type HardwareVirtualization struct {
	cpuFeatures      *CPUFeatures
	iommuEnabled     bool
	sriovDevices     map[string]*SRIOVDevice
	gpuDevices       map[string]*GPUDevice
	numaTopology     *SystemNUMATopology
	hardwareLock     sync.RWMutex
	lastUpdate       time.Time
	cacheValid       time.Duration
}

// CPUFeatures represents CPU virtualization features
type CPUFeatures struct {
	VTx              bool     `json:"vtx"`              // Intel VT-x
	AMDV             bool     `json:"amdv"`             // AMD-V
	EPT              bool     `json:"ept"`              // Extended Page Tables
	NPT              bool     `json:"npt"`              // Nested Page Tables
	VPID             bool     `json:"vpid"`             // Virtual Processor ID
	ASID             bool     `json:"asid"`             // Address Space ID
	AES              bool     `json:"aes"`              // AES-NI
	AVX              bool     `json:"avx"`              // Advanced Vector Extensions
	AVX2             bool     `json:"avx2"`             // Advanced Vector Extensions 2
	TSX              bool     `json:"tsx"`              // Transactional Synchronization Extensions
	SGX              bool     `json:"sgx"`              // Software Guard Extensions
	SupportedFlags   []string `json:"supported_flags"`  // Raw CPU flags
	ModelName        string   `json:"model_name"`       // CPU model name
	Vendor           string   `json:"vendor"`           // CPU vendor
	Cores            int      `json:"cores"`            // Physical cores
	Threads          int      `json:"threads"`          // Logical threads
	MaxFrequency     int64    `json:"max_frequency"`    // Max frequency in MHz
	CacheL1          int64    `json:"cache_l1"`         // L1 cache size in KB
	CacheL2          int64    `json:"cache_l2"`         // L2 cache size in KB
	CacheL3          int64    `json:"cache_l3"`         // L3 cache size in KB
}

// SRIOVDevice represents an SR-IOV capable device
type SRIOVDevice struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	PCIAddress       string            `json:"pci_address"`
	Driver           string            `json:"driver"`
	VendorID         string            `json:"vendor_id"`
	DeviceID         string            `json:"device_id"`
	SubsystemVendor  string            `json:"subsystem_vendor"`
	SubsystemDevice  string            `json:"subsystem_device"`
	IOMMUGroup       int               `json:"iommu_group"`
	MaxVFs           int               `json:"max_vfs"`
	CurrentVFs       int               `json:"current_vfs"`
	AvailableVFs     int               `json:"available_vfs"`
	VirtualFunctions []VirtualFunction `json:"virtual_functions"`
	Capabilities     []string          `json:"capabilities"`
	LinkSpeed        string            `json:"link_speed,omitempty"`
	LinkWidth        string            `json:"link_width,omitempty"`
	PowerState       string            `json:"power_state"`
	Temperature      int               `json:"temperature,omitempty"`
	FirmwareVersion  string            `json:"firmware_version,omitempty"`
}

// VirtualFunction represents an SR-IOV virtual function
type VirtualFunction struct {
	ID         string `json:"id"`
	PCIAddress string `json:"pci_address"`
	Driver     string `json:"driver,omitempty"`
	MACAddress string `json:"mac_address,omitempty"`
	VLAN       int    `json:"vlan,omitempty"`
	InUse      bool   `json:"in_use"`
	AssignedTo string `json:"assigned_to,omitempty"`
	IOMMUGroup int    `json:"iommu_group"`
}

// SystemNUMATopology represents the system's NUMA topology
type SystemNUMATopology struct {
	Nodes           []NUMANodeInfo    `json:"nodes"`
	TotalNodes      int               `json:"total_nodes"`
	TotalCPUs       int               `json:"total_cpus"`
	TotalMemoryMB   int64             `json:"total_memory_mb"`
	InterconnectMap map[string]int    `json:"interconnect_map"` // Distance matrix
	HugePagesSupport bool              `json:"hugepages_support"`
	HugePageSizes   []int             `json:"hugepage_sizes"`   // In KB
}

// NUMANodeInfo represents detailed NUMA node information
type NUMANodeInfo struct {
	ID              int      `json:"id"`
	CPUs            []int    `json:"cpus"`
	MemoryMB        int64    `json:"memory_mb"`
	FreeMemoryMB    int64    `json:"free_memory_mb"`
	HugePagesTotal  int      `json:"hugepages_total"`
	HugePagesFree   int      `json:"hugepages_free"`
	HugePageSize    int      `json:"hugepage_size"` // In KB
	PCIDevices      []string `json:"pci_devices"`
	Distance        []int    `json:"distance"`      // Distance to other nodes
}

// GPUVirtualizationMode represents GPU virtualization modes
type GPUVirtualizationMode string

const (
	GPUModePassthrough GPUVirtualizationMode = "passthrough"
	GPUModeVGPU        GPUVirtualizationMode = "vgpu"
	GPUModeSRIOV       GPUVirtualizationMode = "sriov"
	GPUModeMdev        GPUVirtualizationMode = "mdev"
)

// VGPUProfile represents vGPU profile information
type VGPUProfile struct {
	Name          string `json:"name"`
	Description   string `json:"description"`
	FrameBuffer   int64  `json:"frame_buffer_mb"`
	MaxInstances  int    `json:"max_instances"`
	MaxResX       int    `json:"max_resolution_x"`
	MaxResY       int    `json:"max_resolution_y"`
	MaxDisplays   int    `json:"max_displays"`
	CUDAEnabled   bool   `json:"cuda_enabled"`
	OpenGLVersion string `json:"opengl_version"`
	DirectXVersion string `json:"directx_version"`
}

// NewHardwareVirtualization creates a new hardware virtualization manager
func NewHardwareVirtualization() (*HardwareVirtualization, error) {
	hv := &HardwareVirtualization{
		sriovDevices: make(map[string]*SRIOVDevice),
		gpuDevices:   make(map[string]*GPUDevice),
		cacheValid:   5 * time.Minute,
	}

	// Initialize hardware detection
	if err := hv.detectHardwareFeatures(); err != nil {
		log.Printf("Warning: Failed to detect hardware features: %v", err)
	}

	return hv, nil
}

// detectHardwareFeatures detects available hardware virtualization features
func (hv *HardwareVirtualization) detectHardwareFeatures() error {
	log.Printf("Detecting hardware virtualization features...")

	var err error

	// Detect CPU features
	if hv.cpuFeatures, err = hv.detectCPUFeatures(); err != nil {
		log.Printf("Warning: Failed to detect CPU features: %v", err)
	}

	// Detect IOMMU support
	hv.iommuEnabled = hv.detectIOMMU()

	// Detect SR-IOV devices
	if err := hv.detectSRIOVDevices(); err != nil {
		log.Printf("Warning: Failed to detect SR-IOV devices: %v", err)
	}

	// Detect GPU devices
	if err := hv.detectGPUDevices(); err != nil {
		log.Printf("Warning: Failed to detect GPU devices: %v", err)
	}

	// Detect NUMA topology
	if hv.numaTopology, err = hv.detectNUMATopology(); err != nil {
		log.Printf("Warning: Failed to detect NUMA topology: %v", err)
	}

	hv.lastUpdate = time.Now()
	return nil
}

// detectCPUFeatures detects CPU virtualization features
func (hv *HardwareVirtualization) detectCPUFeatures() (*CPUFeatures, error) {
	features := &CPUFeatures{}

	// Read /proc/cpuinfo
	cpuinfoData, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to read /proc/cpuinfo: %w", err)
	}

	cpuinfo := string(cpuinfoData)
	lines := strings.Split(cpuinfo, "\n")

	for _, line := range lines {
		if strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) != 2 {
				continue
			}

			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			switch key {
			case "model name":
				if features.ModelName == "" {
					features.ModelName = value
				}
			case "vendor_id":
				if features.Vendor == "" {
					features.Vendor = value
				}
			case "cpu cores":
				if cores, err := strconv.Atoi(value); err == nil {
					features.Cores = cores
				}
			case "siblings":
				if threads, err := strconv.Atoi(value); err == nil {
					features.Threads = threads
				}
			case "cpu MHz":
				if freq, err := strconv.ParseFloat(value, 64); err == nil {
					features.MaxFrequency = int64(freq)
				}
			case "cache size":
				// Parse cache size (format: "8192 KB")
				if strings.Contains(value, "KB") {
					cacheSizeStr := strings.TrimSuffix(strings.TrimSpace(value), " KB")
					if cacheSize, err := strconv.ParseInt(cacheSizeStr, 10, 64); err == nil {
						// This is typically L3 cache in /proc/cpuinfo
						features.CacheL3 = cacheSize
					}
				}
			case "flags":
				features.SupportedFlags = strings.Fields(value)
				// Check for specific virtualization features
				for _, flag := range features.SupportedFlags {
					switch flag {
					case "vmx":
						features.VTx = true
					case "svm":
						features.AMDV = true
					case "ept":
						features.EPT = true
					case "npt":
						features.NPT = true
					case "vpid":
						features.VPID = true
					case "aes":
						features.AES = true
					case "avx":
						features.AVX = true
					case "avx2":
						features.AVX2 = true
					case "tsx":
						features.TSX = true
					case "sgx":
						features.SGX = true
					}
				}
			}
		}
	}

	return features, nil
}

// detectIOMMU detects IOMMU support
func (hv *HardwareVirtualization) detectIOMMU() bool {
	// Check for IOMMU in kernel command line
	if cmdlineData, err := os.ReadFile("/proc/cmdline"); err == nil {
		cmdline := string(cmdlineData)
		if strings.Contains(cmdline, "intel_iommu=on") || strings.Contains(cmdline, "amd_iommu=on") {
			// Also check if IOMMU groups exist
			if _, err := os.Stat("/sys/kernel/iommu_groups"); err == nil {
				return true
			}
		}
	}

	// Check IOMMU groups directory
	iommuGroupsDir := "/sys/kernel/iommu_groups"
	if info, err := os.Stat(iommuGroupsDir); err == nil && info.IsDir() {
		// Count IOMMU groups
		if entries, err := os.ReadDir(iommuGroupsDir); err == nil && len(entries) > 0 {
			return true
		}
	}

	return false
}

// detectSRIOVDevices detects SR-IOV capable devices
func (hv *HardwareVirtualization) detectSRIOVDevices() error {
	pciDevicesDir := "/sys/bus/pci/devices"
	
	entries, err := os.ReadDir(pciDevicesDir)
	if err != nil {
		return fmt.Errorf("failed to read PCI devices directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		devicePath := filepath.Join(pciDevicesDir, entry.Name())
		
		// Check if device supports SR-IOV
		sriovCapFile := filepath.Join(devicePath, "sriov_numvfs")
		if _, err := os.Stat(sriovCapFile); err != nil {
			continue // Not SR-IOV capable
		}

		device, err := hv.parseSRIOVDevice(entry.Name(), devicePath)
		if err != nil {
			log.Printf("Warning: Failed to parse SR-IOV device %s: %v", entry.Name(), err)
			continue
		}

		hv.sriovDevices[device.PCIAddress] = device
	}

	return nil
}

// parseSRIOVDevice parses an SR-IOV device from sysfs
func (hv *HardwareVirtualization) parseSRIOVDevice(pciAddr, devicePath string) (*SRIOVDevice, error) {
	device := &SRIOVDevice{
		PCIAddress: pciAddr,
		ID:         pciAddr,
	}

	// Read vendor and device IDs
	if vendorData, err := os.ReadFile(filepath.Join(devicePath, "vendor")); err == nil {
		device.VendorID = strings.TrimSpace(string(vendorData))
	}
	
	if deviceData, err := os.ReadFile(filepath.Join(devicePath, "device")); err == nil {
		device.DeviceID = strings.TrimSpace(string(deviceData))
	}

	// Read driver
	driverLink := filepath.Join(devicePath, "driver")
	if linkTarget, err := os.Readlink(driverLink); err == nil {
		device.Driver = filepath.Base(linkTarget)
	}

	// Read SR-IOV configuration
	if maxVFsData, err := os.ReadFile(filepath.Join(devicePath, "sriov_totalvfs")); err == nil {
		if maxVFs, err := strconv.Atoi(strings.TrimSpace(string(maxVFsData))); err == nil {
			device.MaxVFs = maxVFs
		}
	}

	if currentVFsData, err := os.ReadFile(filepath.Join(devicePath, "sriov_numvfs")); err == nil {
		if currentVFs, err := strconv.Atoi(strings.TrimSpace(string(currentVFsData))); err == nil {
			device.CurrentVFs = currentVFs
			device.AvailableVFs = device.MaxVFs - currentVFs
		}
	}

	// Parse existing virtual functions
	device.VirtualFunctions = hv.parseVirtualFunctions(devicePath)

	// Determine IOMMU group
	iommuGroupLink := filepath.Join(devicePath, "iommu_group")
	if linkTarget, err := os.Readlink(iommuGroupLink); err == nil {
		groupName := filepath.Base(linkTarget)
		if groupID, err := strconv.Atoi(groupName); err == nil {
			device.IOMMUGroup = groupID
		}
	}

	// Read device name from modalias or use vendor/device lookup
	if modaliasData, err := os.ReadFile(filepath.Join(devicePath, "modalias")); err == nil {
		device.Name = strings.TrimSpace(string(modaliasData))
	}

	return device, nil
}

// parseVirtualFunctions parses virtual functions for an SR-IOV device
func (hv *HardwareVirtualization) parseVirtualFunctions(devicePath string) []VirtualFunction {
	var vfs []VirtualFunction

	// Look for virtfn* symlinks
	entries, err := os.ReadDir(devicePath)
	if err != nil {
		return vfs
	}

	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "virtfn") {
			continue
		}

		vfLink := filepath.Join(devicePath, entry.Name())
		if linkTarget, err := os.Readlink(vfLink); err == nil {
			vfPCIAddr := filepath.Base(linkTarget)
			
			vf := VirtualFunction{
				ID:         entry.Name(),
				PCIAddress: vfPCIAddr,
			}

			// Get VF IOMMU group
			vfPath := filepath.Join("/sys/bus/pci/devices", vfPCIAddr)
			iommuGroupLink := filepath.Join(vfPath, "iommu_group")
			if linkTarget, err := os.Readlink(iommuGroupLink); err == nil {
				groupName := filepath.Base(linkTarget)
				if groupID, err := strconv.Atoi(groupName); err == nil {
					vf.IOMMUGroup = groupID
				}
			}

			// Check if VF is in use (has a driver bound)
			driverLink := filepath.Join(vfPath, "driver")
			if linkTarget, err := os.Readlink(driverLink); err == nil {
				vf.Driver = filepath.Base(linkTarget)
				vf.InUse = true
			}

			vfs = append(vfs, vf)
		}
	}

	return vfs
}

// detectGPUDevices detects GPU devices and their virtualization capabilities
func (hv *HardwareVirtualization) detectGPUDevices() error {
	// Detection logic for GPU devices would go here
	// This would involve parsing PCI devices for GPUs and checking for:
	// - NVIDIA vGPU support (via nvidia-ml-py or nvidia-smi)
	// - AMD MxGPU support
	// - Intel GVT-g support
	// For now, we'll create a placeholder implementation
	
	log.Printf("GPU device detection not fully implemented")
	return nil
}

// detectNUMATopology detects system NUMA topology
func (hv *HardwareVirtualization) detectNUMATopology() (*SystemNUMATopology, error) {
	topology := &SystemNUMATopology{
		InterconnectMap: make(map[string]int),
	}

	numaDir := "/sys/devices/system/node"
	entries, err := os.ReadDir(numaDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read NUMA directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "node") {
			continue
		}

		nodeIDStr := strings.TrimPrefix(entry.Name(), "node")
		nodeID, err := strconv.Atoi(nodeIDStr)
		if err != nil {
			continue
		}

		node, err := hv.parseNUMANode(nodeID, filepath.Join(numaDir, entry.Name()))
		if err != nil {
			log.Printf("Warning: Failed to parse NUMA node %d: %v", nodeID, err)
			continue
		}

		topology.Nodes = append(topology.Nodes, *node)
		topology.TotalCPUs += len(node.CPUs)
		topology.TotalMemoryMB += node.MemoryMB
	}

	topology.TotalNodes = len(topology.Nodes)

	// Check for hugepage support
	if _, err := os.Stat("/sys/kernel/mm/hugepages"); err == nil {
		topology.HugePagesSupport = true
		topology.HugePageSizes = hv.getHugePageSizes()
	}

	return topology, nil
}

// parseNUMANode parses a NUMA node from sysfs
func (hv *HardwareVirtualization) parseNUMANode(nodeID int, nodePath string) (*NUMANodeInfo, error) {
	node := &NUMANodeInfo{
		ID: nodeID,
	}

	// Parse CPU list
	if cpuListData, err := os.ReadFile(filepath.Join(nodePath, "cpulist")); err == nil {
		cpuListStr := strings.TrimSpace(string(cpuListData))
		node.CPUs = parseCPUList(cpuListStr)
	}

	// Parse memory info
	if memInfoData, err := os.ReadFile(filepath.Join(nodePath, "meminfo")); err == nil {
		memInfo := string(memInfoData)
		lines := strings.Split(memInfo, "\n")
		
		for _, line := range lines {
			if strings.Contains(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 4 {
					if memKB, err := strconv.ParseInt(fields[3], 10, 64); err == nil {
						node.MemoryMB = memKB / 1024
					}
				}
			} else if strings.Contains(line, "MemFree:") {
				fields := strings.Fields(line)
				if len(fields) >= 4 {
					if memKB, err := strconv.ParseInt(fields[3], 10, 64); err == nil {
						node.FreeMemoryMB = memKB / 1024
					}
				}
			}
		}
	}

	// Parse hugepage info
	hugePagesDir := filepath.Join(nodePath, "hugepages")
	if entries, err := os.ReadDir(hugePagesDir); err == nil {
		for _, entry := range entries {
			if strings.HasPrefix(entry.Name(), "hugepages-") {
				// Parse hugepage size from directory name (e.g., hugepages-2048kB)
				sizeStr := strings.TrimPrefix(entry.Name(), "hugepages-")
				sizeStr = strings.TrimSuffix(sizeStr, "kB")
				if size, err := strconv.Atoi(sizeStr); err == nil {
					node.HugePageSize = size
					
					// Read total and free hugepages
					if totalData, err := os.ReadFile(filepath.Join(hugePagesDir, entry.Name(), "nr_hugepages")); err == nil {
						if total, err := strconv.Atoi(strings.TrimSpace(string(totalData))); err == nil {
							node.HugePagesTotal = total
						}
					}
					
					if freeData, err := os.ReadFile(filepath.Join(hugePagesDir, entry.Name(), "free_hugepages")); err == nil {
						if free, err := strconv.Atoi(strings.TrimSpace(string(freeData))); err == nil {
							node.HugePagesFree = free
						}
					}
					
					break // Use first hugepage size found
				}
			}
		}
	}

	// Parse distance information
	if distanceData, err := os.ReadFile(filepath.Join(nodePath, "distance")); err == nil {
		distanceStr := strings.TrimSpace(string(distanceData))
		distanceFields := strings.Fields(distanceStr)
		
		for _, field := range distanceFields {
			if distance, err := strconv.Atoi(field); err == nil {
				node.Distance = append(node.Distance, distance)
			}
		}
	}

	return node, nil
}

// parseCPUList parses a CPU list string (e.g., "0-3,8,9")
func parseCPUList(cpuList string) []int {
	var cpus []int
	
	parts := strings.Split(cpuList, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if strings.Contains(part, "-") {
			// Handle range (e.g., "0-3")
			rangeParts := strings.Split(part, "-")
			if len(rangeParts) == 2 {
				start, err1 := strconv.Atoi(rangeParts[0])
				end, err2 := strconv.Atoi(rangeParts[1])
				if err1 == nil && err2 == nil {
					for i := start; i <= end; i++ {
						cpus = append(cpus, i)
					}
				}
			}
		} else {
			// Handle single CPU
			if cpu, err := strconv.Atoi(part); err == nil {
				cpus = append(cpus, cpu)
			}
		}
	}
	
	return cpus
}

// getHugePageSizes gets available hugepage sizes
func (hv *HardwareVirtualization) getHugePageSizes() []int {
	var sizes []int
	
	hugePagesDir := "/sys/kernel/mm/hugepages"
	entries, err := os.ReadDir(hugePagesDir)
	if err != nil {
		return sizes
	}
	
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "hugepages-") {
			sizeStr := strings.TrimPrefix(entry.Name(), "hugepages-")
			sizeStr = strings.TrimSuffix(sizeStr, "kB")
			if size, err := strconv.Atoi(sizeStr); err == nil {
				sizes = append(sizes, size)
			}
		}
	}
	
	return sizes
}

// GetCPUFeatures returns detected CPU features
func (hv *HardwareVirtualization) GetCPUFeatures() *CPUFeatures {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()
	return hv.cpuFeatures
}

// IsIOMMUEnabled returns whether IOMMU is enabled
func (hv *HardwareVirtualization) IsIOMMUEnabled() bool {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()
	return hv.iommuEnabled
}

// GetSRIOVDevices returns all SR-IOV capable devices
func (hv *HardwareVirtualization) GetSRIOVDevices() map[string]*SRIOVDevice {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()
	
	// Return a copy to prevent modification
	devices := make(map[string]*SRIOVDevice)
	for k, v := range hv.sriovDevices {
		devices[k] = v
	}
	
	return devices
}

// GetGPUDevices returns all GPU devices
func (hv *HardwareVirtualization) GetGPUDevices() map[string]*GPUDevice {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()
	
	// Return a copy to prevent modification
	devices := make(map[string]*GPUDevice)
	for k, v := range hv.gpuDevices {
		devices[k] = v
	}
	
	return devices
}

// GetNUMATopology returns system NUMA topology
func (hv *HardwareVirtualization) GetNUMATopology() *SystemNUMATopology {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()
	return hv.numaTopology
}

// EnableSRIOV enables SR-IOV on a device
func (hv *HardwareVirtualization) EnableSRIOV(ctx context.Context, pciAddress string, numVFs int) error {
	hv.hardwareLock.Lock()
	defer hv.hardwareLock.Unlock()

	device, exists := hv.sriovDevices[pciAddress]
	if !exists {
		return fmt.Errorf("SR-IOV device %s not found", pciAddress)
	}

	if numVFs > device.MaxVFs {
		return fmt.Errorf("requested VFs (%d) exceeds maximum (%d)", numVFs, device.MaxVFs)
	}

	// Write to sriov_numvfs to enable SR-IOV
	sriovPath := filepath.Join("/sys/bus/pci/devices", pciAddress, "sriov_numvfs")
	if err := os.WriteFile(sriovPath, []byte(strconv.Itoa(numVFs)), 0644); err != nil {
		return fmt.Errorf("failed to enable SR-IOV: %w", err)
	}

	// Update device state
	device.CurrentVFs = numVFs
	device.AvailableVFs = device.MaxVFs - numVFs
	
	// Re-parse virtual functions
	devicePath := filepath.Join("/sys/bus/pci/devices", pciAddress)
	device.VirtualFunctions = hv.parseVirtualFunctions(devicePath)

	log.Printf("Enabled SR-IOV on device %s with %d VFs", pciAddress, numVFs)
	return nil
}

// DisableSRIOV disables SR-IOV on a device
func (hv *HardwareVirtualization) DisableSRIOV(ctx context.Context, pciAddress string) error {
	hv.hardwareLock.Lock()
	defer hv.hardwareLock.Unlock()

	device, exists := hv.sriovDevices[pciAddress]
	if !exists {
		return fmt.Errorf("SR-IOV device %s not found", pciAddress)
	}

	// Write 0 to sriov_numvfs to disable SR-IOV
	sriovPath := filepath.Join("/sys/bus/pci/devices", pciAddress, "sriov_numvfs")
	if err := os.WriteFile(sriovPath, []byte("0"), 0644); err != nil {
		return fmt.Errorf("failed to disable SR-IOV: %w", err)
	}

	// Update device state
	device.CurrentVFs = 0
	device.AvailableVFs = device.MaxVFs
	device.VirtualFunctions = []VirtualFunction{}

	log.Printf("Disabled SR-IOV on device %s", pciAddress)
	return nil
}

// AllocateVF allocates a virtual function from an SR-IOV device
func (hv *HardwareVirtualization) AllocateVF(ctx context.Context, pciAddress, vmID string) (*VirtualFunction, error) {
	hv.hardwareLock.Lock()
	defer hv.hardwareLock.Unlock()

	device, exists := hv.sriovDevices[pciAddress]
	if !exists {
		return nil, fmt.Errorf("SR-IOV device %s not found", pciAddress)
	}

	// Find an available VF
	for i := range device.VirtualFunctions {
		vf := &device.VirtualFunctions[i]
		if !vf.InUse {
			vf.InUse = true
			vf.AssignedTo = vmID
			
			log.Printf("Allocated VF %s from device %s to VM %s", vf.ID, pciAddress, vmID)
			return vf, nil
		}
	}

	return nil, fmt.Errorf("no available virtual functions on device %s", pciAddress)
}

// ReleaseVF releases a virtual function
func (hv *HardwareVirtualization) ReleaseVF(ctx context.Context, pciAddress, vfID string) error {
	hv.hardwareLock.Lock()
	defer hv.hardwareLock.Unlock()

	device, exists := hv.sriovDevices[pciAddress]
	if !exists {
		return fmt.Errorf("SR-IOV device %s not found", pciAddress)
	}

	// Find and release the VF
	for i := range device.VirtualFunctions {
		vf := &device.VirtualFunctions[i]
		if vf.ID == vfID {
			vf.InUse = false
			vf.AssignedTo = ""
			
			log.Printf("Released VF %s from device %s", vfID, pciAddress)
			return nil
		}
	}

	return fmt.Errorf("virtual function %s not found on device %s", vfID, pciAddress)
}

// Refresh refreshes hardware information
func (hv *HardwareVirtualization) Refresh(ctx context.Context) error {
	hv.hardwareLock.Lock()
	defer hv.hardwareLock.Unlock()

	// Clear caches
	hv.sriovDevices = make(map[string]*SRIOVDevice)
	hv.gpuDevices = make(map[string]*GPUDevice)

	// Re-detect hardware
	return hv.detectHardwareFeatures()
}

// IsVirtualizationEnabled returns whether hardware virtualization is enabled
func (hv *HardwareVirtualization) IsVirtualizationEnabled() bool {
	if hv.cpuFeatures == nil {
		return false
	}

	return hv.cpuFeatures.VTx || hv.cpuFeatures.AMDV
}

// GetVirtualizationCapabilities returns a summary of virtualization capabilities
func (hv *HardwareVirtualization) GetVirtualizationCapabilities() map[string]interface{} {
	hv.hardwareLock.RLock()
	defer hv.hardwareLock.RUnlock()

	capabilities := make(map[string]interface{})

	capabilities["cpu_features"] = hv.cpuFeatures
	capabilities["iommu_enabled"] = hv.iommuEnabled
	capabilities["sriov_devices"] = len(hv.sriovDevices)
	capabilities["gpu_devices"] = len(hv.gpuDevices)
	
	if hv.numaTopology != nil {
		capabilities["numa_nodes"] = hv.numaTopology.TotalNodes
		capabilities["hugepages_support"] = hv.numaTopology.HugePagesSupport
	}

	capabilities["virtualization_enabled"] = hv.IsVirtualizationEnabled()

	return capabilities
}