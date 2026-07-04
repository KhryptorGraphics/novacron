package vm

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// VMVolumeAttachment describes a pre-existing storage LUN or volume to attach to a VM.
type VMVolumeAttachment struct {
	VolumeID string `yaml:"volume_id" json:"volume_id"`
	Device   string `yaml:"device,omitempty" json:"device,omitempty"`
	ReadOnly bool   `yaml:"read_only,omitempty" json:"read_only,omitempty"`
	Boot     bool   `yaml:"boot,omitempty" json:"boot,omitempty"`
}

// VMNetworkAttachment describes a network attachment request for a VM.
type VMNetworkAttachment struct {
	NetworkID     string `yaml:"network_id" json:"network_id"`
	InterfaceName string `yaml:"interface_name,omitempty" json:"interface_name,omitempty"`
	MACAddress    string `yaml:"mac_address,omitempty" json:"mac_address,omitempty"`
	IPAddress     string `yaml:"ip_address,omitempty" json:"ip_address,omitempty"`
	Primary       bool   `yaml:"primary,omitempty" json:"primary,omitempty"`
}

// VMPlacementSpec captures placement intent for a VM.
type VMPlacementSpec struct {
	Policy         string   `yaml:"policy,omitempty" json:"policy,omitempty"`
	PreferredNodes []string `yaml:"preferred_nodes,omitempty" json:"preferred_nodes,omitempty"`
	ExcludedNodes  []string `yaml:"excluded_nodes,omitempty" json:"excluded_nodes,omitempty"`
}

// VMMigrationPolicy captures migration intent for a VM.
type VMMigrationPolicy struct {
	Type             string `yaml:"type,omitempty" json:"type,omitempty"`
	BandwidthLimit   int64  `yaml:"bandwidth_limit,omitempty" json:"bandwidth_limit,omitempty"`
	CompressionLevel int    `yaml:"compression_level,omitempty" json:"compression_level,omitempty"`
	Priority         int    `yaml:"priority,omitempty" json:"priority,omitempty"`
}

// VMReplicationPolicy captures replication intent for a VM.
type VMReplicationPolicy struct {
	Enabled bool   `yaml:"enabled,omitempty" json:"enabled,omitempty"`
	Factor  int    `yaml:"factor,omitempty" json:"factor,omitempty"`
	Mode    string `yaml:"mode,omitempty" json:"mode,omitempty"`
}

// Normalized returns a request with the canonical NovaCron VM contract defaults applied.
func (r CreateVMRequest) Normalized() CreateVMRequest {
	if r.Spec.Tags == nil {
		r.Spec.Tags = make(map[string]string)
	}

	if r.Name == "" {
		r.Name = strings.TrimSpace(r.Spec.Name)
	}
	if r.Spec.Name == "" {
		r.Spec.Name = strings.TrimSpace(r.Name)
	}

	for key, value := range r.Tags {
		r.Spec.Tags[key] = value
	}

	normalizeVMType(&r.Spec)

	r.Spec.OwnerID = strings.TrimSpace(r.Spec.OwnerID)
	r.Spec.TenantID = strings.TrimSpace(r.Spec.TenantID)

	if r.Spec.OwnerID != "" {
		r.Spec.Tags["owner_id"] = r.Spec.OwnerID
	}
	if r.Spec.TenantID != "" {
		r.Spec.Tags["tenant_id"] = r.Spec.TenantID
	}

	if r.Spec.Placement != nil {
		r.Spec.Placement.Policy = normalizePlacementPolicy(r.Spec.Placement.Policy)
	}

	if r.Spec.Image == "" && r.Spec.RootFS != "" && r.Spec.Type == VMTypeKVM {
		r.Spec.Image = r.Spec.RootFS
	}
	if r.Spec.RootFS == "" && r.Spec.Image != "" {
		r.Spec.RootFS = r.Spec.Image
	}

	if r.Spec.NetworkID == "" {
		for _, attachment := range r.Spec.NetworkAttachments {
			if attachment.Primary {
				r.Spec.NetworkID = attachment.NetworkID
				break
			}
		}
		if r.Spec.NetworkID == "" && len(r.Spec.NetworkAttachments) > 0 {
			r.Spec.NetworkID = r.Spec.NetworkAttachments[0].NetworkID
		}
	}

	r.Tags = r.Spec.Tags
	return r
}

// Validate checks whether a create request conforms to the canonical VM contract.
// Resource ceilings guard against disk/memory exhaustion from a single create
// request. Env-overridable (a positive integer); unset/invalid -> the default.
// ponytail: read per-call, not cached — Validate isn't hot.
const maxVMNameLen = 128

func maxDiskSizeGB() int { return envPositiveIntOrDefault("NOVACRON_MAX_DISK_GB", 2048) }      // 2 TB
func maxMemoryMB() int   { return envPositiveIntOrDefault("NOVACRON_MAX_MEMORY_MB", 1048576) } // 1 TB

func envPositiveIntOrDefault(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func (r CreateVMRequest) Validate() error {
	name := strings.TrimSpace(r.Name)
	if name == "" {
		name = strings.TrimSpace(r.Spec.Name)
	}
	if name == "" {
		return fmt.Errorf("vm name is required")
	}
	if !r.AllowMissingOwnership {
		if strings.TrimSpace(r.Spec.OwnerID) == "" {
			return fmt.Errorf("owner_id is required")
		}
		if strings.TrimSpace(r.Spec.TenantID) == "" {
			return fmt.Errorf("tenant_id is required")
		}
	}
	if !isSupportedVMType(r.Spec.Type) {
		return fmt.Errorf("unsupported vm type %q", r.Spec.Type)
	}
	if r.Spec.CPUShares < 0 {
		return fmt.Errorf("cpu shares cannot be negative")
	}
	if r.Spec.MemoryMB < 0 {
		return fmt.Errorf("memory_mb cannot be negative")
	}
	if r.Spec.DiskSizeGB < 0 {
		return fmt.Errorf("disk_size_gb cannot be negative")
	}
	// Upper ceilings (0 stays valid — the driver treats it as "use default").
	if r.Spec.DiskSizeGB > maxDiskSizeGB() {
		return fmt.Errorf("disk_size_gb %d exceeds the maximum of %d", r.Spec.DiskSizeGB, maxDiskSizeGB())
	}
	if r.Spec.MemoryMB > maxMemoryMB() {
		return fmt.Errorf("memory_mb %d exceeds the maximum of %d", r.Spec.MemoryMB, maxMemoryMB())
	}
	// Name length + no control characters (defense-in-depth: the name reaches
	// the DB/logs/tags, though not qemu args or paths since the id is generated).
	if len(name) > maxVMNameLen {
		return fmt.Errorf("vm name too long (max %d characters)", maxVMNameLen)
	}
	for _, ch := range name {
		if ch < 0x20 || ch == 0x7f {
			return fmt.Errorf("vm name contains control characters")
		}
	}

	seenVolumeIDs := make(map[string]struct{}, len(r.Spec.VolumeAttachments))
	for _, attachment := range r.Spec.VolumeAttachments {
		volumeID := strings.TrimSpace(attachment.VolumeID)
		if volumeID == "" {
			return fmt.Errorf("volume attachments require volume_id")
		}
		if _, exists := seenVolumeIDs[volumeID]; exists {
			return fmt.Errorf("duplicate volume attachment %q", volumeID)
		}
		seenVolumeIDs[volumeID] = struct{}{}
	}

	primaryNetworks := 0
	seenNetworkIDs := make(map[string]struct{}, len(r.Spec.NetworkAttachments))
	for _, attachment := range r.Spec.NetworkAttachments {
		networkID := strings.TrimSpace(attachment.NetworkID)
		if networkID == "" {
			return fmt.Errorf("network attachments require network_id")
		}
		if _, exists := seenNetworkIDs[networkID]; exists {
			return fmt.Errorf("duplicate network attachment %q", networkID)
		}
		seenNetworkIDs[networkID] = struct{}{}
		if attachment.Primary {
			primaryNetworks++
		}
	}
	if primaryNetworks > 1 {
		return fmt.Errorf("only one primary network attachment is allowed")
	}

	if r.Spec.Placement != nil {
		if !isValidPlacementPolicy(r.Spec.Placement.Policy) {
			return fmt.Errorf("unsupported placement policy %q", r.Spec.Placement.Policy)
		}
		if r.Spec.Placement.Policy == "custom" && len(normalizeNodeIDSet(r.Spec.Placement.PreferredNodes)) == 0 && len(normalizeNodeIDSet(r.Spec.Placement.ExcludedNodes)) == 0 {
			return fmt.Errorf("custom placement policy requires preferred_nodes or excluded_nodes")
		}
	}
	if r.Spec.Migration != nil && !isValidMigrationPolicyType(r.Spec.Migration.Type) {
		return fmt.Errorf("unsupported migration policy %q", r.Spec.Migration.Type)
	}
	if r.Spec.Replication != nil && r.Spec.Replication.Factor < 0 {
		return fmt.Errorf("replication factor cannot be negative")
	}

	return nil
}

func normalizeVMType(config *VMConfig) {
	if config.Tags == nil {
		config.Tags = make(map[string]string)
	}

	switch {
	case config.Type != "":
		config.Tags["vm_type"] = string(config.Type)
	case strings.TrimSpace(config.Tags["vm_type"]) != "":
		config.Type = VMType(strings.TrimSpace(config.Tags["vm_type"]))
	default:
		config.Type = VMTypeKVM
		config.Tags["vm_type"] = string(config.Type)
	}
}

func isSupportedVMType(vmType VMType) bool {
	switch vmType {
	case VMTypeKVM, VMTypeContainer, VMTypeContainerd, VMTypeKataContainers, VMTypeProcess, VMTypeVMware, VMTypeVSphere, VMTypeHyperV, VMTypeXen, VMTypeXenServer, VMTypeProxmox, VMTypeProxmoxVE:
		return true
	default:
		return false
	}
}

func isValidPlacementPolicy(policy string) bool {
	switch normalizePlacementPolicy(policy) {
	case "", "balanced", "consolidated", "performance", "efficiency", "network-aware", "custom":
		return true
	default:
		return false
	}
}

func normalizePlacementPolicy(policy string) string {
	policy = strings.TrimSpace(strings.ToLower(policy))
	switch policy {
	case "":
		return ""
	case "balanced", "consolidated", "performance", "efficiency", "network-aware", "custom":
		return policy
	default:
		return policy
	}
}

func isValidMigrationPolicyType(policyType string) bool {
	switch strings.TrimSpace(policyType) {
	case "", "cold", "warm", "live", "checkpoint":
		return true
	default:
		return false
	}
}
