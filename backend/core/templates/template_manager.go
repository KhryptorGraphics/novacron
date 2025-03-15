package templates

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// TemplateType represents the type of template
type TemplateType string

const (
	// VMTemplate is a virtual machine template
	VMTemplate TemplateType = "vm"
	
	// ContainerTemplate is a container template
	ContainerTemplate TemplateType = "container"
	
	// ApplicationTemplate is an application template (contains multiple VMs/containers)
	ApplicationTemplate TemplateType = "application"
)

// TemplateState represents the state of a template
type TemplateState string

const (
	// DraftState means the template is still being edited
	DraftState TemplateState = "draft"
	
	// ActiveState means the template is ready for use
	ActiveState TemplateState = "active"
	
	// DeprecatedState means the template should not be used for new deployments
	DeprecatedState TemplateState = "deprecated"
	
	// ArchivedState means the template is archived and not available for use
	ArchivedState TemplateState = "archived"
)

// ProvisioningStrategy represents how a template should be provisioned
type ProvisioningStrategy string

const (
	// CloneStrategy creates a full clone of the template
	CloneStrategy ProvisioningStrategy = "clone"
	
	// LinkedCloneStrategy creates a linked clone (shares base with template)
	LinkedCloneStrategy ProvisioningStrategy = "linked_clone"
	
	// InstantiateStrategy creates a new instance from the template
	InstantiateStrategy ProvisioningStrategy = "instantiate"
)

// Template represents a VM or container template
type Template struct {
	// ID is the unique identifier of the template
	ID string `json:"id"`
	
	// Name is the human-readable name of the template
	Name string `json:"name"`
	
	// Description describes the purpose of the template
	Description string `json:"description"`
	
	// Type is the type of template
	Type TemplateType `json:"type"`
	
	// State is the state of the template
	State TemplateState `json:"state"`
	
	// Version is the version of the template
	Version string `json:"version"`
	
	// BaseImageID is the ID of the base image for this template
	BaseImageID string `json:"base_image_id,omitempty"`
	
	// ParentTemplateID is the ID of the parent template (if derived)
	ParentTemplateID string `json:"parent_template_id,omitempty"`
	
	// StorageLocation is where the template is stored
	StorageLocation string `json:"storage_location"`
	
	// DefaultProvisioningStrategy is the default strategy for provisioning
	DefaultProvisioningStrategy ProvisioningStrategy `json:"default_provisioning_strategy"`
	
	// HardwareProfile defines the hardware requirements
	HardwareProfile *HardwareProfile `json:"hardware_profile,omitempty"`
	
	// NetworkProfile defines the network configuration
	NetworkProfile *NetworkProfile `json:"network_profile,omitempty"`
	
	// StorageProfile defines the storage configuration
	StorageProfile *StorageProfile `json:"storage_profile,omitempty"`
	
	// CustomProperties is additional template metadata
	CustomProperties map[string]interface{} `json:"custom_properties,omitempty"`
	
	// Tags are labels for organizing templates
	Tags []string `json:"tags,omitempty"`
	
	// IsPublic indicates if the template is public
	IsPublic bool `json:"is_public"`
	
	// OwnerID is the ID of the user who owns this template
	OwnerID string `json:"owner_id"`
	
	// TenantID is the ID of the tenant this template belongs to
	TenantID string `json:"tenant_id"`
	
	// AccessControl defines who can access this template
	AccessControl *TemplateAccessControl `json:"access_control,omitempty"`
	
	// CreatedAt is when the template was created
	CreatedAt time.Time `json:"created_at"`
	
	// UpdatedAt is when the template was last updated
	UpdatedAt time.Time `json:"updated_at"`
	
	// PublishedAt is when the template was published (made active)
	PublishedAt time.Time `json:"published_at,omitempty"`
	
	// LastUsedAt is when the template was last used for provisioning
	LastUsedAt time.Time `json:"last_used_at,omitempty"`
	
	// Parameters defines customizable parameters for this template
	Parameters []*TemplateParameter `json:"parameters,omitempty"`
}

// HardwareProfile defines the hardware requirements for a template
type HardwareProfile struct {
	// CPUCount is the number of CPUs
	CPUCount int `json:"cpu_count"`
	
	// CPUCores is the number of CPU cores
	CPUCores int `json:"cpu_cores"`
	
	// CPUType is the type of CPU
	CPUType string `json:"cpu_type,omitempty"`
	
	// MemoryMB is the amount of memory in MB
	MemoryMB int `json:"memory_mb"`
	
	// GPUCount is the number of GPUs
	GPUCount int `json:"gpu_count,omitempty"`
	
	// GPUType is the type of GPU
	GPUType string `json:"gpu_type,omitempty"`
	
	// SpecializedHardware is any specialized hardware requirements
	SpecializedHardware map[string]interface{} `json:"specialized_hardware,omitempty"`
}

// NetworkProfile defines the network configuration for a template
type NetworkProfile struct {
	// Interfaces is a list of network interfaces
	Interfaces []*NetworkInterface `json:"interfaces"`
	
	// RequiresPublicIP indicates if a public IP is required
	RequiresPublicIP bool `json:"requires_public_ip"`
	
	// DefaultNetworkPolicies are the default network policies to apply
	DefaultNetworkPolicies []string `json:"default_network_policies,omitempty"`
	
	// AdditionalNetworkRequirements are any additional network requirements
	AdditionalNetworkRequirements map[string]interface{} `json:"additional_network_requirements,omitempty"`
}

// NetworkInterface defines a network interface for a template
type NetworkInterface struct {
	// Name is the name of the interface
	Name string `json:"name"`
	
	// Type is the type of interface
	Type string `json:"type"`
	
	// MacAddress is the MAC address of the interface
	MacAddress string `json:"mac_address,omitempty"`
	
	// VLAN is the VLAN ID for the interface
	VLAN int `json:"vlan,omitempty"`
	
	// IPConfiguration is the IP configuration for the interface
	IPConfiguration *IPConfiguration `json:"ip_configuration,omitempty"`
}

// IPConfiguration defines the IP configuration for a network interface
type IPConfiguration struct {
	// DHCPEnabled indicates if DHCP is enabled
	DHCPEnabled bool `json:"dhcp_enabled"`
	
	// StaticIP is the static IP address
	StaticIP string `json:"static_ip,omitempty"`
	
	// Subnet is the subnet mask
	Subnet string `json:"subnet,omitempty"`
	
	// Gateway is the gateway address
	Gateway string `json:"gateway,omitempty"`
	
	// DNSServers are the DNS servers
	DNSServers []string `json:"dns_servers,omitempty"`
}

// StorageProfile defines the storage configuration for a template
type StorageProfile struct {
	// Disks is a list of disks
	Disks []*DiskProfile `json:"disks"`
	
	// StorageRequirements are the storage requirements
	StorageRequirements map[string]interface{} `json:"storage_requirements,omitempty"`
}

// DiskProfile defines a disk for a template
type DiskProfile struct {
	// ID is the unique identifier of the disk
	ID string `json:"id"`
	
	// Name is the name of the disk
	Name string `json:"name"`
	
	// SizeGB is the size of the disk in GB
	SizeGB int `json:"size_gb"`
	
	// Type is the type of disk
	Type string `json:"type"`
	
	// StorageTier is the storage tier for the disk
	StorageTier string `json:"storage_tier,omitempty"`
	
	// IsBoot indicates if this is a boot disk
	IsBoot bool `json:"is_boot"`
	
	// IsSystemDisk indicates if this is a system disk
	IsSystemDisk bool `json:"is_system_disk"`
	
	// DiskImageID is the ID of the disk image
	DiskImageID string `json:"disk_image_id,omitempty"`
	
	// MountPath is the mount path for the disk (for containers)
	MountPath string `json:"mount_path,omitempty"`
}

// TemplateAccessControl defines access control for a template
type TemplateAccessControl struct {
	// ReadUsers are the users who can read the template
	ReadUsers []string `json:"read_users,omitempty"`
	
	// ReadGroups are the groups who can read the template
	ReadGroups []string `json:"read_groups,omitempty"`
	
	// WriteUsers are the users who can modify the template
	WriteUsers []string `json:"write_users,omitempty"`
	
	// WriteGroups are the groups who can modify the template
	WriteGroups []string `json:"write_groups,omitempty"`
	
	// UseUsers are the users who can use the template for provisioning
	UseUsers []string `json:"use_users,omitempty"`
	
	// UseGroups are the groups who can use the template for provisioning
	UseGroups []string `json:"use_groups,omitempty"`
}

// TemplateParameter defines a customizable parameter for a template
type TemplateParameter struct {
	// ID is the unique identifier of the parameter
	ID string `json:"id"`
	
	// Name is the name of the parameter
	Name string `json:"name"`
	
	// Description describes the parameter
	Description string `json:"description"`
	
	// Type is the type of parameter
	Type string `json:"type"`
	
	// Required indicates if the parameter is required
	Required bool `json:"required"`
	
	// DefaultValue is the default value for the parameter
	DefaultValue interface{} `json:"default_value,omitempty"`
	
	// AllowedValues are the allowed values for the parameter
	AllowedValues []interface{} `json:"allowed_values,omitempty"`
	
	// MinValue is the minimum value for numeric parameters
	MinValue interface{} `json:"min_value,omitempty"`
	
	// MaxValue is the maximum value for numeric parameters
	MaxValue interface{} `json:"max_value,omitempty"`
	
	// Regex is a regular expression for string parameters
	Regex string `json:"regex,omitempty"`
}

// TemplateVersion represents a version of a template
type TemplateVersion struct {
	// TemplateID is the ID of the template
	TemplateID string `json:"template_id"`
	
	// Version is the version number
	Version string `json:"version"`
	
	// Description describes the changes in this version
	Description string `json:"description"`
	
	// CreatedAt is when the version was created
	CreatedAt time.Time `json:"created_at"`
	
	// CreatedBy is the ID of the user who created this version
	CreatedBy string `json:"created_by"`
	
	// State is the state of this version
	State TemplateState `json:"state"`
	
	// StorageLocation is where this version is stored
	StorageLocation string `json:"storage_location"`
	
	// ChangesFromPrevious describes changes from the previous version
	ChangesFromPrevious string `json:"changes_from_previous,omitempty"`
}

// ProvisioningRequest represents a request to provision from a template
type ProvisioningRequest struct {
	// ID is the unique identifier of the request
	ID string `json:"id"`
	
	// Name is the name for the provisioned resource
	Name string `json:"name"`
	
	// Description is a description for the provisioned resource
	Description string `json:"description"`
	
	// TemplateID is the ID of the template to provision from
	TemplateID string `json:"template_id"`
	
	// TemplateVersion is the version of the template to use
	TemplateVersion string `json:"template_version,omitempty"`
	
	// ProvisioningStrategy is the strategy to use for provisioning
	ProvisioningStrategy ProvisioningStrategy `json:"provisioning_strategy"`
	
	// Parameters are parameter values for the provisioning
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	
	// Customizations are customizations to apply during provisioning
	Customizations map[string]interface{} `json:"customizations,omitempty"`
	
	// RequestedBy is the ID of the user who requested the provisioning
	RequestedBy string `json:"requested_by"`
	
	// TenantID is the ID of the tenant for the provisioned resource
	TenantID string `json:"tenant_id"`
	
	// ResourcePoolID is the ID of the resource pool to provision to
	ResourcePoolID string `json:"resource_pool_id,omitempty"`
	
	// Status is the status of the provisioning request
	Status string `json:"status"`
	
	// CreatedAt is when the request was created
	CreatedAt time.Time `json:"created_at"`
	
	// StartedAt is when provisioning started
	StartedAt time.Time `json:"started_at,omitempty"`
	
	// CompletedAt is when provisioning completed
	CompletedAt time.Time `json:"completed_at,omitempty"`
	
	// ResourceID is the ID of the provisioned resource
	ResourceID string `json:"resource_id,omitempty"`
	
	// Error is the error message if provisioning failed
	Error string `json:"error,omitempty"`
}

// TemplateManager manages VM and container templates
type TemplateManager struct {
	// templates is a map of template ID to template
	templates map[string]*Template
	
	// templateVersions is a map of template ID to a map of version to template version
	templateVersions map[string]map[string]*TemplateVersion
	
	// tenantTemplates is a map of tenant ID to template IDs
	tenantTemplates map[string][]string
	
	// provisioningRequests is a map of request ID to provisioning request
	provisioningRequests map[string]*ProvisioningRequest
	
	// mutex protects the maps
	mutex sync.RWMutex
}

// NewTemplateManager creates a new template manager
func NewTemplateManager() *TemplateManager {
	return &TemplateManager{
		templates:            make(map[string]*Template),
		templateVersions:     make(map[string]map[string]*TemplateVersion),
		tenantTemplates:      make(map[string][]string),
		provisioningRequests: make(map[string]*ProvisioningRequest),
	}
}

// CreateTemplate creates a new template
func (m *TemplateManager) CreateTemplate(template *Template) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template ID already exists
	if _, exists := m.templates[template.ID]; exists {
		return fmt.Errorf("template with ID %s already exists", template.ID)
	}
	
	// Set created and updated times
	now := time.Now()
	template.CreatedAt = now
	template.UpdatedAt = now
	
	// Add template
	m.templates[template.ID] = template
	
	// Add to tenant templates
	m.tenantTemplates[template.TenantID] = append(m.tenantTemplates[template.TenantID], template.ID)
	
	// Initialize version map
	m.templateVersions[template.ID] = make(map[string]*TemplateVersion)
	
	// Create initial version
	version := &TemplateVersion{
		TemplateID:      template.ID,
		Version:         template.Version,
		Description:     "Initial version",
		CreatedAt:       now,
		CreatedBy:       template.OwnerID,
		State:           template.State,
		StorageLocation: template.StorageLocation,
	}
	
	m.templateVersions[template.ID][template.Version] = version
	
	return nil
}

// UpdateTemplate updates a template
func (m *TemplateManager) UpdateTemplate(template *Template) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template exists
	existingTemplate, exists := m.templates[template.ID]
	if !exists {
		return fmt.Errorf("template with ID %s does not exist", template.ID)
	}
	
	// Check tenant ID
	if existingTemplate.TenantID != template.TenantID {
		return errors.New("cannot change tenant ID of a template")
	}
	
	// Update timestamps
	template.CreatedAt = existingTemplate.CreatedAt
	template.UpdatedAt = time.Now()
	
	// Preserve published time if already published
	if existingTemplate.PublishedAt.Unix() > 0 {
		template.PublishedAt = existingTemplate.PublishedAt
	}
	
	// Add new version if version is different
	if existingTemplate.Version != template.Version {
		// Create new version
		version := &TemplateVersion{
			TemplateID:        template.ID,
			Version:           template.Version,
			Description:       "Updated version",
			CreatedAt:         template.UpdatedAt,
			CreatedBy:         "system", // Should be set from context
			State:             template.State,
			StorageLocation:   template.StorageLocation,
			ChangesFromPrevious: "Template updated", // Should be more specific
		}
		
		m.templateVersions[template.ID][template.Version] = version
	}
	
	// Update template
	m.templates[template.ID] = template
	
	return nil
}

// DeleteTemplate deletes a template
func (m *TemplateManager) DeleteTemplate(templateID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template exists
	template, exists := m.templates[templateID]
	if !exists {
		return fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Remove from tenant templates
	tenantTemplates := m.tenantTemplates[template.TenantID]
	for i, id := range tenantTemplates {
		if id == templateID {
			m.tenantTemplates[template.TenantID] = append(tenantTemplates[:i], tenantTemplates[i+1:]...)
			break
		}
	}
	
	// Remove template versions
	delete(m.templateVersions, templateID)
	
	// Remove template
	delete(m.templates, templateID)
	
	return nil
}

// GetTemplate gets a template by ID
func (m *TemplateManager) GetTemplate(templateID string) (*Template, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	// Check if template exists
	template, exists := m.templates[templateID]
	if !exists {
		return nil, fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	return template, nil
}

// ListTemplates lists templates
func (m *TemplateManager) ListTemplates(tenantID string, includePublic bool) ([]*Template, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	var templates []*Template
	
	if tenantID != "" {
		// Get templates for tenant
		templateIDs := m.tenantTemplates[tenantID]
		templates = make([]*Template, 0, len(templateIDs))
		
		for _, id := range templateIDs {
			templates = append(templates, m.templates[id])
		}
		
		// Add public templates from other tenants if requested
		if includePublic {
			for id, template := range m.templates {
				if template.TenantID != tenantID && template.IsPublic {
					templates = append(templates, m.templates[id])
				}
			}
		}
	} else {
		// List all templates
		templates = make([]*Template, 0, len(m.templates))
		for _, template := range m.templates {
			templates = append(templates, template)
		}
	}
	
	return templates, nil
}

// CreateTemplateVersion creates a new version of a template
func (m *TemplateManager) CreateTemplateVersion(templateID, version, description string, location string, createdBy string) (*TemplateVersion, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template exists
	template, exists := m.templates[templateID]
	if !exists {
		return nil, fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Check if version already exists
	if _, exists := m.templateVersions[templateID][version]; exists {
		return nil, fmt.Errorf("version %s already exists for template %s", version, templateID)
	}
	
	// Create new version
	templateVersion := &TemplateVersion{
		TemplateID:      templateID,
		Version:         version,
		Description:     description,
		CreatedAt:       time.Now(),
		CreatedBy:       createdBy,
		State:           template.State,
		StorageLocation: location,
	}
	
	// Add version
	m.templateVersions[templateID][version] = templateVersion
	
	// Update template with new version
	template.Version = version
	template.UpdatedAt = templateVersion.CreatedAt
	template.StorageLocation = location
	
	return templateVersion, nil
}

// GetTemplateVersion gets a specific version of a template
func (m *TemplateManager) GetTemplateVersion(templateID, version string) (*TemplateVersion, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	// Check if template exists
	if _, exists := m.templates[templateID]; !exists {
		return nil, fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Check if version exists
	versions, exists := m.templateVersions[templateID]
	if !exists {
		return nil, fmt.Errorf("no versions found for template %s", templateID)
	}
	
	templateVersion, exists := versions[version]
	if !exists {
		return nil, fmt.Errorf("version %s does not exist for template %s", version, templateID)
	}
	
	return templateVersion, nil
}

// ListTemplateVersions lists all versions of a template
func (m *TemplateManager) ListTemplateVersions(templateID string) ([]*TemplateVersion, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	// Check if template exists
	if _, exists := m.templates[templateID]; !exists {
		return nil, fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Get versions
	versions, exists := m.templateVersions[templateID]
	if !exists {
		return nil, fmt.Errorf("no versions found for template %s", templateID)
	}
	
	// Convert map to slice
	result := make([]*TemplateVersion, 0, len(versions))
	for _, version := range versions {
		result = append(result, version)
	}
	
	return result, nil
}

// PublishTemplate publishes a template (makes it active)
func (m *TemplateManager) PublishTemplate(templateID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template exists
	template, exists := m.templates[templateID]
	if !exists {
		return fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Check if template is already published
	if template.State == ActiveState {
		return nil
	}
	
	// Update template state
	template.State = ActiveState
	template.UpdatedAt = time.Now()
	template.PublishedAt = template.UpdatedAt
	
	// Update version state
	templateVersion, exists := m.templateVersions[templateID][template.Version]
	if exists {
		templateVersion.State = ActiveState
	}
	
	return nil
}

// DeprecateTemplate deprecates a template
func (m *TemplateManager) DeprecateTemplate(templateID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if template exists
	template, exists := m.templates[templateID]
	if !exists {
		return fmt.Errorf("template with ID %s does not exist", templateID)
	}
	
	// Update template state
	template.State = DeprecatedState
	template.UpdatedAt = time.Now()
	
	// Update version state
	templateVersion, exists := m.templateVersions[templateID][template.Version]
	if exists {
		templateVersion.State = DeprecatedState
	}
	
	return nil
}

// CreateProvisioningRequest creates a request to provision from a template
func (m *TemplateManager) CreateProvisioningRequest(request *ProvisioningRequest) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if request ID already exists
	if _, exists := m.provisioningRequests[request.ID]; exists {
		return fmt.Errorf("provisioning request with ID %s already exists", request.ID)
	}
	
	// Check if template exists
	template, exists := m.templates[request.TemplateID]
	if !exists {
		return fmt.Errorf("template with ID %s does not exist", request.TemplateID)
	}
	
	// Check template state
	if template.State != ActiveState {
		return fmt.Errorf("template %s is not active", request.TemplateID)
	}
	
	// Set created time and status
	request.CreatedAt = time.Now()
	request.Status = "pending"
	
	// Add request
	m.provisioningRequests[request.ID] = request
	
	return nil
}

// GetProvisioningRequest gets a provisioning request by ID
func (m *TemplateManager) GetProvisioningRequest(requestID string) (*ProvisioningRequest, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	// Check if request exists
	request, exists := m.provisioningRequests[requestID]
	if !exists {
		return nil, fmt.Errorf("provisioning request with ID %s does not exist", requestID)
	}
	
	return request, nil
}

// ListProvisioningRequests lists provisioning requests
func (m *TemplateManager) ListProvisioningRequests(tenantID string, status string) ([]*ProvisioningRequest, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	var requests []*ProvisioningRequest
	
	// Filter by tenant ID and status
	for _, request := range m.provisioningRequests {
		if (tenantID == "" || request.TenantID == tenantID) &&
			(status == "" || request.Status == status) {
			requests = append(requests, request)
		}
	}
	
	return requests, nil
}

// UpdateProvisioningRequest updates a provisioning request
func (m *TemplateManager) UpdateProvisioningRequest(request *ProvisioningRequest) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if request exists
	existingRequest, exists := m.provisioningRequests[request.ID]
	if !exists {
		return fmt.Errorf("provisioning request with ID %s does not exist", request.ID)
	}
	
	// Preserve created time
	request.CreatedAt = existingRequest.CreatedAt
	
	// Update request
	m.provisioningRequests[request.ID] = request
	
	return nil
}

// StartProvisioning starts the provisioning process for a request
func (m *TemplateManager) StartProvisioning(requestID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if request exists
	request, exists := m.provisioningRequests[requestID]
	if !exists {
		return fmt.Errorf("provisioning request with ID %s does not exist", requestID)
	}
	
	// Check if request is in the right state
	if request.Status != "pending" {
		return fmt.Errorf("provisioning request %s is not in pending state", requestID)
	}
	
	// Update request
	request.Status = "provisioning"
	request.StartedAt = time.Now()
	
	return nil
}

// CompleteProvisioning completes the provisioning process for a request
func (m *TemplateManager) CompleteProvisioning(requestID string, resourceID string, successful bool, errorMsg string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if request exists
	request, exists := m.provisioningRequests[requestID]
	if !exists {
		return fmt.Errorf("provisioning request with ID %s does not exist", requestID)
	}
	
	// Check if request is in the right state
	if request.Status != "provisioning" {
		return fmt.Errorf("provisioning request %s is not in provisioning state", requestID)
	}
	
	// Update request
	request.CompletedAt = time.Now()
	
	if successful {
		request.Status = "completed"
		request.ResourceID = resourceID
		
		// Update template last used time
		if template, exists := m.templates[request.TemplateID]; exists {
			template.LastUsedAt = request.CompletedAt
		}
	} else {
		request.Status = "failed"
		request.Error = errorMsg
	}
	
	return nil
}

// CloneTemplate creates a clone of a template
func (m *TemplateManager) CloneTemplate(sourceID, newID, newName, newVersion, ownerID, tenantID string) (*Template, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Check if source template exists
	sourceTemplate, exists := m.templates[sourceID]
	if !exists {
		return nil, fmt.Errorf("source template with ID %s does not exist", sourceID)
	}
	
	// Check if target template ID already exists
	if _, exists := m.templates[newID]; exists {
		return nil, fmt.Errorf("template with ID %s already exists", newID)
	}
	
	// Create a deep copy of the template
	newTemplate := &Template{
		ID:                        newID,
		Name:                      newName,
		Description:               sourceTemplate.Description + " (Clone)",
		Type:                      sourceTemplate.Type,
		State:                     DraftState, // New clones are drafts
		Version:                   newVersion,
		BaseImageID:               sourceTemplate.Base
