// Package azure implements the Azure cloud adapter for NovaCron
package azure

import (
	"context"
	"fmt"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2021-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2021-02-01/network"
	"github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2021-01-01/resources"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure/auth"

	"github.com/khryptorgraphics/novacron/adapters/pkg/interfaces"
)

// Adapter implements the CloudAdapter interface for Azure
type Adapter struct {
	config          *Config
	authorizer      autorest.Authorizer
	vmClient        *compute.VirtualMachinesClient
	diskClient      *compute.DisksClient
	networkClient   *network.VirtualNetworksClient
	subnetClient    *network.SubnetsClient
	nsgClient       *network.SecurityGroupsClient
	resourcesClient *resources.Client
}

// Config represents Azure-specific configuration
type Config struct {
	SubscriptionID string `json:"subscription_id"`
	TenantID       string `json:"tenant_id"`
	ClientID       string `json:"client_id"`
	ClientSecret   string `json:"client_secret"`
	ResourceGroup  string `json:"resource_group"`
	Location       string `json:"location"`
	Environment    string `json:"environment,omitempty"` // AzurePublicCloud, AzureGovernmentCloud, etc.
}

// GetProvider returns the cloud provider name
func (c *Config) GetProvider() string {
	return "azure"
}

// GetRegion returns the Azure location
func (c *Config) GetRegion() string {
	return c.Location
}

// GetCredentials returns the credentials map
func (c *Config) GetCredentials() map[string]string {
	return map[string]string{
		"subscription_id": c.SubscriptionID,
		"tenant_id":       c.TenantID,
		"client_id":       c.ClientID,
		"client_secret":   c.ClientSecret,
	}
}

// GetSettings returns additional settings
func (c *Config) GetSettings() map[string]interface{} {
	return map[string]interface{}{
		"resource_group": c.ResourceGroup,
		"environment":    c.Environment,
	}
}

// Validate validates the Azure configuration
func (c *Config) Validate() error {
	if c.SubscriptionID == "" {
		return fmt.Errorf("subscription_id is required")
	}
	if c.TenantID == "" {
		return fmt.Errorf("tenant_id is required")
	}
	if c.ClientID == "" {
		return fmt.Errorf("client_id is required")
	}
	if c.ClientSecret == "" {
		return fmt.Errorf("client_secret is required")
	}
	if c.ResourceGroup == "" {
		return fmt.Errorf("resource_group is required")
	}
	if c.Location == "" {
		return fmt.Errorf("location is required")
	}
	return nil
}

// NewAdapter creates a new Azure adapter
func NewAdapter() *Adapter {
	return &Adapter{}
}

// Name returns the adapter name
func (a *Adapter) Name() string {
	return "azure-vm-adapter"
}

// Version returns the adapter version
func (a *Adapter) Version() string {
	return "1.0.0"
}

// SupportedRegions returns the list of supported Azure regions
func (a *Adapter) SupportedRegions() []string {
	return []string{
		"eastus", "eastus2", "westus", "westus2", "westus3",
		"centralus", "northcentralus", "southcentralus",
		"northeurope", "westeurope", "francecentral", "francesouth",
		"uksouth", "ukwest", "germanywestcentral",
		"switzerlandnorth", "switzerlandwest", "norwayeast", "norwaywest",
		"southeastasia", "eastasia", "australiaeast", "australiasoutheast",
		"japaneast", "japanwest", "koreacentral", "koreasouth",
		"southindia", "westindia", "centralindia",
		"canadacentral", "canadaeast",
		"brazilsouth", "southafricanorth", "uaenorth",
	}
}

// SupportedInstanceTypes returns the list of supported VM sizes
func (a *Adapter) SupportedInstanceTypes() []string {
	return []string{
		// B-series (Burstable)
		"Standard_B1ls", "Standard_B1s", "Standard_B1ms", "Standard_B2s", "Standard_B2ms",
		"Standard_B4ms", "Standard_B8ms", "Standard_B12ms", "Standard_B16ms", "Standard_B20ms",
		
		// D-series (General purpose)
		"Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3", "Standard_D16s_v3", "Standard_D32s_v3",
		"Standard_D48s_v3", "Standard_D64s_v3",
		
		// E-series (Memory optimized)
		"Standard_E2s_v3", "Standard_E4s_v3", "Standard_E8s_v3", "Standard_E16s_v3", "Standard_E32s_v3",
		"Standard_E48s_v3", "Standard_E64s_v3",
		
		// F-series (Compute optimized)
		"Standard_F2s_v2", "Standard_F4s_v2", "Standard_F8s_v2", "Standard_F16s_v2", "Standard_F32s_v2",
		"Standard_F48s_v2", "Standard_F64s_v2", "Standard_F72s_v2",
	}
}

// Configure configures the Azure adapter
func (a *Adapter) Configure(config interfaces.CloudConfig) error {
	azureConfig, ok := config.(*Config)
	if !ok {
		return fmt.Errorf("invalid config type, expected *azure.Config")
	}

	if err := azureConfig.Validate(); err != nil {
		return fmt.Errorf("config validation failed: %w", err)
	}

	a.config = azureConfig

	// Create authorizer using client credentials
	clientCredentialsConfig := auth.NewClientCredentialsConfig(
		azureConfig.ClientID,
		azureConfig.ClientSecret,
		azureConfig.TenantID,
	)

	authorizer, err := clientCredentialsConfig.Authorizer()
	if err != nil {
		return fmt.Errorf("failed to create authorizer: %w", err)
	}

	a.authorizer = authorizer

	// Initialize clients
	a.vmClient = compute.NewVirtualMachinesClient(azureConfig.SubscriptionID)
	a.vmClient.Authorizer = authorizer

	a.diskClient = compute.NewDisksClient(azureConfig.SubscriptionID)
	a.diskClient.Authorizer = authorizer

	a.networkClient = network.NewVirtualNetworksClient(azureConfig.SubscriptionID)
	a.networkClient.Authorizer = authorizer

	a.subnetClient = network.NewSubnetsClient(azureConfig.SubscriptionID)
	a.subnetClient.Authorizer = authorizer

	a.nsgClient = network.NewSecurityGroupsClient(azureConfig.SubscriptionID)
	a.nsgClient.Authorizer = authorizer

	a.resourcesClient = resources.NewClient(azureConfig.SubscriptionID)
	a.resourcesClient.Authorizer = authorizer

	return nil
}

// ValidateCredentials validates Azure credentials
func (a *Adapter) ValidateCredentials(ctx context.Context) error {
	if a.resourcesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	// Try to list resource groups to validate credentials
	_, err := a.resourcesClient.ListComplete(ctx, "", nil)
	return err
}

// CreateInstance creates a new Azure VM
func (a *Adapter) CreateInstance(ctx context.Context, req *interfaces.CreateInstanceRequest) (*interfaces.Instance, error) {
	if a.vmClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	vmName := req.Name
	if vmName == "" {
		vmName = fmt.Sprintf("vm-%d", time.Now().Unix())
	}

	// Create VM parameters
	vmParams := compute.VirtualMachine{
		Location: &a.config.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			HardwareProfile: &compute.HardwareProfile{
				VMSize: compute.VirtualMachineSizeTypes(req.InstanceType),
			},
			StorageProfile: &compute.StorageProfile{
				ImageReference: &compute.ImageReference{
					// Parse image ID - simplified for demo
					Publisher: stringPtr("Canonical"),
					Offer:     stringPtr("UbuntuServer"),
					Sku:       stringPtr("18.04-LTS"),
					Version:   stringPtr("latest"),
				},
				OsDisk: &compute.OSDisk{
					CreateOption: compute.DiskCreateOptionTypesFromImage,
					ManagedDisk: &compute.ManagedDiskParameters{
						StorageAccountType: compute.StorageAccountTypesStandardLRS,
					},
				},
			},
			OsProfile: &compute.OSProfile{
				ComputerName:  &vmName,
				AdminUsername: stringPtr("azureuser"),
				LinuxConfiguration: &compute.LinuxConfiguration{
					DisablePasswordAuthentication: boolPtr(true),
				},
			},
			NetworkProfile: &compute.NetworkProfile{
				NetworkInterfaces: &[]compute.NetworkInterfaceReference{
					{
						ID: stringPtr(fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkInterfaces/%s-nic",
							a.config.SubscriptionID, a.config.ResourceGroup, vmName)),
						NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
							Primary: boolPtr(true),
						},
					},
				},
			},
		},
	}

	// Add SSH key if provided
	if req.KeyPairName != "" {
		vmParams.OsProfile.LinuxConfiguration.SSH = &compute.SSHConfiguration{
			PublicKeys: &[]compute.SSHPublicKey{
				{
					Path: stringPtr("/home/azureuser/.ssh/authorized_keys"),
					KeyData: stringPtr(req.KeyPairName), // In real implementation, would load actual key
				},
			},
		}
	}

	// Add custom data (user data)
	if req.UserData != "" {
		vmParams.OsProfile.CustomData = &req.UserData
	}

	// Add tags
	if len(req.Tags) > 0 {
		tags := make(map[string]*string)
		for k, v := range req.Tags {
			tags[k] = stringPtr(v)
		}
		vmParams.Tags = tags
	}

	// Configure root volume size if specified
	if req.RootVolumeSize > 0 {
		vmParams.StorageProfile.OsDisk.DiskSizeGB = int32Ptr(int32(req.RootVolumeSize))
	}

	// Create VM
	future, err := a.vmClient.CreateOrUpdate(ctx, a.config.ResourceGroup, vmName, vmParams)
	if err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}

	// Wait for completion
	err = future.WaitForCompletionRef(ctx, a.vmClient.Client)
	if err != nil {
		return nil, fmt.Errorf("failed to wait for VM creation: %w", err)
	}

	// Get created VM
	vm, err := future.Result(*a.vmClient)
	if err != nil {
		return nil, fmt.Errorf("failed to get created VM: %w", err)
	}

	return a.convertAzureVM(&vm), nil
}

// GetInstance retrieves an Azure VM by name
func (a *Adapter) GetInstance(ctx context.Context, instanceID string) (*interfaces.Instance, error) {
	if a.vmClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	vm, err := a.vmClient.Get(ctx, a.config.ResourceGroup, instanceID, compute.InstanceView)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	return a.convertAzureVM(&vm), nil
}

// ListInstances lists Azure VMs with optional filtering
func (a *Adapter) ListInstances(ctx context.Context, filters *interfaces.ListInstanceFilters) ([]*interfaces.Instance, error) {
	if a.vmClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	result, err := a.vmClient.List(ctx, a.config.ResourceGroup)
	if err != nil {
		return nil, fmt.Errorf("failed to list VMs: %w", err)
	}

	var instances []*interfaces.Instance
	for result.NotDone() {
		for _, vm := range result.Values() {
			instance := a.convertAzureVM(&vm)
			
			// Apply filters
			if filters != nil {
				if len(filters.States) > 0 {
					found := false
					for _, state := range filters.States {
						if instance.State == state {
							found = true
							break
						}
					}
					if !found {
						continue
					}
				}
				
				if filters.InstanceType != "" && instance.InstanceType != filters.InstanceType {
					continue
				}
			}
			
			instances = append(instances, instance)
		}
		
		if err := result.NextWithContext(ctx); err != nil {
			break
		}
	}

	return instances, nil
}

// UpdateInstance updates an Azure VM
func (a *Adapter) UpdateInstance(ctx context.Context, instanceID string, updates *interfaces.UpdateInstanceRequest) (*interfaces.Instance, error) {
	if a.vmClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	// Get existing VM
	vm, err := a.vmClient.Get(ctx, a.config.ResourceGroup, instanceID, "")
	if err != nil {
		return nil, fmt.Errorf("failed to get VM for update: %w", err)
	}

	// Update VM size if specified
	if updates.InstanceType != nil {
		vm.HardwareProfile.VMSize = compute.VirtualMachineSizeTypes(*updates.InstanceType)
	}

	// Update tags if specified
	if len(updates.Tags) > 0 {
		if vm.Tags == nil {
			vm.Tags = make(map[string]*string)
		}
		for k, v := range updates.Tags {
			vm.Tags[k] = stringPtr(v)
		}
	}

	// Update VM
	future, err := a.vmClient.CreateOrUpdate(ctx, a.config.ResourceGroup, instanceID, vm)
	if err != nil {
		return nil, fmt.Errorf("failed to update VM: %w", err)
	}

	err = future.WaitForCompletionRef(ctx, a.vmClient.Client)
	if err != nil {
		return nil, fmt.Errorf("failed to wait for VM update: %w", err)
	}

	updatedVM, err := future.Result(*a.vmClient)
	if err != nil {
		return nil, fmt.Errorf("failed to get updated VM: %w", err)
	}

	return a.convertAzureVM(&updatedVM), nil
}

// DeleteInstance deletes an Azure VM
func (a *Adapter) DeleteInstance(ctx context.Context, instanceID string, force bool) error {
	if a.vmClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	future, err := a.vmClient.Delete(ctx, a.config.ResourceGroup, instanceID)
	if err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}

	return future.WaitForCompletionRef(ctx, a.vmClient.Client)
}

// StartInstance starts an Azure VM
func (a *Adapter) StartInstance(ctx context.Context, instanceID string) error {
	if a.vmClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	future, err := a.vmClient.Start(ctx, a.config.ResourceGroup, instanceID)
	if err != nil {
		return fmt.Errorf("failed to start VM: %w", err)
	}

	return future.WaitForCompletionRef(ctx, a.vmClient.Client)
}

// StopInstance stops an Azure VM
func (a *Adapter) StopInstance(ctx context.Context, instanceID string, force bool) error {
	if a.vmClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	// Azure VMs are deallocated when stopped to save costs
	future, err := a.vmClient.Deallocate(ctx, a.config.ResourceGroup, instanceID)
	if err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	return future.WaitForCompletionRef(ctx, a.vmClient.Client)
}

// RebootInstance restarts an Azure VM
func (a *Adapter) RebootInstance(ctx context.Context, instanceID string) error {
	if a.vmClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	future, err := a.vmClient.Restart(ctx, a.config.ResourceGroup, instanceID)
	if err != nil {
		return fmt.Errorf("failed to restart VM: %w", err)
	}

	return future.WaitForCompletionRef(ctx, a.vmClient.Client)
}

// Helper method to convert Azure VM to interface instance
func (a *Adapter) convertAzureVM(vm *compute.VirtualMachine) *interfaces.Instance {
	instance := &interfaces.Instance{
		ID:           *vm.Name,
		Provider:     "azure",
		Region:       a.config.Location,
		State:        a.convertVMState(vm),
		Tags:         make(map[string]string),
		Metadata:     make(map[string]interface{}),
	}

	if vm.Name != nil {
		instance.Name = *vm.Name
	}

	if vm.VirtualMachineProperties != nil {
		if vm.HardwareProfile != nil {
			instance.InstanceType = string(vm.HardwareProfile.VMSize)
		}

		// Get instance view for current state and other runtime info
		if vm.InstanceView != nil {
			for _, status := range *vm.InstanceView.Statuses {
				if status.Code != nil && strings.HasPrefix(*status.Code, "PowerState/") {
					instance.State = a.convertPowerState(*status.Code)
				}
			}
		}

		// Extract network information (simplified)
		if vm.NetworkProfile != nil && vm.NetworkProfile.NetworkInterfaces != nil {
			// In a full implementation, we'd resolve the network interface to get IP addresses
		}
	}

	// Convert tags
	if vm.Tags != nil {
		for k, v := range vm.Tags {
			if v != nil {
				instance.Tags[k] = *v
			}
		}
	}

	return instance
}

// Helper method to convert Azure VM state
func (a *Adapter) convertVMState(vm *compute.VirtualMachine) interfaces.InstanceState {
	if vm.InstanceView == nil || vm.InstanceView.Statuses == nil {
		return interfaces.InstanceStateUnknown
	}

	for _, status := range *vm.InstanceView.Statuses {
		if status.Code != nil && strings.HasPrefix(*status.Code, "PowerState/") {
			return a.convertPowerState(*status.Code)
		}
	}

	return interfaces.InstanceStateUnknown
}

// Helper method to convert Azure power state
func (a *Adapter) convertPowerState(powerState string) interfaces.InstanceState {
	switch powerState {
	case "PowerState/starting":
		return interfaces.InstanceStatePending
	case "PowerState/running":
		return interfaces.InstanceStateRunning
	case "PowerState/stopping":
		return interfaces.InstanceStateStopping
	case "PowerState/stopped":
		return interfaces.InstanceStateStopped
	case "PowerState/deallocating":
		return interfaces.InstanceStateStopping
	case "PowerState/deallocated":
		return interfaces.InstanceStateStopped
	default:
		return interfaces.InstanceStateUnknown
	}
}

// Implement remaining interface methods (simplified for space)

func (a *Adapter) GetInstanceMetrics(ctx context.Context, instanceID string, opts *interfaces.MetricsOptions) (*interfaces.InstanceMetrics, error) {
	// Would integrate with Azure Monitor
	return &interfaces.InstanceMetrics{
		InstanceID: instanceID,
		Period:     opts.Period,
		StartTime:  opts.StartTime,
		EndTime:    opts.EndTime,
	}, nil
}

func (a *Adapter) GetInstanceLogs(ctx context.Context, instanceID string, opts *interfaces.LogOptions) (*interfaces.InstanceLogs, error) {
	// Would integrate with Azure Monitor Logs
	return &interfaces.InstanceLogs{
		InstanceID: instanceID,
		LogGroups:  []*interfaces.LogGroup{},
	}, nil
}

func (a *Adapter) CreateVolume(ctx context.Context, req *interfaces.CreateVolumeRequest) (*interfaces.Volume, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) AttachVolume(ctx context.Context, volumeID, instanceID string) error {
	return fmt.Errorf("not implemented")
}

func (a *Adapter) DetachVolume(ctx context.Context, volumeID string) error {
	return fmt.Errorf("not implemented")
}

func (a *Adapter) DeleteVolume(ctx context.Context, volumeID string) error {
	return fmt.Errorf("not implemented")
}

func (a *Adapter) ListNetworks(ctx context.Context) ([]*interfaces.Network, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) CreateSecurityGroup(ctx context.Context, req *interfaces.CreateSecurityGroupRequest) (*interfaces.SecurityGroup, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) DeleteSecurityGroup(ctx context.Context, groupID string) error {
	return fmt.Errorf("not implemented")
}

func (a *Adapter) GetQuotas(ctx context.Context) (*interfaces.ResourceQuotas, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) GetCosts(ctx context.Context, opts *interfaces.CostOptions) (*interfaces.CostData, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) ExportInstance(ctx context.Context, instanceID string, opts *interfaces.ExportOptions) (*interfaces.ExportResult, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) ImportInstance(ctx context.Context, req *interfaces.ImportInstanceRequest) (*interfaces.Instance, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *Adapter) HealthCheck(ctx context.Context) error {
	return a.ValidateCredentials(ctx)
}

func (a *Adapter) GetStatus(ctx context.Context) (*interfaces.AdapterStatus, error) {
	status := &interfaces.AdapterStatus{
		Name:            a.Name(),
		Version:         a.Version(),
		Provider:        "azure",
		Region:          a.config.Location,
		Status:          "healthy",
		LastHealthCheck: time.Now(),
		Capabilities: []string{
			"create_instance", "get_instance", "list_instances",
			"update_instance", "delete_instance",
			"start_instance", "stop_instance", "reboot_instance",
		},
	}

	if err := a.HealthCheck(ctx); err != nil {
		status.Status = "unhealthy"
		status.ErrorMessage = err.Error()
	}

	return status, nil
}

// Helper functions
func stringPtr(s string) *string {
	return &s
}

func boolPtr(b bool) *bool {
	return &b
}

func int32Ptr(i int32) *int32 {
	return &i
}