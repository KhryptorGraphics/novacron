// Package gcp implements the Google Cloud Platform adapter for NovaCron
package gcp

import (
	"context"
	"fmt"
	"strings"
	"time"

	compute "cloud.google.com/go/compute/apiv1"
	"google.golang.org/api/option"
	computepb "google.golang.org/genproto/googleapis/cloud/compute/v1"
	"google.golang.org/protobuf/proto"

	"github.com/khryptorgraphics/novacron/adapters/pkg/interfaces"
)

// Adapter implements the CloudAdapter interface for Google Cloud Platform
type Adapter struct {
	config            *Config
	instancesClient   *compute.InstancesClient
	disksClient       *compute.DisksClient
	networksClient    *compute.NetworksClient
	subnetworksClient *compute.SubnetworksClient
	firewallsClient   *compute.FirewallsClient
	regionsClient     *compute.RegionsClient
	zonesClient       *compute.ZonesClient
}

// Config represents GCP-specific configuration
type Config struct {
	ProjectID            string `json:"project_id"`
	Zone                 string `json:"zone"`
	Region               string `json:"region"`
	CredentialsJSON      string `json:"credentials_json,omitempty"`
	CredentialsFile      string `json:"credentials_file,omitempty"`
	ServiceAccountEmail  string `json:"service_account_email,omitempty"`
	ImpersonateServiceAccount string `json:"impersonate_service_account,omitempty"`
}

// GetProvider returns the cloud provider name
func (c *Config) GetProvider() string {
	return "gcp"
}

// GetRegion returns the GCP region
func (c *Config) GetRegion() string {
	return c.Region
}

// GetCredentials returns the credentials map
func (c *Config) GetCredentials() map[string]string {
	return map[string]string{
		"project_id":                   c.ProjectID,
		"credentials_json":             c.CredentialsJSON,
		"credentials_file":             c.CredentialsFile,
		"service_account_email":        c.ServiceAccountEmail,
		"impersonate_service_account":  c.ImpersonateServiceAccount,
	}
}

// GetSettings returns additional settings
func (c *Config) GetSettings() map[string]interface{} {
	return map[string]interface{}{
		"zone": c.Zone,
	}
}

// Validate validates the GCP configuration
func (c *Config) Validate() error {
	if c.ProjectID == "" {
		return fmt.Errorf("project_id is required")
	}
	if c.Zone == "" {
		return fmt.Errorf("zone is required")
	}
	
	// Extract region from zone if not provided
	if c.Region == "" {
		zoneParts := strings.Split(c.Zone, "-")
		if len(zoneParts) >= 2 {
			c.Region = strings.Join(zoneParts[:2], "-")
		} else {
			return fmt.Errorf("invalid zone format or region is required")
		}
	}
	
	return nil
}

// NewAdapter creates a new GCP adapter
func NewAdapter() *Adapter {
	return &Adapter{}
}

// Name returns the adapter name
func (a *Adapter) Name() string {
	return "gcp-compute-adapter"
}

// Version returns the adapter version
func (a *Adapter) Version() string {
	return "1.0.0"
}

// SupportedRegions returns the list of supported GCP regions
func (a *Adapter) SupportedRegions() []string {
	return []string{
		"us-central1", "us-east1", "us-east4", "us-west1", "us-west2", "us-west3", "us-west4",
		"europe-north1", "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west6",
		"asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", "asia-northeast3",
		"asia-south1", "asia-southeast1", "asia-southeast2",
		"australia-southeast1", "northamerica-northeast1", "southamerica-east1",
	}
}

// SupportedInstanceTypes returns the list of supported machine types
func (a *Adapter) SupportedInstanceTypes() []string {
	return []string{
		// E2 (cost-optimized)
		"e2-micro", "e2-small", "e2-medium", "e2-standard-2", "e2-standard-4", "e2-standard-8", "e2-standard-16", "e2-standard-32",
		
		// N1 (general purpose)
		"n1-standard-1", "n1-standard-2", "n1-standard-4", "n1-standard-8", "n1-standard-16", "n1-standard-32", "n1-standard-64", "n1-standard-96",
		"n1-highmem-2", "n1-highmem-4", "n1-highmem-8", "n1-highmem-16", "n1-highmem-32", "n1-highmem-64", "n1-highmem-96",
		"n1-highcpu-2", "n1-highcpu-4", "n1-highcpu-8", "n1-highcpu-16", "n1-highcpu-32", "n1-highcpu-64", "n1-highcpu-96",
		
		// N2 (general purpose)
		"n2-standard-2", "n2-standard-4", "n2-standard-8", "n2-standard-16", "n2-standard-32", "n2-standard-48", "n2-standard-64", "n2-standard-80",
		"n2-highmem-2", "n2-highmem-4", "n2-highmem-8", "n2-highmem-16", "n2-highmem-32", "n2-highmem-48", "n2-highmem-64", "n2-highmem-80",
		"n2-highcpu-2", "n2-highcpu-4", "n2-highcpu-8", "n2-highcpu-16", "n2-highcpu-32", "n2-highcpu-48", "n2-highcpu-64", "n2-highcpu-80",
		
		// C2 (compute optimized)
		"c2-standard-4", "c2-standard-8", "c2-standard-16", "c2-standard-30", "c2-standard-60",
	}
}

// Configure configures the GCP adapter
func (a *Adapter) Configure(config interfaces.CloudConfig) error {
	gcpConfig, ok := config.(*Config)
	if !ok {
		return fmt.Errorf("invalid config type, expected *gcp.Config")
	}

	if err := gcpConfig.Validate(); err != nil {
		return fmt.Errorf("config validation failed: %w", err)
	}

	a.config = gcpConfig

	ctx := context.Background()
	
	// Configure client options
	var opts []option.ClientOption
	
	if gcpConfig.CredentialsJSON != "" {
		opts = append(opts, option.WithCredentialsJSON([]byte(gcpConfig.CredentialsJSON)))
	} else if gcpConfig.CredentialsFile != "" {
		opts = append(opts, option.WithCredentialsFile(gcpConfig.CredentialsFile))
	}
	
	if gcpConfig.ImpersonateServiceAccount != "" {
		opts = append(opts, option.ImpersonateCredentials(gcpConfig.ImpersonateServiceAccount))
	}

	// Initialize clients
	var err error
	
	a.instancesClient, err = compute.NewInstancesRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create instances client: %w", err)
	}

	a.disksClient, err = compute.NewDisksRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create disks client: %w", err)
	}

	a.networksClient, err = compute.NewNetworksRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create networks client: %w", err)
	}

	a.subnetworksClient, err = compute.NewSubnetworksRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create subnetworks client: %w", err)
	}

	a.firewallsClient, err = compute.NewFirewallsRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create firewalls client: %w", err)
	}

	a.regionsClient, err = compute.NewRegionsRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create regions client: %w", err)
	}

	a.zonesClient, err = compute.NewZonesRESTClient(ctx, opts...)
	if err != nil {
		return fmt.Errorf("failed to create zones client: %w", err)
	}

	return nil
}

// ValidateCredentials validates GCP credentials
func (a *Adapter) ValidateCredentials(ctx context.Context) error {
	if a.instancesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	// Try to list instances to validate credentials
	req := &computepb.ListInstancesRequest{
		Project: a.config.ProjectID,
		Zone:    a.config.Zone,
	}

	it := a.instancesClient.List(ctx, req)
	_, err := it.Next()
	
	// It's OK if there are no instances, we just want to validate credentials
	if err != nil && !strings.Contains(err.Error(), "no more items") {
		return err
	}

	return nil
}

// CreateInstance creates a new GCP Compute Engine instance
func (a *Adapter) CreateInstance(ctx context.Context, req *interfaces.CreateInstanceRequest) (*interfaces.Instance, error) {
	if a.instancesClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	instanceName := req.Name
	if instanceName == "" {
		instanceName = fmt.Sprintf("instance-%d", time.Now().Unix())
	}

	// Create instance configuration
	instance := &computepb.Instance{
		Name:        proto.String(instanceName),
		MachineType: proto.String(fmt.Sprintf("zones/%s/machineTypes/%s", a.config.Zone, req.InstanceType)),
		Disks: []*computepb.AttachedDisk{
			{
				Boot:       proto.Bool(true),
				AutoDelete: proto.Bool(true),
				InitializeParams: &computepb.AttachedDiskInitializeParams{
					DiskSizeGb:  proto.Int64(int64(max(10, req.RootVolumeSize))),
					SourceImage: proto.String(req.ImageID),
				},
			},
		},
		NetworkInterfaces: []*computepb.NetworkInterface{
			{
				AccessConfigs: []*computepb.AccessConfig{
					{
						Type: proto.String(computepb.AccessConfig_ONE_TO_ONE_NAT.String()),
						Name: proto.String("External NAT"),
					},
				},
			},
		},
		ServiceAccounts: []*computepb.ServiceAccount{
			{
				Email: proto.String("default"),
				Scopes: []string{
					"https://www.googleapis.com/auth/devstorage.read_only",
					"https://www.googleapis.com/auth/logging.write",
					"https://www.googleapis.com/auth/monitoring.write",
				},
			},
		},
	}

	// Set subnet if provided
	if req.SubnetID != "" {
		instance.NetworkInterfaces[0].Subnetwork = proto.String(req.SubnetID)
	}

	// Add metadata for user data
	if req.UserData != "" {
		instance.Metadata = &computepb.Metadata{
			Items: []*computepb.Items{
				{
					Key:   proto.String("user-data"),
					Value: proto.String(req.UserData),
				},
				{
					Key:   proto.String("startup-script"),
					Value: proto.String(req.UserData),
				},
			},
		}
	}

	// Add SSH keys if provided
	if req.KeyPairName != "" {
		if instance.Metadata == nil {
			instance.Metadata = &computepb.Metadata{}
		}
		instance.Metadata.Items = append(instance.Metadata.Items, &computepb.Items{
			Key:   proto.String("ssh-keys"),
			Value: proto.String(fmt.Sprintf("gce-user:%s", req.KeyPairName)),
		})
	}

	// Add labels (GCP's equivalent of tags)
	if len(req.Tags) > 0 {
		labels := make(map[string]string)
		for k, v := range req.Tags {
			// GCP labels have restrictions on keys and values
			key := strings.ToLower(strings.ReplaceAll(k, "_", "-"))
			value := strings.ToLower(strings.ReplaceAll(v, "_", "-"))
			labels[key] = value
		}
		instance.Labels = labels
	}

	// Configure disk type for root volume
	if req.RootVolumeType != "" {
		diskType := fmt.Sprintf("zones/%s/diskTypes/%s", a.config.Zone, req.RootVolumeType)
		instance.Disks[0].InitializeParams.DiskType = proto.String(diskType)
	}

	// Create the instance
	insertReq := &computepb.InsertInstanceRequest{
		Project:          a.config.ProjectID,
		Zone:             a.config.Zone,
		InstanceResource: instance,
	}

	op, err := a.instancesClient.Insert(ctx, insertReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance: %w", err)
	}

	// Wait for the operation to complete
	if err := a.waitForZoneOperation(ctx, op); err != nil {
		return nil, fmt.Errorf("failed to wait for instance creation: %w", err)
	}

	// Get the created instance
	return a.GetInstance(ctx, instanceName)
}

// GetInstance retrieves a GCP instance by name
func (a *Adapter) GetInstance(ctx context.Context, instanceID string) (*interfaces.Instance, error) {
	if a.instancesClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	req := &computepb.GetInstanceRequest{
		Project:  a.config.ProjectID,
		Zone:     a.config.Zone,
		Instance: instanceID,
	}

	instance, err := a.instancesClient.Get(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get instance: %w", err)
	}

	return a.convertGCPInstance(instance), nil
}

// ListInstances lists GCP instances with optional filtering
func (a *Adapter) ListInstances(ctx context.Context, filters *interfaces.ListInstanceFilters) ([]*interfaces.Instance, error) {
	if a.instancesClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	req := &computepb.ListInstancesRequest{
		Project: a.config.ProjectID,
		Zone:    a.config.Zone,
	}

	// Apply filters
	if filters != nil {
		var filterParts []string
		
		// Filter by states
		if len(filters.States) > 0 {
			states := make([]string, 0, len(filters.States))
			for _, state := range filters.States {
				gcpState := a.convertToGCPState(state)
				if gcpState != "" {
					states = append(states, fmt.Sprintf("status = %s", gcpState))
				}
			}
			if len(states) > 0 {
				filterParts = append(filterParts, fmt.Sprintf("(%s)", strings.Join(states, " OR ")))
			}
		}
		
		// Filter by machine type
		if filters.InstanceType != "" {
			machineType := fmt.Sprintf("zones/%s/machineTypes/%s", a.config.Zone, filters.InstanceType)
			filterParts = append(filterParts, fmt.Sprintf("machineType = %s", machineType))
		}
		
		// Filter by labels (tags)
		for key, value := range filters.Tags {
			gcpKey := strings.ToLower(strings.ReplaceAll(key, "_", "-"))
			gcpValue := strings.ToLower(strings.ReplaceAll(value, "_", "-"))
			filterParts = append(filterParts, fmt.Sprintf("labels.%s = %s", gcpKey, gcpValue))
		}
		
		if len(filterParts) > 0 {
			req.Filter = proto.String(strings.Join(filterParts, " AND "))
		}
	}

	var instances []*interfaces.Instance
	it := a.instancesClient.List(ctx, req)
	
	for {
		instance, err := it.Next()
		if err != nil {
			if strings.Contains(err.Error(), "no more items") {
				break
			}
			return nil, fmt.Errorf("failed to list instances: %w", err)
		}
		
		instances = append(instances, a.convertGCPInstance(instance))
	}

	return instances, nil
}

// UpdateInstance updates a GCP instance
func (a *Adapter) UpdateInstance(ctx context.Context, instanceID string, updates *interfaces.UpdateInstanceRequest) (*interfaces.Instance, error) {
	if a.instancesClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	// Get current instance
	current, err := a.GetInstance(ctx, instanceID)
	if err != nil {
		return nil, fmt.Errorf("failed to get current instance: %w", err)
	}

	// Update machine type if specified and instance is stopped
	if updates.InstanceType != nil {
		if current.State != interfaces.InstanceStateStopped {
			return nil, fmt.Errorf("instance must be stopped to change machine type")
		}

		machineType := fmt.Sprintf("zones/%s/machineTypes/%s", a.config.Zone, *updates.InstanceType)
		setMachineTypeReq := &computepb.SetMachineTypeInstanceRequest{
			Project:  a.config.ProjectID,
			Zone:     a.config.Zone,
			Instance: instanceID,
			InstancesSetMachineTypeRequestResource: &computepb.InstancesSetMachineTypeRequest{
				MachineType: proto.String(machineType),
			},
		}

		op, err := a.instancesClient.SetMachineType(ctx, setMachineTypeReq)
		if err != nil {
			return nil, fmt.Errorf("failed to set machine type: %w", err)
		}

		if err := a.waitForZoneOperation(ctx, op); err != nil {
			return nil, fmt.Errorf("failed to wait for machine type change: %w", err)
		}
	}

	// Update labels if specified
	if len(updates.Tags) > 0 {
		instance, err := a.instancesClient.Get(ctx, &computepb.GetInstanceRequest{
			Project:  a.config.ProjectID,
			Zone:     a.config.Zone,
			Instance: instanceID,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get instance for label update: %w", err)
		}

		labels := make(map[string]string)
		if instance.Labels != nil {
			for k, v := range instance.Labels {
				labels[k] = v
			}
		}

		// Add new labels
		for k, v := range updates.Tags {
			key := strings.ToLower(strings.ReplaceAll(k, "_", "-"))
			value := strings.ToLower(strings.ReplaceAll(v, "_", "-"))
			labels[key] = value
		}

		setLabelsReq := &computepb.SetLabelsInstanceRequest{
			Project:  a.config.ProjectID,
			Zone:     a.config.Zone,
			Instance: instanceID,
			InstancesSetLabelsRequestResource: &computepb.InstancesSetLabelsRequest{
				Labels:           labels,
				LabelFingerprint: instance.LabelFingerprint,
			},
		}

		op, err := a.instancesClient.SetLabels(ctx, setLabelsReq)
		if err != nil {
			return nil, fmt.Errorf("failed to set labels: %w", err)
		}

		if err := a.waitForZoneOperation(ctx, op); err != nil {
			return nil, fmt.Errorf("failed to wait for label update: %w", err)
		}
	}

	// Return updated instance
	return a.GetInstance(ctx, instanceID)
}

// DeleteInstance deletes a GCP instance
func (a *Adapter) DeleteInstance(ctx context.Context, instanceID string, force bool) error {
	if a.instancesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	req := &computepb.DeleteInstanceRequest{
		Project:  a.config.ProjectID,
		Zone:     a.config.Zone,
		Instance: instanceID,
	}

	op, err := a.instancesClient.Delete(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to delete instance: %w", err)
	}

	return a.waitForZoneOperation(ctx, op)
}

// StartInstance starts a GCP instance
func (a *Adapter) StartInstance(ctx context.Context, instanceID string) error {
	if a.instancesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	req := &computepb.StartInstanceRequest{
		Project:  a.config.ProjectID,
		Zone:     a.config.Zone,
		Instance: instanceID,
	}

	op, err := a.instancesClient.Start(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to start instance: %w", err)
	}

	return a.waitForZoneOperation(ctx, op)
}

// StopInstance stops a GCP instance
func (a *Adapter) StopInstance(ctx context.Context, instanceID string, force bool) error {
	if a.instancesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	req := &computepb.StopInstanceRequest{
		Project:  a.config.ProjectID,
		Zone:     a.config.Zone,
		Instance: instanceID,
	}

	op, err := a.instancesClient.Stop(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to stop instance: %w", err)
	}

	return a.waitForZoneOperation(ctx, op)
}

// RebootInstance reboots a GCP instance
func (a *Adapter) RebootInstance(ctx context.Context, instanceID string) error {
	if a.instancesClient == nil {
		return fmt.Errorf("adapter not configured")
	}

	req := &computepb.ResetInstanceRequest{
		Project:  a.config.ProjectID,
		Zone:     a.config.Zone,
		Instance: instanceID,
	}

	op, err := a.instancesClient.Reset(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to reboot instance: %w", err)
	}

	return a.waitForZoneOperation(ctx, op)
}

// Helper method to convert GCP instance to interface instance
func (a *Adapter) convertGCPInstance(instance *computepb.Instance) *interfaces.Instance {
	result := &interfaces.Instance{
		Provider: "gcp",
		Region:   a.config.Region,
		Zone:     a.config.Zone,
		Tags:     make(map[string]string),
		Metadata: make(map[string]interface{}),
	}

	if instance.Name != nil {
		result.ID = *instance.Name
		result.Name = *instance.Name
	}

	if instance.Status != nil {
		result.State = a.convertGCPInstanceState(*instance.Status)
	}

	if instance.MachineType != nil {
		// Extract machine type from full path
		parts := strings.Split(*instance.MachineType, "/")
		if len(parts) > 0 {
			result.InstanceType = parts[len(parts)-1]
		}
	}

	if instance.CreationTimestamp != nil {
		if createdAt, err := time.Parse(time.RFC3339, *instance.CreationTimestamp); err == nil {
			result.CreatedAt = createdAt
		}
	}

	// Extract network information
	for _, netInterface := range instance.NetworkInterfaces {
		if netInterface.NetworkIP != nil {
			result.PrivateIP = *netInterface.NetworkIP
		}
		
		for _, accessConfig := range netInterface.AccessConfigs {
			if accessConfig.NatIP != nil {
				result.PublicIP = *accessConfig.NatIP
			}
		}
		
		if netInterface.Subnetwork != nil {
			result.SubnetID = *netInterface.Subnetwork
		}
	}

	// Extract boot disk image
	for _, disk := range instance.Disks {
		if disk.Boot != nil && *disk.Boot {
			if disk.Source != nil {
				result.VolumeIDs = append(result.VolumeIDs, *disk.Source)
			}
		}
	}

	// Convert labels to tags
	if instance.Labels != nil {
		for k, v := range instance.Labels {
			result.Tags[k] = v
		}
	}

	return result
}

// Helper method to convert GCP instance state
func (a *Adapter) convertGCPInstanceState(status string) interfaces.InstanceState {
	switch status {
	case "PROVISIONING", "STAGING":
		return interfaces.InstanceStatePending
	case "RUNNING":
		return interfaces.InstanceStateRunning
	case "STOPPING":
		return interfaces.InstanceStateStopping
	case "STOPPED", "TERMINATED":
		return interfaces.InstanceStateStopped
	case "SUSPENDING", "SUSPENDED":
		return interfaces.InstanceStateStopped
	default:
		return interfaces.InstanceStateUnknown
	}
}

// Helper method to convert interface state to GCP state
func (a *Adapter) convertToGCPState(state interfaces.InstanceState) string {
	switch state {
	case interfaces.InstanceStatePending:
		return "PROVISIONING"
	case interfaces.InstanceStateRunning:
		return "RUNNING"
	case interfaces.InstanceStateStopping:
		return "STOPPING"
	case interfaces.InstanceStateStopped:
		return "TERMINATED"
	default:
		return ""
	}
}

// Helper method to wait for zone operation completion
func (a *Adapter) waitForZoneOperation(ctx context.Context, op *computepb.Operation) error {
	if op.Name == nil {
		return fmt.Errorf("operation name is nil")
	}

	zoneOpsClient, err := compute.NewZoneOperationsRESTClient(ctx)
	if err != nil {
		return fmt.Errorf("failed to create zone operations client: %w", err)
	}
	defer zoneOpsClient.Close()

	for {
		req := &computepb.GetZoneOperationRequest{
			Project:   a.config.ProjectID,
			Zone:      a.config.Zone,
			Operation: *op.Name,
		}

		result, err := zoneOpsClient.Get(ctx, req)
		if err != nil {
			return fmt.Errorf("failed to get operation status: %w", err)
		}

		if result.Status != nil && *result.Status == computepb.Operation_DONE.String() {
			if result.Error != nil {
				return fmt.Errorf("operation failed: %v", result.Error)
			}
			return nil
		}

		// Wait before checking again
		time.Sleep(2 * time.Second)
	}
}

// Implement remaining interface methods (simplified for space)

func (a *Adapter) GetInstanceMetrics(ctx context.Context, instanceID string, opts *interfaces.MetricsOptions) (*interfaces.InstanceMetrics, error) {
	// Would integrate with Cloud Monitoring
	return &interfaces.InstanceMetrics{
		InstanceID: instanceID,
		Period:     opts.Period,
		StartTime:  opts.StartTime,
		EndTime:    opts.EndTime,
	}, nil
}

func (a *Adapter) GetInstanceLogs(ctx context.Context, instanceID string, opts *interfaces.LogOptions) (*interfaces.InstanceLogs, error) {
	// Would integrate with Cloud Logging
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
		Provider:        "gcp",
		Region:          a.config.Region,
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

// Helper function for max
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}