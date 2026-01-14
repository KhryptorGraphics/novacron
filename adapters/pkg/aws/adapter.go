// Package aws implements the AWS cloud adapter for NovaCron
package aws

import (
	"context"
	"fmt"
// 	"strconv"
// 	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/cloudwatch"
	"github.com/aws/aws-sdk-go/service/costexplorer"

	"github.com/khryptorgraphics/novacron/adapters/pkg/interfaces"
)

// Adapter implements the CloudAdapter interface for AWS
type Adapter struct {
	config      *Config
	session     *session.Session
	ec2Client   *ec2.EC2
	cwClient    *cloudwatch.CloudWatch
	costClient  *costexplorer.CostExplorer
}

// Config represents AWS-specific configuration
type Config struct {
	Region          string `json:"region"`
	AccessKeyID     string `json:"access_key_id"`
	SecretAccessKey string `json:"secret_access_key"`
	SessionToken    string `json:"session_token,omitempty"`
	Profile         string `json:"profile,omitempty"`
	RoleArn         string `json:"role_arn,omitempty"`
	ExternalID      string `json:"external_id,omitempty"`
	AssumeRole      bool   `json:"assume_role,omitempty"`
	Endpoint        string `json:"endpoint,omitempty"`
}

// GetProvider returns the cloud provider name
func (c *Config) GetProvider() string {
	return "aws"
}

// GetRegion returns the AWS region
func (c *Config) GetRegion() string {
	return c.Region
}

// GetCredentials returns the credentials map
func (c *Config) GetCredentials() map[string]string {
	return map[string]string{
		"access_key_id":     c.AccessKeyID,
		"secret_access_key": c.SecretAccessKey,
		"session_token":     c.SessionToken,
		"profile":           c.Profile,
		"role_arn":          c.RoleArn,
		"external_id":       c.ExternalID,
	}
}

// GetSettings returns additional settings
func (c *Config) GetSettings() map[string]interface{} {
	return map[string]interface{}{
		"assume_role": c.AssumeRole,
		"endpoint":    c.Endpoint,
	}
}

// Validate validates the AWS configuration
func (c *Config) Validate() error {
	if c.Region == "" {
		return fmt.Errorf("region is required")
	}
	
	if c.AccessKeyID == "" && c.Profile == "" && c.RoleArn == "" {
		return fmt.Errorf("either access_key_id, profile, or role_arn must be provided")
	}
	
	if c.AccessKeyID != "" && c.SecretAccessKey == "" {
		return fmt.Errorf("secret_access_key is required when access_key_id is provided")
	}
	
	return nil
}

// NewAdapter creates a new AWS adapter
func NewAdapter() *Adapter {
	return &Adapter{}
}

// Name returns the adapter name
func (a *Adapter) Name() string {
	return "aws-ec2-adapter"
}

// Version returns the adapter version
func (a *Adapter) Version() string {
	return "1.0.0"
}

// SupportedRegions returns the list of supported AWS regions
func (a *Adapter) SupportedRegions() []string {
	return []string{
		"us-east-1", "us-east-2", "us-west-1", "us-west-2",
		"eu-north-1", "eu-west-3", "eu-west-2", "eu-west-1", "eu-central-1",
		"ap-northeast-3", "ap-northeast-2", "ap-northeast-1",
		"ap-southeast-1", "ap-southeast-2", "ap-south-1",
		"ca-central-1", "sa-east-1",
		"af-south-1", "eu-south-1", "me-south-1",
	}
}

// SupportedInstanceTypes returns the list of supported instance types
func (a *Adapter) SupportedInstanceTypes() []string {
	return []string{
		"t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge",
		"t3a.nano", "t3a.micro", "t3a.small", "t3a.medium", "t3a.large", "t3a.xlarge", "t3a.2xlarge",
		"m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.8xlarge", "m5.12xlarge", "m5.16xlarge", "m5.24xlarge",
		"m5a.large", "m5a.xlarge", "m5a.2xlarge", "m5a.4xlarge", "m5a.8xlarge", "m5a.12xlarge", "m5a.16xlarge", "m5a.24xlarge",
		"c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge", "c5.9xlarge", "c5.12xlarge", "c5.18xlarge", "c5.24xlarge",
		"r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge", "r5.8xlarge", "r5.12xlarge", "r5.16xlarge", "r5.24xlarge",
	}
}

// Configure configures the AWS adapter
func (a *Adapter) Configure(config interfaces.CloudConfig) error {
	awsConfig, ok := config.(*Config)
	if !ok {
		return fmt.Errorf("invalid config type, expected *aws.Config")
	}

	if err := awsConfig.Validate(); err != nil {
		return fmt.Errorf("config validation failed: %w", err)
	}

	a.config = awsConfig

	// Create AWS session
	sessConfig := &aws.Config{
		Region: aws.String(awsConfig.Region),
	}

	if awsConfig.Endpoint != "" {
		sessConfig.Endpoint = aws.String(awsConfig.Endpoint)
	}

	// Configure credentials
	if awsConfig.AccessKeyID != "" {
		sessConfig.Credentials = credentials.NewStaticCredentials(
			awsConfig.AccessKeyID,
			awsConfig.SecretAccessKey,
			awsConfig.SessionToken,
		)
	} else if awsConfig.Profile != "" {
		sessConfig.Credentials = credentials.NewSharedCredentials("", awsConfig.Profile)
	}

	sess, err := session.NewSession(sessConfig)
	if err != nil {
		return fmt.Errorf("failed to create AWS session: %w", err)
	}

	a.session = sess
	a.ec2Client = ec2.New(sess)
	a.cwClient = cloudwatch.New(sess)
	a.costClient = costexplorer.New(sess)

	return nil
}

// ValidateCredentials validates AWS credentials
func (a *Adapter) ValidateCredentials(ctx context.Context) error {
	if a.ec2Client == nil {
		return fmt.Errorf("adapter not configured")
	}

	_, err := a.ec2Client.DescribeRegionsWithContext(ctx, &ec2.DescribeRegionsInput{})
	return err
}

// CreateInstance creates a new EC2 instance
func (a *Adapter) CreateInstance(ctx context.Context, req *interfaces.CreateInstanceRequest) (*interfaces.Instance, error) {
	if a.ec2Client == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	runInput := &ec2.RunInstancesInput{
		ImageId:      aws.String(req.ImageID),
		InstanceType: aws.String(req.InstanceType),
		MinCount:     aws.Int64(int64(max(1, req.MinCount))),
		MaxCount:     aws.Int64(int64(max(1, req.MaxCount))),
	}

	if req.KeyPairName != "" {
		runInput.KeyName = aws.String(req.KeyPairName)
	}

	if len(req.SecurityGroupIDs) > 0 {
		runInput.SecurityGroupIds = aws.StringSlice(req.SecurityGroupIDs)
	}

	if req.SubnetID != "" {
		runInput.SubnetId = aws.String(req.SubnetID)
	}

	if req.UserData != "" {
		runInput.UserData = aws.String(req.UserData)
	}

	if req.IamInstanceProfile != "" {
		runInput.IamInstanceProfile = &ec2.IamInstanceProfileSpecification{
			Name: aws.String(req.IamInstanceProfile),
		}
	}

	// Configure root volume
	if req.RootVolumeSize > 0 || req.RootVolumeType != "" {
		blockDevice := &ec2.BlockDeviceMapping{
			DeviceName: aws.String("/dev/sda1"),
			Ebs: &ec2.EbsBlockDevice{
				DeleteOnTermination: aws.Bool(true),
			},
		}

		if req.RootVolumeSize > 0 {
			blockDevice.Ebs.VolumeSize = aws.Int64(int64(req.RootVolumeSize))
		}

		if req.RootVolumeType != "" {
			blockDevice.Ebs.VolumeType = aws.String(req.RootVolumeType)
		}

		runInput.BlockDeviceMappings = []*ec2.BlockDeviceMapping{blockDevice}
	}

	// Add tags
	if len(req.Tags) > 0 || req.Name != "" {
		tagSpec := &ec2.TagSpecification{
			ResourceType: aws.String("instance"),
			Tags:         []*ec2.Tag{},
		}

		if req.Name != "" {
			tagSpec.Tags = append(tagSpec.Tags, &ec2.Tag{
				Key:   aws.String("Name"),
				Value: aws.String(req.Name),
			})
		}

		for key, value := range req.Tags {
			tagSpec.Tags = append(tagSpec.Tags, &ec2.Tag{
				Key:   aws.String(key),
				Value: aws.String(value),
			})
		}

		runInput.TagSpecifications = []*ec2.TagSpecification{tagSpec}
	}

	result, err := a.ec2Client.RunInstancesWithContext(ctx, runInput)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance: %w", err)
	}

	if len(result.Instances) == 0 {
		return nil, fmt.Errorf("no instances created")
	}

	return a.convertEC2Instance(result.Instances[0]), nil
}

// GetInstance retrieves an EC2 instance by ID
func (a *Adapter) GetInstance(ctx context.Context, instanceID string) (*interfaces.Instance, error) {
	if a.ec2Client == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	result, err := a.ec2Client.DescribeInstancesWithContext(ctx, &ec2.DescribeInstancesInput{
		InstanceIds: []*string{aws.String(instanceID)},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to describe instance: %w", err)
	}

	for _, reservation := range result.Reservations {
		for _, instance := range reservation.Instances {
			if *instance.InstanceId == instanceID {
				return a.convertEC2Instance(instance), nil
			}
		}
	}

	return nil, fmt.Errorf("instance not found: %s", instanceID)
}

// ListInstances lists EC2 instances with optional filtering
func (a *Adapter) ListInstances(ctx context.Context, filters *interfaces.ListInstanceFilters) ([]*interfaces.Instance, error) {
	if a.ec2Client == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	input := &ec2.DescribeInstancesInput{}

	if filters != nil {
		var ec2Filters []*ec2.Filter

		// Filter by states
		if len(filters.States) > 0 {
			states := make([]*string, 0, len(filters.States))
			for _, state := range filters.States {
				states = append(states, aws.String(string(state)))
			}
			ec2Filters = append(ec2Filters, &ec2.Filter{
				Name:   aws.String("instance-state-name"),
				Values: states,
			})
		}

		// Filter by instance type
		if filters.InstanceType != "" {
			ec2Filters = append(ec2Filters, &ec2.Filter{
				Name:   aws.String("instance-type"),
				Values: []*string{aws.String(filters.InstanceType)},
			})
		}

		// Filter by image ID
		if filters.ImageID != "" {
			ec2Filters = append(ec2Filters, &ec2.Filter{
				Name:   aws.String("image-id"),
				Values: []*string{aws.String(filters.ImageID)},
			})
		}

		// Filter by zone
		if filters.Zone != "" {
			ec2Filters = append(ec2Filters, &ec2.Filter{
				Name:   aws.String("placement-availability-zone"),
				Values: []*string{aws.String(filters.Zone)},
			})
		}

		// Filter by tags
		for key, value := range filters.Tags {
			ec2Filters = append(ec2Filters, &ec2.Filter{
				Name:   aws.String(fmt.Sprintf("tag:%s", key)),
				Values: []*string{aws.String(value)},
			})
		}

		input.Filters = ec2Filters

		// Filter by instance IDs
		if len(filters.InstanceIDs) > 0 {
			input.InstanceIds = aws.StringSlice(filters.InstanceIDs)
		}
	}

	var instances []*interfaces.Instance
	err := a.ec2Client.DescribeInstancesPagesWithContext(ctx, input,
		func(page *ec2.DescribeInstancesOutput, lastPage bool) bool {
			for _, reservation := range page.Reservations {
				for _, instance := range reservation.Instances {
					instances = append(instances, a.convertEC2Instance(instance))
				}
			}
			return !lastPage
		})

	if err != nil {
		return nil, fmt.Errorf("failed to list instances: %w", err)
	}

	return instances, nil
}

// UpdateInstance updates an EC2 instance
func (a *Adapter) UpdateInstance(ctx context.Context, instanceID string, updates *interfaces.UpdateInstanceRequest) (*interfaces.Instance, error) {
	if a.ec2Client == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	// Update instance type if specified
	if updates.InstanceType != nil {
		_, err := a.ec2Client.ModifyInstanceAttributeWithContext(ctx, &ec2.ModifyInstanceAttributeInput{
			InstanceId: aws.String(instanceID),
			InstanceType: &ec2.AttributeValue{
				Value: updates.InstanceType,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("failed to update instance type: %w", err)
		}
	}

	// Update tags if specified
	if len(updates.Tags) > 0 {
		var tags []*ec2.Tag
		for key, value := range updates.Tags {
			tags = append(tags, &ec2.Tag{
				Key:   aws.String(key),
				Value: aws.String(value),
			})
		}

		// Add/update Name tag if specified
		if updates.Name != nil {
			tags = append(tags, &ec2.Tag{
				Key:   aws.String("Name"),
				Value: updates.Name,
			})
		}

		_, err := a.ec2Client.CreateTagsWithContext(ctx, &ec2.CreateTagsInput{
			Resources: []*string{aws.String(instanceID)},
			Tags:      tags,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to update tags: %w", err)
		}
	}

	// Return updated instance
	return a.GetInstance(ctx, instanceID)
}

// DeleteInstance terminates an EC2 instance
func (a *Adapter) DeleteInstance(ctx context.Context, instanceID string, force bool) error {
	if a.ec2Client == nil {
		return fmt.Errorf("adapter not configured")
	}

	// Disable termination protection if force is true
	if force {
		_, err := a.ec2Client.ModifyInstanceAttributeWithContext(ctx, &ec2.ModifyInstanceAttributeInput{
			InstanceId: aws.String(instanceID),
			DisableApiTermination: &ec2.AttributeBooleanValue{
				Value: aws.Bool(false),
			},
		})
		if err != nil {
			// Continue even if this fails - the instance might not have termination protection
		}
	}

	_, err := a.ec2Client.TerminateInstancesWithContext(ctx, &ec2.TerminateInstancesInput{
		InstanceIds: []*string{aws.String(instanceID)},
	})
	if err != nil {
		return fmt.Errorf("failed to terminate instance: %w", err)
	}

	return nil
}

// StartInstance starts an EC2 instance
func (a *Adapter) StartInstance(ctx context.Context, instanceID string) error {
	if a.ec2Client == nil {
		return fmt.Errorf("adapter not configured")
	}

	_, err := a.ec2Client.StartInstancesWithContext(ctx, &ec2.StartInstancesInput{
		InstanceIds: []*string{aws.String(instanceID)},
	})
	if err != nil {
		return fmt.Errorf("failed to start instance: %w", err)
	}

	return nil
}

// StopInstance stops an EC2 instance
func (a *Adapter) StopInstance(ctx context.Context, instanceID string, force bool) error {
	if a.ec2Client == nil {
		return fmt.Errorf("adapter not configured")
	}

	_, err := a.ec2Client.StopInstancesWithContext(ctx, &ec2.StopInstancesInput{
		InstanceIds: []*string{aws.String(instanceID)},
		Force:       aws.Bool(force),
	})
	if err != nil {
		return fmt.Errorf("failed to stop instance: %w", err)
	}

	return nil
}

// RebootInstance reboots an EC2 instance
func (a *Adapter) RebootInstance(ctx context.Context, instanceID string) error {
	if a.ec2Client == nil {
		return fmt.Errorf("adapter not configured")
	}

	_, err := a.ec2Client.RebootInstancesWithContext(ctx, &ec2.RebootInstancesInput{
		InstanceIds: []*string{aws.String(instanceID)},
	})
	if err != nil {
		return fmt.Errorf("failed to reboot instance: %w", err)
	}

	return nil
}

// Helper method to convert EC2 instance to interface instance
func (a *Adapter) convertEC2Instance(instance *ec2.Instance) *interfaces.Instance {
	result := &interfaces.Instance{
		ID:           *instance.InstanceId,
		Provider:     "aws",
		Region:       a.config.Region,
		InstanceType: *instance.InstanceType,
		State:        a.convertInstanceState(*instance.State.Name),
		ImageID:      *instance.ImageId,
		CreatedAt:    *instance.LaunchTime,
		Tags:         make(map[string]string),
		Metadata:     make(map[string]interface{}),
	}

	if instance.Placement != nil && instance.Placement.AvailabilityZone != nil {
		result.Zone = *instance.Placement.AvailabilityZone
	}

	if instance.PublicIpAddress != nil {
		result.PublicIP = *instance.PublicIpAddress
	}

	if instance.PrivateIpAddress != nil {
		result.PrivateIP = *instance.PrivateIpAddress
	}

	if instance.KeyName != nil {
		result.KeyPairName = *instance.KeyName
	}

	if instance.SubnetId != nil {
		result.SubnetID = *instance.SubnetId
	}

	if instance.Platform != nil {
		result.Platform = *instance.Platform
	}

	if instance.Architecture != nil {
		result.Architecture = *instance.Architecture
	}

	if instance.Hypervisor != nil {
		result.Hypervisor = *instance.Hypervisor
	}

	if instance.VirtualizationType != nil {
		result.VirtualizationType = *instance.VirtualizationType
	}

	// Extract security groups
	for _, sg := range instance.SecurityGroups {
		if sg.GroupId != nil {
			result.SecurityGroups = append(result.SecurityGroups, *sg.GroupId)
		}
	}

	// Extract volume IDs
	for _, bdm := range instance.BlockDeviceMappings {
		if bdm.Ebs != nil && bdm.Ebs.VolumeId != nil {
			result.VolumeIDs = append(result.VolumeIDs, *bdm.Ebs.VolumeId)
		}
	}

	// Extract tags
	for _, tag := range instance.Tags {
		if tag.Key != nil && tag.Value != nil {
			if *tag.Key == "Name" {
				result.Name = *tag.Value
			} else {
				result.Tags[*tag.Key] = *tag.Value
			}
		}
	}

	return result
}

// Helper method to convert EC2 instance state to interface state
func (a *Adapter) convertInstanceState(state string) interfaces.InstanceState {
	switch state {
	case "pending":
		return interfaces.InstanceStatePending
	case "running":
		return interfaces.InstanceStateRunning
	case "stopping":
		return interfaces.InstanceStateStopping
	case "stopped":
		return interfaces.InstanceStateStopped
	case "shutting-down":
		return interfaces.InstanceStateTerminating
	case "terminated":
		return interfaces.InstanceStateTerminated
	case "rebooting":
		return interfaces.InstanceStateRebooting
	default:
		return interfaces.InstanceStateUnknown
	}
}

// Implement remaining interface methods (simplified for space)

// GetInstanceMetrics retrieves CloudWatch metrics for an instance
func (a *Adapter) GetInstanceMetrics(ctx context.Context, instanceID string, opts *interfaces.MetricsOptions) (*interfaces.InstanceMetrics, error) {
	if a.cwClient == nil {
		return nil, fmt.Errorf("adapter not configured")
	}

	// Implementation would use CloudWatch API to get metrics
	// This is a simplified placeholder
	return &interfaces.InstanceMetrics{
		InstanceID: instanceID,
		Period:     opts.Period,
		StartTime:  opts.StartTime,
		EndTime:    opts.EndTime,
	}, nil
}

// GetInstanceLogs retrieves instance logs
func (a *Adapter) GetInstanceLogs(ctx context.Context, instanceID string, opts *interfaces.LogOptions) (*interfaces.InstanceLogs, error) {
	// AWS EC2 doesn't provide direct log access - would need to integrate with CloudWatch Logs
	return &interfaces.InstanceLogs{
		InstanceID: instanceID,
		LogGroups:  []*interfaces.LogGroup{},
	}, nil
}

// CreateVolume creates an EBS volume
func (a *Adapter) CreateVolume(ctx context.Context, req *interfaces.CreateVolumeRequest) (*interfaces.Volume, error) {
	// Implementation would use EC2 CreateVolume API
	return nil, fmt.Errorf("not implemented")
}

// AttachVolume attaches a volume to an instance
func (a *Adapter) AttachVolume(ctx context.Context, volumeID, instanceID string) error {
	return fmt.Errorf("not implemented")
}

// DetachVolume detaches a volume from an instance
func (a *Adapter) DetachVolume(ctx context.Context, volumeID string) error {
	return fmt.Errorf("not implemented")
}

// DeleteVolume deletes a volume
func (a *Adapter) DeleteVolume(ctx context.Context, volumeID string) error {
	return fmt.Errorf("not implemented")
}

// ListNetworks lists VPCs
func (a *Adapter) ListNetworks(ctx context.Context) ([]*interfaces.Network, error) {
	return nil, fmt.Errorf("not implemented")
}

// CreateSecurityGroup creates a security group
func (a *Adapter) CreateSecurityGroup(ctx context.Context, req *interfaces.CreateSecurityGroupRequest) (*interfaces.SecurityGroup, error) {
	return nil, fmt.Errorf("not implemented")
}

// DeleteSecurityGroup deletes a security group
func (a *Adapter) DeleteSecurityGroup(ctx context.Context, groupID string) error {
	return fmt.Errorf("not implemented")
}

// GetQuotas retrieves AWS service quotas
func (a *Adapter) GetQuotas(ctx context.Context) (*interfaces.ResourceQuotas, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetCosts retrieves cost data using Cost Explorer
func (a *Adapter) GetCosts(ctx context.Context, opts *interfaces.CostOptions) (*interfaces.CostData, error) {
	return nil, fmt.Errorf("not implemented")
}

// ExportInstance exports an instance as an AMI or VM image
func (a *Adapter) ExportInstance(ctx context.Context, instanceID string, opts *interfaces.ExportOptions) (*interfaces.ExportResult, error) {
	return nil, fmt.Errorf("not implemented")
}

// ImportInstance imports a VM image as an AMI
func (a *Adapter) ImportInstance(ctx context.Context, req *interfaces.ImportInstanceRequest) (*interfaces.Instance, error) {
	return nil, fmt.Errorf("not implemented")
}

// HealthCheck performs a health check
func (a *Adapter) HealthCheck(ctx context.Context) error {
	return a.ValidateCredentials(ctx)
}

// GetStatus returns adapter status
func (a *Adapter) GetStatus(ctx context.Context) (*interfaces.AdapterStatus, error) {
	status := &interfaces.AdapterStatus{
		Name:            a.Name(),
		Version:         a.Version(),
		Provider:        "aws",
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