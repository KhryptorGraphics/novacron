package multicloud

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// AWSIntegration provides AWS cloud integration capabilities
type AWSIntegration struct {
	config      AWSConfig
	ec2Client   *ec2.Client
	s3Client    *s3.Client
	cwClient    *cloudwatch.Client
	awsConfig   aws.Config
	mutex       sync.RWMutex
	instances   map[string]*AWSInstance
	migrations  map[string]*AWSMigration
	ctx         context.Context
	cancel      context.CancelFunc
}

// AWSConfig contains AWS-specific configuration
type AWSConfig struct {
	Region          string            `json:"region"`
	AccessKeyID     string            `json:"access_key_id"`
	SecretAccessKey string            `json:"secret_access_key"`
	SessionToken    string            `json:"session_token,omitempty"`
	DefaultVPC      string            `json:"default_vpc"`
	DefaultSubnet   string            `json:"default_subnet"`
	SecurityGroups  []string          `json:"security_groups"`
	S3Bucket        string            `json:"s3_bucket"`
	KeyPairName     string            `json:"key_pair_name"`
	Tags            map[string]string `json:"tags"`
	Endpoint        string            `json:"endpoint,omitempty"` // For LocalStack testing
}

// AWSInstance represents an AWS EC2 instance managed by NovaCron
type AWSInstance struct {
	InstanceID       string                  `json:"instance_id"`
	NovaCronVMID     string                  `json:"novacron_vm_id"`
	InstanceType     string                  `json:"instance_type"`
	State            ec2types.InstanceState  `json:"state"`
	LaunchTime       time.Time               `json:"launch_time"`
	PrivateIPAddress string                  `json:"private_ip"`
	PublicIPAddress  string                  `json:"public_ip"`
	VolumeIDs        []string                `json:"volume_ids"`
	SecurityGroups   []ec2types.GroupIdentifier `json:"security_groups"`
	SubnetID         string                  `json:"subnet_id"`
	VPCID            string                  `json:"vpc_id"`
	Tags             []ec2types.Tag          `json:"tags"`
	Metadata         map[string]string       `json:"metadata"`
}

// AWSMigration represents a VM migration to/from AWS
type AWSMigration struct {
	MigrationID  string                 `json:"migration_id"`
	Direction    MigrationDirection     `json:"direction"`
	VMID         string                 `json:"vm_id"`
	InstanceID   string                 `json:"instance_id,omitempty"`
	Status       MigrationStatus        `json:"status"`
	Progress     float64                `json:"progress"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time,omitempty"`
	Error        string                 `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// MigrationDirection indicates import or export
type MigrationDirection string

const (
	MigrationDirectionImport MigrationDirection = "import" // EC2 → NovaCron
	MigrationDirectionExport MigrationDirection = "export" // NovaCron → EC2
)

// MigrationStatus represents the status of a migration
type MigrationStatus string

const (
	MigrationStatusPending    MigrationStatus = "pending"
	MigrationStatusPreparing  MigrationStatus = "preparing"
	MigrationStatusTransferring MigrationStatus = "transferring"
	MigrationStatusFinalizing MigrationStatus = "finalizing"
	MigrationStatusCompleted  MigrationStatus = "completed"
	MigrationStatusFailed     MigrationStatus = "failed"
	MigrationStatusRollingBack MigrationStatus = "rolling_back"
	MigrationStatusRolledBack MigrationStatus = "rolled_back"
)

// NewAWSIntegration creates a new AWS integration instance
func NewAWSIntegration(cfg AWSConfig) (*AWSIntegration, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Validate configuration
	if err := validateAWSConfig(cfg); err != nil {
		cancel()
		return nil, fmt.Errorf("invalid AWS configuration: %w", err)
	}

	// Load AWS SDK configuration
	var awsCfg aws.Config
	var err error

	if cfg.Endpoint != "" {
		// Custom endpoint (LocalStack, etc.)
		awsCfg, err = config.LoadDefaultConfig(ctx,
			config.WithRegion(cfg.Region),
			config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
				cfg.AccessKeyID,
				cfg.SecretAccessKey,
				cfg.SessionToken,
			)),
		)
	} else {
		// Standard AWS configuration
		awsCfg, err = config.LoadDefaultConfig(ctx,
			config.WithRegion(cfg.Region),
			config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
				cfg.AccessKeyID,
				cfg.SecretAccessKey,
				cfg.SessionToken,
			)),
		)
	}

	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to load AWS configuration: %w", err)
	}

	integration := &AWSIntegration{
		config:     cfg,
		awsConfig:  awsCfg,
		ec2Client:  ec2.NewFromConfig(awsCfg),
		s3Client:   s3.NewFromConfig(awsCfg),
		cwClient:   cloudwatch.NewFromConfig(awsCfg),
		instances:  make(map[string]*AWSInstance),
		migrations: make(map[string]*AWSMigration),
		ctx:        ctx,
		cancel:     cancel,
	}

	log.Printf("AWS integration initialized for region %s", cfg.Region)
	return integration, nil
}

// validateAWSConfig validates AWS configuration
func validateAWSConfig(cfg AWSConfig) error {
	if cfg.Region == "" {
		return fmt.Errorf("region is required")
	}
	if cfg.AccessKeyID == "" {
		return fmt.Errorf("access_key_id is required")
	}
	if cfg.SecretAccessKey == "" {
		return fmt.Errorf("secret_access_key is required")
	}
	if cfg.S3Bucket == "" {
		return fmt.Errorf("s3_bucket is required for VM image storage")
	}
	return nil
}

// DiscoverInstances discovers existing EC2 instances that can be imported
func (a *AWSIntegration) DiscoverInstances(ctx context.Context, filters map[string][]string) ([]*AWSInstance, error) {
	// Build EC2 filters
	ec2Filters := make([]ec2types.Filter, 0)
	for key, values := range filters {
		filterValues := make([]string, len(values))
		copy(filterValues, values)
		ec2Filters = append(ec2Filters, ec2types.Filter{
			Name:   aws.String(key),
			Values: filterValues,
		})
	}

	// Add NovaCron tag filter if not already present
	hasNovaCronFilter := false
	for _, f := range ec2Filters {
		if *f.Name == "tag:ManagedBy" {
			hasNovaCronFilter = true
			break
		}
	}

	if !hasNovaCronFilter {
		ec2Filters = append(ec2Filters, ec2types.Filter{
			Name:   aws.String("tag:ManagedBy"),
			Values: []string{"NovaCron"},
		})
	}

	// Describe instances
	input := &ec2.DescribeInstancesInput{
		Filters: ec2Filters,
	}

	result, err := a.ec2Client.DescribeInstances(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to describe instances: %w", err)
	}

	// Parse results
	instances := make([]*AWSInstance, 0)
	for _, reservation := range result.Reservations {
		for _, instance := range reservation.Instances {
			awsInstance := a.convertEC2Instance(instance)
			instances = append(instances, awsInstance)

			// Cache instance
			a.mutex.Lock()
			a.instances[*instance.InstanceId] = awsInstance
			a.mutex.Unlock()
		}
	}

	log.Printf("Discovered %d AWS EC2 instances", len(instances))
	return instances, nil
}

// convertEC2Instance converts EC2 SDK instance to AWSInstance
func (a *AWSIntegration) convertEC2Instance(instance ec2types.Instance) *AWSInstance {
	awsInstance := &AWSInstance{
		InstanceID:       aws.ToString(instance.InstanceId),
		InstanceType:     string(instance.InstanceType),
		State:            *instance.State,
		LaunchTime:       aws.ToTime(instance.LaunchTime),
		PrivateIPAddress: aws.ToString(instance.PrivateIpAddress),
		PublicIPAddress:  aws.ToString(instance.PublicIpAddress),
		SecurityGroups:   instance.SecurityGroups,
		SubnetID:         aws.ToString(instance.SubnetId),
		VPCID:            aws.ToString(instance.VpcId),
		Tags:             instance.Tags,
		VolumeIDs:        make([]string, 0),
		Metadata:         make(map[string]string),
	}

	// Extract volume IDs
	for _, blockDevice := range instance.BlockDeviceMappings {
		if blockDevice.Ebs != nil {
			awsInstance.VolumeIDs = append(awsInstance.VolumeIDs, *blockDevice.Ebs.VolumeId)
		}
	}

	// Extract NovaCron VM ID from tags
	for _, tag := range instance.Tags {
		if aws.ToString(tag.Key) == "NovaCronVMID" {
			awsInstance.NovaCronVMID = aws.ToString(tag.Value)
		}
		awsInstance.Metadata[aws.ToString(tag.Key)] = aws.ToString(tag.Value)
	}

	return awsInstance
}

// ImportVM imports an EC2 instance into NovaCron
func (a *AWSIntegration) ImportVM(ctx context.Context, instanceID string, options map[string]interface{}) (*AWSMigration, error) {
	migration := &AWSMigration{
		MigrationID: fmt.Sprintf("import-%s-%d", instanceID, time.Now().Unix()),
		Direction:   MigrationDirectionImport,
		InstanceID:  instanceID,
		Status:      MigrationStatusPending,
		StartTime:   time.Now(),
		Metadata:    options,
	}

	a.mutex.Lock()
	a.migrations[migration.MigrationID] = migration
	a.mutex.Unlock()

	// Execute migration asynchronously
	go func() {
		if err := a.executeImportMigration(ctx, migration); err != nil {
			a.mutex.Lock()
			migration.Status = MigrationStatusFailed
			migration.Error = err.Error()
			migration.EndTime = time.Now()
			a.mutex.Unlock()
			log.Printf("Failed to import EC2 instance %s: %v", instanceID, err)
		}
	}()

	return migration, nil
}

// executeImportMigration performs the actual import migration
func (a *AWSIntegration) executeImportMigration(ctx context.Context, migration *AWSMigration) error {
	// Update status
	a.updateMigrationStatus(migration, MigrationStatusPreparing, 10)

	// 1. Get instance details
	instance, err := a.getInstanceDetails(ctx, migration.InstanceID)
	if err != nil {
		return fmt.Errorf("failed to get instance details: %w", err)
	}

	// 2. Create snapshot of EBS volumes
	a.updateMigrationStatus(migration, MigrationStatusPreparing, 20)
	snapshotIDs, err := a.createVolumeSnapshots(ctx, instance.VolumeIDs)
	if err != nil {
		return fmt.Errorf("failed to create volume snapshots: %w", err)
	}

	// 3. Export snapshots to S3
	a.updateMigrationStatus(migration, MigrationStatusTransferring, 40)
	s3Keys, err := a.exportSnapshotsToS3(ctx, snapshotIDs)
	if err != nil {
		return fmt.Errorf("failed to export snapshots to S3: %w", err)
	}

	// 4. Download from S3 to local storage
	a.updateMigrationStatus(migration, MigrationStatusTransferring, 60)
	localPaths, err := a.downloadFromS3(ctx, s3Keys)
	if err != nil {
		return fmt.Errorf("failed to download from S3: %w", err)
	}

	// 5. Create NovaCron VM from imported data
	a.updateMigrationStatus(migration, MigrationStatusFinalizing, 80)
	vmID, err := a.createNovaCronVMFromImport(ctx, instance, localPaths)
	if err != nil {
		return fmt.Errorf("failed to create NovaCron VM: %w", err)
	}

	migration.VMID = vmID

	// 6. Optionally terminate source instance
	if shouldTerminate, ok := migration.Metadata["terminate_source"].(bool); ok && shouldTerminate {
		if err := a.terminateInstance(ctx, migration.InstanceID); err != nil {
			log.Printf("Warning: failed to terminate source instance: %v", err)
		}
	}

	// Complete migration
	a.updateMigrationStatus(migration, MigrationStatusCompleted, 100)
	migration.EndTime = time.Now()

	log.Printf("Successfully imported EC2 instance %s as VM %s", migration.InstanceID, vmID)
	return nil
}

// ExportVM exports a NovaCron VM to EC2
func (a *AWSIntegration) ExportVM(ctx context.Context, vmID string, options map[string]interface{}) (*AWSMigration, error) {
	migration := &AWSMigration{
		MigrationID: fmt.Sprintf("export-%s-%d", vmID, time.Now().Unix()),
		Direction:   MigrationDirectionExport,
		VMID:        vmID,
		Status:      MigrationStatusPending,
		StartTime:   time.Now(),
		Metadata:    options,
	}

	a.mutex.Lock()
	a.migrations[migration.MigrationID] = migration
	a.mutex.Unlock()

	// Execute migration asynchronously
	go func() {
		if err := a.executeExportMigration(ctx, migration); err != nil {
			a.mutex.Lock()
			migration.Status = MigrationStatusFailed
			migration.Error = err.Error()
			migration.EndTime = time.Now()
			a.mutex.Unlock()
			log.Printf("Failed to export VM %s to EC2: %v", vmID, err)
		}
	}()

	return migration, nil
}

// executeExportMigration performs the actual export migration
func (a *AWSIntegration) executeExportMigration(ctx context.Context, migration *AWSMigration) error {
	// Update status
	a.updateMigrationStatus(migration, MigrationStatusPreparing, 10)

	// 1. Get VM details and create snapshot
	vmSnapshotPath, vmMetadata, err := a.createVMSnapshot(ctx, migration.VMID)
	if err != nil {
		return fmt.Errorf("failed to create VM snapshot: %w", err)
	}

	// 2. Upload snapshot to S3
	a.updateMigrationStatus(migration, MigrationStatusTransferring, 30)
	s3Key, err := a.uploadToS3(ctx, vmSnapshotPath, migration.MigrationID)
	if err != nil {
		return fmt.Errorf("failed to upload to S3: %w", err)
	}

	// 3. Import snapshot as EBS volume
	a.updateMigrationStatus(migration, MigrationStatusTransferring, 60)
	volumeID, err := a.importSnapshotAsVolume(ctx, s3Key)
	if err != nil {
		return fmt.Errorf("failed to import snapshot as EBS volume: %w", err)
	}

	// 4. Launch EC2 instance with the volume
	a.updateMigrationStatus(migration, MigrationStatusFinalizing, 80)
	instanceID, err := a.launchEC2Instance(ctx, volumeID, vmMetadata)
	if err != nil {
		return fmt.Errorf("failed to launch EC2 instance: %w", err)
	}

	migration.InstanceID = instanceID

	// 5. Optionally stop/delete source VM
	if shouldDelete, ok := migration.Metadata["delete_source"].(bool); ok && shouldDelete {
		if err := a.deleteSourceVM(ctx, migration.VMID); err != nil {
			log.Printf("Warning: failed to delete source VM: %v", err)
		}
	}

	// Complete migration
	a.updateMigrationStatus(migration, MigrationStatusCompleted, 100)
	migration.EndTime = time.Now()

	log.Printf("Successfully exported VM %s as EC2 instance %s", migration.VMID, instanceID)
	return nil
}

// GetMigrationStatus returns the status of a migration
func (a *AWSIntegration) GetMigrationStatus(migrationID string) (*AWSMigration, error) {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	migration, exists := a.migrations[migrationID]
	if !exists {
		return nil, fmt.Errorf("migration %s not found", migrationID)
	}

	// Return copy
	migrationCopy := *migration
	return &migrationCopy, nil
}

// GetCloudWatchMetrics retrieves CloudWatch metrics for an instance
func (a *AWSIntegration) GetCloudWatchMetrics(ctx context.Context, instanceID string, metricName string, period int32, startTime, endTime time.Time) ([]types.Datapoint, error) {
	input := &cloudwatch.GetMetricStatisticsInput{
		Namespace:  aws.String("AWS/EC2"),
		MetricName: aws.String(metricName),
		Dimensions: []types.Dimension{
			{
				Name:  aws.String("InstanceId"),
				Value: aws.String(instanceID),
			},
		},
		StartTime:  aws.Time(startTime),
		EndTime:    aws.Time(endTime),
		Period:     aws.Int32(period),
		Statistics: []types.Statistic{types.StatisticAverage, types.StatisticMaximum},
	}

	result, err := a.cwClient.GetMetricStatistics(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to get CloudWatch metrics: %w", err)
	}

	return result.Datapoints, nil
}

// ListS3Objects lists objects in the configured S3 bucket
func (a *AWSIntegration) ListS3Objects(ctx context.Context, prefix string) ([]s3types.Object, error) {
	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(a.config.S3Bucket),
		Prefix: aws.String(prefix),
	}

	result, err := a.s3Client.ListObjectsV2(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to list S3 objects: %w", err)
	}

	return result.Contents, nil
}

// Helper methods

func (a *AWSIntegration) updateMigrationStatus(migration *AWSMigration, status MigrationStatus, progress float64) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	migration.Status = status
	migration.Progress = progress
}

func (a *AWSIntegration) getInstanceDetails(ctx context.Context, instanceID string) (*AWSInstance, error) {
	// Check cache first
	a.mutex.RLock()
	if instance, ok := a.instances[instanceID]; ok {
		a.mutex.RUnlock()
		return instance, nil
	}
	a.mutex.RUnlock()

	// Query AWS
	input := &ec2.DescribeInstancesInput{
		InstanceIds: []string{instanceID},
	}

	result, err := a.ec2Client.DescribeInstances(ctx, input)
	if err != nil {
		return nil, err
	}

	if len(result.Reservations) == 0 || len(result.Reservations[0].Instances) == 0 {
		return nil, fmt.Errorf("instance %s not found", instanceID)
	}

	instance := a.convertEC2Instance(result.Reservations[0].Instances[0])

	// Update cache
	a.mutex.Lock()
	a.instances[instanceID] = instance
	a.mutex.Unlock()

	return instance, nil
}

func (a *AWSIntegration) createVolumeSnapshots(ctx context.Context, volumeIDs []string) ([]string, error) {
	snapshotIDs := make([]string, 0, len(volumeIDs))

	for _, volumeID := range volumeIDs {
		input := &ec2.CreateSnapshotInput{
			VolumeId:    aws.String(volumeID),
			Description: aws.String(fmt.Sprintf("NovaCron import snapshot - %s", time.Now().Format(time.RFC3339))),
			TagSpecifications: []ec2types.TagSpecification{
				{
					ResourceType: ec2types.ResourceTypeSnapshot,
					Tags: []ec2types.Tag{
						{Key: aws.String("ManagedBy"), Value: aws.String("NovaCron")},
						{Key: aws.String("Purpose"), Value: aws.String("Import")},
					},
				},
			},
		}

		result, err := a.ec2Client.CreateSnapshot(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("failed to create snapshot for volume %s: %w", volumeID, err)
		}

		snapshotIDs = append(snapshotIDs, *result.SnapshotId)

		// Wait for snapshot to complete (simplified - production would use waiter)
		log.Printf("Created snapshot %s for volume %s", *result.SnapshotId, volumeID)
	}

	return snapshotIDs, nil
}

func (a *AWSIntegration) exportSnapshotsToS3(ctx context.Context, snapshotIDs []string) ([]string, error) {
	// Placeholder - in production, use AWS snapshot export APIs or custom transfer
	s3Keys := make([]string, len(snapshotIDs))
	for i, snapshotID := range snapshotIDs {
		s3Keys[i] = fmt.Sprintf("snapshots/%s.raw", snapshotID)
	}
	return s3Keys, nil
}

func (a *AWSIntegration) downloadFromS3(ctx context.Context, s3Keys []string) ([]string, error) {
	localPaths := make([]string, len(s3Keys))
	for i, key := range s3Keys {
		localPaths[i] = fmt.Sprintf("/tmp/novacron-import/%s", key)
	}
	return localPaths, nil
}

func (a *AWSIntegration) createNovaCronVMFromImport(ctx context.Context, instance *AWSInstance, localPaths []string) (string, error) {
	// Placeholder - integrate with NovaCron VM creation API
	vmID := fmt.Sprintf("vm-imported-%s", instance.InstanceID)
	return vmID, nil
}

func (a *AWSIntegration) terminateInstance(ctx context.Context, instanceID string) error {
	input := &ec2.TerminateInstancesInput{
		InstanceIds: []string{instanceID},
	}
	_, err := a.ec2Client.TerminateInstances(ctx, input)
	return err
}

func (a *AWSIntegration) createVMSnapshot(ctx context.Context, vmID string) (string, map[string]interface{}, error) {
	// Placeholder - integrate with NovaCron snapshot API
	snapshotPath := fmt.Sprintf("/var/lib/novacron/snapshots/%s.qcow2", vmID)
	metadata := map[string]interface{}{
		"vm_id": vmID,
		"timestamp": time.Now(),
	}
	return snapshotPath, metadata, nil
}

func (a *AWSIntegration) uploadToS3(ctx context.Context, filePath, migrationID string) (string, error) {
	// Placeholder - implement S3 multipart upload for large files
	s3Key := fmt.Sprintf("exports/%s/%s", migrationID, filepath.Base(filePath))
	return s3Key, nil
}

func (a *AWSIntegration) importSnapshotAsVolume(ctx context.Context, s3Key string) (string, error) {
	// Placeholder - use AWS snapshot import APIs
	volumeID := fmt.Sprintf("vol-%d", time.Now().Unix())
	return volumeID, nil
}

func (a *AWSIntegration) launchEC2Instance(ctx context.Context, volumeID string, vmMetadata map[string]interface{}) (string, error) {
	// Determine instance type from metadata or use default
	instanceType := ec2types.InstanceTypeT3Medium
	if it, ok := vmMetadata["instance_type"].(string); ok {
		instanceType = ec2types.InstanceType(it)
	}

	input := &ec2.RunInstancesInput{
		ImageId:      aws.String("ami-placeholder"), // Would be determined from volume/snapshot
		InstanceType: instanceType,
		MinCount:     aws.Int32(1),
		MaxCount:     aws.Int32(1),
		SubnetId:     aws.String(a.config.DefaultSubnet),
		SecurityGroupIds: a.config.SecurityGroups,
		TagSpecifications: []ec2types.TagSpecification{
			{
				ResourceType: ec2types.ResourceTypeInstance,
				Tags: []ec2types.Tag{
					{Key: aws.String("ManagedBy"), Value: aws.String("NovaCron")},
					{Key: aws.String("NovaCronVMID"), Value: aws.String(vmMetadata["vm_id"].(string))},
				},
			},
		},
	}

	result, err := a.ec2Client.RunInstances(ctx, input)
	if err != nil {
		return "", fmt.Errorf("failed to launch EC2 instance: %w", err)
	}

	if len(result.Instances) == 0 {
		return "", fmt.Errorf("no instances launched")
	}

	return *result.Instances[0].InstanceId, nil
}

func (a *AWSIntegration) deleteSourceVM(ctx context.Context, vmID string) error {
	// Placeholder - integrate with NovaCron VM deletion API
	log.Printf("Deleting source VM %s", vmID)
	return nil
}

// Shutdown gracefully shuts down the AWS integration
func (a *AWSIntegration) Shutdown(ctx context.Context) error {
	log.Println("Shutting down AWS integration")
	a.cancel()
	return nil
}

// CalculateCost calculates estimated AWS costs for an instance
func (a *AWSIntegration) CalculateCost(ctx context.Context, instanceType string, hours float64) (float64, error) {
	// Simplified cost calculation - in production, use AWS Price List API
	costPerHour := map[string]float64{
		"t3.micro":   0.0104,
		"t3.small":   0.0208,
		"t3.medium":  0.0416,
		"t3.large":   0.0832,
		"m5.large":   0.096,
		"m5.xlarge":  0.192,
		"c5.large":   0.085,
		"c5.xlarge":  0.170,
	}

	rate, ok := costPerHour[instanceType]
	if !ok {
		return 0, fmt.Errorf("unknown instance type: %s", instanceType)
	}

	return rate * hours, nil
}

// Helper to get filepath.Base functionality
func filepath.Base(path string) string {
	// Simple implementation
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			return path[i+1:]
		}
	}
	return path
}
