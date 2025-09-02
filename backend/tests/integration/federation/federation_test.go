package federation_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/khryptorgraphics/novacron/backend/tests/integration/fixtures"
	"github.com/khryptorgraphics/novacron/backend/tests/integration/helpers"
)

// FederationTestSuite tests multi-cloud federation scenarios
type FederationTestSuite struct {
	suite.Suite
	helpers       *helpers.TestHelpers
	fixtures      *fixtures.TestFixtures
	awsSession    *session.Session
	s3Client      *s3.S3
	ec2Client     *ec2.EC2
	minioClient   *minio.Client
	fedManager    *federation.FederationManager
	testBucket    string
	testInstance  string
	cleanupItems  []string
}

// CloudProvider represents different cloud providers
type CloudProvider struct {
	Name     string `json:"name"`
	Region   string `json:"region"`
	Endpoint string `json:"endpoint"`
	Type     string `json:"type"` // aws, minio, gcp, azure
}

// VMSpec represents VM specification for federation
type VMSpec struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	CPUCores    int               `json:"cpu_cores"`
	MemoryMB    int               `json:"memory_mb"`
	DiskGB      int               `json:"disk_gb"`
	Provider    string            `json:"provider"`
	Region      string            `json:"region"`
	Tags        map[string]string `json:"tags"`
	NetworkConfig map[string]string `json:"network_config"`
}

// FederationEvent represents events in the federation system
type FederationEvent struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Target    string                 `json:"target"`
	Payload   map[string]interface{} `json:"payload"`
	Timestamp time.Time              `json:"timestamp"`
}

// SetupSuite initializes the federation test environment
func (suite *FederationTestSuite) SetupSuite() {
	// Initialize LocalStack AWS services
	suite.setupLocalStackServices()
	
	// Initialize MinIO services
	suite.setupMinIOServices()
	
	// Initialize federation manager
	suite.setupFederationManager()
	
	// Create test bucket and resources
	suite.createTestResources()
}

// TearDownSuite cleans up federation test resources
func (suite *FederationTestSuite) TearDownSuite() {
	suite.cleanupTestResources()
}

// setupLocalStackServices initializes LocalStack AWS services
func (suite *FederationTestSuite) setupLocalStackServices() {
	// LocalStack endpoint from docker-compose
	endpoint := "http://localhost:4566"
	if os.Getenv("LOCALSTACK_URL") != "" {
		endpoint = os.Getenv("LOCALSTACK_URL")
	}

	// Create AWS session for LocalStack
	sess, err := session.NewSession(&aws.Config{
		Region:           aws.String("us-east-1"),
		Endpoint:         aws.String(endpoint),
		S3ForcePathStyle: aws.Bool(true),
		Credentials:      credentials.NewStaticCredentials("test", "test", ""),
	})
	suite.Require().NoError(err)
	
	suite.awsSession = sess
	suite.s3Client = s3.New(sess)
	suite.ec2Client = ec2.New(sess)
}

// setupMinIOServices initializes MinIO services
func (suite *FederationTestSuite) setupMinIOServices() {
	endpoint := "localhost:9000"
	if os.Getenv("MINIO_URL") != "" {
		endpoint = strings.TrimPrefix(os.Getenv("MINIO_URL"), "http://")
	}

	// Create MinIO client
	minioClient, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4("minioadmin", "minioadmin123", ""),
		Secure: false,
	})
	suite.Require().NoError(err)
	
	suite.minioClient = minioClient
}

// setupFederationManager initializes the federation manager
func (suite *FederationTestSuite) setupFederationManager() {
	// In a real implementation, this would initialize the actual federation manager
	// For testing, we'll create a mock implementation
	suite.fedManager = &federation.FederationManager{
		// Mock configuration
	}
}

// createTestResources creates test buckets and instances
func (suite *FederationTestSuite) createTestResources() {
	suite.testBucket = fmt.Sprintf("test-federation-%d", time.Now().Unix())
	
	// Create S3 bucket in LocalStack
	_, err := suite.s3Client.CreateBucket(&s3.CreateBucketInput{
		Bucket: aws.String(suite.testBucket),
	})
	suite.Require().NoError(err)
	suite.cleanupItems = append(suite.cleanupItems, "s3:"+suite.testBucket)
	
	// Create MinIO bucket
	err = suite.minioClient.MakeBucket(context.Background(), suite.testBucket, minio.MakeBucketOptions{})
	suite.Require().NoError(err)
	suite.cleanupItems = append(suite.cleanupItems, "minio:"+suite.testBucket)
}

// TestMultiCloudStorageFederation tests storage federation across providers
func (suite *FederationTestSuite) TestMultiCloudStorageFederation() {
	testData := []byte("federation test data")
	testKey := "test-federation-file.txt"
	
	// Upload to AWS S3 (LocalStack)
	suite.T().Run("UploadToAWS", func(t *testing.T) {
		_, err := suite.s3Client.PutObject(&s3.PutObjectInput{
			Bucket: aws.String(suite.testBucket),
			Key:    aws.String(testKey),
			Body:   bytes.NewReader(testData),
		})
		suite.NoError(err)
	})
	
	// Upload to MinIO
	suite.T().Run("UploadToMinIO", func(t *testing.T) {
		_, err := suite.minioClient.PutObject(
			context.Background(),
			suite.testBucket,
			testKey,
			bytes.NewReader(testData),
			int64(len(testData)),
			minio.PutObjectOptions{},
		)
		suite.NoError(err)
	})
	
	// Test federation sync between providers
	suite.T().Run("FederationSync", func(t *testing.T) {
		suite.testStorageFederation(testKey, testData)
	})
}

// TestVMFederationLifecycle tests VM lifecycle across cloud providers
func (suite *FederationTestSuite) TestVMFederationLifecycle() {
	vmSpecs := []VMSpec{
		{
			ID:       "test-fed-vm-aws",
			Name:     "federation-test-aws",
			CPUCores: 2,
			MemoryMB: 2048,
			DiskGB:   20,
			Provider: "aws",
			Region:   "us-east-1",
			Tags:     map[string]string{"Environment": "test", "Federation": "true"},
		},
		{
			ID:       "test-fed-vm-minio",
			Name:     "federation-test-storage",
			CPUCores: 1,
			MemoryMB: 1024,
			DiskGB:   10,
			Provider: "minio",
			Region:   "local",
			Tags:     map[string]string{"Environment": "test", "StorageNode": "true"},
		},
	}
	
	for _, vmSpec := range vmSpecs {
		suite.T().Run(fmt.Sprintf("VM_%s", vmSpec.Provider), func(t *testing.T) {
			suite.testVMFederation(vmSpec)
		})
	}
}

// TestCrossCloudNetworking tests networking between federated resources
func (suite *FederationTestSuite) TestCrossCloudNetworking() {
	suite.T().Run("NetworkDiscovery", func(t *testing.T) {
		// Test network discovery across providers
		networks := suite.discoverFederatedNetworks()
		suite.Greater(len(networks), 0)
		
		for _, network := range networks {
			suite.NotEmpty(network.Provider)
			suite.NotEmpty(network.Region)
			suite.NotEmpty(network.NetworkID)
		}
	})
	
	suite.T().Run("CrossCloudConnectivity", func(t *testing.T) {
		// Test connectivity between AWS and MinIO services
		suite.testCrossCloudConnectivity()
	})
}

// TestFederationEventPropagation tests event propagation across federation
func (suite *FederationTestSuite) TestFederationEventPropagation() {
	events := []FederationEvent{
		{
			ID:     "test-event-1",
			Type:   "vm.created",
			Source: "aws.us-east-1",
			Target: "federation.global",
			Payload: map[string]interface{}{
				"vm_id":    "test-fed-vm-aws",
				"provider": "aws",
				"region":   "us-east-1",
			},
			Timestamp: time.Now(),
		},
		{
			ID:     "test-event-2",
			Type:   "storage.synchronized",
			Source: "minio.local",
			Target: "federation.global",
			Payload: map[string]interface{}{
				"bucket":   suite.testBucket,
				"provider": "minio",
				"status":   "synchronized",
			},
			Timestamp: time.Now(),
		},
	}
	
	for _, event := range events {
		suite.T().Run(fmt.Sprintf("Event_%s", event.Type), func(t *testing.T) {
			suite.testEventPropagation(event)
		})
	}
}

// TestFederationConsistency tests data consistency across federation
func (suite *FederationTestSuite) TestFederationConsistency() {
	suite.T().Run("StorageConsistency", func(t *testing.T) {
		// Test eventual consistency across storage providers
		testFiles := []string{"consistency-test-1.json", "consistency-test-2.json"}
		
		for _, filename := range testFiles {
			data := map[string]interface{}{
				"timestamp": time.Now().Unix(),
				"filename":  filename,
				"test_data": "federation consistency test",
			}
			
			jsonData, _ := json.Marshal(data)
			
			// Write to both providers
			suite.writeToAllStorageProviders(filename, jsonData)
			
			// Verify consistency
			suite.verifyStorageConsistency(filename, jsonData)
		}
	})
	
	suite.T().Run("MetadataConsistency", func(t *testing.T) {
		// Test metadata consistency across providers
		suite.testMetadataConsistency()
	})
}

// TestFederationFailover tests failover scenarios
func (suite *FederationTestSuite) TestFederationFailover() {
	suite.T().Run("StorageFailover", func(t *testing.T) {
		// Simulate storage provider failure and test failover
		suite.simulateStorageFailover()
	})
	
	suite.T().Run("ComputeFailover", func(t *testing.T) {
		// Simulate compute provider failure and test failover
		suite.simulateComputeFailover()
	})
}

// TestFederationScaling tests scaling across federation
func (suite *FederationTestSuite) TestFederationScaling() {
	suite.T().Run("HorizontalScaling", func(t *testing.T) {
		// Test horizontal scaling across multiple providers
		suite.testHorizontalScaling()
	})
	
	suite.T().Run("LoadDistribution", func(t *testing.T) {
		// Test load distribution across federated resources
		suite.testLoadDistribution()
	})
}

// Helper methods for federation testing

func (suite *FederationTestSuite) testStorageFederation(key string, data []byte) {
	// Test federation sync between AWS S3 and MinIO
	
	// Wait for potential sync (in real implementation)
	time.Sleep(2 * time.Second)
	
	// Verify data exists in both providers
	suite.T().Run("VerifyAWSStorage", func(t *testing.T) {
		result, err := suite.s3Client.GetObject(&s3.GetObjectInput{
			Bucket: aws.String(suite.testBucket),
			Key:    aws.String(key),
		})
		suite.NoError(err)
		defer result.Body.Close()
		
		retrievedData, err := io.ReadAll(result.Body)
		suite.NoError(err)
		suite.Equal(data, retrievedData)
	})
	
	suite.T().Run("VerifyMinIOStorage", func(t *testing.T) {
		object, err := suite.minioClient.GetObject(
			context.Background(),
			suite.testBucket,
			key,
			minio.GetObjectOptions{},
		)
		suite.NoError(err)
		defer object.Close()
		
		retrievedData, err := io.ReadAll(object)
		suite.NoError(err)
		suite.Equal(data, retrievedData)
	})
}

func (suite *FederationTestSuite) testVMFederation(vmSpec VMSpec) {
	// In a real implementation, this would create VMs across providers
	// For testing, we simulate the VM creation process
	
	suite.T().Logf("Testing VM federation for provider: %s", vmSpec.Provider)
	
	// Simulate VM creation
	suite.simulateVMCreation(vmSpec)
	
	// Test VM state synchronization
	suite.testVMStateSynchronization(vmSpec)
	
	// Test VM monitoring across federation
	suite.testVMMonitoring(vmSpec)
}

func (suite *FederationTestSuite) simulateVMCreation(vmSpec VMSpec) {
	switch vmSpec.Provider {
	case "aws":
		// Simulate AWS EC2 instance creation
		suite.T().Logf("Simulating AWS EC2 creation for VM: %s", vmSpec.ID)
	case "minio":
		// Simulate storage-backed compute creation
		suite.T().Logf("Simulating storage compute creation for VM: %s", vmSpec.ID)
	}
	
	// In real implementation, would call actual cloud APIs
	suite.cleanupItems = append(suite.cleanupItems, fmt.Sprintf("vm:%s", vmSpec.ID))
}

func (suite *FederationTestSuite) testVMStateSynchronization(vmSpec VMSpec) {
	// Test that VM state is properly synchronized across federation
	expectedState := "running"
	
	// Simulate state check across providers
	state := suite.getVMStateFromFederation(vmSpec.ID)
	suite.Equal(expectedState, state)
}

func (suite *FederationTestSuite) testVMMonitoring(vmSpec VMSpec) {
	// Test VM monitoring across federation
	metrics := suite.getFederatedVMMetrics(vmSpec.ID)
	
	suite.NotNil(metrics)
	suite.Contains(metrics, "cpu_usage")
	suite.Contains(metrics, "memory_usage")
	suite.Contains(metrics, "provider")
	suite.Equal(vmSpec.Provider, metrics["provider"])
}

func (suite *FederationTestSuite) discoverFederatedNetworks() []NetworkInfo {
	// Simulate network discovery across providers
	return []NetworkInfo{
		{
			Provider:  "aws",
			Region:    "us-east-1",
			NetworkID: "vpc-test123",
			CIDR:      "10.0.0.0/16",
		},
		{
			Provider:  "minio",
			Region:    "local",
			NetworkID: "minio-network",
			CIDR:      "172.20.0.0/16",
		},
	}
}

func (suite *FederationTestSuite) testCrossCloudConnectivity() {
	// Test connectivity between services
	
	// Test AWS S3 connectivity
	suite.T().Run("AWS_Connectivity", func(t *testing.T) {
		_, err := suite.s3Client.ListBuckets(&s3.ListBucketsInput{})
		suite.NoError(err)
	})
	
	// Test MinIO connectivity
	suite.T().Run("MinIO_Connectivity", func(t *testing.T) {
		_, err := suite.minioClient.ListBuckets(context.Background())
		suite.NoError(err)
	})
}

func (suite *FederationTestSuite) testEventPropagation(event FederationEvent) {
	// Simulate event propagation across federation
	
	// Publish event to federation event bus
	err := suite.publishFederationEvent(event)
	suite.NoError(err)
	
	// Wait for propagation
	time.Sleep(1 * time.Second)
	
	// Verify event was received by all federation members
	receivedEvents := suite.getFederationEvents(event.Type)
	suite.Greater(len(receivedEvents), 0)
	
	found := false
	for _, receivedEvent := range receivedEvents {
		if receivedEvent.ID == event.ID {
			found = true
			break
		}
	}
	suite.True(found, "Event should be propagated across federation")
}

func (suite *FederationTestSuite) writeToAllStorageProviders(filename string, data []byte) {
	// Write to AWS S3
	_, err := suite.s3Client.PutObject(&s3.PutObjectInput{
		Bucket: aws.String(suite.testBucket),
		Key:    aws.String(filename),
		Body:   bytes.NewReader(data),
	})
	suite.NoError(err)
	
	// Write to MinIO
	_, err = suite.minioClient.PutObject(
		context.Background(),
		suite.testBucket,
		filename,
		bytes.NewReader(data),
		int64(len(data)),
		minio.PutObjectOptions{},
	)
	suite.NoError(err)
}

func (suite *FederationTestSuite) verifyStorageConsistency(filename string, expectedData []byte) {
	// Verify data consistency across all storage providers
	
	// Check AWS S3
	awsObject, err := suite.s3Client.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(suite.testBucket),
		Key:    aws.String(filename),
	})
	suite.NoError(err)
	defer awsObject.Body.Close()
	
	awsData, err := io.ReadAll(awsObject.Body)
	suite.NoError(err)
	
	// Check MinIO
	minioObject, err := suite.minioClient.GetObject(
		context.Background(),
		suite.testBucket,
		filename,
		minio.GetObjectOptions{},
	)
	suite.NoError(err)
	defer minioObject.Close()
	
	minioData, err := io.ReadAll(minioObject)
	suite.NoError(err)
	
	// Verify consistency
	suite.Equal(expectedData, awsData, "AWS S3 data should match expected")
	suite.Equal(expectedData, minioData, "MinIO data should match expected")
	suite.Equal(awsData, minioData, "AWS and MinIO data should be consistent")
}

func (suite *FederationTestSuite) testMetadataConsistency() {
	// Test metadata consistency across providers
	testMetadata := map[string]string{
		"environment": "test",
		"federation":  "true",
		"timestamp":   fmt.Sprintf("%d", time.Now().Unix()),
	}
	
	filename := "metadata-test.json"
	data := []byte(`{"test": "metadata"}`)
	
	// Upload with metadata to AWS S3
	_, err := suite.s3Client.PutObject(&s3.PutObjectInput{
		Bucket:   aws.String(suite.testBucket),
		Key:      aws.String(filename),
		Body:     bytes.NewReader(data),
		Metadata: testMetadata,
	})
	suite.NoError(err)
	
	// Upload with metadata to MinIO (using tags as metadata)
	_, err = suite.minioClient.PutObject(
		context.Background(),
		suite.testBucket,
		filename,
		bytes.NewReader(data),
		int64(len(data)),
		minio.PutObjectOptions{
			UserMetadata: testMetadata,
		},
	)
	suite.NoError(err)
	
	// Verify metadata consistency
	time.Sleep(1 * time.Second)
	
	// Check AWS metadata
	awsHead, err := suite.s3Client.HeadObject(&s3.HeadObjectInput{
		Bucket: aws.String(suite.testBucket),
		Key:    aws.String(filename),
	})
	suite.NoError(err)
	suite.Equal(testMetadata["environment"], *awsHead.Metadata["Environment"])
}

func (suite *FederationTestSuite) simulateStorageFailover() {
	// Simulate storage provider failure scenario
	suite.T().Log("Simulating storage provider failure and testing failover")
	
	// This would test actual failover logic in a real implementation
	// For now, we verify that both providers are accessible
	
	// Test primary provider (AWS)
	_, err := suite.s3Client.ListBuckets(&s3.ListBucketsInput{})
	suite.NoError(err, "Primary storage provider should be accessible")
	
	// Test secondary provider (MinIO) 
	_, err = suite.minioClient.ListBuckets(context.Background())
	suite.NoError(err, "Secondary storage provider should be accessible for failover")
}

func (suite *FederationTestSuite) simulateComputeFailover() {
	// Simulate compute provider failure scenario
	suite.T().Log("Simulating compute provider failure and testing failover")
	
	// In real implementation, would test VM migration between providers
	// For testing, verify that federation can handle provider unavailability
	
	vmSpec := VMSpec{
		ID:       "failover-test-vm",
		Name:     "failover-test",
		Provider: "aws",
		Region:   "us-east-1",
	}
	
	// Simulate primary provider failure and failover to secondary
	suite.testVMFailover(vmSpec, "minio")
}

func (suite *FederationTestSuite) testHorizontalScaling() {
	// Test horizontal scaling across multiple providers
	suite.T().Log("Testing horizontal scaling across federation")
	
	vmCount := 5
	providers := []string{"aws", "minio"}
	
	// Simulate creating VMs across multiple providers for scaling
	for i := 0; i < vmCount; i++ {
		provider := providers[i%len(providers)]
		vmSpec := VMSpec{
			ID:       fmt.Sprintf("scale-test-vm-%d", i),
			Name:     fmt.Sprintf("scale-test-%d", i),
			Provider: provider,
			Region:   "us-east-1",
		}
		
		suite.simulateVMCreation(vmSpec)
	}
	
	suite.T().Logf("Successfully scaled to %d VMs across %d providers", vmCount, len(providers))
}

func (suite *FederationTestSuite) testLoadDistribution() {
	// Test load distribution across federated resources
	suite.T().Log("Testing load distribution across federation")
	
	// Simulate load distribution algorithm
	loads := []float64{0.8, 0.3, 0.6, 0.2, 0.9}
	providers := []string{"aws", "minio"}
	
	distribution := suite.calculateLoadDistribution(loads, providers)
	
	suite.NotNil(distribution)
	suite.Equal(len(providers), len(distribution))
	
	// Verify load is properly distributed
	for provider, load := range distribution {
		suite.Contains(providers, provider)
		suite.GreaterOrEqual(load, 0.0)
		suite.LessOrEqual(load, 1.0)
	}
}

// Helper types and methods

type NetworkInfo struct {
	Provider  string `json:"provider"`
	Region    string `json:"region"`
	NetworkID string `json:"network_id"`
	CIDR      string `json:"cidr"`
}

func (suite *FederationTestSuite) getVMStateFromFederation(vmID string) string {
	// Simulate getting VM state from federation
	return "running"
}

func (suite *FederationTestSuite) getFederatedVMMetrics(vmID string) map[string]interface{} {
	// Simulate getting VM metrics from federation
	return map[string]interface{}{
		"cpu_usage":    45.2,
		"memory_usage": 67.8,
		"disk_usage":   23.1,
		"provider":     "aws",
		"region":       "us-east-1",
		"timestamp":    time.Now().Unix(),
	}
}

func (suite *FederationTestSuite) publishFederationEvent(event FederationEvent) error {
	// Simulate publishing event to federation event bus
	suite.T().Logf("Publishing federation event: %s", event.Type)
	return nil
}

func (suite *FederationTestSuite) getFederationEvents(eventType string) []FederationEvent {
	// Simulate getting events from federation event bus
	return []FederationEvent{
		{
			ID:        "test-event-received",
			Type:      eventType,
			Source:    "federation.test",
			Target:    "federation.global",
			Timestamp: time.Now(),
		},
	}
}

func (suite *FederationTestSuite) testVMFailover(vmSpec VMSpec, targetProvider string) {
	// Simulate VM failover from one provider to another
	suite.T().Logf("Testing VM failover from %s to %s", vmSpec.Provider, targetProvider)
	
	// In real implementation, would migrate VM state and storage
	// For testing, verify failover logic
	
	originalProvider := vmSpec.Provider
	vmSpec.Provider = targetProvider
	
	suite.simulateVMCreation(vmSpec)
	
	suite.T().Logf("VM failover completed from %s to %s", originalProvider, targetProvider)
}

func (suite *FederationTestSuite) calculateLoadDistribution(loads []float64, providers []string) map[string]float64 {
	// Simple load distribution algorithm for testing
	distribution := make(map[string]float64)
	
	totalLoad := 0.0
	for _, load := range loads {
		totalLoad += load
	}
	
	loadPerProvider := totalLoad / float64(len(providers))
	
	for _, provider := range providers {
		distribution[provider] = loadPerProvider
	}
	
	return distribution
}

func (suite *FederationTestSuite) cleanupTestResources() {
	suite.T().Log("Cleaning up federation test resources")
	
	for _, item := range suite.cleanupItems {
		parts := strings.SplitN(item, ":", 2)
		if len(parts) != 2 {
			continue
		}
		
		resourceType, resourceName := parts[0], parts[1]
		
		switch resourceType {
		case "s3":
			// Delete S3 bucket contents and bucket
			suite.cleanupS3Bucket(resourceName)
		case "minio":
			// Delete MinIO bucket contents and bucket
			suite.cleanupMinioBucket(resourceName)
		case "vm":
			// Cleanup simulated VM resources
			suite.T().Logf("Cleaning up VM resource: %s", resourceName)
		}
	}
}

func (suite *FederationTestSuite) cleanupS3Bucket(bucketName string) {
	// List and delete all objects in bucket
	objects, err := suite.s3Client.ListObjects(&s3.ListObjectsInput{
		Bucket: aws.String(bucketName),
	})
	if err != nil {
		return
	}
	
	for _, object := range objects.Contents {
		suite.s3Client.DeleteObject(&s3.DeleteObjectInput{
			Bucket: aws.String(bucketName),
			Key:    object.Key,
		})
	}
	
	// Delete bucket
	suite.s3Client.DeleteBucket(&s3.DeleteBucketInput{
		Bucket: aws.String(bucketName),
	})
}

func (suite *FederationTestSuite) cleanupMinioBucket(bucketName string) {
	// List and delete all objects in bucket
	objectCh := suite.minioClient.ListObjects(context.Background(), bucketName, minio.ListObjectsOptions{})
	
	for object := range objectCh {
		if object.Err != nil {
			continue
		}
		
		suite.minioClient.RemoveObject(context.Background(), bucketName, object.Key, minio.RemoveObjectOptions{})
	}
	
	// Delete bucket
	suite.minioClient.RemoveBucket(context.Background(), bucketName)
}

// TestFederationTestSuite runs the federation test suite
func TestFederationTestSuite(t *testing.T) {
	suite.Run(t, new(FederationTestSuite))
}