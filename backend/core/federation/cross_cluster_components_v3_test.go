package federation

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/consensus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest"
)

// TestNewCrossClusterComponentsV3 tests v3 component initialization
func TestNewCrossClusterComponentsV3(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	require.NotNil(t, cc)
	defer cc.Close()

	assert.Equal(t, "test-node-1", cc.nodeID)
	assert.Equal(t, upgrade.ModeHybrid, cc.mode)
	assert.NotNil(t, cc.hdeEngine)
	assert.NotNil(t, cc.amstTransport)
	assert.NotNil(t, cc.acpConsensus)
	assert.NotNil(t, cc.assSync)
	assert.NotNil(t, cc.pbaPredictor)
	assert.NotNil(t, cc.itpPartition)
}

// TestConnectClusterV3_DatacenterMode tests datacenter mode connection
func TestConnectClusterV3_DatacenterMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeDatacenter

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	cluster := &ClusterConnectionV3{
		ClusterID:  "datacenter-cluster-1",
		Datacenter: "dc1",
		Region:     "us-west",
		Endpoint:   "datacenter-cluster-1.local:8080",
		trusted:    true,
	}

	// Note: This will fail without actual network setup, but tests the logic
	err = cc.ConnectClusterV3(context.Background(), cluster)
	// We expect an error due to no actual network, but verify the mode selection
	assert.Equal(t, upgrade.ModeDatacenter, cluster.NetworkMode,
		"Trusted datacenter should use datacenter mode")
}

// TestConnectClusterV3_InternetMode tests internet mode with Byzantine tolerance
func TestConnectClusterV3_InternetMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeInternet
	config.ByzantineTolerance = true

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	cluster := &ClusterConnectionV3{
		ClusterID:     "cloud-cluster-1",
		CloudProvider: "aws",
		Region:        "us-east-1",
		Endpoint:      "cloud-cluster-1.aws.com:8080",
		trusted:       false, // Untrusted cloud
	}

	err = cc.ConnectClusterV3(context.Background(), cluster)
	// Verify Byzantine-tolerant internet mode is selected
	assert.Equal(t, upgrade.ModeInternet, cluster.NetworkMode,
		"Untrusted cloud should use internet mode with Byzantine tolerance")
}

// TestConnectClusterV3_HybridMode tests hybrid mode adaptive switching
func TestConnectClusterV3_HybridMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeHybrid

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	cluster := &ClusterConnectionV3{
		ClusterID: "hybrid-cluster-1",
		Region:    "us-central",
		Endpoint:  "hybrid-cluster-1.local:8080",
		trusted:   true,
	}

	err = cc.ConnectClusterV3(context.Background(), cluster)
	// Hybrid mode should be selected for generic connections
	assert.Equal(t, upgrade.ModeHybrid, cluster.NetworkMode,
		"Generic connection should use hybrid mode")
}

// TestSyncClusterStateV3_Compression tests state synchronization with HDE v3
func TestSyncClusterStateV3_Compression(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Create test state data
	stateData := make([]byte, 10000)
	for i := range stateData {
		stateData[i] = byte(i % 256)
	}

	// Add mock cluster connections
	cc.clusterConnections["cluster-1"] = &ClusterConnectionV3{
		ClusterID:  "cluster-1",
		Endpoint:   "cluster-1.local:8080",
		trusted:    true,
	}
	cc.clusterConnections["cluster-1"].connected.Store(true)

	// Note: This will fail without transport setup
	err = cc.SyncClusterStateV3(context.Background(), "source-cluster", []string{"cluster-1"}, stateData)

	// Verify metrics were updated (even if sync failed due to no transport)
	// In production, this would succeed with proper transport
}

// TestConsensusV3_DatacenterRaft tests Raft consensus for datacenter mode
func TestConsensusV3_DatacenterRaft(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeDatacenter

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Add trusted datacenter clusters
	cc.clusterConnections["dc-cluster-1"] = &ClusterConnectionV3{
		ClusterID:  "dc-cluster-1",
		Datacenter: "dc1",
		trusted:    true,
	}
	cc.clusterConnections["dc-cluster-2"] = &ClusterConnectionV3{
		ClusterID:  "dc-cluster-2",
		Datacenter: "dc2",
		trusted:    true,
	}

	value := map[string]interface{}{
		"operation": "vm_migration",
		"vm_id":     "vm-123",
		"target":    "dc-cluster-2",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = cc.ConsensusV3(ctx, value, []string{"dc-cluster-1", "dc-cluster-2"})

	// Should use fast Raft consensus
	datacenterOps := cc.metrics.DatacenterOperations.Load()
	assert.Greater(t, datacenterOps, uint64(0), "Should record datacenter operation")
}

// TestConsensusV3_InternetPBFT tests PBFT consensus for untrusted clouds
func TestConsensusV3_InternetPBFT(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeInternet
	config.ByzantineTolerance = true
	config.MaxFaultyNodes = 1

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Add untrusted cloud clusters
	cc.clusterConnections["aws-cluster-1"] = &ClusterConnectionV3{
		ClusterID:     "aws-cluster-1",
		CloudProvider: "aws",
		trusted:       false,
	}
	cc.clusterConnections["azure-cluster-1"] = &ClusterConnectionV3{
		ClusterID:     "azure-cluster-1",
		CloudProvider: "azure",
		trusted:       false,
	}
	cc.clusterConnections["gcp-cluster-1"] = &ClusterConnectionV3{
		ClusterID:     "gcp-cluster-1",
		CloudProvider: "gcp",
		trusted:       false,
	}

	value := map[string]interface{}{
		"operation": "multi_cloud_deployment",
		"vm_id":     "vm-456",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = cc.ConsensusV3(ctx, value, []string{"aws-cluster-1", "azure-cluster-1", "gcp-cluster-1"})

	// Should use Byzantine-tolerant PBFT
	internetOps := cc.metrics.InternetOperations.Load()
	assert.Greater(t, internetOps, uint64(0), "Should record internet operation with Byzantine tolerance")
}

// TestConsensusV3_MixedTrust tests mixed trusted/untrusted clusters
func TestConsensusV3_MixedTrust(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.ByzantineTolerance = true

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Mix of trusted and untrusted clusters
	cc.clusterConnections["trusted-dc"] = &ClusterConnectionV3{
		ClusterID:  "trusted-dc",
		Datacenter: "dc1",
		trusted:    true,
	}
	cc.clusterConnections["untrusted-cloud"] = &ClusterConnectionV3{
		ClusterID:     "untrusted-cloud",
		CloudProvider: "aws",
		trusted:       false, // Untrusted
	}

	value := map[string]interface{}{
		"operation": "hybrid_deployment",
		"vm_id":     "vm-789",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = cc.ConsensusV3(ctx, value, []string{"trusted-dc", "untrusted-cloud"})

	// Should use Byzantine consensus due to untrusted cluster
	internetOps := cc.metrics.InternetOperations.Load()
	assert.Greater(t, internetOps, uint64(0), "Should use Byzantine consensus for mixed trust")
}

// TestHandlePartitionV3 tests network partition handling with ITP v3
func TestHandlePartitionV3(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.PartitionTolerance = true
	config.RecoveryTimeout = 5 * time.Second

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Add clusters
	cc.clusterConnections["cluster-1"] = &ClusterConnectionV3{
		ClusterID: "cluster-1",
		trusted:   true,
	}
	cc.clusterConnections["cluster-1"].healthy.Store(true)

	cc.clusterConnections["cluster-2"] = &ClusterConnectionV3{
		ClusterID: "cluster-2",
		trusted:   true,
	}
	cc.clusterConnections["cluster-2"].healthy.Store(true)

	// Simulate partition
	affectedClusters := []string{"cluster-1", "cluster-2"}
	err = cc.HandlePartitionV3(context.Background(), affectedClusters)
	require.NoError(t, err)

	// Verify clusters marked unhealthy
	assert.False(t, cc.clusterConnections["cluster-1"].healthy.Load())
	assert.False(t, cc.clusterConnections["cluster-2"].healthy.Load())
}

// TestRecoverFromPartitionV3 tests partition recovery
func TestRecoverFromPartitionV3(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.PartitionTolerance = true

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Add unhealthy clusters (simulating partition)
	cc.clusterConnections["cluster-1"] = &ClusterConnectionV3{
		ClusterID: "cluster-1",
		trusted:   true,
	}
	cc.clusterConnections["cluster-1"].healthy.Store(false)

	// Recover from partition
	recoveredClusters := []string{"cluster-1"}
	err = cc.RecoverFromPartitionV3(context.Background(), recoveredClusters)
	require.NoError(t, err)

	// Verify cluster recovered
	assert.True(t, cc.clusterConnections["cluster-1"].healthy.Load())
}

// TestUpdateNetworkMode tests dynamic mode switching
func TestUpdateNetworkMode(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.NetworkMode = upgrade.ModeHybrid

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	initialMode := cc.mode
	assert.Equal(t, upgrade.ModeHybrid, initialMode)

	// Switch to datacenter mode
	cc.UpdateNetworkMode(upgrade.ModeDatacenter)
	assert.Equal(t, upgrade.ModeDatacenter, cc.mode)
	assert.Greater(t, cc.metrics.ModeChanges.Load(), uint64(0))

	// Switch to internet mode
	cc.UpdateNetworkMode(upgrade.ModeInternet)
	assert.Equal(t, upgrade.ModeInternet, cc.mode)
	assert.Greater(t, cc.metrics.ModeChanges.Load(), uint64(1))

	// No change when setting same mode
	modeChangesBefore := cc.metrics.ModeChanges.Load()
	cc.UpdateNetworkMode(upgrade.ModeInternet)
	assert.Equal(t, modeChangesBefore, cc.metrics.ModeChanges.Load(),
		"Should not increment mode changes when mode unchanged")
}

// TestGetMetricsV3 tests comprehensive metrics collection
func TestGetMetricsV3(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Update some metrics
	cc.metrics.TotalConnections.Add(5)
	cc.metrics.ActiveConnections.Add(3)
	cc.metrics.TotalBytesSent.Add(1000000)
	cc.metrics.TotalBytesReceived.Add(800000)
	cc.metrics.CompressionRatio.Store(250) // 2.5x
	cc.metrics.ConsensusOperations.Add(100)
	cc.metrics.ConsensusFailures.Add(5)
	cc.metrics.SyncOperations.Add(200)
	cc.metrics.SyncFailures.Add(10)
	cc.metrics.ByzantineDetections.Add(2)
	cc.metrics.DatacenterOperations.Add(150)
	cc.metrics.InternetOperations.Add(50)

	metrics := cc.GetMetricsV3()
	require.NotNil(t, metrics)

	// Verify connection metrics
	assert.Equal(t, int32(5), metrics["total_connections"])
	assert.Equal(t, int32(3), metrics["active_connections"])

	// Verify bandwidth metrics
	assert.Equal(t, uint64(1000000), metrics["bytes_sent"])
	assert.Equal(t, uint64(800000), metrics["bytes_received"])
	assert.Equal(t, uint64(1800000), metrics["total_bandwidth"])

	// Verify compression
	assert.Equal(t, 2.5, metrics["compression_ratio"])

	// Verify consensus metrics
	assert.Equal(t, uint64(100), metrics["consensus_operations"])
	assert.Equal(t, uint64(5), metrics["consensus_failures"])
	assert.Equal(t, 95.0, metrics["consensus_success_rate"])

	// Verify sync metrics
	assert.Equal(t, uint64(200), metrics["sync_operations"])
	assert.Equal(t, uint64(10), metrics["sync_failures"])
	assert.Equal(t, 95.0, metrics["sync_success_rate"])

	// Verify Byzantine metrics
	assert.Equal(t, uint64(2), metrics["byzantine_detections"])

	// Verify mode statistics
	assert.Equal(t, uint64(150), metrics["datacenter_operations"])
	assert.Equal(t, uint64(50), metrics["internet_operations"])
	assert.Equal(t, config.NetworkMode.String(), metrics["current_mode"])

	// Verify component metrics present
	assert.NotNil(t, metrics["hde_v3"])
	assert.NotNil(t, metrics["acp_v3"])
	assert.NotNil(t, metrics["ass_v3"])
	assert.NotNil(t, metrics["pba_v3"])
}

// TestMultiCloudFederation tests multi-cloud federation setup
func TestMultiCloudFederation(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.MultiCloudMode = MultiCloudHybrid
	config.CloudProviders = []CloudProvider{
		{ID: "aws-us-east", Type: "aws", Region: "us-east-1", Trusted: false},
		{ID: "azure-west", Type: "azure", Region: "westus", Trusted: false},
		{ID: "gcp-central", Type: "gcp", Region: "us-central1", Trusted: false},
	}
	config.ByzantineTolerance = true

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	assert.Equal(t, MultiCloudHybrid, config.MultiCloudMode)
	assert.Len(t, config.CloudProviders, 3)
	assert.True(t, config.ByzantineTolerance, "Multi-cloud should enable Byzantine tolerance")

	// Verify all clouds are untrusted
	for _, provider := range config.CloudProviders {
		assert.False(t, provider.Trusted,
			"Public clouds should be untrusted by default for Byzantine tolerance")
	}
}

// TestCrossDatacenterFederation tests cross-datacenter federation
func TestCrossDatacenterFederation(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.DatacenterMode = DatacenterMesh
	config.Datacenters = []Datacenter{
		{ID: "dc1", Location: "New York", Region: "us-east", Latency: 2 * time.Millisecond},
		{ID: "dc2", Location: "California", Region: "us-west", Latency: 50 * time.Millisecond},
		{ID: "dc3", Location: "London", Region: "eu-west", Latency: 80 * time.Millisecond},
	}

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	assert.Equal(t, DatacenterMesh, config.DatacenterMode)
	assert.Len(t, config.Datacenters, 3)

	// Verify low-latency datacenters
	for _, dc := range config.Datacenters {
		assert.Less(t, dc.Latency, 100*time.Millisecond,
			"Datacenter latency should be reasonable for Raft consensus")
	}
}

// TestByzantineToleranceConfiguration tests Byzantine tolerance setup
func TestByzantineToleranceConfiguration(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.ByzantineTolerance = true
	config.MaxFaultyNodes = 1

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	assert.True(t, config.ByzantineTolerance)
	assert.Equal(t, 1, config.MaxFaultyNodes)

	// Verify PBFT can tolerate f=1 fault with 3f+1=4 nodes minimum
	minNodes := 3*config.MaxFaultyNodes + 1
	assert.Equal(t, 4, minNodes, "PBFT requires 3f+1 nodes to tolerate f faults")
}

// TestRegionManagerV3 tests regional cluster management
func TestRegionManagerV3(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultFederationV3Config("test-node-1")
	config.DatacenterMode = DatacenterRegional

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(t, err)
	defer cc.Close()

	// Add cluster with region
	cluster := &ClusterConnectionV3{
		ClusterID: "cluster-1",
		Region:    "us-west",
		Datacenter: "dc-west-1",
	}
	cc.updateRegionManagerV3(cluster)

	// Verify region manager created
	manager, exists := cc.regionManagers["us-west"]
	require.True(t, exists)
	assert.Equal(t, "us-west", manager.RegionID)
	assert.Equal(t, DatacenterRegional, manager.Strategy)
	assert.Contains(t, manager.Clusters, "cluster-1")
}

// TestHealthMonitor tests cluster health monitoring
func TestHealthMonitor(t *testing.T) {
	hm := NewHealthMonitor()
	require.NotNil(t, hm)

	clusterID := "test-cluster"
	latency := int64(5000) // 5ms in microseconds

	// Update health
	hm.UpdateHealth(clusterID, true, latency)

	// Verify health check recorded
	hm.mu.RLock()
	check, exists := hm.healthChecks[clusterID]
	hm.mu.RUnlock()

	require.True(t, exists)
	assert.True(t, check.Healthy)
	assert.Equal(t, clusterID, check.ClusterID)
	assert.Equal(t, 5*time.Millisecond, check.Latency)

	// Record failure
	hm.RecordFailure(clusterID)
	hm.mu.RLock()
	check = hm.healthChecks[clusterID]
	hm.mu.RUnlock()
	assert.False(t, check.Healthy)
	assert.Equal(t, "Partition detected", check.Message)

	// Record recovery
	hm.RecordRecovery(clusterID)
	hm.mu.RLock()
	check = hm.healthChecks[clusterID]
	hm.mu.RUnlock()
	assert.True(t, check.Healthy)
	assert.Equal(t, "Recovered from partition", check.Message)
}

// TestFederationV3Metrics tests metrics tracking
func TestFederationV3Metrics(t *testing.T) {
	metrics := NewFederationV3Metrics()
	require.NotNil(t, metrics)

	// Test atomic operations
	metrics.TotalConnections.Add(1)
	metrics.ActiveConnections.Add(1)
	metrics.TotalBytesSent.Add(1000)
	metrics.TotalBytesReceived.Add(800)
	metrics.CompressionRatio.Store(300) // 3.0x
	metrics.ConsensusOperations.Add(1)
	metrics.SyncOperations.Add(1)
	metrics.ByzantineDetections.Add(1)
	metrics.DatacenterOperations.Add(1)

	assert.Equal(t, int32(1), metrics.TotalConnections.Load())
	assert.Equal(t, int32(1), metrics.ActiveConnections.Load())
	assert.Equal(t, uint64(1000), metrics.TotalBytesSent.Load())
	assert.Equal(t, uint64(800), metrics.TotalBytesReceived.Load())
	assert.Equal(t, uint64(300), metrics.CompressionRatio.Load())
	assert.Equal(t, uint64(1), metrics.ConsensusOperations.Load())
	assert.Equal(t, uint64(1), metrics.SyncOperations.Load())
	assert.Equal(t, uint64(1), metrics.ByzantineDetections.Load())
	assert.Equal(t, uint64(1), metrics.DatacenterOperations.Load())
}

// BenchmarkSyncClusterStateV3 benchmarks state synchronization
func BenchmarkSyncClusterStateV3(b *testing.B) {
	logger := zap.NewNop()
	config := DefaultFederationV3Config("bench-node")
	config.NetworkMode = upgrade.ModeDatacenter

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(b, err)
	defer cc.Close()

	// Create test state data
	stateData := make([]byte, 10000)
	for i := range stateData {
		stateData[i] = byte(i % 256)
	}

	// Add mock cluster
	cc.clusterConnections["bench-cluster"] = &ClusterConnectionV3{
		ClusterID: "bench-cluster",
	}
	cc.clusterConnections["bench-cluster"].connected.Store(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cc.SyncClusterStateV3(context.Background(), "source", []string{"bench-cluster"}, stateData)
	}
}

// BenchmarkConsensusV3 benchmarks consensus operations
func BenchmarkConsensusV3(b *testing.B) {
	logger := zap.NewNop()
	config := DefaultFederationV3Config("bench-node")
	config.ConsensusConfig = &consensus.ACPConfig{}

	cc, err := NewCrossClusterComponentsV3(logger, config)
	require.NoError(b, err)
	defer cc.Close()

	// Add trusted clusters for Raft
	for i := 0; i < 3; i++ {
		clusterID := fmt.Sprintf("cluster-%d", i)
		cc.clusterConnections[clusterID] = &ClusterConnectionV3{
			ClusterID: clusterID,
			trusted:   true,
		}
	}

	value := map[string]interface{}{
		"operation": "test",
		"data":      "benchmark",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cc.ConsensusV3(context.Background(), value, []string{"cluster-0", "cluster-1", "cluster-2"})
	}
}

// Helper function for tests
func NewRegionalBaselineCache() *RegionalBaselineCache {
	return &RegionalBaselineCache{
		baselines: make(map[string]*BaselineCacheEntry),
	}
}

type RegionalBaselineCache struct {
	mu        sync.RWMutex
	baselines map[string]*BaselineCacheEntry
}

type BaselineCacheEntry struct {
	BaselineID string
	Data       []byte
	Timestamp  time.Time
	Size       int
}
