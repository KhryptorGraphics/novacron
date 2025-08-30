package network

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// Network performance benchmarks for SDN scalability testing

func BenchmarkNetworkManager_CreateNetwork_Sequential(b *testing.B) {
	manager := NewNetworkManager(DefaultNetworkManagerConfig(), "test-node")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spec := NetworkSpec{
			Name: fmt.Sprintf("test-network-%d", i),
			Type: NetworkTypeBridge,
			IPAM: IPAMConfig{
				Subnet:  fmt.Sprintf("192.168.%d.0/24", i%255),
				Gateway: fmt.Sprintf("192.168.%d.1", i%255),
			},
		}
		_, err := manager.CreateNetwork(ctx, spec)
		if err != nil {
			b.Fatalf("Failed to create network: %v", err)
		}
	}
}

func BenchmarkNetworkManager_CreateNetwork_Parallel(b *testing.B) {
	manager := NewNetworkManager(DefaultNetworkManagerConfig(), "test-node")
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			spec := NetworkSpec{
				Name: fmt.Sprintf("test-network-parallel-%d-%d", b.N, i),
				Type: NetworkTypeBridge,
				IPAM: IPAMConfig{
					Subnet:  fmt.Sprintf("10.%d.%d.0/24", (i/255)%255, i%255),
					Gateway: fmt.Sprintf("10.%d.%d.1", (i/255)%255, i%255),
				},
			}
			_, err := manager.CreateNetwork(ctx, spec)
			if err != nil {
				b.Fatalf("Failed to create network: %v", err)
			}
			i++
		}
	})
}

func BenchmarkNetworkManager_GetNetwork(b *testing.B) {
	manager := setupNetworkManagerWithNetworks(b, 1000)

	// Get all network IDs for testing
	networks := manager.ListNetworks()
	if len(networks) == 0 {
		b.Fatal("No networks available for benchmark")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		networkID := networks[i%len(networks)].ID
		_, err := manager.GetNetwork(networkID)
		if err != nil {
			b.Fatalf("Failed to get network: %v", err)
		}
	}
}

func BenchmarkNetworkManager_ListNetworks(b *testing.B) {
	sizes := []int{100, 500, 1000, 5000}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Networks_%d", size), func(b *testing.B) {
			manager := setupNetworkManagerWithNetworks(b, size)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				networks := manager.ListNetworks()
				if len(networks) != size {
					b.Fatalf("Expected %d networks, got %d", size, len(networks))
				}
			}
		})
	}
}

func BenchmarkNetworkManager_ConnectVM_Concurrent(b *testing.B) {
	manager := setupNetworkManagerWithNetworks(b, 100)
	networks := manager.ListNetworks()
	ctx := context.Background()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			networkID := networks[i%len(networks)].ID
			vmID := fmt.Sprintf("vm-%d-%d", b.N, i)
			err := manager.ConnectVM(ctx, networkID, vmID)
			if err != nil {
				b.Fatalf("Failed to connect VM: %v", err)
			}
			i++
		}
	})
}

// Overlay Network Benchmarks

func BenchmarkOverlayManager_CreateNetwork_VXLAN(b *testing.B) {
	overlayManager := NewOverlayManager()
	// Register mock VXLAN driver
	driver := &MockVXLANDriver{}
	overlayManager.RegisterDriver(driver)
	overlayManager.Initialize(context.Background())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network := OverlayNetwork{
			ID:   fmt.Sprintf("overlay-%d", i),
			Name: fmt.Sprintf("overlay-net-%d", i),
			Type: VXLAN,
			CIDR: fmt.Sprintf("172.16.%d.0/24", i%255),
			VNI:  uint32(1000 + i),
			MTU:  1500,
		}
		err := overlayManager.CreateNetwork(context.Background(), network, "mock-vxlan")
		if err != nil {
			b.Fatalf("Failed to create overlay network: %v", err)
		}
	}
}

func BenchmarkOverlayManager_CreateEndpoint(b *testing.B) {
	overlayManager := setupOverlayManagerWithNetworks(b, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		endpoint := EndpointConfig{
			NetworkID:  fmt.Sprintf("overlay-%d", i%100),
			Name:       fmt.Sprintf("endpoint-%d", i),
			MACAddress: generateTestMACAddress(i),
			IPAddress:  fmt.Sprintf("172.16.%d.%d", (i%100), (i%255)+1),
		}
		err := overlayManager.CreateEndpoint(context.Background(), endpoint)
		if err != nil {
			b.Fatalf("Failed to create endpoint: %v", err)
		}
	}
}

// Network Policy Benchmarks (Security)

func BenchmarkNetworkPolicy_CheckConnectivity_Simple(b *testing.B) {
	manager := setupPolicyManagerWithRules(b, 100, 10) // 100 policies, 10 rules each
	sourceIP := parseIP("192.168.1.100")
	destIP := parseIP("192.168.1.200")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := manager.CheckConnectivity(sourceIP, destIP, 8080, 3306, "tcp", "tenant-1")
		if err != nil {
			b.Fatalf("Policy check failed: %v", err)
		}
	}
}

func BenchmarkNetworkPolicy_CheckConnectivity_Complex(b *testing.B) {
	// Test with more complex policy sets
	sizes := []struct {
		policies int
		rules    int
	}{
		{10, 10},    // 100 total rules
		{50, 20},    // 1000 total rules
		{100, 50},   // 5000 total rules
		{200, 100},  // 20000 total rules
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Policies_%d_Rules_%d", size.policies, size.rules), func(b *testing.B) {
			manager := setupPolicyManagerWithRules(b, size.policies, size.rules)
			sourceIP := parseIP("10.0.1.100")
			destIP := parseIP("10.0.2.200")

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tenantID := fmt.Sprintf("tenant-%d", i%size.policies)
				_, err := manager.CheckConnectivity(sourceIP, destIP, uint16(8000+(i%1000)), 3306, "tcp", tenantID)
				if err != nil {
					b.Fatalf("Policy check failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkNetworkPolicy_RuleMatching(b *testing.B) {
	// Test rule matching performance
	rule := &NetworkRule{
		ID:                   "test-rule",
		Type:                 AllowPolicy,
		Direction:            Both,
		Protocol:             "tcp",
		SourceCIDR:           []string{"192.168.0.0/16", "10.0.0.0/8"},
		DestinationCIDR:      []string{"172.16.0.0/12"},
		SourcePortRange:      []string{"1024-65535"},
		DestinationPortRange: []string{"80", "443", "3306", "5432"},
		Priority:             1000,
		Enabled:              true,
	}

	sourceIP := parseIP("192.168.1.100")
	destIP := parseIP("172.16.1.200")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matches := ruleMatches(rule, sourceIP, destIP, uint16(8080), uint16(3306), "tcp")
		if !matches {
			b.Fatalf("Expected rule to match")
		}
	}
}

// WAN Migration Optimizer Benchmarks

func BenchmarkWANOptimizer_CompressionLevels(b *testing.B) {
	levels := []int{1, 3, 6, 9}
	testData := generateTestData(1024 * 1024) // 1MB test data

	for _, level := range levels {
		b.Run(fmt.Sprintf("Level_%d", level), func(b *testing.B) {
			config := DefaultWANMigrationConfig()
			config.CompressionLevel = level
			optimizer := NewWANMigrationOptimizer(config)

			b.ResetTimer()
			b.SetBytes(int64(len(testData)))

			for i := 0; i < b.N; i++ {
				// Benchmark compression performance
				reader := &testDataReader{data: testData}
				writer := &testDataWriter{}

				optimizedWriter, err := optimizer.OptimizeWriter(context.Background(), writer)
				if err != nil {
					b.Fatalf("Failed to create optimized writer: %v", err)
				}

				_, err = optimizedWriter.Write(testData)
				if err != nil {
					b.Fatalf("Failed to write data: %v", err)
				}
				optimizedWriter.Close()
			}
		})
	}
}

func BenchmarkWANOptimizer_EstimateTransferTime(b *testing.B) {
	config := DefaultWANMigrationConfig()
	optimizer := NewWANMigrationOptimizer(config)

	dataSizes := []int64{
		1024 * 1024,      // 1 MB
		100 * 1024 * 1024, // 100 MB
		1024 * 1024 * 1024, // 1 GB
		10 * 1024 * 1024 * 1024, // 10 GB
	}

	for _, size := range dataSizes {
		b.Run(fmt.Sprintf("Size_%dMB", size/(1024*1024)), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				duration := optimizer.EstimateTransferTime(size)
				if duration <= 0 {
					b.Fatalf("Invalid transfer time estimate: %v", duration)
				}
			}
		})
	}
}

// Network-Aware Scheduler Benchmarks

func BenchmarkNetworkScheduler_ScoreNetworkTopology(b *testing.B) {
	scheduler := setupNetworkSchedulerWithTopology(b, 100, 500) // 100 nodes, 500 VMs

	// Add some communication patterns
	for i := 0; i < 100; i++ {
		sourceVM := fmt.Sprintf("vm-%d", i)
		destVM := fmt.Sprintf("vm-%d", (i+1)%100)
		scheduler.TrackVMCommunication(sourceVM, destVM, 100.0, 1000.0)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vmID := fmt.Sprintf("vm-new-%d", i)
		nodeID := fmt.Sprintf("node-%d", i%100)
		score := scheduler.scoreNetworkTopology(vmID, nodeID)
		if score < 0 || score > 1 {
			b.Fatalf("Invalid network topology score: %f", score)
		}
	}
}

func BenchmarkNetworkScheduler_TrackVMCommunication(b *testing.B) {
	scheduler := setupNetworkSchedulerWithTopology(b, 100, 1000)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			sourceVM := fmt.Sprintf("vm-%d", i%1000)
			destVM := fmt.Sprintf("vm-%d", (i+1)%1000)
			bandwidth := float64(rand.Intn(1000)) + 10.0
			packetRate := bandwidth * 10 // Approximate packets per Mbps
			scheduler.TrackVMCommunication(sourceVM, destVM, bandwidth, packetRate)
			i++
		}
	})
}

// SDN Controller Benchmarks

func BenchmarkSDNController_InstallFlowRule(b *testing.B) {
	controller := setupSDNControllerWithTopology(b)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rule := &FlowRule{
			ID:          fmt.Sprintf("rule-%d", i),
			Priority:    5000 + (i % 1000),
			TableID:     0,
			IdleTimeout: 300,
			HardTimeout: 3600,
			Match: FlowMatch{
				EthType: "0x0800",
				IPSrc:   fmt.Sprintf("10.0.%d.0/24", i%255),
				IPDst:   fmt.Sprintf("10.1.%d.0/24", i%255),
			},
			Actions: []FlowAction{
				{Type: ActionOutput, Params: map[string]interface{}{"port": "normal"}},
			},
		}
		err := controller.installFlowRule(rule)
		if err != nil {
			b.Fatalf("Failed to install flow rule: %v", err)
		}
	}
}

func BenchmarkSDNController_ProcessIntent_Parallel(b *testing.B) {
	controller := setupSDNControllerWithTopology(b)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			intent := &Intent{
				ID:          fmt.Sprintf("intent-%d-%d", b.N, i),
				Name:        fmt.Sprintf("test-intent-%d", i),
				Description: "Test intent for benchmarking",
				Priority:    100,
				Constraints: []Constraint{
					{Type: ConstraintTypeLatency, Params: map[string]interface{}{"max_latency_ms": 10}},
				},
				Goals: []Goal{
					{Type: GoalTypeMinimize, Target: 5.0, Operator: GoalOperatorLessThan},
				},
				Scope: IntentScope{Global: true},
			}
			controller.processIntent(intent)
			i++
		}
	})
}

func BenchmarkSDNController_CreateNetworkSlice(b *testing.B) {
	controller := setupSDNControllerWithTopology(b)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slice := &NetworkSlice{
			ID:          fmt.Sprintf("slice-%d", i),
			Name:        fmt.Sprintf("test-slice-%d", i),
			Description: "Benchmark network slice",
			Type:        SliceTypeLowLatency,
			QoSProfile: QoSProfile{
				MaxLatency:    5 * time.Millisecond,
				MinBandwidth:  100 * 1024 * 1024, // 100 Mbps
				MaxJitter:     1 * time.Millisecond,
				MaxPacketLoss: 0.001, // 0.1%
				Availability:  0.999,  // 99.9%
				Priority:      7,
			},
			Resources: SliceResources{
				BandwidthMbps: 100,
				ComputeNodes:  []string{fmt.Sprintf("node-%d", i%10)},
			},
			Endpoints: []SliceEndpoint{
				{
					ID:     fmt.Sprintf("endpoint-%d-1", i),
					NodeID: fmt.Sprintf("node-%d", i%10),
					Type:   "vm",
				},
			},
		}
		err := controller.CreateNetworkSlice(slice)
		if err != nil {
			b.Fatalf("Failed to create network slice: %v", err)
		}
	}
}

// Multi-tenant scaling benchmarks

func BenchmarkMultiTenant_NetworkCreation_1000Tenants(b *testing.B) {
	managers := make([]*NetworkManager, 1000)
	for i := 0; i < 1000; i++ {
		managers[i] = NewNetworkManager(DefaultNetworkManagerConfig(), fmt.Sprintf("node-%d", i%10))
	}

	ctx := context.Background()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			tenantID := i % 1000
			manager := managers[tenantID]
			
			spec := NetworkSpec{
				Name: fmt.Sprintf("tenant-%d-network-%d", tenantID, i/1000),
				Type: NetworkTypeOverlay,
				IPAM: IPAMConfig{
					Subnet:  fmt.Sprintf("10.%d.%d.0/24", tenantID/255, tenantID%255),
					Gateway: fmt.Sprintf("10.%d.%d.1", tenantID/255, tenantID%255),
				},
				Labels: map[string]string{
					"tenant_id": fmt.Sprintf("tenant-%d", tenantID),
				},
			}
			
			_, err := manager.CreateNetwork(ctx, spec)
			if err != nil {
				b.Fatalf("Failed to create network for tenant %d: %v", tenantID, err)
			}
			i++
		}
	})
}

func BenchmarkMultiTenant_PolicyEvaluation_10000VMs(b *testing.B) {
	// Setup 1000 tenants with varying policy complexity
	managers := setupMultiTenantPolicyManagers(b, 1000)

	// Generate random VM connections to test
	connections := generateTestConnections(10000)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			conn := connections[i%len(connections)]
			tenantID := fmt.Sprintf("tenant-%d", i%1000)
			manager := managers[tenantID]
			
			_, err := manager.CheckConnectivity(
				conn.SourceIP, conn.DestIP,
				conn.SourcePort, conn.DestPort,
				conn.Protocol, tenantID,
			)
			if err != nil {
				b.Fatalf("Policy evaluation failed: %v", err)
			}
			i++
		}
	})
}

// Memory usage benchmarks

func BenchmarkNetworkManager_MemoryUsage(b *testing.B) {
	sizes := []int{1000, 5000, 10000}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Networks_%d", size), func(b *testing.B) {
			var m1, m2 MemUsage
			m1 = getMemUsage()
			
			manager := setupNetworkManagerWithNetworks(b, size)
			_ = manager // Use the manager to prevent optimization
			
			m2 = getMemUsage()
			memoryUsed := m2.Alloc - m1.Alloc
			
			b.ReportMetric(float64(memoryUsed)/float64(size), "bytes/network")
			b.Logf("Memory usage for %d networks: %d bytes (%.2f KB per network)", 
				size, memoryUsed, float64(memoryUsed)/(float64(size)*1024))
		})
	}
}

// Latency benchmarks for critical path operations

func BenchmarkNetworkOperations_CriticalPath(b *testing.B) {
	manager := NewNetworkManager(DefaultNetworkManagerConfig(), "test-node")
	ctx := context.Background()

	b.Run("NetworkCreation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			spec := NetworkSpec{
				Name: fmt.Sprintf("latency-test-%d", i),
				Type: NetworkTypeBridge,
				IPAM: IPAMConfig{
					Subnet:  fmt.Sprintf("192.168.%d.0/24", i%255),
					Gateway: fmt.Sprintf("192.168.%d.1", i%255),
				},
			}
			b.StartTimer()
			
			start := time.Now()
			_, err := manager.CreateNetwork(ctx, spec)
			duration := time.Since(start)
			
			if err != nil {
				b.Fatalf("Failed to create network: %v", err)
			}
			
			b.ReportMetric(float64(duration.Microseconds()), "μs/op")
		}
	})
}

// Helper functions for benchmark setup

func setupNetworkManagerWithNetworks(b *testing.B, count int) *NetworkManager {
	manager := NewNetworkManager(DefaultNetworkManagerConfig(), "test-node")
	ctx := context.Background()

	for i := 0; i < count; i++ {
		spec := NetworkSpec{
			Name: fmt.Sprintf("setup-network-%d", i),
			Type: NetworkTypeBridge,
			IPAM: IPAMConfig{
				Subnet:  fmt.Sprintf("192.168.%d.0/24", i%255),
				Gateway: fmt.Sprintf("192.168.%d.1", i%255),
			},
		}
		_, err := manager.CreateNetwork(ctx, spec)
		if err != nil {
			b.Fatalf("Failed to setup network %d: %v", i, err)
		}
	}
	return manager
}

func setupOverlayManagerWithNetworks(b *testing.B, count int) *OverlayManager {
	manager := NewOverlayManager()
	driver := &MockVXLANDriver{}
	manager.RegisterDriver(driver)
	manager.Initialize(context.Background())

	for i := 0; i < count; i++ {
		network := OverlayNetwork{
			ID:   fmt.Sprintf("overlay-%d", i),
			Name: fmt.Sprintf("overlay-net-%d", i),
			Type: VXLAN,
			CIDR: fmt.Sprintf("172.16.%d.0/24", i%255),
			VNI:  uint32(1000 + i),
			MTU:  1500,
		}
		err := manager.CreateNetwork(context.Background(), network, "mock-vxlan")
		if err != nil {
			b.Fatalf("Failed to setup overlay network %d: %v", i, err)
		}
	}
	return manager
}

// Mock drivers and test data structures

type MockVXLANDriver struct{}

func (d *MockVXLANDriver) Name() string { return "mock-vxlan" }

func (d *MockVXLANDriver) Initialize(ctx context.Context) error { return nil }

func (d *MockVXLANDriver) Capabilities() DriverCapabilities {
	return DriverCapabilities{
		SupportedTypes:              []OverlayType{VXLAN},
		MaxMTU:                      9000,
		SupportsL2Extension:         true,
		SupportsNetworkPolicies:     true,
		SupportsQoS:                 true,
		SupportsServiceMesh:         false,
	}
}

func (d *MockVXLANDriver) CreateNetwork(ctx context.Context, network OverlayNetwork) error {
	// Simulate network creation delay
	time.Sleep(10 * time.Microsecond)
	return nil
}

func (d *MockVXLANDriver) DeleteNetwork(ctx context.Context, networkID string) error {
	return nil
}

func (d *MockVXLANDriver) UpdateNetwork(ctx context.Context, network OverlayNetwork) error {
	return nil
}

func (d *MockVXLANDriver) GetNetwork(ctx context.Context, networkID string) (OverlayNetwork, error) {
	return OverlayNetwork{}, nil
}

func (d *MockVXLANDriver) ListNetworks(ctx context.Context) ([]OverlayNetwork, error) {
	return []OverlayNetwork{}, nil
}

func (d *MockVXLANDriver) CreateEndpoint(ctx context.Context, endpoint EndpointConfig) error {
	time.Sleep(5 * time.Microsecond)
	return nil
}

func (d *MockVXLANDriver) DeleteEndpoint(ctx context.Context, networkID, endpointName string) error {
	return nil
}

func (d *MockVXLANDriver) GetEndpoint(ctx context.Context, networkID, endpointName string) (EndpointConfig, error) {
	return EndpointConfig{}, nil
}

func (d *MockVXLANDriver) ListEndpoints(ctx context.Context, networkID string) ([]EndpointConfig, error) {
	return []EndpointConfig{}, nil
}

func (d *MockVXLANDriver) ApplyNetworkPolicy(ctx context.Context, networkID string, policy NetworkPolicy) error {
	return nil
}

func (d *MockVXLANDriver) RemoveNetworkPolicy(ctx context.Context, networkID, policyID string) error {
	return nil
}

func (d *MockVXLANDriver) Shutdown(ctx context.Context) error {
	return nil
}

// Test data generators and utilities

type testConnection struct {
	SourceIP   net.IP
	DestIP     net.IP
	SourcePort uint16
	DestPort   uint16
	Protocol   string
}

func generateTestConnections(count int) []testConnection {
	connections := make([]testConnection, count)
	for i := 0; i < count; i++ {
		connections[i] = testConnection{
			SourceIP:   parseIP(fmt.Sprintf("10.0.%d.%d", (i/255)%255, i%255)),
			DestIP:     parseIP(fmt.Sprintf("10.1.%d.%d", (i/255)%255, i%255)),
			SourcePort: uint16(32768 + (i % 32767)),
			DestPort:   uint16(80 + (i % 100)),
			Protocol:   "tcp",
		}
	}
	return connections
}

func generateTestMACAddress(i int) string {
	return fmt.Sprintf("52:54:00:%02x:%02x:%02x", 
		(i>>16)&0xff, (i>>8)&0xff, i&0xff)
}

func generateTestData(size int) []byte {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}
	return data
}

type testDataReader struct {
	data []byte
	pos  int
}

func (r *testDataReader) Read(p []byte) (n int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

type testDataWriter struct {
	data []byte
}

func (w *testDataWriter) Write(p []byte) (n int, error) {
	w.data = append(w.data, p...)
	return len(p), nil
}

// Memory usage tracking
type MemUsage struct {
	Alloc      uint64
	TotalAlloc uint64
	Sys        uint64
	NumGC      uint32
}

func getMemUsage() MemUsage {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemUsage{
		Alloc:      m.Alloc,
		TotalAlloc: m.TotalAlloc,
		Sys:        m.Sys,
		NumGC:      m.NumGC,
	}
}