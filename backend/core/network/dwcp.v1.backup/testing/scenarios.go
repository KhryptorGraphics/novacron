package testing

import (
	"fmt"
	"time"
)

// TestScenario represents a complete test scenario
type TestScenario struct {
	Name        string
	Description string
	Topology    *NetworkTopology
	Workload    *Workload
	Duration    time.Duration
	Assertions  []Assertion
	Tags        []string
}

// Workload represents a test workload
type Workload struct {
	Type          WorkloadType
	VMs           int
	VMSize        int64 // bytes
	Operations    int
	Concurrency   int
	Pattern       WorkloadPattern
	ThinkTime     time.Duration
}

// WorkloadType defines the type of workload
type WorkloadType string

const (
	WorkloadMigration  WorkloadType = "migration"
	WorkloadReplication WorkloadType = "replication"
	WorkloadBackup     WorkloadType = "backup"
	WorkloadDisasterRecovery WorkloadType = "disaster_recovery"
)

// WorkloadPattern defines the workload pattern
type WorkloadPattern string

const (
	PatternConstant    WorkloadPattern = "constant"
	PatternBursty      WorkloadPattern = "bursty"
	PatternSinusoidal  WorkloadPattern = "sinusoidal"
	PatternRealWorld   WorkloadPattern = "real_world"
)

// Assertion represents a test assertion
type Assertion struct {
	Type      AssertionType
	Threshold float64
	Unit      string
	Critical  bool
}

// AssertionType defines the type of assertion
type AssertionType string

const (
	AssertionBandwidthUtilization AssertionType = "bandwidth_utilization"
	AssertionMigrationTime        AssertionType = "migration_time"
	AssertionCompressionRatio     AssertionType = "compression_ratio"
	AssertionThroughput           AssertionType = "throughput"
	AssertionLatency              AssertionType = "latency"
	AssertionPacketLoss           AssertionType = "packet_loss"
	AssertionCPUUsage             AssertionType = "cpu_usage"
	AssertionMemoryUsage          AssertionType = "memory_usage"
	AssertionSuccessRate          AssertionType = "success_rate"
)

// NewCrossRegionScenario creates a cross-region migration scenario
func NewCrossRegionScenario() *TestScenario {
	return &TestScenario{
		Name:        "Cross-Region Migration",
		Description: "Migrate VMs across continents with realistic WAN conditions",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"us-east": {
					ID:       "us-east-1",
					Region:   "us-east",
					Location: GeoLocation{Latitude: 40.7128, Longitude: -74.0060}, // New York
					Nodes: []*Node{
						{ID: "us-east-node-1", Datacenter: "us-east-1", IPAddress: "10.0.1.1"},
						{ID: "us-east-node-2", Datacenter: "us-east-1", IPAddress: "10.0.1.2"},
					},
				},
				"eu-west": {
					ID:       "eu-west-1",
					Region:   "eu-west",
					Location: GeoLocation{Latitude: 51.5074, Longitude: -0.1278}, // London
					Nodes: []*Node{
						{ID: "eu-west-node-1", Datacenter: "eu-west-1", IPAddress: "10.0.2.1"},
						{ID: "eu-west-node-2", Datacenter: "eu-west-1", IPAddress: "10.0.2.2"},
					},
				},
				"ap-south": {
					ID:       "ap-south-1",
					Region:   "ap-south",
					Location: GeoLocation{Latitude: 19.0760, Longitude: 72.8777}, // Mumbai
					Nodes: []*Node{
						{ID: "ap-south-node-1", Datacenter: "ap-south-1", IPAddress: "10.0.3.1"},
						{ID: "ap-south-node-2", Datacenter: "ap-south-1", IPAddress: "10.0.3.2"},
					},
				},
			},
			Links: map[string]*Link{
				"us-east-eu-west": {
					Source:      "us-east-1",
					Destination: "eu-west-1",
					Latency: LatencyProfile{
						BaseLatency:  80 * time.Millisecond,
						Jitter:       10 * time.Millisecond,
						Distribution: DistributionNormal,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    10000, // 10 Gbps
						Utilization: 0.3,
						Burstable:   true,
					},
					PacketLoss: LossProfile{
						Rate:         0.001, // 0.1%
						BurstLength:  3,
						Distribution: DistributionUniform,
					},
				},
				"us-east-ap-south": {
					Source:      "us-east-1",
					Destination: "ap-south-1",
					Latency: LatencyProfile{
						BaseLatency:  200 * time.Millisecond,
						Jitter:       20 * time.Millisecond,
						Distribution: DistributionPareto,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    5000, // 5 Gbps
						Utilization: 0.5,
						Burstable:   false,
					},
					PacketLoss: LossProfile{
						Rate:         0.005, // 0.5%
						BurstLength:  5,
						Distribution: DistributionPareto,
					},
				},
				"eu-west-ap-south": {
					Source:      "eu-west-1",
					Destination: "ap-south-1",
					Latency: LatencyProfile{
						BaseLatency:  150 * time.Millisecond,
						Jitter:       15 * time.Millisecond,
						Distribution: DistributionNormal,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    8000, // 8 Gbps
						Utilization: 0.4,
						Burstable:   true,
					},
					PacketLoss: LossProfile{
						Rate:         0.002, // 0.2%
						BurstLength:  4,
						Distribution: DistributionUniform,
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadMigration,
			VMs:         10,
			VMSize:      4 * 1024 * 1024 * 1024, // 4 GB
			Operations:  10,
			Concurrency: 2,
			Pattern:     PatternRealWorld,
			ThinkTime:   5 * time.Second,
		},
		Duration: 30 * time.Minute,
		Assertions: []Assertion{
			{
				Type:      AssertionBandwidthUtilization,
				Threshold: 0.85,
				Unit:      "ratio",
				Critical:  true,
			},
			{
				Type:      AssertionMigrationTime,
				Threshold: 300, // 5 minutes
				Unit:      "seconds",
				Critical:  true,
			},
			{
				Type:      AssertionCompressionRatio,
				Threshold: 10.0,
				Unit:      "ratio",
				Critical:  false,
			},
			{
				Type:      AssertionSuccessRate,
				Threshold: 0.99,
				Unit:      "ratio",
				Critical:  true,
			},
		},
		Tags: []string{"migration", "cross-region", "wan"},
	}
}

// NewHighLatencyScenario creates a high-latency scenario
func NewHighLatencyScenario() *TestScenario {
	return &TestScenario{
		Name:        "High Latency Migration",
		Description: "Test migration performance under extreme latency conditions",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"dc1": {
					ID:       "dc1",
					Region:   "primary",
					Location: GeoLocation{Latitude: 37.7749, Longitude: -122.4194}, // San Francisco
				},
				"dc2": {
					ID:       "dc2",
					Region:   "remote",
					Location: GeoLocation{Latitude: -33.8688, Longitude: 151.2093}, // Sydney
				},
			},
			Links: map[string]*Link{
				"dc1-dc2": {
					Source:      "dc1",
					Destination: "dc2",
					Latency: LatencyProfile{
						BaseLatency:  300 * time.Millisecond,
						Jitter:       50 * time.Millisecond,
						Distribution: DistributionPareto,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    1000, // 1 Gbps
						Utilization: 0.6,
					},
					PacketLoss: LossProfile{
						Rate:        0.01, // 1%
						BurstLength: 10,
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadMigration,
			VMs:         5,
			VMSize:      8 * 1024 * 1024 * 1024, // 8 GB
			Operations:  5,
			Concurrency: 1,
			Pattern:     PatternConstant,
		},
		Duration: 1 * time.Hour,
		Assertions: []Assertion{
			{
				Type:      AssertionThroughput,
				Threshold: 500, // 500 Mbps minimum
				Unit:      "mbps",
				Critical:  true,
			},
			{
				Type:      AssertionCompressionRatio,
				Threshold: 15.0,
				Unit:      "ratio",
				Critical:  false,
			},
		},
		Tags: []string{"migration", "high-latency", "stress"},
	}
}

// NewPacketLossScenario creates a packet loss scenario
func NewPacketLossScenario() *TestScenario {
	return &TestScenario{
		Name:        "Packet Loss Resilience",
		Description: "Test DWCP resilience under packet loss conditions",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"dc1": {ID: "dc1", Region: "primary"},
				"dc2": {ID: "dc2", Region: "secondary"},
			},
			Links: map[string]*Link{
				"dc1-dc2": {
					Source:      "dc1",
					Destination: "dc2",
					Latency: LatencyProfile{
						BaseLatency: 50 * time.Millisecond,
						Jitter:      10 * time.Millisecond,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    10000, // 10 Gbps
						Utilization: 0.3,
					},
					PacketLoss: LossProfile{
						Rate:        0.05, // 5% loss
						BurstLength: 20,
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadMigration,
			VMs:         20,
			VMSize:      2 * 1024 * 1024 * 1024, // 2 GB
			Operations:  20,
			Concurrency: 4,
			Pattern:     PatternBursty,
		},
		Duration: 20 * time.Minute,
		Assertions: []Assertion{
			{
				Type:      AssertionSuccessRate,
				Threshold: 0.95,
				Unit:      "ratio",
				Critical:  true,
			},
			{
				Type:      AssertionPacketLoss,
				Threshold: 0.05,
				Unit:      "ratio",
				Critical:  false,
			},
		},
		Tags: []string{"migration", "packet-loss", "resilience"},
	}
}

// NewBandwidthConstrainedScenario creates a bandwidth-constrained scenario
func NewBandwidthConstrainedScenario() *TestScenario {
	return &TestScenario{
		Name:        "Bandwidth Constrained Migration",
		Description: "Test migration with limited bandwidth",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"dc1": {ID: "dc1", Region: "primary"},
				"dc2": {ID: "dc2", Region: "secondary"},
			},
			Links: map[string]*Link{
				"dc1-dc2": {
					Source:      "dc1",
					Destination: "dc2",
					Latency: LatencyProfile{
						BaseLatency: 30 * time.Millisecond,
						Jitter:      5 * time.Millisecond,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    100, // 100 Mbps (constrained)
						Utilization: 0.8,
						Burstable:   false,
					},
					PacketLoss: LossProfile{
						Rate: 0.001,
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadMigration,
			VMs:         10,
			VMSize:      4 * 1024 * 1024 * 1024, // 4 GB
			Operations:  10,
			Concurrency: 1, // Sequential due to bandwidth
			Pattern:     PatternConstant,
		},
		Duration: 2 * time.Hour,
		Assertions: []Assertion{
			{
				Type:      AssertionCompressionRatio,
				Threshold: 20.0, // High compression needed
				Unit:      "ratio",
				Critical:  true,
			},
			{
				Type:      AssertionBandwidthUtilization,
				Threshold: 0.95,
				Unit:      "ratio",
				Critical:  true,
			},
		},
		Tags: []string{"migration", "bandwidth-constrained", "compression"},
	}
}

// NewDisasterRecoveryScenario creates a disaster recovery scenario
func NewDisasterRecoveryScenario() *TestScenario {
	return &TestScenario{
		Name:        "Disaster Recovery Replication",
		Description: "Test continuous replication for disaster recovery",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"primary": {
					ID:       "primary",
					Region:   "primary",
					Location: GeoLocation{Latitude: 37.7749, Longitude: -122.4194},
				},
				"dr": {
					ID:       "dr",
					Region:   "disaster-recovery",
					Location: GeoLocation{Latitude: 47.6062, Longitude: -122.3321}, // Seattle
				},
			},
			Links: map[string]*Link{
				"primary-dr": {
					Source:      "primary",
					Destination: "dr",
					Latency: LatencyProfile{
						BaseLatency:  20 * time.Millisecond,
						Jitter:       3 * time.Millisecond,
						Distribution: DistributionNormal,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    10000, // 10 Gbps dedicated
						Utilization: 0.2,
						Burstable:   true,
					},
					PacketLoss: LossProfile{
						Rate: 0.0001, // Very low loss
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadReplication,
			VMs:         50,
			VMSize:      8 * 1024 * 1024 * 1024, // 8 GB
			Operations:  1000,                    // Continuous
			Concurrency: 10,
			Pattern:     PatternSinusoidal, // Varying load
			ThinkTime:   1 * time.Second,
		},
		Duration: 24 * time.Hour,
		Assertions: []Assertion{
			{
				Type:      AssertionLatency,
				Threshold: 100, // 100ms max replication lag
				Unit:      "milliseconds",
				Critical:  true,
			},
			{
				Type:      AssertionSuccessRate,
				Threshold: 0.9999, // Four nines
				Unit:      "ratio",
				Critical:  true,
			},
			{
				Type:      AssertionThroughput,
				Threshold: 5000, // 5 Gbps sustained
				Unit:      "mbps",
				Critical:  true,
			},
		},
		Tags: []string{"replication", "disaster-recovery", "continuous"},
	}
}

// GetAllScenarios returns all predefined scenarios
func GetAllScenarios() []*TestScenario {
	return []*TestScenario{
		NewCrossRegionScenario(),
		NewHighLatencyScenario(),
		NewPacketLossScenario(),
		NewBandwidthConstrainedScenario(),
		NewDisasterRecoveryScenario(),
	}
}

// GetScenarioByName returns a scenario by name
func GetScenarioByName(name string) *TestScenario {
	scenarios := GetAllScenarios()
	for _, scenario := range scenarios {
		if scenario.Name == name {
			return scenario
		}
	}
	return nil
}

// GetScenariosByTag returns scenarios with a specific tag
func GetScenariosByTag(tag string) []*TestScenario {
	scenarios := GetAllScenarios()
	result := make([]*TestScenario, 0)

	for _, scenario := range scenarios {
		for _, t := range scenario.Tags {
			if t == tag {
				result = append(result, scenario)
				break
			}
		}
	}

	return result
}

// String returns a string representation of the scenario
func (s *TestScenario) String() string {
	return fmt.Sprintf("Scenario: %s\n  Description: %s\n  Duration: %v\n  VMs: %d\n  Assertions: %d",
		s.Name, s.Description, s.Duration, s.Workload.VMs, len(s.Assertions))
}
