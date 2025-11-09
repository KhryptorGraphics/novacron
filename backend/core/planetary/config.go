package planetary

import (
	"time"
)

// PlanetaryConfig defines the configuration for planetary-scale coordination
type PlanetaryConfig struct {
	// LEO Satellite Configuration
	EnableLEO              bool          `json:"enable_leo" yaml:"enable_leo"`
	StarlinkAPIKey         string        `json:"starlink_api_key" yaml:"starlink_api_key"`
	StarlinkAPIEndpoint    string        `json:"starlink_api_endpoint" yaml:"starlink_api_endpoint"`
	OneWebAPIKey           string        `json:"oneweb_api_key" yaml:"oneweb_api_key"`
	OneWebAPIEndpoint      string        `json:"oneweb_api_endpoint" yaml:"oneweb_api_endpoint"`
	KuiperAPIKey           string        `json:"kuiper_api_key" yaml:"kuiper_api_key"`
	KuiperAPIEndpoint      string        `json:"kuiper_api_endpoint" yaml:"kuiper_api_endpoint"`
	TelesatAPIKey          string        `json:"telesat_api_key" yaml:"telesat_api_key"`
	TelesatAPIEndpoint     string        `json:"telesat_api_endpoint" yaml:"telesat_api_endpoint"`

	// Mesh Networking
	MeshNetworking         bool          `json:"mesh_networking" yaml:"mesh_networking"`
	EnableDTN              bool          `json:"enable_dtn" yaml:"enable_dtn"`
	BundleProtocolVersion  string        `json:"bundle_protocol_version" yaml:"bundle_protocol_version"`
	OpportunisticRouting   bool          `json:"opportunistic_routing" yaml:"opportunistic_routing"`
	StoreAndForward        bool          `json:"store_and_forward" yaml:"store_and_forward"`

	// Space-Based Computing
	SpaceBasedCompute      bool          `json:"space_based_compute" yaml:"space_based_compute"`
	OrbitalDataCenters     bool          `json:"orbital_data_centers" yaml:"orbital_data_centers"`
	RadiationHardening     bool          `json:"radiation_hardening" yaml:"radiation_hardening"`
	ZeroGOptimizations     bool          `json:"zero_g_optimizations" yaml:"zero_g_optimizations"`

	// Interplanetary Communication
	InterplanetaryReady    bool          `json:"interplanetary_ready" yaml:"interplanetary_ready"`
	MarsRelayEnabled       bool          `json:"mars_relay_enabled" yaml:"mars_relay_enabled"`
	MoonBaseEnabled        bool          `json:"moon_base_enabled" yaml:"moon_base_enabled"`
	LaserCommsEnabled      bool          `json:"laser_comms_enabled" yaml:"laser_comms_enabled"`
	DeepSpaceDTN           bool          `json:"deep_space_dtn" yaml:"deep_space_dtn"`

	// Coverage and Performance Targets
	CoverageTarget         float64       `json:"coverage_target" yaml:"coverage_target"`           // 0.9999 (99.99%)
	MaxGlobalLatency       time.Duration `json:"max_global_latency" yaml:"max_global_latency"`     // 100ms
	SatelliteHandoffTime   time.Duration `json:"satellite_handoff_time" yaml:"satellite_handoff_time"` // 100ms
	MeshConvergenceTime    time.Duration `json:"mesh_convergence_time" yaml:"mesh_convergence_time"` // 1 second
	AvailabilityTarget     float64       `json:"availability_target" yaml:"availability_target"`   // 0.99999 (five nines)

	// Underwater Cable Integration
	EnableSubmarineCables  bool          `json:"enable_submarine_cables" yaml:"enable_submarine_cables"`
	HybridRouting          bool          `json:"hybrid_routing" yaml:"hybrid_routing"`
	CableFaultDetection    bool          `json:"cable_fault_detection" yaml:"cable_fault_detection"`

	// Regional Configuration
	MinRegions             int           `json:"min_regions" yaml:"min_regions"`                   // 100+
	DynamicRegions         bool          `json:"dynamic_regions" yaml:"dynamic_regions"`
	RemoteAreaCoverage     bool          `json:"remote_area_coverage" yaml:"remote_area_coverage"`
	ArcticCoverage         bool          `json:"arctic_coverage" yaml:"arctic_coverage"`
	AntarcticaCoverage     bool          `json:"antarctica_coverage" yaml:"antarctica_coverage"`

	// Disaster Recovery
	PlanetaryDR            bool          `json:"planetary_dr" yaml:"planetary_dr"`
	AutoFailover           bool          `json:"auto_failover" yaml:"auto_failover"`
	EmergencyRouting       bool          `json:"emergency_routing" yaml:"emergency_routing"`
	GeopoliticalRouting    bool          `json:"geopolitical_routing" yaml:"geopolitical_routing"`

	// Monitoring and Metrics
	GlobalLatencyHeatmap   bool          `json:"global_latency_heatmap" yaml:"global_latency_heatmap"`
	SatelliteTracking      bool          `json:"satellite_tracking" yaml:"satellite_tracking"`
	CoverageVisualization  bool          `json:"coverage_visualization" yaml:"coverage_visualization"`
	TrafficDistribution    bool          `json:"traffic_distribution" yaml:"traffic_distribution"`
}

// DefaultPlanetaryConfig returns the default planetary configuration
func DefaultPlanetaryConfig() *PlanetaryConfig {
	return &PlanetaryConfig{
		EnableLEO:              true,
		StarlinkAPIEndpoint:    "https://api.starlink.com/v1",
		OneWebAPIEndpoint:      "https://api.oneweb.net/v1",
		KuiperAPIEndpoint:      "https://api.kuiper.aws.com/v1",
		TelesatAPIEndpoint:     "https://api.telesat.com/v1",

		MeshNetworking:         true,
		EnableDTN:              true,
		BundleProtocolVersion:  "7",
		OpportunisticRouting:   true,
		StoreAndForward:        true,

		SpaceBasedCompute:      true,
		OrbitalDataCenters:     true,
		RadiationHardening:     true,
		ZeroGOptimizations:     true,

		InterplanetaryReady:    true,
		MarsRelayEnabled:       true,
		MoonBaseEnabled:        true,
		LaserCommsEnabled:      true,
		DeepSpaceDTN:           true,

		CoverageTarget:         0.9999,     // 99.99%
		MaxGlobalLatency:       100 * time.Millisecond,
		SatelliteHandoffTime:   100 * time.Millisecond,
		MeshConvergenceTime:    1 * time.Second,
		AvailabilityTarget:     0.99999,    // 99.999%

		EnableSubmarineCables:  true,
		HybridRouting:          true,
		CableFaultDetection:    true,

		MinRegions:             100,
		DynamicRegions:         true,
		RemoteAreaCoverage:     true,
		ArcticCoverage:         true,
		AntarcticaCoverage:     true,

		PlanetaryDR:            true,
		AutoFailover:           true,
		EmergencyRouting:       true,
		GeopoliticalRouting:    true,

		GlobalLatencyHeatmap:   true,
		SatelliteTracking:      true,
		CoverageVisualization:  true,
		TrafficDistribution:    true,
	}
}

// ConstellationConfig defines configuration for a satellite constellation
type ConstellationConfig struct {
	Name              string        `json:"name"`
	TotalSatellites   int           `json:"total_satellites"`
	ActiveSatellites  int           `json:"active_satellites"`
	Altitude          float64       `json:"altitude"`           // km
	Inclination       float64       `json:"inclination"`        // degrees
	APIEndpoint       string        `json:"api_endpoint"`
	APIKey            string        `json:"api_key"`
	LatencyTarget     time.Duration `json:"latency_target"`     // ms
	Bandwidth         float64       `json:"bandwidth"`          // Gbps per satellite
	Coverage          float64       `json:"coverage"`           // percentage of Earth
	HandoffTime       time.Duration `json:"handoff_time"`       // ms
	Enabled           bool          `json:"enabled"`
}

// GetConstellations returns all configured satellite constellations
func (c *PlanetaryConfig) GetConstellations() []ConstellationConfig {
	constellations := []ConstellationConfig{}

	if c.StarlinkAPIKey != "" {
		constellations = append(constellations, ConstellationConfig{
			Name:              "Starlink",
			TotalSatellites:   42000,
			ActiveSatellites:  5000,
			Altitude:          550,
			Inclination:       53,
			APIEndpoint:       c.StarlinkAPIEndpoint,
			APIKey:            c.StarlinkAPIKey,
			LatencyTarget:     20 * time.Millisecond,
			Bandwidth:         20,
			Coverage:          0.95,
			HandoffTime:       80 * time.Millisecond,
			Enabled:           true,
		})
	}

	if c.OneWebAPIKey != "" {
		constellations = append(constellations, ConstellationConfig{
			Name:              "OneWeb",
			TotalSatellites:   648,
			ActiveSatellites:  648,
			Altitude:          1200,
			Inclination:       87.9,
			APIEndpoint:       c.OneWebAPIEndpoint,
			APIKey:            c.OneWebAPIKey,
			LatencyTarget:     40 * time.Millisecond,
			Bandwidth:         8,
			Coverage:          0.90,
			HandoffTime:       100 * time.Millisecond,
			Enabled:           true,
		})
	}

	if c.KuiperAPIKey != "" {
		constellations = append(constellations, ConstellationConfig{
			Name:              "Kuiper",
			TotalSatellites:   3236,
			ActiveSatellites:  500,
			Altitude:          630,
			Inclination:       51.9,
			APIEndpoint:       c.KuiperAPIEndpoint,
			APIKey:            c.KuiperAPIKey,
			LatencyTarget:     30 * time.Millisecond,
			Bandwidth:         15,
			Coverage:          0.92,
			HandoffTime:       90 * time.Millisecond,
			Enabled:           true,
		})
	}

	if c.TelesatAPIKey != "" {
		constellations = append(constellations, ConstellationConfig{
			Name:              "Telesat",
			TotalSatellites:   298,
			ActiveSatellites:  200,
			Altitude:          1000,
			Inclination:       99.5,
			APIEndpoint:       c.TelesatAPIEndpoint,
			APIKey:            c.TelesatAPIKey,
			LatencyTarget:     35 * time.Millisecond,
			Bandwidth:         10,
			Coverage:          0.85,
			HandoffTime:       95 * time.Millisecond,
			Enabled:           true,
		})
	}

	return constellations
}

// Validate validates the planetary configuration
func (c *PlanetaryConfig) Validate() error {
	if c.CoverageTarget < 0 || c.CoverageTarget > 1 {
		return ErrInvalidCoverageTarget
	}

	if c.AvailabilityTarget < 0 || c.AvailabilityTarget > 1 {
		return ErrInvalidAvailabilityTarget
	}

	if c.MaxGlobalLatency <= 0 {
		return ErrInvalidLatencyTarget
	}

	if c.MinRegions < 1 {
		return ErrInvalidMinRegions
	}

	if c.EnableLEO && c.StarlinkAPIKey == "" && c.OneWebAPIKey == "" &&
	   c.KuiperAPIKey == "" && c.TelesatAPIKey == "" {
		return ErrNoSatelliteAPI
	}

	return nil
}
