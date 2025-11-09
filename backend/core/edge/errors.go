package edge

import "errors"

var (
	// Configuration errors
	ErrInvalidDiscoveryInterval  = errors.New("invalid discovery interval")
	ErrInvalidMaxLatency         = errors.New("invalid max latency")
	ErrInvalidPlacementWeights   = errors.New("placement weights must sum to 1.0")
	ErrInvalidMinResources       = errors.New("invalid minimum resources")
	ErrInvalidMigrationTimeout   = errors.New("invalid migration timeout")

	// Discovery errors
	ErrEdgeNodeNotFound          = errors.New("edge node not found")
	ErrDiscoveryTimeout          = errors.New("edge discovery timeout")
	ErrInsufficientEdgeNodes     = errors.New("insufficient edge nodes available")
	ErrEdgeNodeUnhealthy         = errors.New("edge node unhealthy")

	// Placement errors
	ErrNoSuitableEdgeNode        = errors.New("no suitable edge node found")
	ErrLatencyExceedsThreshold   = errors.New("latency exceeds threshold")
	ErrInsufficientEdgeResources = errors.New("insufficient edge resources")
	ErrPlacementTimeout          = errors.New("placement decision timeout")

	// Migration errors
	ErrMigrationInProgress       = errors.New("migration already in progress")
	ErrMigrationFailed           = errors.New("migration failed")
	ErrMigrationTimeout          = errors.New("migration timeout")
	ErrInvalidMigrationTarget    = errors.New("invalid migration target")

	// MEC errors
	ErrMECNotEnabled             = errors.New("MEC not enabled")
	ErrMECPlatformUnavailable    = errors.New("MEC platform unavailable")
	ErrNetworkSliceUnavailable   = errors.New("network slice unavailable")

	// IoT errors
	ErrIoTGatewayNotEnabled      = errors.New("IoT gateway not enabled")
	ErrInsufficientIoTResources  = errors.New("insufficient IoT gateway resources")
	ErrARMArchitectureRequired   = errors.New("ARM architecture required")

	// Network errors
	ErrEdgeMeshNotEnabled        = errors.New("edge mesh not enabled")
	ErrVPNTunnelFailed           = errors.New("VPN tunnel failed")
	ErrBandwidthExceeded         = errors.New("bandwidth exceeded")

	// Policy errors
	ErrDataResidencyViolation    = errors.New("data residency policy violation")
	ErrRegionNotAllowed          = errors.New("region not allowed")
	ErrComplianceViolation       = errors.New("compliance violation")
	ErrLatencySLAViolation       = errors.New("latency SLA violation")
)
