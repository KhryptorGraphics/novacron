package planetary

import "time"

// Common types used across planetary subpackages

var (
	// Reuse error types from errors.go
	_ = ErrSatelliteNotAvailable
)

// PlanetaryConfigInterface allows subpackages to use config without circular import
type PlanetaryConfigInterface interface {
	GetCoverageTarget() float64
	GetMaxGlobalLatency() time.Duration
	IsLEOEnabled() bool
	IsDTNEnabled() bool
	IsSpaceComputeEnabled() bool
	IsInterplanetaryReady() bool
}

// Implement interface for PlanetaryConfig
func (c *PlanetaryConfig) GetCoverageTarget() float64 {
	return c.CoverageTarget
}

func (c *PlanetaryConfig) GetMaxGlobalLatency() time.Duration {
	return c.MaxGlobalLatency
}

func (c *PlanetaryConfig) IsLEOEnabled() bool {
	return c.EnableLEO
}

func (c *PlanetaryConfig) IsDTNEnabled() bool {
	return c.EnableDTN
}

func (c *PlanetaryConfig) IsSpaceComputeEnabled() bool {
	return c.SpaceBasedCompute
}

func (c *PlanetaryConfig) IsInterplanetaryReady() bool {
	return c.InterplanetaryReady
}
