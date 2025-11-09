package planetary

import (
	"errors"
	"fmt"
)

var (
	// Configuration Errors
	ErrInvalidCoverageTarget      = errors.New("invalid coverage target")
	ErrInvalidAvailabilityTarget  = errors.New("invalid availability target")
	ErrInvalidLatencyTarget       = errors.New("invalid latency target")
	ErrInvalidMinRegions          = errors.New("invalid minimum regions")
	ErrNoSatelliteAPI             = errors.New("no satellite API configured")

	// Satellite Errors
	ErrSatelliteNotAvailable      = errors.New("satellite not available")
	ErrSatelliteHandoffFailed     = errors.New("satellite handoff failed")
	ErrNoVisibleSatellites        = errors.New("no visible satellites")
	ErrBeamSteeringFailed         = errors.New("beam steering failed")
	ErrDopplerCompensationFailed  = errors.New("doppler compensation failed")

	// Mesh Errors
	ErrMeshNotConverged           = errors.New("mesh network not converged")
	ErrNoRouteToDest              = errors.New("no route to destination")
	ErrBundleDeliveryFailed       = errors.New("bundle delivery failed")
	ErrDTNNotEnabled              = errors.New("DTN not enabled")

	// Regional Errors
	ErrRegionNotFound             = errors.New("region not found")
	ErrRegionIsolated             = errors.New("region isolated")
	ErrInsufficientRegions        = errors.New("insufficient regions")

	// Space Computing Errors
	ErrSpaceComputeNotAvailable   = errors.New("space compute not available")
	ErrRadiationError             = errors.New("radiation error detected")
	ErrThermalLimitExceeded       = errors.New("thermal limit exceeded")
	ErrSolarPowerInsufficient     = errors.New("solar power insufficient")

	// Interplanetary Errors
	ErrMarsRelayUnavailable       = errors.New("Mars relay unavailable")
	ErrInterplanetaryLatency      = errors.New("interplanetary latency exceeded")
	ErrDeepSpaceCommsDown         = errors.New("deep space communications down")

	// Cable Errors
	ErrCableFault                 = errors.New("submarine cable fault")
	ErrNoCableRoute               = errors.New("no cable route available")

	// Coverage Errors
	ErrCoverageInsufficient       = errors.New("coverage insufficient")
	ErrDeadZoneDetected           = errors.New("dead zone detected")

	// Routing Errors
	ErrNoViablePath               = errors.New("no viable path found")
	ErrRoutingOptimizationFailed  = errors.New("routing optimization failed")
	ErrGeopoliticalBlock          = errors.New("geopolitical routing blocked")
)

// PlanetaryError represents a planetary-scale error with context
type PlanetaryError struct {
	Op      string
	Region  string
	Err     error
	Context map[string]interface{}
}

func (e *PlanetaryError) Error() string {
	if e.Region != "" {
		return fmt.Sprintf("planetary %s [region=%s]: %v", e.Op, e.Region, e.Err)
	}
	return fmt.Sprintf("planetary %s: %v", e.Op, e.Err)
}

func (e *PlanetaryError) Unwrap() error {
	return e.Err
}

// NewPlanetaryError creates a new planetary error with context
func NewPlanetaryError(op, region string, err error, context map[string]interface{}) *PlanetaryError {
	return &PlanetaryError{
		Op:      op,
		Region:  region,
		Err:     err,
		Context: context,
	}
}
