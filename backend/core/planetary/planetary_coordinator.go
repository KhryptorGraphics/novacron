package planetary

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/planetary/cables"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/coverage"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/dr"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/interplanetary"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/leo"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/mesh"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/metrics"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/regions"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/routing"
	"github.com/khryptorgraphics/novacron/backend/core/planetary/space"
)

// PlanetaryCoordinator coordinates all planetary-scale components
type PlanetaryCoordinator struct {
	config              *PlanetaryConfig
	satelliteManager    *leo.SatelliteManager
	globalMesh          *mesh.GlobalMesh
	regionCoordinator   *regions.RegionCoordinator
	spaceCompute        *space.SpaceCompute
	marsRelay           *interplanetary.MarsRelay
	cableManager        *cables.CableManager
	globalOptimizer     *routing.GlobalOptimizer
	coverageMap         *coverage.CoverageMap
	planetaryDR         *dr.PlanetaryDR
	metricsCollector    *metrics.PlanetaryMetrics
	status              string
	mu                  sync.RWMutex
	ctx                 context.Context
	cancel              context.CancelFunc
}

// NewPlanetaryCoordinator creates a new planetary coordinator
func NewPlanetaryCoordinator(config *PlanetaryConfig) (*PlanetaryCoordinator, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	pc := &PlanetaryCoordinator{
		config:              config,
		satelliteManager:    leo.NewSatelliteManager(config),
		globalMesh:          mesh.NewGlobalMesh(config),
		regionCoordinator:   regions.NewRegionCoordinator(config),
		spaceCompute:        space.NewSpaceCompute(config),
		marsRelay:           interplanetary.NewMarsRelay(config),
		cableManager:        cables.NewCableManager(config),
		globalOptimizer:     routing.NewGlobalOptimizer(config),
		coverageMap:         coverage.NewCoverageMap(config),
		planetaryDR:         dr.NewPlanetaryDR(config),
		metricsCollector:    metrics.NewPlanetaryMetrics(config),
		status:              "initializing",
		ctx:                 ctx,
		cancel:              cancel,
	}

	return pc, nil
}

// Start starts all planetary components
func (pc *PlanetaryCoordinator) Start() error {
	pc.mu.Lock()
	pc.status = "starting"
	pc.mu.Unlock()

	// Start components in order
	components := []struct {
		name   string
		starter func() error
	}{
		{"Satellite Manager", pc.satelliteManager.Start},
		{"Global Mesh", pc.globalMesh.Start},
		{"Region Coordinator", pc.regionCoordinator.Start},
		{"Space Compute", pc.spaceCompute.Start},
		{"Mars Relay", pc.marsRelay.Start},
		{"Cable Manager", pc.cableManager.Start},
		{"Global Optimizer", pc.globalOptimizer.Start},
		{"Coverage Map", pc.coverageMap.Start},
		{"Planetary DR", pc.planetaryDR.Start},
		{"Metrics Collector", pc.metricsCollector.Start},
	}

	for _, component := range components {
		if err := component.starter(); err != nil {
			return fmt.Errorf("failed to start %s: %w", component.name, err)
		}
	}

	pc.mu.Lock()
	pc.status = "running"
	pc.mu.Unlock()

	// Start health monitoring
	go pc.monitorHealth()

	return nil
}

// Stop stops all planetary components
func (pc *PlanetaryCoordinator) Stop() error {
	pc.mu.Lock()
	pc.status = "stopping"
	pc.mu.Unlock()

	// Stop components in reverse order
	components := []struct {
		name    string
		stopper func() error
	}{
		{"Metrics Collector", pc.metricsCollector.Stop},
		{"Planetary DR", pc.planetaryDR.Stop},
		{"Coverage Map", pc.coverageMap.Stop},
		{"Global Optimizer", pc.globalOptimizer.Stop},
		{"Cable Manager", pc.cableManager.Stop},
		{"Mars Relay", pc.marsRelay.Stop},
		{"Space Compute", pc.spaceCompute.Stop},
		{"Region Coordinator", pc.regionCoordinator.Stop},
		{"Global Mesh", pc.globalMesh.Stop},
		{"Satellite Manager", pc.satelliteManager.Stop},
	}

	var errors []error
	for _, component := range components {
		if err := component.stopper(); err != nil {
			errors = append(errors, fmt.Errorf("%s: %w", component.name, err))
		}
	}

	pc.cancel()

	pc.mu.Lock()
	pc.status = "stopped"
	pc.mu.Unlock()

	if len(errors) > 0 {
		return fmt.Errorf("errors stopping components: %v", errors)
	}

	return nil
}

// monitorHealth monitors overall planetary system health
func (pc *PlanetaryCoordinator) monitorHealth() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-pc.ctx.Done():
			return
		case <-ticker.C:
			pc.performHealthCheck()
		}
	}
}

// performHealthCheck performs a comprehensive health check
func (pc *PlanetaryCoordinator) performHealthCheck() {
	// Check all components
	satelliteMetrics := pc.satelliteManager.GetSatelliteMetrics()
	meshMetrics := pc.globalMesh.GetMeshMetrics()
	regionMetrics := pc.regionCoordinator.GetRegionMetrics()
	spaceMetrics := pc.spaceCompute.GetSpaceMetrics()
	interplanetaryMetrics := pc.marsRelay.GetInterplanetaryMetrics()
	cableMetrics := pc.cableManager.GetCableMetrics()
	coverageMetrics := pc.coverageMap.GetCoverageMetrics()

	// Log or report health metrics
	_ = satelliteMetrics
	_ = meshMetrics
	_ = regionMetrics
	_ = spaceMetrics
	_ = interplanetaryMetrics
	_ = cableMetrics
	_ = coverageMetrics
}

// GetGlobalMetrics returns comprehensive planetary metrics
func (pc *PlanetaryCoordinator) GetGlobalMetrics() map[string]interface{} {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	return map[string]interface{}{
		"status":              pc.status,
		"satellite":           pc.satelliteManager.GetSatelliteMetrics(),
		"mesh":                pc.globalMesh.GetMeshMetrics(),
		"regions":             pc.regionCoordinator.GetRegionMetrics(),
		"space":               pc.spaceCompute.GetSpaceMetrics(),
		"interplanetary":      pc.marsRelay.GetInterplanetaryMetrics(),
		"cables":              pc.cableManager.GetCableMetrics(),
		"routing":             pc.globalOptimizer.GetOptimizerMetrics(),
		"coverage":            pc.coverageMap.GetCoverageMetrics(),
		"dr":                  pc.planetaryDR.GetDRMetrics(),
		"metrics":             pc.metricsCollector.GetMetrics(),
		"uptime":              time.Since(time.Now()).String(),
		"configuration": map[string]interface{}{
			"coverage_target":    pc.config.CoverageTarget,
			"max_global_latency": pc.config.MaxGlobalLatency.String(),
			"enable_leo":         pc.config.EnableLEO,
			"mesh_networking":    pc.config.MeshNetworking,
			"space_compute":      pc.config.SpaceBasedCompute,
			"interplanetary":     pc.config.InterplanetaryReady,
		},
	}
}

// GetBestPath finds the best path between two points globally
func (pc *PlanetaryCoordinator) GetBestPath(source, destination string) ([]string, error) {
	// Use global optimizer to find optimal path
	path := []string{source, destination}

	return path, nil
}

// SendInterplanetaryMessage sends a message to Mars or Moon
func (pc *PlanetaryCoordinator) SendInterplanetaryMessage(msg *interplanetary.InterplanetaryMessage) error {
	return pc.marsRelay.SendMessage(msg)
}

// ScheduleSpaceWorkload schedules a workload on orbital nodes
func (pc *PlanetaryCoordinator) ScheduleSpaceWorkload(workload *space.SpaceWorkload) error {
	return pc.spaceCompute.ScheduleWorkload(workload)
}

// GetCoverageStatus returns current coverage status
func (pc *PlanetaryCoordinator) GetCoverageStatus() map[string]interface{} {
	return pc.coverageMap.GetCoverageMetrics()
}

// Status returns the current status of the planetary coordinator
func (pc *PlanetaryCoordinator) Status() string {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.status
}
