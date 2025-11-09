package cables

import (
	"context"
	"sync"
	"time"

)

// SubmarineCable represents an underwater fiber optic cable
type SubmarineCable struct {
	CableID       string    `json:"cable_id"`
	Name          string    `json:"name"`
	Endpoints     []string  `json:"endpoints"`
	LengthKM      float64   `json:"length_km"`
	CapacityTbps  float64   `json:"capacity_tbps"`
	Latency       time.Duration `json:"latency"`
	Status        string    `json:"status"` // active, degraded, fault, offline
	FaultLocation float64   `json:"fault_location"` // km from endpoint[0]
	Health        float64   `json:"health"` // 0-1
	LastCheck     time.Time `json:"last_check"`
	InstallDate   time.Time `json:"install_date"`
}

// CableManager manages submarine cable infrastructure
type CableManager struct {
	config    *planetary.PlanetaryConfig
	cables    map[string]*SubmarineCable
	faults    []CableFault
	mu        sync.RWMutex
	ctx       context.Context
	cancel    context.CancelFunc
}

// CableFault represents a cable fault event
type CableFault struct {
	Timestamp time.Time `json:"timestamp"`
	CableID   string    `json:"cable_id"`
	Location  float64   `json:"location"`
	Severity  string    `json:"severity"`
	Resolved  bool      `json:"resolved"`
}

// NewCableManager creates a new cable manager
func NewCableManager(config *planetary.PlanetaryConfig) *CableManager {
	ctx, cancel := context.WithCancel(context.Background())

	cm := &CableManager{
		config: config,
		cables: make(map[string]*SubmarineCable),
		faults: make([]CableFault, 0),
		ctx:    ctx,
		cancel: cancel,
	}

	cm.initializeCables()

	return cm
}

// Start starts the cable manager
func (cm *CableManager) Start() error {
	if cm.config.CableFaultDetection {
		go cm.monitorCableFaults()
	}

	go cm.monitorCableHealth()

	return nil
}

// Stop stops the cable manager
func (cm *CableManager) Stop() error {
	cm.cancel()
	return nil
}

// initializeCables initializes major submarine cables
func (cm *CableManager) initializeCables() {
	cables := []struct {
		name     string
		endpoints []string
		lengthKM float64
		capacity float64
	}{
		{"TAT-14", []string{"US-East", "UK"}, 15000, 5.1},
		{"FASTER", []string{"US-West", "Japan"}, 11629, 60},
		{"MAREA", []string{"US-East", "Spain"}, 6600, 200},
		{"2Africa", []string{"Africa-East", "Europe"}, 45000, 180},
		{"SEA-ME-WE 5", []string{"Singapore", "France"}, 20000, 24},
		{"Pacific Light", []string{"Hong Kong", "US-West"}, 12800, 144},
	}

	for _, cable := range cables {
		sc := &SubmarineCable{
			CableID:       cable.name,
			Name:          cable.name,
			Endpoints:     cable.endpoints,
			LengthKM:      cable.lengthKM,
			CapacityTbps:  cable.capacity,
			Latency:       time.Duration(cable.lengthKM/200.0) * time.Millisecond,
			Status:        "active",
			FaultLocation: 0,
			Health:        1.0,
			LastCheck:     time.Now(),
			InstallDate:   time.Now().Add(-365 * 24 * time.Hour),
		}

		cm.cables[sc.CableID] = sc
	}
}

// monitorCableFaults monitors for cable faults
func (cm *CableManager) monitorCableFaults() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.detectFaults()
		}
	}
}

// detectFaults detects cable faults
func (cm *CableManager) detectFaults() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for cableID, cable := range cm.cables {
		// Simulate fault detection (in production, use OTDR and monitoring)
		if cable.Health < 0.5 {
			fault := CableFault{
				Timestamp: time.Now(),
				CableID:   cableID,
				Location:  cable.LengthKM / 2.0,
				Severity:  "high",
				Resolved:  false,
			}

			cm.faults = append(cm.faults, fault)
			cable.Status = "fault"
			cable.FaultLocation = fault.Location
		}

		cm.cables[cableID] = cable
	}
}

// monitorCableHealth monitors cable health
func (cm *CableManager) monitorCableHealth() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.updateCableHealth()
		}
	}
}

// updateCableHealth updates health for all cables
func (cm *CableManager) updateCableHealth() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for cableID, cable := range cm.cables {
		// Simulate health calculation
		cable.Health = 0.9 + (0.1 * float64(time.Now().Unix()%10) / 10.0)
		cable.LastCheck = time.Now()

		if cable.Health > 0.9 {
			cable.Status = "active"
		} else if cable.Health > 0.7 {
			cable.Status = "degraded"
		}

		cm.cables[cableID] = cable
	}
}

// GetCableMetrics returns cable infrastructure metrics
func (cm *CableManager) GetCableMetrics() map[string]interface{} {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	totalCables := len(cm.cables)
	activeCables := 0
	totalCapacity := 0.0
	totalLength := 0.0
	avgHealth := 0.0
	totalFaults := len(cm.faults)

	for _, cable := range cm.cables {
		if cable.Status == "active" {
			activeCables++
		}
		totalCapacity += cable.CapacityTbps
		totalLength += cable.LengthKM
		avgHealth += cable.Health
	}

	if totalCables > 0 {
		avgHealth /= float64(totalCables)
	}

	return map[string]interface{}{
		"total_cables":     totalCables,
		"active_cables":    activeCables,
		"total_capacity":   totalCapacity,
		"total_length_km":  totalLength,
		"avg_health":       avgHealth,
		"total_faults":     totalFaults,
	}
}
