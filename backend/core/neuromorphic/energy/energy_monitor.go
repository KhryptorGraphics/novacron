package energy

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// EnergyMonitor tracks energy consumption of neuromorphic computing
type EnergyMonitor struct {
	mu                sync.RWMutex
	samplingRate      time.Duration
	measurements      []*EnergyMeasurement
	totalEnergy       float64
	totalInferences   int64
	baselinePower     float64
	metricsChannel    chan *EnergyMetrics
	stopChannel       chan struct{}
}

// EnergyMeasurement represents a single energy measurement
type EnergyMeasurement struct {
	Timestamp        time.Time `json:"timestamp"`
	PowerConsumption float64   `json:"power_consumption_mw"`
	Voltage          float64   `json:"voltage_v"`
	Current          float64   `json:"current_ma"`
	Temperature      float64   `json:"temperature_c"`
	Activity         string    `json:"activity"`
}

// EnergyMetrics represents aggregated energy metrics
type EnergyMetrics struct {
	TotalEnergy         float64 `json:"total_energy_mj"`          // millijoules
	AveragePower        float64 `json:"average_power_mw"`
	PeakPower           float64 `json:"peak_power_mw"`
	EnergyPerInference  float64 `json:"energy_per_inference_mj"`
	InferencesPerJoule  float64 `json:"inferences_per_joule"`
	CarbonFootprint     float64 `json:"carbon_footprint_mg_co2"`  // milligrams CO2
	Efficiency          float64 `json:"efficiency_percent"`
	ComparisonToGPU     float64 `json:"comparison_to_gpu_x"`      // times better than GPU
	ComparisonToCNN     float64 `json:"comparison_to_cnn_x"`      // times better than CNN
}

// PowerMode represents different power modes
type PowerMode string

const (
	PowerNormal   PowerMode = "normal"
	PowerLow      PowerMode = "low-power"
	PowerUltraLow PowerMode = "ultra-low"
)

// NewEnergyMonitor creates a new energy monitor
func NewEnergyMonitor(samplingRate time.Duration) *EnergyMonitor {
	em := &EnergyMonitor{
		samplingRate:   samplingRate,
		measurements:   make([]*EnergyMeasurement, 0),
		baselinePower:  10.0, // 10mW baseline
		metricsChannel: make(chan *EnergyMetrics, 100),
		stopChannel:    make(chan struct{}),
	}

	// Start monitoring
	go em.monitor()

	return em
}

// RecordMeasurement records an energy measurement
func (em *EnergyMonitor) RecordMeasurement(power, voltage, current, temp float64, activity string) {
	em.mu.Lock()
	defer em.mu.Unlock()

	measurement := &EnergyMeasurement{
		Timestamp:        time.Now(),
		PowerConsumption: power,
		Voltage:          voltage,
		Current:          current,
		Temperature:      temp,
		Activity:         activity,
	}

	em.measurements = append(em.measurements, measurement)

	// Calculate energy (power * time)
	energyMJ := power * em.samplingRate.Seconds() * 1000 // millijoules
	em.totalEnergy += energyMJ
}

// RecordInference records an inference event
func (em *EnergyMonitor) RecordInference() {
	em.mu.Lock()
	defer em.mu.Unlock()
	em.totalInferences++
}

// GetMetrics returns current energy metrics
func (em *EnergyMonitor) GetMetrics() *EnergyMetrics {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if len(em.measurements) == 0 {
		return &EnergyMetrics{}
	}

	// Calculate average and peak power
	totalPower := 0.0
	peakPower := 0.0

	for _, m := range em.measurements {
		totalPower += m.PowerConsumption
		if m.PowerConsumption > peakPower {
			peakPower = m.PowerConsumption
		}
	}

	avgPower := totalPower / float64(len(em.measurements))

	// Energy per inference
	energyPerInference := 0.0
	inferencesPerJoule := 0.0
	if em.totalInferences > 0 {
		energyPerInference = em.totalEnergy / float64(em.totalInferences)
		inferencesPerJoule = 1000.0 / energyPerInference // 1 joule = 1000 millijoules
	}

	// Carbon footprint (assume 500g CO2/kWh)
	energyKWh := em.totalEnergy / 1000.0 / 3600.0 / 1000.0 // mJ to kWh
	carbonFootprint := energyKWh * 500.0 * 1000.0 // mg CO2

	// Efficiency (compared to theoretical minimum)
	theoreticalMin := 0.01 // 0.01 mJ per inference (theoretical limit)
	efficiency := 0.0
	if energyPerInference > 0 {
		efficiency = (theoreticalMin / energyPerInference) * 100.0
	}

	// Comparison to GPU (typical GPU: 200W, 1000 inferences/sec = 200mJ/inference)
	gpuEnergyPerInference := 200.0 // mJ
	comparisonToGPU := gpuEnergyPerInference / energyPerInference

	// Comparison to CNN on CPU (typical: 50mJ/inference)
	cnnEnergyPerInference := 50.0 // mJ
	comparisonToCNN := cnnEnergyPerInference / energyPerInference

	metrics := &EnergyMetrics{
		TotalEnergy:        em.totalEnergy,
		AveragePower:       avgPower,
		PeakPower:          peakPower,
		EnergyPerInference: energyPerInference,
		InferencesPerJoule: inferencesPerJoule,
		CarbonFootprint:    carbonFootprint,
		Efficiency:         efficiency,
		ComparisonToGPU:    comparisonToGPU,
		ComparisonToCNN:    comparisonToCNN,
	}

	return metrics
}

// monitor continuously monitors energy consumption
func (em *EnergyMonitor) monitor() {
	ticker := time.NewTicker(em.samplingRate)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Simulate power measurement (in practice, would read from hardware)
			power := em.baselinePower + math.Sin(float64(time.Now().UnixNano())/1e9)*5.0
			voltage := 3.3
			current := power / voltage
			temp := 25.0 + math.Sin(float64(time.Now().UnixNano())/1e9)*10.0

			em.RecordMeasurement(power, voltage, current, temp, "idle")

			// Send metrics
			metrics := em.GetMetrics()
			select {
			case em.metricsChannel <- metrics:
			default:
			}

		case <-em.stopChannel:
			return
		}
	}
}

// GetMetricsChannel returns the metrics channel
func (em *EnergyMonitor) GetMetricsChannel() <-chan *EnergyMetrics {
	return em.metricsChannel
}

// Reset resets the energy monitor
func (em *EnergyMonitor) Reset() {
	em.mu.Lock()
	defer em.mu.Unlock()

	em.measurements = make([]*EnergyMeasurement, 0)
	em.totalEnergy = 0
	em.totalInferences = 0
}

// EstimateBatteryLife estimates battery life for edge devices
func (em *EnergyMonitor) EstimateBatteryLife(batteryCapacityMAh, voltage float64) float64 {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if len(em.measurements) == 0 {
		return 0
	}

	// Calculate average current
	totalCurrent := 0.0
	for _, m := range em.measurements {
		totalCurrent += m.Current
	}
	avgCurrent := totalCurrent / float64(len(em.measurements))

	// Battery life in hours
	batteryLife := batteryCapacityMAh / avgCurrent

	return batteryLife
}

// ComparePowerModes compares different power modes
func (em *EnergyMonitor) ComparePowerModes() map[PowerMode]float64 {
	// Estimated power consumption for different modes
	return map[PowerMode]float64{
		PowerNormal:   100.0, // 100mW
		PowerLow:      10.0,  // 10mW (10x reduction)
		PowerUltraLow: 1.0,   // 1mW (100x reduction)
	}
}

// GetEnergyReport generates a detailed energy report
func (em *EnergyMonitor) GetEnergyReport() map[string]interface{} {
	metrics := em.GetMetrics()

	report := map[string]interface{}{
		"total_energy_mj":         metrics.TotalEnergy,
		"average_power_mw":        metrics.AveragePower,
		"peak_power_mw":           metrics.PeakPower,
		"energy_per_inference_mj": metrics.EnergyPerInference,
		"inferences_per_joule":    metrics.InferencesPerJoule,
		"carbon_footprint_mg_co2": metrics.CarbonFootprint,
		"efficiency_percent":      metrics.Efficiency,
		"vs_gpu":                  fmt.Sprintf("%.1fx better", metrics.ComparisonToGPU),
		"vs_cnn":                  fmt.Sprintf("%.1fx better", metrics.ComparisonToCNN),
		"total_inferences":        em.totalInferences,
		"measurement_count":       len(em.measurements),
	}

	return report
}

// Close stops the energy monitor
func (em *EnergyMonitor) Close() error {
	close(em.stopChannel)
	close(em.metricsChannel)
	return nil
}
