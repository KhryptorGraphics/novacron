package hardware

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// HardwareType represents neuromorphic hardware types
type HardwareType string

const (
	Loihi2    HardwareType = "loihi2"
	TrueNorth HardwareType = "truenorth"
	Akida     HardwareType = "akida"
	Spinnaker HardwareType = "spinnaker"
	Neurogrid HardwareType = "neurogrid"
)

// HardwareCapabilities defines hardware-specific capabilities
type HardwareCapabilities struct {
	MaxNeurons          int64   `json:"max_neurons"`
	MaxSynapsesPerNeuron int64  `json:"max_synapses_per_neuron"`
	PowerConsumption    float64 `json:"power_consumption_mw"`
	ClockFrequency      float64 `json:"clock_frequency_mhz"`
	SupportedModels     []string `json:"supported_models"`
	OnChipLearning      bool    `json:"on_chip_learning"`
	SpikeLatency        time.Duration `json:"spike_latency"`
}

// HardwareDevice represents a neuromorphic hardware device
type HardwareDevice struct {
	ID           string               `json:"id"`
	Type         HardwareType         `json:"type"`
	Capabilities HardwareCapabilities `json:"capabilities"`
	Status       string               `json:"status"`
	Temperature  float64              `json:"temperature"`
	PowerUsage   float64              `json:"power_usage"`
	Utilization  float64              `json:"utilization"`
	LastSeen     time.Time            `json:"last_seen"`
}

// HardwareManager manages neuromorphic hardware devices
type HardwareManager struct {
	mu              sync.RWMutex
	devices         map[string]*HardwareDevice
	capabilities    map[HardwareType]HardwareCapabilities
	activeDevice    *HardwareDevice
	monitoringChan  chan *HardwareMetrics
	stopChan        chan struct{}
}

// HardwareMetrics represents hardware performance metrics
type HardwareMetrics struct {
	DeviceID        string    `json:"device_id"`
	Timestamp       time.Time `json:"timestamp"`
	PowerUsage      float64   `json:"power_usage_mw"`
	Temperature     float64   `json:"temperature_c"`
	Utilization     float64   `json:"utilization_percent"`
	SpikeRate       float64   `json:"spike_rate_hz"`
	InferenceLatency float64  `json:"inference_latency_us"`
	ActiveNeurons   int64     `json:"active_neurons"`
}

// NewHardwareManager creates a new hardware manager
func NewHardwareManager() *HardwareManager {
	hm := &HardwareManager{
		devices:        make(map[string]*HardwareDevice),
		capabilities:   getDefaultCapabilities(),
		monitoringChan: make(chan *HardwareMetrics, 100),
		stopChan:       make(chan struct{}),
	}

	// Start monitoring
	go hm.monitorDevices()

	return hm
}

// RegisterDevice registers a neuromorphic hardware device
func (hm *HardwareManager) RegisterDevice(ctx context.Context, hwType HardwareType, deviceID string) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	caps, ok := hm.capabilities[hwType]
	if !ok {
		return fmt.Errorf("unsupported hardware type: %s", hwType)
	}

	device := &HardwareDevice{
		ID:           deviceID,
		Type:         hwType,
		Capabilities: caps,
		Status:       "online",
		Temperature:  25.0,
		PowerUsage:   0.0,
		Utilization:  0.0,
		LastSeen:     time.Now(),
	}

	hm.devices[deviceID] = device

	// Set as active if first device
	if hm.activeDevice == nil {
		hm.activeDevice = device
	}

	return nil
}

// GetActiveDevice returns the currently active device
func (hm *HardwareManager) GetActiveDevice() (*HardwareDevice, error) {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	if hm.activeDevice == nil {
		return nil, fmt.Errorf("no active device")
	}

	return hm.activeDevice, nil
}

// SetActiveDevice sets the active device
func (hm *HardwareManager) SetActiveDevice(deviceID string) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	device, ok := hm.devices[deviceID]
	if !ok {
		return fmt.Errorf("device not found: %s", deviceID)
	}

	hm.activeDevice = device
	return nil
}

// AllocateNeurons allocates neurons on the device
func (hm *HardwareManager) AllocateNeurons(ctx context.Context, count int64) ([]int64, error) {
	device, err := hm.GetActiveDevice()
	if err != nil {
		return nil, err
	}

	if count > device.Capabilities.MaxNeurons {
		return nil, fmt.Errorf("requested neurons (%d) exceeds device capacity (%d)",
			count, device.Capabilities.MaxNeurons)
	}

	// Allocate neuron IDs
	neurons := make([]int64, count)
	for i := int64(0); i < count; i++ {
		neurons[i] = i
	}

	hm.mu.Lock()
	device.Utilization = float64(count) / float64(device.Capabilities.MaxNeurons) * 100
	hm.mu.Unlock()

	return neurons, nil
}

// SendSpikes sends spikes to the hardware
func (hm *HardwareManager) SendSpikes(ctx context.Context, spikes []Spike) error {
	device, err := hm.GetActiveDevice()
	if err != nil {
		return err
	}

	// Simulate spike processing
	hm.mu.Lock()
	device.PowerUsage = float64(len(spikes)) * 0.1 // 0.1mW per spike
	device.LastSeen = time.Now()
	hm.mu.Unlock()

	// Record metrics
	metrics := &HardwareMetrics{
		DeviceID:         device.ID,
		Timestamp:        time.Now(),
		PowerUsage:       device.PowerUsage,
		Temperature:      device.Temperature,
		Utilization:      device.Utilization,
		SpikeRate:        float64(len(spikes)) / 0.001, // spikes per second
		InferenceLatency: float64(device.Capabilities.SpikeLatency.Microseconds()),
	}

	select {
	case hm.monitoringChan <- metrics:
	default:
		// Channel full, skip
	}

	return nil
}

// GetMetrics returns the metrics channel
func (hm *HardwareManager) GetMetrics() <-chan *HardwareMetrics {
	return hm.monitoringChan
}

// monitorDevices monitors hardware device health
func (hm *HardwareManager) monitorDevices() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hm.mu.Lock()
			for _, device := range hm.devices {
				// Simulate temperature changes
				if device.PowerUsage > 100 {
					device.Temperature += 0.5
				} else if device.Temperature > 25 {
					device.Temperature -= 0.1
				}

				// Check thermal limits
				if device.Temperature > 85 {
					device.Status = "thermal-throttle"
				} else {
					device.Status = "online"
				}
			}
			hm.mu.Unlock()

		case <-hm.stopChan:
			return
		}
	}
}

// Close stops the hardware manager
func (hm *HardwareManager) Close() error {
	close(hm.stopChan)
	close(hm.monitoringChan)
	return nil
}

// Spike represents a neural spike event
type Spike struct {
	NeuronID  int64     `json:"neuron_id"`
	Timestamp float64   `json:"timestamp"`
	Weight    float64   `json:"weight"`
}

// getDefaultCapabilities returns default hardware capabilities
func getDefaultCapabilities() map[HardwareType]HardwareCapabilities {
	return map[HardwareType]HardwareCapabilities{
		Loihi2: {
			MaxNeurons:           1_000_000,
			MaxSynapsesPerNeuron: 4096,
			PowerConsumption:     100.0, // 100mW
			ClockFrequency:       100.0,
			SupportedModels:      []string{"lif", "izhikevich"},
			OnChipLearning:       true,
			SpikeLatency:         10 * time.Microsecond,
		},
		TrueNorth: {
			MaxNeurons:           1_000_000,
			MaxSynapsesPerNeuron: 256,
			PowerConsumption:     70.0,
			ClockFrequency:       1000.0,
			SupportedModels:      []string{"lif"},
			OnChipLearning:       false,
			SpikeLatency:         1 * time.Microsecond,
		},
		Akida: {
			MaxNeurons:           1_200_000,
			MaxSynapsesPerNeuron: 1024,
			PowerConsumption:     200.0,
			ClockFrequency:       400.0,
			SupportedModels:      []string{"lif", "izhikevich"},
			OnChipLearning:       true,
			SpikeLatency:         5 * time.Microsecond,
		},
		Spinnaker: {
			MaxNeurons:           2_000_000,
			MaxSynapsesPerNeuron: 1000,
			PowerConsumption:     1000.0,
			ClockFrequency:       200.0,
			SupportedModels:      []string{"lif", "izhikevich", "hodgkin-huxley"},
			OnChipLearning:       true,
			SpikeLatency:         100 * time.Microsecond,
		},
		Neurogrid: {
			MaxNeurons:           1_000_000,
			MaxSynapsesPerNeuron: 10000,
			PowerConsumption:     3000.0,
			ClockFrequency:       1.0,
			SupportedModels:      []string{"hodgkin-huxley"},
			OnChipLearning:       false,
			SpikeLatency:         1 * time.Millisecond,
		},
	}
}
