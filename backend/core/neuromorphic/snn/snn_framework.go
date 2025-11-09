package snn

import (
	"context"
	"math"
	"sync"
)

// NeuronModel represents different neuron models
type NeuronModel string

const (
	LIF          NeuronModel = "lif"           // Leaky Integrate-and-Fire
	Izhikevich   NeuronModel = "izhikevich"    // Izhikevich model
	HodgkinHuxley NeuronModel = "hodgkin-huxley" // Hodgkin-Huxley model
)

// Neuron represents a spiking neuron
type Neuron struct {
	ID              int64       `json:"id"`
	Model           NeuronModel `json:"model"`
	Potential       float64     `json:"potential"`
	Threshold       float64     `json:"threshold"`
	RestPotential   float64     `json:"rest_potential"`
	TimeConstant    float64     `json:"time_constant"`
	RefractoryPeriod float64    `json:"refractory_period"`
	LastSpikeTime   float64     `json:"last_spike_time"`

	// Izhikevich parameters
	A, B, C, D      float64     `json:"a,b,c,d"`
	U               float64     `json:"u"` // recovery variable

	// Hodgkin-Huxley parameters
	GNa, GK, GL     float64     `json:"g_na,g_k,g_l"`
	M, N, H         float64     `json:"m,n,h"`
}

// Synapse represents a connection between neurons
type Synapse struct {
	ID           int64   `json:"id"`
	PreNeuronID  int64   `json:"pre_neuron_id"`
	PostNeuronID int64   `json:"post_neuron_id"`
	Weight       float64 `json:"weight"`
	Delay        float64 `json:"delay"`
	Plasticity   bool    `json:"plasticity"`

	// STDP parameters
	LastPreSpike  float64 `json:"last_pre_spike"`
	LastPostSpike float64 `json:"last_post_spike"`
}

// SNNNetwork represents a spiking neural network
type SNNNetwork struct {
	mu              sync.RWMutex
	Neurons         map[int64]*Neuron
	Synapses        map[int64]*Synapse
	SpikeQueue      []Spike
	CurrentTime     float64
	TimeStep        float64

	// Learning parameters
	EnableSTDP      bool
	STDPTauPlus     float64
	STDPTauMinus    float64
	STDPAPlus       float64
	STDPAMinus      float64

	// Homeostasis
	EnableHomeostasis bool
	TargetRate        float64
}

// Spike represents a spike event
type Spike struct {
	NeuronID  int64   `json:"neuron_id"`
	Timestamp float64 `json:"timestamp"`
	Weight    float64 `json:"weight"`
}

// STDPConfig defines STDP learning configuration
type STDPConfig struct {
	Enable     bool    `json:"enable"`
	TauPlus    float64 `json:"tau_plus"`   // ms
	TauMinus   float64 `json:"tau_minus"`  // ms
	APlus      float64 `json:"a_plus"`     // learning rate for potentiation
	AMinus     float64 `json:"a_minus"`    // learning rate for depression
}

// NewSNNNetwork creates a new spiking neural network
func NewSNNNetwork(timeStep float64, stdpConfig *STDPConfig) *SNNNetwork {
	snn := &SNNNetwork{
		Neurons:           make(map[int64]*Neuron),
		Synapses:          make(map[int64]*Synapse),
		SpikeQueue:        make([]Spike, 0),
		CurrentTime:       0.0,
		TimeStep:          timeStep,
		EnableSTDP:        stdpConfig.Enable,
		STDPTauPlus:       stdpConfig.TauPlus,
		STDPTauMinus:      stdpConfig.TauMinus,
		STDPAPlus:         stdpConfig.APlus,
		STDPAMinus:        stdpConfig.AMinus,
		EnableHomeostasis: false,
		TargetRate:        10.0, // 10 Hz
	}

	return snn
}

// AddNeuron adds a neuron to the network
func (snn *SNNNetwork) AddNeuron(model NeuronModel) int64 {
	snn.mu.Lock()
	defer snn.mu.Unlock()

	id := int64(len(snn.Neurons))

	neuron := &Neuron{
		ID:               id,
		Model:            model,
		Potential:        -65.0, // resting potential
		Threshold:        -50.0, // spike threshold
		RestPotential:    -65.0,
		TimeConstant:     20.0,  // ms
		RefractoryPeriod: 2.0,   // ms
		LastSpikeTime:    -1000.0,
	}

	// Set model-specific parameters
	switch model {
	case Izhikevich:
		// Regular spiking parameters
		neuron.A = 0.02
		neuron.B = 0.2
		neuron.C = -65.0
		neuron.D = 8.0
		neuron.U = neuron.B * neuron.Potential

	case HodgkinHuxley:
		neuron.GNa = 120.0 // mS/cm^2
		neuron.GK = 36.0
		neuron.GL = 0.3
		neuron.M = 0.05
		neuron.N = 0.32
		neuron.H = 0.6
	}

	snn.Neurons[id] = neuron
	return id
}

// AddSynapse adds a synapse between neurons
func (snn *SNNNetwork) AddSynapse(preID, postID int64, weight, delay float64) int64 {
	snn.mu.Lock()
	defer snn.mu.Unlock()

	id := int64(len(snn.Synapses))

	synapse := &Synapse{
		ID:           id,
		PreNeuronID:  preID,
		PostNeuronID: postID,
		Weight:       weight,
		Delay:        delay,
		Plasticity:   snn.EnableSTDP,
		LastPreSpike: -1000.0,
		LastPostSpike: -1000.0,
	}

	snn.Synapses[id] = synapse
	return id
}

// Step simulates one time step
func (snn *SNNNetwork) Step(ctx context.Context, inputSpikes []Spike) ([]Spike, error) {
	snn.mu.Lock()
	defer snn.mu.Unlock()

	// Add input spikes to queue
	snn.SpikeQueue = append(snn.SpikeQueue, inputSpikes...)

	// Process all neurons
	outputSpikes := make([]Spike, 0)

	for _, neuron := range snn.Neurons {
		// Skip if in refractory period
		if snn.CurrentTime-neuron.LastSpikeTime < neuron.RefractoryPeriod {
			continue
		}

		// Compute input current from synapses
		current := 0.0
		for _, syn := range snn.Synapses {
			if syn.PostNeuronID == neuron.ID {
				// Check for spike from pre-synaptic neuron
				for _, spike := range snn.SpikeQueue {
					if spike.NeuronID == syn.PreNeuronID {
						deliveryTime := spike.Timestamp + syn.Delay
						if math.Abs(deliveryTime-snn.CurrentTime) < snn.TimeStep {
							current += syn.Weight

							// Update STDP
							if syn.Plasticity {
								snn.updateSTDP(syn, spike.Timestamp, neuron.LastSpikeTime)
							}
						}
					}
				}
			}
		}

		// Update neuron dynamics
		spiked := false
		switch neuron.Model {
		case LIF:
			spiked = snn.updateLIF(neuron, current)
		case Izhikevich:
			spiked = snn.updateIzhikevich(neuron, current)
		case HodgkinHuxley:
			spiked = snn.updateHodgkinHuxley(neuron, current)
		}

		if spiked {
			spike := Spike{
				NeuronID:  neuron.ID,
				Timestamp: snn.CurrentTime,
				Weight:    1.0,
			}
			outputSpikes = append(outputSpikes, spike)
			neuron.LastSpikeTime = snn.CurrentTime

			// Update post-synaptic STDP
			if snn.EnableSTDP {
				for _, syn := range snn.Synapses {
					if syn.PostNeuronID == neuron.ID && syn.Plasticity {
						syn.LastPostSpike = snn.CurrentTime
					}
				}
			}
		}
	}

	// Clean old spikes from queue
	snn.cleanSpikeQueue()

	// Add output spikes to queue
	snn.SpikeQueue = append(snn.SpikeQueue, outputSpikes...)

	// Advance time
	snn.CurrentTime += snn.TimeStep

	return outputSpikes, nil
}

// updateLIF updates Leaky Integrate-and-Fire neuron
func (snn *SNNNetwork) updateLIF(neuron *Neuron, current float64) bool {
	// dV/dt = (-(V - V_rest) + R*I) / tau
	dv := (-(neuron.Potential - neuron.RestPotential) + current) / neuron.TimeConstant
	neuron.Potential += dv * snn.TimeStep

	// Check for spike
	if neuron.Potential >= neuron.Threshold {
		neuron.Potential = neuron.RestPotential
		return true
	}

	return false
}

// updateIzhikevich updates Izhikevich neuron model
func (snn *SNNNetwork) updateIzhikevich(neuron *Neuron, current float64) bool {
	// dv/dt = 0.04v^2 + 5v + 140 - u + I
	// du/dt = a(bv - u)

	v := neuron.Potential
	u := neuron.U

	dv := (0.04*v*v + 5*v + 140 - u + current) * snn.TimeStep
	du := neuron.A * (neuron.B*v - u) * snn.TimeStep

	neuron.Potential += dv
	neuron.U += du

	// Check for spike
	if neuron.Potential >= 30.0 {
		neuron.Potential = neuron.C
		neuron.U += neuron.D
		return true
	}

	return false
}

// updateHodgkinHuxley updates Hodgkin-Huxley model (simplified)
func (snn *SNNNetwork) updateHodgkinHuxley(neuron *Neuron, current float64) bool {
	v := neuron.Potential

	// Simplified H-H model
	alphaN := 0.01 * (v + 55) / (1 - math.Exp(-(v+55)/10))
	betaN := 0.125 * math.Exp(-(v + 65) / 80)

	dn := (alphaN*(1-neuron.N) - betaN*neuron.N) * snn.TimeStep
	neuron.N += dn

	// Current from ion channels
	iNa := neuron.GNa * math.Pow(neuron.M, 3) * neuron.H * (v - 50)
	iK := neuron.GK * math.Pow(neuron.N, 4) * (v + 77)
	iL := neuron.GL * (v + 54.387)

	dv := (current - iNa - iK - iL) * snn.TimeStep
	neuron.Potential += dv

	// Check for spike
	if neuron.Potential >= neuron.Threshold {
		neuron.Potential = neuron.RestPotential
		return true
	}

	return false
}

// updateSTDP updates synaptic weights using STDP
func (snn *SNNNetwork) updateSTDP(syn *Synapse, preTime, postTime float64) {
	dt := postTime - preTime

	if dt > 0 {
		// Post after pre: potentiation
		dw := snn.STDPAPlus * math.Exp(-dt/snn.STDPTauPlus)
		syn.Weight += dw
	} else {
		// Pre after post: depression
		dw := -snn.STDPAMinus * math.Exp(dt/snn.STDPTauMinus)
		syn.Weight += dw
	}

	// Clamp weights
	if syn.Weight > 10.0 {
		syn.Weight = 10.0
	} else if syn.Weight < 0.0 {
		syn.Weight = 0.0
	}
}

// cleanSpikeQueue removes old spikes from queue
func (snn *SNNNetwork) cleanSpikeQueue() {
	maxDelay := 10.0 // ms
	cutoff := snn.CurrentTime - maxDelay

	newQueue := make([]Spike, 0)
	for _, spike := range snn.SpikeQueue {
		if spike.Timestamp >= cutoff {
			newQueue = append(newQueue, spike)
		}
	}

	snn.SpikeQueue = newQueue
}

// Run runs the SNN for a specified duration
func (snn *SNNNetwork) Run(ctx context.Context, duration float64, inputFunc func(float64) []Spike) ([]Spike, error) {
	allSpikes := make([]Spike, 0)
	steps := int(duration / snn.TimeStep)

	for i := 0; i < steps; i++ {
		// Get input spikes for this time step
		var inputSpikes []Spike
		if inputFunc != nil {
			inputSpikes = inputFunc(snn.CurrentTime)
		}

		// Simulate one step
		outputSpikes, err := snn.Step(ctx, inputSpikes)
		if err != nil {
			return nil, err
		}

		allSpikes = append(allSpikes, outputSpikes...)

		// Check context cancellation
		select {
		case <-ctx.Done():
			return allSpikes, ctx.Err()
		default:
		}
	}

	return allSpikes, nil
}

// GetSpikeRate calculates the average spike rate
func (snn *SNNNetwork) GetSpikeRate(neuronID int64, window float64) float64 {
	snn.mu.RLock()
	defer snn.mu.RUnlock()

	count := 0
	cutoff := snn.CurrentTime - window

	for _, spike := range snn.SpikeQueue {
		if spike.NeuronID == neuronID && spike.Timestamp >= cutoff {
			count++
		}
	}

	return float64(count) / (window / 1000.0) // Convert to Hz
}

// Reset resets the network to initial state
func (snn *SNNNetwork) Reset() {
	snn.mu.Lock()
	defer snn.mu.Unlock()

	snn.CurrentTime = 0.0
	snn.SpikeQueue = make([]Spike, 0)

	for _, neuron := range snn.Neurons {
		neuron.Potential = neuron.RestPotential
		neuron.LastSpikeTime = -1000.0

		if neuron.Model == Izhikevich {
			neuron.U = neuron.B * neuron.Potential
		}
	}
}

// GetMetrics returns network metrics
func (snn *SNNNetwork) GetMetrics() map[string]interface{} {
	snn.mu.RLock()
	defer snn.mu.RUnlock()

	totalSpikes := len(snn.SpikeQueue)
	avgSpikeRate := 0.0
	if snn.CurrentTime > 0 {
		avgSpikeRate = float64(totalSpikes) / (snn.CurrentTime / 1000.0) / float64(len(snn.Neurons))
	}

	return map[string]interface{}{
		"neurons":         len(snn.Neurons),
		"synapses":        len(snn.Synapses),
		"current_time":    snn.CurrentTime,
		"total_spikes":    totalSpikes,
		"avg_spike_rate":  avgSpikeRate,
		"enable_stdp":     snn.EnableSTDP,
	}
}
