package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/partition"
)

// NetworkSimulator simulates DWCP network for RL agent training
type NetworkSimulator struct {
	baseLatency    float64
	baseBandwidth  float64
	congestionProb float64
	currentState   *partition.EnvironmentState
	rewardCalc     *partition.RewardCalculator
	episodeCount   int
	totalSteps     int
}

// NewNetworkSimulator creates a new network simulator
func NewNetworkSimulator() *NetworkSimulator {
	rand.Seed(time.Now().UnixNano())

	return &NetworkSimulator{
		baseLatency:    10.0,  // 10ms base latency
		baseBandwidth:  100.0, // 100 Mbps base bandwidth
		congestionProb: 0.1,   // 10% chance of congestion events
		rewardCalc:     partition.NewRewardCalculator(),
		episodeCount:   0,
		totalSteps:     0,
	}
}

// TrainingMetrics tracks training progress
type TrainingMetrics struct {
	Episode        int
	TotalReward    float64
	AvgReward      float64
	Steps          int
	Epsilon        float64
	BufferSize     int
	SuccessRate    float64
	AvgThroughput  float64
	AvgLatency     float64
	Timestamp      time.Time
}

// SimulateEpisode runs a full training episode
func (sim *NetworkSimulator) SimulateEpisode(agent *partition.DQNAgent, maxSteps int) (*TrainingMetrics, error) {
	// Reset environment
	state := sim.resetEnvironment()

	totalReward := 0.0
	steps := 0
	successCount := 0
	totalThroughput := 0.0
	totalLatency := 0.0

	for steps < maxSteps {
		// Agent selects action
		decision, err := agent.SelectAction(state)
		if err != nil {
			return nil, fmt.Errorf("failed to select action: %w", err)
		}

		// Execute action in environment
		nextState, reward, done := sim.step(state, decision)

		// Store experience in agent's memory
		agent.Remember(state, decision.Action, reward, nextState, done)

		// Update metrics
		totalReward += reward
		if reward > 0 {
			successCount++
		}

		// Track performance
		outcome := sim.calculateOutcome(decision, state)
		totalThroughput += outcome.ActualThroughput
		totalLatency += outcome.ActualLatency

		// Move to next state
		state = nextState
		steps++
		sim.totalSteps++

		if done {
			break
		}

		// Train agent periodically
		if agent.ReplayBuffer.Size() > 32 {
			// Trigger replay training (would be done in Python)
			// For Go, we just collect experiences
		}
	}

	// Update epsilon after episode
	agent.UpdateEpsilon()

	sim.episodeCount++

	metrics := &TrainingMetrics{
		Episode:       sim.episodeCount,
		TotalReward:   totalReward,
		AvgReward:     totalReward / float64(steps),
		Steps:         steps,
		Epsilon:       agent.Epsilon,
		BufferSize:    agent.ReplayBuffer.Size(),
		SuccessRate:   float64(successCount) / float64(steps),
		AvgThroughput: totalThroughput / float64(steps),
		AvgLatency:    totalLatency / float64(steps),
		Timestamp:     time.Now(),
	}

	return metrics, nil
}

func (sim *NetworkSimulator) resetEnvironment() *partition.EnvironmentState {
	state := partition.NewEnvironmentState()

	// Add randomness to create diverse training scenarios
	for i := 0; i < 4; i++ {
		state.StreamBandwidth[i] = sim.baseBandwidth * (0.7 + rand.Float64()*0.6)
		state.StreamLatency[i] = sim.baseLatency * (0.7 + rand.Float64()*0.6)
		state.StreamCongestion[i] = rand.Float64() * 0.4
		state.StreamSuccessRate[i] = 0.85 + rand.Float64()*0.14
	}

	state.TaskQueueDepth = rand.Intn(50) + 1
	state.TaskSize = int(rand.Float64() * 1e9)
	state.TaskPriority = rand.Float64()
	state.TimeOfDay = float64(time.Now().Hour()) / 24.0

	sim.currentState = state
	return state
}

func (sim *NetworkSimulator) step(state *partition.EnvironmentState, decision *partition.TaskPartitionDecision) (*partition.EnvironmentState, float64, bool) {
	// Calculate outcome of action
	outcome := sim.calculateOutcome(decision, state)

	// Calculate reward
	reward := sim.rewardCalc.Calculate(outcome)

	// Update environment state
	nextState := sim.updateState(state, decision, outcome)

	// Check if episode is done
	done := nextState.TaskQueueDepth == 0

	return nextState, reward, done
}

func (sim *NetworkSimulator) calculateOutcome(decision *partition.TaskPartitionDecision, state *partition.EnvironmentState) *partition.ActionOutcome {
	outcome := &partition.ActionOutcome{
		BaselineThroughput: sim.baseBandwidth,
		TargetLatency:      sim.baseLatency,
	}

	numStreams := len(decision.StreamIDs)
	if numStreams == 0 {
		return outcome
	}

	// Calculate actual throughput and latency
	totalThroughput := 0.0
	maxLatency := 0.0

	for i, streamID := range decision.StreamIDs {
		if streamID >= 4 {
			continue
		}

		// Throughput proportional to chunk size and stream capacity
		chunkProportion := float64(decision.ChunkSizes[i]) / float64(state.TaskSize)
		streamThroughput := state.StreamBandwidth[streamID] * state.StreamSuccessRate[streamID] * chunkProportion

		// Add variance
		streamThroughput *= (0.9 + rand.Float64()*0.2)

		totalThroughput += streamThroughput

		// Latency is max of all streams (parallel transfer)
		streamLatency := state.StreamLatency[streamID] * (1 + state.StreamCongestion[streamID])
		streamLatency *= (0.9 + rand.Float64()*0.2) // Add variance

		if streamLatency > maxLatency {
			maxLatency = streamLatency
		}
	}

	outcome.ActualThroughput = totalThroughput
	outcome.ActualLatency = maxLatency

	// Calculate stream imbalance
	if numStreams > 1 {
		loads := make([]float64, numStreams)
		for i, streamID := range decision.StreamIDs {
			loads[i] = state.StreamCongestion[streamID]
		}
		outcome.StreamImbalance = calculateStdDev(loads)
	}

	// Simulate completion and retransmissions
	successProb := 1.0
	for _, streamID := range decision.StreamIDs {
		successProb *= state.StreamSuccessRate[streamID]
	}

	outcome.Completed = rand.Float64() < successProb
	if !outcome.Completed {
		outcome.Retransmissions = rand.Intn(3) + 1
	}

	return outcome
}

func (sim *NetworkSimulator) updateState(state *partition.EnvironmentState, decision *partition.TaskPartitionDecision, outcome *partition.ActionOutcome) *partition.EnvironmentState {
	nextState := *state

	// Update stream congestion based on usage
	for _, streamID := range decision.StreamIDs {
		if streamID >= 4 {
			continue
		}

		// Increase congestion on used streams
		nextState.StreamCongestion[streamID] = math.Min(1.0, nextState.StreamCongestion[streamID]+0.05)

		// Update success rate based on outcome
		if outcome.Completed {
			nextState.StreamSuccessRate[streamID] = 0.95*nextState.StreamSuccessRate[streamID] + 0.05
		} else {
			nextState.StreamSuccessRate[streamID] = 0.95*nextState.StreamSuccessRate[streamID] + 0.025
		}
	}

	// Decay congestion on unused streams
	for i := 0; i < 4; i++ {
		isUsed := false
		for _, streamID := range decision.StreamIDs {
			if streamID == i {
				isUsed = true
				break
			}
		}

		if !isUsed {
			nextState.StreamCongestion[i] = math.Max(0, nextState.StreamCongestion[i]-0.03)
		}
	}

	// Random network fluctuations
	for i := 0; i < 4; i++ {
		nextState.StreamBandwidth[i] += (rand.Float64()*10 - 5)
		nextState.StreamBandwidth[i] = math.Max(10, math.Min(200, nextState.StreamBandwidth[i]))

		nextState.StreamLatency[i] += (rand.Float64()*2 - 1)
		nextState.StreamLatency[i] = math.Max(1, math.Min(50, nextState.StreamLatency[i]))
	}

	// Random congestion events
	if rand.Float64() < sim.congestionProb {
		affectedStream := rand.Intn(4)
		nextState.StreamCongestion[affectedStream] = math.Min(1.0, nextState.StreamCongestion[affectedStream]+0.2)
	}

	// Update task queue
	if outcome.Completed && nextState.TaskQueueDepth > 0 {
		nextState.TaskQueueDepth--

		// Generate new task
		if nextState.TaskQueueDepth > 0 {
			nextState.TaskSize = int(rand.Float64() * 1e9)
			nextState.TaskPriority = rand.Float64()
		}
	}

	// Update time
	nextState.TimeOfDay = float64(time.Now().Hour()) / 24.0

	return &nextState
}

// RunTraining runs full training loop
func (sim *NetworkSimulator) RunTraining(agent *partition.DQNAgent, episodes int, maxStepsPerEpisode int) error {
	log.Printf("Starting training for %d episodes...", episodes)

	allMetrics := make([]*TrainingMetrics, 0, episodes)
	exportInterval := 100 // Export experiences every 100 episodes

	for ep := 0; ep < episodes; ep++ {
		metrics, err := sim.SimulateEpisode(agent, maxStepsPerEpisode)
		if err != nil {
			return fmt.Errorf("episode %d failed: %w", ep, err)
		}

		allMetrics = append(allMetrics, metrics)

		// Log progress
		if ep%10 == 0 {
			log.Printf("Episode %d/%d - Reward: %.2f, Avg: %.2f, Steps: %d, Epsilon: %.3f, Success: %.2f%%",
				metrics.Episode, episodes, metrics.TotalReward, metrics.AvgReward,
				metrics.Steps, metrics.Epsilon, metrics.SuccessRate*100)
		}

		// Export experiences for Python training
		if ep%exportInterval == 0 && ep > 0 {
			if err := sim.exportExperiences(agent, fmt.Sprintf("experiences_ep%d.json", ep)); err != nil {
				log.Printf("Warning: Failed to export experiences: %v", err)
			}
		}
	}

	// Save final training metrics
	if err := sim.saveMetrics(allMetrics, "training_metrics.json"); err != nil {
		return fmt.Errorf("failed to save metrics: %w", err)
	}

	log.Printf("Training completed successfully!")
	return nil
}

func (sim *NetworkSimulator) exportExperiences(agent *partition.DQNAgent, filename string) error {
	// Sample experiences from replay buffer
	experiences := agent.ReplayBuffer.Sample(1000)

	// Convert to JSON-serializable format
	type ExperienceJSON struct {
		State     []float32 `json:"state"`
		Action    int       `json:"action"`
		Reward    float64   `json:"reward"`
		NextState []float32 `json:"next_state"`
		Done      bool      `json:"done"`
	}

	jsonExps := make([]ExperienceJSON, len(experiences))
	for i, exp := range experiences {
		jsonExps[i] = ExperienceJSON{
			State:     exp.State,
			Action:    int(exp.Action),
			Reward:    exp.Reward,
			NextState: exp.NextState,
			Done:      exp.Done,
		}
	}

	data, err := json.MarshalIndent(jsonExps, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

func (sim *NetworkSimulator) saveMetrics(metrics []*TrainingMetrics, filename string) error {
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

func calculateStdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return math.Sqrt(variance)
}

func main() {
	// Example usage
	sim := NewNetworkSimulator()

	agent, err := partition.NewDQNAgent("models/dqn_v1.onnx")
	if err != nil {
		log.Printf("Warning: Could not load model: %v", err)
		log.Printf("Agent will operate in exploration mode")
	}

	// Run training
	if err := sim.RunTraining(agent, 1000, 100); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
}