package partition

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"sync"
	"time"
)

// OnlineLearner manages continuous learning and model updates
type OnlineLearner struct {
	agent            *DQNAgent
	replayBuffer     *ReplayBuffer
	updateFrequency  time.Duration
	lastUpdate       time.Time
	experienceCount  int
	minExperiences   int
	trainingScript   string
	modelPath        string
	isTraining       bool
	mu               sync.RWMutex

	// Metrics
	updateCount      int
	avgReward        float64
	performanceGain  float64
}

// OnlineLearnerConfig configures the online learning system
type OnlineLearnerConfig struct {
	UpdateFrequency  time.Duration
	MinExperiences   int
	TrainingScript   string
	ModelPath        string
	EnableAutoUpdate bool
}

// NewOnlineLearner creates a new online learning system
func NewOnlineLearner(agent *DQNAgent, config *OnlineLearnerConfig) *OnlineLearner {
	if config == nil {
		config = &OnlineLearnerConfig{
			UpdateFrequency:  24 * time.Hour,    // Update daily
			MinExperiences:   1000,              // Minimum experiences before update
			TrainingScript:   "training/train_dqn.py",
			ModelPath:        "models/dqn_online",
			EnableAutoUpdate: true,
		}
	}

	learner := &OnlineLearner{
		agent:           agent,
		replayBuffer:    agent.replayBuffer,
		updateFrequency: config.UpdateFrequency,
		lastUpdate:      time.Now(),
		minExperiences:  config.MinExperiences,
		trainingScript:  config.TrainingScript,
		modelPath:       config.ModelPath,
		isTraining:      false,
		updateCount:     0,
	}

	if config.EnableAutoUpdate {
		go learner.autoUpdateLoop()
	}

	return learner
}

// CollectExperience adds a new experience to the learning system
func (ol *OnlineLearner) CollectExperience(state *EnvironmentState, action Action, reward float64, nextState *EnvironmentState, done bool) {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	// Store in agent's memory
	ol.agent.Remember(state, action, reward, nextState, done)

	ol.experienceCount++
	ol.avgReward = 0.9*ol.avgReward + 0.1*reward

	// Check if it's time to update the model
	if ol.shouldUpdate() {
		go ol.triggerUpdate()
	}
}

// shouldUpdate determines if model update should be triggered
func (ol *OnlineLearner) shouldUpdate() bool {
	if ol.isTraining {
		return false
	}

	// Check if enough experiences collected
	if ol.replayBuffer.Size() < ol.minExperiences {
		return false
	}

	// Check if enough time has passed
	if time.Since(ol.lastUpdate) < ol.updateFrequency {
		return false
	}

	return true
}

// triggerUpdate initiates the model update process
func (ol *OnlineLearner) triggerUpdate() {
	ol.mu.Lock()
	if ol.isTraining {
		ol.mu.Unlock()
		return
	}
	ol.isTraining = true
	ol.mu.Unlock()

	defer func() {
		ol.mu.Lock()
		ol.isTraining = false
		ol.mu.Unlock()
	}()

	log.Printf("Starting online model update (experiences: %d)...", ol.experienceCount)

	// Export current experiences
	exportPath := fmt.Sprintf("%s_experiences.json", ol.modelPath)
	if err := ol.exportExperiences(exportPath); err != nil {
		log.Printf("Failed to export experiences: %v", err)
		return
	}

	// Trigger retraining
	if err := ol.runTraining(); err != nil {
		log.Printf("Training failed: %v", err)
		return
	}

	// Load updated model
	if err := ol.loadUpdatedModel(); err != nil {
		log.Printf("Failed to load updated model: %v", err)
		return
	}

	ol.mu.Lock()
	ol.lastUpdate = time.Now()
	ol.updateCount++
	ol.mu.Unlock()

	log.Printf("Online model update completed successfully (update #%d)", ol.updateCount)
}

// exportExperiences exports replay buffer to JSON for Python training
func (ol *OnlineLearner) exportExperiences(path string) error {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	// Sample experiences from buffer
	batchSize := min(ol.replayBuffer.Size(), 5000)
	experiences := ol.replayBuffer.Sample(batchSize)

	// Convert to JSON format
	type ExperienceData struct {
		State     []float32 `json:"state"`
		Action    int       `json:"action"`
		Reward    float64   `json:"reward"`
		NextState []float32 `json:"next_state"`
		Done      bool      `json:"done"`
		TDError   float64   `json:"td_error"`
	}

	data := make([]ExperienceData, len(experiences))
	for i, exp := range experiences {
		data[i] = ExperienceData{
			State:     exp.State,
			Action:    int(exp.Action),
			Reward:    exp.Reward,
			NextState: exp.NextState,
			Done:      exp.Done,
			TDError:   exp.TDError,
		}
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal experiences: %w", err)
	}

	if err := ioutil.WriteFile(path, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write experiences: %w", err)
	}

	log.Printf("Exported %d experiences to %s", len(experiences), path)
	return nil
}

// runTraining executes the Python training script
func (ol *OnlineLearner) runTraining() error {
	// Prepare training command
	cmd := exec.Command("python3", ol.trainingScript,
		"--load-experiences", fmt.Sprintf("%s_experiences.json", ol.modelPath),
		"--output-model", ol.modelPath,
		"--episodes", "100", // Quick online update
		"--batch-size", "64",
	)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	log.Printf("Running training: %s", cmd.String())

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("training command failed: %w", err)
	}

	return nil
}

// loadUpdatedModel loads the newly trained model
func (ol *OnlineLearner) loadUpdatedModel() error {
	modelFile := fmt.Sprintf("%s.onnx", ol.modelPath)

	// Check if model file exists
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", modelFile)
	}

	// Load model
	if err := ol.agent.LoadModel(modelFile); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	// Reduce exploration rate since we have a better model
	ol.agent.UpdateEpsilon()

	log.Printf("Loaded updated model from %s", modelFile)
	return nil
}

// autoUpdateLoop periodically checks for model updates
func (ol *OnlineLearner) autoUpdateLoop() {
	ticker := time.NewTicker(1 * time.Hour) // Check every hour
	defer ticker.Stop()

	for range ticker.C {
		ol.mu.RLock()
		shouldUpdate := ol.shouldUpdate()
		ol.mu.RUnlock()

		if shouldUpdate {
			ol.triggerUpdate()
		}
	}
}

// GetStatus returns the current status of the online learner
func (ol *OnlineLearner) GetStatus() map[string]interface{} {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	return map[string]interface{}{
		"is_training":       ol.isTraining,
		"update_count":      ol.updateCount,
		"experience_count":  ol.experienceCount,
		"buffer_size":       ol.replayBuffer.Size(),
		"avg_reward":        ol.avgReward,
		"last_update":       ol.lastUpdate,
		"time_until_update": ol.updateFrequency - time.Since(ol.lastUpdate),
		"min_experiences":   ol.minExperiences,
	}
}

// ForceUpdate forces an immediate model update
func (ol *OnlineLearner) ForceUpdate() error {
	ol.mu.RLock()
	if ol.isTraining {
		ol.mu.RUnlock()
		return fmt.Errorf("training already in progress")
	}

	if ol.replayBuffer.Size() < ol.minExperiences/2 {
		ol.mu.RUnlock()
		return fmt.Errorf("insufficient experiences: %d < %d", ol.replayBuffer.Size(), ol.minExperiences/2)
	}
	ol.mu.RUnlock()

	go ol.triggerUpdate()
	return nil
}

// EvaluateModel evaluates the current model performance
func (ol *OnlineLearner) EvaluateModel(episodes int) (*EvaluationResults, error) {
	log.Printf("Evaluating model over %d episodes...", episodes)

	results := &EvaluationResults{
		Episodes:    episodes,
		Rewards:     make([]float64, 0, episodes),
		Throughputs: make([]float64, 0, episodes),
		Latencies:   make([]float64, 0, episodes),
	}

	// Create test environment
	env := NewEnvironmentSimulator()

	// Disable exploration for evaluation
	originalEpsilon := ol.agent.epsilon
	ol.agent.mu.Lock()
	ol.agent.epsilon = 0
	ol.agent.mu.Unlock()

	defer func() {
		ol.agent.mu.Lock()
		ol.agent.epsilon = originalEpsilon
		ol.agent.mu.Unlock()
	}()

	for ep := 0; ep < episodes; ep++ {
		state := env.Reset()
		totalReward := 0.0
		totalThroughput := 0.0
		totalLatency := 0.0
		steps := 0

		for steps < 100 {
			decision, err := ol.agent.SelectAction(state)
			if err != nil {
				return nil, err
			}

			nextState, reward, done := env.Step(decision.Action)

			totalReward += reward
			totalThroughput += float64(state.TaskSize) / decision.ExpectedTime.Seconds()
			totalLatency += decision.ExpectedTime.Seconds()

			state = nextState
			steps++

			if done {
				break
			}
		}

		results.Rewards = append(results.Rewards, totalReward)
		results.Throughputs = append(results.Throughputs, totalThroughput/float64(steps))
		results.Latencies = append(results.Latencies, totalLatency/float64(steps))
	}

	// Calculate statistics
	results.calculateStats()

	log.Printf("Evaluation complete: Mean Reward = %.2f, Mean Throughput = %.2f Mbps",
		results.MeanReward, results.MeanThroughput)

	return results, nil
}

// EvaluationResults stores evaluation metrics
type EvaluationResults struct {
	Episodes        int
	Rewards         []float64
	Throughputs     []float64
	Latencies       []float64
	MeanReward      float64
	StdReward       float64
	MeanThroughput  float64
	StdThroughput   float64
	MeanLatency     float64
	StdLatency      float64
	SuccessRate     float64
}

func (er *EvaluationResults) calculateStats() {
	er.MeanReward = mean(er.Rewards)
	er.StdReward = stddev(er.Rewards, er.MeanReward)

	er.MeanThroughput = mean(er.Throughputs)
	er.StdThroughput = stddev(er.Throughputs, er.MeanThroughput)

	er.MeanLatency = mean(er.Latencies)
	er.StdLatency = stddev(er.Latencies, er.MeanLatency)

	// Count successful episodes (positive reward)
	successCount := 0
	for _, r := range er.Rewards {
		if r > 0 {
			successCount++
		}
	}
	er.SuccessRate = float64(successCount) / float64(len(er.Rewards))
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func stddev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	return variance
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}