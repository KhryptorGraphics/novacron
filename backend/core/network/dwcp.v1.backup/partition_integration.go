package dwcp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/partition"
	"go.uber.org/zap"
)

// TaskPartitioner provides intelligent task partitioning using Deep RL
type TaskPartitioner struct {
	agent          *partition.DQNAgent
	onlineLearner  *partition.OnlineLearner
	envState       *partition.EnvironmentState
	logger         *zap.Logger
	enabled        bool
	mu             sync.RWMutex

	// Metrics
	totalDecisions  int64
	successfulTasks int64
	failedTasks     int64
	avgReward       float64
}

// NewTaskPartitioner creates a new intelligent task partitioner
func NewTaskPartitioner(modelPath string, logger *zap.Logger) (*TaskPartitioner, error) {
	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	// Initialize DQN agent
	agent, err := partition.NewDQNAgent(modelPath)
	if err != nil {
		logger.Warn("Failed to initialize DQN agent, using heuristic mode",
			zap.Error(err),
			zap.String("model_path", modelPath))
		// Continue with heuristic mode
	}

	// Initialize online learner
	learnerConfig := &partition.OnlineLearnerConfig{
		UpdateFrequency:  24 * time.Hour,
		MinExperiences:   1000,
		TrainingScript:   "/home/kp/novacron/backend/core/network/dwcp/partition/training/train_dqn.py",
		ModelPath:        modelPath,
		EnableAutoUpdate: true,
	}

	onlineLearner := partition.NewOnlineLearner(agent, learnerConfig)

	tp := &TaskPartitioner{
		agent:         agent,
		onlineLearner: onlineLearner,
		envState:      partition.NewEnvironmentState(),
		logger:        logger,
		enabled:       true,
	}

	logger.Info("Task partitioner initialized",
		zap.String("model_path", modelPath),
		zap.Bool("online_learning", true))

	return tp, nil
}

// Task represents a workload that needs partitioning
type Task struct {
	ID           string
	Size         int
	Priority     float64
	Deadline     time.Time
	RequiresFEC  bool
	MinBandwidth float64
}

// PartitionTask makes an intelligent decision on how to partition a task
func (tp *TaskPartitioner) PartitionTask(task *Task) (*partition.TaskPartitionDecision, error) {
	tp.mu.Lock()
	defer tp.mu.Unlock()

	if !tp.enabled {
		// Fallback to simple round-robin
		return tp.simplePartition(task), nil
	}

	// Update environment state with current network conditions
	tp.updateEnvironmentState(task)

	// Get RL agent decision
	decision, err := tp.agent.SelectAction(tp.envState)
	if err != nil {
		tp.logger.Error("Failed to get partition decision, falling back to heuristic",
			zap.Error(err),
			zap.String("task_id", task.ID))
		return tp.simplePartition(task), nil
	}

	tp.totalDecisions++

	tp.logger.Debug("Partition decision made",
		zap.String("task_id", task.ID),
		zap.Int("task_size", task.Size),
		zap.Ints("streams", decision.StreamIDs),
		zap.Ints("chunk_sizes", decision.ChunkSizes),
		zap.Float64("confidence", decision.Confidence),
		zap.Duration("expected_time", decision.ExpectedTime),
		zap.Bool("exploration", decision.ExplorationUsed))

	return decision, nil
}

// ReportOutcome reports the outcome of a task execution for learning
func (tp *TaskPartitioner) ReportOutcome(taskID string, decision *partition.TaskPartitionDecision,
	actualThroughput float64, actualLatency time.Duration, success bool) {

	tp.mu.Lock()
	defer tp.mu.Unlock()

	if success {
		tp.successfulTasks++
	} else {
		tp.failedTasks++
	}

	// Calculate reward
	outcome := &partition.ActionOutcome{
		ActualThroughput:   actualThroughput,
		BaselineThroughput: 100.0, // Base expectation
		ActualLatency:      actualLatency.Seconds() * 1000, // Convert to ms
		TargetLatency:      decision.ExpectedTime.Seconds() * 1000,
		StreamImbalance:    tp.calculateStreamImbalance(decision.StreamIDs),
		Completed:          success,
		Retransmissions:    0, // Would be tracked by transport layer
	}

	rewardCalc := partition.NewRewardCalculator()
	reward := rewardCalc.Calculate(outcome)

	// Update moving average
	tp.avgReward = 0.95*tp.avgReward + 0.05*reward

	// Create next state (current state after task execution)
	nextState := tp.createNextState(decision, success)

	// Collect experience for online learning
	tp.onlineLearner.CollectExperience(
		tp.envState,
		decision.Action,
		reward,
		nextState,
		false, // Episodes don't "end" in production
	)

	tp.logger.Debug("Task outcome reported",
		zap.String("task_id", taskID),
		zap.Float64("reward", reward),
		zap.Float64("avg_reward", tp.avgReward),
		zap.Bool("success", success),
		zap.Float64("throughput", actualThroughput),
		zap.Duration("latency", actualLatency))

	// Update current state for next decision
	tp.envState = nextState
}

// updateEnvironmentState updates the environment state based on current network conditions
func (tp *TaskPartitioner) updateEnvironmentState(task *Task) {
	// In production, these would come from actual network metrics
	// For now, we maintain the state from the last update

	tp.envState.TaskSize = task.Size
	tp.envState.TaskPriority = task.Priority
	tp.envState.TimeOfDay = float64(time.Now().Hour()) / 24.0

	// Task queue depth would be tracked by the scheduler
	// tp.envState.TaskQueueDepth = scheduler.GetQueueDepth()
}

// createNextState creates the next state after task execution
func (tp *TaskPartitioner) createNextState(decision *partition.TaskPartitionDecision, success bool) *partition.EnvironmentState {
	nextState := *tp.envState

	// Update stream metrics based on usage
	for _, streamID := range decision.StreamIDs {
		if streamID >= 4 {
			continue
		}

		// Increase congestion on used streams
		if nextState.StreamCongestion[streamID] < 1.0 {
			nextState.StreamCongestion[streamID] += 0.05
		}

		// Update success rate
		if success {
			nextState.StreamSuccessRate[streamID] = 0.95*nextState.StreamSuccessRate[streamID] + 0.05
		} else {
			nextState.StreamSuccessRate[streamID] = 0.95*nextState.StreamSuccessRate[streamID] + 0.025
		}
	}

	// Decay congestion on unused streams
	for i := 0; i < 4; i++ {
		used := false
		for _, sid := range decision.StreamIDs {
			if sid == i {
				used = true
				break
			}
		}
		if !used && nextState.StreamCongestion[i] > 0 {
			nextState.StreamCongestion[i] -= 0.03
			if nextState.StreamCongestion[i] < 0 {
				nextState.StreamCongestion[i] = 0
			}
		}
	}

	return &nextState
}

// calculateStreamImbalance calculates load imbalance across streams
func (tp *TaskPartitioner) calculateStreamImbalance(streamIDs []int) float64 {
	if len(streamIDs) <= 1 {
		return 0
	}

	var loads []float64
	for _, sid := range streamIDs {
		if sid < 4 {
			loads = append(loads, tp.envState.StreamCongestion[sid])
		}
	}

	if len(loads) == 0 {
		return 0
	}

	// Calculate standard deviation
	mean := 0.0
	for _, l := range loads {
		mean += l
	}
	mean /= float64(len(loads))

	variance := 0.0
	for _, l := range loads {
		diff := l - mean
		variance += diff * diff
	}
	variance /= float64(len(loads))

	return variance // Return variance (sqrt would be stddev)
}

// simplePartition provides a fallback partitioning strategy
func (tp *TaskPartitioner) simplePartition(task *Task) *partition.TaskPartitionDecision {
	// Find the best stream based on current state
	bestStream := 0
	bestScore := 0.0

	for i := 0; i < 4; i++ {
		score := tp.envState.StreamBandwidth[i] * tp.envState.StreamSuccessRate[i] /
			(tp.envState.StreamLatency[i] * (1 + tp.envState.StreamCongestion[i]))

		if score > bestScore {
			bestScore = score
			bestStream = i
		}
	}

	// For large tasks, consider splitting
	if task.Size > 100*1024*1024 { // > 100 MB
		// Use top 2 streams
		streams := tp.getTopNStreams(2)
		chunkSizes := make([]int, len(streams))
		for i := range chunkSizes {
			if i == len(chunkSizes)-1 {
				// Last stream gets remainder
				chunkSizes[i] = task.Size - (task.Size/len(streams))*(len(streams)-1)
			} else {
				chunkSizes[i] = task.Size / len(streams)
			}
		}

		return &partition.TaskPartitionDecision{
			StreamIDs:    streams,
			ChunkSizes:   chunkSizes,
			Confidence:   0.5, // Lower confidence for heuristic
			ExpectedTime: tp.estimateTime(task.Size, streams),
		}
	}

	// Single stream for small tasks
	return &partition.TaskPartitionDecision{
		StreamIDs:    []int{bestStream},
		ChunkSizes:   []int{task.Size},
		Confidence:   0.5,
		ExpectedTime: tp.estimateTime(task.Size, []int{bestStream}),
	}
}

// getTopNStreams returns the N best streams based on current metrics
func (tp *TaskPartitioner) getTopNStreams(n int) []int {
	type streamScore struct {
		id    int
		score float64
	}

	scores := make([]streamScore, 4)
	for i := 0; i < 4; i++ {
		score := tp.envState.StreamBandwidth[i] * tp.envState.StreamSuccessRate[i] /
			(tp.envState.StreamLatency[i] * (1 + tp.envState.StreamCongestion[i]))
		scores[i] = streamScore{id: i, score: score}
	}

	// Simple selection sort for top N
	for i := 0; i < n && i < 4; i++ {
		maxIdx := i
		for j := i + 1; j < 4; j++ {
			if scores[j].score > scores[maxIdx].score {
				maxIdx = j
			}
		}
		scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
	}

	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = scores[i].id
	}

	return result
}

// estimateTime estimates completion time for a partitioning decision
func (tp *TaskPartitioner) estimateTime(taskSize int, streams []int) time.Duration {
	maxTime := 0.0

	for _, streamID := range streams {
		if streamID >= 4 {
			continue
		}

		chunkSize := taskSize / len(streams)
		bandwidth := tp.envState.StreamBandwidth[streamID] * 1e6 / 8 // Mbps to bytes/s
		latency := tp.envState.StreamLatency[streamID] / 1000        // ms to seconds

		streamTime := latency + (float64(chunkSize)/bandwidth)*(1+tp.envState.StreamCongestion[streamID])
		streamTime *= (2 - tp.envState.StreamSuccessRate[streamID]) // Account for retransmissions

		if streamTime > maxTime {
			maxTime = streamTime
		}
	}

	return time.Duration(maxTime * float64(time.Second))
}

// UpdateNetworkMetrics updates the environment state with fresh network metrics
func (tp *TaskPartitioner) UpdateNetworkMetrics(streamID int, bandwidth, latency, congestion, successRate float64) {
	tp.mu.Lock()
	defer tp.mu.Unlock()

	if streamID < 0 || streamID >= 4 {
		return
	}

	tp.envState.StreamBandwidth[streamID] = bandwidth
	tp.envState.StreamLatency[streamID] = latency
	tp.envState.StreamCongestion[streamID] = congestion
	tp.envState.StreamSuccessRate[streamID] = successRate
}

// GetMetrics returns partitioner metrics
func (tp *TaskPartitioner) GetMetrics() map[string]interface{} {
	tp.mu.RLock()
	defer tp.mu.RUnlock()

	successRate := 0.0
	if tp.totalDecisions > 0 {
		successRate = float64(tp.successfulTasks) / float64(tp.totalDecisions)
	}

	metrics := map[string]interface{}{
		"enabled":          tp.enabled,
		"total_decisions":  tp.totalDecisions,
		"successful_tasks": tp.successfulTasks,
		"failed_tasks":     tp.failedTasks,
		"success_rate":     successRate,
		"avg_reward":       tp.avgReward,
	}

	// Add agent metrics
	agentMetrics := tp.agent.GetMetrics()
	for k, v := range agentMetrics {
		metrics["agent_"+k] = v
	}

	// Add online learner status
	learnerStatus := tp.onlineLearner.GetStatus()
	for k, v := range learnerStatus {
		metrics["learner_"+k] = v
	}

	return metrics
}

// ForceModelUpdate triggers an immediate model update
func (tp *TaskPartitioner) ForceModelUpdate() error {
	return tp.onlineLearner.ForceUpdate()
}

// Evaluate evaluates the current model performance
func (tp *TaskPartitioner) Evaluate(episodes int) (*partition.EvaluationResults, error) {
	return tp.onlineLearner.EvaluateModel(episodes)
}

// Destroy cleans up resources
func (tp *TaskPartitioner) Destroy() {
	tp.agent.Destroy()
}

// Integration with DWCP Manager

// AddTaskPartitioner adds ITP to the DWCP manager
func (m *Manager) AddTaskPartitioner(modelPath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	partitioner, err := NewTaskPartitioner(modelPath, m.logger)
	if err != nil {
		return fmt.Errorf("failed to create task partitioner: %w", err)
	}

	// Store partitioner (would need to add field to Manager struct)
	// m.partitioner = partitioner

	m.logger.Info("Task partitioner added to DWCP manager",
		zap.String("model_path", modelPath))

	return nil
}

// PartitionTask uses ITP to intelligently partition a task
func (m *Manager) PartitionTask(ctx context.Context, task *Task) (*partition.TaskPartitionDecision, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.enabled || !m.started {
		return nil, fmt.Errorf("DWCP not enabled or not started")
	}

	// Would use m.partitioner here
	// return m.partitioner.PartitionTask(task)

	// For now, return a placeholder
	return &partition.TaskPartitionDecision{
		StreamIDs:    []int{0},
		ChunkSizes:   []int{task.Size},
		Confidence:   0.5,
		ExpectedTime: 1 * time.Second,
	}, nil
}