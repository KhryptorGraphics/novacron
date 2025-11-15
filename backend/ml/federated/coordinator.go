// Package federated implements TCS-FEEL federated learning coordinator
package federated

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sync"
	"time"
)

// Model represents a machine learning model
type Model struct {
	Weights    map[string][]float64 `json:"weights"`
	Version    int                  `json:"version"`
	Accuracy   float64              `json:"accuracy"`
	UpdateTime time.Time            `json:"update_time"`
}

// ModelUpdate represents a client's model update
type ModelUpdate struct {
	ClientID   int                  `json:"client_id"`
	Weights    map[string][]float64 `json:"weights"`
	DataSize   int                  `json:"data_size"`
	Loss       float64              `json:"loss"`
	Accuracy   float64              `json:"accuracy"`
	UpdateTime time.Time            `json:"update_time"`
}

// Client represents a federated learning client
type Client struct {
	ID               int                  `json:"id"`
	DataSize         int                  `json:"data_size"`
	DataDistribution []float64            `json:"data_distribution"`
	ComputeCapacity  float64              `json:"compute_capacity"`
	Bandwidth        float64              `json:"bandwidth"`
	Latency          float64              `json:"latency"`
	Reliability      float64              `json:"reliability"`
	CurrentModel     *Model               `json:"current_model,omitempty"`
	LastUpdate       *ModelUpdate         `json:"last_update,omitempty"`
}

// TrainingRound represents a federated learning round
type TrainingRound struct {
	RoundNumber       int                `json:"round_number"`
	SelectedClients   []int              `json:"selected_clients"`
	GlobalModel       *Model             `json:"global_model"`
	StartTime         time.Time          `json:"start_time"`
	EndTime           time.Time          `json:"end_time"`
	AverageAccuracy   float64            `json:"average_accuracy"`
	CommCost          float64            `json:"communication_cost"`
	ConvergenceSpeed  float64            `json:"convergence_speed"`
}

// FederatedCoordinator manages federated learning process
type FederatedCoordinator struct {
	mu                sync.RWMutex
	clients           map[int]*Client
	globalModel       *Model
	topology          *TopologyOptimizer // Python integration
	currentRound      int
	trainingHistory   []*TrainingRound
	targetAccuracy    float64
	maxRounds         int
	aggregationMethod string // "fedavg", "fedprox", "weighted"

	// Channels for async operations
	updateChan    chan *ModelUpdate
	errorChan     chan error
	shutdownChan  chan struct{}
}

// TopologyOptimizer interface for Python integration
type TopologyOptimizer interface {
	OptimizeTopology(roundNumber int, budgetConstraint float64) ([]*Client, error)
	UpdateClientPerformance(clientID int, quality, reliability float64) error
	GetTopologyStats() (map[string]interface{}, error)
}

// NewFederatedCoordinator creates a new coordinator
func NewFederatedCoordinator(
	targetAccuracy float64,
	maxRounds int,
	aggregationMethod string,
) *FederatedCoordinator {
	return &FederatedCoordinator{
		clients:           make(map[int]*Client),
		globalModel:       initializeGlobalModel(),
		currentRound:      0,
		trainingHistory:   make([]*TrainingRound, 0),
		targetAccuracy:    targetAccuracy,
		maxRounds:         maxRounds,
		aggregationMethod: aggregationMethod,
		updateChan:        make(chan *ModelUpdate, 100),
		errorChan:         make(chan error, 10),
		shutdownChan:      make(chan struct{}),
	}
}

// RegisterClient adds a client to the federation
func (c *FederatedCoordinator) RegisterClient(client *Client) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.clients[client.ID]; exists {
		return fmt.Errorf("client %d already registered", client.ID)
	}

	// Initialize client with global model
	client.CurrentModel = c.globalModel.Clone()
	c.clients[client.ID] = client

	log.Printf("[FedCoord] Registered client %d (data_size=%d, compute=%.2f GFLOPS)",
		client.ID, client.DataSize, client.ComputeCapacity)

	return nil
}

// SetTopologyOptimizer sets the Python topology optimizer
func (c *FederatedCoordinator) SetTopologyOptimizer(optimizer TopologyOptimizer) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.topology = optimizer
}

// TrainRound executes one federated learning round
func (c *FederatedCoordinator) TrainRound(ctx context.Context) (*TrainingRound, error) {
	c.mu.Lock()
	c.currentRound++
	roundNumber := c.currentRound
	c.mu.Unlock()

	log.Printf("\n=== Starting Round %d ===", roundNumber)
	startTime := time.Now()

	round := &TrainingRound{
		RoundNumber: roundNumber,
		StartTime:   startTime,
		GlobalModel: c.globalModel.Clone(),
	}

	// Step 1: Select clients using TCS-FEEL topology optimization
	selectedClients, err := c.selectClients(roundNumber)
	if err != nil {
		return nil, fmt.Errorf("client selection failed: %w", err)
	}

	round.SelectedClients = extractClientIDs(selectedClients)
	log.Printf("[Round %d] Selected %d clients: %v",
		roundNumber, len(selectedClients), round.SelectedClients)

	// Step 2: Distribute global model to selected clients
	if err := c.distributeModel(ctx, selectedClients); err != nil {
		return nil, fmt.Errorf("model distribution failed: %w", err)
	}

	// Step 3: Collect local training updates
	updates, err := c.collectUpdates(ctx, selectedClients)
	if err != nil {
		return nil, fmt.Errorf("update collection failed: %w", err)
	}

	// Step 4: Aggregate updates into global model
	if err := c.aggregateUpdates(updates); err != nil {
		return nil, fmt.Errorf("aggregation failed: %w", err)
	}

	// Step 5: Evaluate global model
	avgAccuracy, commCost := c.evaluateRound(updates)

	// Step 6: Update client performance metrics
	c.updateClientMetrics(updates)

	// Finalize round
	round.EndTime = time.Now()
	round.AverageAccuracy = avgAccuracy
	round.CommCost = commCost
	round.ConvergenceSpeed = c.calculateConvergenceSpeed(round)

	c.mu.Lock()
	c.trainingHistory = append(c.trainingHistory, round)
	c.mu.Unlock()

	log.Printf("[Round %d] Complete - Accuracy: %.2f%%, CommCost: %.2f, Time: %v",
		roundNumber, avgAccuracy*100, commCost, round.EndTime.Sub(round.StartTime))

	return round, nil
}

// selectClients uses TCS-FEEL to select optimal clients
func (c *FederatedCoordinator) selectClients(roundNumber int) ([]*Client, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.topology == nil {
		// Fallback: random selection
		return c.randomClientSelection(10), nil
	}

	// Use topology optimizer (Python integration)
	budgetConstraint := 1000.0 // Example budget
	selectedClients, err := c.topology.OptimizeTopology(roundNumber, budgetConstraint)
	if err != nil {
		log.Printf("[WARN] Topology optimization failed: %v, using fallback", err)
		return c.randomClientSelection(10), nil
	}

	return selectedClients, nil
}

// randomClientSelection fallback method
func (c *FederatedCoordinator) randomClientSelection(n int) []*Client {
	clients := make([]*Client, 0, len(c.clients))
	for _, client := range c.clients {
		clients = append(clients, client)
	}

	// Simple random selection
	if len(clients) <= n {
		return clients
	}

	selected := make([]*Client, n)
	for i := 0; i < n; i++ {
		selected[i] = clients[i]
	}
	return selected
}

// distributeModel sends global model to selected clients
func (c *FederatedCoordinator) distributeModel(
	ctx context.Context,
	clients []*Client,
) error {
	c.mu.RLock()
	globalModel := c.globalModel.Clone()
	c.mu.RUnlock()

	var wg sync.WaitGroup
	errChan := make(chan error, len(clients))

	for _, client := range clients {
		wg.Add(1)
		go func(cl *Client) {
			defer wg.Done()

			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
				// Simulate network latency
				time.Sleep(time.Duration(cl.Latency) * time.Millisecond)

				// Update client's model
				c.mu.Lock()
				cl.CurrentModel = globalModel.Clone()
				c.mu.Unlock()

				log.Printf("[Client %d] Received global model v%d",
					cl.ID, globalModel.Version)
			}
		}(client)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

// collectUpdates gathers model updates from clients
func (c *FederatedCoordinator) collectUpdates(
	ctx context.Context,
	clients []*Client,
) ([]*ModelUpdate, error) {
	updateChan := make(chan *ModelUpdate, len(clients))
	errChan := make(chan error, len(clients))
	var wg sync.WaitGroup

	for _, client := range clients {
		wg.Add(1)
		go func(cl *Client) {
			defer wg.Done()

			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
				// Simulate local training
				update := c.simulateLocalTraining(cl)
				updateChan <- update

				log.Printf("[Client %d] Training complete - Accuracy: %.2f%%, Loss: %.4f",
					cl.ID, update.Accuracy*100, update.Loss)
			}
		}(client)
	}

	wg.Wait()
	close(updateChan)
	close(errChan)

	// Collect updates
	updates := make([]*ModelUpdate, 0, len(clients))
	for update := range updateChan {
		updates = append(updates, update)
	}

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, err
		}
	}

	return updates, nil
}

// simulateLocalTraining simulates client-side training
func (c *FederatedCoordinator) simulateLocalTraining(client *Client) *ModelUpdate {
	// Simulate training time based on data size and compute
	trainingTime := float64(client.DataSize) / (client.ComputeCapacity * 1000)
	time.Sleep(time.Duration(trainingTime*100) * time.Millisecond)

	// Simulate accuracy improvement (based on data quality)
	baseAccuracy := c.globalModel.Accuracy
	improvement := 0.01 * client.Reliability * (float64(client.DataSize) / 5000.0)
	newAccuracy := math.Min(baseAccuracy+improvement, 0.99)

	// Create update with simulated weights
	update := &ModelUpdate{
		ClientID:   client.ID,
		Weights:    c.perturbWeights(c.globalModel.Weights, improvement),
		DataSize:   client.DataSize,
		Loss:       (1.0 - newAccuracy) * 0.5, // Simplified loss
		Accuracy:   newAccuracy,
		UpdateTime: time.Now(),
	}

	return update
}

// perturbWeights simulates weight updates
func (c *FederatedCoordinator) perturbWeights(
	weights map[string][]float64,
	improvement float64,
) map[string][]float64 {
	newWeights := make(map[string][]float64)
	for layer, w := range weights {
		newW := make([]float64, len(w))
		for i, val := range w {
			// Small perturbation based on improvement
			delta := (2.0*math.Sin(float64(i)) - 1.0) * improvement * 0.1
			newW[i] = val + delta
		}
		newWeights[layer] = newW
	}
	return newWeights
}

// aggregateUpdates combines client updates into global model
func (c *FederatedCoordinator) aggregateUpdates(updates []*ModelUpdate) error {
	if len(updates) == 0 {
		return fmt.Errorf("no updates to aggregate")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	switch c.aggregationMethod {
	case "fedavg":
		return c.federatedAveraging(updates)
	case "weighted":
		return c.weightedAggregation(updates)
	default:
		return c.federatedAveraging(updates)
	}
}

// federatedAveraging implements FedAvg algorithm
func (c *FederatedCoordinator) federatedAveraging(updates []*ModelUpdate) error {
	// Calculate total data size
	totalData := 0
	for _, update := range updates {
		totalData += update.DataSize
	}

	// Initialize aggregated weights
	aggregated := make(map[string][]float64)

	// Weighted average based on data size
	for _, update := range updates {
		weight := float64(update.DataSize) / float64(totalData)

		for layer, weights := range update.Weights {
			if _, exists := aggregated[layer]; !exists {
				aggregated[layer] = make([]float64, len(weights))
			}

			for i, val := range weights {
				aggregated[layer][i] += val * weight
			}
		}
	}

	// Update global model
	c.globalModel.Weights = aggregated
	c.globalModel.Version++
	c.globalModel.UpdateTime = time.Now()

	// Calculate average accuracy
	avgAccuracy := 0.0
	for _, update := range updates {
		avgAccuracy += update.Accuracy * (float64(update.DataSize) / float64(totalData))
	}
	c.globalModel.Accuracy = avgAccuracy

	log.Printf("[Aggregation] FedAvg complete - New model v%d, Accuracy: %.2f%%",
		c.globalModel.Version, c.globalModel.Accuracy*100)

	return nil
}

// weightedAggregation with quality-based weighting
func (c *FederatedCoordinator) weightedAggregation(updates []*ModelUpdate) error {
	// Calculate weights based on data size and accuracy
	totalWeight := 0.0
	weights := make([]float64, len(updates))

	for i, update := range updates {
		// Weight = data_size * accuracy
		w := float64(update.DataSize) * update.Accuracy
		weights[i] = w
		totalWeight += w
	}

	// Normalize weights
	for i := range weights {
		weights[i] /= totalWeight
	}

	// Aggregate
	aggregated := make(map[string][]float64)
	for i, update := range updates {
		for layer, layerWeights := range update.Weights {
			if _, exists := aggregated[layer]; !exists {
				aggregated[layer] = make([]float64, len(layerWeights))
			}

			for j, val := range layerWeights {
				aggregated[layer][j] += val * weights[i]
			}
		}
	}

	c.globalModel.Weights = aggregated
	c.globalModel.Version++
	c.globalModel.UpdateTime = time.Now()

	// Weighted average accuracy
	avgAccuracy := 0.0
	for i, update := range updates {
		avgAccuracy += update.Accuracy * weights[i]
	}
	c.globalModel.Accuracy = avgAccuracy

	return nil
}

// evaluateRound calculates round metrics
func (c *FederatedCoordinator) evaluateRound(updates []*ModelUpdate) (float64, float64) {
	totalData := 0
	avgAccuracy := 0.0
	commCost := 0.0

	for _, update := range updates {
		totalData += update.DataSize

		// Weighted accuracy
		weight := float64(update.DataSize)
		avgAccuracy += update.Accuracy * weight

		// Communication cost (bytes transferred)
		commCost += float64(len(update.Weights)) * float64(update.DataSize)
	}

	if totalData > 0 {
		avgAccuracy /= float64(totalData)
	}

	return avgAccuracy, commCost
}

// updateClientMetrics updates client performance tracking
func (c *FederatedCoordinator) updateClientMetrics(updates []*ModelUpdate) {
	for _, update := range updates {
		// Update quality = accuracy
		quality := update.Accuracy

		// Update reliability based on performance
		c.mu.Lock()
		if client, exists := c.clients[update.ClientID]; exists {
			client.LastUpdate = update

			// Exponential moving average for reliability
			alpha := 0.3
			newReliability := alpha*quality + (1-alpha)*client.Reliability
			client.Reliability = newReliability

			// Update topology optimizer if available
			if c.topology != nil {
				go c.topology.UpdateClientPerformance(
					update.ClientID,
					quality,
					newReliability,
				)
			}
		}
		c.mu.Unlock()
	}
}

// calculateConvergenceSpeed estimates convergence metric
func (c *FederatedCoordinator) calculateConvergenceSpeed(round *TrainingRound) float64 {
	if len(c.trainingHistory) < 2 {
		return 1.0
	}

	// Compare with previous round
	prevRound := c.trainingHistory[len(c.trainingHistory)-2]
	accuracyImprovement := round.AverageAccuracy - prevRound.AverageAccuracy

	// Convergence speed = improvement / time
	duration := round.EndTime.Sub(round.StartTime).Seconds()
	if duration > 0 {
		return accuracyImprovement / duration
	}

	return 0.0
}

// GetStatus returns current coordinator status
func (c *FederatedCoordinator) GetStatus() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return map[string]interface{}{
		"current_round":    c.currentRound,
		"total_clients":    len(c.clients),
		"global_accuracy":  c.globalModel.Accuracy,
		"model_version":    c.globalModel.Version,
		"target_accuracy":  c.targetAccuracy,
		"rounds_completed": len(c.trainingHistory),
	}
}

// Helper functions

func initializeGlobalModel() *Model {
	// Initialize with random weights
	weights := map[string][]float64{
		"layer1": make([]float64, 128),
		"layer2": make([]float64, 64),
		"output": make([]float64, 10),
	}

	// Random initialization
	for layer := range weights {
		for i := range weights[layer] {
			weights[layer][i] = (2.0*math.Sin(float64(i)) - 1.0) * 0.1
		}
	}

	return &Model{
		Weights:    weights,
		Version:    1,
		Accuracy:   0.75, // Initial accuracy
		UpdateTime: time.Now(),
	}
}

func (m *Model) Clone() *Model {
	weights := make(map[string][]float64)
	for layer, w := range m.Weights {
		weights[layer] = append([]float64{}, w...)
	}

	return &Model{
		Weights:    weights,
		Version:    m.Version,
		Accuracy:   m.Accuracy,
		UpdateTime: m.UpdateTime,
	}
}

func extractClientIDs(clients []*Client) []int {
	ids := make([]int, len(clients))
	for i, client := range clients {
		ids[i] = client.ID
	}
	return ids
}

// JSON marshaling support

func (c *FederatedCoordinator) MarshalJSON() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return json.Marshal(map[string]interface{}{
		"current_round":   c.currentRound,
		"clients":         len(c.clients),
		"global_accuracy": c.globalModel.Accuracy,
		"model_version":   c.globalModel.Version,
	})
}
