package federated

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// FLCoordinator coordinates federated learning across distributed clients
type FLCoordinator struct {
	config       *FederatedConfig
	clients      map[string]*Client
	globalModel  *Model
	aggregator   *ModelAggregator
	privacyMech  *PrivacyMechanism
	mu           sync.RWMutex
	currentRound int
	history      []RoundHistory
}

// FederatedConfig defines federated learning configuration
type FederatedConfig struct {
	NumClients        int
	ClientFraction    float64 // Fraction of clients to sample per round
	LocalEpochs       int     // Epochs each client trains
	GlobalRounds      int     // Total federated rounds
	PrivacyBudget     float64 // Epsilon for differential privacy
	SecureAggregation bool    // Enable secure aggregation
	MinClients        int     // Minimum clients per round
	Timeout           time.Duration
	Algorithm         string  // "fedavg", "fedprox", "fedadam"
}

// Client represents a federated learning client
type Client struct {
	ID            string
	Model         *Model
	DataSize      int
	LastUpdate    time.Time
	Metrics       map[string]float64
	IsActive      bool
	Region        string
}

// Model represents a neural network model
type Model struct {
	Weights    [][]float64
	Biases     []float64
	Architecture string
	Version    int
}

// RoundHistory tracks federated learning round metrics
type RoundHistory struct {
	Round           int
	Timestamp       time.Time
	NumClients      int
	GlobalLoss      float64
	GlobalAccuracy  float64
	AggregationTime time.Duration
	ClientMetrics   map[string]float64
}

// ModelAggregator aggregates client models
type ModelAggregator struct {
	algorithm string
}

// PrivacyMechanism implements differential privacy
type PrivacyMechanism struct {
	epsilon       float64
	delta         float64
	noiseScale    float64
	clippingNorm  float64
}

// Update represents a client model update
type Update struct {
	ClientID  string
	Weights   [][]float64
	DataSize  int
	Loss      float64
	Accuracy  float64
}

// NewFLCoordinator creates a new federated learning coordinator
func NewFLCoordinator(config *FederatedConfig) *FLCoordinator {
	if config == nil {
		config = DefaultFederatedConfig()
	}

	coordinator := &FLCoordinator{
		config:      config,
		clients:     make(map[string]*Client),
		globalModel: initializeGlobalModel(),
		aggregator:  NewModelAggregator(config.Algorithm),
		history:     make([]RoundHistory, 0),
	}

	if config.PrivacyBudget > 0 {
		coordinator.privacyMech = NewPrivacyMechanism(config.PrivacyBudget, 1e-5)
	}

	return coordinator
}

// DefaultFederatedConfig returns default federated learning configuration
func DefaultFederatedConfig() *FederatedConfig {
	return &FederatedConfig{
		NumClients:        100,
		ClientFraction:    0.1,  // 10% of clients per round
		LocalEpochs:       5,
		GlobalRounds:      100,
		PrivacyBudget:     1.0,  // epsilon for DP
		SecureAggregation: true,
		MinClients:        10,
		Timeout:           10 * time.Minute,
		Algorithm:         "fedavg",
	}
}

// RegisterClient registers a new federated learning client
func (fc *FLCoordinator) RegisterClient(clientID, region string, dataSize int) error {
	fc.mu.Lock()
	defer fc.mu.Unlock()

	if _, exists := fc.clients[clientID]; exists {
		return fmt.Errorf("client %s already registered", clientID)
	}

	client := &Client{
		ID:         clientID,
		Model:      fc.copyModel(fc.globalModel),
		DataSize:   dataSize,
		LastUpdate: time.Now(),
		Metrics:    make(map[string]float64),
		IsActive:   true,
		Region:     region,
	}

	fc.clients[clientID] = client
	return nil
}

// Train runs federated learning training
func (fc *FLCoordinator) Train(ctx context.Context) error {
	for round := 0; round < fc.config.GlobalRounds; round++ {
		fc.currentRound = round

		// Select clients for this round
		selectedClients := fc.selectClients()
		if len(selectedClients) < fc.config.MinClients {
			return fmt.Errorf("insufficient clients: %d < %d", len(selectedClients), fc.config.MinClients)
		}

		// Distribute global model to selected clients
		for _, client := range selectedClients {
			client.Model = fc.copyModel(fc.globalModel)
		}

		// Parallel client training
		updatesChan := make(chan Update, len(selectedClients))
		var wg sync.WaitGroup

		for _, client := range selectedClients {
			wg.Add(1)
			go func(c *Client) {
				defer wg.Done()
				update := fc.clientTrain(ctx, c)
				updatesChan <- update
			}(client)
		}

		// Wait for all clients to complete
		go func() {
			wg.Wait()
			close(updatesChan)
		}()

		// Collect updates
		updates := make([]Update, 0)
		for update := range updatesChan {
			updates = append(updates, update)
		}

		// Aggregate updates
		startAgg := time.Now()
		if err := fc.aggregateUpdates(updates); err != nil {
			return fmt.Errorf("aggregation failed: %w", err)
		}
		aggTime := time.Since(startAgg)

		// Evaluate global model
		globalMetrics := fc.evaluateGlobalModel()

		// Record round history
		history := RoundHistory{
			Round:           round,
			Timestamp:       time.Now(),
			NumClients:      len(updates),
			GlobalLoss:      globalMetrics["loss"],
			GlobalAccuracy:  globalMetrics["accuracy"],
			AggregationTime: aggTime,
			ClientMetrics:   fc.aggregateClientMetrics(updates),
		}

		fc.mu.Lock()
		fc.history = append(fc.history, history)
		fc.mu.Unlock()

		// Log progress
		fmt.Printf("Round %d/%d: Accuracy=%.4f, Loss=%.4f, Clients=%d\n",
			round+1, fc.config.GlobalRounds, globalMetrics["accuracy"], globalMetrics["loss"], len(updates))

		// Check convergence
		if fc.hasConverged() {
			fmt.Printf("Converged at round %d\n", round+1)
			break
		}
	}

	return nil
}

// selectClients selects clients for federated round
func (fc *FLCoordinator) selectClients() []*Client {
	fc.mu.RLock()
	defer fc.mu.RUnlock()

	// Get active clients
	activeClients := make([]*Client, 0)
	for _, client := range fc.clients {
		if client.IsActive {
			activeClients = append(activeClients, client)
		}
	}

	// Sample fraction of clients
	numSelect := int(float64(len(activeClients)) * fc.config.ClientFraction)
	if numSelect < fc.config.MinClients {
		numSelect = fc.config.MinClients
	}
	if numSelect > len(activeClients) {
		numSelect = len(activeClients)
	}

	// Random selection (can be stratified by region/data)
	selected := make([]*Client, numSelect)
	indices := randomSample(len(activeClients), numSelect)
	for i, idx := range indices {
		selected[i] = activeClients[idx]
	}

	return selected
}

// clientTrain simulates client-side training
func (fc *FLCoordinator) clientTrain(ctx context.Context, client *Client) Update {
	// Simulate local training
	// In practice, this would be actual model training on client data

	trainCtx, cancel := context.WithTimeout(ctx, fc.config.Timeout)
	defer cancel()

	done := make(chan Update)

	go func() {
		// Simulate training with local epochs
		for epoch := 0; epoch < fc.config.LocalEpochs; epoch++ {
			// Simulate gradient updates
			fc.simulateLocalUpdate(client)
		}

		// Compute local metrics
		loss, accuracy := fc.evaluateClientModel(client)

		update := Update{
			ClientID: client.ID,
			Weights:  client.Model.Weights,
			DataSize: client.DataSize,
			Loss:     loss,
			Accuracy: accuracy,
		}

		done <- update
	}()

	select {
	case update := <-done:
		return update
	case <-trainCtx.Done():
		return Update{
			ClientID: client.ID,
			Weights:  client.Model.Weights,
			DataSize: 0,
		}
	}
}

// simulateLocalUpdate simulates a local gradient update
func (fc *FLCoordinator) simulateLocalUpdate(client *Client) {
	// Simplified simulation of SGD update
	learningRate := 0.01

	for i := range client.Model.Weights {
		for j := range client.Model.Weights[i] {
			// Add random gradient (simulation)
			gradient := (math.NormFloat64() * 0.1)
			client.Model.Weights[i][j] -= learningRate * gradient
		}
	}
}

// aggregateUpdates aggregates client model updates
func (fc *FLCoordinator) aggregateUpdates(updates []Update) error {
	if len(updates) == 0 {
		return fmt.Errorf("no updates to aggregate")
	}

	// Apply differential privacy if configured
	if fc.privacyMech != nil {
		for i := range updates {
			updates[i].Weights = fc.privacyMech.AddNoise(updates[i].Weights)
		}
	}

	// Aggregate based on algorithm
	aggregated, err := fc.aggregator.Aggregate(updates)
	if err != nil {
		return err
	}

	fc.mu.Lock()
	fc.globalModel.Weights = aggregated
	fc.globalModel.Version++
	fc.mu.Unlock()

	return nil
}

// evaluateClientModel evaluates a client model
func (fc *FLCoordinator) evaluateClientModel(client *Client) (float64, float64) {
	// Simplified evaluation
	// In practice, evaluate on client's local test data

	loss := math.Abs(math.NormFloat64() * 0.5)
	accuracy := 0.7 + math.NormFloat64()*0.1

	// Clip to valid range
	accuracy = math.Max(0, math.Min(1, accuracy))

	return loss, accuracy
}

// evaluateGlobalModel evaluates the global model
func (fc *FLCoordinator) evaluateGlobalModel() map[string]float64 {
	// Simplified global evaluation
	// In practice, evaluate on held-out test data

	metrics := make(map[string]float64)

	// Simulate improving metrics over rounds
	baseAccuracy := 0.6
	improvement := float64(fc.currentRound) * 0.003
	noise := math.NormFloat64() * 0.02

	metrics["accuracy"] = math.Min(0.95, baseAccuracy+improvement+noise)
	metrics["loss"] = math.Max(0.05, 1.0-metrics["accuracy"]+math.Abs(noise))
	metrics["f1_score"] = metrics["accuracy"] * 0.95

	return metrics
}

// aggregateClientMetrics aggregates metrics from clients
func (fc *FLCoordinator) aggregateClientMetrics(updates []Update) map[string]float64 {
	metrics := make(map[string]float64)

	totalData := 0
	weightedLoss := 0.0
	weightedAccuracy := 0.0

	for _, update := range updates {
		totalData += update.DataSize
		weightedLoss += update.Loss * float64(update.DataSize)
		weightedAccuracy += update.Accuracy * float64(update.DataSize)
	}

	if totalData > 0 {
		metrics["avg_loss"] = weightedLoss / float64(totalData)
		metrics["avg_accuracy"] = weightedAccuracy / float64(totalData)
	}

	metrics["num_clients"] = float64(len(updates))

	return metrics
}

// hasConverged checks if training has converged
func (fc *FLCoordinator) hasConverged() bool {
	if len(fc.history) < 10 {
		return false
	}

	// Check if accuracy improvement is minimal
	recent := fc.history[len(fc.history)-5:]
	maxAcc := recent[0].GlobalAccuracy
	minAcc := recent[0].GlobalAccuracy

	for _, h := range recent {
		if h.GlobalAccuracy > maxAcc {
			maxAcc = h.GlobalAccuracy
		}
		if h.GlobalAccuracy < minAcc {
			minAcc = h.GlobalAccuracy
		}
	}

	improvement := maxAcc - minAcc
	return improvement < 0.001 // Converged if improvement < 0.1%
}

// GetGlobalModel returns the current global model
func (fc *FLCoordinator) GetGlobalModel() *Model {
	fc.mu.RLock()
	defer fc.mu.RUnlock()
	return fc.copyModel(fc.globalModel)
}

// GetHistory returns training history
func (fc *FLCoordinator) GetHistory() []RoundHistory {
	fc.mu.RLock()
	defer fc.mu.RUnlock()
	history := make([]RoundHistory, len(fc.history))
	copy(history, fc.history)
	return history
}

// copyModel creates a deep copy of a model
func (fc *FLCoordinator) copyModel(model *Model) *Model {
	newModel := &Model{
		Weights:      make([][]float64, len(model.Weights)),
		Biases:       make([]float64, len(model.Biases)),
		Architecture: model.Architecture,
		Version:      model.Version,
	}

	for i := range model.Weights {
		newModel.Weights[i] = make([]float64, len(model.Weights[i]))
		copy(newModel.Weights[i], model.Weights[i])
	}

	copy(newModel.Biases, model.Biases)

	return newModel
}

// NewModelAggregator creates a new model aggregator
func NewModelAggregator(algorithm string) *ModelAggregator {
	return &ModelAggregator{algorithm: algorithm}
}

// Aggregate aggregates model updates
func (ma *ModelAggregator) Aggregate(updates []Update) ([][]float64, error) {
	switch ma.algorithm {
	case "fedavg":
		return ma.fedAvg(updates)
	case "fedprox":
		return ma.fedProx(updates)
	case "fedadam":
		return ma.fedAdam(updates)
	default:
		return ma.fedAvg(updates)
	}
}

// fedAvg implements FedAvg aggregation
func (ma *ModelAggregator) fedAvg(updates []Update) ([][]float64, error) {
	if len(updates) == 0 {
		return nil, fmt.Errorf("no updates")
	}

	// Weighted average by data size
	totalData := 0
	for _, update := range updates {
		totalData += update.DataSize
	}

	aggregated := make([][]float64, len(updates[0].Weights))
	for i := range aggregated {
		aggregated[i] = make([]float64, len(updates[0].Weights[i]))
	}

	for _, update := range updates {
		weight := float64(update.DataSize) / float64(totalData)
		for i := range update.Weights {
			for j := range update.Weights[i] {
				aggregated[i][j] += update.Weights[i][j] * weight
			}
		}
	}

	return aggregated, nil
}

// fedProx implements FedProx aggregation
func (ma *ModelAggregator) fedProx(updates []Update) ([][]float64, error) {
	// FedProx is similar to FedAvg but with proximal term
	// For simplification, use FedAvg here
	return ma.fedAvg(updates)
}

// fedAdam implements FedAdam aggregation
func (ma *ModelAggregator) fedAdam(updates []Update) ([][]float64, error) {
	// FedAdam uses adaptive learning rates
	// For simplification, use FedAvg here
	return ma.fedAvg(updates)
}

// NewPrivacyMechanism creates a new privacy mechanism
func NewPrivacyMechanism(epsilon, delta float64) *PrivacyMechanism {
	// Calculate noise scale for (epsilon, delta)-DP
	noiseScale := math.Sqrt(2 * math.Log(1.25/delta)) / epsilon

	return &PrivacyMechanism{
		epsilon:      epsilon,
		delta:        delta,
		noiseScale:   noiseScale,
		clippingNorm: 1.0,
	}
}

// AddNoise adds differential privacy noise to weights
func (pm *PrivacyMechanism) AddNoise(weights [][]float64) [][]float64 {
	// Clip weights to bound sensitivity
	clipped := pm.clipWeights(weights)

	// Add Gaussian noise
	noisy := make([][]float64, len(clipped))
	for i := range clipped {
		noisy[i] = make([]float64, len(clipped[i]))
		for j := range clipped[i] {
			noise := math.NormFloat64() * pm.noiseScale
			noisy[i][j] = clipped[i][j] + noise
		}
	}

	return noisy
}

// clipWeights clips weights to bound L2 norm
func (pm *PrivacyMechanism) clipWeights(weights [][]float64) [][]float64 {
	// Calculate L2 norm
	norm := 0.0
	for i := range weights {
		for j := range weights[i] {
			norm += weights[i][j] * weights[i][j]
		}
	}
	norm = math.Sqrt(norm)

	// Clip if norm exceeds threshold
	if norm <= pm.clippingNorm {
		return weights
	}

	clipped := make([][]float64, len(weights))
	scale := pm.clippingNorm / norm
	for i := range weights {
		clipped[i] = make([]float64, len(weights[i]))
		for j := range weights[i] {
			clipped[i][j] = weights[i][j] * scale
		}
	}

	return clipped
}

// Helper functions

func initializeGlobalModel() *Model {
	// Initialize a simple model
	return &Model{
		Weights: [][]float64{
			{0.1, 0.2, 0.3},
			{0.4, 0.5, 0.6},
		},
		Biases:       []float64{0.1, 0.2},
		Architecture: "simple_nn",
		Version:      0,
	}
}

func randomSample(n, k int) []int {
	if k > n {
		k = n
	}

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yates shuffle
	for i := n - 1; i > 0; i-- {
		j := randInt(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:k]
}

func randInt(n int) int {
	nBig, _ := rand.Int(rand.Reader, big.NewInt(int64(n)))
	return int(nBig.Int64())
}

func math.NormFloat64() float64 {
	// Box-Muller transform
	u1 := math.Float64()
	u2 := math.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func math.Float64() float64 {
	nBig, _ := rand.Int(rand.Reader, big.NewInt(1<<53))
	return float64(nBig.Int64()) / float64(1<<53)
}
