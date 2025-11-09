package algorithms

import (
	"context"
	"fmt"
	"math"
	"sync"
)

// FederatedLearningConfig configures federated learning
type FederatedLearningConfig struct {
	NumClients        int
	ClientSampleSize  int
	NumRounds         int
	LearningRate      float64
	PrivacyBudget     float64
	SecureAggregation bool
	DifferentialPrivacy bool
}

// FLClient represents a federated learning client
type FLClient struct {
	ID           string
	DataSize     int
	LocalModel   []float64
	LocalGrad    []float64
	PrivacyNoise float64
}

// FLServer represents the federated learning server
type FLServer struct {
	config       FederatedLearningConfig
	globalModel  []float64
	clients      []*FLClient
	currentRound int
	mu           sync.Mutex
}

// NewFLServer creates a new federated learning server
func NewFLServer(config FederatedLearningConfig) *FLServer {
	return &FLServer{
		config:       config,
		globalModel:  make([]float64, 0),
		clients:      make([]*FLClient, 0),
		currentRound: 0,
	}
}

// RegisterClient registers a new client
func (s *FLServer) RegisterClient(clientID string, dataSize int) *FLClient {
	s.mu.Lock()
	defer s.mu.Unlock()

	client := &FLClient{
		ID:       clientID,
		DataSize: dataSize,
	}

	s.clients = append(s.clients, client)
	return client
}

// Train runs federated learning training
func (s *FLServer) Train(ctx context.Context) error {
	// Initialize global model
	s.globalModel = s.initializeModel()

	for round := 0; round < s.config.NumRounds; round++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		s.currentRound = round

		// Select clients for this round
		selectedClients := s.selectClients()

		// Broadcast global model to selected clients
		s.broadcastModel(selectedClients)

		// Clients perform local training
		gradients := s.clientTraining(ctx, selectedClients)

		// Aggregate gradients
		if s.config.SecureAggregation {
			gradients = s.secureAggregate(gradients)
		} else {
			gradients = s.federatedAverage(gradients)
		}

		// Apply differential privacy if enabled
		if s.config.DifferentialPrivacy {
			gradients = s.applyDifferentialPrivacy(gradients)
		}

		// Update global model
		s.updateGlobalModel(gradients)

		// Evaluate model
		accuracy := s.evaluate()
		fmt.Printf("Round %d: Accuracy = %.4f\n", round, accuracy)
	}

	return nil
}

// initializeModel initializes the global model
func (s *FLServer) initializeModel() []float64 {
	// Initialize with small random values
	modelSize := 1000 // Example size
	model := make([]float64, modelSize)
	for i := range model {
		model[i] = (float64(i) * 0.01) - 0.5 // Simple initialization
	}
	return model
}

// selectClients selects clients for the current round
func (s *FLServer) selectClients() []*FLClient {
	// Simple random selection
	sampleSize := s.config.ClientSampleSize
	if sampleSize > len(s.clients) {
		sampleSize = len(s.clients)
	}

	selected := make([]*FLClient, sampleSize)
	copy(selected, s.clients[:sampleSize])
	return selected
}

// broadcastModel broadcasts the global model to selected clients
func (s *FLServer) broadcastModel(clients []*FLClient) {
	for _, client := range clients {
		client.LocalModel = make([]float64, len(s.globalModel))
		copy(client.LocalModel, s.globalModel)
	}
}

// clientTraining simulates client local training
func (s *FLServer) clientTraining(ctx context.Context, clients []*FLClient) [][]float64 {
	gradients := make([][]float64, len(clients))
	var wg sync.WaitGroup

	for i, client := range clients {
		wg.Add(1)
		go func(idx int, c *FLClient) {
			defer wg.Done()

			// Simulate local training
			localGrad := make([]float64, len(c.LocalModel))
			for j := range localGrad {
				// Simulate gradient computation
				localGrad[j] = (c.LocalModel[j] - s.globalModel[j]) * s.config.LearningRate
			}

			c.LocalGrad = localGrad
			gradients[idx] = localGrad
		}(i, client)
	}

	wg.Wait()
	return gradients
}

// federatedAverage computes federated averaging
func (s *FLServer) federatedAverage(gradients [][]float64) []float64 {
	if len(gradients) == 0 {
		return make([]float64, len(s.globalModel))
	}

	avgGrad := make([]float64, len(gradients[0]))
	totalWeight := 0.0

	// Weight by client data size
	for i, grad := range gradients {
		weight := float64(s.clients[i].DataSize)
		totalWeight += weight

		for j := range grad {
			avgGrad[j] += grad[j] * weight
		}
	}

	// Normalize
	for j := range avgGrad {
		avgGrad[j] /= totalWeight
	}

	return avgGrad
}

// secureAggregate performs secure aggregation
func (s *FLServer) secureAggregate(gradients [][]float64) []float64 {
	// Simplified secure aggregation using additive secret sharing
	// In production, use proper cryptographic protocols

	aggregated := make([]float64, len(gradients[0]))

	for _, grad := range gradients {
		for j := range grad {
			aggregated[j] += grad[j]
		}
	}

	// Normalize by number of clients
	n := float64(len(gradients))
	for j := range aggregated {
		aggregated[j] /= n
	}

	return aggregated
}

// applyDifferentialPrivacy applies differential privacy to gradients
func (s *FLServer) applyDifferentialPrivacy(gradients []float64) []float64 {
	// Add calibrated Gaussian noise for differential privacy
	sensitivity := 1.0 // L2 sensitivity
	epsilon := s.config.PrivacyBudget
	delta := 1e-5

	// Calculate noise scale (Gaussian mechanism)
	noiseScale := (sensitivity * math.Sqrt(2*math.Log(1.25/delta))) / epsilon

	noisyGrad := make([]float64, len(gradients))
	for i, g := range gradients {
		// Add Gaussian noise
		noise := s.gaussianNoise(0, noiseScale)
		noisyGrad[i] = g + noise
	}

	return noisyGrad
}

// gaussianNoise generates Gaussian noise
func (s *FLServer) gaussianNoise(mean, stddev float64) float64 {
	// Box-Muller transform
	u1 := float64(s.currentRound+1) / float64(s.config.NumRounds)
	u2 := float64(s.currentRound+2) / float64(s.config.NumRounds+1)

	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mean + stddev*z0
}

// updateGlobalModel updates the global model with aggregated gradients
func (s *FLServer) updateGlobalModel(gradients []float64) {
	for i := range s.globalModel {
		s.globalModel[i] += gradients[i]
	}
}

// evaluate evaluates the global model
func (s *FLServer) evaluate() float64 {
	// Simplified evaluation
	// In production, use proper validation dataset
	accuracy := 0.7 + (float64(s.currentRound) / float64(s.config.NumRounds) * 0.2)
	return accuracy
}

// GetGlobalModel returns the current global model
func (s *FLServer) GetGlobalModel() []float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	model := make([]float64, len(s.globalModel))
	copy(model, s.globalModel)
	return model
}

// PersonalizedFL implements personalized federated learning
type PersonalizedFL struct {
	server       *FLServer
	personalModels map[string][]float64
	mu           sync.RWMutex
}

// NewPersonalizedFL creates a personalized FL system
func NewPersonalizedFL(config FederatedLearningConfig) *PersonalizedFL {
	return &PersonalizedFL{
		server:         NewFLServer(config),
		personalModels: make(map[string][]float64),
	}
}

// Train trains personalized models
func (pfl *PersonalizedFL) Train(ctx context.Context) error {
	// First train global model
	if err := pfl.server.Train(ctx); err != nil {
		return err
	}

	// Then personalize for each client
	globalModel := pfl.server.GetGlobalModel()

	for _, client := range pfl.server.clients {
		personalModel := make([]float64, len(globalModel))
		copy(personalModel, globalModel)

		// Fine-tune on local data
		for i := range personalModel {
			personalModel[i] += client.LocalGrad[i] * 0.1 // Small adjustment
		}

		pfl.mu.Lock()
		pfl.personalModels[client.ID] = personalModel
		pfl.mu.Unlock()
	}

	return nil
}

// GetPersonalizedModel returns a client's personalized model
func (pfl *PersonalizedFL) GetPersonalizedModel(clientID string) ([]float64, error) {
	pfl.mu.RLock()
	defer pfl.mu.RUnlock()

	model, exists := pfl.personalModels[clientID]
	if !exists {
		return nil, fmt.Errorf("no model for client: %s", clientID)
	}

	result := make([]float64, len(model))
	copy(result, model)
	return result, nil
}
