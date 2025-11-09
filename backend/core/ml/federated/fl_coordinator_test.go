package federated

import (
	"context"
	"testing"
	"time"
)

func TestFederatedLearning(t *testing.T) {
	config := &FederatedConfig{
		NumClients:     10,
		ClientFraction: 0.5,
		LocalEpochs:    3,
		GlobalRounds:   5,
		PrivacyBudget:  1.0,
		MinClients:     3,
		Timeout:        30 * time.Second,
	}

	coordinator := NewFLCoordinator(config)

	// Register clients
	for i := 0; i < config.NumClients; i++ {
		clientID := fmt.Sprintf("client_%d", i)
		err := coordinator.RegisterClient(clientID, "region1", 100+i*10)
		if err != nil {
			t.Fatalf("failed to register client: %v", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	err := coordinator.Train(ctx)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	globalModel := coordinator.GetGlobalModel()
	if globalModel.Version < config.GlobalRounds {
		t.Logf("Converged early at version %d", globalModel.Version)
	}

	history := coordinator.GetHistory()
	if len(history) == 0 {
		t.Error("no training history")
	}

	lastRound := history[len(history)-1]
	t.Logf("Final accuracy: %.4f, loss: %.4f",
		lastRound.GlobalAccuracy, lastRound.GlobalLoss)

	if lastRound.GlobalAccuracy < 0.6 {
		t.Error("final accuracy too low")
	}
}

func TestDifferentialPrivacy(t *testing.T) {
	pm := NewPrivacyMechanism(1.0, 1e-5)

	weights := [][]float64{
		{0.5, 0.3},
		{0.2, 0.8},
	}

	noisy := pm.AddNoise(weights)

	// Check that noise was added
	different := false
	for i := range weights {
		for j := range weights[i] {
			if weights[i][j] != noisy[i][j] {
				different = true
				break
			}
		}
	}

	if !different {
		t.Error("no noise was added")
	}

	t.Logf("Privacy mechanism: epsilon=%.2f, noise scale=%.4f",
		pm.epsilon, pm.noiseScale)
}
