package integration_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"novacron/backend/core/network/dwcp/v3"
	"novacron/backend/ml"
	"novacron/backend/api"
)

// TestFullStackIntegration tests complete end-to-end flow
func TestFullStackIntegration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Setup test environment
	env := setupTestEnvironment(t)
	defer env.Cleanup()

	t.Run("CompleteTransactionFlow", func(t *testing.T) {
		// Step 1: Client submits transaction via API
		client := api.NewClient(env.APIEndpoint)
		tx := &api.Transaction{
			From:   "user-123",
			To:     "user-456",
			Amount: 100.0,
			Data:   map[string]interface{}{"memo": "test-payment"},
		}

		submitResp, err := client.SubmitTransaction(ctx, tx)
		require.NoError(t, err)
		assert.NotEmpty(t, submitResp.TxID)

		// Step 2: ML models predict optimal routing
		mlService := env.MLService
		prediction, err := mlService.PredictOptimalRoute(ctx, &ml.RouteRequest{
			TransactionID: submitResp.TxID,
			Source:        tx.From,
			Destination:   tx.To,
			Priority:      "normal",
		})
		require.NoError(t, err)
		assert.NotEmpty(t, prediction.Route)
		assert.Greater(t, prediction.Confidence, 0.8, "ML confidence should be high")

		// Step 3: Consensus protocol commits transaction
		dwcpNode := env.DWCPNodes[0]
		block := &dwcp.Block{
			Height:    env.GetNextBlockHeight(),
			Timestamp: time.Now().Unix(),
			Transactions: []dwcp.Transaction{
				{
					ID:   submitResp.TxID,
					Data: tx.Data,
				},
			},
		}

		consensus, err := dwcpNode.ProposeBlock(ctx, block)
		require.NoError(t, err)
		assert.True(t, consensus, "Consensus should be achieved")

		// Step 4: State is replicated across nodes
		time.Sleep(2 * time.Second) // Allow replication

		for i, node := range env.DWCPNodes {
			state := node.GetState()
			assert.Equal(t, block.Height, state.Height, "Node %d height mismatch", i)
			assert.Contains(t, state.Transactions, submitResp.TxID, "Node %d missing tx", i)
		}

		// Step 5: Client receives confirmation
		status, err := client.GetTransactionStatus(ctx, submitResp.TxID)
		require.NoError(t, err)
		assert.Equal(t, "confirmed", status.Status)
		assert.Equal(t, block.Height, status.BlockHeight)
	})

	t.Run("MLDrivenConsensus", func(t *testing.T) {
		// Test ML model integration with consensus
		mlService := env.MLService

		// Submit multiple transactions
		txIDs := make([]string, 10)
		for i := 0; i < 10; i++ {
			tx := &api.Transaction{
				From:   "batch-sender",
				To:     "batch-receiver",
				Amount: float64(i * 10),
			}
			resp, err := env.APIClient.SubmitTransaction(ctx, tx)
			require.NoError(t, err)
			txIDs[i] = resp.TxID
		}

		// ML predicts batch processing order
		batchPrediction, err := mlService.OptimizeBatch(ctx, &ml.BatchRequest{
			TransactionIDs: txIDs,
			Optimization:   "latency",
		})
		require.NoError(t, err)
		assert.Equal(t, len(txIDs), len(batchPrediction.Order))

		// Process in ML-optimized order
		orderedTxs := reorderTransactions(txIDs, batchPrediction.Order)
		block := &dwcp.Block{
			Height:       env.GetNextBlockHeight(),
			Timestamp:    time.Now().Unix(),
			Transactions: orderedTxs,
		}

		start := time.Now()
		consensus, err := env.DWCPNodes[0].ProposeBlock(ctx, block)
		duration := time.Since(start)

		require.NoError(t, err)
		assert.True(t, consensus)
		assert.Less(t, duration, 2*time.Second, "ML optimization should reduce latency")
	})

	t.Run("MultiProtocolConsensus", func(t *testing.T) {
		// Test switching between consensus protocols based on workload
		protocols := []string{"ProBFT", "Bullshark", "T-PBFT"}

		for _, protocol := range protocols {
			t.Run(protocol, func(t *testing.T) {
				// Reconfigure network for protocol
				env.ReconfigureProtocol(protocol)

				tx := &api.Transaction{
					From:   "protocol-test",
					To:     "protocol-receiver",
					Amount: 50.0,
				}

				resp, err := env.APIClient.SubmitTransaction(ctx, tx)
				require.NoError(t, err)

				// Wait for consensus
				time.Sleep(1 * time.Second)

				status, err := env.APIClient.GetTransactionStatus(ctx, resp.TxID)
				require.NoError(t, err)
				assert.Equal(t, "confirmed", status.Status)
				assert.Equal(t, protocol, status.ConsensusProtocol)
			})
		}
	})

	t.Run("FailoverAndRecovery", func(t *testing.T) {
		// Simulate node failure during transaction
		tx := &api.Transaction{
			From:   "failover-test",
			To:     "failover-receiver",
			Amount: 75.0,
		}

		resp, err := env.APIClient.SubmitTransaction(ctx, tx)
		require.NoError(t, err)

		// Kill primary node
		env.DWCPNodes[0].Stop()
		time.Sleep(500 * time.Millisecond)

		// Secondary node should take over
		status, err := env.APIClient.GetTransactionStatus(ctx, resp.TxID)
		require.NoError(t, err)
		assert.Equal(t, "confirmed", status.Status, "Failover should complete tx")

		// Restart primary and verify sync
		env.DWCPNodes[0].Start(ctx)
		time.Sleep(2 * time.Second)

		primaryState := env.DWCPNodes[0].GetState()
		secondaryState := env.DWCPNodes[1].GetState()
		assert.Equal(t, secondaryState.Height, primaryState.Height, "States should sync")
	})

	t.Run("HighLoadStressTest", func(t *testing.T) {
		// Stress test with high transaction volume
		txCount := 1000
		client := env.APIClient

		// Submit transactions concurrently
		results := make(chan error, txCount)
		start := time.Now()

		for i := 0; i < txCount; i++ {
			go func(id int) {
				tx := &api.Transaction{
					From:   "stress-sender",
					To:     "stress-receiver",
					Amount: float64(id),
				}
				_, err := client.SubmitTransaction(ctx, tx)
				results <- err
			}(i)
		}

		// Collect results
		errorCount := 0
		for i := 0; i < txCount; i++ {
			if err := <-results; err != nil {
				errorCount++
			}
		}
		duration := time.Since(start)

		// Verify performance
		assert.Less(t, errorCount, txCount/100, "Error rate should be < 1%")
		tps := float64(txCount) / duration.Seconds()
		assert.Greater(t, tps, 100.0, "Should handle > 100 tx/s")

		t.Logf("Processed %d transactions in %v (%.2f tx/s)", txCount, duration, tps)
	})

	t.Run("MLModelAccuracy", func(t *testing.T) {
		// Test ML model prediction accuracy
		mlService := env.MLService

		// Generate test dataset
		testCases := generateMLTestCases(100)
		correctPredictions := 0

		for _, tc := range testCases {
			prediction, err := mlService.PredictOptimalRoute(ctx, tc.Input)
			require.NoError(t, err)

			if prediction.Route == tc.ExpectedRoute {
				correctPredictions++
			}
		}

		accuracy := float64(correctPredictions) / float64(len(testCases))
		assert.Greater(t, accuracy, 0.9, "ML accuracy should be > 90%")
		t.Logf("ML Model Accuracy: %.2f%%", accuracy*100)
	})
}

// TestEnvironment encapsulates test infrastructure
type TestEnvironment struct {
	APIEndpoint  string
	APIClient    *api.Client
	MLService    *ml.Service
	DWCPNodes    []*dwcp.Node
	blockHeight  int64
	t            *testing.T
}

func setupTestEnvironment(t *testing.T) *TestEnvironment {
	ctx := context.Background()

	// Setup DWCP nodes
	nodeCount := 10
	nodes := make([]*dwcp.Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:     i,
			TotalNodes: nodeCount,
			Protocol:   "ProBFT",
		})
		go nodes[i].Start(ctx)
	}
	time.Sleep(1 * time.Second)

	// Setup ML service
	mlService := ml.NewService(&ml.Config{
		ModelPath: "/tmp/test-models",
		Features:  []string{"latency", "throughput", "reliability"},
	})

	// Setup API server
	apiServer := api.NewServer(&api.Config{
		Port:        8080,
		DWCPNodes:   nodes,
		MLService:   mlService,
	})
	go apiServer.Start()
	time.Sleep(500 * time.Millisecond)

	return &TestEnvironment{
		APIEndpoint:  "http://localhost:8080",
		APIClient:    api.NewClient("http://localhost:8080"),
		MLService:    mlService,
		DWCPNodes:    nodes,
		blockHeight:  0,
		t:            t,
	}
}

func (env *TestEnvironment) GetNextBlockHeight() int64 {
	env.blockHeight++
	return env.blockHeight
}

func (env *TestEnvironment) ReconfigureProtocol(protocol string) {
	for _, node := range env.DWCPNodes {
		node.SetProtocol(protocol)
	}
	time.Sleep(500 * time.Millisecond)
}

func (env *TestEnvironment) Cleanup() {
	for _, node := range env.DWCPNodes {
		node.Stop()
	}
}

func reorderTransactions(txIDs []string, order []int) []dwcp.Transaction {
	result := make([]dwcp.Transaction, len(txIDs))
	for i, idx := range order {
		result[i] = dwcp.Transaction{ID: txIDs[idx]}
	}
	return result
}

func generateMLTestCases(count int) []struct {
	Input         *ml.RouteRequest
	ExpectedRoute string
} {
	// Generate synthetic test cases
	cases := make([]struct {
		Input         *ml.RouteRequest
		ExpectedRoute string
	}, count)

	for i := 0; i < count; i++ {
		cases[i] = struct {
			Input         *ml.RouteRequest
			ExpectedRoute string
		}{
			Input: &ml.RouteRequest{
				TransactionID: "test-" + string(rune(i)),
				Source:        "src-" + string(rune(i%10)),
				Destination:   "dst-" + string(rune((i+1)%10)),
				Priority:      "normal",
			},
			ExpectedRoute: "route-optimal",
		}
	}

	return cases
}
