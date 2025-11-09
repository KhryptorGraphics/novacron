package rl_routing

import (
	"context"
	"testing"
	"time"
)

func TestDQNRouterInitialization(t *testing.T) {
	topology := map[string][]string{
		"node1": {"node2", "node3"},
		"node2": {"node1", "node3", "node4"},
		"node3": {"node1", "node2", "node4"},
		"node4": {"node2", "node3"},
	}

	router := NewDQNRouter(topology)
	err := router.Initialize()

	if err != nil {
		t.Errorf("Failed to initialize DQN router: %v", err)
	}

	if router.qNetwork == nil {
		t.Error("Q-network not initialized")
	}

	if router.targetNetwork == nil {
		t.Error("Target network not initialized")
	}
}

func TestMakeRoutingDecision(t *testing.T) {
	topology := map[string][]string{
		"node1": {"node2", "node3"},
		"node2": {"node1", "node3", "node4"},
		"node3": {"node1", "node2", "node4"},
		"node4": {"node2", "node3"},
	}

	router := NewDQNRouter(topology)
	router.Initialize()

	state := State{
		CurrentNode:    "node1",
		DestNode:       "node4",
		LinkLatencies:  map[string]float64{"node1-node2": 10, "node1-node3": 15},
		LinkBandwidth:  map[string]float64{"node1-node2": 1000, "node1-node3": 800},
		PacketPriority: 5,
		QueueDepths:    map[string]int{"node1-node2": 10, "node1-node3": 5},
		TimeOfDay:      14,
	}

	ctx := context.Background()
	start := time.Now()
	action, err := router.MakeRoutingDecision(ctx, state)
	latency := time.Since(start)

	if err != nil {
		t.Errorf("Routing decision failed: %v", err)
	}

	if action.NextHop == "" {
		t.Error("No next hop selected")
	}

	// Check latency requirement (<1ms)
	if latency > 1*time.Millisecond {
		t.Errorf("Routing decision too slow: %v (target: <1ms)", latency)
	}

	t.Logf("Routing decision made in %v", latency)
	t.Logf("Selected next hop: %s", action.NextHop)
}

func TestExperienceReplay(t *testing.T) {
	topology := map[string][]string{
		"node1": {"node2"},
		"node2": {"node1"},
	}

	router := NewDQNRouter(topology)
	router.Initialize()

	// Add experiences
	for i := 0; i < 100; i++ {
		exp := Experience{
			State: State{
				CurrentNode: "node1",
				DestNode:    "node2",
			},
			Action: Action{
				NextHop: "node2",
				LinkID:  "node1-node2",
			},
			Reward:    -0.01,
			NextState: State{CurrentNode: "node2"},
			Done:      true,
			Timestamp: time.Now(),
		}
		router.AddExperience(exp)
	}

	if len(router.replayBuffer) < 100 {
		t.Errorf("Experience buffer size incorrect: got %d, want >= 100", len(router.replayBuffer))
	}

	// Test training
	router.Train()

	// Verify step count updated
	if router.stepCount == 0 {
		t.Error("Training did not update step count")
	}
}

func TestRewardCalculation(t *testing.T) {
	router := NewDQNRouter(nil)

	tests := []struct {
		name       string
		latency    float64
		packetLoss float64
		bandwidth  float64
		wantSign   bool // true for positive, false for negative
	}{
		{"Low latency, no loss", 10, 0, 1000, true},
		{"High latency, high loss", 100, 0.05, 100, false},
		{"Medium performance", 50, 0.01, 500, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reward := router.CalculateReward(
				Action{},
				tt.latency,
				tt.packetLoss,
				tt.bandwidth,
			)

			if tt.wantSign && reward < 0 {
				t.Errorf("Expected positive reward, got %f", reward)
			} else if !tt.wantSign && reward > 0 {
				t.Errorf("Expected negative reward, got %f", reward)
			}

			t.Logf("Reward: %f", reward)
		})
	}
}

func TestEpsilonGreedyExploration(t *testing.T) {
	topology := map[string][]string{
		"node1": {"node2", "node3"},
		"node2": {"node1", "node3"},
		"node3": {"node1", "node2"},
	}

	router := NewDQNRouter(topology)
	router.Initialize()
	router.epsilon = 1.0 // Force exploration

	state := State{
		CurrentNode: "node1",
		DestNode:    "node3",
	}

	ctx := context.Background()

	// Test exploration (random selection)
	actions := make(map[string]int)
	for i := 0; i < 100; i++ {
		action, _ := router.MakeRoutingDecision(ctx, state)
		actions[action.NextHop]++
	}

	// Should have selected different neighbors
	if len(actions) < 2 {
		t.Error("Exploration not working - always selecting same action")
	}

	// Test exploitation
	router.epsilon = 0.0 // Force exploitation

	exploitActions := make(map[string]int)
	for i := 0; i < 100; i++ {
		action, _ := router.MakeRoutingDecision(ctx, state)
		exploitActions[action.NextHop]++
	}

	// Should consistently select best action
	maxCount := 0
	for _, count := range exploitActions {
		if count > maxCount {
			maxCount = count
		}
	}

	if maxCount < 90 {
		t.Error("Exploitation not consistent")
	}
}

func TestGetMetrics(t *testing.T) {
	router := NewDQNRouter(nil)
	router.Initialize()

	metrics := router.GetMetrics()

	expectedMetrics := []string{
		"avg_decision_time_us",
		"decision_count",
		"success_rate",
		"epsilon",
		"replay_buffer_size",
		"step_count",
	}

	for _, metric := range expectedMetrics {
		if _, exists := metrics[metric]; !exists {
			t.Errorf("Missing metric: %s", metric)
		}
	}
}

// Benchmarks

func BenchmarkMakeRoutingDecision(b *testing.B) {
	topology := map[string][]string{
		"node1": {"node2", "node3", "node4"},
		"node2": {"node1", "node3", "node4"},
		"node3": {"node1", "node2", "node4"},
		"node4": {"node1", "node2", "node3"},
	}

	router := NewDQNRouter(topology)
	router.Initialize()

	state := State{
		CurrentNode:   "node1",
		DestNode:      "node4",
		LinkLatencies: map[string]float64{"node1-node2": 10, "node1-node3": 15},
		LinkBandwidth: map[string]float64{"node1-node2": 1000, "node1-node3": 800},
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.MakeRoutingDecision(ctx, state)
	}
}

func BenchmarkTrain(b *testing.B) {
	topology := map[string][]string{
		"node1": {"node2", "node3"},
		"node2": {"node1", "node3"},
		"node3": {"node1", "node2"},
	}

	router := NewDQNRouter(topology)
	router.Initialize()

	// Fill replay buffer
	for i := 0; i < 1000; i++ {
		exp := Experience{
			State: State{
				CurrentNode: "node1",
				DestNode:    "node3",
			},
			Action: Action{
				NextHop: "node2",
			},
			Reward:    -0.01,
			NextState: State{CurrentNode: "node2"},
			Done:      false,
		}
		router.AddExperience(exp)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.Train()
	}
}

func BenchmarkStateToFeatures(b *testing.B) {
	router := NewDQNRouter(nil)
	router.Initialize()

	state := State{
		CurrentNode:   "node1",
		DestNode:      "node4",
		LinkLatencies: map[string]float64{"link1": 10, "link2": 15},
		LinkBandwidth: map[string]float64{"link1": 1000, "link2": 800},
		QueueDepths:   map[string]int{"queue1": 10, "queue2": 20},
		TimeOfDay:     14,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = router.stateToFeatures(state)
	}
}