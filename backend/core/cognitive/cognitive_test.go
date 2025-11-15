// Package cognitive_test provides comprehensive tests for cognitive AI
package cognitive_test

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cognitive"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/advisor"
	contextpkg "github.com/khryptorgraphics/novacron/backend/core/cognitive/context"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/explanation"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/knowledge"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	// TODO: Create cognitive/memory package
	// "github.com/khryptorgraphics/novacron/backend/core/cognitive/memory"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/metrics"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/multimodal"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/nli"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/parser"
	"github.com/khryptorgraphics/novacron/backend/core/cognitive/reasoning"
)

// TestIntentParserAccuracy tests intent parsing accuracy
func TestIntentParserAccuracy(t *testing.T) {
	llmClient := &parser.SimpleLLMClient{}
	intentParser := parser.NewIntentParser(llmClient)

	testCases := []struct {
		input          string
		expectedAction string
		minConfidence  float64
	}{
		{"Deploy a VM in us-east-1", "deploy", 0.8},
		{"Migrate all VMs from AWS to GCP", "migrate", 0.8},
		{"Why is my app slow?", "diagnose", 0.7},
		{"Show me all VMs", "query", 0.8},
		{"Scale up the web service", "scale", 0.8},
	}

	ctx := context.Background()
	successCount := 0

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			intent, err := intentParser.Parse(ctx, tc.input, nil)
			require.NoError(t, err)
			assert.NotNil(t, intent)
			assert.GreaterOrEqual(t, intent.Confidence, tc.minConfidence)
			if intent.Action == tc.expectedAction && intent.Confidence >= tc.minConfidence {
				successCount++
			}
		})
	}

	accuracy := float64(successCount) / float64(len(testCases))
	assert.GreaterOrEqual(t, accuracy, 0.95, "Intent parsing accuracy should be >= 95%%")
}

// TestNaturalLanguageInterface tests NLI end-to-end
func TestNaturalLanguageInterface(t *testing.T) {
	config := cognitive.DefaultCognitiveConfig()
	llmClient := &nli.MockLLMClient{}
	intentParser := parser.NewIntentParser(&parser.SimpleLLMClient{})
	nliInterface := nli.NewNaturalLanguageInterface(config, llmClient, intentParser)

	ctx := context.Background()
	userID := "test-user"
	sessionID := ""

	// Test conversation flow
	t.Run("ProcessMessage", func(t *testing.T) {
		resp, err := nliInterface.ProcessMessage(ctx, userID, sessionID, "Deploy a VM in us-east-1")
		require.NoError(t, err)
		assert.NotNil(t, resp)
		assert.True(t, resp.Success)
		assert.NotEmpty(t, resp.SessionID)
		assert.LessOrEqual(t, resp.Latency, int64(100), "Response latency should be <= 100ms")
		sessionID = resp.SessionID
	})

	t.Run("MultiTurnConversation", func(t *testing.T) {
		// Second turn
		resp, err := nliInterface.ProcessMessage(ctx, userID, sessionID, "Make it t2.large")
		require.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, sessionID, resp.SessionID)
	})

	t.Run("MetricsValidation", func(t *testing.T) {
		metrics := nliInterface.GetMetrics()
		assert.NotNil(t, metrics)
		assert.GreaterOrEqual(t, metrics.IntentAccuracy, 0.8)
		assert.LessOrEqual(t, metrics.AvgResponseLatency, 100.0)
	})
}

// TestReasoningEngine tests logical reasoning
func TestReasoningEngine(t *testing.T) {
	config := cognitive.DefaultCognitiveConfig()
	engine := reasoning.NewReasoningEngine(config)

	ctx := context.Background()

	t.Run("BasicReasoning", func(t *testing.T) {
		facts := []string{
			"hasHighLatency(app-frontend)",
			"hasDatabaseDependency(app-frontend, db-main)",
			"connectionPoolExhausted(db-main)",
		}

		result, err := engine.Reason(ctx, "What should I do?", facts)
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.NotEmpty(t, result.Conclusion)
		assert.GreaterOrEqual(t, result.Confidence, 0.8)
		assert.NotEmpty(t, result.Explanation)
	})

	t.Run("ReasoningLatency", func(t *testing.T) {
		startTime := time.Now()
		facts := []string{"hasHighCost(workload-1)", "isStateless(workload-1)"}
		_, err := engine.Reason(ctx, "How to reduce cost?", facts)
		latency := time.Since(startTime)

		require.NoError(t, err)
		assert.LessOrEqual(t, latency, 100*time.Millisecond, "Reasoning latency should be < 100ms")
	})

	t.Run("ReasoningCorrectness", func(t *testing.T) {
		metrics := engine.GetMetrics()
		correctnessRate := float64(metrics.SuccessfulInferences) / float64(metrics.TotalInferences)
		assert.GreaterOrEqual(t, correctnessRate, 0.9, "Reasoning correctness should be >= 90%%")
	})
}

// TestKnowledgeGraph tests knowledge graph operations
func TestKnowledgeGraph(t *testing.T) {
	config := cognitive.DefaultCognitiveConfig()
	driver := knowledge.NewMockGraphDriver()
	kg := knowledge.NewKnowledgeGraph(config, driver)

	ctx := context.Background()

	t.Run("ConnectAndAddEntity", func(t *testing.T) {
		err := kg.Connect(ctx)
		require.NoError(t, err)

		entity := &cognitive.KnowledgeEntity{
			ID:   "vm-123",
			Type: "VM",
			Properties: map[string]interface{}{
				"region": "us-east-1",
				"size":   "t2.medium",
			},
		}

		err = kg.AddEntity(ctx, entity)
		require.NoError(t, err)
	})

	t.Run("AddRelation", func(t *testing.T) {
		entity2 := &cognitive.KnowledgeEntity{
			ID:   "db-456",
			Type: "Database",
			Properties: map[string]interface{}{
				"engine": "postgres",
			},
		}
		require.NoError(t, kg.AddEntity(ctx, entity2))

		relation := &cognitive.KnowledgeRelation{
			ID:   "rel-1",
			Type: "Depends-On",
			From: "vm-123",
			To:   "db-456",
			Properties: map[string]interface{}{
				"strength": "strong",
			},
		}

		err := kg.AddRelation(ctx, relation)
		require.NoError(t, err)
	})

	t.Run("QueryAndRetrieve", func(t *testing.T) {
		entity, err := kg.GetEntity(ctx, "vm-123")
		require.NoError(t, err)
		assert.NotNil(t, entity)
		assert.Equal(t, "VM", entity.Type)
	})

	t.Run("BestPractices", func(t *testing.T) {
		err := kg.AddBestPractice(ctx, "security", "Enable Encryption", "Always enable encryption at rest")
		require.NoError(t, err)

		practices, err := kg.GetBestPractices(ctx, "security")
		require.NoError(t, err)
		assert.NotEmpty(t, practices)
	})
}

// TestContextManager tests context management
func TestContextManager(t *testing.T) {
	cm := contextpkg.NewContextManager()

	userID := "user-123"

	t.Run("UserContext", func(t *testing.T) {
		cm.UpdateUserContext(userID, func(ctx *contextpkg.UserContext) {
			ctx.Role = "admin"
			ctx.Preferences["theme"] = "dark"
		})

		userCtx := cm.GetUserContext(userID)
		assert.NotNil(t, userCtx)
		assert.Equal(t, "admin", userCtx.Role)
		assert.Equal(t, "dark", userCtx.Preferences["theme"])
	})

	t.Run("RecordUserAction", func(t *testing.T) {
		cm.RecordUserAction(userID, "deploy_vm", map[string]interface{}{"vm_id": "vm-123"})

		userCtx := cm.GetUserContext(userID)
		assert.NotEmpty(t, userCtx.History)
	})

	t.Run("ContextSwitch", func(t *testing.T) {
		// Create second user
		user2ID := "user-456"
		cm.UpdateUserContext(user2ID, func(ctx *contextpkg.UserContext) {
			ctx.Role = "developer"
		})

		startTime := time.Now()
		err := cm.SwitchContext(context.Background(), userID, user2ID)
		latency := time.Since(startTime)

		require.NoError(t, err)
		assert.LessOrEqual(t, latency, 10*time.Millisecond, "Context switch latency should be < 10ms")
	})

	t.Run("FullContext", func(t *testing.T) {
		fullCtx := cm.GetFullContext(userID)
		assert.NotNil(t, fullCtx.User)
		assert.NotNil(t, fullCtx.System)
		assert.NotNil(t, fullCtx.Business)
		assert.NotNil(t, fullCtx.Temporal)
		assert.NotNil(t, fullCtx.Geospatial)
	})
}

// TestProactiveAdvisor tests recommendation generation
func TestProactiveAdvisor(t *testing.T) {
	config := cognitive.DefaultCognitiveConfig()
	pa := advisor.NewProactiveAdvisor(config)

	ctx := context.Background()

	t.Run("GenerateRecommendations", func(t *testing.T) {
		systemState := map[string]interface{}{
			"cost":     12000,
			"vms":      50,
			"security": "medium",
		}

		recommendations, err := pa.AnalyzeAndRecommend(ctx, systemState)
		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Verify all recommendations meet confidence threshold
		for _, rec := range recommendations {
			assert.GreaterOrEqual(t, rec.Confidence, config.MinConfidenceScore)
		}
	})

	t.Run("RecommendationTypes", func(t *testing.T) {
		systemState := map[string]interface{}{}
		recommendations, err := pa.AnalyzeAndRecommend(ctx, systemState)
		require.NoError(t, err)

		// Should have recommendations from different analyzers
		typesSeen := make(map[string]bool)
		for _, rec := range recommendations {
			typesSeen[rec.Type] = true
		}

		assert.True(t, len(typesSeen) >= 2, "Should have multiple recommendation types")
	})

	t.Run("RecommendationRelevance", func(t *testing.T) {
		// Simulate acceptance
		pa.RecordAcceptance("rec-1", true)
		pa.RecordAcceptance("rec-2", true)
		pa.RecordAcceptance("rec-3", false)

		metrics := pa.GetMetrics()
		acceptanceRate := float64(metrics.AcceptedRecommendations) / float64(metrics.TotalRecommendations)
		assert.GreaterOrEqual(t, acceptanceRate, 0.5)
	})
}

// TestConversationalMemory tests memory with RAG
func TestConversationalMemory(t *testing.T) {
	config := cognitive.DefaultCognitiveConfig()
	embedder := &memory.MockEmbedder{}
	vectorStore := memory.NewMockVectorStore()
	mem := memory.NewConversationalMemory(config, embedder, vectorStore)

	ctx := context.Background()

	t.Run("StoreAndRetrieve", func(t *testing.T) {
		err := mem.Store(ctx, "Deploy VM in us-east-1", "conversation", map[string]interface{}{"user": "test"})
		require.NoError(t, err)

		memories, err := mem.Retrieve(ctx, "deployment", 5)
		require.NoError(t, err)
		assert.NotEmpty(t, memories)
	})

	t.Run("EpisodicMemory", func(t *testing.T) {
		episode := &memory.Episode{
			ID:        "episode-1",
			SessionID: "session-1",
			Summary:   "Successful VM deployment",
			Learnings: []string{"User prefers t2.large instances", "Prefer us-east-1 region"},
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now(),
		}

		err := mem.StoreEpisode(ctx, episode)
		require.NoError(t, err)

		retrieved, err := mem.GetEpisode("episode-1")
		require.NoError(t, err)
		assert.Equal(t, episode.Summary, retrieved.Summary)
	})

	t.Run("PatternLearning", func(t *testing.T) {
		pattern := &memory.Pattern{
			ID:          "pattern-1",
			Description: "User always deploys in us-east-1",
			Frequency:   1,
			Confidence:  0.8,
		}

		mem.LearnPattern(pattern)
		patterns := mem.GetPatterns(0.7)
		assert.NotEmpty(t, patterns)
	})

	t.Run("Preferences", func(t *testing.T) {
		mem.SetPreference("default_region", "us-east-1")
		mem.SetPreference("instance_type", "t2.medium")

		region, exists := mem.GetPreference("default_region")
		assert.True(t, exists)
		assert.Equal(t, "us-east-1", region)
	})
}

// TestMultiModalInterface tests multi-modal input/output
func TestMultiModalInterface(t *testing.T) {
	config := &multimodal.InterfaceConfig{
		EnableVoice:   true,
		EnableVision:  true,
		EnableGesture: true,
	}
	mmi := multimodal.NewMultiModalInterface(config)

	ctx := context.Background()

	t.Run("TextInput", func(t *testing.T) {
		result, err := mmi.ProcessInput(ctx, "text", []byte("Deploy a VM"))
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "text", result.InputType)
	})

	t.Run("VoiceInput", func(t *testing.T) {
		result, err := mmi.ProcessInput(ctx, "voice", []byte("audio data"))
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.NotEmpty(t, result.TranscribedText)
	})

	t.Run("ImageInput", func(t *testing.T) {
		result, err := mmi.ProcessInput(ctx, "image", []byte("image data"))
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.NotNil(t, result.ImageAnalysis)
	})

	t.Run("TextOutput", func(t *testing.T) {
		output, err := mmi.GenerateOutput(ctx, "text", "Hello, world!")
		require.NoError(t, err)
		assert.NotNil(t, output)
	})

	t.Run("VoiceOutput", func(t *testing.T) {
		output, err := mmi.GenerateOutput(ctx, "voice", "Hello, world!")
		require.NoError(t, err)
		assert.NotNil(t, output)
	})
}

// TestExplanationGenerator tests explanation generation
func TestExplanationGenerator(t *testing.T) {
	llmClient := &explanation.SimpleLLMClient{}
	eg := explanation.NewExplanationGenerator(llmClient)

	ctx := context.Background()

	t.Run("ExplainDecision", func(t *testing.T) {
		reasoning := &cognitive.ReasoningResult{
			Conclusion: "Scale out VM",
			Confidence: 0.92,
			Steps: []cognitive.ReasoningStep{
				{
					Rule:       "High CPU triggers scale out",
					Conclusion: "Scale out recommended",
					Confidence: 0.92,
				},
			},
		}

		explanation, err := eg.ExplainDecision(ctx, "Scale out", reasoning)
		require.NoError(t, err)
		assert.NotEmpty(t, explanation)
	})

	t.Run("ExplainWhatIf", func(t *testing.T) {
		explanation, err := eg.ExplainWhatIf(ctx, "Migrate to GCP", "Currently on AWS", map[string]interface{}{"cost": "$10k/mo"})
		require.NoError(t, err)
		assert.NotEmpty(t, explanation)
	})
}

// TestMetricsCollector tests metrics tracking
func TestMetricsCollector(t *testing.T) {
	mc := metrics.NewMetricsCollector()

	t.Run("RecordMetrics", func(t *testing.T) {
		mc.RecordIntentParsing(0.95, true)
		mc.RecordResponseLatency(50.0)
		mc.RecordReasoningResult(true, 0.92)
		mc.RecordRecommendation(true)
		mc.RecordContextSwitch(5.0)
		mc.RecordUserSatisfaction(4.8)
	})

	t.Run("ValidateMetrics", func(t *testing.T) {
		report := mc.ValidateMetrics()
		assert.NotNil(t, report)
		assert.NotEmpty(t, report.Checks)

		// Check individual targets
		for _, check := range report.Checks {
			t.Logf("%s: %.2f (target: %.2f) - Pass: %v", check.Metric, check.Current, check.Target, check.Pass)
		}
	})

	t.Run("ExportMetrics", func(t *testing.T) {
		exported := mc.ExportMetrics()
		assert.NotNil(t, exported)
		assert.Contains(t, exported, "intent_accuracy")
		assert.Contains(t, exported, "task_completion_rate")
	})
}

// TestEndToEndCognitiveFlow tests complete cognitive AI workflow
func TestEndToEndCognitiveFlow(t *testing.T) {
	// Setup complete cognitive system
	config := cognitive.DefaultCognitiveConfig()
	llmClient := &nli.MockLLMClient{}
	intentParser := parser.NewIntentParser(&parser.SimpleLLMClient{})
	nliInterface := nli.NewNaturalLanguageInterface(config, llmClient, intentParser)
	reasoningEngine := reasoning.NewReasoningEngine(config)
	proactiveAdvisor := advisor.NewProactiveAdvisor(config)
	metricsCollector := metrics.NewMetricsCollector()

	ctx := context.Background()

	t.Run("CompleteUserInteraction", func(t *testing.T) {
		// User input
		startTime := time.Now()
		resp, err := nliInterface.ProcessMessage(ctx, "user-1", "", "My app is slow")
		require.NoError(t, err)
		assert.True(t, resp.Success)

		// Record metrics
		metricsCollector.RecordIntentParsing(resp.Intent.Confidence, true)
		metricsCollector.RecordResponseLatency(float64(time.Since(startTime).Milliseconds()))

		// Reasoning
		facts := []string{"hasHighLatency(app)", "hasDatabaseDependency(app, db)"}
		reasoningResult, err := reasoningEngine.Reason(ctx, "diagnose", facts)
		require.NoError(t, err)

		metricsCollector.RecordReasoningResult(true, reasoningResult.Confidence)

		// Get recommendations
		recommendations, err := proactiveAdvisor.AnalyzeAndRecommend(ctx, map[string]interface{}{})
		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Validate overall performance
		report := metricsCollector.ValidateMetrics()
		assert.GreaterOrEqual(t, report.PassRate, 0.8, "Overall pass rate should be >= 80%%")
	})
}

// MockLLMClient is a mock LLM client for testing
type MockLLMClient struct{}

func (m *MockLLMClient) Complete(ctx context.Context, messages []nli.Message) (*nli.CompletionResponse, error) {
	return &nli.CompletionResponse{
		Text:       "I understand you want to deploy a VM. I'll help you with that.",
		Confidence: 0.95,
		Model:      "mock-gpt-4",
		Tokens:     50,
	}, nil
}

func (m *MockLLMClient) Embed(ctx context.Context, text string) ([]float64, error) {
	embedding := make([]float64, 128)
	for i := range embedding {
		embedding[i] = float64(i) / 128.0
	}
	return embedding, nil
}
