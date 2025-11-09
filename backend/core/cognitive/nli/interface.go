// Package nli provides natural language interface for infrastructure control
package nli

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/yourusername/novacron/backend/core/cognitive"
)

// NaturalLanguageInterface handles natural language understanding
type NaturalLanguageInterface struct {
	config      *cognitive.CognitiveConfig
	llmClient   LLMClient
	parser      IntentParser
	sessions    map[string]*cognitive.ConversationContext
	sessionLock sync.RWMutex
	metrics     *cognitive.CognitiveMetrics
	metricsLock sync.RWMutex
}

// LLMClient interface for language model integration
type LLMClient interface {
	Complete(ctx context.Context, messages []Message) (*CompletionResponse, error)
	Embed(ctx context.Context, text string) ([]float64, error)
}

// Message represents a conversation message
type Message struct {
	Role    string `json:"role"`    // system, user, assistant
	Content string `json:"content"`
}

// CompletionResponse represents LLM response
type CompletionResponse struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	Model      string  `json:"model"`
	Tokens     int     `json:"tokens"`
}

// IntentParser extracts structured intent from natural language
type IntentParser interface {
	Parse(ctx context.Context, text string, context *cognitive.ConversationContext) (*cognitive.Intent, error)
}

// NewNaturalLanguageInterface creates a new NLI
func NewNaturalLanguageInterface(config *cognitive.CognitiveConfig, llmClient LLMClient, parser IntentParser) *NaturalLanguageInterface {
	return &NaturalLanguageInterface{
		config:    config,
		llmClient: llmClient,
		parser:    parser,
		sessions:  make(map[string]*cognitive.ConversationContext),
		metrics:   &cognitive.CognitiveMetrics{},
	}
}

// ProcessMessage handles a natural language input
func (nli *NaturalLanguageInterface) ProcessMessage(ctx context.Context, userID, sessionID, message string) (*NLIResponse, error) {
	startTime := time.Now()

	// Get or create session context
	session := nli.getOrCreateSession(userID, sessionID)

	// Check conversation turn limit
	if session.TurnCount >= nli.config.MaxConversationTurns {
		return &NLIResponse{
			Response:  "This conversation has reached the maximum turn limit. Please start a new session.",
			SessionID: sessionID,
			Success:   false,
		}, nil
	}

	// Parse intent
	intent, err := nli.parser.Parse(ctx, message, session)
	if err != nil {
		return nil, fmt.Errorf("failed to parse intent: %w", err)
	}

	// Update metrics
	nli.updateMetrics(intent.Confidence, true)

	// Check for ambiguities
	if len(intent.Ambiguities) > 0 {
		clarification := nli.generateClarification(intent)
		nli.addTurn(session, message, clarification, intent, nil)
		return &NLIResponse{
			Response:     clarification,
			SessionID:    sessionID,
			Intent:       intent,
			NeedsClarification: true,
			Success:      false,
		}, nil
	}

	// Generate response using LLM
	response, err := nli.generateResponse(ctx, session, message, intent)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	// Add turn to history
	actions := nli.extractActions(intent)
	nli.addTurn(session, message, response, intent, actions)

	// Update latency metric
	latency := time.Since(startTime).Milliseconds()
	nli.updateLatencyMetric(float64(latency))

	return &NLIResponse{
		Response:  response,
		SessionID: sessionID,
		Intent:    intent,
		Actions:   actions,
		Success:   true,
		Latency:   latency,
	}, nil
}

// NLIResponse represents the NLI output
type NLIResponse struct {
	Response          string              `json:"response"`
	SessionID         string              `json:"session_id"`
	Intent            *cognitive.Intent   `json:"intent,omitempty"`
	Actions           []string            `json:"actions,omitempty"`
	NeedsClarification bool               `json:"needs_clarification"`
	Success           bool                `json:"success"`
	Latency           int64               `json:"latency_ms"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// getOrCreateSession retrieves or creates a conversation session
func (nli *NaturalLanguageInterface) getOrCreateSession(userID, sessionID string) *cognitive.ConversationContext {
	nli.sessionLock.Lock()
	defer nli.sessionLock.Unlock()

	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	session, exists := nli.sessions[sessionID]
	if !exists {
		session = &cognitive.ConversationContext{
			SessionID:    sessionID,
			UserID:       userID,
			TurnCount:    0,
			History:      []cognitive.ConversationTurn{},
			SharedState:  make(map[string]interface{}),
			CreatedAt:    time.Now(),
			LastActivity: time.Now(),
		}
		nli.sessions[sessionID] = session
	}

	session.LastActivity = time.Now()
	return session
}

// addTurn adds a conversation turn to the session
func (nli *NaturalLanguageInterface) addTurn(session *cognitive.ConversationContext, userMsg, aiResp string, intent *cognitive.Intent, actions []string) {
	nli.sessionLock.Lock()
	defer nli.sessionLock.Unlock()

	turn := cognitive.ConversationTurn{
		UserMessage: userMsg,
		AIResponse:  aiResp,
		Intent:      intent,
		Actions:     actions,
		Timestamp:   time.Now(),
		Metadata:    make(map[string]interface{}),
	}

	session.History = append(session.History, turn)
	session.TurnCount++
	session.CurrentIntent = intent
}

// generateClarification creates a clarification question
func (nli *NaturalLanguageInterface) generateClarification(intent *cognitive.Intent) string {
	var clarifications []string

	for _, ambiguity := range intent.Ambiguities {
		clarifications = append(clarifications, fmt.Sprintf("- %s", ambiguity))
	}

	return fmt.Sprintf("I need clarification on the following:\n%s\n\nCould you provide more details?",
		strings.Join(clarifications, "\n"))
}

// generateResponse creates an AI response using the LLM
func (nli *NaturalLanguageInterface) generateResponse(ctx context.Context, session *cognitive.ConversationContext, userMsg string, intent *cognitive.Intent) (string, error) {
	// Build conversation history
	messages := []Message{
		{
			Role: "system",
			Content: `You are an AI infrastructure assistant for NovaCron. You help users manage distributed systems using natural language.
You can:
- Deploy and manage VMs across multiple clouds
- Optimize costs and performance
- Diagnose issues and provide recommendations
- Migrate workloads between providers
- Ensure security and compliance

Be concise, helpful, and proactive. Ask for confirmation before destructive actions.`,
		},
	}

	// Add conversation history (last 5 turns for context)
	historyStart := len(session.History) - 5
	if historyStart < 0 {
		historyStart = 0
	}
	for _, turn := range session.History[historyStart:] {
		messages = append(messages,
			Message{Role: "user", Content: turn.UserMessage},
			Message{Role: "assistant", Content: turn.AIResponse},
		)
	}

	// Add current message with intent context
	intentJSON, _ := json.Marshal(intent)
	messages = append(messages, Message{
		Role:    "user",
		Content: fmt.Sprintf("%s\n\nParsed Intent: %s", userMsg, string(intentJSON)),
	})

	// Get LLM completion
	resp, err := nli.llmClient.Complete(ctx, messages)
	if err != nil {
		return "", fmt.Errorf("LLM completion failed: %w", err)
	}

	return resp.Text, nil
}

// extractActions extracts actionable items from intent
func (nli *NaturalLanguageInterface) extractActions(intent *cognitive.Intent) []string {
	var actions []string

	switch intent.Action {
	case "deploy":
		actions = append(actions, fmt.Sprintf("Deploy %v", intent.Entities["resource"]))
	case "migrate":
		actions = append(actions, fmt.Sprintf("Migrate from %v to %v", intent.Entities["source"], intent.Entities["target"]))
	case "optimize":
		actions = append(actions, fmt.Sprintf("Optimize %v", intent.Entities["target"]))
	case "diagnose":
		actions = append(actions, fmt.Sprintf("Diagnose issue: %v", intent.Entities["problem"]))
	case "scale":
		actions = append(actions, fmt.Sprintf("Scale %v to %v", intent.Entities["resource"], intent.Entities["target_size"]))
	}

	return actions
}

// updateMetrics updates NLI performance metrics
func (nli *NaturalLanguageInterface) updateMetrics(confidence float64, success bool) {
	nli.metricsLock.Lock()
	defer nli.metricsLock.Unlock()

	nli.metrics.TotalIntents++

	// Update intent accuracy (running average)
	alpha := 0.1
	nli.metrics.IntentAccuracy = alpha*confidence + (1-alpha)*nli.metrics.IntentAccuracy

	if success {
		// Update task completion rate
		successRate := float64(1.0)
		nli.metrics.TaskCompletionRate = alpha*successRate + (1-alpha)*nli.metrics.TaskCompletionRate
	}
}

// updateLatencyMetric updates response latency metric
func (nli *NaturalLanguageInterface) updateLatencyMetric(latency float64) {
	nli.metricsLock.Lock()
	defer nli.metricsLock.Unlock()

	alpha := 0.1
	nli.metrics.AvgResponseLatency = alpha*latency + (1-alpha)*nli.metrics.AvgResponseLatency
}

// GetMetrics returns current metrics
func (nli *NaturalLanguageInterface) GetMetrics() *cognitive.CognitiveMetrics {
	nli.metricsLock.RLock()
	defer nli.metricsLock.RUnlock()

	// Return a copy
	metricsCopy := *nli.metrics
	return &metricsCopy
}

// EndSession ends a conversation session
func (nli *NaturalLanguageInterface) EndSession(sessionID string) error {
	nli.sessionLock.Lock()
	defer nli.sessionLock.Unlock()

	delete(nli.sessions, sessionID)
	return nil
}

// GetSession retrieves a conversation session
func (nli *NaturalLanguageInterface) GetSession(sessionID string) (*cognitive.ConversationContext, error) {
	nli.sessionLock.RLock()
	defer nli.sessionLock.RUnlock()

	session, exists := nli.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	return session, nil
}

// CleanupStaleSessions removes inactive sessions
func (nli *NaturalLanguageInterface) CleanupStaleSessions(maxAge time.Duration) int {
	nli.sessionLock.Lock()
	defer nli.sessionLock.Unlock()

	cutoff := time.Now().Add(-maxAge)
	removed := 0

	for id, session := range nli.sessions {
		if session.LastActivity.Before(cutoff) {
			delete(nli.sessions, id)
			removed++
		}
	}

	return removed
}
