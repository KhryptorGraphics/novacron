// Package cognitive provides AI-powered natural language infrastructure control
package cognitive

import (
	"time"
)

// CognitiveConfig defines configuration for the cognitive AI system
type CognitiveConfig struct {
	// Natural Language Interface
	EnableNLI            bool          `yaml:"enable_nli" json:"enable_nli"`
	LLMModel             string        `yaml:"llm_model" json:"llm_model"`                         // "gpt-4-turbo"
	LLMAPIKey            string        `yaml:"llm_api_key" json:"llm_api_key"`
	MaxConversationTurns int           `yaml:"max_conversation_turns" json:"max_conversation_turns"` // 10

	// Reasoning
	ReasoningTimeout     time.Duration `yaml:"reasoning_timeout" json:"reasoning_timeout"` // 100ms
	EnableSymbolicAI     bool          `yaml:"enable_symbolic_ai" json:"enable_symbolic_ai"`
	MaxReasoningDepth    int           `yaml:"max_reasoning_depth" json:"max_reasoning_depth"` // 5

	// Knowledge Graph
	KnowledgeGraphURL    string        `yaml:"knowledge_graph_url" json:"knowledge_graph_url"`
	KnowledgeGraphType   string        `yaml:"knowledge_graph_type" json:"knowledge_graph_type"` // "neo4j" or "arangodb"
	KnowledgeGraphUser   string        `yaml:"knowledge_graph_user" json:"knowledge_graph_user"`
	KnowledgeGraphPass   string        `yaml:"knowledge_graph_pass" json:"knowledge_graph_pass"`

	// Proactive Advice
	EnableProactiveAdvice bool         `yaml:"enable_proactive_advice" json:"enable_proactive_advice"`
	AdviceCheckInterval   time.Duration `yaml:"advice_check_interval" json:"advice_check_interval"` // 5m
	MinConfidenceScore    float64       `yaml:"min_confidence_score" json:"min_confidence_score"`   // 0.85

	// Multi-Modal
	VoiceEnabled         bool          `yaml:"voice_enabled" json:"voice_enabled"`
	VisionEnabled        bool          `yaml:"vision_enabled" json:"vision_enabled"`
	SpeechToTextAPI      string        `yaml:"speech_to_text_api" json:"speech_to_text_api"`
	TextToSpeechAPI      string        `yaml:"text_to_speech_api" json:"text_to_speech_api"`

	// Memory
	ShortTermMemorySize  int           `yaml:"short_term_memory_size" json:"short_term_memory_size"` // 100
	LongTermMemorySize   int           `yaml:"long_term_memory_size" json:"long_term_memory_size"`   // 10000
	EnableRAG            bool          `yaml:"enable_rag" json:"enable_rag"`
	EmbeddingModel       string        `yaml:"embedding_model" json:"embedding_model"` // "text-embedding-3-large"

	// Performance
	CacheTTL             time.Duration `yaml:"cache_ttl" json:"cache_ttl"` // 1h
	MaxConcurrentRequests int          `yaml:"max_concurrent_requests" json:"max_concurrent_requests"` // 100
}

// DefaultCognitiveConfig returns default configuration
func DefaultCognitiveConfig() *CognitiveConfig {
	return &CognitiveConfig{
		EnableNLI:             true,
		LLMModel:              "gpt-4-turbo",
		MaxConversationTurns:  10,
		ReasoningTimeout:      100 * time.Millisecond,
		EnableSymbolicAI:      true,
		MaxReasoningDepth:     5,
		KnowledgeGraphType:    "neo4j",
		EnableProactiveAdvice: true,
		AdviceCheckInterval:   5 * time.Minute,
		MinConfidenceScore:    0.85,
		VoiceEnabled:          false,
		VisionEnabled:         false,
		ShortTermMemorySize:   100,
		LongTermMemorySize:    10000,
		EnableRAG:             true,
		EmbeddingModel:        "text-embedding-3-large",
		CacheTTL:              1 * time.Hour,
		MaxConcurrentRequests: 100,
	}
}

// Intent represents a parsed user intention
type Intent struct {
	Action      string                 `json:"action"`       // deploy, migrate, optimize, diagnose
	Entities    map[string]interface{} `json:"entities"`     // VMs, regions, metrics
	Constraints map[string]interface{} `json:"constraints"`  // latency, cost, security
	Confidence  float64                `json:"confidence"`   // 0.0-1.0
	Ambiguities []string               `json:"ambiguities"`  // Unclear aspects
	Context     *ConversationContext   `json:"context"`
}

// ConversationContext maintains multi-turn conversation state
type ConversationContext struct {
	SessionID     string                 `json:"session_id"`
	UserID        string                 `json:"user_id"`
	TurnCount     int                    `json:"turn_count"`
	History       []ConversationTurn     `json:"history"`
	CurrentIntent *Intent                `json:"current_intent"`
	SharedState   map[string]interface{} `json:"shared_state"`
	CreatedAt     time.Time              `json:"created_at"`
	LastActivity  time.Time              `json:"last_activity"`
}

// ConversationTurn represents one exchange
type ConversationTurn struct {
	UserMessage string                 `json:"user_message"`
	AIResponse  string                 `json:"ai_response"`
	Intent      *Intent                `json:"intent"`
	Actions     []string               `json:"actions"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ReasoningResult contains the output of the reasoning engine
type ReasoningResult struct {
	Conclusion  string                 `json:"conclusion"`
	Confidence  float64                `json:"confidence"`
	Explanation string                 `json:"explanation"`
	Steps       []ReasoningStep        `json:"steps"`
	Alternatives []Alternative         `json:"alternatives"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ReasoningStep represents one step in logical reasoning
type ReasoningStep struct {
	Rule        string                 `json:"rule"`
	Premises    []string               `json:"premises"`
	Conclusion  string                 `json:"conclusion"`
	Confidence  float64                `json:"confidence"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Alternative represents an alternative solution
type Alternative struct {
	Description string                 `json:"description"`
	Pros        []string               `json:"pros"`
	Cons        []string               `json:"cons"`
	Confidence  float64                `json:"confidence"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Recommendation represents a proactive suggestion
type Recommendation struct {
	Type        string                 `json:"type"`         // cost, security, performance, capacity
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Impact      string                 `json:"impact"`       // High, Medium, Low
	Effort      string                 `json:"effort"`       // High, Medium, Low
	Savings     float64                `json:"savings"`      // Estimated cost savings
	Confidence  float64                `json:"confidence"`
	Actions     []string               `json:"actions"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
}

// KnowledgeEntity represents an entity in the knowledge graph
type KnowledgeEntity struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`       // VM, Network, Storage, User, Policy
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// KnowledgeRelation represents a relationship in the knowledge graph
type KnowledgeRelation struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`       // Depends-On, Communicates-With, Belongs-To
	From       string                 `json:"from"`       // Source entity ID
	To         string                 `json:"to"`         // Target entity ID
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
}

// CognitiveMetrics tracks cognitive AI performance
type CognitiveMetrics struct {
	IntentAccuracy          float64 `json:"intent_accuracy"`           // >95%
	TaskCompletionRate      float64 `json:"task_completion_rate"`      // >90%
	UserSatisfaction        float64 `json:"user_satisfaction"`         // >4.5/5
	ReasoningCorrectness    float64 `json:"reasoning_correctness"`     // >90%
	AvgResponseLatency      float64 `json:"avg_response_latency_ms"`   // <100ms
	RecommendationAcceptance float64 `json:"recommendation_acceptance"` // >85%
	ContextSwitchLatency    float64 `json:"context_switch_latency_ms"` // <10ms
	TotalConversations      int64   `json:"total_conversations"`
	TotalIntents            int64   `json:"total_intents"`
	TotalRecommendations    int64   `json:"total_recommendations"`
}
