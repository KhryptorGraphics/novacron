package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

// RuleEngine provides rule-based decision making
type RuleEngine struct {
	rules  []*DecisionRule
	logger *zap.Logger
	mu     sync.RWMutex
}

// DecisionRule represents a decision rule
type DecisionRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Conditions  []*RuleCondition       `json:"conditions"`
	Actions     []RuleAction           `json:"actions"`
	Priority    int                    `json:"priority"`
	Enabled     bool                   `json:"enabled"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// RuleCondition represents a condition in a rule
type RuleCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

// RuleAction represents an action in a rule
type RuleAction struct {
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"parameters"`
}

// NewRuleEngine creates a new rule engine
func NewRuleEngine(logger *zap.Logger) *RuleEngine {
	engine := &RuleEngine{
		rules:  make([]*DecisionRule, 0),
		logger: logger,
	}

	// Load default rules
	engine.loadDefaultRules()

	return engine
}

// loadDefaultRules loads default decision rules
func (r *RuleEngine) loadDefaultRules() {
	defaultRules := []*DecisionRule{
		{
			ID:          "rule-cpu-high",
			Name:        "High CPU utilization",
			Description: "Scale up when CPU is high",
			Conditions: []*RuleCondition{
				{Field: "cpu_utilization", Operator: "gt", Value: 80.0},
			},
			Actions: []RuleAction{
				{Type: "recommend_scale_up", Parameters: map[string]interface{}{"resource": "cpu"}},
			},
			Priority: 1,
			Enabled:  true,
		},
		{
			ID:          "rule-cost-optimize",
			Name:        "Cost optimization opportunity",
			Description: "Recommend cost optimization when utilization is low",
			Conditions: []*RuleCondition{
				{Field: "cpu_utilization", Operator: "lt", Value: 30.0},
				{Field: "duration_hours", Operator: "gt", Value: 24.0},
			},
			Actions: []RuleAction{
				{Type: "recommend_downsize", Parameters: map[string]interface{}{"reason": "underutilization"}},
			},
			Priority: 2,
			Enabled:  true,
		},
		{
			ID:          "rule-incident-critical",
			Name:        "Critical incident response",
			Description: "Immediate action for critical incidents",
			Conditions: []*RuleCondition{
				{Field: "incident_severity", Operator: "eq", Value: "critical"},
			},
			Actions: []RuleAction{
				{Type: "immediate_response", Parameters: map[string]interface{}{"escalate": true}},
			},
			Priority: 0,
			Enabled:  true,
		},
	}

	r.rules = append(r.rules, defaultRules...)
}

// ScoreOptions scores options using rules
func (r *RuleEngine) ScoreOptions(ctx context.Context, request *DecisionRequest) map[string]float64 {
	scores := make(map[string]float64)

	for _, option := range request.Options {
		score := r.evaluateOption(ctx, request, option)
		scores[option.ID] = score
	}

	return scores
}

// evaluateOption evaluates a single option against rules
func (r *RuleEngine) evaluateOption(ctx context.Context, request *DecisionRequest, option *DecisionOption) float64 {
	score := 0.0
	matchedRules := 0

	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, rule := range r.rules {
		if !rule.Enabled {
			continue
		}

		if r.matchesRule(request, option, rule) {
			// Higher priority rules have more weight
			ruleWeight := 1.0 / float64(rule.Priority+1)
			score += ruleWeight
			matchedRules++
		}
	}

	// Normalize score
	if matchedRules > 0 {
		score = score / float64(matchedRules)
	}

	return score
}

// matchesRule checks if option matches a rule
func (r *RuleEngine) matchesRule(request *DecisionRequest, option *DecisionOption, rule *DecisionRule) bool {
	for _, condition := range rule.Conditions {
		// Check in context first
		value, ok := request.Context[condition.Field]
		if !ok {
			// Check in option predictions
			value, ok = option.Predictions[condition.Field]
			if !ok {
				return false
			}
		}

		if !r.evaluateCondition(value, condition) {
			return false
		}
	}

	return true
}

// evaluateCondition evaluates a single condition
func (r *RuleEngine) evaluateCondition(value interface{}, condition *RuleCondition) bool {
	switch condition.Operator {
	case "gt":
		return r.compareValues(value, condition.Value) > 0
	case "lt":
		return r.compareValues(value, condition.Value) < 0
	case "gte":
		return r.compareValues(value, condition.Value) >= 0
	case "lte":
		return r.compareValues(value, condition.Value) <= 0
	case "eq":
		return r.compareValues(value, condition.Value) == 0
	case "ne":
		return r.compareValues(value, condition.Value) != 0
	default:
		return false
	}
}

// compareValues compares two values
func (r *RuleEngine) compareValues(v1, v2 interface{}) int {
	// Type conversion and comparison
	f1, ok1 := r.toFloat64(v1)
	f2, ok2 := r.toFloat64(v2)

	if ok1 && ok2 {
		if f1 > f2 {
			return 1
		} else if f1 < f2 {
			return -1
		}
		return 0
	}

	// String comparison as fallback
	s1 := fmt.Sprintf("%v", v1)
	s2 := fmt.Sprintf("%v", v2)

	if s1 > s2 {
		return 1
	} else if s1 < s2 {
		return -1
	}
	return 0
}

// toFloat64 converts interface to float64
func (r *RuleEngine) toFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	default:
		return 0, false
	}
}

// AddRule adds a new rule
func (r *RuleEngine) AddRule(rule *DecisionRule) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.rules = append(r.rules, rule)

	r.logger.Info("Rule added",
		zap.String("id", rule.ID),
		zap.String("name", rule.Name))
}

// MLDecisionEngine provides ML-based decision making
type MLDecisionEngine struct {
	models map[DecisionType]*MLModel
	logger *zap.Logger
	mu     sync.RWMutex
}

// MLModel represents a machine learning model
type MLModel struct {
	Type       DecisionType `json:"type"`
	Version    string       `json:"version"`
	Accuracy   float64      `json:"accuracy"`
	LastTrained time.Time   `json:"last_trained"`
	Features   []string     `json:"features"`
}

// NewMLDecisionEngine creates a new ML decision engine
func NewMLDecisionEngine(logger *zap.Logger) *MLDecisionEngine {
	engine := &MLDecisionEngine{
		models: make(map[DecisionType]*MLModel),
		logger: logger,
	}

	// Initialize models
	engine.initializeModels()

	return engine
}

// initializeModels initializes ML models
func (m *MLDecisionEngine) initializeModels() {
	models := []*MLModel{
		{
			Type:        DecisionTypeScaling,
			Version:     "1.0.0",
			Accuracy:    0.92,
			LastTrained: time.Now().Add(-24 * time.Hour),
			Features:    []string{"cpu_utilization", "memory_utilization", "request_rate"},
		},
		{
			Type:        DecisionTypeOptimization,
			Version:     "1.0.0",
			Accuracy:    0.88,
			LastTrained: time.Now().Add(-48 * time.Hour),
			Features:    []string{"cost", "performance", "utilization"},
		},
	}

	for _, model := range models {
		m.models[model.Type] = model
	}
}

// ScoreOptions scores options using ML models
func (m *MLDecisionEngine) ScoreOptions(ctx context.Context, request *DecisionRequest) map[string]float64 {
	scores := make(map[string]float64)

	model, exists := m.models[request.Type]
	if !exists {
		// Return neutral scores if no model available
		for _, option := range request.Options {
			scores[option.ID] = 0.5
		}
		return scores
	}

	for _, option := range request.Options {
		score := m.predictScore(ctx, model, request, option)
		scores[option.ID] = score
	}

	return scores
}

// predictScore predicts score for an option
func (m *MLDecisionEngine) predictScore(ctx context.Context, model *MLModel, request *DecisionRequest, option *DecisionOption) float64 {
	// Extract features
	features := m.extractFeatures(model, request, option)

	// Simple neural network simulation
	// In production, this would call a real ML model
	score := 0.5

	for i, feature := range features {
		weight := 0.3 + (float64(i) * 0.1)
		score += feature * weight
	}

	// Normalize to 0-1
	score = 1.0 / (1.0 + math.Exp(-score)) // Sigmoid

	// Adjust by model accuracy
	score *= model.Accuracy

	return score
}

// extractFeatures extracts features for ML model
func (m *MLDecisionEngine) extractFeatures(model *MLModel, request *DecisionRequest, option *DecisionOption) []float64 {
	features := make([]float64, len(model.Features))

	for i, featureName := range model.Features {
		value := 0.0

		// Try to get from context
		if v, ok := request.Context[featureName]; ok {
			if fv, ok := v.(float64); ok {
				value = fv
			}
		}

		// Try to get from predictions
		if v, ok := option.Predictions[featureName]; ok {
			value = v
		}

		// Normalize feature (simple min-max normalization)
		features[i] = value / 100.0
	}

	return features
}

// TrainModel trains an ML model
func (m *MLDecisionEngine) TrainModel(ctx context.Context, decisionType DecisionType, trainingData []map[string]interface{}) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	model := m.models[decisionType]
	if model == nil {
		return fmt.Errorf("model not found for type: %s", decisionType)
	}

	// Simulate training
	// In production, this would train a real model
	model.LastTrained = time.Now()
	model.Accuracy = 0.85 + rand.Float64()*0.1

	m.logger.Info("Model trained",
		zap.String("type", string(decisionType)),
		zap.Float64("accuracy", model.Accuracy))

	return nil
}

// ContextManager enriches decision context
type ContextManager struct {
	logger *zap.Logger
}

// NewContextManager creates a new context manager
func NewContextManager(logger *zap.Logger) *ContextManager {
	return &ContextManager{logger: logger}
}

// EnrichContext enriches decision context with additional information
func (c *ContextManager) EnrichContext(ctx context.Context, baseContext map[string]interface{}) map[string]interface{} {
	enriched := make(map[string]interface{})

	// Copy base context
	for k, v := range baseContext {
		enriched[k] = v
	}

	// Add temporal context
	now := time.Now()
	enriched["timestamp"] = now
	enriched["hour_of_day"] = now.Hour()
	enriched["day_of_week"] = now.Weekday().String()
	enriched["is_business_hours"] = c.isBusinessHours(now)

	// Add load context (would come from real monitoring)
	enriched["system_load"] = c.getSystemLoad(ctx)
	enriched["request_rate"] = c.getRequestRate(ctx)

	// Add cost context
	enriched["current_cost_rate"] = c.getCurrentCostRate(ctx)

	// Add historical context
	enriched["avg_utilization_24h"] = c.getAverageUtilization(ctx, 24*time.Hour)

	return enriched
}

func (c *ContextManager) isBusinessHours(t time.Time) bool {
	hour := t.Hour()
	weekday := t.Weekday()

	return weekday >= time.Monday && weekday <= time.Friday &&
		hour >= 9 && hour < 17
}

func (c *ContextManager) getSystemLoad(ctx context.Context) float64 {
	return 0.65 + rand.Float64()*0.2
}

func (c *ContextManager) getRequestRate(ctx context.Context) float64 {
	return 1000.0 + rand.Float64()*500.0
}

func (c *ContextManager) getCurrentCostRate(ctx context.Context) float64 {
	return 5.0 + rand.Float64()*2.0
}

func (c *ContextManager) getAverageUtilization(ctx context.Context, duration time.Duration) float64 {
	return 60.0 + rand.Float64()*20.0
}

// AuditLog logs decisions for compliance
type AuditLog struct {
	entries []*AuditEntry
	logger  *zap.Logger
	mu      sync.RWMutex
}

// AuditEntry represents an audit log entry
type AuditEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Type      string                 `json:"type"`
	RequestID string                 `json:"request_id"`
	Data      map[string]interface{} `json:"data"`
}

// NewAuditLog creates a new audit log
func NewAuditLog(logger *zap.Logger) *AuditLog {
	return &AuditLog{
		entries: make([]*AuditEntry, 0),
		logger:  logger,
	}
}

// LogDecision logs a decision
func (a *AuditLog) LogDecision(ctx context.Context, request *DecisionRequest, result *DecisionResult) {
	a.mu.Lock()
	defer a.mu.Unlock()

	entry := &AuditEntry{
		Timestamp: time.Now(),
		Type:      "decision",
		RequestID: request.ID,
		Data: map[string]interface{}{
			"request": request,
			"result":  result,
		},
	}

	a.entries = append(a.entries, entry)

	// Log to structured logger
	a.logger.Info("Decision logged",
		zap.String("request_id", request.ID),
		zap.String("decision_type", string(request.Type)))
}

// LogExecution logs decision execution
func (a *AuditLog) LogExecution(ctx context.Context, result *DecisionResult) {
	a.mu.Lock()
	defer a.mu.Unlock()

	entry := &AuditEntry{
		Timestamp: time.Now(),
		Type:      "execution",
		RequestID: result.RequestID,
		Data: map[string]interface{}{
			"result": result,
		},
	}

	a.entries = append(a.entries, entry)

	a.logger.Info("Execution logged",
		zap.String("request_id", result.RequestID))
}

// GetHistory returns audit history
func (a *AuditLog) GetHistory(ctx context.Context, filters map[string]interface{}, limit int) ([]*DecisionResult, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	results := make([]*DecisionResult, 0)

	// Return recent decisions (simplified)
	count := 0
	for i := len(a.entries) - 1; i >= 0 && count < limit; i-- {
		entry := a.entries[i]
		if entry.Type == "decision" {
			if result, ok := entry.Data["result"].(*DecisionResult); ok {
				results = append(results, result)
				count++
			}
		}
	}

	return results, nil
}

// GetDecision retrieves a specific decision
func (a *AuditLog) GetDecision(ctx context.Context, requestID string) (*DecisionResult, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	for _, entry := range a.entries {
		if entry.RequestID == requestID && entry.Type == "decision" {
			if result, ok := entry.Data["result"].(*DecisionResult); ok {
				return result, nil
			}
		}
	}

	return nil, fmt.Errorf("decision not found: %s", requestID)
}

// HumanInTheLoopManager manages human approval workflows
type HumanInTheLoopManager struct {
	logger *zap.Logger
}

// NewHumanInTheLoopManager creates a new human-in-the-loop manager
func NewHumanInTheLoopManager(logger *zap.Logger) *HumanInTheLoopManager {
	return &HumanInTheLoopManager{logger: logger}
}

// RequestApproval requests human approval for a decision
func (h *HumanInTheLoopManager) RequestApproval(ctx context.Context, request *DecisionRequest, result *DecisionResult) (bool, error) {
	h.logger.Info("Requesting human approval",
		zap.String("request_id", request.ID),
		zap.String("approval_level", result.ApprovalLevel))

	// In production, this would:
	// 1. Send notification to approvers
	// 2. Create approval task in workflow system
	// 3. Wait for approval (with timeout)
	// 4. Return approval status

	// For now, simulate approval based on confidence
	approved := result.Confidence > 0.75

	h.logger.Info("Approval result",
		zap.String("request_id", request.ID),
		zap.Bool("approved", approved))

	return approved, nil
}

// SendNotification sends notification to approvers
func (h *HumanInTheLoopManager) SendNotification(ctx context.Context, request *DecisionRequest, result *DecisionResult) error {
	// Simulate notification
	notificationData, _ := json.Marshal(map[string]interface{}{
		"request_id": request.ID,
		"type":       request.Type,
		"urgency":    request.Urgency,
		"decision":   result.SelectedOption.Description,
		"confidence": result.Confidence,
	})

	h.logger.Info("Notification sent",
		zap.String("request_id", request.ID),
		zap.String("data", string(notificationData)))

	return nil
}
