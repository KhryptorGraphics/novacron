// Package decision provides autonomous decision-making capabilities
package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// DecisionEngine makes autonomous decisions with hybrid rule-based and ML approaches
type DecisionEngine struct {
	ruleEngine     *RuleEngine
	mlEngine       *MLDecisionEngine
	contextManager *ContextManager
	auditLog       *AuditLog
	humanInLoop    *HumanInTheLoopManager
	logger         *zap.Logger
	mu             sync.RWMutex
}

// DecisionRequest represents a decision to be made
type DecisionRequest struct {
	ID          string                 `json:"id"`
	Type        DecisionType           `json:"type"`
	Context     map[string]interface{} `json:"context"`
	Options     []*DecisionOption      `json:"options"`
	Constraints []*Constraint          `json:"constraints"`
	Objectives  []*Objective           `json:"objectives"`
	Urgency     UrgencyLevel           `json:"urgency"`
	CreatedAt   time.Time              `json:"created_at"`
}

// DecisionType defines types of decisions
type DecisionType string

const (
	DecisionTypeScaling       DecisionType = "scaling"
	DecisionTypeProvisioning  DecisionType = "provisioning"
	DecisionTypeIncident      DecisionType = "incident_response"
	DecisionTypeOptimization  DecisionType = "optimization"
	DecisionTypeSecurity      DecisionType = "security"
	DecisionTypeCompliance    DecisionType = "compliance"
	DecisionTypeMaintenance   DecisionType = "maintenance"
	DecisionTypeDisasterRecov DecisionType = "disaster_recovery"
)

// DecisionOption represents a possible decision option
type DecisionOption struct {
	ID          string                 `json:"id"`
	Description string                 `json:"description"`
	Actions     []Action               `json:"actions"`
	Predictions map[string]float64     `json:"predictions"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Action represents an action to take
type Action struct {
	Type       string                 `json:"type"`
	Target     string                 `json:"target"`
	Parameters map[string]interface{} `json:"parameters"`
	Order      int                    `json:"order"`
}

// Constraint defines a constraint on decisions
type Constraint struct {
	Type        string  `json:"type"`
	Description string  `json:"description"`
	Value       float64 `json:"value"`
	Operator    string  `json:"operator"` // gt, lt, eq, gte, lte
	Priority    int     `json:"priority"`
}

// Objective defines an objective to optimize
type Objective struct {
	Type        string  `json:"type"`
	Description string  `json:"description"`
	Weight      float64 `json:"weight"`
	Direction   string  `json:"direction"` // maximize, minimize
}

// UrgencyLevel defines decision urgency
type UrgencyLevel string

const (
	UrgencyLow      UrgencyLevel = "low"
	UrgencyMedium   UrgencyLevel = "medium"
	UrgencyHigh     UrgencyLevel = "high"
	UrgencyCritical UrgencyLevel = "critical"
)

// DecisionResult represents the result of a decision
type DecisionResult struct {
	RequestID       string                 `json:"request_id"`
	SelectedOption  *DecisionOption        `json:"selected_option"`
	Score           float64                `json:"score"`
	Confidence      float64                `json:"confidence"`
	Reasoning       []string               `json:"reasoning"`
	AlternativeOpts []*ScoredOption        `json:"alternative_options"`
	RequiresApproval bool                  `json:"requires_approval"`
	ApprovalLevel   string                 `json:"approval_level,omitempty"`
	DecisionMethod  string                 `json:"decision_method"` // rule_based, ml_based, hybrid
	Context         map[string]interface{} `json:"context"`
	Timestamp       time.Time              `json:"timestamp"`
}

// ScoredOption represents an option with its score
type ScoredOption struct {
	Option *DecisionOption `json:"option"`
	Score  float64         `json:"score"`
}

// NewDecisionEngine creates a new decision engine
func NewDecisionEngine(logger *zap.Logger) *DecisionEngine {
	engine := &DecisionEngine{
		logger: logger,
	}

	engine.ruleEngine = NewRuleEngine(logger)
	engine.mlEngine = NewMLDecisionEngine(logger)
	engine.contextManager = NewContextManager(logger)
	engine.auditLog = NewAuditLog(logger)
	engine.humanInLoop = NewHumanInTheLoopManager(logger)

	return engine
}

// MakeDecision makes a decision based on the request
func (e *DecisionEngine) MakeDecision(ctx context.Context, request *DecisionRequest) (*DecisionResult, error) {
	e.logger.Info("Making decision",
		zap.String("request_id", request.ID),
		zap.String("type", string(request.Type)),
		zap.String("urgency", string(request.Urgency)))

	// 1. Enrich context
	enrichedContext := e.contextManager.EnrichContext(ctx, request.Context)
	request.Context = enrichedContext

	// 2. Score options using both rule-based and ML approaches
	ruleScores := e.ruleEngine.ScoreOptions(ctx, request)
	mlScores := e.mlEngine.ScoreOptions(ctx, request)

	// 3. Combine scores (hybrid approach)
	combinedScores := e.combineScores(ruleScores, mlScores, request)

	// 4. Select best option
	result := e.selectBestOption(request, combinedScores)

	// 5. Generate reasoning
	result.Reasoning = e.generateReasoning(request, result, ruleScores, mlScores)

	// 6. Determine if approval required
	result.RequiresApproval = e.requiresApproval(request, result)
	if result.RequiresApproval {
		result.ApprovalLevel = e.determineApprovalLevel(request, result)
	}

	// 7. Audit the decision
	e.auditLog.LogDecision(ctx, request, result)

	// 8. Handle human-in-the-loop if required
	if result.RequiresApproval && request.Urgency != UrgencyCritical {
		approved, err := e.humanInLoop.RequestApproval(ctx, request, result)
		if err != nil || !approved {
			result.SelectedOption = nil
			result.Reasoning = append(result.Reasoning, "Decision pending human approval")
		}
	}

	e.logger.Info("Decision made",
		zap.String("request_id", request.ID),
		zap.Float64("score", result.Score),
		zap.Float64("confidence", result.Confidence),
		zap.Bool("requires_approval", result.RequiresApproval))

	return result, nil
}

// combineScores combines rule-based and ML scores
func (e *DecisionEngine) combineScores(ruleScores, mlScores map[string]float64, request *DecisionRequest) map[string]float64 {
	combined := make(map[string]float64)

	// Weight based on urgency and confidence
	ruleWeight := 0.6
	mlWeight := 0.4

	// Adjust weights based on urgency
	if request.Urgency == UrgencyCritical {
		ruleWeight = 0.8 // Trust rules more for critical decisions
		mlWeight = 0.2
	}

	for optionID := range ruleScores {
		ruleScore := ruleScores[optionID]
		mlScore := mlScores[optionID]

		combined[optionID] = ruleScore*ruleWeight + mlScore*mlWeight
	}

	return combined
}

// selectBestOption selects the best option based on scores
func (e *DecisionEngine) selectBestOption(request *DecisionRequest, scores map[string]float64) *DecisionResult {
	var bestOption *DecisionOption
	var bestScore float64
	alternatives := make([]*ScoredOption, 0)

	for _, option := range request.Options {
		score := scores[option.ID]

		if score > bestScore {
			if bestOption != nil {
				alternatives = append(alternatives, &ScoredOption{
					Option: bestOption,
					Score:  bestScore,
				})
			}
			bestOption = option
			bestScore = score
		} else {
			alternatives = append(alternatives, &ScoredOption{
				Option: option,
				Score:  score,
			})
		}
	}

	// Calculate confidence based on score gap
	confidence := 0.5
	if len(alternatives) > 0 {
		secondBestScore := alternatives[0].Score
		scoreGap := bestScore - secondBestScore
		confidence = 0.5 + (scoreGap / 2.0)
		if confidence > 1.0 {
			confidence = 1.0
		}
	} else {
		confidence = 0.9 // Only one option
	}

	return &DecisionResult{
		RequestID:       request.ID,
		SelectedOption:  bestOption,
		Score:           bestScore,
		Confidence:      confidence,
		AlternativeOpts: alternatives,
		DecisionMethod:  "hybrid",
		Context:         request.Context,
		Timestamp:       time.Now(),
	}
}

// generateReasoning generates human-readable reasoning
func (e *DecisionEngine) generateReasoning(request *DecisionRequest, result *DecisionResult, ruleScores, mlScores map[string]float64) []string {
	reasoning := make([]string, 0)

	if result.SelectedOption == nil {
		return reasoning
	}

	// Basic reasoning
	reasoning = append(reasoning, fmt.Sprintf("Selected option: %s", result.SelectedOption.Description))
	reasoning = append(reasoning, fmt.Sprintf("Overall score: %.2f", result.Score))
	reasoning = append(reasoning, fmt.Sprintf("Confidence: %.0f%%", result.Confidence*100))

	// Rule-based reasoning
	ruleScore := ruleScores[result.SelectedOption.ID]
	reasoning = append(reasoning, fmt.Sprintf("Rule-based score: %.2f", ruleScore))

	// ML reasoning
	mlScore := mlScores[result.SelectedOption.ID]
	reasoning = append(reasoning, fmt.Sprintf("ML prediction score: %.2f", mlScore))

	// Constraint satisfaction
	constraintsMet := e.checkConstraints(request.Constraints, result.SelectedOption)
	reasoning = append(reasoning, fmt.Sprintf("Constraints satisfied: %d/%d", constraintsMet, len(request.Constraints)))

	// Objective alignment
	objectiveScore := e.scoreObjectives(request.Objectives, result.SelectedOption)
	reasoning = append(reasoning, fmt.Sprintf("Objective alignment: %.2f", objectiveScore))

	return reasoning
}

// requiresApproval determines if decision requires human approval
func (e *DecisionEngine) requiresApproval(request *DecisionRequest, result *DecisionResult) bool {
	// Always require approval for low confidence
	if result.Confidence < 0.7 {
		return true
	}

	// Critical decisions require approval unless urgent
	criticalTypes := map[DecisionType]bool{
		DecisionTypeSecurity:      true,
		DecisionTypeCompliance:    true,
		DecisionTypeDisasterRecov: true,
	}

	if criticalTypes[request.Type] && request.Urgency != UrgencyCritical {
		return true
	}

	// High-cost decisions require approval
	if result.SelectedOption != nil {
		if cost, ok := result.SelectedOption.Predictions["cost"]; ok {
			if cost > 1000 { // $1000 threshold
				return true
			}
		}
	}

	return false
}

// determineApprovalLevel determines the required approval level
func (e *DecisionEngine) determineApprovalLevel(request *DecisionRequest, result *DecisionResult) string {
	if result.SelectedOption == nil {
		return "manager"
	}

	cost, _ := result.SelectedOption.Predictions["cost"]

	switch {
	case cost > 10000:
		return "director"
	case cost > 5000:
		return "senior_manager"
	case cost > 1000:
		return "manager"
	default:
		if result.Confidence < 0.5 {
			return "manager"
		}
		return "team_lead"
	}
}

// checkConstraints checks how many constraints are satisfied
func (e *DecisionEngine) checkConstraints(constraints []*Constraint, option *DecisionOption) int {
	met := 0

	for _, constraint := range constraints {
		value, ok := option.Predictions[constraint.Type]
		if !ok {
			continue
		}

		satisfied := false
		switch constraint.Operator {
		case "gt":
			satisfied = value > constraint.Value
		case "lt":
			satisfied = value < constraint.Value
		case "gte":
			satisfied = value >= constraint.Value
		case "lte":
			satisfied = value <= constraint.Value
		case "eq":
			satisfied = value == constraint.Value
		}

		if satisfied {
			met++
		}
	}

	return met
}

// scoreObjectives scores how well an option meets objectives
func (e *DecisionEngine) scoreObjectives(objectives []*Objective, option *DecisionOption) float64 {
	totalScore := 0.0
	totalWeight := 0.0

	for _, objective := range objectives {
		value, ok := option.Predictions[objective.Type]
		if !ok {
			continue
		}

		// Normalize value (simple linear normalization)
		normalizedValue := value / 100.0
		if normalizedValue > 1.0 {
			normalizedValue = 1.0
		}

		score := normalizedValue
		if objective.Direction == "minimize" {
			score = 1.0 - score
		}

		totalScore += score * objective.Weight
		totalWeight += objective.Weight
	}

	if totalWeight == 0 {
		return 0
	}

	return totalScore / totalWeight
}

// ExecuteDecision executes a decision
func (e *DecisionEngine) ExecuteDecision(ctx context.Context, result *DecisionResult) error {
	if result.SelectedOption == nil {
		return fmt.Errorf("no option selected")
	}

	e.logger.Info("Executing decision",
		zap.String("request_id", result.RequestID),
		zap.String("option", result.SelectedOption.ID))

	// Execute actions in order
	for _, action := range result.SelectedOption.Actions {
		if err := e.executeAction(ctx, action); err != nil {
			return fmt.Errorf("failed to execute action: %w", err)
		}
	}

	// Record execution in audit log
	e.auditLog.LogExecution(ctx, result)

	return nil
}

// executeAction executes a single action
func (e *DecisionEngine) executeAction(ctx context.Context, action Action) error {
	e.logger.Info("Executing action",
		zap.String("type", action.Type),
		zap.String("target", action.Target))

	// Placeholder for actual action execution
	// This would integrate with various system controllers

	return nil
}

// GetDecisionHistory returns decision history
func (e *DecisionEngine) GetDecisionHistory(ctx context.Context, filters map[string]interface{}, limit int) ([]*DecisionResult, error) {
	return e.auditLog.GetHistory(ctx, filters, limit)
}

// ExplainDecision provides detailed explanation of a past decision
func (e *DecisionEngine) ExplainDecision(ctx context.Context, requestID string) (*DecisionExplanation, error) {
	result, err := e.auditLog.GetDecision(ctx, requestID)
	if err != nil {
		return nil, err
	}

	explanation := &DecisionExplanation{
		RequestID:   requestID,
		Decision:    result,
		Reasoning:   result.Reasoning,
		Timestamp:   result.Timestamp,
		FactorsUsed: e.extractFactors(result),
	}

	return explanation, nil
}

// DecisionExplanation provides detailed explanation
type DecisionExplanation struct {
	RequestID   string                 `json:"request_id"`
	Decision    *DecisionResult        `json:"decision"`
	Reasoning   []string               `json:"reasoning"`
	FactorsUsed map[string]interface{} `json:"factors_used"`
	Timestamp   time.Time              `json:"timestamp"`
}

func (e *DecisionEngine) extractFactors(result *DecisionResult) map[string]interface{} {
	factors := make(map[string]interface{})

	// Extract key decision factors
	factors["score"] = result.Score
	factors["confidence"] = result.Confidence
	factors["method"] = result.DecisionMethod

	if result.SelectedOption != nil {
		factors["predictions"] = result.SelectedOption.Predictions
	}

	return factors
}
