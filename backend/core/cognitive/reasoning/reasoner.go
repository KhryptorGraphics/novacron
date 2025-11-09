// Package reasoning provides symbolic AI reasoning engine
package reasoning

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/yourusername/novacron/backend/core/cognitive"
)

// ReasoningEngine performs logical reasoning and planning
type ReasoningEngine struct {
	config      *cognitive.CognitiveConfig
	ruleBase    *RuleBase
	factBase    *FactBase
	planner     *PDDLPlanner
	cache       map[string]*cognitive.ReasoningResult
	cacheLock   sync.RWMutex
	metricsLock sync.RWMutex
	metrics     ReasoningMetrics
}

// RuleBase stores logical rules
type RuleBase struct {
	rules map[string]*Rule
	lock  sync.RWMutex
}

// Rule represents a logical inference rule
type Rule struct {
	ID          string
	Name        string
	Premises    []string   // Logical premises
	Conclusion  string     // Logical conclusion
	Confidence  float64    // Rule reliability
	Domain      string     // Application domain (deployment, optimization, etc.)
	Priority    int        // Rule priority for conflict resolution
	CreatedAt   time.Time
}

// FactBase stores known facts
type FactBase struct {
	facts map[string]*Fact
	lock  sync.RWMutex
}

// Fact represents a known fact
type Fact struct {
	Predicate   string                 // e.g., "hasHighLatency"
	Arguments   []string               // e.g., ["vm-123", "us-east-1"]
	Confidence  float64                // Certainty 0.0-1.0
	Source      string                 // Where this fact came from
	Timestamp   time.Time
	Metadata    map[string]interface{}
}

// PDDLPlanner performs automated planning
type PDDLPlanner struct {
	domain      *PlanningDomain
	maxDepth    int
	timeout     time.Duration
}

// PlanningDomain defines the planning domain
type PlanningDomain struct {
	Operators   map[string]*Operator
	Predicates  []string
}

// Operator represents a planning operator
type Operator struct {
	Name         string
	Preconditions []string
	Effects      []string
	Cost         float64
}

// ReasoningMetrics tracks reasoning performance
type ReasoningMetrics struct {
	TotalInferences     int64
	SuccessfulInferences int64
	AvgLatencyMs        float64
	CacheHits           int64
	CacheMisses         int64
}

// NewReasoningEngine creates a new reasoning engine
func NewReasoningEngine(config *cognitive.CognitiveConfig) *ReasoningEngine {
	return &ReasoningEngine{
		config:   config,
		ruleBase: NewRuleBase(),
		factBase: NewFactBase(),
		planner:  NewPDDLPlanner(config.MaxReasoningDepth, config.ReasoningTimeout),
		cache:    make(map[string]*cognitive.ReasoningResult),
	}
}

// NewRuleBase creates a new rule base with default rules
func NewRuleBase() *RuleBase {
	rb := &RuleBase{
		rules: make(map[string]*Rule),
	}

	// Add default infrastructure reasoning rules
	rb.AddRule(&Rule{
		ID:         "high-latency-db-pool",
		Name:       "High latency caused by database connection pool",
		Premises:   []string{"hasHighLatency(?app)", "hasDatabaseDependency(?app, ?db)", "connectionPoolExhausted(?db)"},
		Conclusion: "shouldIncreaseConnectionPool(?db)",
		Confidence: 0.92,
		Domain:     "performance",
		Priority:   10,
	})

	rb.AddRule(&Rule{
		ID:         "cost-optimization-spot",
		Name:       "Cost optimization via spot instances",
		Premises:   []string{"hasHighCost(?workload)", "isStateless(?workload)", "toleratesInterruption(?workload)"},
		Conclusion: "shouldMigrateToSpot(?workload)",
		Confidence: 0.88,
		Domain:     "cost",
		Priority:   8,
	})

	rb.AddRule(&Rule{
		ID:         "security-encryption",
		Name:       "Security requires encryption",
		Premises:   []string{"handlesUserData(?service)", "notEncrypted(?service)"},
		Conclusion: "shouldEnableEncryption(?service)",
		Confidence: 0.98,
		Domain:     "security",
		Priority:   15,
	})

	rb.AddRule(&Rule{
		ID:         "scale-out-high-cpu",
		Name:       "Scale out on high CPU",
		Premises:   []string{"hasCPUUsageAbove(?vm, 80)", "canScaleHorizontally(?vm)"},
		Conclusion: "shouldScaleOut(?vm)",
		Confidence: 0.90,
		Domain:     "scaling",
		Priority:   12,
	})

	return rb
}

// AddRule adds a new rule
func (rb *RuleBase) AddRule(rule *Rule) {
	rb.lock.Lock()
	defer rb.lock.Unlock()
	rule.CreatedAt = time.Now()
	rb.rules[rule.ID] = rule
}

// GetRule retrieves a rule by ID
func (rb *RuleBase) GetRule(id string) (*Rule, bool) {
	rb.lock.RLock()
	defer rb.lock.RUnlock()
	rule, exists := rb.rules[id]
	return rule, exists
}

// MatchRules finds rules that match given facts
func (rb *RuleBase) MatchRules(facts []string) []*Rule {
	rb.lock.RLock()
	defer rb.lock.RUnlock()

	var matched []*Rule
	for _, rule := range rb.rules {
		if rb.premisesMatch(rule.Premises, facts) {
			matched = append(matched, rule)
		}
	}

	return matched
}

// premisesMatch checks if rule premises match facts
func (rb *RuleBase) premisesMatch(premises, facts []string) bool {
	// Simple matching - in production, use proper unification
	factSet := make(map[string]bool)
	for _, fact := range facts {
		factSet[fact] = true
	}

	for _, premise := range premises {
		// Check if premise exists in facts (ignoring variables for now)
		found := false
		premiseBase := strings.Split(premise, "(")[0]
		for fact := range factSet {
			if strings.HasPrefix(fact, premiseBase) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// NewFactBase creates a new fact base
func NewFactBase() *FactBase {
	return &FactBase{
		facts: make(map[string]*Fact),
	}
}

// AddFact adds a fact
func (fb *FactBase) AddFact(fact *Fact) {
	fb.lock.Lock()
	defer fb.lock.Unlock()
	fact.Timestamp = time.Now()
	key := fmt.Sprintf("%s(%s)", fact.Predicate, strings.Join(fact.Arguments, ","))
	fb.facts[key] = fact
}

// GetFacts retrieves all facts
func (fb *FactBase) GetFacts() []*Fact {
	fb.lock.RLock()
	defer fb.lock.RUnlock()

	facts := make([]*Fact, 0, len(fb.facts))
	for _, fact := range fb.facts {
		facts = append(facts, fact)
	}
	return facts
}

// NewPDDLPlanner creates a new planner
func NewPDDLPlanner(maxDepth int, timeout time.Duration) *PDDLPlanner {
	return &PDDLPlanner{
		domain:   NewPlanningDomain(),
		maxDepth: maxDepth,
		timeout:  timeout,
	}
}

// NewPlanningDomain creates a planning domain
func NewPlanningDomain() *PlanningDomain {
	pd := &PlanningDomain{
		Operators:  make(map[string]*Operator),
		Predicates: []string{},
	}

	// Add default operators
	pd.AddOperator(&Operator{
		Name:          "deploy-vm",
		Preconditions: []string{"hasCapacity(?region)", "hasQuota(?user)"},
		Effects:       []string{"vmDeployed(?vm, ?region)", "usedQuota(?user, 1)"},
		Cost:          1.0,
	})

	pd.AddOperator(&Operator{
		Name:          "scale-vm",
		Preconditions: []string{"vmExists(?vm)", "hasCapacity(?region)"},
		Effects:       []string{"vmScaled(?vm)", "increasedCapacity(?vm)"},
		Cost:          0.5,
	})

	return pd
}

// AddOperator adds a planning operator
func (pd *PlanningDomain) AddOperator(op *Operator) {
	pd.Operators[op.Name] = op
}

// Reason performs logical reasoning on a query
func (re *ReasoningEngine) Reason(ctx context.Context, query string, facts []string) (*cognitive.ReasoningResult, error) {
	startTime := time.Now()

	// Check cache
	cacheKey := fmt.Sprintf("%s:%v", query, facts)
	if cached := re.getFromCache(cacheKey); cached != nil {
		re.recordMetrics(time.Since(startTime), true, true)
		return cached, nil
	}

	// Add facts to fact base
	for _, factStr := range facts {
		re.factBase.AddFact(re.parseFact(factStr))
	}

	// Find matching rules
	matchedRules := re.ruleBase.MatchRules(facts)

	// Apply forward chaining
	result := re.forwardChain(matchedRules, facts)

	// Generate explanation
	result.Explanation = re.generateExplanation(result)

	// Find alternatives
	result.Alternatives = re.findAlternatives(query, facts)

	// Cache result
	re.addToCache(cacheKey, result)

	re.recordMetrics(time.Since(startTime), false, len(matchedRules) > 0)

	return result, nil
}

// forwardChain performs forward chaining inference
func (re *ReasoningEngine) forwardChain(rules []*Rule, facts []string) *cognitive.ReasoningResult {
	var steps []cognitive.ReasoningStep
	var conclusions []string
	totalConfidence := 1.0

	for _, rule := range rules {
		step := cognitive.ReasoningStep{
			Rule:       rule.Name,
			Premises:   rule.Premises,
			Conclusion: rule.Conclusion,
			Confidence: rule.Confidence,
			Metadata:   map[string]interface{}{"rule_id": rule.ID},
		}
		steps = append(steps, step)
		conclusions = append(conclusions, rule.Conclusion)
		totalConfidence *= rule.Confidence
	}

	var finalConclusion string
	if len(conclusions) > 0 {
		finalConclusion = conclusions[0]
	} else {
		finalConclusion = "No conclusion could be drawn from available facts"
		totalConfidence = 0.0
	}

	return &cognitive.ReasoningResult{
		Conclusion:  finalConclusion,
		Confidence:  totalConfidence,
		Steps:       steps,
		Alternatives: []cognitive.Alternative{},
		Metadata:    map[string]interface{}{"rule_count": len(rules)},
	}
}

// generateExplanation creates a human-readable explanation
func (re *ReasoningEngine) generateExplanation(result *cognitive.ReasoningResult) string {
	if len(result.Steps) == 0 {
		return "No reasoning steps were performed."
	}

	var explanation strings.Builder
	explanation.WriteString("Reasoning process:\n")

	for i, step := range result.Steps {
		explanation.WriteString(fmt.Sprintf("%d. Applied rule: %s\n", i+1, step.Rule))
		explanation.WriteString(fmt.Sprintf("   Based on: %s\n", strings.Join(step.Premises, ", ")))
		explanation.WriteString(fmt.Sprintf("   Concluded: %s (confidence: %.2f)\n", step.Conclusion, step.Confidence))
	}

	explanation.WriteString(fmt.Sprintf("\nFinal conclusion: %s (overall confidence: %.2f)", result.Conclusion, result.Confidence))

	return explanation.String()
}

// findAlternatives finds alternative solutions
func (re *ReasoningEngine) findAlternatives(query string, facts []string) []cognitive.Alternative {
	// This is simplified - in production, explore different rule chains
	return []cognitive.Alternative{
		{
			Description: "Alternative approach using different optimization strategy",
			Pros:        []string{"Lower risk", "Faster implementation"},
			Cons:        []string{"Higher cost", "Less optimal"},
			Confidence:  0.75,
			Metadata:    map[string]interface{}{"type": "alternative"},
		},
	}
}

// parseFact parses a fact string
func (re *ReasoningEngine) parseFact(factStr string) *Fact {
	// Simple parser: predicate(arg1, arg2, ...)
	openParen := strings.Index(factStr, "(")
	if openParen == -1 {
		return &Fact{
			Predicate:  factStr,
			Arguments:  []string{},
			Confidence: 1.0,
			Source:     "user",
		}
	}

	predicate := factStr[:openParen]
	argsStr := strings.TrimSuffix(factStr[openParen+1:], ")")
	var args []string
	if argsStr != "" {
		args = strings.Split(argsStr, ",")
		for i, arg := range args {
			args[i] = strings.TrimSpace(arg)
		}
	}

	return &Fact{
		Predicate:  predicate,
		Arguments:  args,
		Confidence: 1.0,
		Source:     "user",
	}
}

// getFromCache retrieves from cache
func (re *ReasoningEngine) getFromCache(key string) *cognitive.ReasoningResult {
	re.cacheLock.RLock()
	defer re.cacheLock.RUnlock()
	return re.cache[key]
}

// addToCache adds to cache
func (re *ReasoningEngine) addToCache(key string, result *cognitive.ReasoningResult) {
	re.cacheLock.Lock()
	defer re.cacheLock.Unlock()
	re.cache[key] = result
}

// recordMetrics records reasoning metrics
func (re *ReasoningEngine) recordMetrics(latency time.Duration, cacheHit, success bool) {
	re.metricsLock.Lock()
	defer re.metricsLock.Unlock()

	re.metrics.TotalInferences++
	if success {
		re.metrics.SuccessfulInferences++
	}
	if cacheHit {
		re.metrics.CacheHits++
	} else {
		re.metrics.CacheMisses++
	}

	// Update average latency
	alpha := 0.1
	re.metrics.AvgLatencyMs = alpha*float64(latency.Milliseconds()) + (1-alpha)*re.metrics.AvgLatencyMs
}

// GetMetrics returns reasoning metrics
func (re *ReasoningEngine) GetMetrics() ReasoningMetrics {
	re.metricsLock.RLock()
	defer re.metricsLock.RUnlock()
	return re.metrics
}

// Plan generates a plan to achieve a goal
func (re *ReasoningEngine) Plan(ctx context.Context, initialState, goalState []string) ([]string, error) {
	// Simple forward planning
	plan := []string{}

	// Find operators that can achieve goal
	for _, goal := range goalState {
		for _, op := range re.planner.domain.Operators {
			if re.operatorAchievesGoal(op, goal) {
				plan = append(plan, op.Name)
			}
		}
	}

	return plan, nil
}

// operatorAchievesGoal checks if operator achieves goal
func (re *ReasoningEngine) operatorAchievesGoal(op *Operator, goal string) bool {
	for _, effect := range op.Effects {
		if strings.Contains(effect, goal) {
			return true
		}
	}
	return false
}
