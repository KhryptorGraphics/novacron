// Package explanation provides natural language explanations
package explanation

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/yourusername/novacron/backend/core/cognitive"
)

// ExplanationGenerator creates human-readable explanations
type ExplanationGenerator struct {
	llmClient LLMClient
	templates map[string]string
}

// LLMClient interface for language model integration
type LLMClient interface {
	Complete(ctx context.Context, prompt string) (string, error)
}

// NewExplanationGenerator creates a new explanation generator
func NewExplanationGenerator(llmClient LLMClient) *ExplanationGenerator {
	return &ExplanationGenerator{
		llmClient: llmClient,
		templates: initializeTemplates(),
	}
}

// initializeTemplates creates explanation templates
func initializeTemplates() map[string]string {
	return map[string]string{
		"why_decision": `Explain why the following decision was made:
Decision: %s
Context: %s
Reasoning: %s

Provide a clear, concise explanation.`,

		"what_if": `Explain what would happen if:
Scenario: %s
Current state: %s
Constraints: %s

Describe the likely outcomes.`,

		"how_works": `Explain how this works:
Component: %s
Context: %s

Provide a step-by-step explanation.`,

		"counterfactual": `Explain what would have happened differently if:
Alternative: %s
Actual outcome: %s
Context: %s

Compare the scenarios.`,
	}
}

// ExplainDecision explains why a decision was made
func (eg *ExplanationGenerator) ExplainDecision(ctx context.Context, decision string, reasoning *cognitive.ReasoningResult) (string, error) {
	// Build context from reasoning steps
	var steps []string
	for i, step := range reasoning.Steps {
		steps = append(steps, fmt.Sprintf("%d. %s → %s (confidence: %.2f)", i+1, strings.Join(step.Premises, ", "), step.Conclusion, step.Confidence))
	}

	contextStr := strings.Join(steps, "\n")

	prompt := fmt.Sprintf(eg.templates["why_decision"], decision, contextStr, reasoning.Explanation)

	explanation, err := eg.llmClient.Complete(ctx, prompt)
	if err != nil {
		// Fallback to template-based explanation
		return eg.fallbackDecisionExplanation(decision, reasoning), nil
	}

	return explanation, nil
}

// ExplainWhatIf explains hypothetical scenarios
func (eg *ExplanationGenerator) ExplainWhatIf(ctx context.Context, scenario, currentState string, constraints map[string]interface{}) (string, error) {
	constraintsStr := eg.formatConstraints(constraints)

	prompt := fmt.Sprintf(eg.templates["what_if"], scenario, currentState, constraintsStr)

	explanation, err := eg.llmClient.Complete(ctx, prompt)
	if err != nil {
		return eg.fallbackWhatIfExplanation(scenario, currentState), nil
	}

	return explanation, nil
}

// ExplainHowWorks explains how something works
func (eg *ExplanationGenerator) ExplainHowWorks(ctx context.Context, component, context string) (string, error) {
	prompt := fmt.Sprintf(eg.templates["how_works"], component, context)

	explanation, err := eg.llmClient.Complete(ctx, prompt)
	if err != nil {
		return eg.fallbackHowWorksExplanation(component), nil
	}

	return explanation, nil
}

// ExplainCounterfactual provides counterfactual explanations
func (eg *ExplanationGenerator) ExplainCounterfactual(ctx context.Context, alternative, actualOutcome, context string) (string, error) {
	prompt := fmt.Sprintf(eg.templates["counterfactual"], alternative, actualOutcome, context)

	explanation, err := eg.llmClient.Complete(ctx, prompt)
	if err != nil {
		return eg.fallbackCounterfactualExplanation(alternative, actualOutcome), nil
	}

	return explanation, nil
}

// GenerateVisualization creates visualization data
func (eg *ExplanationGenerator) GenerateVisualization(ctx context.Context, vizType string, data map[string]interface{}) (*Visualization, error) {
	viz := &Visualization{
		Type:      vizType,
		Data:      data,
		CreatedAt: time.Now(),
	}

	switch vizType {
	case "dependency_graph":
		viz.Config = map[string]interface{}{
			"layout":    "hierarchical",
			"direction": "top-bottom",
		}
	case "timeline":
		viz.Config = map[string]interface{}{
			"xAxis": "time",
			"yAxis": "value",
		}
	case "cost_breakdown":
		viz.Config = map[string]interface{}{
			"type": "pie",
		}
	}

	return viz, nil
}

// Visualization represents a visual explanation
type Visualization struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Config    map[string]interface{} `json:"config"`
	CreatedAt time.Time              `json:"created_at"`
}

// fallbackDecisionExplanation provides template-based explanation
func (eg *ExplanationGenerator) fallbackDecisionExplanation(decision string, reasoning *cognitive.ReasoningResult) string {
	var explanation strings.Builder

	explanation.WriteString(fmt.Sprintf("Decision: %s\n\n", decision))
	explanation.WriteString("This decision was made based on the following reasoning:\n\n")

	for i, step := range reasoning.Steps {
		explanation.WriteString(fmt.Sprintf("%d. %s\n", i+1, step.Conclusion))
		explanation.WriteString(fmt.Sprintf("   Confidence: %.2f%%\n\n", step.Confidence*100))
	}

	explanation.WriteString(fmt.Sprintf("Overall confidence: %.2f%%", reasoning.Confidence*100))

	return explanation.String()
}

// fallbackWhatIfExplanation provides template-based what-if explanation
func (eg *ExplanationGenerator) fallbackWhatIfExplanation(scenario, currentState string) string {
	return fmt.Sprintf(`If %s, the following would likely occur:

Current state: %s

Expected changes:
- The system would need to adjust resource allocation
- Performance characteristics may change
- Costs could be affected

Please review the detailed impact analysis for specific metrics.`, scenario, currentState)
}

// fallbackHowWorksExplanation provides template-based how-it-works explanation
func (eg *ExplanationGenerator) fallbackHowWorksExplanation(component string) string {
	return fmt.Sprintf(`%s works by:

1. Receiving input from upstream components
2. Processing the input according to configured rules
3. Producing output for downstream components
4. Monitoring and reporting metrics

For detailed technical documentation, please refer to the system architecture guide.`, component)
}

// fallbackCounterfactualExplanation provides template-based counterfactual explanation
func (eg *ExplanationGenerator) fallbackCounterfactualExplanation(alternative, actualOutcome string) string {
	return fmt.Sprintf(`If %s had been chosen instead:

Actual outcome: %s

The alternative would have resulted in different system behavior, with trade-offs in:
- Performance characteristics
- Cost implications
- Operational complexity
- Risk profile

A detailed comparison analysis can be generated upon request.`, alternative, actualOutcome)
}

// formatConstraints formats constraints for display
func (eg *ExplanationGenerator) formatConstraints(constraints map[string]interface{}) string {
	var parts []string
	for key, value := range constraints {
		parts = append(parts, fmt.Sprintf("%s: %v", key, value))
	}
	return strings.Join(parts, ", ")
}

// SimpleLLMClient is a mock implementation
type SimpleLLMClient struct{}

// Complete implements mock completion
func (c *SimpleLLMClient) Complete(ctx context.Context, prompt string) (string, error) {
	return "This is a mock explanation. In production, this would use GPT-4 to generate detailed, context-aware explanations.", nil
}
