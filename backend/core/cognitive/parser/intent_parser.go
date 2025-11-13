// Package parser provides intent parsing from natural language
package parser

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cognitive"
)

// IntentParser extracts structured intents from natural language
type IntentParser struct {
	llmClient LLMClient
	patterns  map[string]*regexp.Regexp
}

// LLMClient interface for language model integration
type LLMClient interface {
	Complete(ctx context.Context, prompt string) (string, error)
}

// NewIntentParser creates a new intent parser
func NewIntentParser(llmClient LLMClient) *IntentParser {
	return &IntentParser{
		llmClient: llmClient,
		patterns:  initializePatterns(),
	}
}

// initializePatterns creates regex patterns for quick entity extraction
func initializePatterns() map[string]*regexp.Regexp {
	return map[string]*regexp.Regexp{
		"region":     regexp.MustCompile(`\b(us-east-1|us-west-2|eu-west-1|ap-southeast-1|AWS|GCP|Azure)\b`),
		"latency":    regexp.MustCompile(`\b(\d+)\s*(ms|milliseconds?)\b`),
		"cost":       regexp.MustCompile(`\$(\d+(?:\.\d{2})?)`),
		"percentage": regexp.MustCompile(`(\d+)%`),
		"vm_count":   regexp.MustCompile(`\b(\d+)\s*(VMs?|virtual machines?|instances?)\b`),
		"size":       regexp.MustCompile(`\b(small|medium|large|x-large|t2\.\w+|c5\.\w+)\b`),
	}
}

// Parse extracts intent from natural language
func (p *IntentParser) Parse(ctx context.Context, text string, context *cognitive.ConversationContext) (*cognitive.Intent, error) {
	// First, try pattern-based extraction for speed
	entities := p.extractEntitiesWithPatterns(text)

	// Use LLM for complex intent understanding
	llmIntent, err := p.extractIntentWithLLM(ctx, text, context)
	if err != nil {
		return nil, fmt.Errorf("LLM intent extraction failed: %w", err)
	}

	// Merge pattern-based and LLM-based results
	intent := p.mergeIntents(llmIntent, entities)

	// Validate and check for ambiguities
	intent.Ambiguities = p.findAmbiguities(intent, text, context)

	return intent, nil
}

// extractEntitiesWithPatterns uses regex for quick entity extraction
func (p *IntentParser) extractEntitiesWithPatterns(text string) map[string]interface{} {
	entities := make(map[string]interface{})

	for entityType, pattern := range p.patterns {
		matches := pattern.FindAllStringSubmatch(text, -1)
		if len(matches) > 0 {
			if len(matches) == 1 {
				entities[entityType] = matches[0][1]
			} else {
				var values []string
				for _, match := range matches {
					values = append(values, match[1])
				}
				entities[entityType] = values
			}
		}
	}

	return entities
}

// extractIntentWithLLM uses LLM for sophisticated intent understanding
func (p *IntentParser) extractIntentWithLLM(ctx context.Context, text string, context *cognitive.ConversationContext) (*cognitive.Intent, error) {
	// Build prompt with conversation context
	var historyContext string
	if context != nil && len(context.History) > 0 {
		lastTurns := context.History
		if len(lastTurns) > 3 {
			lastTurns = lastTurns[len(lastTurns)-3:]
		}

		var turns []string
		for _, turn := range lastTurns {
			turns = append(turns, fmt.Sprintf("User: %s\nAssistant: %s", turn.UserMessage, turn.AIResponse))
		}
		historyContext = "Conversation history:\n" + strings.Join(turns, "\n\n")
	}

	prompt := fmt.Sprintf(`Extract structured intent from the following user message. Return a JSON object with:
- action: The main action (deploy, migrate, optimize, diagnose, scale, query, delete, update, configure)
- entities: Key entities mentioned (VMs, regions, services, metrics, etc.)
- constraints: Constraints or requirements (latency, cost, security, compliance)
- confidence: Confidence score 0.0-1.0
- needs_confirmation: Boolean if this requires user confirmation before execution

%s

Current message: "%s"

Respond with ONLY valid JSON, no additional text:`, historyContext, text)

	response, err := p.llmClient.Complete(ctx, prompt)
	if err != nil {
		return nil, err
	}

	// Parse JSON response
	var intent cognitive.Intent
	if err := json.Unmarshal([]byte(response), &intent); err != nil {
		return nil, fmt.Errorf("failed to parse LLM intent JSON: %w", err)
	}

	return &intent, nil
}

// mergeIntents combines pattern-based and LLM-based intents
func (p *IntentParser) mergeIntents(llmIntent *cognitive.Intent, patternEntities map[string]interface{}) *cognitive.Intent {
	if llmIntent.Entities == nil {
		llmIntent.Entities = make(map[string]interface{})
	}
	if llmIntent.Constraints == nil {
		llmIntent.Constraints = make(map[string]interface{})
	}

	// Merge pattern entities (pattern-based takes precedence for specific types)
	for key, value := range patternEntities {
		if _, exists := llmIntent.Entities[key]; !exists {
			llmIntent.Entities[key] = value
		}
	}

	return llmIntent
}

// findAmbiguities identifies unclear aspects of the intent
func (p *IntentParser) findAmbiguities(intent *cognitive.Intent, text string, context *cognitive.ConversationContext) []string {
	var ambiguities []string

	// Check if action is clear
	if intent.Action == "" || intent.Action == "unknown" {
		ambiguities = append(ambiguities, "The intended action is unclear. Do you want to deploy, migrate, optimize, or diagnose?")
	}

	// Check if critical entities are missing for specific actions
	switch intent.Action {
	case "deploy":
		if _, hasRegion := intent.Entities["region"]; !hasRegion {
			ambiguities = append(ambiguities, "Which region should I deploy to?")
		}
		if _, hasType := intent.Entities["vm_type"]; !hasType {
			ambiguities = append(ambiguities, "What type/size of VM should I deploy?")
		}

	case "migrate":
		if _, hasSource := intent.Entities["source"]; !hasSource {
			ambiguities = append(ambiguities, "Which source location should I migrate from?")
		}
		if _, hasTarget := intent.Entities["target"]; !hasTarget {
			ambiguities = append(ambiguities, "Which target location should I migrate to?")
		}

	case "optimize":
		if _, hasTarget := intent.Entities["optimization_target"]; !hasTarget {
			ambiguities = append(ambiguities, "What should I optimize? (cost, performance, security)")
		}

	case "diagnose":
		if _, hasProblem := intent.Entities["problem"]; !hasProblem {
			if _, hasSymptom := intent.Entities["symptom"]; !hasSymptom {
				ambiguities = append(ambiguities, "What specific issue are you experiencing?")
			}
		}
	}

	// Check confidence threshold
	if intent.Confidence < 0.7 {
		ambiguities = append(ambiguities, "I'm not very confident I understood correctly. Could you rephrase?")
	}

	return ambiguities
}

// ClassifyAction determines the action type from text
func (p *IntentParser) ClassifyAction(text string) string {
	textLower := strings.ToLower(text)

	actionPatterns := map[string][]string{
		"deploy":    {"deploy", "create", "launch", "start", "provision", "spin up"},
		"migrate":   {"migrate", "move", "transfer", "relocate", "shift"},
		"optimize":  {"optimize", "improve", "enhance", "tune", "reduce cost", "save money"},
		"diagnose":  {"diagnose", "debug", "troubleshoot", "why", "what's wrong", "issue", "problem", "slow"},
		"scale":     {"scale", "increase", "decrease", "grow", "shrink", "resize"},
		"query":     {"show", "list", "get", "what", "which", "how many"},
		"delete":    {"delete", "remove", "terminate", "destroy", "stop"},
		"update":    {"update", "modify", "change", "configure", "set"},
		"configure": {"configure", "setup", "enable", "disable", "adjust"},
	}

	for action, keywords := range actionPatterns {
		for _, keyword := range keywords {
			if strings.Contains(textLower, keyword) {
				return action
			}
		}
	}

	return "unknown"
}

// ExtractEntities extracts named entities from text
func (p *IntentParser) ExtractEntities(ctx context.Context, text string) (map[string]interface{}, error) {
	// Use pattern-based extraction
	entities := p.extractEntitiesWithPatterns(text)

	// Enhance with LLM if needed
	if len(entities) < 2 {
		prompt := fmt.Sprintf(`Extract all relevant entities from this infrastructure-related message: "%s"

Return JSON with entities like: {"vm_ids": [...], "regions": [...], "metrics": [...], "services": [...], etc.}`, text)

		response, err := p.llmClient.Complete(ctx, prompt)
		if err != nil {
			return entities, nil // Fall back to pattern-based
		}

		var llmEntities map[string]interface{}
		if err := json.Unmarshal([]byte(response), &llmEntities); err == nil {
			for k, v := range llmEntities {
				entities[k] = v
			}
		}
	}

	return entities, nil
}

// ValidateIntent checks if an intent is valid and complete
func (p *IntentParser) ValidateIntent(intent *cognitive.Intent) error {
	if intent.Action == "" {
		return fmt.Errorf("action is required")
	}

	if intent.Confidence < 0 || intent.Confidence > 1 {
		return fmt.Errorf("confidence must be between 0 and 1")
	}

	// Validate destructive actions require high confidence
	destructiveActions := map[string]bool{
		"delete":  true,
		"destroy": true,
		"migrate": true,
	}

	if destructiveActions[intent.Action] && intent.Confidence < 0.9 {
		return fmt.Errorf("destructive action %s requires confidence >= 0.9", intent.Action)
	}

	return nil
}

// SimpleLLMClient is a mock LLM client for testing
type SimpleLLMClient struct{}

// Complete implements a simple mock completion
func (c *SimpleLLMClient) Complete(ctx context.Context, prompt string) (string, error) {
	// Mock response for testing
	mockIntent := cognitive.Intent{
		Action:     "deploy",
		Entities:   map[string]interface{}{"region": "us-east-1", "vm_type": "t2.medium"},
		Constraints: map[string]interface{}{"latency": "50ms"},
		Confidence: 0.95,
	}

	jsonData, err := json.Marshal(mockIntent)
	if err != nil {
		return "", err
	}

	return string(jsonData), nil
}
