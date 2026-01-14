package policy

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"gopkg.in/yaml.v2"
)

// DefaultPolicyParser implements the PolicyParser interface
type DefaultPolicyParser struct {
	keywords map[string]TokenType
}

// NewDefaultPolicyParser creates a new policy parser
func NewDefaultPolicyParser() *DefaultPolicyParser {
	keywords := map[string]TokenType{
		"policy":     TokenTypeKeyword,
		"rule":       TokenTypeKeyword,
		"when":       TokenTypeKeyword,
		"then":       TokenTypeKeyword,
		"if":         TokenTypeKeyword,
		"and":        TokenTypeKeyword,
		"or":         TokenTypeKeyword,
		"not":        TokenTypeKeyword,
		"matches":    TokenTypeKeyword,
		"contains":   TokenTypeKeyword,
		"in":         TokenTypeKeyword,
		"exists":     TokenTypeKeyword,
		"action":     TokenTypeKeyword,
		"schedule":   TokenTypeKeyword,
		"selector":   TokenTypeKeyword,
		"true":       TokenTypeKeyword,
		"false":      TokenTypeKeyword,
	}

	return &DefaultPolicyParser{
		keywords: keywords,
	}
}

// ParsePolicy parses a policy from DSL string
func (p *DefaultPolicyParser) ParsePolicy(dsl string) (*OrchestrationPolicy, error) {
	// First try to parse as YAML
	if yamlPolicy, err := p.parseYAMLPolicy(dsl); err == nil {
		return yamlPolicy, nil
	}

	// Then try to parse as JSON
	if jsonPolicy, err := p.parseJSONPolicy(dsl); err == nil {
		return jsonPolicy, nil
	}

	// Finally try to parse as custom DSL
	return p.parseCustomDSL(dsl)
}

// ParseRule parses a single rule from DSL string
func (p *DefaultPolicyParser) ParseRule(dsl string) (*PolicyRule, error) {
	// Tokenize the DSL
	tokens, err := p.tokenize(dsl)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	// Parse tokens into rule structure
	return p.parseRuleFromTokens(tokens)
}

// ParseCondition parses a condition from DSL string
func (p *DefaultPolicyParser) ParseCondition(dsl string) (*RuleCondition, error) {
	// Simple condition parsing
	// Format: field operator value
	// Example: "cpu_usage > 0.8", "labels.env == 'production'"

	parts := p.splitCondition(dsl)
	if len(parts) != 3 {
		return nil, fmt.Errorf("invalid condition format, expected 'field operator value'")
	}

	field := strings.TrimSpace(parts[0])
	operatorStr := strings.TrimSpace(parts[1])
	valueStr := strings.TrimSpace(parts[2])

	// Parse operator
	operator, err := p.parseOperator(operatorStr)
	if err != nil {
		return nil, fmt.Errorf("invalid operator '%s': %w", operatorStr, err)
	}

	// Parse value
	value, err := p.parseValue(valueStr)
	if err != nil {
		return nil, fmt.Errorf("invalid value '%s': %w", valueStr, err)
	}

	// Determine condition type based on field
	conditionType := p.inferConditionType(field)

	return &RuleCondition{
		ID:       uuid.New().String(),
		Type:     conditionType,
		Field:    field,
		Operator: operator,
		Value:    value,
	}, nil
}

// ValidateSyntax validates the syntax of policy DSL
func (p *DefaultPolicyParser) ValidateSyntax(dsl string) (*SyntaxValidationResult, error) {
	result := &SyntaxValidationResult{
		Valid:  true,
		Errors: []SyntaxError{},
		Tokens: []Token{},
	}

	// Tokenize
	tokens, err := p.tokenize(dsl)
	if err != nil {
		result.Valid = false
		result.Errors = append(result.Errors, SyntaxError{
			Message: err.Error(),
			Line:    1,
			Column:  1,
		})
		return result, nil
	}

	result.Tokens = tokens

	// Basic syntax validation
	if err := p.validateTokenSequence(tokens); err != nil {
		result.Valid = false
		result.Errors = append(result.Errors, SyntaxError{
			Message: err.Error(),
			Line:    1,
			Column:  1,
		})
	}

	return result, nil
}

// Private methods

func (p *DefaultPolicyParser) parseYAMLPolicy(dsl string) (*OrchestrationPolicy, error) {
	var policyDSL PolicyDSL
	if err := yaml.Unmarshal([]byte(dsl), &policyDSL); err != nil {
		return nil, err
	}

	return p.convertDSLToPolicy(&policyDSL)
}

func (p *DefaultPolicyParser) parseJSONPolicy(dsl string) (*OrchestrationPolicy, error) {
	var policyDSL PolicyDSL
	if err := json.Unmarshal([]byte(dsl), &policyDSL); err != nil {
		return nil, err
	}

	return p.convertDSLToPolicy(&policyDSL)
}

func (p *DefaultPolicyParser) parseCustomDSL(dsl string) (*OrchestrationPolicy, error) {
	// Simple custom DSL parser
	lines := strings.Split(dsl, "\n")
	policy := &OrchestrationPolicy{
		ID:        uuid.New().String(),
		Version:   "1.0",
		Enabled:   true,
		Priority:  1,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Rules:     []*PolicyRule{},
	}

	currentRule := (*PolicyRule)(nil)

	for lineNum, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}

		if err := p.parseCustomDSLLine(line, policy, &currentRule, lineNum+1); err != nil {
			return nil, fmt.Errorf("error on line %d: %w", lineNum+1, err)
		}
	}

	// Add the last rule if exists
	if currentRule != nil {
		policy.Rules = append(policy.Rules, currentRule)
	}

	return policy, nil
}

func (p *DefaultPolicyParser) parseCustomDSLLine(line string, policy *OrchestrationPolicy, currentRule **PolicyRule, lineNum int) error {
	// Parse different line types
	if strings.HasPrefix(line, "policy ") {
		// Policy declaration: policy "name" { ... }
		return p.parsePolicyDeclaration(line, policy)
	} else if strings.HasPrefix(line, "rule ") {
		// Rule declaration: rule "name" { ... }
		if *currentRule != nil {
			policy.Rules = append(policy.Rules, *currentRule)
		}
		
		rule, err := p.parseRuleDeclaration(line)
		if err != nil {
			return err
		}
		*currentRule = rule
		return nil
	} else if strings.HasPrefix(line, "when ") || strings.HasPrefix(line, "if ") {
		// Condition: when cpu_usage > 0.8
		if *currentRule == nil {
			return fmt.Errorf("condition without rule context")
		}
		return p.parseConditionLine(line, *currentRule)
	} else if strings.HasPrefix(line, "then ") || strings.HasPrefix(line, "action ") {
		// Action: then scale up by 2
		if *currentRule == nil {
			return fmt.Errorf("action without rule context")
		}
		return p.parseActionLine(line, *currentRule)
	} else if strings.Contains(line, ":") {
		// Property assignment: priority: 5
		return p.parsePropertyLine(line, policy, *currentRule)
	}

	return fmt.Errorf("unrecognized line format: %s", line)
}

func (p *DefaultPolicyParser) parsePolicyDeclaration(line string, policy *OrchestrationPolicy) error {
	// Extract policy name
	re := regexp.MustCompile(`policy\s+"([^"]+)"`)
	matches := re.FindStringSubmatch(line)
	if len(matches) > 1 {
		policy.Name = matches[1]
	}
	return nil
}

func (p *DefaultPolicyParser) parseRuleDeclaration(line string) (*PolicyRule, error) {
	rule := &PolicyRule{
		ID:         uuid.New().String(),
		Enabled:    true,
		Priority:   1,
		Conditions: []*RuleCondition{},
		Actions:    []*RuleAction{},
	}

	// Extract rule name and type
	re := regexp.MustCompile(`rule\s+"([^"]+)"\s*(?:type\s+(\w+))?`)
	matches := re.FindStringSubmatch(line)
	if len(matches) > 1 {
		rule.Name = matches[1]
	}
	if len(matches) > 2 && matches[2] != "" {
		rule.Type = PolicyRuleType(matches[2])
	} else {
		rule.Type = RuleTypeCustom
	}

	return rule, nil
}

func (p *DefaultPolicyParser) parseConditionLine(line string, rule *PolicyRule) error {
	// Remove "when " or "if " prefix
	conditionStr := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "when "), "if "))
	
	condition, err := p.ParseCondition(conditionStr)
	if err != nil {
		return err
	}

	rule.Conditions = append(rule.Conditions, condition)
	return nil
}

func (p *DefaultPolicyParser) parseActionLine(line string, rule *PolicyRule) error {
	// Remove "then " or "action " prefix
	actionStr := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "then "), "action "))
	
	action, err := p.parseAction(actionStr)
	if err != nil {
		return err
	}

	rule.Actions = append(rule.Actions, action)
	return nil
}

func (p *DefaultPolicyParser) parsePropertyLine(line string, policy *OrchestrationPolicy, rule *PolicyRule) error {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid property format")
	}

	key := strings.TrimSpace(parts[0])
	value := strings.TrimSpace(parts[1])

	// Parse value
	parsedValue, err := p.parseValue(value)
	if err != nil {
		return err
	}

	// Apply property to appropriate object
	if rule != nil {
		return p.setRuleProperty(rule, key, parsedValue)
	} else {
		return p.setPolicyProperty(policy, key, parsedValue)
	}
}

func (p *DefaultPolicyParser) setRuleProperty(rule *PolicyRule, key string, value interface{}) error {
	switch key {
	case "priority":
		if priority, ok := value.(int); ok {
			rule.Priority = priority
		}
	case "enabled":
		if enabled, ok := value.(bool); ok {
			rule.Enabled = enabled
		}
	case "type":
		if ruleType, ok := value.(string); ok {
			rule.Type = PolicyRuleType(ruleType)
		}
	default:
		if rule.Parameters == nil {
			rule.Parameters = make(map[string]interface{})
		}
		rule.Parameters[key] = value
	}
	return nil
}

func (p *DefaultPolicyParser) setPolicyProperty(policy *OrchestrationPolicy, key string, value interface{}) error {
	switch key {
	case "priority":
		if priority, ok := value.(int); ok {
			policy.Priority = priority
		}
	case "enabled":
		if enabled, ok := value.(bool); ok {
			policy.Enabled = enabled
		}
	case "description":
		if description, ok := value.(string); ok {
			policy.Description = description
		}
	case "namespace":
		if namespace, ok := value.(string); ok {
			policy.Namespace = namespace
		}
	default:
		if policy.Metadata == nil {
			policy.Metadata = make(map[string]interface{})
		}
		policy.Metadata[key] = value
	}
	return nil
}

func (p *DefaultPolicyParser) parseAction(actionStr string) (*RuleAction, error) {
	action := &RuleAction{
		ID:         uuid.New().String(),
		Parameters: make(map[string]interface{}),
	}

	// Parse action format: "scale up by 2", "migrate to node-01", "alert severity high"
	words := strings.Fields(actionStr)
	if len(words) == 0 {
		return nil, fmt.Errorf("empty action")
	}

	// First word is the action type
	action.Type = ActionType(words[0])

	// Parse parameters based on action type
	switch action.Type {
	case ActionTypeScale:
		if len(words) >= 3 && words[1] == "up" {
			action.Parameters["direction"] = "up"
			if len(words) >= 4 && words[2] == "by" {
				if count, err := strconv.Atoi(words[3]); err == nil {
					action.Parameters["count"] = count
				}
			}
		} else if len(words) >= 3 && words[1] == "down" {
			action.Parameters["direction"] = "down"
			if len(words) >= 4 && words[2] == "by" {
				if count, err := strconv.Atoi(words[3]); err == nil {
					action.Parameters["count"] = count
				}
			}
		}

	case ActionTypeMigrate:
		if len(words) >= 3 && words[1] == "to" {
			action.Parameters["destination"] = words[2]
		}

	case ActionTypeAlert:
		for i := 1; i < len(words); i += 2 {
			if i+1 < len(words) {
				action.Parameters[words[i]] = words[i+1]
			}
		}

	default:
		// Generic parameter parsing
		for i := 1; i < len(words); i += 2 {
			if i+1 < len(words) {
				key := words[i]
				value := words[i+1]
				if parsedValue, err := p.parseValue(value); err == nil {
					action.Parameters[key] = parsedValue
				} else {
					action.Parameters[key] = value
				}
			}
		}
	}

	return action, nil
}

func (p *DefaultPolicyParser) convertDSLToPolicy(policyDSL *PolicyDSL) (*OrchestrationPolicy, error) {
	policy := &OrchestrationPolicy{
		ID:        uuid.New().String(),
		Version:   policyDSL.Version,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Enabled:   true,
		Rules:     []*PolicyRule{},
	}

	if policyDSL.Spec == nil {
		return nil, fmt.Errorf("policy spec is required")
	}

	policy.Name = policyDSL.Spec.Name
	policy.Description = policyDSL.Spec.Description

	// Parse selector
	if policyDSL.Spec.Selector != nil {
		selector, err := p.parseSelector(policyDSL.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("failed to parse selector: %w", err)
		}
		policy.Selector = selector
	}

	// Parse rules
	for _, ruleMap := range policyDSL.Spec.Rules {
		rule, err := p.parseRuleFromMap(ruleMap)
		if err != nil {
			return nil, fmt.Errorf("failed to parse rule: %w", err)
		}
		policy.Rules = append(policy.Rules, rule)
	}

	return policy, nil
}

func (p *DefaultPolicyParser) parseSelector(selectorMap map[string]interface{}) (*PolicySelector, error) {
	selector := &PolicySelector{}

	if matchLabels, ok := selectorMap["matchLabels"].(map[string]interface{}); ok {
		selector.MatchLabels = make(map[string]string)
		for k, v := range matchLabels {
			if str, ok := v.(string); ok {
				selector.MatchLabels[k] = str
			}
		}
	}

	if resourceTypes, ok := selectorMap["resourceTypes"].([]interface{}); ok {
		for _, rt := range resourceTypes {
			if str, ok := rt.(string); ok {
				selector.ResourceTypes = append(selector.ResourceTypes, ResourceType(str))
			}
		}
	}

	if celExpr, ok := selectorMap["celExpression"].(string); ok {
		selector.CELExpression = celExpr
	}

	return selector, nil
}

func (p *DefaultPolicyParser) parseRuleFromMap(ruleMap map[string]interface{}) (*PolicyRule, error) {
	rule := &PolicyRule{
		ID:         uuid.New().String(),
		Enabled:    true,
		Priority:   1,
		Conditions: []*RuleCondition{},
		Actions:    []*RuleAction{},
		Parameters: make(map[string]interface{}),
	}

	if name, ok := ruleMap["name"].(string); ok {
		rule.Name = name
	}

	if ruleType, ok := ruleMap["type"].(string); ok {
		rule.Type = PolicyRuleType(ruleType)
	}

	if priority, ok := ruleMap["priority"].(int); ok {
		rule.Priority = priority
	} else if priority, ok := ruleMap["priority"].(float64); ok {
		rule.Priority = int(priority)
	}

	if enabled, ok := ruleMap["enabled"].(bool); ok {
		rule.Enabled = enabled
	}

	// Parse conditions
	if conditions, ok := ruleMap["conditions"].([]interface{}); ok {
		for _, conditionInterface := range conditions {
			if conditionMap, ok := conditionInterface.(map[string]interface{}); ok {
				condition, err := p.parseConditionFromMap(conditionMap)
				if err != nil {
					return nil, fmt.Errorf("failed to parse condition: %w", err)
				}
				rule.Conditions = append(rule.Conditions, condition)
			}
		}
	}

	// Parse actions
	if actions, ok := ruleMap["actions"].([]interface{}); ok {
		for _, actionInterface := range actions {
			if actionMap, ok := actionInterface.(map[string]interface{}); ok {
				action, err := p.parseActionFromMap(actionMap)
				if err != nil {
					return nil, fmt.Errorf("failed to parse action: %w", err)
				}
				rule.Actions = append(rule.Actions, action)
			}
		}
	}

	return rule, nil
}

func (p *DefaultPolicyParser) parseConditionFromMap(conditionMap map[string]interface{}) (*RuleCondition, error) {
	condition := &RuleCondition{
		ID: uuid.New().String(),
	}

	if condType, ok := conditionMap["type"].(string); ok {
		condition.Type = ConditionType(condType)
	}

	if field, ok := conditionMap["field"].(string); ok {
		condition.Field = field
	}

	if operator, ok := conditionMap["operator"].(string); ok {
		condition.Operator = ConditionOperator(operator)
	}

	if value, ok := conditionMap["value"]; ok {
		condition.Value = value
	}

	return condition, nil
}

func (p *DefaultPolicyParser) parseActionFromMap(actionMap map[string]interface{}) (*RuleAction, error) {
	action := &RuleAction{
		ID:         uuid.New().String(),
		Parameters: make(map[string]interface{}),
	}

	if actionType, ok := actionMap["type"].(string); ok {
		action.Type = ActionType(actionType)
	}

	if target, ok := actionMap["target"].(string); ok {
		action.Target = target
	}

	if parameters, ok := actionMap["parameters"].(map[string]interface{}); ok {
		action.Parameters = parameters
	}

	return action, nil
}

func (p *DefaultPolicyParser) parseRuleFromTokens(tokens []Token) (*PolicyRule, error) {
	// Simple token-based rule parser
	rule := &PolicyRule{
		ID:         uuid.New().String(),
		Enabled:    true,
		Priority:   1,
		Conditions: []*RuleCondition{},
		Actions:    []*RuleAction{},
	}

	// This is a simplified implementation
	// In a real parser, you would build an AST and traverse it
	for _, token := range tokens {
		if token.Type == TokenTypeKeyword && token.Value == "rule" {
			// Found rule keyword, expect rule name next
		}
	}

	return rule, nil
}

func (p *DefaultPolicyParser) tokenize(dsl string) ([]Token, error) {
	var tokens []Token
	lines := strings.Split(dsl, "\n")

	for lineNum, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		lineTokens, err := p.tokenizeLine(line, lineNum+1)
		if err != nil {
			return nil, err
		}

		tokens = append(tokens, lineTokens...)
	}

	return tokens, nil
}

func (p *DefaultPolicyParser) tokenizeLine(line string, lineNum int) ([]Token, error) {
	var tokens []Token
	
	// Simple tokenizer using regex patterns
	patterns := []struct {
		regex *regexp.Regexp
		tokenType TokenType
	}{
		{regexp.MustCompile(`^"([^"]*)"`), TokenTypeString},
		{regexp.MustCompile(`^'([^']*)' `), TokenTypeString},
		{regexp.MustCompile(`^\d+(\.\d+)?`), TokenTypeNumber},
		{regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_.]*`), TokenTypeIdentifier},
		{regexp.MustCompile(`^(==|!=|<=|>=|<|>|&&|\|\||!)`), TokenTypeOperator},
		{regexp.MustCompile(`^[(){}[\],;:]`), TokenTypeDelimiter},
		{regexp.MustCompile(`^//.*`), TokenTypeComment},
		{regexp.MustCompile(`^#.*`), TokenTypeComment},
	}

	pos := 0
	for pos < len(line) {
		// Skip whitespace
		for pos < len(line) && (line[pos] == ' ' || line[pos] == '\t') {
			pos++
		}

		if pos >= len(line) {
			break
		}

		matched := false
		for _, pattern := range patterns {
			if match := pattern.regex.FindString(line[pos:]); match != "" {
				value := match
				if pattern.tokenType == TokenTypeString {
					// Remove quotes
					value = strings.Trim(match, `"'`)
				}

				// Check if identifier is a keyword
				tokenType := pattern.tokenType
				if tokenType == TokenTypeIdentifier {
					if _, isKeyword := p.keywords[value]; isKeyword {
						tokenType = TokenTypeKeyword
					}
				}

				tokens = append(tokens, Token{
					Type:   tokenType,
					Value:  value,
					Line:   lineNum,
					Column: pos + 1,
				})

				pos += len(match)
				matched = true
				break
			}
		}

		if !matched {
			return nil, fmt.Errorf("unexpected character '%c' at line %d, column %d", line[pos], lineNum, pos+1)
		}
	}

	return tokens, nil
}

func (p *DefaultPolicyParser) validateTokenSequence(tokens []Token) error {
	// Basic token sequence validation
	if len(tokens) == 0 {
		return fmt.Errorf("empty token sequence")
	}

	// Check for balanced delimiters
	stack := []rune{}
	for _, token := range tokens {
		if token.Type == TokenTypeDelimiter {
			switch token.Value {
			case "(", "{", "[":
				stack = append(stack, rune(token.Value[0]))
			case ")", "}", "]":
				if len(stack) == 0 {
					return fmt.Errorf("unmatched closing delimiter '%s'", token.Value)
				}
				expected := stack[len(stack)-1]
				stack = stack[:len(stack)-1]

				var expectedClosing rune
				switch expected {
				case '(':
					expectedClosing = ')'
				case '{':
					expectedClosing = '}'
				case '[':
					expectedClosing = ']'
				}

				if rune(token.Value[0]) != expectedClosing {
					return fmt.Errorf("mismatched delimiter, expected '%c' but found '%s'", expectedClosing, token.Value)
				}
			}
		}
	}

	if len(stack) > 0 {
		return fmt.Errorf("unmatched opening delimiter")
	}

	return nil
}

func (p *DefaultPolicyParser) splitCondition(condition string) []string {
	// Split condition by operators, preserving quoted strings
	operators := []string{"==", "!=", "<=", ">=", "<", ">", "matches", "contains", "in", "exists"}
	
	for _, op := range operators {
		if strings.Contains(condition, op) {
			parts := strings.SplitN(condition, op, 2)
			if len(parts) == 2 {
				return []string{parts[0], op, parts[1]}
			}
		}
	}

	// Default split by spaces if no operator found
	words := strings.Fields(condition)
	if len(words) >= 3 {
		return []string{words[0], words[1], strings.Join(words[2:], " ")}
	}

	return words
}

func (p *DefaultPolicyParser) parseOperator(operatorStr string) (ConditionOperator, error) {
	operatorMap := map[string]ConditionOperator{
		"==":          OperatorEquals,
		"!=":          OperatorNotEquals,
		">":           OperatorGreaterThan,
		">=":          OperatorGreaterThanOrEqual,
		"<":           OperatorLessThan,
		"<=":          OperatorLessThanOrEqual,
		"in":          OperatorIn,
		"not_in":      OperatorNotIn,
		"contains":    OperatorContains,
		"not_contains": OperatorNotContains,
		"matches":     OperatorMatches,
		"exists":      OperatorExists,
	}

	if op, exists := operatorMap[operatorStr]; exists {
		return op, nil
	}

	return "", fmt.Errorf("unknown operator: %s", operatorStr)
}

func (p *DefaultPolicyParser) parseValue(valueStr string) (interface{}, error) {
	// Remove surrounding quotes if present
	valueStr = strings.Trim(valueStr, `"'`)

	// Try to parse as number
	if intVal, err := strconv.Atoi(valueStr); err == nil {
		return intVal, nil
	}

	if floatVal, err := strconv.ParseFloat(valueStr, 64); err == nil {
		return floatVal, nil
	}

	// Try to parse as boolean
	if boolVal, err := strconv.ParseBool(valueStr); err == nil {
		return boolVal, nil
	}

	// Return as string
	return valueStr, nil
}

func (p *DefaultPolicyParser) inferConditionType(field string) ConditionType {
	if strings.HasPrefix(field, "labels.") {
		return ConditionTypeLabel
	}

	if strings.HasPrefix(field, "tags.") {
		return ConditionTypeTag
	}

	if strings.Contains(field, "_usage") || strings.Contains(field, "_utilization") {
		return ConditionTypeMetric
	}

	if field == "state" || field == "status" {
		return ConditionTypeState
	}

	return ConditionTypeCustom
}