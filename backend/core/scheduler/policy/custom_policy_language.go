package policy

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

// CustomPolicyLanguageParser extends the basic PolicyParser with advanced features
// for parsing policies written in the custom policy language
type CustomPolicyLanguageParser struct {
	// Base parser
	*PolicyParser
	
	// ExpressionCompiler compiles policy expressions
	ExpressionCompiler ExpressionCompiler
	
	// ValidationRules for validating policies
	ValidationRules []*ValidationRule
}

// ValidationRule represents a rule for validating policies
type ValidationRule struct {
	// Name is the name of the rule
	Name string
	
	// Description describes the rule
	Description string
	
	// Validate validates a policy against this rule
	Validate func(policy *SchedulingPolicy) (bool, string)
}

// NewCustomPolicyLanguageParser creates a new custom policy language parser
func NewCustomPolicyLanguageParser() *CustomPolicyLanguageParser {
	return &CustomPolicyLanguageParser{
		PolicyParser:      NewPolicyParser(),
		ExpressionCompiler: &DefaultExpressionCompiler{},
		ValidationRules:   defaultValidationRules(),
	}
}

// defaultValidationRules returns the default validation rules
func defaultValidationRules() []*ValidationRule {
	return []*ValidationRule{
		{
			Name:        "policy-id-format",
			Description: "Policy ID must be lowercase alphanumeric with hyphens",
			Validate: func(policy *SchedulingPolicy) (bool, string) {
				match, _ := regexp.MatchString("^[a-z0-9-]+$", policy.ID)
				if !match {
					return false, "Policy ID must be lowercase alphanumeric with hyphens"
				}
				return true, ""
			},
		},
		{
			Name:        "policy-name-length",
			Description: "Policy name must be between 3 and 100 characters",
			Validate: func(policy *SchedulingPolicy) (bool, string) {
				if len(policy.Name) < 3 || len(policy.Name) > 100 {
					return false, "Policy name must be between 3 and 100 characters"
				}
				return true, ""
			},
		},
		{
			Name:        "policy-description-length",
			Description: "Policy description must be between 10 and 1000 characters",
			Validate: func(policy *SchedulingPolicy) (bool, string) {
				if len(policy.Description) < 10 || len(policy.Description) > 1000 {
					return false, "Policy description must be between 10 and 1000 characters"
				}
				return true, ""
			},
		},
		{
			Name:        "policy-rules-count",
			Description: "Policy must have at least one rule",
			Validate: func(policy *SchedulingPolicy) (bool, string) {
				if len(policy.Rules) == 0 {
					return false, "Policy must have at least one rule"
				}
				return true, ""
			},
		},
	}
}

// ParseCustomPolicy parses a policy definition written in the custom policy language
func (p *CustomPolicyLanguageParser) ParseCustomPolicy(policyDef string, author string) (*SchedulingPolicy, error) {
	// First, parse the policy using the base parser
	policy, err := p.PolicyParser.ParsePolicy(policyDef, author)
	if err != nil {
		return nil, err
	}
	
	// Validate the policy
	validationErrors := p.ValidatePolicy(policy)
	if len(validationErrors) > 0 {
		return nil, fmt.Errorf("policy validation failed: %s", strings.Join(validationErrors, "; "))
	}
	
	return policy, nil
}

// ValidatePolicy validates a policy against all validation rules
func (p *CustomPolicyLanguageParser) ValidatePolicy(policy *SchedulingPolicy) []string {
	errors := make([]string, 0)
	
	for _, rule := range p.ValidationRules {
		valid, message := rule.Validate(policy)
		if !valid {
			errors = append(errors, message)
		}
	}
	
	return errors
}

// FormatPolicy formats a policy as a string in the custom policy language
func (p *CustomPolicyLanguageParser) FormatPolicy(policy *SchedulingPolicy) string {
	var sb strings.Builder
	
	// Policy header
	sb.WriteString(fmt.Sprintf("policy \"%s\" {\n", policy.Name))
	sb.WriteString(fmt.Sprintf("    id = \"%s\"\n", policy.ID))
	sb.WriteString(fmt.Sprintf("    type = \"%s\"\n", policy.Type))
	sb.WriteString(fmt.Sprintf("    description = \"%s\"\n", policy.Description))
	sb.WriteString("\n")
	
	// Metadata
	if len(policy.Metadata) > 0 {
		sb.WriteString("    metadata {\n")
		for key, value := range policy.Metadata {
			sb.WriteString(fmt.Sprintf("        %s = \"%v\"\n", key, value))
		}
		sb.WriteString("    }\n\n")
	}
	
	// Target selector
	if policy.TargetSelector != nil {
		sb.WriteString("    target {\n")
		sb.WriteString(fmt.Sprintf("        selector = %s\n", policy.TargetSelector.String()))
		sb.WriteString("    }\n\n")
	}
	
	// Parameters
	if len(policy.Parameters) > 0 {
		sb.WriteString("    parameters {\n")
		for name, param := range policy.Parameters {
			sb.WriteString(fmt.Sprintf("        parameter \"%s\" {\n", name))
			sb.WriteString(fmt.Sprintf("            type = \"%s\"\n", param.Type))
			sb.WriteString(fmt.Sprintf("            description = \"%s\"\n", param.Description))
			if param.DefaultValue != nil {
				sb.WriteString(fmt.Sprintf("            default = %v\n", param.DefaultValue))
			}
			if param.MinValue != nil {
				sb.WriteString(fmt.Sprintf("            min = %v\n", param.MinValue))
			}
			if param.MaxValue != nil {
				sb.WriteString(fmt.Sprintf("            max = %v\n", param.MaxValue))
			}
			if len(param.AllowedValues) > 0 {
				sb.WriteString("            allowed_values = [")
				for i, value := range param.AllowedValues {
					if i > 0 {
						sb.WriteString(", ")
					}
					sb.WriteString(fmt.Sprintf("\"%v\"", value))
				}
				sb.WriteString("]\n")
			}
			sb.WriteString("        }\n")
		}
		sb.WriteString("    }\n\n")
	}
	
	// Rules
	sb.WriteString("    rules {\n")
	for _, rule := range policy.Rules {
		sb.WriteString(fmt.Sprintf("        rule \"%s\" {\n", rule.Name))
		sb.WriteString(fmt.Sprintf("            id = \"%s\"\n", rule.ID))
		sb.WriteString(fmt.Sprintf("            description = \"%s\"\n", rule.Description))
		sb.WriteString(fmt.Sprintf("            hard_constraint = %t\n", rule.IsHardConstraint))
		
		// Condition
		if rule.Condition != nil {
			sb.WriteString("            when {\n")
			sb.WriteString(fmt.Sprintf("                %s\n", rule.Condition.String()))
			sb.WriteString("            }\n")
		}
		
		// Actions
		sb.WriteString("            then {\n")
		for _, action := range rule.Actions {
			switch act := action.(type) {
			case *ScoreAction:
				sb.WriteString(fmt.Sprintf("                score %s \"%s\"\n",
					act.ScoreExpression.String(), act.Reason))
			case *FilterAction:
				sb.WriteString(fmt.Sprintf("                filter \"%s\"\n", act.Reason))
			case *LogAction:
				sb.WriteString(fmt.Sprintf("                log \"%s\" level=\"%s\"\n",
					act.Message, act.Level))
			}
		}
		sb.WriteString("            }\n")
		
		sb.WriteString("        }\n")
	}
	sb.WriteString("    }\n")
	
	sb.WriteString("}\n")
	
	return sb.String()
}

// ParseRuleFromString parses a rule definition from a string
func (p *CustomPolicyLanguageParser) ParseRuleFromString(ruleDef string) (*PolicyRule, error) {
	// Extract rule header
	headerMatch := regexp.MustCompile(`rule\s+"([^"]+)"\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(ruleDef)
	if len(headerMatch) < 3 {
		return nil, fmt.Errorf("invalid rule format: missing rule header")
	}
	
	ruleName := headerMatch[1]
	ruleBody := headerMatch[2]
	
	// Extract rule ID
	idMatch := regexp.MustCompile(`id\s*=\s*"([^"]+)"`).FindStringSubmatch(ruleBody)
	ruleID := ""
	if len(idMatch) > 1 {
		ruleID = idMatch[1]
	} else {
		// Generate ID from name
		ruleID = strings.ToLower(strings.ReplaceAll(ruleName, " ", "-"))
	}
	
	// Extract rule description
	descMatch := regexp.MustCompile(`description\s*=\s*"([^"]+)"`).FindStringSubmatch(ruleBody)
	ruleDescription := ""
	if len(descMatch) > 1 {
		ruleDescription = descMatch[1]
	}
	
	// Extract hard constraint flag
	hardConstraintMatch := regexp.MustCompile(`hard_constraint\s*=\s*(true|false)`).FindStringSubmatch(ruleBody)
	isHardConstraint := false
	if len(hardConstraintMatch) > 1 {
		isHardConstraint = hardConstraintMatch[1] == "true"
	}
	
	// Extract condition
	conditionMatch := regexp.MustCompile(`when\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(ruleBody)
	var condition Expression
	if len(conditionMatch) > 1 {
		conditionExpr := strings.TrimSpace(conditionMatch[1])
		var err error
		condition, err = p.ExpressionCompiler.CompileExpression(conditionExpr)
		if err != nil {
			return nil, fmt.Errorf("error compiling condition: %v", err)
		}
	}
	
	// Extract actions
	actionsMatch := regexp.MustCompile(`then\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(ruleBody)
	actions := make([]Action, 0)
	if len(actionsMatch) > 1 {
		actionsBody := actionsMatch[1]
		
		// Extract score actions
		scoreMatches := regexp.MustCompile(`score\s+(.*?)\s+"([^"]+)"`).FindAllStringSubmatch(actionsBody, -1)
		for _, match := range scoreMatches {
			if len(match) > 2 {
				scoreExpr, err := p.ExpressionCompiler.CompileExpression(match[1])
				if err != nil {
					return nil, fmt.Errorf("error compiling score expression: %v", err)
				}
				
				actions = append(actions, &ScoreAction{
					ScoreExpression: scoreExpr,
					Reason:          match[2],
				})
			}
		}
		
		// Extract filter actions
		filterMatches := regexp.MustCompile(`filter\s+"([^"]+)"`).FindAllStringSubmatch(actionsBody, -1)
		for _, match := range filterMatches {
			if len(match) > 1 {
				actions = append(actions, &FilterAction{
					Reason: match[1],
				})
			}
		}
		
		// Extract log actions
		logMatches := regexp.MustCompile(`log\s+"([^"]+)"\s+level="([^"]+)"`).FindAllStringSubmatch(actionsBody, -1)
		for _, match := range logMatches {
			if len(match) > 2 {
				actions = append(actions, &LogAction{
					Message: match[1],
					Level:   match[2],
				})
			}
		}
	}
	
	// Create the rule
	rule := &PolicyRule{
		ID:              ruleID,
		Name:            ruleName,
		Description:     ruleDescription,
		IsHardConstraint: isHardConstraint,
		Condition:       condition,
		Actions:         actions,
	}
	
	return rule, nil
}

// AddValidationRule adds a validation rule
func (p *CustomPolicyLanguageParser) AddValidationRule(rule *ValidationRule) {
	p.ValidationRules = append(p.ValidationRules, rule)
}

// GeneratePolicyTemplate generates a template for a new policy
func (p *CustomPolicyLanguageParser) GeneratePolicyTemplate(policyType PolicyType, name string) string {
	var sb strings.Builder
	
	// Generate ID from name
	id := strings.ToLower(strings.ReplaceAll(name, " ", "-"))
	
	// Policy header
	sb.WriteString(fmt.Sprintf("policy \"%s\" {\n", name))
	sb.WriteString(fmt.Sprintf("    id = \"%s\"\n", id))
	sb.WriteString(fmt.Sprintf("    type = \"%s\"\n", policyType))
	sb.WriteString("    description = \"Description of the policy\"\n\n")
	
	// Metadata
	sb.WriteString("    metadata {\n")
	sb.WriteString("        created_by = \"template-generator\"\n")
	sb.WriteString(fmt.Sprintf("        created_at = \"%s\"\n", time.Now().Format(time.RFC3339)))
	sb.WriteString("    }\n\n")
	
	// Parameters
	sb.WriteString("    parameters {\n")
	sb.WriteString("        parameter \"weight\" {\n")
	sb.WriteString("            type = \"float\"\n")
	sb.WriteString("            description = \"Weight for scoring\"\n")
	sb.WriteString("            default = 1.0\n")
	sb.WriteString("            min = 0.0\n")
	sb.WriteString("            max = 10.0\n")
	sb.WriteString("        }\n")
	sb.WriteString("    }\n\n")
	
	// Rules
	sb.WriteString("    rules {\n")
	sb.WriteString("        rule \"Example Rule\" {\n")
	sb.WriteString("            id = \"example-rule\"\n")
	sb.WriteString("            description = \"An example rule\"\n")
	sb.WriteString("            hard_constraint = false\n")
	sb.WriteString("            when {\n")
	sb.WriteString("                vm.labels[\"example\"] == \"true\"\n")
	sb.WriteString("            }\n")
	sb.WriteString("            then {\n")
	sb.WriteString("                score param.weight * 10 \"Example score\"\n")
	sb.WriteString("                log \"Applied example rule\" level=\"info\"\n")
	sb.WriteString("            }\n")
	sb.WriteString("        }\n")
	sb.WriteString("    }\n")
	
	sb.WriteString("}\n")
	
	return sb.String()
}

// GenerateRuleTemplate generates a template for a new rule
func (p *CustomPolicyLanguageParser) GenerateRuleTemplate(name string, isHardConstraint bool) string {
	var sb strings.Builder
	
	// Generate ID from name
	id := strings.ToLower(strings.ReplaceAll(name, " ", "-"))
	
	// Rule header
	sb.WriteString(fmt.Sprintf("rule \"%s\" {\n", name))
	sb.WriteString(fmt.Sprintf("    id = \"%s\"\n", id))
	sb.WriteString("    description = \"Description of the rule\"\n")
	sb.WriteString(fmt.Sprintf("    hard_constraint = %t\n", isHardConstraint))
	
	// Condition
	sb.WriteString("    when {\n")
	sb.WriteString("        // Define condition here\n")
	sb.WriteString("        true\n")
	sb.WriteString("    }\n")
	
	// Actions
	sb.WriteString("    then {\n")
	if isHardConstraint {
		sb.WriteString("        filter \"Reason for filtering\"\n")
	} else {
		sb.WriteString("        score 10.0 \"Reason for score\"\n")
	}
	sb.WriteString("        log \"Rule applied\" level=\"info\"\n")
	sb.WriteString("    }\n")
	
	sb.WriteString("}\n")
	
	return sb.String()
}
