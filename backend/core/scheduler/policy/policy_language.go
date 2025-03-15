package policy

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// PolicyParser parses policy definitions written in the policy DSL
type PolicyParser struct {
	// ExpressionCompiler is used to compile expressions in the policy
	ExpressionCompiler ExpressionCompiler
}

// NewPolicyParser creates a new policy parser
func NewPolicyParser() *PolicyParser {
	return &PolicyParser{
		ExpressionCompiler: &DefaultExpressionCompiler{},
	}
}

// ParsePolicy parses a policy definition string into a SchedulingPolicy
func (p *PolicyParser) ParsePolicy(policyDef string, author string) (*SchedulingPolicy, error) {
	// Extract policy header section
	headerMatch := regexp.MustCompile(`policy\s+"([^"]+)"\s*\{([\s\S]*?)\n\s*rules\s*\{`).FindStringSubmatch(policyDef)
	if len(headerMatch) < 3 {
		return nil, fmt.Errorf("invalid policy format: missing policy header")
	}

	policyName := headerMatch[1]
	headerSection := headerMatch[2]

	// Parse header attributes
	policyID := ""
	policyDescription := ""
	policyType := PlacementPolicy // Default
	metadataMap := make(map[string]interface{})

	// Look for description
	descMatch := regexp.MustCompile(`description\s*=\s*"([^"]+)"`).FindStringSubmatch(headerSection)
	if len(descMatch) > 1 {
		policyDescription = descMatch[1]
	}

	// Look for ID
	idMatch := regexp.MustCompile(`id\s*=\s*"([^"]+)"`).FindStringSubmatch(headerSection)
	if len(idMatch) > 1 {
		policyID = idMatch[1]
	} else {
		// Generate ID from name if not specified
		policyID = generatePolicyID(policyName)
	}

	// Look for type
	typeMatch := regexp.MustCompile(`type\s*=\s*"([^"]+)"`).FindStringSubmatch(headerSection)
	if len(typeMatch) > 1 {
		policyTypeStr := typeMatch[1]
		switch policyTypeStr {
		case "placement":
			policyType = PlacementPolicy
		case "migration":
			policyType = MigrationPolicy
		case "resource_allocation":
			policyType = ResourceAllocationPolicy
		case "maintenance":
			policyType = MaintenancePolicy
		case "composite":
			policyType = CompositePolicy
		default:
			return nil, fmt.Errorf("unknown policy type: %s", policyTypeStr)
		}
	}

	// Look for target selector
	var targetSelectorExpr Expression
	targetSelectorExpr = &LiteralExpression{Value: true} // Default matches everything
	targetMatch := regexp.MustCompile(`target\s*=\s*(.+)$`).FindStringSubmatch(headerSection)
	if len(targetMatch) > 1 {
		targetExprStr := targetMatch[1]
		var err error
		targetSelectorExpr, err = p.ExpressionCompiler.Compile(targetExprStr)
		if err != nil {
			return nil, fmt.Errorf("invalid target selector expression: %v", err)
		}
	}

	// Look for metadata
	metadataMatches := regexp.MustCompile(`metadata\s*\{\s*([\s\S]*?)\s*\}`).FindStringSubmatch(headerSection)
	if len(metadataMatches) > 1 {
		metadataSection := metadataMatches[1]
		// Extract key-value pairs
		metadataRegex := regexp.MustCompile(`\s*(\w+)\s*=\s*"([^"]+)"\s*`)
		metadataEntries := metadataRegex.FindAllStringSubmatch(metadataSection, -1)
		for _, entry := range metadataEntries {
			if len(entry) > 2 {
				metadataMap[entry[1]] = entry[2]
			}
		}
	}

	// Extract rules section
	rulesMatch := regexp.MustCompile(`rules\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(policyDef)
	if len(rulesMatch) < 2 {
		return nil, fmt.Errorf("invalid policy format: missing rules section")
	}
	rulesSection := rulesMatch[1]

	// Parse individual rules
	rules, err := p.parseRules(rulesSection)
	if err != nil {
		return nil, fmt.Errorf("error parsing rules: %v", err)
	}

	// Extract parameters section
	parameters := make(map[string]*PolicyParameter)
	paramsMatch := regexp.MustCompile(`parameters\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(policyDef)
	if len(paramsMatch) > 1 {
		paramsSection := paramsMatch[1]
		// Parse individual parameters
		params, err := p.parseParameters(paramsSection)
		if err != nil {
			return nil, fmt.Errorf("error parsing parameters: %v", err)
		}
		parameters = params
	}

	// Create the policy
	policy := &SchedulingPolicy{
		ID:             policyID,
		Name:           policyName,
		Description:    policyDescription,
		Type:           policyType,
		Status:         PolicyStatusDraft,
		Rules:          rules,
		Parameters:     parameters,
		TargetSelector: targetSelectorExpr,
		Metadata:       metadataMap,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
		CreatedBy:      author,
		UpdatedBy:      author,
	}

	return policy, nil
}

// parseRules parses the rules section of a policy
func (p *PolicyParser) parseRules(rulesSection string) ([]*PolicyRule, error) {
	rules := make([]*PolicyRule, 0)

	// Extract individual rule blocks
	ruleBlockRegex := regexp.MustCompile(`rule\s+"([^"]+)"\s*\{([\s\S]*?)\n\s*\}`)
	ruleBlocks := ruleBlockRegex.FindAllStringSubmatch(rulesSection, -1)

	for i, ruleBlock := range ruleBlocks {
		if len(ruleBlock) < 3 {
			continue
		}

		ruleName := ruleBlock[1]
		ruleContent := ruleBlock[2]

		// Parse rule attributes
		ruleID := fmt.Sprintf("rule-%d", i+1)
		ruleDescription := ""
		rulePriority := 100 - i // Higher index = lower priority
		ruleWeight := 1
		isHardConstraint := false

		// Look for ID
		idMatch := regexp.MustCompile(`id\s*=\s*"([^"]+)"`).FindStringSubmatch(ruleContent)
		if len(idMatch) > 1 {
			ruleID = idMatch[1]
		}

		// Look for description
		descMatch := regexp.MustCompile(`description\s*=\s*"([^"]+)"`).FindStringSubmatch(ruleContent)
		if len(descMatch) > 1 {
			ruleDescription = descMatch[1]
		}

		// Look for priority
		priorityMatch := regexp.MustCompile(`priority\s*=\s*(\d+)`).FindStringSubmatch(ruleContent)
		if len(priorityMatch) > 1 {
			priority, err := strconv.Atoi(priorityMatch[1])
			if err == nil {
				rulePriority = priority
			}
		}

		// Look for weight
		weightMatch := regexp.MustCompile(`weight\s*=\s*(\d+)`).FindStringSubmatch(ruleContent)
		if len(weightMatch) > 1 {
			weight, err := strconv.Atoi(weightMatch[1])
			if err == nil {
				ruleWeight = weight
			}
		}

		// Look for hard constraint flag
		hardConstraintMatch := regexp.MustCompile(`hard_constraint\s*=\s*(true|false)`).FindStringSubmatch(ruleContent)
		if len(hardConstraintMatch) > 1 {
			isHardConstraint = hardConstraintMatch[1] == "true"
		}

		// Extract condition
		conditionMatch := regexp.MustCompile(`when\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(ruleContent)
		if len(conditionMatch) < 2 {
			return nil, fmt.Errorf("rule %s is missing condition", ruleName)
		}
		conditionExprStr := strings.TrimSpace(conditionMatch[1])

		conditionExpr, err := p.ExpressionCompiler.Compile(conditionExprStr)
		if err != nil {
			return nil, fmt.Errorf("invalid condition in rule %s: %v", ruleName, err)
		}

		// Extract actions
		actionsMatch := regexp.MustCompile(`then\s*\{([\s\S]*?)\n\s*\}`).FindStringSubmatch(ruleContent)
		if len(actionsMatch) < 2 {
			return nil, fmt.Errorf("rule %s is missing actions", ruleName)
		}
		actionsStr := actionsMatch[1]

		actions, err := p.parseActions(actionsStr, ruleName)
		if err != nil {
			return nil, fmt.Errorf("error parsing actions in rule %s: %v", ruleName, err)
		}

		// Create the rule
		rule := &PolicyRule{
			ID:               ruleID,
			Name:             ruleName,
			Description:      ruleDescription,
			Condition:        conditionExpr,
			Actions:          actions,
			Priority:         rulePriority,
			Weight:           ruleWeight,
			IsHardConstraint: isHardConstraint,
		}

		rules = append(rules, rule)
	}

	return rules, nil
}

// parseActions parses the actions section of a rule
func (p *PolicyParser) parseActions(actionsStr string, ruleName string) ([]PolicyAction, error) {
	actions := make([]PolicyAction, 0)

	// Extract score actions
	scoreRegex := regexp.MustCompile(`score\s+(.+?)\s+"([^"]+)"`)
	scoreMatches := scoreRegex.FindAllStringSubmatch(actionsStr, -1)

	for _, scoreMatch := range scoreMatches {
		if len(scoreMatch) < 3 {
			continue
		}

		scoreExprStr := scoreMatch[1]
		scoreReason := scoreMatch[2]

		scoreExpr, err := p.ExpressionCompiler.Compile(scoreExprStr)
		if err != nil {
			return nil, fmt.Errorf("invalid score expression: %v", err)
		}

		scoreAction := &ScoreAction{
			ScoreExpression: scoreExpr,
			Reason:          scoreReason,
		}

		actions = append(actions, scoreAction)
	}

	// Extract filter actions
	filterRegex := regexp.MustCompile(`filter\s+"([^"]+)"`)
	filterMatches := filterRegex.FindAllStringSubmatch(actionsStr, -1)

	for _, filterMatch := range filterMatches {
		if len(filterMatch) < 2 {
			continue
		}

		filterReason := filterMatch[1]

		filterAction := &FilterAction{
			Reason: filterReason,
		}

		actions = append(actions, filterAction)
	}

	// Extract log actions
	logRegex := regexp.MustCompile(`log\s+"([^"]+)"\s+level\s*=\s*"([^"]+)"`)
	logMatches := logRegex.FindAllStringSubmatch(actionsStr, -1)

	for _, logMatch := range logMatches {
		if len(logMatch) < 3 {
			continue
		}

		logMessage := logMatch[1]
		logLevel := logMatch[2]

		logAction := &LogAction{
			Message: logMessage,
			Level:   logLevel,
			AdditionalData: map[string]interface{}{
				"rule": ruleName,
			},
		}

		actions = append(actions, logAction)
	}

	return actions, nil
}

// parseParameters parses the parameters section of a policy
func (p *PolicyParser) parseParameters(paramsSection string) (map[string]*PolicyParameter, error) {
	parameters := make(map[string]*PolicyParameter)

	// Extract individual parameter blocks
	paramBlockRegex := regexp.MustCompile(`param\s+"([^"]+)"\s*\{([\s\S]*?)\n\s*\}`)
	paramBlocks := paramBlockRegex.FindAllStringSubmatch(paramsSection, -1)

	for _, paramBlock := range paramBlocks {
		if len(paramBlock) < 3 {
			continue
		}

		paramName := paramBlock[1]
		paramContent := paramBlock[2]

		// Parse parameter attributes
		paramType := "string" // Default type
		var paramDefaultValue interface{} = ""
		paramDescription := ""
		paramConstraints := make(map[string]interface{})

		// Look for type
		typeMatch := regexp.MustCompile(`type\s*=\s*"([^"]+)"`).FindStringSubmatch(paramContent)
		if len(typeMatch) > 1 {
			paramType = typeMatch[1]
		}

		// Look for default value
		defaultMatch := regexp.MustCompile(`default\s*=\s*([^,\s\}]+)`).FindStringSubmatch(paramContent)
		if len(defaultMatch) > 1 {
			defaultStr := defaultMatch[1]
			defaultValue, err := parseDefaultValue(defaultStr, paramType)
			if err != nil {
				return nil, fmt.Errorf("invalid default value for parameter %s: %v", paramName, err)
			}
			paramDefaultValue = defaultValue
		}

		// Look for description
		descMatch := regexp.MustCompile(`description\s*=\s*"([^"]+)"`).FindStringSubmatch(paramContent)
		if len(descMatch) > 1 {
			paramDescription = descMatch[1]
		}

		// Look for min value constraint for numeric types
		minMatch := regexp.MustCompile(`min\s*=\s*([^,\s\}]+)`).FindStringSubmatch(paramContent)
		if len(minMatch) > 1 && (paramType == "float" || paramType == "int") {
			minStr := minMatch[1]
			if paramType == "float" {
				minVal, err := strconv.ParseFloat(minStr, 64)
				if err == nil {
					paramConstraints["min"] = minVal
				}
			} else {
				minVal, err := strconv.Atoi(minStr)
				if err == nil {
					paramConstraints["min"] = minVal
				}
			}
		}

		// Look for max value constraint for numeric types
		maxMatch := regexp.MustCompile(`max\s*=\s*([^,\s\}]+)`).FindStringSubmatch(paramContent)
		if len(maxMatch) > 1 && (paramType == "float" || paramType == "int") {
			maxStr := maxMatch[1]
			if paramType == "float" {
				maxVal, err := strconv.ParseFloat(maxStr, 64)
				if err == nil {
					paramConstraints["max"] = maxVal
				}
			} else {
				maxVal, err := strconv.Atoi(maxStr)
				if err == nil {
					paramConstraints["max"] = maxVal
				}
			}
		}

		// Look for allowed values constraint
		allowedMatch := regexp.MustCompile(`allowed\s*=\s*\[\s*(.*?)\s*\]`).FindStringSubmatch(paramContent)
		if len(allowedMatch) > 1 {
			allowedStr := allowedMatch[1]
			allowedValues := parseAllowedValues(allowedStr, paramType)
			paramConstraints["allowed"] = allowedValues
		}

		// Create the parameter
		parameter := &PolicyParameter{
			Name:         paramName,
			Description:  paramDescription,
			Type:         paramType,
			DefaultValue: paramDefaultValue,
			CurrentValue: paramDefaultValue, // Start with default value
			Constraints:  paramConstraints,
		}

		parameters[paramName] = parameter
	}

	return parameters, nil
}

// parseDefaultValue parses a default value string based on the parameter type
func parseDefaultValue(valueStr string, paramType string) (interface{}, error) {
	switch paramType {
	case "string":
		// Strip quotes if present
		if strings.HasPrefix(valueStr, "\"") && strings.HasSuffix(valueStr, "\"") {
			return valueStr[1 : len(valueStr)-1], nil
		}
		return valueStr, nil
	case "int":
		return strconv.Atoi(valueStr)
	case "float":
		return strconv.ParseFloat(valueStr, 64)
	case "bool":
		return strconv.ParseBool(valueStr)
	default:
		return nil, fmt.Errorf("unsupported parameter type: %s", paramType)
	}
}

// parseAllowedValues parses a list of allowed values based on the parameter type
func parseAllowedValues(valuesStr string, paramType string) []interface{} {
	// Split by commas, handle quoted strings
	parts := strings.Split(valuesStr, ",")
	results := make([]interface{}, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		switch paramType {
		case "string":
			// Strip quotes if present
			if strings.HasPrefix(part, "\"") && strings.HasSuffix(part, "\"") {
				part = part[1 : len(part)-1]
			}
			results = append(results, part)
		case "int":
			val, err := strconv.Atoi(part)
			if err == nil {
				results = append(results, val)
			}
		case "float":
			val, err := strconv.ParseFloat(part, 64)
			if err == nil {
				results = append(results, val)
			}
		case "bool":
			val, err := strconv.ParseBool(part)
			if err == nil {
				results = append(results, val)
			}
		}
	}

	return results
}

// Helper function to generate a policy ID from its name
func generatePolicyID(name string) string {
	// Convert to lowercase and replace spaces with dashes
	id := strings.ToLower(name)
	id = strings.ReplaceAll(id, " ", "-")
	// Remove any remaining non-alphanumeric characters
	idRegex := regexp.MustCompile(`[^a-z0-9\-]`)
	id = idRegex.ReplaceAllString(id, "")
	return id
}

// FormatPolicy formats a SchedulingPolicy back into the policy DSL format
func FormatPolicy(policy *SchedulingPolicy) string {
	var sb strings.Builder

	// Header
	sb.WriteString(fmt.Sprintf("policy \"%s\" {\n", policy.Name))

	// ID
	sb.WriteString(fmt.Sprintf("    id = \"%s\"\n", policy.ID))

	// Type
	sb.WriteString(fmt.Sprintf("    type = \"%s\"\n", string(policy.Type)))

	// Description
	if policy.Description != "" {
		sb.WriteString(fmt.Sprintf("    description = \"%s\"\n", policy.Description))
	}

	// Target selector
	if policy.TargetSelector != nil {
		sb.WriteString(fmt.Sprintf("    target = %s\n", policy.TargetSelector.String()))
	}

	// Metadata
	if len(policy.Metadata) > 0 {
		sb.WriteString("    metadata {\n")
		for key, value := range policy.Metadata {
			sb.WriteString(fmt.Sprintf("        %s = \"%v\"\n", key, value))
		}
		sb.WriteString("    }\n")
	}

	// Rules
	sb.WriteString("\n    rules {\n")
	for _, rule := range policy.Rules {
		sb.WriteString(fmt.Sprintf("        rule \"%s\" {\n", rule.Name))

		// Rule properties
		sb.WriteString(fmt.Sprintf("            id = \"%s\"\n", rule.ID))
		if rule.Description != "" {
			sb.WriteString(fmt.Sprintf("            description = \"%s\"\n", rule.Description))
		}
		sb.WriteString(fmt.Sprintf("            priority = %d\n", rule.Priority))
		sb.WriteString(fmt.Sprintf("            weight = %d\n", rule.Weight))
		sb.WriteString(fmt.Sprintf("            hard_constraint = %t\n", rule.IsHardConstraint))

		// Condition
		sb.WriteString("            when {\n")
		sb.WriteString(fmt.Sprintf("                %s\n", rule.Condition.String()))
		sb.WriteString("            }\n")

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

	// Parameters
	if len(policy.Parameters) > 0 {
		sb.WriteString("\n    parameters {\n")
		for name, param := range policy.Parameters {
			sb.WriteString(fmt.Sprintf("        param \"%s\" {\n", name))
			sb.WriteString(fmt.Sprintf("            type = \"%s\"\n", param.Type))
			sb.WriteString(fmt.Sprintf("            default = %v\n", formatDefaultValue(param.DefaultValue, param.Type)))
			if param.Description != "" {
				sb.WriteString(fmt.Sprintf("            description = \"%s\"\n", param.Description))
			}

			// Constraints
			if minVal, ok := param.Constraints["min"]; ok {
				sb.WriteString(fmt.Sprintf("            min = %v\n", minVal))
			}
			if maxVal, ok := param.Constraints["max"]; ok {
				sb.WriteString(fmt.Sprintf("            max = %v\n", maxVal))
			}
			if allowedVals, ok := param.Constraints["allowed"].([]interface{}); ok {
				sb.WriteString("            allowed = [")
				for i, val := range allowedVals {
					if i > 0 {
						sb.WriteString(", ")
					}
					sb.WriteString(fmt.Sprintf("%v", formatDefaultValue(val, param.Type)))
				}
				sb.WriteString("]\n")
			}

			sb.WriteString("        }\n")
		}
		sb.WriteString("    }\n")
	}

	// Close policy
	sb.WriteString("}\n")

	return sb.String()
}

// formatDefaultValue formats a default value for the policy DSL
func formatDefaultValue(value interface{}, paramType string) string {
	switch paramType {
	case "string":
		return fmt.Sprintf("\"%v\"", value)
	default:
		return fmt.Sprintf("%v", value)
	}
}
