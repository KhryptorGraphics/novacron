package security

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/khryptorgraphics/novacron/backend/pkg/errors"
)

// ValidationRule represents a validation rule
type ValidationRule struct {
	Name    string
	Message string
	Rule    func(value string) bool
}

// InputValidator provides input validation functionality
type InputValidator struct {
	rules map[string][]ValidationRule
}

// NewInputValidator creates a new input validator
func NewInputValidator() *InputValidator {
	return &InputValidator{
		rules: make(map[string][]ValidationRule),
	}
}

// AddRule adds a validation rule for a field
func (v *InputValidator) AddRule(field string, rule ValidationRule) {
	v.rules[field] = append(v.rules[field], rule)
}

// Validate validates input against registered rules
func (v *InputValidator) Validate(field, value string) error {
	rules, exists := v.rules[field]
	if !exists {
		return nil // No rules defined for this field
	}
	
	for _, rule := range rules {
		if !rule.Rule(value) {
			return errors.NewValidationError(field, rule.Message)
		}
	}
	
	return nil
}

// ValidateAll validates multiple fields at once
func (v *InputValidator) ValidateAll(inputs map[string]string) error {
	var validationErrors []string
	
	for field, value := range inputs {
		if err := v.Validate(field, value); err != nil {
			validationErrors = append(validationErrors, err.Error())
		}
	}
	
	if len(validationErrors) > 0 {
		return errors.NewAppErrorWithDetails(
			errors.ErrInvalidInput,
			"Validation failed",
			strings.Join(validationErrors, "; "),
		)
	}
	
	return nil
}

// Common validation rules

// RequiredRule validates that a field is not empty
func RequiredRule() ValidationRule {
	return ValidationRule{
		Name:    "required",
		Message: "field is required",
		Rule: func(value string) bool {
			return strings.TrimSpace(value) != ""
		},
	}
}

// MinLengthRule validates minimum length
func MinLengthRule(length int) ValidationRule {
	return ValidationRule{
		Name:    fmt.Sprintf("min_length_%d", length),
		Message: fmt.Sprintf("must be at least %d characters long", length),
		Rule: func(value string) bool {
			return len(value) >= length
		},
	}
}

// MaxLengthRule validates maximum length
func MaxLengthRule(length int) ValidationRule {
	return ValidationRule{
		Name:    fmt.Sprintf("max_length_%d", length),
		Message: fmt.Sprintf("must not exceed %d characters", length),
		Rule: func(value string) bool {
			return len(value) <= length
		},
	}
}

// EmailRule validates email format
func EmailRule() ValidationRule {
	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	return ValidationRule{
		Name:    "email",
		Message: "must be a valid email address",
		Rule: func(value string) bool {
			return emailRegex.MatchString(value)
		},
	}
}

// AlphanumericRule validates alphanumeric characters only
func AlphanumericRule() ValidationRule {
	return ValidationRule{
		Name:    "alphanumeric",
		Message: "must contain only letters and numbers",
		Rule: func(value string) bool {
			for _, r := range value {
				if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
					return false
				}
			}
			return true
		},
	}
}

// NoSQLInjectionRule validates against common SQL injection patterns
func NoSQLInjectionRule() ValidationRule {
	sqlPatterns := []string{
		"'", "\"", ";", "--", "/*", "*/", "xp_", "sp_", 
		"union", "select", "insert", "delete", "update", "drop", "create", "alter",
		"exec", "execute", "script", "<script", "javascript:",
	}
	
	return ValidationRule{
		Name:    "no_sql_injection",
		Message: "contains potentially dangerous characters or keywords",
		Rule: func(value string) bool {
			lowerValue := strings.ToLower(value)
			for _, pattern := range sqlPatterns {
				if strings.Contains(lowerValue, pattern) {
					return false
				}
			}
			return true
		},
	}
}

// NoXSSRule validates against XSS patterns
func NoXSSRule() ValidationRule {
	xssPatterns := []string{
		"<script", "</script", "<iframe", "</iframe", "<object", "</object",
		"javascript:", "vbscript:", "onload=", "onerror=", "onclick=",
		"onmouseover=", "onfocus=", "onblur=", "onchange=", "onsubmit=",
	}
	
	return ValidationRule{
		Name:    "no_xss",
		Message: "contains potentially dangerous script content",
		Rule: func(value string) bool {
			lowerValue := strings.ToLower(value)
			for _, pattern := range xssPatterns {
				if strings.Contains(lowerValue, pattern) {
					return false
				}
			}
			return true
		},
	}
}

// VMNameRule validates VM names
func VMNameRule() ValidationRule {
	vmNameRegex := regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9-_]{2,63}$`)
	return ValidationRule{
		Name:    "vm_name",
		Message: "must start with a letter, contain only letters, numbers, hyphens, and underscores, and be 3-64 characters long",
		Rule: func(value string) bool {
			return vmNameRegex.MatchString(value)
		},
	}
}

// IPv4Rule validates IPv4 addresses
func IPv4Rule() ValidationRule {
	ipv4Regex := regexp.MustCompile(`^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$`)
	return ValidationRule{
		Name:    "ipv4",
		Message: "must be a valid IPv4 address",
		Rule: func(value string) bool {
			return ipv4Regex.MatchString(value)
		},
	}
}

// PortRule validates port numbers
func PortRule() ValidationRule {
	return ValidationRule{
		Name:    "port",
		Message: "must be a valid port number (1-65535)",
		Rule: func(value string) bool {
			portRegex := regexp.MustCompile(`^([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$`)
			return portRegex.MatchString(value)
		},
	}
}

// PasswordStrengthRule validates password strength
func PasswordStrengthRule(minLength int, requireMixed, requireNumbers, requireSymbols bool) ValidationRule {
	return ValidationRule{
		Name:    "password_strength",
		Message: getPasswordRequirements(minLength, requireMixed, requireNumbers, requireSymbols),
		Rule: func(value string) bool {
			if len(value) < minLength {
				return false
			}
			
			if requireMixed {
				hasUpper, hasLower := false, false
				for _, r := range value {
					if unicode.IsUpper(r) {
						hasUpper = true
					}
					if unicode.IsLower(r) {
						hasLower = true
					}
				}
				if !hasUpper || !hasLower {
					return false
				}
			}
			
			if requireNumbers {
				hasNumber := false
				for _, r := range value {
					if unicode.IsDigit(r) {
						hasNumber = true
						break
					}
				}
				if !hasNumber {
					return false
				}
			}
			
			if requireSymbols {
				hasSymbol := false
				symbols := "!@#$%^&*()_+-=[]{}|;':\",./<>?"
				for _, r := range value {
					for _, s := range symbols {
						if r == s {
							hasSymbol = true
							break
						}
					}
					if hasSymbol {
						break
					}
				}
				if !hasSymbol {
					return false
				}
			}
			
			return true
		},
	}
}

// FilePathRule validates file paths for security
func FilePathRule() ValidationRule {
	return ValidationRule{
		Name:    "file_path",
		Message: "must be a valid file path without directory traversal attempts",
		Rule: func(value string) bool {
			// Check for directory traversal attempts
			dangerousPatterns := []string{
				"..", "~", "//", "\\\\", "/etc/", "/proc/", "/sys/",
				"C:\\", "D:\\", "%", "$", "`",
			}
			
			for _, pattern := range dangerousPatterns {
				if strings.Contains(value, pattern) {
					return false
				}
			}
			
			// Must be relative path or absolute within safe directories
			if strings.HasPrefix(value, "/") {
				safeRoots := []string{
					"/var/lib/novacron/",
					"/tmp/novacron/",
					"/opt/novacron/",
				}
				
				allowed := false
				for _, root := range safeRoots {
					if strings.HasPrefix(value, root) {
						allowed = true
						break
					}
				}
				return allowed
			}
			
			return true
		},
	}
}

// Helper function to generate password requirements message
func getPasswordRequirements(minLength int, requireMixed, requireNumbers, requireSymbols bool) string {
	requirements := []string{
		fmt.Sprintf("at least %d characters", minLength),
	}
	
	if requireMixed {
		requirements = append(requirements, "both uppercase and lowercase letters")
	}
	if requireNumbers {
		requirements = append(requirements, "at least one number")
	}
	if requireSymbols {
		requirements = append(requirements, "at least one special character")
	}
	
	return "must contain " + strings.Join(requirements, ", ")
}

// SanitizeInput sanitizes user input by removing potentially dangerous characters
func SanitizeInput(input string) string {
	// Remove null bytes
	input = strings.ReplaceAll(input, "\x00", "")
	
	// Remove control characters except newlines and tabs
	var sanitized strings.Builder
	for _, r := range input {
		if unicode.IsPrint(r) || r == '\n' || r == '\t' {
			sanitized.WriteRune(r)
		}
	}
	
	// Trim whitespace
	return strings.TrimSpace(sanitized.String())
}

// EscapeHTML escapes HTML characters to prevent XSS
func EscapeHTML(input string) string {
	replacer := strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		"\"", "&quot;",
		"'", "&#39;",
	)
	return replacer.Replace(input)
}

// Default validators for common use cases

// NewVMValidator creates a validator for VM-related inputs
func NewVMValidator() *InputValidator {
	validator := NewInputValidator()
	
	validator.AddRule("name", RequiredRule())
	validator.AddRule("name", VMNameRule())
	validator.AddRule("name", NoSQLInjectionRule())
	validator.AddRule("name", NoXSSRule())
	
	validator.AddRule("description", MaxLengthRule(500))
	validator.AddRule("description", NoSQLInjectionRule())
	validator.AddRule("description", NoXSSRule())
	
	validator.AddRule("cpu_shares", RequiredRule())
	validator.AddRule("memory_mb", RequiredRule())
	
	return validator
}

// NewUserValidator creates a validator for user-related inputs
func NewUserValidator(passwordMinLength int, requireMixed, requireNumbers, requireSymbols bool) *InputValidator {
	validator := NewInputValidator()
	
	validator.AddRule("username", RequiredRule())
	validator.AddRule("username", MinLengthRule(3))
	validator.AddRule("username", MaxLengthRule(50))
	validator.AddRule("username", AlphanumericRule())
	validator.AddRule("username", NoSQLInjectionRule())
	
	validator.AddRule("email", RequiredRule())
	validator.AddRule("email", EmailRule())
	validator.AddRule("email", MaxLengthRule(255))
	validator.AddRule("email", NoSQLInjectionRule())
	
	validator.AddRule("password", RequiredRule())
	validator.AddRule("password", PasswordStrengthRule(passwordMinLength, requireMixed, requireNumbers, requireSymbols))
	
	validator.AddRule("first_name", RequiredRule())
	validator.AddRule("first_name", MaxLengthRule(100))
	validator.AddRule("first_name", NoSQLInjectionRule())
	validator.AddRule("first_name", NoXSSRule())
	
	validator.AddRule("last_name", RequiredRule())
	validator.AddRule("last_name", MaxLengthRule(100))
	validator.AddRule("last_name", NoSQLInjectionRule())
	validator.AddRule("last_name", NoXSSRule())
	
	return validator
}