package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"regexp"
	"strings"
	"time"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/bcrypt"
)

// PasswordSecurityConfig defines password security settings
type PasswordSecurityConfig struct {
	// MinLength minimum password length
	MinLength int
	// MaxLength maximum password length
	MaxLength int
	// RequireUppercase requires at least one uppercase letter
	RequireUppercase bool
	// RequireLowercase requires at least one lowercase letter
	RequireLowercase bool
	// RequireNumbers requires at least one number
	RequireNumbers bool
	// RequireSpecialChars requires at least one special character
	RequireSpecialChars bool
	// MinSpecialChars minimum number of special characters
	MinSpecialChars int
	// ForbidCommonPasswords forbids common weak passwords
	ForbidCommonPasswords bool
	// ForbidPersonalInfo forbids personal information in password
	ForbidPersonalInfo bool
	// MaxAge maximum password age before forced rotation
	MaxAge time.Duration
	// HistorySize number of previous passwords to remember
	HistorySize int
	// HashAlgorithm preferred hashing algorithm (bcrypt, argon2)
	HashAlgorithm string
	// BcryptCost bcrypt cost parameter
	BcryptCost int
	// Argon2Memory argon2 memory parameter (KB)
	Argon2Memory uint32
	// Argon2Time argon2 time parameter
	Argon2Time uint32
	// Argon2Threads argon2 threads parameter
	Argon2Threads uint8
	// Argon2KeyLen argon2 key length
	Argon2KeyLen uint32
	// SaltLength salt length in bytes
	SaltLength int
}

// PasswordHash represents a password hash with metadata
type PasswordHash struct {
	Hash      string    `json:"hash"`
	Salt      string    `json:"salt"`
	Algorithm string    `json:"algorithm"`
	CreatedAt time.Time `json:"created_at"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

// PasswordHistory represents password history for a user
type PasswordHistory struct {
	UserID    string         `json:"user_id"`
	Hashes    []PasswordHash `json:"hashes"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

// PasswordSecurityService handles secure password operations
type PasswordSecurityService struct {
	config         PasswordSecurityConfig
	commonPasswords map[string]bool
	history        map[string]*PasswordHistory
}

// NewPasswordSecurityService creates a new password security service
func NewPasswordSecurityService(config PasswordSecurityConfig) *PasswordSecurityService {
	if config.MinLength == 0 {
		config.MinLength = 12
	}
	if config.MaxLength == 0 {
		config.MaxLength = 128
	}
	if config.HashAlgorithm == "" {
		config.HashAlgorithm = "argon2"
	}
	if config.BcryptCost == 0 {
		config.BcryptCost = 12
	}
	if config.Argon2Memory == 0 {
		config.Argon2Memory = 64 * 1024 // 64MB
	}
	if config.Argon2Time == 0 {
		config.Argon2Time = 3
	}
	if config.Argon2Threads == 0 {
		config.Argon2Threads = 2
	}
	if config.Argon2KeyLen == 0 {
		config.Argon2KeyLen = 32
	}
	if config.SaltLength == 0 {
		config.SaltLength = 32
	}
	if config.MaxAge == 0 {
		config.MaxAge = 90 * 24 * time.Hour // 90 days
	}
	if config.HistorySize == 0 {
		config.HistorySize = 12
	}

	service := &PasswordSecurityService{
		config:  config,
		history: make(map[string]*PasswordHistory),
	}

	if config.ForbidCommonPasswords {
		service.commonPasswords = loadCommonPasswords()
	}

	return service
}

// ValidatePassword validates password against security policies
func (p *PasswordSecurityService) ValidatePassword(password string, userInfo *User) error {
	// Length validation
	if len(password) < p.config.MinLength {
		return fmt.Errorf("password must be at least %d characters long", p.config.MinLength)
	}
	if len(password) > p.config.MaxLength {
		return fmt.Errorf("password must not exceed %d characters", p.config.MaxLength)
	}

	// Character class validation
	if p.config.RequireUppercase && !regexp.MustCompile(`[A-Z]`).MatchString(password) {
		return fmt.Errorf("password must contain at least one uppercase letter")
	}
	if p.config.RequireLowercase && !regexp.MustCompile(`[a-z]`).MatchString(password) {
		return fmt.Errorf("password must contain at least one lowercase letter")
	}
	if p.config.RequireNumbers && !regexp.MustCompile(`[0-9]`).MatchString(password) {
		return fmt.Errorf("password must contain at least one number")
	}

	// Special character validation
	if p.config.RequireSpecialChars {
		specialCharRegex := regexp.MustCompile(`[!@#$%^&*()_+\-=\[\]{}|;':",./<>?~` + "`" + `]`)
		matches := specialCharRegex.FindAllString(password, -1)
		if len(matches) < p.config.MinSpecialChars {
			if p.config.MinSpecialChars == 1 {
				return fmt.Errorf("password must contain at least one special character")
			} else {
				return fmt.Errorf("password must contain at least %d special characters", p.config.MinSpecialChars)
			}
		}
	}

	// Common password validation
	if p.config.ForbidCommonPasswords && p.isCommonPassword(password) {
		return fmt.Errorf("password is too common, please choose a more unique password")
	}

	// Personal information validation
	if p.config.ForbidPersonalInfo && userInfo != nil {
		if p.containsPersonalInfo(password, userInfo) {
			return fmt.Errorf("password must not contain personal information")
		}
	}

	// Password history validation
	if p.config.HistorySize > 0 && userInfo != nil {
		if p.isInHistory(password, userInfo.ID) {
			return fmt.Errorf("password has been used recently, please choose a different password")
		}
	}

	return nil
}

// HashPassword creates a secure password hash
func (p *PasswordSecurityService) HashPassword(password string) (*PasswordHash, error) {
	salt, err := p.generateSalt()
	if err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}

	now := time.Now()
	hash := &PasswordHash{
		Salt:      salt,
		Algorithm: p.config.HashAlgorithm,
		CreatedAt: now,
		Params:    make(map[string]interface{}),
	}

	switch p.config.HashAlgorithm {
	case "bcrypt":
		hashBytes, err := bcrypt.GenerateFromPassword([]byte(password+salt), p.config.BcryptCost)
		if err != nil {
			return nil, fmt.Errorf("bcrypt hash generation failed: %w", err)
		}
		hash.Hash = base64.StdEncoding.EncodeToString(hashBytes)
		hash.Params["cost"] = p.config.BcryptCost

	case "argon2":
		saltBytes, err := base64.StdEncoding.DecodeString(salt)
		if err != nil {
			return nil, fmt.Errorf("salt decoding failed: %w", err)
		}
		hashBytes := argon2.IDKey([]byte(password), saltBytes, p.config.Argon2Time, p.config.Argon2Memory, p.config.Argon2Threads, p.config.Argon2KeyLen)
		hash.Hash = base64.StdEncoding.EncodeToString(hashBytes)
		hash.Params["memory"] = p.config.Argon2Memory
		hash.Params["time"] = p.config.Argon2Time
		hash.Params["threads"] = p.config.Argon2Threads
		hash.Params["keyLen"] = p.config.Argon2KeyLen

	default:
		return nil, fmt.Errorf("unsupported hash algorithm: %s", p.config.HashAlgorithm)
	}

	return hash, nil
}

// VerifyPassword verifies a password against a hash
func (p *PasswordSecurityService) VerifyPassword(password string, hash *PasswordHash) (bool, error) {
	switch hash.Algorithm {
	case "bcrypt":
		hashBytes, err := base64.StdEncoding.DecodeString(hash.Hash)
		if err != nil {
			return false, fmt.Errorf("hash decoding failed: %w", err)
		}
		err = bcrypt.CompareHashAndPassword(hashBytes, []byte(password+hash.Salt))
		return err == nil, nil

	case "argon2":
		saltBytes, err := base64.StdEncoding.DecodeString(hash.Salt)
		if err != nil {
			return false, fmt.Errorf("salt decoding failed: %w", err)
		}
		expectedHash, err := base64.StdEncoding.DecodeString(hash.Hash)
		if err != nil {
			return false, fmt.Errorf("hash decoding failed: %w", err)
		}

		// Extract parameters
		memory := uint32(hash.Params["memory"].(float64))
		time := uint32(hash.Params["time"].(float64))
		threads := uint8(hash.Params["threads"].(float64))
		keyLen := uint32(hash.Params["keyLen"].(float64))

		actualHash := argon2.IDKey([]byte(password), saltBytes, time, memory, threads, keyLen)
		return subtle.ConstantTimeCompare(expectedHash, actualHash) == 1, nil

	default:
		return false, fmt.Errorf("unsupported hash algorithm: %s", hash.Algorithm)
	}
}

// AddToHistory adds a password hash to user's history
func (p *PasswordSecurityService) AddToHistory(userID string, hash *PasswordHash) {
	history, exists := p.history[userID]
	if !exists {
		history = &PasswordHistory{
			UserID:    userID,
			Hashes:    make([]PasswordHash, 0, p.config.HistorySize),
			CreatedAt: time.Now(),
		}
		p.history[userID] = history
	}

	// Add new hash
	history.Hashes = append(history.Hashes, *hash)
	history.UpdatedAt = time.Now()

	// Maintain history size limit
	if len(history.Hashes) > p.config.HistorySize {
		history.Hashes = history.Hashes[len(history.Hashes)-p.config.HistorySize:]
	}
}

// IsPasswordExpired checks if password needs rotation
func (p *PasswordSecurityService) IsPasswordExpired(hash *PasswordHash) bool {
	return time.Since(hash.CreatedAt) > p.config.MaxAge
}

// GenerateSecurePassword generates a secure password meeting policy requirements
func (p *PasswordSecurityService) GenerateSecurePassword() (string, error) {
	const (
		uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		lowercase = "abcdefghijklmnopqrstuvwxyz"
		numbers   = "0123456789"
		special   = "!@#$%^&*()_+-=[]{}|;':,./<>?"
	)

	// Build character set
	var charset string
	var required []string

	if p.config.RequireUppercase {
		charset += uppercase
		required = append(required, uppercase)
	}
	if p.config.RequireLowercase {
		charset += lowercase
		required = append(required, lowercase)
	}
	if p.config.RequireNumbers {
		charset += numbers
		required = append(required, numbers)
	}
	if p.config.RequireSpecialChars {
		charset += special
		required = append(required, special)
	}

	if charset == "" {
		charset = uppercase + lowercase + numbers + special
	}

	length := p.config.MinLength
	if length < 12 {
		length = 12
	}

	for attempts := 0; attempts < 100; attempts++ {
		password := make([]byte, length)

		// Ensure required character types
		for i, reqSet := range required {
			if i < length {
				randIndex, err := p.randomInt(len(reqSet))
				if err != nil {
					return "", err
				}
				password[i] = reqSet[randIndex]
			}
		}

		// Fill remaining positions
		for i := len(required); i < length; i++ {
			randIndex, err := p.randomInt(len(charset))
			if err != nil {
				return "", err
			}
			password[i] = charset[randIndex]
		}

		// Shuffle password
		for i := len(password) - 1; i > 0; i-- {
			j, err := p.randomInt(i + 1)
			if err != nil {
				return "", err
			}
			password[i], password[j] = password[j], password[i]
		}

		passwordStr := string(password)
		
		// Validate generated password
		if err := p.ValidatePassword(passwordStr, nil); err == nil {
			return passwordStr, nil
		}
	}

	return "", fmt.Errorf("failed to generate secure password after 100 attempts")
}

// generateSalt generates a cryptographically secure random salt
func (p *PasswordSecurityService) generateSalt() (string, error) {
	salt := make([]byte, p.config.SaltLength)
	_, err := rand.Read(salt)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(salt), nil
}

// randomInt generates a cryptographically secure random integer
func (p *PasswordSecurityService) randomInt(max int) (int, error) {
	b := make([]byte, 4)
	_, err := rand.Read(b)
	if err != nil {
		return 0, err
	}
	return int(uint32(b[0])<<24|uint32(b[1])<<16|uint32(b[2])<<8|uint32(b[3])) % max, nil
}

// isCommonPassword checks if password is in common password list
func (p *PasswordSecurityService) isCommonPassword(password string) bool {
	if p.commonPasswords == nil {
		return false
	}
	_, exists := p.commonPasswords[strings.ToLower(password)]
	return exists
}

// containsPersonalInfo checks if password contains user's personal information
func (p *PasswordSecurityService) containsPersonalInfo(password, user *User) bool {
	passwordLower := strings.ToLower(password)

	// Check username
	if strings.Contains(passwordLower, strings.ToLower(user.Username)) {
		return true
	}

	// Check email
	if strings.Contains(passwordLower, strings.ToLower(user.Email)) {
		return true
	}

	// Check first/last name
	if user.FirstName != "" && strings.Contains(passwordLower, strings.ToLower(user.FirstName)) {
		return true
	}
	if user.LastName != "" && strings.Contains(passwordLower, strings.ToLower(user.LastName)) {
		return true
	}

	return false
}

// isInHistory checks if password has been used recently
func (p *PasswordSecurityService) isInHistory(password, userID string) bool {
	history, exists := p.history[userID]
	if !exists {
		return false
	}

	for _, hash := range history.Hashes {
		valid, err := p.VerifyPassword(password, &hash)
		if err == nil && valid {
			return true
		}
	}

	return false
}

// loadCommonPasswords loads common weak passwords list
func loadCommonPasswords() map[string]bool {
	// In production, this would load from a file or database
	commonPasswords := map[string]bool{
		"password":    true,
		"123456":      true,
		"password123": true,
		"admin":       true,
		"qwerty":      true,
		"letmein":     true,
		"welcome":     true,
		"monkey":      true,
		"dragon":      true,
		"secret":      true,
		"1234567":     true,
		"12345678":    true,
		"123456789":   true,
		"1234567890":  true,
		"qwertyuiop":  true,
		"asdfghjkl":   true,
		"zxcvbnm":     true,
	}
	return commonPasswords
}

// DefaultPasswordSecurityConfig returns secure default configuration
func DefaultPasswordSecurityConfig() PasswordSecurityConfig {
	return PasswordSecurityConfig{
		MinLength:             12,
		MaxLength:             128,
		RequireUppercase:      true,
		RequireLowercase:      true,
		RequireNumbers:        true,
		RequireSpecialChars:   true,
		MinSpecialChars:       1,
		ForbidCommonPasswords: true,
		ForbidPersonalInfo:    true,
		MaxAge:                90 * 24 * time.Hour,
		HistorySize:           12,
		HashAlgorithm:         "argon2",
		BcryptCost:            12,
		Argon2Memory:          64 * 1024,
		Argon2Time:            3,
		Argon2Threads:         2,
		Argon2KeyLen:          32,
		SaltLength:            32,
	}
}