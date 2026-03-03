package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/pquerna/otp"
	"github.com/pquerna/otp/totp"
	"github.com/skip2/go-qrcode"
)

// TwoFactorService manages 2FA operations
type TwoFactorService struct {
	mu              sync.RWMutex
	userSecrets     map[string]*UserTwoFactor
	backupCodes     map[string][]string
	rateLimiter     map[string]*RateLimit
	issuer          string
	encryptionKey   []byte
}

// UserTwoFactor stores 2FA data for a user
type UserTwoFactor struct {
	UserID      string    `json:"user_id"`
	Secret      string    `json:"secret"`
	Enabled     bool      `json:"enabled"`
	SetupAt     time.Time `json:"setup_at"`
	LastUsed    time.Time `json:"last_used"`
	BackupCodes []string  `json:"backup_codes"`
	Algorithm   string    `json:"algorithm"`
	Digits      int       `json:"digits"`
	Period      int       `json:"period"`
}

// RateLimit tracks verification attempts
type RateLimit struct {
	Attempts  int       `json:"attempts"`
	LastReset time.Time `json:"last_reset"`
	Blocked   bool      `json:"blocked"`
	BlockedUntil time.Time `json:"blocked_until"`
}

// TwoFactorSetupResponse contains setup information
type TwoFactorSetupResponse struct {
	Secret      string   `json:"secret"`
	QRCodeURL   string   `json:"qr_code_url"`
	BackupCodes []string `json:"backup_codes"`
	Manual      string   `json:"manual_entry_key"`
}

// TwoFactorVerifyRequest represents a verification request
type TwoFactorVerifyRequest struct {
	UserID string `json:"user_id"`
	Code   string `json:"code"`
	IsBackupCode bool `json:"is_backup_code,omitempty"`
}

// TwoFactorVerifyResponse represents a verification response
type TwoFactorVerifyResponse struct {
	Valid         bool     `json:"valid"`
	RemainingCodes int     `json:"remaining_backup_codes,omitempty"`
	Error         string   `json:"error,omitempty"`
}

const (
	MaxVerificationAttempts = 5
	RateLimitWindow        = 15 * time.Minute
	BlockDuration         = 30 * time.Minute
	BackupCodeLength      = 8
	BackupCodeCount       = 10
)

// NewTwoFactorService creates a new 2FA service
func NewTwoFactorService(issuer string, encryptionKey []byte) *TwoFactorService {
	return &TwoFactorService{
		userSecrets:   make(map[string]*UserTwoFactor),
		backupCodes:   make(map[string][]string),
		rateLimiter:   make(map[string]*RateLimit),
		issuer:        issuer,
		encryptionKey: encryptionKey,
	}
}

// SetupTwoFactor initiates 2FA setup for a user
func (tfs *TwoFactorService) SetupTwoFactor(userID, accountName string) (*TwoFactorSetupResponse, error) {
	if userID == "" || accountName == "" {
		return nil, fmt.Errorf("user ID and account name are required")
	}

	// Generate secret
	key, err := totp.Generate(totp.GenerateOpts{
		Issuer:      tfs.issuer,
		AccountName: accountName,
		SecretSize:  32,
		Digits:      otp.DigitsSix,
		Algorithm:   otp.AlgorithmSHA1,
		Period:      30,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate TOTP key: %w", err)
	}

	// Generate backup codes
	backupCodes, err := tfs.generateBackupCodes()
	if err != nil {
		return nil, fmt.Errorf("failed to generate backup codes: %w", err)
	}

	// Store user 2FA data (not enabled yet)
	tfs.mu.Lock()
	tfs.userSecrets[userID] = &UserTwoFactor{
		UserID:      userID,
		Secret:      key.Secret(),
		Enabled:     false,
		SetupAt:     time.Now(),
		BackupCodes: backupCodes,
		Algorithm:   string(otp.AlgorithmSHA1),
		Digits:      int(otp.DigitsSix),
		Period:      30,
	}
	tfs.mu.Unlock()

	// Generate manual entry key (formatted for user convenience)
	manualKey := tfs.formatManualEntryKey(key.Secret())

	response := &TwoFactorSetupResponse{
		Secret:      key.Secret(),
		QRCodeURL:   key.URL(),
		BackupCodes: backupCodes,
		Manual:      manualKey,
	}

	log.Printf("2FA setup initiated for user: %s", userID)
	return response, nil
}

// GenerateQRCode generates a QR code image for the setup
func (tfs *TwoFactorService) GenerateQRCode(userID string) ([]byte, error) {
	tfs.mu.RLock()
	userTwoFactor, exists := tfs.userSecrets[userID]
	tfs.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("2FA not set up for user %s", userID)
	}

	// Reconstruct the URL for QR code generation
	key, err := otp.NewKeyFromURL(fmt.Sprintf(
		"otpauth://totp/%s:%s?secret=%s&issuer=%s&algorithm=SHA1&digits=6&period=30",
		tfs.issuer, userID, userTwoFactor.Secret, tfs.issuer,
	))
	if err != nil {
		return nil, fmt.Errorf("failed to create OTP key: %w", err)
	}

	// Generate QR code
	qrCode, err := qrcode.Encode(key.URL(), qrcode.Medium, 256)
	if err != nil {
		return nil, fmt.Errorf("failed to generate QR code: %w", err)
	}

	return qrCode, nil
}

// VerifyAndEnable verifies a code and enables 2FA if successful
func (tfs *TwoFactorService) VerifyAndEnable(userID, code string) error {
	if err := tfs.checkRateLimit(userID); err != nil {
		return err
	}

	valid, err := tfs.verifyCode(userID, code, false)
	if err != nil {
		tfs.recordFailedAttempt(userID)
		return err
	}

	if !valid {
		tfs.recordFailedAttempt(userID)
		return fmt.Errorf("invalid verification code")
	}

	// Enable 2FA
	tfs.mu.Lock()
	if userTwoFactor, exists := tfs.userSecrets[userID]; exists {
		userTwoFactor.Enabled = true
		userTwoFactor.LastUsed = time.Now()
	}
	tfs.mu.Unlock()

	// Reset rate limiting on successful enable
	tfs.resetRateLimit(userID)

	log.Printf("2FA enabled for user: %s", userID)
	return nil
}

// VerifyCode verifies a 2FA code for authentication
func (tfs *TwoFactorService) VerifyCode(req TwoFactorVerifyRequest) (*TwoFactorVerifyResponse, error) {
	if err := tfs.checkRateLimit(req.UserID); err != nil {
		return &TwoFactorVerifyResponse{
			Valid: false,
			Error: err.Error(),
		}, nil
	}

	valid, err := tfs.verifyCode(req.UserID, req.Code, req.IsBackupCode)
	if err != nil {
		tfs.recordFailedAttempt(req.UserID)
		return &TwoFactorVerifyResponse{
			Valid: false,
			Error: err.Error(),
		}, nil
	}

	if !valid {
		tfs.recordFailedAttempt(req.UserID)
		return &TwoFactorVerifyResponse{
			Valid: false,
			Error: "Invalid code",
		}, nil
	}

	// Successful verification - reset rate limit
	tfs.resetRateLimit(req.UserID)

	// Update last used timestamp
	tfs.mu.Lock()
	if userTwoFactor, exists := tfs.userSecrets[req.UserID]; exists {
		userTwoFactor.LastUsed = time.Now()
	}
	tfs.mu.Unlock()

	// Get remaining backup codes count
	remainingCodes := tfs.getRemainingBackupCodesCount(req.UserID)

	return &TwoFactorVerifyResponse{
		Valid:          true,
		RemainingCodes: remainingCodes,
	}, nil
}

// verifyCode internal method to verify codes
func (tfs *TwoFactorService) verifyCode(userID, code string, isBackupCode bool) (bool, error) {
	tfs.mu.RLock()
	userTwoFactor, exists := tfs.userSecrets[userID]
	tfs.mu.RUnlock()

	if !exists {
		return false, fmt.Errorf("2FA not set up for user")
	}

	if !userTwoFactor.Enabled && !isBackupCode {
		// During setup, we allow TOTP verification
		return totp.Validate(code, userTwoFactor.Secret), nil
	}

	if isBackupCode {
		return tfs.verifyBackupCode(userID, code)
	}

	// Verify TOTP code with time window tolerance
	valid, err := totp.ValidateCustom(
		code,
		userTwoFactor.Secret,
		time.Now().UTC(),
		totp.ValidateOpts{
			Period:    uint(userTwoFactor.Period),
			Skew:      1, // Allow 1 period skew (30 seconds before/after)
			Digits:    otp.Digits(userTwoFactor.Digits),
			Algorithm: otp.AlgorithmSHA1,
		},
	)
	if err != nil {
		return false, fmt.Errorf("TOTP validation failed: %v", err)
	}

	return valid, nil
}

// verifyBackupCode verifies and consumes a backup code
func (tfs *TwoFactorService) verifyBackupCode(userID, code string) (bool, error) {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	if !exists {
		return false, fmt.Errorf("2FA not set up for user")
	}

	// Find and remove the backup code if it matches
	for i, backupCode := range userTwoFactor.BackupCodes {
		if subtle.ConstantTimeCompare([]byte(code), []byte(backupCode)) == 1 {
			// Remove the used backup code
			userTwoFactor.BackupCodes = append(
				userTwoFactor.BackupCodes[:i],
				userTwoFactor.BackupCodes[i+1:]...,
			)
			log.Printf("Backup code used for user: %s, remaining: %d", userID, len(userTwoFactor.BackupCodes))
			return true, nil
		}
	}

	return false, nil
}

// RegenerateBackupCodes generates new backup codes for a user
func (tfs *TwoFactorService) RegenerateBackupCodes(userID string) ([]string, error) {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	if !exists {
		return nil, fmt.Errorf("2FA not set up for user")
	}

	if !userTwoFactor.Enabled {
		return nil, fmt.Errorf("2FA not enabled for user")
	}

	backupCodes, err := tfs.generateBackupCodes()
	if err != nil {
		return nil, fmt.Errorf("failed to generate backup codes: %w", err)
	}

	userTwoFactor.BackupCodes = backupCodes
	log.Printf("Backup codes regenerated for user: %s", userID)

	return backupCodes, nil
}

// GetBackupCodes returns the current backup codes for a user
func (tfs *TwoFactorService) GetBackupCodes(userID string) ([]string, error) {
	tfs.mu.RLock()
	defer tfs.mu.RUnlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	if !exists {
		return nil, fmt.Errorf("2FA not set up for user")
	}

	if !userTwoFactor.Enabled {
		return nil, fmt.Errorf("2FA not enabled for user")
	}

	// Return a copy to prevent modification
	codes := make([]string, len(userTwoFactor.BackupCodes))
	copy(codes, userTwoFactor.BackupCodes)

	return codes, nil
}

// DisableTwoFactor disables 2FA for a user
func (tfs *TwoFactorService) DisableTwoFactor(userID string) error {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	if !exists {
		return fmt.Errorf("2FA not set up for user")
	}

	userTwoFactor.Enabled = false
	userTwoFactor.BackupCodes = nil

	// Clean up rate limiting
	delete(tfs.rateLimiter, userID)

	log.Printf("2FA disabled for user: %s", userID)
	return nil
}

// IsEnabled checks if 2FA is enabled for a user
func (tfs *TwoFactorService) IsEnabled(userID string) bool {
	tfs.mu.RLock()
	defer tfs.mu.RUnlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	return exists && userTwoFactor.Enabled
}

// GetUserTwoFactorInfo returns 2FA information for a user (without secrets)
func (tfs *TwoFactorService) GetUserTwoFactorInfo(userID string) (*UserTwoFactor, error) {
	tfs.mu.RLock()
	defer tfs.mu.RUnlock()

	userTwoFactor, exists := tfs.userSecrets[userID]
	if !exists {
		return nil, fmt.Errorf("2FA not set up for user")
	}

	// Return sanitized copy without secrets
	info := &UserTwoFactor{
		UserID:    userTwoFactor.UserID,
		Enabled:   userTwoFactor.Enabled,
		SetupAt:   userTwoFactor.SetupAt,
		LastUsed:  userTwoFactor.LastUsed,
		Algorithm: userTwoFactor.Algorithm,
		Digits:    userTwoFactor.Digits,
		Period:    userTwoFactor.Period,
	}

	return info, nil
}

// generateBackupCodes generates secure backup codes
func (tfs *TwoFactorService) generateBackupCodes() ([]string, error) {
	codes := make([]string, BackupCodeCount)

	for i := 0; i < BackupCodeCount; i++ {
		code, err := tfs.generateSecureCode(BackupCodeLength)
		if err != nil {
			return nil, err
		}
		codes[i] = code
	}

	return codes, nil
}

// generateSecureCode generates a secure random code
func (tfs *TwoFactorService) generateSecureCode(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}

	// Use base32 encoding for better human readability
	code := base64.StdEncoding.EncodeToString(bytes)
	// Remove padding and take only the required length
	code = strings.TrimRight(code, "=")
	if len(code) > length {
		code = code[:length]
	}

	return strings.ToUpper(code), nil
}

// formatManualEntryKey formats the secret for manual entry
func (tfs *TwoFactorService) formatManualEntryKey(secret string) string {
	// Format as groups of 4 characters for better readability
	var formatted strings.Builder
	for i, char := range secret {
		if i > 0 && i%4 == 0 {
			formatted.WriteString(" ")
		}
		formatted.WriteRune(char)
	}
	return formatted.String()
}

// checkRateLimit checks if a user is rate limited
func (tfs *TwoFactorService) checkRateLimit(userID string) error {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	limit, exists := tfs.rateLimiter[userID]
	if !exists {
		tfs.rateLimiter[userID] = &RateLimit{
			Attempts:  0,
			LastReset: time.Now(),
		}
		return nil
	}

	now := time.Now()

	// Check if user is currently blocked
	if limit.Blocked && now.Before(limit.BlockedUntil) {
		return fmt.Errorf("too many failed attempts, blocked until %v", limit.BlockedUntil.Format(time.RFC3339))
	}

	// Reset block if time has passed
	if limit.Blocked && now.After(limit.BlockedUntil) {
		limit.Blocked = false
		limit.Attempts = 0
		limit.LastReset = now
	}

	// Reset attempts if window has passed
	if now.Sub(limit.LastReset) > RateLimitWindow {
		limit.Attempts = 0
		limit.LastReset = now
	}

	return nil
}

// recordFailedAttempt records a failed verification attempt
func (tfs *TwoFactorService) recordFailedAttempt(userID string) {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	limit, exists := tfs.rateLimiter[userID]
	if !exists {
		limit = &RateLimit{
			Attempts:  0,
			LastReset: time.Now(),
		}
		tfs.rateLimiter[userID] = limit
	}

	limit.Attempts++

	if limit.Attempts >= MaxVerificationAttempts {
		limit.Blocked = true
		limit.BlockedUntil = time.Now().Add(BlockDuration)
		log.Printf("User %s blocked due to too many failed 2FA attempts", userID)
	}
}

// resetRateLimit resets rate limiting for a user
func (tfs *TwoFactorService) resetRateLimit(userID string) {
	tfs.mu.Lock()
	defer tfs.mu.Unlock()

	delete(tfs.rateLimiter, userID)
}

// getRemainingBackupCodesCount returns the number of remaining backup codes
func (tfs *TwoFactorService) getRemainingBackupCodesCount(userID string) int {
	tfs.mu.RLock()
	defer tfs.mu.RUnlock()

	if userTwoFactor, exists := tfs.userSecrets[userID]; exists {
		return len(userTwoFactor.BackupCodes)
	}
	return 0
}

// GetStats returns statistics about 2FA usage
func (tfs *TwoFactorService) GetStats() map[string]interface{} {
	tfs.mu.RLock()
	defer tfs.mu.RUnlock()

	stats := map[string]interface{}{
		"total_users":      len(tfs.userSecrets),
		"enabled_users":    0,
		"rate_limited":     0,
		"blocked_users":    0,
	}

	enabledCount := 0
	rateLimitedCount := 0
	blockedCount := 0

	for _, user := range tfs.userSecrets {
		if user.Enabled {
			enabledCount++
		}
	}

	for _, limit := range tfs.rateLimiter {
		if limit.Attempts > 0 {
			rateLimitedCount++
		}
		if limit.Blocked {
			blockedCount++
		}
	}

	stats["enabled_users"] = enabledCount
	stats["rate_limited"] = rateLimitedCount
	stats["blocked_users"] = blockedCount

	return stats
}