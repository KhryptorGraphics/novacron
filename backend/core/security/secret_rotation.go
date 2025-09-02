package security

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"sync"
	"time"
)

// SecretRotationPolicy defines rotation requirements
type SecretRotationPolicy struct {
	MaxAge           time.Duration `json:"max_age"`
	RotationInterval time.Duration `json:"rotation_interval"`
	MinEntropy       int           `json:"min_entropy"`
	NotifyBefore     time.Duration `json:"notify_before"`
	AutoRotate       bool          `json:"auto_rotate"`
	RequireApproval  bool          `json:"require_approval"`
}

// SecretVersion represents a versioned secret
type SecretVersion struct {
	Version     string    `json:"version"`
	Value       string    `json:"-"` // Never log the actual value
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	RotatedFrom string    `json:"rotated_from,omitempty"`
	RotatedBy   string    `json:"rotated_by,omitempty"`
	Status      string    `json:"status"` // active, rotating, expired, revoked
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// RotationSchedule tracks rotation schedules
type RotationSchedule struct {
	SecretKey      string               `json:"secret_key"`
	NextRotation   time.Time            `json:"next_rotation"`
	LastRotation   time.Time            `json:"last_rotation"`
	Policy         SecretRotationPolicy `json:"policy"`
	NotificationSent bool               `json:"notification_sent"`
}

// SecretRotationManager manages secret lifecycle
type SecretRotationManager struct {
	provider     SecretProvider
	auditor      AuditLogger
	notifier     NotificationService
	policies     map[string]SecretRotationPolicy
	schedules    map[string]*RotationSchedule
	versions     map[string][]SecretVersion
	mu           sync.RWMutex
	stopChan     chan struct{}
	wg           sync.WaitGroup
}

// NotificationService for rotation notifications
type NotificationService interface {
	NotifyRotationPending(ctx context.Context, secretKey string, timeUntil time.Duration) error
	NotifyRotationComplete(ctx context.Context, secretKey string, newVersion string) error
	NotifyRotationFailed(ctx context.Context, secretKey string, err error) error
}

// NewSecretRotationManager creates a rotation manager
func NewSecretRotationManager(
	provider SecretProvider,
	auditor AuditLogger,
	notifier NotificationService,
) *SecretRotationManager {
	return &SecretRotationManager{
		provider:  provider,
		auditor:   auditor,
		notifier:  notifier,
		policies:  make(map[string]SecretRotationPolicy),
		schedules: make(map[string]*RotationSchedule),
		versions:  make(map[string][]SecretVersion),
		stopChan:  make(chan struct{}),
	}
}

// RegisterPolicy registers a rotation policy for a secret
func (m *SecretRotationManager) RegisterPolicy(secretKey string, policy SecretRotationPolicy) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate policy
	if policy.MaxAge < time.Hour {
		return fmt.Errorf("max age must be at least 1 hour")
	}
	if policy.RotationInterval < time.Minute {
		return fmt.Errorf("rotation interval must be at least 1 minute")
	}

	m.policies[secretKey] = policy
	
	// Create initial schedule
	m.schedules[secretKey] = &RotationSchedule{
		SecretKey:    secretKey,
		NextRotation: time.Now().Add(policy.RotationInterval),
		LastRotation: time.Now(),
		Policy:       policy,
	}

	return nil
}

// Start begins the rotation scheduler
func (m *SecretRotationManager) Start(ctx context.Context) error {
	m.wg.Add(1)
	go m.rotationScheduler(ctx)
	return nil
}

// Stop stops the rotation scheduler
func (m *SecretRotationManager) Stop() {
	close(m.stopChan)
	m.wg.Wait()
}

// rotationScheduler runs the rotation schedule
func (m *SecretRotationManager) rotationScheduler(ctx context.Context) {
	defer m.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.checkRotations(ctx)
		}
	}
}

// checkRotations checks and performs due rotations
func (m *SecretRotationManager) checkRotations(ctx context.Context) {
	m.mu.RLock()
	schedules := make([]*RotationSchedule, 0, len(m.schedules))
	for _, s := range m.schedules {
		schedules = append(schedules, s)
	}
	m.mu.RUnlock()

	now := time.Now()
	for _, schedule := range schedules {
		// Check if notification needed
		notifyTime := schedule.NextRotation.Add(-schedule.Policy.NotifyBefore)
		if now.After(notifyTime) && !schedule.NotificationSent {
			timeUntil := schedule.NextRotation.Sub(now)
			if err := m.notifier.NotifyRotationPending(ctx, schedule.SecretKey, timeUntil); err != nil {
				// Log but don't fail rotation
				fmt.Printf("Failed to send rotation notification: %v\n", err)
			}
			schedule.NotificationSent = true
		}

		// Check if rotation needed
		if now.After(schedule.NextRotation) {
			if schedule.Policy.AutoRotate && !schedule.Policy.RequireApproval {
				if err := m.RotateSecret(ctx, schedule.SecretKey, "system"); err != nil {
					m.notifier.NotifyRotationFailed(ctx, schedule.SecretKey, err)
				}
			}
		}
	}
}

// RotateSecret performs secret rotation
func (m *SecretRotationManager) RotateSecret(ctx context.Context, secretKey string, rotatedBy string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	policy, exists := m.policies[secretKey]
	if !exists {
		return fmt.Errorf("no rotation policy for secret: %s", secretKey)
	}

	// Get current secret
	currentValue, err := m.provider.GetSecret(ctx, secretKey)
	if err != nil {
		return fmt.Errorf("failed to get current secret: %w", err)
	}

	// Generate new secret
	newValue, err := m.generateNewSecret(policy)
	if err != nil {
		return fmt.Errorf("failed to generate new secret: %w", err)
	}

	// Create new version
	newVersion := SecretVersion{
		Version:     fmt.Sprintf("v%d", time.Now().Unix()),
		Value:       newValue,
		CreatedAt:   time.Now(),
		ExpiresAt:   time.Now().Add(policy.MaxAge),
		RotatedFrom: m.getCurrentVersion(secretKey),
		RotatedBy:   rotatedBy,
		Status:      "rotating",
	}

	// Store versions
	if _, ok := m.versions[secretKey]; !ok {
		m.versions[secretKey] = []SecretVersion{}
	}
	m.versions[secretKey] = append(m.versions[secretKey], newVersion)

	// Perform rotation in provider
	if err := m.provider.SetSecret(ctx, secretKey, newValue); err != nil {
		newVersion.Status = "failed"
		m.auditor.LogSecretRotation(ctx, rotatedBy, secretKey, m.getCurrentVersion(secretKey), newVersion.Version, ResultFailure)
		return fmt.Errorf("failed to set new secret: %w", err)
	}

	// Mark as active
	newVersion.Status = "active"
	
	// Mark old versions as rotated
	for i := range m.versions[secretKey] {
		if m.versions[secretKey][i].Status == "active" && m.versions[secretKey][i].Version != newVersion.Version {
			m.versions[secretKey][i].Status = "rotated"
		}
	}

	// Update schedule
	if schedule, ok := m.schedules[secretKey]; ok {
		schedule.LastRotation = time.Now()
		schedule.NextRotation = time.Now().Add(policy.RotationInterval)
		schedule.NotificationSent = false
	}

	// Audit log
	m.auditor.LogSecretRotation(ctx, rotatedBy, secretKey, m.getCurrentVersion(secretKey), newVersion.Version, ResultSuccess)

	// Notify completion
	m.notifier.NotifyRotationComplete(ctx, secretKey, newVersion.Version)

	return nil
}

// generateNewSecret generates a new secret value
func (m *SecretRotationManager) generateNewSecret(policy SecretRotationPolicy) (string, error) {
	// Calculate byte length for desired entropy
	byteLength := policy.MinEntropy / 8
	if byteLength < 32 {
		byteLength = 32 // Minimum 256 bits
	}

	bytes := make([]byte, byteLength)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("failed to generate random bytes: %w", err)
	}

	return base64.URLEncoding.EncodeToString(bytes), nil
}

// getCurrentVersion gets the current active version
func (m *SecretRotationManager) getCurrentVersion(secretKey string) string {
	versions, ok := m.versions[secretKey]
	if !ok {
		return "v0"
	}

	for _, v := range versions {
		if v.Status == "active" {
			return v.Version
		}
	}

	return "v0"
}

// GetSecretWithVersion retrieves a specific version
func (m *SecretRotationManager) GetSecretWithVersion(ctx context.Context, secretKey, version string) (*SecretVersion, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	versions, ok := m.versions[secretKey]
	if !ok {
		return nil, fmt.Errorf("no versions for secret: %s", secretKey)
	}

	for _, v := range versions {
		if v.Version == version {
			// Audit access to specific version
			m.auditor.LogSecretAccess(ctx, "system", secretKey, ActionRead, ResultSuccess, map[string]interface{}{
				"version": version,
			})
			return &v, nil
		}
	}

	return nil, fmt.Errorf("version not found: %s", version)
}

// GetRotationHistory retrieves rotation history
func (m *SecretRotationManager) GetRotationHistory(ctx context.Context, secretKey string) ([]SecretVersion, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	versions, ok := m.versions[secretKey]
	if !ok {
		return nil, fmt.Errorf("no history for secret: %s", secretKey)
	}

	// Return copy without actual values
	history := make([]SecretVersion, len(versions))
	for i, v := range versions {
		history[i] = v
		history[i].Value = "" // Never expose values in history
	}

	return history, nil
}

// ApproveRotation approves a pending rotation
func (m *SecretRotationManager) ApproveRotation(ctx context.Context, secretKey, approvedBy string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	schedule, ok := m.schedules[secretKey]
	if !ok {
		return fmt.Errorf("no schedule for secret: %s", secretKey)
	}

	if !schedule.Policy.RequireApproval {
		return fmt.Errorf("secret does not require approval for rotation")
	}

	// Perform the rotation
	return m.RotateSecret(ctx, secretKey, approvedBy)
}

// RevokeVersion revokes a specific version
func (m *SecretRotationManager) RevokeVersion(ctx context.Context, secretKey, version, revokedBy string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	versions, ok := m.versions[secretKey]
	if !ok {
		return fmt.Errorf("no versions for secret: %s", secretKey)
	}

	for i, v := range versions {
		if v.Version == version {
			m.versions[secretKey][i].Status = "revoked"
			m.versions[secretKey][i].Metadata = map[string]interface{}{
				"revoked_by": revokedBy,
				"revoked_at": time.Now(),
			}

			// Audit the revocation
			m.auditor.LogSecretModification(ctx, revokedBy, secretKey, ActionDelete, ResultSuccess, map[string]interface{}{
				"version": version,
				"action":  "revoke",
			})

			return nil
		}
	}

	return fmt.Errorf("version not found: %s", version)
}

// GetPendingRotations returns secrets pending rotation
func (m *SecretRotationManager) GetPendingRotations(ctx context.Context) ([]RotationSchedule, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pending := []RotationSchedule{}
	now := time.Now()

	for _, schedule := range m.schedules {
		if now.After(schedule.NextRotation) || (schedule.Policy.RequireApproval && schedule.NotificationSent) {
			pending = append(pending, *schedule)
		}
	}

	return pending, nil
}

// DefaultNotificationService implements basic notifications
type DefaultNotificationService struct {
	logger *slog.Logger
}

func NewDefaultNotificationService() *DefaultNotificationService {
	return &DefaultNotificationService{
		logger: slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}
}

func (n *DefaultNotificationService) NotifyRotationPending(ctx context.Context, secretKey string, timeUntil time.Duration) error {
	n.logger.Info("Secret rotation pending",
		"secret", secretKey,
		"time_until", timeUntil.String(),
	)
	return nil
}

func (n *DefaultNotificationService) NotifyRotationComplete(ctx context.Context, secretKey string, newVersion string) error {
	n.logger.Info("Secret rotation complete",
		"secret", secretKey,
		"new_version", newVersion,
	)
	return nil
}

func (n *DefaultNotificationService) NotifyRotationFailed(ctx context.Context, secretKey string, err error) error {
	n.logger.Error("Secret rotation failed",
		"secret", secretKey,
		"error", err,
	)
	return nil
}