package security

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ConsentManagementSystem handles user consent for data processing
type ConsentManagementSystem struct {
	db            *sql.DB
	auditLogger   AuditLogger
	notifications *NotificationService
	legalBasis    map[Purpose]string
	retentionPolicies map[DataCategory]time.Duration
	mu            sync.RWMutex
}

// ConsentRecord represents a user's consent for data processing
type ConsentRecord struct {
	ConsentID       string                 `json:"consent_id"`
	UserID          string                 `json:"user_id"`
	Purpose         Purpose                `json:"purpose"`
	DataCategories  []DataCategory         `json:"data_categories"`
	ConsentGiven    bool                   `json:"consent_given"`
	ConsentMethod   ConsentMethod          `json:"consent_method"`
	LegalBasis      string                 `json:"legal_basis"`
	Timestamp       time.Time              `json:"timestamp"`
	ExpiryDate      *time.Time             `json:"expiry_date,omitempty"`
	WithdrawnAt     *time.Time             `json:"withdrawn_at,omitempty"`
	Version         int                    `json:"version"`
	Metadata        map[string]interface{} `json:"metadata"`
	GeographicScope []string               `json:"geographic_scope"`
	IPAddress       string                 `json:"ip_address"`
	UserAgent       string                 `json:"user_agent"`
}

type ConsentMethod string

const (
	ConsentMethodExplicit   ConsentMethod = "explicit"    // Clear affirmative action
	ConsentMethodImplied    ConsentMethod = "implied"     // Continued use of service
	ConsentMethodOptOut     ConsentMethod = "opt_out"     // Pre-checked with opt-out option
	ConsentMethodLegitimate ConsentMethod = "legitimate"  // Legitimate interest
	ConsentMethodContract   ConsentMethod = "contract"    // Contractual necessity
	ConsentMethodLegal      ConsentMethod = "legal"       // Legal obligation
)

// DataPortabilityService handles data export requests
type DataPortabilityService struct {
	db              *sql.DB
	storageManager  *StorageManager
	encryptionMgr   *EncryptionManager
	formatters      map[string]DataFormatter
	auditLogger     AuditLogger
}

type DataExportRequest struct {
	RequestID     string            `json:"request_id"`
	UserID        string            `json:"user_id"`
	RequestType   ExportType        `json:"request_type"`
	DataCategories []DataCategory   `json:"data_categories"`
	Format        ExportFormat      `json:"format"`
	Status        ExportStatus      `json:"status"`
	RequestedAt   time.Time         `json:"requested_at"`
	CompletedAt   *time.Time        `json:"completed_at,omitempty"`
	ExpiryDate    time.Time         `json:"expiry_date"`
	DownloadURL   string            `json:"download_url,omitempty"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type ExportType string

const (
	ExportTypePortability ExportType = "portability"  // Data portability request
	ExportTypeAccess      ExportType = "access"       // Right of access request
	ExportTypeCorrection  ExportType = "correction"   // Data correction verification
)

type ExportFormat string

const (
	FormatJSON ExportFormat = "json"
	FormatXML  ExportFormat = "xml"
	FormatCSV  ExportFormat = "csv"
	FormatPDF  ExportFormat = "pdf"
)

type ExportStatus string

const (
	StatusPending    ExportStatus = "pending"
	StatusProcessing ExportStatus = "processing"
	StatusCompleted  ExportStatus = "completed"
	StatusFailed     ExportStatus = "failed"
	StatusExpired    ExportStatus = "expired"
)

// DataErasureService handles right to be forgotten requests
type DataErasureService struct {
	db                *sql.DB
	storageManager    *StorageManager
	dependencyMapper  *DataDependencyMapper
	auditLogger       AuditLogger
	legalRetention    *LegalRetentionService
}

type ErasureRequest struct {
	RequestID       string                 `json:"request_id"`
	UserID          string                 `json:"user_id"`
	ErasureType     ErasureType           `json:"erasure_type"`
	Reason          ErasureReason         `json:"reason"`
	DataCategories  []DataCategory        `json:"data_categories"`
	Status          ErasureStatus         `json:"status"`
	RequestedAt     time.Time             `json:"requested_at"`
	VerifiedAt      *time.Time            `json:"verified_at,omitempty"`
	CompletedAt     *time.Time            `json:"completed_at,omitempty"`
	RetentionOverride []RetentionOverride `json:"retention_override,omitempty"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type ErasureType string

const (
	ErasureComplete ErasureType = "complete"    // Complete account deletion
	ErasurePartial  ErasureType = "partial"     // Specific data categories
	ErasureAnonymize ErasureType = "anonymize"  // Anonymize instead of delete
)

type ErasureReason string

const (
	ReasonWithdrawConsent    ErasureReason = "consent_withdrawn"
	ReasonNoLongerNecessary  ErasureReason = "no_longer_necessary"
	ReasonObjectToProcessing ErasureReason = "object_to_processing"
	ReasonUnlawfulProcessing ErasureReason = "unlawful_processing"
	ReasonComplianceObligation ErasureReason = "compliance_obligation"
)

type ErasureStatus string

const (
	ErasureStatusPending   ErasureStatus = "pending"
	ErasureStatusVerifying ErasureStatus = "verifying"
	ErasureStatusApproved  ErasureStatus = "approved"
	ErasureStatusRejected  ErasureStatus = "rejected"
	ErasureStatusProcessing ErasureStatus = "processing"
	ErasureStatusCompleted ErasureStatus = "completed"
	ErasureStatusFailed    ErasureStatus = "failed"
)

// DataRetentionService manages automated data lifecycle
type DataRetentionService struct {
	db               *sql.DB
	retentionPolicies map[DataCategory]*RetentionPolicy
	scheduler        *RetentionScheduler
	auditLogger      AuditLogger
	legalRetention   *LegalRetentionService
}

type RetentionPolicy struct {
	PolicyID        string        `json:"policy_id"`
	DataCategory    DataCategory  `json:"data_category"`
	Purpose         Purpose       `json:"purpose"`
	RetentionPeriod time.Duration `json:"retention_period"`
	GracePeriod     time.Duration `json:"grace_period"`
	Action          RetentionAction `json:"action"`
	LegalBasis      string        `json:"legal_basis"`
	Jurisdiction    []string      `json:"jurisdiction"`
	Exceptions      []RetentionException `json:"exceptions"`
}

type RetentionAction string

const (
	ActionDelete     RetentionAction = "delete"
	ActionAnonymize  RetentionAction = "anonymize"
	ActionArchive    RetentionAction = "archive"
	ActionPurge      RetentionAction = "purge"
)

type RetentionException struct {
	Reason      string        `json:"reason"`
	Extension   time.Duration `json:"extension"`
	LegalBasis  string        `json:"legal_basis"`
	ReviewDate  time.Time     `json:"review_date"`
}

// DataAnonymizationService handles data anonymization
type DataAnonymizationService struct {
	anonymizers     map[DataCategory]Anonymizer
	qualityChecker  *AnonymizationQualityChecker
	auditLogger     AuditLogger
}

type Anonymizer interface {
	Anonymize(ctx context.Context, data interface{}) (interface{}, error)
	GetAnonymizationLevel() AnonymizationLevel
	ValidateAnonymization(original, anonymized interface{}) error
}

type AnonymizationLevel string

const (
	LevelPseudonymization AnonymizationLevel = "pseudonymization"
	LevelKAnonymity       AnonymizationLevel = "k_anonymity"
	LevelLDiversity       AnonymizationLevel = "l_diversity"
	LevelTCloseness       AnonymizationLevel = "t_closeness"
	LevelDifferentialPrivacy AnonymizationLevel = "differential_privacy"
)

// BreachNotificationService handles data breach notifications
type BreachNotificationService struct {
	db              *sql.DB
	notifications   *NotificationService
	regulatoryAPI   *RegulatoryNotificationService
	auditLogger     AuditLogger
	templates       map[string]*NotificationTemplate
}

type DataBreachIncident struct {
	IncidentID          string                 `json:"incident_id"`
	Severity            BreachSeverity         `json:"severity"`
	BreachType          BreachType             `json:"breach_type"`
	AffectedDataTypes   []DataCategory         `json:"affected_data_types"`
	AffectedUserCount   int                    `json:"affected_user_count"`
	DetectedAt          time.Time              `json:"detected_at"`
	ContainedAt         *time.Time             `json:"contained_at,omitempty"`
	NotificationSent    bool                   `json:"notification_sent"`
	RegulatoryNotified  bool                   `json:"regulatory_notified"`
	UserNotificationRequired bool              `json:"user_notification_required"`
	RiskAssessment      *BreachRiskAssessment  `json:"risk_assessment"`
	Metadata            map[string]interface{} `json:"metadata"`
}

type BreachSeverity string

const (
	BreachSeverityLow      BreachSeverity = "low"
	BreachSeverityMedium   BreachSeverity = "medium"
	BreachSeverityHigh     BreachSeverity = "high"
	BreachSeverityCritical BreachSeverity = "critical"
)

type BreachType string

const (
	BreachTypeConfidentiality BreachType = "confidentiality"
	BreachTypeIntegrity       BreachType = "integrity"
	BreachTypeAvailability    BreachType = "availability"
)

// NewConsentManagementSystem creates a consent management system
func NewConsentManagementSystem(db *sql.DB, auditLogger AuditLogger) *ConsentManagementSystem {
	cms := &ConsentManagementSystem{
		db:          db,
		auditLogger: auditLogger,
		legalBasis:  make(map[Purpose]string),
		retentionPolicies: make(map[DataCategory]time.Duration),
	}
	
	// Initialize legal basis mapping
	cms.initializeLegalBasisMapping()
	
	return cms
}

func (cms *ConsentManagementSystem) initializeLegalBasisMapping() {
	cms.legalBasis[PurposeMatching] = "Legitimate Interest"
	cms.legalBasis[PurposeMessaging] = "Contract Performance"
	cms.legalBasis[PurposeLocation] = "Explicit Consent"
	cms.legalBasis[PurposePayment] = "Contract Performance"
	cms.legalBasis[PurposeAnalytics] = "Legitimate Interest"
	cms.legalBasis[PurposeMarketing] = "Explicit Consent"
	cms.legalBasis[PurposeSupport] = "Legitimate Interest"
	cms.legalBasis[PurposeSafety] = "Legal Obligation"
}

// RecordConsent records user consent for data processing
func (cms *ConsentManagementSystem) RecordConsent(ctx context.Context, userID string, purpose Purpose, dataCategories []DataCategory, consentGiven bool, method ConsentMethod, clientIP, userAgent string) (*ConsentRecord, error) {
	cms.mu.Lock()
	defer cms.mu.Unlock()
	
	consentID := uuid.New().String()
	timestamp := time.Now()
	
	// Determine expiry date based on legal requirements
	var expiryDate *time.Time
	if purpose == PurposeMarketing {
		// Marketing consent expires after 2 years
		expiry := timestamp.Add(2 * 365 * 24 * time.Hour)
		expiryDate = &expiry
	}
	
	record := &ConsentRecord{
		ConsentID:       consentID,
		UserID:          userID,
		Purpose:         purpose,
		DataCategories:  dataCategories,
		ConsentGiven:    consentGiven,
		ConsentMethod:   method,
		LegalBasis:      cms.legalBasis[purpose],
		Timestamp:       timestamp,
		ExpiryDate:      expiryDate,
		Version:         1,
		Metadata:        make(map[string]interface{}),
		GeographicScope: []string{"EU", "US"}, // Based on user location
		IPAddress:       clientIP,
		UserAgent:       userAgent,
	}
	
	// Store in database
	query := `
		INSERT INTO consent_records (
			consent_id, user_id, purpose, data_categories, consent_given,
			consent_method, legal_basis, timestamp, expiry_date, version,
			metadata, geographic_scope, ip_address, user_agent
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
	`
	
	dataCategoriesJSON, _ := json.Marshal(dataCategories)
	metadataJSON, _ := json.Marshal(record.Metadata)
	geographicScopeJSON, _ := json.Marshal(record.GeographicScope)
	
	_, err := cms.db.ExecContext(ctx, query,
		record.ConsentID, record.UserID, record.Purpose, dataCategoriesJSON,
		record.ConsentGiven, record.ConsentMethod, record.LegalBasis,
		record.Timestamp, record.ExpiryDate, record.Version,
		metadataJSON, geographicScopeJSON, record.IPAddress, record.UserAgent,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to store consent record: %w", err)
	}
	
	// Audit log consent recording
	cms.auditLogger.LogSecretModification(ctx, userID, "consent_record",
		ActionWrite, ResultSuccess, map[string]interface{}{
			"consent_id": consentID,
			"purpose":    purpose,
			"consent_given": consentGiven,
			"method":     method,
		})
	
	return record, nil
}

// WithdrawConsent handles consent withdrawal
func (cms *ConsentManagementSystem) WithdrawConsent(ctx context.Context, userID string, purpose Purpose) error {
	cms.mu.Lock()
	defer cms.mu.Unlock()
	
	timestamp := time.Now()
	
	// Update consent record
	query := `
		UPDATE consent_records 
		SET consent_given = false, withdrawn_at = $1
		WHERE user_id = $2 AND purpose = $3 AND consent_given = true AND withdrawn_at IS NULL
	`
	
	_, err := cms.db.ExecContext(ctx, query, timestamp, userID, purpose)
	if err != nil {
		return fmt.Errorf("failed to withdraw consent: %w", err)
	}
	
	// Audit log consent withdrawal
	cms.auditLogger.LogSecretModification(ctx, userID, "consent_withdrawal",
		ActionUpdate, ResultSuccess, map[string]interface{}{
			"purpose":      purpose,
			"withdrawn_at": timestamp,
		})
	
	// Trigger data processing review
	go cms.reviewDataProcessingAfterWithdrawal(userID, purpose)
	
	return nil
}

func (cms *ConsentManagementSystem) reviewDataProcessingAfterWithdrawal(userID string, purpose Purpose) {
	// Implementation would review if data processing can continue under different legal basis
	// or if data needs to be deleted/anonymized
}

// NewDataPortabilityService creates a data portability service
func NewDataPortabilityService(db *sql.DB, storageManager *StorageManager, encryptionMgr *EncryptionManager, auditLogger AuditLogger) *DataPortabilityService {
	return &DataPortabilityService{
		db:             db,
		storageManager: storageManager,
		encryptionMgr:  encryptionMgr,
		formatters:     make(map[string]DataFormatter),
		auditLogger:    auditLogger,
	}
}

// ProcessDataExportRequest handles data export requests
func (dps *DataPortabilityService) ProcessDataExportRequest(ctx context.Context, userID string, exportType ExportType, dataCategories []DataCategory, format ExportFormat) (*DataExportRequest, error) {
	requestID := uuid.New().String()
	timestamp := time.Now()
	expiryDate := timestamp.Add(30 * 24 * time.Hour) // 30 days to download
	
	request := &DataExportRequest{
		RequestID:      requestID,
		UserID:         userID,
		RequestType:    exportType,
		DataCategories: dataCategories,
		Format:         format,
		Status:         StatusPending,
		RequestedAt:    timestamp,
		ExpiryDate:     expiryDate,
		Metadata:       make(map[string]interface{}),
	}
	
	// Store request in database
	query := `
		INSERT INTO data_export_requests (
			request_id, user_id, request_type, data_categories, format,
			status, requested_at, expiry_date, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	
	dataCategoriesJSON, _ := json.Marshal(dataCategories)
	metadataJSON, _ := json.Marshal(request.Metadata)
	
	_, err := dps.db.ExecContext(ctx, query,
		request.RequestID, request.UserID, request.RequestType,
		dataCategoriesJSON, request.Format, request.Status,
		request.RequestedAt, request.ExpiryDate, metadataJSON,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to store export request: %w", err)
	}
	
	// Start async processing
	go dps.processExportAsync(requestID)
	
	// Audit log export request
	dps.auditLogger.LogSecretAccess(ctx, userID, "data_export_request",
		ActionRead, ResultSuccess, map[string]interface{}{
			"request_id":      requestID,
			"export_type":     exportType,
			"data_categories": dataCategories,
			"format":          format,
		})
	
	return request, nil
}

func (dps *DataPortabilityService) processExportAsync(requestID string) {
	// Implementation would:
	// 1. Update status to processing
	// 2. Collect user data from all systems
	// 3. Format data according to requested format
	// 4. Encrypt export file
	// 5. Generate secure download URL
	// 6. Update request with download URL
	// 7. Send notification to user
}

// NewDataErasureService creates a data erasure service
func NewDataErasureService(db *sql.DB, storageManager *StorageManager, auditLogger AuditLogger) *DataErasureService {
	return &DataErasureService{
		db:               db,
		storageManager:   storageManager,
		dependencyMapper: NewDataDependencyMapper(),
		auditLogger:      auditLogger,
		legalRetention:   NewLegalRetentionService(),
	}
}

// ProcessErasureRequest handles right to be forgotten requests
func (des *DataErasureService) ProcessErasureRequest(ctx context.Context, userID string, erasureType ErasureType, reason ErasureReason, dataCategories []DataCategory) (*ErasureRequest, error) {
	requestID := uuid.New().String()
	timestamp := time.Now()
	
	request := &ErasureRequest{
		RequestID:      requestID,
		UserID:         userID,
		ErasureType:    erasureType,
		Reason:         reason,
		DataCategories: dataCategories,
		Status:         ErasureStatusPending,
		RequestedAt:    timestamp,
		Metadata:       make(map[string]interface{}),
	}
	
	// Check for legal retention requirements
	overrides, err := des.legalRetention.CheckRetentionRequirements(userID, dataCategories)
	if err != nil {
		return nil, fmt.Errorf("failed to check retention requirements: %w", err)
	}
	request.RetentionOverride = overrides
	
	// Store request in database
	query := `
		INSERT INTO data_erasure_requests (
			request_id, user_id, erasure_type, reason, data_categories,
			status, requested_at, retention_override, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	
	dataCategoriesJSON, _ := json.Marshal(dataCategories)
	retentionOverrideJSON, _ := json.Marshal(overrides)
	metadataJSON, _ := json.Marshal(request.Metadata)
	
	_, err = des.db.ExecContext(ctx, query,
		request.RequestID, request.UserID, request.ErasureType,
		request.Reason, dataCategoriesJSON, request.Status,
		request.RequestedAt, retentionOverrideJSON, metadataJSON,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to store erasure request: %w", err)
	}
	
	// Start verification process
	go des.processErasureAsync(requestID)
	
	// Audit log erasure request
	des.auditLogger.LogSecretModification(ctx, userID, "data_erasure_request",
		ActionDelete, ResultSuccess, map[string]interface{}{
			"request_id":      requestID,
			"erasure_type":    erasureType,
			"reason":          reason,
			"data_categories": dataCategories,
		})
	
	return request, nil
}

func (des *DataErasureService) processErasureAsync(requestID string) {
	// Implementation would:
	// 1. Verify user identity
	// 2. Check legal obligations
	// 3. Map data dependencies
	// 4. Execute erasure/anonymization
	// 5. Verify completion
	// 6. Update request status
	// 7. Generate compliance certificate
}

// Additional service implementations and helper types...

type DataFormatter interface {
	Format(data interface{}) ([]byte, error)
	GetMimeType() string
}

type StorageManager struct {
	// Storage management implementation
}

type DataDependencyMapper struct {
	// Data dependency mapping implementation
}

type LegalRetentionService struct {
	// Legal retention service implementation
}

type RetentionOverride struct {
	DataCategory  DataCategory  `json:"data_category"`
	LegalBasis    string        `json:"legal_basis"`
	RetainUntil   time.Time     `json:"retain_until"`
	Reason        string        `json:"reason"`
	ReviewDate    time.Time     `json:"review_date"`
}

type RetentionScheduler struct {
	// Automated retention processing
}

type AnonymizationQualityChecker struct {
	// Quality assurance for anonymization
}

type NotificationService struct {
	// User notification service
}

type RegulatoryNotificationService struct {
	// Regulatory authority notifications
}

type NotificationTemplate struct {
	Subject  string `json:"subject"`
	Body     string `json:"body"`
	Language string `json:"language"`
}

type BreachRiskAssessment struct {
	RiskLevel           string                 `json:"risk_level"`
	ImpactAssessment    string                 `json:"impact_assessment"`
	MitigationMeasures  []string              `json:"mitigation_measures"`
	UserNotificationRequired bool              `json:"user_notification_required"`
	RegulatoryDeadline  *time.Time            `json:"regulatory_deadline,omitempty"`
	Metadata           map[string]interface{} `json:"metadata"`
}

func NewDataDependencyMapper() *DataDependencyMapper {
	return &DataDependencyMapper{}
}

func NewLegalRetentionService() *LegalRetentionService {
	return &LegalRetentionService{}
}

func (lrs *LegalRetentionService) CheckRetentionRequirements(userID string, dataCategories []DataCategory) ([]RetentionOverride, error) {
	var overrides []RetentionOverride
	
	for _, category := range dataCategories {
		switch category {
		case DataFinancial:
			// Financial records must be retained for 7 years
			overrides = append(overrides, RetentionOverride{
				DataCategory: category,
				LegalBasis:   "Legal Obligation - Financial Regulations",
				RetainUntil:  time.Now().Add(7 * 365 * 24 * time.Hour),
				Reason:       "Tax and financial reporting requirements",
				ReviewDate:   time.Now().Add(6 * 365 * 24 * time.Hour),
			})
		}
	}
	
	return overrides, nil
}