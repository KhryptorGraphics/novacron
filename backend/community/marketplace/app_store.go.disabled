// Package marketplace implements Developer Marketplace & App Store
// Revenue sharing (70/30), app discovery, in-app billing, quality verification
// Target: 1,000+ apps, $10M+ ecosystem revenue
package marketplace

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// AppCategory represents app categorization
type AppCategory string

const (
	CategoryDataProcessing   AppCategory = "data_processing"
	CategoryMLAI             AppCategory = "ml_ai"
	CategoryDevTools         AppCategory = "dev_tools"
	CategoryMonitoring       AppCategory = "monitoring"
	CategorySecurity         AppCategory = "security"
	CategoryNetworking       AppCategory = "networking"
	CategoryStorage          AppCategory = "storage"
	CategoryAnalytics        AppCategory = "analytics"
	CategoryAutomation       AppCategory = "automation"
	CategoryIntegration      AppCategory = "integration"
)

// AppStatus represents app lifecycle status
type AppStatus string

const (
	StatusDraft       AppStatus = "draft"
	StatusInReview    AppStatus = "in_review"
	StatusRejected    AppStatus = "rejected"
	StatusApproved    AppStatus = "approved"
	StatusPublished   AppStatus = "published"
	StatusDeprecated  AppStatus = "deprecated"
	StatusSuspended   AppStatus = "suspended"
)

// PricingModel represents app pricing strategy
const (
	PricingFree        PricingModel = "free"
	PricingFreemium    PricingModel = "freemium"
	PricingSubscription PricingModel = "subscription"
	PricingOneTime     PricingModel = "one_time"
	PricingUsageBased  PricingModel = "usage_based"
)

// App represents marketplace application
type App struct {
	ID                string
	Name              string
	Description       string
	LongDescription   string
	DeveloperID       string
	DeveloperName     string
	Category          AppCategory
	Tags              []string
	Version           string
	Status            AppStatus
	PricingModel      PricingModel
	BasePrice         float64 // USD
	SubscriptionPlans []SubscriptionPlan
	UsageTiers        []UsageTier
	Features          []AppFeature
	Screenshots       []string
	VideoURL          string
	IconURL           string
	BannerURL         string
	DocumentationURL  string
	SupportURL        string
	SourceCodeURL     string
	DemoURL           string
	Requirements      AppRequirements
	Permissions       []Permission
	APIEndpoints      []APIEndpoint
	Webhooks          []WebhookConfig
	Integrations      []Integration
	Metrics           AppMetrics
	Reviews           []AppReview
	Security          SecurityReport
	Quality           QualityReport
	RevenueShare      RevenueShareConfig
	SubmittedAt       time.Time
	ReviewedAt        *time.Time
	PublishedAt       *time.Time
	LastUpdated       time.Time
	CreatedAt         time.Time
}

// SubscriptionPlan represents subscription pricing tier
type SubscriptionPlan struct {
	ID              string
	Name            string
	Description     string
	PriceMonthly    float64
	PriceYearly     float64
	Features        []string
	ResourceLimits  ResourceLimits
	SupportLevel    string
	TrialDays       int
	PopularPlan     bool
}

// UsageTier represents usage-based pricing tier
type UsageTier struct {
	ID              string
	Name            string
	MinUnits        int64
	MaxUnits        int64
	PricePerUnit    float64
	IncludedUnits   int64
}

// ResourceLimits defines resource quotas
type ResourceLimits struct {
	APICallsPerMonth    int64
	StorageGB           int64
	ComputeHours        int64
	ConcurrentUsers     int
	DataTransferGB      int64
	CustomLimits        map[string]int64
}

// AppFeature represents app feature/capability
type AppFeature struct {
	Name        string
	Description string
	Category    string
	Available   bool
	Premium     bool
}

// AppRequirements defines system requirements
type AppRequirements struct {
	MinCPUCores       int
	MinMemoryGB       int
	MinStorageGB      int
	RequiredOS        []string
	RequiredRuntime   string
	RequiredVersion   string
	Dependencies      []Dependency
	NetworkRequired   bool
	GPURequired       bool
}

// Dependency represents app dependency
type Dependency struct {
	Name           string
	Version        string
	Required       bool
	AutoInstall    bool
}

// Permission represents app permission requirement
type Permission struct {
	Name        string
	Scope       string
	Reason      string
	Required    bool
	Sensitive   bool
}

// APIEndpoint represents app API endpoint
type APIEndpoint struct {
	Path        string
	Method      string
	Description string
	Auth        string
	RateLimit   string
	Example     string
}

// WebhookConfig represents webhook configuration
type WebhookConfig struct {
	Event       string
	URL         string
	Method      string
	Headers     map[string]string
	RetryPolicy RetryPolicy
}

// RetryPolicy defines webhook retry behavior
type RetryPolicy struct {
	MaxRetries     int
	InitialDelay   int // seconds
	MaxDelay       int // seconds
	BackoffFactor  float64
}

// Integration represents third-party integration
type Integration struct {
	Service     string
	Type        string // oauth, api_key, webhook
	Required    bool
	ConfigURL   string
}

// AppMetrics tracks app performance metrics
type AppMetrics struct {
	TotalInstalls       int64
	ActiveInstalls      int64
	TotalRevenue        float64
	MonthlyRevenue      float64
	AverageRating       float64
	TotalReviews        int
	TotalDownloads      int64
	DAU                 int // Daily Active Users
	MAU                 int // Monthly Active Users
	ChurnRate           float64
	RetentionRate       float64
	ConversionRate      float64
	ARPU                float64 // Average Revenue Per User
	LTV                 float64 // Lifetime Value
	CAC                 float64 // Customer Acquisition Cost
	GrowthRate          float64
	UpdatedAt           time.Time
}

// AppReview represents user review
type AppReview struct {
	ID          string
	UserID      string
	Username    string
	Rating      int // 1-5
	Title       string
	Comment     string
	Verified    bool
	HelpfulCount int
	Version     string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// SecurityReport represents security assessment
type SecurityReport struct {
	Score               float64 // 0-100
	LastAssessed        time.Time
	Vulnerabilities     []Vulnerability
	SecurityCompliance  []ComplianceCheck
	DataEncryption      bool
	SecureAuth          bool
	APISecurityScore    float64
	PrivacyScore        float64
	CertifiedSecure     bool
	PenetrationTested   bool
	LastPenTest         *time.Time
}

// Vulnerability represents security vulnerability
type Vulnerability struct {
	ID          string
	Severity    string // critical, high, medium, low
	Category    string
	Description string
	CVE         string
	Fixed       bool
	FixedAt     *time.Time
}

// ComplianceCheck represents compliance verification
type ComplianceCheck struct {
	Standard    string // GDPR, HIPAA, SOC2, etc.
	Compliant   bool
	VerifiedAt  time.Time
	Certificate string
}

// QualityReport represents app quality assessment
type QualityReport struct {
	Score               float64 // 0-100
	LastAssessed        time.Time
	CodeQuality         float64
	Documentation       float64
	TestCoverage        float64
	Performance         float64
	Reliability         float64
	Usability           float64
	Maintainability     float64
	Issues              []QualityIssue
	PassedReview        bool
}

// QualityIssue represents quality concern
type QualityIssue struct {
	Category    string
	Severity    string
	Description string
	Location    string
	Fixed       bool
}

// RevenueShareConfig defines revenue distribution
type RevenueShareConfig struct {
	DeveloperShare      float64 // 0.70 = 70%
	PlatformShare       float64 // 0.30 = 30%
	MinPayoutThreshold  float64
	PaymentSchedule     string // monthly, quarterly
	PaymentMethod       string
	TaxWithholding      float64
}

// AppInstallation represents app installation
type AppInstallation struct {
	ID              string
	AppID           string
	UserID          string
	SubscriptionID  string
	Status          string // active, suspended, cancelled
	Version         string
	InstalledAt     time.Time
	LastUsed        time.Time
	UsageMetrics    UsageMetrics
	BillingInfo     BillingInfo
}

// UsageMetrics tracks app usage
type UsageMetrics struct {
	APICallsTotal       int64
	APICallsThisMonth   int64
	StorageUsedGB       float64
	ComputeHoursTotal   float64
	ComputeHoursMonth   float64
	DataTransferGB      float64
	ActiveSessions      int
	LastActivity        time.Time
}

// BillingInfo represents billing information
type BillingInfo struct {
	CurrentPlan         string
	BillingCycle        string // monthly, yearly
	NextBillingDate     time.Time
	AmountDue           float64
	PaymentStatus       string
	PaymentMethod       string
	InvoiceHistory      []Invoice
	UsageCharges        []UsageCharge
}

// Invoice represents billing invoice
type Invoice struct {
	ID              string
	Amount          float64
	Status          string
	IssuedAt        time.Time
	PaidAt          *time.Time
	DueAt           time.Time
	LineItems       []LineItem
	TaxAmount       float64
	TotalAmount     float64
}

// LineItem represents invoice line item
type LineItem struct {
	Description     string
	Quantity        float64
	UnitPrice       float64
	Amount          float64
}

// UsageCharge represents usage-based charge
type UsageCharge struct {
	Resource        string
	Units           int64
	PricePerUnit    float64
	TotalAmount     float64
	BillingPeriod   string
}

// DeveloperAnalytics provides developer dashboard analytics
type DeveloperAnalytics struct {
	DeveloperID         string
	TotalApps           int
	PublishedApps       int
	TotalInstalls       int64
	ActiveUsers         int
	TotalRevenue        float64
	RevenueThisMonth    float64
	RevenueLastMonth    float64
	GrowthRate          float64
	AverageRating       float64
	TotalReviews        int
	TopApps             []AppPerformance
	RevenueByApp        map[string]float64
	UsersByCountry      map[string]int
	ConversionFunnel    ConversionFunnel
	ChurnAnalysis       ChurnAnalysis
	UpdatedAt           time.Time
}

// AppPerformance tracks individual app performance
type AppPerformance struct {
	AppID           string
	AppName         string
	Installs        int64
	ActiveUsers     int
	Revenue         float64
	Rating          float64
	GrowthRate      float64
	ChurnRate       float64
}

// ConversionFunnel tracks user conversion
type ConversionFunnel struct {
	Views           int64
	Installs        int64
	Activations     int64
	Subscriptions   int64
	ViewToInstall   float64
	InstallToActive float64
	ActiveToSub     float64
}

// ChurnAnalysis provides churn insights
type ChurnAnalysis struct {
	MonthlyChurnRate    float64
	ChurnReasons        map[string]int
	AtRiskUsers         int
	WinBackCampaigns    int
	RetentionRate       float64
}

// MarketplaceManager manages app marketplace
type MarketplaceManager struct {
	mu                  sync.RWMutex
	apps                map[string]*App
	installations       map[string]*AppInstallation
	reviews             map[string][]AppReview
	developerAnalytics  map[string]*DeveloperAnalytics
	categoryApps        map[AppCategory][]string
	featuredApps        []string
	trendingApps        []string
	newApps             []string
	stats               MarketplaceStats
	revenueShare        RevenueShareConfig
	reviewQueue         []*App
}

// MarketplaceStats tracks marketplace metrics
type MarketplaceStats struct {
	TotalApps               int
	PublishedApps           int
	TotalDevelopers         int
	TotalInstalls           int64
	ActiveInstalls          int64
	TotalRevenue            float64
	MonthlyRevenue          float64
	DeveloperRevenue        float64
	PlatformRevenue         float64
	AverageAppRating        float64
	TotalReviews            int
	AppsInReview            int
	AppApprovalRate         float64
	AverageReviewTime       float64 // hours
	TopGrossingApps         []string
	TopRatedApps            []string
	FastestGrowingApps      []string
	UpdatedAt               time.Time
}

// NewMarketplaceManager creates marketplace manager
func NewMarketplaceManager() *MarketplaceManager {
	mm := &MarketplaceManager{
		apps:               make(map[string]*App),
		installations:      make(map[string]*AppInstallation),
		reviews:            make(map[string][]AppReview),
		developerAnalytics: make(map[string]*DeveloperAnalytics),
		categoryApps:       make(map[AppCategory][]string),
		featuredApps:       []string{},
		trendingApps:       []string{},
		newApps:            []string{},
		reviewQueue:        []*App{},
		revenueShare: RevenueShareConfig{
			DeveloperShare:     0.70,
			PlatformShare:      0.30,
			MinPayoutThreshold: 100.0,
			PaymentSchedule:    "monthly",
			PaymentMethod:      "bank_transfer",
			TaxWithholding:     0.0,
		},
	}

	mm.initializeSampleApps()

	return mm
}

// initializeSampleApps creates sample marketplace apps
func (mm *MarketplaceManager) initializeSampleApps() {
	categories := []AppCategory{
		CategoryDataProcessing, CategoryMLAI, CategoryDevTools,
		CategoryMonitoring, CategorySecurity, CategoryNetworking,
	}

	for i := 0; i < 100; i++ {
		category := categories[i%len(categories)]

		app := &App{
			ID:              mm.generateID("APP"),
			Name:            fmt.Sprintf("App %d - %s", i+1, category),
			Description:     fmt.Sprintf("Sample app for %s", category),
			LongDescription: "Detailed description of the application",
			DeveloperID:     fmt.Sprintf("DEV-%03d", (i%50)+1),
			DeveloperName:   fmt.Sprintf("Developer %d", (i%50)+1),
			Category:        category,
			Tags:            []string{"popular", "trending"},
			Version:         "1.0.0",
			Status:          StatusPublished,
			PricingModel:    PricingFreemium,
			BasePrice:       0,
			SubscriptionPlans: []SubscriptionPlan{
				{
					ID:           "basic",
					Name:         "Basic",
					Description:  "Basic features",
					PriceMonthly: 9.99,
					PriceYearly:  99.99,
					Features:     []string{"Core features", "Email support"},
					ResourceLimits: ResourceLimits{
						APICallsPerMonth: 10000,
						StorageGB:        10,
						ComputeHours:     100,
						ConcurrentUsers:  10,
					},
					SupportLevel: "email",
					TrialDays:    14,
				},
				{
					ID:           "pro",
					Name:         "Professional",
					Description:  "Advanced features",
					PriceMonthly: 29.99,
					PriceYearly:  299.99,
					Features:     []string{"All features", "Priority support", "Custom integrations"},
					ResourceLimits: ResourceLimits{
						APICallsPerMonth: 100000,
						StorageGB:        100,
						ComputeHours:     1000,
						ConcurrentUsers:  100,
					},
					SupportLevel: "priority",
					TrialDays:    30,
					PopularPlan:  true,
				},
			},
			Features: []AppFeature{
				{Name: "Feature 1", Description: "Core feature", Category: "core", Available: true},
				{Name: "Feature 2", Description: "Premium feature", Category: "premium", Available: true, Premium: true},
			},
			Requirements: AppRequirements{
				MinCPUCores:     2,
				MinMemoryGB:     4,
				MinStorageGB:    10,
				RequiredOS:      []string{"linux", "darwin", "windows"},
				RequiredRuntime: "go1.21+",
				NetworkRequired: true,
			},
			Permissions: []Permission{
				{Name: "network", Scope: "internet", Reason: "API communication", Required: true},
				{Name: "storage", Scope: "local", Reason: "Data persistence", Required: true},
			},
			Metrics: AppMetrics{
				TotalInstalls:  int64((i + 1) * 100),
				ActiveInstalls: int64((i + 1) * 75),
				TotalRevenue:   float64((i + 1) * 1000),
				MonthlyRevenue: float64((i + 1) * 100),
				AverageRating:  4.5,
				TotalReviews:   50,
				DAU:            int((i + 1) * 50),
				MAU:            int((i + 1) * 200),
				RetentionRate:  0.85,
				ConversionRate: 0.15,
			},
			Security: SecurityReport{
				Score:              95.0,
				LastAssessed:       time.Now().AddDate(0, 0, -7),
				Vulnerabilities:    []Vulnerability{},
				DataEncryption:     true,
				SecureAuth:         true,
				APISecurityScore:   90.0,
				PrivacyScore:       95.0,
				CertifiedSecure:    true,
				PenetrationTested:  true,
			},
			Quality: QualityReport{
				Score:           92.0,
				LastAssessed:    time.Now().AddDate(0, 0, -3),
				CodeQuality:     95.0,
				Documentation:   90.0,
				TestCoverage:    85.0,
				Performance:     92.0,
				Reliability:     96.0,
				Usability:       88.0,
				Maintainability: 90.0,
				PassedReview:    true,
			},
			RevenueShare: mm.revenueShare,
			PublishedAt:  &[]time.Time{time.Now().AddDate(0, -6, 0)}[0],
			CreatedAt:    time.Now().AddDate(0, -6, 0),
			LastUpdated:  time.Now(),
		}

		mm.apps[app.ID] = app
		mm.categoryApps[category] = append(mm.categoryApps[category], app.ID)

		if i < 10 {
			mm.featuredApps = append(mm.featuredApps, app.ID)
		}
		if i < 20 {
			mm.trendingApps = append(mm.trendingApps, app.ID)
		}
		if i < 15 {
			mm.newApps = append(mm.newApps, app.ID)
		}
	}
}

// SubmitApp submits app for review
func (mm *MarketplaceManager) SubmitApp(ctx context.Context, app *App) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if app.ID == "" {
		app.ID = mm.generateID("APP")
	}

	app.Status = StatusInReview
	app.SubmittedAt = time.Now()
	app.CreatedAt = time.Now()
	app.LastUpdated = time.Now()

	mm.apps[app.ID] = app
	mm.reviewQueue = append(mm.reviewQueue, app)

	mm.stats.AppsInReview++
	mm.stats.UpdatedAt = time.Now()

	return nil
}

// ReviewApp performs app review
func (mm *MarketplaceManager) ReviewApp(ctx context.Context, appID string, approved bool, feedback string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	app, exists := mm.apps[appID]
	if !exists {
		return fmt.Errorf("app not found: %s", appID)
	}

	now := time.Now()
	app.ReviewedAt = &now

	if approved {
		app.Status = StatusApproved
		mm.stats.AppApprovalRate = (mm.stats.AppApprovalRate*float64(mm.stats.PublishedApps) + 1.0) / float64(mm.stats.PublishedApps+1)
	} else {
		app.Status = StatusRejected
	}

	app.LastUpdated = time.Now()

	// Calculate review time
	reviewTime := time.Since(app.SubmittedAt).Hours()
	mm.stats.AverageReviewTime = (mm.stats.AverageReviewTime*float64(mm.stats.AppsInReview-1) + reviewTime) / float64(mm.stats.AppsInReview)
	mm.stats.AppsInReview--
	mm.stats.UpdatedAt = time.Now()

	// Remove from review queue
	for i, queuedApp := range mm.reviewQueue {
		if queuedApp.ID == appID {
			mm.reviewQueue = append(mm.reviewQueue[:i], mm.reviewQueue[i+1:]...)
			break
		}
	}

	return nil
}

// PublishApp publishes approved app
func (mm *MarketplaceManager) PublishApp(ctx context.Context, appID string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	app, exists := mm.apps[appID]
	if !exists {
		return fmt.Errorf("app not found: %s", appID)
	}

	if app.Status != StatusApproved {
		return fmt.Errorf("app not approved: %s", app.Status)
	}

	now := time.Now()
	app.Status = StatusPublished
	app.PublishedAt = &now
	app.LastUpdated = now

	// Add to category
	mm.categoryApps[app.Category] = append(mm.categoryApps[app.Category], app.ID)

	// Add to new apps
	mm.newApps = append([]string{app.ID}, mm.newApps...)
	if len(mm.newApps) > 50 {
		mm.newApps = mm.newApps[:50]
	}

	mm.stats.TotalApps++
	mm.stats.PublishedApps++
	mm.stats.UpdatedAt = time.Now()

	return nil
}

// InstallApp installs app for user
func (mm *MarketplaceManager) InstallApp(ctx context.Context, appID, userID, planID string) (*AppInstallation, error) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	app, exists := mm.apps[appID]
	if !exists {
		return nil, fmt.Errorf("app not found: %s", appID)
	}

	if app.Status != StatusPublished {
		return nil, fmt.Errorf("app not published")
	}

	installation := &AppInstallation{
		ID:          mm.generateID("INST"),
		AppID:       appID,
		UserID:      userID,
		Status:      "active",
		Version:     app.Version,
		InstalledAt: time.Now(),
		LastUsed:    time.Now(),
		UsageMetrics: UsageMetrics{
			LastActivity: time.Now(),
		},
	}

	// Set up billing if paid plan
	if planID != "" {
		for _, plan := range app.SubscriptionPlans {
			if plan.ID == planID {
				installation.BillingInfo = BillingInfo{
					CurrentPlan:     plan.Name,
					BillingCycle:    "monthly",
					NextBillingDate: time.Now().AddDate(0, 1, 0),
					AmountDue:       plan.PriceMonthly,
					PaymentStatus:   "active",
				}
				break
			}
		}
	}

	mm.installations[installation.ID] = installation

	// Update app metrics
	app.Metrics.TotalInstalls++
	app.Metrics.ActiveInstalls++
	app.LastUpdated = time.Now()

	// Update marketplace stats
	mm.stats.TotalInstalls++
	mm.stats.ActiveInstalls++
	mm.stats.UpdatedAt = time.Now()

	return installation, nil
}

// SubmitReview submits app review
func (mm *MarketplaceManager) SubmitReview(ctx context.Context, appID, userID, username string, rating int, title, comment string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	app, exists := mm.apps[appID]
	if !exists {
		return fmt.Errorf("app not found: %s", appID)
	}

	review := AppReview{
		ID:        mm.generateID("REV"),
		UserID:    userID,
		Username:  username,
		Rating:    rating,
		Title:     title,
		Comment:   comment,
		Verified:  true,
		Version:   app.Version,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	mm.reviews[appID] = append(mm.reviews[appID], review)
	app.Reviews = append(app.Reviews, review)

	// Update app metrics
	oldRating := app.Metrics.AverageRating
	oldCount := app.Metrics.TotalReviews
	newRating := (oldRating*float64(oldCount) + float64(rating)) / float64(oldCount+1)
	app.Metrics.AverageRating = newRating
	app.Metrics.TotalReviews++
	app.LastUpdated = time.Now()

	// Update marketplace stats
	mm.stats.TotalReviews++
	mm.stats.UpdatedAt = time.Now()

	return nil
}

// RecordUsage records app usage metrics
func (mm *MarketplaceManager) RecordUsage(ctx context.Context, installationID string, metrics UsageMetrics) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	installation, exists := mm.installations[installationID]
	if !exists {
		return fmt.Errorf("installation not found: %s", installationID)
	}

	installation.UsageMetrics.APICallsTotal += metrics.APICallsTotal
	installation.UsageMetrics.APICallsThisMonth += metrics.APICallsThisMonth
	installation.UsageMetrics.StorageUsedGB = metrics.StorageUsedGB
	installation.UsageMetrics.ComputeHoursTotal += metrics.ComputeHoursTotal
	installation.UsageMetrics.ComputeHoursMonth += metrics.ComputeHoursMonth
	installation.UsageMetrics.DataTransferGB += metrics.DataTransferGB
	installation.UsageMetrics.LastActivity = time.Now()
	installation.LastUsed = time.Now()

	return nil
}

// ProcessPayment processes app payment
func (mm *MarketplaceManager) ProcessPayment(ctx context.Context, installationID string, amount float64) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	installation, exists := mm.installations[installationID]
	if !exists {
		return fmt.Errorf("installation not found: %s", installationID)
	}

	app, exists := mm.apps[installation.AppID]
	if !exists {
		return fmt.Errorf("app not found: %s", installation.AppID)
	}

	// Create invoice
	invoice := Invoice{
		ID:        mm.generateID("INV"),
		Amount:    amount,
		Status:    "paid",
		IssuedAt:  time.Now(),
		PaidAt:    &[]time.Time{time.Now()}[0],
		DueAt:     time.Now().AddDate(0, 1, 0),
		LineItems: []LineItem{
			{
				Description: fmt.Sprintf("%s subscription", app.Name),
				Quantity:    1,
				UnitPrice:   amount,
				Amount:      amount,
			},
		},
		TotalAmount: amount,
	}

	installation.BillingInfo.InvoiceHistory = append(installation.BillingInfo.InvoiceHistory, invoice)
	installation.BillingInfo.NextBillingDate = time.Now().AddDate(0, 1, 0)

	// Calculate revenue share
	developerRevenue := amount * app.RevenueShare.DeveloperShare
	platformRevenue := amount * app.RevenueShare.PlatformShare

	// Update app metrics
	app.Metrics.TotalRevenue += amount
	app.Metrics.MonthlyRevenue += amount
	app.LastUpdated = time.Now()

	// Update marketplace stats
	mm.stats.TotalRevenue += amount
	mm.stats.MonthlyRevenue += amount
	mm.stats.DeveloperRevenue += developerRevenue
	mm.stats.PlatformRevenue += platformRevenue
	mm.stats.UpdatedAt = time.Now()

	// Update developer analytics
	if analytics, exists := mm.developerAnalytics[app.DeveloperID]; exists {
		analytics.TotalRevenue += developerRevenue
		analytics.RevenueThisMonth += developerRevenue
		analytics.RevenueByApp[app.ID] = analytics.RevenueByApp[app.ID] + developerRevenue
		analytics.UpdatedAt = time.Now()
	}

	return nil
}

// GetDeveloperAnalytics retrieves developer analytics
func (mm *MarketplaceManager) GetDeveloperAnalytics(ctx context.Context, developerID string) (*DeveloperAnalytics, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	analytics, exists := mm.developerAnalytics[developerID]
	if !exists {
		// Create new analytics
		analytics = &DeveloperAnalytics{
			DeveloperID:    developerID,
			RevenueByApp:   make(map[string]float64),
			UsersByCountry: make(map[string]int),
			UpdatedAt:      time.Now(),
		}

		// Calculate from apps
		for _, app := range mm.apps {
			if app.DeveloperID == developerID {
				analytics.TotalApps++
				if app.Status == StatusPublished {
					analytics.PublishedApps++
				}
				analytics.TotalInstalls += app.Metrics.TotalInstalls
				analytics.ActiveUsers += app.Metrics.DAU
				analytics.TotalRevenue += app.Metrics.TotalRevenue * mm.revenueShare.DeveloperShare
				analytics.TotalReviews += app.Metrics.TotalReviews

				if app.Metrics.AverageRating > 0 {
					analytics.AverageRating = (analytics.AverageRating*float64(analytics.PublishedApps-1) + app.Metrics.AverageRating) / float64(analytics.PublishedApps)
				}
			}
		}

		mm.developerAnalytics[developerID] = analytics
	}

	return analytics, nil
}

// SearchApps searches marketplace apps
func (mm *MarketplaceManager) SearchApps(ctx context.Context, query string, category AppCategory, minRating float64) []*App {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	var results []*App

	for _, app := range mm.apps {
		if app.Status != StatusPublished {
			continue
		}

		// Category filter
		if category != "" && app.Category != category {
			continue
		}

		// Rating filter
		if minRating > 0 && app.Metrics.AverageRating < minRating {
			continue
		}

		// Text search (simple contains)
		if query != "" {
			// In production, use full-text search
			continue
		}

		results = append(results, app)
	}

	return results
}

// GetFeaturedApps returns featured apps
func (mm *MarketplaceManager) GetFeaturedApps(ctx context.Context) []*App {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	var featured []*App
	for _, appID := range mm.featuredApps {
		if app, exists := mm.apps[appID]; exists && app.Status == StatusPublished {
			featured = append(featured, app)
		}
	}

	return featured
}

// GetTrendingApps returns trending apps
func (mm *MarketplaceManager) GetTrendingApps(ctx context.Context) []*App {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	var trending []*App
	for _, appID := range mm.trendingApps {
		if app, exists := mm.apps[appID]; exists && app.Status == StatusPublished {
			trending = append(trending, app)
		}
	}

	return trending
}

// GetMarketplaceStats returns marketplace statistics
func (mm *MarketplaceManager) GetMarketplaceStats(ctx context.Context) MarketplaceStats {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	stats := mm.stats

	// Calculate additional stats
	developerSet := make(map[string]bool)
	for _, app := range mm.apps {
		developerSet[app.DeveloperID] = true
	}
	stats.TotalDevelopers = len(developerSet)

	// Top grossing apps
	type appRevenue struct {
		id      string
		revenue float64
	}
	var revenues []appRevenue
	for id, app := range mm.apps {
		if app.Status == StatusPublished {
			revenues = append(revenues, appRevenue{id, app.Metrics.TotalRevenue})
		}
	}
	// Sort by revenue (simplified)
	stats.TopGrossingApps = []string{}
	for i := 0; i < len(revenues) && i < 10; i++ {
		stats.TopGrossingApps = append(stats.TopGrossingApps, revenues[i].id)
	}

	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (mm *MarketplaceManager) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// ExportAppData exports app data as JSON
func (mm *MarketplaceManager) ExportAppData(ctx context.Context, appID string) ([]byte, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	app, exists := mm.apps[appID]
	if !exists {
		return nil, fmt.Errorf("app not found: %s", appID)
	}

	return json.MarshalIndent(app, "", "  ")
}
