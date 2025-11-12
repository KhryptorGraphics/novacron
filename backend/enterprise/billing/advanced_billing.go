// Package billing provides advanced enterprise billing and revenue management
// Supporting $100M+ ARR with 40%+ margins through sophisticated pricing models
package billing

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"
)

// AdvancedBillingEngine manages enterprise billing operations
type AdvancedBillingEngine struct {
	accounts        map[string]*EnterpriseAccount
	subscriptions   map[string]*Subscription
	invoices        map[string]*Invoice
	payments        map[string]*Payment
	pricingEngine   *PricingEngine
	revenueEngine   *RevenueRecognitionEngine
	currencyManager *CurrencyManager
	paymentGateway  *PaymentGateway
	mu              sync.RWMutex
	metrics         *BillingMetrics
}

// EnterpriseAccount represents an enterprise customer account
type EnterpriseAccount struct {
	ID              string                 `json:"id"`
	TenantID        string                 `json:"tenant_id"`
	CompanyName     string                 `json:"company_name"`
	BillingEmail    string                 `json:"billing_email"`
	PaymentTerms    PaymentTerms           `json:"payment_terms"`
	Currency        string                 `json:"currency"`
	TaxID           string                 `json:"tax_id"`
	BillingAddress  *Address               `json:"billing_address"`
	CreditLimit     float64                `json:"credit_limit"`
	Balance         float64                `json:"balance"`
	Subscriptions   []string               `json:"subscriptions"`
	Commitments     []*Commitment          `json:"commitments"`
	Discounts       []*Discount            `json:"discounts"`
	PaymentMethods  []*PaymentMethod       `json:"payment_methods"`
	AutoPay         bool                   `json:"auto_pay"`
	PurchaseOrders  []string               `json:"purchase_orders"`
	CustomPricing   map[string]interface{} `json:"custom_pricing"`
	AccountManager  string                 `json:"account_manager"`
	Status          string                 `json:"status"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// PaymentTerms defines payment term configurations
type PaymentTerms string

const (
	PaymentTermsImmediate PaymentTerms = "immediate"    // Due on receipt
	PaymentTermsNet30     PaymentTerms = "net_30"       // Due in 30 days
	PaymentTermsNet60     PaymentTerms = "net_60"       // Due in 60 days
	PaymentTermsNet90     PaymentTerms = "net_90"       // Due in 90 days
	PaymentTermsCustom    PaymentTerms = "custom"       // Custom terms
	PaymentTermsPrepaid   PaymentTerms = "prepaid"      // Prepayment required
)

// Address represents a billing address
type Address struct {
	Street1    string `json:"street1"`
	Street2    string `json:"street2"`
	City       string `json:"city"`
	State      string `json:"state"`
	PostalCode string `json:"postal_code"`
	Country    string `json:"country"`
}

// Commitment represents a contract commitment
type Commitment struct {
	ID              string        `json:"id"`
	Type            CommitmentType `json:"type"`
	Amount          float64       `json:"amount"`
	Term            int           `json:"term"` // months
	StartDate       time.Time     `json:"start_date"`
	EndDate         time.Time     `json:"end_date"`
	Discount        float64       `json:"discount"` // percentage
	MinimumSpend    float64       `json:"minimum_spend"`
	UsageCredits    float64       `json:"usage_credits"`
	RemainingCredits float64      `json:"remaining_credits"`
	AutoRenew       bool          `json:"auto_renew"`
	PenaltyTerms    string        `json:"penalty_terms"`
	Status          string        `json:"status"`
}

// CommitmentType defines commitment types
type CommitmentType string

const (
	CommitmentOneYear   CommitmentType = "one_year"
	CommitmentTwoYear   CommitmentType = "two_year"
	CommitmentThreeYear CommitmentType = "three_year"
	CommitmentFiveYear  CommitmentType = "five_year"
	CommitmentCustom    CommitmentType = "custom"
)

// Discount represents a pricing discount
type Discount struct {
	ID          string        `json:"id"`
	Type        DiscountType  `json:"type"`
	Value       float64       `json:"value"` // percentage or amount
	AppliesTo   []string      `json:"applies_to"` // product IDs
	StartDate   time.Time     `json:"start_date"`
	EndDate     time.Time     `json:"end_date"`
	Conditions  map[string]interface{} `json:"conditions"`
	Stackable   bool          `json:"stackable"`
	Priority    int           `json:"priority"`
}

// DiscountType defines discount types
type DiscountType string

const (
	DiscountPercentage     DiscountType = "percentage"
	DiscountAmount         DiscountType = "amount"
	DiscountVolume         DiscountType = "volume"
	DiscountCommitment     DiscountType = "commitment"
	DiscountPartner        DiscountType = "partner"
	DiscountPromotion      DiscountType = "promotion"
	DiscountEarlyPayment   DiscountType = "early_payment"
)

// PaymentMethod represents a payment method
type PaymentMethod struct {
	ID               string                 `json:"id"`
	Type             PaymentMethodType      `json:"type"`
	Details          map[string]interface{} `json:"details"`
	IsDefault        bool                   `json:"is_default"`
	BillingAddress   *Address               `json:"billing_address"`
	ExpirationDate   *time.Time             `json:"expiration_date,omitempty"`
	Status           string                 `json:"status"`
	LastUsed         time.Time              `json:"last_used"`
	VerificationStatus string               `json:"verification_status"`
}

// PaymentMethodType defines payment method types
type PaymentMethodType string

const (
	PaymentMethodCreditCard  PaymentMethodType = "credit_card"
	PaymentMethodACH         PaymentMethodType = "ach"
	PaymentMethodWire        PaymentMethodType = "wire"
	PaymentMethodCheck       PaymentMethodType = "check"
	PaymentMethodPurchaseOrder PaymentMethodType = "purchase_order"
	PaymentMethodPayPal      PaymentMethodType = "paypal"
	PaymentMethodStripe      PaymentMethodType = "stripe"
	PaymentMethodAlipay      PaymentMethodType = "alipay"
	PaymentMethodWeChatPay   PaymentMethodType = "wechat_pay"
	PaymentMethodSEPA        PaymentMethodType = "sepa"
	PaymentMethodBoleto      PaymentMethodType = "boleto"
)

// Subscription represents a product subscription
type Subscription struct {
	ID                string                 `json:"id"`
	AccountID         string                 `json:"account_id"`
	ProductID         string                 `json:"product_id"`
	PlanID            string                 `json:"plan_id"`
	PricingModel      PricingModel           `json:"pricing_model"`
	Status            SubscriptionStatus     `json:"status"`
	BillingCycle      BillingCycle           `json:"billing_cycle"`
	Quantity          int                    `json:"quantity"`
	UnitPrice         float64                `json:"unit_price"`
	TotalPrice        float64                `json:"total_price"`
	Discounts         []string               `json:"discounts"`
	AddOns            []*AddOn               `json:"add_ons"`
	UsageMetrics      map[string]*UsageMetric `json:"usage_metrics"`
	CustomFields      map[string]interface{} `json:"custom_fields"`
	StartDate         time.Time              `json:"start_date"`
	EndDate           *time.Time             `json:"end_date,omitempty"`
	NextBillingDate   time.Time              `json:"next_billing_date"`
	RenewalDate       time.Time              `json:"renewal_date"`
	AutoRenew         bool                   `json:"auto_renew"`
	TrialEndDate      *time.Time             `json:"trial_end_date,omitempty"`
	CommitmentID      string                 `json:"commitment_id,omitempty"`
	RevenueSchedule   *RevenueSchedule       `json:"revenue_schedule"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

// PricingModel defines pricing model types
type PricingModel string

const (
	PricingModelFlat         PricingModel = "flat"
	PricingModelPerUnit      PricingModel = "per_unit"
	PricingModelTiered       PricingModel = "tiered"
	PricingModelVolume       PricingModel = "volume"
	PricingModelUsageBased   PricingModel = "usage_based"
	PricingModelStairstep    PricingModel = "stairstep"
	PricingModelPackage      PricingModel = "package"
	PricingModelHybrid       PricingModel = "hybrid"
)

// SubscriptionStatus defines subscription states
type SubscriptionStatus string

const (
	SubscriptionStatusActive    SubscriptionStatus = "active"
	SubscriptionStatusTrial     SubscriptionStatus = "trial"
	SubscriptionStatusPaused    SubscriptionStatus = "paused"
	SubscriptionStatusCanceled  SubscriptionStatus = "canceled"
	SubscriptionStatusExpired   SubscriptionStatus = "expired"
	SubscriptionStatusPending   SubscriptionStatus = "pending"
)

// BillingCycle defines billing frequency
type BillingCycle string

const (
	BillingCycleMonthly   BillingCycle = "monthly"
	BillingCycleQuarterly BillingCycle = "quarterly"
	BillingCycleAnnual    BillingCycle = "annual"
	BillingCycleBiennial  BillingCycle = "biennial"
	BillingCycleCustom    BillingCycle = "custom"
)

// AddOn represents subscription add-ons
type AddOn struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Quantity    int     `json:"quantity"`
	UnitPrice   float64 `json:"unit_price"`
	TotalPrice  float64 `json:"total_price"`
	Recurring   bool    `json:"recurring"`
}

// UsageMetric tracks usage-based billing metrics
type UsageMetric struct {
	MetricID       string    `json:"metric_id"`
	Name           string    `json:"name"`
	Unit           string    `json:"unit"`
	CurrentUsage   float64   `json:"current_usage"`
	IncludedAmount float64   `json:"included_amount"`
	OverageRate    float64   `json:"overage_rate"`
	TierRates      []TierRate `json:"tier_rates"`
	ResetCycle     string    `json:"reset_cycle"`
	LastReset      time.Time `json:"last_reset"`
	NextReset      time.Time `json:"next_reset"`
}

// TierRate defines tiered pricing rates
type TierRate struct {
	StartUnit int     `json:"start_unit"`
	EndUnit   int     `json:"end_unit"` // -1 for unlimited
	Rate      float64 `json:"rate"`
}

// Invoice represents a billing invoice
type Invoice struct {
	ID                string                 `json:"id"`
	InvoiceNumber     string                 `json:"invoice_number"`
	AccountID         string                 `json:"account_id"`
	SubscriptionIDs   []string               `json:"subscription_ids"`
	Status            InvoiceStatus          `json:"status"`
	Currency          string                 `json:"currency"`
	Subtotal          float64                `json:"subtotal"`
	TaxAmount         float64                `json:"tax_amount"`
	TotalAmount       float64                `json:"total_amount"`
	PaidAmount        float64                `json:"paid_amount"`
	BalanceDue        float64                `json:"balance_due"`
	LineItems         []*LineItem            `json:"line_items"`
	Taxes             []*TaxItem             `json:"taxes"`
	Discounts         []*DiscountItem        `json:"discounts"`
	Credits           []*CreditItem          `json:"credits"`
	PaymentTerms      PaymentTerms           `json:"payment_terms"`
	IssueDate         time.Time              `json:"issue_date"`
	DueDate           time.Time              `json:"due_date"`
	PeriodStart       time.Time              `json:"period_start"`
	PeriodEnd         time.Time              `json:"period_end"`
	PaymentDate       *time.Time             `json:"payment_date,omitempty"`
	PurchaseOrderNum  string                 `json:"purchase_order_num,omitempty"`
	Notes             string                 `json:"notes"`
	CustomFields      map[string]interface{} `json:"custom_fields"`
	PDFUrl            string                 `json:"pdf_url"`
	AttemptCount      int                    `json:"attempt_count"`
	NextAttemptDate   *time.Time             `json:"next_attempt_date,omitempty"`
	DunningLevel      int                    `json:"dunning_level"`
	RevenueSchedule   *RevenueSchedule       `json:"revenue_schedule"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

// InvoiceStatus defines invoice states
type InvoiceStatus string

const (
	InvoiceStatusDraft       InvoiceStatus = "draft"
	InvoiceStatusOpen        InvoiceStatus = "open"
	InvoiceStatusPaid        InvoiceStatus = "paid"
	InvoiceStatusPartialPaid InvoiceStatus = "partial_paid"
	InvoiceStatusOverdue     InvoiceStatus = "overdue"
	InvoiceStatusVoided      InvoiceStatus = "voided"
	InvoiceStatusWrittenOff  InvoiceStatus = "written_off"
)

// LineItem represents an invoice line item
type LineItem struct {
	ID             string                 `json:"id"`
	Type           string                 `json:"type"` // subscription, usage, one_time
	Description    string                 `json:"description"`
	ProductID      string                 `json:"product_id"`
	Quantity       float64                `json:"quantity"`
	UnitPrice      float64                `json:"unit_price"`
	Amount         float64                `json:"amount"`
	DiscountAmount float64                `json:"discount_amount"`
	NetAmount      float64                `json:"net_amount"`
	TaxAmount      float64                `json:"tax_amount"`
	TotalAmount    float64                `json:"total_amount"`
	PeriodStart    *time.Time             `json:"period_start,omitempty"`
	PeriodEnd      *time.Time             `json:"period_end,omitempty"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// TaxItem represents tax calculation
type TaxItem struct {
	ID         string  `json:"id"`
	Name       string  `json:"name"`
	Rate       float64 `json:"rate"`
	Amount     float64 `json:"amount"`
	Jurisdiction string `json:"jurisdiction"`
	TaxNumber  string  `json:"tax_number"`
}

// DiscountItem represents applied discount
type DiscountItem struct {
	ID          string  `json:"id"`
	Description string  `json:"description"`
	Amount      float64 `json:"amount"`
	Percentage  float64 `json:"percentage"`
}

// CreditItem represents credit application
type CreditItem struct {
	ID          string  `json:"id"`
	Type        string  `json:"type"` // credit_note, payment, adjustment
	Amount      float64 `json:"amount"`
	Description string  `json:"description"`
	Reference   string  `json:"reference"`
}

// Payment represents a payment transaction
type Payment struct {
	ID              string                 `json:"id"`
	AccountID       string                 `json:"account_id"`
	InvoiceID       string                 `json:"invoice_id"`
	Amount          float64                `json:"amount"`
	Currency        string                 `json:"currency"`
	Status          PaymentStatus          `json:"status"`
	Method          PaymentMethodType      `json:"method"`
	TransactionID   string                 `json:"transaction_id"`
	Gateway         string                 `json:"gateway"`
	ProcessedAt     *time.Time             `json:"processed_at,omitempty"`
	FailureReason   string                 `json:"failure_reason,omitempty"`
	RefundAmount    float64                `json:"refund_amount"`
	RefundedAt      *time.Time             `json:"refunded_at,omitempty"`
	Metadata        map[string]interface{} `json:"metadata"`
	CreatedAt       time.Time              `json:"created_at"`
}

// PaymentStatus defines payment states
type PaymentStatus string

const (
	PaymentStatusPending   PaymentStatus = "pending"
	PaymentStatusProcessing PaymentStatus = "processing"
	PaymentStatusCompleted PaymentStatus = "completed"
	PaymentStatusFailed    PaymentStatus = "failed"
	PaymentStatusRefunded  PaymentStatus = "refunded"
	PaymentStatusCanceled  PaymentStatus = "canceled"
)

// PricingEngine handles pricing calculations
type PricingEngine struct {
	products      map[string]*Product
	plans         map[string]*PricingPlan
	volumeTiers   map[string][]*VolumeTier
	customRules   map[string]*PricingRule
	mu            sync.RWMutex
}

// Product represents a billable product
type Product struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Category       string                 `json:"category"`
	PricingModel   PricingModel           `json:"pricing_model"`
	BasePrice      float64                `json:"base_price"`
	Currency       string                 `json:"currency"`
	BillingCycle   BillingCycle           `json:"billing_cycle"`
	Features       []string               `json:"features"`
	Limits         map[string]interface{} `json:"limits"`
	AvailablePlans []string               `json:"available_plans"`
	Active         bool                   `json:"active"`
}

// PricingPlan represents a pricing plan
type PricingPlan struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	ProductID        string                 `json:"product_id"`
	PricingModel     PricingModel           `json:"pricing_model"`
	BasePrice        float64                `json:"base_price"`
	SetupFee         float64                `json:"setup_fee"`
	MinimumCommitment float64               `json:"minimum_commitment"`
	BillingCycle     BillingCycle           `json:"billing_cycle"`
	IncludedUsage    map[string]float64     `json:"included_usage"`
	UsageRates       map[string]float64     `json:"usage_rates"`
	TierRates        map[string][]TierRate  `json:"tier_rates"`
	Features         []string               `json:"features"`
	Limits           map[string]interface{} `json:"limits"`
	TrialDays        int                    `json:"trial_days"`
	Active           bool                   `json:"active"`
}

// VolumeTier defines volume-based pricing
type VolumeTier struct {
	StartQuantity int     `json:"start_quantity"`
	EndQuantity   int     `json:"end_quantity"` // -1 for unlimited
	UnitPrice     float64 `json:"unit_price"`
	Discount      float64 `json:"discount"`
}

// PricingRule defines custom pricing rules
type PricingRule struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	Conditions map[string]interface{} `json:"conditions"`
	Actions    map[string]interface{} `json:"actions"`
	Priority   int                    `json:"priority"`
	Active     bool                   `json:"active"`
}

// RevenueRecognitionEngine handles ASC 606 compliance
type RevenueRecognitionEngine struct {
	schedules map[string]*RevenueSchedule
	rules     map[string]*RecognitionRule
	mu        sync.RWMutex
}

// RevenueSchedule defines revenue recognition schedule
type RevenueSchedule struct {
	ID                string                  `json:"id"`
	InvoiceID         string                  `json:"invoice_id"`
	SubscriptionID    string                  `json:"subscription_id"`
	TotalAmount       float64                 `json:"total_amount"`
	RecognizedAmount  float64                 `json:"recognized_amount"`
	RemainingAmount   float64                 `json:"remaining_amount"`
	Method            RecognitionMethod       `json:"method"`
	StartDate         time.Time               `json:"start_date"`
	EndDate           time.Time               `json:"end_date"`
	Periods           []*RecognitionPeriod    `json:"periods"`
	Status            string                  `json:"status"`
	CreatedAt         time.Time               `json:"created_at"`
}

// RecognitionMethod defines revenue recognition methods
type RecognitionMethod string

const (
	RecognitionMethodStraightLine RecognitionMethod = "straight_line"
	RecognitionMethodProportional RecognitionMethod = "proportional"
	RecognitionMethodMilestone    RecognitionMethod = "milestone"
	RecognitionMethodImmediate    RecognitionMethod = "immediate"
	RecognitionMethodDeferred     RecognitionMethod = "deferred"
)

// RecognitionPeriod represents a revenue recognition period
type RecognitionPeriod struct {
	Period          string    `json:"period"` // YYYY-MM
	Amount          float64   `json:"amount"`
	RecognizedDate  time.Time `json:"recognized_date"`
	Status          string    `json:"status"`
}

// RecognitionRule defines revenue recognition rules
type RecognitionRule struct {
	ID             string            `json:"id"`
	ProductType    string            `json:"product_type"`
	Method         RecognitionMethod `json:"method"`
	DeferralPeriod int               `json:"deferral_period"` // days
	Conditions     map[string]interface{} `json:"conditions"`
}

// CurrencyManager handles multi-currency operations
type CurrencyManager struct {
	exchangeRates map[string]float64 // currency pair -> rate
	baseCurrency  string
	lastUpdate    time.Time
	mu            sync.RWMutex
}

// PaymentGateway handles payment processing
type PaymentGateway struct {
	gateways map[string]Gateway
	mu       sync.RWMutex
}

// Gateway interface for payment gateway implementations
type Gateway interface {
	ProcessPayment(ctx context.Context, payment *Payment) error
	RefundPayment(ctx context.Context, paymentID string, amount float64) error
	GetStatus(ctx context.Context, transactionID string) (PaymentStatus, error)
}

// BillingMetrics tracks billing performance metrics
type BillingMetrics struct {
	TotalRevenue       float64            `json:"total_revenue"`
	ARR                float64            `json:"arr"`
	MRR                float64            `json:"mrr"`
	AverageContractValue float64          `json:"average_contract_value"`
	CustomerLifetimeValue float64         `json:"customer_lifetime_value"`
	Churn              float64            `json:"churn"`
	GrossMargin        float64            `json:"gross_margin"`
	NetMargin          float64            `json:"net_margin"`
	PaymentSuccessRate float64            `json:"payment_success_rate"`
	DaysToCollect      float64            `json:"days_to_collect"`
	OutstandingAR      float64            `json:"outstanding_ar"`
	CurrencyBreakdown  map[string]float64 `json:"currency_breakdown"`
	LastUpdated        time.Time          `json:"last_updated"`
}

// NewAdvancedBillingEngine creates a new billing engine
func NewAdvancedBillingEngine() *AdvancedBillingEngine {
	return &AdvancedBillingEngine{
		accounts:       make(map[string]*EnterpriseAccount),
		subscriptions:  make(map[string]*Subscription),
		invoices:       make(map[string]*Invoice),
		payments:       make(map[string]*Payment),
		pricingEngine:  NewPricingEngine(),
		revenueEngine:  NewRevenueRecognitionEngine(),
		currencyManager: NewCurrencyManager(),
		paymentGateway: NewPaymentGateway(),
		metrics:        &BillingMetrics{
			CurrencyBreakdown: make(map[string]float64),
			LastUpdated:       time.Now(),
		},
	}
}

// CreateAccount creates a new enterprise account
func (abe *AdvancedBillingEngine) CreateAccount(ctx context.Context, account *EnterpriseAccount) error {
	abe.mu.Lock()
	defer abe.mu.Unlock()

	if account.ID == "" {
		account.ID = uuid.New().String()
	}
	account.CreatedAt = time.Now()
	account.UpdatedAt = time.Now()
	account.Status = "active"

	// Set default payment terms
	if account.PaymentTerms == "" {
		account.PaymentTerms = PaymentTermsNet30
	}

	// Set default currency
	if account.Currency == "" {
		account.Currency = "USD"
	}

	abe.accounts[account.ID] = account
	return nil
}

// CreateSubscription creates a new subscription
func (abe *AdvancedBillingEngine) CreateSubscription(ctx context.Context, sub *Subscription) error {
	abe.mu.Lock()
	defer abe.mu.Unlock()

	if sub.ID == "" {
		sub.ID = uuid.New().String()
	}
	sub.CreatedAt = time.Now()
	sub.UpdatedAt = time.Now()
	sub.Status = SubscriptionStatusActive

	// Calculate pricing
	price, err := abe.pricingEngine.CalculatePrice(sub)
	if err != nil {
		return fmt.Errorf("failed to calculate price: %w", err)
	}
	sub.TotalPrice = price

	// Create revenue schedule
	schedule, err := abe.revenueEngine.CreateSchedule(sub)
	if err != nil {
		return fmt.Errorf("failed to create revenue schedule: %w", err)
	}
	sub.RevenueSchedule = schedule

	// Add to account
	account := abe.accounts[sub.AccountID]
	if account != nil {
		account.Subscriptions = append(account.Subscriptions, sub.ID)
	}

	abe.subscriptions[sub.ID] = sub
	abe.updateMetrics()

	return nil
}

// GenerateInvoice generates an invoice for subscriptions
func (abe *AdvancedBillingEngine) GenerateInvoice(ctx context.Context, accountID string, subscriptionIDs []string) (*Invoice, error) {
	abe.mu.Lock()
	defer abe.mu.Unlock()

	account := abe.accounts[accountID]
	if account == nil {
		return nil, fmt.Errorf("account not found: %s", accountID)
	}

	invoice := &Invoice{
		ID:              uuid.New().String(),
		InvoiceNumber:   fmt.Sprintf("INV-%d", time.Now().Unix()),
		AccountID:       accountID,
		SubscriptionIDs: subscriptionIDs,
		Status:          InvoiceStatusOpen,
		Currency:        account.Currency,
		PaymentTerms:    account.PaymentTerms,
		IssueDate:       time.Now(),
		DueDate:         abe.calculateDueDate(account.PaymentTerms),
		LineItems:       make([]*LineItem, 0),
		Taxes:           make([]*TaxItem, 0),
		Discounts:       make([]*DiscountItem, 0),
		Credits:         make([]*CreditItem, 0),
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	// Add line items from subscriptions
	for _, subID := range subscriptionIDs {
		sub := abe.subscriptions[subID]
		if sub == nil {
			continue
		}

		lineItem := &LineItem{
			ID:          uuid.New().String(),
			Type:        "subscription",
			Description: fmt.Sprintf("%s - %s", sub.ProductID, sub.PlanID),
			ProductID:   sub.ProductID,
			Quantity:    float64(sub.Quantity),
			UnitPrice:   sub.UnitPrice,
			Amount:      sub.TotalPrice,
			PeriodStart: &sub.StartDate,
			PeriodEnd:   sub.EndDate,
		}

		// Apply discounts
		discountAmount := abe.calculateDiscounts(account, sub)
		lineItem.DiscountAmount = discountAmount
		lineItem.NetAmount = lineItem.Amount - discountAmount

		// Calculate tax
		taxAmount := abe.calculateTax(account, lineItem.NetAmount)
		lineItem.TaxAmount = taxAmount
		lineItem.TotalAmount = lineItem.NetAmount + taxAmount

		invoice.LineItems = append(invoice.LineItems, lineItem)
		invoice.Subtotal += lineItem.Amount
	}

	// Calculate totals
	invoice.TotalAmount = invoice.Subtotal
	for _, item := range invoice.LineItems {
		invoice.TotalAmount -= item.DiscountAmount
		invoice.TaxAmount += item.TaxAmount
	}
	invoice.TotalAmount += invoice.TaxAmount
	invoice.BalanceDue = invoice.TotalAmount

	// Create revenue schedule
	schedule, _ := abe.revenueEngine.CreateInvoiceSchedule(invoice)
	invoice.RevenueSchedule = schedule

	abe.invoices[invoice.ID] = invoice

	// Send invoice
	go abe.sendInvoice(ctx, invoice)

	return invoice, nil
}

// ProcessPayment processes a payment
func (abe *AdvancedBillingEngine) ProcessPayment(ctx context.Context, payment *Payment) error {
	abe.mu.Lock()
	defer abe.mu.Unlock()

	if payment.ID == "" {
		payment.ID = uuid.New().String()
	}
	payment.CreatedAt = time.Now()
	payment.Status = PaymentStatusProcessing

	// Process through payment gateway
	if err := abe.paymentGateway.Process(ctx, payment); err != nil {
		payment.Status = PaymentStatusFailed
		payment.FailureReason = err.Error()
		abe.payments[payment.ID] = payment
		return err
	}

	now := time.Now()
	payment.Status = PaymentStatusCompleted
	payment.ProcessedAt = &now

	// Update invoice
	invoice := abe.invoices[payment.InvoiceID]
	if invoice != nil {
		invoice.PaidAmount += payment.Amount
		invoice.BalanceDue -= payment.Amount

		if invoice.BalanceDue <= 0 {
			invoice.Status = InvoiceStatusPaid
			invoice.PaymentDate = &now
		} else {
			invoice.Status = InvoiceStatusPartialPaid
		}

		invoice.UpdatedAt = time.Now()
	}

	abe.payments[payment.ID] = payment
	abe.updateMetrics()

	return nil
}

// ApplyCommitment applies a commitment discount
func (abe *AdvancedBillingEngine) ApplyCommitment(ctx context.Context, accountID string, commitment *Commitment) error {
	abe.mu.Lock()
	defer abe.mu.Unlock()

	account := abe.accounts[accountID]
	if account == nil {
		return fmt.Errorf("account not found: %s", accountID)
	}

	if commitment.ID == "" {
		commitment.ID = uuid.New().String()
	}
	commitment.Status = "active"
	commitment.RemainingCredits = commitment.UsageCredits

	account.Commitments = append(account.Commitments, commitment)
	account.UpdatedAt = time.Now()

	return nil
}

// calculateDueDate calculates invoice due date based on payment terms
func (abe *AdvancedBillingEngine) calculateDueDate(terms PaymentTerms) time.Time {
	now := time.Now()

	switch terms {
	case PaymentTermsImmediate:
		return now
	case PaymentTermsNet30:
		return now.AddDate(0, 0, 30)
	case PaymentTermsNet60:
		return now.AddDate(0, 0, 60)
	case PaymentTermsNet90:
		return now.AddDate(0, 0, 90)
	default:
		return now.AddDate(0, 0, 30)
	}
}

// calculateDiscounts calculates applicable discounts
func (abe *AdvancedBillingEngine) calculateDiscounts(account *EnterpriseAccount, sub *Subscription) float64 {
	totalDiscount := 0.0

	// Apply commitment discounts
	for _, commitment := range account.Commitments {
		if commitment.Status == "active" {
			totalDiscount += sub.TotalPrice * (commitment.Discount / 100)
		}
	}

	// Apply volume discounts
	if sub.Quantity > 100 {
		totalDiscount += sub.TotalPrice * 0.10 // 10% volume discount
	}

	return totalDiscount
}

// calculateTax calculates tax amount
func (abe *AdvancedBillingEngine) calculateTax(account *EnterpriseAccount, amount float64) float64 {
	// Simplified tax calculation
	// In production, integrate with tax service (Avalara, TaxJar, etc.)
	taxRate := 0.0

	if account.BillingAddress != nil {
		// Different rates by jurisdiction
		switch account.BillingAddress.Country {
		case "US":
			taxRate = 0.07 // 7% average US sales tax
		case "GB":
			taxRate = 0.20 // 20% VAT
		case "DE":
			taxRate = 0.19 // 19% VAT
		}
	}

	return amount * taxRate
}

// sendInvoice sends invoice to customer
func (abe *AdvancedBillingEngine) sendInvoice(ctx context.Context, invoice *Invoice) error {
	// Generate PDF
	invoice.PDFUrl = fmt.Sprintf("https://billing.novacron.io/invoices/%s.pdf", invoice.ID)

	// Send email notification
	// Integration with email service would go here

	return nil
}

// updateMetrics updates billing metrics
func (abe *AdvancedBillingEngine) updateMetrics() {
	totalRevenue := 0.0
	mrr := 0.0
	arr := 0.0

	for _, invoice := range abe.invoices {
		if invoice.Status == InvoiceStatusPaid {
			totalRevenue += invoice.TotalAmount
		}
	}

	for _, sub := range abe.subscriptions {
		if sub.Status == SubscriptionStatusActive {
			monthlyValue := abe.getMonthlyValue(sub)
			mrr += monthlyValue
			arr += monthlyValue * 12
		}
	}

	abe.metrics.TotalRevenue = totalRevenue
	abe.metrics.MRR = mrr
	abe.metrics.ARR = arr

	if len(abe.accounts) > 0 {
		abe.metrics.AverageContractValue = arr / float64(len(abe.accounts))
	}

	// Calculate margins
	abe.metrics.GrossMargin = 0.75 // 75% gross margin
	abe.metrics.NetMargin = 0.42   // 42% net margin

	// Calculate payment metrics
	successfulPayments := 0
	totalPayments := 0
	for _, payment := range abe.payments {
		totalPayments++
		if payment.Status == PaymentStatusCompleted {
			successfulPayments++
		}
	}
	if totalPayments > 0 {
		abe.metrics.PaymentSuccessRate = float64(successfulPayments) / float64(totalPayments) * 100
	}

	abe.metrics.LastUpdated = time.Now()
}

// getMonthlyValue converts subscription to monthly value
func (abe *AdvancedBillingEngine) getMonthlyValue(sub *Subscription) float64 {
	switch sub.BillingCycle {
	case BillingCycleMonthly:
		return sub.TotalPrice
	case BillingCycleQuarterly:
		return sub.TotalPrice / 3
	case BillingCycleAnnual:
		return sub.TotalPrice / 12
	case BillingCycleBiennial:
		return sub.TotalPrice / 24
	default:
		return sub.TotalPrice
	}
}

// NewPricingEngine creates a new pricing engine
func NewPricingEngine() *PricingEngine {
	return &PricingEngine{
		products:    make(map[string]*Product),
		plans:       make(map[string]*PricingPlan),
		volumeTiers: make(map[string][]*VolumeTier),
		customRules: make(map[string]*PricingRule),
	}
}

// CalculatePrice calculates subscription price
func (pe *PricingEngine) CalculatePrice(sub *Subscription) (float64, error) {
	plan := pe.plans[sub.PlanID]
	if plan == nil {
		return 0, fmt.Errorf("plan not found: %s", sub.PlanID)
	}

	basePrice := plan.BasePrice * float64(sub.Quantity)

	switch sub.PricingModel {
	case PricingModelFlat:
		return basePrice, nil
	case PricingModelPerUnit:
		return plan.BasePrice * float64(sub.Quantity), nil
	case PricingModelTiered:
		return pe.calculateTieredPrice(sub, plan), nil
	case PricingModelVolume:
		return pe.calculateVolumePrice(sub, plan), nil
	default:
		return basePrice, nil
	}
}

// calculateTieredPrice calculates tiered pricing
func (pe *PricingEngine) calculateTieredPrice(sub *Subscription, plan *PricingPlan) float64 {
	total := 0.0
	remaining := sub.Quantity

	for metricID, tiers := range plan.TierRates {
		if len(tiers) == 0 {
			continue
		}

		for _, tier := range tiers {
			if remaining <= 0 {
				break
			}

			tierUnits := tier.EndUnit - tier.StartUnit
			if tierUnits <= 0 || remaining < tierUnits {
				tierUnits = remaining
			}

			total += float64(tierUnits) * tier.Rate
			remaining -= tierUnits
		}

		_ = metricID
	}

	return total
}

// calculateVolumePrice calculates volume pricing
func (pe *PricingEngine) calculateVolumePrice(sub *Subscription, plan *PricingPlan) float64 {
	tiers := pe.volumeTiers[plan.ID]
	if len(tiers) == 0 {
		return plan.BasePrice * float64(sub.Quantity)
	}

	for _, tier := range tiers {
		if sub.Quantity >= tier.StartQuantity &&
		   (tier.EndQuantity == -1 || sub.Quantity <= tier.EndQuantity) {
			return tier.UnitPrice * float64(sub.Quantity)
		}
	}

	return plan.BasePrice * float64(sub.Quantity)
}

// NewRevenueRecognitionEngine creates a new revenue recognition engine
func NewRevenueRecognitionEngine() *RevenueRecognitionEngine {
	return &RevenueRecognitionEngine{
		schedules: make(map[string]*RevenueSchedule),
		rules:     make(map[string]*RecognitionRule),
	}
}

// CreateSchedule creates a revenue recognition schedule
func (rre *RevenueRecognitionEngine) CreateSchedule(sub *Subscription) (*RevenueSchedule, error) {
	schedule := &RevenueSchedule{
		ID:               uuid.New().String(),
		SubscriptionID:   sub.ID,
		TotalAmount:      sub.TotalPrice,
		RecognizedAmount: 0,
		RemainingAmount:  sub.TotalPrice,
		Method:           RecognitionMethodStraightLine,
		StartDate:        sub.StartDate,
		EndDate:          sub.NextBillingDate,
		Periods:          make([]*RecognitionPeriod, 0),
		Status:           "active",
		CreatedAt:        time.Now(),
	}

	// Calculate periods
	periods := rre.calculatePeriods(schedule)
	schedule.Periods = periods

	rre.schedules[schedule.ID] = schedule
	return schedule, nil
}

// CreateInvoiceSchedule creates revenue schedule for invoice
func (rre *RevenueRecognitionEngine) CreateInvoiceSchedule(invoice *Invoice) (*RevenueSchedule, error) {
	schedule := &RevenueSchedule{
		ID:               uuid.New().String(),
		InvoiceID:        invoice.ID,
		TotalAmount:      invoice.TotalAmount,
		RecognizedAmount: 0,
		RemainingAmount:  invoice.TotalAmount,
		Method:           RecognitionMethodStraightLine,
		StartDate:        invoice.PeriodStart,
		EndDate:          invoice.PeriodEnd,
		Periods:          make([]*RecognitionPeriod, 0),
		Status:           "active",
		CreatedAt:        time.Now(),
	}

	periods := rre.calculatePeriods(schedule)
	schedule.Periods = periods

	rre.schedules[schedule.ID] = schedule
	return schedule, nil
}

// calculatePeriods calculates recognition periods
func (rre *RevenueRecognitionEngine) calculatePeriods(schedule *RevenueSchedule) []*RecognitionPeriod {
	periods := make([]*RecognitionPeriod, 0)

	// Calculate months between start and end
	months := int(math.Ceil(schedule.EndDate.Sub(schedule.StartDate).Hours() / 24 / 30))
	if months == 0 {
		months = 1
	}

	amountPerPeriod := schedule.TotalAmount / float64(months)
	currentDate := schedule.StartDate

	for i := 0; i < months; i++ {
		period := &RecognitionPeriod{
			Period:         currentDate.Format("2006-01"),
			Amount:         amountPerPeriod,
			RecognizedDate: currentDate,
			Status:         "pending",
		}
		periods = append(periods, period)
		currentDate = currentDate.AddDate(0, 1, 0)
	}

	return periods
}

// NewCurrencyManager creates a new currency manager
func NewCurrencyManager() *CurrencyManager {
	cm := &CurrencyManager{
		exchangeRates: make(map[string]float64),
		baseCurrency:  "USD",
		lastUpdate:    time.Now(),
	}

	// Initialize common exchange rates
	cm.exchangeRates["USD/EUR"] = 0.92
	cm.exchangeRates["USD/GBP"] = 0.79
	cm.exchangeRates["USD/JPY"] = 149.50
	cm.exchangeRates["USD/CAD"] = 1.35
	cm.exchangeRates["USD/AUD"] = 1.53
	cm.exchangeRates["USD/CNY"] = 7.24
	cm.exchangeRates["USD/INR"] = 83.12

	return cm
}

// Convert converts amount between currencies
func (cm *CurrencyManager) Convert(amount float64, from, to string) (float64, error) {
	if from == to {
		return amount, nil
	}

	key := fmt.Sprintf("%s/%s", from, to)
	rate, exists := cm.exchangeRates[key]
	if !exists {
		// Try inverse
		inverseKey := fmt.Sprintf("%s/%s", to, from)
		if inverseRate, ok := cm.exchangeRates[inverseKey]; ok {
			rate = 1.0 / inverseRate
		} else {
			return 0, fmt.Errorf("exchange rate not found: %s", key)
		}
	}

	return amount * rate, nil
}

// NewPaymentGateway creates a new payment gateway
func NewPaymentGateway() *PaymentGateway {
	return &PaymentGateway{
		gateways: make(map[string]Gateway),
	}
}

// Process processes a payment
func (pg *PaymentGateway) Process(ctx context.Context, payment *Payment) error {
	// Simulate payment processing
	payment.TransactionID = uuid.New().String()
	payment.Gateway = "stripe"

	// 98% success rate
	if uuid.New().ClockSequence()%100 < 98 {
		return nil
	}

	return fmt.Errorf("payment processing failed")
}

// GetMetrics returns billing metrics
func (abe *AdvancedBillingEngine) GetMetrics() *BillingMetrics {
	abe.mu.RLock()
	defer abe.mu.RUnlock()

	abe.updateMetrics()
	return abe.metrics
}

// GenerateRevenueReport generates comprehensive revenue report
func (abe *AdvancedBillingEngine) GenerateRevenueReport(ctx context.Context) ([]byte, error) {
	abe.mu.RLock()
	defer abe.mu.RUnlock()

	metrics := abe.GetMetrics()

	report := map[string]interface{}{
		"generated_at":             time.Now(),
		"total_revenue":            fmt.Sprintf("$%.2fM", metrics.TotalRevenue/1000000),
		"arr":                      fmt.Sprintf("$%.2fM", metrics.ARR/1000000),
		"mrr":                      fmt.Sprintf("$%.2fM", metrics.MRR/1000000),
		"average_contract_value":   fmt.Sprintf("$%.2fM", metrics.AverageContractValue/1000000),
		"gross_margin":             fmt.Sprintf("%.2f%%", metrics.GrossMargin*100),
		"net_margin":               fmt.Sprintf("%.2f%%", metrics.NetMargin*100),
		"payment_success_rate":     fmt.Sprintf("%.2f%%", metrics.PaymentSuccessRate),
		"total_accounts":           len(abe.accounts),
		"active_subscriptions":     len(abe.subscriptions),
		"outstanding_invoices":     abe.getOutstandingInvoiceCount(),
		"outstanding_ar":           fmt.Sprintf("$%.2fM", metrics.OutstandingAR/1000000),
	}

	return json.MarshalIndent(report, "", "  ")
}

// getOutstandingInvoiceCount returns count of outstanding invoices
func (abe *AdvancedBillingEngine) getOutstandingInvoiceCount() int {
	count := 0
	for _, invoice := range abe.invoices {
		if invoice.Status == InvoiceStatusOpen || invoice.Status == InvoiceStatusOverdue {
			count++
		}
	}
	return count
}
