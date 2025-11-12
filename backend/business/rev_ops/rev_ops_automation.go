// Package rev_ops provides revenue operations automation
package rev_ops

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// RevOpsAutomation manages quote-to-cash automation
type RevOpsAutomation struct {
	mu                  sync.RWMutex
	quotes              map[string]*Quote
	contracts           map[string]*Contract
	invoices            map[string]*Invoice
	payments            map[string]*Payment
	subscriptions       map[string]*Subscription
	recognitionEngine   *RevenueRecognition
	billingEngine       *BillingEngine
	collectionsEngine   *CollectionsEngine
	reconciliationEngine *ReconciliationEngine
	metrics             *RevOpsMetrics
	config              RevOpsConfig
}

// Quote represents sales quote
type Quote struct {
	ID                  string                 `json:"id"`
	OpportunityID       string                 `json:"opportunity_id"`
	CustomerID          string                 `json:"customer_id"`
	Status              string                 `json:"status"`              // draft, sent, accepted, rejected, expired
	Version             int                    `json:"version"`
	LineItems           []QuoteLineItem        `json:"line_items"`
	Subtotal            float64                `json:"subtotal"`
	Discounts           []QuoteDiscount        `json:"discounts"`
	Tax                 float64                `json:"tax"`
	Total               float64                `json:"total"`
	ARR                 float64                `json:"arr"`
	TCV                 float64                `json:"tcv"`                 // Total Contract Value
	Currency            string                 `json:"currency"`
	ExchangeRate        float64                `json:"exchange_rate"`
	PaymentTerms        string                 `json:"payment_terms"`
	ContractLength      int                    `json:"contract_length"`     // Months
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
	BillingFrequency    string                 `json:"billing_frequency"`   // monthly, quarterly, annual
	AutoRenewal         bool                   `json:"auto_renewal"`
	ApprovalRequired    bool                   `json:"approval_required"`
	ApprovedBy          string                 `json:"approved_by"`
	ApprovedAt          *time.Time             `json:"approved_at,omitempty"`
	SentAt              *time.Time             `json:"sent_at,omitempty"`
	AcceptedAt          *time.Time             `json:"accepted_at,omitempty"`
	ExpiresAt           time.Time              `json:"expires_at"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// QuoteLineItem represents individual line item
type QuoteLineItem struct {
	ProductID           string                 `json:"product_id"`
	ProductName         string                 `json:"product_name"`
	Description         string                 `json:"description"`
	Quantity            int                    `json:"quantity"`
	UnitPrice           float64                `json:"unit_price"`
	Discount            float64                `json:"discount"`            // Percentage
	NetPrice            float64                `json:"net_price"`
	TotalPrice          float64                `json:"total_price"`
	ARRContribution     float64                `json:"arr_contribution"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
}

// QuoteDiscount represents discount applied
type QuoteDiscount struct {
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Percentage          float64                `json:"percentage"`
	Amount              float64                `json:"amount"`
	ApprovalLevel       string                 `json:"approval_level"`
}

// Contract represents legal agreement
type Contract struct {
	ID                  string                 `json:"id"`
	QuoteID             string                 `json:"quote_id"`
	CustomerID          string                 `json:"customer_id"`
	Status              string                 `json:"status"`              // draft, active, suspended, terminated, expired
	ContractNumber      string                 `json:"contract_number"`
	Type                string                 `json:"type"`                // new, renewal, amendment
	ARR                 float64                `json:"arr"`
	TCV                 float64                `json:"tcv"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
	SignedDate          *time.Time             `json:"signed_date,omitempty"`
	RenewalDate         time.Time              `json:"renewal_date"`
	AutoRenewal         bool                   `json:"auto_renewal"`
	NoticeRequired      int                    `json:"notice_required"`     // Days
	PaymentTerms        string                 `json:"payment_terms"`
	BillingSchedule     []BillingSchedule      `json:"billing_schedule"`
	Terms               ContractTerms          `json:"terms"`
	Amendments          []ContractAmendment    `json:"amendments"`
	DocumentURL         string                 `json:"document_url"`
	SignedByCustomer    string                 `json:"signed_by_customer"`
	SignedByVendor      string                 `json:"signed_by_vendor"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// BillingSchedule defines when to bill
type BillingSchedule struct {
	Date                time.Time              `json:"date"`
	Amount              float64                `json:"amount"`
	Description         string                 `json:"description"`
	Status              string                 `json:"status"`              // pending, invoiced, paid
	InvoiceID           string                 `json:"invoice_id,omitempty"`
}

// ContractTerms defines legal terms
type ContractTerms struct {
	PaymentTerms        string                 `json:"payment_terms"`
	SLA                 SLATerms               `json:"sla"`
	SupportLevel        string                 `json:"support_level"`
	DataRetention       int                    `json:"data_retention"`      // Days
	TerminationClause   string                 `json:"termination_clause"`
	LiabilityLimit      float64                `json:"liability_limit"`
	Warranties          []string               `json:"warranties"`
	Compliance          []string               `json:"compliance"`          // SOC2, GDPR, etc.
}

// SLATerms defines service level agreement
type SLATerms struct {
	Uptime              float64                `json:"uptime"`              // 99.9%
	ResponseTime        int                    `json:"response_time"`       // Minutes
	ResolutionTime      int                    `json:"resolution_time"`     // Hours
	Credits             []SLACredit            `json:"credits"`
}

// SLACredit defines penalties
type SLACredit struct {
	UptimeThreshold     float64                `json:"uptime_threshold"`
	CreditPercentage    float64                `json:"credit_percentage"`
}

// ContractAmendment tracks changes
type ContractAmendment struct {
	ID                  string                 `json:"id"`
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	ARRImpact           float64                `json:"arr_impact"`
	EffectiveDate       time.Time              `json:"effective_date"`
	ApprovedBy          string                 `json:"approved_by"`
	CreatedAt           time.Time              `json:"created_at"`
}

// Invoice represents billing invoice
type Invoice struct {
	ID                  string                 `json:"id"`
	InvoiceNumber       string                 `json:"invoice_number"`
	ContractID          string                 `json:"contract_id"`
	CustomerID          string                 `json:"customer_id"`
	Status              string                 `json:"status"`              // draft, sent, paid, overdue, void
	Type                string                 `json:"type"`                // recurring, one-time, usage
	LineItems           []InvoiceLineItem      `json:"line_items"`
	Subtotal            float64                `json:"subtotal"`
	Tax                 float64                `json:"tax"`
	Total               float64                `json:"total"`
	AmountPaid          float64                `json:"amount_paid"`
	AmountDue           float64                `json:"amount_due"`
	Currency            string                 `json:"currency"`
	ExchangeRate        float64                `json:"exchange_rate"`
	BillingPeriodStart  time.Time              `json:"billing_period_start"`
	BillingPeriodEnd    time.Time              `json:"billing_period_end"`
	IssueDate           time.Time              `json:"issue_date"`
	DueDate             time.Time              `json:"due_date"`
	PaidDate            *time.Time             `json:"paid_date,omitempty"`
	PaymentMethod       string                 `json:"payment_method"`
	Notes               string                 `json:"notes"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// InvoiceLineItem represents invoice line
type InvoiceLineItem struct {
	Description         string                 `json:"description"`
	Quantity            float64                `json:"quantity"`
	UnitPrice           float64                `json:"unit_price"`
	Amount              float64                `json:"amount"`
	RevenueType         string                 `json:"revenue_type"`        // subscription, usage, professional_services
	RecognitionSchedule []RecognitionEntry     `json:"recognition_schedule"`
}

// RecognitionEntry for ASC 606 compliance
type RecognitionEntry struct {
	Date                time.Time              `json:"date"`
	Amount              float64                `json:"amount"`
	Recognized          bool                   `json:"recognized"`
	RecognizedAt        *time.Time             `json:"recognized_at,omitempty"`
}

// Payment represents payment transaction
type Payment struct {
	ID                  string                 `json:"id"`
	InvoiceID           string                 `json:"invoice_id"`
	CustomerID          string                 `json:"customer_id"`
	Amount              float64                `json:"amount"`
	Currency            string                 `json:"currency"`
	ExchangeRate        float64                `json:"exchange_rate"`
	Method              string                 `json:"method"`              // credit_card, ach, wire, check
	Status              string                 `json:"status"`              // pending, completed, failed, refunded
	TransactionID       string                 `json:"transaction_id"`
	ProcessedAt         *time.Time             `json:"processed_at,omitempty"`
	FailureReason       string                 `json:"failure_reason,omitempty"`
	RefundID            string                 `json:"refund_id,omitempty"`
	CreatedAt           time.Time              `json:"created_at"`
}

// Subscription tracks recurring revenue
type Subscription struct {
	ID                  string                 `json:"id"`
	CustomerID          string                 `json:"customer_id"`
	ContractID          string                 `json:"contract_id"`
	Status              string                 `json:"status"`              // active, paused, cancelled, expired
	ProductID           string                 `json:"product_id"`
	Quantity            int                    `json:"quantity"`
	BillingCycle        string                 `json:"billing_cycle"`       // monthly, annual
	Amount              float64                `json:"amount"`
	ARR                 float64                `json:"arr"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             *time.Time             `json:"end_date,omitempty"`
	NextBillingDate     time.Time              `json:"next_billing_date"`
	LastBillingDate     *time.Time             `json:"last_billing_date,omitempty"`
	AutoRenew           bool                   `json:"auto_renew"`
	CancellationDate    *time.Time             `json:"cancellation_date,omitempty"`
	CancellationReason  string                 `json:"cancellation_reason,omitempty"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// RevenueRecognition handles ASC 606 compliance
type RevenueRecognition struct {
	mu                  sync.RWMutex
	schedules           map[string]*RecognitionSchedule
	metrics             RecognitionMetrics
}

// RecognitionSchedule tracks revenue recognition
type RecognitionSchedule struct {
	ID                  string                 `json:"id"`
	ContractID          string                 `json:"contract_id"`
	TotalAmount         float64                `json:"total_amount"`
	RecognizedAmount    float64                `json:"recognized_amount"`
	DeferredAmount      float64                `json:"deferred_amount"`
	Entries             []RecognitionEntry     `json:"entries"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
	Method              string                 `json:"method"`              // straight-line, performance-based
}

// RecognitionMetrics tracks revenue recognition
type RecognitionMetrics struct {
	TotalRevenue        float64                `json:"total_revenue"`
	RecognizedRevenue   float64                `json:"recognized_revenue"`
	DeferredRevenue     float64                `json:"deferred_revenue"`
	CurrentMonthRevenue float64                `json:"current_month_revenue"`
	CurrentQuarterRevenue float64              `json:"current_quarter_revenue"`
}

// BillingEngine handles invoice generation
type BillingEngine struct {
	mu                  sync.RWMutex
	currencies          map[string]float64     // Exchange rates
	taxRates            map[string]float64     // Tax rates by region
	metrics             BillingMetrics
}

// BillingMetrics tracks billing performance
type BillingMetrics struct {
	TotalInvoiced       float64                `json:"total_invoiced"`
	TotalCollected      float64                `json:"total_collected"`
	TotalOutstanding    float64                `json:"total_outstanding"`
	AvgCollectionTime   int                    `json:"avg_collection_time"` // Days
	OnTimePaymentRate   float64                `json:"on_time_payment_rate"`
}

// CollectionsEngine manages payment collection
type CollectionsEngine struct {
	mu                  sync.RWMutex
	overdueInvoices     []*Invoice
	collectionActions   []CollectionAction
	metrics             CollectionMetrics
}

// CollectionAction represents collection activity
type CollectionAction struct {
	InvoiceID           string                 `json:"invoice_id"`
	Type                string                 `json:"type"`                // email, call, escalation
	Description         string                 `json:"description"`
	ScheduledAt         time.Time              `json:"scheduled_at"`
	CompletedAt         *time.Time             `json:"completed_at,omitempty"`
	Outcome             string                 `json:"outcome"`
}

// CollectionMetrics tracks collections performance
type CollectionMetrics struct {
	TotalOverdue        float64                `json:"total_overdue"`
	OverdueCount        int                    `json:"overdue_count"`
	DSO                 int                    `json:"dso"`                 // Days Sales Outstanding
	CollectionRate      float64                `json:"collection_rate"`
	BadDebtRate         float64                `json:"bad_debt_rate"`
}

// ReconciliationEngine handles financial reconciliation
type ReconciliationEngine struct {
	mu                  sync.RWMutex
	reconciliations     []Reconciliation
	discrepancies       []Discrepancy
	metrics             ReconciliationMetrics
}

// Reconciliation represents reconciliation record
type Reconciliation struct {
	ID                  string                 `json:"id"`
	Period              string                 `json:"period"`              // 2024-01
	Type                string                 `json:"type"`                // invoice_to_payment, arr_to_recognized
	ExpectedAmount      float64                `json:"expected_amount"`
	ActualAmount        float64                `json:"actual_amount"`
	Variance            float64                `json:"variance"`
	Status              string                 `json:"status"`              // pending, reconciled, discrepancy
	ReconciledAt        *time.Time             `json:"reconciled_at,omitempty"`
	ReconciledBy        string                 `json:"reconciled_by,omitempty"`
	CreatedAt           time.Time              `json:"created_at"`
}

// Discrepancy represents reconciliation issue
type Discrepancy struct {
	ID                  string                 `json:"id"`
	ReconciliationID    string                 `json:"reconciliation_id"`
	Type                string                 `json:"type"`
	Amount              float64                `json:"amount"`
	Description         string                 `json:"description"`
	Status              string                 `json:"status"`              // open, investigating, resolved
	Resolution          string                 `json:"resolution"`
	ResolvedAt          *time.Time             `json:"resolved_at,omitempty"`
	CreatedAt           time.Time              `json:"created_at"`
}

// ReconciliationMetrics tracks reconciliation performance
type ReconciliationMetrics struct {
	TotalReconciliations int                   `json:"total_reconciliations"`
	PendingReconciliations int                 `json:"pending_reconciliations"`
	Discrepancies       int                    `json:"discrepancies"`
	AccuracyRate        float64                `json:"accuracy_rate"`
}

// RevOpsMetrics tracks overall performance
type RevOpsMetrics struct {
	mu                  sync.RWMutex
	QuotesGenerated     int64                  `json:"quotes_generated"`
	QuotesAccepted      int64                  `json:"quotes_accepted"`
	ContractsActive     int64                  `json:"contracts_active"`
	InvoicesGenerated   int64                  `json:"invoices_generated"`
	PaymentsProcessed   int64                  `json:"payments_processed"`
	TotalARR            float64                `json:"total_arr"`
	TotalBilled         float64                `json:"total_billed"`
	TotalCollected      float64                `json:"total_collected"`
	AvgQuoteToContract  int                    `json:"avg_quote_to_contract"` // Days
	AvgContractToInvoice int                   `json:"avg_contract_to_invoice"` // Days
}

// RevOpsConfig configures automation
type RevOpsConfig struct {
	EnableAutoInvoicing bool                   `json:"enable_auto_invoicing"`
	EnableAutoCollection bool                  `json:"enable_auto_collections"`
	PaymentGracePeriod  int                    `json:"payment_grace_period"` // Days
	SupportedCurrencies []string               `json:"supported_currencies"` // 50+ currencies
	TaxCompliance       []string               `json:"tax_compliance"`
}

// NewRevOpsAutomation creates automation engine
func NewRevOpsAutomation(config RevOpsConfig) *RevOpsAutomation {
	return &RevOpsAutomation{
		quotes:              make(map[string]*Quote),
		contracts:           make(map[string]*Contract),
		invoices:            make(map[string]*Invoice),
		payments:            make(map[string]*Payment),
		subscriptions:       make(map[string]*Subscription),
		recognitionEngine:   initializeRecognitionEngine(),
		billingEngine:       initializeBillingEngine(),
		collectionsEngine:   initializeCollectionsEngine(),
		reconciliationEngine: initializeReconciliationEngine(),
		metrics:             &RevOpsMetrics{},
		config:              config,
	}
}

// GenerateQuote creates quote from opportunity
func (r *RevOpsAutomation) GenerateQuote(ctx context.Context, opportunityID string, lineItems []QuoteLineItem) (*Quote, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	quote := &Quote{
		ID:              fmt.Sprintf("quote-%s-%d", opportunityID, time.Now().Unix()),
		OpportunityID:   opportunityID,
		Status:          "draft",
		Version:         1,
		LineItems:       lineItems,
		Currency:        "USD",
		ExchangeRate:    1.0,
		PaymentTerms:    "Net 30",
		BillingFrequency: "annual",
		AutoRenewal:     true,
		ExpiresAt:       time.Now().AddDate(0, 0, 30),
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	// Calculate totals
	subtotal := 0.0
	arr := 0.0
	for _, item := range lineItems {
		subtotal += item.TotalPrice
		arr += item.ARRContribution
	}

	quote.Subtotal = subtotal
	quote.ARR = arr
	quote.Tax = subtotal * 0.08 // Simplified tax
	quote.Total = subtotal + quote.Tax
	quote.TCV = quote.Total

	// Store quote
	r.quotes[quote.ID] = quote

	// Update metrics
	r.metrics.mu.Lock()
	r.metrics.QuotesGenerated++
	r.metrics.mu.Unlock()

	return quote, nil
}

// Helper initialization functions
func initializeRecognitionEngine() *RevenueRecognition {
	return &RevenueRecognition{
		schedules: make(map[string]*RecognitionSchedule),
	}
}

func initializeBillingEngine() *BillingEngine {
	return &BillingEngine{
		currencies: map[string]float64{
			"USD": 1.0,
			"EUR": 0.92,
			"GBP": 0.79,
			"JPY": 149.50,
			// 50+ more currencies in production
		},
		taxRates: make(map[string]float64),
	}
}

func initializeCollectionsEngine() *CollectionsEngine {
	return &CollectionsEngine{
		overdueInvoices:   make([]*Invoice, 0),
		collectionActions: make([]CollectionAction, 0),
	}
}

func initializeReconciliationEngine() *ReconciliationEngine {
	return &ReconciliationEngine{
		reconciliations: make([]Reconciliation, 0),
		discrepancies:   make([]Discrepancy, 0),
	}
}

// ExportMetrics exports revenue operations metrics
func (r *RevOpsAutomation) ExportMetrics() map[string]interface{} {
	r.metrics.mu.RLock()
	defer r.metrics.mu.RUnlock()

	return map[string]interface{}{
		"quotes_generated":    r.metrics.QuotesGenerated,
		"quotes_accepted":     r.metrics.QuotesAccepted,
		"contracts_active":    r.metrics.ContractsActive,
		"invoices_generated":  r.metrics.InvoicesGenerated,
		"payments_processed":  r.metrics.PaymentsProcessed,
		"total_arr":           r.metrics.TotalARR,
		"total_billed":        r.metrics.TotalBilled,
		"total_collected":     r.metrics.TotalCollected,
	}
}

// MarshalJSON implements json.Marshaler
func (r *RevOpsAutomation) MarshalJSON() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return json.Marshal(map[string]interface{}{
		"quotes":        len(r.quotes),
		"contracts":     len(r.contracts),
		"invoices":      len(r.invoices),
		"payments":      len(r.payments),
		"subscriptions": len(r.subscriptions),
		"metrics":       r.ExportMetrics(),
	})
}
