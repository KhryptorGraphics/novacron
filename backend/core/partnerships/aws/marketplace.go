// Package aws provides AWS Marketplace integration for NovaCron DWCP v3
package aws

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/marketplacemetering"
	"github.com/aws/aws-sdk-go-v2/service/marketplacecatalog"
	"github.com/aws/aws-sdk-go-v2/service/sts"
	"github.com/google/uuid"
)

// MarketplaceManager handles AWS Marketplace integration
type MarketplaceManager struct {
	meteringClient *marketplacemetering.Client
	catalogClient  *marketplacecatalog.Client
	stsClient      *sts.Client
	productCode    string
	publicKeyVersion int32
	cfg            aws.Config
}

// MarketplaceConfig configures AWS Marketplace integration
type MarketplaceConfig struct {
	ProductCode      string
	PublicKeyVersion int32
	Region           string
}

// UsageRecord represents a metering usage record
type UsageRecord struct {
	CustomerIdentifier string
	Dimension          string
	Quantity           int64
	Timestamp          time.Time
	UsageAllocations   []UsageAllocation
}

// UsageAllocation represents usage allocation across tags
type UsageAllocation struct {
	AllocatedUsageQuantity int64
	Tags                   []Tag
}

// Tag represents a marketplace tag
type Tag struct {
	Key   string
	Value string
}

// SubscriptionInfo contains marketplace subscription details
type SubscriptionInfo struct {
	CustomerID        string
	ProductCode       string
	SubscriptionArn   string
	Status            string
	StartDate         time.Time
	EndDate           *time.Time
	AutoRenew         bool
	Dimensions        map[string]int64
	EntitlementStatus map[string]bool
}

// ListingMetadata contains marketplace listing information
type ListingMetadata struct {
	ProductID     string
	ProductCode   string
	Name          string
	Description   string
	ShortDesc     string
	LogoURL       string
	Version       string
	ReleaseNotes  string
	Category      string
	SupportURL    string
	DocumentURL   string
	VideoURLs     []string
	Highlights    []string
	SearchKeywords []string
	PricingModel  PricingModel
}

// PricingModel represents marketplace pricing
type PricingModel struct {
	Type       string // "hourly", "monthly", "annual", "usage-based"
	Dimensions []PricingDimension
	FreeTrial  *FreeTrialConfig
}

// PricingDimension represents a billable dimension
type PricingDimension struct {
	Key         string
	Name        string
	Description string
	Unit        string
	PricePerUnit float64
	MinQuantity  int64
	MaxQuantity  int64
}

// FreeTrialConfig configures free trial
type FreeTrialConfig struct {
	DurationDays int
	Dimensions   map[string]int64
}

// NewMarketplaceManager creates a new AWS Marketplace manager
func NewMarketplaceManager(ctx context.Context, cfg MarketplaceConfig) (*MarketplaceManager, error) {
	awsCfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(cfg.Region))
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	return &MarketplaceManager{
		meteringClient:   marketplacemetering.NewFromConfig(awsCfg),
		catalogClient:    marketplacecatalog.NewFromConfig(awsCfg),
		stsClient:        sts.NewFromConfig(awsCfg),
		productCode:      cfg.ProductCode,
		publicKeyVersion: cfg.PublicKeyVersion,
		cfg:              awsCfg,
	}, nil
}

// MeterUsage sends usage records to AWS Marketplace Metering Service
func (m *MarketplaceManager) MeterUsage(ctx context.Context, record UsageRecord) error {
	// Validate record
	if record.CustomerIdentifier == "" {
		return fmt.Errorf("customer identifier is required")
	}
	if record.Dimension == "" {
		return fmt.Errorf("dimension is required")
	}
	if record.Quantity <= 0 {
		return fmt.Errorf("quantity must be positive")
	}

	// Prepare usage allocations
	var allocations []marketplacemetering.UsageAllocation
	for _, alloc := range record.UsageAllocations {
		var tags []marketplacemetering.Tag
		for _, tag := range alloc.Tags {
			tags = append(tags, marketplacemetering.Tag{
				Key:   aws.String(tag.Key),
				Value: aws.String(tag.Value),
			})
		}
		allocations = append(allocations, marketplacemetering.UsageAllocation{
			AllocatedUsageQuantity: aws.Int32(int32(alloc.AllocatedUsageQuantity)),
			Tags:                   tags,
		})
	}

	// Send metering record
	input := &marketplacemetering.MeterUsageInput{
		ProductCode:        aws.String(m.productCode),
		Timestamp:          aws.Time(record.Timestamp),
		UsageDimension:     aws.String(record.Dimension),
		UsageQuantity:      aws.Int32(int32(record.Quantity)),
		DryRun:             aws.Bool(false),
		UsageAllocations:   allocations,
	}

	result, err := m.meteringClient.MeterUsage(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to meter usage: %w", err)
	}

	// Log metering ID for audit
	fmt.Printf("Metered usage: MeteringRecordId=%s\n", *result.MeteringRecordId)

	return nil
}

// BatchMeterUsage sends batch usage records (up to 25 records)
func (m *MarketplaceManager) BatchMeterUsage(ctx context.Context, records []UsageRecord) error {
	if len(records) > 25 {
		return fmt.Errorf("batch size exceeds maximum of 25 records")
	}

	var usageRecords []marketplacemetering.UsageRecord
	for _, record := range records {
		var allocations []marketplacemetering.UsageAllocation
		for _, alloc := range record.UsageAllocations {
			var tags []marketplacemetering.Tag
			for _, tag := range alloc.Tags {
				tags = append(tags, marketplacemetering.Tag{
					Key:   aws.String(tag.Key),
					Value: aws.String(tag.Value),
				})
			}
			allocations = append(allocations, marketplacemetering.UsageAllocation{
				AllocatedUsageQuantity: aws.Int32(int32(alloc.AllocatedUsageQuantity)),
				Tags:                   tags,
			})
		}

		usageRecords = append(usageRecords, marketplacemetering.UsageRecord{
			CustomerIdentifier: aws.String(record.CustomerIdentifier),
			Dimension:          aws.String(record.Dimension),
			Quantity:           aws.Int32(int32(record.Quantity)),
			Timestamp:          aws.Time(record.Timestamp),
			UsageAllocations:   allocations,
		})
	}

	input := &marketplacemetering.BatchMeterUsageInput{
		ProductCode:  aws.String(m.productCode),
		UsageRecords: usageRecords,
	}

	result, err := m.meteringClient.BatchMeterUsage(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to batch meter usage: %w", err)
	}

	// Log results
	fmt.Printf("Batch metering: Processed=%d, Unprocessed=%d\n",
		len(result.Results), len(result.UnprocessedRecords))

	// Handle unprocessed records
	if len(result.UnprocessedRecords) > 0 {
		return fmt.Errorf("batch metering had %d unprocessed records", len(result.UnprocessedRecords))
	}

	return nil
}

// ResolveCustomer resolves registration token to customer information
func (m *MarketplaceManager) ResolveCustomer(ctx context.Context, registrationToken string) (*SubscriptionInfo, error) {
	input := &marketplacemetering.ResolveCustomerInput{
		RegistrationToken: aws.String(registrationToken),
	}

	result, err := m.meteringClient.ResolveCustomer(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve customer: %w", err)
	}

	return &SubscriptionInfo{
		CustomerID:  aws.ToString(result.CustomerIdentifier),
		ProductCode: aws.ToString(result.ProductCode),
	}, nil
}

// RegisterUsage registers customer usage (container-based pricing)
func (m *MarketplaceManager) RegisterUsage(ctx context.Context, customerID string, nonce string) error {
	input := &marketplacemetering.RegisterUsageInput{
		ProductCode:      aws.String(m.productCode),
		PublicKeyVersion: aws.Int32(m.publicKeyVersion),
		Nonce:            aws.String(nonce),
	}

	result, err := m.meteringClient.RegisterUsage(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to register usage: %w", err)
	}

	fmt.Printf("Registered usage: Signature=%s\n", aws.ToString(result.Signature))

	return nil
}

// GetSubscriptionInfo retrieves subscription information
func (m *MarketplaceManager) GetSubscriptionInfo(ctx context.Context, customerID string) (*SubscriptionInfo, error) {
	// Use AWS License Manager or custom API to retrieve subscription details
	// This is a placeholder implementation
	return &SubscriptionInfo{
		CustomerID:  customerID,
		ProductCode: m.productCode,
		Status:      "ACTIVE",
		StartDate:   time.Now().AddDate(0, -1, 0),
		AutoRenew:   true,
		Dimensions: map[string]int64{
			"vm-hours":      1000,
			"storage-gb":    5000,
			"transfer-gb":   10000,
		},
		EntitlementStatus: map[string]bool{
			"advanced-features": true,
			"premium-support":   true,
		},
	}, nil
}

// CreateListing creates or updates marketplace listing
func (m *MarketplaceManager) CreateListing(ctx context.Context, metadata ListingMetadata) error {
	// Prepare change set
	changeSet := map[string]interface{}{
		"Version": map[string]interface{}{
			"VersionTitle": metadata.Version,
			"ReleaseNotes": metadata.ReleaseNotes,
		},
		"Description": map[string]interface{}{
			"ProductTitle":       metadata.Name,
			"ShortDescription":   metadata.ShortDesc,
			"LongDescription":    metadata.Description,
			"Highlights":         metadata.Highlights,
			"ProductCode":        metadata.ProductCode,
			"SearchKeywords":     metadata.SearchKeywords,
			"Categories":         []string{metadata.Category},
			"LogoUrl":            metadata.LogoURL,
			"VideoUrls":          metadata.VideoURLs,
			"SupportDescription": "24/7 support available",
		},
		"Dimensions": m.buildDimensionsChangeSet(metadata.PricingModel),
	}

	changeSetJSON, err := json.Marshal(changeSet)
	if err != nil {
		return fmt.Errorf("failed to marshal change set: %w", err)
	}

	input := &marketplacecatalog.StartChangeSetInput{
		Catalog: aws.String("AWSMarketplace"),
		ChangeSet: []marketplacecatalog.Change{
			{
				ChangeType: aws.String("UpdateInformation"),
				Entity: &marketplacecatalog.Entity{
					Type:       aws.String("ServerProduct@1.0"),
					Identifier: aws.String(metadata.ProductID),
				},
				Details: aws.String(string(changeSetJSON)),
			},
		},
		ChangeSetName: aws.String(fmt.Sprintf("NovaCron-Update-%s", uuid.New().String())),
	}

	result, err := m.catalogClient.StartChangeSet(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to start change set: %w", err)
	}

	fmt.Printf("Started change set: %s\n", aws.ToString(result.ChangeSetId))

	return nil
}

// buildDimensionsChangeSet builds pricing dimension change set
func (m *MarketplaceManager) buildDimensionsChangeSet(pricing PricingModel) map[string]interface{} {
	dimensions := make([]map[string]interface{}, 0, len(pricing.Dimensions))

	for _, dim := range pricing.Dimensions {
		dimensions = append(dimensions, map[string]interface{}{
			"Key":         dim.Key,
			"Name":        dim.Name,
			"Description": dim.Description,
			"Unit":        dim.Unit,
			"Types":       []string{"Metered"},
		})
	}

	return map[string]interface{}{
		"Dimensions": dimensions,
	}
}

// MonitorMeteringHealth monitors metering service health
func (m *MarketplaceManager) MonitorMeteringHealth(ctx context.Context) error {
	// Test metering connectivity with a zero-quantity record
	testRecord := UsageRecord{
		CustomerIdentifier: "test-customer",
		Dimension:          "vm-hours",
		Quantity:           0,
		Timestamp:          time.Now(),
	}

	// Use DryRun to test without actual metering
	input := &marketplacemetering.MeterUsageInput{
		ProductCode:    aws.String(m.productCode),
		Timestamp:      aws.Time(testRecord.Timestamp),
		UsageDimension: aws.String(testRecord.Dimension),
		UsageQuantity:  aws.Int32(0),
		DryRun:         aws.Bool(true),
	}

	_, err := m.meteringClient.MeterUsage(ctx, input)
	if err != nil {
		return fmt.Errorf("metering health check failed: %w", err)
	}

	return nil
}

// GetCallerIdentity gets AWS account information
func (m *MarketplaceManager) GetCallerIdentity(ctx context.Context) (string, error) {
	result, err := m.stsClient.GetCallerIdentity(ctx, &sts.GetCallerIdentityInput{})
	if err != nil {
		return "", fmt.Errorf("failed to get caller identity: %w", err)
	}

	return aws.ToString(result.Account), nil
}
