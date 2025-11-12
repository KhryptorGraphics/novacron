// Package verticals implements Vertical Market Domination Platform
// for achieving industry-specific penetration targets across Financial Services,
// Healthcare, Telecommunications, Retail, Manufacturing, and Energy sectors.
package verticals

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// VerticalMarket represents an industry vertical
type VerticalMarket struct {
	VerticalID           string                 `json:"vertical_id"`
	VerticalName         string                 `json:"vertical_name"`
	TotalMarketSize      float64                `json:"total_market_size"`      // TAM for vertical
	TargetPenetration    float64                `json:"target_penetration"`     // Target % of top companies
	CurrentPenetration   float64                `json:"current_penetration"`    // Current % achieved
	TopCompaniesCount    int                    `json:"top_companies_count"`    // e.g., top 100 banks
	CustomersAcquired    int                    `json:"customers_acquired"`
	TotalRevenue         float64                `json:"total_revenue"`
	AverageACV           float64                `json:"average_acv"`
	GrowthRate           float64                `json:"growth_rate"`
	ComplianceRequirements []string            `json:"compliance_requirements"`
	KeyUseCases          []string               `json:"key_use_cases"`
	CompetitiveLandscape map[string]float64     `json:"competitive_landscape"` // Competitor -> market share
	IndustryTrends       []string               `json:"industry_trends"`
	CustomAttributes     map[string]interface{} `json:"custom_attributes"`
	LastUpdated          time.Time              `json:"last_updated"`
}

// VerticalCustomer represents a customer in a specific vertical
type VerticalCustomer struct {
	CustomerID          string                 `json:"customer_id"`
	CompanyName         string                 `json:"company_name"`
	VerticalID          string                 `json:"vertical_id"`
	IndustryRank        int                    `json:"industry_rank"`        // Rank in industry (e.g., #3 bank)
	AnnualRevenue       float64                `json:"annual_revenue"`       // Customer's company revenue
	EmployeeCount       int                    `json:"employee_count"`
	ContractValue       float64                `json:"contract_value"`
	DeploymentSize      int                    `json:"deployment_size"`      // Node count
	UseCases            []string               `json:"use_cases"`
	ComplianceStatus    map[string]bool        `json:"compliance_status"`    // Regulation -> compliant
	ReferenceStatus     string                 `json:"reference_status"`     // willing, conditional, no
	StrategicValue      float64                `json:"strategic_value"`      // Reference value multiplier
	ExpansionPotential  float64                `json:"expansion_potential"`
	CompetitorReplaced  string                 `json:"competitor_replaced"`
	AcquisitionDate     time.Time              `json:"acquisition_date"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// ComplianceFramework represents industry regulations
type ComplianceFramework struct {
	FrameworkID          string    `json:"framework_id"`
	FrameworkName        string    `json:"framework_name"`
	Description          string    `json:"description"`
	ApplicableVerticals  []string  `json:"applicable_verticals"`
	Requirements         []string  `json:"requirements"`
	CertificationNeeded  bool      `json:"certification_needed"`
	AuditFrequency       string    `json:"audit_frequency"`
	PenaltyForNonCompliance string `json:"penalty_for_non_compliance"`
	ComplianceStatus     string    `json:"compliance_status"` // certified, in_progress, planned
	LastAuditDate        time.Time `json:"last_audit_date"`
	NextAuditDate        time.Time `json:"next_audit_date"`
	LastUpdated          time.Time `json:"last_updated"`
}

// VerticalSolution represents industry-specific product offering
type VerticalSolution struct {
	SolutionID          string                 `json:"solution_id"`
	SolutionName        string                 `json:"solution_name"`
	TargetVertical      string                 `json:"target_vertical"`
	Description         string                 `json:"description"`
	KeyFeatures         []string               `json:"key_features"`
	ComplianceSupport   []string               `json:"compliance_support"`
	IntegrationPartners []string               `json:"integration_partners"`
	Pricing             map[string]float64     `json:"pricing"`
	ROIMetrics          map[string]float64     `json:"roi_metrics"`
	CaseStudies         []string               `json:"case_studies"`
	CompetitiveAdvantages []string             `json:"competitive_advantages"`
	CustomAttributes    map[string]interface{} `json:"custom_attributes"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// VerticalDominationPlatform manages industry-specific market penetration
type VerticalDominationPlatform struct {
	mu                  sync.RWMutex
	verticals           map[string]*VerticalMarket
	customers           map[string]*VerticalCustomer
	complianceFrameworks map[string]*ComplianceFramework
	solutions           map[string]*VerticalSolution
	penetrationMetrics  *PenetrationMetrics
	ctx                 context.Context
	cancel              context.CancelFunc
}

// PenetrationMetrics tracks vertical market performance
type PenetrationMetrics struct {
	OverallPenetration       float64            `json:"overall_penetration"`
	PenetrationByVertical    map[string]float64 `json:"penetration_by_vertical"`
	RevenueByVertical        map[string]float64 `json:"revenue_by_vertical"`
	GrowthByVertical         map[string]float64 `json:"growth_by_vertical"`
	Top100Penetration        map[string]int     `json:"top100_penetration"`      // Vertical -> count of top 100
	ComplianceCertifications map[string]int     `json:"compliance_certifications"`
	ReferenceCustomers       map[string]int     `json:"reference_customers"`
	CompetitiveWins          map[string]int     `json:"competitive_wins"`
	LastUpdated              time.Time          `json:"last_updated"`
}

// NewVerticalDominationPlatform creates the vertical market domination system
func NewVerticalDominationPlatform() *VerticalDominationPlatform {
	ctx, cancel := context.WithCancel(context.Background())

	platform := &VerticalDominationPlatform{
		verticals:           make(map[string]*VerticalMarket),
		customers:           make(map[string]*VerticalCustomer),
		complianceFrameworks: make(map[string]*ComplianceFramework),
		solutions:           make(map[string]*VerticalSolution),
		penetrationMetrics:  &PenetrationMetrics{},
		ctx:                 ctx,
		cancel:              cancel,
	}

	platform.initializeVerticals()
	platform.initializeComplianceFrameworks()
	platform.initializeSolutions()

	return platform
}

// initializeVerticals sets up industry verticals with targets
func (p *VerticalDominationPlatform) initializeVerticals() {
	p.verticals["financial-services"] = &VerticalMarket{
		VerticalID:         "financial-services",
		VerticalName:       "Financial Services & Banking",
		TotalMarketSize:    25_000_000_000, // $25B TAM
		TargetPenetration:  0.80,           // 80% of top 100 banks
		CurrentPenetration: 0.45,           // 45% current
		TopCompaniesCount:  100,            // Top 100 banks
		CustomersAcquired:  45,
		TotalRevenue:       3_000_000_000,
		AverageACV:         6_000_000,
		GrowthRate:         0.35,
		ComplianceRequirements: []string{"PCI-DSS", "SOX", "Basel III", "GDPR", "FINRA"},
		KeyUseCases: []string{
			"Core banking infrastructure",
			"Trading platforms",
			"Payment processing",
			"Risk management systems",
			"Fraud detection",
			"Mobile banking",
		},
		CompetitiveLandscape: map[string]float64{
			"ibm-mainframe":  0.30,
			"oracle-exadata": 0.25,
			"vmware":         0.20,
			"others":         0.25,
		},
		IndustryTrends: []string{
			"Digital transformation acceleration",
			"Real-time payments",
			"Open banking APIs",
			"AI-powered fraud detection",
			"Cloud migration",
		},
		LastUpdated: time.Now(),
	}

	p.verticals["healthcare"] = &VerticalMarket{
		VerticalID:         "healthcare",
		VerticalName:       "Healthcare & Life Sciences",
		TotalMarketSize:    18_000_000_000,
		TargetPenetration:  0.70, // 70% of top 100 hospitals
		CurrentPenetration: 0.40,
		TopCompaniesCount:  100,
		CustomersAcquired:  40,
		TotalRevenue:       2_000_000_000,
		AverageACV:         5_000_000,
		GrowthRate:         0.40,
		ComplianceRequirements: []string{"HIPAA", "HITECH", "FDA 21 CFR Part 11", "GDPR"},
		KeyUseCases: []string{
			"Electronic health records (EHR)",
			"Medical imaging (PACS)",
			"Genomics research",
			"Drug discovery platforms",
			"Telemedicine infrastructure",
			"Clinical trials management",
		},
		CompetitiveLandscape: map[string]float64{
			"epic-systems":  0.25,
			"cerner":        0.20,
			"vmware":        0.18,
			"others":        0.37,
		},
		IndustryTrends: []string{
			"Telemedicine explosion",
			"AI diagnostics",
			"Precision medicine",
			"Interoperability standards",
			"Value-based care",
		},
		LastUpdated: time.Now(),
	}

	p.verticals["telecommunications"] = &VerticalMarket{
		VerticalID:         "telecommunications",
		VerticalName:       "Telecommunications & 5G",
		TotalMarketSize:    20_000_000_000,
		TargetPenetration:  0.75, // 75% of global carriers
		CurrentPenetration: 0.50,
		TopCompaniesCount:  50, // Top 50 global carriers
		CustomersAcquired:  38,
		TotalRevenue:       2_500_000_000,
		AverageACV:         8_000_000,
		GrowthRate:         0.45,
		ComplianceRequirements: []string{"ETSI", "3GPP", "GSMA", "FCC", "GDPR"},
		KeyUseCases: []string{
			"5G core network (vRAN)",
			"Network functions virtualization (NFV)",
			"Edge computing",
			"Content delivery networks",
			"IoT platforms",
			"Network slicing",
		},
		CompetitiveLandscape: map[string]float64{
			"ericsson":  0.22,
			"nokia":     0.20,
			"huawei":    0.15,
			"others":    0.43,
		},
		IndustryTrends: []string{
			"5G rollout acceleration",
			"Open RAN adoption",
			"Edge computing growth",
			"Network automation",
			"Private 5G networks",
		},
		LastUpdated: time.Now(),
	}

	p.verticals["retail"] = &VerticalMarket{
		VerticalID:         "retail",
		VerticalName:       "Retail & E-Commerce",
		TotalMarketSize:    15_000_000_000,
		TargetPenetration:  0.60, // 60% of Fortune 500 retailers
		CurrentPenetration: 0.35,
		TopCompaniesCount:  100, // Fortune 500 retailers
		CustomersAcquired:  35,
		TotalRevenue:       1_500_000_000,
		AverageACV:         4_000_000,
		GrowthRate:         0.50,
		ComplianceRequirements: []string{"PCI-DSS", "GDPR", "CCPA", "SOX"},
		KeyUseCases: []string{
			"E-commerce platforms",
			"Point-of-sale systems",
			"Inventory management",
			"Supply chain optimization",
			"Customer analytics",
			"Omnichannel retail",
		},
		CompetitiveLandscape: map[string]float64{
			"salesforce": 0.25,
			"sap":        0.20,
			"aws":        0.18,
			"others":     0.37,
		},
		IndustryTrends: []string{
			"Omnichannel transformation",
			"AI-powered personalization",
			"Contactless payments",
			"Social commerce",
			"Sustainability focus",
		},
		LastUpdated: time.Now(),
	}

	p.verticals["manufacturing"] = &VerticalMarket{
		VerticalID:         "manufacturing",
		VerticalName:       "Manufacturing & Industrial IoT",
		TotalMarketSize:    22_000_000_000,
		TargetPenetration:  0.65, // 65% of Industrial IoT deployments
		CurrentPenetration: 0.42,
		TopCompaniesCount:  200, // Top 200 manufacturers
		CustomersAcquired:  84,
		TotalRevenue:       2_800_000_000,
		AverageACV:         5_500_000,
		GrowthRate:         0.38,
		ComplianceRequirements: []string{"ISO 9001", "ISO 27001", "ISA/IEC 62443", "GDPR"},
		KeyUseCases: []string{
			"Industrial IoT platforms",
			"Predictive maintenance",
			"Digital twin simulations",
			"Supply chain visibility",
			"Quality control automation",
			"Smart factory orchestration",
		},
		CompetitiveLandscape: map[string]float64{
			"siemens":   0.20,
			"ge":        0.18,
			"rockwell":  0.15,
			"others":    0.47,
		},
		IndustryTrends: []string{
			"Industry 4.0 adoption",
			"Edge computing in factories",
			"AI quality control",
			"Sustainable manufacturing",
			"Supply chain resilience",
		},
		LastUpdated: time.Now(),
	}

	p.verticals["energy"] = &VerticalMarket{
		VerticalID:         "energy",
		VerticalName:       "Energy & Smart Grid",
		TotalMarketSize:    16_000_000_000,
		TargetPenetration:  0.70, // 70% of smart grid deployments
		CurrentPenetration: 0.48,
		TopCompaniesCount:  150, // Top 150 utilities
		CustomersAcquired:  72,
		TotalRevenue:       2_200_000_000,
		AverageACV:         4_500_000,
		GrowthRate:         0.42,
		ComplianceRequirements: []string{"NERC CIP", "ISO 50001", "FERC", "EPA"},
		KeyUseCases: []string{
			"Smart grid management",
			"Renewable energy integration",
			"Distribution automation",
			"Demand response systems",
			"EV charging infrastructure",
			"Energy trading platforms",
		},
		CompetitiveLandscape: map[string]float64{
			"ge":        0.22,
			"schneider": 0.20,
			"abb":       0.18,
			"others":    0.40,
		},
		IndustryTrends: []string{
			"Renewable energy growth",
			"Grid modernization",
			"EV charging networks",
			"Distributed energy resources",
			"Energy storage systems",
		},
		LastUpdated: time.Now(),
	}
}

// initializeComplianceFrameworks sets up regulatory requirements
func (p *VerticalDominationPlatform) initializeComplianceFrameworks() {
	p.complianceFrameworks["pci-dss"] = &ComplianceFramework{
		FrameworkID:          "pci-dss",
		FrameworkName:        "PCI-DSS 4.0",
		Description:          "Payment Card Industry Data Security Standard",
		ApplicableVerticals:  []string{"financial-services", "retail"},
		Requirements: []string{
			"Encrypted cardholder data storage",
			"Secure network architecture",
			"Access control implementation",
			"Regular security monitoring",
		},
		CertificationNeeded: true,
		AuditFrequency:      "annual",
		PenaltyForNonCompliance: "Up to $500K per incident + card network fines",
		ComplianceStatus:    "certified",
		LastAuditDate:       time.Date(2024, 9, 1, 0, 0, 0, 0, time.UTC),
		NextAuditDate:       time.Date(2025, 9, 1, 0, 0, 0, 0, time.UTC),
		LastUpdated:         time.Now(),
	}

	p.complianceFrameworks["hipaa"] = &ComplianceFramework{
		FrameworkID:          "hipaa",
		FrameworkName:        "HIPAA/HITECH",
		Description:          "Health Insurance Portability and Accountability Act",
		ApplicableVerticals:  []string{"healthcare"},
		Requirements: []string{
			"PHI encryption at rest and in transit",
			"Access controls and audit logs",
			"Business associate agreements",
			"Breach notification procedures",
		},
		CertificationNeeded: true,
		AuditFrequency:      "annual",
		PenaltyForNonCompliance: "Up to $1.5M per violation category per year",
		ComplianceStatus:    "certified",
		LastAuditDate:       time.Date(2024, 10, 15, 0, 0, 0, 0, time.UTC),
		NextAuditDate:       time.Date(2025, 10, 15, 0, 0, 0, 0, time.UTC),
		LastUpdated:         time.Now(),
	}

	p.complianceFrameworks["nerc-cip"] = &ComplianceFramework{
		FrameworkID:          "nerc-cip",
		FrameworkName:        "NERC CIP",
		Description:          "North American Electric Reliability Corporation Critical Infrastructure Protection",
		ApplicableVerticals:  []string{"energy"},
		Requirements: []string{
			"Electronic security perimeters",
			"System security management",
			"Personnel and training requirements",
			"Incident reporting and response",
		},
		CertificationNeeded: true,
		AuditFrequency:      "annual",
		PenaltyForNonCompliance: "Up to $1M per violation per day",
		ComplianceStatus:    "certified",
		LastAuditDate:       time.Date(2024, 8, 20, 0, 0, 0, 0, time.UTC),
		NextAuditDate:       time.Date(2025, 8, 20, 0, 0, 0, 0, time.UTC),
		LastUpdated:         time.Now(),
	}
}

// initializeSolutions creates industry-specific product offerings
func (p *VerticalDominationPlatform) initializeSolutions() {
	p.solutions["banking-cloud"] = &VerticalSolution{
		SolutionID:     "banking-cloud",
		SolutionName:   "NovaCron Banking Cloud",
		TargetVertical: "financial-services",
		Description:    "Compliant, high-performance infrastructure for core banking and trading platforms",
		KeyFeatures: []string{
			"PCI-DSS certified infrastructure",
			"Real-time transaction processing (100K+ TPS)",
			"HSM integration for key management",
			"Disaster recovery with RPO=0, RTO<60s",
			"Multi-region active-active deployment",
		},
		ComplianceSupport: []string{"PCI-DSS", "SOX", "Basel III", "GDPR", "FINRA"},
		IntegrationPartners: []string{"Fiserv", "Temenos", "FIS", "Jack Henry"},
		Pricing: map[string]float64{
			"base_platform":    200000, // $200K/month
			"per_node":         5000,
			"compliance_addon": 50000,
		},
		ROIMetrics: map[string]float64{
			"cost_savings":         0.40, // 40% infrastructure cost reduction
			"time_to_market":       0.60, // 60% faster deployment
			"availability_sla":     0.9999, // 99.99% uptime
			"transaction_speedup":  3.5,  // 3.5x transaction performance
		},
		CaseStudies: []string{
			"Top 5 US Bank: 70% cost reduction, 4x performance",
			"European Fintech: 80% faster time-to-market",
		},
		CompetitiveAdvantages: []string{
			"30% lower TCO vs mainframes",
			"Built-in PCI-DSS compliance",
			"Real-time replication",
		},
		LastUpdated: time.Now(),
	}

	p.solutions["healthcare-platform"] = &VerticalSolution{
		SolutionID:     "healthcare-platform",
		SolutionName:   "NovaCron Healthcare Platform",
		TargetVertical: "healthcare",
		Description:    "HIPAA-compliant infrastructure for EHR, PACS, and clinical research",
		KeyFeatures: []string{
			"HIPAA/HITECH certified",
			"High-performance medical imaging (DICOM)",
			"Genomics processing acceleration",
			"PHI encryption and access controls",
			"Multi-site clinical trial coordination",
		},
		ComplianceSupport: []string{"HIPAA", "HITECH", "FDA 21 CFR Part 11", "GDPR"},
		IntegrationPartners: []string{"Epic", "Cerner", "Philips", "GE Healthcare"},
		Pricing: map[string]float64{
			"base_platform":    150000,
			"per_node":         4000,
			"hipaa_addon":      40000,
		},
		ROIMetrics: map[string]float64{
			"cost_savings":      0.45,
			"imaging_speedup":   5.0, // 5x faster PACS
			"genomics_speedup":  10.0, // 10x genomics processing
			"availability_sla":  0.9999,
		},
		CaseStudies: []string{
			"Top 10 US Hospital: 5x PACS performance, 50% cost reduction",
			"Cancer Research Center: 10x genomics processing speedup",
		},
		CompetitiveAdvantages: []string{
			"Built-in HIPAA compliance",
			"Medical imaging optimization",
			"Genomics acceleration",
		},
		LastUpdated: time.Now(),
	}
}

// AddVerticalCustomer adds a customer to vertical tracking
func (p *VerticalDominationPlatform) AddVerticalCustomer(customer *VerticalCustomer) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Validate vertical exists
	if _, exists := p.verticals[customer.VerticalID]; !exists {
		return fmt.Errorf("vertical %s not found", customer.VerticalID)
	}

	customer.LastUpdated = time.Now()
	p.customers[customer.CustomerID] = customer

	// Update vertical metrics
	p.updateVerticalMetrics(customer.VerticalID)

	return nil
}

// updateVerticalMetrics recalculates penetration for a vertical
func (p *VerticalDominationPlatform) updateVerticalMetrics(verticalID string) {
	vertical := p.verticals[verticalID]

	customersInVertical := 0
	totalRevenue := 0.0

	for _, customer := range p.customers {
		if customer.VerticalID == verticalID {
			customersInVertical++
			totalRevenue += customer.ContractValue
		}
	}

	vertical.CustomersAcquired = customersInVertical
	vertical.TotalRevenue = totalRevenue
	vertical.CurrentPenetration = float64(customersInVertical) / float64(vertical.TopCompaniesCount)
	vertical.LastUpdated = time.Now()
}

// CalculatePenetrationMetrics computes vertical market performance
func (p *VerticalDominationPlatform) CalculatePenetrationMetrics() *PenetrationMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	penetrationByVertical := make(map[string]float64)
	revenueByVertical := make(map[string]float64)
	growthByVertical := make(map[string]float64)
	top100Penetration := make(map[string]int)
	complianceCertifications := make(map[string]int)
	referenceCustomers := make(map[string]int)
	competitiveWins := make(map[string]int)

	var totalCustomers int
	var totalTopCompanies int

	for verticalID, vertical := range p.verticals {
		penetrationByVertical[verticalID] = vertical.CurrentPenetration
		revenueByVertical[verticalID] = vertical.TotalRevenue
		growthByVertical[verticalID] = vertical.GrowthRate
		top100Penetration[verticalID] = vertical.CustomersAcquired

		totalCustomers += vertical.CustomersAcquired
		totalTopCompanies += vertical.TopCompaniesCount

		// Count compliance certifications
		for _, framework := range p.complianceFrameworks {
			for _, applicableVertical := range framework.ApplicableVerticals {
				if applicableVertical == verticalID && framework.ComplianceStatus == "certified" {
					complianceCertifications[verticalID]++
				}
			}
		}

		// Count reference customers
		for _, customer := range p.customers {
			if customer.VerticalID == verticalID && customer.ReferenceStatus == "willing" {
				referenceCustomers[verticalID]++
			}
			if customer.VerticalID == verticalID && customer.CompetitorReplaced != "" {
				competitiveWins[verticalID]++
			}
		}
	}

	overallPenetration := 0.0
	if totalTopCompanies > 0 {
		overallPenetration = float64(totalCustomers) / float64(totalTopCompanies)
	}

	metrics := &PenetrationMetrics{
		OverallPenetration:       overallPenetration,
		PenetrationByVertical:    penetrationByVertical,
		RevenueByVertical:        revenueByVertical,
		GrowthByVertical:         growthByVertical,
		Top100Penetration:        top100Penetration,
		ComplianceCertifications: complianceCertifications,
		ReferenceCustomers:       referenceCustomers,
		CompetitiveWins:          competitiveWins,
		LastUpdated:              time.Now(),
	}

	p.penetrationMetrics = metrics
	return metrics
}

// GetVerticalsByPenetration returns verticals ranked by penetration gap
func (p *VerticalDominationPlatform) GetVerticalsByPenetration() []*VerticalMarket {
	p.mu.RLock()
	defer p.mu.RUnlock()

	verticals := make([]*VerticalMarket, 0, len(p.verticals))
	for _, vertical := range p.verticals {
		verticals = append(verticals, vertical)
	}

	// Sort by penetration gap (target - current)
	sort.Slice(verticals, func(i, j int) bool {
		gapI := verticals[i].TargetPenetration - verticals[i].CurrentPenetration
		gapJ := verticals[j].TargetPenetration - verticals[j].CurrentPenetration
		return gapI > gapJ
	})

	return verticals
}

// GetTopCustomersByVertical returns highest-value customers in a vertical
func (p *VerticalDominationPlatform) GetTopCustomersByVertical(verticalID string, limit int) []*VerticalCustomer {
	p.mu.RLock()
	defer p.mu.RUnlock()

	customers := make([]*VerticalCustomer, 0)
	for _, customer := range p.customers {
		if customer.VerticalID == verticalID {
			customers = append(customers, customer)
		}
	}

	sort.Slice(customers, func(i, j int) bool {
		return customers[i].ContractValue > customers[j].ContractValue
	})

	if limit > len(customers) {
		limit = len(customers)
	}

	return customers[:limit]
}

// ProjectVerticalGrowth forecasts vertical penetration
func (p *VerticalDominationPlatform) ProjectVerticalGrowth(verticalID string, quarters int) []VerticalProjection {
	p.mu.RLock()
	defer p.mu.RUnlock()

	vertical := p.verticals[verticalID]
	projections := make([]VerticalProjection, quarters)

	currentPenetration := vertical.CurrentPenetration

	for i := 0; i < quarters; i++ {
		remainingQuarters := float64(quarters - i)
		quarterlyGrowth := math.Pow(vertical.TargetPenetration/currentPenetration, 1/remainingQuarters)

		currentPenetration *= quarterlyGrowth

		projections[i] = VerticalProjection{
			Quarter:            i + 1,
			ProjectedPenetration: currentPenetration,
			ProjectedCustomers: int(currentPenetration * float64(vertical.TopCompaniesCount)),
			ProjectedRevenue:   currentPenetration * float64(vertical.TopCompaniesCount) * vertical.AverageACV,
			Confidence:         p.calculateVerticalConfidence(i),
			LastUpdated:        time.Now(),
		}
	}

	return projections
}

// calculateVerticalConfidence determines forecast confidence
func (p *VerticalDominationPlatform) calculateVerticalConfidence(quartersOut int) float64 {
	baseConfidence := 0.90
	decayRate := 0.025
	return baseConfidence * math.Exp(-decayRate*float64(quartersOut))
}

// VerticalProjection represents forecasted vertical performance
type VerticalProjection struct {
	Quarter              int       `json:"quarter"`
	ProjectedPenetration float64   `json:"projected_penetration"`
	ProjectedCustomers   int       `json:"projected_customers"`
	ProjectedRevenue     float64   `json:"projected_revenue"`
	Confidence           float64   `json:"confidence"`
	LastUpdated          time.Time `json:"last_updated"`
}

// GenerateVerticalReport creates comprehensive vertical market report
func (p *VerticalDominationPlatform) GenerateVerticalReport() *VerticalReport {
	p.mu.RLock()
	defer p.mu.RUnlock()

	metrics := p.CalculatePenetrationMetrics()
	verticals := p.GetVerticalsByPenetration()

	// Get projections for all verticals
	projections := make(map[string][]VerticalProjection)
	for verticalID := range p.verticals {
		projections[verticalID] = p.ProjectVerticalGrowth(verticalID, 12)
	}

	report := &VerticalReport{
		Metrics:         metrics,
		Verticals:       verticals,
		Solutions:       p.getSolutionsList(),
		Compliance:      p.getComplianceList(),
		Projections:     projections,
		HealthStatus:    p.assessVerticalHealth(),
		Recommendations: p.generateVerticalRecommendations(),
		GeneratedAt:     time.Now(),
	}

	return report
}

// getSolutionsList returns all vertical solutions
func (p *VerticalDominationPlatform) getSolutionsList() []*VerticalSolution {
	solutions := make([]*VerticalSolution, 0, len(p.solutions))
	for _, solution := range p.solutions {
		solutions = append(solutions, solution)
	}
	return solutions
}

// getComplianceList returns all compliance frameworks
func (p *VerticalDominationPlatform) getComplianceList() []*ComplianceFramework {
	frameworks := make([]*ComplianceFramework, 0, len(p.complianceFrameworks))
	for _, framework := range p.complianceFrameworks {
		frameworks = append(frameworks, framework)
	}
	return frameworks
}

// assessVerticalHealth evaluates vertical market progress
func (p *VerticalDominationPlatform) assessVerticalHealth() string {
	metrics := p.penetrationMetrics

	if metrics.OverallPenetration >= 0.60 {
		return "excellent"
	} else if metrics.OverallPenetration >= 0.50 {
		return "good"
	} else if metrics.OverallPenetration >= 0.40 {
		return "fair"
	}

	return "needs_attention"
}

// generateVerticalRecommendations creates strategic recommendations
func (p *VerticalDominationPlatform) generateVerticalRecommendations() []string {
	recommendations := []string{}

	// Analyze each vertical
	for _, vertical := range p.verticals {
		gap := vertical.TargetPenetration - vertical.CurrentPenetration
		if gap > 0.20 { // More than 20% gap
			recommendations = append(recommendations,
				fmt.Sprintf("🎯 %s: %.0f%% penetration gap - accelerate to %.0f%% target",
					vertical.VerticalName, gap*100, vertical.TargetPenetration*100))
		}

		// Check compliance
		certCount := 0
		for _, framework := range p.complianceFrameworks {
			for _, applicableVertical := range framework.ApplicableVerticals {
				if applicableVertical == vertical.VerticalID && framework.ComplianceStatus == "certified" {
					certCount++
				}
			}
		}

		if certCount < len(vertical.ComplianceRequirements) {
			recommendations = append(recommendations,
				fmt.Sprintf("📋 %s: Complete remaining %d compliance certifications",
					vertical.VerticalName, len(vertical.ComplianceRequirements)-certCount))
		}
	}

	return recommendations
}

// VerticalReport represents comprehensive vertical market intelligence
type VerticalReport struct {
	Metrics         *PenetrationMetrics              `json:"metrics"`
	Verticals       []*VerticalMarket                `json:"verticals"`
	Solutions       []*VerticalSolution              `json:"solutions"`
	Compliance      []*ComplianceFramework           `json:"compliance"`
	Projections     map[string][]VerticalProjection  `json:"projections"`
	HealthStatus    string                           `json:"health_status"`
	Recommendations []string                         `json:"recommendations"`
	GeneratedAt     time.Time                        `json:"generated_at"`
}

// ExportMetrics exports vertical metrics in JSON format
func (p *VerticalDominationPlatform) ExportMetrics() ([]byte, error) {
	report := p.GenerateVerticalReport()
	return json.MarshalIndent(report, "", "  ")
}

// Close shuts down the vertical domination platform
func (p *VerticalDominationPlatform) Close() error {
	p.cancel()
	return nil
}
