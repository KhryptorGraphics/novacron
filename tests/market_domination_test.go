// Package tests implements comprehensive test suites for Phase 12 Market Domination platform
package tests

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/khryptorgraphics/novacron/backend/business/revenue"
	"github.com/khryptorgraphics/novacron/backend/competitive"
	"github.com/khryptorgraphics/novacron/backend/business/verticals"
	"github.com/khryptorgraphics/novacron/backend/partners"
)

// TestRevenueAccelerationEngine tests the $1B ARR revenue acceleration engine
func TestRevenueAccelerationEngine(t *testing.T) {
	engine := revenue.NewRevenueAccelerationEngine()
	defer engine.Close()

	t.Run("InitialTargets", func(t *testing.T) {
		// Verify initial targets
		assert.Equal(t, 1_000_000_000.0, engine.GetTarget().ARRGoal, "$1B ARR target")
		assert.Equal(t, 120_000_000.0, engine.GetTarget().CurrentARR, "$120M current ARR")
		assert.Equal(t, 1.50, engine.GetTarget().ExpansionNRR, "150% net revenue retention")
		assert.Equal(t, 5_000_000.0, engine.GetTarget().EnterpriseACVTarget, "$5M+ ACV target")
		assert.Equal(t, 50000, engine.GetTarget().NewCustomersRequired, "50,000 new customers")
	})

	t.Run("CustomerSegments", func(t *testing.T) {
		// Test customer segmentation
		segments := engine.GetSegments()
		require.NotEmpty(t, segments, "Should have customer segments")

		// Verify Fortune 500 segment
		var f500Segment *revenue.CustomerSegment
		for _, seg := range segments {
			if seg.SegmentName == "Fortune 500 Enterprise" {
				f500Segment = seg
				break
			}
		}
		require.NotNil(t, f500Segment, "Fortune 500 segment should exist")
		assert.Equal(t, 150, f500Segment.CustomerCount, "150 Fortune 500 customers")
		assert.GreaterOrEqual(t, f500Segment.NetRetention, 1.50, "150%+ NRR")
	})

	t.Run("EnterpriseDeal", func(t *testing.T) {
		// Test $5M+ enterprise deal
		deal := &revenue.EnterpriseDeal{
			DealID:              "enterprise-001",
			AccountName:         "Fortune 100 Bank",
			Fortune500Rank:      50,
			TotalContractValue:  30_000_000,
			AnnualValue:         10_000_000,
			ContractTerm:        36,
			CloseDate:           time.Now().AddDate(0, 2, 0),
			SalesStage:          "negotiation",
			Probability:         0.85,
			CompetitorDisplaced: "VMware",
			VerticalMarket:      "financial-services",
			NodeCount:           1000,
			ExpansionPotential:  20_000_000,
			StrategicValue:      2.5,
		}

		err := engine.AddEnterpriseDeal(deal)
		require.NoError(t, err, "Should add enterprise deal")

		// Verify deal tracking
		topDeals := engine.GetTopDeals(10)
		require.NotEmpty(t, topDeals, "Should have deals")
		assert.GreaterOrEqual(t, topDeals[0].AnnualValue, 5_000_000.0, "$5M+ ACV")
	})

	t.Run("ExpansionRevenue", func(t *testing.T) {
		// Test expansion revenue tracking
		expansion := &revenue.ExpansionRevenue{
			CustomerID:          "customer-001",
			CurrentARR:          2_000_000,
			ExpansionARR:        3_000_000,
			ExpansionType:       "upsell",
			OpportunityValue:    1_000_000,
			ExpansionVelocity:   45,
			ProductAdoption:     0.85,
			HealthScore:         0.92,
			ChurnRisk:           0.05,
			ExpansionReadiness:  0.95,
			RecommendedActions:  []string{"Propose additional node expansion", "Cross-sell DR solution"},
			ExpectedCloseDate:   time.Now().AddDate(0, 1, 15),
		}

		engine.TrackExpansionOpportunity(expansion)

		// Verify expansion tracking
		opportunities := engine.GetExpansionOpportunities(10)
		require.NotEmpty(t, opportunities, "Should have expansion opportunities")
	})

	t.Run("RevenueMetrics", func(t *testing.T) {
		// Test revenue metrics calculation
		metrics := engine.CalculateRevenueMetrics()
		require.NotNil(t, metrics, "Should calculate metrics")

		assert.Greater(t, metrics.CurrentARR, 0.0, "Should have current ARR")
		assert.GreaterOrEqual(t, metrics.NetRevenueRetention, 1.40, "140%+ NRR")
		assert.GreaterOrEqual(t, metrics.LTVCACRatio, 8.0, "8:1+ LTV:CAC ratio")
		assert.GreaterOrEqual(t, metrics.OnTrackPercentage, 75.0, "75%+ on track")
	})

	t.Run("RevenueProjection", func(t *testing.T) {
		// Test 24-month revenue projection to $1B
		projections := engine.ProjectRevenue(24)
		require.Len(t, projections, 24, "Should have 24 months projection")

		// Verify growth trajectory
		lastProjection := projections[23]
		assert.GreaterOrEqual(t, lastProjection.ProjectedARR, 900_000_000.0, "Close to $1B by month 24")
		assert.GreaterOrEqual(t, lastProjection.Confidence, 0.50, "Reasonable confidence")
	})

	t.Run("RevenueReport", func(t *testing.T) {
		// Test comprehensive revenue report
		report := engine.GenerateRevenueReport()
		require.NotNil(t, report, "Should generate report")

		assert.NotEmpty(t, report.Segments, "Should have segments")
		assert.NotEmpty(t, report.AccelerationStrategies, "Should have strategies")
		assert.NotEmpty(t, report.RevenueProjections, "Should have projections")
		assert.NotEmpty(t, report.Recommendations, "Should have recommendations")
	})
}

// TestMarketShareTracker tests the 50%+ market share tracking system
func TestMarketShareTracker(t *testing.T) {
	tracker := competitive.NewMarketShareTracker()
	defer tracker.Close()

	t.Run("MarketShareTarget", func(t *testing.T) {
		target := tracker.GetTarget()
		assert.Equal(t, 0.35, target.CurrentShare, "35% current market share")
		assert.Equal(t, 0.50, target.TargetShare, "50%+ target market share")
		assert.Equal(t, 1, target.MarketPosition, "#1 market position")
	})

	t.Run("Competitors", func(t *testing.T) {
		// Test competitive landscape
		competitors := tracker.GetTopCompetitors(10)
		require.NotEmpty(t, competitors, "Should have competitors")

		// Verify VMware competitor
		var vmware *competitive.CompetitorProfile
		for _, comp := range competitors {
			if comp.CompetitorID == "vmware" {
				vmware = comp
				break
			}
		}
		require.NotNil(t, vmware, "VMware competitor should exist")
		assert.GreaterOrEqual(t, vmware.CompetitiveWinRate, 0.70, "70%+ win rate vs VMware")
		assert.True(t, vmware.DisplacementTarget, "VMware is displacement target")
	})

	t.Run("CompetitiveWin", func(t *testing.T) {
		// Test competitive win tracking
		win := &competitive.CompetitiveWin{
			WinID:               "win-001",
			AccountName:         "Global Bank",
			CompetitorDisplaced: "vmware",
			DealValue:           8_000_000,
			WinReason:           []string{"Lower TCO", "Better performance", "Simpler management"},
			DisplacementType:    "full_replacement",
			TimeToWin:           120,
			DiscountOffered:     0.20,
			MigrationComplexity: "medium",
			CustomerSatisfaction: 0.95,
			WinDate:             time.Now(),
		}

		tracker.RecordCompetitiveWin(win)

		// Verify win tracking
		recentWins := tracker.GetCompetitiveWins(10)
		require.NotEmpty(t, recentWins, "Should have competitive wins")
	})

	t.Run("AcquisitionPipeline", func(t *testing.T) {
		// Test M&A acquisition pipeline
		pipeline := tracker.GetAcquisitionPipeline()
		require.NotEmpty(t, pipeline, "Should have acquisition targets")

		// Verify Nutanix acquisition target
		var nutanix *competitive.AcquisitionTarget
		for _, target := range pipeline {
			if target.TargetID == "nutanix" {
				nutanix = target
				break
			}
		}
		require.NotNil(t, nutanix, "Nutanix should be acquisition target")
		assert.Greater(t, nutanix.EstimatedValuation, 0.0, "Should have valuation")
		assert.Greater(t, nutanix.SynergyValue, 0.0, "Should have synergy value")
	})

	t.Run("MarketMetrics", func(t *testing.T) {
		// Test market metrics calculation
		metrics := tracker.CalculateMarketMetrics()
		require.NotNil(t, metrics, "Should calculate metrics")

		assert.GreaterOrEqual(t, metrics.CurrentMarketShare, 0.30, "30%+ market share")
		assert.GreaterOrEqual(t, metrics.CompetitiveWinRate, 0.85, "85%+ competitive win rate")
		assert.GreaterOrEqual(t, metrics.MarketLeadershipScore, 0.70, "70%+ leadership score")
	})

	t.Run("MarketProjection", func(t *testing.T) {
		// Test market share projection to 50%+
		projections := tracker.ProjectMarketShare(12)
		require.Len(t, projections, 12, "Should have 12 quarters projection")

		// Verify growth trajectory to 50%+
		lastProjection := projections[11]
		assert.GreaterOrEqual(t, lastProjection.ProjectedShare, 0.48, "Close to 50% by Q12")
	})

	t.Run("MarketReport", func(t *testing.T) {
		// Test comprehensive market report
		report := tracker.GenerateMarketReport()
		require.NotNil(t, report, "Should generate report")

		assert.NotEmpty(t, report.TopCompetitors, "Should have competitors")
		assert.NotEmpty(t, report.AcquisitionTargets, "Should have acquisition targets")
		assert.NotEmpty(t, report.MarketProjections, "Should have projections")
		assert.NotEmpty(t, report.Recommendations, "Should have recommendations")
	})
}

// TestVerticalDominationPlatform tests industry vertical market penetration
func TestVerticalDominationPlatform(t *testing.T) {
	platform := verticals.NewVerticalDominationPlatform()
	defer platform.Close()

	t.Run("VerticalMarkets", func(t *testing.T) {
		// Test vertical markets setup
		allVerticals := platform.GetVerticalsByPenetration()
		require.NotEmpty(t, allVerticals, "Should have vertical markets")

		// Verify Financial Services vertical
		var financial *verticals.VerticalMarket
		for _, v := range allVerticals {
			if v.VerticalID == "financial-services" {
				financial = v
				break
			}
		}
		require.NotNil(t, financial, "Financial services vertical should exist")
		assert.Equal(t, 0.80, financial.TargetPenetration, "80% target penetration")
		assert.Equal(t, 100, financial.TopCompaniesCount, "Top 100 banks")
	})

	t.Run("VerticalCustomer", func(t *testing.T) {
		// Test vertical customer tracking
		customer := &verticals.VerticalCustomer{
			CustomerID:          "vert-cust-001",
			CompanyName:         "Top 5 US Bank",
			VerticalID:          "financial-services",
			IndustryRank:        5,
			AnnualRevenue:       100_000_000_000,
			EmployeeCount:       250000,
			ContractValue:       8_000_000,
			DeploymentSize:      1500,
			UseCases:            []string{"Core banking", "Trading platforms", "Risk management"},
			ComplianceStatus:    map[string]bool{"PCI-DSS": true, "SOX": true, "Basel III": true},
			ReferenceStatus:     "willing",
			StrategicValue:      3.0,
			ExpansionPotential:  15_000_000,
			CompetitorReplaced:  "VMware",
			AcquisitionDate:     time.Now().AddDate(-1, 0, 0),
		}

		err := platform.AddVerticalCustomer(customer)
		require.NoError(t, err, "Should add vertical customer")
	})

	t.Run("ComplianceFrameworks", func(t *testing.T) {
		// Test compliance framework tracking
		report := platform.GenerateVerticalReport()
		require.NotEmpty(t, report.Compliance, "Should have compliance frameworks")

		// Verify PCI-DSS compliance
		var pciDSS *verticals.ComplianceFramework
		for _, framework := range report.Compliance {
			if framework.FrameworkID == "pci-dss" {
				pciDSS = framework
				break
			}
		}
		require.NotNil(t, pciDSS, "PCI-DSS framework should exist")
		assert.Equal(t, "certified", pciDSS.ComplianceStatus, "Should be certified")
	})

	t.Run("VerticalSolutions", func(t *testing.T) {
		// Test industry-specific solutions
		report := platform.GenerateVerticalReport()
		require.NotEmpty(t, report.Solutions, "Should have vertical solutions")

		// Verify Banking Cloud solution
		var bankingCloud *verticals.VerticalSolution
		for _, solution := range report.Solutions {
			if solution.SolutionID == "banking-cloud" {
				bankingCloud = solution
				break
			}
		}
		require.NotNil(t, bankingCloud, "Banking Cloud solution should exist")
		assert.NotEmpty(t, bankingCloud.ComplianceSupport, "Should have compliance support")
	})

	t.Run("PenetrationMetrics", func(t *testing.T) {
		// Test penetration metrics calculation
		metrics := platform.CalculatePenetrationMetrics()
		require.NotNil(t, metrics, "Should calculate metrics")

		assert.Greater(t, metrics.OverallPenetration, 0.0, "Should have penetration")
		assert.NotEmpty(t, metrics.PenetrationByVertical, "Should have per-vertical penetration")
	})

	t.Run("VerticalProjection", func(t *testing.T) {
		// Test vertical growth projection
		projections := platform.ProjectVerticalGrowth("financial-services", 12)
		require.Len(t, projections, 12, "Should have 12 quarters projection")

		// Verify growth trajectory
		lastProjection := projections[11]
		assert.GreaterOrEqual(t, lastProjection.ProjectedPenetration, 0.75, "75%+ penetration by Q12")
	})

	t.Run("VerticalReport", func(t *testing.T) {
		// Test comprehensive vertical report
		report := platform.GenerateVerticalReport()
		require.NotNil(t, report, "Should generate report")

		assert.NotEmpty(t, report.Verticals, "Should have verticals")
		assert.NotEmpty(t, report.Solutions, "Should have solutions")
		assert.NotEmpty(t, report.Compliance, "Should have compliance frameworks")
		assert.NotEmpty(t, report.Recommendations, "Should have recommendations")
	})
}

// TestPartnerEcosystemPlatform tests 5,000+ partner management
func TestPartnerEcosystemPlatform(t *testing.T) {
	platform := partners.NewPartnerEcosystemPlatform()
	defer platform.Close()

	t.Run("PartnerPrograms", func(t *testing.T) {
		// Test partner program tiers
		report := platform.GenerateEcosystemReport()
		require.NotEmpty(t, report.Programs, "Should have partner programs")

		// Verify Platinum program
		var platinum *partners.PartnerProgram
		for _, program := range report.Programs {
			if program.Tier == partners.TierPlatinum {
				platinum = program
				break
			}
		}
		require.NotNil(t, platinum, "Platinum program should exist")
		assert.GreaterOrEqual(t, platinum.CommissionRates["direct"], 0.25, "25%+ commission")
		assert.Greater(t, platinum.MDFBudget, 0.0, "Should have MDF budget")
	})

	t.Run("PartnerProfile", func(t *testing.T) {
		// Test partner profile management
		partner := &partners.PartnerProfile{
			PartnerID:          "partner-001",
			CompanyName:        "Enterprise Solutions Inc",
			PartnerType:        partners.TypeSystemIntegrator,
			Tier:               partners.TierGold,
			Status:             "active",
			EmployeeCount:      500,
			AnnualRevenue:      50_000_000,
			Geography:          []string{"North America", "Europe"},
			Verticals:          []string{"financial-services", "healthcare"},
			TechnicalCerts:     25,
			TotalRevenue:       5_000_000,
			QuarterlyRevenue:   1_500_000,
			DealCount:          15,
			AverageDealSize:    500_000,
			WinRate:            0.75,
			ProgramCompliance:  0.95,
			StrategicPartner:   true,
			PartnerManager:     "PAM-001",
			JoinedDate:         time.Now().AddDate(-2, 0, 0),
		}

		err := platform.AddPartner(partner)
		require.NoError(t, err, "Should add partner")

		// Verify partner tracking
		topPartners := platform.GetTopPartners(10)
		require.NotEmpty(t, topPartners, "Should have partners")
	})

	t.Run("PartnerDeal", func(t *testing.T) {
		// Test partner deal registration
		deal := &partners.PartnerDeal{
			DealID:            "pdeal-001",
			PartnerID:         "partner-001",
			AccountName:       "Enterprise Corp",
			DealValue:         1_000_000,
			DealType:          "co_sell",
			Stage:             "proposal",
			Probability:       0.70,
			ExpectedCloseDate: time.Now().AddDate(0, 2, 0),
			Status:            "open",
		}

		err := platform.RegisterDeal(deal)
		require.NoError(t, err, "Should register deal")
	})

	t.Run("CoSellingOpportunity", func(t *testing.T) {
		// Test co-selling engagement
		opp := &partners.CoSellingOpportunity{
			OpportunityID:     "cosell-001",
			PartnerID:         "partner-001",
			AccountName:       "Fortune 500 Company",
			OpportunityValue:  5_000_000,
			PartnerRole:       "lead",
			CoSellingStage:    "qualification",
			ResourcesNeeded:   []string{"Sales engineer", "Solution architect"},
			SalesTeamAssigned: "Enterprise-Team-A",
			Status:            "active",
		}

		err := platform.CreateCoSellingOpportunity(opp)
		require.NoError(t, err, "Should create co-selling opportunity")
	})

	t.Run("EcosystemMetrics", func(t *testing.T) {
		// Test ecosystem metrics calculation
		metrics := platform.CalculateEcosystemMetrics()
		require.NotNil(t, metrics, "Should calculate metrics")

		assert.Greater(t, metrics.TotalPartners, 0, "Should have partners")
		assert.NotEmpty(t, metrics.PartnersByTier, "Should have partners by tier")
		assert.NotEmpty(t, metrics.PartnersByType, "Should have partners by type")
	})

	t.Run("PartnerProjection", func(t *testing.T) {
		// Test partner growth projection to 5,000+
		projections := platform.ProjectPartnerGrowth(12)
		require.Len(t, projections, 12, "Should have 12 quarters projection")

		// Verify growth trajectory
		lastProjection := projections[11]
		assert.GreaterOrEqual(t, lastProjection.ProjectedPartners, 4000, "4000+ partners by Q12")
	})

	t.Run("EcosystemReport", func(t *testing.T) {
		// Test comprehensive ecosystem report
		report := platform.GenerateEcosystemReport()
		require.NotNil(t, report, "Should generate report")

		assert.NotEmpty(t, report.Programs, "Should have programs")
		assert.NotEmpty(t, report.Projections, "Should have projections")
		assert.NotEmpty(t, report.Recommendations, "Should have recommendations")
	})
}

// BenchmarkRevenueCalculation benchmarks revenue metric calculations
func BenchmarkRevenueCalculation(b *testing.B) {
	engine := revenue.NewRevenueAccelerationEngine()
	defer engine.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.CalculateRevenueMetrics()
	}
}

// BenchmarkMarketShareTracking benchmarks market share calculations
func BenchmarkMarketShareTracking(b *testing.B) {
	tracker := competitive.NewMarketShareTracker()
	defer tracker.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tracker.CalculateMarketMetrics()
	}
}

// BenchmarkVerticalPenetration benchmarks vertical penetration calculations
func BenchmarkVerticalPenetration(b *testing.B) {
	platform := verticals.NewVerticalDominationPlatform()
	defer platform.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		platform.CalculatePenetrationMetrics()
	}
}

// BenchmarkPartnerEcosystem benchmarks partner ecosystem calculations
func BenchmarkPartnerEcosystem(b *testing.B) {
	platform := partners.NewPartnerEcosystemPlatform()
	defer platform.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		platform.CalculateEcosystemMetrics()
	}
}
