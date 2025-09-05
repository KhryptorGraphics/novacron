package controllers

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/pricing"
	"go.uber.org/zap"
)

// CostOptimizer manages cost-aware scaling decisions
type CostOptimizer struct {
	mu sync.RWMutex
	
	// Cloud provider clients
	ec2Client     *ec2.EC2
	pricingClient *pricing.Pricing
	
	// Pricing models
	onDemandPricing  *PricingModel
	spotPricing      *SpotPricingModel
	reservedPricing  *ReservedPricingModel
	savingsPlan      *SavingsPlanModel
	
	// Spot instance management
	spotManager      *SpotInstanceManager
	bidOptimizer     *BidOptimizer
	
	// Cost tracking
	costTracker      *CostTracker
	budgetManager    *BudgetManager
	
	// Optimization engine
	optimizer        *CostOptimizationEngine
	recommendations  []*CostRecommendation
	
	// ML models
	pricePredictorML   *SpotPricePredictor
	demandPredictor  *DemandPredictor
	
	// Configuration
	config          *CostOptimizerConfig
	logger          *zap.Logger
}

// SpotInstanceManager handles spot instance lifecycle
type SpotInstanceManager struct {
	mu sync.RWMutex
	
	// Spot fleet management
	spotFleets      map[string]*SpotFleet
	bidStrategies   map[string]BidStrategy
	
	// Interruption handling
	interruptHandler *InterruptionHandler
	checkpointer     *StateCheckpointer
	
	// Availability tracking
	availabilityZones map[string]*ZoneAvailability
	instanceTypes     map[string]*InstanceTypeAvailability
	
	// Fallback management
	fallbackManager  *FallbackManager
	
	// Diversification
	diversifier      *InstanceDiversifier
}

// BidOptimizer optimizes spot instance bidding
type BidOptimizer struct {
	// ML models for price prediction
	pricePredictor   *SpotPricePredictor
	volatilityModel  *VolatilityPredictor
	
	// Bidding strategies
	strategies       map[string]BiddingAlgorithm
	activeStrategy   BiddingAlgorithm
	
	// Historical data
	priceHistory     *PriceHistory
	interruptHistory *InterruptionHistory
	
	// Risk management
	riskTolerance    float64
	maxBidMultiplier float64
}

// CostOptimizationEngine performs cost optimization
type CostOptimizationEngine struct {
	// Optimization algorithms
	linearOptimizer  *LinearProgramSolver
	geneticOptimizer *GeneticAlgorithm
	simulatedAnnealing *SimulatedAnnealing
	
	// Constraints
	budgetConstraint    float64
	performanceConstraint float64
	availabilityConstraint float64
}

// PricingModel represents instance pricing
type PricingModel struct {
	Region          string
	InstancePrices  map[string]float64 // instance type -> hourly price
	DataTransferCost float64
	StorageCost     float64
	LastUpdated     time.Time
}

// SpotPricingModel tracks spot instance pricing
type SpotPricingModel struct {
	CurrentPrices   map[string]map[string]float64 // AZ -> instance type -> price
	PriceHistory    map[string][]*PricePoint
	Volatility      map[string]float64
	PredictedPrices map[string]map[string]float64
}

// SpotFleet represents a spot fleet configuration
type SpotFleet struct {
	ID               string
	TargetCapacity   int
	CurrentCapacity  int
	LaunchTemplates  []*LaunchTemplate
	AllocationStrategy string
	InstancePools    int
	OnDemandBase     int
	SpotMaxPrice     float64
	Status           string
}

// InterruptionHandler manages spot interruptions
type InterruptionHandler struct {
	// Interruption detection
	detector        *InterruptionDetector
	
	// Grace period management
	gracePeriod     time.Duration
	
	// Checkpoint and migration
	checkpointer    *StateCheckpointer
	migrator        *WorkloadMigrator
	
	// Notification channels
	notifications   chan *InterruptionNotice
}

// NewCostOptimizer creates a cost optimization system
func NewCostOptimizer(config *CostOptimizerConfig, logger *zap.Logger) *CostOptimizer {
	co := &CostOptimizer{
		config: config,
		logger: logger,
		recommendations: make([]*CostRecommendation, 0),
	}
	
	// Initialize pricing models
	co.onDemandPricing = &PricingModel{
		Region:         config.Region,
		InstancePrices: make(map[string]float64),
	}
	
	co.spotPricing = &SpotPricingModel{
		CurrentPrices:   make(map[string]map[string]float64),
		PriceHistory:    make(map[string][]*PricePoint),
		Volatility:      make(map[string]float64),
		PredictedPrices: make(map[string]map[string]float64),
	}
	
	co.reservedPricing = NewReservedPricingModel()
	co.savingsPlan = NewSavingsPlanModel()
	
	// Initialize spot manager
	co.spotManager = &SpotInstanceManager{
		spotFleets:        make(map[string]*SpotFleet),
		bidStrategies:     make(map[string]BidStrategy),
		availabilityZones: make(map[string]*ZoneAvailability),
		instanceTypes:     make(map[string]*InstanceTypeAvailability),
		interruptHandler:  NewInterruptionHandler(config.GracePeriod),
		checkpointer:      NewStateCheckpointer(),
		fallbackManager:   NewFallbackManager(),
		diversifier:       NewInstanceDiversifier(),
	}
	
	// Initialize bid optimizer
	co.bidOptimizer = &BidOptimizer{
		pricePredictor:   NewSpotPricePredictor(),
		volatilityModel:  NewVolatilityPredictor(),
		strategies:       make(map[string]BiddingAlgorithm),
		priceHistory:     NewPriceHistory(),
		interruptHistory: NewInterruptionHistory(),
		riskTolerance:    config.RiskTolerance,
		maxBidMultiplier: config.MaxBidMultiplier,
	}
	
	// Initialize optimization engine
	co.optimizer = &CostOptimizationEngine{
		linearOptimizer:        NewLinearProgramSolver(),
		geneticOptimizer:       NewGeneticAlgorithm(),
		simulatedAnnealing:     NewSimulatedAnnealing(),
		budgetConstraint:       config.MaxHourlyCost,
		performanceConstraint:  config.MinPerformance,
		availabilityConstraint: config.MinAvailability,
	}
	
	// Initialize cost tracking
	co.costTracker = NewCostTracker()
	co.budgetManager = NewBudgetManager(config.Budgets)
	
	// Initialize ML models
	co.pricePredictorML = NewSpotPricePredictor()
	co.demandPredictor = NewDemandPredictor()
	
	// Initialize bidding strategies
	co.initializeBiddingStrategies()
	
	// Start background tasks
	go co.updatePricingData()
	go co.monitorSpotInterruptions()
	go co.optimizeContinuously()
	
	return co
}

// OptimizeInstanceMix determines optimal mix of instance types
func (co *CostOptimizer) OptimizeInstanceMix(ctx context.Context, requirements *ScalingRequirements) (*InstanceMix, error) {
	co.mu.Lock()
	defer co.mu.Unlock()
	
	// Get current pricing
	onDemandPrices := co.onDemandPricing.InstancePrices
	spotPrices := co.getCurrentSpotPrices()
	
	// Predict future spot prices
	predictedPrices := co.pricePredictorML.PredictPrices(requirements.Duration)
	
	// Calculate volatility for each instance type
	volatility := co.calculateVolatility(spotPrices)
	
	// Optimize instance mix using linear programming
	mix := co.optimizer.OptimizeMix(&OptimizationInput{
		Requirements:     requirements,
		OnDemandPrices:  onDemandPrices,
		SpotPrices:      spotPrices,
		PredictedPrices: predictedPrices,
		Volatility:      volatility,
		Budget:          co.config.MaxHourlyCost,
	})
	
	// Apply diversification for resilience
	mix = co.spotManager.diversifier.Diversify(mix)
	
	// Validate against budget
	if err := co.validateBudget(mix); err != nil {
		return nil, err
	}
	
	return mix, nil
}

// CreateSpotFleet creates an optimized spot fleet
func (co *CostOptimizer) CreateSpotFleet(ctx context.Context, config *SpotFleetConfig) (*SpotFleet, error) {
	// Determine optimal bid prices
	bidPrices := co.calculateOptimalBids(config.InstanceTypes)
	
	// Create launch templates with diversification
	launchTemplates := co.createLaunchTemplates(config, bidPrices)
	
	// Create spot fleet request
	fleet := &SpotFleet{
		ID:                 generateFleetID(),
		TargetCapacity:     config.TargetCapacity,
		LaunchTemplates:    launchTemplates,
		AllocationStrategy: co.selectAllocationStrategy(config),
		InstancePools:      co.calculateInstancePools(config),
		OnDemandBase:       config.OnDemandBase,
		SpotMaxPrice:       co.calculateMaxPrice(bidPrices),
	}
	
	// Register fleet with manager
	co.spotManager.spotFleets[fleet.ID] = fleet
	
	// Set up interruption handling
	co.spotManager.interruptHandler.RegisterFleet(fleet.ID)
	
	// Start monitoring
	go co.monitorFleet(fleet)
	
	return fleet, nil
}

// HandleSpotInterruption handles spot instance interruption
func (co *CostOptimizer) HandleSpotInterruption(notice *InterruptionNotice) error {
	co.logger.Warn("Spot interruption detected",
		zap.String("instance", notice.InstanceID),
		zap.Duration("time_remaining", notice.TimeRemaining))
	
	// Checkpoint current state
	if err := co.spotManager.checkpointer.Checkpoint(notice.InstanceID); err != nil {
		co.logger.Error("Failed to checkpoint", zap.Error(err))
	}
	
	// Find replacement capacity
	replacement := co.findReplacementCapacity(notice)
	
	// Launch replacement instances
	if replacement != nil {
		if err := co.launchReplacement(replacement); err != nil {
			// Fallback to on-demand
			return co.fallbackToOnDemand(notice)
		}
	}
	
	// Migrate workload
	if err := co.migrateWorkload(notice.InstanceID, replacement.InstanceID); err != nil {
		return fmt.Errorf("failed to migrate workload: %v", err)
	}
	
	return nil
}

// calculateOptimalBids determines optimal bid prices
func (co *CostOptimizer) calculateOptimalBids(instanceTypes []string) map[string]float64 {
	bids := make(map[string]float64)
	
	for _, instanceType := range instanceTypes {
		// Get price history
		history := co.bidOptimizer.priceHistory.GetHistory(instanceType)
		
		// Predict future price
		prediction := co.pricePredictorML.Predict(instanceType, 1*time.Hour)
		
		// Calculate volatility
		volatility := co.bidOptimizer.volatilityModel.Calculate(history)
		
		// Apply bidding strategy
		bid := co.bidOptimizer.activeStrategy.CalculateBid(&BidInput{
			InstanceType:    instanceType,
			PriceHistory:    history,
			PredictedPrice:  prediction.Price,
			Volatility:      volatility,
			OnDemandPrice:   co.onDemandPricing.InstancePrices[instanceType],
			RiskTolerance:   co.bidOptimizer.riskTolerance,
		})
		
		// Apply bid ceiling
		maxBid := co.onDemandPricing.InstancePrices[instanceType] * co.bidOptimizer.maxBidMultiplier
		if bid > maxBid {
			bid = maxBid
		}
		
		bids[instanceType] = bid
	}
	
	return bids
}

// initializeBiddingStrategies sets up bidding algorithms
func (co *CostOptimizer) initializeBiddingStrategies() {
	// Conservative strategy - bid at 70% of on-demand
	co.bidOptimizer.strategies["conservative"] = &ConservativeBidding{
		Multiplier: 0.7,
	}
	
	// Aggressive strategy - bid at 90% of on-demand
	co.bidOptimizer.strategies["aggressive"] = &AggressiveBidding{
		Multiplier: 0.9,
	}
	
	// Adaptive strategy - ML-based bidding
	co.bidOptimizer.strategies["adaptive"] = &AdaptiveBidding{
		Predictor: co.pricePredictorML,
		MinMultiplier: 0.5,
		MaxMultiplier: 0.95,
	}
	
	// Volatility-aware strategy
	co.bidOptimizer.strategies["volatility_aware"] = &VolatilityAwareBidding{
		BaseMultiplier: 0.8,
		VolatilityAdjustment: 0.2,
	}
	
	// Set default strategy
	co.bidOptimizer.activeStrategy = co.bidOptimizer.strategies[co.config.BidStrategy]
}

// updatePricingData continuously updates pricing information
func (co *CostOptimizer) updatePricingData() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		// Update on-demand prices
		co.updateOnDemandPrices()
		
		// Update spot prices
		co.updateSpotPrices()
		
		// Update reserved instance pricing
		co.updateReservedPricing()
		
		// Train ML models
		co.trainPricePredictors()
	}
}

// monitorSpotInterruptions monitors for spot interruptions
func (co *CostOptimizer) monitorSpotInterruptions() {
	for notice := range co.spotManager.interruptHandler.notifications {
		// Handle interruption
		if err := co.HandleSpotInterruption(notice); err != nil {
			co.logger.Error("Failed to handle interruption", zap.Error(err))
		}
	}
}

// optimizeContinuously performs continuous cost optimization
func (co *CostOptimizer) optimizeContinuously() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		// Analyze current costs
		currentCost := co.costTracker.GetHourlyCost()
		
		// Check budget compliance
		if overBudget := co.budgetManager.CheckBudget(currentCost); overBudget {
			co.handleBudgetOverrun()
		}
		
		// Generate optimization recommendations
		recommendations := co.generateRecommendations()
		
		// Apply high-confidence recommendations automatically
		for _, rec := range recommendations {
			if rec.Confidence > 0.9 && rec.AutoApply {
				co.applyRecommendation(rec)
			}
		}
		
		// Store recommendations for review
		co.mu.Lock()
		co.recommendations = recommendations
		co.mu.Unlock()
	}
}

// Recommendation generates cost optimization recommendations
func (co *CostOptimizer) generateRecommendations() []*CostRecommendation {
	recommendations := []*CostRecommendation{}
	
	// Check for reserved instance opportunities
	if rec := co.checkReservedInstanceOpportunity(); rec != nil {
		recommendations = append(recommendations, rec)
	}
	
	// Check for spot instance opportunities
	if rec := co.checkSpotOpportunity(); rec != nil {
		recommendations = append(recommendations, rec)
	}
	
	// Check for right-sizing opportunities
	if rec := co.checkRightSizingOpportunity(); rec != nil {
		recommendations = append(recommendations, rec)
	}
	
	// Check for savings plan opportunities
	if rec := co.checkSavingsPlanOpportunity(); rec != nil {
		recommendations = append(recommendations, rec)
	}
	
	// Sort by potential savings
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].EstimatedSavings > recommendations[j].EstimatedSavings
	})
	
	return recommendations
}

// Helper structures

type CostOptimizerConfig struct {
	Region           string
	MaxHourlyCost    float64
	MinPerformance   float64
	MinAvailability  float64
	RiskTolerance    float64
	MaxBidMultiplier float64
	BidStrategy      string
	GracePeriod      time.Duration
	Budgets          []*Budget
}

type InstanceMix struct {
	OnDemandInstances map[string]int // instance type -> count
	SpotInstances     map[string]int
	ReservedInstances map[string]int
	TotalCost         float64
	SavingsPercent    float64
}

type SpotFleetConfig struct {
	TargetCapacity int
	InstanceTypes  []string
	OnDemandBase   int
	MaxPrice       float64
	Duration       time.Duration
}

type InterruptionNotice struct {
	InstanceID    string
	TimeRemaining time.Duration
	Timestamp     time.Time
}

type CostRecommendation struct {
	ID               string
	Type             string
	Description      string
	EstimatedSavings float64
	Implementation   string
	Risk             string
	Confidence       float64
	AutoApply        bool
}

type BidStrategy interface {
	CalculateBid(input *BidInput) float64
}

type BidInput struct {
	InstanceType   string
	PriceHistory   []*PricePoint
	PredictedPrice float64
	Volatility     float64
	OnDemandPrice  float64
	RiskTolerance  float64
}

type PricePoint struct {
	Timestamp time.Time
	Price     float64
	AZ        string
}