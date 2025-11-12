package security

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"
)

// AIThreatDetector implements AI-powered threat detection
type AIThreatDetector struct {
	mu                  sync.RWMutex
	mlModels            map[string]*MLModel
	anomalyDetector     *AnomalyDetector
	behaviorAnalyzer    *BehaviorAnalyzer
	threatIntel         *ThreatIntelligence
	responseEngine      *AutomatedResponseEngine
	featureExtractor    *FeatureExtractor
	config              *AIThreatConfig
	metrics             *AIThreatMetrics
}

// AIThreatConfig configuration for AI threat detection
type AIThreatConfig struct {
	// ML models
	EnableMLDetection      bool
	ModelUpdateInterval    time.Duration
	MinModelAccuracy       float64
	EnableOnlineLearning   bool

	// Anomaly detection
	AnomalyThreshold       float64
	BaselineWindow         time.Duration
	EnableBehaviorAnalysis bool

	// Threat intelligence
	EnableThreatIntel      bool
	ThreatFeedURLs         []string
	ThreatFeedUpdateInterval time.Duration

	// Automated response
	EnableAutoResponse     bool
	ResponseConfidenceMin  float64
	EscalationThreshold    float64

	// Performance
	MaxConcurrentAnalysis  int
	AnalysisTimeout        time.Duration
	FeatureCacheSize       int
}

// MLModel represents a machine learning model
type MLModel struct {
	ID              string
	Name            string
	Type            MLModelType
	Version         string
	Accuracy        float64
	Precision       float64
	Recall          float64
	F1Score         float64
	Parameters      map[string]interface{}
	TrainingData    *TrainingDataset
	LastTrained     time.Time
	LastEvaluated   time.Time
	PredictionCount int64
	Status          ModelStatus
}

// MLModelType defines model types
type MLModelType string

const (
	ModelTypeRandomForest    MLModelType = "random_forest"
	ModelTypeNeuralNetwork   MLModelType = "neural_network"
	ModelTypeSVM             MLModelType = "svm"
	ModelTypeIsolationForest MLModelType = "isolation_forest"
	ModelTypeAutoencoder     MLModelType = "autoencoder"
	ModelTypeLSTM            MLModelType = "lstm"
)

// ModelStatus defines model status
type ModelStatus string

const (
	ModelStatusTraining  ModelStatus = "training"
	ModelStatusActive    ModelStatus = "active"
	ModelStatusEvaluating ModelStatus = "evaluating"
	ModelStatusDeprecated ModelStatus = "deprecated"
)

// TrainingDataset represents training dataset
type TrainingDataset struct {
	ID             string
	SampleCount    int
	FeatureCount   int
	LabelCount     int
	CreatedAt      time.Time
	LastUpdated    time.Time
}

// AnomalyDetector detects anomalous behavior
type AnomalyDetector struct {
	mu              sync.RWMutex
	baselines       map[string]*Baseline
	anomalies       map[string]*Anomaly
	detectors       []AnomalyDetectionAlgorithm
	threshold       float64
}

// Baseline represents normal behavior baseline
type Baseline struct {
	EntityID        string
	EntityType      string
	Metrics         map[string]*MetricBaseline
	SampleCount     int
	LastUpdated     time.Time
	Confidence      float64
}

// MetricBaseline represents metric baseline
type MetricBaseline struct {
	Name            string
	Mean            float64
	StdDev          float64
	Min             float64
	Max             float64
	Percentile95    float64
	Percentile99    float64
	SampleCount     int
}

// Anomaly represents detected anomaly
type Anomaly struct {
	ID              string
	EntityID        string
	EntityType      string
	Type            AnomalyType
	Severity        ThreatSeverity
	Score           float64
	Confidence      float64
	Features        map[string]float64
	Deviation       map[string]float64
	DetectedAt      time.Time
	Resolved        bool
	ResolvedAt      time.Time
	Response        *ThreatResponse
}

// AnomalyType defines anomaly types
type AnomalyType string

const (
	AnomalyTypePoint       AnomalyType = "point"       // Single data point anomaly
	AnomalyTypeContextual  AnomalyType = "contextual"  // Context-dependent anomaly
	AnomalyTypeCollective  AnomalyType = "collective"  // Group behavior anomaly
	AnomalyTypeTemporal    AnomalyType = "temporal"    // Time-series anomaly
)

// ThreatSeverity defines threat severity levels
type ThreatSeverity string

const (
	SeverityCritical ThreatSeverity = "critical"
	SeverityHigh     ThreatSeverity = "high"
	SeverityMedium   ThreatSeverity = "medium"
	SeverityLow      ThreatSeverity = "low"
	SeverityInfo     ThreatSeverity = "info"
)

// AnomalyDetectionAlgorithm interface for anomaly detection
type AnomalyDetectionAlgorithm interface {
	Detect(features map[string]float64, baseline *Baseline) (score float64, err error)
	GetName() string
}

// BehaviorAnalyzer analyzes entity behavior
type BehaviorAnalyzer struct {
	mu              sync.RWMutex
	profiles        map[string]*BehaviorProfile
	patterns        map[string]*BehaviorPattern
	analyzer        *PatternAnalyzer
}

// BehaviorPattern represents behavior pattern
type BehaviorPattern struct {
	ID              string
	Name            string
	Type            PatternType
	Features        []string
	Frequency       float64
	Confidence      float64
	FirstSeen       time.Time
	LastSeen        time.Time
	OccurrenceCount int64
	Malicious       bool
	Severity        ThreatSeverity
}

// PatternType defines pattern types
type PatternType string

const (
	PatternTypeNormal      PatternType = "normal"
	PatternTypeSuspicious  PatternType = "suspicious"
	PatternTypeMalicious   PatternType = "malicious"
	PatternTypeReconnaissance PatternType = "reconnaissance"
	PatternTypeExfiltration PatternType = "exfiltration"
	PatternTypeLateralMovement PatternType = "lateral_movement"
)

// PatternAnalyzer analyzes patterns
type PatternAnalyzer struct {
	mu              sync.RWMutex
	rules           map[string]*PatternRule
	sequences       map[string]*SequencePattern
}

// PatternRule represents pattern matching rule
type PatternRule struct {
	ID              string
	Name            string
	Conditions      []PatternCondition
	Action          ThreatResponseAction
	Priority        int
	Enabled         bool
}

// PatternCondition represents pattern condition
type PatternCondition struct {
	Field           string
	Operator        string
	Value           interface{}
	Weight          float64
}

// SequencePattern represents sequential behavior pattern
type SequencePattern struct {
	ID              string
	Events          []string
	TimeWindow      time.Duration
	Threshold       int
	Severity        ThreatSeverity
}

// ThreatIntelligence integrates threat intelligence
type ThreatIntelligence struct {
	mu              sync.RWMutex
	feeds           map[string]*ThreatFeed
	indicators      map[string]*ThreatIndicator
	reputation      *ReputationService
	enrichment      *ThreatEnrichment
}

// ThreatFeed represents threat intelligence feed
type ThreatFeed struct {
	ID              string
	Name            string
	URL             string
	Type            FeedType
	UpdateInterval  time.Duration
	LastUpdated     time.Time
	IndicatorCount  int
	Reliability     float64
	Active          bool
}

// FeedType defines feed types
type FeedType string

const (
	FeedTypeIP         FeedType = "ip"
	FeedTypeDomain     FeedType = "domain"
	FeedTypeURL        FeedType = "url"
	FeedTypeHash       FeedType = "hash"
	FeedTypeCVE        FeedType = "cve"
	FeedTypeBehavior   FeedType = "behavior"
)

// ThreatIndicator represents threat indicator
type ThreatIndicator struct {
	ID              string
	Type            IndicatorType
	Value           string
	Severity        ThreatSeverity
	Confidence      float64
	Source          string
	FirstSeen       time.Time
	LastSeen        time.Time
	Tags            []string
	Context         map[string]interface{}
}

// IndicatorType defines indicator types
type IndicatorType string

const (
	IndicatorTypeIP        IndicatorType = "ip"
	IndicatorTypeDomain    IndicatorType = "domain"
	IndicatorTypeURL       IndicatorType = "url"
	IndicatorTypeHash      IndicatorType = "hash"
	IndicatorTypeEmail     IndicatorType = "email"
	IndicatorTypeUserAgent IndicatorType = "user_agent"
)

// ReputationService provides reputation scoring
type ReputationService struct {
	mu              sync.RWMutex
	scores          map[string]*ReputationScore
	cache           map[string]*CachedReputation
}

// ReputationScore represents reputation score
type ReputationScore struct {
	Entity          string
	Score           float64
	Factors         map[string]float64
	LastUpdated     time.Time
	SampleCount     int
}

// CachedReputation represents cached reputation
type CachedReputation struct {
	Score           float64
	Timestamp       time.Time
	TTL             time.Duration
}

// ThreatEnrichment enriches threat data
type ThreatEnrichment struct {
	mu              sync.RWMutex
	enrichers       []ThreatEnricher
	cache           map[string]*EnrichedThreat
}

// ThreatEnricher interface for threat enrichment
type ThreatEnricher interface {
	Enrich(ctx context.Context, threat *Threat) (*EnrichedThreat, error)
	GetName() string
}

// EnrichedThreat represents enriched threat
type EnrichedThreat struct {
	Threat          *Threat
	Geolocation     *GeolocationData
	ASN             string
	Organization    string
	ThreatActors    []string
	Campaigns       []string
	TTPs            []string // MITRE ATT&CK TTPs
	CVEs            []string
	RelatedThreats  []string
	EnrichmentTime  time.Time
}

// GeolocationData represents geolocation
type GeolocationData struct {
	Country         string
	Region          string
	City            string
	Latitude        float64
	Longitude       float64
}

// AutomatedResponseEngine handles automated responses
type AutomatedResponseEngine struct {
	mu              sync.RWMutex
	responders      map[ThreatResponseAction]ThreatResponder
	responses       map[string]*ThreatResponse
	playbooks       map[string]*ResponsePlaybook
	escalation      *EscalationManager
}

// ThreatResponder interface for threat response
type ThreatResponder interface {
	Respond(ctx context.Context, threat *Threat) (*ThreatResponse, error)
	GetAction() ThreatResponseAction
}

// ThreatResponse represents threat response
type ThreatResponse struct {
	ID              string
	ThreatID        string
	Action          ThreatResponseAction
	Status          ResponseStatus
	StartedAt       time.Time
	CompletedAt     time.Time
	Success         bool
	Details         map[string]interface{}
	Artifacts       []string
}

// ThreatResponseAction defines response actions
type ThreatResponseAction string

const (
	ActionBlock            ThreatResponseAction = "block"
	ActionQuarantine       ThreatResponseAction = "quarantine"
	ActionAlert            ThreatResponseAction = "alert"
	ActionIsolate          ThreatResponseAction = "isolate"
	ActionThrottle         ThreatResponseAction = "throttle"
	ActionLog              ThreatResponseAction = "log"
	ActionEscalate         ThreatResponseAction = "escalate"
	ActionAutoRemediate    ThreatResponseAction = "auto_remediate"
)

// ResponseStatus defines response status
type ResponseStatus string

const (
	ResponsePending    ResponseStatus = "pending"
	ResponseInProgress ResponseStatus = "in_progress"
	ResponseCompleted  ResponseStatus = "completed"
	ResponseFailed     ResponseStatus = "failed"
)

// ResponsePlaybook represents response playbook
type ResponsePlaybook struct {
	ID              string
	Name            string
	ThreatTypes     []ThreatType
	Steps           []ResponseStep
	AutoExecute     bool
	RequireApproval bool
}

// ThreatType defines threat types
type ThreatType string

const (
	ThreatTypeMalware      ThreatType = "malware"
	ThreatTypeIntrusion    ThreatType = "intrusion"
	ThreatTypeDDoS         ThreatType = "ddos"
	ThreatTypeExfiltration ThreatType = "exfiltration"
	ThreatTypePhishing     ThreatType = "phishing"
	ThreatTypeRansomware   ThreatType = "ransomware"
	ThreatTypeAPT          ThreatType = "apt"
)

// ResponseStep represents response step
type ResponseStep struct {
	Order           int
	Action          ThreatResponseAction
	Parameters      map[string]interface{}
	Timeout         time.Duration
	RetryOnFailure  bool
	MaxRetries      int
}

// EscalationManager manages threat escalation
type EscalationManager struct {
	mu              sync.RWMutex
	escalationPaths map[ThreatSeverity]*EscalationPath
	escalations     map[string]*Escalation
}

// EscalationPath defines escalation path
type EscalationPath struct {
	Severity        ThreatSeverity
	Levels          []EscalationLevel
	Timeout         time.Duration
}

// EscalationLevel represents escalation level
type EscalationLevel struct {
	Level           int
	Contacts        []string
	Actions         []ThreatResponseAction
	Timeout         time.Duration
}

// Escalation represents active escalation
type Escalation struct {
	ID              string
	ThreatID        string
	Severity        ThreatSeverity
	CurrentLevel    int
	StartedAt       time.Time
	LastEscalated   time.Time
	Resolved        bool
	ResolvedAt      time.Time
}

// FeatureExtractor extracts features for ML
type FeatureExtractor struct {
	mu              sync.RWMutex
	extractors      map[string]FeatureExtractorFunc
	cache           map[string]*CachedFeatures
	cacheSize       int
}

// FeatureExtractorFunc extracts features
type FeatureExtractorFunc func(data interface{}) (map[string]float64, error)

// CachedFeatures represents cached features
type CachedFeatures struct {
	Features        map[string]float64
	Timestamp       time.Time
	TTL             time.Duration
}

// Threat represents detected threat
type Threat struct {
	ID              string
	Type            ThreatType
	Severity        ThreatSeverity
	Confidence      float64
	Source          string
	Target          string
	Description     string
	Indicators      []string
	Features        map[string]interface{}
	DetectedAt      time.Time
	LastSeen        time.Time
	Status          ThreatStatus
	MitigationSteps []string
}

// ThreatStatus defines threat status
type ThreatStatus string

const (
	ThreatActive    ThreatStatus = "active"
	ThreatMitigated ThreatStatus = "mitigated"
	ThreatResolved  ThreatStatus = "resolved"
	ThreatFalsePositive ThreatStatus = "false_positive"
)

// AIThreatMetrics contains metrics
type AIThreatMetrics struct {
	mu                      sync.RWMutex
	ThreatsDetected         int64
	AnomaliesDetected       int64
	FalsePositives          int64
	TruePositives           int64
	ResponsesExecuted       int64
	AverageDetectionTime    time.Duration
	AverageResponseTime     time.Duration
	ModelAccuracy           float64
	ThreatsByType           map[ThreatType]int64
	ThreatsBySeverity       map[ThreatSeverity]int64
	LastUpdated             time.Time
}

// NewAIThreatDetector creates AI threat detector
func NewAIThreatDetector(config *AIThreatConfig) *AIThreatDetector {
	detector := &AIThreatDetector{
		mlModels:         make(map[string]*MLModel),
		anomalyDetector:  NewAnomalyDetector(config.AnomalyThreshold),
		behaviorAnalyzer: NewBehaviorAnalyzer(),
		threatIntel:      NewThreatIntelligence(),
		responseEngine:   NewAutomatedResponseEngine(),
		featureExtractor: NewFeatureExtractor(config.FeatureCacheSize),
		config:           config,
		metrics: &AIThreatMetrics{
			ThreatsByType:     make(map[ThreatType]int64),
			ThreatsBySeverity: make(map[ThreatSeverity]int64),
		},
	}

	detector.initializeMLModels()
	detector.startBackgroundTasks()

	return detector
}

// NewAnomalyDetector creates anomaly detector
func NewAnomalyDetector(threshold float64) *AnomalyDetector {
	return &AnomalyDetector{
		baselines: make(map[string]*Baseline),
		anomalies: make(map[string]*Anomaly),
		detectors: make([]AnomalyDetectionAlgorithm, 0),
		threshold: threshold,
	}
}

// NewBehaviorAnalyzer creates behavior analyzer
func NewBehaviorAnalyzer() *BehaviorAnalyzer {
	return &BehaviorAnalyzer{
		profiles: make(map[string]*BehaviorProfile),
		patterns: make(map[string]*BehaviorPattern),
		analyzer: &PatternAnalyzer{
			rules:     make(map[string]*PatternRule),
			sequences: make(map[string]*SequencePattern),
		},
	}
}

// NewThreatIntelligence creates threat intelligence
func NewThreatIntelligence() *ThreatIntelligence {
	return &ThreatIntelligence{
		feeds:      make(map[string]*ThreatFeed),
		indicators: make(map[string]*ThreatIndicator),
		reputation: &ReputationService{
			scores: make(map[string]*ReputationScore),
			cache:  make(map[string]*CachedReputation),
		},
		enrichment: &ThreatEnrichment{
			enrichers: make([]ThreatEnricher, 0),
			cache:     make(map[string]*EnrichedThreat),
		},
	}
}

// NewAutomatedResponseEngine creates response engine
func NewAutomatedResponseEngine() *AutomatedResponseEngine {
	return &AutomatedResponseEngine{
		responders: make(map[ThreatResponseAction]ThreatResponder),
		responses:  make(map[string]*ThreatResponse),
		playbooks:  make(map[string]*ResponsePlaybook),
		escalation: &EscalationManager{
			escalationPaths: make(map[ThreatSeverity]*EscalationPath),
			escalations:     make(map[string]*Escalation),
		},
	}
}

// NewFeatureExtractor creates feature extractor
func NewFeatureExtractor(cacheSize int) *FeatureExtractor {
	return &FeatureExtractor{
		extractors: make(map[string]FeatureExtractorFunc),
		cache:      make(map[string]*CachedFeatures),
		cacheSize:  cacheSize,
	}
}

// DetectThreat detects threats in data
func (aitd *AIThreatDetector) DetectThreat(ctx context.Context, data interface{}) (*Threat, error) {
	startTime := time.Now()

	// Extract features
	features, err := aitd.featureExtractor.Extract(data)
	if err != nil {
		return nil, fmt.Errorf("failed to extract features: %w", err)
	}

	// Run ML models
	mlScore, err := aitd.runMLDetection(features)
	if err != nil {
		return nil, fmt.Errorf("ML detection failed: %w", err)
	}

	// Detect anomalies
	anomalyScore, anomaly := aitd.anomalyDetector.Detect(features)

	// Analyze behavior
	behaviorScore := aitd.behaviorAnalyzer.Analyze(features)

	// Check threat intelligence
	intelMatch := aitd.threatIntel.Check(features)

	// Calculate composite threat score
	threatScore := (mlScore*0.4 + anomalyScore*0.3 + behaviorScore*0.2)
	if intelMatch {
		threatScore += 0.1
	}

	// Determine if threat
	if threatScore > 0.7 {
		threat := &Threat{
			ID:          uuid.New().String(),
			Type:        aitd.classifyThreatType(features),
			Severity:    aitd.calculateSeverity(threatScore),
			Confidence:  threatScore,
			DetectedAt:  time.Now(),
			Status:      ThreatActive,
			Features:    convertFeaturesToInterface(features),
		}

		// Update metrics
		aitd.metrics.mu.Lock()
		aitd.metrics.ThreatsDetected++
		aitd.metrics.AverageDetectionTime = time.Since(startTime)
		aitd.metrics.ThreatsByType[threat.Type]++
		aitd.metrics.ThreatsBySeverity[threat.Severity]++
		aitd.metrics.LastUpdated = time.Now()
		aitd.metrics.mu.Unlock()

		// Automated response
		if aitd.config.EnableAutoResponse && threatScore > aitd.config.ResponseConfidenceMin {
			go aitd.responseEngine.Respond(ctx, threat)
		}

		return threat, nil
	}

	if anomaly != nil {
		aitd.metrics.mu.Lock()
		aitd.metrics.AnomaliesDetected++
		aitd.metrics.mu.Unlock()
	}

	return nil, nil
}

// Extract extracts features
func (fe *FeatureExtractor) Extract(data interface{}) (map[string]float64, error) {
	fe.mu.RLock()
	defer fe.mu.RUnlock()

	features := make(map[string]float64)

	for name, extractor := range fe.extractors {
		extracted, err := extractor(data)
		if err != nil {
			continue
		}

		for k, v := range extracted {
			features[fmt.Sprintf("%s_%s", name, k)] = v
		}
	}

	return features, nil
}

// runMLDetection runs ML detection
func (aitd *AIThreatDetector) runMLDetection(features map[string]float64) (float64, error) {
	aitd.mu.RLock()
	defer aitd.mu.RUnlock()

	var totalScore float64
	var modelCount int

	for _, model := range aitd.mlModels {
		if model.Status != ModelStatusActive {
			continue
		}

		score := aitd.predictWithModel(model, features)
		totalScore += score
		modelCount++
	}

	if modelCount == 0 {
		return 0, fmt.Errorf("no active ML models")
	}

	return totalScore / float64(modelCount), nil
}

// predictWithModel makes prediction
func (aitd *AIThreatDetector) predictWithModel(model *MLModel, features map[string]float64) float64 {
	// Simulate ML prediction
	// In production, use actual ML inference

	var sum float64
	for _, v := range features {
		sum += v
	}

	// Normalize
	score := sum / float64(len(features))
	score = math.Min(1.0, math.Max(0.0, score))

	model.PredictionCount++
	return score
}

// Detect detects anomalies
func (ad *AnomalyDetector) Detect(features map[string]float64) (float64, *Anomaly) {
	ad.mu.RLock()
	defer ad.mu.RUnlock()

	// Statistical anomaly detection
	score := ad.calculateAnomalyScore(features)

	if score > ad.threshold {
		anomaly := &Anomaly{
			ID:         uuid.New().String(),
			Type:       AnomalyTypePoint,
			Score:      score,
			Confidence: score,
			Features:   features,
			DetectedAt: time.Now(),
			Resolved:   false,
		}

		return score, anomaly
	}

	return score, nil
}

// calculateAnomalyScore calculates anomaly score
func (ad *AnomalyDetector) calculateAnomalyScore(features map[string]float64) float64 {
	// Z-score based anomaly detection
	var totalDeviation float64
	var count int

	for _, baseline := range ad.baselines {
		for name, value := range features {
			if metric, ok := baseline.Metrics[name]; ok {
				zscore := math.Abs((value - metric.Mean) / metric.StdDev)
				totalDeviation += zscore
				count++
			}
		}
	}

	if count == 0 {
		return 0
	}

	return math.Min(1.0, totalDeviation/float64(count)/3.0)
}

// Analyze analyzes behavior
func (ba *BehaviorAnalyzer) Analyze(features map[string]float64) float64 {
	ba.mu.RLock()
	defer ba.mu.RUnlock()

	// Pattern matching
	var matchScore float64

	for _, pattern := range ba.patterns {
		if pattern.Malicious {
			match := ba.matchPattern(features, pattern)
			if match > matchScore {
				matchScore = match
			}
		}
	}

	return matchScore
}

// matchPattern matches behavior pattern
func (ba *BehaviorAnalyzer) matchPattern(features map[string]float64, pattern *BehaviorPattern) float64 {
	// Simple pattern matching simulation
	// In production, use sophisticated pattern matching

	var matches int
	for _, feature := range pattern.Features {
		if _, ok := features[feature]; ok {
			matches++
		}
	}

	return float64(matches) / float64(len(pattern.Features))
}

// Check checks threat intelligence
func (ti *ThreatIntelligence) Check(features map[string]float64) bool {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	// Check indicators
	for _, indicator := range ti.indicators {
		// In production, match against actual indicators
		if indicator.Confidence > 0.8 {
			return true
		}
	}

	return false
}

// Respond executes automated response
func (are *AutomatedResponseEngine) Respond(ctx context.Context, threat *Threat) (*ThreatResponse, error) {
	are.mu.Lock()
	defer are.mu.Unlock()

	response := &ThreatResponse{
		ID:        uuid.New().String(),
		ThreatID:  threat.ID,
		Action:    aitd.determineAction(threat),
		Status:    ResponseInProgress,
		StartedAt: time.Now(),
		Details:   make(map[string]interface{}),
	}

	are.responses[response.ID] = response

	// Execute response
	go func() {
		// Simulate response execution
		time.Sleep(100 * time.Millisecond)

		response.Status = ResponseCompleted
		response.CompletedAt = time.Now()
		response.Success = true
	}()

	return response, nil
}

// classifyThreatType classifies threat type
func (aitd *AIThreatDetector) classifyThreatType(features map[string]float64) ThreatType {
	// Simple classification based on features
	// In production, use ML classification

	if features["network_requests"] > 100 {
		return ThreatTypeDDoS
	}
	if features["data_transfer"] > 1000 {
		return ThreatTypeExfiltration
	}

	return ThreatTypeIntrusion
}

// calculateSeverity calculates threat severity
func (aitd *AIThreatDetector) calculateSeverity(score float64) ThreatSeverity {
	switch {
	case score >= 0.9:
		return SeverityCritical
	case score >= 0.7:
		return SeverityHigh
	case score >= 0.5:
		return SeverityMedium
	case score >= 0.3:
		return SeverityLow
	default:
		return SeverityInfo
	}
}

// determineAction determines response action
func (aitd *AIThreatDetector) determineAction(threat *Threat) ThreatResponseAction {
	switch threat.Severity {
	case SeverityCritical:
		return ActionBlock
	case SeverityHigh:
		return ActionQuarantine
	case SeverityMedium:
		return ActionAlert
	default:
		return ActionLog
	}
}

// initializeMLModels initializes ML models
func (aitd *AIThreatDetector) initializeMLModels() {
	aitd.mu.Lock()
	defer aitd.mu.Unlock()

	// Random Forest model
	aitd.mlModels["random_forest"] = &MLModel{
		ID:       "rf-001",
		Name:     "Random Forest Threat Detector",
		Type:     ModelTypeRandomForest,
		Version:  "1.0.0",
		Accuracy: 0.95,
		Status:   ModelStatusActive,
	}

	// Neural Network model
	aitd.mlModels["neural_net"] = &MLModel{
		ID:       "nn-001",
		Name:     "Neural Network Threat Detector",
		Type:     ModelTypeNeuralNetwork,
		Version:  "1.0.0",
		Accuracy: 0.93,
		Status:   ModelStatusActive,
	}

	// Isolation Forest for anomaly detection
	aitd.mlModels["isolation_forest"] = &MLModel{
		ID:       "if-001",
		Name:     "Isolation Forest Anomaly Detector",
		Type:     ModelTypeIsolationForest,
		Version:  "1.0.0",
		Accuracy: 0.91,
		Status:   ModelStatusActive,
	}
}

// startBackgroundTasks starts background tasks
func (aitd *AIThreatDetector) startBackgroundTasks() {
	// Model retraining
	go aitd.runModelRetraining()

	// Threat intel updates
	go aitd.runThreatIntelUpdates()

	// Metrics collection
	go aitd.runMetricsCollection()
}

// runModelRetraining runs model retraining
func (aitd *AIThreatDetector) runModelRetraining() {
	ticker := time.NewTicker(aitd.config.ModelUpdateInterval)
	defer ticker.Stop()

	for range ticker.C {
		if !aitd.config.EnableOnlineLearning {
			continue
		}

		aitd.mu.Lock()
		for _, model := range aitd.mlModels {
			if model.Status == ModelStatusActive {
				// Simulate retraining
				model.LastTrained = time.Now()
			}
		}
		aitd.mu.Unlock()
	}
}

// runThreatIntelUpdates runs threat intel updates
func (aitd *AIThreatDetector) runThreatIntelUpdates() {
	ticker := time.NewTicker(aitd.config.ThreatFeedUpdateInterval)
	defer ticker.Stop()

	for range ticker.C {
		if !aitd.config.EnableThreatIntel {
			continue
		}

		aitd.threatIntel.mu.Lock()
		for _, feed := range aitd.threatIntel.feeds {
			if feed.Active {
				feed.LastUpdated = time.Now()
			}
		}
		aitd.threatIntel.mu.Unlock()
	}
}

// runMetricsCollection runs metrics collection
func (aitd *AIThreatDetector) runMetricsCollection() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		aitd.metrics.mu.Lock()

		// Calculate accuracy
		if aitd.metrics.TruePositives+aitd.metrics.FalsePositives > 0 {
			aitd.metrics.ModelAccuracy = float64(aitd.metrics.TruePositives) /
				float64(aitd.metrics.TruePositives+aitd.metrics.FalsePositives)
		}

		aitd.metrics.LastUpdated = time.Now()
		aitd.metrics.mu.Unlock()
	}
}

// GetMetrics returns metrics
func (aitd *AIThreatDetector) GetMetrics() *AIThreatMetrics {
	aitd.metrics.mu.RLock()
	defer aitd.metrics.mu.RUnlock()

	metricsCopy := *aitd.metrics
	return &metricsCopy
}

// Helper function
func convertFeaturesToInterface(features map[string]float64) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range features {
		result[k] = v
	}
	return result
}
