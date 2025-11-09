package automl

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// AutoMLEngine provides automated machine learning capabilities
type AutoMLEngine struct {
	config      *AutoMLConfig
	models      map[string]ModelBuilder
	evaluator   *ModelEvaluator
	featureEng  *FeatureEngineer
	mu          sync.RWMutex
	bestModel   *TrainedModel
	experiments []Experiment
}

// AutoMLConfig defines AutoML configuration
type AutoMLConfig struct {
	MaxTrials          int           // Maximum number of trials
	TimeoutPerTrial    time.Duration // Timeout per trial
	TargetMetric       string        // "accuracy", "f1", "auc", "rmse"
	MetricGoal         string        // "maximize" or "minimize"
	ValidationSplit    float64       // Validation split ratio
	CVFolds            int           // Cross-validation folds
	EarlyStoppingRounds int          // Early stopping patience
	ParallelTrials     int           // Parallel trial execution
	AutoFeatureEng     bool          // Enable automatic feature engineering
	ModelTypes         []string      // Model types to try
}

// ModelBuilder interface for model construction
type ModelBuilder interface {
	Build(params map[string]interface{}) Model
	DefaultParams() map[string]interface{}
	ParamSpace() map[string]ParamRange
}

// Model interface for ML models
type Model interface {
	Fit(X [][]float64, y []float64) error
	Predict(X [][]float64) ([]float64, error)
	GetParams() map[string]interface{}
}

// TrainedModel represents a trained model with metadata
type TrainedModel struct {
	Model       Model
	Metrics     map[string]float64
	Params      map[string]interface{}
	ModelType   string
	TrainTime   time.Duration
	Features    []string
	Timestamp   time.Time
}

// Experiment tracks an AutoML trial
type Experiment struct {
	ID         string
	ModelType  string
	Params     map[string]interface{}
	Metrics    map[string]float64
	TrainTime  time.Duration
	Status     string
	Error      error
	Timestamp  time.Time
}

// ParamRange defines parameter search space
type ParamRange struct {
	Type   string      // "int", "float", "categorical"
	Min    float64     // For int/float
	Max    float64     // For int/float
	Values []interface{} // For categorical
	Scale  string      // "linear", "log"
}

// FeatureEngineer handles automatic feature engineering
type FeatureEngineer struct {
	transforms []FeatureTransform
	mu         sync.RWMutex
}

// FeatureTransform interface for feature transformations
type FeatureTransform interface {
	Transform(X [][]float64) ([][]float64, error)
	GetFeatureNames(inputNames []string) []string
}

// ModelEvaluator evaluates model performance
type ModelEvaluator struct {
	config *AutoMLConfig
}

// NewAutoMLEngine creates a new AutoML engine
func NewAutoMLEngine(config *AutoMLConfig) *AutoMLEngine {
	if config == nil {
		config = DefaultAutoMLConfig()
	}

	engine := &AutoMLEngine{
		config:      config,
		models:      make(map[string]ModelBuilder),
		evaluator:   NewModelEvaluator(config),
		featureEng:  NewFeatureEngineer(),
		experiments: make([]Experiment, 0),
	}

	// Register default models
	engine.RegisterModel("random_forest", &RandomForestBuilder{})
	engine.RegisterModel("xgboost", &XGBoostBuilder{})
	engine.RegisterModel("neural_net", &NeuralNetBuilder{})
	engine.RegisterModel("linear", &LinearModelBuilder{})

	return engine
}

// DefaultAutoMLConfig returns default AutoML configuration
func DefaultAutoMLConfig() *AutoMLConfig {
	return &AutoMLConfig{
		MaxTrials:          100,
		TimeoutPerTrial:    5 * time.Minute,
		TargetMetric:       "accuracy",
		MetricGoal:         "maximize",
		ValidationSplit:    0.2,
		CVFolds:            5,
		EarlyStoppingRounds: 10,
		ParallelTrials:     4,
		AutoFeatureEng:     true,
		ModelTypes:         []string{"random_forest", "xgboost", "neural_net"},
	}
}

// RegisterModel registers a model builder
func (e *AutoMLEngine) RegisterModel(name string, builder ModelBuilder) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.models[name] = builder
}

// Fit runs AutoML to find the best model
func (e *AutoMLEngine) Fit(ctx context.Context, X [][]float64, y []float64, featureNames []string) (*TrainedModel, error) {
	startTime := time.Now()

	// Auto feature engineering
	if e.config.AutoFeatureEng {
		X, featureNames = e.featureEng.AutoEngineer(X, featureNames)
	}

	// Split data
	XTrain, XVal, yTrain, yVal := e.splitData(X, y)

	// Parallel trial execution
	trialsChan := make(chan Experiment, e.config.MaxTrials)
	resultsChan := make(chan Experiment, e.config.MaxTrials)

	// Worker pool
	var wg sync.WaitGroup
	for i := 0; i < e.config.ParallelTrials; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for trial := range trialsChan {
				result := e.executeTrial(ctx, trial, XTrain, yTrain, XVal, yVal)
				resultsChan <- result
			}
		}()
	}

	// Generate trials
	go func() {
		trialID := 0
		for _, modelType := range e.getModelTypes() {
			builder, ok := e.models[modelType]
			if !ok {
				continue
			}

			// Generate hyperparameter configurations
			paramSpace := builder.ParamSpace()
			configs := e.generateParamConfigs(paramSpace, e.config.MaxTrials/len(e.getModelTypes()))

			for _, params := range configs {
				if trialID >= e.config.MaxTrials {
					break
				}

				trial := Experiment{
					ID:        fmt.Sprintf("trial_%d", trialID),
					ModelType: modelType,
					Params:    params,
					Status:    "pending",
					Timestamp: time.Now(),
				}

				trialsChan <- trial
				trialID++
			}
		}
		close(trialsChan)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Process results
	var bestExperiment *Experiment
	for result := range resultsChan {
		e.mu.Lock()
		e.experiments = append(e.experiments, result)

		if result.Status == "completed" {
			if bestExperiment == nil || e.isBetter(result.Metrics, bestExperiment.Metrics) {
				bestExperiment = &result
			}
		}
		e.mu.Unlock()
	}

	if bestExperiment == nil {
		return nil, fmt.Errorf("no successful trials")
	}

	// Train final model on full training data
	builder := e.models[bestExperiment.ModelType]
	finalModel := builder.Build(bestExperiment.Params)

	if err := finalModel.Fit(append(XTrain, XVal...), append(yTrain, yVal...)); err != nil {
		return nil, fmt.Errorf("failed to train final model: %w", err)
	}

	e.bestModel = &TrainedModel{
		Model:      finalModel,
		Metrics:    bestExperiment.Metrics,
		Params:     bestExperiment.Params,
		ModelType:  bestExperiment.ModelType,
		TrainTime:  time.Since(startTime),
		Features:   featureNames,
		Timestamp:  time.Now(),
	}

	return e.bestModel, nil
}

// executeTrial executes a single AutoML trial
func (e *AutoMLEngine) executeTrial(ctx context.Context, trial Experiment, XTrain, XVal [][]float64, yTrain, yVal []float64) Experiment {
	startTime := time.Now()

	// Check timeout
	trialCtx, cancel := context.WithTimeout(ctx, e.config.TimeoutPerTrial)
	defer cancel()

	done := make(chan bool)
	var model Model
	var err error

	go func() {
		builder, ok := e.models[trial.ModelType]
		if !ok {
			trial.Error = fmt.Errorf("unknown model type: %s", trial.ModelType)
			trial.Status = "failed"
			done <- true
			return
		}

		model = builder.Build(trial.Params)
		err = model.Fit(XTrain, yTrain)
		done <- true
	}()

	select {
	case <-done:
		if err != nil {
			trial.Error = err
			trial.Status = "failed"
			return trial
		}
	case <-trialCtx.Done():
		trial.Error = fmt.Errorf("trial timeout")
		trial.Status = "timeout"
		return trial
	}

	// Evaluate model
	metrics, err := e.evaluator.Evaluate(model, XVal, yVal)
	if err != nil {
		trial.Error = err
		trial.Status = "failed"
		return trial
	}

	trial.Metrics = metrics
	trial.TrainTime = time.Since(startTime)
	trial.Status = "completed"

	return trial
}

// splitData splits data into training and validation sets
func (e *AutoMLEngine) splitData(X [][]float64, y []float64) ([][]float64, [][]float64, []float64, []float64) {
	n := len(X)
	splitIdx := int(float64(n) * (1 - e.config.ValidationSplit))

	// Shuffle indices
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	XTrain := make([][]float64, splitIdx)
	XVal := make([][]float64, n-splitIdx)
	yTrain := make([]float64, splitIdx)
	yVal := make([]float64, n-splitIdx)

	for i := 0; i < splitIdx; i++ {
		XTrain[i] = X[indices[i]]
		yTrain[i] = y[indices[i]]
	}

	for i := splitIdx; i < n; i++ {
		XVal[i-splitIdx] = X[indices[i]]
		yVal[i-splitIdx] = y[indices[i]]
	}

	return XTrain, XVal, yTrain, yVal
}

// generateParamConfigs generates hyperparameter configurations
func (e *AutoMLEngine) generateParamConfigs(paramSpace map[string]ParamRange, numConfigs int) []map[string]interface{} {
	configs := make([]map[string]interface{}, numConfigs)

	for i := 0; i < numConfigs; i++ {
		config := make(map[string]interface{})
		for name, prange := range paramSpace {
			config[name] = e.sampleParam(prange)
		}
		configs[i] = config
	}

	return configs
}

// sampleParam samples a parameter from its range
func (e *AutoMLEngine) sampleParam(prange ParamRange) interface{} {
	switch prange.Type {
	case "int":
		val := prange.Min + rand.Float64()*(prange.Max-prange.Min)
		return int(val)
	case "float":
		if prange.Scale == "log" {
			logMin := math.Log(prange.Min)
			logMax := math.Log(prange.Max)
			return math.Exp(logMin + rand.Float64()*(logMax-logMin))
		}
		return prange.Min + rand.Float64()*(prange.Max-prange.Min)
	case "categorical":
		return prange.Values[rand.Intn(len(prange.Values))]
	default:
		return nil
	}
}

// isBetter checks if new metrics are better than current best
func (e *AutoMLEngine) isBetter(newMetrics, currentMetrics map[string]float64) bool {
	newScore := newMetrics[e.config.TargetMetric]
	currentScore := currentMetrics[e.config.TargetMetric]

	if e.config.MetricGoal == "maximize" {
		return newScore > currentScore
	}
	return newScore < currentScore
}

// getModelTypes returns configured model types
func (e *AutoMLEngine) getModelTypes() []string {
	if len(e.config.ModelTypes) > 0 {
		return e.config.ModelTypes
	}

	types := make([]string, 0, len(e.models))
	for t := range e.models {
		types = append(types, t)
	}
	return types
}

// GetBestModel returns the best trained model
func (e *AutoMLEngine) GetBestModel() *TrainedModel {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.bestModel
}

// GetExperiments returns all experiments
func (e *AutoMLEngine) GetExperiments() []Experiment {
	e.mu.RLock()
	defer e.mu.RUnlock()

	experiments := make([]Experiment, len(e.experiments))
	copy(experiments, e.experiments)

	// Sort by metric
	sort.Slice(experiments, func(i, j int) bool {
		if experiments[i].Status != "completed" {
			return false
		}
		if experiments[j].Status != "completed" {
			return true
		}
		return e.isBetter(experiments[i].Metrics, experiments[j].Metrics)
	})

	return experiments
}

// NewModelEvaluator creates a new model evaluator
func NewModelEvaluator(config *AutoMLConfig) *ModelEvaluator {
	return &ModelEvaluator{config: config}
}

// Evaluate evaluates model performance
func (ev *ModelEvaluator) Evaluate(model Model, X [][]float64, y []float64) (map[string]float64, error) {
	predictions, err := model.Predict(X)
	if err != nil {
		return nil, err
	}

	metrics := make(map[string]float64)

	// Calculate metrics
	metrics["accuracy"] = ev.calculateAccuracy(predictions, y)
	metrics["mse"] = ev.calculateMSE(predictions, y)
	metrics["rmse"] = math.Sqrt(metrics["mse"])
	metrics["mae"] = ev.calculateMAE(predictions, y)
	metrics["r2"] = ev.calculateR2(predictions, y)

	return metrics, nil
}

// calculateAccuracy calculates classification accuracy
func (ev *ModelEvaluator) calculateAccuracy(pred, actual []float64) float64 {
	correct := 0
	for i := range pred {
		if math.Round(pred[i]) == actual[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(pred))
}

// calculateMSE calculates mean squared error
func (ev *ModelEvaluator) calculateMSE(pred, actual []float64) float64 {
	sum := 0.0
	for i := range pred {
		diff := pred[i] - actual[i]
		sum += diff * diff
	}
	return sum / float64(len(pred))
}

// calculateMAE calculates mean absolute error
func (ev *ModelEvaluator) calculateMAE(pred, actual []float64) float64 {
	sum := 0.0
	for i := range pred {
		sum += math.Abs(pred[i] - actual[i])
	}
	return sum / float64(len(pred))
}

// calculateR2 calculates R-squared score
func (ev *ModelEvaluator) calculateR2(pred, actual []float64) float64 {
	mean := 0.0
	for _, v := range actual {
		mean += v
	}
	mean /= float64(len(actual))

	ssRes := 0.0
	ssTot := 0.0
	for i := range actual {
		ssRes += (actual[i] - pred[i]) * (actual[i] - pred[i])
		ssTot += (actual[i] - mean) * (actual[i] - mean)
	}

	if ssTot == 0 {
		return 0
	}
	return 1 - (ssRes / ssTot)
}

// NewFeatureEngineer creates a new feature engineer
func NewFeatureEngineer() *FeatureEngineer {
	return &FeatureEngineer{
		transforms: make([]FeatureTransform, 0),
	}
}

// AutoEngineer performs automatic feature engineering
func (fe *FeatureEngineer) AutoEngineer(X [][]float64, featureNames []string) ([][]float64, []string) {
	// Add polynomial features
	X, featureNames = fe.addPolynomialFeatures(X, featureNames, 2)

	// Add interaction features
	X, featureNames = fe.addInteractionFeatures(X, featureNames)

	// Normalize features
	X = fe.normalizeFeatures(X)

	return X, featureNames
}

// addPolynomialFeatures adds polynomial features
func (fe *FeatureEngineer) addPolynomialFeatures(X [][]float64, names []string, degree int) ([][]float64, []string) {
	if degree < 2 {
		return X, names
	}

	newX := make([][]float64, len(X))
	newNames := append([]string{}, names...)

	for i := range X {
		newRow := append([]float64{}, X[i]...)

		// Add squared terms
		for j := range X[i] {
			newRow = append(newRow, X[i][j]*X[i][j])
		}

		newX[i] = newRow
	}

	for _, name := range names {
		newNames = append(newNames, name+"^2")
	}

	return newX, newNames
}

// addInteractionFeatures adds interaction features
func (fe *FeatureEngineer) addInteractionFeatures(X [][]float64, names []string) ([][]float64, []string) {
	if len(X) == 0 || len(X[0]) < 2 {
		return X, names
	}

	newX := make([][]float64, len(X))
	newNames := append([]string{}, names...)

	for i := range X {
		newRow := append([]float64{}, X[i]...)

		// Add pairwise interactions
		for j := 0; j < len(X[i]); j++ {
			for k := j + 1; k < len(X[i]); k++ {
				newRow = append(newRow, X[i][j]*X[i][k])
			}
		}

		newX[i] = newRow
	}

	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			newNames = append(newNames, names[i]+"*"+names[j])
		}
	}

	return newX, newNames
}

// normalizeFeatures normalizes features to [0, 1]
func (fe *FeatureEngineer) normalizeFeatures(X [][]float64) [][]float64 {
	if len(X) == 0 {
		return X
	}

	nFeatures := len(X[0])
	mins := make([]float64, nFeatures)
	maxs := make([]float64, nFeatures)

	// Find min/max
	for j := 0; j < nFeatures; j++ {
		mins[j] = X[0][j]
		maxs[j] = X[0][j]
		for i := 1; i < len(X); i++ {
			if X[i][j] < mins[j] {
				mins[j] = X[i][j]
			}
			if X[i][j] > maxs[j] {
				maxs[j] = X[i][j]
			}
		}
	}

	// Normalize
	normalized := make([][]float64, len(X))
	for i := range X {
		normalized[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			if maxs[j] > mins[j] {
				normalized[i][j] = (X[i][j] - mins[j]) / (maxs[j] - mins[j])
			} else {
				normalized[i][j] = 0
			}
		}
	}

	return normalized
}
