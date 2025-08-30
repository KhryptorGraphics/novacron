// Package ml provides machine learning model training and optimization for orchestration
package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ModelType represents different types of ML models
type ModelType string

const (
	ModelTypePlacementPredictor ModelType = "placement_predictor"
	ModelTypeScalingPredictor   ModelType = "scaling_predictor"
	ModelTypeResourcePredictor  ModelType = "resource_predictor"
	ModelTypeFailurePredictor   ModelType = "failure_predictor"
)

// TrainingConfig configures model training parameters
type TrainingConfig struct {
	ModelType           ModelType     `json:"model_type"`
	TrainingDataPath    string        `json:"training_data_path"`
	ValidationSplit     float64       `json:"validation_split"`
	TestSplit          float64       `json:"test_split"`
	BatchSize          int           `json:"batch_size"`
	LearningRate       float64       `json:"learning_rate"`
	Epochs             int           `json:"epochs"`
	EarlyStoppingRounds int          `json:"early_stopping_rounds"`
	ModelOutputPath    string        `json:"model_output_path"`
	HyperparameterTuning bool        `json:"hyperparameter_tuning"`
	CrossValidationFolds int         `json:"cross_validation_folds"`
}

// TrainingMetrics tracks model training progress
type TrainingMetrics struct {
	Epoch           int       `json:"epoch"`
	TrainingLoss    float64   `json:"training_loss"`
	ValidationLoss  float64   `json:"validation_loss"`
	TrainingAccuracy float64  `json:"training_accuracy"`
	ValidationAccuracy float64 `json:"validation_accuracy"`
	LearningRate    float64   `json:"learning_rate"`
	Timestamp       time.Time `json:"timestamp"`
}

// ModelPerformance tracks model performance metrics
type ModelPerformance struct {
	ModelType      ModelType `json:"model_type"`
	ModelVersion   string    `json:"model_version"`
	Accuracy       float64   `json:"accuracy"`
	Precision      float64   `json:"precision"`
	Recall         float64   `json:"recall"`
	F1Score        float64   `json:"f1_score"`
	AUC            float64   `json:"auc"`
	MeanSquaredError float64 `json:"mean_squared_error"`
	MeanAbsoluteError float64 `json:"mean_absolute_error"`
	TestDataSize   int       `json:"test_data_size"`
	TrainingTime   time.Duration `json:"training_time"`
	InferenceTime  time.Duration `json:"inference_time"`
	ModelSizeMB    float64   `json:"model_size_mb"`
	CreatedAt      time.Time `json:"created_at"`
}

// ModelTrainer handles ML model training and optimization
type ModelTrainer struct {
	logger       *logrus.Logger
	models       map[ModelType]*TrainedModel
	trainingData map[ModelType]*TrainingDataset
	mu           sync.RWMutex
}

// TrainedModel represents a trained ML model
type TrainedModel struct {
	Type          ModelType          `json:"type"`
	Version       string             `json:"version"`
	Parameters    map[string]float64 `json:"parameters"`
	Hyperparams   map[string]interface{} `json:"hyperparameters"`
	Performance   ModelPerformance   `json:"performance"`
	CreatedAt     time.Time          `json:"created_at"`
	LastUsedAt    time.Time          `json:"last_used_at"`
	UsageCount    int64              `json:"usage_count"`
}

// TrainingDataset holds training data for ML models
type TrainingDataset struct {
	Type         ModelType     `json:"type"`
	Features     [][]float64   `json:"features"`
	Labels       []float64     `json:"labels"`
	FeatureNames []string      `json:"feature_names"`
	Size         int           `json:"size"`
	LastUpdated  time.Time     `json:"last_updated"`
}

// NewModelTrainer creates a new model trainer
func NewModelTrainer(logger *logrus.Logger) *ModelTrainer {
	return &ModelTrainer{
		logger:       logger,
		models:       make(map[ModelType]*TrainedModel),
		trainingData: make(map[ModelType]*TrainingDataset),
	}
}

// TrainModel trains a new ML model with the given configuration
func (mt *ModelTrainer) TrainModel(ctx context.Context, config TrainingConfig) (*TrainedModel, error) {
	startTime := time.Now()
	
	mt.logger.WithFields(logrus.Fields{
		"model_type":    config.ModelType,
		"learning_rate": config.LearningRate,
		"epochs":        config.Epochs,
	}).Info("Starting model training")

	// Load training data
	dataset, err := mt.loadTrainingData(config)
	if err != nil {
		return nil, fmt.Errorf("failed to load training data: %w", err)
	}

	// Split data
	trainData, valData, testData := mt.splitDataset(dataset, config)

	// Initialize model parameters
	model := &TrainedModel{
		Type:        config.ModelType,
		Version:     fmt.Sprintf("v%d", time.Now().Unix()),
		Parameters:  mt.initializeParameters(config.ModelType),
		Hyperparams: mt.getHyperparameters(config),
		CreatedAt:   time.Now(),
	}

	var bestModel *TrainedModel
	var bestValidationLoss float64 = math.Inf(1)
	var earlyStopCounter int

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Train epoch
		trainLoss, trainAccuracy := mt.trainEpoch(model, trainData, config)
		
		// Validate
		valLoss, valAccuracy := mt.validateModel(model, valData)

		// Record metrics
		metrics := TrainingMetrics{
			Epoch:             epoch,
			TrainingLoss:      trainLoss,
			ValidationLoss:    valLoss,
			TrainingAccuracy:  trainAccuracy,
			ValidationAccuracy: valAccuracy,
			LearningRate:      config.LearningRate,
			Timestamp:         time.Now(),
		}

		mt.logger.WithFields(logrus.Fields{
			"epoch":      epoch,
			"train_loss": trainLoss,
			"val_loss":   valLoss,
			"train_acc":  trainAccuracy,
			"val_acc":    valAccuracy,
		}).Debug("Training epoch completed")

		// Early stopping
		if valLoss < bestValidationLoss {
			bestValidationLoss = valLoss
			bestModel = mt.copyModel(model)
			earlyStopCounter = 0
		} else {
			earlyStopCounter++
			if config.EarlyStoppingRounds > 0 && earlyStopCounter >= config.EarlyStoppingRounds {
				mt.logger.WithField("epoch", epoch).Info("Early stopping triggered")
				break
			}
		}

		// Adaptive learning rate
		if epoch%10 == 0 && epoch > 0 {
			config.LearningRate *= 0.95
		}
	}

	// Use best model
	if bestModel != nil {
		model = bestModel
	}

	// Evaluate on test data
	performance := mt.evaluateModel(model, testData)
	performance.TrainingTime = time.Since(startTime)
	performance.ModelVersion = model.Version
	performance.CreatedAt = time.Now()

	model.Performance = performance

	// Store model
	mt.mu.Lock()
	mt.models[config.ModelType] = model
	mt.mu.Unlock()

	mt.logger.WithFields(logrus.Fields{
		"model_type":      config.ModelType,
		"model_version":   model.Version,
		"training_time":   performance.TrainingTime,
		"test_accuracy":   performance.Accuracy,
		"test_f1_score":   performance.F1Score,
	}).Info("Model training completed")

	return model, nil
}

// PerformHyperparameterTuning performs hyperparameter optimization
func (mt *ModelTrainer) PerformHyperparameterTuning(ctx context.Context, baseConfig TrainingConfig) (*TrainingConfig, error) {
	mt.logger.Info("Starting hyperparameter tuning")

	// Define hyperparameter search space
	learningRates := []float64{0.001, 0.01, 0.1}
	batchSizes := []int{32, 64, 128}
	
	bestConfig := baseConfig
	bestScore := -math.Inf(1)

	for _, lr := range learningRates {
		for _, bs := range batchSizes {
			config := baseConfig
			config.LearningRate = lr
			config.BatchSize = bs
			config.Epochs = 20 // Reduced epochs for tuning

			score, err := mt.performCrossValidation(ctx, config)
			if err != nil {
				mt.logger.WithError(err).Warn("Cross-validation failed for hyperparameters")
				continue
			}

			mt.logger.WithFields(logrus.Fields{
				"learning_rate": lr,
				"batch_size":    bs,
				"cv_score":      score,
			}).Debug("Hyperparameter combination evaluated")

			if score > bestScore {
				bestScore = score
				bestConfig = config
			}
		}
	}

	mt.logger.WithFields(logrus.Fields{
		"best_learning_rate": bestConfig.LearningRate,
		"best_batch_size":    bestConfig.BatchSize,
		"best_cv_score":      bestScore,
	}).Info("Hyperparameter tuning completed")

	return &bestConfig, nil
}

// PerformABTesting performs A/B testing between model versions
func (mt *ModelTrainer) PerformABTesting(modelA, modelB *TrainedModel, testDataset *TrainingDataset) (*ABTestResult, error) {
	mt.logger.WithFields(logrus.Fields{
		"model_a_version": modelA.Version,
		"model_b_version": modelB.Version,
	}).Info("Starting A/B test")

	// Evaluate both models on test dataset
	perfA := mt.evaluateModel(modelA, testDataset)
	perfB := mt.evaluateModel(modelB, testDataset)

	// Statistical significance test (simplified)
	significant := mt.calculateStatisticalSignificance(perfA, perfB)

	result := &ABTestResult{
		ModelAVersion:      modelA.Version,
		ModelBVersion:      modelB.Version,
		ModelAPerformance:  perfA,
		ModelBPerformance:  perfB,
		Winner:            mt.determineWinner(perfA, perfB),
		StatisticallySignificant: significant,
		TestDataSize:      testDataset.Size,
		TestDate:          time.Now(),
	}

	mt.logger.WithFields(logrus.Fields{
		"winner":                    result.Winner,
		"statistically_significant": result.StatisticallySignificant,
		"model_a_accuracy":          perfA.Accuracy,
		"model_b_accuracy":          perfB.Accuracy,
	}).Info("A/B test completed")

	return result, nil
}

// GetModel returns a trained model by type
func (mt *ModelTrainer) GetModel(modelType ModelType) (*TrainedModel, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	model, exists := mt.models[modelType]
	if !exists {
		return nil, fmt.Errorf("model of type %s not found", modelType)
	}

	// Update usage statistics
	model.LastUsedAt = time.Now()
	model.UsageCount++

	return model, nil
}

// UpdateTrainingData updates training data for a model type
func (mt *ModelTrainer) UpdateTrainingData(modelType ModelType, features [][]float64, labels []float64, featureNames []string) error {
	mt.mu.Lock()
	defer mt.mu.Unlock()

	dataset := &TrainingDataset{
		Type:         modelType,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Size:         len(features),
		LastUpdated:  time.Now(),
	}

	mt.trainingData[modelType] = dataset

	mt.logger.WithFields(logrus.Fields{
		"model_type":    modelType,
		"samples":       len(features),
		"features":      len(featureNames),
	}).Info("Training data updated")

	return nil
}

// ABTestResult contains results from A/B testing
type ABTestResult struct {
	ModelAVersion            string           `json:"model_a_version"`
	ModelBVersion            string           `json:"model_b_version"`
	ModelAPerformance        ModelPerformance `json:"model_a_performance"`
	ModelBPerformance        ModelPerformance `json:"model_b_performance"`
	Winner                   string           `json:"winner"`
	StatisticallySignificant bool             `json:"statistically_significant"`
	TestDataSize            int              `json:"test_data_size"`
	TestDate                time.Time        `json:"test_date"`
}

// Helper methods

func (mt *ModelTrainer) loadTrainingData(config TrainingConfig) (*TrainingDataset, error) {
	// Check if we have cached training data
	mt.mu.RLock()
	dataset, exists := mt.trainingData[config.ModelType]
	mt.mu.RUnlock()

	if exists && time.Since(dataset.LastUpdated) < 24*time.Hour {
		return dataset, nil
	}

	// Generate synthetic training data for demonstration
	return mt.generateSyntheticData(config.ModelType), nil
}

func (mt *ModelTrainer) generateSyntheticData(modelType ModelType) *TrainingDataset {
	size := 1000
	features := make([][]float64, size)
	labels := make([]float64, size)
	
	switch modelType {
	case ModelTypePlacementPredictor:
		featureNames := []string{"cpu_cores", "memory_mb", "disk_gb", "network_mbps", "node_load"}
		for i := 0; i < size; i++ {
			// Generate realistic placement features
			features[i] = []float64{
				float64(1 + i%16),           // cpu_cores
				float64(1024 + i%32*1024),   // memory_mb  
				float64(50 + i%500),         // disk_gb
				float64(100 + i%900),        // network_mbps
				float64(i%100) / 100.0,      // node_load
			}
			
			// Generate placement success probability
			labels[i] = mt.calculatePlacementSuccess(features[i])
		}
		
		return &TrainingDataset{
			Type:         modelType,
			Features:     features,
			Labels:       labels,
			FeatureNames: featureNames,
			Size:         size,
			LastUpdated:  time.Now(),
		}
		
	case ModelTypeScalingPredictor:
		featureNames := []string{"cpu_utilization", "memory_utilization", "request_rate", "response_time", "queue_length"}
		for i := 0; i < size; i++ {
			features[i] = []float64{
				float64(i%100) / 100.0,      // cpu_utilization
				float64(i%100) / 100.0,      // memory_utilization
				float64(100 + i%1000),       // request_rate
				float64(50 + i%500),         // response_time
				float64(i % 50),             // queue_length
			}
			
			labels[i] = mt.calculateScalingNeed(features[i])
		}
		
		return &TrainingDataset{
			Type:         modelType,
			Features:     features,
			Labels:       labels,
			FeatureNames: featureNames,
			Size:         size,
			LastUpdated:  time.Now(),
		}
		
	default:
		// Generic synthetic data
		featureNames := []string{"feature1", "feature2", "feature3", "feature4", "feature5"}
		for i := 0; i < size; i++ {
			features[i] = []float64{
				float64(i) / float64(size),
				math.Sin(float64(i) / 10.0),
				math.Cos(float64(i) / 10.0),
				float64(i%10) / 10.0,
				float64(i%5) / 5.0,
			}
			
			labels[i] = features[i][0]*0.5 + features[i][1]*0.3 + features[i][2]*0.2
		}
		
		return &TrainingDataset{
			Type:         modelType,
			Features:     features,
			Labels:       labels,
			FeatureNames: featureNames,
			Size:         size,
			LastUpdated:  time.Now(),
		}
	}
}

func (mt *ModelTrainer) calculatePlacementSuccess(features []float64) float64 {
	// Simplified placement success calculation
	cpuScore := math.Min(features[0]/16.0, 1.0)
	memScore := math.Min(features[1]/32768.0, 1.0)
	loadPenalty := features[4] // node_load
	
	return (cpuScore + memScore) * 0.5 * (1.0 - loadPenalty*0.5)
}

func (mt *ModelTrainer) calculateScalingNeed(features []float64) float64 {
	cpuUtil := features[0]
	memUtil := features[1]
	
	// Scale up if utilization is high
	if cpuUtil > 0.8 || memUtil > 0.8 {
		return 1.0 // Scale up
	} else if cpuUtil < 0.3 && memUtil < 0.3 {
		return -1.0 // Scale down
	}
	
	return 0.0 // No scaling needed
}

func (mt *ModelTrainer) splitDataset(dataset *TrainingDataset, config TrainingConfig) (*TrainingDataset, *TrainingDataset, *TrainingDataset) {
	size := dataset.Size
	trainEnd := int(float64(size) * (1.0 - config.ValidationSplit - config.TestSplit))
	valEnd := int(float64(size) * (1.0 - config.TestSplit))
	
	return &TrainingDataset{
			Features: dataset.Features[:trainEnd],
			Labels:   dataset.Labels[:trainEnd],
			Size:     trainEnd,
		}, &TrainingDataset{
			Features: dataset.Features[trainEnd:valEnd],
			Labels:   dataset.Labels[trainEnd:valEnd],
			Size:     valEnd - trainEnd,
		}, &TrainingDataset{
			Features: dataset.Features[valEnd:],
			Labels:   dataset.Labels[valEnd:],
			Size:     size - valEnd,
		}
}

func (mt *ModelTrainer) initializeParameters(modelType ModelType) map[string]float64 {
	// Initialize model parameters based on type
	params := make(map[string]float64)
	
	switch modelType {
	case ModelTypePlacementPredictor:
		params["weight_cpu"] = 0.3
		params["weight_memory"] = 0.3
		params["weight_disk"] = 0.2
		params["weight_network"] = 0.1
		params["weight_load"] = 0.1
		
	case ModelTypeScalingPredictor:
		params["cpu_threshold"] = 0.7
		params["memory_threshold"] = 0.8
		params["scale_factor"] = 1.5
		
	default:
		// Generic linear model parameters
		for i := 0; i < 5; i++ {
			params[fmt.Sprintf("weight_%d", i)] = 0.2
		}
	}
	
	return params
}

func (mt *ModelTrainer) getHyperparameters(config TrainingConfig) map[string]interface{} {
	return map[string]interface{}{
		"learning_rate": config.LearningRate,
		"batch_size":    config.BatchSize,
		"epochs":        config.Epochs,
	}
}

func (mt *ModelTrainer) trainEpoch(model *TrainedModel, trainData *TrainingDataset, config TrainingConfig) (float64, float64) {
	// Simplified training implementation
	totalLoss := 0.0
	correct := 0
	
	for i := 0; i < trainData.Size; i++ {
		prediction := mt.predict(model, trainData.Features[i])
		loss := math.Pow(prediction-trainData.Labels[i], 2)
		totalLoss += loss
		
		if math.Abs(prediction-trainData.Labels[i]) < 0.1 {
			correct++
		}
		
		// Update parameters (simplified gradient descent)
		mt.updateParameters(model, trainData.Features[i], trainData.Labels[i], prediction, config.LearningRate)
	}
	
	avgLoss := totalLoss / float64(trainData.Size)
	accuracy := float64(correct) / float64(trainData.Size)
	
	return avgLoss, accuracy
}

func (mt *ModelTrainer) validateModel(model *TrainedModel, valData *TrainingDataset) (float64, float64) {
	totalLoss := 0.0
	correct := 0
	
	for i := 0; i < valData.Size; i++ {
		prediction := mt.predict(model, valData.Features[i])
		loss := math.Pow(prediction-valData.Labels[i], 2)
		totalLoss += loss
		
		if math.Abs(prediction-valData.Labels[i]) < 0.1 {
			correct++
		}
	}
	
	avgLoss := totalLoss / float64(valData.Size)
	accuracy := float64(correct) / float64(valData.Size)
	
	return avgLoss, accuracy
}

func (mt *ModelTrainer) predict(model *TrainedModel, features []float64) float64 {
	// Simplified prediction based on model type
	switch model.Type {
	case ModelTypePlacementPredictor:
		return features[0]*model.Parameters["weight_cpu"] +
			features[1]*model.Parameters["weight_memory"] +
			features[2]*model.Parameters["weight_disk"] +
			features[3]*model.Parameters["weight_network"] +
			features[4]*model.Parameters["weight_load"]
			
	case ModelTypeScalingPredictor:
		cpuUtil := features[0]
		memUtil := features[1]
		
		if cpuUtil > model.Parameters["cpu_threshold"] || memUtil > model.Parameters["memory_threshold"] {
			return 1.0
		} else if cpuUtil < 0.3 && memUtil < 0.3 {
			return -1.0
		}
		return 0.0
		
	default:
		// Generic linear prediction
		result := 0.0
		for i, feature := range features {
			if i < len(features) {
				weight := model.Parameters[fmt.Sprintf("weight_%d", i)]
				result += feature * weight
			}
		}
		return result
	}
}

func (mt *ModelTrainer) updateParameters(model *TrainedModel, features []float64, actual, predicted, learningRate float64) {
	// Simplified parameter update
	error := actual - predicted
	
	switch model.Type {
	case ModelTypePlacementPredictor:
		model.Parameters["weight_cpu"] += learningRate * error * features[0]
		model.Parameters["weight_memory"] += learningRate * error * features[1]
		model.Parameters["weight_disk"] += learningRate * error * features[2]
		model.Parameters["weight_network"] += learningRate * error * features[3]
		model.Parameters["weight_load"] += learningRate * error * features[4]
		
	default:
		for i, feature := range features {
			if i < len(features) {
				key := fmt.Sprintf("weight_%d", i)
				model.Parameters[key] += learningRate * error * feature
			}
		}
	}
}

func (mt *ModelTrainer) evaluateModel(model *TrainedModel, testData *TrainingDataset) ModelPerformance {
	var predictions []float64
	var trueLabels []float64
	
	totalLoss := 0.0
	correct := 0
	
	startTime := time.Now()
	
	for i := 0; i < testData.Size; i++ {
		prediction := mt.predict(model, testData.Features[i])
		predictions = append(predictions, prediction)
		trueLabels = append(trueLabels, testData.Labels[i])
		
		loss := math.Pow(prediction-testData.Labels[i], 2)
		totalLoss += loss
		
		if math.Abs(prediction-testData.Labels[i]) < 0.1 {
			correct++
		}
	}
	
	inferenceTime := time.Since(startTime)
	
	accuracy := float64(correct) / float64(testData.Size)
	mse := totalLoss / float64(testData.Size)
	mae := mt.calculateMAE(predictions, trueLabels)
	
	return ModelPerformance{
		ModelType:         model.Type,
		Accuracy:         accuracy,
		MeanSquaredError: mse,
		MeanAbsoluteError: mae,
		TestDataSize:     testData.Size,
		InferenceTime:    inferenceTime,
		ModelSizeMB:      mt.calculateModelSize(model),
	}
}

func (mt *ModelTrainer) calculateMAE(predictions, actual []float64) float64 {
	sum := 0.0
	for i := range predictions {
		sum += math.Abs(predictions[i] - actual[i])
	}
	return sum / float64(len(predictions))
}

func (mt *ModelTrainer) calculateModelSize(model *TrainedModel) float64 {
	// Estimate model size in MB
	parameterCount := len(model.Parameters)
	return float64(parameterCount * 8) / (1024 * 1024) // 8 bytes per float64
}

func (mt *ModelTrainer) copyModel(model *TrainedModel) *TrainedModel {
	params := make(map[string]float64)
	for k, v := range model.Parameters {
		params[k] = v
	}
	
	hyperparams := make(map[string]interface{})
	for k, v := range model.Hyperparams {
		hyperparams[k] = v
	}
	
	return &TrainedModel{
		Type:        model.Type,
		Version:     model.Version,
		Parameters:  params,
		Hyperparams: hyperparams,
		Performance: model.Performance,
		CreatedAt:   model.CreatedAt,
	}
}

func (mt *ModelTrainer) performCrossValidation(ctx context.Context, config TrainingConfig) (float64, error) {
	// Simplified cross-validation
	dataset, err := mt.loadTrainingData(config)
	if err != nil {
		return 0, err
	}
	
	folds := config.CrossValidationFolds
	if folds <= 0 {
		folds = 5
	}
	
	foldSize := dataset.Size / folds
	scores := make([]float64, folds)
	
	for fold := 0; fold < folds; fold++ {
		// Create train/validation split for this fold
		valStart := fold * foldSize
		valEnd := (fold + 1) * foldSize
		
		trainFeatures := append(dataset.Features[:valStart], dataset.Features[valEnd:]...)
		trainLabels := append(dataset.Labels[:valStart], dataset.Labels[valEnd:]...)
		valFeatures := dataset.Features[valStart:valEnd]
		valLabels := dataset.Labels[valStart:valEnd]
		
		trainData := &TrainingDataset{
			Features: trainFeatures,
			Labels:   trainLabels,
			Size:     len(trainFeatures),
		}
		
		valData := &TrainingDataset{
			Features: valFeatures,
			Labels:   valLabels,
			Size:     len(valFeatures),
		}
		
		// Train model on fold
		model := &TrainedModel{
			Type:       config.ModelType,
			Parameters: mt.initializeParameters(config.ModelType),
		}
		
		// Quick training for CV
		for epoch := 0; epoch < config.Epochs/4; epoch++ {
			mt.trainEpoch(model, trainData, config)
		}
		
		// Evaluate on validation set
		_, accuracy := mt.validateModel(model, valData)
		scores[fold] = accuracy
	}
	
	// Return average score
	sum := 0.0
	for _, score := range scores {
		sum += score
	}
	
	return sum / float64(len(scores)), nil
}

func (mt *ModelTrainer) calculateStatisticalSignificance(perfA, perfB ModelPerformance) bool {
	// Simplified statistical significance test
	// In practice, would use proper statistical tests like t-test
	diff := math.Abs(perfA.Accuracy - perfB.Accuracy)
	return diff > 0.05 // 5% difference threshold
}

func (mt *ModelTrainer) determineWinner(perfA, perfB ModelPerformance) string {
	if perfA.Accuracy > perfB.Accuracy {
		return "Model A"
	} else if perfB.Accuracy > perfA.Accuracy {
		return "Model B"
	}
	return "Tie"
}