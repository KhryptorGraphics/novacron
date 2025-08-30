package predictive

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ModelType represents the type of predictive model
type ModelType string

const (
	ModelTypeARIMA ModelType = "arima"
	ModelTypeLSTM  ModelType = "lstm"
	ModelTypeLinear ModelType = "linear"
	ModelTypeExponentialSmoothing ModelType = "exponential_smoothing"
)

// TimeSeriesPoint represents a single data point in a time series
type TimeSeriesPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Tags      map[string]string `json:"tags,omitempty"`
}

// TimeSeriesData represents a collection of time series points
type TimeSeriesData struct {
	MetricName string              `json:"metric_name"`
	Points     []TimeSeriesPoint   `json:"points"`
	Tags       map[string]string   `json:"tags"`
}

// Prediction represents a forecast prediction
type Prediction struct {
	Timestamp   time.Time `json:"timestamp"`
	Value       float64   `json:"value"`
	Confidence  float64   `json:"confidence"`
	Lower       float64   `json:"lower_bound"`
	Upper       float64   `json:"upper_bound"`
	ModelType   ModelType `json:"model_type"`
}

// ForecastResult represents the result of a forecasting operation
type ForecastResult struct {
	MetricName    string       `json:"metric_name"`
	ModelType     ModelType    `json:"model_type"`
	Predictions   []Prediction `json:"predictions"`
	ModelAccuracy float64      `json:"model_accuracy"`
	GeneratedAt   time.Time    `json:"generated_at"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// ModelConfig represents configuration for a predictive model
type ModelConfig struct {
	Type              ModelType              `json:"type"`
	Parameters        map[string]interface{} `json:"parameters"`
	TrainingWindow    time.Duration          `json:"training_window"`
	PredictionHorizon time.Duration          `json:"prediction_horizon"`
	MinDataPoints     int                    `json:"min_data_points"`
	MaxDataPoints     int                    `json:"max_data_points"`
	UpdateFrequency   time.Duration          `json:"update_frequency"`
}

// PredictiveModel interface for all forecasting models
type PredictiveModel interface {
	// Train trains the model with historical data
	Train(ctx context.Context, data *TimeSeriesData) error
	
	// Predict generates forecasts for the specified horizon
	Predict(ctx context.Context, horizon time.Duration, interval time.Duration) (*ForecastResult, error)
	
	// Update updates the model with new data points
	Update(ctx context.Context, points []TimeSeriesPoint) error
	
	// GetAccuracy returns the model's accuracy score
	GetAccuracy() float64
	
	// GetType returns the model type
	GetType() ModelType
	
	// GetConfig returns the model configuration
	GetConfig() ModelConfig
	
	// IsReady returns true if the model is trained and ready for predictions
	IsReady() bool
}

// ARIMAModel implements Auto-Regressive Integrated Moving Average model
type ARIMAModel struct {
	config     ModelConfig
	trained    bool
	accuracy   float64
	coeffs     []float64
	residuals  []float64
	lastPoints []TimeSeriesPoint
	
	// ARIMA parameters (p, d, q)
	p int // autoregressive order
	d int // differencing order  
	q int // moving average order
	
	// Fitted parameters
	arCoeffs []float64 // autoregressive coefficients
	maCoeffs []float64 // moving average coefficients
	constant float64   // constant term
}

// NewARIMAModel creates a new ARIMA model
func NewARIMAModel(config ModelConfig) *ARIMAModel {
	// Extract ARIMA parameters from config
	p := 1 // default values
	d := 1
	q := 1
	
	if params, ok := config.Parameters["p"]; ok {
		if pVal, ok := params.(int); ok {
			p = pVal
		}
	}
	
	if params, ok := config.Parameters["d"]; ok {
		if dVal, ok := params.(int); ok {
			d = dVal
		}
	}
	
	if params, ok := config.Parameters["q"]; ok {
		if qVal, ok := params.(int); ok {
			q = qVal
		}
	}
	
	return &ARIMAModel{
		config: config,
		p:      p,
		d:      d,
		q:      q,
	}
}

// Train implements PredictiveModel interface
func (a *ARIMAModel) Train(ctx context.Context, data *TimeSeriesData) error {
	if len(data.Points) < a.config.MinDataPoints {
		return fmt.Errorf("insufficient data points: got %d, need at least %d", 
			len(data.Points), a.config.MinDataPoints)
	}
	
	// Sort points by timestamp
	sort.Slice(data.Points, func(i, j int) bool {
		return data.Points[i].Timestamp.Before(data.Points[j].Timestamp)
	})
	
	// Extract values
	values := make([]float64, len(data.Points))
	for i, point := range data.Points {
		values[i] = point.Value
	}
	
	// Apply differencing to achieve stationarity
	diffValues := a.applyDifferencing(values, a.d)
	
	// Fit ARIMA model using simplified approach
	if err := a.fitModel(diffValues); err != nil {
		return fmt.Errorf("failed to fit ARIMA model: %w", err)
	}
	
	// Calculate model accuracy
	a.accuracy = a.calculateAccuracy(values)
	
	// Store recent points for prediction
	recentCount := a.p + a.q + 10 // keep extra for better predictions
	if recentCount > len(data.Points) {
		recentCount = len(data.Points)
	}
	
	startIdx := len(data.Points) - recentCount
	a.lastPoints = make([]TimeSeriesPoint, recentCount)
	copy(a.lastPoints, data.Points[startIdx:])
	
	a.trained = true
	return nil
}

// Predict implements PredictiveModel interface
func (a *ARIMAModel) Predict(ctx context.Context, horizon time.Duration, interval time.Duration) (*ForecastResult, error) {
	if !a.trained {
		return nil, fmt.Errorf("model not trained")
	}
	
	if len(a.lastPoints) == 0 {
		return nil, fmt.Errorf("no historical data available for prediction")
	}
	
	// Calculate number of predictions needed
	numPredictions := int(horizon / interval)
	if numPredictions <= 0 {
		return nil, fmt.Errorf("invalid prediction horizon or interval")
	}
	
	predictions := make([]Prediction, numPredictions)
	lastTimestamp := a.lastPoints[len(a.lastPoints)-1].Timestamp
	
	// Extract recent values for prediction
	recentValues := make([]float64, len(a.lastPoints))
	for i, point := range a.lastPoints {
		recentValues[i] = point.Value
	}
	
	// Generate predictions
	for i := 0; i < numPredictions; i++ {
		predictTimestamp := lastTimestamp.Add(time.Duration(i+1) * interval)
		
		// Simple ARIMA prediction using autoregressive component
		prediction := a.predictNextValue(recentValues)
		confidence := a.calculateConfidence(i + 1)
		
		// Calculate confidence intervals (simplified approach)
		stdErr := a.calculateStandardError()
		lower := prediction - 1.96*stdErr
		upper := prediction + 1.96*stdErr
		
		predictions[i] = Prediction{
			Timestamp:  predictTimestamp,
			Value:      prediction,
			Confidence: confidence,
			Lower:      lower,
			Upper:      upper,
			ModelType:  ModelTypeARIMA,
		}
		
		// Update recent values for next prediction
		recentValues = append(recentValues[1:], prediction)
	}
	
	return &ForecastResult{
		ModelType:     ModelTypeARIMA,
		Predictions:   predictions,
		ModelAccuracy: a.accuracy,
		GeneratedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"arima_order": fmt.Sprintf("(%d,%d,%d)", a.p, a.d, a.q),
			"data_points": len(a.lastPoints),
		},
	}, nil
}

// Update implements PredictiveModel interface
func (a *ARIMAModel) Update(ctx context.Context, points []TimeSeriesPoint) error {
	if !a.trained {
		return fmt.Errorf("model not trained")
	}
	
	// Add new points to lastPoints
	a.lastPoints = append(a.lastPoints, points...)
	
	// Keep only recent points within training window
	maxPoints := a.config.MaxDataPoints
	if maxPoints > 0 && len(a.lastPoints) > maxPoints {
		startIdx := len(a.lastPoints) - maxPoints
		a.lastPoints = a.lastPoints[startIdx:]
	}
	
	return nil
}

// GetAccuracy implements PredictiveModel interface
func (a *ARIMAModel) GetAccuracy() float64 {
	return a.accuracy
}

// GetType implements PredictiveModel interface
func (a *ARIMAModel) GetType() ModelType {
	return ModelTypeARIMA
}

// GetConfig implements PredictiveModel interface
func (a *ARIMAModel) GetConfig() ModelConfig {
	return a.config
}

// IsReady implements PredictiveModel interface
func (a *ARIMAModel) IsReady() bool {
	return a.trained
}

// applyDifferencing applies differencing to achieve stationarity
func (a *ARIMAModel) applyDifferencing(values []float64, order int) []float64 {
	result := make([]float64, len(values))
	copy(result, values)
	
	for d := 0; d < order; d++ {
		if len(result) <= 1 {
			break
		}
		
		diffed := make([]float64, len(result)-1)
		for i := 1; i < len(result); i++ {
			diffed[i-1] = result[i] - result[i-1]
		}
		result = diffed
	}
	
	return result
}

// fitModel fits the ARIMA model using simplified least squares approach
func (a *ARIMAModel) fitModel(values []float64) error {
	if len(values) < a.p+a.q+1 {
		return fmt.Errorf("insufficient data for fitting ARIMA(%d,%d,%d)", a.p, a.d, a.q)
	}
	
	// Initialize coefficients
	a.arCoeffs = make([]float64, a.p)
	a.maCoeffs = make([]float64, a.q)
	
	// Simplified fitting using method of moments/least squares
	// For production use, consider using more sophisticated methods like MLE
	
	// Fit AR component using Yule-Walker equations (simplified)
	if a.p > 0 {
		if err := a.fitARComponent(values); err != nil {
			return fmt.Errorf("failed to fit AR component: %w", err)
		}
	}
	
	// Calculate residuals and fit MA component
	residuals := a.calculateResiduals(values)
	if a.q > 0 {
		if err := a.fitMAComponent(residuals); err != nil {
			return fmt.Errorf("failed to fit MA component: %w", err)
		}
	}
	
	a.residuals = residuals
	return nil
}

// fitARComponent fits the autoregressive component
func (a *ARIMAModel) fitARComponent(values []float64) error {
	// Calculate autocorrelations
	autocorrs := a.calculateAutocorrelations(values, a.p)
	
	// Solve Yule-Walker equations (simplified approach)
	for i := 0; i < a.p; i++ {
		if i < len(autocorrs) {
			a.arCoeffs[i] = autocorrs[i] * 0.5 // simplified coefficient
		}
	}
	
	return nil
}

// fitMAComponent fits the moving average component
func (a *ARIMAModel) fitMAComponent(residuals []float64) error {
	// Simplified MA fitting - use sample autocorrelations of residuals
	autocorrs := a.calculateAutocorrelations(residuals, a.q)
	
	for i := 0; i < a.q; i++ {
		if i < len(autocorrs) {
			a.maCoeffs[i] = autocorrs[i] * 0.3 // simplified coefficient
		}
	}
	
	return nil
}

// calculateAutocorrelations calculates sample autocorrelations
func (a *ARIMAModel) calculateAutocorrelations(values []float64, maxLag int) []float64 {
	n := len(values)
	if n < 2 {
		return []float64{}
	}
	
	// Calculate mean
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(n)
	
	// Calculate variance
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(n - 1)
	
	if variance == 0 {
		return make([]float64, maxLag)
	}
	
	// Calculate autocorrelations
	autocorrs := make([]float64, maxLag)
	for lag := 1; lag <= maxLag; lag++ {
		if lag >= n {
			autocorrs[lag-1] = 0
			continue
		}
		
		covariance := 0.0
		count := 0
		for i := 0; i < n-lag; i++ {
			covariance += (values[i] - mean) * (values[i+lag] - mean)
			count++
		}
		
		if count > 0 {
			covariance /= float64(count)
			autocorrs[lag-1] = covariance / variance
		}
	}
	
	return autocorrs
}

// calculateResiduals calculates model residuals
func (a *ARIMAModel) calculateResiduals(values []float64) []float64 {
	residuals := make([]float64, len(values))
	
	for i := range values {
		predicted := a.predictValueAt(values, i)
		residuals[i] = values[i] - predicted
	}
	
	return residuals
}

// predictValueAt predicts value at specific index using AR component
func (a *ARIMAModel) predictValueAt(values []float64, index int) float64 {
	if index < a.p {
		return values[index] // can't predict without enough history
	}
	
	prediction := a.constant
	
	// AR component
	for i := 0; i < a.p; i++ {
		if index-i-1 >= 0 {
			prediction += a.arCoeffs[i] * values[index-i-1]
		}
	}
	
	return prediction
}

// predictNextValue predicts the next value in the series
func (a *ARIMAModel) predictNextValue(recentValues []float64) float64 {
	if len(recentValues) < a.p {
		// Fallback to simple moving average
		sum := 0.0
		for _, v := range recentValues {
			sum += v
		}
		return sum / float64(len(recentValues))
	}
	
	prediction := a.constant
	
	// AR component
	for i := 0; i < a.p && i < len(recentValues); i++ {
		prediction += a.arCoeffs[i] * recentValues[len(recentValues)-1-i]
	}
	
	// MA component (simplified - would need recent residuals)
	if len(a.residuals) > 0 {
		for i := 0; i < a.q && i < len(a.residuals); i++ {
			residualIdx := len(a.residuals) - 1 - i
			if residualIdx >= 0 {
				prediction += a.maCoeffs[i] * a.residuals[residualIdx]
			}
		}
	}
	
	return prediction
}

// calculateConfidence calculates prediction confidence based on forecast horizon
func (a *ARIMAModel) calculateConfidence(stepsAhead int) float64 {
	// Confidence decreases with prediction distance
	baseConfidence := a.accuracy
	decayFactor := math.Exp(-0.1 * float64(stepsAhead))
	return baseConfidence * decayFactor
}

// calculateStandardError calculates prediction standard error
func (a *ARIMAModel) calculateStandardError() float64 {
	if len(a.residuals) == 0 {
		return 1.0 // default error
	}
	
	// Calculate residual standard deviation
	mean := 0.0
	for _, r := range a.residuals {
		mean += r
	}
	mean /= float64(len(a.residuals))
	
	variance := 0.0
	for _, r := range a.residuals {
		diff := r - mean
		variance += diff * diff
	}
	variance /= float64(len(a.residuals) - 1)
	
	return math.Sqrt(variance)
}

// calculateAccuracy calculates model accuracy using various metrics
func (a *ARIMAModel) calculateAccuracy(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}
	
	// Use cross-validation approach for accuracy estimation
	totalError := 0.0
	predictions := 0
	
	// Use last 20% of data for validation
	testStart := int(0.8 * float64(len(values)))
	if testStart < a.p+1 {
		testStart = a.p + 1
	}
	
	for i := testStart; i < len(values); i++ {
		predicted := a.predictValueAt(values[:i], i-1)
		actual := values[i]
		
		error := math.Abs(predicted - actual)
		if actual != 0 {
			error = error / math.Abs(actual) // relative error
		}
		
		totalError += error
		predictions++
	}
	
	if predictions == 0 {
		return 0.0
	}
	
	meanError := totalError / float64(predictions)
	// Convert error to accuracy (0-1 scale)
	accuracy := math.Max(0, 1.0-meanError)
	return accuracy
}

// ExponentialSmoothingModel implements exponential smoothing for simpler forecasting
type ExponentialSmoothingModel struct {
	config   ModelConfig
	trained  bool
	accuracy float64
	alpha    float64 // smoothing parameter
	level    float64 // current level
	trend    float64 // current trend (for double exponential)
	season   []float64 // seasonal components (for triple exponential)
	
	lastPoints []TimeSeriesPoint
	useDouble  bool // use double exponential smoothing
	useTriple  bool // use triple exponential smoothing (Holt-Winters)
	seasonLength int // length of seasonal cycle
}

// NewExponentialSmoothingModel creates a new exponential smoothing model
func NewExponentialSmoothingModel(config ModelConfig) *ExponentialSmoothingModel {
	alpha := 0.3 // default smoothing parameter
	if params, ok := config.Parameters["alpha"]; ok {
		if alphaVal, ok := params.(float64); ok {
			alpha = alphaVal
		}
	}
	
	useDouble := false
	if params, ok := config.Parameters["double"]; ok {
		if doubleVal, ok := params.(bool); ok {
			useDouble = doubleVal
		}
	}
	
	useTriple := false
	if params, ok := config.Parameters["triple"]; ok {
		if tripleVal, ok := params.(bool); ok {
			useTriple = tripleVal
		}
	}
	
	seasonLength := 24 // default to 24 hours
	if params, ok := config.Parameters["season_length"]; ok {
		if seasonVal, ok := params.(int); ok {
			seasonLength = seasonVal
		}
	}
	
	return &ExponentialSmoothingModel{
		config:       config,
		alpha:        alpha,
		useDouble:    useDouble,
		useTriple:    useTriple,
		seasonLength: seasonLength,
	}
}

// Train implements PredictiveModel interface for exponential smoothing
func (e *ExponentialSmoothingModel) Train(ctx context.Context, data *TimeSeriesData) error {
	if len(data.Points) < e.config.MinDataPoints {
		return fmt.Errorf("insufficient data points: got %d, need at least %d", 
			len(data.Points), e.config.MinDataPoints)
	}
	
	// Sort points by timestamp
	sort.Slice(data.Points, func(i, j int) bool {
		return data.Points[i].Timestamp.Before(data.Points[j].Timestamp)
	})
	
	// Extract values
	values := make([]float64, len(data.Points))
	for i, point := range data.Points {
		values[i] = point.Value
	}
	
	// Initialize model
	e.level = values[0]
	if e.useDouble || e.useTriple {
		// Initialize trend
		if len(values) >= 2 {
			e.trend = values[1] - values[0]
		}
	}
	
	if e.useTriple {
		// Initialize seasonal components
		e.season = make([]float64, e.seasonLength)
		if len(values) >= e.seasonLength {
			for i := 0; i < e.seasonLength; i++ {
				e.season[i] = values[i] / e.level
			}
		} else {
			// Fill with 1.0 (no seasonal effect)
			for i := 0; i < e.seasonLength; i++ {
				e.season[i] = 1.0
			}
		}
	}
	
	// Fit model by iterating through data
	for i := 1; i < len(values); i++ {
		e.updateModel(values[i], i)
	}
	
	// Calculate accuracy
	e.accuracy = e.calculateAccuracy(values)
	
	// Store recent points
	recentCount := 100 // keep last 100 points
	if recentCount > len(data.Points) {
		recentCount = len(data.Points)
	}
	
	startIdx := len(data.Points) - recentCount
	e.lastPoints = make([]TimeSeriesPoint, recentCount)
	copy(e.lastPoints, data.Points[startIdx:])
	
	e.trained = true
	return nil
}

// updateModel updates the exponential smoothing model with a new value
func (e *ExponentialSmoothingModel) updateModel(value float64, index int) {
	if e.useTriple {
		// Holt-Winters (triple exponential smoothing)
		seasonIndex := index % e.seasonLength
		oldLevel := e.level
		
		// Update level
		e.level = e.alpha*(value/e.season[seasonIndex]) + (1-e.alpha)*(e.level+e.trend)
		
		// Update trend  
		beta := 0.2 // trend smoothing parameter
		e.trend = beta*(e.level-oldLevel) + (1-beta)*e.trend
		
		// Update seasonal component
		gamma := 0.1 // seasonal smoothing parameter
		e.season[seasonIndex] = gamma*(value/e.level) + (1-gamma)*e.season[seasonIndex]
		
	} else if e.useDouble {
		// Double exponential smoothing (Holt's method)
		oldLevel := e.level
		
		// Update level
		e.level = e.alpha*value + (1-e.alpha)*(e.level+e.trend)
		
		// Update trend
		beta := 0.2 // trend smoothing parameter  
		e.trend = beta*(e.level-oldLevel) + (1-beta)*e.trend
		
	} else {
		// Simple exponential smoothing
		e.level = e.alpha*value + (1-e.alpha)*e.level
	}
}

// Predict implements PredictiveModel interface for exponential smoothing
func (e *ExponentialSmoothingModel) Predict(ctx context.Context, horizon time.Duration, interval time.Duration) (*ForecastResult, error) {
	if !e.trained {
		return nil, fmt.Errorf("model not trained")
	}
	
	// Calculate number of predictions needed
	numPredictions := int(horizon / interval)
	if numPredictions <= 0 {
		return nil, fmt.Errorf("invalid prediction horizon or interval")
	}
	
	predictions := make([]Prediction, numPredictions)
	lastTimestamp := e.lastPoints[len(e.lastPoints)-1].Timestamp
	
	for i := 0; i < numPredictions; i++ {
		predictTimestamp := lastTimestamp.Add(time.Duration(i+1) * interval)
		
		var prediction float64
		if e.useTriple {
			// Holt-Winters prediction
			seasonIndex := (len(e.lastPoints) + i) % e.seasonLength
			prediction = (e.level + float64(i+1)*e.trend) * e.season[seasonIndex]
		} else if e.useDouble {
			// Double exponential prediction  
			prediction = e.level + float64(i+1)*e.trend
		} else {
			// Simple exponential prediction
			prediction = e.level
		}
		
		// Simple confidence calculation
		confidence := math.Max(0.1, e.accuracy*math.Exp(-0.05*float64(i+1)))
		
		// Confidence intervals (simplified)
		stdErr := e.calculateStandardError()
		lower := prediction - 1.96*stdErr
		upper := prediction + 1.96*stdErr
		
		predictions[i] = Prediction{
			Timestamp:  predictTimestamp,
			Value:      prediction,
			Confidence: confidence,
			Lower:      lower,
			Upper:      upper,
			ModelType:  ModelTypeExponentialSmoothing,
		}
	}
	
	return &ForecastResult{
		ModelType:     ModelTypeExponentialSmoothing,
		Predictions:   predictions,
		ModelAccuracy: e.accuracy,
		GeneratedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"alpha":         e.alpha,
			"double":        e.useDouble,
			"triple":        e.useTriple,
			"season_length": e.seasonLength,
		},
	}, nil
}

// Update implements PredictiveModel interface for exponential smoothing
func (e *ExponentialSmoothingModel) Update(ctx context.Context, points []TimeSeriesPoint) error {
	if !e.trained {
		return fmt.Errorf("model not trained")
	}
	
	// Update model with new points
	for _, point := range points {
		index := len(e.lastPoints)
		e.updateModel(point.Value, index)
	}
	
	// Add new points to history
	e.lastPoints = append(e.lastPoints, points...)
	
	// Keep only recent points
	maxPoints := e.config.MaxDataPoints
	if maxPoints > 0 && len(e.lastPoints) > maxPoints {
		startIdx := len(e.lastPoints) - maxPoints
		e.lastPoints = e.lastPoints[startIdx:]
	}
	
	return nil
}

// GetAccuracy implements PredictiveModel interface
func (e *ExponentialSmoothingModel) GetAccuracy() float64 {
	return e.accuracy
}

// GetType implements PredictiveModel interface
func (e *ExponentialSmoothingModel) GetType() ModelType {
	return ModelTypeExponentialSmoothing
}

// GetConfig implements PredictiveModel interface
func (e *ExponentialSmoothingModel) GetConfig() ModelConfig {
	return e.config
}

// IsReady implements PredictiveModel interface
func (e *ExponentialSmoothingModel) IsReady() bool {
	return e.trained
}

// calculateStandardError calculates prediction standard error for exponential smoothing
func (e *ExponentialSmoothingModel) calculateStandardError() float64 {
	// Simplified error calculation
	return 0.1 * e.level // 10% of current level as error estimate
}

// calculateAccuracy calculates model accuracy for exponential smoothing
func (e *ExponentialSmoothingModel) calculateAccuracy(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}
	
	// Calculate accuracy using one-step-ahead predictions
	totalError := 0.0
	predictions := 0
	
	// Reset model state for accuracy calculation
	tempLevel := values[0]
	tempTrend := 0.0
	if len(values) >= 2 {
		tempTrend = values[1] - values[0]
	}
	
	for i := 1; i < len(values); i++ {
		var predicted float64
		if e.useDouble {
			predicted = tempLevel + tempTrend
		} else {
			predicted = tempLevel
		}
		
		actual := values[i]
		error := math.Abs(predicted - actual)
		if actual != 0 {
			error = error / math.Abs(actual)
		}
		
		totalError += error
		predictions++
		
		// Update temporary state
		if e.useDouble {
			oldLevel := tempLevel
			tempLevel = e.alpha*actual + (1-e.alpha)*(tempLevel+tempTrend)
			tempTrend = 0.2*(tempLevel-oldLevel) + 0.8*tempTrend
		} else {
			tempLevel = e.alpha*actual + (1-e.alpha)*tempLevel
		}
	}
	
	if predictions == 0 {
		return 0.0
	}
	
	meanError := totalError / float64(predictions)
	accuracy := math.Max(0, 1.0-meanError)
	return accuracy
}

// ModelFactory creates predictive models based on configuration
type ModelFactory struct{}

// CreateModel creates a predictive model of the specified type
func (f *ModelFactory) CreateModel(config ModelConfig) (PredictiveModel, error) {
	switch config.Type {
	case ModelTypeARIMA:
		return NewARIMAModel(config), nil
	case ModelTypeExponentialSmoothing:
		return NewExponentialSmoothingModel(config), nil
	case ModelTypeLinear:
		return NewLinearModel(config), nil
	default:
		return nil, fmt.Errorf("unsupported model type: %s", config.Type)
	}
}

// LinearModel implements simple linear regression for trend forecasting
type LinearModel struct {
	config     ModelConfig
	trained    bool
	accuracy   float64
	slope      float64
	intercept  float64
	lastPoints []TimeSeriesPoint
}

// NewLinearModel creates a new linear regression model
func NewLinearModel(config ModelConfig) *LinearModel {
	return &LinearModel{
		config: config,
	}
}

// Train implements PredictiveModel interface for linear model
func (l *LinearModel) Train(ctx context.Context, data *TimeSeriesData) error {
	if len(data.Points) < l.config.MinDataPoints {
		return fmt.Errorf("insufficient data points: got %d, need at least %d", 
			len(data.Points), l.config.MinDataPoints)
	}
	
	// Sort points by timestamp
	sort.Slice(data.Points, func(i, j int) bool {
		return data.Points[i].Timestamp.Before(data.Points[j].Timestamp)
	})
	
	// Convert timestamps to numeric values (hours since first point)
	firstTime := data.Points[0].Timestamp
	x := make([]float64, len(data.Points))
	y := make([]float64, len(data.Points))
	
	for i, point := range data.Points {
		x[i] = point.Timestamp.Sub(firstTime).Hours()
		y[i] = point.Value
	}
	
	// Fit linear regression using least squares
	l.slope, l.intercept = l.fitLinearRegression(x, y)
	
	// Calculate accuracy
	l.accuracy = l.calculateLinearAccuracy(x, y)
	
	// Store points
	l.lastPoints = make([]TimeSeriesPoint, len(data.Points))
	copy(l.lastPoints, data.Points)
	
	l.trained = true
	return nil
}

// fitLinearRegression fits a linear regression model using least squares
func (l *LinearModel) fitLinearRegression(x, y []float64) (slope, intercept float64) {
	n := len(x)
	if n == 0 {
		return 0, 0
	}
	
	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := 0; i < n; i++ {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(n)
	meanY /= float64(n)
	
	// Calculate slope and intercept
	numerator := 0.0
	denominator := 0.0
	for i := 0; i < n; i++ {
		xDiff := x[i] - meanX
		yDiff := y[i] - meanY
		numerator += xDiff * yDiff
		denominator += xDiff * xDiff
	}
	
	if denominator != 0 {
		slope = numerator / denominator
	}
	intercept = meanY - slope*meanX
	
	return slope, intercept
}

// calculateLinearAccuracy calculates R-squared for linear model
func (l *LinearModel) calculateLinearAccuracy(x, y []float64) float64 {
	n := len(y)
	if n == 0 {
		return 0.0
	}
	
	// Calculate mean of y
	meanY := 0.0
	for _, val := range y {
		meanY += val
	}
	meanY /= float64(n)
	
	// Calculate total sum of squares and residual sum of squares
	totalSumSquares := 0.0
	residualSumSquares := 0.0
	
	for i := 0; i < n; i++ {
		predicted := l.intercept + l.slope*x[i]
		totalSumSquares += (y[i] - meanY) * (y[i] - meanY)
		residualSumSquares += (y[i] - predicted) * (y[i] - predicted)
	}
	
	// Calculate R-squared
	if totalSumSquares == 0 {
		return 0.0
	}
	
	rSquared := 1.0 - (residualSumSquares / totalSumSquares)
	return math.Max(0, rSquared)
}

// Predict implements PredictiveModel interface for linear model
func (l *LinearModel) Predict(ctx context.Context, horizon time.Duration, interval time.Duration) (*ForecastResult, error) {
	if !l.trained {
		return nil, fmt.Errorf("model not trained")
	}
	
	if len(l.lastPoints) == 0 {
		return nil, fmt.Errorf("no historical data available")
	}
	
	// Calculate number of predictions needed
	numPredictions := int(horizon / interval)
	if numPredictions <= 0 {
		return nil, fmt.Errorf("invalid prediction horizon or interval")
	}
	
	predictions := make([]Prediction, numPredictions)
	firstTime := l.lastPoints[0].Timestamp
	lastTime := l.lastPoints[len(l.lastPoints)-1].Timestamp
	
	for i := 0; i < numPredictions; i++ {
		predictTimestamp := lastTime.Add(time.Duration(i+1) * interval)
		
		// Convert timestamp to numeric value
		x := predictTimestamp.Sub(firstTime).Hours()
		
		// Linear prediction
		prediction := l.intercept + l.slope*x
		
		// Simple confidence (decreases with distance)
		confidence := l.accuracy * math.Exp(-0.01*float64(i+1))
		
		// Confidence intervals (simplified)
		stdErr := l.calculateLinearStandardError()
		lower := prediction - 1.96*stdErr
		upper := prediction + 1.96*stdErr
		
		predictions[i] = Prediction{
			Timestamp:  predictTimestamp,
			Value:      prediction,
			Confidence: confidence,
			Lower:      lower,
			Upper:      upper,
			ModelType:  ModelTypeLinear,
		}
	}
	
	return &ForecastResult{
		ModelType:     ModelTypeLinear,
		Predictions:   predictions,
		ModelAccuracy: l.accuracy,
		GeneratedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"slope":     l.slope,
			"intercept": l.intercept,
		},
	}, nil
}

// Update implements PredictiveModel interface for linear model
func (l *LinearModel) Update(ctx context.Context, points []TimeSeriesPoint) error {
	if !l.trained {
		return fmt.Errorf("model not trained")
	}
	
	// Add new points
	l.lastPoints = append(l.lastPoints, points...)
	
	// Keep only recent points
	maxPoints := l.config.MaxDataPoints
	if maxPoints > 0 && len(l.lastPoints) > maxPoints {
		startIdx := len(l.lastPoints) - maxPoints
		l.lastPoints = l.lastPoints[startIdx:]
	}
	
	// Retrain with new data periodically
	// For now, just update the points - could implement incremental learning
	
	return nil
}

// GetAccuracy implements PredictiveModel interface
func (l *LinearModel) GetAccuracy() float64 {
	return l.accuracy
}

// GetType implements PredictiveModel interface
func (l *LinearModel) GetType() ModelType {
	return ModelTypeLinear
}

// GetConfig implements PredictiveModel interface
func (l *LinearModel) GetConfig() ModelConfig {
	return l.config
}

// IsReady implements PredictiveModel interface
func (l *LinearModel) IsReady() bool {
	return l.trained
}

// calculateLinearStandardError calculates standard error for linear model
func (l *LinearModel) calculateLinearStandardError() float64 {
	if len(l.lastPoints) < 2 {
		return 1.0
	}
	
	// Calculate standard error of estimate
	firstTime := l.lastPoints[0].Timestamp
	sumSquaredErrors := 0.0
	n := len(l.lastPoints)
	
	for _, point := range l.lastPoints {
		x := point.Timestamp.Sub(firstTime).Hours()
		predicted := l.intercept + l.slope*x
		error := point.Value - predicted
		sumSquaredErrors += error * error
	}
	
	if n <= 2 {
		return 1.0
	}
	
	return math.Sqrt(sumSquaredErrors / float64(n-2))
}