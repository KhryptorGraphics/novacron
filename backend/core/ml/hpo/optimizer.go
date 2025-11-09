package hpo

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// HPOOptimizer performs hyperparameter optimization
type HPOOptimizer struct {
	config     *HPOConfig
	space      map[string]ParamDef
	evaluator  ObjectiveFunction
	trials     []Trial
	bestTrial  *Trial
	mu         sync.RWMutex
}

// HPOConfig defines HPO configuration
type HPOConfig struct {
	Algorithm           string        // "bayesian", "hyperband", "grid", "random"
	MaxTrials           int           // Maximum number of trials
	ParallelTrials      int           // Parallel trial execution
	MetricGoal          string        // "minimize" or "maximize"
	EarlyStoppingRounds int           // Early stopping patience
	TimeoutPerTrial     time.Duration // Timeout per trial
	Seed                int64         // Random seed
}

// ParamDef defines a hyperparameter
type ParamDef struct {
	Type   string      // "int", "float", "categorical"
	Min    float64     // For int/float
	Max    float64     // For int/float
	Values []interface{} // For categorical
	Scale  string      // "linear", "log"
}

// Trial represents a single HPO trial
type Trial struct {
	ID         int
	Params     map[string]interface{}
	Metrics    map[string]float64
	Status     string // "pending", "running", "completed", "failed"
	StartTime  time.Time
	Duration   time.Duration
	Error      error
}

// ObjectiveFunction evaluates a parameter configuration
type ObjectiveFunction func(params map[string]interface{}) (map[string]float64, error)

// BayesianOptimizer implements Bayesian optimization
type BayesianOptimizer struct {
	gp          *GaussianProcess
	acquisitionFunc string // "ei", "ucb", "poi"
	kappa       float64    // UCB exploration parameter
	xi          float64    // EI/POI exploration parameter
}

// HyperbandOptimizer implements Hyperband algorithm
type HyperbandOptimizer struct {
	maxResource int
	eta         float64
	brackets    []HyperbandBracket
}

// HyperbandBracket represents a Hyperband bracket
type HyperbandBracket struct {
	n           int
	r           int
	configs     []map[string]interface{}
	performance []float64
}

// GaussianProcess implements GP for Bayesian optimization
type GaussianProcess struct {
	X      [][]float64
	Y      []float64
	kernel KernelFunction
	noise  float64
	mu     sync.RWMutex
}

// KernelFunction computes kernel between points
type KernelFunction func(x1, x2 []float64) float64

// NewHPOOptimizer creates a new HPO optimizer
func NewHPOOptimizer(config *HPOConfig, space map[string]ParamDef, evaluator ObjectiveFunction) *HPOOptimizer {
	if config == nil {
		config = DefaultHPOConfig()
	}

	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}

	return &HPOOptimizer{
		config:    config,
		space:     space,
		evaluator: evaluator,
		trials:    make([]Trial, 0),
	}
}

// DefaultHPOConfig returns default HPO configuration
func DefaultHPOConfig() *HPOConfig {
	return &HPOConfig{
		Algorithm:           "bayesian",
		MaxTrials:           100,
		ParallelTrials:      4,
		MetricGoal:          "minimize",
		EarlyStoppingRounds: 10,
		TimeoutPerTrial:     5 * time.Minute,
		Seed:                0,
	}
}

// Optimize runs hyperparameter optimization
func (opt *HPOOptimizer) Optimize(ctx context.Context, targetMetric string) (*Trial, error) {
	switch opt.config.Algorithm {
	case "bayesian":
		return opt.bayesianOptimize(ctx, targetMetric)
	case "hyperband":
		return opt.hyperbandOptimize(ctx, targetMetric)
	case "grid":
		return opt.gridSearch(ctx, targetMetric)
	case "random":
		return opt.randomSearch(ctx, targetMetric)
	default:
		return nil, fmt.Errorf("unknown algorithm: %s", opt.config.Algorithm)
	}
}

// bayesianOptimize performs Bayesian optimization
func (opt *HPOOptimizer) bayesianOptimize(ctx context.Context, targetMetric string) (*Trial, error) {
	bo := &BayesianOptimizer{
		gp:          NewGaussianProcess(RBFKernel(1.0), 0.01),
		acquisitionFunc: "ei",
		kappa:       2.576, // 99% confidence
		xi:          0.01,
	}

	// Initial random trials
	initialTrials := 10
	for i := 0; i < initialTrials; i++ {
		params := opt.sampleRandom()
		trial := opt.executeTrial(ctx, i, params, targetMetric)

		opt.mu.Lock()
		opt.trials = append(opt.trials, trial)
		if trial.Status == "completed" {
			X := opt.paramsToVector(params)
			bo.gp.Fit(X, trial.Metrics[targetMetric])
		}
		opt.mu.Unlock()
	}

	// Bayesian optimization loop
	for i := initialTrials; i < opt.config.MaxTrials; i++ {
		// Select next parameters using acquisition function
		params := opt.selectNextBayesian(bo, targetMetric)

		trial := opt.executeTrial(ctx, i, params, targetMetric)

		opt.mu.Lock()
		opt.trials = append(opt.trials, trial)
		if trial.Status == "completed" {
			X := opt.paramsToVector(params)
			bo.gp.Fit(X, trial.Metrics[targetMetric])

			if opt.bestTrial == nil || opt.isBetter(trial.Metrics[targetMetric], opt.bestTrial.Metrics[targetMetric]) {
				opt.bestTrial = &trial
			}
		}
		opt.mu.Unlock()

		// Early stopping
		if opt.shouldEarlyStop(targetMetric) {
			break
		}
	}

	return opt.bestTrial, nil
}

// hyperbandOptimize performs Hyperband optimization
func (opt *HPOOptimizer) hyperbandOptimize(ctx context.Context, targetMetric string) (*Trial, error) {
	hb := &HyperbandOptimizer{
		maxResource: 81,  // Max epochs
		eta:         3.0, // Reduction factor
	}

	maxIter := int(math.Log(float64(hb.maxResource)) / math.Log(hb.eta))

	for s := maxIter; s >= 0; s-- {
		n := int(math.Ceil(float64(hb.maxResource) / float64(s+1) * math.Pow(hb.eta, float64(s))))
		r := hb.maxResource * int(math.Pow(hb.eta, float64(-s)))

		// Create bracket
		bracket := HyperbandBracket{
			n:           n,
			r:           r,
			configs:     make([]map[string]interface{}, n),
			performance: make([]float64, n),
		}

		// Generate random configurations
		for i := 0; i < n; i++ {
			bracket.configs[i] = opt.sampleRandom()
		}

		// Successive halving
		for i := 0; i <= s; i++ {
			ni := int(math.Floor(float64(n) * math.Pow(hb.eta, float64(-i))))
			ri := r * int(math.Pow(hb.eta, float64(i)))

			// Evaluate configurations
			for j := 0; j < ni; j++ {
				params := bracket.configs[j]
				params["epochs"] = ri

				trial := opt.executeTrial(ctx, len(opt.trials), params, targetMetric)

				opt.mu.Lock()
				opt.trials = append(opt.trials, trial)
				if trial.Status == "completed" {
					bracket.performance[j] = trial.Metrics[targetMetric]

					if opt.bestTrial == nil || opt.isBetter(trial.Metrics[targetMetric], opt.bestTrial.Metrics[targetMetric]) {
						opt.bestTrial = &trial
					}
				}
				opt.mu.Unlock()
			}

			// Select top performers
			if i < s {
				indices := opt.argsort(bracket.performance[:ni])
				keep := int(math.Floor(float64(ni) / hb.eta))

				newConfigs := make([]map[string]interface{}, keep)
				newPerf := make([]float64, keep)
				for k := 0; k < keep; k++ {
					newConfigs[k] = bracket.configs[indices[k]]
					newPerf[k] = bracket.performance[indices[k]]
				}

				bracket.configs = newConfigs
				bracket.performance = newPerf
			}
		}
	}

	return opt.bestTrial, nil
}

// gridSearch performs grid search
func (opt *HPOOptimizer) gridSearch(ctx context.Context, targetMetric string) (*Trial, error) {
	// Generate grid
	grid := opt.generateGrid()

	// Parallel evaluation
	trialsChan := make(chan Trial, len(grid))
	resultsChan := make(chan Trial, len(grid))

	// Worker pool
	var wg sync.WaitGroup
	for i := 0; i < opt.config.ParallelTrials; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for trial := range trialsChan {
				result := opt.executeTrial(ctx, trial.ID, trial.Params, targetMetric)
				resultsChan <- result
			}
		}()
	}

	// Submit trials
	go func() {
		for i, params := range grid {
			trial := Trial{
				ID:     i,
				Params: params,
				Status: "pending",
			}
			trialsChan <- trial
		}
		close(trialsChan)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	for trial := range resultsChan {
		opt.mu.Lock()
		opt.trials = append(opt.trials, trial)
		if trial.Status == "completed" {
			if opt.bestTrial == nil || opt.isBetter(trial.Metrics[targetMetric], opt.bestTrial.Metrics[targetMetric]) {
				opt.bestTrial = &trial
			}
		}
		opt.mu.Unlock()
	}

	return opt.bestTrial, nil
}

// randomSearch performs random search
func (opt *HPOOptimizer) randomSearch(ctx context.Context, targetMetric string) (*Trial, error) {
	for i := 0; i < opt.config.MaxTrials; i++ {
		params := opt.sampleRandom()
		trial := opt.executeTrial(ctx, i, params, targetMetric)

		opt.mu.Lock()
		opt.trials = append(opt.trials, trial)
		if trial.Status == "completed" {
			if opt.bestTrial == nil || opt.isBetter(trial.Metrics[targetMetric], opt.bestTrial.Metrics[targetMetric]) {
				opt.bestTrial = &trial
			}
		}
		opt.mu.Unlock()

		if opt.shouldEarlyStop(targetMetric) {
			break
		}
	}

	return opt.bestTrial, nil
}

// executeTrial executes a single trial
func (opt *HPOOptimizer) executeTrial(ctx context.Context, id int, params map[string]interface{}, targetMetric string) Trial {
	trial := Trial{
		ID:        id,
		Params:    params,
		Status:    "running",
		StartTime: time.Now(),
	}

	trialCtx, cancel := context.WithTimeout(ctx, opt.config.TimeoutPerTrial)
	defer cancel()

	done := make(chan bool)
	var metrics map[string]float64
	var err error

	go func() {
		metrics, err = opt.evaluator(params)
		done <- true
	}()

	select {
	case <-done:
		if err != nil {
			trial.Status = "failed"
			trial.Error = err
		} else {
			trial.Status = "completed"
			trial.Metrics = metrics
		}
	case <-trialCtx.Done():
		trial.Status = "failed"
		trial.Error = fmt.Errorf("trial timeout")
	}

	trial.Duration = time.Since(trial.StartTime)
	return trial
}

// sampleRandom samples random parameters
func (opt *HPOOptimizer) sampleRandom() map[string]interface{} {
	params := make(map[string]interface{})

	for name, def := range opt.space {
		params[name] = opt.sampleParam(def)
	}

	return params
}

// sampleParam samples a single parameter
func (opt *HPOOptimizer) sampleParam(def ParamDef) interface{} {
	switch def.Type {
	case "int":
		val := def.Min + rand.Float64()*(def.Max-def.Min)
		return int(val)
	case "float":
		if def.Scale == "log" {
			logMin := math.Log(def.Min)
			logMax := math.Log(def.Max)
			return math.Exp(logMin + rand.Float64()*(logMax-logMin))
		}
		return def.Min + rand.Float64()*(def.Max-def.Min)
	case "categorical":
		return def.Values[rand.Intn(len(def.Values))]
	default:
		return nil
	}
}

// selectNextBayesian selects next params using Bayesian optimization
func (opt *HPOOptimizer) selectNextBayesian(bo *BayesianOptimizer, targetMetric string) map[string]interface{} {
	// Generate candidates
	numCandidates := 1000
	var bestParams map[string]interface{}
	bestAcq := math.Inf(-1)

	for i := 0; i < numCandidates; i++ {
		params := opt.sampleRandom()
		x := opt.paramsToVector(params)

		// Compute acquisition function
		acq := bo.acquisitionValue(x, opt.config.MetricGoal == "minimize")

		if acq > bestAcq {
			bestAcq = acq
			bestParams = params
		}
	}

	return bestParams
}

// paramsToVector converts params to vector
func (opt *HPOOptimizer) paramsToVector(params map[string]interface{}) []float64 {
	vec := make([]float64, 0)

	// Sort keys for consistency
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := params[k]
		switch val := v.(type) {
		case int:
			vec = append(vec, float64(val))
		case float64:
			vec = append(vec, val)
		case string:
			// Encode categorical as hash
			vec = append(vec, float64(len(val)))
		}
	}

	return vec
}

// generateGrid generates grid of parameters
func (opt *HPOOptimizer) generateGrid() []map[string]interface{} {
	// Get all parameter values
	paramLists := make(map[string][]interface{})

	for name, def := range opt.space {
		switch def.Type {
		case "int":
			step := int((def.Max - def.Min) / 5)
			if step == 0 {
				step = 1
			}
			values := make([]interface{}, 0)
			for v := int(def.Min); v <= int(def.Max); v += step {
				values = append(values, v)
			}
			paramLists[name] = values
		case "float":
			step := (def.Max - def.Min) / 5
			values := make([]interface{}, 0)
			for v := def.Min; v <= def.Max; v += step {
				values = append(values, v)
			}
			paramLists[name] = values
		case "categorical":
			paramLists[name] = def.Values
		}
	}

	// Generate Cartesian product
	return opt.cartesianProduct(paramLists)
}

// cartesianProduct generates Cartesian product of parameter values
func (opt *HPOOptimizer) cartesianProduct(paramLists map[string][]interface{}) []map[string]interface{} {
	if len(paramLists) == 0 {
		return []map[string]interface{}{{}}
	}

	// Get first parameter
	var firstName string
	var firstValues []interface{}
	for name, values := range paramLists {
		firstName = name
		firstValues = values
		break
	}

	// Remove first parameter
	remaining := make(map[string][]interface{})
	for name, values := range paramLists {
		if name != firstName {
			remaining[name] = values
		}
	}

	// Recursive call
	subResults := opt.cartesianProduct(remaining)

	// Combine
	results := make([]map[string]interface{}, 0)
	for _, val := range firstValues {
		for _, subResult := range subResults {
			result := make(map[string]interface{})
			result[firstName] = val
			for k, v := range subResult {
				result[k] = v
			}
			results = append(results, result)
		}
	}

	return results
}

// isBetter checks if new metric is better than current
func (opt *HPOOptimizer) isBetter(newMetric, currentMetric float64) bool {
	if opt.config.MetricGoal == "minimize" {
		return newMetric < currentMetric
	}
	return newMetric > currentMetric
}

// shouldEarlyStop checks if optimization should stop early
func (opt *HPOOptimizer) shouldEarlyStop(targetMetric string) bool {
	if opt.config.EarlyStoppingRounds <= 0 {
		return false
	}

	if len(opt.trials) < opt.config.EarlyStoppingRounds {
		return false
	}

	// Check if no improvement in last N trials
	recent := opt.trials[len(opt.trials)-opt.config.EarlyStoppingRounds:]
	hasImprovement := false

	for _, trial := range recent {
		if trial.Status == "completed" && opt.bestTrial != nil {
			if opt.isBetter(trial.Metrics[targetMetric], opt.bestTrial.Metrics[targetMetric]) {
				hasImprovement = true
				break
			}
		}
	}

	return !hasImprovement
}

// argsort returns indices that would sort array
func (opt *HPOOptimizer) argsort(arr []float64) []int {
	indices := make([]int, len(arr))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		if opt.config.MetricGoal == "minimize" {
			return arr[indices[i]] < arr[indices[j]]
		}
		return arr[indices[i]] > arr[indices[j]]
	})

	return indices
}

// GetBestTrial returns the best trial
func (opt *HPOOptimizer) GetBestTrial() *Trial {
	opt.mu.RLock()
	defer opt.mu.RUnlock()
	return opt.bestTrial
}

// GetTrials returns all trials
func (opt *HPOOptimizer) GetTrials() []Trial {
	opt.mu.RLock()
	defer opt.mu.RUnlock()
	return opt.trials
}

// NewGaussianProcess creates a new Gaussian Process
func NewGaussianProcess(kernel KernelFunction, noise float64) *GaussianProcess {
	return &GaussianProcess{
		X:      make([][]float64, 0),
		Y:      make([]float64, 0),
		kernel: kernel,
		noise:  noise,
	}
}

// Fit fits GP to data
func (gp *GaussianProcess) Fit(x []float64, y float64) {
	gp.mu.Lock()
	defer gp.mu.Unlock()
	gp.X = append(gp.X, x)
	gp.Y = append(gp.Y, y)
}

// Predict predicts mean and variance at x
func (gp *GaussianProcess) Predict(x []float64) (float64, float64) {
	gp.mu.RLock()
	defer gp.mu.RUnlock()

	if len(gp.X) == 0 {
		return 0, 1
	}

	// Compute kernel matrix
	K := make([][]float64, len(gp.X))
	for i := range K {
		K[i] = make([]float64, len(gp.X))
		for j := range K[i] {
			K[i][j] = gp.kernel(gp.X[i], gp.X[j])
			if i == j {
				K[i][j] += gp.noise
			}
		}
	}

	// Compute k*
	kStar := make([]float64, len(gp.X))
	for i := range kStar {
		kStar[i] = gp.kernel(gp.X[i], x)
	}

	// Compute k**
	kStarStar := gp.kernel(x, x)

	// Solve K * alpha = y
	alpha := gp.solve(K, gp.Y)

	// Compute mean
	mean := 0.0
	for i := range alpha {
		mean += alpha[i] * kStar[i]
	}

	// Compute variance
	v := gp.solve(K, kStar)
	variance := kStarStar
	for i := range v {
		variance -= kStar[i] * v[i]
	}

	return mean, variance
}

// solve solves linear system using Cholesky decomposition
func (gp *GaussianProcess) solve(A [][]float64, b []float64) []float64 {
	n := len(A)
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}

	// Cholesky decomposition
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += L[i][k] * L[j][k]
			}

			if i == j {
				L[i][j] = math.Sqrt(A[i][i] - sum + 1e-6)
			} else {
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}

	// Forward substitution
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < i; j++ {
			sum += L[i][j] * y[j]
		}
		y[i] = (b[i] - sum) / L[i][i]
	}

	// Backward substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += L[j][i] * x[j]
		}
		x[i] = (y[i] - sum) / L[i][i]
	}

	return x
}

// RBFKernel returns RBF kernel function
func RBFKernel(lengthScale float64) KernelFunction {
	return func(x1, x2 []float64) float64 {
		dist := 0.0
		for i := range x1 {
			diff := x1[i] - x2[i]
			dist += diff * diff
		}
		return math.Exp(-dist / (2 * lengthScale * lengthScale))
	}
}

// acquisitionValue computes acquisition function value
func (bo *BayesianOptimizer) acquisitionValue(x []float64, minimize bool) float64 {
	mean, variance := bo.gp.Predict(x)
	std := math.Sqrt(variance)

	switch bo.acquisitionFunc {
	case "ucb":
		if minimize {
			return -(mean - bo.kappa*std)
		}
		return mean + bo.kappa*std
	case "ei":
		return bo.expectedImprovement(mean, std, minimize)
	case "poi":
		return bo.probabilityOfImprovement(mean, std, minimize)
	default:
		return mean + bo.kappa*std
	}
}

// expectedImprovement computes Expected Improvement
func (bo *BayesianOptimizer) expectedImprovement(mean, std float64, minimize bool) float64 {
	if len(bo.gp.Y) == 0 {
		return 0
	}

	fBest := bo.gp.Y[0]
	for _, y := range bo.gp.Y {
		if minimize && y < fBest {
			fBest = y
		} else if !minimize && y > fBest {
			fBest = y
		}
	}

	if std == 0 {
		return 0
	}

	imp := mean - fBest - bo.xi
	if minimize {
		imp = fBest - mean - bo.xi
	}

	z := imp / std
	ei := imp*normalCDF(z) + std*normalPDF(z)

	return ei
}

// probabilityOfImprovement computes Probability of Improvement
func (bo *BayesianOptimizer) probabilityOfImprovement(mean, std float64, minimize bool) float64 {
	if len(bo.gp.Y) == 0 {
		return 0
	}

	fBest := bo.gp.Y[0]
	for _, y := range bo.gp.Y {
		if minimize && y < fBest {
			fBest = y
		} else if !minimize && y > fBest {
			fBest = y
		}
	}

	if std == 0 {
		return 0
	}

	imp := mean - fBest - bo.xi
	if minimize {
		imp = fBest - mean - bo.xi
	}

	return normalCDF(imp / std)
}

// normalPDF computes normal probability density function
func normalPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// normalCDF computes normal cumulative distribution function
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
