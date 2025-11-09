package algorithms

import (
	"fmt"
	"math"
	"sync"
)

// DPMechanism defines differential privacy mechanism interface
type DPMechanism interface {
	AddNoise(value float64) float64
	GetPrivacyBudget() float64
}

// LaplaceMechanism implements Laplace mechanism for DP
type LaplaceMechanism struct {
	epsilon    float64
	sensitivity float64
	counter    int
	mu         sync.Mutex
}

// NewLaplaceMechanism creates a new Laplace mechanism
func NewLaplaceMechanism(epsilon, sensitivity float64) *LaplaceMechanism {
	return &LaplaceMechanism{
		epsilon:     epsilon,
		sensitivity: sensitivity,
	}
}

// AddNoise adds Laplace noise to a value
func (lm *LaplaceMechanism) AddNoise(value float64) float64 {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// Laplace noise: scale = sensitivity / epsilon
	scale := lm.sensitivity / lm.epsilon

	// Generate Laplace random variable
	noise := lm.laplaceSample(0, scale)

	lm.counter++
	return value + noise
}

// laplaceSample generates a Laplace distributed random variable
func (lm *LaplaceMechanism) laplaceSample(mu, b float64) float64 {
	// Using inverse transform: X = mu - b * sign(U) * ln(1 - 2|U|)
	// Simplified version
	u := float64(lm.counter%100) / 100.0
	if u > 0.5 {
		return mu - b*math.Log(2*(1-u))
	}
	return mu + b*math.Log(2*u)
}

// GetPrivacyBudget returns consumed privacy budget
func (lm *LaplaceMechanism) GetPrivacyBudget() float64 {
	return lm.epsilon * float64(lm.counter)
}

// GaussianMechanism implements Gaussian mechanism for DP
type GaussianMechanism struct {
	epsilon    float64
	delta      float64
	sensitivity float64
	counter    int
	mu         sync.Mutex
}

// NewGaussianMechanism creates a new Gaussian mechanism
func NewGaussianMechanism(epsilon, delta, sensitivity float64) *GaussianMechanism {
	return &GaussianMechanism{
		epsilon:     epsilon,
		delta:       delta,
		sensitivity: sensitivity,
	}
}

// AddNoise adds Gaussian noise to a value
func (gm *GaussianMechanism) AddNoise(value float64) float64 {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	// Gaussian noise: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
	sigma := (gm.sensitivity * math.Sqrt(2*math.Log(1.25/gm.delta))) / gm.epsilon

	noise := gm.gaussianSample(0, sigma)

	gm.counter++
	return value + noise
}

// gaussianSample generates a Gaussian distributed random variable
func (gm *GaussianMechanism) gaussianSample(mu, sigma float64) float64 {
	// Box-Muller transform
	u1 := float64(gm.counter%100+1) / 101.0
	u2 := float64((gm.counter+1)%100+1) / 101.0

	z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mu + sigma*z0
}

// GetPrivacyBudget returns consumed privacy budget
func (gm *GaussianMechanism) GetPrivacyBudget() float64 {
	return gm.epsilon * float64(gm.counter)
}

// DPQueryEngine provides differentially private queries
type DPQueryEngine struct {
	mechanism  DPMechanism
	budgetUsed float64
	budgetMax  float64
	mu         sync.RWMutex
}

// NewDPQueryEngine creates a new DP query engine
func NewDPQueryEngine(mechanism DPMechanism, budgetMax float64) *DPQueryEngine {
	return &DPQueryEngine{
		mechanism: mechanism,
		budgetMax: budgetMax,
	}
}

// Count performs a differentially private count query
func (dpe *DPQueryEngine) Count(data []float64) (float64, error) {
	if !dpe.checkBudget() {
		return 0, fmt.Errorf("privacy budget exceeded")
	}

	trueCount := float64(len(data))
	noisyCount := dpe.mechanism.AddNoise(trueCount)

	dpe.updateBudget()
	return noisyCount, nil
}

// Sum performs a differentially private sum query
func (dpe *DPQueryEngine) Sum(data []float64) (float64, error) {
	if !dpe.checkBudget() {
		return 0, fmt.Errorf("privacy budget exceeded")
	}

	trueSum := 0.0
	for _, v := range data {
		trueSum += v
	}

	noisySum := dpe.mechanism.AddNoise(trueSum)

	dpe.updateBudget()
	return noisySum, nil
}

// Average performs a differentially private average query
func (dpe *DPQueryEngine) Average(data []float64) (float64, error) {
	if !dpe.checkBudget() {
		return 0, fmt.Errorf("privacy budget exceeded")
	}

	if len(data) == 0 {
		return 0, nil
	}

	// Use composition: count + sum
	noisyCount, err := dpe.Count(data)
	if err != nil {
		return 0, err
	}

	noisySum, err := dpe.Sum(data)
	if err != nil {
		return 0, err
	}

	if noisyCount == 0 {
		return 0, nil
	}

	return noisySum / noisyCount, nil
}

// checkBudget checks if budget is available
func (dpe *DPQueryEngine) checkBudget() bool {
	dpe.mu.RLock()
	defer dpe.mu.RUnlock()

	return dpe.budgetUsed < dpe.budgetMax
}

// updateBudget updates consumed budget
func (dpe *DPQueryEngine) updateBudget() {
	dpe.mu.Lock()
	defer dpe.mu.Unlock()

	dpe.budgetUsed = dpe.mechanism.GetPrivacyBudget()
}

// GetRemainingBudget returns remaining privacy budget
func (dpe *DPQueryEngine) GetRemainingBudget() float64 {
	dpe.mu.RLock()
	defer dpe.mu.RUnlock()

	return dpe.budgetMax - dpe.budgetUsed
}

// LocalDifferentialPrivacy implements local DP
type LocalDifferentialPrivacy struct {
	epsilon float64
}

// NewLocalDifferentialPrivacy creates a local DP instance
func NewLocalDifferentialPrivacy(epsilon float64) *LocalDifferentialPrivacy {
	return &LocalDifferentialPrivacy{
		epsilon: epsilon,
	}
}

// RandomizedResponse implements randomized response for local DP
func (ldp *LocalDifferentialPrivacy) RandomizedResponse(value bool) bool {
	// Probability of truth: p = e^epsilon / (e^epsilon + 1)
	p := math.Exp(ldp.epsilon) / (math.Exp(ldp.epsilon) + 1)

	// Simplified randomization
	random := float64(1) / 2.0 // placeholder
	if random < p {
		return value
	}
	return !value
}

// UnaryEncoding implements unary encoding for local DP
func (ldp *LocalDifferentialPrivacy) UnaryEncoding(value int, domain int) []bool {
	encoded := make([]bool, domain)

	// Encode value
	if value >= 0 && value < domain {
		encoded[value] = true
	}

	// Add noise to each bit
	for i := range encoded {
		encoded[i] = ldp.RandomizedResponse(encoded[i])
	}

	return encoded
}

// RAPPOR implements RAPPOR (Randomized Aggregatable Privacy-Preserving Ordinal Response)
type RAPPOR struct {
	epsilon    float64
	numBits    int
	numCohorts int
}

// NewRAPPOR creates a new RAPPOR instance
func NewRAPPOR(epsilon float64, numBits, numCohorts int) *RAPPOR {
	return &RAPPOR{
		epsilon:    epsilon,
		numBits:    numBits,
		numCohorts: numCohorts,
	}
}

// Encode encodes a value using RAPPOR
func (r *RAPPOR) Encode(value string) []bool {
	// Hash value to bloom filter
	bloomFilter := r.hashToBloom(value)

	// Add permanent randomness
	permanent := r.addPermanentNoise(bloomFilter)

	// Add instantaneous randomness
	return r.addInstantaneousNoise(permanent)
}

// hashToBloom hashes value to bloom filter
func (r *RAPPOR) hashToBloom(value string) []bool {
	filter := make([]bool, r.numBits)

	// Simple hash function (in production use proper hash)
	for i, c := range value {
		index := (int(c) + i) % r.numBits
		filter[index] = true
	}

	return filter
}

// addPermanentNoise adds permanent randomness
func (r *RAPPOR) addPermanentNoise(filter []bool) []bool {
	noisy := make([]bool, len(filter))
	copy(noisy, filter)

	// Flip bits with probability f
	f := 0.5 * (1 - math.Exp(-r.epsilon/2))

	for i := range noisy {
		if float64(i)/float64(len(noisy)) < f {
			noisy[i] = !noisy[i]
		}
	}

	return noisy
}

// addInstantaneousNoise adds instantaneous randomness
func (r *RAPPOR) addInstantaneousNoise(filter []bool) []bool {
	noisy := make([]bool, len(filter))

	p := 1.0 / (1.0 + math.Exp(r.epsilon))
	q := 1.0 - p

	for i := range filter {
		if filter[i] {
			// Keep with probability q
			noisy[i] = float64(i)/float64(len(filter)) < q
		} else {
			// Flip with probability p
			noisy[i] = float64(i)/float64(len(filter)) < p
		}
	}

	return noisy
}

// PrivacySandwich implements privacy sandwich technique
type PrivacySandwich struct {
	publicEpsilon  float64
	privateEpsilon float64
}

// NewPrivacySandwich creates a privacy sandwich instance
func NewPrivacySandwich(publicEpsilon, privateEpsilon float64) *PrivacySandwich {
	return &PrivacySandwich{
		publicEpsilon:  publicEpsilon,
		privateEpsilon: privateEpsilon,
	}
}

// Process processes data with privacy sandwich
func (ps *PrivacySandwich) Process(publicData, privateData []float64) ([]float64, error) {
	// Public data processing (less privacy)
	publicMech := NewLaplaceMechanism(ps.publicEpsilon, 1.0)

	// Private data processing (more privacy)
	privateMech := NewLaplaceMechanism(ps.privateEpsilon, 1.0)

	result := make([]float64, len(publicData)+len(privateData))

	for i, v := range publicData {
		result[i] = publicMech.AddNoise(v)
	}

	for i, v := range privateData {
		result[len(publicData)+i] = privateMech.AddNoise(v)
	}

	return result, nil
}
