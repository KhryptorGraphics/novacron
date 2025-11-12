// Package common provides shared utilities and framework for NovaCron research prototypes
// This framework supports experimentation, benchmarking, and validation across all research domains
package common

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"
)

// ExperimentStatus represents the lifecycle state of a research experiment
type ExperimentStatus string

const (
	StatusPending    ExperimentStatus = "pending"
	StatusRunning    ExperimentStatus = "running"
	StatusCompleted  ExperimentStatus = "completed"
	StatusFailed     ExperimentStatus = "failed"
	StatusValidating ExperimentStatus = "validating"
)

// Experiment represents a single research experiment with metadata
type Experiment struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Domain      string                 `json:"domain"` // quantum, blockchain, ai, etc.
	Status      ExperimentStatus       `json:"status"`
	Config      map[string]interface{} `json:"config"`
	Results     *ExperimentResults     `json:"results,omitempty"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     *time.Time             `json:"end_time,omitempty"`
	Duration    time.Duration          `json:"duration,omitempty"`
	Error       string                 `json:"error,omitempty"`
	Tags        []string               `json:"tags"`
}

// ExperimentResults captures comprehensive experiment outcomes
type ExperimentResults struct {
	Success     bool                   `json:"success"`
	Metrics     map[string]float64     `json:"metrics"`
	Artifacts   []string               `json:"artifacts"`
	Logs        []string               `json:"logs"`
	Metadata    map[string]interface{} `json:"metadata"`
	Validations []ValidationResult     `json:"validations"`
}

// ValidationResult captures validation outcomes
type ValidationResult struct {
	Name      string    `json:"name"`
	Passed    bool      `json:"passed"`
	Score     float64   `json:"score"`
	Threshold float64   `json:"threshold"`
	Details   string    `json:"details"`
	Timestamp time.Time `json:"timestamp"`
}

// ResearchLab manages experiments and coordinates research activities
type ResearchLab struct {
	mu          sync.RWMutex
	experiments map[string]*Experiment
	logger      *log.Logger
	dataDir     string
}

// NewResearchLab creates a new research laboratory instance
func NewResearchLab(dataDir string, logger *log.Logger) (*ResearchLab, error) {
	if logger == nil {
		logger = log.New(os.Stdout, "[RESEARCH] ", log.LstdFlags|log.Lshortfile)
	}

	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	return &ResearchLab{
		experiments: make(map[string]*Experiment),
		logger:      logger,
		dataDir:     dataDir,
	}, nil
}

// RegisterExperiment registers a new experiment
func (rl *ResearchLab) RegisterExperiment(exp *Experiment) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if exp.ID == "" {
		exp.ID = generateExperimentID(exp.Domain, exp.Name)
	}

	if exp.Status == "" {
		exp.Status = StatusPending
	}

	exp.StartTime = time.Now()

	rl.experiments[exp.ID] = exp
	rl.logger.Printf("Registered experiment: %s (%s)", exp.Name, exp.ID)

	return rl.saveExperiment(exp)
}

// RunExperiment executes an experiment with the provided function
func (rl *ResearchLab) RunExperiment(ctx context.Context, expID string, runFunc func(context.Context, *Experiment) (*ExperimentResults, error)) error {
	rl.mu.Lock()
	exp, exists := rl.experiments[expID]
	if !exists {
		rl.mu.Unlock()
		return fmt.Errorf("experiment not found: %s", expID)
	}
	exp.Status = StatusRunning
	rl.mu.Unlock()

	rl.logger.Printf("Starting experiment: %s", exp.Name)

	results, err := runFunc(ctx, exp)

	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	exp.EndTime = &now
	exp.Duration = now.Sub(exp.StartTime)
	exp.Results = results

	if err != nil {
		exp.Status = StatusFailed
		exp.Error = err.Error()
		rl.logger.Printf("Experiment failed: %s - %v", exp.Name, err)
	} else {
		exp.Status = StatusCompleted
		rl.logger.Printf("Experiment completed: %s (%.2fs)", exp.Name, exp.Duration.Seconds())
	}

	return rl.saveExperiment(exp)
}

// ValidateExperiment runs validation checks on experiment results
func (rl *ResearchLab) ValidateExperiment(expID string, validators []Validator) error {
	rl.mu.Lock()
	exp, exists := rl.experiments[expID]
	if !exists {
		rl.mu.Unlock()
		return fmt.Errorf("experiment not found: %s", expID)
	}
	exp.Status = StatusValidating
	rl.mu.Unlock()

	if exp.Results == nil {
		return fmt.Errorf("no results to validate for experiment: %s", expID)
	}

	validations := make([]ValidationResult, 0, len(validators))
	allPassed := true

	for _, validator := range validators {
		result := validator.Validate(exp)
		validations = append(validations, result)
		if !result.Passed {
			allPassed = false
		}
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	exp.Results.Validations = validations
	if allPassed {
		exp.Status = StatusCompleted
		rl.logger.Printf("All validations passed for experiment: %s", exp.Name)
	} else {
		exp.Status = StatusFailed
		exp.Error = "validation failed"
		rl.logger.Printf("Validation failed for experiment: %s", exp.Name)
	}

	return rl.saveExperiment(exp)
}

// GetExperiment retrieves an experiment by ID
func (rl *ResearchLab) GetExperiment(expID string) (*Experiment, error) {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	exp, exists := rl.experiments[expID]
	if !exists {
		return nil, fmt.Errorf("experiment not found: %s", expID)
	}

	return exp, nil
}

// ListExperiments returns all experiments, optionally filtered by domain
func (rl *ResearchLab) ListExperiments(domain string) []*Experiment {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	experiments := make([]*Experiment, 0, len(rl.experiments))
	for _, exp := range rl.experiments {
		if domain == "" || exp.Domain == domain {
			experiments = append(experiments, exp)
		}
	}

	return experiments
}

// ExportResults exports experiment results to JSON file
func (rl *ResearchLab) ExportResults(expID string, outputPath string) error {
	exp, err := rl.GetExperiment(expID)
	if err != nil {
		return err
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(exp); err != nil {
		return fmt.Errorf("failed to encode results: %w", err)
	}

	rl.logger.Printf("Exported results for experiment %s to %s", expID, outputPath)
	return nil
}

// saveExperiment persists experiment to disk
func (rl *ResearchLab) saveExperiment(exp *Experiment) error {
	filePath := fmt.Sprintf("%s/%s.json", rl.dataDir, exp.ID)

	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to save experiment: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	return encoder.Encode(exp)
}

// Validator interface for experiment validation
type Validator interface {
	Validate(exp *Experiment) ValidationResult
}

// MetricValidator validates that a metric meets a threshold
type MetricValidator struct {
	MetricName string
	Threshold  float64
	Comparator string // "gt", "lt", "gte", "lte", "eq"
}

// Validate implements Validator interface
func (mv *MetricValidator) Validate(exp *Experiment) ValidationResult {
	result := ValidationResult{
		Name:      fmt.Sprintf("metric_%s", mv.MetricName),
		Threshold: mv.Threshold,
		Timestamp: time.Now(),
	}

	if exp.Results == nil || exp.Results.Metrics == nil {
		result.Passed = false
		result.Details = "no metrics available"
		return result
	}

	value, exists := exp.Results.Metrics[mv.MetricName]
	if !exists {
		result.Passed = false
		result.Details = fmt.Sprintf("metric %s not found", mv.MetricName)
		return result
	}

	result.Score = value
	result.Passed = mv.compareValue(value)

	if result.Passed {
		result.Details = fmt.Sprintf("metric %s (%.4f) passed threshold (%.4f)", mv.MetricName, value, mv.Threshold)
	} else {
		result.Details = fmt.Sprintf("metric %s (%.4f) failed threshold (%.4f)", mv.MetricName, value, mv.Threshold)
	}

	return result
}

func (mv *MetricValidator) compareValue(value float64) bool {
	switch mv.Comparator {
	case "gt":
		return value > mv.Threshold
	case "lt":
		return value < mv.Threshold
	case "gte":
		return value >= mv.Threshold
	case "lte":
		return value <= mv.Threshold
	case "eq":
		return value == mv.Threshold
	default:
		return value >= mv.Threshold // default to gte
	}
}

// Benchmark represents a performance benchmark
type Benchmark struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Iterations  int                    `json:"iterations"`
	Results     []BenchmarkResult      `json:"results"`
	Summary     BenchmarkSummary       `json:"summary"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// BenchmarkResult captures a single benchmark run
type BenchmarkResult struct {
	Iteration int           `json:"iteration"`
	Duration  time.Duration `json:"duration"`
	Success   bool          `json:"success"`
	Error     string        `json:"error,omitempty"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
}

// BenchmarkSummary provides statistical summary of benchmark results
type BenchmarkSummary struct {
	TotalRuns      int           `json:"total_runs"`
	SuccessfulRuns int           `json:"successful_runs"`
	FailedRuns     int           `json:"failed_runs"`
	AvgDuration    time.Duration `json:"avg_duration"`
	MinDuration    time.Duration `json:"min_duration"`
	MaxDuration    time.Duration `json:"max_duration"`
	P50Duration    time.Duration `json:"p50_duration"`
	P95Duration    time.Duration `json:"p95_duration"`
	P99Duration    time.Duration `json:"p99_duration"`
}

// BenchmarkRunner executes benchmarks
type BenchmarkRunner struct {
	logger *log.Logger
}

// NewBenchmarkRunner creates a new benchmark runner
func NewBenchmarkRunner(logger *log.Logger) *BenchmarkRunner {
	if logger == nil {
		logger = log.New(os.Stdout, "[BENCHMARK] ", log.LstdFlags)
	}
	return &BenchmarkRunner{logger: logger}
}

// Run executes a benchmark function multiple times
func (br *BenchmarkRunner) Run(ctx context.Context, bench *Benchmark, fn func(context.Context, int) error) error {
	br.logger.Printf("Starting benchmark: %s (%d iterations)", bench.Name, bench.Iterations)

	bench.Results = make([]BenchmarkResult, 0, bench.Iterations)

	for i := 0; i < bench.Iterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		result := BenchmarkResult{Iteration: i}
		start := time.Now()

		err := fn(ctx, i)
		result.Duration = time.Since(start)

		if err != nil {
			result.Success = false
			result.Error = err.Error()
		} else {
			result.Success = true
		}

		bench.Results = append(bench.Results, result)
	}

	bench.Summary = br.calculateSummary(bench.Results)
	br.logger.Printf("Benchmark completed: %s - Avg: %v, P95: %v, Success rate: %.2f%%",
		bench.Name,
		bench.Summary.AvgDuration,
		bench.Summary.P95Duration,
		float64(bench.Summary.SuccessfulRuns)/float64(bench.Summary.TotalRuns)*100)

	return nil
}

// calculateSummary computes statistical summary from benchmark results
func (br *BenchmarkRunner) calculateSummary(results []BenchmarkResult) BenchmarkSummary {
	summary := BenchmarkSummary{
		TotalRuns: len(results),
	}

	if len(results) == 0 {
		return summary
	}

	var totalDuration time.Duration
	durations := make([]time.Duration, 0, len(results))

	for _, result := range results {
		if result.Success {
			summary.SuccessfulRuns++
		} else {
			summary.FailedRuns++
		}

		totalDuration += result.Duration
		durations = append(durations, result.Duration)

		if summary.MinDuration == 0 || result.Duration < summary.MinDuration {
			summary.MinDuration = result.Duration
		}
		if result.Duration > summary.MaxDuration {
			summary.MaxDuration = result.Duration
		}
	}

	summary.AvgDuration = totalDuration / time.Duration(len(results))

	// Sort for percentile calculation
	for i := 0; i < len(durations); i++ {
		for j := i + 1; j < len(durations); j++ {
			if durations[i] > durations[j] {
				durations[i], durations[j] = durations[j], durations[i]
			}
		}
	}

	summary.P50Duration = durations[len(durations)*50/100]
	summary.P95Duration = durations[len(durations)*95/100]
	summary.P99Duration = durations[len(durations)*99/100]

	return summary
}

// generateExperimentID generates a unique experiment ID
func generateExperimentID(domain, name string) string {
	return fmt.Sprintf("%s-%s-%d", domain, name, time.Now().UnixNano())
}

// Logger utility for research modules
type ResearchLogger struct {
	*log.Logger
	logFile io.WriteCloser
}

// NewResearchLogger creates a logger that writes to both stdout and file
func NewResearchLogger(domain string, logDir string) (*ResearchLogger, error) {
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}

	logPath := fmt.Sprintf("%s/%s-%s.log", logDir, domain, time.Now().Format("20060102-150405"))
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}

	multiWriter := io.MultiWriter(os.Stdout, logFile)
	logger := log.New(multiWriter, fmt.Sprintf("[%s] ", domain), log.LstdFlags|log.Lshortfile)

	return &ResearchLogger{
		Logger:  logger,
		logFile: logFile,
	}, nil
}

// Close closes the log file
func (rl *ResearchLogger) Close() error {
	if rl.logFile != nil {
		return rl.logFile.Close()
	}
	return nil
}
