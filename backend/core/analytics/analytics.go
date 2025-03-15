package analytics

import (
	"fmt"
	"sync"
	"time"

	"c:/10/novacron/backend/core/monitoring"
)

// AnalyticsEngine is the main analytics engine
type AnalyticsEngine struct {
	metricRegistry     *monitoring.MetricRegistry
	historyManager     *monitoring.MetricHistoryManager
	reporters          map[string]Reporter
	analyzers          map[string]Analyzer
	visualizers        map[string]Visualizer
	processors         map[string]Processor
	pipelineRegistry   *PipelineRegistry
	mutex              sync.RWMutex
	processingInterval time.Duration
	stopChan           chan struct{}
	wg                 sync.WaitGroup
}

// NewAnalyticsEngine creates a new analytics engine
func NewAnalyticsEngine(registry *monitoring.MetricRegistry, historyManager *monitoring.MetricHistoryManager, processingInterval time.Duration) *AnalyticsEngine {
	return &AnalyticsEngine{
		metricRegistry:     registry,
		historyManager:     historyManager,
		reporters:          make(map[string]Reporter),
		analyzers:          make(map[string]Analyzer),
		visualizers:        make(map[string]Visualizer),
		processors:         make(map[string]Processor),
		pipelineRegistry:   NewPipelineRegistry(),
		processingInterval: processingInterval,
		stopChan:           make(chan struct{}),
	}
}

// Start starts the analytics engine
func (e *AnalyticsEngine) Start() error {
	e.wg.Add(1)
	go e.run()
	return nil
}

// Stop stops the analytics engine
func (e *AnalyticsEngine) Stop() error {
	close(e.stopChan)
	e.wg.Wait()
	return nil
}

// run runs the analytics engine
func (e *AnalyticsEngine) run() {
	defer e.wg.Done()

	ticker := time.NewTicker(e.processingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-e.stopChan:
			return
		case <-ticker.C:
			e.processPipelines()
		}
	}
}

// processPipelines processes all analytics pipelines
func (e *AnalyticsEngine) processPipelines() {
	pipelines := e.pipelineRegistry.ListPipelines()
	for _, pipeline := range pipelines {
		if pipeline.IsEnabled() {
			go e.processPipeline(pipeline)
		}
	}
}

// processPipeline processes a single analytics pipeline
func (e *AnalyticsEngine) processPipeline(pipeline *Pipeline) {
	// Record start time for performance tracking
	startTime := time.Now()

	// Skip if pipeline is already running
	if !pipeline.StartProcessing() {
		return
	}

	// Ensure we mark the pipeline as not running when done
	defer pipeline.FinishProcessing()

	// Set up the context
	ctx := &PipelineContext{
		StartTime:      startTime,
		MetricRegistry: e.metricRegistry,
		HistoryManager: e.historyManager,
		Data:           make(map[string]interface{}),
	}

	// Run the pipeline stages in sequence
	var err error

	// 1. Run data collectors
	for _, collectorID := range pipeline.CollectorIDs {
		if processor, exists := e.processors[collectorID]; exists {
			err = processor.Process(ctx)
			if err != nil {
				pipeline.RecordError(err)
				return
			}
		}
	}

	// 2. Run analyzers
	for _, analyzerID := range pipeline.AnalyzerIDs {
		if analyzer, exists := e.analyzers[analyzerID]; exists {
			err = analyzer.Analyze(ctx)
			if err != nil {
				pipeline.RecordError(err)
				return
			}
		}
	}

	// 3. Run visualizers
	for _, visualizerID := range pipeline.VisualizerIDs {
		if visualizer, exists := e.visualizers[visualizerID]; exists {
			err = visualizer.Visualize(ctx)
			if err != nil {
				pipeline.RecordError(err)
				return
			}
		}
	}

	// 4. Run reporters
	for _, reporterID := range pipeline.ReporterIDs {
		if reporter, exists := e.reporters[reporterID]; exists {
			err = reporter.Report(ctx)
			if err != nil {
				pipeline.RecordError(err)
				return
			}
		}
	}

	// Record execution time
	pipeline.LastExecutionTime = time.Since(startTime)
	pipeline.LastExecutionSuccess = true
}

// RegisterReporter registers a reporter
func (e *AnalyticsEngine) RegisterReporter(id string, reporter Reporter) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.reporters[id]; exists {
		return fmt.Errorf("reporter already exists: %s", id)
	}

	e.reporters[id] = reporter
	return nil
}

// RegisterAnalyzer registers an analyzer
func (e *AnalyticsEngine) RegisterAnalyzer(id string, analyzer Analyzer) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.analyzers[id]; exists {
		return fmt.Errorf("analyzer already exists: %s", id)
	}

	e.analyzers[id] = analyzer
	return nil
}

// RegisterVisualizer registers a visualizer
func (e *AnalyticsEngine) RegisterVisualizer(id string, visualizer Visualizer) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.visualizers[id]; exists {
		return fmt.Errorf("visualizer already exists: %s", id)
	}

	e.visualizers[id] = visualizer
	return nil
}

// RegisterProcessor registers a data processor
func (e *AnalyticsEngine) RegisterProcessor(id string, processor Processor) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.processors[id]; exists {
		return fmt.Errorf("processor already exists: %s", id)
	}

	e.processors[id] = processor
	return nil
}

// CreatePipeline creates a new analytics pipeline
func (e *AnalyticsEngine) CreatePipeline(id, name, description string) (*Pipeline, error) {
	pipeline := NewPipeline(id, name, description)
	err := e.pipelineRegistry.RegisterPipeline(pipeline)
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// GetPipeline gets a pipeline by ID
func (e *AnalyticsEngine) GetPipeline(id string) (*Pipeline, error) {
	return e.pipelineRegistry.GetPipeline(id)
}

// DeletePipeline deletes a pipeline
func (e *AnalyticsEngine) DeletePipeline(id string) error {
	return e.pipelineRegistry.RemovePipeline(id)
}

// ListPipelines lists all pipelines
func (e *AnalyticsEngine) ListPipelines() []*Pipeline {
	return e.pipelineRegistry.ListPipelines()
}

// ListReporters lists all reporters
func (e *AnalyticsEngine) ListReporters() []string {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	reporters := make([]string, 0, len(e.reporters))
	for id := range e.reporters {
		reporters = append(reporters, id)
	}
	return reporters
}

// ListAnalyzers lists all analyzers
func (e *AnalyticsEngine) ListAnalyzers() []string {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	analyzers := make([]string, 0, len(e.analyzers))
	for id := range e.analyzers {
		analyzers = append(analyzers, id)
	}
	return analyzers
}

// ListVisualizers lists all visualizers
func (e *AnalyticsEngine) ListVisualizers() []string {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	visualizers := make([]string, 0, len(e.visualizers))
	for id := range e.visualizers {
		visualizers = append(visualizers, id)
	}
	return visualizers
}

// ListProcessors lists all processors
func (e *AnalyticsEngine) ListProcessors() []string {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	processors := make([]string, 0, len(e.processors))
	for id := range e.processors {
		processors = append(processors, id)
	}
	return processors
}

// Processor processes data from metrics
type Processor interface {
	// Process processes data and updates the context
	Process(ctx *PipelineContext) error

	// GetMetadata returns metadata about the processor
	GetMetadata() ProcessorMetadata
}

// Analyzer analyzes processed data
type Analyzer interface {
	// Analyze analyzes data and updates the context
	Analyze(ctx *PipelineContext) error

	// GetMetadata returns metadata about the analyzer
	GetMetadata() AnalyzerMetadata
}

// Visualizer creates visualizations from analyzed data
type Visualizer interface {
	// Visualize creates visualizations and updates the context
	Visualize(ctx *PipelineContext) error

	// GetMetadata returns metadata about the visualizer
	GetMetadata() VisualizerMetadata
}

// Reporter reports results from analyzed and visualized data
type Reporter interface {
	// Report generates reports and updates the context
	Report(ctx *PipelineContext) error

	// GetMetadata returns metadata about the reporter
	GetMetadata() ReporterMetadata
}

// ProcessorMetadata contains metadata about a processor
type ProcessorMetadata struct {
	// ID is the unique identifier for the processor
	ID string `json:"id"`

	// Name is the human-readable name of the processor
	Name string `json:"name"`

	// Description is a description of the processor
	Description string `json:"description"`

	// RequiredMetrics are the metrics required by the processor
	RequiredMetrics []string `json:"requiredMetrics,omitempty"`

	// ProducedData are the data items produced by the processor
	ProducedData []string `json:"producedData,omitempty"`
}

// AnalyzerMetadata contains metadata about an analyzer
type AnalyzerMetadata struct {
	// ID is the unique identifier for the analyzer
	ID string `json:"id"`

	// Name is the human-readable name of the analyzer
	Name string `json:"name"`

	// Description is a description of the analyzer
	Description string `json:"description"`

	// RequiredData are the data items required by the analyzer
	RequiredData []string `json:"requiredData,omitempty"`

	// ProducedData are the data items produced by the analyzer
	ProducedData []string `json:"producedData,omitempty"`
}

// VisualizerMetadata contains metadata about a visualizer
type VisualizerMetadata struct {
	// ID is the unique identifier for the visualizer
	ID string `json:"id"`

	// Name is the human-readable name of the visualizer
	Name string `json:"name"`

	// Description is a description of the visualizer
	Description string `json:"description"`

	// RequiredData are the data items required by the visualizer
	RequiredData []string `json:"requiredData,omitempty"`

	// ProducedData are the data items produced by the visualizer
	ProducedData []string `json:"producedData,omitempty"`

	// OutputFormat is the format of the visualizer output
	OutputFormat string `json:"outputFormat,omitempty"`
}

// ReporterMetadata contains metadata about a reporter
type ReporterMetadata struct {
	// ID is the unique identifier for the reporter
	ID string `json:"id"`

	// Name is the human-readable name of the reporter
	Name string `json:"name"`

	// Description is a description of the reporter
	Description string `json:"description"`

	// RequiredData are the data items required by the reporter
	RequiredData []string `json:"requiredData,omitempty"`

	// OutputFormats are the formats supported by the reporter
	OutputFormats []string `json:"outputFormats,omitempty"`
}

// PipelineContext is the context for a pipeline execution
type PipelineContext struct {
	// StartTime is when the pipeline execution started
	StartTime time.Time `json:"startTime"`

	// MetricRegistry is the metric registry for accessing metrics
	MetricRegistry *monitoring.MetricRegistry `json:"-"`

	// HistoryManager is the history manager for accessing historical metrics
	HistoryManager *monitoring.MetricHistoryManager `json:"-"`

	// Data is the data shared between pipeline stages
	Data map[string]interface{} `json:"-"`
}

// Pipeline is an analytics pipeline
type Pipeline struct {
	// ID is the unique identifier for the pipeline
	ID string `json:"id"`

	// Name is the human-readable name of the pipeline
	Name string `json:"name"`

	// Description is a description of the pipeline
	Description string `json:"description"`

	// Enabled indicates if the pipeline is enabled
	Enabled bool `json:"enabled"`

	// CollectorIDs are the IDs of the collectors to run
	CollectorIDs []string `json:"collectorIds"`

	// AnalyzerIDs are the IDs of the analyzers to run
	AnalyzerIDs []string `json:"analyzerIds"`

	// VisualizerIDs are the IDs of the visualizers to run
	VisualizerIDs []string `json:"visualizerIds"`

	// ReporterIDs are the IDs of the reporters to run
	ReporterIDs []string `json:"reporterIds"`

	// Schedule is when the pipeline should run
	Schedule string `json:"schedule,omitempty"`

	// LastRun is when the pipeline was last run
	LastRun time.Time `json:"lastRun,omitempty"`

	// LastExecutionTime is how long the last execution took
	LastExecutionTime time.Duration `json:"lastExecutionTime,omitempty"`

	// LastExecutionSuccess indicates if the last execution was successful
	LastExecutionSuccess bool `json:"lastExecutionSuccess"`

	// LastError is the last error that occurred
	LastError string `json:"lastError,omitempty"`

	// TenantID is the ID of the tenant this pipeline belongs to
	TenantID string `json:"tenantId,omitempty"`

	// Tags are additional tags for the pipeline
	Tags []string `json:"tags,omitempty"`

	// Parameters are parameters for the pipeline
	Parameters map[string]interface{} `json:"parameters,omitempty"`

	// running indicates if the pipeline is currently running
	running bool

	// mutex protects the pipeline
	mutex sync.RWMutex
}

// NewPipeline creates a new pipeline
func NewPipeline(id, name, description string) *Pipeline {
	return &Pipeline{
		ID:            id,
		Name:          name,
		Description:   description,
		Enabled:       true,
		CollectorIDs:  make([]string, 0),
		AnalyzerIDs:   make([]string, 0),
		VisualizerIDs: make([]string, 0),
		ReporterIDs:   make([]string, 0),
		Parameters:    make(map[string]interface{}),
		Tags:          make([]string, 0),
	}
}

// IsEnabled checks if the pipeline is enabled
func (p *Pipeline) IsEnabled() bool {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.Enabled
}

// SetEnabled sets whether the pipeline is enabled
func (p *Pipeline) SetEnabled(enabled bool) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.Enabled = enabled
}

// IsRunning checks if the pipeline is running
func (p *Pipeline) IsRunning() bool {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.running
}

// StartProcessing marks the pipeline as running
func (p *Pipeline) StartProcessing() bool {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.running {
		return false
	}
	p.running = true
	p.LastRun = time.Now()
	return true
}

// FinishProcessing marks the pipeline as not running
func (p *Pipeline) FinishProcessing() {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.running = false
}

// RecordError records an error
func (p *Pipeline) RecordError(err error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.LastError = err.Error()
	p.LastExecutionSuccess = false
}

// AddCollector adds a collector to the pipeline
func (p *Pipeline) AddCollector(id string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.CollectorIDs = append(p.CollectorIDs, id)
}

// AddAnalyzer adds an analyzer to the pipeline
func (p *Pipeline) AddAnalyzer(id string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.AnalyzerIDs = append(p.AnalyzerIDs, id)
}

// AddVisualizer adds a visualizer to the pipeline
func (p *Pipeline) AddVisualizer(id string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.VisualizerIDs = append(p.VisualizerIDs, id)
}

// AddReporter adds a reporter to the pipeline
func (p *Pipeline) AddReporter(id string) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.ReporterIDs = append(p.ReporterIDs, id)
}

// PipelineRegistry manages analytics pipelines
type PipelineRegistry struct {
	pipelines map[string]*Pipeline
	mutex     sync.RWMutex
}

// NewPipelineRegistry creates a new pipeline registry
func NewPipelineRegistry() *PipelineRegistry {
	return &PipelineRegistry{
		pipelines: make(map[string]*Pipeline),
	}
}

// RegisterPipeline registers a pipeline
func (r *PipelineRegistry) RegisterPipeline(pipeline *Pipeline) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if _, exists := r.pipelines[pipeline.ID]; exists {
		return fmt.Errorf("pipeline already exists: %s", pipeline.ID)
	}

	r.pipelines[pipeline.ID] = pipeline
	return nil
}

// GetPipeline gets a pipeline by ID
func (r *PipelineRegistry) GetPipeline(id string) (*Pipeline, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	pipeline, exists := r.pipelines[id]
	if !exists {
		return nil, fmt.Errorf("pipeline not found: %s", id)
	}

	return pipeline, nil
}

// RemovePipeline removes a pipeline
func (r *PipelineRegistry) RemovePipeline(id string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if _, exists := r.pipelines[id]; !exists {
		return fmt.Errorf("pipeline not found: %s", id)
	}

	delete(r.pipelines, id)
	return nil
}

// ListPipelines lists all pipelines
func (r *PipelineRegistry) ListPipelines() []*Pipeline {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	pipelines := make([]*Pipeline, 0, len(r.pipelines))
	for _, pipeline := range r.pipelines {
		pipelines = append(pipelines, pipeline)
	}

	return pipelines
}
