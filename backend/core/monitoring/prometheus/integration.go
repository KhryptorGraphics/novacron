package prometheus

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/common/model"
)

// PrometheusIntegration handles Prometheus metrics collection and exposition
type PrometheusIntegration struct {
	// Prometheus client for querying external Prometheus
	client    api.Client
	queryAPI  v1.API
	serverURL string

	// Local Prometheus registry for custom metrics
	registry *prometheus.Registry
	
	// HTTP server for metrics exposition
	httpServer *http.Server
	
	// Metrics collectors
	collectors map[string]prometheus.Collector
	
	// Configuration
	config *PrometheusConfig
	
	// High-frequency data collection
	highFreqCollectors map[string]*HighFrequencyCollector
	dataStreams       map[string]chan *MetricSample
	
	// Concurrency control
	mutex           sync.RWMutex
	collectorsMutex sync.RWMutex
	streamsMutex    sync.RWMutex
	
	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// PrometheusConfig represents the configuration for Prometheus integration
type PrometheusConfig struct {
	// Server configuration
	ServerURL      string `json:"server_url"`
	ListenAddress  string `json:"listen_address"`
	MetricsPath    string `json:"metrics_path"`
	
	// Query configuration
	QueryTimeout   time.Duration `json:"query_timeout"`
	QueryStep      time.Duration `json:"query_step"`
	MaxSamples     int           `json:"max_samples"`
	
	// High-frequency collection
	HighFrequencyEnabled   bool          `json:"high_frequency_enabled"`
	HighFrequencyInterval  time.Duration `json:"high_frequency_interval"`
	HighFrequencyRetention time.Duration `json:"high_frequency_retention"`
	
	// Federation
	FederationEnabled bool     `json:"federation_enabled"`
	FederationTargets []string `json:"federation_targets"`
	
	// Storage
	LocalStorageEnabled   bool          `json:"local_storage_enabled"`
	LocalStoragePath      string        `json:"local_storage_path"`
	LocalRetentionPeriod  time.Duration `json:"local_retention_period"`
	
	// Security
	EnableBasicAuth bool   `json:"enable_basic_auth"`
	Username        string `json:"username"`
	Password        string `json:"password"`
	TLSCertFile     string `json:"tls_cert_file"`
	TLSKeyFile      string `json:"tls_key_file"`
}

// DefaultPrometheusConfig returns a default configuration
func DefaultPrometheusConfig() *PrometheusConfig {
	return &PrometheusConfig{
		ServerURL:              "http://localhost:9090",
		ListenAddress:          ":9091",
		MetricsPath:            "/metrics",
		QueryTimeout:           30 * time.Second,
		QueryStep:              15 * time.Second,
		MaxSamples:             50000,
		HighFrequencyEnabled:   true,
		HighFrequencyInterval:  1 * time.Second,
		HighFrequencyRetention: 1 * time.Hour,
		LocalStorageEnabled:    false,
		LocalRetentionPeriod:   24 * time.Hour,
	}
}

// MetricSample represents a single metric sample
type MetricSample struct {
	Metric    map[string]string `json:"metric"`
	Value     float64           `json:"value"`
	Timestamp time.Time         `json:"timestamp"`
}

// QueryResult represents the result of a Prometheus query
type QueryResult struct {
	Type   model.ValueType `json:"type"`
	Result model.Value     `json:"result"`
	Error  string          `json:"error,omitempty"`
}

// HighFrequencyCollector collects metrics at high frequency
type HighFrequencyCollector struct {
	query    string
	interval time.Duration
	stream   chan *MetricSample
	ticker   *time.Ticker
	done     chan struct{}
	api      v1.API
}

// NewPrometheusIntegration creates a new Prometheus integration
func NewPrometheusIntegration(config *PrometheusConfig) (*PrometheusIntegration, error) {
	if config == nil {
		config = DefaultPrometheusConfig()
	}

	// Create Prometheus client
	client, err := api.NewClient(api.Config{
		Address: config.ServerURL,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Prometheus client: %w", err)
	}

	// Create local registry
	registry := prometheus.NewRegistry()

	ctx, cancel := context.WithCancel(context.Background())

	integration := &PrometheusIntegration{
		client:             client,
		queryAPI:           v1.NewAPI(client),
		serverURL:          config.ServerURL,
		registry:           registry,
		config:             config,
		collectors:         make(map[string]prometheus.Collector),
		highFreqCollectors: make(map[string]*HighFrequencyCollector),
		dataStreams:        make(map[string]chan *MetricSample),
		ctx:                ctx,
		cancel:             cancel,
	}

	// Register default collectors
	integration.registerDefaultCollectors()

	return integration, nil
}

// Start starts the Prometheus integration
func (p *PrometheusIntegration) Start() error {
	log.Println("Starting Prometheus integration...")

	// Start HTTP server for metrics exposition
	if err := p.startHTTPServer(); err != nil {
		return fmt.Errorf("failed to start HTTP server: %w", err)
	}

	// Start high-frequency collectors if enabled
	if p.config.HighFrequencyEnabled {
		p.startHighFrequencyCollection()
	}

	log.Printf("Prometheus integration started on %s", p.config.ListenAddress)
	return nil
}

// Stop stops the Prometheus integration
func (p *PrometheusIntegration) Stop() error {
	log.Println("Stopping Prometheus integration...")

	p.cancel()
	p.wg.Wait()

	// Stop HTTP server
	if p.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := p.httpServer.Shutdown(ctx); err != nil {
			log.Printf("Error shutting down HTTP server: %v", err)
		}
	}

	// Stop high-frequency collectors
	p.collectorsMutex.Lock()
	for _, collector := range p.highFreqCollectors {
		collector.stop()
	}
	p.collectorsMutex.Unlock()

	// Close data streams
	p.streamsMutex.Lock()
	for _, stream := range p.dataStreams {
		close(stream)
	}
	p.streamsMutex.Unlock()

	log.Println("Prometheus integration stopped")
	return nil
}

// Query executes a Prometheus query
func (p *PrometheusIntegration) Query(ctx context.Context, query string, timestamp time.Time) (*QueryResult, error) {
	result, warnings, err := p.queryAPI.Query(ctx, query, timestamp)
	if err != nil {
		return &QueryResult{Error: err.Error()}, err
	}

	if len(warnings) > 0 {
		log.Printf("Query warnings: %v", warnings)
	}

	return &QueryResult{
		Type:   result.Type(),
		Result: result,
	}, nil
}

// QueryRange executes a Prometheus range query
func (p *PrometheusIntegration) QueryRange(ctx context.Context, query string, startTime, endTime time.Time, step time.Duration) (*QueryResult, error) {
	queryRange := v1.Range{
		Start: startTime,
		End:   endTime,
		Step:  step,
	}

	result, warnings, err := p.queryAPI.QueryRange(ctx, query, queryRange)
	if err != nil {
		return &QueryResult{Error: err.Error()}, err
	}

	if len(warnings) > 0 {
		log.Printf("Query range warnings: %v", warnings)
	}

	return &QueryResult{
		Type:   result.Type(),
		Result: result,
	}, nil
}

// RegisterCollector registers a custom Prometheus collector
func (p *PrometheusIntegration) RegisterCollector(name string, collector prometheus.Collector) error {
	p.collectorsMutex.Lock()
	defer p.collectorsMutex.Unlock()

	if _, exists := p.collectors[name]; exists {
		return fmt.Errorf("collector %s already registered", name)
	}

	if err := p.registry.Register(collector); err != nil {
		return fmt.Errorf("failed to register collector %s: %w", name, err)
	}

	p.collectors[name] = collector
	log.Printf("Registered Prometheus collector: %s", name)
	return nil
}

// UnregisterCollector unregisters a custom Prometheus collector
func (p *PrometheusIntegration) UnregisterCollector(name string) error {
	p.collectorsMutex.Lock()
	defer p.collectorsMutex.Unlock()

	collector, exists := p.collectors[name]
	if !exists {
		return fmt.Errorf("collector %s not found", name)
	}

	if !p.registry.Unregister(collector) {
		return fmt.Errorf("failed to unregister collector %s", name)
	}

	delete(p.collectors, name)
	log.Printf("Unregistered Prometheus collector: %s", name)
	return nil
}

// StartHighFrequencyCollection starts high-frequency collection for a query
func (p *PrometheusIntegration) StartHighFrequencyCollection(name, query string, interval time.Duration) (<-chan *MetricSample, error) {
	p.collectorsMutex.Lock()
	defer p.collectorsMutex.Unlock()

	if _, exists := p.highFreqCollectors[name]; exists {
		return nil, fmt.Errorf("high-frequency collector %s already exists", name)
	}

	stream := make(chan *MetricSample, 1000)
	collector := &HighFrequencyCollector{
		query:    query,
		interval: interval,
		stream:   stream,
		ticker:   time.NewTicker(interval),
		done:     make(chan struct{}),
		api:      p.queryAPI,
	}

	p.highFreqCollectors[name] = collector
	p.dataStreams[name] = stream

	go collector.run(p.ctx)

	log.Printf("Started high-frequency collector: %s (interval: %s)", name, interval)
	return stream, nil
}

// StopHighFrequencyCollection stops high-frequency collection for a query
func (p *PrometheusIntegration) StopHighFrequencyCollection(name string) error {
	p.collectorsMutex.Lock()
	defer p.collectorsMutex.Unlock()

	collector, exists := p.highFreqCollectors[name]
	if !exists {
		return fmt.Errorf("high-frequency collector %s not found", name)
	}

	collector.stop()
	delete(p.highFreqCollectors, name)

	p.streamsMutex.Lock()
	if stream, exists := p.dataStreams[name]; exists {
		close(stream)
		delete(p.dataStreams, name)
	}
	p.streamsMutex.Unlock()

	log.Printf("Stopped high-frequency collector: %s", name)
	return nil
}

// GetMetricLabels retrieves the labels for a metric
func (p *PrometheusIntegration) GetMetricLabels(ctx context.Context, metricName string) ([]string, error) {
	labelNames, warnings, err := p.queryAPI.LabelNames(ctx, nil, time.Time{}, time.Time{})
	if err != nil {
		return nil, fmt.Errorf("failed to get label names: %w", err)
	}

	if len(warnings) > 0 {
		log.Printf("Label names warnings: %v", warnings)
	}

	return labelNames, nil
}

// GetLabelValues retrieves the values for a specific label
func (p *PrometheusIntegration) GetLabelValues(ctx context.Context, labelName string) ([]string, error) {
	labelValues, warnings, err := p.queryAPI.LabelValues(ctx, labelName, nil, time.Time{}, time.Time{})
	if err != nil {
		return nil, fmt.Errorf("failed to get label values: %w", err)
	}

	if len(warnings) > 0 {
		log.Printf("Label values warnings: %v", warnings)
	}

	return model.LabelValues(labelValues).Strings(), nil
}

// GetTargets retrieves Prometheus targets
func (p *PrometheusIntegration) GetTargets(ctx context.Context) (v1.TargetsResult, error) {
	targets, err := p.queryAPI.Targets(ctx)
	if err != nil {
		return v1.TargetsResult{}, fmt.Errorf("failed to get targets: %w", err)
	}

	return targets, nil
}

// GetConfig retrieves Prometheus configuration
func (p *PrometheusIntegration) GetConfig(ctx context.Context) (v1.ConfigResult, error) {
	config, err := p.queryAPI.Config(ctx)
	if err != nil {
		return v1.ConfigResult{}, fmt.Errorf("failed to get config: %w", err)
	}

	return config, nil
}

// BuildQuery builds a Prometheus query with common patterns
func (p *PrometheusIntegration) BuildQuery(metric string, labels map[string]string, function string, duration time.Duration) string {
	query := metric

	// Add label selectors
	if len(labels) > 0 {
		query += "{"
		first := true
		for k, v := range labels {
			if !first {
				query += ","
			}
			query += fmt.Sprintf(`%s="%s"`, k, v)
			first = false
		}
		query += "}"
	}

	// Apply function if specified
	if function != "" && duration > 0 {
		durationStr := duration.String()
		switch function {
		case "rate":
			query = fmt.Sprintf("rate(%s[%s])", query, durationStr)
		case "increase":
			query = fmt.Sprintf("increase(%s[%s])", query, durationStr)
		case "avg_over_time":
			query = fmt.Sprintf("avg_over_time(%s[%s])", query, durationStr)
		case "max_over_time":
			query = fmt.Sprintf("max_over_time(%s[%s])", query, durationStr)
		case "min_over_time":
			query = fmt.Sprintf("min_over_time(%s[%s])", query, durationStr)
		case "sum_over_time":
			query = fmt.Sprintf("sum_over_time(%s[%s])", query, durationStr)
		case "delta":
			query = fmt.Sprintf("delta(%s[%s])", query, durationStr)
		case "deriv":
			query = fmt.Sprintf("deriv(%s[%s])", query, durationStr)
		}
	}

	return query
}

// Helper methods

func (p *PrometheusIntegration) startHTTPServer() error {
	mux := http.NewServeMux()
	
	// Add metrics handler
	mux.Handle(p.config.MetricsPath, promhttp.HandlerFor(p.registry, promhttp.HandlerOpts{}))
	
	// Add health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Add federation endpoint if enabled
	if p.config.FederationEnabled {
		mux.HandleFunc("/federate", p.handleFederation)
	}

	p.httpServer = &http.Server{
		Addr:    p.config.ListenAddress,
		Handler: mux,
	}

	go func() {
		if p.config.TLSCertFile != "" && p.config.TLSKeyFile != "" {
			if err := p.httpServer.ListenAndServeTLS(p.config.TLSCertFile, p.config.TLSKeyFile); err != nil && err != http.ErrServerClosed {
				log.Printf("HTTPS server error: %v", err)
			}
		} else {
			if err := p.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				log.Printf("HTTP server error: %v", err)
			}
		}
	}()

	return nil
}

func (p *PrometheusIntegration) startHighFrequencyCollection() {
	// Start collectors for critical metrics
	criticalMetrics := map[string]string{
		"cpu_usage":      "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[1m])) * 100)",
		"memory_usage":   "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
		"disk_usage":     "100 - ((node_filesystem_avail_bytes{mountpoint=\"/\"} / node_filesystem_size_bytes{mountpoint=\"/\"}) * 100)",
		"network_rx":     "rate(node_network_receive_bytes_total[1m])",
		"network_tx":     "rate(node_network_transmit_bytes_total[1m])",
		"vm_count":       "count(libvirt_domain_info_state)",
		"vm_cpu_usage":   "avg(libvirt_domain_info_cpu_time_seconds_total)",
		"vm_memory_usage": "avg(libvirt_domain_info_memory_usage_bytes)",
	}

	for name, query := range criticalMetrics {
		if _, err := p.StartHighFrequencyCollection(name, query, p.config.HighFrequencyInterval); err != nil {
			log.Printf("Failed to start high-frequency collection for %s: %v", name, err)
		}
	}
}

func (p *PrometheusIntegration) registerDefaultCollectors() {
	// Register Go runtime metrics
	p.registry.MustRegister(prometheus.NewGoCollector())
	p.registry.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
}

func (p *PrometheusIntegration) handleFederation(w http.ResponseWriter, r *http.Request) {
	// Simple federation endpoint implementation
	// In a production environment, this would implement full Prometheus federation
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	
	// Get metrics from registry
	metricFamilies, err := p.registry.Gather()
	if err != nil {
		http.Error(w, fmt.Sprintf("Error gathering metrics: %v", err), http.StatusInternalServerError)
		return
	}

	for _, mf := range metricFamilies {
		if mf.GetName() != "" {
			fmt.Fprintf(w, "# HELP %s %s\n", mf.GetName(), mf.GetHelp())
			fmt.Fprintf(w, "# TYPE %s %s\n", mf.GetName(), mf.GetType().String())
			
			for _, m := range mf.GetMetric() {
				// Format metric line
				labels := ""
				for _, l := range m.GetLabel() {
					if labels != "" {
						labels += ","
					}
					labels += fmt.Sprintf(`%s="%s"`, l.GetName(), l.GetValue())
				}
				
				if labels != "" {
					labels = "{" + labels + "}"
				}
				
				var value float64
				switch mf.GetType() {
				case prometheus.CounterValue:
					value = m.GetCounter().GetValue()
				case prometheus.GaugeValue:
					value = m.GetGauge().GetValue()
				}
				
				timestamp := ""
				if m.GetTimestampMs() != 0 {
					timestamp = fmt.Sprintf(" %d", m.GetTimestampMs())
				}
				
				fmt.Fprintf(w, "%s%s %g%s\n", mf.GetName(), labels, value, timestamp)
			}
		}
	}
}

// HighFrequencyCollector methods

func (c *HighFrequencyCollector) run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-c.done:
			return
		case <-c.ticker.C:
			c.collect()
		}
	}
}

func (c *HighFrequencyCollector) collect() {
	result, _, err := c.api.Query(context.Background(), c.query, time.Now())
	if err != nil {
		log.Printf("High-frequency query error: %v", err)
		return
	}

	switch result.Type() {
	case model.ValVector:
		vector := result.(model.Vector)
		for _, sample := range vector {
			metric := make(map[string]string)
			for k, v := range sample.Metric {
				metric[string(k)] = string(v)
			}
			
			metricSample := &MetricSample{
				Metric:    metric,
				Value:     float64(sample.Value),
				Timestamp: sample.Timestamp.Time(),
			}
			
			select {
			case c.stream <- metricSample:
			default:
				// Channel is full, drop the sample
				log.Printf("Dropped metric sample due to full channel")
			}
		}
	case model.ValScalar:
		scalar := result.(*model.Scalar)
		metricSample := &MetricSample{
			Metric:    map[string]string{"__name__": "scalar"},
			Value:     float64(scalar.Value),
			Timestamp: scalar.Timestamp.Time(),
		}
		
		select {
		case c.stream <- metricSample:
		default:
			log.Printf("Dropped scalar sample due to full channel")
		}
	}
}

func (c *HighFrequencyCollector) stop() {
	c.ticker.Stop()
	close(c.done)
}