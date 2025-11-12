// Package enterprise provides Datadog integration for enhanced monitoring
package enterprise

import (
	"context"
	"fmt"
	"time"

	"github.com/DataDog/datadog-api-client-go/v2/api/datadog"
	"github.com/DataDog/datadog-api-client-go/v2/api/datadogV1"
	"github.com/DataDog/datadog-api-client-go/v2/api/datadogV2"
)

// DatadogClient handles Datadog monitoring integration
type DatadogClient struct {
	apiClient *datadog.APIClient
	ctx       context.Context
}

// DatadogConfig configures Datadog integration
type DatadogConfig struct {
	APIKey string
	APPKey string
	Site   string // datadoghq.com, datadoghq.eu, etc.
}

// Metric represents a Datadog metric
type Metric struct {
	Name      string
	Type      string // gauge, count, rate, histogram
	Value     float64
	Timestamp int64
	Tags      []string
	Host      string
}

// Event represents a Datadog event
type Event struct {
	Title       string
	Text        string
	Timestamp   int64
	Priority    string // normal, low
	AlertType   string // info, warning, error, success
	Tags        []string
	Aggregation string
}

// Monitor represents a Datadog monitor
type Monitor struct {
	Name    string
	Type    string // metric alert, service check, etc.
	Query   string
	Message string
	Tags    []string
	Options MonitorOptions
}

// MonitorOptions configures monitor behavior
type MonitorOptions struct {
	Thresholds         Thresholds
	NotifyNoData       bool
	NoDataTimeframe    int
	RenotifyInterval   int
	EvaluationDelay    int
	RequireFullWindow  bool
	IncludeTags        bool
}

// Thresholds defines alert thresholds
type Thresholds struct {
	Critical        *float64
	CriticalRecovery *float64
	Warning         *float64
	WarningRecovery *float64
	OK              *float64
}

// Dashboard represents a Datadog dashboard
type Dashboard struct {
	Title       string
	Description string
	Widgets     []Widget
	LayoutType  string // ordered, free
}

// Widget represents a dashboard widget
type Widget struct {
	Definition WidgetDefinition
	Layout     *WidgetLayout
}

// WidgetDefinition defines widget content
type WidgetDefinition struct {
	Type     string // timeseries, query_value, toplist, etc.
	Requests []WidgetRequest
	Title    string
}

// WidgetRequest defines data request
type WidgetRequest struct {
	Query      string
	Aggregator string
	Style      *WidgetStyle
}

// WidgetStyle defines widget styling
type WidgetStyle struct {
	Palette    string
	LineType   string
	LineWidth  string
}

// WidgetLayout defines widget position
type WidgetLayout struct {
	X      int64
	Y      int64
	Width  int64
	Height int64
}

// NewDatadogClient creates a new Datadog client
func NewDatadogClient(cfg DatadogConfig) (*DatadogClient, error) {
	ctx := context.WithValue(
		context.Background(),
		datadog.ContextAPIKeys,
		map[string]datadog.APIKey{
			"apiKeyAuth": {Key: cfg.APIKey},
			"appKeyAuth": {Key: cfg.APPKey},
		},
	)

	configuration := datadog.NewConfiguration()
	if cfg.Site != "" {
		configuration.SetHost(fmt.Sprintf("https://api.%s", cfg.Site))
	}

	apiClient := datadog.NewAPIClient(configuration)

	return &DatadogClient{
		apiClient: apiClient,
		ctx:       ctx,
	}, nil
}

// SendMetric sends a single metric to Datadog
func (d *DatadogClient) SendMetric(ctx context.Context, metric Metric) error {
	api := datadogV2.NewMetricsApi(d.apiClient)

	if metric.Timestamp == 0 {
		metric.Timestamp = time.Now().Unix()
	}

	series := datadogV2.MetricSeries{
		Metric: metric.Name,
		Type:   datadogV2.MetricIntakeType(metric.Type).Ptr(),
		Points: []datadogV2.MetricPoint{
			{
				Timestamp: datadog.PtrInt64(metric.Timestamp),
				Value:     datadog.PtrFloat64(metric.Value),
			},
		},
		Tags: &metric.Tags,
	}

	if metric.Host != "" {
		series.Resources = &[]datadogV2.MetricResource{
			{
				Name: datadog.PtrString(metric.Host),
				Type: datadog.PtrString("host"),
			},
		}
	}

	body := datadogV2.MetricPayload{
		Series: []datadogV2.MetricSeries{series},
	}

	_, _, err := api.SubmitMetrics(d.ctx, body, *datadogV2.NewSubmitMetricsOptionalParameters())
	if err != nil {
		return fmt.Errorf("failed to send metric: %w", err)
	}

	return nil
}

// SendMetrics sends multiple metrics in a batch
func (d *DatadogClient) SendMetrics(ctx context.Context, metrics []Metric) error {
	api := datadogV2.NewMetricsApi(d.apiClient)

	series := make([]datadogV2.MetricSeries, 0, len(metrics))

	for _, metric := range metrics {
		if metric.Timestamp == 0 {
			metric.Timestamp = time.Now().Unix()
		}

		s := datadogV2.MetricSeries{
			Metric: metric.Name,
			Type:   datadogV2.MetricIntakeType(metric.Type).Ptr(),
			Points: []datadogV2.MetricPoint{
				{
					Timestamp: datadog.PtrInt64(metric.Timestamp),
					Value:     datadog.PtrFloat64(metric.Value),
				},
			},
			Tags: &metric.Tags,
		}

		if metric.Host != "" {
			s.Resources = &[]datadogV2.MetricResource{
				{
					Name: datadog.PtrString(metric.Host),
					Type: datadog.PtrString("host"),
				},
			}
		}

		series = append(series, s)
	}

	body := datadogV2.MetricPayload{
		Series: series,
	}

	_, _, err := api.SubmitMetrics(d.ctx, body, *datadogV2.NewSubmitMetricsOptionalParameters())
	if err != nil {
		return fmt.Errorf("failed to send metrics: %w", err)
	}

	return nil
}

// SendEvent sends an event to Datadog
func (d *DatadogClient) SendEvent(ctx context.Context, event Event) error {
	api := datadogV1.NewEventsApi(d.apiClient)

	if event.Timestamp == 0 {
		event.Timestamp = time.Now().Unix()
	}

	body := datadogV1.EventCreateRequest{
		Title: event.Title,
		Text:  event.Text,
		Tags:  &event.Tags,
	}

	if event.Timestamp > 0 {
		body.DateHappened = datadog.PtrInt64(event.Timestamp)
	}
	if event.Priority != "" {
		priority := datadogV1.EventPriority(event.Priority)
		body.Priority = &priority
	}
	if event.AlertType != "" {
		alertType := datadogV1.EventAlertType(event.AlertType)
		body.AlertType = &alertType
	}
	if event.Aggregation != "" {
		body.AggregationKey = &event.Aggregation
	}

	_, _, err := api.CreateEvent(d.ctx, body)
	if err != nil {
		return fmt.Errorf("failed to send event: %w", err)
	}

	return nil
}

// CreateMonitor creates a Datadog monitor
func (d *DatadogClient) CreateMonitor(ctx context.Context, monitor Monitor) (int64, error) {
	api := datadogV1.NewMonitorsApi(d.apiClient)

	body := datadogV1.Monitor{
		Name:    &monitor.Name,
		Type:    datadogV1.MonitorType(monitor.Type),
		Query:   monitor.Query,
		Message: &monitor.Message,
		Tags:    &monitor.Tags,
		Options: &datadogV1.MonitorOptions{
			NotifyNoData:      &monitor.Options.NotifyNoData,
			NoDataTimeframe:   datadog.PtrInt64(int64(monitor.Options.NoDataTimeframe)),
			RenotifyInterval:  datadog.PtrInt64(int64(monitor.Options.RenotifyInterval)),
			EvaluationDelay:   datadog.PtrInt64(int64(monitor.Options.EvaluationDelay)),
			RequireFullWindow: &monitor.Options.RequireFullWindow,
			IncludeTags:       &monitor.Options.IncludeTags,
		},
	}

	// Set thresholds
	if monitor.Options.Thresholds.Critical != nil {
		body.Options.Thresholds = &datadogV1.MonitorThresholds{
			Critical:         monitor.Options.Thresholds.Critical,
			CriticalRecovery: monitor.Options.Thresholds.CriticalRecovery,
			Warning:          monitor.Options.Thresholds.Warning,
			WarningRecovery:  monitor.Options.Thresholds.WarningRecovery,
			Ok:               monitor.Options.Thresholds.OK,
		}
	}

	result, _, err := api.CreateMonitor(d.ctx, body)
	if err != nil {
		return 0, fmt.Errorf("failed to create monitor: %w", err)
	}

	return *result.Id, nil
}

// SendVMMetrics sends VM operation metrics
func (d *DatadogClient) SendVMMetrics(ctx context.Context, vmID string, cpuUsage, memoryUsage, diskIO float64) error {
	tags := []string{
		fmt.Sprintf("vm_id:%s", vmID),
		"product:novacron",
		"component:vm-manager",
	}

	metrics := []Metric{
		{
			Name:  "novacron.vm.cpu_usage",
			Type:  "gauge",
			Value: cpuUsage,
			Tags:  tags,
		},
		{
			Name:  "novacron.vm.memory_usage",
			Type:  "gauge",
			Value: memoryUsage,
			Tags:  tags,
		},
		{
			Name:  "novacron.vm.disk_io",
			Type:  "gauge",
			Value: diskIO,
			Tags:  tags,
		},
	}

	return d.SendMetrics(ctx, metrics)
}

// SendMigrationMetrics sends migration metrics
func (d *DatadogClient) SendMigrationMetrics(ctx context.Context, migrationID string, phase string, bytesTransferred int64, duration float64) error {
	tags := []string{
		fmt.Sprintf("migration_id:%s", migrationID),
		fmt.Sprintf("phase:%s", phase),
		"product:novacron",
		"component:migration-engine",
	}

	metrics := []Metric{
		{
			Name:  "novacron.migration.bytes_transferred",
			Type:  "count",
			Value: float64(bytesTransferred),
			Tags:  tags,
		},
		{
			Name:  "novacron.migration.duration",
			Type:  "gauge",
			Value: duration,
			Tags:  tags,
		},
	}

	return d.SendMetrics(ctx, metrics)
}

// CreateNovaCronMonitors creates recommended monitors for NovaCron
func (d *DatadogClient) CreateNovaCronMonitors(ctx context.Context) error {
	monitors := []Monitor{
		{
			Name:    "NovaCron - High Migration Failure Rate",
			Type:    "metric alert",
			Query:   "avg(last_1h):avg:novacron.migration.failed{*} by {migration_type} > 5",
			Message: "Migration failure rate is high. Investigate immediately. @oncall-team",
			Tags:    []string{"product:novacron", "component:migration", "severity:critical"},
			Options: MonitorOptions{
				Thresholds: Thresholds{
					Critical: datadog.PtrFloat64(5),
					Warning:  datadog.PtrFloat64(3),
				},
				NotifyNoData:      true,
				NoDataTimeframe:   60,
				RenotifyInterval:  120,
				RequireFullWindow: false,
			},
		},
		{
			Name:    "NovaCron - High CPU Usage",
			Type:    "metric alert",
			Query:   "avg(last_15m):avg:novacron.vm.cpu_usage{*} by {vm_id} > 90",
			Message: "VM {{vm_id.name}} has high CPU usage. Consider scaling. @ops-team",
			Tags:    []string{"product:novacron", "component:vm", "severity:warning"},
			Options: MonitorOptions{
				Thresholds: Thresholds{
					Critical: datadog.PtrFloat64(90),
					Warning:  datadog.PtrFloat64(80),
				},
				NotifyNoData:     false,
				RenotifyInterval: 60,
			},
		},
		{
			Name:    "NovaCron - Memory Exhaustion",
			Type:    "metric alert",
			Query:   "avg(last_10m):avg:novacron.vm.memory_usage{*} by {vm_id} > 95",
			Message: "VM {{vm_id.name}} is running out of memory. Immediate action required. @oncall-team",
			Tags:    []string{"product:novacron", "component:vm", "severity:critical"},
			Options: MonitorOptions{
				Thresholds: Thresholds{
					Critical: datadog.PtrFloat64(95),
					Warning:  datadog.PtrFloat64(85),
				},
				NotifyNoData:     true,
				NoDataTimeframe:  30,
				RenotifyInterval: 30,
			},
		},
	}

	for _, monitor := range monitors {
		monitorID, err := d.CreateMonitor(ctx, monitor)
		if err != nil {
			return fmt.Errorf("failed to create monitor '%s': %w", monitor.Name, err)
		}
		fmt.Printf("Created monitor: %s (ID: %d)\n", monitor.Name, monitorID)
	}

	return nil
}
