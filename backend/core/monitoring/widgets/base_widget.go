package widgets

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// WidgetType represents the type of widget
type WidgetType string

const (
	WidgetTypeGauge     WidgetType = "gauge"
	WidgetTypeChart     WidgetType = "chart"
	WidgetTypeTable     WidgetType = "table"
	WidgetTypeHeatmap   WidgetType = "heatmap"
	WidgetTypeStatus    WidgetType = "status"
	WidgetTypeStat      WidgetType = "stat"
	WidgetTypeProgress  WidgetType = "progress"
	WidgetTypeAlert     WidgetType = "alert"
	WidgetTypeLog       WidgetType = "log"
	WidgetTypeTopology  WidgetType = "topology"
	WidgetTypePieChart  WidgetType = "pie_chart"
	WidgetTypeScatter   WidgetType = "scatter"
	WidgetTypeHistogram WidgetType = "histogram"
	WidgetTypeCustom    WidgetType = "custom"
)

// DataFormat represents the format of widget data
type DataFormat string

const (
	DataFormatTimeSeries DataFormat = "time_series"
	DataFormatTable      DataFormat = "table"
	DataFormatSingle     DataFormat = "single_value"
	DataFormatHistogram  DataFormat = "histogram"
	DataFormatHeatmap    DataFormat = "heatmap"
	DataFormatTopology   DataFormat = "topology"
	DataFormatLogs       DataFormat = "logs"
	DataFormatCustom     DataFormat = "custom"
	DataFormatJSON       DataFormat = "json"
	DataFormatTimeSeriesJSON DataFormat = "time_series_json"
)

// ChartType represents different chart visualization types
type ChartType string

const (
	ChartTypeLine      ChartType = "line"
	ChartTypeArea      ChartType = "area"
	ChartTypeBar       ChartType = "bar"
	ChartTypeColumn    ChartType = "column"
	ChartTypePie       ChartType = "pie"
	ChartTypeDoughnut  ChartType = "doughnut"
	ChartTypeScatter   ChartType = "scatter"
	ChartTypeBubble    ChartType = "bubble"
	ChartTypeCandlestick ChartType = "candlestick"
)

// StatusLevel represents different status levels
type StatusLevel string

const (
	StatusLevelOK       StatusLevel = "ok"
	StatusLevelWarning  StatusLevel = "warning"
	StatusLevelCritical StatusLevel = "critical"
	StatusLevelUnknown  StatusLevel = "unknown"
)

// WidgetData represents data for a widget
type WidgetData struct {
	// Basic information
	WidgetID  string     `json:"widget_id"`
	Type      WidgetType `json:"type"`
	Format    DataFormat `json:"format"`
	UpdatedAt time.Time  `json:"updated_at"`

	// Data payload
	Value       interface{}            `json:"value,omitempty"`        // For single values
	Series      []DataSeries           `json:"series,omitempty"`       // For time series data
	Rows        []map[string]interface{} `json:"rows,omitempty"`       // For tabular data
	Points      []DataPoint            `json:"points,omitempty"`       // For scatter/bubble charts
	Histogram   []HistogramBucket      `json:"histogram,omitempty"`    // For histogram data
	HeatmapData [][]float64            `json:"heatmap_data,omitempty"` // For heatmap data
	Logs        []LogEntry             `json:"logs,omitempty"`         // For log data
	Topology    TopologyData           `json:"topology,omitempty"`     // For topology data

	// Metadata
	Labels    map[string]string      `json:"labels,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Status    string                 `json:"status"` // "loading", "success", "error"
	Error     string                 `json:"error,omitempty"`
	Unit      string                 `json:"unit,omitempty"`
	
	// Aggregation info
	AggregationWindow string `json:"aggregation_window,omitempty"`
	SampleCount       int    `json:"sample_count,omitempty"`
}

// DataSeries represents a time series of data points
type DataSeries struct {
	Name      string                 `json:"name"`
	Labels    map[string]string      `json:"labels,omitempty"`
	Data      []TimeSeriesPoint      `json:"data"`
	Color     string                 `json:"color,omitempty"`
	Type      ChartType              `json:"type,omitempty"`
	YAxis     string                 `json:"y_axis,omitempty"` // "left", "right"
	Unit      string                 `json:"unit,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// TimeSeriesPoint represents a single point in time series
type TimeSeriesPoint struct {
	Timestamp time.Time   `json:"timestamp"`
	Value     float64     `json:"value"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// DataPoint represents a point for scatter/bubble charts
type DataPoint struct {
	X         float64               `json:"x"`
	Y         float64               `json:"y"`
	Size      float64               `json:"size,omitempty"`      // For bubble charts
	Color     string                `json:"color,omitempty"`
	Label     string                `json:"label,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// HistogramBucket represents a histogram bucket
type HistogramBucket struct {
	LowerBound float64 `json:"lower_bound"`
	UpperBound float64 `json:"upper_bound"`
	Count      int64   `json:"count"`
	Frequency  float64 `json:"frequency"`
}

// LogEntry represents a log entry
type LogEntry struct {
	Timestamp time.Time         `json:"timestamp"`
	Level     string            `json:"level"`
	Message   string            `json:"message"`
	Source    string            `json:"source,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
}

// TopologyData represents network topology data
type TopologyData struct {
	Nodes []TopologyNode `json:"nodes"`
	Edges []TopologyEdge `json:"edges"`
}

// TopologyNode represents a node in topology
type TopologyNode struct {
	ID       string                 `json:"id"`
	Label    string                 `json:"label"`
	Type     string                 `json:"type"`
	Status   StatusLevel            `json:"status"`
	Position *Position              `json:"position,omitempty"`
	Size     *Size                  `json:"size,omitempty"`
	Color    string                 `json:"color,omitempty"`
	Icon     string                 `json:"icon,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// TopologyEdge represents an edge in topology
type TopologyEdge struct {
	ID       string                 `json:"id"`
	Source   string                 `json:"source"`
	Target   string                 `json:"target"`
	Label    string                 `json:"label,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Status   StatusLevel            `json:"status,omitempty"`
	Weight   float64                `json:"weight,omitempty"`
	Color    string                 `json:"color,omitempty"`
	Style    string                 `json:"style,omitempty"` // "solid", "dashed", "dotted"
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Position represents a 2D position
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Size represents dimensions
type Size struct {
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// Widget represents the base widget interface
type Widget interface {
	// Basic properties
	GetID() string
	GetType() WidgetType
	GetTitle() string
	GetDescription() string

	// Configuration
	GetConfig() map[string]interface{}
	SetConfig(config map[string]interface{}) error
	ValidateConfig() error

	// Data handling
	ProcessData(ctx context.Context, rawData interface{}) (*WidgetData, error)
	GetSupportedDataFormats() []DataFormat
	GetDataRequirements() DataRequirements

	// Rendering
	RenderHTML(data *WidgetData, options RenderOptions) (string, error)
	RenderJSON(data *WidgetData) (json.RawMessage, error)
	
	// Responsive behavior
	GetResponsiveBreakpoints() map[string]ResponsiveConfig
	AdaptToDevice(deviceType DeviceType) error
}

// DataRequirements specifies what data a widget needs
type DataRequirements struct {
	MinDataPoints    int               `json:"min_data_points"`
	MaxDataPoints    int               `json:"max_data_points"`
	RequiredFields   []string          `json:"required_fields"`
	OptionalFields   []string          `json:"optional_fields"`
	SupportedFormats []DataFormat      `json:"supported_formats"`
	TimeRange        *TimeRange        `json:"time_range,omitempty"`
	Aggregations     []string          `json:"aggregations,omitempty"`
	Constraints      map[string]interface{} `json:"constraints,omitempty"`
}

// TimeRange represents a time range
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// RenderOptions represents rendering options
type RenderOptions struct {
	Width       int               `json:"width"`
	Height      int               `json:"height"`
	Theme       string            `json:"theme"`
	Interactive bool              `json:"interactive"`
	DeviceType  DeviceType        `json:"device_type"`
	Format      string            `json:"format"` // "html", "svg", "png"
	CustomCSS   string            `json:"custom_css,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

// DeviceType represents different device types
type DeviceType string

const (
	DeviceTypeDesktop DeviceType = "desktop"
	DeviceTypeTablet  DeviceType = "tablet"
	DeviceTypeMobile  DeviceType = "mobile"
)

// ResponsiveConfig represents responsive configuration
type ResponsiveConfig struct {
	Hidden     bool              `json:"hidden"`
	Width      int               `json:"width"`
	Height     int               `json:"height"`
	Config     map[string]interface{} `json:"config"`
	Properties map[string]interface{} `json:"properties"`
}

// BaseWidget provides common functionality for all widgets
type BaseWidget struct {
	ID          string                 `json:"id"`
	Type        WidgetType             `json:"type"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Config      map[string]interface{} `json:"config"`
	
	// Responsive configuration
	ResponsiveBreakpoints map[string]ResponsiveConfig `json:"responsive_breakpoints,omitempty"`
	CurrentDevice         DeviceType                  `json:"current_device,omitempty"`
}

// NewBaseWidget creates a new base widget
func NewBaseWidget(id string, widgetType WidgetType, title string) *BaseWidget {
	return &BaseWidget{
		ID:          id,
		Type:        widgetType,
		Title:       title,
		Config:      make(map[string]interface{}),
		ResponsiveBreakpoints: map[string]ResponsiveConfig{
			"mobile": {
				Width:  300,
				Height: 200,
			},
			"tablet": {
				Width:  600,
				Height: 400,
			},
			"desktop": {
				Width:  800,
				Height: 600,
			},
		},
		CurrentDevice: DeviceTypeDesktop,
	}
}

// GetID returns the widget ID
func (w *BaseWidget) GetID() string {
	return w.ID
}

// GetType returns the widget type
func (w *BaseWidget) GetType() WidgetType {
	return w.Type
}

// GetTitle returns the widget title
func (w *BaseWidget) GetTitle() string {
	return w.Title
}

// GetDescription returns the widget description
func (w *BaseWidget) GetDescription() string {
	return w.Description
}

// GetConfig returns the widget configuration
func (w *BaseWidget) GetConfig() map[string]interface{} {
	return w.Config
}

// SetConfig sets the widget configuration
func (w *BaseWidget) SetConfig(config map[string]interface{}) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}
	w.Config = config
	return nil
}

// GetResponsiveBreakpoints returns responsive breakpoints
func (w *BaseWidget) GetResponsiveBreakpoints() map[string]ResponsiveConfig {
	return w.ResponsiveBreakpoints
}

// AdaptToDevice adapts the widget to a device type
func (w *BaseWidget) AdaptToDevice(deviceType DeviceType) error {
	w.CurrentDevice = deviceType
	return nil
}

// GetDataRequirements returns the data requirements for this widget
func (w *BaseWidget) GetDataRequirements() DataRequirements {
	// Default data requirements - can be overridden by specific widget types
	return DataRequirements{
		MinDataPoints:    1,
		MaxDataPoints:    1000,
		RequiredFields:   []string{"value"},
		OptionalFields:   []string{"timestamp", "label"},
		SupportedFormats: []DataFormat{DataFormatJSON, DataFormatTimeSeriesJSON},
		Constraints:      make(map[string]interface{}),
	}
}

// GetSupportedDataFormats returns the supported data formats
func (w *BaseWidget) GetSupportedDataFormats() []DataFormat {
	return []DataFormat{DataFormatJSON, DataFormatTimeSeriesJSON, DataFormatSingle}
}

// ProcessData processes raw data for this widget
func (w *BaseWidget) ProcessData(ctx context.Context, rawData interface{}) (*WidgetData, error) {
	// Default implementation - should be overridden by specific widget types
	return &WidgetData{
		Value: rawData,
		Metadata: make(map[string]interface{}),
	}, nil
}

// ValidateConfig validates the widget configuration
func (w *BaseWidget) ValidateConfig() error {
	// Default implementation - can be overridden
	return nil
}

// RenderHTML renders the widget as HTML
func (w *BaseWidget) RenderHTML(data *WidgetData, options RenderOptions) (string, error) {
	return fmt.Sprintf(`<div id="%s" class="widget"><h3>%s</h3><p>Data: %v</p></div>`, w.ID, w.Title, data.Value), nil
}

// RenderJSON renders the widget as JSON
func (w *BaseWidget) RenderJSON(data *WidgetData) (json.RawMessage, error) {
	result := map[string]interface{}{
		"id": w.ID,
		"type": w.Type,
		"title": w.Title,
		"data": data,
	}
	bytes, err := json.Marshal(result)
	return json.RawMessage(bytes), err
}

// GetConfigValue gets a configuration value with type checking
func (w *BaseWidget) GetConfigValue(key string, defaultValue interface{}) interface{} {
	if value, exists := w.Config[key]; exists {
		return value
	}
	return defaultValue
}

// GetConfigString gets a string configuration value
func (w *BaseWidget) GetConfigString(key string, defaultValue string) string {
	if value, exists := w.Config[key]; exists {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return defaultValue
}

// GetConfigInt gets an integer configuration value
func (w *BaseWidget) GetConfigInt(key string, defaultValue int) int {
	if value, exists := w.Config[key]; exists {
		switch v := value.(type) {
		case int:
			return v
		case float64:
			return int(v)
		}
	}
	return defaultValue
}

// GetConfigFloat gets a float configuration value
func (w *BaseWidget) GetConfigFloat(key string, defaultValue float64) float64 {
	if value, exists := w.Config[key]; exists {
		switch v := value.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		}
	}
	return defaultValue
}

// GetConfigBool gets a boolean configuration value
func (w *BaseWidget) GetConfigBool(key string, defaultValue bool) bool {
	if value, exists := w.Config[key]; exists {
		if b, ok := value.(bool); ok {
			return b
		}
	}
	return defaultValue
}

// WidgetFactory creates widgets of different types
type WidgetFactory struct {
	constructors map[WidgetType]func() Widget
	validators   map[WidgetType]func(config map[string]interface{}) error
}

// NewWidgetFactory creates a new widget factory
func NewWidgetFactory() *WidgetFactory {
	return &WidgetFactory{
		constructors: make(map[WidgetType]func() Widget),
		validators:   make(map[WidgetType]func(config map[string]interface{}) error),
	}
}

// RegisterWidget registers a widget type
func (f *WidgetFactory) RegisterWidget(widgetType WidgetType, constructor func() Widget, validator func(config map[string]interface{}) error) {
	f.constructors[widgetType] = constructor
	if validator != nil {
		f.validators[widgetType] = validator
	}
}

// CreateWidget creates a widget of the specified type
func (f *WidgetFactory) CreateWidget(widgetType WidgetType, id, title string, config map[string]interface{}) (Widget, error) {
	constructor, exists := f.constructors[widgetType]
	if !exists {
		return nil, fmt.Errorf("widget type %s not registered", widgetType)
	}

	// Validate configuration
	if validator, exists := f.validators[widgetType]; exists {
		if err := validator(config); err != nil {
			return nil, fmt.Errorf("widget configuration validation failed: %w", err)
		}
	}

	widget := constructor()
	
	// Set basic properties (assuming the widget has these methods)
	if baseWidget, ok := widget.(*BaseWidget); ok {
		baseWidget.ID = id
		baseWidget.Title = title
		baseWidget.Type = widgetType
		baseWidget.Config = config
	}

	return widget, nil
}

// GetSupportedTypes returns all supported widget types
func (f *WidgetFactory) GetSupportedTypes() []WidgetType {
	var types []WidgetType
	for widgetType := range f.constructors {
		types = append(types, widgetType)
	}
	return types
}

// WidgetValidator provides validation for widgets
type WidgetValidator struct{}

// NewWidgetValidator creates a new widget validator
func NewWidgetValidator() *WidgetValidator {
	return &WidgetValidator{}
}

// ValidateWidget validates a widget configuration
func (v *WidgetValidator) ValidateWidget(widget Widget) error {
	if widget.GetID() == "" {
		return fmt.Errorf("widget ID cannot be empty")
	}
	
	if widget.GetType() == "" {
		return fmt.Errorf("widget type cannot be empty")
	}

	return widget.ValidateConfig()
}

// ValidateDataRequirements validates data against widget requirements
func (v *WidgetValidator) ValidateDataRequirements(data *WidgetData, requirements DataRequirements) error {
	// Check data format
	formatSupported := false
	for _, format := range requirements.SupportedFormats {
		if data.Format == format {
			formatSupported = true
			break
		}
	}
	if !formatSupported {
		return fmt.Errorf("data format %s not supported", data.Format)
	}

	// Check data points count
	dataPointCount := 0
	switch data.Format {
	case DataFormatTimeSeries:
		for _, series := range data.Series {
			dataPointCount += len(series.Data)
		}
	case DataFormatTable:
		dataPointCount = len(data.Rows)
	case DataFormatSingle:
		dataPointCount = 1
	}

	if requirements.MinDataPoints > 0 && dataPointCount < requirements.MinDataPoints {
		return fmt.Errorf("insufficient data points: got %d, need at least %d", dataPointCount, requirements.MinDataPoints)
	}

	if requirements.MaxDataPoints > 0 && dataPointCount > requirements.MaxDataPoints {
		return fmt.Errorf("too many data points: got %d, maximum allowed %d", dataPointCount, requirements.MaxDataPoints)
	}

	return nil
}

// WidgetRenderer provides common rendering functionality
type WidgetRenderer struct {
	templates map[WidgetType]string
	themes    map[string]map[string]interface{}
}

// NewWidgetRenderer creates a new widget renderer
func NewWidgetRenderer() *WidgetRenderer {
	return &WidgetRenderer{
		templates: make(map[WidgetType]string),
		themes:    make(map[string]map[string]interface{}),
	}
}

// RegisterTemplate registers a template for a widget type
func (r *WidgetRenderer) RegisterTemplate(widgetType WidgetType, template string) {
	r.templates[widgetType] = template
}

// RegisterTheme registers a theme
func (r *WidgetRenderer) RegisterTheme(name string, theme map[string]interface{}) {
	r.themes[name] = theme
}

// RenderWidget renders a widget to HTML
func (r *WidgetRenderer) RenderWidget(widget Widget, data *WidgetData, options RenderOptions) (string, error) {
	template, exists := r.templates[widget.GetType()]
	if !exists {
		return "", fmt.Errorf("no template found for widget type %s", widget.GetType())
	}

	// Apply theme
	if options.Theme != "" {
		if theme, exists := r.themes[options.Theme]; exists {
			// Apply theme properties to options
			for key, value := range theme {
				if options.Options == nil {
					options.Options = make(map[string]interface{})
				}
				options.Options[key] = value
			}
		}
	}

	// Render using template (simplified implementation)
	// In a real implementation, this would use a template engine like Go's html/template
	return fmt.Sprintf("<!-- Widget: %s -->\n%s", widget.GetTitle(), template), nil
}

// Common utility functions

// FormatValue formats a value for display
func FormatValue(value interface{}, unit string, precision int) string {
	switch v := value.(type) {
	case float64:
		format := fmt.Sprintf("%%.%df", precision)
		formatted := fmt.Sprintf(format, v)
		if unit != "" {
			return fmt.Sprintf("%s %s", formatted, unit)
		}
		return formatted
	case int:
		if unit != "" {
			return fmt.Sprintf("%d %s", v, unit)
		}
		return fmt.Sprintf("%d", v)
	case string:
		return v
	default:
		return fmt.Sprintf("%v", value)
	}
}

// ParseColor parses a color string
func ParseColor(color string) (r, g, b uint8, err error) {
	if len(color) == 7 && color[0] == '#' {
		_, err = fmt.Sscanf(color[1:], "%02x%02x%02x", &r, &g, &b)
		return
	}
	return 0, 0, 0, fmt.Errorf("invalid color format: %s", color)
}

// GenerateColorPalette generates a color palette
func GenerateColorPalette(count int) []string {
	colors := []string{
		"#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
		"#9966FF", "#FF9F40", "#C9CBCF", "#4BC0C0",
		"#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
	}
	
	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = colors[i%len(colors)]
	}
	return result
}