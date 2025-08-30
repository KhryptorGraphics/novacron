package widgets

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"
)

// ChartWidget represents a chart visualization widget
type ChartWidget struct {
	*BaseWidget
}

// ChartConfig represents configuration for a chart widget
type ChartConfig struct {
	// Chart type
	ChartType ChartType `json:"chart_type"`
	
	// Axes configuration
	XAxis XAxisConfig `json:"x_axis"`
	YAxis YAxisConfig `json:"y_axis"`
	Y2Axis *YAxisConfig `json:"y2_axis,omitempty"` // Secondary Y-axis
	
	// Series configuration
	SeriesConfig map[string]SeriesConfig `json:"series_config"`
	
	// Display options
	ShowLegend    bool   `json:"show_legend"`
	LegendPosition string `json:"legend_position"` // "top", "bottom", "left", "right"
	ShowGrid      bool   `json:"show_grid"`
	ShowTooltips  bool   `json:"show_tooltips"`
	Stacked       bool   `json:"stacked"`
	Fill          bool   `json:"fill"`
	Smooth        bool   `json:"smooth"`
	
	// Colors and styling
	Colors        []string `json:"colors"`
	BackgroundColor string `json:"background_color"`
	GridColor     string   `json:"grid_color"`
	TextColor     string   `json:"text_color"`
	
	// Data processing
	AggregationWindow string `json:"aggregation_window"` // "1m", "5m", "1h"
	AggregationMethod string `json:"aggregation_method"` // "avg", "sum", "min", "max"
	MaxDataPoints     int    `json:"max_data_points"`
	
	// Animation
	Animated       bool `json:"animated"`
	AnimationDuration int `json:"animation_duration"` // milliseconds
	
	// Zoom and pan
	Zoomable bool `json:"zoomable"`
	Pannable bool `json:"pannable"`
	
	// Annotations
	Annotations []ChartAnnotation `json:"annotations"`
	
	// Thresholds
	Thresholds []Threshold `json:"thresholds"`
}

// XAxisConfig represents X-axis configuration
type XAxisConfig struct {
	Type        string `json:"type"`         // "time", "linear", "category"
	Label       string `json:"label"`
	ShowLabels  bool   `json:"show_labels"`
	ShowTicks   bool   `json:"show_ticks"`
	TickFormat  string `json:"tick_format"`
	Min         *float64 `json:"min,omitempty"`
	Max         *float64 `json:"max,omitempty"`
	AutoScale   bool   `json:"auto_scale"`
}

// YAxisConfig represents Y-axis configuration
type YAxisConfig struct {
	Label      string   `json:"label"`
	Unit       string   `json:"unit"`
	ShowLabels bool     `json:"show_labels"`
	ShowTicks  bool     `json:"show_ticks"`
	TickFormat string   `json:"tick_format"`
	Min        *float64 `json:"min,omitempty"`
	Max        *float64 `json:"max,omitempty"`
	AutoScale  bool     `json:"auto_scale"`
	LogScale   bool     `json:"log_scale"`
	Position   string   `json:"position"` // "left", "right"
}

// SeriesConfig represents configuration for a data series
type SeriesConfig struct {
	Name        string    `json:"name"`
	Type        ChartType `json:"type"`
	Color       string    `json:"color"`
	LineWidth   int       `json:"line_width"`
	PointRadius int       `json:"point_radius"`
	Fill        bool      `json:"fill"`
	YAxis       string    `json:"y_axis"` // "y1", "y2"
	Visible     bool      `json:"visible"`
	
	// Line-specific
	Smooth bool   `json:"smooth"`
	Dash   []int  `json:"dash,omitempty"` // dash pattern for lines
	
	// Bar-specific
	BarWidth float64 `json:"bar_width"`
	
	// Area-specific
	FillOpacity float64 `json:"fill_opacity"`
}

// ChartAnnotation represents an annotation on the chart
type ChartAnnotation struct {
	Type     string    `json:"type"`      // "line", "point", "area", "text"
	X        float64   `json:"x"`
	Y        *float64  `json:"y,omitempty"`
	X2       *float64  `json:"x2,omitempty"` // For area annotations
	Y2       *float64  `json:"y2,omitempty"` // For area annotations
	Text     string    `json:"text,omitempty"`
	Color    string    `json:"color"`
	Style    string    `json:"style,omitempty"` // "solid", "dashed", "dotted"
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Threshold represents a threshold line on the chart
type Threshold struct {
	Value       float64     `json:"value"`
	Label       string      `json:"label"`
	Color       string      `json:"color"`
	Style       string      `json:"style"` // "solid", "dashed", "dotted"
	Operator    string      `json:"operator"` // "gt", "lt", "gte", "lte"
	Severity    StatusLevel `json:"severity"`
	ShowInLegend bool       `json:"show_in_legend"`
}

// DefaultChartConfig returns default configuration for chart widget
func DefaultChartConfig() *ChartConfig {
	return &ChartConfig{
		ChartType: ChartTypeLine,
		XAxis: XAxisConfig{
			Type:       "time",
			ShowLabels: true,
			ShowTicks:  true,
			AutoScale:  true,
			TickFormat: "HH:mm",
		},
		YAxis: YAxisConfig{
			ShowLabels: true,
			ShowTicks:  true,
			AutoScale:  true,
			Position:   "left",
		},
		SeriesConfig:      make(map[string]SeriesConfig),
		ShowLegend:        true,
		LegendPosition:    "bottom",
		ShowGrid:          true,
		ShowTooltips:      true,
		Colors:            []string{"#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"},
		BackgroundColor:   "#FFFFFF",
		GridColor:         "#E0E0E0",
		TextColor:         "#333333",
		MaxDataPoints:     1000,
		Animated:          true,
		AnimationDuration: 1000,
		Zoomable:          true,
		Pannable:          true,
	}
}

// NewChartWidget creates a new chart widget
func NewChartWidget() Widget {
	base := NewBaseWidget("", WidgetTypeChart, "")
	
	// Set default configuration
	defaultConfig := DefaultChartConfig()
	configMap := make(map[string]interface{})
	configBytes, _ := json.Marshal(defaultConfig)
	json.Unmarshal(configBytes, &configMap)
	base.Config = configMap
	
	return &ChartWidget{
		BaseWidget: base,
	}
}

// ValidateConfig validates the chart configuration
func (w *ChartWidget) ValidateConfig() error {
	config := w.getConfig()
	
	// Validate chart type
	validChartTypes := []ChartType{
		ChartTypeLine, ChartTypeArea, ChartTypeBar, ChartTypeColumn,
		ChartTypePie, ChartTypeDoughnut, ChartTypeScatter,
	}
	
	validType := false
	for _, t := range validChartTypes {
		if config.ChartType == t {
			validType = true
			break
		}
	}
	if !validType {
		return fmt.Errorf("invalid chart type: %s", config.ChartType)
	}
	
	// Validate X-axis type
	validXAxisTypes := []string{"time", "linear", "category"}
	validXAxis := false
	for _, t := range validXAxisTypes {
		if config.XAxis.Type == t {
			validXAxis = true
			break
		}
	}
	if !validXAxis {
		return fmt.Errorf("invalid X-axis type: %s", config.XAxis.Type)
	}
	
	// Validate legend position
	validPositions := []string{"top", "bottom", "left", "right"}
	validPosition := false
	for _, p := range validPositions {
		if config.LegendPosition == p {
			validPosition = true
			break
		}
	}
	if !validPosition {
		return fmt.Errorf("invalid legend position: %s", config.LegendPosition)
	}
	
	// Validate max data points
	if config.MaxDataPoints <= 0 {
		return fmt.Errorf("max_data_points must be positive, got %d", config.MaxDataPoints)
	}
	
	return nil
}

// ProcessData processes raw data for the chart widget
func (w *ChartWidget) ProcessData(ctx context.Context, rawData interface{}) (*WidgetData, error) {
	var series []DataSeries
	var metadata map[string]interface{}
	
	switch data := rawData.(type) {
	case []DataSeries:
		series = data
	case map[string]interface{}:
		if seriesData, exists := data["series"]; exists {
			if seriesSlice, ok := seriesData.([]DataSeries); ok {
				series = seriesSlice
			} else {
				return nil, fmt.Errorf("invalid series data format")
			}
		} else {
			return nil, fmt.Errorf("series field not found in data")
		}
		
		// Extract metadata if present
		if m, exists := data["metadata"].(map[string]interface{}); exists {
			metadata = m
		}
	case *WidgetData:
		// Data is already processed
		series = data.Series
		metadata = data.Metadata
	default:
		return nil, fmt.Errorf("unsupported data type: %T", rawData)
	}
	
	if len(series) == 0 {
		return nil, fmt.Errorf("no data series provided")
	}
	
	config := w.getConfig()
	
	// Process and aggregate data if needed
	processedSeries := w.processSeriesData(series, config)
	
	// Apply data limits
	if config.MaxDataPoints > 0 {
		processedSeries = w.limitDataPoints(processedSeries, config.MaxDataPoints)
	}
	
	// Calculate statistics
	stats := w.calculateStatistics(processedSeries)
	
	// Prepare metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["statistics"] = stats
	metadata["series_count"] = len(processedSeries)
	metadata["chart_type"] = config.ChartType
	metadata["total_points"] = w.countDataPoints(processedSeries)
	
	// Add threshold violations if configured
	if len(config.Thresholds) > 0 {
		violations := w.checkThresholdViolations(processedSeries, config.Thresholds)
		metadata["threshold_violations"] = violations
	}
	
	return &WidgetData{
		WidgetID:  w.GetID(),
		Type:      WidgetTypeChart,
		Format:    DataFormatTimeSeries,
		Series:    processedSeries,
		Metadata:  metadata,
		Status:    "success",
	}, nil
}

// GetSupportedDataFormats returns supported data formats
func (w *ChartWidget) GetSupportedDataFormats() []DataFormat {
	return []DataFormat{DataFormatTimeSeries}
}

// GetDataRequirements returns data requirements for the chart widget
func (w *ChartWidget) GetDataRequirements() DataRequirements {
	return DataRequirements{
		MinDataPoints:    1,
		MaxDataPoints:    10000,
		RequiredFields:   []string{"series"},
		OptionalFields:   []string{"metadata"},
		SupportedFormats: []DataFormat{DataFormatTimeSeries},
	}
}

// RenderHTML renders the chart widget as HTML
func (w *ChartWidget) RenderHTML(data *WidgetData, options RenderOptions) (string, error) {
	if len(data.Series) == 0 {
		return "", fmt.Errorf("no data to render")
	}
	
	config := w.getConfig()
	
	// Generate unique ID for this chart instance
	chartID := fmt.Sprintf("chart_%s_%d", w.GetID(), time.Now().UnixNano())
	
	width := options.Width
	if width <= 0 {
		width = 800
	}
	height := options.Height
	if height <= 0 {
		height = 400
	}
	
	// Generate Chart.js configuration
	chartConfig := w.generateChartJSConfig(data, config, options)
	configJSON, err := json.Marshal(chartConfig)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chart config: %w", err)
	}
	
	html := fmt.Sprintf(`
<div class="chart-widget" style="width: %dpx; height: %dpx;">
	<canvas id="%s" width="%d" height="%d"></canvas>
</div>
<script>
(function() {
	var ctx = document.getElementById('%s').getContext('2d');
	var config = %s;
	var chart = new Chart(ctx, config);
	
	// Store chart instance for later access
	if (!window.chartInstances) {
		window.chartInstances = {};
	}
	window.chartInstances['%s'] = chart;
})();
</script>`,
		width, height, chartID, width, height,
		chartID, string(configJSON), chartID)
	
	return html, nil
}

// RenderJSON renders the chart widget data as JSON
func (w *ChartWidget) RenderJSON(data *WidgetData) (json.RawMessage, error) {
	if data == nil {
		return nil, fmt.Errorf("no data to render")
	}
	
	config := w.getConfig()
	
	result := map[string]interface{}{
		"type":   "chart",
		"series": data.Series,
		"config": map[string]interface{}{
			"chartType":     config.ChartType,
			"showLegend":    config.ShowLegend,
			"showGrid":      config.ShowGrid,
			"showTooltips":  config.ShowTooltips,
			"colors":        config.Colors,
			"animated":      config.Animated,
			"zoomable":      config.Zoomable,
			"pannable":      config.Pannable,
			"xAxis":         config.XAxis,
			"yAxis":         config.YAxis,
			"thresholds":    config.Thresholds,
			"annotations":   config.Annotations,
		},
		"metadata": data.Metadata,
	}
	
	return json.Marshal(result)
}

// Helper methods

func (w *ChartWidget) getConfig() *ChartConfig {
	config := &ChartConfig{}
	configBytes, _ := json.Marshal(w.Config)
	json.Unmarshal(configBytes, config)
	
	// Apply defaults for missing values
	defaults := DefaultChartConfig()
	if config.ChartType == "" {
		config.ChartType = defaults.ChartType
	}
	if len(config.Colors) == 0 {
		config.Colors = defaults.Colors
	}
	if config.MaxDataPoints == 0 {
		config.MaxDataPoints = defaults.MaxDataPoints
	}
	
	return config
}

func (w *ChartWidget) processSeriesData(series []DataSeries, config *ChartConfig) []DataSeries {
	processedSeries := make([]DataSeries, len(series))
	
	for i, s := range series {
		processedSeries[i] = DataSeries{
			Name:   s.Name,
			Labels: s.Labels,
			Data:   w.applySorting(s.Data),
			Color:  s.Color,
			Type:   s.Type,
			YAxis:  s.YAxis,
			Unit:   s.Unit,
		}
		
		// Apply series-specific configuration
		if seriesConfig, exists := config.SeriesConfig[s.Name]; exists {
			if seriesConfig.Color != "" {
				processedSeries[i].Color = seriesConfig.Color
			}
			if seriesConfig.Type != "" {
				processedSeries[i].Type = seriesConfig.Type
			}
		} else if processedSeries[i].Color == "" {
			// Assign default color
			colorIndex := i % len(config.Colors)
			processedSeries[i].Color = config.Colors[colorIndex]
		}
		
		// Apply aggregation if configured
		if config.AggregationWindow != "" && config.AggregationMethod != "" {
			processedSeries[i].Data = w.aggregateData(processedSeries[i].Data, config.AggregationWindow, config.AggregationMethod)
		}
	}
	
	return processedSeries
}

func (w *ChartWidget) applySorting(data []TimeSeriesPoint) []TimeSeriesPoint {
	// Sort by timestamp
	sorted := make([]TimeSeriesPoint, len(data))
	copy(sorted, data)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})
	return sorted
}

func (w *ChartWidget) limitDataPoints(series []DataSeries, maxPoints int) []DataSeries {
	limited := make([]DataSeries, len(series))
	
	for i, s := range series {
		limited[i] = s
		if len(s.Data) > maxPoints {
			// Simple decimation - take every nth point
			step := len(s.Data) / maxPoints
			if step < 1 {
				step = 1
			}
			
			var decimated []TimeSeriesPoint
			for j := 0; j < len(s.Data); j += step {
				decimated = append(decimated, s.Data[j])
			}
			limited[i].Data = decimated
		}
	}
	
	return limited
}

func (w *ChartWidget) aggregateData(data []TimeSeriesPoint, window, method string) []TimeSeriesPoint {
	if len(data) == 0 {
		return data
	}
	
	// Parse window duration
	duration, err := time.ParseDuration(window)
	if err != nil {
		return data // Return original data if window is invalid
	}
	
	// Group data points by time window
	groups := make(map[int64][]TimeSeriesPoint)
	
	for _, point := range data {
		windowStart := point.Timestamp.Truncate(duration).Unix()
		groups[windowStart] = append(groups[windowStart], point)
	}
	
	// Aggregate each group
	var aggregated []TimeSeriesPoint
	for windowStart, points := range groups {
		aggregatedPoint := TimeSeriesPoint{
			Timestamp: time.Unix(windowStart, 0),
			Value:     w.aggregateValues(points, method),
		}
		aggregated = append(aggregated, aggregatedPoint)
	}
	
	// Sort by timestamp
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Timestamp.Before(aggregated[j].Timestamp)
	})
	
	return aggregated
}

func (w *ChartWidget) aggregateValues(points []TimeSeriesPoint, method string) float64 {
	if len(points) == 0 {
		return 0
	}
	
	switch method {
	case "sum":
		sum := 0.0
		for _, p := range points {
			sum += p.Value
		}
		return sum
	case "avg":
		sum := 0.0
		for _, p := range points {
			sum += p.Value
		}
		return sum / float64(len(points))
	case "min":
		min := points[0].Value
		for _, p := range points[1:] {
			if p.Value < min {
				min = p.Value
			}
		}
		return min
	case "max":
		max := points[0].Value
		for _, p := range points[1:] {
			if p.Value > max {
				max = p.Value
			}
		}
		return max
	default:
		// Default to average
		sum := 0.0
		for _, p := range points {
			sum += p.Value
		}
		return sum / float64(len(points))
	}
}

func (w *ChartWidget) calculateStatistics(series []DataSeries) map[string]interface{} {
	stats := make(map[string]interface{})
	
	for _, s := range series {
		if len(s.Data) == 0 {
			continue
		}
		
		values := make([]float64, len(s.Data))
		for i, point := range s.Data {
			values[i] = point.Value
		}
		
		seriesStats := map[string]float64{
			"count": float64(len(values)),
			"sum":   w.sum(values),
			"avg":   w.average(values),
			"min":   w.min(values),
			"max":   w.max(values),
			"std":   w.standardDeviation(values),
		}
		
		stats[s.Name] = seriesStats
	}
	
	return stats
}

func (w *ChartWidget) countDataPoints(series []DataSeries) int {
	total := 0
	for _, s := range series {
		total += len(s.Data)
	}
	return total
}

func (w *ChartWidget) checkThresholdViolations(series []DataSeries, thresholds []Threshold) []map[string]interface{} {
	var violations []map[string]interface{}
	
	for _, threshold := range thresholds {
		for _, s := range series {
			for _, point := range s.Data {
				violated := false
				
				switch threshold.Operator {
				case "gt":
					violated = point.Value > threshold.Value
				case "gte":
					violated = point.Value >= threshold.Value
				case "lt":
					violated = point.Value < threshold.Value
				case "lte":
					violated = point.Value <= threshold.Value
				}
				
				if violated {
					violations = append(violations, map[string]interface{}{
						"threshold": threshold,
						"series":    s.Name,
						"timestamp": point.Timestamp,
						"value":     point.Value,
					})
				}
			}
		}
	}
	
	return violations
}

func (w *ChartWidget) generateChartJSConfig(data *WidgetData, config *ChartConfig, options RenderOptions) map[string]interface{} {
	datasets := make([]map[string]interface{}, len(data.Series))
	
	for i, series := range data.Series {
		dataset := map[string]interface{}{
			"label":           series.Name,
			"data":            w.convertToChartJSData(series.Data, config.XAxis.Type),
			"backgroundColor": series.Color,
			"borderColor":     series.Color,
			"fill":           config.Fill,
		}
		
		// Apply series-specific configuration
		if seriesConfig, exists := config.SeriesConfig[series.Name]; exists {
			if seriesConfig.LineWidth > 0 {
				dataset["borderWidth"] = seriesConfig.LineWidth
			}
			if seriesConfig.PointRadius > 0 {
				dataset["pointRadius"] = seriesConfig.PointRadius
			}
			if seriesConfig.Fill {
				dataset["fill"] = true
				dataset["backgroundColor"] = w.adjustOpacity(series.Color, seriesConfig.FillOpacity)
			}
		}
		
		datasets[i] = dataset
	}
	
	chartConfig := map[string]interface{}{
		"type": string(config.ChartType),
		"data": map[string]interface{}{
			"datasets": datasets,
		},
		"options": map[string]interface{}{
			"responsive": true,
			"plugins": map[string]interface{}{
				"legend": map[string]interface{}{
					"display":  config.ShowLegend,
					"position": config.LegendPosition,
				},
				"tooltip": map[string]interface{}{
					"enabled": config.ShowTooltips,
				},
			},
			"scales": w.generateScaleConfig(config),
			"animation": map[string]interface{}{
				"duration": config.AnimationDuration,
			},
		},
	}
	
	// Add zoom/pan plugins if enabled
	if config.Zoomable || config.Pannable {
		chartConfig["options"].(map[string]interface{})["plugins"].(map[string]interface{})["zoom"] = map[string]interface{}{
			"zoom": map[string]interface{}{
				"wheel": map[string]interface{}{
					"enabled": config.Zoomable,
				},
				"pinch": map[string]interface{}{
					"enabled": config.Zoomable,
				},
			},
			"pan": map[string]interface{}{
				"enabled": config.Pannable,
			},
		}
	}
	
	return chartConfig
}

func (w *ChartWidget) convertToChartJSData(data []TimeSeriesPoint, xAxisType string) []map[string]interface{} {
	result := make([]map[string]interface{}, len(data))
	
	for i, point := range data {
		dataPoint := map[string]interface{}{
			"y": point.Value,
		}
		
		if xAxisType == "time" {
			dataPoint["x"] = point.Timestamp.Format(time.RFC3339)
		} else {
			dataPoint["x"] = point.Timestamp.Unix()
		}
		
		result[i] = dataPoint
	}
	
	return result
}

func (w *ChartWidget) generateScaleConfig(config *ChartConfig) map[string]interface{} {
	scales := map[string]interface{}{
		"x": map[string]interface{}{
			"type":    config.XAxis.Type,
			"display": config.XAxis.ShowLabels,
			"title": map[string]interface{}{
				"display": config.XAxis.Label != "",
				"text":    config.XAxis.Label,
			},
			"grid": map[string]interface{}{
				"display": config.ShowGrid,
				"color":   config.GridColor,
			},
		},
		"y": map[string]interface{}{
			"display": config.YAxis.ShowLabels,
			"title": map[string]interface{}{
				"display": config.YAxis.Label != "",
				"text":    config.YAxis.Label,
			},
			"grid": map[string]interface{}{
				"display": config.ShowGrid,
				"color":   config.GridColor,
			},
		},
	}
	
	// Add Y-axis range if specified
	if config.YAxis.Min != nil || config.YAxis.Max != nil {
		scales["y"].(map[string]interface{})["min"] = config.YAxis.Min
		scales["y"].(map[string]interface{})["max"] = config.YAxis.Max
	}
	
	// Add secondary Y-axis if configured
	if config.Y2Axis != nil {
		scales["y2"] = map[string]interface{}{
			"type":     "linear",
			"display":  config.Y2Axis.ShowLabels,
			"position": "right",
			"title": map[string]interface{}{
				"display": config.Y2Axis.Label != "",
				"text":    config.Y2Axis.Label,
			},
			"grid": map[string]interface{}{
				"drawOnChartArea": false,
			},
		}
	}
	
	return scales
}

func (w *ChartWidget) adjustOpacity(color string, opacity float64) string {
	// Simple opacity adjustment for hex colors
	if len(color) == 7 && color[0] == '#' {
		return fmt.Sprintf("%s%02x", color, int(255*opacity))
	}
	return color
}

// Statistical helper functions
func (w *ChartWidget) sum(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum
}

func (w *ChartWidget) average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return w.sum(values) / float64(len(values))
}

func (w *ChartWidget) min(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func (w *ChartWidget) max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func (w *ChartWidget) standardDeviation(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	
	avg := w.average(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - avg
		sumSquares += diff * diff
	}
	
	variance := sumSquares / float64(len(values)-1)
	return math.Sqrt(variance)
}