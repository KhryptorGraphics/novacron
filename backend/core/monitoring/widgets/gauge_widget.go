package widgets

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
)

// GaugeWidget represents a gauge visualization widget
type GaugeWidget struct {
	*BaseWidget
}

// GaugeConfig represents configuration for a gauge widget
type GaugeConfig struct {
	Min        float64 `json:"min"`
	Max        float64 `json:"max"`
	Unit       string  `json:"unit"`
	Precision  int     `json:"precision"`
	
	// Thresholds
	WarningThreshold  float64 `json:"warning_threshold"`
	CriticalThreshold float64 `json:"critical_threshold"`
	
	// Visual properties
	Size         int     `json:"size"`           // Gauge size in pixels
	ShowValue    bool    `json:"show_value"`     // Show numeric value
	ShowLabel    bool    `json:"show_label"`     // Show label
	Animated     bool    `json:"animated"`       // Animate value changes
	GaugeType    string  `json:"gauge_type"`     // "full", "semi", "arc"
	
	// Colors
	BackgroundColor string   `json:"background_color"`
	ForegroundColor string   `json:"foreground_color"`
	WarningColor    string   `json:"warning_color"`
	CriticalColor   string   `json:"critical_color"`
	Colors          []string `json:"colors"` // Custom color stops
	
	// Advanced
	StartAngle float64 `json:"start_angle"` // Starting angle in degrees
	EndAngle   float64 `json:"end_angle"`   // Ending angle in degrees
	Thickness  float64 `json:"thickness"`   // Gauge thickness (0-1)
}

// DefaultGaugeConfig returns default configuration for gauge widget
func DefaultGaugeConfig() *GaugeConfig {
	return &GaugeConfig{
		Min:               0,
		Max:               100,
		Unit:              "%",
		Precision:         1,
		WarningThreshold:  70,
		CriticalThreshold: 90,
		Size:              200,
		ShowValue:         true,
		ShowLabel:         true,
		Animated:          true,
		GaugeType:         "semi",
		BackgroundColor:   "#E0E0E0",
		ForegroundColor:   "#4CAF50",
		WarningColor:      "#FF9800",
		CriticalColor:     "#F44336",
		StartAngle:        -90,
		EndAngle:          90,
		Thickness:         0.1,
	}
}

// NewGaugeWidget creates a new gauge widget
func NewGaugeWidget() Widget {
	base := NewBaseWidget("", WidgetTypeGauge, "")
	
	// Set default configuration
	defaultConfig := DefaultGaugeConfig()
	configMap := make(map[string]interface{})
	configBytes, _ := json.Marshal(defaultConfig)
	json.Unmarshal(configBytes, &configMap)
	base.Config = configMap
	
	return &GaugeWidget{
		BaseWidget: base,
	}
}

// ValidateConfig validates the gauge configuration
func (w *GaugeWidget) ValidateConfig() error {
	config := w.getConfig()
	
	if config.Min >= config.Max {
		return fmt.Errorf("min value (%f) must be less than max value (%f)", config.Min, config.Max)
	}
	
	if config.WarningThreshold < config.Min || config.WarningThreshold > config.Max {
		return fmt.Errorf("warning threshold (%f) must be between min (%f) and max (%f)", 
			config.WarningThreshold, config.Min, config.Max)
	}
	
	if config.CriticalThreshold < config.Min || config.CriticalThreshold > config.Max {
		return fmt.Errorf("critical threshold (%f) must be between min (%f) and max (%f)", 
			config.CriticalThreshold, config.Min, config.Max)
	}
	
	if config.Size <= 0 {
		return fmt.Errorf("size must be positive, got %d", config.Size)
	}
	
	if config.Thickness <= 0 || config.Thickness > 1 {
		return fmt.Errorf("thickness must be between 0 and 1, got %f", config.Thickness)
	}
	
	validGaugeTypes := []string{"full", "semi", "arc"}
	validType := false
	for _, t := range validGaugeTypes {
		if config.GaugeType == t {
			validType = true
			break
		}
	}
	if !validType {
		return fmt.Errorf("invalid gauge type: %s, must be one of: %v", config.GaugeType, validGaugeTypes)
	}
	
	return nil
}

// ProcessData processes raw data for the gauge widget
func (w *GaugeWidget) ProcessData(ctx context.Context, rawData interface{}) (*WidgetData, error) {
	var value float64
	var labels map[string]string
	var metadata map[string]interface{}
	
	switch data := rawData.(type) {
	case float64:
		value = data
	case int:
		value = float64(data)
	case map[string]interface{}:
		if v, exists := data["value"]; exists {
			switch val := v.(type) {
			case float64:
				value = val
			case int:
				value = float64(val)
			default:
				return nil, fmt.Errorf("invalid value type: %T", val)
			}
		} else {
			return nil, fmt.Errorf("value field not found in data")
		}
		
		// Extract labels if present
		if l, exists := data["labels"].(map[string]string); exists {
			labels = l
		}
		
		// Extract metadata if present
		if m, exists := data["metadata"].(map[string]interface{}); exists {
			metadata = m
		}
	case *WidgetData:
		// Data is already processed
		if data.Value == nil {
			return nil, fmt.Errorf("no value in widget data")
		}
		switch val := data.Value.(type) {
		case float64:
			value = val
		case int:
			value = float64(val)
		default:
			return nil, fmt.Errorf("invalid value type in widget data: %T", val)
		}
		labels = data.Labels
		metadata = data.Metadata
	default:
		return nil, fmt.Errorf("unsupported data type: %T", rawData)
	}
	
	config := w.getConfig()
	
	// Clamp value to min/max range
	if value < config.Min {
		value = config.Min
	}
	if value > config.Max {
		value = config.Max
	}
	
	// Determine status level based on thresholds
	status := StatusLevelOK
	if value >= config.CriticalThreshold {
		status = StatusLevelCritical
	} else if value >= config.WarningThreshold {
		status = StatusLevelWarning
	}
	
	// Calculate percentage for gauge display
	percentage := (value - config.Min) / (config.Max - config.Min) * 100
	
	// Prepare metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["percentage"] = percentage
	metadata["status_level"] = status
	metadata["formatted_value"] = FormatValue(value, config.Unit, config.Precision)
	metadata["min"] = config.Min
	metadata["max"] = config.Max
	metadata["warning_threshold"] = config.WarningThreshold
	metadata["critical_threshold"] = config.CriticalThreshold
	
	return &WidgetData{
		WidgetID:  w.GetID(),
		Type:      WidgetTypeGauge,
		Format:    DataFormatSingle,
		Value:     value,
		Labels:    labels,
		Metadata:  metadata,
		Status:    "success",
		Unit:      config.Unit,
	}, nil
}

// GetSupportedDataFormats returns supported data formats
func (w *GaugeWidget) GetSupportedDataFormats() []DataFormat {
	return []DataFormat{DataFormatSingle}
}

// GetDataRequirements returns data requirements for the gauge widget
func (w *GaugeWidget) GetDataRequirements() DataRequirements {
	return DataRequirements{
		MinDataPoints:    1,
		MaxDataPoints:    1,
		RequiredFields:   []string{"value"},
		OptionalFields:   []string{"labels", "metadata"},
		SupportedFormats: []DataFormat{DataFormatSingle},
	}
}

// RenderHTML renders the gauge widget as HTML
func (w *GaugeWidget) RenderHTML(data *WidgetData, options RenderOptions) (string, error) {
	if data.Value == nil {
		return "", fmt.Errorf("no data to render")
	}
	
	config := w.getConfig()
	_, ok := data.Value.(float64)
	if !ok {
		return "", fmt.Errorf("invalid value type: %T", data.Value)
	}
	
	percentage := data.Metadata["percentage"].(float64)
	formattedValue := data.Metadata["formatted_value"].(string)
	statusLevel := data.Metadata["status_level"].(StatusLevel)
	
	// Determine color based on status
	color := config.ForegroundColor
	switch statusLevel {
	case StatusLevelWarning:
		color = config.WarningColor
	case StatusLevelCritical:
		color = config.CriticalColor
	}
	
	// Generate SVG gauge
	size := config.Size
	if options.Width > 0 && options.Width < size {
		size = options.Width
	}
	
	center := float64(size) / 2
	radius := center - 20
	thickness := radius * config.Thickness
	
	// Calculate angles
	startAngle := config.StartAngle * math.Pi / 180
	endAngle := config.EndAngle * math.Pi / 180
	angleRange := endAngle - startAngle
	currentAngle := startAngle + (angleRange * percentage / 100)
	
	html := fmt.Sprintf(`
<div class="gauge-widget" style="width: %dpx; height: %dpx;">
	<svg width="%d" height="%d" viewBox="0 0 %d %d">
		<!-- Background arc -->
		<path d="M %f %f A %f %f 0 %d 1 %f %f"
			  fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>
		
		<!-- Value arc -->
		<path d="M %f %f A %f %f 0 %d 1 %f %f"
			  fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>
		
		<!-- Center dot -->
		<circle cx="%f" cy="%f" r="3" fill="%s"/>
	</svg>`,
		size, size, size, size, size, size,
		
		// Background arc
		center+radius*math.Cos(startAngle), center+radius*math.Sin(startAngle),
		radius, radius,
		boolToInt(angleRange > math.Pi), // large-arc-flag
		center+radius*math.Cos(endAngle), center+radius*math.Sin(endAngle),
		config.BackgroundColor, thickness,
		
		// Value arc
		center+radius*math.Cos(startAngle), center+radius*math.Sin(startAngle),
		radius, radius,
		boolToInt(percentage/100*angleRange > math.Pi), // large-arc-flag
		center+radius*math.Cos(currentAngle), center+radius*math.Sin(currentAngle),
		color, thickness,
		
		// Center
		center, center, color,
	)
	
	// Add value text
	if config.ShowValue {
		html += fmt.Sprintf(`
	<text x="%f" y="%f" text-anchor="middle" dominant-baseline="middle" 
		  font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="%s">
		%s
	</text>`,
			center, center+10, color, formattedValue)
	}
	
	// Add label
	if config.ShowLabel && w.GetTitle() != "" {
		html += fmt.Sprintf(`
	<text x="%f" y="%f" text-anchor="middle" dominant-baseline="middle" 
		  font-family="Arial, sans-serif" font-size="12" fill="#666">
		%s
	</text>`,
			center, center+30, w.GetTitle())
	}
	
	html += `
	</svg>
</div>`
	
	// Add CSS for animations if enabled
	if config.Animated {
		html += `
<style>
.gauge-widget path {
	transition: stroke-dasharray 1s ease-in-out;
}
</style>`
	}
	
	return html, nil
}

// RenderJSON renders the gauge widget data as JSON
func (w *GaugeWidget) RenderJSON(data *WidgetData) (json.RawMessage, error) {
	if data == nil {
		return nil, fmt.Errorf("no data to render")
	}
	
	config := w.getConfig()
	
	result := map[string]interface{}{
		"type":       "gauge",
		"value":      data.Value,
		"percentage": data.Metadata["percentage"],
		"status":     data.Metadata["status_level"],
		"formatted":  data.Metadata["formatted_value"],
		"config": map[string]interface{}{
			"min":        config.Min,
			"max":        config.Max,
			"unit":       config.Unit,
			"thresholds": map[string]float64{
				"warning":  config.WarningThreshold,
				"critical": config.CriticalThreshold,
			},
			"visual": map[string]interface{}{
				"size":       config.Size,
				"gaugeType":  config.GaugeType,
				"animated":   config.Animated,
				"showValue":  config.ShowValue,
				"showLabel":  config.ShowLabel,
				"colors": map[string]string{
					"background": config.BackgroundColor,
					"foreground": config.ForegroundColor,
					"warning":    config.WarningColor,
					"critical":   config.CriticalColor,
				},
			},
		},
		"metadata": data.Metadata,
	}
	
	return json.Marshal(result)
}

// getConfig returns the typed configuration
func (w *GaugeWidget) getConfig() *GaugeConfig {
	config := &GaugeConfig{}
	configBytes, _ := json.Marshal(w.Config)
	json.Unmarshal(configBytes, config)
	
	// Apply defaults for missing values
	defaults := DefaultGaugeConfig()
	if config.Min == 0 && config.Max == 0 {
		config.Min = defaults.Min
		config.Max = defaults.Max
	}
	if config.Unit == "" {
		config.Unit = defaults.Unit
	}
	if config.GaugeType == "" {
		config.GaugeType = defaults.GaugeType
	}
	if config.BackgroundColor == "" {
		config.BackgroundColor = defaults.BackgroundColor
	}
	if config.ForegroundColor == "" {
		config.ForegroundColor = defaults.ForegroundColor
	}
	if config.WarningColor == "" {
		config.WarningColor = defaults.WarningColor
	}
	if config.CriticalColor == "" {
		config.CriticalColor = defaults.CriticalColor
	}
	
	return config
}

// Helper function to convert bool to int for SVG arc flags
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// UpdateThresholds updates the warning and critical thresholds
func (w *GaugeWidget) UpdateThresholds(warning, critical float64) error {
	config := w.getConfig()
	
	if warning < config.Min || warning > config.Max {
		return fmt.Errorf("warning threshold must be between min and max values")
	}
	
	if critical < config.Min || critical > config.Max {
		return fmt.Errorf("critical threshold must be between min and max values")
	}
	
	if warning >= critical {
		return fmt.Errorf("warning threshold must be less than critical threshold")
	}
	
	w.Config["warning_threshold"] = warning
	w.Config["critical_threshold"] = critical
	
	return nil
}

// SetRange sets the min and max values for the gauge
func (w *GaugeWidget) SetRange(min, max float64) error {
	if min >= max {
		return fmt.Errorf("min value must be less than max value")
	}
	
	w.Config["min"] = min
	w.Config["max"] = max
	
	// Adjust thresholds if they're outside the new range
	config := w.getConfig()
	if config.WarningThreshold < min || config.WarningThreshold > max {
		w.Config["warning_threshold"] = min + (max-min)*0.7 // 70% of range
	}
	if config.CriticalThreshold < min || config.CriticalThreshold > max {
		w.Config["critical_threshold"] = min + (max-min)*0.9 // 90% of range
	}
	
	return nil
}

// SetColors sets the color scheme for the gauge
func (w *GaugeWidget) SetColors(background, foreground, warning, critical string) error {
	// Validate colors (basic validation)
	colors := []string{background, foreground, warning, critical}
	for _, color := range colors {
		if color != "" && !isValidColor(color) {
			return fmt.Errorf("invalid color format: %s", color)
		}
	}
	
	if background != "" {
		w.Config["background_color"] = background
	}
	if foreground != "" {
		w.Config["foreground_color"] = foreground
	}
	if warning != "" {
		w.Config["warning_color"] = warning
	}
	if critical != "" {
		w.Config["critical_color"] = critical
	}
	
	return nil
}

// isValidColor checks if a color string is valid (basic check)
func isValidColor(color string) bool {
	// Check hex colors
	if len(color) == 7 && color[0] == '#' {
		for i := 1; i < 7; i++ {
			c := color[i]
			if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
				return false
			}
		}
		return true
	}
	
	// Check named colors (basic list)
	namedColors := []string{
		"red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
		"black", "white", "gray", "grey", "transparent",
	}
	
	for _, named := range namedColors {
		if color == named {
			return true
		}
	}
	
	return false
}